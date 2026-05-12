# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-8B vLLM generator for tt-inference-server.

Works with both the new vllm fork (has molmo2.py with Molmo2MultiModalProcessor)
and the reference vllm fork (molmo.py only, sends raw frames to prefill_forward).

Raw frame handling: if pixel_values_videos arrives as [n, 3, H, W] (reference vllm),
we unfold to patch format [n, 729, 588] before passing to TtMolmo2Model.
"""

from pathlib import Path
from typing import List, Mapping, Optional

import torch
from loguru import logger

import ttnn
from models.common.warmup import WarmupForwardMixin
from models.demos.molmo2.tt.model_config import Molmo2Config
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.generator import create_submeshes
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY

# New vllm fork has molmo2.py; reference fork does not.
try:
    from vllm.model_executor.models.molmo2 import (
        Molmo2DummyInputsBuilder,
        Molmo2MultiModalProcessor,
        Molmo2ProcessingInfo,
    )

    class _TT_Molmo2ProcessingInfo(Molmo2ProcessingInfo):
        def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
            # TT limit: 1 video OR up to 23 images per request.
            # 23 = max_seq_len(36864) // (max_crops(8) * tokens_per_crop_image(196))
            # Image pooling=[2,2] → 196 tokens/crop; video pooling=[3,3] → 81 tokens/frame.
            return {"video": 1, "image": 23}

    class _TT_Molmo2DummyInputsBuilder(Molmo2DummyInputsBuilder):
        """Native image and video dummy inputs for vLLM init (token budget).

        Images use the native <|image|> path (not redirected to video).
        Chat template: 1 image → '<|image|>', N images → 'Image 1<|image|>...'
        """

        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            num_v = mm_counts.get("video", 0)
            num_i = mm_counts.get("image", 0)
            # Video and image cannot be mixed — prefer video if both present.
            if num_v > 0:
                return "<|video|>" * num_v
            if num_i == 1:
                return "<|image|>"
            if num_i > 1:
                return "".join(f"Image {i + 1}<|image|>" for i in range(num_i))
            return ""

        def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
            num_v = mm_counts.get("video", 0)
            num_i = mm_counts.get("image", 0)
            # Video and image cannot be mixed: HF Molmo2Processor only handles one
            # modality at a time. When both are requested (budget sizing), prefer video.
            if num_v > 0:
                return super().get_dummy_mm_data(seq_len, {"video": num_v}, mm_options)
            if num_i > 0:
                return super().get_dummy_mm_data(seq_len, {"image": num_i}, mm_options)
            return {}

    _registry_decorator = MULTIMODAL_REGISTRY.register_processor(
        Molmo2MultiModalProcessor,
        info=_TT_Molmo2ProcessingInfo,
        dummy_inputs=_TT_Molmo2DummyInputsBuilder,
    )
except ImportError:
    _registry_decorator = lambda cls: cls  # no-op for reference vllm

import os as _os

WEIGHT_CACHE_PATH = Path(_os.environ.get("MOLMO2_WEIGHT_CACHE", f"/tmp/molmo2_weight_cache_u{_os.getuid()}"))
_PATCH_SIZE = 14
_PATCH_FEATURES = _PATCH_SIZE * _PATCH_SIZE * 3  # 588


def _raw_frames_to_patches(frames: torch.Tensor) -> torch.Tensor:
    """Convert raw normalized frames [n, 3, H, W] → patch format [n, n_patches, 588].

    Used when the reference vllm sends raw frames instead of pre-processed patches.
    """
    x = frames.unfold(2, _PATCH_SIZE, _PATCH_SIZE).unfold(3, _PATCH_SIZE, _PATCH_SIZE)
    # x: [n, c, h/p, w/p, p, p] → [n, n_patches, 588]
    return x.permute(0, 2, 3, 4, 5, 1).reshape(frames.shape[0], -1, _PATCH_FEATURES)


def allocate_molmo2_kv_cache(kv_cache_shape, dtype, num_layers, model, cfg):
    """Replace each layer's KV cache with new TTNN tensors."""
    cache_shape = (cfg.max_batch_size, cfg.n_local_kv_heads, cfg.max_seq_len, cfg.head_dim)
    for layer_idx in range(num_layers):
        cache_kv = torch.zeros(cache_shape, dtype=torch.bfloat16)
        model.layers[layer_idx].attention.layer_past = [
            ttnn.as_tensor(
                cache_kv,
                device=model.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                cache_file_name=None,
            )
            for _ in range(2)  # K and V
        ]
    return [layer.attention.layer_past for layer in model.layers]


@_registry_decorator
class Molmo2ForConditionalGeneration(WarmupForwardMixin, SupportsMultiModal):
    """TT Molmo2 vLLM generator wrapping TtMolmo2Model."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, models, cfgs, submesh_devices, full_mesh_device, processor):
        assert len(models) == len(cfgs) == len(submesh_devices)
        self.models: List = list(models)
        self.cfgs: List = list(cfgs)
        self.submesh_devices: List = list(submesh_devices)
        self.dp = len(models)
        self.batch_per_dp = cfgs[0].max_batch_size
        # Single-replica back-compat aliases (helpers that still reach for
        # self.model / self.cfg / self.mesh_device pick up replica 0).
        self.model = models[0]
        self.cfg = cfgs[0]
        self.mesh_device = full_mesh_device
        self.processor = processor
        self._decode_trace_captured = [True] * self.dp  # captured eagerly during init

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
    ):
        """Called by TTModelLoader after vLLM resolves TTMolmo2ForConditionalGeneration."""
        import os

        from transformers import AutoModelForImageTextToText, AutoProcessor

        from models.demos.molmo2.tt.model import TtMolmo2Model

        hf_model_id = getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", "")
        # Use HF_MODEL env var (set by run_vllm_api_server.py to local symlink) to load
        # weights from local path instead of downloading from HF.
        hf_path = os.environ.get("HF_MODEL", hf_model_id)
        logger.info(f"Initializing Molmo2 TT model from {hf_path}")

        logger.info("Loading HF state dict (bfloat16)...")
        hf = AutoModelForImageTextToText.from_pretrained(
            hf_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        state_dict = hf.state_dict()
        del hf

        processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)

        # Detect Galaxy (8,4) and split into 4 (1,8) submeshes for DP=4 inside one
        # EngineCore. We override the loader-provided tt_data_parallel because in
        # tt-inference-server we set vllm's data_parallel_size=1 (the alternative
        # spawns 4 EngineCore processes which exhausts Galaxy PCIe TLBs).
        try:
            mesh_shape = tuple(mesh_device.shape)
        except Exception:
            mesh_shape = None
        if tt_data_parallel == 1 and mesh_shape == (8, 4):
            logger.info("Galaxy mesh detected (8,4); forcing internal tt_data_parallel=4")
            tt_data_parallel = 4

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
        assert (
            len(submesh_devices) == tt_data_parallel
        ), f"create_submeshes returned {len(submesh_devices)} submeshes, expected {tt_data_parallel}"
        batch_per_dp = max(1, max_batch_size // tt_data_parallel)
        logger.info(
            f"Building {tt_data_parallel} TtMolmo2Model replica(s) on submeshes "
            f"of shape {[tuple(m.shape) for m in submesh_devices]}, batch_per_dp={batch_per_dp}"
        )

        WEIGHT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

        models: List = []
        cfgs: List = []
        for dp_idx, submesh in enumerate(submesh_devices):
            cfg = Molmo2Config(mesh_device=submesh)
            cfg.max_batch_size = batch_per_dp
            cfg.max_seq_len = max_seq_len
            ccl = TT_CCL(submesh)
            logger.info(f"Replica {dp_idx + 1}/{tt_data_parallel}: building TtMolmo2Model")
            model = TtMolmo2Model(
                mesh_device=submesh,
                tt_ccl=ccl,
                state_dict=state_dict,
                weight_cache_path=WEIGHT_CACHE_PATH,
                dtype=ttnn.bfloat16,
                configuration=cfg,
            )
            models.append(model)
            cfgs.append(cfg)
        del state_dict
        logger.info(f"All {tt_data_parallel} TtMolmo2Model replica(s) ready")

        # Two-step warmup: JIT compile prefill kernels + vision kernels. Applied to
        # every replica (single mesh or each submesh). This makes prefill deterministic
        # — cold-JIT allocations otherwise vary the memory layout across requests and
        # cause bf16 reduction-order non-determinism in prefill outputs.
        #
        # We deliberately do NOT call:
        #   - warmup_all_buckets(use_trace=True): prefill traces conflict with the
        #     decode trace for video/image eager prefill (commit 1dd8490d7d2).
        #   - warmup_decode_trace(): captures decode trace at PREFILL_BUCKETS[-1]
        #     =32768. The "position-agnostic" claim breaks under the server (concurrent
        #     users on one model) — produces garbage decode at S≪32768. Lazy capture
        #     at first request's actual position works.
        from models.demos.molmo2.tt.model import PREFILL_BUCKETS

        for dp_idx, m in enumerate(models):
            logger.info(f"[warmup {dp_idx + 1}/{len(models)}] JIT prefill buckets {PREFILL_BUCKETS}")
            m.warmup_all_buckets(bucket_sizes=PREFILL_BUCKETS, use_trace=False)
            logger.info(f"[warmup {dp_idx + 1}/{len(models)}] vision compile")
            m.warmup_vision_compile()
        logger.info("All replicas warmed up (JIT prefill + vision). Decode trace will capture lazily.")

        return cls(
            models=models,
            cfgs=cfgs,
            submesh_devices=submesh_devices,
            full_mesh_device=mesh_device,
            processor=processor,
        )

    @property
    def cache_path(self):
        return WEIGHT_CACHE_PATH

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate KV cache on every replica; vLLM only calls this once on rank 0.

        Returns a list-of-lists (per-replica layer caches). vLLM treats it as
        opaque since Molmo2 sets supports_prefix_caching=False.
        """
        return [
            allocate_molmo2_kv_cache(kv_cache_shape, dtype, num_layers, model=m, cfg=c)
            for m, c in zip(self.models, self.cfgs)
        ]

    def _route(self, user_id: int) -> int:
        """Map a user/batch index to a DP replica index."""
        return min(user_id // self.batch_per_dp, self.dp - 1)

    def _unwrap(self, val, idx=0):
        """Unwrap up to two layers of list nesting, optionally indexing instead of [0]."""
        if isinstance(val, list):
            val = val[idx] if idx < len(val) else (val[0] if val else None)
        if isinstance(val, list):
            val = val[0] if val else None
        return val

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, enable_trace=False, **kwargs):
        """Run prefill for each user in the batch; route per-user to a DP replica.

        tokens: [batch, padded_seq_len] int32
        prompt_lens: [batch] int — actual length before padding
        pixel_values_videos may be:
          [n, 729, 588] pre-processed patches (new vllm / our processor)
          [n, 3, H, W]  raw normalized frames  (reference vllm)
        """
        import time

        token_type_ids_all = kwargs.get("token_type_ids")
        pixel_values_videos_all = kwargs.get("pixel_values_videos")
        video_token_pooling_all = kwargs.get("video_token_pooling")
        pixel_values_all = kwargs.get("pixel_values")
        image_token_pooling_all = kwargs.get("image_token_pooling")

        batch_size = tokens.shape[0]
        out_logits = []
        for u in range(batch_size):
            rank = self._route(u)
            model = self.models[rank]
            local_uid = u % self.batch_per_dp

            token_type_ids = self._unwrap(token_type_ids_all, idx=u)
            pixel_values_videos = self._unwrap(pixel_values_videos_all, idx=u)
            video_token_pooling = self._unwrap(video_token_pooling_all, idx=u)
            pixel_values = self._unwrap(pixel_values_all, idx=u)
            image_token_pooling = self._unwrap(image_token_pooling_all, idx=u)

            pv = None
            pool_idx = None
            if pixel_values_videos is not None:
                t = pixel_values_videos.float()
                if t.dim() == 4 and t.shape[1] == 3:
                    logger.info(f"Converting raw frames {t.shape} → patches")
                    t = _raw_frames_to_patches(t)
                pv = t.unsqueeze(0)
                if video_token_pooling is not None:
                    pool_idx = video_token_pooling.unsqueeze(0)
            elif pixel_values is not None:
                t = pixel_values.float()
                if t.dim() == 4 and t.shape[1] == 3:
                    t = _raw_frames_to_patches(t)
                pv = t.unsqueeze(0)
                if image_token_pooling is not None:
                    pool_idx = image_token_pooling.unsqueeze(0)

            seq_len = int(prompt_lens[u].item()) if hasattr(prompt_lens[u], "item") else int(prompt_lens[u])
            input_ids = tokens[u : u + 1, :seq_len]

            if token_type_ids is None and pv is not None:
                _IMAGE_PATCH_ID = 151938
                _IM_START = 151936
                _IM_END = 151937
                token_type_ids = (
                    (input_ids == _IMAGE_PATCH_ID) | (input_ids == _IM_START) | (input_ids == _IM_END)
                ).long()

            model.reset_kv_cache(user_id=local_uid)
            vision_str = "video" if pixel_values_videos is not None else "image" if pixel_values is not None else "none"
            logger.info(f"Prefill rank={rank} user={u} local_uid={local_uid}: S={seq_len}, vision={vision_str}")

            t0 = time.perf_counter()
            logits = model.forward_prefill(
                input_ids=input_ids,
                pixel_values=pv,
                pooled_patches_idx=pool_idx,
                token_type_ids=token_type_ids,
                user_id=local_uid,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"Prefill done (rank={rank}): {elapsed_ms:.0f}ms")
            self._last_prefill_ms = elapsed_ms
            self._decode_step_count = 0
            self._decode_total_ms = 0.0
            out_logits.append(logits)

        # Stack along batch dim, then add a unit seq-len dim so the runner can do
        # prefill_output[:, -1, :]. Bare-tensor return (no tuple) keeps both
        # tt_model_runner code paths happy (mixed-batch + single-request).
        stacked = torch.cat([l if l.dim() == 2 else l.unsqueeze(0) for l in out_logits], dim=0)  # [batch, vocab]
        return stacked.unsqueeze(1)  # [batch, 1, vocab]

    def _capture_decode_trace(self, prefill_seq_len: int):
        self.model._decode_trace_tensors = self.model._allocate_decode_trace_tensors()
        self.model._decode_trace_id, self.model._decode_trace_output = self.model._capture_decode_trace(
            self.model._decode_trace_tensors, prefill_seq_len
        )
        self._decode_trace_captured = True
        logger.info("Decode trace captured")

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=False,
        read_from_device=True,
        sampling_params=None,
        **kwargs,
    ):
        """Run single-token decode for each user; route per-user to its DP replica.

        Padding slots (start_pos < 0, from runner pad-to-max_num_reqs) get a
        zeros logit shaped to match the real outputs (vocab inferred from a real
        forward).
        """
        import time

        batch_size = tokens.shape[0]
        out_logits = [None] * batch_size

        # Collect active (non-padding) users routed to their DP rank.
        active = []
        for u in range(batch_size):
            position = int(start_pos[u].item()) if hasattr(start_pos[u], "item") else int(start_pos[u])
            if position < 0:
                continue  # padding slot
            rank = self._route(u)
            token_id = int(tokens[u, 0].item())
            active.append((u, rank, token_id, position))

        if active:
            t0 = time.perf_counter()
            # Loop 0: lazy decode-trace capture per rank (at first decode position).
            # The trace's SDPA reads KV cache up to current_pos; capturing at the
            # actual S rather than PREFILL_BUCKETS[-1]=32768 avoids reading garbage.
            for u, rank, _, position in active:
                m = self.models[rank]
                if m._decode_trace_id is None:
                    m._decode_trace_tensors = m._allocate_decode_trace_tensors()
                    m._decode_trace_id, m._decode_trace_output = m._capture_decode_trace(
                        m._decode_trace_tensors, position
                    )

            # Loop 1: update inputs on every active rank's stable trace buffers.
            for u, rank, token_id, position in active:
                self.models[rank]._update_decode_inputs(token_id, position)

            # Loop 2: dispatch every active rank's decode trace non-blocking — all
            # submeshes run concurrently (mirrors tt_transformers Generator at
            # tt/generator.py:1292). Single-rank galaxy_t3k path: 1 dispatch.
            for u, rank, _, _ in active:
                self.models[rank]._dispatch_decode_trace(blocking=False)

            # Loop 3: read each rank's logits. ttnn.to_torch is blocking, so this
            # also synchronizes the non-blocking dispatches above.
            for u, rank, _, _ in active:
                out_logits[u] = self.models[rank]._read_decode_output().unsqueeze(0)

            step_ms = (time.perf_counter() - t0) * 1000
            self._decode_step_count = getattr(self, "_decode_step_count", 0) + 1
            self._decode_total_ms = getattr(self, "_decode_total_ms", 0.0) + step_ms
            logger.info(
                f"Decode active={len(active)} step {self._decode_step_count}: "
                f"positions={[p for _,_,_,p in active]} {step_ms:.0f}ms"
            )

        real = next((l for l in out_logits if l is not None), None)
        if real is None:
            return torch.zeros(0, 0)
        vocab = real.shape[-1]
        for u in range(batch_size):
            if out_logits[u] is None:
                out_logits[u] = torch.zeros(1, vocab, dtype=real.dtype)
        return torch.cat(out_logits, dim=0)  # [batch, vocab]
