# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-8B vLLM generator for tt-inference-server.

Works with both the new vllm fork (has molmo2.py with Molmo2MultiModalProcessor)
and the reference vllm fork (molmo.py only, sends raw frames to prefill_forward).

Raw frame handling: if pixel_values_videos arrives as [n, 3, H, W] (reference vllm),
we unfold to patch format [n, 729, 588] before passing to TtMolmo2Model.
"""

from pathlib import Path
from typing import Mapping, Optional

import torch
from loguru import logger

import ttnn
from models.common.warmup import WarmupForwardMixin
from models.demos.molmo2.tt.model_config import Molmo2Config
from models.tt_transformers.tt.ccl import TT_CCL
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
            # Support both image and video — prefill_forward handles both via
            # pixel_values (image) and pixel_values_videos (video) kwargs.
            return {"video": 1, "image": 1}

    _registry_decorator = MULTIMODAL_REGISTRY.register_processor(
        Molmo2MultiModalProcessor,
        info=_TT_Molmo2ProcessingInfo,
        dummy_inputs=Molmo2DummyInputsBuilder,
    )
except ImportError:
    _registry_decorator = lambda cls: cls  # no-op for reference vllm

WEIGHT_CACHE_PATH = Path("/tmp/molmo2_weight_cache")
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

    def __init__(self, model, cfg, mesh_device, processor):
        self.model = model
        self.cfg = cfg
        self.mesh_device = mesh_device
        self.processor = processor
        self._decode_trace_captured = False

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

        cfg = Molmo2Config(mesh_device=mesh_device)
        cfg.max_batch_size = max_batch_size
        cfg.max_seq_len = max_seq_len

        ccl = TT_CCL(mesh_device)
        WEIGHT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

        model = TtMolmo2Model(
            mesh_device=mesh_device,
            tt_ccl=ccl,
            state_dict=state_dict,
            weight_cache_path=WEIGHT_CACHE_PATH,
            dtype=ttnn.bfloat16,
            configuration=cfg,
        )
        del state_dict
        logger.info("TtMolmo2Model ready")

        # Pre-compile JIT kernels for all prefill bucket sizes and vision ops
        # before the server starts serving — avoids stall on first inference.
        # Use the same PREFILL_BUCKETS as the demo (test_10_videos.py).
        from models.demos.molmo2.tt.model import PREFILL_BUCKETS

        logger.info(f"Pre-compiling prefill JIT kernels for buckets {PREFILL_BUCKETS}...")
        model.warmup_all_buckets(bucket_sizes=PREFILL_BUCKETS, use_trace=False)
        # After warmup_all_buckets, KV cache is filled by the last bucket's forward_prefill.
        # Run one decode step to JIT-compile the decode kernel — avoids first-inference stall.
        logger.info("Pre-compiling decode JIT kernel...")
        _ = model.forward_decode_step(0, PREFILL_BUCKETS[-1])
        logger.info("Pre-compiling vision JIT kernels...")
        model.warmup_vision_compile()

        # warmup_all_buckets runs text-only (pixel_values=None), skipping the vision
        # feature injection path (ttnn.add for image patches). Run forward_prefill with
        # dummy vision inputs for EVERY bucket to JIT all decoder/add paths in vision mode.
        # Without this, each new bucket triggers a ~25s JIT stall on first vision request.
        logger.info("Pre-compiling vision-integrated prefill for all buckets...")
        _n_patches, _k_pool, _n_pooled = 729, 9, 81  # 1 frame: 81 pooled positions
        _dummy_pv = torch.zeros(1, 8, _n_patches, 588)  # 8 crops (max ViT batch)
        _dummy_pool_idx = torch.zeros(1, _n_pooled, _k_pool, dtype=torch.long)
        for _S_warmup in PREFILL_BUCKETS:
            if _S_warmup < _n_pooled:
                continue  # bucket too small to hold n_pooled image tokens
            _dummy_ids = torch.zeros(1, _S_warmup, dtype=torch.long)
            _dummy_ids[0, :_n_pooled] = cfg.image_patch_id
            # Pass token_type_ids so build_molmo2_prefill_mask (ttnn.mul/maximum/where)
            # is JIT-compiled here rather than on the first real inference.
            # Limit to S_warmup <= 8192: the [S,S] mask tensor at S=16384 is 536MB and
            # at S=32768 is 2GB — too large to allocate 2 copies during mask build.
            # Real vision inputs max at S=4233 (51 frames × 83 tok) → bucket 8192.
            _dummy_tti = None
            if _S_warmup <= 8192:
                _dummy_tti = torch.zeros(1, _S_warmup, dtype=torch.long)
                _dummy_tti[0, :_n_pooled] = 1  # mark image positions as type=1
            logger.info(f"  vision-integrated prefill bucket {_S_warmup}...")
            _ = model.forward_prefill(
                input_ids=_dummy_ids,
                pixel_values=_dummy_pv,
                pooled_patches_idx=_dummy_pool_idx,
                token_type_ids=_dummy_tti,
                user_id=0,
            )
        logger.info("JIT warmup complete — server ready to serve")

        return cls(model=model, cfg=cfg, mesh_device=mesh_device, processor=processor)

    @property
    def cache_path(self):
        return WEIGHT_CACHE_PATH

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_molmo2_kv_cache(*args, **kwargs, model=self.model, cfg=self.cfg)

    def _unwrap(self, val):
        """Unwrap up to two layers of list nesting."""
        if isinstance(val, list):
            val = val[0] if val else None
        if isinstance(val, list):
            val = val[0] if val else None
        return val

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, enable_trace=False, **kwargs):
        """Run prefill for a single user (batch_size=1).

        tokens: [batch, padded_seq_len] int32
        prompt_lens: [batch] int — actual length before padding
        pixel_values_videos may be:
          [n, 729, 588] pre-processed patches (new vllm / our processor)
          [n, 3, H, W]  raw normalized frames  (reference vllm)
        """
        pv = None
        pool_idx = None
        token_type_ids = self._unwrap(kwargs.get("token_type_ids"))

        pixel_values_videos = self._unwrap(kwargs.get("pixel_values_videos"))
        video_token_pooling = self._unwrap(kwargs.get("video_token_pooling"))
        pixel_values = self._unwrap(kwargs.get("pixel_values"))
        image_token_pooling = self._unwrap(kwargs.get("image_token_pooling"))

        if pixel_values_videos is not None:
            t = pixel_values_videos.float()
            # Reference vllm sends raw frames [n, 3, H, W]; convert to patches
            if t.dim() == 4 and t.shape[1] == 3:
                logger.info(f"Converting raw frames {t.shape} → patches")
                t = _raw_frames_to_patches(t)
            pv = t.unsqueeze(0)  # [1, n_frames, 729, 588]
            if video_token_pooling is not None:
                pool_idx = video_token_pooling.unsqueeze(0)
        elif pixel_values is not None:
            t = pixel_values.float()
            if t.dim() == 4 and t.shape[1] == 3:
                t = _raw_frames_to_patches(t)
            pv = t.unsqueeze(0)
            if image_token_pooling is not None:
                pool_idx = image_token_pooling.unsqueeze(0)

        seq_len = int(prompt_lens[0].item()) if hasattr(prompt_lens[0], "item") else int(prompt_lens[0])
        input_ids = tokens[:1, :seq_len]

        # Reconstruct token_type_ids matching HF processor's output exactly:
        # type=1 for image_patch_id (151938) AND frame markers <im_start> (151936) / <im_end> (151937).
        # The HF processor marks all three as type=1 (4233 positions for 51 frames = 51×83).
        # Without frame markers: only 4131 type=1 positions — markers get causal-only attention,
        # breaking the bidirectional image↔marker attention the model was trained with.
        if token_type_ids is None and pv is not None:
            _IMAGE_PATCH_ID = 151938
            _IM_START = 151936  # <im_start> frame boundary token
            _IM_END = 151937  # <im_end>   frame boundary token
            token_type_ids = ((input_ids == _IMAGE_PATCH_ID) | (input_ids == _IM_START) | (input_ids == _IM_END)).long()

        import time

        self.model.reset_kv_cache(user_id=0)
        vision_str = "video" if pixel_values_videos is not None else "image" if pixel_values is not None else "none"
        logger.info(f"Prefill: S={seq_len}, vision={vision_str}")

        t0 = time.perf_counter()
        logits = self.model.forward_prefill(
            input_ids=input_ids,
            pixel_values=pv,
            pooled_patches_idx=pool_idx,
            token_type_ids=token_type_ids,
            user_id=0,
        )
        self._last_prefill_ms = (time.perf_counter() - t0) * 1000
        self._decode_step_count = 0
        self._decode_total_ms = 0.0
        logger.info(f"Prefill done: {self._last_prefill_ms:.0f}ms")

        # Decode trace is pre-captured at server bringup (initialize_vllm_model).
        # No lazy capture here — nothing should JIT or trace during inference.

        return logits, None  # (logits, rope_deltas=None)

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
        """Run single-token decode."""
        token_id = int(tokens[0, 0].item())
        position = int(start_pos[0].item()) if hasattr(start_pos[0], "item") else int(start_pos[0])

        import time

        t0 = time.perf_counter()
        # Use forward_decode_step (eager, no trace) — decode JIT is pre-compiled during
        # initialize_vllm_model warmup so no first-inference stall happens.
        logits = self.model.forward_decode_step(token_id, position).squeeze(0)
        step_ms = (time.perf_counter() - t0) * 1000

        self._decode_step_count = getattr(self, "_decode_step_count", 0) + 1
        self._decode_total_ms = getattr(self, "_decode_total_ms", 0.0) + step_ms
        logger.info(
            f"Decode step {self._decode_step_count}: pos={position} {step_ms:.0f}ms  "
            f"(total decode {self._decode_total_ms:.0f}ms, "
            f"prefill {getattr(self,'_last_prefill_ms',0):.0f}ms)"
        )

        return logits.unsqueeze(0)  # [1, vocab_size]
