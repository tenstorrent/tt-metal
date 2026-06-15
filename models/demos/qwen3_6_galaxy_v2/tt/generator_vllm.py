# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM generator wrapper for Qwen3.6-27B (text-only) on BH Galaxy.

Mirrors the construction in demo/text_demo_qwen36.py: local v2 TtTransformer +
TtQwen36ModelArgs. Weights are loaded from raw safetensors because the
checkpoint's `qwen3_5` architecture is not in any public transformers release
and cannot be loaded by the standard HF model loader.
"""
import os
from pathlib import Path

import torch
from tqdm import tqdm

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
from models.demos.qwen3_6_galaxy_v2.tt.load_checkpoints import load_hf_state_dict
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs


def _resolve_ckpt_dir() -> Path:
    """Local checkpoint dir. The server sets HF_MODEL to a local symlink dir."""
    hf_model = os.getenv("HF_MODEL", "Qwen/Qwen3.6-27B")
    p = Path(hf_model)
    if p.is_dir():
        return p
    # Fall back to a resolved HF snapshot.
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_model))


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: TtTransformer, tt_cache_path):
    """Paged KV cache allocation for qwen3.6.

    CRITICAL: the external (vLLM-allocated) KV cache MUST match the layout the
    model's paged ops were validated against — the model-internal cache from
    ``TtLlamaAttention.init_kv_cache`` (the demo / test_decode_eager_64L_pcc.py
    path). That layout is ROW-SHARDED across the 8 mesh rows:

        torch shape  = [num_blocks, n_kv_full=8, block_size, head_dim]
        mesh_mapper  = ShardTensor2dMesh(dims=(1, None), mesh_shape=cluster_shape)
        => per chip   [num_blocks, 1, block_size, head_dim], KV head h on row h,
           replicated across the 4 columns.

    vLLM passes ``kv_cache_shape = (num_blocks, num_kv_heads_per_dev=1,
    block_size, head_size)`` (already TP-divided). Allocating that REPLICATED
    (the old code) gives every device an identical 1-head buffer with REPLICATE
    metadata — which does NOT match the row-sharded layout the paged
    fill_cache / decode-SDPA + attention all-gather expect, so decode reads the
    wrong KV heads and degenerates into garbage (prefill's first token is
    computed pre-cache, so it stays correct — the observed symptom). Rebuild the
    cache here with the model's n_kv_full row-shard, keeping vLLM's num_blocks /
    block_size / head_size so the block_table indices stay valid.
    """
    kv_dtype = ttnn.bfloat8_b if os.getenv("QWEN36_KV_BF8", "0") == "1" else ttnn.bfloat16
    num_blocks, _vllm_num_kv_per_dev, block_size, head_size = kv_cache_shape
    n_kv_full = model.args.n_kv_heads  # 8 (padded); sharded 1/row across the 8 rows
    cluster_shape = model.args.cluster_shape
    submesh_devices = [model.mesh_device]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros((num_blocks, n_kv_full, block_size, head_size), dtype=dtype)
        row_shard_kv = ttnn.ShardTensor2dMesh(submesh, dims=(1, None), mesh_shape=cluster_shape)
        kv_tt = []
        for _ in tqdm(range(num_layers), desc=f"Allocating TT kv caches (submesh {mesh_idx+1})"):
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    mesh_mapper=row_shard_kv,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_dtype,
                    cache_file_name=tt_cache_path
                    / f"empty_{kv}cache_paged_rowshard_{num_blocks}_{n_kv_full}_{block_size}_{head_size}",
                )
                for kv in ["k", "v"]
            ]
            kv_tt.append(kv_tt_i)
        kv_cache.append(kv_tt)
    return kv_cache


def initialize_vllm_text_transformer_qwen36(
    hf_config,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
):
    instruct = "instruct" in str(getattr(hf_config, "_name_or_path", "")).lower()
    args = TtQwen36ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if n_layers is not None:
        args.n_layers = n_layers

    # Raw safetensors load — the standard HF model loader cannot parse `qwen3_5`.
    ckpt_dir = _resolve_ckpt_dir()
    state_dict = load_hf_state_dict(str(ckpt_dir))

    weight_cache_path = args.weight_cache_path(dtype)
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    tt_model = TtTransformer(
        args=args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        use_paged_kv_cache=True,
        mode="prefill",
    )
    return tt_model, args


class Qwen3_5ForConditionalGeneration(Generator):
    """Text-only vLLM serving class for Qwen3.6-27B. Name matches the HF arch
    so platform.py's `TT` prefix resolves to this class."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=262144,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        assert (
            tt_data_parallel == 1
        ), f"Qwen3.6 v2 galaxy is TP-only; data parallel > 1 unsupported, got tt_data_parallel={tt_data_parallel}"
        # Bake the known-good decode-CCL config as DEFAULTS so the SERVER gets the same coherent
        # path as the demo (test_qwen36_demo_generator_batch1) without per-deploy env wiring;
        # setdefault keeps any explicit override. Read during the model build + decode below.
        #   FORCE_SWITCH_DECODE/DECODE_L1_RESIDUAL : decode-mode tt_ccl tail + 32-row L1 residual norm
        #   LM_HEAD_PLAIN_DECODE                   : decode lm_head via minimal_matmul (coherence fix)
        #   SEQ_CORES_PER_HEAD / *_TUNED / CCL_NUM_LINKS_DELTA / RESIDUAL_BUF_BF16 : prefill+perf tuning
        for _k, _v in {
            "QWEN36_FORCE_SWITCH_DECODE": "1",
            "QWEN36_DECODE_L1_RESIDUAL": "1",
            "QWEN36_RESIDUAL_BUF_BF16": "1",
            "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
            "QWEN36_SEQ_CORES_PER_HEAD": "4",
            "QWEN36_FULLATTN_WO_TUNED": "1",
            "QWEN36_DELTA_OP_TUNED": "1",
            "QWEN36_CCL_NUM_LINKS_DELTA": "2",
        }.items():
            os.environ.setdefault(_k, _v)
        tt_model, model_args = initialize_vllm_text_transformer_qwen36(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
        )
        inst = cls(tt_model, model_args, mesh_device)
        # Prefill trace-capture hits a GDN/DeltaNet L1 circular-buffer clash
        # (V2-9 trace-capture blocker); run prefill eager via the Generator's
        # _disable_prefill_tracing hook. DECODE is traced (worker
        # override_tt_config.trace_mode=true) — the demo's proven decode path;
        # eager decode was never multi-step-verified and produced garbage past
        # token 1. _disable_decode_tracing is intentionally left unset.
        inst._disable_prefill_tracing = True
        # Skip the built-in prefill warmup: it is hardcoded for batch-32 (loops
        # batch in (1,32) + forces on-device sampling which asserts
        # max_batch_size % 32 == 0). For batch-1 serving we sample on host
        # (sample_on_device_mode=None), so warmup is unnecessary; the first
        # request prefills directly (eager).
        inst.prefill_warmup_completed = True
        return inst

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)


# =============================================================================
# Multimodal (image + video) vLLM serving class for Qwen3.6-27B VL.
#
# Mirrors the SERVER-PATH recipe validated in demo/mm_perf_qwen36.py:
#   PREFILL: on-device vision encoder -> host embed lookup + splice -> M-RoPE
#            cos/sin upload -> Generator.prefill_forward_text_embeds(inputs_embeds,
#            rot_mats) (populates paged KV + DeltaNet state, returns host logits).
#   DECODE : Generator.set_decode_rope_offset(...) decouples decode RoPE from the
#            KV index (vision tokens compress positions), then the SAME traced
#            on-device-sampling decode path as the text server.
#
# vLLM wiring (verified empirically against this plugin/runner):
#   * The served config resolves model_type=qwen3_5_vl -> Qwen3_6VLConfig (KEEPS
#     vision_config). The spec sets hf_overrides architectures to the NATIVE
#     Qwen3VLForConditionalGeneration so vLLM's capability check sees a native
#     multimodal generative class (multimodal_config gets set; the native
#     Qwen3VL HF processor expands image/video placeholders). platform.py then
#     prepends "TT" -> TTQwen3VLForConditionalGeneration, which the plugin maps
#     to THIS class for model instantiation.
#   * The processor is bound by the RESOLVED (TT-prefixed) model class at
#     model-load time, so this class carries the @register_processor decorator.
#   * The plugin runner gathers pixel_values + image_grid_thw (+ video fields)
#     into multi_modal_kwargs and passes them through to prefill_forward; this
#     class consumes them. prefill_forward returns a plain [1, 1, vocab] logits
#     tensor (NOT a (logits, rope_deltas) tuple — this runner indexes the
#     prefill output directly); decode rope state is carried model-internally.
# =============================================================================


try:
    from vllm.model_executor.models.interfaces import SupportsMultiModal as _SupportsMultiModal
except Exception:  # vLLM not importable (off-device tooling); keep module importable.

    class _SupportsMultiModal:  # type: ignore[no-redef]
        pass


def _try_register_vl_processor(cls):
    """Decorate `cls` with the native Qwen3VL multimodal processor, if vLLM is
    importable. Done lazily (not at module import) so that off-device / no-vllm
    imports of this module (e.g. the text path) keep working."""
    try:
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLDummyInputsBuilder,
            Qwen3VLMultiModalProcessor,
            Qwen3VLProcessingInfo,
        )
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except Exception:
        return cls

    class TT_Qwen36VLProcessingInfo(Qwen3VLProcessingInfo):
        def get_supported_mm_limits(self):
            # Multi-image + video enabled: the on-device splice
            # (splice_modalities_into_embeddings) handles multiple image-token
            # runs, the vision encoder is block-diagonal per image/frame
            # (cu_seqlens), and prefill_forward concatenates pixel_values across
            # images with image_grid_thw [num_images, 3] (native qwen3_vl
            # convention). Validated for up to 4 images + 1 video per request.
            return {"image": 4, "video": 1}

    return MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=TT_Qwen36VLProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )(cls)


class _Qwen3_6VLForConditionalGenerationBase(Generator, _SupportsMultiModal):
    """Multimodal serving class for Qwen3.6-27B VL (image + video).

    Built on top of the text Generator + the on-device vision encoder + the
    host embed-splice pipeline (validated in mm_perf_qwen36.py).
    """

    # Default placeholder token ids (qwen3.6 / Qwen3VL); refreshed from config.
    IMAGE_TOKEN_ID = 248056
    VIDEO_TOKEN_ID = 248057
    VISION_START_TOKEN_ID = 248053

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):
        # MUST match the native Qwen3VL placeholders so vLLM's chat path inserts
        # the exact tokens the Qwen3VL processor expands/replaces. Without this,
        # SupportsMultiModal's default placeholder is inserted and the processor
        # fails with "Failed to apply prompt replacement for mm_items[...]".
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *args, **kwargs):
        self._vision_encoder = kwargs.pop("vision_encoder", None)
        self._vision_args = kwargs.pop("vision_args", None)
        self._ccl_manager = kwargs.pop("ccl_manager", None)
        self._text_embed_weight = kwargs.pop("text_embed_weight", None)
        # M-RoPE params (mirrors mm_perf build_mrope_cos_sin call).
        self._mrope_partial_rotary_dim = kwargs.pop("mrope_partial_rotary_dim", 64)
        self._mrope_section = kwargs.pop("mrope_section", [11, 11, 10])
        self._mrope_theta = kwargs.pop("mrope_theta", 10_000_000.0)
        self._mrope_head_dim = kwargs.pop("mrope_head_dim", 256)
        self._spatial_merge_size = kwargs.pop("spatial_merge_size", 2)
        super().__init__(*args, **kwargs)

    # ----------------------------------------------------------------- init
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=262144,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
        vllm_config=None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        assert (
            tt_data_parallel == 1
        ), f"Qwen3.6 v2 galaxy is TP-only; data parallel > 1 unsupported, got tt_data_parallel={tt_data_parallel}"
        # Same correctness-critical decode-CCL defaults as the text server.
        for _k, _v in {
            "QWEN36_FORCE_SWITCH_DECODE": "1",
            "QWEN36_DECODE_L1_RESIDUAL": "1",
            "QWEN36_RESIDUAL_BUF_BF16": "1",
            "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
            "QWEN36_SEQ_CORES_PER_HEAD": "4",
            "QWEN36_FULLATTN_WO_TUNED": "1",
            "QWEN36_DELTA_OP_TUNED": "1",
            "QWEN36_CCL_NUM_LINKS_DELTA": "2",
        }.items():
            os.environ.setdefault(_k, _v)

        # --- Text decoder (load state_dict ONCE; reuse for embed weight) ---
        instruct = "instruct" in str(getattr(hf_config, "_name_or_path", "")).lower()
        args = TtQwen36ModelArgs(
            mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        if n_layers is not None:
            args.n_layers = n_layers
        ckpt_dir = _resolve_ckpt_dir()
        state_dict = load_hf_state_dict(str(ckpt_dir))
        weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
        weight_cache_path.mkdir(parents=True, exist_ok=True)
        tt_model = TtTransformer(
            args=args,
            dtype=ttnn.bfloat8_b,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            use_paged_kv_cache=True,
            mode="prefill",
        )

        # Host embedding table for the text tokens (CPU lookup -> splice).
        text_embed_weight = state_dict["model.language_model.embed_tokens.weight"].float()

        # --- Vision encoder (on-device) + CCL (mirror mm_perf) ---
        from models.demos.qwen3_6_galaxy_v2.tt.vision_encoder import Qwen36VisionEncoder
        from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
        from models.tt_dit.parallel.manager import CCLManager

        ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
        vision_args = Qwen36VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
        vision_encoder = Qwen36VisionEncoder(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            model_args=vision_args,
            dtype=ttnn.bfloat16,
        )

        # M-RoPE params from the served HF config (fall back to qwen3.6 defaults).
        tc = getattr(hf_config, "text_config", None) or hf_config
        rp = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None) or {}
        head_dim = int(getattr(tc, "head_dim", 256) or 256)
        prf = float(rp.get("partial_rotary_factor", 0.25)) if isinstance(rp, dict) else 0.25
        partial_rotary_dim = int(head_dim * prf)
        mrope_section = list(rp.get("mrope_section", [11, 11, 10])) if isinstance(rp, dict) else [11, 11, 10]
        rope_theta = float(rp.get("rope_theta", 10_000_000.0)) if isinstance(rp, dict) else 10_000_000.0
        vc = getattr(hf_config, "vision_config", None)
        spatial_merge_size = int(getattr(vc, "spatial_merge_size", 2) or 2) if vc is not None else 2

        inst = cls(
            tt_model,
            args,
            mesh_device,
            vision_encoder=vision_encoder,
            vision_args=vision_args,
            ccl_manager=ccl_manager,
            text_embed_weight=text_embed_weight,
            mrope_partial_rotary_dim=partial_rotary_dim,
            mrope_section=mrope_section,
            mrope_theta=rope_theta,
            mrope_head_dim=head_dim,
            spatial_merge_size=spatial_merge_size,
        )
        # Prefill runs eager (GDN trace-capture blocker); decode is traced.
        inst._disable_prefill_tracing = True
        inst.prefill_warmup_completed = True
        # Refresh placeholder ids from config if present.
        inst.IMAGE_TOKEN_ID = int(getattr(hf_config, "image_token_id", inst.IMAGE_TOKEN_ID) or inst.IMAGE_TOKEN_ID)
        inst.VIDEO_TOKEN_ID = int(getattr(hf_config, "video_token_id", inst.VIDEO_TOKEN_ID) or inst.VIDEO_TOKEN_ID)
        inst.VISION_START_TOKEN_ID = int(
            getattr(hf_config, "vision_start_token_id", inst.VISION_START_TOKEN_ID) or inst.VISION_START_TOKEN_ID
        )
        return inst

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)

    # ----------------------------------------------------- mm input helpers
    @staticmethod
    def _flatten_user_list(values):
        """vLLM/runner passes mm fields as a list (per request) of lists (per
        item). Flatten to a single list of per-item tensors, dropping None."""
        out = []
        if values is None:
            return out
        for per_user in values:
            if per_user is None:
                continue
            if isinstance(per_user, (list, tuple)):
                out.extend([v for v in per_user if v is not None])
            else:
                out.append(per_user)
        return out

    def _stack_grid(self, grid_items):
        """Stack per-image/video grid_thw tensors into [num, 3] int32."""
        norm = []
        for g in grid_items:
            t = torch.as_tensor(g)
            t = t.reshape(-1, 3) if t.numel() % 3 == 0 and t.ndim != 2 else t
            if t.ndim == 1:
                t = t.view(1, 3)
            norm.append(t.to(torch.int32))
        return torch.cat(norm, dim=0) if norm else None

    # ---------------------------------------------------------- prefill
    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,  # pre-padding real lengths after text+image processing
        enable_trace=False,
        **kwargs,  # pixel_values / image_grid_thw (+ video variants)
    ):
        from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin
        from models.demos.qwen3_6_galaxy_v2.tt.generator import get_padded_prefill_len
        from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_pipeline import splice_modalities_into_embeddings
        from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import get_rope_index

        assert tokens.shape[0] == 1, "Qwen3.6 VL serving is batch-1 (single user) per prefill"
        real_len = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[-1])
        input_ids = tokens[:, :real_len].to(torch.long)  # [1, real_len]

        # --- Gather + run vision (image and/or video) on device ---
        pixel_values = self._flatten_user_list(kwargs.get("pixel_values"))
        image_grids = self._flatten_user_list(kwargs.get("image_grid_thw"))
        video_pixels = self._flatten_user_list(kwargs.get("pixel_values_videos"))
        video_grids = self._flatten_user_list(kwargs.get("video_grid_thw"))

        image_grid_thw = self._stack_grid(image_grids) if image_grids else None
        video_grid_thw = self._stack_grid(video_grids) if video_grids else None

        image_features = None
        if pixel_values:
            pv = torch.cat([torch.as_tensor(p).reshape(-1, torch.as_tensor(p).shape[-1]) for p in pixel_values], dim=0)
            assert image_grid_thw is not None, "pixel_values present but no image_grid_thw"
            image_features = self._vision_encoder.forward(pv, image_grid_thw)  # torch [N_img_tok, H]
        video_features = None
        if video_pixels:
            vp = torch.cat([torch.as_tensor(p).reshape(-1, torch.as_tensor(p).shape[-1]) for p in video_pixels], dim=0)
            assert video_grid_thw is not None, "pixel_values_videos present but no video_grid_thw"
            video_features = self._vision_encoder.forward(vp, video_grid_thw)  # torch [N_vid_tok, H]

        # --- Host text embedding + splice (validated path; mm_perf parity) ---
        text_embeddings = torch.nn.functional.embedding(input_ids, self._text_embed_weight)  # [1, S, H]
        if image_features is not None or video_features is not None:
            fused = splice_modalities_into_embeddings(
                text_embeddings,
                input_ids,
                image_features=image_features,
                video_features=video_features,
                image_token_id=self.IMAGE_TOKEN_ID,
                video_token_id=self.VIDEO_TOKEN_ID,
            )
        else:
            fused = text_embeddings  # text-only request

        # --- 3D M-RoPE positions (host golden, trace-safe input) ---
        position_ids_3d, _deltas = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=self.IMAGE_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            vision_start_token_id=self.VISION_START_TOKEN_ID,
            spatial_merge_size=self._spatial_merge_size,
        )  # [3, 1, S]

        # --- Pad to prefill bucket (fused embeds + 3D positions) ---
        S_unpadded = fused.shape[1]
        S = get_padded_prefill_len(S_unpadded)
        if S > S_unpadded:
            pad = S - S_unpadded
            fused = torch.cat([fused, torch.zeros(*fused.shape[:-2], pad, fused.shape[-1], dtype=fused.dtype)], dim=-2)
            last = int(position_ids_3d[:, :, -1:].max().item())
            pad_pos = (
                torch.arange(last + 1, last + 1 + pad, dtype=position_ids_3d.dtype).view(1, 1, pad).expand(3, 1, pad)
            )
            position_ids_3d = torch.cat([position_ids_3d, pad_pos], dim=-1)
        # Padded token ids for page/last-token bookkeeping.
        ids_padded = (
            torch.cat([input_ids, torch.zeros(1, S - S_unpadded, dtype=torch.long)], dim=1)
            if S > S_unpadded
            else input_ids
        )

        # --- M-RoPE cos/sin (real 3D positions), replicated upload ---
        cos_ref, sin_ref = build_mrope_cos_sin(
            positions_3d=position_ids_3d[:, 0, :],
            head_dim=self._mrope_head_dim,
            partial_rotary_factor=self._mrope_partial_rotary_dim / self._mrope_head_dim,
            mrope_section=self._mrope_section,
            theta=self._mrope_theta,
        )

        def _upload(t):
            return ttnn.from_torch(
                t.unsqueeze(0),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        rot_mats = (_upload(cos_ref), _upload(sin_ref))

        # --- Server-path prefill ---
        prefill_logits = self.prefill_forward_text_embeds(
            ids_padded,
            inputs_embeds=fused,
            rot_mats=rot_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[S_unpadded],
        )

        # --- Decouple decode RoPE position from the KV index for coherence ---
        rope_pos_next = int(position_ids_3d[:, :, :S_unpadded].max().item()) + 1
        self.set_decode_rope_offset(rope_pos_next - S_unpadded)

        # Runner indexes prefill output as [batch, seq, vocab]; reshape 2D->3D.
        logits = torch.as_tensor(prefill_logits)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)  # [1, 1, vocab]
        return logits

    # ----------------------------------------------------------- decode
    def decode_forward(self, *args, **kwargs):
        # rope offset already set in prefill_forward (model-internal state).
        return super().decode_forward(*args, **kwargs)


# Bind the native Qwen3VL multimodal processor to the resolved TT class.
Qwen3_6VLForConditionalGeneration = _try_register_vl_processor(_Qwen3_6VLForConditionalGenerationBase)
