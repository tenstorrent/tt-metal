# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Qwen3.5/3.6 on Blackhole — a thin tt_transformers Generator subclass.

Hybrid model: 8 paged-KV attention + 24 GDN recurrent-state layers. GDN forbids token-padding and
isn't position-general, so prefill is model-owned (masked-bucket for short prompts / chunk-outer
trace for long, via prefill_dispatch) while Generator drives decode only. GDN + KV state is
model-bound, so the kv_cache contract param is accepted but unused.
"""

import dataclasses
import math
import os
from collections import defaultdict
from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch, warmup_decode_buckets
from models.demos.blackhole.qwen36.tt.model import serve_device_decode_env_enabled
from models.demos.blackhole.qwen36.tt.spec_engine import rows_to_engine, slot_map_after_remap, slot_map_inverse
from models.demos.blackhole.qwen36.tt.tp_common import (
    fabric_router_config,
    spec_chain_enabled,
    spec_device_logprobs_enabled,
    spec_device_mask_enabled,
    spec_device_penalties_enabled,
    spec_device_topp_enabled,
    spec_fused_enabled,
    spec_fused_sampled_enabled,
    spec_hostset_enabled,
    spec_prefill_overlap_enabled,
    verify_device_topk_m,
)
from models.tt_transformers.tt.generator import Generator

# Sampler top-k limit (models/common/sampling TTSampling.max_top_k default; ttnn.sampling walks k <= 32 candidates).
_MAX_DEVICE_TOP_K = 32
# Positional order of Generator.decode_forward's leading parameters (the plugin passes keywords; direct callers may not).
_DECODE_POSITIONAL = (
    "tokens",
    "start_pos",
    "page_table",
    "kv_cache",
    "enable_trace",
    "read_from_device",
    "sampling_params",
    "prompt_tokens",
    "output_tokens",
    "slot_remap",
    "defer_device_sampling",
)
# Host-authoritative defaults for direct callers (DECODE_RELOAD_CONTRACT.md "Command defaults"); the plugin always
# sends all four explicitly.
_RELOAD_COMMAND_DEFAULTS = {
    "reload_inputs": True,
    "reload_page_table": False,
    "reload_sampling_params": False,
    "reset_sampling_state": False,
}


def _spec_prefix_cache_enabled():
    """QWEN36_SPEC_PREFIX_CACHE (default 1): prefix caching on the MTP spec class; 0 = refuse it as before."""
    return os.environ.get("QWEN36_SPEC_PREFIX_CACHE", "1") == "1"


def _spec_chunked_prefill_enabled():
    """QWEN36_SPEC_CHUNKED_PREFILL=1 (default off) AND the spec GDN prefix cache on (the flags that decide whether
    _maybe_build_spec_prefix_cache builds model._prefix_cache): the MTP class accepts chunked prefill (chunk snapshots live in
    that cache, see spec_engine.prefill ``final``)."""
    return (
        os.environ.get("QWEN36_SPEC_CHUNKED_PREFILL", "0") == "1"
        and _spec_prefix_cache_enabled()
        and os.environ.get("QWEN36_PREFIX_CACHE", "1") == "1"
    )


def _mtp_chunked_capabilities():
    """model_capabilities keys of the MTP class that depend on QWEN36_SPEC_CHUNKED_PREFILL (read at import)."""
    on = _spec_chunked_prefill_enabled()
    return {"supports_chunked_prefill": on, "accepts_intermediate_prefill_mask": on}


def _ns_slotmap_enabled():
    """QWEN36_NS_SLOTMAP (default 1): the non-spec resident decode keeps a runner-slot -> device-row map on the host
    instead of moving GDN state on the device for every plugin slot_remap; 0 = the device remap of old."""
    return os.environ.get("QWEN36_NS_SLOTMAP", "1") != "0"


_NS_WIDEN_DTYPES = tuple(getattr(torch, n) for n in ("uint16", "uint32", "uint64") if hasattr(torch, n))


def _ns_widen(x):
    """(x as int64 if torch cannot index / index_put / cat its unsigned dtype, else x unchanged), original dtype."""
    return (x.to(torch.int64) if x.dtype in _NS_WIDEN_DTYPES else x), x.dtype


def _ns_scatter_rows(x, inv, n, fill):
    """Runner-ordered per-row field -> device row order (full width ``n`` = len(inv)). Tensors keep dtype and trailing
    dims, lists stay lists; a field shorter than ``n`` is padded with ``fill`` (tensors) / its last item (lists)."""
    if isinstance(x, torch.Tensor):
        if x.dim() == 0 or x.shape[0] not in range(1, n + 1):
            return x
        x, dt = _ns_widen(x)
        if x.shape[0] < n:
            pad = torch.full((n - x.shape[0],) + tuple(x.shape[1:]), fill, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        return rows_to_engine(x, inv).to(dt)
    if isinstance(x, (list, tuple)) and 0 < len(x) <= n:
        x = list(x) + [x[-1]] * (n - len(x))
        return [x[s] for s in inv]
    return x


def _ns_gather_rows(x, phys):
    """Device-ordered per-row output -> runner order: out[i] = x[phys[i]] (neutral zeros where phys[i] is beyond the
    rows the step produced). Always len(phys) rows; dtype and trailing dims kept."""
    if not isinstance(x, torch.Tensor) or x.dim() == 0:
        return x
    idx = torch.as_tensor(phys, dtype=torch.int64)
    keep = idx < x.shape[0]
    x, dt = _ns_widen(x)
    out = torch.zeros((len(phys),) + tuple(x.shape[1:]), dtype=x.dtype)
    out[keep] = x[idx[keep]]
    return out.to(dt)


def _live_rows(start_pos, n):
    """Rows holding a live request (vLLM pads idle decode rows with position -1), or None if unknown / disabled."""
    if os.environ.get("QWEN36_GDN_REMAP_LIVE_ONLY", "1") != "1" or start_pos is None:
        return None
    pos = start_pos.flatten().tolist() if hasattr(start_pos, "flatten") else list(start_pos)
    return [int(p) != -1 for p in pos] if len(pos) == n else None


def _plugin_candidates(cand):
    """The plugin's ``SpecCandidates`` (docs/SPEC_DECODE_CONTRACT.md) from the engine's ``DeviceCandidates``: the same three tensors
    (ids int32 [B, T, M], values fp32 [B, T, M], lse fp32 [B, T]). The plugin owns the type; this module imports it lazily.
    """
    from vllm_tt_plugin.spec_decode import SpecCandidates

    return SpecCandidates(ids=cand.ids, values=cand.values, lse=cand.lse)


def _spec_mask_capability():
    """Import-time part of the model capability supports_spec_device_mask (human-approved 2026-10-28): the fused sampled step, the model-side
    kill switch QWEN36_SPEC_DEVICE_MASK, and the traces the mask trace builds on (the stats trace and the certificate accept:
    QWEN36_SPEC_DEVICE_TOPP). The adapter then clears the key after warmup when the engine did not build the mask traces.
    """
    return bool(
        spec_fused_enabled()
        and spec_fused_sampled_enabled()
        and spec_device_mask_enabled()
        and spec_device_topp_enabled()
    )


def _build_model_capabilities():
    """The plugin-visible capabilities. QWEN36_SERVE_DEVICE_DECODE=0 returns exactly the pre-1.6S dict."""
    caps = {
        # Automatic prefix caching: vLLM reuses the attention KV blocks and passes start_pos (cached tokens, a multiple
        # of the block size 64); the model keeps its own GDN-state snapshots (tt/prefix_cache.py). QWEN36_PREFIX_CACHE=0
        # disables it. Run vLLM with --block-size 64 and --enable-prefix-caching.
        "supports_prefix_caching": os.environ.get("QWEN36_PREFIX_CACHE", "1") == "1",
        # vLLM chunked prefill: a prompt longer than the per-step token budget (--max-num-batched-tokens) is prefilled
        # over several engine steps, with decode steps in between. Batched TP resumes each chunk from a GDN snapshot
        # taken at the previous chunk's end (model._chunk_state_cache); the B=1 TP path recomputes from 0 (correct,
        # slower: keep --max-num-batched-tokens >= max_model_len at max_num_seqs 1). Opt-in (QWEN36_CHUNKED_PREFILL=1):
        # declared, the plugin keeps vLLM's chunked prefill on and sets disable_chunked_mm_input, which vLLM's
        # hybrid-model prefix caching (mamba_cache_mode 'align') rejects at startup, so it also needs
        # --no-enable-prefix-caching (README "Chunked prefill").
        "supports_chunked_prefill": os.environ.get("QWEN36_CHUNKED_PREFILL", "0") == "1",
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }
    if serve_device_decode_env_enabled():
        caps.update(
            supports_async_decode=True,
            # TTSampling.max_top_k: k outside [1, 32] on a sampled row is routed to the host sampler by the plugin.
            max_device_top_k=_MAX_DEVICE_TOP_K,
            # Penalty bookkeeping (tt_penalties.update_output_tokens) reshapes and deallocates the sampled-token
            # tensor, which is now the persistent decode token input; not validated on device for this model, so
            # penalty requests use the host sampler (which reloads every step).
            supports_device_penalties=False,
        )
    # Fabric packet payload (QWEN36_FABRIC_PAYLOAD_BYTES, default 8192): merged by the plugin into its
    # set_fabric_config kwargs; launch overrides of the fabric type still win.
    router = fabric_router_config()
    if router is not None:
        caps["fabric_config"] = {"router_config": router}
    return caps


_PREFILL_WARMUP_CHUNK = 2048
_PREFILL_WARMUP_BUCKET = 4096
_BLOCK_SIZE = 64


class TT_Qwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Serve a single visual item per request (B=1, max_concurrency=1). Image and video are both
        # supported, but only ONE modality per request: the model's vision splice keys off a single
        # placeholder token id (image_token_id XOR video_token_id), so a mixed image+video prompt
        # cannot be spliced correctly.
        return {"image": 1, "video": 1}


class TT_Qwen3_5SpecProcessingInfo(TT_Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # the spec engine prefills image / video requests too (vision rows spliced in the traced prefill); one item per
        # request, one modality (see TT_Qwen3_5ProcessingInfo)
        return {"image": 1, "video": 1}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36ForCausalLM(Generator, SupportsMultiModal):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    # Decode bucketing keeps several traces live and refreshes the selected
    # bucket's inputs before replay, so their I/O buffers may safely overlap.
    _tt_allow_decode_trace_buffer_reuse = True
    decode_input_update_contract = 1
    _ENABLE_MTP = False  # MTP contract subclass overrides

    # QWEN36_SERVE_DEVICE_DECODE=0: supports_async_decode=False -- the decode inputs (host-packed RoPE, [B,1] token
    # input) are reloaded every step, so there is no on-device token/position continuity, and replaying a stale
    # token would corrupt Qwen's non-idempotent GDN scan. supports_sample_on_device=True: on-device sampling is
    # decode-only.
    # QWEN36_SERVE_DEVICE_DECODE=1 (default): the serving decode is device-resident (token feedback in place, RoPE
    # lookup + position increment in-trace, see model.py / _decode_forward_resident), so async decode is safe:
    # every replay consumes the correct (token, position) exactly once (_decode_forward_resident docstring).
    model_capabilities = _build_model_capabilities()

    def _validate_device_sampling_request(self, requested):
        if not requested:
            return
        for model in self.model:
            if model.sampling is not None:
                continue
            mesh_shape = tuple(int(dim) for dim in model.mesh_device.shape)
            logits_per_device = math.ceil(model.args.vocab_size / model.num_devices)
            raise RuntimeError(
                "Qwen3.6 on-device sampling requires a certified TP topology (1x4 or 1x8) "
                f"with at most 65536 logits/device; got mesh={mesh_shape}, "
                f"vocab={model.args.vocab_size}, logits/device={logits_per_device}. "
                "Unset sample_on_device_mode for host sampling."
            )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """All-user KV capacity (the shared paged-KV token pool).

        QWEN36_MAX_TOKENS_ALL_USERS overrides it with a FIXED pool (set per device+model from the
        tt-inference-server spec's env_vars, mirroring GEMMA4_MAX_TOKENS_ALL_USERS). This decouples
        the pool from max_model_len × max_num_seqs so ONE config serves both a single long request
        (up to max_model_len) and a batch of shorter ones (sum of lengths ≤ pool) — e.g. 524288 =
        1×256K or 8×64K. Without the override, fall back to max_model_len × max_num_seqs (the old
        per-config product) so existing single-mode specs are unchanged."""
        override = os.environ.get("QWEN36_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # Weights dir: MODEL_WEIGHTS_DIR → HF_MODEL → hf_config._name_or_path; a hub id resolves to a local snapshot.
        name_or_path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
            from huggingface_hub import snapshot_download

            # When offline/CI, resolve from the local cache only so snapshot_download
            # reads the cached refs instead of reaching the HF API (refused by HF_HUB_OFFLINE=1).
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            name_or_path = snapshot_download(name_or_path, local_files_only=offline)
        args, model, _ = create_tt_model(
            mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            hf_model=name_or_path,
            # vLLM does not drive spec decode yet (vllm-tt-plugin#110): skip the MTP head, its weights and its KV cache.
            enable_mtp=cls._ENABLE_MTP,
        )
        # Attach the TT vision tower so prefill can splice image/video embeddings (multimodal path).
        # No-op cost for text-only requests; get_image_features / get_video_features are only invoked
        # when a request actually carries pixel_values / pixel_values_videos.
        model.init_vision_model()
        inst = cls([model], [args], mesh_device)
        # vLLM's --[no-]enable-prefix-caching, read from the config the worker publishes while loading the model; the model-side
        # GDN snapshot cache is only worth its ~1.2 GB/device (and the per-prefill snapshot saves) when vLLM will reuse prefixes.
        # Non-vLLM callers (tests, demos) have no current config and keep the default (enabled). The plugin builds the
        # model OUTSIDE vLLM's config context, so allocate_kv_cache (inside it) reads these again.
        inst._read_vllm_serving_flags()
        inst._vllm_mm_max_pixels = cls._read_vllm_max_pixels()
        return inst

    @staticmethod
    def _vllm_config_or_none():
        """vLLM's current config, or None outside a set_current_vllm_config() context: the plugin calls
        initialize_vllm_model outside one, and allocate_kv_cache inside one (WorkerWrapperBase.initialize_from_config).
        """
        try:
            from vllm.config import get_current_vllm_config_or_none

            vc = get_current_vllm_config_or_none()
        except Exception:
            return None
        return vc if vc is not None and getattr(vc, "model_config", None) is not None else None

    def _read_vllm_serving_flags(self):
        """(Re)read vLLM's prefix caching / chunked prefill settings; keeps the current values without a vLLM config."""
        vc = self._vllm_config_or_none()
        if vc is None and hasattr(self, "_vllm_prefix_caching"):
            return
        self._vllm_prefix_caching = True if vc is None else bool(vc.cache_config.enable_prefix_caching)
        self._vllm_chunked_prefill = self._read_vllm_chunked_prefill(vc)
        if vc is not None:
            logger.info(
                f"vLLM serving config: enable_prefix_caching={self._vllm_prefix_caching}, chunked prefill "
                f"(budget, long_prefill_threshold, align)={self._vllm_chunked_prefill}"
            )

    @staticmethod
    def _read_vllm_chunked_prefill(vc=None):
        """(per-step token budget, long_prefill_token_threshold, prefill_chunk_alignment) when vLLM can split a prompt
        across engine steps; False when it cannot (QWEN36_CHUNKED_PREFILL is not 1, chunked prefill off, or the budget
        covers max_model_len); (None, 0, 128) without a vLLM config (tests, demos: budget unknown)."""
        if os.environ.get("QWEN36_CHUNKED_PREFILL", "0") != "1":
            return False
        if vc is None:
            return (None, 0, 128)
        sc, max_len = vc.scheduler_config, int(vc.model_config.max_model_len)
        budget = int(getattr(sc, "max_num_scheduled_tokens", None) or sc.max_num_batched_tokens)
        thr = int(getattr(sc, "long_prefill_token_threshold", 0) or 0)
        if not sc.enable_chunked_prefill or (budget >= max_len and not 0 < thr < max_len):
            return False
        tt_cfg = (getattr(vc, "additional_config", None) or {}).get("tt", {}) or {}
        align = int(tt_cfg.get("prefill_chunk_alignment", 128))
        if align <= 0 or align % 128:
            logger.warning(
                f"prefill_chunk_alignment={align}: non-final prefill chunks may end off a 128 boundary, where the GDN "
                f"state cannot be resumed; those chunks are recomputed from an earlier snapshot (correct, slower)"
            )
        return (budget, thr, align)

    @staticmethod
    def _read_vllm_max_pixels():
        """max_pixels from vLLM's --mm-processor-kwargs (model_config.mm_processor_kwargs, mirrored on
        model_config.multimodal_config.mm_processor_kwargs), or None when unset / no current vLLM config."""
        try:
            from vllm.config import get_current_vllm_config

            mc = get_current_vllm_config().model_config
            for src in (getattr(mc, "multimodal_config", None), mc):
                kw = getattr(src, "mm_processor_kwargs", None) if src is not None else None
                if kw and kw.get("max_pixels") is not None:
                    return int(kw["max_pixels"])
        except Exception:
            pass
        return None

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs.

        batch_size = max_batch_size (vLLM's max_num_seqs, threaded through initialize_vllm_model):
        the paged KV blocks (kv_cache_shape) already cover all users, and this sizes the per-slot
        GDN recurrent/conv state [B,...] + the decode kv grid. B==1 is the single-sequence path."""
        self._read_vllm_serving_flags()  # the plugin allocates inside vLLM's config context (warmup reads the flags)
        batch_size = self.model[0].args.max_batch_size
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=batch_size)

    @staticmethod
    def _has_visual(kwargs, pixel_key, u=0):
        """True only when user `u` carries REAL visual data for this modality. vLLM attaches an empty pixel_values
        placeholder to text requests for a multimodal-registered model, so a plain ``is not None`` check
        misclassifies text as multimodal (key absent / empty list / user entry None or empty => text-only)."""
        v = kwargs.get(pixel_key)
        if v is None or len(v) <= u or v[u] is None:
            return False
        item = v[u]
        return not (isinstance(item, (list, tuple)) and len(item) == 0)

    @staticmethod
    def _gather_user_visual(kwargs, u, pixel_key, grid_key):
        """Pull user `u`'s patches + (t,h,w) grids for one modality out of the per-request vLLM kwargs.

        Returns (pixel_values, grid_thw) or None when the request carries nothing for this modality.
        Multiple items for the user arrive as lists; concat the patches and stack the grids (same
        shape get_image_features / get_video_features expect: [num_patches, patch_dim] + [N, 3]).
        """
        if not Qwen36ForCausalLM._has_visual(kwargs, pixel_key, u) or kwargs.get(grid_key) is None:
            return None
        pixel_values = kwargs[pixel_key][u]
        grid_thw = kwargs[grid_key][u]
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.concat(list(pixel_values), dim=0)
        if isinstance(grid_thw, (list, tuple)):
            grid_thw = torch.stack([g.to(dtype=torch.int32) for g in grid_thw], dim=0)
        return pixel_values, grid_thw

    def _compute_vision_tokens(self, model, kwargs, u=0):
        """Run the vision tower for user `u`'s request, if it carries images or video.

        Returns the packed embeddings (ttnn [num_vision_tokens, H]) for prefill to splice in, or None for a
        text-only request. Image and video share the vision tower; dispatching to get_video_features (vs
        get_image_features) tells the model to splice into video_token_id placeholders and build the video M-RoPE.
        A request carries at most one visual modality (see get_supported_mm_limits); video takes precedence.
        NOTE: the tower stashes MODEL-GLOBAL grids, so call this right before user u's prefill (never ahead of time
        for several users).
        """
        video = self._gather_user_visual(kwargs, u, "pixel_values_videos", "video_grid_thw")
        if video is not None:
            self._check_vision_bucket(video[1])
            return model.get_video_features(*video)

        image = self._gather_user_visual(kwargs, u, "pixel_values", "image_grid_thw")
        if image is not None:
            self._check_vision_bucket(image[1])
            return model.get_image_features(*image)

        return None

    @staticmethod
    def _vision_bucket(num_patches):
        """Padded patch bucket the tower runs on (same rule as tt/vision/model.py: ceil-to-2048 of count + 1 step)."""
        return (int(num_patches) // 2048 + 1) * 2048

    def _check_vision_bucket(self, grid_thw):
        """Refuse an item whose padded bucket was not compiled before trace capture (warmup_vision).

        Running the unwarmed tower after capture compiles programs that a later trace replay can clobber, which HANGS
        the device (seen: tt_all_reduce). Raising here still fails this request's step (the engine reports the error)
        but avoids the device hang. No check when warmup_vision never ran (tests / demos) or was disabled.
        """
        warmed = getattr(self, "_vision_warmed_max_bucket", None)
        if not warmed:
            return
        for g in grid_thw.reshape(-1, 3):
            n = int(g[0]) * int(g[1]) * int(g[2])
            if self._vision_bucket(n) > warmed:
                raise ValueError(
                    f"Vision input of {n} patches (grid {tuple(int(x) for x in g)}, bucket {self._vision_bucket(n)}) "
                    f"exceeds the largest vision bucket warmed before trace capture ({warmed} patches); running it "
                    f"would compile the tower after capture and hang the device. Set vLLM "
                    f"--mm-processor-kwargs '{{\"max_pixels\": N}}' (e.g. 1048576) so every accepted input fits a warmed "
                    f"bucket, or extend QWEN36_VISION_WARMUP_GRIDS."
                )

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        # Prefill replays other traces / rewrites slot state: any resident decode inputs are stale from here on, so
        # the next decode must be a commanded reload (_decode_forward_resident refuses a steady step).
        self._decode_bucket_last = None
        model = self.model[0]
        if model.num_devices > 1 and model.args.max_batch_size > 1:
            # Batched prefill into decode slots; multimodal users get their vision tokens lazily, per user.
            kwargs.pop("rope_deltas_all_users", None)  # per-slot M-RoPE deltas are model-owned (set at prefill)
            return self._prefill_forward_tp_batched(
                model,
                tokens,
                page_table,
                prompt_lens,
                kwargs.get("empty_slots"),
                start_pos=kwargs.get("start_pos"),
                visual_kwargs=kwargs,
            )
        # Prefix caching (start_pos > 0) is only exploited by the batched slot path above. The B=1 TP path and the
        # single-device path ignore start_pos and recompute the whole prompt from position 0: correct (the cached KV
        # blocks are rewritten with the same values), just slower. Chunked prefill makes that cost grow with every chunk.
        sp = kwargs.get("start_pos")
        if sp is not None and not getattr(self, "_warned_start_pos_recompute", False):
            if any(int(x) > 0 for x in (sp.tolist() if hasattr(sp, "tolist") else sp)):
                self._warned_start_pos_recompute = True
                logger.warning(
                    "prefill with start_pos > 0 at max_batch_size 1: recomputed from position 0. With chunked prefill "
                    "every chunk recomputes its whole prefix; keep --max-num-batched-tokens >= max_model_len here."
                )
        vision_tokens = self._compute_vision_tokens(model, kwargs)
        if model.num_devices > 1:
            return self._prefill_forward_tp(model, tokens, page_table, prompt_lens, vision_tokens=vision_tokens)
        seq_len = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        logger.info(f"Prefilling User 1 up to {seq_len} tokens")
        # Multimodal works WITH the captured trace here: prefill_dispatch routes to the traced
        # path, which splices the image/video rows via a fixed-shape ttnn.where over persistent
        # buffers (compiled at warmup, updated per request by copy_host_to_device — no request-time
        # compile).
        logits = prefill_dispatch(
            model,
            tokens,
            page_table,
            prompt_lens,
            use_trace=kwargs.get("enable_trace", False),
            vision_tokens=vision_tokens,
        )
        logits = ttnn.to_torch(logits)
        # The vLLM runner unpacks (logits, rope_deltas) because the HF config has mrope_section.
        # Zero deltas are returned for all modalities: the multimodal M-RoPE delta is applied
        # entirely model-side (build_request_rope stashes self.rope.rope_delta during prefill, and
        # every decode path offsets the rope position by it), so the value handed back to vLLM is
        # unused for device-side rope and stays zero.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        logger.info(f"Finished prefill up to {seq_len} tokens, starting decode...")
        return logits, rope_deltas

    def _prefill_forward_tp(self, model, tokens, page_table, prompt_lens, vision_tokens=None):
        """TP (B=1) paged prefill via the model-owned masked fixed-bucket path.

        prefill_traced_chunked rounds the prompt up to a fixed bucket and masks the GDN to the
        EXACT valid_len, so prefill runs one of a bounded, pre-warmed program set (the
        compile-clobbers-trace fix) — for <=2048 prompts it is entirely the masked bucket (no
        chunk trace needed). Longer prompts replay the chunk-outer trace (Milestone B). Returns
        host logits [1, 1, vocab] gathered to a single replica."""
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        if tokens.shape[1] > T:
            tokens = tokens[:, :T]
        logger.info(f"Prefilling User 1 up to {T} tokens (TP masked-bucket/chunked)")
        # Multimodal is supported on TP too: prefill_traced_chunked splices the image/video rows via
        # a fixed-shape ttnn.where over hidden-sharded persistent buffers (the vision rows are
        # gathered to full hidden on host, placed along seq, then re-sharded), so no request-time
        # compile clobbers the parked trace.
        # Host logits [1,1,vocab]: one trace replay for any T < 2048 whose bucket was captured at warmup
        # (QWEN36_PREFILL_BUCKET_TRACE=1; one D2H of the pre-gather vocab shards), else the eager masked-bucket /
        # chunk-trace path (replicated device logits, read back one replica).
        logits = model.prefill_traced_chunked(
            tokens, page_table, actual_len=T, vision_tokens=vision_tokens, return_host_logits=True
        )
        logger.info(f"Finished prefill up to {T} tokens, starting decode...")
        return logits, torch.zeros(1, dtype=torch.long)

    def _prefill_forward_tp_batched(
        self, model, tokens, page_table, prompt_lens, empty_slots, start_pos=None, visual_kwargs=None
    ):
        """TP batched (max_num_seqs>1) prefill: prefill each request in this step into its decode slot.

        vLLM prefills new requests while other slots decode, so each user's B=1 state is written into
        row empty_slots[u] of the batched GDN buffers without disturbing the live rows (model-owned,
        via prefill_paged_slots). Attention fills each request's blocks via its page-table row.

        tokens:      torch [N, max_T] (rows are the N requests scheduled this prefill step).
        page_table:  torch [N, max_blocks] — row u = request u's blocks.
        prompt_lens: per-request real lengths (row u trimmed to prompt_lens[u]).
        empty_slots: per-request decode slot; defaults to range(N) (mirrors Generator.prefill_forward_text).
        start_pos:   optional per-request number of prefix-cached tokens (vLLM APC; rows of tokens / page_table are
                     still the FULL prompt / page-table row). Resumed from the model's GDN snapshots when available.
        visual_kwargs: the vLLM kwargs carrying per-request pixel_values[u] / image_grid_thw[u] (and the *_videos
                     variants); a user with real visual data is prefilled multimodally (prefix cache bypassed).
        Returns ([N, 1, vocab] host logits, [N] zero rope_deltas — the M-RoPE delta is kept model-side per slot).
        """
        N = tokens.shape[0]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [tokens.shape[1]] * N
        if empty_slots is None:
            empty_slots = list(range(N))
        empty_slots = [int(s) for s in empty_slots]
        ns_commit = None
        if _ns_slotmap_enabled() and self._serve_device_decode_active():  # the map lives in the resident decode only
            empty_slots, ns_commit = self._ns_prefill_rows(model, empty_slots)
        token_ids_list = [tokens[u : u + 1, : plens[u]].to(torch.int32) for u in range(N)]
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        logger.info(f"Prefilling {N} user(s) into slots {empty_slots} (TP batched masked-bucket)")
        if start_pos is not None:
            start_pos = [int(x) for x in (start_pos.tolist() if hasattr(start_pos, "tolist") else start_pos)]
            assert len(start_pos) == N, "one start_pos per request"
        vision_list = None
        if visual_kwargs is not None:
            vl = []
            for u in range(N):
                has_mm = any(self._has_visual(visual_kwargs, k, u) for k in ("pixel_values", "pixel_values_videos"))
                vl.append((lambda u=u: self._compute_vision_tokens(model, visual_kwargs, u)) if has_mm else None)
            if any(v is not None for v in vl):
                vision_list = vl
        host_logits = model.prefill_paged_slots(
            token_ids_list, pt, empty_slots, valid_lens=plens, start_positions=start_pos, vision_tokens_list=vision_list
        )
        if ns_commit is not None:
            self._ns_phys, self._ns_owned = ns_commit
        logits = torch.cat([hl.reshape(1, 1, -1) for hl in host_logits], dim=0)  # [N, 1, vocab]
        logger.info(f"Finished batched prefill of {N} user(s), starting decode...")
        return logits, torch.zeros(N, dtype=torch.long)

    # Host slot map (QWEN36_NS_SLOTMAP). ``_ns_phys[runner slot] = device row`` is a permutation of range(W); a request's
    # GDN / rope-delta state is written to its device row at prefill and never moves. ``_ns_owned`` = device rows held by
    # live requests (prefill adds, release_request / a reload decode step rewrite it). Created lazily: the class is also
    # built without __init__ side effects (tests, MTP subclass) and reset at the end of warmup_model_decode.
    def _ns_reset(self):
        W = int(self.model[0].args.max_batch_size)
        self._ns_phys = list(range(W))
        self._ns_owned = set()

    def _ns_ensure(self):
        W = int(self.model[0].args.max_batch_size)
        if getattr(self, "_ns_phys", None) is None or len(self._ns_phys) != W:
            self._ns_reset()
        return W

    def _ns_prefill_rows(self, model, slots):
        """Runner slots -> device rows for a prefill: each takes the lowest free device row, swapping the two free
        runner slots' map entries (both are free, so no live request moves). Returns (rows, (new_phys, new_owned)); the
        state is committed by the caller after the prefill succeeded."""
        self._ns_ensure()
        phys, owned = list(self._ns_phys), set(self._ns_owned)
        rows = []
        # The plugin only gives free runner slots to a prefill, so phys[s] is never a live device row and is always a safe
        # choice. _ns_owned can hold stale rows (a request that ended without release_request): that may only cost
        # compactness (a higher row, a bigger bucket), never an error.
        given = set()
        for s in slots:
            candidates = {d for d in range(len(phys)) if d not in owned and d not in given}
            d = min(candidates | {phys[s]})
            if d != phys[s]:
                k = phys.index(d)
                phys[s], phys[k] = phys[k], phys[s]
            given.add(d)
            owned.add(d)
            rows.append(d)
        return rows, (phys, owned)

    def release_request(self, slot):
        """Plugin hook (optional, called with the request's current runner slot when it ends): its device row is free."""
        if _ns_slotmap_enabled():
            self._ns_ensure()
            self._ns_owned.discard(self._ns_phys[int(slot)])

    def process_decode_output_host(self, tt_out, is_tokens=False, sample_rows=None):
        out = super().process_decode_output_host(tt_out, is_tokens=is_tokens, sample_rows=sample_rows)
        phys = getattr(self, "_ns_phys", None)
        if not _ns_slotmap_enabled() or phys is None or phys == list(range(len(phys))):
            return out
        # Safe under a changing map: the plugin drains every in-flight decode output before any step that changes the
        # map (layout change or prefill), so each output is processed under the map it was produced with.
        if sample_rows is not None:
            raise ValueError("compact host rows are not supported with a non-identity host slot map")
        if isinstance(out, tuple):
            first, second = out
            if isinstance(second, tuple):
                second = tuple(_ns_gather_rows(t, phys) for t in second)
            else:
                second = _ns_gather_rows(second, phys)
            return _ns_gather_rows(first, phys), second
        return _ns_gather_rows(out, phys)

    def _serve_device_decode_active(self):
        """Device-resident serving decode: env flag on AND the model built the resident inputs (TP mesh)."""
        return serve_device_decode_env_enabled() and bool(getattr(self.model[0], "_serve_device_decode", False))

    @staticmethod
    def _pick_decode_bucket(tokens, start_pos):
        """Smallest power-of-2 width >= the active prefix [0:num_active) (rows are front-packed), capped at the
        padded width. Same rule on every step, remap or not (tests/test_decode_bucketing.py::_pick_bucket)."""
        width = int(tokens.shape[0])
        if os.environ.get("TT_DECODE_BUCKETING", "1") != "1":
            return width
        num_active = int((start_pos != -1).sum()) if start_pos is not None else width
        num_active = max(1, min(num_active, width))
        return min(width, 1 << max(0, (num_active - 1).bit_length()))

    def _bind_bucket_store(self, B):
        """Select the per-width trace metadata / inputs / output (+ the sampling trace namespace) of bucket B."""
        store = getattr(self, "_bucket_trace_store", None)
        if store is None:
            store = self._bucket_trace_store = {}
        if B not in store:
            store[B] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None))
        self.trace_ids_decode, self.trace_inputs_decode, self.trace_output_decode = store[B]
        # Key the sampling trace by bucket width too: Generator binds it to one logits tensor
        # by identity, and each decode-bucket width has its own.
        for _m in self.model:
            _sm = getattr(_m, "sampling", None)
            if _sm is not None and hasattr(_sm, "set_trace_bucket"):
                _sm.set_trace_bucket(B)

    def decode_forward(self, *args, **kwargs):
        if self._serve_device_decode_active():
            return self._decode_forward_resident(*args, **kwargs)
        return self._decode_forward_legacy(*args, **kwargs)

    def _decode_forward_resident(self, *args, **kwargs):
        """QWEN36_SERVE_DEVICE_DECODE=1 decode: honors the four decode_input_update_contract=1 commands exactly.

        reload_inputs         -> Generator copies token / position / RoPE-index / page-table host inputs into the
                                 bucket's persistent trace inputs.
        reload_page_table     -> Generator copies ONLY the page table (token / cur_pos / rope_idx stay resident).
        reload_sampling_params / reset_sampling_state -> forwarded to the Generator's sampler (apply_decode_state,
                                 seed alignment); nothing here infers extra reloads or skips commanded ones.

        Why each async replay consumes the correct (token, position) exactly once, so the GDN scan (non-idempotent:
        a stale / duplicated token permanently corrupts the recurrent + conv state) stays exact:
          * token: the sampler writes the sampled token IN PLACE into trace_inputs_decode[...][0] (the [1,1,1,32]
            buffer the next replay embeds; _tt_supports_decode_token_feedback), so replay k+1 reads t_k from device;
          * position: each device-sampling replay advances cur_pos / rope_idx exactly once, in-trace, after their
            readers; sampling and readback never advance it; idle rows (-1) stay -1;
          * no extra replay: serving issues exactly one model replay + one sampling replay per accepted decode (the
            warmup replays happen before the first request and are followed by a reload on the first decode);
          * idle / padding rows are don't-care: GDN state is per slot and a slot is re-initialised by prefill before
            it is read again; their cur_pos stays -1 so no valid KV position is written;
          * slot_remap is applied exactly once: GDN state here (before the replay reads it), the sampler's seed /
            penalty state inside Generator.decode_forward; every remap implies a layout change, hence a reload;
          * every transition (first decode, layout change, bucket switch, prefill, sampling-mode change) is a
            plugin-commanded reload_inputs; a bucket switch without one is refused (stale resident inputs).
        All argument validation happens BEFORE any state mutation (the GDN remap is not idempotent, and the plugin
        advances its slot map only after an accepted call).
        """
        if len(args) > len(_DECODE_POSITIONAL):
            raise TypeError(f"decode_forward takes at most {len(_DECODE_POSITIONAL)} positional arguments")
        for name, value in zip(_DECODE_POSITIONAL, args):
            if name in kwargs:
                raise TypeError(f"decode_forward got multiple values for argument '{name}'")
            kwargs[name] = value
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires explicit reload commands; reset_batch is legacy")
        for name, default in _RELOAD_COMMAND_DEFAULTS.items():
            kwargs.setdefault(name, default)
            if not isinstance(kwargs[name], bool):
                raise TypeError(f"{name} must be a bool, got {type(kwargs[name]).__name__}")
        reload_inputs = kwargs["reload_inputs"]
        if reload_inputs and kwargs["reload_page_table"]:
            raise ValueError("reload_page_table must be false when reload_inputs is true")
        if kwargs["reset_sampling_state"] and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions (reload_inputs=True)")
        device_sampling = kwargs.get("sampling_params") is not None or bool(kwargs.get("defer_device_sampling", False))
        if not device_sampling and not reload_inputs:
            raise ValueError("Host sampling requires authoritative token and position inputs (reload_inputs=True)")
        slot_remap = kwargs.get("slot_remap")
        if slot_remap is not None and not reload_inputs:
            raise ValueError(
                "slot_remap moves slot-indexed state; the resident token / position rows are in the old slot order, "
                "so a remap requires reload_inputs=True"
            )
        tokens = kwargs.get("tokens")
        if tokens is None:
            raise TypeError("decode_forward requires tokens")
        start_pos = kwargs.get("start_pos")
        ns_phys = None
        if _ns_slotmap_enabled():
            # Host slot map: no device remap. Scatter every per-row input to device-row order (the plugin's runner
            # order is only a view), bucket by the highest live device row, forward slot_remap=None. State is
            # set just before the Generator call (restored if it raises).
            W = self._ns_ensure()
            if slot_remap is not None:
                remap = [int(j) for j in slot_remap]
                if len(remap) != W or sorted(remap) != list(range(W)):
                    raise ValueError(f"slot_remap must be a permutation of range({W}), got {remap}")
                ns_phys = slot_map_after_remap(self._ns_phys, remap)
            else:
                ns_phys = list(self._ns_phys)
            if ns_phys != list(range(W)):
                inv = slot_map_inverse(ns_phys)
                tokens = kwargs["tokens"] = _ns_scatter_rows(tokens, inv, W, 0)
                if start_pos is not None:
                    start_pos = kwargs["start_pos"] = _ns_scatter_rows(start_pos, inv, W, -1)
                if kwargs.get("page_table") is not None:
                    kwargs["page_table"] = _ns_scatter_rows(kwargs["page_table"], inv, W, 0)
                for name, fill in (("prompt_tokens", 0), ("output_tokens", 0), ("grammar_bitmask", -1)):
                    if kwargs.get(name) is not None:
                        kwargs[name] = _ns_scatter_rows(kwargs[name], inv, W, fill)
                sp = kwargs.get("sampling_params")
                if sp is not None and dataclasses.is_dataclass(sp):
                    kwargs["sampling_params"] = dataclasses.replace(
                        sp, **{f.name: _ns_scatter_rows(getattr(sp, f.name), inv, W, 0) for f in dataclasses.fields(sp)}
                    )
            kwargs["slot_remap"] = None
            slot_remap = None
        width = int(tokens.shape[0])
        # Bucket by the active prefix with ONE rule on every step (a remap step no longer forces full width: the
        # remap is applied to the full slot space below and rows are front-packed, so slicing afterwards is exact).
        if ns_phys is not None and os.environ.get("TT_DECODE_BUCKETING", "1") == "1":
            live = [] if start_pos is None else (start_pos.flatten() != -1).nonzero().flatten().tolist()
            bucket = min(width, 1 << max(0, max(live, default=0).bit_length()))  # next pow2 >= max live row + 1
        else:
            bucket = self._pick_decode_bucket(tokens, start_pos)
        previous = getattr(self, "_decode_bucket_last", None)
        if not reload_inputs and bucket != previous:
            raise ValueError(
                f"decode bucket changed ({previous} -> {bucket}) with reload_inputs=False: the new bucket's resident "
                "inputs are stale (replaying them would feed a stale token to the GDN scan); the layout change must "
                "carry reload_inputs=True"
            )

        model = self.model[0]
        # Batched fused GDN decode: make the conv state of this width class valid FIRST (in-place sync when the width
        # class changed since the last call; no-op otherwise / when the fused path is off). After this the conv tag is
        # the single needed format, so the remap below gathers only that one (both ops are row-wise: order-free).
        if hasattr(model, "prepare_gdn_decode_width"):
            model.prepare_gdn_decode_width(bucket)
        # Batched serving: apply vLLM's condense slot_remap to the per-slot GDN recurrent/conv state BEFORE the
        # decode trace reads it (the plugin remaps its own buffers; the sampler's seed/penalty state is remapped by
        # Generator.decode_forward; GDN state is model-internal, so mirror the reindex here). Exactly once.
        if slot_remap is not None and model.num_devices > 1 and model.args.max_batch_size > 1:
            live = _live_rows(start_pos, model.args.max_batch_size)
            model._remap_gdn_slots(slot_remap, live=live)
            model.remap_slot_rope_delta(slot_remap)

        if bucket < width:
            kwargs["tokens"] = tokens[:bucket]
            if start_pos is not None:
                kwargs["start_pos"] = start_pos[:bucket]
            if kwargs.get("page_table") is not None:
                kwargs["page_table"] = kwargs["page_table"][:bucket]
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen, device-resident serving decode)")
        self._bind_bucket_store(bucket)
        # The plugin commits its own map only after decode_forward returns, but Generator.decode_forward with
        # read_from_device=True gathers the output (process_decode_output_host -> _ns_gather_rows) INSIDE the call, so the
        # NEW map must be live before it. A raise must leave our map unchanged: restore it and re-raise.
        old_ns = (getattr(self, "_ns_phys", None), getattr(self, "_ns_owned", None))
        if ns_phys is not None:
            self._ns_phys = ns_phys
            if reload_inputs:  # start_pos is authoritative on a reload step
                self._ns_owned = (
                    set() if start_pos is None else set((start_pos.flatten() != -1).nonzero().flatten().tolist())
                )
        try:
            result = super().decode_forward(**kwargs)
        except BaseException:
            if ns_phys is not None:
                self._ns_phys, self._ns_owned = old_ns
            raise
        self._decode_bucket_last = bucket
        return result

    def _decode_forward_legacy(self, *args, **kwargs):
        args = list(args)

        def _read(name, pos):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if pos < len(args) else None

        def _write(name, pos, val):
            if name in kwargs:
                kwargs[name] = val
            elif pos < len(args):
                args[pos] = val

        # Traced decode (single-device and TP): trace captured at pos 0 in warmup, replayed here.
        # Valid for TP — GDN state is in fixed in-place buffers, and prefill only replays pre-warmed programs.
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen)")
        model = self.model[0]
        # Batched serving: apply vLLM's condense slot_remap to the per-slot GDN recurrent/conv state
        # BEFORE the decode trace reads it. The plugin remaps its own buffers (and the seed RNG via
        # super().decode_forward), but GDN state is model-internal, so mirror the same reindex here.
        # slot_remap is passed through unchanged so the seed-RNG remap inside super() still runs.
        # The remap is deferred until after prepare_gdn_decode_width(bucket) (below) so it gathers only the single
        # conv format that tag leaves valid; both ops are row-wise, so the order does not change the result.
        gdn_remap = None
        if model.num_devices > 1 and model.args.max_batch_size > 1:
            gdn_remap = _read("slot_remap", 9)
            gdn_live = _live_rows(_read("start_pos", 1), model.args.max_batch_size) if gdn_remap is not None else None
        # Decode bucketing (default on; TT_DECODE_BUCKETING=0 off): slice host inputs to the
        # smallest power-of-2 width >= active prefix [0:num_active) before the base forward.
        # No runner edit / output re-pad — plugin reads unpadded_batch_size in slot order.
        # Each width keeps its own trace metadata, inputs, output, and pre-capture staged inputs.
        tokens = _read("tokens", 0)
        if os.environ.get("TT_DECODE_BUCKETING", "1") == "1" and tokens is not None:
            start_pos = _read("start_pos", 1)
            width = int(tokens.shape[0])
            num_active = int((start_pos != -1).sum()) if start_pos is not None else width
            num_active = max(1, min(num_active, width))
            bucket = min(width, 1 << max(0, (num_active - 1).bit_length()))  # smallest pow2 >= num_active
            # Keep full width when slot_remap is set: remap indexes the full slot space (tokens /
            # GDN). Rare (row moves only); bucketing resumes next step. Check kw + positional #9.
            if _read("slot_remap", 9) is not None:
                bucket = width
            if bucket < width:
                _write("tokens", 0, tokens[:bucket])
                if start_pos is not None:
                    _write("start_pos", 1, start_pos[:bucket])
                page_table = _read("page_table", 2)
                if page_table is not None:
                    _write("page_table", 2, page_table[:bucket])
                tokens = tokens[:bucket]

        if tokens is not None:
            B = int(tokens.shape[0])
            store = getattr(self, "_bucket_trace_store", None)
            if store is None:
                store = self._bucket_trace_store = {}
            if B not in store:
                store[B] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None), {})
            (
                self.trace_ids_decode,
                self.trace_inputs_decode,
                self.trace_output_decode,
                self._prepared_decode_traces,
            ) = store[B]
            # Key the sampling trace by bucket width too: Generator binds it to one logits tensor
            # by identity, and each decode-bucket width has its own.
            for _m in self.model:
                _sm = getattr(_m, "sampling", None)
                if _sm is not None and hasattr(_sm, "set_trace_bucket"):
                    _sm.set_trace_bucket(B)
            # Batched fused GDN decode: make the conv state of this width class valid (in-place sync when the width
            # class changed since the last call; no-op otherwise / when the fused path is off), BEFORE the remap so
            # the remap gathers only the one valid format.
            if hasattr(model, "prepare_gdn_decode_width"):
                model.prepare_gdn_decode_width(B)
        if gdn_remap is not None:
            model._remap_gdn_slots(gdn_remap, live=gdn_live)
            model.remap_slot_rope_delta(gdn_remap)
        return super().decode_forward(*args, **kwargs)

    # Vision-tower warmup grids (t x h x w patches, h and w even). The tower is shape-stable (get_*_features run every
    # device op on the PADDED bucket, ceil-to-2048 of the patch count + 1 step: 1024 -> 2048, 3072 -> 4096,
    # 4096 -> 6144, 6400 -> 8192 ...; the exact request size only trims rows on host), so one grid per bucket compiles
    # everything a request of that bucket will run. A program first dispatched AFTER the prefill / decode traces were
    # captured can have its binary clobbered by a later trace replay, which hangs the next vision call -- hence this
    # runs in the eager warmup phase. Buckets not listed still work, but compile at request time (hang risk).
    # Override: QWEN36_VISION_WARMUP_GRIDS="1x32x32;1x80x80" (empty string disables).
    _VISION_WARMUP_GRIDS = "1x32x32;1x48x64;1x64x64;1x80x80"

    def warmup_vision(self, grids=None):
        """Compile + run the vision tower once per grid (image and video dispatch) BEFORE any trace is captured."""
        model = self.model[0]
        if getattr(model, "vision_model", None) is None:
            return
        vc = model.vision_args.hf_config.vision_config
        if grids is None:
            spec = os.environ.get("QWEN36_VISION_WARMUP_GRIDS", self._VISION_WARMUP_GRIDS)
            grids = [tuple(int(x) for x in g.split("x")) for g in spec.split(";") if g.strip()]
            max_pixels = getattr(self, "_vllm_mm_max_pixels", None)
            if max_pixels is None:
                if grids:
                    logger.warning(
                        "vLLM --mm-processor-kwargs max_pixels is not set: image/video inputs needing a vision bucket "
                        f"above the warmed maximum ({max(self._vision_bucket(t * h * w) for t, h, w in grids)} patches) "
                        "will be REJECTED (an unwarmed bucket would hang the device). Set "
                        "--mm-processor-kwargs '{\"max_pixels\": 1048576}' (or your serving limit)."
                    )
            else:
                # Patches of any accepted item: image <= max_pixels/patch^2; a video (frames x H x W <= max_pixels,
                # t = frames/temporal_patch_size) is <= max_pixels/(patch^2 * temporal_patch_size); the image bound
                # covers both (conservative).
                need = -(-int(max_pixels) // vc.patch_size**2)
                need_bucket = self._vision_bucket(need)
                if not any(self._vision_bucket(t * h * w) == need_bucket for t, h, w in grids):
                    h = 2 * math.ceil(math.sqrt(need) / 2)
                    w = 2 * math.ceil(need / h / 2)
                    while self._vision_bucket(h * w) > need_bucket and w > 2:  # keep the grid inside the needed bucket
                        w -= 2
                    grids.append((1, h, w))
                    logger.info(
                        f"max_pixels={max_pixels}: up to {need} patches -> also warming bucket {need_bucket} "
                        f"with grid 1x{h}x{w}"
                    )
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        for t, h, w in grids:
            pix = torch.zeros(t * h * w, patch_dim, dtype=torch.float32)
            grid = torch.tensor([[t, h, w]], dtype=torch.int32)
            ttnn.deallocate(model.get_image_features(pix, grid))  # image / video share the device work
        self._vision_warmed_max_bucket = max((self._vision_bucket(t * h * w) for t, h, w in grids), default=None)
        model._req_image_grid_thw = None
        model._req_video_grid_thw = None
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"Vision tower warmed for grids {grids}")

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # The eager call (enable_trace=False) allocates and compiles everything; the traced call only
        # records, so nothing persistent is allocated once a trace is live (#56474).
        # Guard name must match the plugin's reset.
        self._decode_bucket_last = None
        model = self.model[0]
        if enable_trace and model._chunked_trace_id is not None:
            return
        # Size the chunk-trace page table to the full KV cache (not a hardcoded 4096) so served ISL
        # isn't capped; still captures one chunk — just a bigger page-table tensor.
        if kv_cache:
            # Round to a multiple of 32: paged/chunked SDPA needs the page-table stick % 32 == 0.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        # Batched serving (max_num_seqs>1): the decode buffers are [B,...], but prefill runs B=1. Bind
        # the PERSISTENT B=1 GDN prefill scratch and capture the chunk trace against IT, so long prompts
        # (>chunk_size) replay the traced chunk-outer path per user instead of the slower eager fallback.
        # The scratch is not freed (prefill_paged_slots rebinds it per request); the batched decode
        # buffers are restored before the decode-trace warmup captures at [B,...].
        batched = model.num_devices > 1 and model.args.max_batch_size > 1
        if not enable_trace:
            self.warmup_vision()  # eager phase: before any trace exists
        logger.info(
            f"Qwen prefill warmup ({'record' if enable_trace else 'prepare'}): chunk={_PREFILL_WARMUP_CHUNK}, "
            f"page_table_blocks={num_blocks}{', batched B=1 scratch' if batched else ''}"
        )
        env_pc = os.environ.get("QWEN36_PREFIX_CACHE", "1") == "1"
        vllm_pc = getattr(self, "_vllm_prefix_caching", True)
        if batched and model._prefix_cache is None and model.short_prefill_trace_enabled():
            logger.info(
                f"GDN prefix-state cache {'enabled' if env_pc and vllm_pc else 'disabled'} "
                f"(QWEN36_PREFIX_CACHE={'1' if env_pc else '0'}, vLLM enable_prefix_caching={vllm_pc})"
            )
        if batched and model._prefix_cache is None and env_pc and vllm_pc and model.short_prefill_trace_enabled():
            # GDN prefix-state cache (APC): allocates the snapshot slots (+ the B=1 scratch they mirror) NOW, before any
            # prefill/decode trace is captured. The batched slot path is the only one that resumes from snapshots.
            from models.demos.blackhole.qwen36.tt.prefix_cache import GdnPrefixStateCache

            model._prefix_cache = GdnPrefixStateCache(model)
        chunked = getattr(self, "_vllm_chunked_prefill", (None, 0, 128))
        if batched and chunked and model.short_prefill_trace_enabled() and model._chunk_state_cache is None:
            # Chunked-prefill continuation store: one GDN snapshot per in-flight prompt chunk end (vLLM schedules the
            # running partial prefill first, so normally one at a time). Allocated NOW, before any trace is captured.
            from models.demos.blackhole.qwen36.tt.prefix_cache import GdnPrefixStateCache

            n = int(os.environ.get("QWEN36_CHUNK_STATE_SLOTS", "4"))
            model._chunk_state_cache = GdnPrefixStateCache(model, num_slots=n, name="chunked-prefill continuation")
            model._chunk_budget = chunked
            logger.info(
                f"chunked prefill: continuation store {n} slots, (budget, long_prefill_threshold, align)={chunked}"
            )
        prev = model._bind_gdn_prefill_scratch() if batched else None
        try:
            model.prepare_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)
            if enable_trace:
                model.record_prefill_trace_chunked(self.mesh_device)
                # Traced short prefill for ANY prompt length < 2048 (QWEN36_PREFILL_BUCKET_TRACE, default 1): one B=1 trace
                # per bucket (QWEN36_PREFILL_TRACE_BUCKETS, default 128,256,512,1024,2048; 2048 serves lengths 1025..2047,
                # exactly 2048 tokens keeps the chunk trace). Captured HERE, with the SAME GDN binding the requests run
                # with (the scratch when batched, the decode buffers at max_batch_size==1), and only because this call
                # has enable_trace=True (the plugin's trace_mode == "all"). Block 0 is vLLM's null block, never
                # allocated to a request: padded K/V writes of a bucket land there.
                if model.short_prefill_trace_enabled() and model.num_devices > 1 and model._lmhead_vocab_sharded:
                    model.prefill_trash_block = 0
                    model.capture_prefill_traces_short(self.mesh_device, page_table)
        finally:
            if prev is not None:
                model._unbind_gdn_prefill_scratch(prev)
        if batched:
            model.warmup_gdn_slot_ops()

    def warmup_model_decode(self, *args, **kwargs):
        # Defer to WarmupForwardMixin, which warms the paged-SDPA + GDN decode path at pos 0.
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        self._decode_bucket_last = None  # warmup replays leave the resident decode inputs at warmup values
        self._validate_device_sampling_request(kwargs.get("can_sample_on_device", False))
        result = warmup_decode_buckets(self, super().warmup_model_decode, *args, **kwargs)
        self._ns_reset()  # warmup rows are not requests: identity map, no owned rows
        if kwargs.get("enable_trace"):
            self._capture_gdn_slot_write_traces()
        self._maybe_forbid_post_capture_compile(kwargs.get("enable_trace"))
        return result

    def _maybe_forbid_post_capture_compile(self, enable_trace):
        """Arm the post-capture compile guard (QWEN36_FORBID_POST_CAPTURE_COMPILE=1, traced warmup only)."""
        if enable_trace and os.environ.get("QWEN36_FORBID_POST_CAPTURE_COMPILE") == "1":
            self.mesh_device.set_program_cache_misses_allowed(False)
            logger.info("post-capture program compiles are now forbidden")

    def _capture_gdn_slot_write_traces(self):
        """After the decode traces, before the post-capture compile guard: capture the per-slot GDN slot-write traces. The
        layers normally sit in the packed conv format here (decode ran); otherwise enter it for the capture only. Replay
        is re-validated per call (packed format + baked-address fingerprint), so a later format change just goes eager.
        """
        model = self.model[0]
        if not (model.num_devices > 1 and model.args.max_batch_size > 1) or model._gdn_prefill_scratch is None:
            return
        dn_states = [layer.attention for layer in model.layers if not layer.is_full_attention]
        prevs = [dn.warmup_enter_packed() if dn._conv_fmt != "packed" else None for dn in dn_states]
        try:
            model.capture_gdn_slot_write_traces()
        finally:
            for dn, p in zip(dn_states, prevs):
                dn.warmup_exit_packed(p)


# BH-Galaxy / P150x8: with data parallel each DP rank opens a (1, 8) mesh, so TP is 8 (not the 32-chip product).
_MESH_TP = {"N150": 1, "N300": 2, "P150x4": 4, "N150x4": 4, "T3K": 8, "P150x8": 8, "BH-Galaxy": 8}


def _mesh_tp(default=4):
    """Tensor-parallel degree from MESH_DEVICE ("P150x4" -> 4, "(1, 4)" -> 4)."""
    s = os.environ.get("MESH_DEVICE")
    if not s:
        return default
    if s in _MESH_TP:
        return _MESH_TP[s]
    try:
        dims = [int(x) for x in s.strip("() ").replace("x", ",").split(",") if x.strip()]
        return math.prod(dims) if dims else default
    except ValueError:
        return default


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5SpecProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36MTPContractForCausalLM(Qwen36ForCausalLM):
    """Plugin spec-decode contract adapter (B * (K+1) <= 32) over the MTP engine in tt/spec_engine.py."""

    _ENABLE_MTP = True
    _SPEC_K = int(os.environ.get("QWEN36_SPEC_K", "3"))  # vLLM warmup does not carry K
    _SUPPORTED_K = (3, 7, 11)
    # decode_forward is overridden and does not execute the reload commands (DECODE_RELOAD_CONTRACT legacy mode)
    decode_input_update_contract = 0

    # verify answers argmax_ids (greedy) or logits ([B, 1+K, V] for host acceptance);
    # the first token after prefill is host-sampled
    # Base capabilities (O's _build_model_capabilities, incl. fabric_config) with everything the spec engine cannot do
    # forced off: its prefill asserts start_pos == 0 (no prefix-cache hits, no chunked prefill), every step is a verify
    # (no device-resident async decode) and the first token is host-sampled.
    model_capabilities = {
        **_build_model_capabilities(),
        # QWEN36_SPEC_PREFIX_CACHE=1 (default): APC via the model's GDN snapshot cache (tt/prefix_cache.py); the spec
        # prefill resumes from a snapshot at or below start_pos (B == 1 recomputes from 0).
        "supports_prefix_caching": _spec_prefix_cache_enabled(),
        # QWEN36_SPEC_CHUNKED_PREFILL=1 + prefix cache: non-final chunks snapshot the GDN state at e-128 and the next chunk
        # resumes there (spec_engine.prefill final=False); the plugin passes intermediate_prefill_mask.
        **_mtp_chunked_capabilities(),
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose", "hidden_feed"),
        "spec_hidden_handoff": ("on_device",),
        # Same condition as supports_async_spec_decode: the platform gate disables async scheduling for the whole launch otherwise.
        # Only plain narrow decode steps (_decode_narrow, host logits, no device sampling so never overlapped) reach it besides verifies.
        "supports_async_decode": bool(spec_chain_enabled() and spec_fused_enabled()),
        # QWEN36_SPEC_CHAIN=1 and QWEN36_SPEC_FUSED=1 (stage S4b): fused verifies are enqueue-only (decode_forward defer_read), read on
        # the plugin's reader thread (read_decode_output) and may chain on the step in flight. Read at import, like the other flags here.
        "supports_async_spec_decode": bool(spec_chain_enabled() and spec_fused_enabled()),
        # QWEN36_SPEC_PREFILL_OVERLAP=1 on top of that: prefill_forward may run while one deferred verify step is pending; the plugin
        # finishes it (propose_draft_tokens with the handle) after the prefill and the next decode materializes it.
        "supports_prefill_over_pending_spec": bool(
            spec_chain_enabled() and spec_fused_enabled() and spec_prefill_overlap_enabled()
        ),
        # decode_forward(finishing_slots=[runner slots]): requests left out of a step only because the step in flight ends them at
        # max_tokens. The engine frees their rows without parking (spec_engine._apply_finishing). Only meaningful with spec.
        "supports_finishing_slots": True,
        # Device logprobs (num_logprobs 0..20 in spec_sampling, raw logprobs from the top-64 + logsumexp payload) and the
        # full-vocabulary top-p certificate (top_k 0 with top_p < 1). Both need the fused sampled accept and the stats trace.
        "supports_spec_device_logprobs": bool(
            spec_fused_enabled() and spec_fused_sampled_enabled() and spec_device_logprobs_enabled()
        ),
        "supports_spec_device_top_p": bool(
            spec_fused_enabled() and spec_fused_sampled_enabled() and spec_device_topp_enabled()
        ),
        # Phase 3 (human-approved 2026-10-28): the fused sampled step applies a packed token mask (grammar / allowed_token_ids /
        # bad_words / min_tokens, one composed int32 [b, 1+K, W] ``mask`` key of spec_sampling, or the two-call decode_forward_begin /
        # decode_forward_finish). Import-time part: the fused sampled step, the mask kill switch (QWEN36_SPEC_DEVICE_MASK) and the traces it
        # builds on (the stats trace and the certificate accept: QWEN36_SPEC_DEVICE_TOPP). ``_publish_mask_capability`` then sets the key
        # from the engine after the traces are captured (False when the mask traces were not built).
        "supports_spec_device_mask": _spec_mask_capability(),
        # Phase 3 part 2 (PEN-1, tt/spec_penalties.py): repetition / presence / frequency penalties computed on the device inside the fused
        # step (per-row counts + prompt mask state, a pen trace before the top-k trace, an update trace after the accept), chained like any
        # sampled step. Penalized tokens are fp32 results rounded to bf16 (about half a bf16 ulp from the host reference); a logprobs row
        # with a penalty is not served. The engine builds the traces with the verify traces and raises at startup when it cannot, so the
        # flag set is the whole condition. Approved by name on 2026-10-28.
        "supports_spec_device_penalties": bool(
            spec_fused_enabled() and spec_fused_sampled_enabled() and spec_device_penalties_enabled()
        ),
        "output_tokens_per_step": 1,
        "supports_sample_on_device": False,
    }

    @staticmethod
    def _serving_conflict(vc):
        """Reason string if vLLM's config enables prefix caching or a chunked prefill that would split a prompt (the spec
        prefill needs whole prompts from position 0), else None. ``vc`` None (no vLLM config) -> None."""
        if vc is None:
            return None
        if not _spec_prefix_cache_enabled() and bool(
            getattr(getattr(vc, "cache_config", None), "enable_prefix_caching", False)
        ):
            return "prefix caching is enabled; the spec engine's prefill needs start_pos == 0 (use --no-enable-prefix-caching)"
        sc = getattr(vc, "scheduler_config", None)
        if sc is not None and getattr(sc, "enable_chunked_prefill", False) and not _spec_chunked_prefill_enabled():
            max_len = int(vc.model_config.max_model_len)
            budget = int(getattr(sc, "max_num_scheduled_tokens", None) or sc.max_num_batched_tokens)
            thr = int(getattr(sc, "long_prefill_token_threshold", 0) or 0)
            if budget < max_len or 0 < thr < max_len:
                return (
                    f"chunked prefill can split prompts (max_num_batched_tokens={budget} < max_model_len={max_len}); "
                    "the spec engine's prefill needs whole prompts (raise --max-num-batched-tokens to >= max_model_len "
                    "or pass --no-enable-chunked-prefill)"
                )
        return None

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        # runs inside vLLM's config context (initialize_vllm_model does not): last chance to refuse an unsupported config
        conflict = self._serving_conflict(self._vllm_config_or_none())
        if conflict:
            raise RuntimeError(f"Qwen36MTPContractForCausalLM: {conflict}")
        return super().allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Config-time plan; never raises and never touches devices."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        k = cls._SPEC_K
        conflict = cls._serving_conflict(vllm_config)
        if conflict:
            return SpecReject(reason=conflict, supported_k=())
        if k not in cls._SUPPORTED_K:
            return SpecReject(reason=f"QWEN36_SPEC_K={k} unsupported; need one of {cls._SUPPORTED_K}", supported_k=())
        if max_num_seqs * (k + 1) > 32:
            return SpecReject(
                reason=f"max_num_seqs={max_num_seqs} x (K+1)={k + 1} verify rows exceed the 32-row tile"
                " (K is fixed by QWEN36_SPEC_K)",
                supported_k=tuple(x for x in cls._SUPPORTED_K if max_num_seqs * (x + 1) <= 32),
            )
        if requested_k < k:
            return SpecReject(reason=f"requested_k={requested_k} < engine K={k}", supported_k=(k,))
        bs = getattr(getattr(vllm_config, "cache_config", None), "block_size", None)
        if bs is not None and int(bs) != _BLOCK_SIZE:
            return SpecReject(
                reason=f"block_size={bs} unsupported; the engine's page tables use {_BLOCK_SIZE}", supported_k=()
            )
        try:
            from models.demos.blackhole.qwen36.tt.spec_engine import spec_memory_costs

            mc = vllm_config.model_config
            tc = mc.hf_config
            tc = tc.get_text_config() if hasattr(tc, "get_text_config") else tc
            if int(getattr(tc, "mtp_num_hidden_layers", 0) or 0) < 1:
                return SpecReject(reason="checkpoint has no MTP weights (mtp_num_hidden_layers < 1)", supported_k=())
            costs = spec_memory_costs(tc, _mesh_tp(), max_num_seqs, k, max_seq_len=mc.max_model_len)
            # costs are for the whole batch: per_seq scales with B, fixed is shared
            per_seq = math.ceil((costs["per_seq"]["total"] + costs["fixed"]["total"]) / max_num_seqs)
            per_token = int(costs["per_token"]["total"])
            mib = 1 << 20
            logger.info(
                f"[spec] B={max_num_seqs} cost/seq {per_seq / mib:.1f} MiB (batch: per_seq "
                f"{costs['per_seq']['total'] / mib:.1f}: gdn_ring {costs['per_seq']['gdn_ring'] / mib:.1f}, "
                f"conv {costs['per_seq']['gdn_conv_windows'] / mib:.1f}; fixed {costs['fixed']['total'] / mib:.1f}), "
                f"per_token {per_token} B"
            )
        except Exception as e:
            return SpecReject(reason=f"cost estimate failed: {e}", supported_k=())
        # QWEN36_VERIFY_DEVICE_TOPK (default on) adds the "topk" accept mode (exact top-M candidates from the device instead of
        # the full logits on sampled steps, note 26) and tells the plugin the candidate count. Passed only when on, so
        # a plugin without the field still admits the class with the flag off (QWEN36_VERIFY_DEVICE_TOPK=0).
        topk_m = verify_device_topk_m()
        topk_kw = {"topk_candidates": topk_m} if topk_m else {}
        return SpecPlan(
            effective_k=k,
            lanes_per_request=1,
            extra_bytes_per_seq=per_seq,
            extra_bytes_per_token=per_token,
            accept_modes=("argmax_ids", "logits")
            + (("topk",) if topk_m else ())
            + (("fused_sample",) if spec_fused_enabled() else ()),
            drafter_state="internal",
            supports_narrow_decode=True,
            **topk_kw,
        )

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs):
        self = super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs)
        model = self.model[0]
        assert model.mtp is not None, "MTP head not built (checkpoint has no MTP weights?)"
        model.set_gdn_fused_decode(True)
        # O's lean attention decode cut draft acceptance when the seed path used it (model.py _forward_seed_bucket_tp
        # forces lean_decode=False for that reason); keep the drafter on the validated non-lean path unless opted in.
        if os.environ.get("QWEN36_MTP_LEAN_DECODE", "0") != "1":
            model.mtp.attention.lean_decode = False
        return self

    def _engine(self):
        from models.demos.blackhole.qwen36.tt.spec_engine import get_spec_engine

        return get_spec_engine(self.model[0], self.model[0].args.max_batch_size, self._SPEC_K)

    def _table_width(self):
        from models.demos.blackhole.qwen36.tt.spec_engine import spec_table_width

        return spec_table_width(self.model[0].args.max_seq_len, self._SPEC_K, _BLOCK_SIZE)

    def _table(self, pt):
        """Zero-pad (or trim) a vLLM page table [rows, w] to the engine width; block 0 is vLLM's null block."""
        pt = pt if isinstance(pt, torch.Tensor) else ttnn.to_torch(pt)
        w = self._table_width()
        if pt.shape[1] >= w:
            return pt[:, :w].to(torch.int32)
        out = torch.zeros(pt.shape[0], w, dtype=torch.int32)
        out[:, : pt.shape[1]] = pt
        return out

    def _maybe_build_spec_prefix_cache(self):
        """GDN prefix-state cache for spec prefill hits, built like the base warmup does: before ANY spec buffer / trace
        exists (its first scratch bind can free spec buffers). Needs the batched B=1 scratch (max_batch_size > 1, TP).
        """
        model = self.model[0]
        if not _spec_prefix_cache_enabled() or os.environ.get("QWEN36_PREFIX_CACHE", "1") != "1":
            return
        if not getattr(self, "_vllm_prefix_caching", True):
            return
        if model._prefix_cache is not None or model.num_devices <= 1 or model.args.max_batch_size <= 1:
            return
        from models.demos.blackhole.qwen36.tt.prefix_cache import GdnPrefixStateCache

        model._prefix_cache = GdnPrefixStateCache(model)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        if not enable_trace:
            # eager phase: compile + run the vision tower (built by initialize_vllm_model) BEFORE any trace exists, as
            # the base class does; the base warmup_model_prefill is not called here.
            self.warmup_vision()
            self._maybe_build_spec_prefix_cache()
        eng = self._engine()
        if enable_trace:
            if not eng.prefill_recorded:
                eng.record_prefill()
            return
        if eng.prefill_prepared:
            return
        w = self._table_width()
        pool = int(kv_cache[0][0].shape[0]) if kv_cache else w
        assert w <= pool, (
            f"vLLM KV pool too small: {pool} blocks < table width {w} (max_model_len + K + 1 = "
            f"{self.model[0].args.max_seq_len} + {self._SPEC_K} + 1 tokens)"
        )
        # stay below the pool size: the last block is the MTP scratch block
        # (row u starts at a distinct block: the verify's KV-collision check at warm position 0)
        B = self.model[0].args.max_batch_size
        scratch = torch.stack([(torch.arange(w) * B + u) % max(1, pool - 1) for u in range(B)]).to(torch.int32)
        eng.prepare_prefill(scratch)

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, *args, **kwargs):
        # Plain decode traces are never used: every step is a verify (narrow steps run as 1-column verifies).
        eng = self._engine()
        if enable_trace:
            if not eng.captured:
                eng.capture_decode()
            self._publish_mask_capability(eng)
            # all spec traces (prefill record, verify, drafter) are parked by now
            self._maybe_forbid_post_capture_compile(enable_trace)
        elif not eng.prepared:
            eng.prepare_decode()

    def _ensure_ready(self, kv_cache=None):
        eng = self._engine()
        if eng.prefill_recorded and eng.captured:
            self._publish_mask_capability(eng)
            return eng
        logger.warning("Qwen36 MTP engine not warmed up (enable_model_warmup=False?); warming lazily")
        # plugin warmup order: allocate and compile everything before any trace is recorded
        if not eng.prefill_prepared:
            self.warmup_model_prefill(kv_cache, False)
        if not eng.prepared:
            eng.prepare_decode()
        if not eng.prefill_recorded:
            eng.record_prefill()
        if not eng.captured:
            eng.capture_decode()
        self._publish_mask_capability(eng)
        return eng

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        N = tokens.shape[0]
        start_pos = kwargs.get("start_pos")
        use_pc = _spec_prefix_cache_enabled()
        if not use_pc:
            assert start_pos is None or all(int(s) == 0 for s in start_pos), "no chunked prefill / prefix caching"
        starts = [int(x) for x in start_pos] if use_pc and start_pos is not None else None
        mask = kwargs.get("intermediate_prefill_mask")
        nonfinal = [bool(mask[i]) if mask is not None else False for i in range(N)]
        if any(nonfinal) and not _spec_chunked_prefill_enabled():
            raise RuntimeError(
                "non-final prefill chunk but QWEN36_SPEC_CHUNKED_PREFILL=1 with the spec prefix cache is off"
            )
        eng = self._ensure_ready(kv_cache)
        model = self.model[0]
        slots = [int(x) for x in kwargs.get("empty_slots") or range(N)]
        table = self._table(page_table[:N])
        rows = []
        for i in range(N):
            L = int(prompt_lens[i]) if prompt_lens is not None else tokens.shape[1]
            # Multimodal request: the tower runs right before this user's prefill (it stashes model-global grids), as in
            # the non-spec batched path. Same semantics: video takes precedence, bucket checked against the warmup.
            has_mm = any(self._has_visual(kwargs, k, i) for k in ("pixel_values", "pixel_values_videos"))
            if has_mm and nonfinal[i]:
                raise RuntimeError(
                    f"multimodal prefill row {i} is a non-final chunk: the MTP spec path cannot split a multimodal prompt "
                    "(the scheduler must not chunk it)"
                )
            vision_tokens = self._compute_vision_tokens(model, kwargs, i) if has_mm else None
            if nonfinal[i]:
                assert starts is not None, "non-final prefill chunk without start_pos"
                out = eng.prefill(slots[i], tokens[i, :L], table[i : i + 1], start_pos=starts[i], final=False)
                rows.append(out.float().reshape(-1))  # zero row: the plugin ignores a non-final chunk's output
                continue
            if vision_tokens is not None:
                # No prefix-cache resume: recompute from 0 (a preempted / resumed request may carry start_pos > 0).
                out = eng.prefill(
                    slots[i], tokens[i, :L], table[i : i + 1], start_pos=None, vision_tokens=vision_tokens
                )
            elif starts is None:
                out = eng.prefill(slots[i], tokens[i, :L], table[i : i + 1])
            else:  # start_pos = num_computed_tokens (cached prefix, or a preemption resume)
                out = eng.prefill(slots[i], tokens[i, :L], table[i : i + 1], start_pos=starts[i])
            rows.append(out.float().reshape(-1))
        # Zero rope deltas, as the non-spec class: the plugin unpacks (logits, rope_deltas) for an M-RoPE model, and the
        # per-request delta is model-side (build_request_rope sets rope.rope_delta during the prefill).
        return torch.stack(rows).reshape(N, 1, -1), torch.zeros(N, dtype=torch.long)

    def decode_forward_begin(self, *args, **kwargs):
        """First half of a split fused step (phase 3 stage P3a; needs the capability supports_spec_device_mask). The arguments are those
        of ``decode_forward`` for spec_mode "fused_sample", by keyword, with these differences: ``spec_sampling`` is REQUIRED and has
        no ``mask`` key (the mask comes in the finish); ``chain_rows`` / ``chain_tip`` (phase 3 stage P6, part of the same capability)
        have the meaning they have in ``decode_forward`` and need ``defer_read=True``: the begin then enqueues the chain trace and the
        forward-only main trace, so the forward starts when the previous step ends (the previous step must be finished; a chain that
        breaks makes this a host-staged split step; QWEN36_SPEC_DEVICE_MASK=0 raises ValueError for a chained begin).
        ``defer_read`` as for decode_forward (True: the step is read by the reader thread; False: the finish synchronizes and returns
        the result; a top-p certificate row needs False). Does the host pre-work and enqueues the main verify trace only. Returns
        VerifyOutput(spec_mode, handle=pending FusedHandle): ``handle.pending`` is True, ``handle.wait()`` raises, so the reader thread
        must get the handle only after ``decode_forward_finish``. One split step at a time (a second begin, a decode_forward or a
        prefill before the finish raises RuntimeError)."""
        return self.decode_forward(*args, _split=True, **kwargs)

    def decode_forward_finish(self, tt_out, mask=None):
        """Second half of a split fused step. ``tt_out``: what ``decode_forward_begin`` returned (the VerifyOutput or its handle).
        ``mask``: the composed token mask int32 [b, 1+K, W] (runner slots, b <= the padded rows, W <= 7760 words, bit v % 32 of word
        v // 32 = token v allowed, all ones = no constraint, a row of zeros raises ValueError) or None (no constraint). Writes the mask,
        replays the mask trace when a live row is constrained, then enqueues the rest of the step (topk, stats, accept, drafter, reads,
        event). Returns VerifyOutput(spec_mode, handle=the same handle, readable now) for defer_read=True, else the full
        VerifyOutput of a synchronous fused step (accepted_token_ids / counts, candidates)."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        handle = getattr(tt_out, "handle", tt_out)
        eng = self._ensure_ready()
        if mask is not None:
            mask = self._check_mask(mask, int(eng.B))
        res = eng.decode_forward_finish(handle, mask)
        if res.handle is not None:
            return VerifyOutput(spec_mode=res.spec_mode, handle=res.handle)
        cand = getattr(res, "candidates", None)
        return VerifyOutput(
            spec_mode=res.spec_mode,
            argmax_ids=res.argmax_ids,
            accepted_token_ids=res.accepted_token_ids,
            accepted_counts=res.accepted_counts,
            hidden=None,
            **({} if cand is None else {"candidates": _plugin_candidates(cand)}),
        )

    def _publish_mask_capability(self, eng):
        """Set supports_spec_device_mask from the captured engine (the class dict is shared, so the plugin sees it after warmup): False
        when the mask traces were not built (the import-time value already covers the switches)."""
        caps = type(self).model_capabilities
        ok = getattr(eng, "_mask_ok", None)
        if ok is not None and caps.get("supports_spec_device_mask", False) and not ok():
            caps["supports_spec_device_mask"] = False
            logger.warning(
                "supports_spec_device_mask is False: the engine has no mask traces (see the [spec_engine] mask trace log)"
            )

    def decode_forward(self, *args, **kwargs):
        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        split = bool(kwargs.pop("_split", False))  # decode_forward_begin: the first half of a split step

        def _read(name, pos):
            return kwargs[name] if name in kwargs else (args[pos] if pos < len(args) else None)

        tokens, start_pos, page_table = _read("tokens", 0), _read("start_pos", 1), _read("page_table", 2)
        remap = _read("slot_remap", _DECODE_POSITIONAL.index("slot_remap"))
        narrow = "spec_mode" not in kwargs
        if split:
            if narrow or kwargs["spec_mode"] != "fused_sample":
                raise ValueError("decode_forward_begin needs spec_mode 'fused_sample'")
            if not self.model_capabilities.get("supports_spec_device_mask", False):
                raise ValueError("decode_forward_begin needs the supports_spec_device_mask capability")
            if kwargs.get("spec_sampling") is None:
                raise ValueError("decode_forward_begin needs spec_sampling (a masked step is a sampled step)")
            if "mask" in kwargs["spec_sampling"]:
                raise ValueError(
                    "decode_forward_begin takes no spec_sampling['mask']: pass the mask to decode_forward_finish"
                )
        if narrow:
            return self._decode_narrow(tokens, start_pos, page_table, kwargs, remap)
        num_valid_drafts = kwargs.pop("num_valid_drafts")
        accepted_counts = kwargs.pop("accepted_counts")
        spec_mode = kwargs.pop("spec_mode")
        defer_read = bool(kwargs.pop("defer_read", False))
        chain_rows, chain_tip = kwargs.pop("chain_rows", None), kwargs.pop("chain_tip", None)
        finishing_slots = kwargs.pop("finishing_slots", None)
        if finishing_slots is not None:
            finishing_slots = [int(s) for s in finishing_slots]
            if not self.model_capabilities.get("supports_finishing_slots", False):
                raise ValueError("finishing_slots needs the supports_finishing_slots capability")
        if defer_read or chain_rows is not None or chain_tip is not None:
            if spec_mode != "fused_sample":
                raise ValueError(
                    f"defer_read / chain_rows / chain_tip need spec_mode 'fused_sample', got {spec_mode!r}"
                )
            if not self.model_capabilities.get("supports_async_spec_decode", False):
                raise ValueError(
                    "defer_read needs QWEN36_SPEC_CHAIN=1 and QWEN36_SPEC_FUSED=1 (supports_async_spec_decode)"
                )
            if chain_rows is not None and not defer_read:
                raise ValueError("chain_rows needs defer_read=True")
        B = int(tokens.shape[0])
        if chain_rows is not None:  # chain rows carry accepted_counts -1; everything else is checked as ever
            if tuple(chain_rows.shape) != (B,) or chain_rows.dtype != torch.bool:
                raise ValueError(
                    f"chain_rows must be a bool tensor of shape [{B}], got {chain_rows.dtype} {tuple(chain_rows.shape)}"
                )
            if bool((accepted_counts[chain_rows] != -1).any()):
                raise ValueError("a chain row must carry accepted_counts -1")
            counts_chk = torch.where(chain_rows, torch.ones_like(accepted_counts), accepted_counts)
            check_spec_side_tensors(num_valid_drafts, counts_chk, B, self._SPEC_K)
        else:
            check_spec_side_tensors(num_valid_drafts, accepted_counts, B, self._SPEC_K)
        assert kwargs.get("page_tables_per_layer") is None, "per-layer page tables unsupported"
        spec_sampling = self._check_spec_sampling(kwargs.pop("spec_sampling", None), spec_mode, int(tokens.shape[0]))
        eng = self._ensure_ready()
        res = (eng.decode_forward_begin if split else eng.decode_forward)(
            tokens,
            start_pos,
            num_valid_drafts=num_valid_drafts,
            accepted_counts=accepted_counts,
            page_table=self._table(page_table),
            spec_mode=spec_mode,
            slot_remap=remap,
            # stage S2 (fused_sample with sampled rows); sent only when present, so a greedy / older call keeps its shape
            **({} if spec_sampling is None else {"spec_sampling": spec_sampling}),
            **({"defer_read": True} if defer_read else {}),
            **({} if chain_rows is None else {"chain_rows": chain_rows, "chain_tip": chain_tip}),
            **({} if not finishing_slots else {"finishing_slots": finishing_slots}),
        )
        logits = res.logits if spec_mode == "logits" else None
        if split or (
            spec_mode == "fused_sample" and defer_read
        ):  # stage S4b: no host tensors; the reader thread resolves the handle (a split step: after decode_forward_finish)
            return VerifyOutput(spec_mode=res.spec_mode, handle=res.handle)
        cand = getattr(res, "candidates", None)
        if spec_mode == "fused_sample":  # QWEN36_SPEC_FUSED: greedy rows (S1) or, with spec_sampling, sampled rows (S2)
            return VerifyOutput(
                spec_mode=res.spec_mode,
                argmax_ids=res.argmax_ids,
                accepted_token_ids=res.accepted_token_ids,
                accepted_counts=res.accepted_counts,
                hidden=None,
                # phase 2: the logprob payload, only when a row asked for it (an older plugin never does, so its VerifyOutput
                # never sees the keyword)
                **({} if cand is None else {"candidates": _plugin_candidates(cand)}),
            )
        if spec_mode == "topk":
            # The candidates plus the lazy full-logits fallback (one device round trip; the plugin walks acceptance on
            # the runner thread before it proposes, see supports_async_spec_decode=False).
            return VerifyOutput(
                spec_mode=res.spec_mode,
                argmax_ids=res.argmax_ids,
                hidden=None,
                topk_values=res.topk_values,
                topk_ids=res.topk_ids,
                fetch_logits=res.fetch_logits,
            )
        return VerifyOutput(spec_mode=res.spec_mode, argmax_ids=res.argmax_ids, hidden=None, logits=logits)

    def read_decode_output(self, tt_out, async_read=False, **kwargs):
        """Reader-thread read of a deferred fused verify (stage S4b): ``read_decode_output(handle)`` blocks on the step's device event
        and returns FusedReadout(accepted_token_ids [B,1+K] int32, accepted_counts [B] int32, p0 [B] int32). It makes no other device
        call and changes no engine state (the handle's own result cache only). With ``supports_spec_device_mask`` the readout also
        carries ``drafts`` (int32 [B, K], runner slots: the drafts the next chained step verifies as columns 1..K, from
        ``FusedHandle.drafts()``) when the plugin's FusedReadout has that field. Any other ``tt_out`` takes the Generator's path.
        """
        from models.demos.blackhole.qwen36.tt.spec_engine import FusedHandle

        if isinstance(tt_out, FusedHandle):
            from vllm_tt_plugin.spec_decode import FusedReadout

            ids, counts, p0 = tt_out.wait()
            cand = tt_out.candidates()  # None unless a row of the step asked for logprobs (phase 2)
            # phase 3 stage P6: the drafts of the step (pure host view); an older plugin's FusedReadout has no such field
            has_drafts = "drafts" in getattr(FusedReadout, "__dataclass_fields__", {})
            return FusedReadout(
                accepted_token_ids=ids,
                accepted_counts=counts,
                p0=p0,
                **({} if cand is None else {"candidates": _plugin_candidates(cand)}),
                **({"drafts": tt_out.drafts()} if has_drafts else {}),
            )
        if async_read and hasattr(tt_out, "spec_mode") and getattr(tt_out, "handle", None) is None:
            # Non-deferred VerifyOutput (sampled fused_sample, topk): already host when every present tensor field is.
            fields = [
                getattr(tt_out, n, None)
                for n in (
                    "argmax_ids",
                    "accepted_token_ids",
                    "accepted_counts",
                    "logits",
                    "topk_values",
                    "topk_ids",
                )
            ]
            cand = getattr(tt_out, "candidates", None)  # the logprob payload is host when its three tensors are
            if cand is not None:
                fields += [cand.ids, cand.values, cand.lse]
            if all(f is None or isinstance(f, torch.Tensor) for f in fields):
                return tt_out, []
        if async_read and isinstance(tt_out, torch.Tensor):
            # Already host (narrow decode logits, non-deferred verify): the Generator's path only reads ttnn tensors.
            # The plugin's finalize_decode takes host output with no events to wait on.
            return tt_out, []
        return super().read_decode_output(tt_out, async_read=async_read, **kwargs)

    def _check_spec_sampling(self, ss, spec_mode, B):
        """Validate the plugin's ``spec_sampling`` (stage S2; docs/SPEC_DECODE_CONTRACT.md): host tensors over the padded rows
        B: temperature fp32 [B], top_k int32 [B], top_p fp32 [B], greedy bool [B], uniforms fp32 [B, K+2]. Only a
        fused_sample verify may carry it. A sampled row with top_k 0 and top_p < 1 (full-vocabulary nucleus; the engine's
        certificate row) needs the capability supports_spec_device_top_p. Returns the dict (tensors) or None."""
        if ss is None:
            return None
        if spec_mode != "fused_sample":
            raise ValueError(f"spec_sampling is only valid with spec_mode 'fused_sample', got {spec_mode!r}")
        K = self._SPEC_K
        want = {
            "temperature": ((B,), torch.float32),
            "top_k": ((B,), torch.int32),
            "top_p": ((B,), torch.float32),
            "greedy": ((B,), torch.bool),
            "uniforms": ((B, K + 2), torch.float32),
        }
        if "num_logprobs" in ss:  # phase 2 (optional): int32 [B], -2 = no logprobs, 0..max_logprobs = top-N
            want["num_logprobs"] = ((B,), torch.int32)
        if set(ss) - {"mask", "penalties"} != set(
            want
        ):  # "mask" (phase 3, checked by _check_mask below) and "penalties" (phase 3 part 2, checked by _check_spec_penalties) are optional
            raise ValueError(f"spec_sampling keys {sorted(set(ss) - {'mask', 'penalties'})} != {sorted(want)}")
        for name, (shape, dtype) in want.items():
            t = ss[name]
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != dtype:
                got = (tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__
                raise ValueError(f"spec_sampling[{name!r}] must be a {dtype} tensor of shape {shape}, got {got}")
        u = ss["uniforms"]
        if bool(((u < 0) | (u >= 1.0)).any()):
            raise ValueError("spec_sampling['uniforms'] must lie in [0, 1)")
        cert = (
            ~ss["greedy"] & (ss["top_k"] <= 0) & (ss["top_p"] < 1.0)
        )  # top_k 0 = the whole vocabulary (the plugin's "off")
        if bool(cert.any()) and not self.model_capabilities.get("supports_spec_device_top_p", False):
            raise ValueError(
                "spec_sampling top_k = 0 with top_p < 1 on a sampled row needs the supports_spec_device_top_p capability"
            )
        if "num_logprobs" in ss:
            nl = ss["num_logprobs"]
            if bool(((nl < -2) | (nl > 20)).any()):
                raise ValueError("spec_sampling['num_logprobs'] must lie in [-2, 20] (-2 = no logprobs)")
            if bool((nl >= 0).any()) and not self.model_capabilities.get("supports_spec_device_logprobs", False):
                raise ValueError(
                    "spec_sampling['num_logprobs'] >= 0 needs the supports_spec_device_logprobs capability"
                )
        if "mask" in ss:
            self._check_mask(ss["mask"], B)
        if ss.get("penalties") is not None:
            self._check_spec_penalties(ss, B)
        return ss

    def _check_mask(self, mask, B):
        """Validate the composed token mask of a step (phase 3; spec_sampling["mask"] or decode_forward_finish): int32 [b, 1+K, W] on the
        host, runner slots, 1 <= b <= B (users the plugin does not send are all ones), W <= 7760 words (a narrower mask leaves the
        missing ids disallowed), standard packing (bit v % 32 of word v // 32 = token v allowed; -1 words = no constraint). Needs the
        capability supports_spec_device_mask. A row with no allowed token (all zero words) raises ValueError: the plugin must route such
        a step to logits mode (the engine host walk raises on it too). Returns ``mask``."""
        from models.demos.blackhole.qwen36.tt.spec_masks import MASK_WORDS

        if not self.model_capabilities.get("supports_spec_device_mask", False):
            raise ValueError("a token mask needs the supports_spec_device_mask capability (QWEN36_SPEC_DEVICE_MASK=1)")
        K = self._SPEC_K
        ok = isinstance(mask, torch.Tensor) and mask.dtype == torch.int32 and mask.dim() == 3
        if not (ok and mask.shape[1] == K + 1 and 1 <= mask.shape[0] <= B and 1 <= mask.shape[2] <= MASK_WORDS):
            got = (tuple(mask.shape), mask.dtype) if isinstance(mask, torch.Tensor) else type(mask).__name__
            raise ValueError(f"the token mask must be an int32 tensor [1..{B}, {K + 1}, 1..{MASK_WORDS}], got {got}")
        if bool((mask == 0).all(dim=-1).any()):
            raise ValueError("the token mask has a row that allows no token (run such a step in logits mode)")
        return mask

    def _check_spec_penalties(self, ss, B):
        """Validate ``spec_sampling["penalties"]`` (phase 3 part 2; docs/SPEC_DECODE_CONTRACT.md): a dict with ``repetition``, ``frequency``
        and ``presence`` fp32 [B] (one value per runner slot; 1 / 0 / 0 = no penalty, a padding row sends those) and the optional
        ``rebuild`` {runner slot: (prompt ids, output ids)} (1-D integer tensors): the plugin sends it for a request whose device state is
        not current (join, resume, a slot move, a step the host walked), only on a step that does not continue a chain for that row. A row
        with a non-neutral penalty must be a sampled row (top_k 1, top_p 0 for a greedy request) and must not ask for logprobs (the raw
        logprobs come before the penalties and the device returns the penalized candidates). Needs supports_spec_device_penalties.
        """
        pen = ss["penalties"]
        if not isinstance(pen, dict) or not {"repetition", "frequency", "presence"} <= set(pen) <= {
            "repetition",
            "frequency",
            "presence",
            "rebuild",
        }:
            raise ValueError(
                f"spec_sampling['penalties'] must be a dict with repetition / frequency / presence [B] and an optional rebuild, got "
                f"{sorted(pen) if isinstance(pen, dict) else type(pen).__name__}"
            )
        for name in ("repetition", "frequency", "presence"):
            t = pen[name]
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != (B,) or t.dtype != torch.float32:
                got = (tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__
                raise ValueError(
                    f"spec_sampling['penalties'][{name!r}] must be a float32 tensor of shape ({B},), got {got}"
                )
        rep, freq, pres = pen["repetition"], pen["frequency"], pen["presence"]
        if not bool(torch.isfinite(torch.stack([rep, freq, pres])).all()) or bool((rep <= 0).any()):
            raise ValueError(
                f"spec_sampling['penalties'] must be finite with repetition > 0: repetition {rep.tolist()}, frequency {freq.tolist()}, "
                f"presence {pres.tolist()}"
            )
        live = (rep != 1.0) | (freq != 0.0) | (pres != 0.0)
        rebuild = pen.get("rebuild") or {}
        if (bool(live.any()) or rebuild) and not self.model_capabilities.get("supports_spec_device_penalties", False):
            raise ValueError("spec_sampling['penalties'] needs the supports_spec_device_penalties capability")
        if bool((live & ss["greedy"]).any()):
            raise ValueError(
                f"spec_sampling rows {(live & ss['greedy']).nonzero().reshape(-1).tolist()} carry penalties and are marked greedy; send a "
                "greedy penalty row as a sampled row with top_k 1 and top_p 0"
            )
        nl = ss.get("num_logprobs")
        if nl is not None and bool((live & (nl >= 0)).any()):
            raise ValueError(
                f"spec_sampling rows {(live & (nl >= 0)).nonzero().reshape(-1).tolist()} ask for logprobs and carry penalties: the device "
                "returns penalized candidates, the raw logprobs are not served for such a row"
            )
        for slot, lists in rebuild.items():
            if not 0 <= int(slot) < B or len(lists) != 2:
                raise ValueError(
                    f"spec_sampling['penalties']['rebuild'] entry {slot!r} must be slot in range({B}) -> (prompt, output)"
                )
            for ids in lists:
                if not isinstance(ids, torch.Tensor) or ids.dim() != 1 or ids.dtype not in (torch.int32, torch.int64):
                    raise ValueError(
                        f"spec_sampling['penalties']['rebuild'][{slot}] needs 1-D int32 / int64 tensors, got {type(ids).__name__}"
                    )

    def _decode_narrow(self, tokens, start_pos, page_table, kwargs, remap=None):
        """Ordinary [B, 1] decode step (no drafts) run as a 1-column verify; ``accepted_counts`` resolves a
        previous multi-token commit.
        Returns host float logits [B, 1, V]: the runner host-samples ``tt_out[rows, -1, :]``
        (model_runner.py:3276; async_decode.py:103 accepts a torch tensor as host output)."""
        nv, ac = kwargs.get("num_valid_drafts"), kwargs.get("accepted_counts")
        B = int(tokens.shape[0])
        if nv is not None and bool(torch.as_tensor(nv).any()):
            raise ValueError("narrow decode step carries drafts")
        if ac is not None:
            from vllm_tt_plugin.spec_decode import check_spec_side_tensors

            check_spec_side_tensors(torch.zeros(B, dtype=torch.int32), ac, B, self._SPEC_K, call="decode")
        # Contract 4d: an ordinary step after a multi-token commit carries the count; it picks the state slot.
        acc = torch.ones(B, dtype=torch.int32) if ac is None else torch.as_tensor(ac).reshape(B).to(torch.int32)
        assert kwargs.get("page_tables_per_layer") is None, "per-layer page tables unsupported"
        K = self._SPEC_K
        tok = torch.full((B, K + 1), -1, dtype=torch.int32)
        tok[:, 0] = torch.as_tensor(tokens).reshape(B).to(torch.int32)
        pos = torch.full((B, K + 1), -1, dtype=torch.int32)
        pos[:, 0] = torch.as_tensor(start_pos).reshape(B).to(torch.int32)
        eng = self._ensure_ready()
        res = eng.decode_forward(
            tok,
            pos,
            num_valid_drafts=torch.zeros(B, dtype=torch.int32),
            accepted_counts=acc,
            page_table=self._table(page_table),
            spec_mode="logits",
            slot_remap=remap,
            **({"narrow_col0": True} if spec_hostset_enabled() else {}),  # QWEN36_SPEC_HOSTSET: read column 0 only
            **(
                {"finishing_slots": [int(x) for x in kwargs["finishing_slots"]]}
                if kwargs.get("finishing_slots")
                else {}
            ),
        )
        return res.logits[:, :1].float()

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput, check_spec_side_tensors

        B = int(committed_tokens.shape[0])
        check_spec_side_tensors(torch.zeros(B, dtype=torch.int32), accepted_counts, B, self._SPEC_K, call="propose")
        res = self._engine().propose_draft_tokens(
            num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=hidden
        )
        return DraftOutput(draft_token_ids=res.draft_token_ids, num_valid=res.num_valid)

    def release_request(self, slot):
        self._engine().release(int(slot))

    def note_state_slots_moved(self, moves):
        """State moved with slot_remap in decode_forward; nothing to do."""

    def release_persistent_capture(self):
        logger.info("[spec] release_persistent_capture: start")
        eng = getattr(self.model[0], "_spec_engine", None)
        if eng is not None:
            eng.shutdown()
        parent = getattr(super(), "release_persistent_capture", None)
        if parent is not None:
            parent()
        logger.info("[spec] release_persistent_capture: done")
