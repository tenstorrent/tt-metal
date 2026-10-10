# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Qwen3.5/3.6 on Blackhole — a thin tt_transformers Generator subclass.

Hybrid model: 8 paged-KV attention + 24 GDN recurrent-state layers. GDN forbids token-padding and
isn't position-general, so prefill is model-owned (masked-bucket for short prompts / chunk-outer
trace for long, via prefill_dispatch) while Generator drives decode only. GDN + KV state is
model-bound, so the kv_cache contract param is accepted but unused.
"""

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


def _build_model_capabilities():
    """The plugin-visible capabilities. QWEN36_SERVE_DEVICE_DECODE=0 returns exactly the pre-1.6S dict."""
    caps = {
        # Automatic prefix caching: vLLM reuses the attention KV blocks and passes start_pos (cached tokens, a multiple
        # of the block size 64); the model keeps its own GDN-state snapshots (tt/prefix_cache.py). QWEN36_PREFIX_CACHE=0
        # disables it. Run vLLM with --block-size 64 and --enable-prefix-caching.
        "supports_prefix_caching": os.environ.get("QWEN36_PREFIX_CACHE", "1") == "1",
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


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36ForCausalLM(Generator, SupportsMultiModal):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    # Decode bucketing keeps several traces live and refreshes the selected
    # bucket's inputs before replay, so their I/O buffers may safely overlap.
    _tt_allow_decode_trace_buffer_reuse = True
    decode_input_update_contract = 1

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
            enable_mtp=False,
        )
        # Attach the TT vision tower so prefill can splice image/video embeddings (multimodal path).
        # No-op cost for text-only requests; get_image_features / get_video_features are only invoked
        # when a request actually carries pixel_values / pixel_values_videos.
        model.init_vision_model()
        inst = cls([model], [args], mesh_device)
        # vLLM's --[no-]enable-prefix-caching, read from the config the worker publishes while loading the model; the model-side
        # GDN snapshot cache is only worth its ~1.2 GB/device (and the per-prefill snapshot saves) when vLLM will reuse prefixes.
        # Non-vLLM callers (tests, demos) have no current config and keep the default (enabled).
        try:
            from vllm.config import get_current_vllm_config

            inst._vllm_prefix_caching = bool(get_current_vllm_config().cache_config.enable_prefix_caching)
        except Exception:
            inst._vllm_prefix_caching = True
        return inst

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs.

        batch_size = max_batch_size (vLLM's max_num_seqs, threaded through initialize_vllm_model):
        the paged KV blocks (kv_cache_shape) already cover all users, and this sizes the per-slot
        GDN recurrent/conv state [B,...] + the decode kv grid. B==1 is the single-sequence path."""
        batch_size = self.model[0].args.max_batch_size
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=batch_size)

    @staticmethod
    def _has_visual(kwargs, pixel_key):
        """True only when the request carries REAL visual data for this modality. vLLM attaches an
        empty pixel_values placeholder to text requests for a multimodal-registered model, so a
        plain ``is not None`` check misclassifies text as multimodal. Mirrors the emptiness test in
        _gather_user_visual (key absent / empty list / first item None => text-only)."""
        v = kwargs.get(pixel_key)
        return v is not None and len(v) > 0 and v[0] is not None

    @staticmethod
    def _gather_user_visual(kwargs, pixel_key, grid_key):
        """Pull this (B=1) user's patches + (t,h,w) grids for one modality out of the vLLM kwargs.

        Returns (pixel_values, grid_thw) or None when the request carries nothing for this modality.
        Multiple items for the user arrive as lists; concat the patches and stack the grids (same
        shape get_image_features / get_video_features expect: [num_patches, patch_dim] + [N, 3]).
        """
        if pixel_key not in kwargs or len(kwargs[pixel_key]) == 0 or kwargs[pixel_key][0] is None:
            return None
        pixel_values = kwargs[pixel_key][0]
        grid_thw = kwargs[grid_key][0]
        if isinstance(pixel_values, list) and len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values, dim=0)
            grid_thw = torch.stack([g.to(dtype=torch.int32) for g in grid_thw], dim=0)
        return pixel_values, grid_thw

    def _compute_vision_tokens(self, model, kwargs):
        """Run the vision tower for this (single-user, B=1) request, if it carries images or video.

        Mirrors the Qwen3-VL generator's multimodal check: pull this user's pixels + grid out of
        the vLLM kwargs and return the packed embeddings (ttnn [num_vision_tokens, H]) for prefill
        to splice in. Returns None for a text-only request, so the whole multimodal path is skipped.

        Image and video share the vision tower; dispatching to get_video_features (vs
        get_image_features) is what tells the model to splice into video_token_id placeholders and
        build the video M-RoPE. A request carries at most one visual modality (see
        get_supported_mm_limits); video takes precedence if both are somehow present.
        """
        video = self._gather_user_visual(kwargs, "pixel_values_videos", "video_grid_thw")
        if video is not None:
            return model.get_video_features(*video)

        image = self._gather_user_visual(kwargs, "pixel_values", "image_grid_thw")
        if image is not None:
            return model.get_image_features(*image)

        return None

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        # Prefill replays other traces / rewrites slot state: any resident decode inputs are stale from here on, so
        # the next decode must be a commanded reload (_decode_forward_resident refuses a steady step).
        self._decode_bucket_last = None
        model = self.model[0]
        if model.num_devices > 1 and model.args.max_batch_size > 1:
            # Batched text prefill into decode slots (MM is B=1). Require real visual data, not a
            # non-None empty pixel_values placeholder from vLLM on text requests.
            assert not self._has_visual(kwargs, "pixel_values") and not self._has_visual(
                kwargs, "pixel_values_videos"
            ), (
                "batched (max_num_seqs>1) serving is text-only; multimodal is single-sequence "
                "(max_concurrency=1). Run the model at max_num_seqs=1 for image/video requests."
            )
            return self._prefill_forward_tp_batched(
                model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"), start_pos=kwargs.get("start_pos")
            )
        # Prefix caching (start_pos > 0) is only exploited by the batched slot path above. The B=1 TP path and the
        # single-device path ignore start_pos and recompute the whole prompt from position 0: correct (the cached KV
        # blocks are rewritten with the same values), just slower.
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

    def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots, start_pos=None):
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
        Returns ([N, 1, vocab] host logits, [N] zero rope_deltas — text M-RoPE delta is 0, applied model-side).
        """
        N = tokens.shape[0]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [tokens.shape[1]] * N
        if empty_slots is None:
            empty_slots = list(range(N))
        empty_slots = [int(s) for s in empty_slots]
        token_ids_list = [tokens[u : u + 1, : plens[u]].to(torch.int32) for u in range(N)]
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        logger.info(f"Prefilling {N} user(s) into slots {empty_slots} (TP batched masked-bucket)")
        if start_pos is not None:
            start_pos = [int(x) for x in (start_pos.tolist() if hasattr(start_pos, "tolist") else start_pos)]
            assert len(start_pos) == N, "one start_pos per request"
        host_logits = model.prefill_paged_slots(
            token_ids_list, pt, empty_slots, valid_lens=plens, start_positions=start_pos
        )
        logits = torch.cat([hl.reshape(1, 1, -1) for hl in host_logits], dim=0)  # [N, 1, vocab]
        logger.info(f"Finished batched prefill of {N} user(s), starting decode...")
        return logits, torch.zeros(N, dtype=torch.long)

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
        width = int(tokens.shape[0])
        # Bucket by the active prefix with ONE rule on every step (a remap step no longer forces full width: the
        # remap is applied to the full slot space below and rows are front-packed, so slicing afterwards is exact).
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
            model._remap_gdn_slots(slot_remap)

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
        result = super().decode_forward(**kwargs)
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
            model._remap_gdn_slots(gdn_remap)
        return super().decode_forward(*args, **kwargs)

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
        return warmup_decode_buckets(self, super().warmup_model_decode, *args, **kwargs)
