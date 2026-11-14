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
from models.demos.blackhole.qwen36.tt.tp_common import fabric_router_config
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
        # Generator's sample_deferred_decode applies the grammar mask; bucketed logits are padded to the sampler width,
        # and the sampled token takes the same feedback buffer as plain device sampling.
        "supports_device_grammar": True,
        # TTSampling.max_top_k: k outside [1, 32] on a sampled row is routed to the host sampler by the plugin.
        "max_device_top_k": _MAX_DEVICE_TOP_K,
    }
    if serve_device_decode_env_enabled():
        caps.update(
            supports_async_decode=True,
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
        # the spec engine's prefill is text-only: vLLM rejects visual items at admission (HTTP 400)
        return {"image": 0, "video": 0}


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
        """(Re)read vLLM's prefix caching / chunked prefill / max_pixels settings; keeps the current values without a
        vLLM config. max_pixels must be current before warmup_vision, which picks the vision buckets from it."""
        vc = self._vllm_config_or_none()
        if vc is None and hasattr(self, "_vllm_prefix_caching"):
            return
        self._vllm_prefix_caching = True if vc is None else bool(vc.cache_config.enable_prefix_caching)
        self._vllm_chunked_prefill = self._read_vllm_chunked_prefill(vc)
        self._vllm_mm_max_pixels = self._read_vllm_max_pixels(vc)
        if vc is not None:
            logger.info(
                f"vLLM serving config: enable_prefix_caching={self._vllm_prefix_caching}, chunked prefill "
                f"(budget, long_prefill_threshold, align)={self._vllm_chunked_prefill}, "
                f"max_pixels={self._vllm_mm_max_pixels}"
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
    def _read_vllm_max_pixels(vc=None):
        """max_pixels from --mm-processor-kwargs of `vc` (default: the current vLLM config), or None when unset or
        there is no config. The kwargs sit on model_config or model_config.multimodal_config."""
        try:
            if vc is None:
                from vllm.config import get_current_vllm_config

                vc = get_current_vllm_config()
            mc = vc.model_config
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

    @classmethod
    def _vision_grid_for_bucket(cls, bucket):
        """A 1 x h x w warmup grid (h, w even) whose patch count falls in `bucket`."""
        n = bucket - 1
        h = 2 * math.ceil(math.sqrt(n) / 2)
        w = 2 * math.ceil(n / h / 2)
        while cls._vision_bucket(h * w) > bucket and w > 2:
            w -= 2
        return (1, h, w)

    def _check_vision_bucket(self, grid_thw):
        """Refuse an item whose bucket warmup_vision did not compile: compiling after trace capture can hang the device.
        No check when warmup_vision never ran (tests, demos)."""
        warmed = getattr(self, "_vision_warmed_buckets", None)
        if not warmed:
            return
        for g in grid_thw.reshape(-1, 3):
            n = int(g[0]) * int(g[1]) * int(g[2])
            if self._vision_bucket(n) not in warmed:
                raise ValueError(
                    f"Vision input of {n} patches (grid {tuple(int(x) for x in g)}, bucket {self._vision_bucket(n)}) "
                    f"needs a vision bucket not warmed before trace capture (warmed: {sorted(warmed)}); running it "
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
            store[B] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None), {})
        # The staged (prepared) decode trace inputs are per width too: Generator keys them by on_device_sampling only,
        # so a shared dict let width 1's eager staging hide widths 2..Bmax, whose inputs were then allocated at capture,
        # behind the already-parked traces (#56474; test_vllm_warmup_leaves_no_late_allocations).
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
            model.remap_slot_rope_delta(gdn_remap)
        return super().decode_forward(*args, **kwargs)

    # Vision-tower warmup grids (t x h x w patches, h and w even). The tower is shape-stable (get_*_features run every
    # device op on the PADDED bucket, ceil-to-2048 of the patch count + 1 step: 1024 -> 2048, 3072 -> 4096,
    # 4096 -> 6144, 6400 -> 8192 ...; the exact request size only trims rows on host), so one grid per bucket compiles
    # everything a request of that bucket will run. A program first dispatched AFTER the prefill / decode traces were
    # captured can have its binary clobbered by a later trace replay, which hangs the next vision call -- hence this
    # runs in the eager warmup phase. Requests needing a bucket that was not warmed are refused (_check_vision_bucket).
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
                        f"outside the warmed ones ({sorted({self._vision_bucket(t * h * w) for t, h, w in grids})} "
                        "patches) will be REJECTED (an unwarmed bucket would hang the device). Set "
                        "--mm-processor-kwargs '{\"max_pixels\": 1048576}' (or your serving limit)."
                    )
            else:
                # An item has at most max_pixels / patch^2 patches (a video fewer), and every bucket up to that one
                # is reachable, so all of them are warmed.
                need = -(-int(max_pixels) // vc.patch_size**2)
                need_bucket = self._vision_bucket(need)
                have = {self._vision_bucket(t * h * w) for t, h, w in grids}
                added = [b for b in range(2048, need_bucket + 1, 2048) if b not in have]
                if added:
                    grids.extend(self._vision_grid_for_bucket(b) for b in added)
                    logger.info(f"max_pixels={max_pixels}: up to {need} patches -> also warming vision buckets {added}")
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        for t, h, w in grids:
            pix = torch.zeros(t * h * w, patch_dim, dtype=torch.float32)
            grid = torch.tensor([[t, h, w]], dtype=torch.int32)
            ttnn.deallocate(model.get_image_features(pix, grid))  # image / video share the device work
        self._vision_warmed_buckets = {self._vision_bucket(t * h * w) for t, h, w in grids}
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
            grouped = (
                batched
                and model.batched_prefill_enabled()
                and model._lmhead_vocab_sharded
                and model._short_trace_masks_supported()
            )
            if grouped and not enable_trace:
                # Grouped short prefill (QWEN36_BATCHED_PREFILL=1): allocate ALL its persistent state (group GDN
                # scratch, per-(bucket, Bg) inputs) and run its body eagerly (compile + lazily-keyed buffers) HERE, in
                # the eager pass, before any trace is parked; the trace pass below only captures.
                model.prefill_trash_block = 0
                try:
                    model.prepare_prefill_batched_short(self.mesh_device, page_table)
                except Exception as e:  # e.g. RuntimeError "TT_FATAL ... Out of Memory": keep serving per-user
                    logger.warning(
                        f"Grouped short prefill disabled for this session: prepare_prefill_batched_short failed "
                        f"({'out of DRAM' if 'out of memory' in str(e).lower() else type(e).__name__}): {e}"
                    )
                    try:
                        model.release_prefill_batched_short()
                    except Exception as e2:
                        logger.warning(f"release_prefill_batched_short after failed prepare also raised: {e2}")
                    model._batched_short_unavailable = True
                    grouped = False
            short = model.short_prefill_trace_enabled() and model.num_devices > 1 and model._lmhead_vocab_sharded
            if short and not enable_trace:
                # Traced short prefill: allocate every bucket's persistent inputs and run its warm passes HERE, in the
                # eager pass (no trace parked yet); the record pass below only captures (#56474).
                model.prefill_trash_block = 0
                model.prepare_prefill_traces_short(self.mesh_device, page_table)
            if enable_trace:
                model.record_prefill_trace_chunked(self.mesh_device)
                # Traced short prefill for ANY prompt length < 2048 (QWEN36_PREFILL_BUCKET_TRACE, default 1): one B=1 trace
                # per bucket (QWEN36_PREFILL_TRACE_BUCKETS, default 128,256,512,1024,2048; 2048 serves lengths 1025..2047,
                # exactly 2048 tokens keeps the chunk trace). Captured HERE, with the SAME GDN binding the requests run
                # with (the scratch when batched, the decode buffers at max_batch_size==1), and only because this call
                # has enable_trace=True (the plugin's trace_mode == "all"). Block 0 is vLLM's null block, never
                # allocated to a request: padded K/V writes of a bucket land there.
                if short:
                    model.prefill_trash_block = 0
                    model.capture_prefill_traces_short(self.mesh_device, page_table)
                if grouped:
                    # One trace per (bucket, Bg) of the grouped body + one row-extract trace per (Bg, row), right after
                    # the chunk / short traces (everything they need was prepared in the eager pass).
                    model.prefill_trash_block = 0
                    model.record_prefill_trace_batched_short(self.mesh_device)
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
        if kwargs.get("enable_trace"):
            self._capture_gdn_slot_write_traces()
        if kwargs.get("enable_trace") and os.environ.get("QWEN36_FORBID_POST_CAPTURE_COMPILE") == "1":
            self.mesh_device.set_program_cache_misses_allowed(False)
            logger.info("post-capture program compiles are now forbidden")
        return result

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


_MESH_TP = {"N150": 1, "N300": 2, "P150x2": 2, "P150x4": 4, "N150x4": 4, "P150x8": 8, "T3K": 8}


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
    model_capabilities = {
        **Qwen36ForCausalLM.model_capabilities,
        # prefill_forward runs every prompt from position 0 (the engine keeps no GDN prefix snapshots)
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose", "hidden_feed"),
        "spec_hidden_handoff": ("on_device",),
        "supports_async_decode": False,
        "supports_async_spec_decode": False,
        "output_tokens_per_step": 1,
        "supports_sample_on_device": False,
        # Device grammar masks the device sampler, which the spec path does not use.
        "supports_device_grammar": False,
    }

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Config-time plan; never raises and never touches devices."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        k = cls._SPEC_K
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
            from models.demos.blackhole.qwen36.tt.spec_engine import paged_kv_dtype_bytes, spec_memory_costs

            mc = vllm_config.model_config
            tc = mc.hf_config
            tc = tc.get_text_config() if hasattr(tc, "get_text_config") else tc
            if int(getattr(tc, "mtp_num_hidden_layers", 0) or 0) < 1:
                return SpecReject(reason="checkpoint has no MTP weights (mtp_num_hidden_layers < 1)", supported_k=())
            costs = spec_memory_costs(
                tc, _mesh_tp(), max_num_seqs, k, max_seq_len=mc.max_model_len, kv_dtype_bytes=paged_kv_dtype_bytes()
            )
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
        return SpecPlan(
            effective_k=k,
            lanes_per_request=1,
            extra_bytes_per_seq=per_seq,
            extra_bytes_per_token=per_token,
            accept_modes=("argmax_ids", "logits"),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs):
        self = super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs)
        model = self.model[0]
        assert model.mtp is not None, "MTP head not built (checkpoint has no MTP weights?)"
        model.set_gdn_fused_decode(True)
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

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
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
        elif not eng.prepared:
            eng.prepare_decode()

    def _ensure_ready(self, kv_cache=None):
        eng = self._engine()
        if eng.prefill_recorded and eng.captured:
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
        return eng

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        N = tokens.shape[0]
        start_pos = kwargs.get("start_pos")
        assert start_pos is None or all(int(s) == 0 for s in start_pos), "no chunked prefill / prefix caching"
        if self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs, "pixel_values_videos"):
            raise ValueError("Qwen36MTPContractForCausalLM is text-only; image/video requests are not supported")
        eng = self._ensure_ready(kv_cache)
        slots = [int(x) for x in kwargs.get("empty_slots") or range(N)]
        table = self._table(page_table[:N])
        rows = []
        for i in range(N):
            L = int(prompt_lens[i]) if prompt_lens is not None else tokens.shape[1]
            rows.append(eng.prefill(slots[i], tokens[i, :L], table[i : i + 1]).float().reshape(-1))
        return torch.stack(rows).reshape(N, 1, -1), torch.zeros(N, dtype=torch.long)

    def decode_forward(self, *args, **kwargs):
        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        def _read(name, pos):
            return kwargs[name] if name in kwargs else (args[pos] if pos < len(args) else None)

        tokens, start_pos, page_table = _read("tokens", 0), _read("start_pos", 1), _read("page_table", 2)
        remap = _read("slot_remap", 9)
        narrow = "spec_mode" not in kwargs
        if narrow:
            return self._decode_narrow(tokens, start_pos, page_table, kwargs, remap)
        num_valid_drafts = kwargs.pop("num_valid_drafts")
        accepted_counts = kwargs.pop("accepted_counts")
        spec_mode = kwargs.pop("spec_mode")
        check_spec_side_tensors(num_valid_drafts, accepted_counts, int(tokens.shape[0]), self._SPEC_K)
        assert kwargs.get("page_tables_per_layer") is None, "per-layer page tables unsupported"
        eng = self._ensure_ready()
        res = eng.decode_forward(
            tokens,
            start_pos,
            num_valid_drafts=num_valid_drafts,
            accepted_counts=accepted_counts,
            page_table=self._table(page_table),
            spec_mode=spec_mode,
            slot_remap=remap,
        )
        logits = res.logits if spec_mode == "logits" else None
        return VerifyOutput(spec_mode=res.spec_mode, argmax_ids=res.argmax_ids, hidden=None, logits=logits)

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
