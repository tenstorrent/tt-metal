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
from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch
from models.tt_transformers.tt.generator import Generator

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

    # supports_async_decode=False: async decode assumes on-device token/position continuity, which
    # corrupts Qwen's GDN scan. supports_sample_on_device=True: on-device sampling is decode-only.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }

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
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, hf_model=name_or_path
        )
        # Attach the TT vision tower so prefill can splice image/video embeddings (multimodal path).
        # No-op cost for text-only requests; get_image_features / get_video_features are only invoked
        # when a request actually carries pixel_values / pixel_values_videos.
        model.init_vision_model()
        return cls([model], [args], mesh_device)

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
        model = self.model[0]
        if model.num_devices > 1 and model.args.max_batch_size > 1:
            # Batched serving (max_num_seqs > 1): prefill each request in this step into its decode
            # slot. Text-only — multimodal is single-sequence (get_supported_mm_limits: B=1). Check
            # for REAL visual data (not just non-None): vLLM passes an empty pixel_values placeholder
            # on text requests to a multimodal-registered model, which a bare `is None` check would
            # misflag and crash the engine on every text request.
            assert not self._has_visual(kwargs, "pixel_values") and not self._has_visual(
                kwargs, "pixel_values_videos"
            ), (
                "batched (max_num_seqs>1) serving is text-only; multimodal is single-sequence "
                "(max_concurrency=1). Run the model at max_num_seqs=1 for image/video requests."
            )
            return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"))
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
        logits = model.prefill_traced_chunked(
            tokens, page_table, actual_len=T, vision_tokens=vision_tokens
        )  # [1,1,vocab] replicated
        logits = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            .reshape(-1, model.args.vocab_size)[:1]
            .float()
            .view(1, 1, -1)
        )
        logger.info(f"Finished prefill up to {T} tokens, starting decode...")
        return logits, torch.zeros(1, dtype=torch.long)

    def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots):
        """TP batched (max_num_seqs>1) prefill: prefill each request in this step into its decode slot.

        vLLM prefills new requests while other slots decode, so each user's B=1 state is written into
        row empty_slots[u] of the batched GDN buffers without disturbing the live rows (model-owned,
        via prefill_paged_slots). Attention fills each request's blocks via its page-table row.

        tokens:      torch [N, max_T] (rows are the N requests scheduled this prefill step).
        page_table:  torch [N, max_blocks] — row u = request u's blocks.
        prompt_lens: per-request real lengths (row u trimmed to prompt_lens[u]).
        empty_slots: per-request decode slot; defaults to range(N) (mirrors Generator.prefill_forward_text).
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
        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)
        logits = torch.cat([hl.reshape(1, 1, -1) for hl in host_logits], dim=0)  # [N, 1, vocab]
        logger.info(f"Finished batched prefill of {N} user(s), starting decode...")
        return logits, torch.zeros(N, dtype=torch.long)

    def decode_forward(self, *args, **kwargs):
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
        if model.num_devices > 1 and model.args.max_batch_size > 1:
            slot_remap = kwargs.get("slot_remap")
            if slot_remap is not None:
                model._remap_gdn_slots(slot_remap)
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # Capture the chunk-prefill trace + warm the masked-bucket set so requests only replay
        # pre-compiled programs (compile-clobbers-trace fix). Guard name must match the plugin's reset.
        if not enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        # Size the chunk-trace page table to the full KV cache (not a hardcoded 4096) so served ISL
        # isn't capped; still captures one chunk — just a bigger page-table tensor.
        if kv_cache:
            # Round to a multiple of 32: paged/chunked SDPA needs the page-table stick % 32 == 0.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        model = self.model[0]
        # Batched serving (max_num_seqs>1): the decode buffers are [B,...], but prefill runs B=1. Bind
        # the PERSISTENT B=1 GDN prefill scratch and capture the chunk trace against IT, so long prompts
        # (>chunk_size) replay the traced chunk-outer path per user instead of the slower eager fallback.
        # The scratch is not freed (prefill_paged_slots rebinds it per request); the batched decode
        # buffers are restored before the decode-trace warmup captures at [B,...].
        batched = model.num_devices > 1 and model.args.max_batch_size > 1
        logger.info(
            f"Starting Qwen prefill warmup: chunk-prefill trace{' (batched, B=1 scratch)' if batched else ''} "
            f"(chunk={_PREFILL_WARMUP_CHUNK}, page_table_blocks={num_blocks})..."
        )
        prev = model._bind_gdn_prefill_scratch() if batched else None
        try:
            model.capture_prefill_trace_chunked(
                self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK, capture_chunk_trace=True
            )
        finally:
            if prev is not None:
                model._unbind_gdn_prefill_scratch(prev)

    def warmup_model_decode(self, *args, **kwargs):
        # Defer to WarmupForwardMixin, which captures the paged-SDPA + GDN decode trace at pos 0.
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        return super().warmup_model_decode(*args, **kwargs)
