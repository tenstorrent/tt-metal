# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local vLLM wrapper for Qwen3.5-9B: a thin tt_transformers Generator subclass.

Qwen3.5-9B is a hybrid model: 8 full-attention layers (paged KV, stateless across prefill) plus
24 Gated DeltaNet (GDN) layers carrying a recurrent + conv state that accumulates across the whole
sequence. Standard tt_transformers models are stateless beyond paged KV, so the standard contract
assumes token-padding is numerically free and the decode trace is position-general. Neither holds
for GDN — which is the root of every place this model must diverge.

What conforms to the standard Generator (Llama/DeepSeek/Qwen-VL):
  - Decode, end to end. The model implements the standard decode contract (prepare_inputs_decode /
    ttnn_decode_forward / process_output_decode); current_pos and page_table are device input
    tensors the standard replay updates per step. The inherited WarmupForwardMixin captures the
    decode trace at position 0 during warmup; Generator.decode_forward replays it at serving.

What must diverge, and why:
  - Prefill bucketing. GDN forbids token-padding inside its recurrent scan, so prefill pads to a
    fixed bucket and passes an EXACT valid_len (the standard contract only plumbs get_last_token,
    floored to a 32-multiple — too lossy for the GDN mask). See model.prefill_masked_bucket.
  - Chunk-outer trace. At 128K a whole-length prefill trace is infeasible; we capture ONE
    2048-token chunk trace and replay it N times, carrying GDN/KV state in place. See
    model.prefill_traced_chunked / capture_prefill_trace_chunked.
  - State-reset guard. The stock trace capture runs the forward twice (compile + capture), which
    advances GDN state non-idempotently. Harmless only because every new sequence re-zeros the
    bound GDN buffers before consuming a token (model._reset_gdn_state_for_new_sequence).

Generator drives decode; all prefill is model-owned (prefill_masked_bucket / prefill_traced_chunked)
via prefill_dispatch. GDN recurrent state and the attention KV caches are model-bound, so the
kv_cache contract param is accepted but unused.
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

    model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}

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
        """All-user KV-cache token capacity = served context length (B=1).

        Qwen3.5/3.6 serve one sequence at a time (max_concurrency=1), so the whole
        paged KV cache belongs to a single user and its capacity is exactly the
        served context length — max_model_len, i.e. the catalog's max_context.
        Deriving from max_model_len, instead of the inherited 131072 fallback, lets
        these models serve at the full requested ISL (e.g. 256K = 262144): the
        chunk-outer prefill and the full-KV page-table sizing in
        warmup_model_prefill already scale to whatever KV cache vLLM allocates.
        The * max_num_seqs keeps the all-user semantics correct if B ever grows.
        """
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
        # Resolution order: MODEL_WEIGHTS_DIR (tt-inference-server Docker convention) →
        # HF_MODEL → vLLM's hf_config._name_or_path (the hub id). Resolve a hub id to a
        # local snapshot dir (AutoConfig on a bare hub id is unreliable in this transformers).
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
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs."""
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

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

    def decode_forward(self, *args, **kwargs):
        # Both single-device and TP serve TRACED decode. The decode trace is valid for TP
        # because GDN state lives in fixed in-place buffers (reset_state_inplace preserves
        # addresses), and it no longer collides with prefill: TP prefill runs the bounded,
        # pre-warmed masked-bucket program set (warmed before the trace parks), so a request
        # never compiles a new program that could clobber the parked decode trace.
        # Standard path (default): the decode trace is captured at position 0 during warmup by the
        # inherited WarmupForwardMixin, then replayed here by Generator.decode_forward — identical
        # to Llama/DeepSeek/Qwen-VL. current_pos and page_table are device input tensors the
        # standard replay updates per step, so a pos-0 capture is position-general.
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen)")
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # Single-device AND TP share this path: capture_prefill_trace_chunked dispatches to its
        # TP fork (_capture_prefill_trace_chunked_tp) when num_devices>1. The capture compiles the
        # per-chunk programs AND warms the bounded masked-bucket program set (short prompts + the
        # long-prompt tail) before the decode trace is parked, so a real request only ever replays
        # already-compiled programs — the compile-clobbers-trace fix — and long prompts replay the
        # chunk-outer trace (bounded host dispatch, the 128K path) instead of the eager fallback.
        #
        # The plugin's warmup_model() is two-phase: it calls this first with
        # enable_trace=False (compile), then resets ``already_warmed_up_prefill``
        # and calls again with enable_trace=True (capture). Only the traced phase
        # captures the chunk-prefill trace; capture_prefill_trace_chunked compiles
        # its own programs before capturing, so the non-traced phase is a no-op.
        # The guard attribute MUST be named ``already_warmed_up_prefill`` so the
        # plugin's between-phase reset (model_runner.warmup_model) clears it.
        if not enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        # Size the captured chunk-trace page table to the FULL allocated KV cache
        # (max_model_len worth of blocks), so served ISL matches the tt-metal demo's
        # 128K — not a hardcoded 4096. The chunk-outer trace still captures only one
        # _PREFILL_WARMUP_CHUNK-token chunk, so this is just a larger page-table
        # tensor, not more compute/trace memory. kv_cache[0][0] is the first attention
        # layer's K cache, shape [max_num_blocks, n_kv_heads, block_size, head_dim].
        if kv_cache:
            # Round up to a multiple of 32: the paged/chunked SDPA requires the page-table
            # width (stick size) to be % 32 == 0 (the allocated block count carries a slack
            # block, e.g. 257, which is not 32-aligned). prefill_traced_chunked pads each
            # request's page table up to this width before replay.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        logger.info(
            f"Starting Qwen prefill warmup: capturing chunk-prefill trace "
            f"(chunk={_PREFILL_WARMUP_CHUNK}, page_table_blocks={num_blocks})..."
        )
        self.model[0].capture_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)

    def warmup_model_decode(self, *args, **kwargs):
        # Standard path (default): defer to the inherited WarmupForwardMixin, which captures the
        # paged-SDPA + GDN decode trace at position 0 during warmup. Qwen sets
        # _supports_on_device_sampling=False, so the orchestrator passes can_sample_on_device=False
        # and exactly one greedy trace is captured; serving replays it with per-step input updates.
        #
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        return super().warmup_model_decode(*args, **kwargs)
