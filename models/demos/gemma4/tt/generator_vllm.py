# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import time
from collections import OrderedDict, defaultdict

import torch
from loguru import logger
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedProcessingInfo
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.config import gemma4_kv_bfp8_enabled
from models.demos.gemma4.tt.attention import _RING_HEADROOM_BLOCK, SPEC_RING_HEADROOM_ENV
from models.demos.gemma4.tt.common import (
    GEMMA4_CP_PREFILL_CHUNK,
    create_tt_model,
    gemma4_cp_prefill_engaged,
    gemma4_env_flag,
)
from models.demos.gemma4.tt.dflash_constants import VERIFY_WIDTH_MARGIN
from models.demos.gemma4.tt.generator import (
    SDPA_CHUNK_ALIGN,
    ChunkedPrefillPageTableGuardMixin,
    _build_vision_state_dict,
    align_num_cached_tokens_to_sdpa,
    filter_warmup_sampling_params,
    gemma4_packed_prefill_enabled,
    items_within_prefill_window,
    mask_page_table_columns_past_allocation,
    max_batched_prefill_users,
    merge_soft_tokens,
    resolve_batched_prefill_chunk_users,
    vision_state_dict_for,
)
from models.demos.gemma4.tt.generator_trace import (
    batched_prefill_trace_bucket,
    maybe_disable_pli_prefill_trace,
    patch_gemma4_trace_model_args,
    resolve_gemma4_prefill_chunk_size,
    resolve_gemma4_prefill_trace_enable,
    should_auto_enable_bounded_sliding,
    trace_short_prefill_max,
    trace_short_prefill_opt_in,
    warmup_gemma4_model_prefill,
)
from models.demos.gemma4.tt.lane_prefill import (
    lane_kv_pool_blocks,
    lane_prefill_batched_enabled,
    lane_prefill_table_upload_enabled,
    lane_prefill_trace_buckets,
    lane_prefill_trace_plan,
    lane_trace_rounds_for_call,
    lane_unique_block_rows,
)
from models.demos.gemma4.tt.mesh_orientation import should_reshape_to_column
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.spec_policy import (
    SPEC_ADAPTIVE_ENV,
    SPEC_PLAIN_STEP_MS_ENV,
    CostLine,
    RecordCandidate,
    SpecPolicy,
    SpecPolicyConfig,
    cost_at,
    describe_costs,
)
from models.demos.gemma4.tt.vision.embedders import (
    Gemma4MultimodalEmbedder,
    Gemma4UnifiedAudioEmbedder,
    Gemma4UnifiedVisionEmbedder,
    has_unified_audio_weights,
    has_unified_vision_weights,
)
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.demos.gemma4.tt.vision.vision_tower import VisionTower
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES, create_submeshes
from models.tt_transformers.tt.generator_vllm import (
    HybridAttentionForCausalLM,
    allocate_vllm_kv_cache,
    allocate_vllm_kv_cache_per_layer,
)


def _vllm_force_full_isl_single_chunk() -> bool:
    """Opt into full-ISL single-chunk prefill (legacy / debug).

    Default OFF: with tenstorrent/vllm#448 the TT scheduler supports
    token-chunked prefill, so the server mirrors metal's policy chunk path
    (``resolve_gemma4_prefill_chunk_size``, typically 4096 on P150x8 / QB2).

    ``GEMMA4_VLLM_SINGLE_CHUNK`` is the server override and wins when set:
      * ``0``/false → policy multi-chunk (even if ``GEMMA4_DEMO_SINGLE_CHUNK=1``
        leaked into the process env from a prior demo shell)
      * ``1``/true  → force full-ISL single-chunk
    If unset, ``GEMMA4_DEMO_SINGLE_CHUNK=1`` still forces single-chunk for
    demo/server parity experiments.
    """
    vllm_sc = os.environ.get("GEMMA4_VLLM_SINGLE_CHUNK")
    if vllm_sc is not None:
        return vllm_sc.lower() in ("1", "true", "yes")
    return os.environ.get("GEMMA4_DEMO_SINGLE_CHUNK", "0") != "0"


def _full_isl_prefill_chunk_size(max_seq_len: int) -> int:
    """Largest full-ISL single-chunk size that fits ``max_model_len``.

    Prefill pads to the next power-of-2 bucket (``get_padded_prefill_len``).
    For non-pow2 pools (e.g. QB2 31B ``max_context=49152``) the largest valid
    single-chunk length is the floor power-of-2 (32768) — not 49152 (pads to
    65536 > pool) and not ceil-pow2 65536.
    """
    max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        return 1 << 11
    # Floor power-of-2, minimum 2^11 (smallest prefill bucket used elsewhere).
    floor_pow2 = 1 << max(max_seq_len.bit_length() - 1, 11)
    return min(floor_pow2, max_seq_len)


class _Gemma4VllmOptimizations:
    @staticmethod
    def get_tensor_dtype(decoder_id, tensor, prefetcher=False):
        del decoder_id, prefetcher
        # Serving KV pools read this override; Gemma4Model.__init__'s bfp8 env
        # gate only covers the demo path.
        if gemma4_kv_bfp8_enabled():
            from models.tt_transformers.tt.model_config import TensorGroup

            if tensor == TensorGroup.KV_CACHE:
                return ttnn.bfloat8_b
        return ttnn.bfloat16


def _gemma4_prefill_trace_unsafe(model, bounded_sliding_kv_cache) -> bool:
    """True when the hybrid bridge feeds *non-uniform* per-layer page tables
    to the paged ops, so a prefill-trace capture must run *through* the
    per-layer page-table routing rather than the plain ``prefill_forward_text``.

    A direct ``prefill_forward_text`` capture binds the traced paged ops to the
    single full page_table shared by every layer. That only matches runtime
    when every layer truly uses that one table. It diverges — and the captured
    trace then addresses the wrong KV slots, corrupting prefill output —
    whenever:

      * bounded sliding is on and the model has ``sliding_attention`` layers
        (:meth:`_pad_sliding_page_tables_for_bounded` widens only the sliding
        layers, so their table no longer matches the full layers'), or
      * the model kv-shares layers (``kv_shared_layer_map`` re-points a shared
        layer's table at its source's).

    When this returns True, :meth:`warmup_model_prefill` routes the warmup
    capture through :meth:`prefill_forward` (which populates the persistent
    per-layer buffers before capture) — exactly how decode warmup routes
    through ``decode_forward``. Models without sliding layers (or with bounded
    sliding off and no kv-share) can capture directly via
    ``prefill_forward_text``, so the gate is structural and self-scoping
    rather than a hard-coded model list.
    """
    if getattr(model, "kv_shared_layer_map", None):
        return True
    # ``Gemma4Model`` stores the *text* config directly as ``hf_config`` and
    # reads ``self.hf_config.layer_types`` in forward, so look there first;
    # only fall back to a nested ``text_config`` if the top level lacks the
    # field (some unified/multimodal configs nest it).
    hf_config = getattr(model, "hf_config", None)
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        text_config = getattr(hf_config, "text_config", None)
        layer_types = getattr(text_config, "layer_types", None)
    layer_types = list(layer_types or [])
    has_sliding = "sliding_attention" in layer_types
    has_full = "full_attention" in layer_types
    # Mixed sliding + full layers ⇒ vLLM's hybrid kv-cache manager builds
    # multiple kv-cache groups and HMA tensor-sharing packs layers from
    # different groups into one physical KV buffer, indexed by *distinct*
    # per-layer page tables (different block IDs into the shared buffer). The
    # prefill-trace warmup captures a single broadcast table for every layer,
    # so shared layers collide on the same slots and corrupt the KV cache on
    # replay — independent of bounded sliding. Bounded sliding adds further
    # per-layer width divergence (sliding tables padded to the window) on top.
    if has_sliding and has_full:
        return True
    if bounded_sliding_kv_cache and has_sliding:
        return True
    return False


def _resolve_vllm_bounded_sliding(max_seq_len, mesh_device, model_path, *, hybrid_groups_enabled: bool) -> bool:
    """Mirror demo: auto policy + ``GEMMA4_BOUNDED_SLIDING_KV_CACHE`` / legacy env."""
    # Hybrid groups default bounded ON, except under prefix caching, whose
    # hybrid substrate is vLLM-managed window blocks (rings OFF).
    _prefix_caching = os.environ.get("GEMMA4_PREFIX_CACHING", "0") != "0"
    _bounded_default = "1" if hybrid_groups_enabled and not _prefix_caching else None
    _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING_KV_CACHE")
    if _bs_env is None and _bounded_default is not None:
        _bs_env = _bounded_default
    if _bs_env is None:
        # Also accept GEMMA4_BOUNDED_SLIDING (demo alias) when unset.
        _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
    if _bs_env is None:
        return should_auto_enable_bounded_sliding(max_seq_len, mesh_device, model_path)
    return _bs_env.lower() in ("1", "true", "yes")


def _assert_within_native_context(max_seq_len: int, model_args) -> None:
    """Refuse a max_seq_len beyond gemma4's native RoPE window (no rope scaling).

    The serving templates waive vLLM's own ceiling check (``VLLM_ALLOW_LONG_MAX_MODEL_LEN=1``);
    ``GEMMA4_ALLOW_BEYOND_NATIVE=1`` opts back in.
    """
    text_config = getattr(model_args, "_hf_text_config", None)
    native = int(getattr(text_config, "max_position_embeddings", 0) or 0)
    if not native or max_seq_len <= native:
        return
    if os.environ.get("GEMMA4_ALLOW_BEYOND_NATIVE", "0").lower() in ("1", "true", "yes"):
        logger.warning(
            "Gemma4 vLLM: max_seq_len={} exceeds the native context {} and "
            "GEMMA4_ALLOW_BEYOND_NATIVE is set -- serving beyond the RoPE table "
            "is unvalidated.",
            max_seq_len,
            native,
        )
        return
    raise ValueError(
        f"max_seq_len {max_seq_len} exceeds gemma-4's native max_position_embeddings "
        f"({native}, no rope scaling). vLLM's ceiling check is disabled by "
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN in the serving templates, so this length would "
        "be accepted and then fail at runtime for any single request past the native "
        "window. Lower max_context / --max-model-len, or set "
        "GEMMA4_ALLOW_BEYOND_NATIVE=1 to accept unvalidated behavior."
    )


def _patch_model_args(
    model_args,
    mesh_device,
    max_batch_size,
    max_seq_len,
    model_path,
    prefill_trace_enabled=True,
    *,
    bounded_sliding=False,
):
    _assert_within_native_context(max_seq_len, model_args)
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    # Prefill chunking (two cooperating layers after tenstorrent/vllm#448):
    #   1) vLLM scheduler token-chunked prefill (enable_chunked_prefill +
    #      max_num_batched_tokens / long_prefill_token_threshold)
    #   2) metal generator max_prefill_chunk_size via policy
    #      (resolve_gemma4_prefill_chunk_size: default prefill_chunk, then
    #      optional prefill_chunk_by_isl for high-ISL tiers)
    #
    # Keep llm.yaml scheduler tokens aligned with the resolved metal chunk for
    # the configured max_context. Override with GEMMA4_GEN_PREFILL_CHUNK, or
    # force full-ISL single-chunk via GEMMA4_VLLM_SINGLE_CHUNK=1.
    chunk_override = int(os.environ.get("GEMMA4_GEN_PREFILL_CHUNK", "0"))
    if chunk_override <= 0 and gemma4_cp_prefill_engaged(mesh_device):
        chunk_override = GEMMA4_CP_PREFILL_CHUNK
    if chunk_override > 0:
        model_args.max_prefill_chunk_size = chunk_override
        logger.info(
            "Gemma4 vLLM: GEMMA4_GEN_PREFILL_CHUNK={} → max_prefill_chunk_size={}",
            chunk_override,
            model_args.max_prefill_chunk_size,
        )
    elif _vllm_force_full_isl_single_chunk():
        model_args.max_prefill_chunk_size = _full_isl_prefill_chunk_size(max_seq_len)
        logger.info(
            "Gemma4 vLLM: full-ISL single-chunk prefill "
            "(max_prefill_chunk_size={}, max_seq_len={}). "
            "Unset GEMMA4_VLLM_SINGLE_CHUNK to use policy multi-chunk.",
            model_args.max_prefill_chunk_size,
            max_seq_len,
        )
    else:
        model_args.max_prefill_chunk_size = resolve_gemma4_prefill_chunk_size(
            max_seq_len,
            mesh_device=mesh_device,
            # Measured P150x8 / QB2 policies supply default 4096 (+ ISL tiers);
            # other boards keep full-ISL until validated (same as demo).
            non_qb2_default=max_seq_len,
            model_name_or_path=model_path,
            bounded_sliding=bounded_sliding,
        )
        logger.info(
            "Gemma4 vLLM: policy prefill chunk={} "
            "(max_seq_len={}, bounded_sliding={}) — align vLLM "
            "max_num_batched_tokens / long_prefill_token_threshold to this value",
            model_args.max_prefill_chunk_size,
            max_seq_len,
            bounded_sliding,
        )
    patch_gemma4_trace_model_args(model_args, prefill_trace_enabled=prefill_trace_enabled)
    model_args.optimizations = _Gemma4VllmOptimizations()
    model_args.mesh_device = mesh_device
    model_args._gemma4_model_path = model_path
    model_args.is_llama_vision = lambda: False


def _gemma4_has_vision_weights(state_dict) -> bool:
    """True when the checkpoint ships a ``model.vision_tower.*`` sub-tree (SigLIP,
    31B) or the unified encoder-free embedder (``model.vision_embedder.*``, 12B)."""
    return has_unified_vision_weights(state_dict) or any(k.startswith("model.vision_tower.") for k in state_dict.keys())


def _build_gemma4_vision(
    mesh_device,
    state_dict,
    model_path,
    text_hidden_size,
    vision_dtype,
    mesh_config=None,
    ccl_manager=None,
    weight_cache_path=None,
):
    """Build the on-device vision tower + multimodal embedder for a Gemma-4
    multimodal checkpoint. Mirrors ``Gemma4Generator.from_pretrained(..., multimodal=True)``.

    Returns ``(vision_tower, embed_vision, image_token_id, pad_token_id)`` or
    ``(None, None, None, None)`` when the checkpoint is text-only.
    """
    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    image_token_id = getattr(hf_config, "image_token_id", None)
    if image_token_id is None:
        return None, None, None, None
    pad_token_id = getattr(getattr(hf_config, "text_config", hf_config), "pad_token_id", 0)
    os.environ.setdefault("HF_MODEL", model_path)
    # Warm text weight cache -> placeholder dict: read the vision tensors for real.
    state_dict = vision_state_dict_for(state_dict, model_path)
    if has_unified_vision_weights(state_dict):
        # 12B "unified": no SigLIP tower; the embedder projects straight to text space.
        embedder = Gemma4UnifiedVisionEmbedder(
            mesh_device,
            state_dict,
            text_hidden_size=text_hidden_size,
            weight_cache_path=weight_cache_path,
        )
        return embedder, None, image_token_id, pad_token_id
    # Share the text model's mesh/CCL setup so the vision tower's TP all-reduces
    # reuse the same semaphores and link/topology tuning.
    vision_args = VisionModelArgs(
        mesh_device,
        dummy_weights=True,
        max_batch_size=1,
        max_seq_len=8192,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
    )
    t0 = time.time()
    vision_state_dict = _build_vision_state_dict(state_dict, vision_args)
    logger.info(
        f"Gemma4 vLLM: vision tower state dict ready ({len(vision_state_dict)} tensors) in {time.time() - t0:.1f}s"
    )
    t0 = time.time()
    vision_tower = VisionTower(
        args=vision_args,
        dtype=vision_dtype,
        state_dict=vision_state_dict,
        tt_ccl=TT_CCL(mesh_device),
        weight_cache_path=vision_args.weight_cache_path(vision_dtype),
    )
    logger.info(f"Gemma4 vLLM: vision tower on device in {time.time() - t0:.1f}s")
    embed_vision = Gemma4MultimodalEmbedder.from_state_dict(
        mesh_device=mesh_device,
        state_dict=state_dict,
        vision_args=vision_args,
        text_hidden_size=text_hidden_size,
        dtype=ttnn.bfloat16,
        weight_cache_path=vision_args.weight_cache_path(ttnn.bfloat16),
    )
    return vision_tower, embed_vision, image_token_id, pad_token_id


def _build_gemma4_audio(mesh_device, state_dict, model_path, text_hidden_size, weight_cache_path=None):
    """Build the on-device encoder-free audio embedder of a unified Gemma-4
    checkpoint (12B). Returns ``(embed_audio, audio_token_id)`` or ``(None, None)``
    when the checkpoint carries no audio projection (31B: ``audio_config`` is None)."""
    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    audio_token_id = getattr(hf_config, "audio_token_id", None)
    audio_config = getattr(hf_config, "audio_config", None)
    if audio_token_id is None or audio_config is None:
        return None, None
    state_dict = vision_state_dict_for(state_dict, model_path)
    if not has_unified_audio_weights(state_dict):
        return None, None
    embedder = Gemma4UnifiedAudioEmbedder(
        mesh_device,
        state_dict,
        text_hidden_size=text_hidden_size,
        rms_eps=float(getattr(audio_config, "rms_norm_eps", 1e-6)),
        weight_cache_path=weight_cache_path,
    )
    return embedder, audio_token_id


# Multimodal fields the plugin forwards per batch slot (vLLM ``Gemma4MultiModalProcessor``
# names, plus the HF ``Gemma4Processor`` names the demo path uses). The payload keys
# decide whether a batch is multimodal; the bridge consumes all of them before the
# text prefill runs (``_stage_multimodal_prefill``) so none reaches ``prepare_inputs_prefill``.
_GEMMA4_MM_PAYLOAD_KEYS = ("pixel_values", "pixel_values_videos", "input_features", "input_features_padded")
_GEMMA4_MM_FIELD_KEYS = _GEMMA4_MM_PAYLOAD_KEYS + (
    "pixel_position_ids",
    "image_position_ids",
    "image_grid_thw",
    "image_sizes",
    "pixel_position_ids_videos",
    "video_position_ids",
    "video_frame_counts",
    "video_num_soft_tokens",
    "video_timestamps",
    "input_features_mask",
    # the per-item placeholder ranges the plugin forwards next to the payloads
    "mm_placeholder_offsets",
    "mm_placeholder_lengths",
)


def _mm_batched(t):
    """A per-item ``[P, D]`` processor tensor as the ``[1, P, D]`` batch the encoders take (frames stay ``[F, P, D]``)."""
    return t.unsqueeze(0) if t.dim() == 2 else t


def _mm_value_present(value) -> bool:
    """True when a forwarded multimodal field carries a payload for some slot (a text
    batch on a multimodal class still carries ``pixel_values=[None] * B``)."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_mm_value_present(v) for v in value)
    return True


def _gemma4_has_mm_inputs(kwargs) -> bool:
    """True when the plugin forwarded an image (``pixel_values``), video
    (``pixel_values_videos``) or audio (``input_features``; vLLM's own parser spells it
    ``input_features_padded``) payload for any user of this prefill batch."""
    return any(_mm_value_present(kwargs.get(key)) for key in _GEMMA4_MM_PAYLOAD_KEYS)


def _per_slot_items(kwargs, key, batch_size):
    """The per-user lists of one forwarded multimodal field, padded with None to the batch."""
    value = kwargs.get(key)
    items = list(value) if value is not None else []
    return items + [None] * (batch_size - len(items))


def gemma4_mm_restart_outside_vision_items(aligned_starts, tokens, offsets_per_user, lengths_per_user, vision_ids):
    """Move each user's SDPA-aligned restart out of any vision item it cuts.

    ``aligned_starts`` are the 128-aligned cached-prefix lengths (``align_num_cached_tokens_to_sdpa``); a
    restart inside an image or video placeholder run would recompute only the tail of that run, whose rows need
    bidirectional attention over the whole run (``use_bidirectional_attention="vision"``) and whose earlier rows
    sit in the cached prefix. The restart moves to the run's start, aligned down again, and repeats while it
    still cuts a run (an earlier item may end inside the new window). An item is a vision item when the prompt
    holds one of ``vision_ids`` at its offset; audio runs stay causal and may be cut. Returns a new list.
    """
    if not vision_ids or tokens is None:
        return list(int(n) for n in aligned_starts)
    vision = {int(v) for v in vision_ids}
    out = []
    for u, start in enumerate(aligned_starts):
        start = int(start)
        offsets = offsets_per_user[u] or ()
        lengths = lengths_per_user[u] or ()
        prompt = tokens[u]
        moved = True
        while start > 0 and moved:
            moved = False
            for off, length in zip(offsets, lengths):
                if off is None or length is None:
                    continue
                off, end = int(off), int(off) + int(length)
                if off < start < end and off < int(prompt.shape[-1]) and int(prompt[off]) in vision:
                    new = (off // SDPA_CHUNK_ALIGN) * SDPA_CHUNK_ALIGN
                    logger.info(
                        "Gemma4 vLLM: user {} restart {} cuts the vision item [{}, {}); restarting at {}",
                        u,
                        start,
                        off,
                        end,
                        new,
                    )
                    start = new
                    moved = True
                    break
        out.append(start)
    return out


def gemma4_mm_items_to_restore(aligned_starts, prompt_lens, offsets_per_user, lengths_per_user):
    """Users whose placeholder metadata names an item the restart must re-embed.

    vLLM strips the payload of every item inside a prefix-cache hit (``strip_covered_mm_data``) and the plugin
    forwards only its placeholder range; an item ending after the user's aligned restart and inside this
    call's window still needs its soft tokens (restored from the cache by ``_g4_modality_soft_tokens``). The
    payload predicate alone misses a batch whose every item was stripped.
    """
    users = []
    for u, start in enumerate(aligned_starts):
        offsets = offsets_per_user[u] or ()
        lengths = lengths_per_user[u] or ()
        window_end = int(prompt_lens[u]) if prompt_lens is not None else None
        for off, length in zip(offsets, lengths):
            if off is None or length is None:
                continue
            end = int(off) + int(length)
            if end > int(start) and (window_end is None or end <= window_end):
                users.append(u)
                break
    return users


def galaxy_lanes_fold():
    """``--data_parallel_size N`` runs as N in-process lanes on one Galaxy instance (GEMMA4_GALAXY_LANES=1)."""
    return gemma4_env_flag("GEMMA4_GALAXY_LANES")


def galaxy_lane_prefill_group():
    """Under the lane fold, lanes prefill as whole groups in one chunk walk (GEMMA4_LANE_PREFILL_ROUTER=1)."""
    return galaxy_lanes_fold() and gemma4_env_flag("GEMMA4_LANE_PREFILL_ROUTER")


def galaxy_lane_prefill_prompts():
    """Prompts per lane per gathered prefill step under the lane router: ``GEMMA4_LANE_PREFILL_PROMPTS`` (default
    1), 0 when the router is off. Declared through ``model_capabilities["lane_prefill_group"]``: the vllm-tt-plugin
    caps each lane's new prefill admissions per step at this count (a plugin that reads the declaration as a
    Boolean admits one per lane, as before), and the router prefills the gathered step as ``[lanes, k, seq]``
    lane-batched walks (see ``_try_lane_parallel_prefill``). A malformed value fails the launch."""
    if not galaxy_lane_prefill_group():
        return 0
    raw = os.environ.get("GEMMA4_LANE_PREFILL_PROMPTS", "1").strip() or "1"
    try:
        prompts = int(raw)
    except ValueError as exc:
        raise ValueError(f"GEMMA4_LANE_PREFILL_PROMPTS={raw!r} is not an integer") from exc
    if prompts < 1:
        raise ValueError(f"GEMMA4_LANE_PREFILL_PROMPTS={prompts} must be >= 1")
    return prompts


def lane_batched_warmup_shape(prompts, lane_slots, max_chunk):
    """``(k, seq)`` of the boot warmup's lane-batched call: up to ``prompts`` users per lane whose power-of-two
    per-user length keeps the stream ``k * seq`` within the prefill chunk (``k == 1`` when there is nothing to
    warm). Serving calls then vary ``k`` and ``seq`` under the same page-table frame and chunk bound."""
    k = max(1, min(int(prompts or 0), int(lane_slots or 1)))
    if k <= 1 or int(max_chunk) < 2 * 128:
        return 1, int(max_chunk)
    seq = 128
    while seq * 2 * k <= int(max_chunk):
        seq *= 2
    k = max(1, min(k, int(max_chunk) // seq))
    return k, seq


def galaxy_lane_prefill_capability():
    """What the class DECLARES to the vllm-tt-plugin as ``lane_prefill_group``: the prompts-per-lane count
    (``galaxy_lane_prefill_prompts``), or 0 under GEMMA4_LANE_PREFILL_CAPABILITY=0 -- the plugin then applies no
    per-lane admission cap and no gather (a step carries whatever waits, bounded by max_num_batched_tokens) while
    the model's own router and its prompts-per-lane policy stay on (window 10, 2026-10-24: does the lane-batched
    frame form from uncapped steps without any plugin change)."""
    if os.environ.get("GEMMA4_LANE_PREFILL_CAPABILITY", "1").strip() == "0":
        return 0
    return galaxy_lane_prefill_prompts()


def _capability_flag(capabilities, key):
    """A Boolean ``model_capabilities`` entry or a zero-argument callable one (the plugin resolves the lane fold the
    same way; ``lane_prefill_group`` is read only by this class)."""
    value = (capabilities or {}).get(key, False)
    return bool(value() if callable(value) else value)


def _capability_count(capabilities, key):
    """An integer ``model_capabilities`` entry (a Boolean counts 0 / 1) or a zero-argument callable one."""
    value = (capabilities or {}).get(key, False)
    if callable(value):
        value = value()
    if isinstance(value, bool):
        return int(value)
    return max(0, int(value))


def lane_prefill_rounds(buckets, plens, *, allow_scratch):
    """Rounds of a merged prefill step: one request per lane, equal prompt lengths within a round.

    ``buckets[lane]`` lists the step's request indices on that lane in admission order; ``plens[q]`` is request
    ``q``'s prompt length. A round is one entry per lane: a request index, or None for a lane that computes on
    scratch rows that round. Without scratch rows only rounds covering every lane form, in bucket order, stopping
    at the first round whose lengths differ (the one-prompt-per-lane layout; the step's other requests prefill
    serially). With them every request is routed: for each prompt length in order of first appearance, the lanes'
    requests of that length pair up round by round and short lanes get scratch rows.
    """
    if not allow_scratch:
        rounds = []
        r = 0
        while all(len(b) > r for b in buckets):
            group = [b[r] for b in buckets]
            if len({plens[q] for q in group}) != 1:
                break
            rounds.append(group)
            r += 1
        return rounds
    all_q = sorted(q for b in buckets for q in b)
    lengths = []
    for q in all_q:
        if plens[q] not in lengths:
            lengths.append(plens[q])
    rounds = []
    for length in lengths:
        per_lane = [[q for q in b if plens[q] == length] for b in buckets]
        for r in range(max(len(x) for x in per_lane)):
            rounds.append([x[r] if r < len(x) else None for x in per_lane])
    return rounds


def lane_prefill_frame_tables(per_layer_tables, rounds, lane_slots, scratch_block_for):
    """Per-layer host page tables for one lane-parallel call of ``k = len(rounds)`` rounds.

    ``k == 1``: the one-row-per-lane layout (row ``lane``; its own persistent key, one device row per column).
    ``k > 1``: the lanes' decode frame (``lanes * lane_slots`` rows, lane ``ln``'s user ``r`` at row
    ``ln * lane_slots + r``), which lane-shards to ``[lane_slots, blocks]`` per column with the call's users at
    rows ``0..k-1`` (the rows a batched walk fills) and keys the frame's persistent buffers, so no call size
    allocates a new key under the live decode traces. Request rows copy the step's table row ``q``; a lane's
    scratch round holds the layer's scratch block (``scratch_block_for(layer)``, what the single-user lane prefill
    gives its non-owner lanes); every other frame row is null.
    """
    k = len(rounds)
    lanes = len(rounds[0])
    out = []
    for i, pt in enumerate(per_layer_tables):
        if pt is None or not isinstance(pt, torch.Tensor):
            out.append(pt)
            continue
        pt2 = pt if pt.dim() > 1 else pt.unsqueeze(0)
        width = int(pt2.shape[-1])
        scratch = int(scratch_block_for(i))
        if k == 1:
            rows = torch.full((lanes, width), scratch, dtype=torch.int32)
            for ln, q in enumerate(rounds[0]):
                if q is not None:
                    rows[ln] = pt2[q].to(torch.int32)
        else:
            rows = torch.zeros((lanes * int(lane_slots), width), dtype=torch.int32)
            for r, rnd in enumerate(rounds):
                for ln, q in enumerate(rnd):
                    row = ln * int(lane_slots) + r
                    if q is None:
                        rows[row] = scratch
                    else:
                        rows[row] = pt2[q].to(torch.int32)
        out.append(rows)
    return out


class Gemma4ForCausalLM(ChunkedPrefillPageTableGuardMixin, HybridAttentionForCausalLM):
    """Gemma4 — hybrid attention (sliding-window + full).

    Gemma4's decoder alternates ``sliding_attention`` and ``full_attention``
    layers per ``hf_config.layer_types``, so the bridge inherits from
    :class:`HybridAttentionForCausalLM` to opt into vLLM's hybrid kv cache
    manager. ``get_kv_cache_spec`` is inherited; layer-routed page tables
    flow through the model's ``_active_page_tables_per_layer`` stash and
    are picked up inside ``Gemma4Model.{ttnn_prefill_forward,
    ttnn_decode_forward}`` (mirrors the gpt-oss bridge).
    """

    #: Speculating subclasses that stop proposing before a verify would wrap the
    #: bounded ring set this; they are advised rather than refused at startup.
    _SPEC_DECLINES_AT_RING_WRAP = False

    decode_input_update_contract = 1

    # Async decode closes the ~15–20% metal↔server B=1 gap (#51186): with
    # ``async_scheduling`` the plugin overlaps CPU scheduling with the previous
    # device step via ``decode_forward(read_from_device=False)`` +
    # ``read_decode_output(async_read=True)`` (inherited from ``Generator``).
    # Requires on-device token feedback + position plus_one (non-PLI only;
    # see ``Gemma4Model._tt_vllm_always_refresh_decode_trace_inputs``).
    #
    # Default ON for non-PLI. The plugin drains and applies pending results
    # before commanding a host-authoritative reload. Kill-switch:
    # ``GEMMA4_SUPPORTS_ASYNC_DECODE=0``. PLI models narrow the instance dict
    # in ``__init__``; that does not reach the platform, so PLI still needs
    # the kill-switch to disable async_scheduling.
    model_capabilities = {
        # vLLM prefix caching (default OFF): needs vLLM-managed sliding KV, so
        # __init__ refuses it with bounded rings; spec rails pin it False.
        "supports_prefix_caching": os.environ.get("GEMMA4_PREFIX_CACHING", "0") != "0",
        "supports_async_decode": os.environ.get("GEMMA4_SUPPORTS_ASYNC_DECODE", "1").lower() in ("1", "true", "yes"),
        # Gemma4ModelArgs exposes no get_attn_sdpa_program_config, so Generator
        # cannot derive the resume offset alignment and must be told it. Same pin
        # align_num_cached_tokens_to_sdpa applies locally.
        "resumed_prefill_token_alignment": SDPA_CHUNK_ALIGN,
        "supports_sample_on_device": True,
        # prefill_forward_text routes a nonzero start_pos to the chunked SDPA and
        # floors the offset to resumed_prefill_token_alignment, so a prompt split
        # across engine steps needs no new prefill code.
        "supports_chunked_prefill": True,
        # Lane serving (vllm-tt-plugin ``_convert_dp_to_lanes``): an env-gated rail,
        # so the plugin evaluates this zero-argument callable at config time.
        "folds_dp_into_lanes": galaxy_lanes_fold,
        # The plugin contract: prompts per lane per gathered step (an int: 0 off, k = up to k
        # new prefills per lane per step). The plugin caps each lane's admissions at this count
        # (a Boolean reading admits one) and gathers lanes briefly (``TT_PREFILL_GROUP_GATHER_MS``);
        # GEMMA4_LANE_PREFILL_CAPABILITY=0 declares 0 (no cap, no gather) with the router kept on.
        "lane_prefill_group": galaxy_lane_prefill_capability,
        # This class's own switches (not plugin contracts): the lane-parallel prefill router and
        # the prompts-per-lane policy it builds its calls with.
        "lane_prefill_router": galaxy_lane_prefill_group,
        "lane_prefill_prompts": galaxy_lane_prefill_prompts,
    }

    # vLLM pads decode to the nearest of these (not always max_num_seqs) so B=1
    # recovers the metal demo SDPA/matmul path (~27 tok/s/user vs ~20 at B=32).
    #
    # This is the *candidate* set. ``warmup_model_decode`` narrows it in place to
    # the buckets it actually captured a decode trace for, because the plugin
    # treats this attribute as the whole contract: padding to a bucket with no
    # captured trace leaves the device in an undefined state (on Wormhole a
    # fast-dispatch hang). Do not publish a wider list than what is warmed.
    tt_supported_decode_batch_sizes = SUPPORTED_PREFILL_BATCH_SIZES

    # Set True at the end of each ``warmup_model_decode`` call (the plugin runs
    # an eager pass and a trace pass), once every bucket of that call ran.
    # Previously the presence of ``tt_warmed_decode_batch_sizes`` doubled as
    # this sentinel; that attribute is gone, so the signal is now explicit.
    _decode_warmup_complete = False
    # True while ``warmup_model_decode`` runs its buckets: a page-table buffer
    # realloc inside that loop (the wider bucket growing the buffers the first
    # bucket allocated) is part of warmup, not a serving-time event.
    _g4_decode_warmup_active = False

    # Hybrid vLLM kv-cache groups: env-gated via ``GEMMA4_HYBRID_KV_CACHE_GROUPS``
    # (default OFF). Toggle from the tt-inference-server model-spec env so the KV
    # mode is config-driven and reversible without a code change.
    #
    # OFF (default): ``get_kv_cache_spec`` emits ``FullAttentionSpec`` for *every*
    # layer, which vLLM merges into a single ``UniformTypeKVCacheSpecs`` group, so
    # the whole block pool backs each request and the full ``max_model_len`` is
    # admissible (verified ~100K ISL). Sliding layers allocate full-length KV
    # unless ``bounded_sliding`` is on — then :meth:`allocate_kv_cache_per_layer`
    # shrinks them to ``sliding_window/block_size * max_batch`` (demo parity) and
    # :meth:`_pad_sliding_page_tables_for_bounded` remaps page tables to dense
    # local block IDs. Without that shrink, auto-bounded 256k OOMs on 31B.
    #
    # ON (``GEMMA4_HYBRID_KV_CACHE_GROUPS=1``): sliding layers emit
    # ``SlidingWindowSpec`` and form their own kv_cache_groups, so the 40 sliding
    # layers hold only the 1024-token window (vLLM window blocks, or device rings under bounded sliding);
    # a request is capped at ~num_blocks // num_groups; full-attn layers share the buffer via effective_block_size.
    #
    # Prefix caching (GEMMA4_PREFIX_CACHING) needs the vLLM-managed window blocks:
    # bounded rings are per-request device state vLLM cannot hit, so __init__ refuses the combination.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = os.environ.get("GEMMA4_HYBRID_KV_CACHE_GROUPS", "0") != "0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Multimodal (vision) components. Populated by ``initialize_vllm_model`` when
        # the checkpoint ships vision weights (auto-enable, mirroring the demo's
        # ``multimodal=True`` default). ``is_multimodal`` gates the vision prefill
        # path; text-only requests always take the inherited text path unchanged.
        self.vision_tower = None
        self.embed_vision = None
        self.image_token_id = None
        self.pad_token_id = None
        self.is_multimodal = False
        # Audio (12B unified: encoder-free projection) and video (frames through the
        # image path) share the scatter merge with images, keyed by placeholder id.
        self.embed_audio = None
        self.audio_token_id = None
        self.video_token_id = None
        # Prefer the flag baked into the TT model at create time (resolve + env);
        # fall back to hybrid-default env when the model was built elsewhere.
        model0 = self.model[0] if getattr(self, "model", None) else None
        if model0 is not None and hasattr(model0, "bounded_sliding_kv_cache"):
            self._bounded_sliding_kv_cache = bool(model0.bounded_sliding_kv_cache)
        else:
            _prefix_caching_on = self.model_capabilities.get("supports_prefix_caching", False)
            _bounded_default = "1" if (self._HYBRID_KV_CACHE_GROUPS_ENABLED and not _prefix_caching_on) else "0"
            self._bounded_sliding_kv_cache = os.environ.get("GEMMA4_BOUNDED_SLIDING_KV_CACHE", _bounded_default) != "0"
        # Bounded rings are per-request device state outside vLLM's accounting
        # (see the class comment); refuse the combination at boot.
        if self.model_capabilities.get("supports_prefix_caching") and self._bounded_sliding_kv_cache:
            raise ValueError(
                "GEMMA4_PREFIX_CACHING=1 is incompatible with bounded sliding "
                f"rings (bounded_sliding_kv_cache=True, hybrid_kv_cache_groups="
                f"{self._HYBRID_KV_CACHE_GROUPS_ENABLED}). Unset "
                "GEMMA4_BOUNDED_SLIDING_KV_CACHE (and GEMMA4_BOUNDED_SLIDING) "
                "or disable prefix caching."
            )
        # PLI models must restage decode inputs from host every step; async lag
        # would restage a stale token. Narrow the instance dict so runtime
        # readers (the decode-sync default below) see False. The vLLM TT plugin
        # snapshots the class attribute during check_and_update_config, before
        # this runs, so this does not disable scheduler_config.async_scheduling;
        # set GEMMA4_SUPPORTS_ASYNC_DECODE=0 for that.
        if model0 is not None and (
            bool(getattr(model0, "hidden_size_per_layer_input", 0))
            or bool(getattr(model0, "_tt_vllm_always_refresh_decode_trace_inputs", False))
        ):
            self.model_capabilities = {
                **self.model_capabilities,
                "supports_async_decode": False,
            }
        # On-device decode sampling only exists when Gemma4Model actually built a
        # SamplingGenerator. That needs the per-device vocab shard to fit ttnn's
        # 64k topk width: Gemma4's vocab is 262144, so TP>=4 shards to <=65536 and
        # works, but TP=2 (WH N300 1x2) shards to 131072 and Gemma4Model leaves
        # ``self.sampling = None``. Narrow the instance dict so runtime readers
        # see False. The platform already snapshot the class attribute, so this
        # does not revise sample_on_device_mode.
        if model0 is not None and not bool(getattr(model0, "_supports_on_device_sampling", True)):
            self.model_capabilities = {
                **self.model_capabilities,
                "supports_sample_on_device": False,
            }
            _tp = getattr(getattr(model0, "mesh_config", None), "tp", "?")
            logger.info(
                f"Gemma4: on-device sampling unavailable on this mesh (tp={_tp}, "
                f"vocab={getattr(model0, 'vocab_size', '?')} shards wider than ttnn topk's 64k); "
                "advertising supports_sample_on_device=False so decode samples on host."
            )
        # Host-side TTFT / decode tok/s for metal↔server parity checks.
        # Compare these to demo ``inference_prefill`` / decode tok/s/user logs.
        self._perf_decode_tokens = 0
        self._perf_decode_s = 0.0
        self._perf_log_every = max(1, int(os.environ.get("GEMMA4_VLLM_PERF_LOG_EVERY", "32")))
        # Batch-keyed decode traces (mixin); must exist before first sample.
        self._prev_decode_batch = None
        # Extra synchronize_device stalls async overlap (#51186). Default off;
        # set GEMMA4_VLLM_DECODE_SYNC_EVERY=log/1 for sync wall-clock parity.
        _sync_default = "0" if self.model_capabilities.get("supports_async_decode") else "log"
        self._perf_decode_sync_every = os.environ.get("GEMMA4_VLLM_DECODE_SYNC_EVERY", _sync_default)

    def _gemma4_tt_sampler(self):
        """The on-device sampler of the first model, or None (host sampling / no sampler)."""
        models = getattr(self, "model", None)
        model = models[0] if isinstance(models, (list, tuple)) and models else models
        return getattr(getattr(model, "sampling", None), "tt_sampling", None)

    def _create_sampling_params(self, can_sample_on_device, batch_size, greedy_only: bool = False):
        """The warmup sampling sweep, minus the request shapes this mesh's sampler opts out of.

        2-D fractured weights shard the vocab per chip, which breaks the penalty and top-k/top-p
        programs; the sampler marks itself ``_allow_penalties_sampling`` / ``_allow_topk_sampling``
        False and rejects those shapes, so the sweep must not ask for them (its own precompile skips
        them the same way). Greedy and ``None`` are always kept.
        """
        configs = super()._create_sampling_params(can_sample_on_device, batch_size, greedy_only=greedy_only)
        return filter_warmup_sampling_params(configs, self._gemma4_tt_sampler())

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        """Warm decode traces at B=1 and B=max (trace-region friendly).

        Full power-of-two warmup (1..32) would multiply resident decode-trace
        DRAM (~360MB at B=32 alone). B=1 recovers metal short-ISL tok/s; B=max
        covers concurrency. Intermediate active counts pad up to B=max.
        Override with ``GEMMA4_DECODE_WARMUP_BATCHES=1,8,32``.
        """
        max_b = int(max_batch_size)
        override = os.environ.get("GEMMA4_DECODE_WARMUP_BATCHES")
        if override:
            sizes = sorted({int(x) for x in override.split(",") if x.strip() and int(x) <= max_b})
        else:
            sizes = sorted({1, max_b} if max_b > 1 else {1})
        if getattr(getattr(self.model[0], "mesh_config", None), "lane_sharded", False):
            # Lane-sharded decode pads every batch to the full lane-major frame,
            # so smaller buckets would only re-capture the B=max trace.
            sizes = [max_b]
        # Restrict to declared supported buckets.
        supported = set(self.tt_supported_decode_batch_sizes)
        sizes = [b for b in sizes if b in supported or b == max_b]
        if not sizes:
            sizes = [max_b]
        # Narrow the declared buckets to exactly what we warm below, so the
        # plugin never pads to a bucket without a captured decode trace. This
        # replaces the previous separate ``tt_warmed_decode_batch_sizes``
        # attribute — an unwarmed bucket is simply not supported (review on
        # tenstorrent/vllm#455).
        self.tt_supported_decode_batch_sizes = tuple(sizes)
        # Smallest → largest so the final batch leaves sampling traces bound to
        # max_batch logits (sampling capture is skipped for smaller buckets).
        self._g4_decode_warmup_active = True
        try:
            for batch in sorted(sizes):
                # Drop sampling traces from the previous batch — they bind logits
                # tensor identity/batch and will ValueError on the next size.
                for m in self.model:
                    sampling = getattr(m, "sampling", None)
                    if sampling is not None and hasattr(sampling, "reset_trace"):
                        sampling.reset_trace()
                logger.info("Gemma4 vLLM: decode warmup batch_size={}", batch)
                super().warmup_model_decode(
                    kv_cache=kv_cache,
                    enable_trace=enable_trace,
                    max_batch_size=batch,
                    num_blocks=num_blocks,
                    can_sample_on_device=can_sample_on_device,
                    **kwargs,
                )
        finally:
            self._g4_decode_warmup_active = False
        self._decode_warmup_complete = True
        # Baseline for the decode-trace allocation guard: the level right after the captures, before
        # any request can leave a persistent allocation in the traces' scratch range.
        if enable_trace:
            levels = self._g4_alloc_levels()
            for key, tids in list(self.trace_ids_decode.items()):
                if tids:
                    self._g4_trace_alloc_baseline[key] = levels
            self._g4_alloc_check_pending = False
            self._g4_alloc_dump("post_warmup_")

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", **kwargs) -> int:
        # The all-user KV-cache pool size is a per-device / per-model tuning knob,
        # not a model constant: with hybrid KV groups disabled every layer
        # allocates a full-length KV buffer, so the pool that fits in DRAM is
        # hardware-specific (e.g. ~49K on QB2/P300x2 for 31B, ~131K for 12B).
        # Keep that value OUT of the model code — set ``GEMMA4_MAX_TOKENS_ALL_USERS``
        # from the tt-inference-server model spec's per-device ``env_vars`` block
        # (gated there by device + model). This generic, value-free hook just
        # honors that override and otherwise defers to the default.
        override = os.environ.get("GEMMA4_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        return super().get_max_tokens_all_users(model_name=model_name, **kwargs)

    def _maybe_disable_pli_prefill_trace(self, enable_trace: bool, batch_size: int = 1) -> bool:
        return maybe_disable_pli_prefill_trace(enable_trace, self.model[0], batch_size=batch_size)

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device,
        greedy_only: bool = False,
    ):
        # #49083 fix: pre-capture the prefill-bucket traces here, at warmup,
        # rather than lazily on the first runtime prefill. A cold *eager*
        # prefill dispatched after a shared-Generator traced-decode session
        # (the release workflow's evals phase) wedges the fetch queue
        # (nlp_concat_heads) at ISL=2048 — capturing every bucket up front so
        # runtime only *replays* removes that trace->eager transition.
        #
        # The hybrid per-layer page tables diverge from the single broadcast
        # table a direct ``prefill_forward_text`` capture would bind, so route
        # the capture through ``prefill_forward`` (``prefill_forward_fn`` below).
        # That sets up per-layer routing and populates the persistent per-layer
        # buffers *before* the traced forward, so the captured paged ops bind
        # those buffers — identical to how decode warmup binds via
        # ``decode_forward``. Runtime ``prefill_forward`` then just refreshes the
        # same buffers' block IDs out-of-trace and replays. ``_mock_tokens``
        # sizes the warmup page table to the runtime width, so the persistent
        # buffers match runtime (and decode-warmup) shapes.
        #
        # GEMMA4_DISABLE_PREFILL_TRACE=1 keeps prefill fully eager (no capture).
        # Bounded sliding: never capture prefill TRACE — mid-forward paged_fill
        # corrupts token-0 on TP (see resolve_gemma4_prefill_trace_enable).
        prefill_forward_fn = None
        if (
            self._bounded_sliding_kv_cache
            and enable_trace
            and trace_short_prefill_opt_in(mesh_shape=getattr(self.model[0].mesh_device, "shape", None))
        ):
            # GEMMA4_TRACE_SHORT_PREFILL=1: capture only the short single-user buckets (the runtime policy in
            # resolve_gemma4_prefill_trace_enable replays exactly those); everything longer stays eager.
            cap = trace_short_prefill_max()
            for model_args_i in self.model_args:
                model_args_i.trace_prefill_supported_seq_lens = [
                    length for length in model_args_i.trace_prefill_supported_seq_lens if length <= cap
                ]
            logger.info(
                f"Gemma4 bounded sliding: short-prefill trace opt-in, capturing buckets "
                f"{self.model_args[0].trace_prefill_supported_seq_lens}"
            )
        elif self._bounded_sliding_kv_cache:
            enable_trace = False
        if enable_trace and _gemma4_prefill_trace_unsafe(self.model[0], self._bounded_sliding_kv_cache):
            prefill_forward_fn = self.prefill_forward
        # Lane-group prefill after the eager decode warmup and before any capture:
        # its persistent buffers must exist before the decode traces are recorded,
        # but a lane-parallel prefill before the decode graph's first run hangs that
        # run at boot (Galaxy, 2026-10-22), so it waits for the plugin's second
        # (trace) pass (see the method).
        if getattr(self, "_decode_warmup_complete", False):
            self._g4_warmup_lane_prefill_group(kv_cache)
        warmup_gemma4_model_prefill(
            self,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
            prefill_forward_fn=prefill_forward_fn,
        )
        # Compile the vision tower + projector + on-device gather once at warmup
        # so the first multimodal prefill doesn't pay the compile cost. Text-only
        # serving skips this (``is_multimodal`` is False). A failed warmup turns
        # the multimodal path off (``_g4_refuse_multimodal``): deferring the compile
        # to the first real request put its allocations behind the live decode
        # traces, the hang the warmup exists to prevent.
        # Once per process: the shared prefill path calls this warmup on every
        # prefill batch (``warmup_prefill`` defaults to True), and re-encoding the
        # dummy image each time cost ~170 ms of TTFT per batch and allocated
        # behind the live decode traces.
        if self.is_multimodal and not getattr(self, "_g4_vision_warmed", False):
            self._g4_vision_warmed = True
            try:
                self.warmup_vision()
            except Exception as e:  # noqa: BLE001 - the path goes off, see _g4_refuse_multimodal
                self._g4_refuse_multimodal(e)

    def _g4_refuse_multimodal(self, error) -> None:
        """A failed vision warmup: the first real encode would compile and allocate behind the live decode traces
        (the hang the warmup exists to prevent), so the multimodal path goes off instead of being deferred. A class
        that serves multimodal requests (``SupportsMultiModal``) cannot do without it and fails the boot; a text-only
        class (vLLM rejects its image requests at the API) records the refusal, and a multimodal batch that still
        reaches it is refused (``_g4_check_multimodal_admissible``). Text serving is unaffected either way."""
        reason = f"{type(error).__name__}: {error}"
        if isinstance(self, SupportsMultiModal):
            raise RuntimeError(
                f"Gemma4 vLLM: the vision warmup failed ({reason}); a multimodal server does not start without it"
            ) from error
        self._g4_multimodal_refused = reason
        logger.error(
            "Gemma4 vLLM: the vision warmup failed ({}); multimodal inputs are refused from here (text serving is "
            "unaffected)",
            reason,
        )

    def _g4_check_multimodal_admissible(self, kwargs) -> None:
        """Refuse a multimodal prefill batch after a failed vision warmup (see ``_g4_refuse_multimodal``)."""
        reason = getattr(self, "_g4_multimodal_refused", None)
        if reason and _gemma4_has_mm_inputs(kwargs):
            raise RuntimeError(f"Gemma4 vLLM: multimodal inputs refused: the vision warmup failed at boot ({reason})")

    def _g4_lane_warmup_failed(self, which: str, error) -> None:
        """A failed lane warmup turns its path off for the process (one line) instead of leaving the first serving
        call to compile and allocate it behind the live decode traces: the lane-group walk off is the lane prefill
        router off (every prefill serial, as without the router); the lane-batched walk off is one round per lane
        call (the eager lane-group walk stays) and no k > 1 frame is captured."""
        reason = f"{type(error).__name__}: {error}"
        if which == "lane-group":
            self._g4_lane_router_off = reason
            logger.error(
                "Gemma4 vLLM: the lane-group prefill warmup failed ({}); the lane prefill router is OFF for this "
                "process (every prefill runs serially)",
                reason,
            )
        else:
            self._g4_lane_batched_off = reason
            logger.error(
                "Gemma4 vLLM: the lane-batched prefill warmup failed ({}); lane-batched walks are OFF for this "
                "process (one round per lane call)",
                reason,
            )

    def _g4_lane_batched_walks(self) -> bool:
        """The router builds k > 1 lane-batched walks: the knob, unless the boot warmup of that walk failed."""
        return lane_prefill_batched_enabled() and not getattr(self, "_g4_lane_batched_off", None)

    def _g4_warmup_lane_prefill_group(self, kv_cache) -> None:
        """Run one lane-group prefill at boot when the lane router is on (``lane_prefill_group``).

        A lane group (one user per lane, one chunk walk) compiles its own programs and allocates
        persistent buffers the per-user path never touches: the lanes-keyed per-layer page tables
        and the lane-shaped CCL buffers. In serving the first group only forms once every lane has
        a fresh prefill in the same step -- on the Galaxy that was minutes into the first load
        phase, after the decode-trace allocation guard had spent its re-capture budget, so the
        replays overwrote what that first group allocated and the second group hung the device.
        Warming the path here puts those allocations below every capture. Once per process, and
        only after the eager decode warmup (``_decode_warmup_complete``): run before the decode
        graph's first step it hung that step at boot, so the plugin's two-pass warmup places it
        between the eager pass and the trace pass.
        """
        if getattr(self, "_g4_lane_prefill_warmed", False) or not self._lane_prefill_group():
            return
        model = self.model[0]
        mesh_cfg = getattr(model, "mesh_config", None)
        if mesh_cfg is None or not getattr(mesh_cfg, "lane_sharded", False):
            return
        if kv_cache is None or kv_cache[0] is None:
            return
        self._g4_lane_prefill_warmed = True
        lanes = int(mesh_cfg.lanes)
        lane_slots = int(getattr(model, "lane_slots", 0) or 32)
        seq_len = int(getattr(self.model_args[0], "max_prefill_chunk_size", 0) or 0)
        if seq_len <= 0:
            return
        mock = self._mock_tokens(lanes, seq_len, kv_cache, 0)
        page_table = mock.get("page_table")
        if page_table is None:
            return
        # The bridge's mock rows are all null block 0; the lane walks give every user its own blocks (the batched
        # paged fill's writers must not share a block: the demo generator's override, the P150x8 batch-32 stall).
        pool = lane_kv_pool_blocks(kv_cache[0])
        page_table = lane_unique_block_rows(lanes, int(page_table.shape[-1]), pool)
        # One slot per lane (slot s lives on lane s // lane_slots): the bounded sliding rings get
        # ring-local tables for those slots and full-attention layers the mock's unique blocks.
        slot_ids = [ln * lane_slots for ln in range(lanes)]
        logger.info("Gemma4 vLLM: warming up the lane-group prefill path ({} lanes x {} tokens)", lanes, seq_len)
        t0 = time.perf_counter()
        try:
            self.prefill_forward_lanes(
                mock["tokens"],
                page_table.reshape(lanes, 1, -1).to(torch.int32),
                kv_cache,
                [seq_len] * lanes,
                slot_ids=slot_ids,
            )
        except Exception as e:  # noqa: BLE001 - the router goes off (_g4_lane_warmup_failed)
            self._g4_lane_warmup_failed("lane-group", e)
            return
        finally:
            if hasattr(model, "_active_page_tables_per_layer"):
                del model._active_page_tables_per_layer
        # k prompts per lane: the lane-batched walk's programs and its page-table frame (the lanes' decode frame,
        # the key every batched call uses) must exist before the decode traces are captured too.
        kw, seq_w = lane_batched_warmup_shape(self._lane_prefill_prompts(), lane_slots, seq_len)
        if kw > 1 and self._g4_lane_batched_walks():
            mock_b = self._mock_tokens(lanes * kw, seq_w, kv_cache, 0)
            page_table_b = mock_b.get("page_table")
            if page_table_b is not None:
                page_table_b = lane_unique_block_rows(lanes * kw, int(page_table_b.shape[-1]), pool)
                logger.info(
                    "Gemma4 vLLM: warming up the lane-batched prefill path ({} lanes x {} users x {} tokens)",
                    lanes,
                    kw,
                    seq_w,
                )
                try:
                    self.prefill_forward_lanes(
                        mock_b["tokens"].reshape(lanes, kw, seq_w),
                        page_table_b.reshape(lanes, kw, -1).to(torch.int32),
                        kv_cache,
                        [seq_w] * (lanes * kw),
                        slot_ids=[ln * lane_slots + r for ln in range(lanes) for r in range(kw)],
                    )
                except Exception as e:  # noqa: BLE001 - the batched walks go off (_g4_lane_warmup_failed)
                    self._g4_lane_warmup_failed("lane-batched", e)
                finally:
                    if hasattr(model, "_active_page_tables_per_layer"):
                        del model._active_page_tables_per_layer
        # The traced lane frames (GEMMA4_LANE_PREFILL_TRACE=1): captured here, inside the boot window and before
        # the decode traces, so their buffers and programs sit below every decode capture; a failed capture leaves
        # that frame eager.
        buckets, dropped = lane_prefill_trace_buckets(
            max_chunk=seq_len,
            lane_slots=lane_slots,
            bounded_sliding=bool(getattr(self, "_bounded_sliding_kv_cache", False)),
            short_max=trace_short_prefill_max(),
        )
        if dropped:
            logger.warning(
                "Gemma4 vLLM: lane prefill trace buckets dropped (chunk {}, slots {}): {}", seq_len, lane_slots, dropped
            )
        for bucket in sorted(buckets):
            if bucket[0] > 1 and not self._g4_lane_batched_walks():
                continue  # the batched walk is off: its frames would replay a path the warmup could not run
            try:
                self._g4_capture_lane_prefill_trace(bucket, kv_cache)
            except Exception as e:  # noqa: BLE001 - the capture closed and released its trace; the frame is off
                logger.warning(
                    "Gemma4 vLLM: lane prefill trace {}x{} capture failed ({}); the frame stays eager (the capture "
                    "was closed and released before the error)",
                    bucket[0],
                    bucket[1],
                    e,
                )
                if hasattr(model, "_active_page_tables_per_layer"):
                    del model._active_page_tables_per_layer
        logger.info("Gemma4 vLLM: lane-group prefill warmup done in {:.1f} s", time.perf_counter() - t0)

    # ── Multimodal path ────────────────────────────────────────────────────
    # Active only when ``is_multimodal`` is True (checkpoint has vision weights). Text-only requests always take
    # the inherited ``prefill_forward_text`` path. ``encode_vision`` / ``warmup_vision`` are the mixin's (shared with
    # the demo generator); the merge of the soft tokens into the prompt embeddings is HOST-side here
    # (``_merged_prompt_embeds``: ``merge_soft_tokens`` over the host text embeddings), staged per user for the
    # eager text prefill (``_stage_multimodal_prefill``); decode reuses the inherited text ``decode_forward``.

    def encode_audio(self, input_features: torch.Tensor, input_features_mask: torch.Tensor = None):
        """Project one audio clip's raw-waveform frames (``[T, 640]`` or ``[1, T, 640]``,
        HF ``Gemma4UnifiedAudioFeatureExtractor``) and return the VALID soft tokens on
        device as ``[num_valid, text_hidden]`` (bf16). ``input_features_mask`` (``[T]``,
        True = real frame) drops the extractor's padding frames."""
        if self.embed_audio is None:
            raise RuntimeError("encode_audio requires a Gemma-4 checkpoint with an audio projection (12B unified).")
        features = input_features.unsqueeze(0) if input_features.dim() == 2 else input_features
        mask = input_features_mask.reshape(1, -1).bool() if input_features_mask is not None else None
        projected_tt, valid = self.embed_audio(features, mask)
        return self._gather_valid_soft_tokens(projected_tt, valid)

    def _vision_to_host(self, embeds_tt):
        """ttnn ``[num_valid, text_hidden]`` -> host bf16 tensor (device 0's replica)."""
        if hasattr(self.mesh_device, "shape"):
            host = ttnn.to_torch(ttnn.get_device_tensors(embeds_tt)[0])  # replicated: any device
        else:
            host = ttnn.to_torch(embeds_tt)
        if hasattr(embeds_tt, "deallocate"):
            embeds_tt.deallocate(True)
        return host.to(torch.bfloat16)

    def _host_text_embeds(self, llm_ids: torch.Tensor, model) -> torch.Tensor:
        """Host ``[N, text_hidden]`` bf16 text embeddings of ``llm_ids`` (``[N]``), scaled by
        ``sqrt(hidden)`` exactly as ``Gemma4Model.embed_tokens`` does on device."""
        weight = getattr(model, "_embed_weight_cpu", None)
        if weight is not None:
            embeds = torch.nn.functional.embedding(llm_ids, weight).float() * model.embed_scale
            return embeds.to(torch.bfloat16)
        # No host copy of the embedding table: look the rows up on device.
        ids_tt = ttnn.from_torch(
            llm_ids.view(1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
        )
        embeds_tt = ttnn.reshape(model.embed_tokens(ids_tt), (int(llm_ids.numel()), model.hidden_size))
        return self._vision_to_host(embeds_tt)

    def _merged_prompt_embeds(self, user_tokens: torch.Tensor, soft_tokens, model_id: int = 0, num_cached: int = 0):
        """Host ``[prompt_len, text_hidden]`` bf16 embeddings of one multimodal prompt.

        ``soft_tokens`` is a list of ``(placeholder_token_id, embeds)`` pairs, one per
        modality present in the prompt (image / video / audio), each ``embeds`` a host
        (or ttnn) ``[num_valid, text_hidden]`` tensor holding that modality's valid
        soft tokens in prompt order. The text rows are embedded and scaled as the
        decoder's ``embed_tokens`` does and every placeholder row is replaced by its
        soft token (``merge_soft_tokens``), which is what HF's ``masked_scatter`` of
        the projected features into ``inputs_embeds`` produces.
        """
        model = self.model[model_id]
        if getattr(model, "hidden_size_per_layer_input", 0):
            raise NotImplementedError(
                "Multimodal prefill + per-layer inputs (PLI) is not supported. "
                "Use a non-PLI multimodal checkpoint (gemma-4 12B / 26B-A4B / 31B IT)."
            )
        input_ids = user_tokens.reshape(-1).to(torch.long)
        soft_tokens = [
            (token_id, embeds if isinstance(embeds, torch.Tensor) else self._vision_to_host(embeds))
            for token_id, embeds in soft_tokens
        ]
        llm_ids, positions, soft = merge_soft_tokens(input_ids, soft_tokens, self.pad_token_id, num_cached=num_cached)
        merged = self._host_text_embeds(llm_ids, model).clone()
        merged[positions.long()] = soft.to(torch.bfloat16)
        return merged

    # Soft tokens of the multimodal items earlier prefills encoded, keyed by the item's placeholder range and the KV
    # blocks it occupies (the identity a prefix-cache hit shares between requests). vLLM strips the payload of every
    # item inside a prefix-cache hit (vllm.multimodal.utils.strip_covered_mm_data) while the SDPA alignment restarts
    # the prefill at the 128-floor of the hit (align_num_cached_tokens_to_sdpa) and recomputes the tokens between,
    # so an item ending in that window needs its soft tokens again or the merge fails the request (a 64-aligned
    # hit at 960 restarts at 896 and an image ending at 960 has 64 placeholders to fill). Bounded per process.
    _G4_MM_SOFT_TOKEN_CACHE_ITEMS = 64

    def _g4_mm_soft_token_cache(self):
        cache = self.__dict__.get("_g4_mm_soft_tokens")
        if cache is None:
            cache = self._g4_mm_soft_tokens = OrderedDict()
        return cache

    def _g4_mm_remember(self, key, soft) -> None:
        cache = self._g4_mm_soft_token_cache()
        cache[key] = soft
        cache.move_to_end(key)
        while len(cache) > self._G4_MM_SOFT_TOKEN_CACHE_ITEMS:
            cache.popitem(last=False)

    @staticmethod
    def _g4_mm_item_key(token_id, offset, length, identity):
        """The cache key of one item: its placeholder id and range plus the ids of the KV blocks the range occupies
        (``identity`` = (the request's page-table row, tokens per column)); None without a usable identity (no host
        table, a range past the row, a null block in the span)."""
        if identity is None or token_id is None or offset is None or length is None:
            return None
        row, stride = identity
        c0, c1 = int(offset) // int(stride), (int(offset) + int(length) - 1) // int(stride)
        if c0 < 0 or c1 >= int(row.shape[-1]):
            return None
        blocks = tuple(int(b) for b in row[c0 : c1 + 1].tolist())
        if any(b <= 0 for b in blocks):
            return None
        return (int(token_id), int(offset), int(length), blocks)

    def _g4_mm_identity_tables(self, kwargs, page_tables_per_layer):
        """``(rows, stride)``: the host page table whose row ``u`` names the KV blocks of user ``u``'s prompt (the
        first full-attention layer's table, which keeps every block of the prompt; the legacy table otherwise) and
        the tokens per column it is laid out at (``_prefill_page_table_block_sizes``), or None without a host table."""
        kv_cache = kwargs.get("kv_cache")
        sizes = self._prefill_page_table_block_sizes(kv_cache) if kv_cache is not None else []
        if not sizes:
            return None
        layer_types = list(getattr(self._text_config(), "layer_types", None) or [])
        for i, pt in enumerate(page_tables_per_layer or []):
            if i < len(sizes) and i < len(layer_types) and layer_types[i] == "full_attention":
                if isinstance(pt, torch.Tensor) and pt.dim() == 2:
                    return pt, int(sizes[i])
        legacy = kwargs.get("page_table")
        if isinstance(legacy, torch.Tensor) and legacy.dim() == 2:
            return legacy, int(sizes[0])
        return None

    def _g4_encode_image_item(self, pixel_values, position_ids):
        return self._vision_to_host(self.encode_vision(_mm_batched(pixel_values), _mm_batched(position_ids)))

    def _g4_encode_video_item(self, frames, position_ids):
        frames, position_ids = _mm_batched(frames), _mm_batched(position_ids)  # [frames, P, D], [frames, P, 2]
        chunks = [
            self._vision_to_host(self.encode_vision(frames[f : f + 1], position_ids[f : f + 1]))
            for f in range(int(frames.shape[0]))
        ]
        return torch.cat(chunks, dim=0)

    def _g4_encode_audio_item(self, features, mask):
        return self._vision_to_host(self.encode_audio(features, mask))

    def _g4_modality_soft_tokens(
        self,
        *,
        user,
        prompt,
        token_id,
        payloads,
        companions,
        offsets,
        lengths,
        window_end,
        aligned_start,
        computed,
        identity,
        encode,
        companion_required,
    ):
        """Host soft tokens ``[n, hidden]`` of one modality of user ``user`` for this prefill call, in prompt order,
        or None when nothing is staged: the items whose placeholders end inside the window (``window_end``), encoded
        from their payloads and remembered under their block identity, plus the items the plugin stripped (inside
        the prefix-cache hit of ``computed`` tokens, payload None) whose placeholders the SDPA-aligned restart at
        ``aligned_start`` still covers, restored from the cache. ``payloads`` / ``companions`` / ``offsets`` /
        ``lengths`` are the request's per-item lists (aligned over all of its items, every modality); a stripped
        item belongs to this modality when the prompt holds ``token_id`` at its offset.
        """
        if token_id is None:
            return None  # a modality this checkpoint has no placeholder id for (its payloads were refused above)
        items = list(payloads) if isinstance(payloads, (list, tuple)) else ([payloads] if payloads is not None else [])
        comps = (
            list(companions)
            if isinstance(companions, (list, tuple))
            else ([companions] if companions is not None else [])
        )
        n = max(len(items), len(offsets or ()), len(lengths or ()))
        if offsets is None and lengths is None:
            inside = set(range(n))  # an older plugin without ranges: every forwarded item, as before
        else:
            inside = set(items_within_prefill_window(offsets, lengths, window_end))
        chunks = []
        for i in range(n):
            payload = items[i] if i < len(items) else None
            offset = offsets[i] if offsets is not None and i < len(offsets) else None
            length = lengths[i] if lengths is not None and i < len(lengths) else None
            if i not in inside:
                continue  # placed past this chunk's end: a later chunk's
            key = self._g4_mm_item_key(token_id, offset, length, identity)
            if payload is not None:
                companion = comps[i] if i < len(comps) else None
                if companion_required and companion is None:
                    raise ValueError(
                        f"user {user}: multimodal item {i} (placeholder id {token_id}) has no position ids"
                    )
                soft = encode(payload, companion)
                if key is not None:
                    self._g4_mm_remember(key, soft)
                chunks.append(soft)
                continue
            if offset is None or length is None or int(offset) >= int(prompt.shape[-1]):
                continue
            if int(prompt[int(offset)]) != int(token_id):
                continue  # another modality's item
            end = int(offset) + int(length)
            if end <= int(aligned_start):
                continue  # wholly inside the aligned cached prefix: its KV stays, the merge needs no soft token
            soft = self._g4_mm_soft_token_cache().get(key) if key is not None else None
            if soft is None:
                raise ValueError(
                    f"user {user}: the multimodal item at [{offset}, {end}) (placeholder id {token_id}) lies inside the "
                    f"prefix-cache hit of {computed} tokens, so the plugin forwarded it without its payload, and the "
                    f"SDPA-aligned prefill restarts at {aligned_start} < {end}; its soft tokens are not in this "
                    "process's cache (encoded by an earlier prefill this engine no longer remembers, or never here)"
                )
            self._g4_mm_soft_token_cache().move_to_end(key)
            chunks.append(soft)
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    def _stage_multimodal_prefill(
        self, tokens, kwargs, prompt_lens, *, num_computed=None, page_tables_per_layer=None
    ) -> int:
        """Encode this prefill batch's multimodal inputs and stage each user's merged
        prompt embeddings for the text prefill.

        The plugin forwards the processor's multimodal fields in ``kwargs``, each a
        per-batch-slot list (None for slots without that modality) whose entries are
        per-item tensors in prompt order:
          * images: ``pixel_values`` ``[P, D]`` + ``pixel_position_ids`` ``[P, 2]``
            (vLLM's name for the processor's ``image_position_ids``);
          * videos: ``pixel_values_videos`` ``[frames, P, D]`` +
            ``pixel_position_ids_videos`` ``[frames, P, 2]``; every frame runs through
            the image path and the frames' soft tokens fill the video placeholders;
          * audio (12B unified): ``input_features`` ``[T, 640]`` raw-waveform frames
            + ``input_features_mask`` ``[T]``.

        Each user with inputs gets the host ``[prompt_len, text_hidden]`` merged
        embedding of its prompt in ``kwargs["pixel_values"]``: the one per-user field
        the shared prefill loop indexes per batch row and passes on to
        ``prefill_forward_single_user_text``, which installs it as the embedding
        override of that user's (eager) prefill. So the multimodal prefill IS the
        text prefill -- same page tables, per-layer tables, chunking and cached
        prefixes -- with only the embedding step replaced. Every other multimodal
        field is dropped so the text prefill sees exactly its usual kwargs.

        ``kwargs["start_pos"]`` is the SDPA-aligned restart the merge covers; ``num_computed`` is vLLM's own
        computed count per user (the prefix-cache hit the plugin stripped payloads by), ``page_tables_per_layer``
        the call's host tables (the block identity of the items, ``_g4_modality_soft_tokens``).
        Returns the number of users with multimodal inputs.
        """
        batch_size = int(tokens.shape[0])
        model_id = 0
        start_pos = kwargs.get("start_pos")
        num_cached = [int(n) for n in start_pos] if start_pos is not None else [0] * batch_size
        computed = [int(n) for n in num_computed] if num_computed is not None else list(num_cached)
        identity = self._g4_mm_identity_tables(kwargs, page_tables_per_layer)

        def _per_slot(*keys):
            for key in keys:
                value = kwargs.get(key)
                if value is not None:
                    items = list(value)
                    return items + [None] * (batch_size - len(items))
            return [None] * batch_size

        off_per_user = _per_slot("mm_placeholder_offsets")
        len_per_user = _per_slot("mm_placeholder_lengths")
        pv_per_user = _per_slot("pixel_values")
        ip_per_user = _per_slot("pixel_position_ids", "image_position_ids")
        vpv_per_user = _per_slot("pixel_values_videos")
        vip_per_user = _per_slot("pixel_position_ids_videos", "video_position_ids")
        af_per_user = _per_slot("input_features", "input_features_padded")
        am_per_user = _per_slot("input_features_mask")

        staged = [None] * batch_size
        for u in range(batch_size):
            # The merge covers the prompt up to this chunk's end: items placed past it wait.
            end = int(prompt_lens[u]) if prompt_lens is not None else int(tokens.shape[-1])
            if _mm_value_present(vpv_per_user[u]) and self.video_token_id is None:
                raise RuntimeError("video input for a checkpoint without a video placeholder id")
            if _mm_value_present(af_per_user[u]) and self.embed_audio is None:
                raise RuntimeError("audio input for a checkpoint without an audio embedder (31B has no audio)")
            row = identity[0][u] if identity is not None and u < int(identity[0].shape[0]) else None
            common = dict(
                user=u,
                prompt=tokens[u],
                offsets=off_per_user[u],
                lengths=len_per_user[u],
                window_end=end,
                aligned_start=num_cached[u],
                computed=computed[u],
                identity=(row, identity[1]) if row is not None else None,
            )
            soft = []
            for token_id, payloads, companions, encode, required in (
                (self.image_token_id, pv_per_user[u], ip_per_user[u], self._g4_encode_image_item, True),
                (self.video_token_id, vpv_per_user[u], vip_per_user[u], self._g4_encode_video_item, True),
                (self.audio_token_id, af_per_user[u], am_per_user[u], self._g4_encode_audio_item, False),
            ):
                embeds = self._g4_modality_soft_tokens(
                    token_id=token_id,
                    payloads=payloads,
                    companions=companions,
                    encode=encode,
                    companion_required=required,
                    **common,
                )
                if embeds is not None:
                    soft.append((token_id, embeds))
            if not soft:
                continue
            staged[u] = self._merged_prompt_embeds(tokens[u, :end], soft, model_id=model_id, num_cached=num_cached[u])

        for key in _GEMMA4_MM_FIELD_KEYS:
            kwargs.pop(key, None)
        kwargs["pixel_values"] = staged
        staged_users = sum(1 for e in staged if e is not None)
        logger.info(f"Gemma4 vLLM: staged multimodal prefill embeddings for {staged_users}/{batch_size} user(s)")
        return staged_users

    def prefill_forward_text(self, *args, enable_trace=True, **kwargs):
        tokens = args[0] if args else kwargs.get("tokens")
        batch_size = tokens.shape[0] if tokens is not None else 1
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace, batch_size=batch_size)
        if tokens is not None:
            batch_seq_len = tokens.shape[1]
            prompt_lens = kwargs.get("prompt_lens")
            start_pos = kwargs.get("start_pos")
            prompt_lens_list = prompt_lens if prompt_lens is not None else [batch_seq_len] * batch_size
            if not isinstance(prompt_lens_list, list):
                prompt_lens_list = prompt_lens_list.tolist()
            num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens_list)
            if start_pos is not None:
                num_cached_per_user = align_num_cached_tokens_to_sdpa(num_cached_per_user)
                kwargs["start_pos"] = num_cached_per_user
                start_pos = num_cached_per_user
            prefill_seq_lens = [
                get_padded_prefill_len(seq_len - num_cached)
                for seq_len, num_cached in zip(prompt_lens_list, num_cached_per_user)
            ]
            page_table = kwargs.get("page_table")
            # Hetero *actual* lens OK: per-slot valid_seq_lens caps KV fill.
            can_batch_prefill = (
                page_table is not None
                and batch_size > 1
                and len(set(prefill_seq_lens)) == 1
                and self.data_parallel == 1
                and not getattr(self.model_args[0], "disable_batched_prefill", False)
                and all(n == 0 for n in num_cached_per_user)
            )
            enable_trace = resolve_gemma4_prefill_trace_enable(
                enable_trace,
                self.model[0],
                self.model_args[0],
                batch_size=batch_size,
                prefill_seq_lens=prefill_seq_lens,
                can_batch_prefill=can_batch_prefill,
                empty_slots=kwargs.get("empty_slots"),
            )
            if not args:
                # GEMMA4_PACKED_PREFILL=1: the cold prompts of one padded bucket run as one packed forward
                # (see ChunkedPrefillPageTableGuardMixin._g4_try_packed_prefill); the rest take the stock path.
                packed_out = self._g4_try_packed_prefill(
                    dict(kwargs, enable_trace=enable_trace),
                    run_rest=lambda **sub: super(Gemma4ForCausalLM, self).prefill_forward_text(**sub),
                )
                if packed_out is not None:
                    return packed_out
        return super().prefill_forward_text(*args, enable_trace=enable_trace, **kwargs)

    def _bounded_sliding_min_page_table_cols(self, kv_cache) -> int | None:
        """Min page-table columns so ``cache_position_modulo`` fits the kernel check.

        ``paged_fill_cache`` requires ``modulo <= effective_block_size * cols``.
        For sliding layers the kernel's block_size is the cache's declared
        ``shape[2]`` (typically 64) — not the HMA-scaled effective size used
        for full-attn views. The modulo is the RING (``bounded_ring_modulo``),
        which is the window plus any speculative headroom, so the floor is
        ``cdiv(ring, block_size)``; sizing it from the bare window fails the
        kernel check as soon as headroom is configured.
        """
        if not self._bounded_sliding_kv_cache or kv_cache is None:
            return None
        ring = self._bounded_ring_size()
        if ring is None:
            return None
        try:
            block_size = int(kv_cache[0][0].shape[2])
        except (TypeError, IndexError, AttributeError):
            return None
        if block_size <= 0:
            return None
        from models.tt_transformers.tt.common import num_blocks_in_seq

        return num_blocks_in_seq(int(ring), block_size)

    def _bounded_ring_size(self) -> int | None:
        """The bounded sliding ring in positions: the window plus the configured headroom."""
        sliding_window = getattr(self._text_config(), "sliding_window", None)
        if sliding_window is None:
            return None
        from models.demos.gemma4.tt.attention import bounded_ring_modulo

        ring = bounded_ring_modulo(int(sliding_window))
        return None if ring is None else int(ring)

    def _get_prefill_user_page_table(
        self,
        page_table,
        kv_cache,
        prefill_len,
        trace_enabled=False,
        prefill_seq_len=None,
        use_batched_prefill=False,
        user_id=None,
        padded_batch_size=None,
        use_full_prompt_len=False,
    ):
        """Override the shared Generator helper to size/slice the
        per-user page table to the *smallest* effective block_size in
        the model, not the cache's declared block_size.

        Background: ``Generator._get_prefill_user_page_table`` slices
        the page_table to ``cdiv(prefill_seq_len, get_block_size(kv_cache))``
        columns, where ``get_block_size`` reads ``kv_cache[0][0].shape[2]``
        — the declared block_size of layer 0's K cache. Under vLLM's
        hybrid kv-cache-groups manager that's the buffer's *allocation*
        block_size, which for Gemma4-E2B is sliding's 128 (sliding
        layers come first in the layer order and their spec wins the
        shared buffer's shape). But full-attention layers operate
        through a view with effective block_size=64 (see
        ``attention/{prefill,decode}.py``), and ``paged_fill_cache``
        validates ``input_seq_len <= max_num_blocks_per_seq *
        effective_block_size``. With the legacy slice the full layer
        sees too few blocks and the validator fires.

        Use the smallest effective block_size across all attention
        layers — same invariant the warmup ``_mock_tokens`` override
        uses — so the slice covers every layer's needs.

        Bounded sliding: never slice below ``sliding_window/block_size``
        columns (batched or not). Batched warmup at seq=128 otherwise
        yields 2 columns and TT_FATALs ``cache_position_modulo`` (1024).
        """
        import torch

        from models.tt_transformers.tt.common import num_blocks_in_seq

        min_bounded_cols = self._bounded_sliding_min_page_table_cols(kv_cache)

        # vLLM token-chunked prefill / APC: ``prefill_len`` is the full prompt
        # while ``prefill_seq_len`` is only the padded *current* chunk. Truncating
        # to the chunk width leaves continuation ``chunk_page_table`` slices as
        # all -1 → ``paged_fill_cache`` skips writing tokens past the first
        # scheduler chunk (LB 12B ~9k coherence cliff, #51186). Keep the full
        # mapping whenever the prompt is longer than this chunk's pad.
        # ``prefill_len`` is a list under batched prefill — only compare scalars.
        if (
            not use_full_prompt_len
            and prefill_seq_len is not None
            and prefill_len is not None
            and not isinstance(prefill_len, (list, tuple))
            and int(prefill_len) > int(prefill_seq_len)
        ):
            use_full_prompt_len = True

        if use_batched_prefill:
            from models.tt_transformers.tt.common import get_block_size

            block_size = get_block_size(kv_cache)
            batch_dim = padded_batch_size if padded_batch_size is not None else self.model_args[0].max_batch_size
            # Batched path always sizes to the padded chunk grid (slot layout).
            num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
            if min_bounded_cols is not None:
                num_blocks = max(num_blocks, min_bounded_cols)
            if page_table.shape[1] < num_blocks:
                pad = torch.zeros(
                    (page_table.shape[0], num_blocks - page_table.shape[1]),
                    dtype=torch.int32,
                )
                page_table = torch.cat([page_table, pad], dim=1)
            page_table = page_table[:, :num_blocks]
            if trace_enabled and page_table.shape[1] < num_blocks:
                padding = torch.zeros(page_table.shape[0], num_blocks - page_table.shape[1], dtype=torch.int32)
                page_table = torch.cat([page_table, padding], dim=1)
            padded_page_table = torch.zeros(batch_dim, page_table.shape[1], dtype=torch.int32)
            assert user_id is not None
            for i, user in enumerate(user_id):
                padded_page_table[user, :] = page_table[i, :]
            return padded_page_table

        # Per-user (non-batched) path: replicate the base behavior but
        # with effective block_size instead of ``get_block_size``.
        cache = kv_cache[0][0]  # layer 0, K (HMA-shared across specs)
        cache_block_size = cache.shape[2]
        cache_head_dim = cache.shape[-1]
        head_dims = {layer.self_attn.config.head_dim for layer in self.model[0].layers}
        max_head_dim = max(head_dims)
        effective_block_size = cache_block_size * cache_head_dim // max_head_dim

        if use_full_prompt_len:
            target_prefill_len = prefill_len
        else:
            target_prefill_len = prefill_seq_len if prefill_seq_len is not None else prefill_len
        num_blocks = num_blocks_in_seq(target_prefill_len, effective_block_size)
        if min_bounded_cols is not None:
            num_blocks = max(num_blocks, min_bounded_cols)
        if page_table.shape[1] < num_blocks:
            padding = torch.zeros(1, num_blocks - page_table.shape[1], dtype=torch.int32)
            page_table = torch.cat([page_table, padding], dim=1)
        return page_table[:, :num_blocks]

    def _mock_tokens(self, batch_size, seq_len, kv_cache, model_id):
        """Override warmup page_table sizing for the hybrid-kv-cache-groups
        path.

        Warmup must produce a page_table whose shape matches the *runtime*
        legacy page_table shape (``model_input.block_tables`` =
        ``block_tables_per_group[0]`` in the plugin), because the decode
        trace captures device tensors at warmup shapes and ``copy_host_to_device``
        asserts shape-equality on every replay. The runtime per-group
        block_table for layer 0's group has width
        ``cdiv(max_model_len, group_block_size_after_unification)``; for
        Gemma4-E2B layer 0 is sliding and the unifier doubled sliding's
        block_size from ``cache_config.block_size`` to match the larger
        full-attn page size (sliding head_dim=256 → 128 block_size;
        full head_dim=512 → 64 block_size). The cache tensor's declared
        ``shape[2]`` is that post-unification block_size, so reading
        directly from layer 0's K-cache shape gives the right value.

        The full-attention layers operate through a view with the smaller
        effective block_size (= ``cache.shape[2] * cache.shape[-1] // full_head_dim``
        = 64), but their per-layer block_table is *padded* to the same width
        by the plugin's ``_block_tables_per_layer`` — so a single warmup
        width still aligns every layer's persistent buffer. The smaller
        effective block_size narrows the kernel's ``input_seq_len <=
        max_num_blocks_per_seq * block_size`` validation budget; warmup
        chunks stay well under that limit, and full-coverage of
        ``max_model_len`` for full-attn would require sizing the per-layer
        table separately (separate work).
        """
        import torch

        from models.tt_transformers.tt.common import num_blocks_in_seq

        ret = {
            "tokens": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "prompt_lens": torch.tensor([seq_len] * batch_size, dtype=torch.long),
            "empty_slots": list(range(batch_size)),
        }

        page_table_warmup = None
        if kv_cache is not None and kv_cache[model_id] is not None:
            cache = kv_cache[model_id][0][0]  # layer 0, K
            cache_block_size = cache.shape[2]
            # Match the plugin's runtime page_table width for layer 0's
            # group: ``cdiv(max_seq_len, declared_block_size)``.
            max_seq_len = self.model_args[model_id].max_seq_len
            num_blocks = num_blocks_in_seq(max_seq_len, cache_block_size)
            page_table_warmup = torch.zeros(batch_size, num_blocks, dtype=torch.int32)

        ret["page_table"] = page_table_warmup
        return ret

    # ── vLLM ``VllmModelForTextGeneration`` protocol shim ────────────────
    #
    # vLLM's ``is_text_generation_model`` predicate checks for
    # ``embed_input_ids``, ``forward(input_ids, positions)``, and
    # ``compute_logits`` on the resolved model class — that's how upstream
    # ``runner_type=="generate"`` validates a model is generative. Other TT
    # models (Gemma3, GptOss, etc.) get away without these because vLLM
    # finds an upstream torch implementation in its registry first and uses
    # *that* class for inspection, while the plugin's ``TT``-prefix logic
    # routes execution to the TT class. Gemma4 has no upstream vLLM impl,
    # so the inspection has to land on this class.
    #
    # Actual execution on the TT path goes through ``prefill_forward`` /
    # ``decode_forward`` (called by the TT runner via the
    # ``HybridAttentionForCausalLM`` overrides above), so these stubs are
    # never invoked. They exist purely to satisfy the protocol check.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; embeddings happen on TT via "
            "prefill_forward / decode_forward, not through this method."
        )

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; the TT runner invokes "
            "prefill_forward / decode_forward, not forward()."
        )

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; logits are produced on TT "
            "and surfaced through prefill_forward / decode_forward."
        )

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Build per-layer KVCacheSpec, honoring Gemma4's per-layer-type
        differences in ``head_dim`` and ``num_kv_heads``.

        The base ``HybridAttentionForCausalLM.get_kv_cache_spec`` assumes
        all layers share one ``head_size`` / ``num_kv_heads`` (only the
        sliding-vs-full *spec class* changes). That's true for Gemma3 but
        not Gemma4: sliding layers use ``head_dim`` (256 on E2B/E4B),
        full layers use ``global_head_dim`` (512). Sliding and full also
        each have their own ``num_key_value_heads`` (with the full count
        falling back to the sliding count when ``num_global_key_value_heads``
        is unset). Emitting one uniform spec made the K tensor produced
        by full-attention layers mismatch the cache shape and trip
        ``Last dim of input tensor must match last dim of cache tensor``
        in ``paged_update_cache``.

        vLLM's hybrid kv cache manager handles the resulting
        non-uniform-shape grouping fine (sliding layers form one group,
        full layers another), so the only thing that needs to differ
        between groups is the spec — block_size stays uniform.
        """
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError(
                f"{cls.__name__}.get_kv_cache_spec requires "
                "hf_config.text_config.layer_types (one of 'full_attention' / "
                "'sliding_attention' per layer); none found on this model"
            )

        sliding_kv_heads = text_config.num_key_value_heads
        sliding_head_dim = text_config.head_dim
        sliding_window = getattr(text_config, "sliding_window", None)
        full_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or sliding_kv_heads
        full_head_dim = getattr(text_config, "global_head_dim", None) or sliding_head_dim

        tp = parallel_config.tensor_parallel_size
        sliding_kv_heads_per_dev = sliding_kv_heads // tp
        full_kv_heads_per_dev = full_kv_heads // tp
        if cls._HYBRID_KV_CACHE_GROUPS_ENABLED:
            # A group with fewer kv heads than devices is replicated per device; report the full-attn
            # head count matching the device's per-token ratio, or the shared buffer's pages mismatch.
            mesh = max(1, (ttnn.get_num_devices() or 1) // max(1, parallel_config.data_parallel_size))
            dev_full = full_head_dim * (full_kv_heads // min(mesh, full_kv_heads))
            dev_slid = sliding_head_dim * (sliding_kv_heads // min(mesh, sliding_kv_heads))
            num = dev_full * sliding_kv_heads * sliding_head_dim
            den = dev_slid * full_head_dim
            if num % den:
                raise ValueError(
                    "hybrid kv-cache groups: cannot report a full-attention head count "
                    f"matching the device per-token ratio (full {full_kv_heads}x{full_head_dim}, "
                    f"sliding {sliding_kv_heads}x{sliding_head_dim}, mesh {mesh})"
                )
            full_kv_heads_per_dev = (num // den) // tp

        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        block_size = cache_config.block_size

        spec_per_layer = {}
        for i, lt in enumerate(layer_types):
            name = f"model.layers.{i}.self_attn"
            if lt == "sliding_attention":
                if sliding_window is None:
                    raise ValueError(
                        f"layer_types[{i}] is 'sliding_attention' but "
                        f"hf_config.sliding_window is None on {cls.__name__}"
                    )
                if cls._HYBRID_KV_CACHE_GROUPS_ENABLED:
                    # Hybrid ON: windowed ``SlidingWindowSpec`` so sliding layers
                    # form their own kv_cache_group(s) and only allocate the
                    # bounded window on device (memory-efficient; see the class
                    # docstring for the single-request ISL-cap tradeoff).
                    spec_per_layer[name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=sliding_kv_heads_per_dev,
                        head_size=sliding_head_dim,
                        dtype=dtype,
                        sliding_window=sliding_window,
                    )
                else:
                    # Hybrid OFF: ``FullAttentionSpec`` for sliding layers too,
                    # keeping their own (sliding) num_kv_heads/head_size. vLLM
                    # merges into one ``UniformTypeKVCacheSpecs`` group so the
                    # full block pool backs every request. Specs still declare
                    # full ``num_blocks``; when bounded_sliding is on,
                    # allocate_kv_cache_per_layer shrinks the physical sliding
                    # buffers and remaps page tables (demo parity).
                    spec_per_layer[name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=sliding_kv_heads_per_dev,
                        head_size=sliding_head_dim,
                        dtype=dtype,
                    )
            elif lt == "full_attention":
                spec_per_layer[name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=full_kv_heads_per_dev,
                    head_size=full_head_dim,
                    dtype=dtype,
                )
            else:
                raise ValueError(
                    f"Unsupported layer_type {lt!r} at layer {i} on "
                    f"{cls.__name__}; expected 'full_attention' or "
                    "'sliding_attention'"
                )
        return spec_per_layer

    @classmethod
    def _lanes_fold(cls):
        """The lane fold this class declares (``model_capabilities["folds_dp_into_lanes"]``)."""
        return _capability_flag(cls.model_capabilities, "folds_dp_into_lanes")

    @classmethod
    def _lane_prefill_group(cls):
        """Lane-parallel prefill router switch (``model_capabilities["lane_prefill_router"]``, this class's own
        key). The plugin's admission cap (the declared ``lane_prefill_group`` count) and gather window
        (``TT_PREFILL_GROUP_GATHER_MS``) make merged prefill steps pure lane groups for it; with the capability
        declared 0 the router still runs on whatever a step carries."""
        return _capability_flag(cls.model_capabilities, "lane_prefill_router")

    @classmethod
    def _lane_prefill_prompts(cls):
        """Prompts per lane per call the router builds with (``model_capabilities["lane_prefill_prompts"]``: 0
        off, 1 one per lane, k up to k per lane in one lane-batched walk), independent of what is declared."""
        return _capability_count(cls.model_capabilities, "lane_prefill_prompts")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        if optimizations not in (None, "performance", "accuracy"):
            raise ValueError("Gemma4 TT optimization profiles: None|performance|accuracy " f"(got {optimizations!r})")
        # ``accuracy`` is accepted for API parity with tt_transformers but must
        # NOT enable linear HiFi4/fp32 — that caused unicode garbage on LB 12B.
        # Production GeLU is always Accurate (see ``compute_config.gelu_variant``).
        if optimizations == "accuracy":
            logger.info(
                "Gemma4 optimizations=accuracy: keeping production defaults "
                "(GeLU=Accurate; no linear HiFi4/fp32 — those regress decode)"
            )

        model_path = hf_config._name_or_path
        lanes_fold = cls._lanes_fold() and tt_data_parallel > 1
        # The plugin opens the 4-chip Blackhole line as MeshShape(1, 4); a MoE build views the same chips as the
        # column mesh MeshShape(4, 1) (TP over the rows), where the routed-expert prefill can run the token-dispatch
        # path (tt/mesh_orientation.py; GEMMA4_QB2_COLUMN_MESH=0 keeps the row mesh).
        _text_cfg = getattr(hf_config, "text_config", hf_config)
        _column = should_reshape_to_column(
            tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1),
            blackhole=is_blackhole(),
            moe=bool(getattr(_text_cfg, "enable_moe_block", False)),
            data_parallel=tt_data_parallel,
            lanes=lanes_fold,
        )
        if _column is not None:
            logger.info(
                f"Gemma4: reshaping the {tuple(mesh_device.shape)} mesh to {_column} (column mesh, TP over the rows)"
            )
            mesh_device.reshape(ttnn.MeshShape(*_column))
        if lanes_fold:
            # Lane fold: one model on the full mesh; lanes are mesh columns, not
            # submeshes, and a global slot's lane is slot // lane_slots.
            submesh_devices = [mesh_device]
        else:
            submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        # Bounded sliding: mirror demo (auto policy + env). Hybrid-groups mode
        # still defaults ON when env unset — see ``_resolve_vllm_bounded_sliding``.
        bounded_sliding_kv_cache = _resolve_vllm_bounded_sliding(
            max_seq_len,
            mesh_device,
            model_path,
            hybrid_groups_enabled=cls._HYBRID_KV_CACHE_GROUPS_ENABLED,
        )
        # Before create_tt_model: it sizes the bounded KV specs from the ring.
        _auto_size_spec_ring(cls, hf_config, bounded_sliding_kv_cache)

        model_args = []
        model = []
        state_dict = None
        for submesh in submesh_devices:
            model_args_i, model_i, _, state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                dtype=ttnn.bfloat16,
                state_dict=state_dict,
                num_layers=n_layers,
                mesh_config=None,
                paged_attention_config=None,
                create_kv_cache=False,
                model_path=model_path,
                bounded_sliding_kv_cache=bounded_sliding_kv_cache,
            )
            prefill_trace_unsafe = _gemma4_prefill_trace_unsafe(model_i, bounded_sliding_kv_cache)
            # GH #49083 fix: pre-capture the prefill device traces at *warmup*
            # (see ``warmup_model_prefill``), routed through ``prefill_forward``
            # for the hybrid per-layer case. A cold *eager* prefill dispatched
            # after a shared-Generator traced-decode session (the release
            # workflow's evals phase) wedges the fetch queue (nlp_concat_heads)
            # at ISL=2048; capturing every bucket up front so runtime only
            # *replays* removes that trace->eager transition. Capturing at warmup
            # (before any traced decode) is what makes it safe — a lazy
            # first-runtime capture still hits the wedge.
            # GEMMA4_DISABLE_PREFILL_TRACE=1 restores the fully-eager prefill.
            prefill_trace_enabled = os.environ.get("GEMMA4_DISABLE_PREFILL_TRACE", "0") != "1"
            if prefill_trace_unsafe:
                logger.info(
                    "Gemma4 vLLM: prefill device trace for {} runs hybrid per-layer "
                    "page tables — warmup pre-captures each bucket through "
                    "prefill_forward (per-layer routing active), so runtime replays "
                    "instead of cold-eager capturing (#49083 fix). prefill_trace_enabled={}.",
                    model_path,
                    prefill_trace_enabled,
                )
            if lanes_fold:
                # The generator sees the global slot space while model and KV stay
                # per column; lane_slots maps a slot to its owner lane.
                _lanes = getattr(getattr(model_i, "mesh_config", None), "lanes", None)
                assert (
                    _lanes == tt_data_parallel
                ), f"lane fold expects mesh lanes == tt_data_parallel: {_lanes} != {tt_data_parallel}"
                model_i.lane_slots = max_batch_size // tt_data_parallel
            _patch_model_args(
                model_args_i,
                submesh,
                max_batch_size=(max_batch_size if lanes_fold else max_batch_size // tt_data_parallel),
                max_seq_len=max_seq_len,
                model_path=model_path,
                prefill_trace_enabled=prefill_trace_enabled,
                bounded_sliding=bounded_sliding_kv_cache,
            )
            if lanes_fold:
                # The batched-prefill flatten path is not lane-aware; prefills stay
                # per-user and the lane router batches same-length bursts.
                model_args_i.disable_batched_prefill = True
            # The shared TT vLLM cache allocator reads ``model.args.optimizations``;
            # mirror the text-transformer wrappers by exposing model_args here.
            model_i.args = model_args_i
            model_args.append(model_args_i)
            model.append(model_i)

        instance = cls(model, model_args, mesh_device)
        # Auto-enable multimodal when the checkpoint ships vision weights (mirrors
        # the demo's ``multimodal=True`` default). The vision tower + embedder are
        # built once, tensor-parallel over the full mesh; text-only serving is
        # unaffected.
        if _gemma4_has_vision_weights(state_dict) or getattr(state_dict, "is_placeholder", False):
            vision_dtype_name = os.environ.get("GEMMA4_VISION_DTYPE", "bfloat8_b")
            vision_dtype = getattr(ttnn, vision_dtype_name, ttnn.bfloat8_b)
            text_hidden_size = model[0].hidden_size
            (
                instance.vision_tower,
                instance.embed_vision,
                instance.image_token_id,
                instance.pad_token_id,
            ) = _build_gemma4_vision(
                mesh_device,
                state_dict,
                model_path,
                text_hidden_size,
                vision_dtype,
                mesh_config=getattr(model[0], "mesh_config", None),
                ccl_manager=getattr(model[0], "ccl_manager", None),
                weight_cache_path=model_args[0].weight_cache_path(ttnn.bfloat16),
            )
            instance.is_multimodal = instance.vision_tower is not None and (
                instance.embed_vision is not None or getattr(instance.vision_tower, "projects_to_text", False)
            )
            instance.embed_audio, instance.audio_token_id = _build_gemma4_audio(
                mesh_device,
                state_dict,
                model_path,
                text_hidden_size,
                weight_cache_path=model_args[0].weight_cache_path(ttnn.bfloat16),
            )
            instance.video_token_id = getattr(Gemma4ModelArgs.load_hf_config(model_path), "video_token_id", None)
            instance.is_multimodal = instance.is_multimodal or instance.embed_audio is not None
            logger.info(
                "Gemma4 vLLM: multimodal enabled (image_token_id={}, video_token_id={}, audio_token_id={}, "
                "pad_token_id={}, vision_dtype={}, audio={})",
                instance.image_token_id,
                instance.video_token_id,
                instance.audio_token_id,
                instance.pad_token_id,
                vision_dtype_name,
                "unified-embedder" if instance.embed_audio is not None else "none",
            )
        return instance

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat16)

    def _chunk_prefill_page_table(self, page_table, *, user_id, model_id=-1, kv_cache=None):
        """Use a full-attention layer's per-layer table for multi-chunk fill.

        vLLM's legacy ``page_table`` is ``block_tables_per_group[0]`` — the
        sliding group. Full-attention ``paged_fill_cache`` writes via
        ``chunk_page_table``, which must be sliced from the *full* group's
        block IDs at that group's **effective** block_size (the same value
        ``paged_fill_cache`` / chunked SDPA use). On 31B TP=4 that is 128
        (HMA-shared buffer declared as sliding ``[4, 64, 256]`` → full view
        ``eff_bs = 4*64*256/(1*512) = 128``), matching vLLM's unified full-
        group page-table column stride. Using head_dim-only scaling (32)
        walks past the allocated columns and fills chunk 2+ from zeros —
        the 16k garbage cliff. Sliding fill still uses ``layer_page_table``
        (``cache_position_modulo`` path in ``attention/prefill.py``).
        """
        del user_id
        from models.demos.gemma4.tt.attention.operations import effective_block_size

        model = self.model[model_id]
        per_layer = getattr(model, "_active_page_tables_per_layer", None)
        if per_layer is None or kv_cache is None:
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        text_config = getattr(model.hf_config, "text_config", model.hf_config)
        layer_types = getattr(text_config, "layer_types", None) or []
        full_idx = next((i for i, lt in enumerate(layer_types) if lt == "full_attention"), None)
        if full_idx is None or full_idx >= len(per_layer) or per_layer[full_idx] is None:
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        full_pt = per_layer[full_idx]
        # Persistent / ttnn tensors are device-side; chunk slicing needs a host
        # torch table. Fall back to legacy if the stash was already converted.
        if not isinstance(full_pt, torch.Tensor):
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        if full_idx >= len(kv_cache) or kv_cache[full_idx] is None:
            return super()._chunk_prefill_page_table(page_table, user_id=0, model_id=model_id, kv_cache=kv_cache)

        # Sequential path: legacy page_table is already the 1-row slice but
        # ``user_id`` is forced to 0. The cursor names the user: this runs before
        # the row is activated and slice content is not unique under lane rings.
        if (
            isinstance(page_table, torch.Tensor)
            and page_table.dim() > 1
            and int(page_table.shape[0]) == 1
            and int(full_pt.shape[0]) > 1
        ):
            row = int(getattr(model, "_sequential_row_cursor", 0))
            full_pt = full_pt[row : row + 1]

        cache = kv_cache[full_idx][0]
        attn = model.layers[full_idx].self_attn
        cfg = attn.config
        weights = getattr(attn, "weights", None)
        tp = getattr(getattr(model, "mesh_config", None), "tp", 1) or 1
        if weights is not None and getattr(weights, "kv_replicated", False):
            nkv_local = 1
        else:
            nkv_local = max(1, int(cfg.num_key_value_heads) // tp)
        full_block_size = int(effective_block_size(cache, int(cfg.head_dim), nkv_local))
        return full_pt, full_block_size

    def _prefill_user_chunk_plan(self, tokens, kwargs):
        """Return ``(chunk_users, prefill_seq_len)`` when B must be micro-batched.

        ``gemma4.tt.generator.Generator.prefill_forward_text`` chunks B>4 to
        avoid the P150x8 all_gather hang. The vLLM bridge's MRO skips that
        class — without this plan, the plugin's true-batched B=32 prefill
        wedges after cold capture.
        """
        if tokens is None or tokens.shape[0] <= 1:
            return None, None
        batch_size = int(tokens.shape[0])
        batch_seq_len = int(tokens.shape[1])
        prompt_lens = kwargs.get("prompt_lens")
        start_pos = kwargs.get("start_pos")
        prompt_lens_list = prompt_lens if prompt_lens is not None else [batch_seq_len] * batch_size
        if not isinstance(prompt_lens_list, list):
            prompt_lens_list = prompt_lens_list.tolist()
        num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens_list)
        prefill_seq_lens = [
            get_padded_prefill_len(seq_len - num_cached)
            for seq_len, num_cached in zip(prompt_lens_list, num_cached_per_user)
        ]
        page_table = kwargs.get("page_table")
        # Same padded bucket is enough: attention/prefill.py caps each slot's
        # paged_fill with valid_seq_lens (from last_token_idx). Hetero actual
        # lengths no longer force sequential.
        can_batch_prefill = (
            page_table is not None
            and batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
            and all(n == 0 for n in num_cached_per_user)
        )
        if not can_batch_prefill:
            return None, None
        padded_batch = next(
            (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
            self.model_args[0].max_batch_size,
        )
        max_users = resolve_batched_prefill_chunk_users(padded_batch, prefill_seq_lens[0])
        if batch_size <= max_users or padded_batch > self.model_args[0].max_batch_size:
            return None, prefill_seq_lens[0]
        return max_users, prefill_seq_lens[0]

    def _slice_prefill_kwargs(self, kwargs, chunk_start, chunk_end):
        """Slice host prefill tensors for a user micro-batch.

        Remapped sliding block IDs are already global (from the full-batch
        remap). Re-slot users to ``0..chunk_size-1`` so tt_transformers'
        padded_batch path stays in-range; physical block IDs stay correct.
        """
        chunk_size = chunk_end - chunk_start
        chunk = dict(kwargs)
        tokens = kwargs["tokens"]
        chunk["tokens"] = tokens[chunk_start:chunk_end]
        page_table = kwargs.get("page_table")
        if page_table is not None:
            chunk["page_table"] = page_table[chunk_start:chunk_end]
        prompt_lens = kwargs.get("prompt_lens")
        if prompt_lens is not None:
            if isinstance(prompt_lens, torch.Tensor):
                chunk["prompt_lens"] = prompt_lens[chunk_start:chunk_end]
            else:
                chunk["prompt_lens"] = list(prompt_lens)[chunk_start:chunk_end]
        start_pos = kwargs.get("start_pos")
        if start_pos is not None:
            chunk["start_pos"] = list(start_pos)[chunk_start:chunk_end]
        # Local slots 0..N-1 (required by padded_batch placement).
        chunk["empty_slots"] = list(range(chunk_size))
        return chunk

    def _merge_prefill_chunk_results(self, batch_size, sampling_params, chunk_results):
        merged_output = None
        merged_tokens = None
        merged_log_probs = None
        for chunk_start, chunk_end, chunk_result in chunk_results:
            if sampling_params is not None:
                chunk_tokens, chunk_log_probs = chunk_result
                if merged_tokens is None:
                    merged_tokens = torch.zeros(
                        (batch_size, *chunk_tokens.shape[1:]),
                        dtype=chunk_tokens.dtype,
                        device=chunk_tokens.device,
                    )
                merged_tokens[chunk_start:chunk_end] = chunk_tokens
                if isinstance(chunk_log_probs, tuple):
                    if merged_log_probs is None:
                        merged_log_probs = (
                            torch.zeros(
                                (batch_size, *chunk_log_probs[0].shape[1:]),
                                dtype=chunk_log_probs[0].dtype,
                                device=chunk_log_probs[0].device,
                            ),
                            torch.zeros(
                                (batch_size, *chunk_log_probs[1].shape[1:]),
                                dtype=chunk_log_probs[1].dtype,
                                device=chunk_log_probs[1].device,
                            ),
                        )
                    merged_log_probs[0][chunk_start:chunk_end] = chunk_log_probs[0]
                    merged_log_probs[1][chunk_start:chunk_end] = chunk_log_probs[1]
                else:
                    if merged_log_probs is None:
                        merged_log_probs = torch.zeros(
                            (batch_size, *chunk_log_probs.shape[1:]),
                            dtype=chunk_log_probs.dtype,
                            device=chunk_log_probs.device,
                        )
                    merged_log_probs[chunk_start:chunk_end] = chunk_log_probs
            else:
                if merged_output is None:
                    merged_output = torch.zeros(
                        (batch_size, *chunk_result.shape[1:]),
                        dtype=chunk_result.dtype,
                        device=chunk_result.device,
                    )
                merged_output[chunk_start:chunk_end] = chunk_result
        if sampling_params is not None:
            return merged_tokens, merged_log_probs
        return merged_output

    def _g4_router_reject(self, reason):
        logger.debug("Gemma4 vLLM: lane router ineligible -- {}", reason)
        return None

    def _try_lane_parallel_prefill(self, kwargs, full_page_tables, raw_page_tables_per_layer=None, enable_trace=True):
        """Split a merged prefill step into lane-parallel groups (GEMMA4_LANE_PREFILL_ROUTER=1).

        Rounds covering every lane at one prompt length prefill in one chunk walk and the rest
        recurse serially; returns the call's full [batch, 1, vocab] logits, or None to fall back.
        """
        if not self._lane_prefill_group():
            return None
        if getattr(self, "_g4_lane_router_off", None):
            return self._g4_router_reject(f"router off after a failed boot warmup ({self._g4_lane_router_off})")
        model = self.model[0]
        mesh_cfg = getattr(model, "mesh_config", None)
        if mesh_cfg is None or not getattr(mesh_cfg, "lane_sharded", False):
            return None
        tokens = kwargs.get("tokens")
        slots = kwargs.get("empty_slots")
        prompt_lens = kwargs.get("prompt_lens")
        if tokens is None or slots is None or prompt_lens is None:
            return self._g4_router_reject(
                f"missing kwargs: tokens={tokens is not None} slots={slots is not None} plens={prompt_lens is not None}"
            )
        if kwargs.get("model_id_warmup") is not None or kwargs.get("warmup_prefill"):
            return self._g4_router_reject("warmup call")
        if kwargs.get("sampling_params") is not None:
            return self._g4_router_reject("device sampling_params present")

        # A text batch on a multimodal class still carries pixel_values=[None]*B,
        # so reject only when some entry is real.
        def _mm_present(v):
            if v is None:
                return False
            if isinstance(v, (list, tuple)):
                return any(x is not None for x in v)
            return True

        if any(_mm_present(kwargs.get(k)) for k in _GEMMA4_MM_PAYLOAD_KEYS + ("images", "image_embeds")):
            return self._g4_router_reject("multimodal kwargs")
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and any(int(p) > 0 for p in torch.as_tensor(start_pos).reshape(-1)):
            return self._g4_router_reject("cached prefix (start_pos > 0)")
        B = int(tokens.shape[0])
        lanes = mesh_cfg.lanes
        lane_slots = int(getattr(model, "lane_slots", 0) or 32)
        # The group walk gets raw-width tokens (the serial path pads inside prepare);
        # pad here or the sliding fill hands SDPA a sub-tile V.
        S = int(tokens.shape[-1])
        S_pad = int(get_padded_prefill_len(S))
        # The lane-frame traces captured at boot ((k, S) buckets): a call whose rounds x padded length match one
        # replays it; with a one-round bucket at this length a single prompt (B < lanes) routes through the frame
        # as one real lane plus scratch lanes instead of the owner-lane eager path (the served c1 row).
        captured = set(getattr(self, "_g4_lane_prefill_traces", None) or {})
        traced_single = (1, S_pad) in captured
        if B < lanes and not traced_single:
            return self._g4_router_reject(f"B={B} < lanes={lanes}")
        if not full_page_tables:
            return self._g4_router_reject("no per-layer page tables")
        g_idxs = [i for i, lt in enumerate(model.hf_config.layer_types) if lt == "full_attention"]
        if not g_idxs or full_page_tables[g_idxs[0]] is None:
            return self._g4_router_reject("no full-attention table")
        bad_pt = [i for i, pt in enumerate(full_page_tables) if pt is None or isinstance(pt, ttnn.Tensor)]
        if bad_pt:
            return self._g4_router_reject(
                f"non-host per-layer tables at layers {bad_pt[:6]} (of {len(full_page_tables)})"
            )

        plens = [int(p) for p in torch.as_tensor(prompt_lens).reshape(-1)][:B]
        lane_of = [int(s) // lane_slots for s in slots]
        buckets = [[q for q in range(B) if lane_of[q] == ln] for ln in range(lanes)]
        # k prompts per lane per gathered step (``lane_prefill_group`` as a count): every request of the step is
        # routed, lanes short of a round compute on scratch rows, and as many rounds as fit the prefill chunk
        # (and the lane's slots) prefill as one lane-batched walk: the rounds' users as one stream per column.
        # k == 1 keeps the one-prompt-per-lane groups: full rounds only, the rest serial.
        k = max(1, int(self._lane_prefill_prompts()))
        max_chunk = int(getattr(self.model_args[0], "max_prefill_chunk_size", 0) or 0)
        rounds_per_call = 1
        if k > 1 and self._g4_lane_batched_walks():
            if max_chunk >= S_pad:
                rounds_per_call = max(1, min(lane_slots, max_chunk // S_pad))
        # a traced call size at this length wins over a larger eager one (replays cost a fraction of a walk)
        rounds_per_call = lane_trace_rounds_for_call(captured, S_pad, rounds_per_call)
        rounds = lane_prefill_rounds(buckets, plens, allow_scratch=k > 1 or traced_single)
        if not rounds:
            return self._g4_router_reject(
                f"no full round: B={B} slots={[int(x) for x in slots][:8]} lane_slots={lane_slots} "
                f"buckets={[len(b) for b in buckets]} lens={[plens[b[0]] for b in buckets if b]}"
            )
        routed = {q for rnd in rounds for q in rnd if q is not None}
        remainder = [q for q in range(B) if q not in routed]
        # The captured frames at this length split the step by the replay cost model (a call per frame, scratch
        # rounds priced like real ones); without one the eager walk takes ``rounds_per_call`` rounds per call.
        plan = lane_prefill_trace_plan(captured, len(rounds), S_pad, single_chunk=S_pad <= max_chunk)
        if plan:
            calls, frames, i = [], [], 0
            for frame_k, n_real in plan:
                calls.append(rounds[i : i + n_real])
                frames.append(frame_k)
                i += n_real
        else:
            calls = [rounds[i : i + rounds_per_call] for i in range(0, len(rounds), rounds_per_call)]
            frames = [None] * len(calls)

        gpt = full_page_tables[g_idxs[0]]
        gpt = gpt if gpt.dim() > 1 else gpt.unsqueeze(0)
        out = torch.zeros(B, 1, model.vocab_size)
        # One line per routed step (info: the served log shows the call sizes the router actually built)
        logger.info(
            "Gemma4 vLLM: lane router step B={} rounds={} calls={} rounds_per_call={} k={} serial={} seq={} "
            "lanes_with_prompts={} frames={}",
            B,
            len(rounds),
            len(calls),
            rounds_per_call,
            k,
            len(remainder),
            S_pad,
            sum(1 for b in buckets if b),
            [f if f is not None else "eager" for f in frames],
        )
        for call, frame_k in zip(calls, frames):
            kc = len(call)
            toks = torch.zeros(lanes, kc, S_pad, dtype=tokens.dtype)
            plens_call = [1] * (lanes * kc)
            tables = torch.zeros(lanes, kc, int(gpt.shape[-1]), dtype=torch.int32)
            for r, rnd in enumerate(call):
                # Scratch rows carry the round's length: the round shares one valid length across lanes.
                length = next(plens[q] for q in rnd if q is not None)
                for ln, q in enumerate(rnd):
                    plens_call[ln * kc + r] = length
                    if q is None:
                        continue
                    toks[ln, r, :S] = tokens[q]
                    # Plugin per-layer tables arrive in local prefill order (row
                    # i = request i), not slot-indexed.
                    tables[ln, r] = gpt[q].to(torch.int32)
            per_layer = lane_prefill_frame_tables(full_page_tables, call, lane_slots, model._lane_scratch_block_for)
            bucket = (frame_k, S_pad) if frame_k is not None else None
            real_rounds = list(range(kc))
            if bucket is not None and bucket[0] > kc:
                # pad the call with all-scratch rounds up to the captured frame (a replayed row each, no prompt)
                pad = bucket[0] - kc
                call = list(call) + [[None] * lanes for _ in range(pad)]
                toks = torch.cat([toks, torch.zeros(lanes, pad, S_pad, dtype=tokens.dtype)], dim=1)
                tables = torch.cat([tables, torch.zeros(lanes, pad, int(gpt.shape[-1]), dtype=torch.int32)], dim=1)
                plens_call = [
                    plens_call[ln * kc + r] if r < kc else plens_call[ln * kc]
                    for ln in range(lanes)
                    for r in range(bucket[0])
                ]
                kc = bucket[0]
                per_layer = lane_prefill_frame_tables(full_page_tables, call, lane_slots, model._lane_scratch_block_for)
            if bucket is not None:
                # the replay uploads the fitted per-layer tables itself and reads the trace's output; the frame is
                # named before the replay so a device hang inside it leaves its path in the log (window 9's last
                # step had the router line and nothing after: the replay line follows the blocking readback)
                logger.debug(
                    "Gemma4 vLLM: lane prefill frame {}x{} replaying a {}-round call ({} real rounds, {} lanes with prompts)",
                    bucket[0],
                    bucket[1],
                    len(call),
                    len(real_rounds),
                    sum(1 for b in buckets if b),
                )
                # only the lanes with a prompt in this call are read back (the served c1 frame: one of four)
                real_lanes = sorted({ln for rnd in call for ln, q in enumerate(rnd) if q is not None})
                lane_kw = {"lanes": real_lanes} if len(real_lanes) < lanes else {}
                out_rows = self._g4_lane_prefill_trace_replay(
                    bucket, toks, tables, per_layer, kwargs.get("kv_cache"), plens_call, rounds=real_rounds, **lane_kw
                )
                if out_rows is not None:
                    self._g4_lane_trace_replays = int(getattr(self, "_g4_lane_trace_replays", 0)) + 1
                    (logger.info if self._g4_lane_trace_replays == 1 else logger.debug)(
                        "Gemma4 vLLM: lane prefill trace replay {}x{} (call {} of this process)",
                        bucket[0],
                        bucket[1],
                        self._g4_lane_trace_replays,
                    )
                    out_rows = out_rows.reshape(lanes, kc, -1)
                    for r, rnd in enumerate(call):
                        for ln, q in enumerate(rnd):
                            if q is not None:
                                out[q] = out_rows[ln, r, : model.vocab_size].reshape(1, -1)
                    continue
            model._active_page_tables_per_layer = per_layer
            # Upload the call's tables: ``ttnn_prefill_forward`` resolves the stash through
            # ``_page_tables_to_ttnn``, which hands back a key's existing device buffers without refreshing
            # them, so without this copy every call after a key's first would prefill through that first
            # call's block ids (GEMMA4_LANE_PREFILL_TABLE_UPLOAD=0 reproduces that path for evidence only).
            if lane_prefill_table_upload_enabled():
                self._install_per_layer_page_tables(model, per_layer, writer="prefill")
            elif not getattr(self, "_g4_lane_table_upload_off_logged", False):
                self._g4_lane_table_upload_off_logged = True
                logger.warning(
                    "Gemma4 vLLM: GEMMA4_LANE_PREFILL_TABLE_UPLOAD=0 -- lane prefill calls keep the per-layer page "
                    "tables of the first call per table key (the pre-fix path; evidence runs only)"
                )
            out_rows = self.prefill_forward_lanes(
                toks if kc > 1 else toks[:, 0], tables, kwargs.get("kv_cache"), plens_call, slot_ids=None
            )
            out_rows = out_rows.reshape(lanes, kc, -1)
            for r, rnd in enumerate(call):
                for ln, q in enumerate(rnd):
                    if q is not None:
                        out[q] = out_rows[ln, r, : model.vocab_size].reshape(1, -1)
        if hasattr(model, "_active_page_tables_per_layer"):
            del model._active_page_tables_per_layer

        if remainder:
            sub_kwargs = dict(kwargs)
            sub_kwargs["tokens"] = tokens[remainder]
            sub_kwargs["prompt_lens"] = [plens[q] for q in remainder]
            sub_kwargs["empty_slots"] = [slots[q] for q in remainder]
            sub_kwargs["enable_trace"] = enable_trace
            if start_pos is not None:
                sp = torch.as_tensor(start_pos).reshape(-1)
                sub_kwargs["start_pos"] = [int(sp[q]) for q in remainder]
            pt_kw = kwargs.get("page_table")
            if isinstance(pt_kw, torch.Tensor) and pt_kw.dim() >= 2 and pt_kw.shape[0] >= B:
                sub_kwargs["page_table"] = pt_kw[remainder]
            sub_pts = None
            if raw_page_tables_per_layer is not None:
                sub_pts = []
                for pt in raw_page_tables_per_layer:
                    if isinstance(pt, torch.Tensor) and pt.dim() >= 2 and pt.shape[0] >= B:
                        sub_pts.append(pt[remainder])
                    else:
                        sub_pts.append(pt)
            logger.debug(
                "Gemma4 vLLM: lane router serial remainder -- {} request(s) on lanes {}",
                len(remainder),
                sorted({lane_of[q] for q in remainder}),
            )
            sub_out = self.prefill_forward(page_tables_per_layer=sub_pts, **sub_kwargs)
            for pos, q in enumerate(remainder):
                out[q] = sub_out[pos]
        return out

    def _prefill_page_table_block_sizes(self, kv_cache):
        """Per-layer tokens per page-table column, from the per-block invariant the attention's paged ops use
        (the declared block size when the layer's view matches its buffer, the effective one for an HMA-shared
        buffer or a cache with a different per-chip head count)."""
        from models.demos.gemma4.tt.generator import layer_paged_block_size
        from models.tt_transformers.tt.common import get_block_size

        layers = getattr(self.model[0], "layers", [])
        # The plugin passes ``kv_caches`` nested per submesh ([dp][layer][k, v]);
        # the generator-level callers pass one submesh's per-layer list.
        if (
            kv_cache
            and len(kv_cache) != len(layers)
            and isinstance(kv_cache[0], (list, tuple))
            and len(kv_cache[0]) == len(layers)
        ):
            kv_cache = kv_cache[0]
        try:
            default = int(get_block_size(kv_cache))
        except Exception:
            return []
        sizes = []
        for i, layer in enumerate(layers):
            bs = default
            attn = getattr(layer, "self_attn", None)
            cfg = getattr(attn, "config", None)
            if cfg is not None and i < len(kv_cache) and kv_cache[i] is not None:
                try:
                    bs = int(layer_paged_block_size(attn, cfg, kv_cache[i][0]))
                except (AttributeError, TypeError):  # a cache object without a shape: keep the declared size
                    bs = default
                # a ValueError (a view the paged ops cannot address) propagates, as on the demo path
            sizes.append(bs)
        return sizes

    def _mask_stale_prefill_page_tables(self, full_page_tables, kwargs):
        """Zero page-table columns past each request's allocation (they hold earlier occupants'
        block ids) in the per-layer tables and the legacy ``page_table``.
        GEMMA4_MASK_STALE_PAGE_TABLE_COLS=0 keeps the raw plugin rows."""
        if not gemma4_env_flag("GEMMA4_MASK_STALE_PAGE_TABLE_COLS", "1"):
            return full_page_tables, kwargs
        prompt_lens = kwargs.get("prompt_lens")
        kv_cache = kwargs.get("kv_cache")
        if prompt_lens is None or kv_cache is None:
            return full_page_tables, kwargs
        sizes = self._prefill_page_table_block_sizes(kv_cache)
        if not sizes:
            if not getattr(self, "_g4_stale_page_table_cols_unresolved", False):
                self._g4_stale_page_table_cols_unresolved = True
                logger.warning(
                    "Gemma4 vLLM: could not resolve per-layer block sizes from kv_cache; "
                    "prefill page tables keep the plugin's raw rows (stale columns unmasked)"
                )
            return full_page_tables, kwargs
        zeroed = 0
        if full_page_tables:
            full_page_tables, zeroed = mask_page_table_columns_past_allocation(full_page_tables, prompt_lens, sizes)
        legacy = kwargs.get("page_table")
        if isinstance(legacy, torch.Tensor):
            (masked_legacy,), n_legacy = mask_page_table_columns_past_allocation([legacy], prompt_lens, sizes[:1])
            if n_legacy:
                kwargs = dict(kwargs)
                kwargs["page_table"] = masked_legacy
                zeroed += n_legacy
        if zeroed:
            if not getattr(self, "_g4_stale_page_table_cols_logged", False):
                self._g4_stale_page_table_cols_logged = True
                logger.info(
                    "Gemma4 vLLM: zeroed {} stale page-table column(s) past the requests' allocations "
                    "before prefill (plugin rows carry earlier occupants' block ids; further hits at debug level)",
                    zeroed,
                )
            else:
                logger.debug("Gemma4 vLLM: zeroed {} stale page-table column(s) before prefill", zeroed)
        return full_page_tables, kwargs

    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        t_entry = time.perf_counter()
        for m in self.model:
            m._g4_pf_ms = {}  # host phases of this call (install / trace / lm_head / readback), see the perf line
        self._g4_alloc_dump("prefill_")
        self._g4_arm_alloc_check("prefill")
        prefill_trace_ids_before = self._g4_prefill_trace_ids(self)
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
            kwargs["tokens"] = tokens

        chunk_users, prefill_seq_len = self._prefill_user_chunk_plan(tokens, kwargs)
        enable_trace = kwargs.pop("enable_trace", True)
        batch_size = int(tokens.shape[0]) if tokens is not None else 1
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace, batch_size=batch_size)

        prompt_lens = kwargs.get("prompt_lens")
        seq_len = None
        if prompt_lens is not None:
            try:
                seq_len = int(max(prompt_lens))
            except (TypeError, ValueError):
                seq_len = None
        if seq_len is None and tokens is not None and hasattr(tokens, "shape"):
            seq_len = int(tokens.shape[-1])

        # Remap the *full* batch first so sliding block IDs stay global. Chunk
        # loops only slice those tables (never re-remap local rows 0..N).
        # Snapshot ring occupancy *before* the remap: the remap inserts the
        # request being prefilled into the slot map, so a check made after it
        # would see a non-empty map even for the very first request and skip
        # the clear it actually needs (stale warmup KV -> garbage for exactly
        # that one user).
        rings_live_before_prefill = len(getattr(self, "_bounded_ring_slot_map", None) or {})
        full_page_tables = self._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        full_page_tables, kwargs = self._mask_stale_prefill_page_tables(full_page_tables, kwargs)
        full_page_tables = self._pad_sliding_page_tables_for_bounded(
            full_page_tables, kwargs.get("kv_cache"), row_slots=kwargs.get("empty_slots")
        )
        full_page_tables = self._pad_page_tables_batch_to_max(full_page_tables)
        if self._bounded_sliding_kv_cache and full_page_tables:
            sliding_idxs = self._sliding_layer_indices()
            if sliding_idxs and full_page_tables[sliding_idxs[0]] is not None:
                kwargs["page_table"] = full_page_tables[sliding_idxs[0]]
            # Bounded rings reuse dense physical blocks [u*W,(u+1)*W) across
            # requests. Clear only on a fresh prefill (all start_pos == 0).
            # start_pos may be a list or numpy/torch vector — never use
            # ``array or []`` (ambiguous truth value for multi-element arrays).
            if not rings_live_before_prefill:
                # Fresh wave (no live rings): drop captured decode traces so the
                # next decode recaptures against this wave's state. Replaying a
                # decode trace captured before the wave -- in practice the
                # warmup-captured traces -- corrupts every batch after the first
                # even when all trace input tensors are restaged each step
                # (measured on P150x8 / 12B / 32k bounded: batch 1 clean, every
                # later batch nondeterministic garbage at any concurrency >= 2;
                # ring fingerprints show prefill hidden states already diverging
                # for identical inputs). With this release the first serving
                # wave recaptures once and 16+ consecutive batches across
                # concurrencies 1..32 and bucket switches replay it cleanly.
                # Program caches persist, so the one recapture costs seconds.
                self._release_decode_traces_for_fresh_wave()
            start_pos_for_clear = kwargs.get("start_pos")
            if start_pos_for_clear is None:
                self._clear_bounded_sliding_kv_rings(kwargs.get("kv_cache"), live_before=rings_live_before_prefill)
            else:
                try:
                    start_vals = [int(p) for p in list(start_pos_for_clear)]
                except TypeError:
                    start_vals = [int(start_pos_for_clear)]
                if all(p == 0 for p in start_vals):
                    self._clear_bounded_sliding_kv_rings(kwargs.get("kv_cache"), live_before=rings_live_before_prefill)

        # Align vLLM chunked-prefill continuations to SDPA q_chunk_size (128).
        # tokens[:, :prompt_lens] still holds the full prefix, so aligning
        # start_pos down re-prefills the unaligned boundary (Galaxy pattern).
        start_pos = kwargs.get("start_pos")
        num_computed = [int(n) for n in start_pos] if start_pos is not None else None
        mm_restore_users = []
        if num_computed is not None:
            aligned = align_num_cached_tokens_to_sdpa(num_computed)
            if self.is_multimodal:
                # The plugin forwards every item's placeholder range next to the payloads (None for the items a
                # prefix-cache hit covered); a restart inside a vision item moves to the item's start, and an
                # item ending after the restart is re-embedded even when no payload came with this batch.
                offsets = _per_slot_items(kwargs, "mm_placeholder_offsets", batch_size)
                lengths = _per_slot_items(kwargs, "mm_placeholder_lengths", batch_size)
                aligned = gemma4_mm_restart_outside_vision_items(
                    aligned,
                    tokens,
                    offsets,
                    lengths,
                    [t for t in (self.image_token_id, getattr(self, "video_token_id", None)) if t is not None],
                )
                mm_restore_users = gemma4_mm_items_to_restore(aligned, prompt_lens, offsets, lengths)
            kwargs["start_pos"] = aligned

        # Multimodal batch: encode the images / video frames / audio and stage each
        # user's merged prompt embeddings for the text prefill below, which runs
        # per user and eagerly (the lane router and batched prefill decline it).
        mm_users = 0
        mm_stage_ms = 0.0
        self._g4_check_multimodal_admissible(kwargs)
        if self.is_multimodal and (_gemma4_has_mm_inputs(kwargs) or mm_restore_users):
            mm_t0 = time.perf_counter()
            mm_users = self._stage_multimodal_prefill(
                tokens, kwargs, prompt_lens, num_computed=num_computed, page_tables_per_layer=full_page_tables
            )
            mm_stage_ms = (time.perf_counter() - mm_t0) * 1000.0
            enable_trace = False

        t0 = time.perf_counter()

        # Opt-in lane-parallel prefill router: a full lane group prefills in one
        # chunk walk; None falls through to the serial path.
        routed = self._try_lane_parallel_prefill(
            kwargs, full_page_tables, raw_page_tables_per_layer=page_tables_per_layer, enable_trace=enable_trace
        )
        if routed is not None:
            dt = time.perf_counter() - t0
            logger.info(
                "[gemma4-vllm-perf] lane-router prefill TTFT={:.1f} ms | batch={} | seq_len={} | phases: {}",
                dt * 1000.0,
                len(routed),
                int(kwargs["tokens"].shape[-1]),
                self._g4_lane_phases(t_entry, t0),
            )
            self._perf_decode_tokens = 0
            self._perf_decode_s = 0.0
            return routed

        # B>4 true-batched prefill hangs on P150x8 after the first all_gather.
        # Micro-batching with remapped local slots (0..chunk) also breaks decode:
        # KV lands in the right physical blocks but per-slot decode state does not.
        # Force the proven per-user prefill loop (global empty_slots) instead.
        force_sequential = chunk_users is not None
        if force_sequential:
            logger.info(
                "Gemma4 vLLM: sequential prefill for batch_size={} " "(true-batched B>{} hangs on P150x8; user_cap={})",
                batch_size,
                max_batched_prefill_users(),
                max_batched_prefill_users(),
            )

        # Decide if this call will truly batch (same *padded* bucket; hetero
        # actual OK via per-slot valid_seq_lens). Sequential keeps per-layer
        # tables and slices to the active row (see mixin
        # ``_activate_sequential_per_layer_row``).
        prompt_lens_list = prompt_lens
        if prompt_lens_list is not None and not isinstance(prompt_lens_list, list):
            prompt_lens_list = list(prompt_lens_list)
        start_pos_for_plan = kwargs.get("start_pos")
        num_cached_for_plan = (
            [int(n) for n in start_pos_for_plan]
            if start_pos_for_plan is not None
            else ([0] * len(prompt_lens_list) if prompt_lens_list is not None else [0])
        )
        if prompt_lens_list is not None:
            prefill_seq_lens_plan = [
                get_padded_prefill_len(int(seq_len) - num_cached)
                for seq_len, num_cached in zip(prompt_lens_list, num_cached_for_plan)
            ]
            padded_lens_equal = len(set(prefill_seq_lens_plan)) == 1
        else:
            prefill_seq_lens_plan = None
            padded_lens_equal = True
        will_batch = (
            batch_size > 1
            and not force_sequential
            and kwargs.get("page_table") is not None
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
            and padded_lens_equal
            and all(n == 0 for n in num_cached_for_plan)
            and mm_users == 0
            # Batched prefill is opt-in via G4_FORCE_BATCH_PREFILL=1: it works
            # under BOTH unbounded and bounded sliding since the 2026-09 fixes
            # (identity-gated, byte-equal vs the sequential/B=1 references):
            #  - per-layer page tables scattered local->slot rows (below) — the
            #    non-identity empty_slots KV misplacement, debt #1 bug A;
            #  - boot-time batched-shape warmup captures (generator_trace) — the
            #    runtime cold-capture replay corruption, debt #1 bug B;
            #  - bounded per-slot ring fills deferred after lm_head, full-length
            #    under modulo, wrap-boundary merged (attention/prefill.py).
            # Historic exclusion rationales ("per-slot decode state", "deferred
            # lm-head extraction") are retired — the extraction was always
            # correct. Default-off pending the open conc32 small-ISL
            # wave-boundary corruption (~1-3/32 after a prior wave, prefill-mode
            # independent; repro: wave_smoke) and per-quadrant memory budgets
            # (see GEMMA4_MAX_BATCHED_PREFILL_USERS / GEMMA4_TAIL_POOL_SLOTS).
            # GEMMA4_PACKED_PREFILL=1 takes the same per-layer-table scatter/install (rows by physical slot)
            # and then runs the packed forward from the wrapper's prefill_forward_text.
            and (
                os.environ.get("G4_FORCE_BATCH_PREFILL", "0") == "1"
                or gemma4_packed_prefill_enabled()
                # GEMMA4_TRACE_BATCHED_PREFILL=1: a call that pads to a boot-captured batched bucket replays it
                or batched_prefill_trace_bucket(
                    batch_size,
                    kwargs.get("empty_slots"),
                    self.model_args[0].max_batch_size,
                    prefill_seq_lens_plan[0] if prefill_seq_lens_plan else None,
                    mesh_shape=getattr(self.model[0].mesh_device, "shape", None),
                )
            )
        )
        use_sequential = batch_size > 1 and not will_batch

        if will_batch:
            slots = kwargs.get("empty_slots")
            slots = [int(s) for s in slots] if slots is not None else None
            if slots is not None and slots != list(range(len(slots))):
                # Batched prefill places tokens at their device slots
                # (``prefill_ids[slot] ← user i``) and the per-layer KV fill
                # indexes tables by slot (``paged_fill_cache(batch_idx=slot)``),
                # but the plugin's per-layer tables arrive in *local* prefill
                # order (row i = user i). With non-identity ``empty_slots`` —
                # exactly when another request is mid-decode — user i's KV was
                # written through table row ``slot_i`` of a local-ordered
                # table: each user's KV landed in the next user's blocks and
                # the first user's blocks were never written (debt #1's single
                # corrupted user). Scatter local rows to slot rows, mirroring
                # ``_get_prefill_user_page_table``'s legacy scatter.
                scattered = []
                for pt in full_page_tables or []:
                    if pt is None or not isinstance(pt, torch.Tensor) or pt.dim() < 2:
                        scattered.append(pt)
                        continue
                    out = torch.zeros_like(pt)
                    for i, slot in enumerate(slots):
                        if i < pt.shape[0] and slot < out.shape[0]:
                            out[slot] = pt[i]
                    scattered.append(out)
                full_page_tables = scattered

        # Always install per-layer tables when available. Under bounded sliding
        # kwargs["page_table"] is the remapped *sliding* table; clearing the
        # per-layer stash makes full-attention layers inherit it (empty thought
        # / ~10% GPQA). Sequential tt_transformers still slices a 1-row legacy
        # page_table and forces user_id=0 — Gemma4Model slices the multi-row
        # per-layer stash down to that active row (see ttnn_prefill_forward).
        per_submesh = self._chunk_page_tables_per_dp(full_page_tables)
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                self._install_per_layer_page_tables(m, pt_for_submesh, writer="prefill")
        else:
            for m in self.model:
                if hasattr(m, "_active_page_tables_per_layer"):
                    del m._active_page_tables_per_layer
        if use_sequential:
            logger.info(
                "Gemma4 vLLM: sequential per-user prefill for batch_size={} "
                "(per-layer page tables kept for hybrid/bounded full-attn)",
                batch_size,
            )

        if prefill_seq_len is not None:
            prefill_seq_lens = [prefill_seq_len]
        elif prefill_seq_lens_plan is not None:
            prefill_seq_lens = prefill_seq_lens_plan
        elif seq_len is not None:
            prefill_seq_lens = [get_padded_prefill_len(seq_len)]
        elif tokens is not None:
            prefill_seq_lens = [get_padded_prefill_len(int(tokens.shape[1]))]
        else:
            prefill_seq_lens = [128]
        can_batch = will_batch
        enable_trace = resolve_gemma4_prefill_trace_enable(
            enable_trace,
            self.model[0],
            self.model_args[0],
            batch_size=1 if use_sequential else batch_size,
            prefill_seq_lens=prefill_seq_lens,
            can_batch_prefill=can_batch,
        )
        kwargs["enable_trace"] = enable_trace

        args0 = self.model_args[0]
        prev_disable = getattr(args0, "disable_batched_prefill", False)
        if use_sequential:
            args0.disable_batched_prefill = True
        # Arm the gemma4-owned batched-consumption trace hygiene (per-slot slice
        # dealloc + RM return + retire list in process_logits_after_prefill_trace)
        # only when this call truly batches; the single-user path must keep the
        # original contract (persistent trace output input, TILE return).
        for m in self.model:
            m._g4_batched_prefill_consumption = will_batch

        try:
            with self._route_per_layer_page_tables(per_submesh):
                out = super().prefill_forward_text(**kwargs)
        finally:
            if use_sequential:
                args0.disable_batched_prefill = prev_disable
            for m in self.model:
                m._g4_batched_prefill_consumption = False
            t_clear = time.perf_counter()
            self._clear_sequential_batch_page_tables()
            clear_ms = (time.perf_counter() - t_clear) * 1000.0

        # Device work is synchronous after the TT forward returns — same
        # wall clock the metal demo attributes to ``inference_prefill`` / TTFT.
        dt = time.perf_counter() - t0
        ttft_ms = dt * 1000.0
        prefill_tok_s = (float(seq_len) / dt) if (seq_len and dt > 0) else 0.0
        chunk = getattr(self.model_args[0], "max_prefill_chunk_size", None)
        # Allocation events of this call for the decode-trace guard (chunked = a chunk walk or a continuation
        # of cached tokens: the sliding-tail stash lives across chunks).
        start_pos_after = kwargs.get("start_pos")
        chunked = any(int(n) > int(chunk or 1 << 30) for n in (prefill_seq_lens or [])) or (
            start_pos_after is not None and any(int(p) > 0 for p in list(start_pos_after))
        )
        guard_reasons = self._g4_arm_after_prefill(
            prefill_trace_ids_before=prefill_trace_ids_before, chunked=chunked, multimodal=mm_users > 0
        )
        logger.info(
            "[gemma4-vllm-perf] prefill TTFT={:.1f} ms | prefill_tok/s={:.2f} | "
            "seq_len={} | max_prefill_chunk_size={} | bounded_sliding={} | batch={} | multimodal_users={} | "
            "mm_stage_ms={:.1f} | phases: {} | guard_arm={}",
            ttft_ms,
            prefill_tok_s,
            seq_len,
            chunk,
            self._bounded_sliding_kv_cache,
            batch_size,
            mm_users,
            mm_stage_ms,
            self._g4_prefill_phases(t_entry, t0, clear_ms),
            ",".join(guard_reasons)
            if guard_reasons
            else ("prefill" if self._g4_guard_arm_mode() == "prefill" else "-"),
        )
        # Reset decode accumulators at the start of each new generate.
        self._perf_decode_tokens = 0
        self._perf_decode_s = 0.0
        return out

    def _g4_lane_phases(self, t_entry, t0):
        """Host phases of a lane-router call for the perf line (ms): the prelude before the router, the per-layer
        page-table installs (ms x count), the trace input staging, the replay (its device wall only under
        GEMMA4_LANE_TRACE_SYNC_PROFILE=1, else the enqueue), the eager gathered lm_head (frames that return hidden
        states) and the logits readback (which waits for the device)."""
        acc = {}
        for m in self.model:
            for k, v in (getattr(m, "_g4_pf_ms", None) or {}).items():
                acc[k] = acc.get(k, 0.0) + float(v)
        return (
            f"prelude={(t0 - t_entry) * 1000.0:.1f} install={acc.get('install', 0.0):.1f}x{int(round(acc.get('install_n', 0.0)))} "
            f"stage={acc.get('stage', 0.0):.1f} replay={acc.get('replay', 0.0):.1f} gather={acc.get('gather', 0.0):.1f} "
            f"readback={acc.get('readback', 0.0):.1f} ms"
        )

    def _g4_prefill_phases(self, t_entry, t0, clear_ms):
        """Host phases of one ``prefill_forward`` call for the perf line (ms): the prelude before the TT forward
        (page-table build, mask, pad, install), the per-layer page-table installs (ms and count over the call), the
        traced calls (input staging + replay + its sync), the deferred lm_head, the logits readback and the
        sequential-table clear. Plain host clocks around the existing calls: no device sync is added."""
        acc = {}
        for m in self.model:
            for k, v in (getattr(m, "_g4_pf_ms", None) or {}).items():
                acc[k] = acc.get(k, 0.0) + float(v)
        return (
            f"prelude={(t0 - t_entry) * 1000.0:.1f} install={acc.get('install', 0.0):.1f}x{int(round(acc.get('install_n', 0.0)))} "
            f"trace={acc.get('trace', 0.0):.1f} lm_head={acc.get('lm_head', 0.0):.1f} readback={acc.get('readback', 0.0):.1f} "
            f"clear={clear_ms:.1f} ms"
        )

    @staticmethod
    def _slice_page_tables_rows(page_tables_per_layer, rows):
        """First ``rows`` rows of each per-layer host table; tables that are the
        same object stay the same object so the H2D fan-out still copies once."""
        if rows is None or rows <= 0 or not page_tables_per_layer:
            return page_tables_per_layer
        sliced_by_id = {}
        out = []
        for pt in page_tables_per_layer:
            if not isinstance(pt, torch.Tensor) or pt.dim() < 2 or int(pt.shape[0]) <= rows:
                out.append(pt)
                continue
            sliced = sliced_by_id.get(id(pt))
            if sliced is None:
                sliced = pt[:rows].contiguous()
                sliced_by_id[id(pt)] = sliced
            out.append(sliced)
        return out

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        # Free tensors retired by the last batched-prefill consumption BEFORE
        # this step's trace replay (the retired list is host-consumed already;
        # leaving the final entry alive across the replay violates the
        # trace-safety contract in process_logits_after_prefill_trace).
        for m in self.model:
            scavenge = getattr(m, "_g4_retire_scavenge", None)
            if scavenge is not None:
                scavenge()
        page_tables_per_layer = self._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        page_tables_per_layer = self._pad_sliding_page_tables_for_bounded(
            page_tables_per_layer, kwargs.get("kv_cache"), authoritative=True
        )
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        if per_submesh is not None:
            # The bucket-B decode trace replays the per-layer buffers keyed by B rows;
            # the plugin pads its tables to max_num_seqs, so slice them to the bucket.
            _tok = kwargs.get("tokens")
            _tok = args[0] if _tok is None and args else _tok
            host_b = int(_tok.shape[0]) // max(1, int(self.data_parallel)) if _tok is not None else None
            per_submesh = [self._slice_page_tables_rows(pt_list, host_b) for pt_list in per_submesh]
            # A prefill step in between overwrites the same bucket buffers, so
            # re-upload after one even without the plugin's reload command.
            if self._reload_per_layer_page_tables(kwargs) or self._page_tables_written_by_prefill(
                self.model, per_submesh
            ):
                for m, pt_for_submesh in zip(self.model, per_submesh):
                    self._install_per_layer_page_tables(m, pt_for_submesh, writer="decode")
        self._g4_note_page_table_realloc()
        t0 = time.perf_counter()
        with self._route_per_layer_page_tables(per_submesh):
            # Route through ``ChunkedPrefillPageTableGuardMixin.decode_forward``
            # (Gemma4's batch-keyed trace path). Do not call
            # ``super(HybridAttentionForCausalLM, ...)`` because that skips the
            # mixin and loses the decode-bucket trace key. Also avoid plain
            # ``HybridAttentionForCausalLM.decode_forward`` (NotImplementedError).
            out = super().decode_forward(*args, **kwargs)
        self._perf_decode_tokens += 1
        do_log = self._perf_decode_tokens % self._perf_log_every == 0
        # Sync only when measuring (or GEMMA4_VLLM_DECODE_SYNC_EVERY=1); token
        # readback already synchronizes for correctness.
        sync_mode = self._perf_decode_sync_every
        should_sync = sync_mode == "1" or sync_mode == "always" or (sync_mode == "log" and do_log)
        if should_sync:
            try:
                mesh = getattr(self.model_args[0], "mesh_device", None)
                if mesh is not None:
                    ttnn.synchronize_device(mesh)
            except Exception:
                pass
        dt = time.perf_counter() - t0
        self._perf_decode_s += dt
        if do_log:
            tok_s_u = self._perf_decode_tokens / self._perf_decode_s if self._perf_decode_s > 0 else 0.0
            ms_tok = (self._perf_decode_s / self._perf_decode_tokens) * 1000.0 if self._perf_decode_tokens else 0.0
            # Without the device sync the timer only covers the host's submit of
            # the step (async decode returns before the device finishes), which
            # reads like a 10x throughput; it is not a decode rate. Keep that
            # variant at DEBUG and say so; the synced measurement stays at INFO.
            if should_sync:
                logger.info(
                    "[gemma4-vllm-perf] decode tok/s/user={:.2f} | ms/token={:.2f} | "
                    "tokens={} | bounded_sliding={} | decode_batch={}",
                    tok_s_u,
                    ms_tok,
                    self._perf_decode_tokens,
                    self._bounded_sliding_kv_cache,
                    getattr(self, "_prev_decode_batch", None),
                )
            else:
                logger.debug(
                    "[gemma4-vllm-perf] decode submit (host time, not a decode rate; "
                    "GEMMA4_VLLM_DECODE_SYNC_EVERY=log measures) ms/submit={:.2f} | "
                    "tokens={} | bounded_sliding={} | decode_batch={}",
                    ms_tok,
                    self._perf_decode_tokens,
                    self._bounded_sliding_kv_cache,
                    getattr(self, "_prev_decode_batch", None),
                )
        return out

    def _g4_note_page_table_realloc(self):
        """Drop the decode traces after a per-layer page-table buffer realloc, graded by when it happened.

        The model raises ``_invalidate_decode_traces_after_page_table_realloc``
        when the persistent buffers grew or a new batch key allocated
        (``Gemma4Model._page_tables_to_ttnn``): captured replays would read the
        old addresses, so the traces are released and the next step
        re-captures. Inside ``warmup_model_decode`` that is the wider bucket
        growing what the first bucket allocated (INFO). After warmup it is a
        serving-time event: with traces captured it is an ERROR, because
        under async decode a grow while a step is in flight rebinds buffers
        the pending read still references (#51186); with none captured it is
        a WARNING. The traces are released rather than forgotten: forgetting
        only their ids leaked the trace-region buffers (a 1.2 GB region was
        spent after 45 re-captures, TT_FATAL at the next capture) and dropped
        the sampled-token feedback binding the helper keeps.
        """
        if not any(getattr(m, "_invalidate_decode_traces_after_page_table_realloc", False) for m in self.model):
            return
        in_warmup = bool(self._g4_decode_warmup_active)
        after_warmup = bool(self._decode_warmup_complete) and not in_warmup
        for m in self.model:
            m._invalidate_decode_traces_after_page_table_realloc = False
        what = ", ".join(sorted({str(getattr(m, "_g4_page_table_realloc_reason", "grow")) for m in self.model}))
        reason = f"per-layer page-table buffer realloc ({what}; addresses changed)"
        released = self._g4_release_decode_traces(reason, in_warmup=in_warmup)
        if in_warmup:
            logger.info(
                "Gemma4 vLLM: decode warmup grew the per-layer page tables ({}); {} trace(s) dropped", what, released
            )
        elif after_warmup and released:
            logger.error(
                "Gemma4 vLLM: cleared {} decode trace(s) after {} [AFTER WARMUP — unsafe under async_scheduling]",
                released,
                reason,
            )
        else:
            logger.warning("Gemma4 vLLM: {}; {} decode trace(s) released", reason, released)

    @property
    def kv_head_shard_devices(self):
        """Devices KV heads shard over (the TP axis), for the plugin's per-device kv-cache spec.

        On a 2D one-instance mesh this is the tp-axis size, not the mesh device count.
        """
        models = getattr(self, "model", None) or []
        mc = getattr(models[0], "mesh_config", None) if models else None
        return getattr(mc, "tp", None) if mc is not None else None

    def allocate_kv_cache(self, *args, **kwargs):
        # Legacy uniform path (vLLM falls back here when ``get_kv_cache_spec``
        # isn't consulted). The hybrid path uses ``allocate_kv_cache_per_layer``
        # inherited from :class:`HybridAttentionForCausalLM`.
        #
        # ``tt_cache_path=None``: never disk-cache the zero-filled KV tensors.
        # All DP ranks share one tt_metal_cache dir, so at DP>1 a rank can load
        # an empty-KV cache file another rank is still writing — a torn read
        # that segfaults in memcpy_to_device (observed on BH Galaxy DP=4 at
        # GEMMA4_MAX_TOKENS_ALL_USERS=524288, where each global-layer file is
        # 539 MB and the write window is seconds). Tilizing zeros in-process
        # costs ~1 s/global layer; the cache saved less than it risked.
        kv_cache = allocate_vllm_kv_cache(
            *args,
            **kwargs,
            dp_model=self.model,
            tt_cache_path=None,
        )
        self._attach_sdpa_splitfree_scratch(kv_cache)
        return kv_cache

    def _attach_sdpa_splitfree_scratch(self, kv_cache):
        """GEMMA4_SDPA_DECODE_SPLITFREE: the vLLM build creates no paged caches in the model (create_kv_cache=False,
        so the model's own attach never runs); the split-independent decode SDPA's persistent scratch is allocated
        and attached here, once the plugin's caches exist (their dtype sizes the plan) and before the decode warmup
        captures the traces (``Gemma4Model.attach_sdpa_splitfree_scratch``; off: the layers keep the stock op)."""
        for idx, model in enumerate(self.model):
            attach = getattr(model, "attach_sdpa_splitfree_scratch", None)
            if attach is None:
                continue
            kv_dtype = None
            sub = kv_cache[idx] if kv_cache is not None and idx < len(kv_cache) else None
            for layer_kv in sub or []:
                if layer_kv and layer_kv[0] is not None:
                    kv_dtype = layer_kv[0].dtype
                    break
            attach(kv_dtype=kv_dtype)

    def _text_config(self):
        model = self.model[0]
        return getattr(model.hf_config, "text_config", model.hf_config)

    def _sliding_layer_indices(self) -> list[int]:
        layer_types = list(getattr(self._text_config(), "layer_types", None) or [])
        return [i for i, lt in enumerate(layer_types) if lt == "sliding_attention"]

    def _bounded_sliding_physical_blocks(self, block_size: int) -> int | None:
        """Sliding pool size: ``(ring/block_size) * B``.

        Used when ``bounded_sliding`` is on so hybrid-OFF UniformType specs
        (full ``num_blocks`` for every layer) do not allocate ~256k-length
        sliding buffers and OOM at long ISL.

        Sized from the RING, not the bare ``sliding_window``. The ring is what
        positions wrap into (``bounded_ring_modulo``), and on the speculative
        path it is larger than the window so candidate writes at p+1..p+K land
        outside the live window instead of evicting history a candidate query
        still needs. Allocating from the window while wrapping modulo the ring
        writes past the pool, which is why setting the headroom env alone was
        never enough -- allocation, page tables and modulo have to move
        together.
        """
        if not self._bounded_sliding_kv_cache or block_size <= 0:
            return None
        sliding_window = getattr(self._text_config(), "sliding_window", None)
        if sliding_window is None:
            return None
        from models.demos.gemma4.tt.attention import bounded_ring_modulo

        ring = bounded_ring_modulo(int(sliding_window))
        if ring is None or int(ring) % block_size != 0:
            return None
        # Under the lane fold KV is per column: size the pool per lane, plus one
        # scratch block for non-owner columns (block 0 is slot 0's live ring).
        blocks = (int(ring) // block_size) * self._sliding_pool_batch()
        if self._lanes_enabled():
            blocks += 1
        return blocks

    def _release_decode_traces_for_fresh_wave(self) -> None:
        """Release captured decode traces (see fresh-wave comment at call site)."""
        released = 0
        try:
            for _key, tids in list(self.trace_ids_decode.items()):
                if not tids:
                    continue
                for mid, tid in tids.items():
                    ttnn.release_trace(self.model_args[mid].mesh_device, tid)
                    released += 1
        except Exception as exc:
            logger.warning("Gemma4 bounded: decode trace release failed: {}", exc)
        if released:
            self.trace_ids_decode = defaultdict(lambda: None)
            self.trace_inputs_decode = defaultdict(lambda: None)
            self.trace_output_decode = defaultdict(lambda: None)
            self._prev_decode_batch = None
            self.prev_page_table = None
            logger.info("Gemma4 bounded: released {} decode trace(s) at fresh wave", released)

    def _clear_bounded_sliding_kv_rings(self, kv_cache, live_before: int = 0) -> None:
        """Zero sliding-layer paged KV buffers before a fresh prefill.

        Bounded mode remaps every user onto a fixed physical ring; those
        buffers are not freshly allocated per request. Stale contents from
        warmup or a prior generate corrupt short-prompt next-token logits
        (notably closing the gemma4 thought channel immediately).

        This zeroes the *whole* sliding pool, i.e. every user's ring. Under
        concurrent serving each arriving request is itself a "fresh prefill"
        (``start_pos == 0``), so doing that unconditionally wipes the KV of
        every request currently decoding — measured as garbage output from
        concurrency 2 upward. ``live_before`` is the ring occupancy sampled
        *before* this prefill's own remap, so the serving path skips the clear
        only when some *other* request still holds a ring; the arriving request
        is handed a ring no live request owns, and SDPA reads only
        ``[cur_pos-W+1, cur_pos]``, which its own prefill has just written, so
        any stale tail left by a finished request is never attended to.
        """
        if not self._bounded_sliding_kv_cache or kv_cache is None:
            return
        if live_before:
            logger.debug(
                "Gemma4 bounded: skipping sliding-ring clear, {} live ring(s) in flight",
                live_before,
            )
            return
        sliding_idxs = set(self._sliding_layer_indices())
        if not sliding_idxs:
            return
        # kv_cache: [submesh][layer] -> [k, v] (or layer -> [k, v] when undped).
        submeshes = kv_cache if isinstance(kv_cache, (list, tuple)) else [kv_cache]
        cleared = 0
        for sub in submeshes:
            if sub is None:
                continue
            layers = sub if isinstance(sub, (list, tuple)) else [sub]
            for li, layer_kv in enumerate(layers):
                if li not in sliding_idxs or layer_kv is None:
                    continue
                pair = layer_kv if isinstance(layer_kv, (list, tuple)) else (layer_kv,)
                for cache_t in pair:
                    if cache_t is None:
                        continue
                    try:
                        z = ttnn.zeros_like(cache_t)
                        ttnn.copy(z, cache_t)
                        z.deallocate(True)
                        cleared += 1
                    except Exception as e:
                        logger.warning(
                            "Gemma4 vLLM: failed to clear bounded sliding KV "
                            "(layer={}, err={}) — stale ring may remain",
                            li,
                            e,
                        )
        if cleared:
            logger.info(
                "Gemma4 vLLM: cleared {} bounded sliding KV buffers before prefill",
                cleared,
            )

    def _shrink_bounded_sliding_kv_specs(self, per_layer_specs):
        """Rewrite sliding-layer ``num_blocks`` to the bounded physical pool."""
        sliding_idxs = set(self._sliding_layer_indices())
        if not sliding_idxs:
            return per_layer_specs
        # Specs are (shape, dtype, tensor_idx); shape[0]=num_blocks, shape[2]=block_size.
        sample_bs = None
        for i, (shape, _, _) in enumerate(per_layer_specs):
            if i in sliding_idxs:
                sample_bs = int(shape[2])
                break
        if sample_bs is None:
            return per_layer_specs
        new_blocks = self._bounded_sliding_physical_blocks(sample_bs)
        if new_blocks is None:
            return per_layer_specs
        if self._lanes_enabled():
            for m in self.model:
                m._lane_scratch_block = new_blocks - 1
        out = []
        shrunk = 0
        for i, (shape, dtype, tensor_idx) in enumerate(per_layer_specs):
            if i in sliding_idxs and int(shape[0]) > new_blocks:
                shape = (new_blocks, *shape[1:])
                shrunk += 1
            out.append((shape, dtype, tensor_idx))
        if shrunk:
            logger.info(
                "Gemma4 vLLM: bounded sliding — sized {}/{} sliding KV buffers "
                "to {} blocks (ring/block_size * max_batch={}), "
                "matching metal demo (avoids full-ISL sliding DRAM OOM).",
                shrunk,
                len(sliding_idxs),
                new_blocks,
                self._sliding_pool_batch(),
            )
        return out

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate per-layer KV cache, then alias KV-shared layers to
        their source layer's buffer.

        Gemma4-E2B / -E4B have a Gemma3n-style "num_kv_shared_layers"
        optimization where the last N layers reuse an earlier layer's
        K/V instead of computing+storing their own. The model side
        encodes this via ``self.kv_shared_layer_map`` (layer_idx →
        source_idx) and ``attention/{prefill,decode}.py`` skips
        ``paged_{fill,update}_cache`` whenever a layer is flagged as
        shared. vLLM's hybrid kv-cache manager is unaware of this
        TT-specific reuse and allocates a distinct buffer for every
        layer; without the post-allocator alias the shared layers'
        SDPA reads land on zero-initialized buffers.

        Important: aliasing the buffer is *necessary but not sufficient*.
        Source and shared layers share an attention *type* (sliding or
        full), but vLLM's hybrid manager constructs more groups than
        just "one per type" — for Gemma4-E2B with 35 layers in the
        4-sliding-then-1-full pattern, vLLM produces 5 groups of 7
        layers each (4 sliding sub-groups + 1 full group), and each
        physical tensor is shared by one layer from each group. That
        means layer 13 (sliding, in group[3]) and layer 15 (sliding,
        in group[0]) have *different* per-layer page_tables, so
        aliasing only the buffer leaves layer 15 reading the wrong
        slot of the shared tensor — whatever group[0]'s layer 10 wrote
        there, not what layer 13 wrote. The buffer alias must be
        paired with a per-layer-page-table alias in
        :meth:`_block_tables_per_layer_with_kv_share` so the shared
        layer indexes the buffer the same way the source did.

        When ``bounded_sliding`` is on (hybrid groups still OFF), sliding
        layer buffers are also shrunk to the demo window pool before
        allocation — see :meth:`_shrink_bounded_sliding_kv_specs`.
        """
        per_layer_specs = self._shrink_bounded_sliding_kv_specs(per_layer_specs)
        # Direct util call instead of super(): pass ``tt_cache_path=None`` so
        # the zero-filled KV tensors are never disk-cached — DP ranks share one
        # cache dir and racing create/load of the same empty-KV file is a
        # torn-read segfault (see :meth:`allocate_kv_cache`).
        kv_cache = allocate_vllm_kv_cache_per_layer(per_layer_specs, dp_model=self.model, tt_cache_path=None)
        for submesh_idx, submesh_kv in enumerate(kv_cache):
            kv_shared_map = getattr(self.model[submesh_idx], "kv_shared_layer_map", None)
            if not kv_shared_map:
                continue
            for layer_idx, source_idx in kv_shared_map.items():
                submesh_kv[layer_idx] = submesh_kv[source_idx]
        self._attach_sdpa_splitfree_scratch(kv_cache)
        return kv_cache

    def _ensure_page_tables_per_layer(self, page_tables_per_layer, page_table):
        """Broadcast legacy ``page_table`` to per-layer whenever the plugin
        only sent the legacy view.

        Parent only broadcasts for hybrid-ON. Gemma4 always routes fills
        through ``_chunk_prefill_page_table`` / per-layer aliases (incl.
        unbounded hybrid-OFF + vLLM APC), so a missing stash falls back to a
        truncated legacy table and continuation chunks write through -1
        (#51186). Broadcasting the (full-width) legacy map is cheap and keeps
        APC absolute fills correct.
        """
        if page_tables_per_layer is not None or page_table is None:
            return page_tables_per_layer
        num_layers = len(self.model[0].layers)
        return [page_table] * num_layers

    def _build_per_layer_page_tables(self, page_tables_per_layer, legacy_page_table):
        """Compose the inherited per-layer broadcast/passthrough with
        the Gemma4-specific kv-share alias.

        Composition logic, kept in one place so
        :meth:`prefill_forward` / :meth:`decode_forward` each take one
        call instead of remembering to chain two helpers:

        1. :meth:`_ensure_page_tables_per_layer`
           — broadcast a legacy single ``page_table`` to per-layer when
           the plugin only sent the legacy view (warmup, tests).
        2. :meth:`_apply_kv_share_to_per_layer_page_tables` — for every
           ``(shared_idx, source_idx)`` in the model's
           ``kv_shared_layer_map``, re-point the shared layer's
           page_table at the source's. See the
           :meth:`allocate_kv_cache_per_layer` docstring for why
           aliasing the buffer alone leaves the shared layer reading
           a different layer's slot of the shared HMA tensor.
        """
        page_tables_per_layer = self._ensure_page_tables_per_layer(page_tables_per_layer, legacy_page_table)
        return self._apply_kv_share_to_per_layer_page_tables(page_tables_per_layer)

    def _apply_kv_share_to_per_layer_page_tables(self, page_tables_per_layer):
        """Replace every kv-shared layer's per-layer page_table with
        its source layer's per-layer page_table.

        The buffer alias in :meth:`allocate_kv_cache_per_layer` makes
        ``caches[shared] is caches[source]``; this method makes
        ``page_tables[shared] is page_tables[source]``. Together they
        ensure the shared layer reads exactly the (buffer, block IDs)
        the source layer wrote — without this, the shared layer reads
        the slot in the HMA-shared buffer that the layer in its own
        kv-cache sub-group wrote, which is some other layer's K/V.
        See [[gemma4-kv-share-page-table-alias]] for the diagnosis
        path.
        """
        if not page_tables_per_layer:
            return page_tables_per_layer
        kv_shared_map = getattr(self.model[0], "kv_shared_layer_map", None) or {}
        if not kv_shared_map:
            return page_tables_per_layer
        out = list(page_tables_per_layer)
        for layer_idx, source_idx in kv_shared_map.items():
            if 0 <= layer_idx < len(out) and 0 <= source_idx < len(out):
                out[layer_idx] = out[source_idx]
        return out

    def _pad_page_tables_batch_to_max(self, page_tables_per_layer):
        """Pad host page-table batch dim up to ``max_batch_size``.

        Decode warmup captures Metal traces against persistent buffers sized
        at max batch. Prefill often passes B=1 or B=31; padding here makes
        the first allocation (and every subsequent copy) match that width so
        we never grow/orphan trace addresses. Unused rows are filled with 0
        (vLLM null block).
        """
        if not page_tables_per_layer:
            return page_tables_per_layer
        max_b = int(self.model_args[0].max_batch_size)
        out = []
        padded_by_id = {}  # layers of one kv-cache group share the host table: pad it once, keep it shared
        for pt in page_tables_per_layer:
            if pt is None or not isinstance(pt, torch.Tensor):
                out.append(pt)
                continue
            pt2 = pt if pt.dim() > 1 else pt.unsqueeze(0)
            if int(pt2.shape[0]) >= max_b:
                out.append(pt2)
                continue
            padded = padded_by_id.get(id(pt))
            if padded is None:
                padded = torch.zeros((max_b, int(pt2.shape[1])), dtype=torch.int32)
                padded[: pt2.shape[0], :] = pt2.to(dtype=torch.int32)
                padded_by_id[id(pt)] = padded
            out.append(padded)
        return out

    def _sliding_pool_batch(self):
        """Rows the sliding pool is sized for: per lane under the lane fold, else max batch."""
        return int(getattr(self.model[0], "lane_slots", 0) or self.model_args[0].max_batch_size)

    def _lanes_enabled(self):
        return bool(getattr(getattr(self.model[0], "mesh_config", None), "lane_sharded", False))

    def _pad_sliding_page_tables_for_bounded(
        self, page_tables_per_layer, kv_cache, authoritative=False, row_slots=None
    ):
        """Remap sliding-layer page tables onto the bounded physical pool.

        With hybrid groups OFF, vLLM hands every layer the same full-ISL
        block table (global IDs into ``num_blocks≈max_model_len/block_size``).
        Bounded mode allocates only ``ring/block_size * B`` physical blocks per
        sliding layer (see :meth:`_shrink_bounded_sliding_kv_specs`), so those
        global IDs would OOB. Rebuild each sliding row with dense local IDs —
        same layout as ``build_hybrid_page_tables`` in the metal demo: user
        ``u`` owns ``[u*W, (u+1)*W)`` where ``W=ring/block_size`` and the ring
        is ``bounded_ring_modulo(sliding_window)``: the window, plus the
        speculative headroom when ``GEMMA4_SPEC_RING_HEADROOM_BLOCKS`` is set.

        Tables are sized to exactly ``W`` columns so ``cache_position_modulo``
        shape checks pass on short prompts without retaining vLLM's full-ISL
        width (unused under modulo wrap). The pool, the modulo and these
        columns must all follow the same ring: ``paged_fill_cache`` requires
        ``modulo <= block_size * columns``.

        Full-attention layers are left alone.
        """
        if not self._bounded_sliding_kv_cache:
            return page_tables_per_layer
        if not page_tables_per_layer:
            return page_tables_per_layer
        model = self.model[0]
        text_config = self._text_config()
        sliding_window = getattr(text_config, "sliding_window", None)
        layer_types = getattr(text_config, "layer_types", None)
        if sliding_window is None or layer_types is None:
            return page_tables_per_layer

        # Prefer a *sliding* layer's K-cache block_size (after shrink). Full
        # layers may share the same declared block_size under UniformType.
        block_size = None
        sliding_idxs = self._sliding_layer_indices()
        if kv_cache is not None and sliding_idxs:
            try:
                block_size = int(kv_cache[0][sliding_idxs[0]][0].shape[2])
            except (TypeError, IndexError, AttributeError):
                block_size = None
        if block_size is None and kv_cache is not None:
            try:
                block_size = int(kv_cache[0][0][0].shape[2])
            except (TypeError, IndexError, AttributeError):
                block_size = None
        if block_size is None:
            try:
                block_size = int(model.layers[0].self_attn.kv_cache[0].shape[2])
            except (TypeError, IndexError, AttributeError):
                return page_tables_per_layer

        ring = self._bounded_ring_size()
        if ring is None or ring % block_size != 0:
            return page_tables_per_layer
        target_cols = int(ring) // block_size

        # Ring slots must follow the *request*, not its row in the current
        # page-table tensor. The plugin compacts decode rows onto the occupied
        # requests (``block_tables_for_rows(req_indices, ...)``) and picks the
        # decode bucket from the request *count*, so a live request's row index
        # shifts whenever the running set changes while its KV stays put.
        # Keying dense block IDs on the row index therefore hands a running
        # request a different physical ring between steps and it reads another
        # user's KV (nondeterministic garbage from concurrency 2 upward). Key on
        # vLLM's own global block ID instead: stable for the request's lifetime,
        # and unique while prefix caching is off (__init__ refuses it with bounded rings).
        max_slots = int(getattr(self.model_args[0], "max_batch_size", 0) or 0)

        # Derive the identity from a full-attention row when one is available.
        # vLLM substitutes its reserved null block (ID 0) for blocks it has
        # evicted, which for a *sliding* group can zero the very entry used as
        # the key; a full-attention row keeps every block for the request's
        # lifetime. Under hybrid-groups-off both rows are the same table, so
        # this only matters if hybrid groups are turned on later.
        ref = None
        for i, pt in enumerate(page_tables_per_layer):
            if not isinstance(pt, torch.Tensor) or i >= len(layer_types):
                continue
            if layer_types[i] != "sliding_attention":
                ref = pt
                break
            if ref is None:
                ref = pt
        lane_slots_n = 0
        if bool(getattr(getattr(model, "mesh_config", None), "lane_sharded", False)):
            lane_slots_n = int(getattr(model, "lane_slots", 0) or 0)
        slots_by_row = (
            self._bounded_ring_slots(
                ref, max_slots or int(ref.shape[0]), authoritative, row_slots=row_slots, lane_slots=lane_slots_n
            )
            if ref is not None
            else None
        )

        out = []
        for i, pt in enumerate(page_tables_per_layer):
            if (
                pt is None
                or i >= len(layer_types)
                or layer_types[i] != "sliding_attention"
                or not hasattr(pt, "shape")
                or not isinstance(pt, torch.Tensor)
            ):
                out.append(pt)
                continue
            batch = int(pt.shape[0])
            if slots_by_row is not None and len(slots_by_row) == batch:
                slots = slots_by_row
            else:
                # Per-layer fallback: keys derive from THIS layer's table. Under
                # hybrid-groups-off every layer shares one table so this equals
                # slots_by_row; with hybrid groups on, a sliding row whose block-0
                # was nulled by vLLM could key a different slot than the
                # full-attention reference — if the fallback ever fires alongside
                # a computed reference, surface it instead of diverging silently.
                if slots_by_row is not None and not getattr(self, "_g4_ring_fallback_warned", False):
                    self._g4_ring_fallback_warned = True
                    logger.warning(
                        f"gemma4 bounded ring: layer {i} page table has {batch} rows but the shared "
                        f"slot reference has {len(slots_by_row)} — falling back to per-layer slot "
                        "derivation. Rings can diverge across layers if row identities differ."
                    )
                slots = self._bounded_ring_slots(
                    pt, max_slots or batch, False, row_slots=row_slots, lane_slots=lane_slots_n
                )
            # Always W columns (demo layout). Keeping vLLM's full-ISL width
            # here thrash-reallocates persistent buffers vs short prefill
            # tables and is unused under cache_position_modulo.
            remapped = torch.zeros((batch, target_cols), dtype=torch.int32)
            for u, slot in enumerate(slots):
                if slot is None:
                    continue  # padded gap row; its position is -1 so it is never read
                remapped[u] = torch.arange(slot * target_cols, (slot + 1) * target_cols, dtype=torch.int32)
            out.append(remapped)
        return out

    @staticmethod
    def _bounded_row_key(row):
        """Stable per-request identity for one page-table row.

        vLLM's global block IDs live as long as the request (bounded rings
        refuse prefix caching, so blocks are not shared between requests), which
        makes the first block ID an identity that survives the row index
        moving. An all-zero row is a padded decode gap, not a request.
        """
        if int(row.max()) == 0:
            return None
        return int(row[0])

    def _bounded_ring_slots(self, pt, max_slots, authoritative, row_slots=None, lane_slots=None):
        """Assign each page-table row a persistent bounded-ring slot.

        ``authoritative`` must be set only by the decode path: a decode step
        sees the whole running set, so slots for departed requests can be
        reclaimed there. A prefill call carries just the arriving request, so
        releasing on it would drop every live request's slot and re-hand them
        different rings — the exact corruption this mapping exists to prevent.
        """
        slot_map = getattr(self, "_bounded_ring_slot_map", None)
        if slot_map is None:
            slot_map = {}
            self._bounded_ring_slot_map = slot_map
        keys = [self._bounded_row_key(pt[u]) for u in range(int(pt.shape[0]))]
        # Lane fold: the ring is slot % lane_slots, since dense insertion-order ids would
        # overrun the per-column pool; prefill rebinds its keys, decode/spec look them up.
        if lane_slots and row_slots is not None:
            slots = []
            for u, k in enumerate(keys):
                if k is None or u >= len(row_slots):
                    slots.append(None)
                    continue
                ring = int(row_slots[u]) % int(lane_slots)
                slot_map[k] = ring
                slots.append(ring)
            return slots
        if lane_slots:
            missing = [k for k in keys if k is not None and k not in slot_map]
            if missing:
                # Dense fallback ids would address past the per-column pool.
                raise ValueError(
                    f"Gemma4 bounded: {len(missing)} ring key(s) were never bound by a prefill on the lanes rail"
                )
            return [None if k is None else slot_map[k] for k in keys]
        # Do NOT release a slot merely because its key is missing from this
        # batch. vLLM does not necessarily schedule every running request in
        # every decode step, so an absent key is not proof the request ended;
        # freeing it there hands a live request's ring to a new arrival. Keep
        # entries until the map is actually full and recycle least-recently-used
        # instead — a request that really finished stops being touched and ages
        # out naturally. ``authoritative`` now only marks a call whose key set
        # is a true running set, which is what refreshes recency.
        fresh = getattr(self, "_g4_ring_keys_since_authoritative", None)
        if fresh is None:
            fresh = set()
            self._g4_ring_keys_since_authoritative = fresh
        if authoritative:
            for k in keys:
                if k is not None and k in slot_map:
                    slot_map[k] = slot_map.pop(k)  # move to end = most recent
            # Remember the last true running set: eviction must prefer keys
            # absent from it — LRU order alone can front a LIVE request that
            # simply was not scheduled in recent decode steps.
            self._g4_last_authoritative_keys = {k for k in keys if k is not None}
            fresh.clear()
        used = set(slot_map.values())
        slots = []
        for k in keys:
            if k is None:
                slots.append(None)
                continue
            slot = slot_map.get(k)
            if slot is None:
                slot = next((s for s in range(max_slots) if s not in used), None)
                if slot is None:
                    # Slots are reclaimed on decode steps, so a batch that
                    # drains completely leaves its keys behind until the next
                    # decode runs. A prefill arriving in that window can find
                    # the map full; the scheduler caps concurrency at
                    # ``max_slots``, so the oldest entry is necessarily a
                    # departed request. Evict it rather than colliding on 0.
                    # Keys bound by a prefill since the last authoritative step are
                    # live requests that no decode step has listed yet; they are
                    # absent from that set exactly like departed keys, so they must
                    # never be the victim. Otherwise a long request that finished
                    # (still in the last running set, hence protected) followed by
                    # a burst that fills every slot evicted the burst's FIRST request
                    # for its 32nd, two live users shared one ring and the first one
                    # decoded garbage (12B bounded, 2026-10-18).
                    last_auth = getattr(self, "_g4_last_authoritative_keys", None)
                    candidates = [vk for vk in slot_map if vk not in fresh] or list(slot_map)
                    victim = next(
                        (vk for vk in candidates if last_auth is not None and vk not in last_auth),
                        candidates[0],
                    )
                    slot = slot_map.pop(victim)
                    logger.warning(
                        "Gemma4 bounded: ring slot map full (max_slots={}); recycling "
                        "least-recently-used key {} to admit a new request",
                        max_slots,
                        victim,
                    )
                slot_map[k] = slot
                used.add(slot)
                if not authoritative:
                    fresh.add(k)
            slots.append(slot)
        return slots


def dflash_pv_bucket_ladder(max_context, horizon=None, verify=None, max_rungs=None):
    """The ENUMERABLE set of packed-verify buckets a dFlash server can serve.

    ``DFlashFusedDecoder.pv_bucket`` keys the fused trace on
    ``round_up_1024(start + horizon + P_v + 64)``, i.e. on the PROMPT LENGTH, so
    a 256K server has ~258 possible buckets -- far too many to pre-capture, which
    is why capture is per session today. Per-session capture is the part that
    does not conform: section 8 bans a capture that happens during serving, or
    one whose shape can only be known once a request exists. A FIXED SET of
    verify widths chosen at config time is conformant -- every width is known
    before the first request and captured in warmup -- so this ladder is what
    makes the fused verify admissible, not merely a capture-count optimisation.

    The set becomes enumerable because a bucket captured LARGER than a request
    needs is numerically EXACT for it: ``_pv_setup`` caps S_k at capture and
    masks every column past the live top to NEG ("columns past the live top are
    NEG -> exact; the captured program never changes shape"). So a coarse ladder
    plus round-UP serves every prompt length, and the only cost of rounding up
    is wasted verify width.

    The ladder doubles from the smallest useful rung to the largest a request
    can reach, which bounds the waste at <2x the exact bucket while keeping the
    rung count logarithmic in the context (8 rungs at 256K, worst waste 1.50x).

    Pure: config-time arithmetic, no device and no model instance. Returns
    ascending 1024-aligned bucket sizes.
    """
    horizon = int(os.environ.get("GEMMA4_DFLASH_SERVE_HORIZON", "2048") if horizon is None else horizon)
    verify = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5") if verify is None else verify)
    p_v = verify + 1
    tail = horizon + p_v + VERIFY_WIDTH_MARGIN  # what pv_bucket adds on top of ``start``

    def _round(n):
        return ((int(n) + 1023) // 1024) * 1024

    smallest = _round(tail)  # a zero-length prompt still needs the tail
    largest = _round(int(max_context) + tail)
    ladder, rung = [], smallest
    while rung < largest:
        ladder.append(rung)
        rung = _round(rung * 2)
    ladder.append(largest)
    if max_rungs is not None and len(ladder) > int(max_rungs):
        # Keep the largest rungs: dropping a SMALL rung only costs wasted verify
        # width on short prompts, while dropping the largest would leave long
        # prompts with no bucket that fits.
        ladder = ladder[-int(max_rungs) :]
    return ladder


def dflash_bucket_for(start, ladder):
    """Smallest ladder rung that covers ``start``'s exact bucket, or None.

    None means the request is longer than the ladder covers and must fall back
    to a per-session capture.
    """
    horizon = int(os.environ.get("GEMMA4_DFLASH_SERVE_HORIZON", "2048"))
    verify = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5"))
    need = int(start) + horizon + verify + 1 + VERIFY_WIDTH_MARGIN
    for rung in ladder:
        if rung >= need:
            return rung
    return None


def dflash_width_set(max_context, num_blocks, horizon=None, verify=None, max_rungs=None, block_size=64):
    """The verify WIDTH SET a dFlash server can actually capture.

    ``dflash_pv_bucket_ladder`` gives the widths the CONTEXT needs; this bounds
    them by what the KV POOL can address. The verify page table is
    ``pv_sk // block_size`` entries wide and ``paged_update_cache`` requires
    ``max_num_blocks_per_seq < max_num_blocks``, so a width past the per-request
    block budget kills the engine during warmup:

        max_num_blocks_per_seq must be less than max_num_blocks:
        max_num_blocks_per_seq=4144, max_num_blocks=4128

    which is the ladder's top rung (265216 = round_up_1024(262144 + 2118)) on a
    262144-token server. The top width is therefore the block budget, not the
    rounded-up context, and the last ``P_v + 64`` positions of the context are
    served by it.

    Pure: config-time arithmetic, no device and no model instance.
    """
    ladder = dflash_pv_bucket_ladder(max_context, horizon=horizon, verify=verify, max_rungs=max_rungs)
    cap = ((int(num_blocks) * int(block_size)) // 1024) * 1024 if num_blocks else 0
    if not cap:
        return ladder
    ladder = [w for w in ladder if w <= cap]
    if not ladder or ladder[-1] != cap:
        ladder.append(cap)
    return ladder


def mtp_pv_width_ladder(max_context, draft_len, bucket=1024, max_rungs=None):
    """Verify widths an MTP serving session can migrate through.

    The packed verify's ``S_k`` is fixed per captured trace, and a replay may
    only attend positions the mask it was captured with covers, so a session
    needs ``S_k >= pos + K + 2`` at every position it reaches. Sizing one
    capture to ``anchor_pos + horizon`` is what makes the horizon a hard
    GENERATION BUDGET; a ladder lets the session move to a wider trace instead.

    Doubling keeps the rung count logarithmic and bounds the wasted width at
    <2x, which matters more here than for dFlash: the MTP masks are rebuilt on
    the host and uploaded at ``[1, 1, P_v, S_k]`` on EVERY replay, so width is
    per-iteration bandwidth, not just device footprint.

    Pure: config-time arithmetic, no device and no model instance.
    """

    def _r(n):
        return ((int(n) + bucket - 1) // bucket) * bucket

    tail = int(draft_len) + 2
    smallest = _r(tail)
    largest = _r(int(max_context) + tail)
    ladder, rung = [], smallest
    while rung < largest:
        ladder.append(rung)
        rung = _r(rung * 2)
    ladder.append(largest)
    if max_rungs is not None and len(ladder) > int(max_rungs):
        # Keep the LARGEST rungs: dropping a small rung only widens short
        # prompts, while dropping the largest would leave long ones uncovered.
        ladder = ladder[-int(max_rungs) :]
    return ladder


def _dflash_drafter_config(snapshot):
    """Read the drafter checkpoint's HF config. Pure: file read, no device."""
    import json as _json

    with open(os.path.join(snapshot, "config.json")) as fh:
        cfg = _json.load(fh)
    return cfg.get("text_config") or cfg


def _dflash_mesh_tp():
    """Tensor-parallel width from ``MESH_DEVICE`` (e.g. ``P150x8`` -> 8).

    Returns 1 when unset or unparseable (Galaxy DP entries set no MESH_DEVICE).
    1 is the SAFE default here: ``local_kv = n_kv // tp``, so tp=1 yields the
    largest per-chip cache and therefore over-reserves rather than under-.
    """
    import re as _re

    m = _re.search(r"[xX](\d+)\s*$", os.environ.get("MESH_DEVICE", "") or "")
    return int(m.group(1)) if m else 1


def _hf_hub_cache_dirs():
    """Every hub root worth searching, most specific first.

    ``~/.cache/huggingface/hub`` is only the DEFAULT. A runner that points the
    cache elsewhere sets HF_HUB_CACHE (the tt-metal vllm-model-tests runner
    uses /mnt/MLPerf/huggingface/hub) or HF_HOME, and a drafter resolved by
    globbing the default alone is simply not found there -- which is rejected
    at config time, so the server fails to start rather than degrading.
    """
    roots = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            roots.append(v)
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(os.path.join(home, "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    seen, out = set(), []
    for r in roots:
        r = os.path.expanduser(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _hf_snapshot_glob(repo_dir):
    """First snapshot of ``repo_dir`` across the hub roots, or None."""
    import glob as _glob

    for root in _hf_hub_cache_dirs():
        hits = _glob.glob(os.path.join(root, repo_dir, "snapshots", "*", ""))
        if hits:
            return hits[0]
    return None


def _hf_cache_is_writable():
    """Whether the HF hub cache can be written, i.e. whether a fetch can land.

    This, not HF_HUB_OFFLINE, is the condition that decides it. The CI weights
    share is mounted :ro by default and :rw when the operator asks for write
    access, and HF_HUB_OFFLINE is set because HF writes metadata (refs/main,
    .no_exist markers) on every HEAD call, which fails on a :ro mount with
    "Read-only file system". When the mount IS writable that objection is gone
    and a fetch is exactly what the operator asked for, so probe the mount
    rather than reading a flag that stands in for it.
    """
    for root in _hf_hub_cache_dirs():
        probe = root if os.path.isdir(root) else os.path.dirname(root.rstrip("/"))
        if probe and os.path.isdir(probe) and os.access(probe, os.W_OK):
            return True, root
    return False, None


def _hf_resolve_repo(repo_id, cache_dir_name):
    """Cache first, then the hub -- the way the TARGET model is resolved.

    transformers ``from_pretrained`` reads HF_HUB_CACHE and downloads what is
    missing. A drafter that only globbed the cache diverged from that: the
    target model appears and the drafter does not, and the failure reads as a
    missing checkpoint rather than "nobody fetched it".

    Returns a local path, or None when the repo is neither cached nor
    fetchable. A read-only cache is not an error here: the caller reports the
    miss with the env var that overrides it, which is more use than an HF
    stack trace.
    """
    hit = _hf_snapshot_glob(cache_dir_name)
    if hit:
        return hit
    writable, root = _hf_cache_is_writable()
    if not writable:
        logger.info(
            f"Gemma4: {repo_id} is not in the HF cache and the cache is read-only, "
            "so it will not be fetched; seed it or re-run with write access"
        )
        return None
    try:
        from huggingface_hub import constants as _hf_constants
        from huggingface_hub import snapshot_download

        # HF_HUB_OFFLINE is set for the read-only case, which does not apply to
        # a writable cache; disable it for this call only so the fetch is
        # allowed. Popping the ENV VAR is not enough and was the bug: hub reads
        # it once at import into constants.HF_HUB_OFFLINE, and every offline
        # gate goes through constants.is_offline_mode(), which returns that
        # module global. So a :rw run still raised LocalEntryNotFoundError
        # ("outgoing traffic has been disabled") and the drafter never
        # downloaded -- exactly the case the writable branch exists to serve.
        # Patch the global too, and restore both.
        saved = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        saved_offline = _hf_constants.HF_HUB_OFFLINE
        _hf_constants.HF_HUB_OFFLINE = False
        try:
            path = snapshot_download(repo_id=repo_id)
        finally:
            _hf_constants.HF_HUB_OFFLINE = saved_offline
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        logger.info(f"Gemma4: fetched {repo_id} into {root}")
        return path
    except Exception as exc:  # network, auth, or a repo that does not exist
        logger.warning(f"Gemma4: could not fetch {repo_id}: {type(exc).__name__}: {exc}")
        return None


def _target_weights_dir():
    """Local directory holding the TARGET model's safetensors.

    The drafter's embedding loader reads the target's embed_tokens straight out
    of the checkpoint, so it needs a real directory. ``MODEL_WEIGHTS_DIR`` is
    what tt-inference-server exports, but tt-metal's own vLLM CI does not set
    it: there ``HF_MODEL`` is a REPO ID and vLLM resolves the snapshot itself.
    Relying on the env var alone built the path "None/model.safetensors.index.json"
    and took the engine down during warmup after the drafter had downloaded
    fine. Fall back through the same places the rest of gemma4 looks, ending at
    the hub cache lookup the drafter itself uses.
    """
    cand = os.environ.get("MODEL_WEIGHTS_DIR")
    if cand and os.path.isdir(cand):
        return cand
    for env in ("HF_MODEL", "GEMMA4_MODEL_PATH"):
        cand = os.environ.get(env)
        if cand and os.path.isdir(cand):
            return cand
        # A repo id ("google/gemma-4-31B-it") resolves through the hub cache,
        # which is exactly how the leg already found the drafter.
        if cand and "/" in cand and not os.path.isabs(cand):
            hit = _hf_snapshot_glob("models--" + cand.strip("/").replace("/", "--"))
            if hit:
                return hit
    return None


def _dflash_default_snapshot():
    """Locate the z-lab drafter snapshot, cached or fetched (target parity)."""
    return _hf_resolve_repo("z-lab/gemma-4-31B-it-DFlash", "models--z-lab--gemma-4-31B-it-DFlash")


#: The it-assistant (MTP drafter) checkpoint of each target checkpoint, keyed by the
#: lower-cased marker its name carries. The 26B-A4B marker goes first: no other
#: target's marker is a substring of it, and the match stops at the first hit.
ASSISTANT_REPO_BY_TARGET = (
    ("26b-a4b", "google/gemma-4-26B-A4B-it-assistant"),
    ("31b", "google/gemma-4-31B-it-assistant"),
    ("12b", "google/gemma-4-12B-it-assistant"),
)


def assistant_repo_for(hf_model):
    """The assistant repo of the target checkpoint ``hf_model`` names (a repo id or a weights path).

    Matched case-insensitively on the whole string, so a weights-symlink
    directory such as ``.../gemma-4-26B-A4B-it`` resolves like the repo id. A
    name that matches no known target, or more than one, raises: defaulting to
    one assistant would load a drafter whose shared-KV layout belongs to
    another target. Pure.
    """
    name = str(hf_model).lower()
    repos = [repo for marker, repo in ASSISTANT_REPO_BY_TARGET if marker in name]
    if not repos:
        raise ValueError(
            f"no assistant checkpoint is known for the target {hf_model!r}; the name must carry one of "
            f"{[marker for marker, _ in ASSISTANT_REPO_BY_TARGET]} (set GEMMA4_ASSISTANT_MODEL to choose one explicitly)"
        )
    if len(repos) > 1:
        raise ValueError(
            f"the target {hf_model!r} names more than one checkpoint ({repos}); set GEMMA4_ASSISTANT_MODEL"
        )
    return repos[0]


def _assistant_default_snapshot(hf_model):
    """Resolve the it-assistant (MTP drafter) checkpoint path.

    On the server ``HF_MODEL`` is a LOCAL weights-symlink dir, so the harness
    default ``f"{HF_MODEL}-assistant"`` yields a bogus path that
    AutoConfig.from_pretrained rejects (HFValidationError). Resolve robustly:
    (1) if ``{hf_model}-assistant`` is an existing dir, use it; else (2) map
    the target's name to its assistant repo (``assistant_repo_for``, which
    raises for an unknown target) and glob the HF cache snapshot.
    """
    cand = f"{hf_model}-assistant"
    if os.path.isdir(cand):
        return cand
    repo = assistant_repo_for(hf_model)
    hit = _hf_resolve_repo(repo, "models--" + repo.replace("/", "--"))
    return hit if hit else cand


def _spec_first_slot(empty_slots):
    """The state slot a B=1 speculative session belongs to, or None.

    The runner passes ``empty_slots`` to prefill and later calls
    ``release_request(slot)`` with the SAME slot, so this is what lets the
    model tell its own request's release from another's. None when the runner
    supplied nothing, which is the pre-existing single-session case.
    """
    if not empty_slots:
        return None
    try:
        return int(list(empty_slots)[0])
    except (TypeError, ValueError):
        return None


def _spec_ring_blocks_for(window, drafts):
    """Headroom blocks giving the smallest legal ring that holds ``drafts`` extra rows.

    The ring must stay a power of two (chunk starts must be multiples of both
    the ring and SDPA's q_chunk_size), so this rounds ``window + drafts`` up to
    one. For the shipped 1024 window and K=5 that is 2048, i.e. 16 blocks.
    """
    need = int(window) + int(drafts)
    ring = 1 << (need - 1).bit_length()
    return (ring - int(window)) // _RING_HEADROOM_BLOCK


def _auto_size_spec_ring(cls, hf_config, bounded_sliding):
    """Size the bounded ring so a speculating class can hold its verify rows.

    A packed verify writes the anchor and K drafts before attention runs. Only
    the K DRAFT writes evict live history: the anchor at p lands in slot
    p % ring, which held p - ring, already outside the window. So the ring needs
    ``K`` slots of headroom, not K + 1 -- the same test the contract rail
    applies per request (``_dflash_ring_advice``).

    Done here because ``bounded_ring_modulo`` is read process-wide by the model
    and trace paths, which do not know whether the serving class speculates,
    and because ``create_tt_model`` sizes the bounded KV specs from it. An
    operator's own value is left alone.

    This used to be the operator's job through the env var, and a server that
    did not set it speculated on an exact ring and served plausible, wrong text
    (tt-metal#57701). The env var was the wrong interface: the model already
    knows the class, K, whether bounded sliding resolved on, and the window.
    """
    if not bounded_sliding:
        return
    drafts = int(getattr(cls, "_SPEC_N", 0) or 0) - 1
    if drafts <= 0:
        return  # not a speculating class, or K=0
    text_config = getattr(hf_config, "text_config", hf_config)
    window = getattr(text_config, "sliding_window", None)
    if window is None:
        return  # full attention everywhere: no ring
    window = int(window)
    if window <= 0 or window % _RING_HEADROOM_BLOCK:
        return
    if os.environ.get(SPEC_RING_HEADROOM_ENV):
        return  # operator pinned a value; _reserve_spec_ring_headroom checks it
    blocks = _spec_ring_blocks_for(window, drafts)
    os.environ[SPEC_RING_HEADROOM_ENV] = str(blocks)
    logger.info(
        f"{cls.__name__}: bounded sliding KV with speculation (K={drafts}) needs ring headroom; "
        f"set {SPEC_RING_HEADROOM_ENV}={blocks} (ring {window + blocks * _RING_HEADROOM_BLOCK} "
        f"for a {window} window). The bounded pool is sized per sliding layer from the ring."
    )


def _reserve_spec_ring_headroom(sliding_window, verify_width, where, *, declines_at_wrap=False):
    """Refuse to speculate on a bounded ring that cannot hold the verify.

    ``_auto_size_spec_ring`` gives every speculating class a ring that passes
    this, so reaching the raise means an operator pinned
    ``GEMMA4_SPEC_RING_HEADROOM_BLOCKS`` too small. It is kept as a backstop
    because the failure it catches is silent: a packed verify writes drafts at
    p+1..p+K before attention runs, and slot (p+j)%ring still holds position
    p+j-ring, which is inside the live window. Every sliding layer -- 50 of 60
    on 31B -- then attends a shortened window, so the COMMITTED tokens are wrong
    from the first generated token of any prompt past ``ring - K``, with no
    failed request and no error (tt-metal#57701).

    ``declines_at_wrap`` classes stop proposing before a verify would wrap and
    continue as plain decode, so an exact ring costs them speed, not
    correctness. They are advised, not refused.
    """
    from models.demos.gemma4.tt.attention import _RING_HEADROOM_BLOCK, SPEC_RING_HEADROOM_ENV, bounded_ring_modulo

    if sliding_window is None:
        return
    window = int(sliding_window)
    if window <= 0 or window % _RING_HEADROOM_BLOCK:
        return
    # Validated before the width check so this guard's error path does not depend
    # on verify_width: a ring that is not a power of two is wrong for any width,
    # and bounded_ring_modulo raising here names the bad env value at config time
    # rather than leaving it to surface from the model or trace path later.
    ring = bounded_ring_modulo(window)

    # Only the drafts evict live history; the anchor write lands on a slot
    # holding ``p - ring``, which the window has already dropped.
    drafts = int(verify_width or 0) - 1
    if drafts <= 0:
        return
    if ring is not None and int(ring) - window >= drafts:
        return  # headroom covers every draft write
    if declines_at_wrap:
        return  # the class stops proposing before the wrap; it says so itself

    blocks = _spec_ring_blocks_for(window, drafts)
    raise RuntimeError(
        f"{where}: bounded sliding KV on a ring of {ring} cannot hold a verify of "
        f"{drafts} drafts on a {window} window. The drafts land in slots that still "
        f"hold live window positions, so every sliding layer attends a shortened "
        f"window and the committed tokens are wrong from the first one past position "
        f"{int(ring) - drafts} (tt-metal#57701). Unset {SPEC_RING_HEADROOM_ENV} to let "
        f"the model size the ring itself, or set it to {blocks} (ring "
        f"{window + blocks * _RING_HEADROOM_BLOCK}); the bounded pool is sized per "
        f"sliding layer from the ring, so lower GEMMA4_MAX_TOKENS_ALL_USERS if it "
        f"does not fit. Serving without bounded sliding KV also avoids this."
    )


class Gemma4DFlashBase(Gemma4ForCausalLM):
    """The dFlash drafter, the fused decoder and its verify widths.

    :class:`Gemma4DFlashContractForCausalLM` (the plugin's speculative
    contract) serves one speculating request at a time through one persistent
    ``DFlashFusedDecoder``. This class owns that decoder's life: the drafter
    load, the width-set preparation and capture in warmup, the session
    bootstrap from prefill taps, the release, and the plan arithmetic the
    plugin admits a launch with; prefill tap capture and decode live in the
    subclass.

    Subclasses set ``_SPEC_V`` (drafts verified per iteration) and ``_SPEC_N``
    (``_SPEC_V + 1``, the packed-verify rows), and may set ``_SPEC_CONTRACT_K``
    to pin the fused decoder's verify count.
    """

    # -- plugin speculative contract (vllm-tt-plugin#110 s.2) -----------------
    @classmethod
    def _spec_plan_one_row(cls, requested_k: int, *, supports_narrow_decode: bool):
        """The plan for one speculating request at ``K``, or the reason it cannot speculate.

        Pure and side-effect free: reads env + the drafter's ``config.json`` and
        allocates nothing. Deliberately takes no ``vllm_config``: the platform
        stores ``get_tt_*`` on it later and two of them default to 1 silently
        (#110 s.2).

        dFlash captures one fused trace per verify width of the width set; the
        supported K is the single configured verify width rather than a range.
        """
        # The PLUGIN owns these types (vllm_tt_plugin.spec_decode, merged in
        # vllm-tt-plugin#120) and its admission path does
        # ``isinstance(outcome, SpecReject)``. Returning a locally defined
        # look-alike would make that check False and a reject would be read as
        # a plan. Imported lazily: tt-metal must not import the plugin at
        # module scope.
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        verify = getattr(cls, "_SPEC_CONTRACT_K", None)
        if verify is None:
            verify = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5"))

        snapshot = os.environ.get("GEMMA4_DFLASH_DRAFTER") or _dflash_default_snapshot()
        if not snapshot:
            # Reject at CONFIG time. Without this the drafter is resolved lazily
            # at the first request and a missing checkpoint kills the engine
            # there instead (EngineDeadError, no useful traceback).
            return SpecReject(
                reason=(
                    "Gemma4 dFlash drafter checkpoint not found: set GEMMA4_DFLASH_DRAFTER or "
                    "fetch it (hf download z-lab/gemma-4-31B-it-DFlash)."
                ),
                supported_k=(verify,),
            )
        try:
            cfg = _dflash_drafter_config(snapshot)
            n_layers = int(cfg["num_hidden_layers"])
            hidden = int(cfg["hidden_size"])
            head_dim = int(cfg["head_dim"])
            n_kv = int(cfg["num_key_value_heads"])
            block_size = int(os.environ.get("GEMMA4_DFLASH_BLOCK", cfg["block_size"]))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return SpecReject(
                reason=(
                    f"Gemma4 dFlash drafter config unreadable at {snapshot}: {exc!r}. Point "
                    "GEMMA4_DFLASH_DRAFTER at a z-lab/gemma-4-31B-it-DFlash snapshot."
                ),
                supported_k=(verify,),
            )

        if not 1 <= verify < block_size:
            return SpecReject(
                reason=(
                    f"Gemma4 dFlash verify count {verify} is outside [1, {block_size - 1}] "
                    f"for drafter block_size={block_size}. Set GEMMA4_DFLASH_VERIFY within that range."
                ),
                supported_k=(),
            )
        if requested_k < verify:
            # Captured shapes use the model's resolved count, independent of
            # how many drafts the scheduler requested.
            return SpecReject(
                reason=(
                    f"Gemma4 dFlash verifies exactly {verify} drafts per iteration "
                    f"(GEMMA4_DFLASH_VERIFY); requested_k={requested_k} is below that and no "
                    "narrower verify bucket is captured."
                ),
                supported_k=(verify,),
            )

        tp = _dflash_mesh_tp()
        replicated = os.environ.get("GEMMA4_DFLASH_REPLICATED", "0") == "1"
        local_kv = n_kv if replicated else max(1, n_kv // tp)

        # Fixed per-session device bytes, PER CHIP (#110 expresses the Qwen3.6
        # figures per chip). Mirrors DFlashFusedDecoder.__init__ allocations:
        # every tensor below is bf16 (2 B) except the int64 position vectors.
        cap = int(os.environ.get("GEMMA4_DFLASH_CTX_CAP", "2048"))
        p_v = verify + 1  # packed-verify rows
        ctx_dev = cap * hidden * 2
        ctx_kv = 2 * n_layers * local_kv * cap * head_dim * 2
        ctx_pos = cap * 8
        fc_prev = p_v * hidden * 2
        commit_pos = p_v * 8
        extra_bytes_per_seq = ctx_dev + ctx_kv + ctx_pos + fc_prev + commit_pos

        return SpecPlan(
            effective_k=verify,
            lanes_per_request=1,
            extra_bytes_per_seq=extra_bytes_per_seq,
            # The drafter carries no paged KV that grows with the sequence: its
            # context cache is the fixed ``cap``-row window above.
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            drafter_target_cache_requires=(),
            supports_narrow_decode=supports_narrow_decode,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec_drafter = None
        self._spec_decoder = None
        self._spec_decoder_bucket = None  # packed-verify width bucket of the cached decoder (reuse key)
        self._spec_pending = None  # (taps, prompt_len) awaiting first decode
        self._spec_active = False
        self._spec_horizon = int(os.environ.get("GEMMA4_DFLASH_SERVE_HORIZON", "2048"))
        if self._bounded_sliding_kv_cache:
            _reserve_spec_ring_headroom(
                getattr(self._text_config(), "sliding_window", None),
                self._SPEC_N,
                type(self).__name__,
                declines_at_wrap=self._SPEC_DECLINES_AT_RING_WRAP,
            )
        # WIDTH SET (vllm-tt-plugin#110 s.8, tt-metal#56048 review step 3): the
        # verify widths are derived from max_model_len at CONFIG time and every
        # one is captured in warmup, so serving never captures and no captured
        # shape depends on a request: the largest width covers max_model_len,
        # so a request migrates to a wider trace instead of hitting a horizon.
        # DEFAULT ON. Validated on a P150x8 31B server at 262144 max_model_len
        # with bounded sliding: mid-generation migration coherent, needle
        # retrieved at 121k and 238k, 32/32 needles at conc-32, and decode rate
        # within noise of the per-session path at both 4k and 121k (see the
        # commit message for the numbers). GEMMA4_DFLASH_WIDTH_SET=0 restores
        # per-session capture -- and with it the horizon generation cap.
        self._spec_width_set = os.environ.get("GEMMA4_DFLASH_WIDTH_SET", "1").lower() in ("1", "true", "yes")
        self._spec_width_ladder = None
        logger.info(
            f"Gemma4DFlash serving: V={self._SPEC_V} (N={self._SPEC_N}/step), "
            f"horizon={self._spec_horizon} new tokens/request, B=1 sessions"
        )

    # -- drafter ------------------------------------------------------------
    def _spec_get_drafter(self):
        if self._spec_drafter is not None:
            return self._spec_drafter
        from models.demos.gemma4.tt.dflash_drafter import DFlashDrafter

        snap = os.environ.get("GEMMA4_DFLASH_DRAFTER") or _dflash_default_snapshot()
        if not snap:
            raise RuntimeError("dFlash drafter snapshot not found; set GEMMA4_DFLASH_DRAFTER")
        model0 = self.model[0]
        weights_dir = _target_weights_dir()
        if not weights_dir:
            raise RuntimeError(
                "dFlash drafter needs the TARGET model's weights directory to load "
                "its embedding table, and none could be resolved. Set "
                "MODEL_WEIGHTS_DIR (or point HF_MODEL/GEMMA4_MODEL_PATH at a local "
                "checkpoint, or seed the hub cache for it)."
            )

        def _embed_loader():
            import json as _json

            from safetensors import safe_open

            idx = _json.load(open(f"{weights_dir}/model.safetensors.index.json"))
            key = next(
                k
                for k in idx["weight_map"]
                if k.endswith("language_model.embed_tokens.weight") or k.endswith("model.embed_tokens.weight")
            )
            with safe_open(f"{weights_dir}/{idx['weight_map'][key]}", framework="pt") as f:
                return f.get_tensor(key)

        self._spec_drafter = DFlashDrafter(
            mesh_device=self.mesh_device,
            drafter_path=snap,
            target_embed_weight_loader=_embed_loader,
            mesh_config=model0.mesh_config,
            ccl_manager=model0.ccl_manager,
            tensor_cache_path=None,
        )
        return self._spec_drafter

    def warmup_model_decode(self, *args, **kwargs):
        """Warm the BATCHED BASELINE decode buckets, then the verify widths.

        Both rails serve every step with several live rows through the plain
        batched baseline, so those buckets need warming exactly like a non-spec
        model. The verify widths are prepared on the eager pass and captured on
        the traced pass, before ordinary trace capture.

        This used to no-op in spec mode because the base warmup reader rejected
        the old sentinel-PADDED block output. That padding is gone -- the
        batched and no-session paths now return RAW DEVICE OUTPUT at width 1 --
        so the base warmup applies again. Skipping it made the first conc>1
        request of a run pay trace capture inside the measured window: on a
        P150x8 31B benchmark the 128/128 conc-32 cell showed TTFT 27.1 s cold
        against 11.7 s once the same shape had been driven first.

        Kill switch: GEMMA4_DFLASH_WARMUP_DECODE=0 restores the old no-op.
        """
        if os.environ.get("GEMMA4_DFLASH_WARMUP_DECODE", "1").lower() not in (
            "1",
            "true",
            "yes",
        ):
            del args, kwargs
            self._decode_warmup_complete = True
            logger.info("Gemma4DFlash: batched decode warmup disabled by env")
            return None
        logger.info("Gemma4DFlash: warming batched baseline decode buckets")
        out = super().warmup_model_decode(*args, **kwargs)
        # Eager warmup prepares persistent dFlash allocations before either
        # ordinary prefill or ordinary decode records temporary addresses.
        if kwargs.get("enable_trace") is False:
            self._spec_get_drafter()
            if self._spec_width_set:
                self._spec_capture_width_set(kwargs.get("kv_cache"), kwargs.get("num_blocks"), prepare_only=True)
        elif self._spec_width_set and kwargs.get("enable_trace"):
            self._spec_capture_width_set(kwargs.get("kv_cache"), kwargs.get("num_blocks"))
        return out

    def _spec_capture_width_set(self, kv_cache, num_blocks, *, prepare_only=False):
        """Capture one fused verify trace per width, before the first request.

        The decoder built here is PERSISTENT for the life of the server: the
        traces belong to its buffers, so every request reseeds this one instance
        rather than constructing its own. That is the cross-request reuse path
        (GEMMA4_DFLASH_DECODER_REUSE), re-measured byte-identical to per-request
        capture on a 31B server -- same acceptance, same replies -- so making it
        the only path costs nothing and saves the per-request capture.

        The scratch page table is all zeros (block 0) at the FULL per-request
        width, which is what sizes v_pt at its maximum; every request refreshes
        the contents. Writing a handful of verify rows into block 0 at positions
        [0, P_v) during capture is harmless: any request that later owns that
        block overwrites those positions with its own prefill before it decodes.
        """
        import time as _time

        from models.demos.gemma4.tt.dflash_drafter import DFlashFusedDecoder

        if kv_cache is None:
            return
        max_seq_len = int(getattr(self.model_args[0], "max_seq_len", 0))
        ladder = dflash_width_set(
            max_seq_len,
            num_blocks,
            horizon=self._spec_horizon,
            verify=self._SPEC_V,
            max_rungs=int(os.environ.get("GEMMA4_DFLASH_WIDTH_RUNGS", "0")) or None,
        )
        kv_layers = kv_cache
        if (
            isinstance(kv_layers, (list, tuple))
            and kv_layers
            and isinstance(kv_layers[0], (list, tuple))
            and kv_layers[0]
            and isinstance(kv_layers[0][0], (list, tuple))
        ):
            kv_layers = kv_layers[0]
        blocks = int(num_blocks) if num_blocks else max(1, max_seq_len // 64)
        scratch_pt = torch.zeros(1, blocks, dtype=torch.int32)
        dec = self._spec_decoder
        if dec is None:
            dec = DFlashFusedDecoder(
                self.model[0],
                self._spec_get_drafter(),
                kv_layers,
                scratch_pt,
                verify_count=getattr(self, "_SPEC_CONTRACT_K", None),
            )
            self._spec_decoder = dec
            self._spec_width_ladder = ladder
        elif self._spec_width_ladder != ladder:
            raise RuntimeError("Gemma4 dFlash verify widths changed after preparation")
        if prepare_only:
            dec.prepare_widths(ladder)
            return
        t0 = _time.time()
        cost = dec.capture_widths(ladder)
        logger.info(
            f"Gemma4DFlash: captured {len(cost)} verify widths in {_time.time()-t0:.1f}s "
            f"(max_model_len={max_seq_len}, widths={ladder}, per-width={ {k: round(v, 2) for k, v in cost.items()} })"
        )

    # -- prefill: capture taps (untraced) ------------------------------------
    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False):
        """Warm the prefill buckets eagerly: dFlash rails never replay a prefill trace.

        Taps come from a python hook a traced replay skips, so a captured trace is dead weight (#57853).
        """
        super().warmup_model_prefill(
            kv_cache,
            enable_trace=False,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
        )

    # -- session: bootstrap from prefill taps, release ---------------------
    def _spec_bootstrap(self, anchor_id, start, page_table, kv_cache, page_tables_per_layer=None):
        import time as _time

        from models.demos.gemma4.tt.dflash_drafter import DFlashFusedDecoder

        taps, n = self._spec_pending
        self._spec_pending = None
        if start != n:
            logger.warning(f"Gemma4DFlash: first decode start_pos {start} != prompt_len {n}")
        kv_layers = kv_cache
        if (
            isinstance(kv_layers, (list, tuple))
            and kv_layers
            and isinstance(kv_layers[0], (list, tuple))
            and kv_layers[0]
            and isinstance(kv_layers[0][0], (list, tuple))
        ):
            kv_layers = kv_layers[0]
        model0 = self.model[0]
        # BOUNDED sliding target (auto-enabled at >=131072): install the hybrid
        # per-layer page tables so the fused verify's v_ptl / _pv_setup read the
        # small RING pool for sliding layers and the flat global table for full-
        # attention layers -- the exact set the metal harness installs via
        # build_hybrid_page_tables (dflash_isl_sweep.py). Without it the verify
        # falls through to the flat table for EVERY layer, so sliding layers read
        # the wrong physical blocks, every draft is rejected, and acceptance
        # collapses (~1 tok/iter) at >=131072. Unbounded (<131072) leaves it unset
        # (flat table is already correct), preserving the current path.
        if self._bounded_sliding_kv_cache and page_table is not None:
            # No fallback: continuing with the flat table makes sliding layers
            # read the wrong physical blocks and silently collapses acceptance
            # (~1 tok/iter) -- fail the request loudly instead (review finding
            # on tt-metal#56048).
            _ptpl = self._build_per_layer_page_tables(page_tables_per_layer, page_table)
            _ptpl = self._pad_sliding_page_tables_for_bounded(_ptpl, kv_cache, authoritative=True)
            if not _ptpl:
                raise RuntimeError(
                    "Gemma4DFlash: bounded per-layer page-table install produced "
                    "no tables; refusing to speculate against the flat table"
                )
            model0._active_page_tables_per_layer = _ptpl
        pt = page_table[:1] if page_table is not None else None
        horizon = self._spec_horizon
        t0 = _time.time()
        # REUSE the cached fused decoder when the new request shares its packed-
        # verify width bucket (pv_sk buckets by start/1024): the captured trace
        # reads persistent input buffers, so refreshing the KV page tables +
        # re-ingesting the prompt taps + re-uploading the anchor/pv inputs
        # re-points it at this request WITHOUT the ~2.6 s capture and WITHOUT
        # allocating/freeing a fresh decoder's buffers (which fragments DRAM).
        # A bucket change (different prompt length band) releases + re-captures.
        dec = self._spec_decoder
        # Cross-request decoder reuse: DEFAULT OFF, conservatively -- the two
        # paths are now EQUIVALENT, so OFF is simply the unchanged behaviour and
        # not a considered preference.
        #
        # The earlier rationale here ("reseeded sessions draft worse": capture
        # 4.2-5.5 vs reseeded 1.9-3.7 tokens/iter, reuse ON ~2x slower at
        # ISL >= 1024) was an ARTIFACT of the page-table lifetime bug and is
        # retracted. That bug corrupted the no-reuse branch only -- release
        # deleted the per-layer tables the bootstrap had just installed -- so it
        # made freshly CAPTURED sessions look bad and, when the numbers happened
        # to fall the other way, reuse look bad. reseed() was never the problem.
        #
        # Re-measured on a P150x8 31B server AFTER that fix, 40,624-token prompt,
        # 4 identical requests, temperature 0 -- byte-identical replies both ways:
        #   acceptance   OFF 3.05/4.12/4.16/4.16   ON 3.05/4.12/4.16/4.16
        #   wall/request OFF 13.8-13.9 s           ON 12.7-12.8 s
        #   median TPOT  OFF 29.16 ms              ON 30.16 ms  (random ISL 32768)
        # ON saves the ~1.1 s capture per request and costs nothing measurable in
        # TPOT, so GEMMA4_DFLASH_DECODER_REUSE=1 is a safe win on workloads that
        # stay inside one packed-verify width bucket; it is not the default only
        # because bucket churn re-captures anyway and the DRAM-fragmentation
        # question for long-lived servers has not been measured.
        # With the WIDTH SET every width is already captured on the persistent
        # decoder, so a request never captures: it selects its width, refreshes
        # the page tables and reseeds. A width the set does not cover would be a
        # configuration error (the largest rung covers max_model_len), so fall
        # through to a capture rather than serve a wrong width.
        if self._spec_width_set and dec is not None and pt is not None:
            w = dec.width_for(int(start))
            if w is not None:
                dec.refresh_page_tables(pt)
                dec.select_width(int(start))
                dec.prefill_ingest(taps, n)
                dec.reseed(int(anchor_id), int(start))
                self._spec_decoder_bucket = w
                self._spec_active = True
                self._spec_first_step = True
                self._spec_last_pt = None
                logger.info(
                    f"Gemma4DFlash session: width-set reseed {_time.time()-t0:.2f}s "
                    f"(anchor={int(anchor_id)}, start={start}, width={w})"
                )
                return
            # Should be unreachable: the largest rung covers max_model_len.
            # Fall back to per-session capture, and turn the width set OFF for
            # the rest of the process rather than leaving a decoder whose traces
            # nothing will release (the capture below replaces it).
            logger.warning(
                f"Gemma4DFlash: no captured verify width covers start={int(start)}; "
                "capturing one for this session and disabling the width set"
            )
            self._spec_width_set = False
            self._spec_width_ladder = None
        _reuse_ok = os.environ.get("GEMMA4_DFLASH_DECODER_REUSE", "0").lower() in ("1", "true", "yes")
        reused = (
            _reuse_ok
            and dec is not None
            and pt is not None
            and dec.pv_bucket(int(start), horizon) == self._spec_decoder_bucket
        )
        if reused:
            dec.refresh_page_tables(pt)
            dec.prefill_ingest(taps, n)
            dec.reseed(int(anchor_id), int(start))
        else:
            # Keep the per-layer tables installed above: the new decoder reads
            # them in __init__ (see _spec_release_decoder's drop_page_tables).
            self._spec_release_decoder(drop_page_tables=False)
            dec = DFlashFusedDecoder(
                model0, self._spec_get_drafter(), kv_layers, pt, verify_count=getattr(self, "_SPEC_CONTRACT_K", None)
            )
            dec.prefill_ingest(taps, n)
            dec.capture(int(anchor_id), int(start), max_new=horizon)
            self._spec_decoder = dec
            self._spec_decoder_bucket = dec.pv_bucket(int(start), horizon)
        self._spec_active = True
        self._spec_first_step = True
        self._spec_last_pt = None
        logger.info(
            f"Gemma4DFlash session: {'REUSE' if reused else 'capture'} {_time.time()-t0:.2f}s "
            f"(anchor={int(anchor_id)}, start={start}, bucket={self._spec_decoder_bucket})"
        )

    def _spec_release_decoder(self, drop_page_tables=True, *, teardown=False):
        dec = self._spec_decoder
        if not teardown and self._spec_width_set and getattr(dec, "_pv_widths", None):
            # Session release preserves startup widths. Final teardown bypasses
            # retention and releases their traces while the mesh is open.
            self._spec_active = False
            self._spec_decoder_bucket = None
            if drop_page_tables:
                try:
                    if hasattr(self.model[0], "_active_page_tables_per_layer"):
                        del self.model[0]._active_page_tables_per_layer
                except Exception:
                    pass
            return
        self._spec_decoder = None
        self._spec_decoder_bucket = None
        self._spec_active = False
        # Drop the bounded per-layer tables installed for this session so a later
        # BATCHED baseline decode (adaptive fallback) rebuilds its own set instead
        # of reading this request's stale ring tables. Re-installed on next
        # bootstrap; kept alive DURING the session because refresh reads it.
        #
        # ``drop_page_tables=False`` is for the one caller that releases the OLD
        # decoder AFTER installing the NEW request's tables (_spec_bootstrap's
        # no-reuse branch). Dropping them there deletes the install that the
        # DFlashFusedDecoder constructed on the next line depends on: its v_ptl
        # comes from model._active_page_tables_per_layer, so the new session
        # would fall back to the FLAT table for sliding layers and decode
        # garbage at long context.
        if drop_page_tables:
            try:
                if hasattr(self.model[0], "_active_page_tables_per_layer"):
                    del self.model[0]._active_page_tables_per_layer
            except Exception:
                pass
        if dec is None:
            return
        tids = {id(t): t for t in [getattr(dec, "trace", None)] if t is not None}
        for _rec in (getattr(dec, "_pv_widths", None) or {}).values():
            if _rec.get("trace") is not None:
                tids.setdefault(id(_rec["trace"]), _rec["trace"])
        for _tid in tids.values():
            try:
                ttnn.release_trace(self.mesh_device, _tid)
            except Exception as e:
                logger.warning(f"Gemma4DFlash: trace release failed: {e!r}")
        for attr in ("ctx_k", "ctx_v"):
            for t in getattr(dec, attr, None) or []:
                try:
                    t.deallocate(True)
                except Exception:
                    pass
        for _rec in (getattr(dec, "_pv_widths", None) or {}).values():
            for _t in (_rec.get("pv_iota"), *(_rec.get("cache_by_type") or {}).values()):
                if _t is not None:
                    try:
                        _t.deallocate(True)
                    except Exception:
                        pass
        for attr in (
            "ctx_dev",
            "fc_prev",
            "merge_idx",
            "commit_pos",
            "noise_rows",
            "anchor_row",
            "anchor_tok",
            "blk_pos",
            "mask_full",
            "mask_slide",
            "pv_iota",
            "pv_mask_slide",
            "pv_pos",
            "pv_widx_all",
            "out_ids",
        ):
            t = getattr(dec, attr, None)
            if t is not None:
                try:
                    t.deallocate(True)
                except Exception:
                    pass

    def read_decode_output(self, tt_out, async_read=False, *_, **__):
        # A step the drafter session answered returns committed host tokens
        # from decode_forward -- nothing to read, pass them straight through
        # (no events). Every other output is a DEVICE tensor: the batched and
        # no-session plain decode steps return raw device output for the async
        # read overlap. Route those to the base reader (events under
        # async_read) so the deferred pipeline reads them.
        import torch

        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *_, **__)

    def release_persistent_capture(self) -> None:
        self._spec_pending = None
        try:
            self._spec_release_decoder(teardown=True)
        finally:
            super().release_persistent_capture()


def _dflash_env_flag(name, default):
    """True when ``name`` holds one of the on-values the GEMMA4_* gates accept."""
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


@dataclasses.dataclass
class RetainedProposal:
    """What one fused replay produced, kept for the step that verifies it.

    The fused body drafts K tokens after ``anchor_token`` and, in the same
    replay, runs the target over the anchor and those drafts. ``posterior``
    holds the target's argmax at each of the 1+K positions. The step that
    carries the drafts answers from it without another target forward pass;
    ``consumed`` records that it answered once.
    """

    anchor_token: int
    anchor_position: int
    drafts: list
    posterior: list
    consumed: bool = False
    # How many of ``drafts`` the verify that consumed this proposal carried (0 until then, or when it
    # carried none): the next proposal reads the runner's accepted count against it.
    verified: int = 0


class Gemma4SpecContractBase(Gemma4ForCausalLM):
    """What every Gemma 4 adapter on the plugin's speculative-decoding contract shares.

    The contract (vllm-tt-plugin ``docs/SPEC_DECODE_CONTRACT.md``) hands a
    drafter to the runner one step at a time: ``output_tokens_per_step`` is 1,
    the runner builds the ``[B, 1+K]`` candidate block from the drafts the
    adapter proposed and owns the accept walk. Everything here is independent
    of which drafter proposes: the request identity (state slot plus a
    per-slot generation, so a reused slot never matches a stale owner), the
    lifecycle hooks the runner calls, the row and block helpers, the plain
    decode a non-speculating row runs, its resident-input guard, the
    argmax-eligibility test, the sampling-state consumption of a step the
    drafter answered, the ``DraftOutput`` builders and the bounded-ring
    advice.

    Subclasses own the drafter and the session shape:
    :class:`Gemma4DFlashContractForCausalLM` keeps one fused-decoder session
    for a solo request, :class:`Gemma4MTPContractForCausalLM` keeps one
    session per state slot. They implement the three identity hooks
    (``_spec_follow_slot_moves``, ``_spec_release_slot``,
    ``_spec_release_all``) and call ``_spec_init_state`` from their own
    state initialisation.
    """

    def _spec_init_state(self):
        self._spec_slot_gen = {}
        self._spec_resident_inputs_stale = False
        self._spec_last_boundary = None  # (point, perf_counter, verify) of the runner's last call boundary
        self._spec_ring = self._spec_ring_geometry()
        # The acceptance-adaptive speculation policy (``tt/spec_policy.py``): its knobs (read once here, a
        # bad value raises at construction), the batch prior and the exploration counters. ``_SPEC_N`` is
        # the rail's verify block height (K + 1).
        n = getattr(self, "_SPEC_N", None)
        if n is None:
            raise RuntimeError(f"{type(self).__name__} declares no _SPEC_N (verify block height) for the policy")
        self._spec_policy = SpecPolicy(SpecPolicyConfig.from_env(), int(n) - 1)
        # The cost table measured at warmup ({("step", shape, S_k): CostLine, ("reseed", shape): CostLine}) and
        # the plain decode step's CostLine per decode bucket ({bucket: CostLine}); None until measured, which
        # keeps the fixed behaviour serving. Both are the online calibration's priors (``_spec_record_ms``).
        self._spec_costs = None
        self._spec_plain_cost = None
        self._spec_cost_warning_logged = False
        # The instrumentation (``_spec_interval_begin`` / ``_spec_interval_end``): the previous proposal
        # (its begin time, the mode it decided, its live rows, the mode decided before it), what happened
        # since (prefills, a reseed or bootstrap), the phases timed in the current proposal and the stats
        # window.
        self._spec_last_propose = None
        self._spec_interval_logged = 0  # record intervals written at DEBUG (the first 200 of a run)
        self._spec_prefills_since = 0
        self._spec_dropped_since = 0  # rows (sessions) that ended since the last proposal: an idle interval test
        self._spec_entry_since = False
        self._spec_call_phases = {}
        self._spec_last_phase_ms = 0.0
        self._spec_window_steps = 0
        self._spec_mode_now = None
        self._spec_table_now = None  # the warmup table's ms for the mode just decided, at the step's position
        # (draft length of the last logged decision, its time, switches suppressed since): one INFO line per
        # switch, rate-limited to one per SPEC_POLICY_LOG_INTERVAL_S.
        self._spec_policy_logged = (None, 0.0, 0)

    # -- the acceptance-adaptive policy: what both rails share --------------------------
    def _spec_plain_buckets(self):
        """The decode buckets the plain step is timed at: the warmed ``tt_supported_decode_batch_sizes`` up to
        the engine batch, else the engine batch alone."""
        max_b = int(self.model_args[0].max_batch_size)
        sizes = getattr(self, "tt_supported_decode_batch_sizes", None) or ()
        buckets = sorted({int(b) for b in sizes if 1 <= int(b) <= max_b})
        return buckets or [max_b]

    def _spec_plain_bucket(self, live_rows):
        """The decode bucket ``live_rows`` run the plain step at: the smallest timed bucket covering them."""
        buckets = sorted(self._spec_plain_cost) if self._spec_plain_cost else [int(self.model_args[0].max_batch_size)]
        for bucket in buckets:
            if bucket >= int(live_rows):
                return bucket
        return buckets[-1]

    def _spec_plain_ms_for(self, live_rows, position):
        """``(ms, bucket)``: the plain step of ``live_rows`` at ``position``, measured in serving once its bucket
        is calibrated, else the warmup line of that bucket."""
        bucket = self._spec_plain_bucket(live_rows)
        prior = cost_at(self._spec_plain_cost[bucket], position)
        return self._spec_policy.calibration.step_ms(("plain", bucket), prior), bucket

    def _spec_record_ms(self, shape, width, position, partial=False):
        """One step of record ``shape`` at rung ``width``: the warmup line at ``position`` (every row of the shape
        live; ``partial``: the one-row line of a partial replay where the warmup found one cheaper, else the same
        line), scaled by the mode's measured ratio once calibrated; None when the table has no line for the rung."""
        key = ("step", tuple(shape), int(width))
        line = self._spec_costs.get(("partial", tuple(shape), int(width))) if partial else None
        if line is None:
            line = self._spec_costs.get(key)
        if line is None:
            return None
        return self._spec_policy.calibration.step_ms(key, cost_at(line, position))

    def _spec_record_table_ms(self, shape, width, position):
        """The warmup table's ms for record ``shape`` at rung ``width`` and ``position`` (the ratio's base)."""
        line = self._spec_costs.get(("step", tuple(shape), int(width))) if self._spec_costs else None
        return None if line is None else cost_at(line, position)

    def _spec_plain_table_ms(self, live_rows, position):
        """The warmup table's ms for the plain step of ``live_rows`` at ``position`` (the ratio's base)."""
        if not self._spec_plain_cost:
            return None
        return cost_at(self._spec_plain_cost[self._spec_plain_bucket(live_rows)], position)

    def _spec_reseed_ms(self, shape, position, prior=None):
        """The reseed of ``shape`` (the entry cost): the warmup line at ``position`` (or ``prior`` when the table
        has none, as for the dFlash bootstrap) scaled by the reseed's measured ratio once measured enough; None
        without either."""
        table = self._spec_reseed_table_ms(shape, position, prior)
        if table is None:
            return None
        return self._spec_policy.calibration.reseed_ms(tuple(shape), table)

    def _spec_reseed_table_ms(self, shape, position, prior=None):
        line = self._spec_costs.get(("reseed", tuple(shape))) if self._spec_costs else None
        return cost_at(line, position) if line is not None else prior

    def _spec_timed(self, phase, fn, *args, **kwargs):
        """Run ``fn`` and account its wall under ``phase`` (the current proposal's phases and the stats)."""
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1e3
        self._spec_last_phase_ms = ms
        self._spec_call_phases[phase] = self._spec_call_phases.get(phase, 0.0) + ms
        self._spec_policy.calibration.note_phase(phase, ms)
        return out

    def _spec_boundary(self, point, verify=False):
        """The runner's call timeline around the rail, as phases of the stats line: the wall from the end of a
        proposal to the start of the decode step that answers it (``gap_pv_v`` when that step is a verify,
        ``gap_pv_p`` for a plain step) and from the end of a decode step to the next proposal (``gap_vp_v`` /
        ``gap_vp_p`` by the step that ended). None of it is the rail's own work: it is the runner's iteration
        between its calls (the scheduler, the model input, the output processing, the drain), the serialized
        host cost of a speculative round that the device replay and the rail's phases do not account for."""
        now = time.perf_counter()
        last = self._spec_last_boundary
        policy = getattr(self, "_spec_policy", None)
        if last is not None and policy is not None:
            kind, t, was_verify = last
            if kind == "propose_end" and point == "decode_start":
                policy.calibration.note_phase("gap_pv_v" if verify else "gap_pv_p", (now - t) * 1e3)
            elif kind == "decode_end" and point == "propose_start":
                policy.calibration.note_phase("gap_vp_v" if was_verify else "gap_vp_p", (now - t) * 1e3)
        self._spec_last_boundary = (point, now, bool(verify))

    def _spec_note_prefill(self, empty_slots=None):
        """A prefill ran; ``empty_slots`` (the dFlash rail, which keeps no per-row session) are rows that ended."""
        self._spec_prefills_since += 1
        if empty_slots is not None:
            self._spec_dropped_since += len(list(empty_slots))
        self._spec_policy.calibration.note_prefill()

    def _spec_note_row_ended(self):
        """A row's session ended (the MTP rail): the interval that holds it is idle when every row did."""
        self._spec_dropped_since += 1

    def _spec_note_entry(self, shape, ms, table_ms=None):
        """A reseed (or bootstrap) of ``shape`` took ``ms`` against the table's ``table_ms``: the interval it
        falls in is not a clean sample."""
        self._spec_entry_since = True
        self._spec_policy.calibration.note_reseed(tuple(shape), ms, table_ms)

    @staticmethod
    def _spec_committed(counts, live, width):
        """``(tokens, rows)`` the runner committed in the interval that just closed: per live row the count
        clamped to the block width (0 for a row that did not step: just prefilled, or a slot the runner still
        lists), summed over the rows, and the number of rows that committed at least one token. Without counts
        every live row stepped once (the plain decode)."""
        if counts is None:
            return len(live), len(live)
        flat = counts.reshape(-1) if hasattr(counts, "reshape") else counts
        per_row = [max(0, min(int(flat[r]), int(width))) for r in live]
        return sum(per_row), sum(1 for c in per_row if c >= 1)

    def _spec_interval_begin(self, live_rows, tokens, committing=None):
        """At the top of a proposal: the wall since the previous proposal began is one step of the mode that
        proposal decided, counted dirty or clean (a prefill ran inside it, a reseed started it, the mode changed
        at its start or the live rows changed: the cause is kept for the stats line), unless it is idle: a
        prefill ran and every row the previous proposal served ended before this one, so the wall is the gap
        between requests, not the mode's work; ``tokens`` is what the runner committed in it."""
        now = time.perf_counter()
        last = self._spec_last_propose
        if last is not None:
            t_begin, mode, prev_live, before, table_ms = last
            idle = bool(self._spec_prefills_since) and self._spec_dropped_since >= int(prev_live)
            # the tokens per row of the closed interval are over the rows that committed in it (``committing``),
            # not the rows the proposal saw: a slot the runner lists but did not step dilutes the acceptance pool
            rows_for_pool = int(committing) if committing else int(prev_live)
            if self._spec_prefills_since:
                cause = "prefill"
            elif self._spec_entry_since:
                cause = "reseed"
            elif before != mode:
                cause = "switch"
            elif int(prev_live) != int(live_rows):
                cause = "rows"
            elif table_ms is None:
                cause = "untabled"
            else:
                cause = None
            self._spec_policy.calibration.note_step(
                mode, (now - t_begin) * 1e3, tokens, cause is None, table_ms, rows_for_pool, cause, idle=idle
            )
            if mode[0] == "step" and self._spec_interval_logged < 200:  # the first record intervals of a run
                self._spec_interval_logged += 1
                logger.debug(
                    f"Gemma4 spec interval: mode={mode} wall={(now - t_begin) * 1e3:.1f} ms tokens={tokens} "
                    f"live={int(live_rows)} prev_live={int(prev_live)} committing={committing} cause={cause}"
                )
        self._spec_prefills_since = 0
        self._spec_dropped_since = 0
        self._spec_entry_since = False
        self._spec_call_phases = {}
        self._spec_mode_now = None
        self._spec_table_now = None
        return now

    def _spec_interval_end(self, prefix, t_begin, mode, live_rows):
        """At the end of a proposal that decided ``mode`` for ``live_rows``: account the host bookkeeping
        (the proposal's wall less its timed phases) and open the next interval; every
        ``GEMMA4_SPEC_STATS_STEPS`` proposals the stats line is written."""
        total = (time.perf_counter() - t_begin) * 1e3
        self._spec_policy.calibration.note_phase("host", max(0.0, total - sum(self._spec_call_phases.values())))
        before = self._spec_last_propose[1] if self._spec_last_propose is not None else None
        self._spec_last_propose = (t_begin, tuple(mode), int(live_rows), before, self._spec_table_now)
        self._spec_window_steps += 1
        if self._spec_window_steps >= self._spec_policy.config.stats_steps:
            self._spec_log_stats(prefix)

    def _spec_log_stats(self, prefix, final=False):
        """``<prefix> adaptive stats: window N steps | <modes> | reseeds | phases | prefills | calibrated`` every
        window, and ``<prefix> adaptive summary: engine life N steps | ...`` on release."""
        calib = self._spec_policy.calibration
        if final:
            total = sum(stats.steps for stats in calib.modes.values())
            logger.info(f"{prefix} adaptive summary: engine life {total} steps | {calib.describe(window=False)}")
            return
        logger.info(f"{prefix} adaptive stats: window {self._spec_window_steps} steps | {calib.describe(window=True)}")
        calib.flush_window()
        self._spec_window_steps = 0

    def _spec_top_position(self):
        """The highest position a step's decision is costed at: the context minus one verify block."""
        return max(0, int(self.model_args[0].max_seq_len) - int(self._SPEC_N))

    def _spec_time_plain_step(self, warmup_kwargs, page_table, position=0, repeats=2, batch=None):
        """One plain decode step's cost in ms at ``position`` for the decode bucket ``batch`` (the engine batch
        when None): the minimum of ``repeats`` timed traced replays with readback.

        The call is the decode warmup's own (``WarmupForwardMixin``): the
        base decode on zero tokens at ``position`` through the first
        ``batch`` rows of ``page_table`` (block 0), the device-sampled trace
        of that bucket the serving steps replay when the warmup sampled on
        device, read back on return, so the number covers the input upload,
        the replay and the token readback.
        """
        max_b = int(batch or warmup_kwargs.get("max_batch_size") or self.model_args[0].max_batch_size)
        page_table = page_table[:max_b]
        tokens = torch.zeros(max_b, 1, dtype=torch.int32)
        start_pos = torch.full((max_b,), int(position), dtype=torch.int32)
        params = self._create_sampling_params(
            bool(warmup_kwargs.get("can_sample_on_device")), max_b, greedy_only=bool(warmup_kwargs.get("greedy_only"))
        )
        param = next((p for p in params if p is not None), None)
        best = None
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            Gemma4ForCausalLM.decode_forward(
                self,
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=warmup_kwargs.get("kv_cache"),
                enable_trace=True,
                read_from_device=True,
                sampling_params=param,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=param is not None,
                reset_sampling_state=False,
            )
            dt = (time.perf_counter() - t0) * 1e3
            best = dt if best is None else min(best, dt)
        return best

    def _spec_plain_step_cost(self, warmup_kwargs, page_table, top_position):
        """The plain step's ``CostLine`` from position 0 to ``top_position`` per decode bucket
        (``{bucket: CostLine}``): ``GEMMA4_SPEC_PLAIN_STEP_MS`` when set (one number is flat, two are the
        line's ends, the same for every bucket), else timed at both positions for every warmed bucket."""
        env = self._spec_policy.config.plain_step_ms
        top = int(top_position)
        table = {}
        for bucket in self._spec_plain_buckets():
            if env is not None:
                table[bucket] = CostLine(0, float(env[0]), top, float(env[-1]))
                continue
            at_zero = self._spec_time_plain_step(warmup_kwargs, page_table, position=0, batch=bucket)
            at_top = (
                self._spec_time_plain_step(warmup_kwargs, page_table, position=top, batch=bucket)
                if top > 0
                else at_zero
            )
            table[bucket] = CostLine(0, float(at_zero), top, float(at_top))
        return table

    def _spec_log_cost_table(self, prefix, costs, plain, timings, seconds):
        cfg = self._spec_policy.config
        source = SPEC_PLAIN_STEP_MS_ENV if cfg.plain_step_ms is not None else "timed traced replays"
        logger.info(
            f"{prefix} adaptive K: cost table ({timings} timings in {seconds:.1f}s; ms, lines in the position, "
            f"plain step from {source}; min_gain={cfg.min_gain:.2f}, exit_gain={cfg.exit_gain:.2f}, "
            f"entry_horizon={cfg.entry_horizon}, retry={cfg.retry_steps} steps at {cfg.retry_overhead:.3f} overhead, "
            f"probe={cfg.probe_steps}, ewma={cfg.ewma:.2f}, calib_min={cfg.calib_min}, "
            f"calib_probe={cfg.calib_probe_steps}, stats every {cfg.stats_steps}): {describe_costs(costs, plain)}"
        )

    def _spec_cost_table_ready(self, prefix):
        """Whether the policy can be consulted: it is on and the warmup left a cost table (else warn once)."""
        if not self._spec_policy.config.adaptive:
            return False
        if self._spec_costs and self._spec_plain_cost is not None:
            return True
        if not self._spec_cost_warning_logged:
            logger.warning(
                f"{prefix} adaptive K: no cost table (the warmup measured none and {SPEC_PLAIN_STEP_MS_ENV} is "
                "unset); speculation runs as without the policy"
            )
            self._spec_cost_warning_logged = True
        return False

    def _spec_policy_note(self, prefix, choice, previous, n, cap_text, stats, plain_ms):
        """One INFO line when the chosen draft length differs from the previous decision's, at most one per
        ``SPEC_POLICY_LOG_INTERVAL_S``; the switches a line could not be written for are counted on the next."""
        chosen_k = 0 if choice is None else int(choice.shape[1])
        if chosen_k == previous:
            return
        last_k, last_time, suppressed = self._spec_policy_logged
        now = time.monotonic()
        if last_k is not None and now - last_time < SPEC_POLICY_LOG_INTERVAL_S:
            self._spec_policy_logged = (last_k, last_time, suppressed + 1)
            return
        with_data = [s.ewma_accepted for s in stats if s.ewma_accepted is not None]
        accepted = f"{sum(with_data) / len(with_data):.2f}" if with_data else "prior"
        cond = [sum(s.effective_cond(s.k_max)[j] for s in stats) / n for j in range(stats[0].k_max)]
        cond_text = "/".join(f"{c:.2f}" for c in cond)
        was = "none" if previous is None else f"K={previous}"
        skipped = f", {suppressed} switches unlogged" if suppressed else ""
        if choice is None:
            why = self._spec_policy.last_reason or "declined"
            logger.info(
                f"{prefix} adaptive K: {n} rows, cap {cap_text} -> plain step ({why}; was {was}{skipped}); "
                f"EWMA accepted {accepted}, per-position {cond_text}, plain {plain_ms:.1f} ms"
            )
        else:
            entry = f", entry {choice.entry_ms:.1f} ms" if choice.entry_ms > 0.0 else ""
            members = f" for {len(choice.members)}/{n} rows" if choice.members is not None else ""
            calib = self._spec_policy.calibration
            mode = ("step", tuple(choice.shape), int(choice.width))
            credit, samples = calib.pool_credit(mode)
            pool_text = f", pool {credit:.2f}/{samples}" if credit is not None else ", pool none"
            run = calib.run_samples(mode)
            if run:
                pool_text += f", run {calib.tokens_per_row(mode):.2f}/{run}"
            logger.info(
                f"{prefix} adaptive K: {n} rows, cap {cap_text} -> record {choice.shape} S_k={choice.width}{members} "
                f"({choice.reason}; was {was}{skipped}); EWMA accepted {accepted}, per-position {cond_text}, "
                f"expected {choice.tokens_per_row:.2f} tokens/row at {choice.cost_ms:.1f} ms "
                f"({choice.cost_ms / plain_ms:.2f}x plain {plain_ms:.1f}{entry}{pool_text}), gain {choice.gain:.2f}x"
            )
        self._spec_policy_logged = (chosen_k, now, 0)

    # -- bounded ring ----------------------------------------------------------
    def _spec_ring_geometry(self):
        """``(ring, window)`` of the bounded sliding layers, or None when none is bounded."""
        models = getattr(self, "model", None) or []
        layers = getattr(models[0], "layers", None) if models else None
        for layer in layers or []:
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            ring = getattr(cfg, "cache_position_modulo", None)
            if ring is not None:
                return int(ring), int(cfg.sliding_window)
        return None

    def _spec_ring_advice(self, p_v, where):
        """Warn when an exact-window ring makes proposals decline past ``ring - p_v``."""
        if self._spec_ring is None:
            return
        ring, window = self._spec_ring
        p_v = int(p_v)
        if ring - window >= p_v - 1:
            return
        from models.demos.gemma4.tt.attention import _RING_HEADROOM_BLOCK, SPEC_RING_HEADROOM_ENV

        logger.warning(
            f"{where}: the bounded ring ({ring}) equals the sliding "
            f"window ({window}) and a verify writes P_v={p_v} rows, so proposals decline once a "
            f"request passes position {ring - p_v} and it continues as plain decode. "
            f"{SPEC_RING_HEADROOM_ENV}={window // _RING_HEADROOM_BLOCK} (ring {2 * window}) would "
            "lift that, once the bounded page tables are sized from the ring (tt-metal#57573); "
            "it also needs a lower GEMMA4_MAX_TOKENS_ALL_USERS to fit."
        )

    def _spec_decline_reason(self, position, p_v, width_for):
        """Why no verify block of ``p_v`` rows can be written at ``position``, or None.

        The block occupies rows ``position .. position + p_v - 1``: the anchor
        and the drafts. Each must be covered by a captured width
        (``width_for(position)`` is None when none is), by ``max_seq_len``,
        and on an exact-window ring must not wrap onto a slot the anchor's
        own query still reads.
        """
        position = int(position)
        p_v = int(p_v)
        if width_for(position) is None:
            return f"no captured verify width covers position {position}"
        max_seq_len = int(getattr(self.model_args[0], "max_seq_len", 0) or 0)
        if max_seq_len and position + p_v > max_seq_len:
            return f"verify rows {position}..{position + p_v - 1} exceed max_seq_len {max_seq_len}"
        if self._spec_ring is not None:
            ring, window = self._spec_ring
            if ring - window < p_v - 1 and position + p_v > ring:
                return f"verify rows would wrap the exact-window ring {ring} (window {window}, P_v {p_v})"
        return None

    # -- identity: state slot plus generation ---------------------------------
    def _spec_open_owner(self, slot):
        """Open a new generation of ``slot`` and return its owner key."""
        if slot is None:
            return None
        slot = int(slot)
        gen = self._spec_slot_gen.get(slot, 0) + 1
        self._spec_slot_gen[slot] = gen
        return (slot, gen)

    def _spec_owner_of(self, slot):
        """The owner key of ``slot``'s current generation, without opening a new one."""
        slot = int(slot)
        return (slot, self._spec_slot_gen.get(slot, 0))

    def _spec_owns(self, owner, row):
        if owner is None or row is None:
            return False
        slot, gen = owner
        return int(row) == slot and self._spec_slot_gen.get(slot) == gen

    def _spec_close_slot_generation(self, slot):
        """Close ``slot``'s generation: owners keyed on the old one stop matching."""
        slot = int(slot)
        self._spec_slot_gen[slot] = self._spec_slot_gen.get(slot, 0) + 1

    def _spec_follow_slot_moves(self, moved):
        """Re-key the subclass's owners after the runner moved slots ``{old: new}``."""
        raise NotImplementedError(f"{type(self).__name__} must implement _spec_follow_slot_moves")

    def _spec_release_slot(self, row):
        """Drop the subclass's state for ``row``; its generation is closed afterwards."""
        raise NotImplementedError(f"{type(self).__name__} must implement _spec_release_slot")

    def _spec_release_all(self):
        """Drop the subclass's state for every slot (``release_request(None)``)."""
        raise NotImplementedError(f"{type(self).__name__} must implement _spec_release_all")

    def note_state_slots_moved(self, moves) -> None:
        if not moves:
            return
        try:
            moved = {int(old): int(new) for old, new in dict(moves).items()}
        except (AttributeError, TypeError, ValueError):
            return
        gens = dict(self._spec_slot_gen)
        for old, new in moved.items():
            gens[new] = self._spec_slot_gen.get(old, 0)
        self._spec_slot_gen = gens
        self._spec_follow_slot_moves(moved)

    def release_request(self, row: int) -> None:
        if row is None:
            self._spec_release_all()
            return
        row = int(row)
        self._spec_release_slot(row)
        # Close the slot's generation: a request prefilled into it opens a new one.
        self._spec_close_slot_generation(row)

    def release_persistent_capture(self) -> None:
        self._spec_resident_inputs_stale = False
        super().release_persistent_capture()

    # -- rows and blocks -------------------------------------------------------
    @staticmethod
    def _spec_inputs(args, kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        start_pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        return tokens, start_pos

    @staticmethod
    def _spec_rows(tokens):
        if tokens is None:
            return 1
        return int(tokens.shape[0]) if tokens.dim() > 1 else int(tokens.numel())

    @staticmethod
    def _spec_live_rows(positions, rows):
        """Indices of rows that belong to real requests: the runner pads with -1."""
        if positions is None:
            return list(range(int(rows)))
        try:
            first = positions.reshape(int(rows), -1)[:, 0]
        except Exception:
            return list(range(int(rows)))
        return [int(r) for r in torch.nonzero(first >= 0).reshape(-1).tolist()]

    @staticmethod
    def _spec_row(values, rows, row):
        """Row ``row`` of a ``[rows, W]`` or flat ``[rows]`` tensor as ints."""
        v = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        return [int(x) for x in v.reshape(int(rows), -1)[int(row)].tolist()]

    @staticmethod
    def _spec_slot_of(row, kwargs):
        """The state slot row ``row`` reads on this step.

        The runner permutes per-slot state with ``slot_remap`` (``slot_remap[i]``
        is the slot whose state row ``i`` reads) and reports the move through
        ``note_state_slots_moved`` only after the model accepted the step, so
        during the step a request's row and its owner slot can differ.
        """
        remap = kwargs.get("slot_remap")
        if remap is None:
            return int(row)
        try:
            return int(remap[int(row)])
        except (IndexError, TypeError, ValueError):
            return int(row)

    @staticmethod
    def _spec_column_zero(args, kwargs, rows, exclude_rows=None):
        """Narrow a ``[rows, 1+K]`` block to the plain decode's ``[rows, 1]`` tokens and ``[rows]`` positions.

        Column 0 is each row's last committed token at its own position, which
        is the plain decode's input; padding rows keep their -1 position.
        Every row in ``exclude_rows`` is given position -1 too, so the target
        skips it.
        """
        args = list(args)
        kwargs = dict(kwargs)
        excluded = [int(r) for r in (exclude_rows or ())]

        def col0(t):
            if t is None:
                return None
            v = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
            if int(rows) <= 0 or v.numel() % int(rows):
                return v
            return v.reshape(int(rows), -1)[:, 0].contiguous()

        def tokens(t):
            v = col0(t)
            return v if v is None else v.reshape(int(rows), 1)

        def positions(t):
            v = col0(t)
            if v is not None and excluded:
                v = v.clone()
                for r in excluded:
                    if 0 <= r < int(v.numel()):
                        v[r] = -1
            return v

        for name, index, fn in (("tokens", 0, tokens), ("start_pos", 1, positions)):
            if kwargs.get(name) is not None:
                kwargs[name] = fn(kwargs[name])
            elif len(args) > index:
                args[index] = fn(args[index])
        return tuple(args), kwargs

    # -- argmax eligibility ----------------------------------------------------
    @staticmethod
    def _spec_row_requests_logprobs(sampling_params, row):
        log_probs = getattr(sampling_params, "enable_log_probs", None)
        if isinstance(log_probs, torch.Tensor) and log_probs.ndim == 0:
            log_probs = log_probs.item()
        elif log_probs is not None and not isinstance(log_probs, (bool, int, float)):
            log_probs = log_probs[int(row)]
        return bool(log_probs)

    @classmethod
    def _spec_row_can_use_argmax(cls, sampling_params, row, grammar_bitmask=None):
        """Whether target argmax IDs satisfy ``row``'s sampling and output requirements.

        A drafter session answers with a replay's argmax ids, so only such a
        row may take them. A penalty or a grammar bitmask changes the logits
        before the argmax; any bitmask rules the row out, without reading
        which rows it restricts. top_k and top_p need no check, because they
        only narrow a distribution that temperature 0 has already collapsed
        to one token. Requested logprobs need the ordinary sampler's metadata
        as well as IDs. Each sampling field is one value for every row or a
        per-row sequence.
        """
        if grammar_bitmask is not None or cls._spec_row_requests_logprobs(sampling_params, row):
            return False
        for name, neutral in (
            ("temperature", 0.0),
            ("presence_penalty", 0.0),
            ("frequency_penalty", 0.0),
            ("repetition_penalty", 1.0),
        ):
            value = getattr(sampling_params, name)
            if not isinstance(value, (int, float)):
                value = value[int(row)]
            if float(value) != neutral:
                return False
        return True

    # -- the plain decode and what a drafter-served step leaves behind ----------
    def _spec_note_served_step(self):
        """Record that the drafter answered a row this step without the plain decode.

        That decode's device-resident token and position for the row are
        stale from now on (``_spec_plain_decode``).
        """
        self._spec_resident_inputs_stale = True

    def _spec_prepare_sampling_state(self, kwargs, start_pos, rows):
        """Consume the caller's sampling-state commands when a replay answered the step.

        The device sampler did not run, so its parameter, history and slot
        commands are applied here exactly once; a host-sampled step (no
        ``sampling_params``) only follows the slot remap, as the plain decode
        does on that path.
        """
        sampling = kwargs.get("sampling_params")
        if sampling is None:
            self._apply_sampling_slot_remap(kwargs.get("slot_remap"))
            return
        positions = (
            torch.chunk(start_pos.reshape(int(rows), -1)[:, 0], self.data_parallel, 0)
            if start_pos is not None
            else None
        )
        self._prepare_decode_sampling_state(
            sampling,
            start_pos=positions,
            prompt_tokens=kwargs.get("prompt_tokens"),
            output_tokens=kwargs.get("output_tokens"),
            slot_remap=kwargs.get("slot_remap"),
            reload_inputs=kwargs["reload_inputs"],
            reload_sampling_params=kwargs["reload_sampling_params"],
            reset_sampling_state=kwargs["reset_sampling_state"],
        )

    @staticmethod
    def _spec_is_host(tt_out):
        """The runner's own test for a decode output that needs no device read."""
        if isinstance(tt_out, torch.Tensor):
            return True
        if isinstance(tt_out, tuple):
            return all(t is None or isinstance(t, torch.Tensor) for t in tt_out)
        return False

    def _spec_convert(self, host_groups, rows, is_tokens):
        """Convert each group's host output with this step's row limit."""
        groups = list(host_groups) if isinstance(host_groups, (list, tuple)) else [host_groups]
        per = max(1, int(rows) // max(1, len(groups)))
        parts = []
        for model, group in zip(self.model, groups):
            tensor = group[0] if isinstance(group, tuple) else group
            parts.append(model.process_output_decode(tensor, per, S=1, is_tokens=is_tokens))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def _spec_resident_decode_declared(self):
        """Whether the runner keeps this adapter's plain-decode inputs device-resident.

        ``TTAsyncDecodeController.plan_decode_reload`` reads the instance's
        ``supports_async_decode`` and, when it is true, sends a steady
        device-sampled traced decode ``reload_inputs=False`` whatever the
        scheduling mode: the declaration selects the resident mode, and async
        scheduling only adds overlap on top. The instance dict is read because
        the base narrows it (PLI models) after the platform's class snapshot.
        """
        return bool(self.model_capabilities.get("supports_async_decode", False))

    def _spec_plain_decode(self, args, kwargs, page_tables_per_layer, leaves_inputs_stale=False):
        """Run the plain decode; refuse resident inputs a drafter-served step left stale.

        The traced decode keeps each row's token and position on the device and
        reads them from the host only under ``reload_inputs``. A row answered
        from the drafter session, or run at position -1 beside a peer
        (``leaves_inputs_stale``), left them behind. The runner reloads the
        first ordinary decode after a verify (``plan_decode_reload``: a
        verification followed by an ordinary decode is a transition) and
        every decode of an adapter without ``supports_async_decode``; a
        caller that sends ``reload_inputs=False`` over stale inputs gets an
        error here instead of a decode from the wrong token.
        """
        if self._spec_resident_inputs_stale and not kwargs["reload_inputs"]:
            raise RuntimeError(
                f"{type(self).__name__}: a drafter-served step left the plain decode's "
                "device-resident tokens and positions stale, and this decode was sent "
                "reload_inputs=False. The runner reloads the inputs of the first ordinary decode after a "
                "verify and of every decode of an adapter without supports_async_decode; an ordinary "
                "step the drafter answered is not a transition the runner sees (contract s.4d leaves "
                "that reload to the model, whose resident buffers can only be refilled from host inputs "
                "the same contract declares stale under reload_inputs=False), which is why a resident "
                "adapter answers no ordinary step from the drafter."
            )
        tt_out = Gemma4ForCausalLM.decode_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        self._spec_resident_inputs_stale = leaves_inputs_stale
        return tt_out

    def _spec_plain_host(self, args, kwargs, page_tables_per_layer, rows, exclude_rows, is_tokens):
        """The plain decode's host output over column 0, ids or logits per ``is_tokens``."""
        args, kwargs = self._spec_column_zero(args, kwargs, rows, exclude_rows)
        tt_out = self._spec_plain_decode(args, kwargs, page_tables_per_layer, leaves_inputs_stale=bool(exclude_rows))
        if self._spec_is_host(tt_out):
            host = tt_out[0] if isinstance(tt_out, tuple) else tt_out
        else:
            host = self._spec_convert(
                Gemma4ForCausalLM.read_decode_output(self, tt_out, async_read=False), rows, is_tokens=is_tokens
            )
        return host if isinstance(host, torch.Tensor) else torch.as_tensor(host)

    def _spec_plain_ids(self, args, kwargs, page_tables_per_layer, rows, exclude_rows=None):
        """Each row's next-token id from the plain decode, int32 ``[rows]``.

        Device-sampled ids are used as they are: each is the row's own draw,
        which is the argmax for a greedy row. The caller supplies device
        sampling parameters; a host-sampled step keeps its logits outside
        this path (``_spec_plain_logits``).
        """
        host = self._spec_plain_host(args, kwargs, page_tables_per_layer, rows, exclude_rows, is_tokens=True)
        ids = host.reshape(-1)
        out = torch.zeros(int(rows), dtype=torch.int32)
        n = min(int(rows), int(ids.numel()))
        out[:n] = ids.reshape(-1)[:n].to(torch.int32)
        return out

    def _spec_plain_logits(self, args, kwargs, page_tables_per_layer, rows, exclude_rows=None):
        """Each row's next-token logits from the plain decode, float ``[rows, V]``.

        The decode runs without device sampling (``sampling_params`` None), the
        path a host-sampled step takes, so its output is the ``[1, 1, rows, V]``
        decode logits the base reader converts per row.
        """
        kwargs = dict(kwargs)
        kwargs["sampling_params"] = None
        host = self._spec_plain_host(args, kwargs, page_tables_per_layer, rows, exclude_rows, is_tokens=False)
        if host.dim() < 2:
            raise ValueError(
                f"{type(self).__name__}: the plain decode returned {tuple(host.shape)} where "
                f"[{rows}, 1, V] logits were expected"
            )
        vocab = int(host.shape[-1])
        logits = host.reshape(-1, vocab)
        if int(logits.shape[0]) < int(rows):
            raise ValueError(
                f"{type(self).__name__}: the plain decode returned logits for {int(logits.shape[0])} rows, "
                f"the step has {rows}"
            )
        return logits[: int(rows)]

    # -- DraftOutput -------------------------------------------------------------
    @staticmethod
    def _spec_draft_output(draft_token_ids, num_valid):
        """A ``DraftOutput``, carrying per-row counts when the plugin has them."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        if "num_valid" in getattr(DraftOutput, "__dataclass_fields__", {}):
            return DraftOutput(draft_token_ids=draft_token_ids, num_valid=num_valid)
        return DraftOutput(draft_token_ids=draft_token_ids)

    @classmethod
    def _spec_decline(cls, rows, k):
        """A proposal that offers no row a draft.

        The ids are always ``[rows, K]`` in-vocabulary values, so the tensor
        cannot say "nothing this step"; ``num_valid`` 0 does, and the runner
        records nothing for such a row.
        """
        return cls._spec_draft_output(
            torch.zeros((int(rows), int(k)), dtype=torch.int32),
            torch.zeros(int(rows), dtype=torch.int32),
        )

    def read_decode_output(self, tt_out, async_read=False, *_, **__):
        # A step the drafter answered returns host tokens from decode_forward:
        # nothing to read, pass them straight through (no events). Every other
        # output is a DEVICE tensor from the plain decode; route it to the base
        # reader (events under async_read) so the deferred pipeline reads it.
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *_, **__)


class Gemma4DFlashContractForCausalLM(Gemma4SpecContractBase, Gemma4DFlashBase):
    """dFlash speculation driven by the plugin's speculative-decoding contract.

    The fused decoder drafts and verifies in one device replay; this class
    hands it to the runner one iteration at a time: ``output_tokens_per_step``
    is 1, the runner builds the candidate block from the drafts this class
    proposed and owns the accept walk.

    Steps for one greedy request that decodes alone:

    1. ``prefill_forward`` runs the prompt eagerly and keeps the residual taps
       of the drafter's target layers as ``_spec_pending``. A prompt split
       across scheduler steps accumulates its taps over those calls.
    2. The first ``decode_forward`` arrives as an ordinary step (narrow
       decode). ``decode_forward`` seeds the fused decoder from the taps
       (``_spec_bootstrap``), runs one fused replay, applies the caller's
       sampling-state commands, and returns the replay's first posterior id.
    3. ``propose_draft_tokens`` commits the runner's accepted count and anchor,
       refreshes the verify page tables, runs one fused replay and keeps its
       drafts and posterior as a :class:`RetainedProposal`.
    4. The next ``decode_forward`` carries those drafts (``spec_mode`` set).
       ``decode_forward`` checks the block against the retained proposal,
       applies the caller's sampling-state commands, and answers from the
       posterior without another target or drafter forward pass.
    5. ``propose_draft_tokens`` and ``decode_forward`` repeat steps 3 and 4:
       one fused replay per step, never a second prefill.

    Speculation ends, and does not resume, when another request shares a
    step with the owner, when the owner was prefilled together with another
    request, when an ordinary step uses host sampling, or when a proposal declines
    (see ``propose_draft_tokens``). The row then continues as plain decode,
    which is what the block rail does at batch size above one. When device
    sampling parameters are supplied, a row must pass ``_spec_row_can_use_argmax``
    before it can start or continue a draftless session. The session answers
    with target argmax ids; plain decode samples from the requested distribution.
    Verification requires device sampling parameters and cannot return logprobs.
    The runner must send host-sampled requests through ordinary decode, because
    this adapter receives neither their sampling controls nor their logits processors.
    A step the owner is not part of leaves its session alone: the scheduler keeps the
    drafts of a request it did not schedule and sends them on that request's
    next step, which the retained proposal must still answer.

    Identity: the request that owns the pending taps or the live session is
    known by its state slot (``empty_slots`` at prefill, moved by
    ``note_state_slots_moved``, closed by ``release_request``) together with
    a per-slot generation count (:class:`Gemma4SpecContractBase`), so a
    reused slot never matches a stale owner. The fused decoder addresses
    batch row 0 (``page_table[:1]`` in ``_spec_bootstrap``, ``lpt[0]`` in
    ``refresh_page_tables``), so only a request decoding at row 0 is
    bootstrapped.

    Block tables: vLLM's ``BlockTable.add_row`` and ``move_row`` write only a
    row's current block count, so the plugin's fixed-width slice can carry a
    previous occupant's block ids in the columns past a prompt's last block.
    Every prefill on this rail is eager and the eager paged fill stops at the
    valid length, so those columns are never addressed here;
    ``_dflash_mask_prefill_tables`` zeroes them anyway so a traced chunk
    cannot write another request's blocks.
    """

    _SPEC_CONTRACT_K = int(os.environ.get("GEMMA4_DFLASH_VERIFY", "5"))
    # The inherited width ladder, ring-headroom warning and generation budget
    # read these; on this rail the verify count is the contract K.
    _SPEC_V = _SPEC_CONTRACT_K
    _SPEC_N = _SPEC_CONTRACT_K + 1
    # This rail stops proposing before a verify would wrap the ring and carries
    # on as plain decode (``_dflash_ring_advice``), so an exact ring costs it
    # speed, not correctness. The inherited startup guard must not refuse it.
    _SPEC_DECLINES_AT_RING_WRAP = True

    model_capabilities = {
        **Gemma4ForCausalLM.model_capabilities,
        # Pinned off: the drafter is seeded from the residual taps of the
        # prompt's prefill forward, and a prefix-cache hit prefills only the
        # uncached suffix, so its taps would not cover the context.
        "supports_prefix_caching": False,
        "output_tokens_per_step": 1,
        # A step this class does not speculate on runs the plain decode; the
        # device sampler keeps that step from pulling [B, vocab] logits to host.
        "supports_sample_on_device": True,
        # The TT sampler represents at most 32 candidates. Use host sampling
        # for sampled contract requests that ask for a wider top_k.
        "max_device_top_k": 32,
        # Keep penalty processing on vLLM's per-request host sampling path
        # when contract verification mixes greedy and sampled rows.
        "supports_device_penalties": False,
        # TTAsyncDecodeController.plan_decode_reload sends reload_inputs=True
        # when supports_async_decode is False, including after fused output, and
        # _spec_plain_decode refuses a decode after fused output without it.
        # Keep async decode disabled until the runner tracks fused input residency.
        "supports_async_decode": False,
        "supports_async_spec_decode": False,
        "supports_spec_decode": True,
        # Drafts come from the device and the target hidden state never leaves
        # it, so the runner holds no handle and ``hidden`` is unused.
        "spec_requirements": ("device_propose",),
        "spec_hidden_handoff": ("on_device",),
    }

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Admit any ``max_num_seqs``: speculation is decided per step.

        The block rail's plan refuses ``max_num_seqs > 1`` because the fused
        verify packs one request's candidates into one batch row. That is a
        statement about a step, not about the server: a step with several live
        rows proposes nothing and runs the plain decode. The plan's arithmetic
        (effective_k, lanes, byte accounting) is inherited so it exists once.
        """
        del vllm_config, max_num_seqs  # see Gemma4DFlashBase._spec_plan_one_row
        from vllm_tt_plugin.spec_decode import SpecReject

        verify = cls._SPEC_CONTRACT_K
        if os.environ.get("GEMMA4_DFLASH_PACKED", "1") != "1":
            return SpecReject(
                reason=(
                    "Gemma4 dFlash on the speculative contract needs packed verification "
                    "(GEMMA4_DFLASH_PACKED=1): only the packed body writes effective_k + 1 KV "
                    "rows per step, which is the lookahead the runner reserves."
                ),
                supported_k=(),
            )
        for env in ("GEMMA4_DFLASH_WIDTH_SET", "GEMMA4_DFLASH_WARMUP_DECODE"):
            if not _dflash_env_flag(env, "1"):
                return SpecReject(
                    reason=(
                        f"Gemma4 dFlash on the speculative contract needs {env}=1: the verify "
                        "widths are prepared and captured in warmup, before ordinary trace "
                        "capture, and never during serving."
                    ),
                    supported_k=(verify,),
                )
        # A step with no draft anywhere runs at the plain decode's [B, 1] shape.
        return cls._spec_plan_one_row(requested_k, supports_narrow_decode=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dflash_init_state()
        self._dflash_ring_advice()
        logger.info(
            f"Gemma4DFlash speculative contract: K={self._SPEC_CONTRACT_K}, one fused replay per "
            "solo step, output_tokens_per_step=1"
        )

    def _dflash_init_state(self):
        self._spec_init_state()
        self._dflash_retained = None
        self._dflash_pending_owner = None
        self._dflash_live_owner = None
        self._dflash_owner_tables = None
        self._dflash_last_pt = None
        self._dflash_steps = 0
        self._dflash_committed = 0
        # The live session's acceptance statistics (``spec_policy.AcceptanceStats``): opened from the batch
        # prior at bootstrap, fed by the accepted counts, dropped with the session.
        self._dflash_stats = None
        # The slots the last decode step served and the slots prefilled since it (released ones removed):
        # what a solo prefill checks before capturing taps (``_dflash_taps_cannot_pay``).
        self._dflash_live_slots = set()
        self._dflash_prefilled_slots = set()

    # -- warmup: the policy's cost table -------------------------------------------
    def warmup_model_decode(self, *args, **kwargs):
        """The base's buckets and the width set (``Gemma4DFlashBase``), then, on the traced pass, the
        adaptive policy's cost table (``_dflash_measure_costs``)."""
        out = super().warmup_model_decode(*args, **kwargs)
        if kwargs.get("enable_trace") and self._spec_width_set and self._spec_decoder is not None:
            self._dflash_measure_costs(kwargs)
        return out

    def _dflash_measure_costs(self, warmup_kwargs):
        """Time every captured verify width and the plain step; the policy's cost table.

        Runs once, right after the width-set capture, on the capture's own
        inputs (``DFlashFusedDecoder.measure_widths``: anchor 1 at the first
        and the last position of every width through the scratch table,
        block 0). The plain step is ``GEMMA4_SPEC_PLAIN_STEP_MS`` when set,
        else timed traced replays of the base decode on the decode warmup's
        zero table at position 0 and at the top position
        (``_spec_time_plain_step``). With ``GEMMA4_SPEC_ADAPTIVE=0`` nothing
        is measured. This rail has one record, ``(1, K)``, per width.
        """
        if not self._spec_policy.config.adaptive:
            logger.info(f"Gemma4DFlash adaptive K: off ({SPEC_ADAPTIVE_ENV}=0); every solo greedy request speculates")
            return
        dec = self._spec_decoder
        if not any(record.get("trace") is not None for record in dec._pv_widths.values()):
            raise RuntimeError("Gemma4DFlash adaptive K: the cost table is measured after the width-set capture")
        k = int(self._SPEC_CONTRACT_K)
        top = self._spec_top_position()
        t0 = time.perf_counter()
        costs = {("step", (1, k), int(width)): CostLine.of(line) for width, line in dec.measure_widths(top).items()}
        max_b = int(warmup_kwargs.get("max_batch_size") or self.model_args[0].max_batch_size)
        num_blocks = warmup_kwargs.get("num_blocks")
        blocks = int(num_blocks) if num_blocks else int(dec.page_table_torch.shape[-1])
        plain = self._spec_plain_step_cost(warmup_kwargs, torch.zeros(max_b, blocks, dtype=torch.int32), top)
        self._spec_costs = costs
        self._spec_plain_cost = plain
        timings = 2 * len(costs) + (0 if self._spec_policy.config.plain_step_ms is not None else 2 * len(plain))
        self._spec_log_cost_table("Gemma4DFlash", costs, plain, timings, time.perf_counter() - t0)

    # -- the policy on this rail: speculate or not, per step ---------------------------
    def _dflash_policy_candidate(self, position, entering):
        """This rail's one record, ``(1, K)``, on the width ``position`` selects, costed at ``position``; the
        bootstrap replay is the entry cost when ``entering``. None when no captured width covers the position
        or the table has no line for that width."""
        width = self._spec_decoder.width_for(int(position))
        if width is None:
            return None
        shape = (1, int(self._SPEC_CONTRACT_K))
        cost = self._spec_record_ms(shape, width, position)
        if cost is None:
            return None
        # The bootstrap (ingest and first replay) is the entry: the table has no line for it, so its prior
        # is one step of the table; serving measures it (``_dflash_bootstrap``).
        table = self._spec_record_table_ms(shape, width, position)
        entry = self._spec_reseed_ms(shape, position, prior=table) if entering else 0.0
        measured = self._spec_policy.calibration.tokens_per_row(("step", shape, int(width)))
        return RecordCandidate(shape, int(width), cost, entry, tokens_per_row=measured)

    def _dflash_policy_admits(self, position):
        """Whether a session may start at ``position``: the batch prior's expected tokens pay for the replays
        over the entry horizon, or an amortized retry is due. Without a cost table every solo greedy request
        speculates, as before the policy."""
        if not self._spec_cost_table_ready("Gemma4DFlash"):
            return True
        candidate = self._dflash_policy_candidate(position, entering=True)
        if candidate is None:
            return True
        policy = self._spec_policy
        plain_ms, _bucket = self._spec_plain_ms_for(1, position)
        # The bootstrap is the request's one trial: a decline ends this rail's session for good.
        if policy.would_speculate([candidate], [policy.prior], plain_ms, policy.request_steps, fresh=True):
            return True
        self._spec_policy_note("Gemma4DFlash", None, policy.last_k, 1, str(candidate.shape), [policy.prior], plain_ms)
        return False

    def _dflash_choose(self, position):
        """Whether this proposal replays; False ends the session for the plain step (``propose_draft_tokens``)."""
        if not self._spec_cost_table_ready("Gemma4DFlash"):
            return True
        candidate = self._dflash_policy_candidate(position, entering=False)
        if candidate is None:
            return True
        policy = self._spec_policy
        plain_ms, _bucket = self._spec_plain_ms_for(1, position)
        previous = policy.last_k
        stats = [self._dflash_stats]
        remaining = None
        if policy.request_steps is not None:
            remaining = max(1, int(round(policy.request_steps)) - int(self._dflash_steps))
        fresh = self._dflash_stats is not None and self._dflash_stats.spec_steps == 0  # no verify yet: the trial
        choice = policy.decide([candidate], stats, plain_ms, int(self._SPEC_CONTRACT_K), remaining, fresh=fresh)
        self._spec_policy_note("Gemma4DFlash", choice, previous, 1, str(candidate.shape), stats, plain_ms)
        return choice is not None

    def _dflash_ring_advice(self):
        self._spec_ring_advice(self._SPEC_CONTRACT_K + 1, "Gemma4DFlash speculative contract")

    # The dFlash host tests drive these drafter-agnostic helpers under the
    # names this class had before they moved to Gemma4SpecContractBase.
    _dflash_live_rows = staticmethod(Gemma4SpecContractBase._spec_live_rows)

    @classmethod
    def _dflash_row_can_use_argmax(cls, sampling_params, row, grammar_bitmask=None):
        return cls._spec_row_can_use_argmax(sampling_params, row, grammar_bitmask)

    @staticmethod
    def _dflash_column_zero(args, kwargs, rows, exclude_row=None):
        return Gemma4SpecContractBase._spec_column_zero(
            args, kwargs, rows, None if exclude_row is None else [exclude_row]
        )

    # -- identity: the base keys owners on slot plus generation ---------------
    def _spec_follow_slot_moves(self, moved):
        def follow(owner):
            if owner is None or owner[0] not in moved:
                return owner
            return (moved[owner[0]], owner[1])

        self._dflash_pending_owner = follow(self._dflash_pending_owner)
        self._dflash_live_owner = follow(self._dflash_live_owner)
        self._spec_owner_slot = self._dflash_live_owner[0] if self._dflash_live_owner else None
        self._dflash_live_slots = {moved.get(s, s) for s in self._dflash_live_slots}
        self._dflash_prefilled_slots = {moved.get(s, s) for s in self._dflash_prefilled_slots}

    def _spec_release_slot(self, row):
        pending_mine = self._spec_owns(self._dflash_pending_owner, row)
        live_mine = self._spec_owns(self._dflash_live_owner, row)
        if pending_mine:
            self._dflash_drop_pending()
        if live_mine:
            self._dflash_end_session()
        self._dflash_live_slots.discard(int(row))
        self._dflash_prefilled_slots.discard(int(row))

    def _spec_release_all(self):
        self._dflash_drop_pending()
        self._dflash_end_session()
        self._dflash_live_slots = set()
        self._dflash_prefilled_slots = set()

    def release_persistent_capture(self) -> None:
        if self._spec_last_propose is not None:
            self._spec_log_stats("Gemma4DFlash", final=True)
        self._dflash_retained = None
        self._dflash_live_owner = None
        self._dflash_pending_owner = None
        self._dflash_owner_tables = None
        self._dflash_stats = None
        self._spec_costs = None
        self._spec_plain_cost = None
        super().release_persistent_capture()

    # -- session state ---------------------------------------------------------
    def _dflash_drop_pending(self):
        pending = self._spec_pending
        self._spec_pending = None
        self._dflash_pending_owner = None
        if pending is None:
            return
        for tap in pending[0]:
            try:
                tap.deallocate(True)
            except Exception:
                pass

    def _dflash_end_session(self):
        if self._dflash_steps:
            # One line per session, comparable with the block rail's decode
            # summary: every step of the session counts, the first ordinary
            # step included, and tokens are what the runner committed.
            self._spec_policy.note_request_steps(self._dflash_steps)
            stats = self._dflash_stats
            accepted = (
                f"{stats.ewma_accepted:.2f} accepted drafts/verify (EWMA over {stats.spec_steps})"
                if stats is not None and stats.ewma_accepted is not None
                else "no verified drafts"
            )
            logger.info(
                f"Gemma4DFlash contract summary: {self._dflash_steps} steps, {self._dflash_committed} tokens, "
                f"{self._dflash_committed / self._dflash_steps:.2f} tokens/step, {accepted}"
            )
        self._dflash_steps = 0
        self._dflash_committed = 0
        self._dflash_stats = None
        self._dflash_retained = None
        self._dflash_live_owner = None
        self._dflash_owner_tables = None
        self._dflash_last_pt = None
        self._spec_owner_slot = None
        if self._spec_active:
            self._spec_release_decoder()
        self._spec_active = False

    def _dflash_owner_in(self, slots):
        return any(self._spec_owns(self._dflash_live_owner, slot) for slot in slots)

    def _dflash_leave_row(self, slot, why):
        """Drop ``slot``'s own pending taps and end its own session.

        Another request's taps or session stay: that request is not in this
        step and may still decode alone.
        """
        pending = self._spec_owns(self._dflash_pending_owner, slot)
        live = self._spec_owns(self._dflash_live_owner, slot)
        if pending or live:
            logger.info(f"Gemma4DFlash speculative contract: leaving speculation for slot {slot} ({why})")
        if pending:
            self._dflash_drop_pending()
        if live:
            self._dflash_end_session()

    def _dflash_leave_speculation(self, why, slots=None):
        """Drop the pending taps and end the live session.

        ``slots`` are the state slots this step serves. When given and none of
        them owns the live session, the session is kept for the owner's next
        step (see the class docstring).
        """
        keep = self._spec_active and slots is not None and not self._dflash_owner_in(slots)
        if self._spec_pending is not None or (self._spec_active and not keep):
            logger.info(f"Gemma4DFlash speculative contract: leaving speculation ({why})")
        self._dflash_drop_pending()
        if not keep:
            self._dflash_end_session()
        self._dflash_disarm_taps()

    def _dflash_disarm_taps(self):
        try:
            self.model[0].dflash_capture_taps(None)
        except Exception:
            pass

    def _dflash_note_step(self, kwargs, page_tables_per_layer):
        """Record a step the drafter session answered.

        Keeps the owner's tables for the next page-table refresh. The answered
        row skipped the plain decode, so that decode's device-resident token
        and position for the row are stale (``_spec_plain_decode``).
        """
        self._dflash_owner_tables = (kwargs.get("page_table"), page_tables_per_layer, kwargs.get("kv_cache"))
        self._spec_note_served_step()

    # -- prefill: capture taps, accumulate over scheduler chunks --------------
    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        if args:
            kwargs["tokens"] = args[0]
            args = args[1:]
        self._spec_note_prefill(kwargs.get("empty_slots"))
        tokens = kwargs.get("tokens")
        rows = int(tokens.shape[0]) if tokens is not None else 1
        prompt_lens = kwargs.get("prompt_lens")
        end = int(prompt_lens[0]) if prompt_lens is not None else (int(tokens.shape[1]) if tokens is not None else 0)
        start_pos = kwargs.get("start_pos")
        cached = int(list(start_pos)[0]) if start_pos is not None else 0
        kwargs["page_table"], page_tables_per_layer = self._dflash_mask_prefill_tables(
            kwargs.get("page_table"), page_tables_per_layer, prompt_lens, tokens, kwargs.get("kv_cache")
        )
        # Every prefill on this rail is eager: the residual taps come from a
        # python hook that a traced replay skips, and a plain prefill must not
        # run traced next to an eager one against the same persistent buffers.
        # GEMMA4_DFLASH_TRACE_PLAIN_PREFILL=1 (experimental) lets a prefill that
        # takes no taps keep the trace setting the caller passed.
        requested_trace = kwargs.get("enable_trace", True)
        kwargs["enable_trace"] = False
        is_warmup = bool(kwargs.get("warmup_prefill")) or (tokens is not None and int(tokens.abs().sum()) == 0)
        if is_warmup:
            self._dflash_disarm_taps()
            return Gemma4ForCausalLM.prefill_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        slot = _spec_first_slot(kwargs.get("empty_slots"))
        prefilled_others = self._dflash_prefilled_slots - {int(slot)} if slot is not None else set()
        self._dflash_prefilled_slots |= {int(s) for s in (kwargs.get("empty_slots") or [])}
        continues = (
            cached > 0
            and self._spec_pending is not None
            and self._spec_owns(self._dflash_pending_owner, slot)
            and int(self._spec_pending[1]) == cached
        )
        if not continues:
            self._dflash_drop_pending()
        speculable = rows == 1 and self._dflash_prompt_speculable(end)
        if speculable and cached > 0 and not continues:
            # Another prefill replaced this request's earlier taps. This chunk's
            # taps alone would seed the drafter as if they were the whole prompt.
            speculable = False
        if speculable and cached > 0 and align_num_cached_tokens_to_sdpa([cached])[0] != cached:
            # The generator re-prefills the tokens between the aligned start and
            # the chunk start, so their taps would appear twice and shift every
            # later row of the drafter's window.
            speculable = False
        if speculable and cached > 0 and self._dflash_mm_restart_moves(kwargs, tokens, cached):
            # Same duplication through the multimodal restart: the base moves a
            # continuation's restart to the start of a vision item it cuts.
            speculable = False
        if speculable and not continues:
            why = self._dflash_taps_cannot_pay(slot, end, prefilled_others)
            if why is not None:
                logger.debug(f"Gemma4DFlash speculative contract: no taps for slot {slot} at {end}: {why}")
                speculable = False
        if not speculable:
            self._dflash_drop_pending()
            self._dflash_disarm_taps()
            if requested_trace and _dflash_env_flag("GEMMA4_DFLASH_TRACE_PLAIN_PREFILL", "0"):
                kwargs["enable_trace"] = requested_trace
            return Gemma4ForCausalLM.prefill_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        if not continues:
            self._dflash_pending_owner = self._spec_open_owner(slot)
        drafter = self._spec_get_drafter()
        model0 = self.model[0]
        model0.dflash_capture_taps(drafter.target_layer_ids, keep_last=12)
        try:
            out = Gemma4ForCausalLM.prefill_forward(self, *args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        finally:
            taps = model0.pop_dflash_taps()
            model0.dflash_capture_taps(None)
        previous = list(self._spec_pending[0]) if continues and self._spec_pending is not None else []
        taps = self._dflash_trim_taps(previous + list(taps), len(drafter.target_layer_ids))
        self._spec_pending = (taps, end)
        self._spec_active = self._spec_active and self._dflash_live_owner is not None
        return out

    def _dflash_mm_restart_moves(self, kwargs, tokens, cached):
        """Whether the base prefill restarts this continuation below ``cached``.

        ``Gemma4ForCausalLM.prefill_forward`` moves an SDPA-aligned restart
        that cuts an image or video placeholder run to that run's start
        (``gemma4_mm_restart_outside_vision_items``) and re-prefills the rows
        between. Their taps would be captured a second time, so such a chunk
        is served without speculation, like an unaligned one.
        """
        if not self.is_multimodal:
            return False
        vision_ids = [t for t in (self.image_token_id, getattr(self, "video_token_id", None)) if t is not None]
        if not vision_ids:
            return False
        offsets = _per_slot_items(kwargs, "mm_placeholder_offsets", 1)
        lengths = _per_slot_items(kwargs, "mm_placeholder_lengths", 1)
        moved = gemma4_mm_restart_outside_vision_items([int(cached)], tokens, offsets, lengths, vision_ids)
        return int(moved[0]) != int(cached)

    def _dflash_taps_cannot_pay(self, slot, position, prefilled_others):
        """Why a solo prefill's taps could not bootstrap a session, or None when they may. One session serves an
        engine and it ends when a peer shares a step, so taps pay only for a row that decodes alone from its
        first step: not while a session is live on another row, not when other rows were prefilled since the
        last decode step (``prefilled_others``: they share this row's first step), not while rows the last
        decode step served are still live (``_dflash_live_slots`` less the released ones), and not when the
        batch prior declines a bootstrap at ``position`` (``_dflash_policy_admits``: the entry band or the
        trial floor). Before this gate every solo prefill captured and trimmed its taps and most were dropped
        at the row's first step (the c8 and c32 points: one bootstrap per wave, 7-31 captures)."""
        owner = self._dflash_live_owner
        if self._spec_active and owner is not None and not self._spec_owns(owner, slot):
            return "a session is live on another row"
        if prefilled_others:
            return f"slots {sorted(prefilled_others)} prefilled since the last step share its first step"
        live_others = self._dflash_live_slots - ({int(slot)} if slot is not None else set())
        if live_others:
            return f"slots {sorted(live_others)} the last step served are still live"
        if not self._dflash_policy_admits(position):
            return "the batch prior declines a bootstrap"
        return None

    def _dflash_prompt_speculable(self, prompt_len):
        limit = int(os.environ.get("GEMMA4_DFLASH_MAX_SPEC_ISL", "0"))
        if limit > 0 and int(prompt_len) > limit:
            logger.info(
                f"Gemma4DFlash speculative contract: prompt {prompt_len} > GEMMA4_DFLASH_MAX_SPEC_ISL "
                f"{limit}; serving as plain decode"
            )
            return False
        return True

    def _dflash_trim_taps(self, taps, per_group):
        """Keep the newest tap groups that cover the drafter's context window.

        One group per prefill forward, ``per_group`` tensors each. Groups
        wholly below the last ``ctx_cap`` positions never reach the mirror
        (``prefill_ingest`` slices the window from the end), so they are freed
        here instead of accumulating across scheduler chunks.
        """
        if per_group <= 0 or len(taps) % per_group:
            return list(taps)
        cap = int(getattr(self._spec_decoder, "cap", 0) or os.environ.get("GEMMA4_DFLASH_CTX_CAP", "2048"))
        groups = [taps[i : i + per_group] for i in range(0, len(taps), per_group)]
        kept, rows = [], 0
        for group in reversed(groups):
            if rows >= cap:
                for tap in group:
                    try:
                        tap.deallocate(True)
                    except Exception:
                        pass
                continue
            kept.append(group)
            rows += int(group[0].shape[2])
        return [tap for group in reversed(kept) for tap in group]

    def _dflash_mask_prefill_tables(self, page_table, page_tables_per_layer, prompt_lens, tokens, kv_cache):
        """Zero block-table columns past each prompt's last block.

        vLLM writes only a row's current block count, so the plugin's
        fixed-width slice can carry a previous occupant's block ids past that
        count. A fill that ran to the padded bucket would write through them
        into blocks another request may own. Bounded sliding layers address
        ring slots and are left alone.
        """
        if kv_cache is None or tokens is None:
            return page_table, page_tables_per_layer
        try:
            block_size = int(self._effective_paged_block_size(kv_cache) or 0)
        except Exception:
            block_size = 0
        if block_size <= 0:
            return page_table, page_tables_per_layer
        rows = int(tokens.shape[0])
        if prompt_lens is not None:
            ends = [int(n) for n in list(prompt_lens)][:rows]
        else:
            ends = [int(tokens.shape[1])] * rows
        first_free = [(end + block_size - 1) // block_size for end in ends]

        def mask(table):
            if not isinstance(table, torch.Tensor) or table.dim() != 2:
                return table
            table = table.clone()
            for row, col in enumerate(first_free):
                if row < int(table.shape[0]) and col < int(table.shape[1]):
                    table[row, col:] = 0
            if int(table.shape[0]) > len(first_free):
                table[len(first_free) :, :] = 0
            return table

        page_table = mask(page_table)
        if page_tables_per_layer:
            ring = set(self._sliding_layer_indices()) if self._bounded_sliding_kv_cache else set()
            page_tables_per_layer = [
                table if index in ring else mask(table) for index, table in enumerate(page_tables_per_layer)
            ]
        return page_table, page_tables_per_layer

    # -- decode ----------------------------------------------------------------
    def decode_forward(
        self,
        *args,
        page_tables_per_layer=None,
        reload_inputs: bool,
        reload_page_table: bool,
        reload_sampling_params: bool,
        reset_sampling_state: bool,
        **kwargs,
    ):
        kwargs.update(
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
        )
        spec_mode = kwargs.pop("spec_mode", None)
        self._spec_boundary("decode_start", spec_mode is not None)
        num_valid = kwargs.pop("num_valid_drafts", None)
        kwargs.pop("accepted_counts", None)  # consumed by propose, where the commit is
        kwargs.pop("draft_token_ids", None)
        tokens, start_pos = self._spec_inputs(args, kwargs)
        rows = self._spec_rows(tokens)
        live = self._spec_live_rows(start_pos, rows)
        self._dflash_live_slots = {int(self._spec_slot_of(r, kwargs)) for r in live}
        self._dflash_prefilled_slots = set()
        try:
            if spec_mode is None:
                return self._dflash_ordinary_step(args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live)
            sampling = kwargs.get("sampling_params")
            if sampling is None:
                raise ValueError(
                    "Gemma4 dFlash argmax verification cannot apply host sampling controls. "
                    "Route host-sampled requests through ordinary decode with a plugin that "
                    "supports sampled-step fallback, or disable speculative decoding."
                )
            if kwargs.get("grammar_bitmask") is not None:
                raise ValueError(
                    "Gemma4 dFlash argmax verification cannot apply a grammar bitmask. "
                    "Disable speculative decoding for structured-output requests."
                )
            if any(self._spec_row_requests_logprobs(sampling, row) for row in live):
                raise ValueError(
                    "Gemma4 dFlash argmax verification cannot return requested logprobs. "
                    "Disable speculative decoding for requests that require logprobs."
                )
            return self._dflash_verify_step(
                args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live, num_valid
            )
        finally:
            self._spec_boundary("decode_end", spec_mode is not None)

    def _dflash_ordinary_step(self, args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live):
        """An ordinary decode step: the runner asked for no verification.

        With one live greedy row this is where speculation starts (from
        pending taps) or continues (a retained proposal the scheduler sent no
        drafts for). Both answer with a fused replay's first posterior id, the
        target's greedy choice after the row's token, in the flat int32 form
        the device sampler produces. Every other case is the plain decode.
        """

        def plain():
            return self._spec_plain_decode(args, kwargs, page_tables_per_layer)

        if len(live) != 1:
            self._dflash_leave_speculation("several live rows", [self._spec_slot_of(r, kwargs) for r in live])
            return plain()
        row = live[0]
        sampling = kwargs.get("sampling_params")
        if sampling is None:
            # Host sampling reads logits; the fused replay yields ids only.
            self._dflash_leave_row(self._spec_slot_of(row, kwargs), "host sampling")
            return plain()
        if not self._spec_row_can_use_argmax(sampling, row, kwargs.get("grammar_bitmask")):
            self._dflash_leave_row(
                self._spec_slot_of(row, kwargs), "argmax IDs cannot satisfy the row's sampling output"
            )
            return plain()
        token = self._dflash_solo_next_token(row, tokens, start_pos, rows, kwargs, page_tables_per_layer)
        if token is None:
            return plain()
        self._spec_prepare_sampling_state(kwargs, start_pos, rows)
        out = torch.zeros(int(rows), dtype=torch.int32)
        out[row] = int(token)
        return out

    def _dflash_solo_next_token(self, row, tokens, start_pos, rows, kwargs, page_tables_per_layer):
        """The one live row's next token from the drafter session, or None.

        None means the row cannot speculate on this step and the caller runs
        the plain decode; the session state is already cleaned up for that.
        """
        anchor = self._spec_row(tokens, rows, row)[0]
        position = self._spec_row(start_pos, rows, row)[0] if start_pos is not None else None
        slot = self._spec_slot_of(row, kwargs)
        if self._spec_active:
            if not self._spec_owns(self._dflash_live_owner, slot):
                # Another request's session; its owner is not in this step and
                # keeps it. This row's taps cannot bootstrap past it.
                self._dflash_drop_pending()
                return None
            retained = self._dflash_retained
            if (
                retained is not None
                and not retained.consumed
                and retained.anchor_token == anchor
                and (position is None or retained.anchor_position == position)
            ):
                # The scheduler sent no drafts for this proposal (it drops
                # them near max_model_len). The replay's first id is still
                # the answer, and the next propose commits one token.
                retained.consumed = True
                self._dflash_note_step(kwargs, page_tables_per_layer)
                return retained.posterior[0]
            self._dflash_end_session()
            return None
        if self._spec_pending is None:
            return None
        if not self._spec_owns(self._dflash_pending_owner, slot):
            # Taps of a request that never decoded alone (finished, preempted).
            self._dflash_drop_pending()
            return None
        if row != 0 or position is None or not self._dflash_can_bootstrap(position, kwargs.get("page_table")):
            self._dflash_drop_pending()
            return None
        if not self._dflash_policy_admits(position):
            # The batch prior says the replay would not pay: the request runs plain from its first
            # step (the taps are dropped; this rail cannot start a session later without them).
            self._dflash_drop_pending()
            return None
        self._dflash_bootstrap(row, anchor, position, kwargs, page_tables_per_layer)
        _drafts, posterior = self._spec_decoder.contract_replay(first=self._spec_first_step)
        self._spec_first_step = False
        return int(posterior[0])

    def _dflash_can_bootstrap(self, position, page_table):
        dec = self._spec_decoder
        if dec is None or page_table is None or not self._spec_width_set:
            return False
        # ``_spec_bootstrap`` captures a per-session trace when no prepared
        # width covers the position. That capture writes through an all-zero
        # scratch page table, and under bounded sliding block 0 is a live ring
        # block, so this rail never reaches it.
        return self._dflash_decline_reason(dec, int(position)) is None

    def _dflash_bootstrap(self, row, anchor, position, kwargs, page_tables_per_layer):
        owner = self._dflash_pending_owner
        self._spec_timed(
            "reseed",
            self._spec_bootstrap,
            int(anchor),
            int(position),
            kwargs.get("page_table"),
            kwargs.get("kv_cache"),
            page_tables_per_layer=page_tables_per_layer,
        )
        shape = (1, int(self._SPEC_CONTRACT_K))
        width = self._spec_decoder.width_for(int(position))
        self._spec_note_entry(
            shape,
            self._spec_last_phase_ms,
            None if width is None else self._spec_record_table_ms(shape, width, int(position)),
        )
        self._dflash_pending_owner = None
        self._dflash_live_owner = owner
        self._spec_owner_slot = owner[0] if owner else None
        self._dflash_retained = None
        self._dflash_stats = self._spec_policy.new_row()
        # The bootstrap refreshed the verify page tables from this table.
        page_table = kwargs.get("page_table")
        self._dflash_last_pt = page_table[:1].clone() if page_table is not None else None
        self._dflash_note_step(kwargs, page_tables_per_layer)

    def _dflash_decline_reason(self, dec, position):
        """Why no verify block can be written at ``position``, or None (``_spec_decline_reason``)."""
        return self._spec_decline_reason(position, int(dec.P_v), dec.width_for)

    def _dflash_verify_step(self, args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live, num_valid):
        """A step the runner marked speculative: a ``[rows, 1+K]`` candidate block.

        Returns ``VerifyOutput(spec_mode="argmax_ids")``. The row carrying
        drafts is answered from the retained proposal once the block matches
        it; a mismatch raises, because answering would report a posterior for
        tokens the device never evaluated. Rows without drafts take column 0
        from device sampling in plain decode. A solo row can use the session
        when ``_spec_row_can_use_argmax`` accepts its sampling and output
        requirements. ``decode_forward`` rejects host sampling, grammar
        bitmasks and logprobs before this function can consume a retained
        proposal.
        """
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, VerifyOutput

        width = self._SPEC_CONTRACT_K + 1
        valid = num_valid.reshape(-1).tolist() if num_valid is not None else [0] * int(rows)
        drafted = [r for r in live if r < len(valid) and int(valid[r]) > 0]
        ids = torch.full((int(rows), width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        if len(drafted) > 1:
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: rows {drafted} carry drafts, but one drafter session exists"
            )
        if drafted:
            row = drafted[0]
            retained = self._dflash_take_retained(
                row, self._spec_slot_of(row, kwargs), tokens, start_pos, rows, int(valid[row])
            )
            posterior = retained.posterior[:width]
            ids[row, : len(posterior)] = torch.tensor(posterior, dtype=torch.int32)
            if len(live) == 1:
                self._dflash_note_step(kwargs, page_tables_per_layer)
                self._spec_prepare_sampling_state(kwargs, start_pos, rows)
                return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
            # A peer joined after the proposal. The device evaluated the drafts
            # already, so the drafted row is answered and kept out of the
            # peers' plain decode; speculation ends with this step.
            plain = self._spec_plain_ids(args, kwargs, page_tables_per_layer, rows, exclude_rows=[row])
            for r in range(int(rows)):
                if r != row:
                    ids[r, 0] = plain[r]
            self._dflash_leave_speculation("peer joined while a proposal was outstanding")
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
        if len(live) == 1:
            row = live[0]
            sampling = kwargs["sampling_params"]
            if not self._spec_row_can_use_argmax(sampling, row):
                self._dflash_leave_row(
                    self._spec_slot_of(row, kwargs), "argmax IDs cannot satisfy the row's sampling output"
                )
            else:
                token = self._dflash_solo_next_token(row, tokens, start_pos, rows, kwargs, page_tables_per_layer)
                if token is not None:
                    self._spec_prepare_sampling_state(kwargs, start_pos, rows)
                    ids[row, 0] = int(token)
                    return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
        else:
            self._dflash_leave_speculation("several live rows", [self._spec_slot_of(r, kwargs) for r in live])
        ids[:, 0] = self._spec_plain_ids(args, kwargs, page_tables_per_layer, rows)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)

    def _dflash_take_retained(self, row, slot, tokens, start_pos, rows, valid):
        """The retained proposal that answers ``row``'s block, marked consumed.

        ``slot`` is the state slot the row reads this step. Raises when that
        slot owns no session, when no unconsumed proposal is retained, or when
        the block's anchor, position or first ``valid`` draft columns differ
        from the proposal. Truncation to fewer drafts is fine: the posterior at
        column j depends only on columns before it.
        """
        block = self._spec_row(tokens, rows, row)
        position = self._spec_row(start_pos, rows, row)[0] if start_pos is not None else None
        where = f"row {row} (slot {slot}), position {position}, anchor {block[0]}"
        if not self._spec_active or not self._spec_owns(self._dflash_live_owner, slot):
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify carries {valid} drafts for {where}, "
                "but this row owns no drafter session"
            )
        retained = self._dflash_retained
        if retained is None or retained.consumed:
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify carries {valid} drafts for {where}, "
                "but no proposal is retained for them"
            )
        if retained.anchor_token != block[0] or (position is not None and retained.anchor_position != position):
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify block for {where} does not continue the "
                f"retained proposal (anchor {retained.anchor_token} at position {retained.anchor_position})"
            )
        if valid > len(retained.drafts) or valid > len(block) - 1:
            raise RuntimeError(
                f"Gemma4DFlash speculative contract: verify carries {valid} drafts for {where}, "
                f"but the retained proposal has {len(retained.drafts)}"
            )
        sent = block[1 : 1 + valid]
        if sent != retained.drafts[:valid]:
            raise ValueError(
                "Gemma4DFlash speculative contract: verify was sent draft ids this model did not "
                f"propose: block={sent} proposed={retained.drafts[:valid]} ({where})"
            )
        retained.consumed = True
        retained.verified = int(valid)
        return retained

    # -- propose ---------------------------------------------------------------
    def propose_draft_tokens(self, num_drafts, committed, positions, counts, hidden=None):
        """``[rows, K]`` int32 drafts for the next step, drafted on device.

        Only the one live row of a solo step receives drafts, and only while
        it owns the live session. The runner's accepted count and anchor are
        committed first (the fused body merges the previous replay's rows at
        the start of the next one), the verify page tables are refreshed to
        the row's current block table, and the anchor position is checked
        against the captured widths, ``max_seq_len`` and the bounded ring
        before the replay. A decline returns ``num_valid`` 0 for every row and
        ends the session; the row continues as plain decode. So does the
        adaptive policy's choice of the plain step (``_dflash_choose``): a
        plain step gives the drafter no residual tap, so this rail cannot
        resume a request's session, and the policy's retry lands on the
        first proposal of a later request (the plain steps in between are
        counted here, ``note_plain_steps``).
        """
        self._spec_boundary("propose_start")
        try:
            del hidden
            k = int(num_drafts)
            rows = int(committed.shape[0]) if committed is not None and committed.dim() > 1 else 1
            live = self._spec_live_rows(positions, rows)
            width = int(committed.shape[1]) if committed is not None and committed.dim() > 1 else 1
            tokens, committing = self._spec_committed(counts, live, width)
            t_begin = self._spec_interval_begin(len(live), tokens, committing)
            self._spec_mode_now = ("plain", self._spec_plain_bucket(max(1, len(live))))
            top = max((int(self._spec_row(positions, rows, r)[0]) for r in live), default=0)
            self._spec_table_now = self._spec_plain_table_ms(max(1, len(live)), top)
            try:
                return self._dflash_propose(k, rows, live, committed, positions, counts)
            finally:
                self._spec_interval_end("Gemma4DFlash", t_begin, self._spec_mode_now, len(live))
        finally:
            self._spec_boundary("propose_end")

    def _dflash_propose(self, k, rows, live, committed, positions, counts):
        """``propose_draft_tokens`` proper; ``_spec_mode_now`` ends as the mode the step serves."""
        if len(live) != 1:
            self._dflash_leave_speculation("several live rows at proposal", live)
            return self._spec_decline(rows, k)
        row = live[0]
        if not self._spec_active or not self._spec_owns(self._dflash_live_owner, row):
            self._spec_policy.note_plain_steps()
            return self._spec_decline(rows, k)
        retained = self._dflash_retained
        if retained is not None and not retained.consumed:
            self._dflash_decline_session("the previous proposal was never answered")
            return self._spec_decline(rows, k)
        dec = self._spec_decoder
        block = self._spec_row(committed, rows, row)
        n = int(counts.reshape(-1)[row]) if counts is not None else 1
        n = max(1, min(n, len(block)))
        self._dflash_steps += 1
        self._dflash_committed += n
        if retained is not None and retained.consumed and retained.verified > 0 and self._dflash_stats is not None:
            # The count answers the verify that consumed the proposal: n - 1 of its ``verified`` drafts
            # were accepted (the plugin's accept walk commits a prefix plus one token).
            self._spec_policy.observe(self._dflash_stats, n - 1, retained.verified)
        anchor = block[n - 1]
        anchor_position = self._spec_row(positions, rows, row)[n - 1] if positions is not None else None
        dec.contract_commit(n, anchor)
        position = int(dec.start)
        if anchor_position is not None and position != int(anchor_position):
            self._dflash_decline_session(
                f"decoder position {position} disagrees with the runner's anchor position {anchor_position}"
            )
            return self._spec_decline(rows, k)
        self._spec_timed("refresh", self._dflash_refresh_tables, dec)
        reason = self._dflash_decline_reason(dec, position)
        if reason is not None:
            self._dflash_decline_session(reason)
            return self._spec_decline(rows, k)
        if not self._dflash_choose(position):
            self._dflash_end_session()
            return self._spec_decline(rows, k)
        width = int(dec.select_width(position) or dec.width_for(position))
        drafts, posterior = self._spec_timed("replay", dec.contract_replay, first=self._spec_first_step)
        self._spec_first_step = False
        shape = (1, int(self._SPEC_CONTRACT_K))
        self._spec_mode_now = ("step", shape, width)
        self._spec_table_now = self._spec_record_table_ms(shape, width, position)
        drafts = [int(t) for t in list(drafts)[:k]]
        self._dflash_retained = RetainedProposal(
            anchor_token=int(anchor),
            anchor_position=position,
            drafts=drafts,
            posterior=[int(t) for t in posterior],
        )
        out = torch.zeros((rows, k), dtype=torch.int32)
        out[row, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        valid[row] = len(drafts)
        return self._spec_draft_output(out, valid)

    def _dflash_decline_session(self, reason):
        logger.info(f"Gemma4DFlash speculative contract: proposal declined, session ends ({reason})")
        self._dflash_end_session()

    def _dflash_refresh_tables(self, dec):
        """Point the persistent verify page tables at the owner's current blocks.

        The runner allocates a block roughly every ``block_size`` tokens. The
        verify page-table buffers keep what they held at bootstrap until
        refreshed, and a KV write past that table's last block lands in the
        null block. Under bounded sliding the per-layer tables are rebuilt and
        installed first, because ``refresh_page_tables`` reads the installed
        ring tables for the sliding layers.
        """
        tables = self._dflash_owner_tables
        if tables is None:
            return
        page_table, page_tables_per_layer, kv_cache = tables
        if page_table is None:
            return
        row = page_table[:1] if page_table.dim() > 1 else page_table.reshape(1, -1)
        previous = self._dflash_last_pt
        if previous is not None and previous.shape == row.shape and torch.equal(previous, row):
            return
        if self._bounded_sliding_kv_cache:
            per_layer = self._build_per_layer_page_tables(page_tables_per_layer, page_table)
            per_layer = self._pad_sliding_page_tables_for_bounded(per_layer, kv_cache, authoritative=True)
            if per_layer:
                self.model[0]._active_page_tables_per_layer = per_layer
        dec.refresh_page_tables(row)
        self._dflash_last_pt = row.clone()


# -- the assistant drafter on the speculative contract ------------------------------
SPEC_DRAFT_LEN_ENV = "GEMMA4_SPEC_DRAFT_LEN"
# The records: ``max_rows:k`` entries, one captured (rows, drafts) record each
# plus (1, K). The table is the UPPER bound of the draft length per batch size;
# the acceptance-adaptive policy (``tt/spec_policy.py``, defaults on, shared by
# the dFlash rail whose one record is its K=5 block) picks the record to replay
# each step from the rows' measured acceptance and the cost table the decoder
# times at warmup, and may pick a shorter record or the plain step. Its knobs,
# read once at adapter construction (``Gemma4SpecContractBase._spec_init_state``):
#   GEMMA4_SPEC_ADAPTIVE=1        0 = the fixed behaviour: the smallest record covering the live rows (MTP),
#                                 every solo greedy request speculates (dFlash)
#   GEMMA4_SPEC_MIN_GAIN=1.25     enter a record only when expected tokens per cost beat the plain step by this
#                                 factor and the saving over GEMMA4_SPEC_ENTRY_HORIZON=64 steps covers the reseed
#                                 (an unmeasured record only on informed statistics: the optimistic default
#                                 prior gets a one-step trial instead)
#   GEMMA4_SPEC_EXIT_GAIN=0.95    hold the record while its gain stays above this; leave after two steps below.
#                                 While a record serves, its credit is what the current run delivered (the mean
#                                 tokens per row of its own steps), not the pool of earlier runs
#   GEMMA4_SPEC_RETRY_STEPS=8     the fewest plain steps before a retry: a record measured at break-even or
#                                 better re-enters once its entry as served (the reseed at a first proposal;
#                                 mid-request the cycle -- reseed, overtaken replay, reload -- times the reseeds
#                                 expected before every row's first draft matches) pays over the horizon,
#                                 unless its last run failed; else the cheapest record is exploration...
#   GEMMA4_SPEC_RETRY_OVERHEAD=0.01  ...drawing on the rung's exploration bucket, like every reseeding entry out
#                                 of the plain regime but one on measured credit whose last run paid: the bucket
#                                 fills at this fraction of every plain step's cost the rung serves, holds 1.5
#                                 trials at most (the trial priced with every draft rejected), pays every failed
#                                 run's realized loss and is never reset, so random text trials once per the
#                                 refill of a trial (~15 s) and a rung never measured waits until its trial has
#                                 filled; the log's reason "budget" is a step that waits for it
#   GEMMA4_SPEC_TRIAL_GAIN=0.6    a new request's first proposal replays the cheapest record once (its trial)
#                                 when the gain estimate is at least this -- GEMMA4_SPEC_MIN_GAIN once serving has
#                                 GEMMA4_SPEC_CALIB_MIN samples of the record at the rung, fresh or stale -- and
#                                 the reseed pays over the horizon at that estimate; below it the request runs
#                                 plain until the retry. In a batch with informed rows the trial also needs its hypothesis (the
#                                 new rows accept every draft) to clear GEMMA4_SPEC_MIN_GAIN: the trial reseeds
#                                 every row, and a row joining a batch that measured below plain cannot lift it
#   GEMMA4_SPEC_TRIAL_STEPS=3     steps a trial or retry holds its record (one from an uninformed prior) before
#                                 the hold rule decides on what those steps delivered; a trial whose overtaken
#                                 replay's first draft missed the committed token ends (one reseed per trial)
#   GEMMA4_SPEC_PROBE_STEPS=32    after this many steps below the cap, one step at the next larger draft length (0 = off)
#   GEMMA4_SPEC_VERIFY_CHECK=0    the first N served replays of every (shape, rung) also run a plain forward of each
#                                 row's anchor and log whether its argmax matches the packed verify's row 0 (a
#                                 wrong-keys detector; the zero-table warmup measurement replays are never checked)
#   GEMMA4_SPEC_EWMA=0.1          the acceptance EWMA weight of one verify
#   GEMMA4_SPEC_PLAIN_STEP_MS     the plain decode step cost (``ms`` flat, or ``ms_at_0,ms_at_top``); unset = timed
#                                 at warmup at position 0 and at the top position, per warmed decode bucket (every
#                                 cost is a line in the position)
#   GEMMA4_SPEC_CALIB_MIN=8       counted serving samples of a mode (the plain step at a bucket, a record at a rung)
#                                 after which the median ratio of its measured walls to the table scales the
#                                 table at the step's position in the decision; every interval counts, dirty
#                                 ones included (the prefill, switch, reseed or row change inside it is the
#                                 served cost of the mode in force), except idle intervals (the batch emptied)
#                                 and clean outliers above 3x, which are reported, never used; the measured
#                                 tokens per row (pooled over the records of one draft length at the rung: the
#                                 current run's own steps while it serves, the last run's mean between runs)
#                                 and the reseed as served follow
#   GEMMA4_SPEC_POOL_STALE_STEPS=128  intervals without a sample after which a pool is stale and the acceptance
#                                 model decides for its records (a trial on it still needs GEMMA4_SPEC_MIN_GAIN)
#   GEMMA4_SPEC_CALIB_PROBE_STEPS=256  steps a speculating batch serves without a plain sample before one plain
#                                 step refreshes the plain estimate (overhead-gated like the retry; 0 = off); the
#                                 hold then resumes its record (reseeding) under the hold rule, no exit
#   GEMMA4_SPEC_STATS_STEPS=256   proposals per "adaptive stats:" line (steps, tokens, clean walls per mode,
#                                 reseeds, phases, prefills); "adaptive summary:" on release covers the engine life
# Without a cost table (the decoder could not measure, and no plain cost is
# given) the fixed policy serves and a warning says so.
SPEC_K_BY_ROWS_ENV = "GEMMA4_SPEC_K_BY_ROWS"
SPEC_K_BY_ROWS_DEFAULT = "5:5"  # K=5 up to 5 live rows; above the table the step is the plain decode (K=0)
SPEC_MAX_ROWS_ENV = "GEMMA4_SPEC_MAX_ROWS"
SPEC_EXTRA_SHAPES_ENV = (
    "GEMMA4_SPEC_EXTRA_SHAPES"  # extra rows:k records beside the table's (a band's second draft length)
)
SPEC_MAX_ISL_ENV = "GEMMA4_SPEC_MAX_ISL"
SPEC_WIDTH_RUNGS_ENV = "GEMMA4_SPEC_WIDTH_RUNGS"
# The policy's INFO line on a record switch is rate-limited to one per this many seconds (0 = every switch).
SPEC_POLICY_LOG_INTERVAL_S = float(os.environ.get("GEMMA4_SPEC_POLICY_LOG_INTERVAL_S", "1.0"))
# The packed verify argmax reduces one 32-row tile: a record of ``rows``
# requests at ``k`` drafts packs ``rows * (k + 1)`` verify rows into it.
SPEC_PACKED_ROWS_LIMIT = 32


def _spec_draft_len_env():
    """``GEMMA4_SPEC_DRAFT_LEN`` as an int; ``auto`` and unset mean the default 5."""
    raw = os.environ.get(SPEC_DRAFT_LEN_ENV, "5").strip().lower()
    if raw in ("", "auto"):
        return 5
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{SPEC_DRAFT_LEN_ENV} must be an integer or 'auto', got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{SPEC_DRAFT_LEN_ENV} must be at least 1, got {value}")
    return value


def spec_k_by_rows_table(text, k_max):
    """Parse ``"5:5,8:3,16:1"`` into ``((5, 5), (8, 3), (16, 1))``, every k clamped to ``k_max``.

    ``max_rows`` must strictly ascend (the plugin's ``SpecPlan`` rule) and
    every entry must pack into the device argmax tile
    (``rows * (k + 1) <= 32``). Pure: config-time arithmetic.
    """
    entries = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        rows_text, sep, k_text = item.partition(":")
        if not sep:
            raise ValueError(f"{SPEC_K_BY_ROWS_ENV} entry {item!r} is not max_rows:k (table {text!r})")
        try:
            rows, k = int(rows_text), int(k_text)
        except ValueError as exc:
            raise ValueError(f"{SPEC_K_BY_ROWS_ENV} entry {item!r} is not two integers (table {text!r})") from exc
        if rows < 1 or k < 1:
            raise ValueError(f"{SPEC_K_BY_ROWS_ENV} entry {item!r} needs max_rows >= 1 and k >= 1 (table {text!r})")
        if entries and rows <= entries[-1][0]:
            raise ValueError(f"{SPEC_K_BY_ROWS_ENV} max_rows must strictly ascend, got {text!r}")
        k = min(k, int(k_max))
        if rows * (k + 1) > SPEC_PACKED_ROWS_LIMIT:
            raise ValueError(
                f"{SPEC_K_BY_ROWS_ENV} entry {item!r} packs {rows} x {k + 1} = {rows * (k + 1)} verify rows, "
                f"above the device argmax tile of {SPEC_PACKED_ROWS_LIMIT} (table {text!r})"
            )
        entries.append((rows, k))
    if not entries:
        raise ValueError(f"{SPEC_K_BY_ROWS_ENV} is empty ({text!r})")
    return tuple(entries)


def spec_plan_k_by_rows(table, max_num_seqs):
    """The plan's ``k_by_rows``: the table, plus a ``(max_num_seqs, 1)`` tail when it stops short.

    Admission requires the last ``max_rows`` to reach ``max_num_seqs``. The
    tail only satisfies that rule: live-row counts above
    ``GEMMA4_SPEC_MAX_ROWS`` are declined at proposal time, not drafted at 1.
    """
    k_by_rows = tuple(table)
    if int(max_num_seqs) > k_by_rows[-1][0]:
        k_by_rows += ((int(max_num_seqs), 1),)
    return k_by_rows


def spec_extra_shapes(text, k_max):
    """Parse ``GEMMA4_SPEC_EXTRA_SHAPES`` (``"5:3,1:2"``) into ``((5, 3), (1, 2))``: records captured beside the
    table's so a row band can compare a second draft length (the adaptive policy ranks every captured record
    covering the rows with a draft length up to the band's cap). They never raise a band's cap and change
    nothing in the plugin's plan. Each entry packs into the device argmax tile and drafts at most ``k_max``."""
    shapes = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        rows_text, sep, k_text = item.partition(":")
        try:
            rows, k = int(rows_text), int(k_text)
        except ValueError as exc:
            raise ValueError(f"{SPEC_EXTRA_SHAPES_ENV} entry {item!r} is not rows:k (table {text!r})") from exc
        if not sep or rows < 1 or k < 1:
            raise ValueError(f"{SPEC_EXTRA_SHAPES_ENV} entry {item!r} needs rows >= 1 and k >= 1 (table {text!r})")
        if k > int(k_max):
            raise ValueError(f"{SPEC_EXTRA_SHAPES_ENV} entry {item!r} drafts above {SPEC_DRAFT_LEN_ENV}={k_max}")
        if rows * (k + 1) > SPEC_PACKED_ROWS_LIMIT:
            raise ValueError(
                f"{SPEC_EXTRA_SHAPES_ENV} entry {item!r} packs {rows} x {k + 1} = {rows * (k + 1)} verify rows, "
                f"above the device argmax tile of {SPEC_PACKED_ROWS_LIMIT} (table {text!r})"
            )
        shapes.append((rows, k))
    return tuple(shapes)


def spec_capture_shapes(table, k_max, max_rows, extra=()):
    """The ``(rows, drafts)`` records to capture, ascending rows then drafts.

    ``(1, k_max)`` for a solo request plus one record per table entry, its
    rows clamped to ``max_rows`` (the smaller of ``GEMMA4_SPEC_MAX_ROWS`` and
    the engine's batch). Entries that clamp onto the same row count keep the
    larger draft length, since one record per row count is enough for the
    table (its record is a band's default and cap). ``extra`` records
    (``spec_extra_shapes``) join as they are, rows clamped the same way: a
    band's second draft length the policy can rank against the table's.
    """
    shapes = {1: int(k_max)}
    for rows, k in table:
        rows = min(int(rows), int(max_rows))
        if rows < 1:
            continue
        shapes[rows] = max(shapes.get(rows, 0), int(k))
    out = set(shapes.items())
    for rows, k in extra:
        rows = min(int(rows), int(max_rows))
        if rows >= 1:
            out.add((rows, int(k)))
    return tuple(sorted(out))


@dataclasses.dataclass
class MTPRetainedRow:
    """One row of a ``serving_step`` result, kept for the verify that consumes it.

    ``replay`` is the whole ``ServingStepResult``; ``shape_row`` is this
    request's index among the replay's live rows and ``row_id`` the row id
    the decoder was given for it (the next replay gathers this row's seed
    hidden by that id, so a moved slot reseeds instead). ``argmax_ids``
    holds the target's argmax at the ``K_v + 1`` packed positions
    ``[anchor, d1..dK]``. ``logits`` is this row's ``[K_v + 1, V]`` bf16
    host copy of the verify logits when the proposal read them back
    (``Gemma4MTPContractForCausalLM.propose_draft_tokens``), else None:
    the decoder's verify-logits buffer is shared by every record and is
    rewritten by its next reseed or step, which ``serial`` identifies.
    ``verified`` is how many of ``drafts`` the verify that consumed this row
    carried (0 until consumed, or when it carried none): the next proposal
    reads the accepted count against it. ``overtaken`` marks a proposal no
    verify consumed because an ordinary decode the runner had submitted
    before it was published committed the row past it (contract s.4d) with
    exactly the drafter's first draft: the replay's verify hidden at column 1
    then seeds the next proposal, as column ``count - 1`` would after a verify.
    """

    replay: object
    shape_row: int
    row_id: int
    serial: int
    anchor_token: int
    anchor_position: int
    drafts: list
    argmax_ids: list
    logits: object = None
    consumed: bool = False
    verified: int = 0
    overtaken: bool = False


class MTPRowNeedsLogits:
    """Which rows of a decode step must be verified in logits mode, evaluated per row on first use.

    The eager dict cost every decode step a sampling-parameter walk per live row (32 rows: ~0.7 ms) that
    only a replay's readback ever read; an all-declined step now evaluates nothing. ``rows`` may be a
    zero-argument callable returning the live rows: they are read on first use too, so a step that never
    consults this (the plain decode of an all-declined batch) never lists them."""

    __slots__ = ("_can_use", "_sampling", "_bitmask", "_rows", "_cache")

    def __init__(self, can_use, sampling, bitmask, rows):
        self._can_use = can_use
        self._sampling = sampling
        self._bitmask = bitmask
        self._rows = rows if callable(rows) else tuple(int(r) for r in rows)
        self._cache = {}

    def _live(self):
        if callable(self._rows):
            self._rows = tuple(int(r) for r in self._rows())
        return self._rows

    def get(self, row, default=True):
        row = int(row)
        if row not in self._live():
            return default
        value = self._cache.get(row)
        if value is None:
            value = self._sampling is None or not self._can_use(self._sampling, row, self._bitmask)
            self._cache[row] = value
        return value

    def __getitem__(self, row):
        return self.get(row)

    def any(self):
        return any(self.get(r) for r in self._live())

    def values(self):
        return [self.get(r) for r in self._live()]

    def items(self):
        return [(r, self.get(r)) for r in self._live()]

    def __eq__(self, other):
        try:
            return dict(self.items()) == dict(other.items() if hasattr(other, "items") else other)
        except (TypeError, ValueError):
            return NotImplemented

    __hash__ = None


@dataclasses.dataclass
class MTPSession:
    """The drafter state of one state slot: its owner, its retained proposal, its counters (``dropped``:
    proposals the runner committed past without verifying, contract s.4d) and its acceptance statistics
    (``spec_policy.AcceptanceStats``, opened from the batch prior and reset with the session)."""

    owner: tuple
    anchor: int = None
    position: int = None
    retained: MTPRetainedRow = None
    steps: int = 0
    committed: int = 0
    dropped: int = 0
    stats: object = None
    trialed: bool = False  # the request had its one trial (or entered on its own): no second trial
    # The all-declined proposals served without a walk that ``steps`` already counts (``_mtp_sync_session``).
    epoch: int = 0


class Gemma4MTPContractForCausalLM(Gemma4SpecContractBase):
    """The it-assistant (KV-shared) drafter on the plugin's speculative-decoding contract.

    The drafter reads the target's own KV caches, so prefill stays on the
    normal traced path and needs no taps. Every live row of a step shares
    ONE fused replay in ``propose_draft_tokens``: per row a seed hidden (the
    verify hidden of the row's new anchor, gathered on device, or a target
    forward at the anchor after an ordinary step), ``K_v`` assistant steps,
    and one packed target verify of ``[anchor, d1..dK]`` whose argmax ids
    and logits are retained per state slot. The next ``decode_forward``
    answers from that retained posterior after checking the block it was
    sent is a prefix of the proposal; a row the step carries without a
    proposal (a request whose first decode is this verify, or one that was
    declined) takes column 0 from a plain decode restricted to those rows.

    Records: one captured ``(rows, drafts)`` record per ``k_by_rows`` entry
    plus ``(1, k_max)``, each with its own verify-width ladder. The smallest
    record covering a step's live rows caps its draft length; within that
    cap the acceptance-adaptive policy (``tt/spec_policy.py``) picks the
    record whose expected tokens per measured step cost is best, or the
    plain step when none beats it by ``GEMMA4_SPEC_MIN_GAIN``, from the
    accepted counts the runner hands back per row and the cost table the
    decoder times at warmup (``_mtp_measure_costs``). Above
    ``GEMMA4_SPEC_MAX_ROWS`` it proposes nothing and every row runs the plain
    decode until the batch shrinks, when the next proposal reseeds and
    drafting resumes.

    Sessions are per state slot: opened at the first proposal for a slot,
    moved with ``note_state_slots_moved``, dropped by ``release_request`` and
    by a prefill into the slot (which also closes the slot's generation).
    """

    _SPEC_DRAFT_LEN = _spec_draft_len_env()
    # ``_auto_size_spec_ring`` and the bounded-ring guard read the packed rows per request.
    _SPEC_N = _SPEC_DRAFT_LEN + 1
    # Proposals decline before a verify would wrap an exact-window ring.
    _SPEC_DECLINES_AT_RING_WRAP = True

    model_capabilities = {
        **Gemma4ForCausalLM.model_capabilities,
        "output_tokens_per_step": 1,
        # A step this class does not speculate on runs the plain decode; the
        # device sampler keeps that step from pulling [B, vocab] logits to host.
        "supports_sample_on_device": True,
        # The TT sampler represents at most 32 candidates. Use host sampling
        # for sampled requests that ask for a wider top_k.
        "max_device_top_k": 32,
        # Keep penalty processing on vLLM's per-request host sampling path.
        "supports_device_penalties": False,
        # Asynchronous scheduling (contract s.4c). ``supports_async_decode`` is
        # the base's env-gated declaration (GEMMA4_SUPPORTS_ASYNC_DECODE): the
        # ordinary steps of this class are the base's resident traced decode,
        # and under that declaration no ordinary step is answered from the
        # drafter (``_mtp_ordinary_step``), so the only drafter-served steps
        # are verifies, after which the runner reloads the resident inputs.
        # ``supports_async_spec_decode`` promises what the deferred verify
        # needs: ``read_decode_output`` is handed the verify's own host tensor
        # ([B, 1+K] ids or [B, 1+K, V] logits, assembled from the retained
        # replay inside decode_forward) and passes it through unread, and the
        # drafter's hidden state never leaves the decoder's device pools,
        # which only the next propose_draft_tokens rewrites (``hidden`` is
        # None). GEMMA4_SUPPORTS_ASYNC_SPEC_DECODE=0 pins it off; the platform
        # then refuses async scheduling with speculation (--no-async-scheduling).
        "supports_async_spec_decode": _dflash_env_flag("GEMMA4_SUPPORTS_ASYNC_SPEC_DECODE", "1"),
        "supports_spec_decode": True,
        # Drafts come from the device and the target hidden state never leaves
        # it, so the runner holds no handle and ``hidden`` is unused.
        "spec_requirements": ("device_propose",),
        "spec_hidden_handoff": ("on_device",),
    }

    # -- plugin speculative contract: the plan ---------------------------------
    @staticmethod
    def _mtp_assistant_snapshot():
        """The assistant checkpoint directory, or None when it is not on disk.

        ``GEMMA4_ASSISTANT_MODEL`` wins; otherwise the target named by
        ``HF_MODEL`` selects its assistant (``assistant_repo_for``), which
        raises ``ValueError`` for a target without a known assistant.
        """
        path = os.environ.get("GEMMA4_ASSISTANT_MODEL") or _assistant_default_snapshot(
            os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
        )
        if path and os.path.isdir(str(path)):
            return str(path)
        return None

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """What this adapter serves at ``(max_num_seqs, requested_k)``, or why it cannot.

        Pure: reads env and the assistant's ``config.json``, allocates nothing,
        and never raises (the plugin quotes a ``SpecReject`` to the operator).
        ``K = min(requested_k, GEMMA4_SPEC_DRAFT_LEN)``; the draft cap per
        batch size comes from ``GEMMA4_SPEC_K_BY_ROWS`` clamped to ``K``, and
        ``effective_k`` is the largest cap of that table.
        """
        del vllm_config  # ordinary fields only would be allowed (contract s.2); none is needed
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        k_max = cls._SPEC_DRAFT_LEN
        k = min(int(requested_k), k_max)
        if k < 1:
            return SpecReject(
                reason=f"Gemma4MTP speculative contract: requested_k={requested_k} is below 1",
                supported_k=(k_max,),
            )
        try:
            snapshot = cls._mtp_assistant_snapshot()
        except ValueError as exc:
            return SpecReject(reason=f"Gemma4MTP speculative contract: {exc}", supported_k=())
        if snapshot is None:
            return SpecReject(
                reason=(
                    "Gemma4MTP speculative contract: assistant (drafter) snapshot not found. Set "
                    "GEMMA4_ASSISTANT_MODEL to a local snapshot, or make the weights cache writable so "
                    "google/gemma-4-<size>-it-assistant can be fetched."
                ),
                supported_k=(),
            )
        try:
            import json as _json

            with open(os.path.join(snapshot, "config.json")) as fh:
                cfg = _json.load(fh)
            hidden = int(cfg["backbone_hidden_size"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return SpecReject(
                reason=(
                    f"Gemma4MTP speculative contract: assistant config unreadable at {snapshot} "
                    f"(backbone_hidden_size): {exc!r}"
                ),
                supported_k=(),
            )
        try:
            table = spec_k_by_rows_table(os.environ.get(SPEC_K_BY_ROWS_ENV, SPEC_K_BY_ROWS_DEFAULT), k)
        except ValueError as exc:
            return SpecReject(reason=f"Gemma4MTP speculative contract: {exc}", supported_k=(k,))
        k_by_rows = spec_plan_k_by_rows(table, max_num_seqs)
        return SpecPlan(
            effective_k=max(cap for _, cap in table),
            lanes_per_request=1,
            # Per request: the seed hidden (backbone width, bf16) and its anchor
            # token and position (int64). The drafter's KV is the target's.
            extra_bytes_per_seq=hidden * 2 + 2 * 8,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids", "logits"),
            drafter_state="shared_with_target",
            # The assistant cross-attends the target's last sliding and last
            # full caches at absolute positions (wrapped by the bounded ring).
            drafter_target_cache_requires=("named_layer_caches", "absolute_positions"),
            supports_narrow_decode=True,
            k_by_rows=k_by_rows,
        )

    # -- construction --------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec_init_state()
        self._mtp_init_state()
        if self._bounded_sliding_kv_cache:
            _reserve_spec_ring_headroom(
                getattr(self._text_config(), "sliding_window", None),
                self._SPEC_N,
                type(self).__name__,
                declines_at_wrap=self._SPEC_DECLINES_AT_RING_WRAP,
            )
        self._spec_ring_advice(self._SPEC_N, "Gemma4MTP speculative contract")
        logger.info(
            f"Gemma4MTP speculative contract: K={self._SPEC_DRAFT_LEN}, one fused replay per step over "
            "every live row, output_tokens_per_step=1"
        )

    def _mtp_init_state(self):
        self._mtp_assistant = None
        self._mtp_decoder = None
        self._mtp_captured = False
        self._mtp_shapes = ()  # (rows, drafts) records captured, ascending rows then drafts
        self._mtp_table_shapes = ()  # the table's records among them: a band's default and cap
        self._mtp_widths = {}  # record -> verify-width ladder
        self._mtp_max_rows = 0
        self._mtp_sessions = {}  # state slot -> MTPSession
        self._mtp_trailing_logged = 0
        # Slots released since their last prefill: the runner proposes once more for a released request (its
        # last committed block), and that proposal nobody verifies must open no session and replay nothing.
        self._mtp_released_slots = set()
        # The proposals every row declined without a walk (``_mtp_fast_decline``): each is one plain step of
        # every live row, counted into a session when it is next walked or dropped (``_mtp_sync_session``); a
        # slot prefilled inside such a stretch opens its session with the steps it has had (``_mtp_slot_epoch``:
        # slot -> the count at its prefill).
        self._mtp_plain_epoch = 0
        self._mtp_slot_epoch = {}
        self._mtp_last_replay = None  # the latest ServingStepResult
        # Counts the decoder calls that rewrite its shared verify-logits buffer
        # (one bump per proposal that reseeds and/or replays).
        self._mtp_replay_serial = 0
        # Per decode row of the latest step: True when target argmax ids cannot
        # answer that row (no device sampling, a sampled row, logprobs, a grammar).
        self._mtp_row_needs_logits = {}
        self._mtp_step_tables = None  # (page_table, page_tables_per_layer, kv_cache) of the latest decode step
        self._mtp_installed_tables = None  # the per-layer record tables last handed to the decoder
        self._mtp_installed_keys = None  # (storage, version, shape, dtype) of the tables they were cloned from
        self._mtp_decline_logged = None
        # Proposals the runner committed past before verifying them (contract s.4d: an ordinary decode
        # submitted before the proposal was published drops it rather than verifying it).
        self._mtp_dropped_proposals = 0

    def _mtp_resolve_shapes(self, max_batch_size):
        """The records to capture and the row cap, from the env table and the engine's batch."""
        k_max = self._SPEC_DRAFT_LEN
        table = spec_k_by_rows_table(os.environ.get(SPEC_K_BY_ROWS_ENV, SPEC_K_BY_ROWS_DEFAULT), k_max)
        raw = os.environ.get(SPEC_MAX_ROWS_ENV)
        max_rows = int(raw) if raw else table[-1][0]
        if max_rows < 1:
            raise ValueError(f"{SPEC_MAX_ROWS_ENV} must be at least 1, got {max_rows}")
        max_rows = min(max_rows, int(max_batch_size))
        extra = spec_extra_shapes(os.environ.get(SPEC_EXTRA_SHAPES_ENV, ""), k_max)
        # the table's own records are a band's default and cap (``_mtp_shape_for``); the extra ones are candidates
        self._mtp_table_shapes = spec_capture_shapes(table, k_max, max_rows)
        return spec_capture_shapes(table, k_max, max_rows, extra), max_rows

    # -- warmup: prepare every record eagerly, then capture ----------------------------
    def warmup_model_decode(self, *args, **kwargs):
        """Base plain-decode warmup (every bucket), then the drafter: prepare on the eager pass, capture on the traced one.

        Every graph the records replay runs eagerly in ``serving_prepare``
        before the traced pass captures anything, which is what keeps a
        collective's first compile out of a capture.
        """
        out = super().warmup_model_decode(*args, **kwargs)
        if "enable_trace" not in kwargs:
            raise ValueError("Gemma4MTP speculative contract: warmup_model_decode needs enable_trace as a keyword")
        if kwargs["enable_trace"] is False:
            self._mtp_prepare(kwargs.get("kv_cache"), kwargs.get("num_blocks"))
            return out
        if self._mtp_decoder is None:
            self._mtp_prepare(kwargs.get("kv_cache"), kwargs.get("num_blocks"))
        self._mtp_capture()
        self._mtp_measure_costs(kwargs)
        return out

    def _mtp_prepare(self, kv_cache, num_blocks):
        """Build the assistant and the decoder at ``max_rows`` and run every record eagerly."""
        from models.demos.gemma4.tt.common import create_assistant_model
        from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

        if self._mtp_decoder is not None:
            raise RuntimeError("Gemma4MTP speculative contract: the drafter was prepared twice")
        if kv_cache is None:
            raise ValueError("Gemma4MTP speculative contract: warmup needs the KV cache")
        max_seq_len = int(getattr(self.model_args[0], "max_seq_len", 0) or 0)
        if not max_seq_len:
            raise ValueError("Gemma4MTP speculative contract: warmup needs max_seq_len")
        snapshot = self._mtp_assistant_snapshot()
        if snapshot is None:
            raise RuntimeError(
                "Gemma4MTP speculative contract: assistant (drafter) snapshot not found. Set "
                "GEMMA4_ASSISTANT_MODEL to a local snapshot, or make the weights cache writable so "
                "google/gemma-4-<size>-it-assistant can be fetched."
            )
        shapes, max_rows = self._mtp_resolve_shapes(int(self.model_args[0].max_batch_size))
        model0 = self.model[0]
        kv_layers = kv_cache
        if (
            isinstance(kv_layers, (list, tuple))
            and kv_layers
            and isinstance(kv_layers[0], (list, tuple))
            and kv_layers[0]
            and isinstance(kv_layers[0][0], (list, tuple))
        ):
            kv_layers = kv_layers[0]
        # Sized for the whole context: a drafter built from one request's
        # position cannot reach a later, longer request.
        _, self._mtp_assistant = create_assistant_model(
            mesh_device=self.mesh_device,
            target_model=model0,
            mesh_config=model0.mesh_config,
            ccl_manager=model0.ccl_manager,
            assistant_path=snapshot,
            max_local_batch_size=max_rows,
            max_seq_len=max_seq_len + 64,
        )
        blocks = int(num_blocks) if num_blocks else max(1, max_seq_len // 64)
        # Block 0 is vLLM's null block: the eager and captured records write there.
        scratch = torch.zeros(max_rows, blocks, dtype=torch.int32)
        decoder = SpeculativeDecoder(
            target_model=model0,
            assistant_model=self._mtp_assistant,
            mesh_device=self.mesh_device,
            tt_kv_cache=kv_layers,
            page_table_torch=scratch,
            stop_tokens=set(),
            draft_len=self._SPEC_DRAFT_LEN,
        )
        ladder_context = max_seq_len
        max_isl = int(os.environ.get(SPEC_MAX_ISL_ENV, "0") or 0)
        if max_isl > 0:
            ladder_context = min(max_seq_len, max_isl)
        max_rungs = int(os.environ.get(SPEC_WIDTH_RUNGS_ENV, "0") or 0) or None
        # The decoder's budgeted default ladder (geometric x4 from 2048 to the top rung, 5 rungs
        # at 262144): records cost ~22-32 MB of trace region each, width-independent.
        widths = None
        t0 = time.time()
        decoder.serving_prepare(list(shapes), widths, max_rows, max_context=ladder_context, max_rungs=max_rungs)
        self._mtp_decoder = decoder
        self._mtp_shapes = shapes
        self._mtp_widths = decoder.serving_widths()
        self._mtp_max_rows = max_rows
        logger.info(
            f"Gemma4MTP speculative contract: prepared {len(shapes)} records in {time.time() - t0:.1f}s "
            f"(max_rows={max_rows}, records={list(shapes)}, widths={self._mtp_widths})"
        )

    def _mtp_capture(self):
        if self._mtp_decoder is None:
            raise RuntimeError("Gemma4MTP speculative contract: serving_capture before serving_prepare")
        if self._mtp_captured:
            raise RuntimeError("Gemma4MTP speculative contract: the records were captured twice")
        t0 = time.time()
        self._mtp_decoder.serving_capture()
        self._mtp_captured = True
        logger.info(
            f"Gemma4MTP speculative contract: captured {len(self._mtp_shapes)} records in {time.time() - t0:.1f}s"
        )

    # -- warmup: the cost table of the adaptive record choice ---------------------------
    def _mtp_measure_costs(self, warmup_kwargs):
        """Time every record and the plain step on the captured decoder; the adaptive policy's cost table.

        Runs once, right after the capture, on the decode warmup's own inputs
        (zero tokens, positions and tables: block 0). The records are timed
        by ``SpeculativeDecoder.serving_measure_costs`` through the per-layer
        tables of that zero table (a bounded target's ring pools, installed
        the way a step installs them); the plain step is
        ``GEMMA4_SPEC_PLAIN_STEP_MS`` when set, else one timed traced replay
        of the base decode (``_spec_time_plain_step``). With
        ``GEMMA4_SPEC_ADAPTIVE=0`` nothing is measured.
        """
        if not self._spec_policy.config.adaptive:
            logger.info(f"Gemma4MTP adaptive K: off ({SPEC_ADAPTIVE_ENV}=0); the smallest covering record serves")
            return
        if self._mtp_decoder is None or not self._mtp_captured:
            raise RuntimeError("Gemma4MTP adaptive K: the cost table is measured after the capture")
        max_b = int(warmup_kwargs["max_batch_size"])
        num_blocks = warmup_kwargs.get("num_blocks")
        blocks = int(num_blocks) if num_blocks else int(self._mtp_decoder.page_table_torch.shape[-1])
        kv_cache = warmup_kwargs.get("kv_cache")
        zero_table = torch.zeros(max_b, blocks, dtype=torch.int32)
        top = self._spec_top_position()
        t0 = time.perf_counter()
        self._mtp_step_tables = (zero_table, None, kv_cache)
        try:
            _full, layers = self._mtp_step_page_tables()
            self._mtp_decoder.serving_refresh_page_tables(layers)
            self._mtp_remember_tables(layers)
            costs = {key: CostLine.of(value) for key, value in self._mtp_decoder.serving_measure_costs(top).items()}
            plain = self._spec_plain_step_cost(warmup_kwargs, zero_table, top)
        finally:
            self._mtp_step_tables = None
        self._spec_costs = costs
        self._spec_plain_cost = plain
        timings = 2 * len(costs) + (0 if self._spec_policy.config.plain_step_ms is not None else 2 * len(plain))
        self._spec_log_cost_table("Gemma4MTP", costs, plain, timings, time.perf_counter() - t0)

    # -- sessions ------------------------------------------------------------------
    def _mtp_session_at(self, slot):
        """The session owned by ``slot``'s current generation, or None."""
        session = self._mtp_sessions.get(int(slot))
        if session is None:
            return None
        if not self._spec_owns(session.owner, slot):
            raise RuntimeError(
                f"Gemma4MTP speculative contract: slot {slot} holds a session of generation "
                f"{session.owner[1]} but its generation is {self._spec_slot_gen.get(int(slot), 0)}"
            )
        return session

    def _mtp_open_session(self, slot):
        """A session for ``slot``'s current generation, starting with the plain steps its row has had since its
        prefill while every proposal declined without a walk (``_mtp_slot_epoch``)."""
        epoch = self._mtp_plain_epoch
        behind = max(0, epoch - self._mtp_slot_epoch.pop(int(slot), epoch))
        session = MTPSession(
            owner=self._spec_owner_of(slot),
            stats=self._spec_policy.new_row(),
            steps=behind,
            committed=behind,
            epoch=epoch,
        )
        self._mtp_sessions[int(slot)] = session
        return session

    def _mtp_sync_session(self, session):
        """Count the proposals that declined without a walk since ``session`` was last walked: one plain step
        (one committed token) of its row each (``_mtp_fast_decline``)."""
        behind = self._mtp_plain_epoch - session.epoch
        if behind > 0:
            session.steps += behind
            session.committed += behind
            session.epoch = self._mtp_plain_epoch

    def _mtp_drop_session(self, slot, why):
        session = self._mtp_sessions.pop(int(slot), None)
        if session is None:
            return
        self._mtp_sync_session(session)
        self._spec_note_row_ended()
        if session.steps:
            self._spec_policy.note_request_steps(session.steps)
            stats = session.stats
            accepted = (
                f"{stats.ewma_accepted:.2f} accepted drafts/verify (EWMA over {stats.spec_steps})"
                if stats is not None and stats.ewma_accepted is not None
                else "no verified drafts"
            )
            dropped = f", {session.dropped} proposals dropped unverified" if session.dropped else ""
            logger.info(
                f"Gemma4MTP contract summary: slot {slot} ({why}) {session.steps} steps, "
                f"{session.committed} tokens, {session.committed / session.steps:.2f} tokens/step, {accepted}{dropped}"
            )

    def _spec_follow_slot_moves(self, moved):
        sessions = {}
        for slot, session in self._mtp_sessions.items():
            new = moved.get(slot, slot)
            if new in sessions:
                raise RuntimeError(f"Gemma4MTP speculative contract: slot moves {moved} put two sessions on slot {new}")
            session.owner = (new, session.owner[1])
            sessions[new] = session
        self._mtp_sessions = sessions
        self._mtp_released_slots = {moved.get(slot, slot) for slot in self._mtp_released_slots}
        self._mtp_slot_epoch = {moved.get(slot, slot): epoch for slot, epoch in self._mtp_slot_epoch.items()}

    def _spec_release_slot(self, row):
        self._mtp_drop_session(row, "release")
        self._mtp_released_slots.add(int(row))
        self._mtp_slot_epoch.pop(int(row), None)

    def _spec_release_all(self):
        for slot in list(self._mtp_sessions):
            self._mtp_drop_session(slot, "release")

    def release_persistent_capture(self) -> None:
        for slot in list(self._mtp_sessions):
            self._mtp_drop_session(slot, "shutdown")
        if self._spec_last_propose is not None:
            self._spec_log_stats("Gemma4MTP", final=True)
        if self._mtp_dropped_proposals:
            # Each one cost a drafter replay that answered nothing (an ordinary decode the runner had
            # submitted before the proposal was published, contract s.4d); a steady rate here means the
            # scheduler overlaps more steps than the drafter can serve.
            logger.info(
                f"Gemma4MTP speculative contract: {self._mtp_dropped_proposals} proposals dropped unverified "
                "over the engine's life"
            )
            self._mtp_dropped_proposals = 0
        self._mtp_last_replay = None
        self._mtp_step_tables = None
        self._mtp_installed_tables = None
        self._mtp_installed_keys = None
        self._mtp_plain_epoch = 0
        self._mtp_slot_epoch = {}
        self._spec_costs = None
        self._spec_plain_cost = None
        try:
            if self._mtp_decoder is not None:
                self._mtp_decoder.serving_release()
        finally:
            self._mtp_decoder = None
            self._mtp_captured = False
            super().release_persistent_capture()

    # -- prefill: a slot's new occupant opens a new generation ----------------------
    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        empty_slots = kwargs.get("empty_slots")
        for slot in list(empty_slots) if empty_slots is not None else ():
            self._mtp_drop_session(int(slot), "prefill")
            self._spec_close_slot_generation(int(slot))
            self._mtp_released_slots.discard(int(slot))
            self._mtp_slot_epoch[int(slot)] = self._mtp_plain_epoch
        self._spec_note_prefill()
        return super().prefill_forward(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)

    # -- decode ------------------------------------------------------------------------
    def decode_forward(
        self,
        *args,
        page_tables_per_layer=None,
        reload_inputs: bool,
        reload_page_table: bool,
        reload_sampling_params: bool,
        reset_sampling_state: bool,
        **kwargs,
    ):
        kwargs.update(
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
        )
        spec_mode = kwargs.pop("spec_mode", None)
        self._spec_boundary("decode_start", spec_mode is not None)
        num_valid = kwargs.pop("num_valid_drafts", None)
        # The commit it describes reaches propose_draft_tokens; the KV holds
        # every candidate position already, so no per-candidate state is selected.
        kwargs.pop("accepted_counts", None)
        kwargs.pop("draft_token_ids", None)
        tokens, start_pos = self._spec_inputs(args, kwargs)
        if tokens is None:
            raise ValueError("Gemma4MTP speculative contract: decode_forward expects tokens")
        rows = self._spec_rows(tokens)
        resolved = []

        def live_rows():
            # The live rows, listed once and only by a step that walks them (a retained proposal, a verify, a
            # replay's readback): the plain decode of an all-declined batch never reads the positions.
            if not resolved:
                resolved.append(self._spec_live_rows(start_pos, rows))
            return resolved[0]

        self._mtp_step_tables = (kwargs.get("page_table"), page_tables_per_layer, kwargs.get("kv_cache"))
        # What this step's rows will need from their next verify: a row the
        # runner samples on the host (no sampling_params, which is also every
        # logits verify), a sampled or penalised row, one requesting logprobs
        # or one under a grammar is verified in logits mode, so the proposal
        # that follows this step must read the verify logits back for it.
        sampling = kwargs.get("sampling_params")
        bitmask = kwargs.get("grammar_bitmask")
        self._mtp_row_needs_logits = MTPRowNeedsLogits(self._spec_row_can_use_argmax, sampling, bitmask, live_rows)
        if self._mtp_unread_replay_pending() and (sampling is None or self._mtp_row_needs_logits.any()):
            # A request that needs logits reaches a decode step before its own
            # proposal replays, so this is where every greedy-only replay still
            # in the buffer gets its host copy, ahead of that replay.
            self._mtp_read_back_live_replay()
        try:
            if spec_mode is None:
                return self._mtp_ordinary_step(args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live_rows)
            if spec_mode not in ("argmax_ids", "logits"):
                raise ValueError(f"Gemma4MTP speculative contract: unsupported spec_mode {spec_mode!r}")
            # Timed as its own phase: a verify answered from the retained replay runs no device work, so its wall is
            # the host assembly of the answer and the sampling state (the stats line's "verify" ms/call).
            return self._spec_timed(
                "verify",
                self._mtp_verify_step,
                args,
                kwargs,
                page_tables_per_layer,
                tokens,
                start_pos,
                rows,
                live_rows(),
                num_valid,
                spec_mode,
            )
        finally:
            self._spec_boundary("decode_end", spec_mode is not None)

    def _mtp_any_retained(self):
        """Whether any session holds a retained proposal (the only state an ordinary step has to walk)."""
        return any(session.retained is not None for session in self._mtp_sessions.values())

    def _mtp_unread_replay_pending(self):
        """Whether the latest replay left a retained proposal without its host logits (``_mtp_read_back_live_replay``)."""
        serial = self._mtp_replay_serial
        for session in self._mtp_sessions.values():
            retained = session.retained
            if retained is not None and not retained.consumed and retained.logits is None and retained.serial == serial:
                return True
        return False

    def _mtp_retained_for(self, session, anchor, position):
        """The unconsumed retained row that continues ``(anchor, position)``, or None."""
        if session is None or session.retained is None:
            return None
        retained = session.retained
        if retained.consumed or retained.anchor_token != anchor or retained.anchor_position != position:
            return None
        return retained

    def _mtp_ordinary_step(self, args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live):
        """An ordinary decode step: the runner asked for no verification.

        Under the resident decode (``supports_async_decode``) every live row
        runs the plain decode and a retained proposal is dropped: the runner
        keeps the plain decode's device-resident token and position across
        ordinary steps and reloads them only after a verify, so a row this
        step answered from the drafter would leave the next steady decode on
        stale inputs (``_spec_plain_decode``). Otherwise a greedy-eligible row
        whose retained proposal continues its token (the scheduler sent no
        drafts for it) is answered from that proposal's column 0; every other
        live row runs the plain decode, restricted to those rows, and its
        retained proposal, if any, is dropped because the plain decode moved
        the row past it.
        """
        if not self._mtp_any_retained():
            # Every row declined (or never proposed): the plain decode, with no per-row walk.
            return self._spec_plain_decode(args, kwargs, page_tables_per_layer)
        live = live() if callable(live) else live
        sampling = kwargs.get("sampling_params")
        resident = self._spec_resident_decode_declared()
        answered = {}
        for r in live:
            session = self._mtp_session_at(self._spec_slot_of(r, kwargs))
            if session is None:
                continue
            anchor = self._spec_row(tokens, rows, r)[0]
            position = self._spec_row(start_pos, rows, r)[0] if start_pos is not None else None
            retained = self._mtp_retained_for(session, anchor, position)
            if (
                not resident
                and retained is not None
                and sampling is not None
                and self._spec_row_can_use_argmax(sampling, r, kwargs.get("grammar_bitmask"))
            ):
                retained.consumed = True
                answered[r] = retained.argmax_ids[0]
            else:
                if session.retained is not None and not session.retained.consumed:
                    # The plain decode moves the row past its proposal: dropped, never verified (s.4d).
                    self._mtp_dropped_proposals += 1
                    session.dropped += 1
                session.retained = None
        if not answered:
            return self._spec_plain_decode(args, kwargs, page_tables_per_layer)
        plain_rows = [r for r in live if r not in answered]
        if plain_rows:
            out = self._spec_plain_ids(args, kwargs, page_tables_per_layer, rows, exclude_rows=list(answered))
        else:
            self._spec_prepare_sampling_state(kwargs, start_pos, rows)
            self._spec_note_served_step()
            out = torch.zeros(int(rows), dtype=torch.int32)
        for r, token in answered.items():
            out[r] = int(token)
        return out

    def _mtp_take_retained(self, session, slot, row, block, position, valid, width):
        """The retained row that answers ``row``'s block, marked consumed.

        Raises when the slot owns no session, no unconsumed proposal is
        retained, or the block's anchor, position or first ``valid`` draft
        columns differ from the proposal. Truncation to fewer drafts is fine:
        the posterior at column j depends only on columns before it.
        """
        where = f"row {row} (slot {slot}), position {position}, anchor {block[0]}"
        if session is None:
            raise RuntimeError(
                f"Gemma4MTP speculative contract: verify carries {valid} drafts for {where}, "
                "but this row owns no drafter session"
            )
        retained = session.retained
        if retained is None or retained.consumed:
            raise RuntimeError(
                f"Gemma4MTP speculative contract: verify carries {valid} drafts for {where}, "
                "but no proposal is retained for them"
            )
        if retained.anchor_token != block[0] or (position is not None and retained.anchor_position != position):
            raise RuntimeError(
                f"Gemma4MTP speculative contract: verify block for {where} does not continue the "
                f"retained proposal (anchor {retained.anchor_token} at position {retained.anchor_position})"
            )
        if valid > len(retained.drafts) or valid > width - 1:
            raise RuntimeError(
                f"Gemma4MTP speculative contract: verify carries {valid} drafts for {where}, "
                f"but the retained proposal has {len(retained.drafts)} and the block {width - 1} columns"
            )
        sent = block[1 : 1 + valid]
        if sent != retained.drafts[:valid]:
            raise ValueError(
                "Gemma4MTP speculative contract: verify was sent draft ids this model did not "
                f"propose: block={sent} proposed={retained.drafts[:valid]} ({where})"
            )
        retained.consumed = True
        retained.verified = int(valid)
        return retained

    def _mtp_vocab_size(self):
        return int(self.model[0].vocab_size)

    def _mtp_verify_step(self, args, kwargs, page_tables_per_layer, tokens, start_pos, rows, live, num_valid, mode):
        """A step the runner marked speculative: a ``[rows, 1+K]`` candidate block.

        Rows carrying drafts are answered from their retained proposal after
        the prefix check; a draftless row whose proposal continues its
        token is answered from that proposal's column 0; the remaining live
        rows (a peer that joined after the proposals, a declined row, the
        resolving verify after an all-rows decline) take column 0 from the
        plain decode restricted to them. ``argmax_ids`` returns the ids,
        ``logits`` the host ``[rows, 1+K, V]`` bf16 block assembled from the
        retained device logits, zero where no column was evaluated.
        """
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, VerifyOutput

        width = int(tokens.shape[1]) if tokens.dim() > 1 else 1
        if width < 1 or width > self._SPEC_N:
            raise ValueError(f"Gemma4MTP speculative contract: verify block width {width} outside [1, {self._SPEC_N}]")
        valid = num_valid.reshape(-1).tolist() if num_valid is not None else [0] * int(rows)
        sampling = kwargs.get("sampling_params")
        if mode == "logits" and sampling is not None:
            raise ValueError(
                "Gemma4MTP speculative contract: a logits verify carries no device sampling parameters "
                "(contract s.4h), but sampling_params were sent"
            )
        answered = {}
        for r in live:
            slot = self._spec_slot_of(r, kwargs)
            session = self._mtp_session_at(slot)
            block = self._spec_row(tokens, rows, r)
            position = self._spec_row(start_pos, rows, r)[0] if start_pos is not None else None
            n = int(valid[r]) if r < len(valid) else 0
            if n > 0:
                answered[r] = self._mtp_take_retained(session, slot, r, block, position, n, width)
                continue
            retained = self._mtp_retained_for(session, block[0], position)
            if retained is not None:
                retained.consumed = True
                answered[r] = retained
            elif session is not None:
                session.retained = None
        plain_rows = [r for r in live if r not in answered]
        excluded = list(answered)
        if mode == "argmax_ids":
            ids = torch.full((int(rows), width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            for r, retained in answered.items():
                cols = min(width, len(retained.argmax_ids))
                ids[r, :cols] = torch.tensor(retained.argmax_ids[:cols], dtype=torch.int32)
            if plain_rows:
                if sampling is not None:
                    plain = self._spec_plain_ids(args, kwargs, page_tables_per_layer, rows, exclude_rows=excluded)
                else:
                    plain = self._spec_plain_logits(args, kwargs, page_tables_per_layer, rows, exclude_rows=excluded)
                    plain = plain.argmax(dim=-1).to(torch.int32)
                for r in plain_rows:
                    ids[r, 0] = int(plain[r])
            else:
                self._spec_prepare_sampling_state(kwargs, start_pos, rows)
                self._spec_note_served_step()
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
        vocab = self._mtp_vocab_size()
        logits = torch.zeros((int(rows), width, vocab), dtype=torch.bfloat16)
        late = {}
        for r, retained in answered.items():
            host = retained.logits
            if host is None:
                # The replay ran for greedy-eligible rows only, so the proposal
                # read nothing back, and no later decode step carried a row
                # this adapter could see needs logits (``_mtp_read_back_live_replay``).
                # The decoder's shared verify-logits buffer still holds that
                # replay until its next reseed or step.
                if retained.serial != self._mtp_replay_serial:
                    raise RuntimeError(
                        f"Gemma4MTP speculative contract: row {r} is verified in logits mode for its proposal "
                        f"at position {retained.anchor_position} (replay {retained.serial}), but that replay was "
                        f"not read back and replay {self._mtp_replay_serial} has since overwritten the shared "
                        "verify-logits buffer. Only one sequence reaches this: a row that looked greedy-eligible "
                        "on every decode step it was part of (temperature 0, no penalty, no logprobs, no grammar "
                        "bitmask in its sampling params) is verified in logits mode because the runner applies a "
                        "control this adapter cannot see in sampling_params (allowed_token_ids, bad_words, "
                        "min_tokens), and its proposal was overwritten by another request's replay with no "
                        "host-sampled or flagged decode step in between"
                    )
                block = late.get(id(retained.replay))
                if block is None:
                    block = self._mtp_read_logits(retained.replay, vocab)
                    late[id(retained.replay)] = block
                host = block[retained.shape_row]
            cols = min(width, int(host.shape[0]))
            logits[r, :cols] = host[:cols]
        if plain_rows:
            plain = self._spec_plain_logits(args, kwargs, page_tables_per_layer, rows, exclude_rows=excluded)
            if int(plain.shape[-1]) != vocab:
                raise ValueError(
                    f"Gemma4MTP speculative contract: the plain decode returned {int(plain.shape[-1])} "
                    f"vocabulary columns, the verify block has {vocab}"
                )
            for r in plain_rows:
                logits[r, 0] = plain[r].to(torch.bfloat16)
        else:
            self._spec_prepare_sampling_state(kwargs, start_pos, rows)
            self._spec_note_served_step()
        return VerifyOutput(spec_mode="logits", logits=logits, hidden=None)

    def _mtp_read_back_live_replay(self):
        """Give every retained row of the current replay its host logits, once for the replay.

        The decoder's shared verify-logits buffer holds the latest replay
        until the next reseed or step, both of which happen inside
        ``propose_draft_tokens``. A proposal reads the buffer back only when
        one of its own rows needs logits; a row verified later in logits mode
        because a sampled, penalised, logprobs or host-sampled request joined
        the batch gets its copy here, on that request's first decode step,
        which precedes the request's own proposal and therefore the overwrite.
        """
        host = None
        for session in self._mtp_sessions.values():
            retained = session.retained
            if (
                retained is None
                or retained.consumed
                or retained.logits is not None
                or retained.serial != self._mtp_replay_serial
            ):
                continue
            if host is None:
                host = self._mtp_read_logits(retained.replay, self._mtp_vocab_size())
            retained.logits = host[retained.shape_row]

    def _mtp_read_logits(self, result, vocab):
        """``result.logits_to_host()`` as ``[live, K_v + 1, V]`` bf16, shape-checked against the record."""
        host = result.logits_to_host()
        expected = (int(result.live), int(result.shape[1]) + 1, int(vocab))
        if tuple(host.shape) != expected:
            raise ValueError(
                f"Gemma4MTP speculative contract: logits_to_host returned {tuple(host.shape)}, "
                f"expected [live, K_v + 1, V] = {expected}"
            )
        return host.to(torch.bfloat16)

    # -- propose: one fused replay over every drafting row -------------------------------
    def _mtp_shape_for(self, live_rows):
        """The smallest TABLE record whose rows cover ``live_rows`` (the band's default, whose draft length caps
        the choice), or None; an extra record (``GEMMA4_SPEC_EXTRA_SHAPES``) is a candidate, never the default."""
        for shape in self._mtp_table_shapes:
            if shape[0] >= int(live_rows):
                return shape
        return None

    def _mtp_width_for(self, shape, position):
        """The narrowest rung of ``shape``'s ladder covering a verify at ``position``, or None.

        A rung must hold the anchor, the drafts and one more position
        (``S_k >= position + K_v + 2``, the ladder's own rule).
        """
        need = int(position) + int(shape[1]) + 2
        for width in sorted(self._mtp_widths[shape]):
            if width >= need:
                return width
        return None

    def _mtp_step_page_tables(self):
        """The latest decode step's flat ``[B_wire, blocks]`` table and the full per-layer set.

        The per-layer set is built exactly as the plain decode builds its own
        (kv-share aliases, bounded ring padding) and kept at full width: the
        decoder selects its rows by the row ids each call passes.
        """
        if self._mtp_step_tables is None:
            raise RuntimeError("Gemma4MTP speculative contract: propose_draft_tokens before any decode step")
        page_table, page_tables_per_layer, kv_cache = self._mtp_step_tables
        if page_table is None:
            raise ValueError("Gemma4MTP speculative contract: the decode step carried no page_table")
        if not isinstance(page_table, torch.Tensor):
            raise TypeError(f"Gemma4MTP speculative contract: page table of type {type(page_table).__name__}")
        full = page_table if page_table.dim() > 1 else page_table.reshape(1, -1)
        per_layer = self._build_per_layer_page_tables(page_tables_per_layer, page_table)
        if self._bounded_sliding_kv_cache:
            per_layer = self._pad_sliding_page_tables_for_bounded(per_layer, kv_cache, authoritative=True)
        return full, list(per_layer)

    @staticmethod
    def _mtp_record_rows(full, b_v, row_ids):
        """The record's flat ``[B_v, blocks]`` table for one decoder call: row i = ``full[row_ids[i]]``.

        Every row past ``len(row_ids)`` is a padding row: zeros, i.e. block 0,
        vLLM's null block, which the decoder's padding rows write into. The
        decoder checks exactly that layout against the ``rows`` it is given,
        so a reseed of a subset of the record's rows gets its own table.
        """
        flat = torch.zeros((int(b_v), int(full.shape[1])), dtype=torch.int32)
        for i, r in enumerate(row_ids):
            if r >= int(full.shape[0]):
                raise ValueError(
                    f"Gemma4MTP speculative contract: row {r} is outside the {int(full.shape[0])}-row page table"
                )
            flat[i] = full[r].to(torch.int32)
        return flat

    @staticmethod
    def _mtp_table_key(table):
        """What identifies a host table's content without reading it: its storage, in-place version, shape and
        dtype (a view shares its base's version counter, so a block appended in place changes the key)."""
        if table is None:
            return None
        return (table.data_ptr(), int(table._version), tuple(table.shape), str(table.dtype))

    def _mtp_remember_tables(self, layers):
        """Keep what the decoder was handed: one clone per distinct table object (the layers of a KV group share
        one host table) and each layer's key, so the next step's check is a key compare per layer and at most one
        content compare per distinct object. Cloning every layer cost 48 x 512 KB a refresh at 32 rows."""
        clones = {}
        kept = []
        for table in layers:
            if table is None:
                kept.append(None)
                continue
            clone = clones.get(id(table))
            if clone is None:
                clone = clones[id(table)] = table.clone()
            kept.append(clone)
        self._mtp_installed_tables = kept
        self._mtp_installed_keys = [self._mtp_table_key(table) for table in layers]

    def _mtp_tables_changed(self, layers):
        """Whether the step's per-layer tables differ from the ones last handed to the decoder: a layer whose
        table is the same storage at the same version is unchanged without a read; a new object is compared
        once per distinct (old, new) pair (every layer of a KV group shares one table, so 48 layers cost two or
        three compares, not 48); when nothing changed the new objects' keys are kept for the next step."""
        previous = self._mtp_installed_tables
        if previous is None or len(previous) != len(layers):
            return True
        keys = self._mtp_installed_keys or []
        new_keys = []
        compared = {}
        for i, (a, b) in enumerate(zip(previous, layers)):
            if a is None or b is None:
                if (a is None) != (b is None):
                    return True
                new_keys.append(None)
                continue
            key = self._mtp_table_key(b)
            new_keys.append(key)
            if i < len(keys) and keys[i] == key:
                continue
            pair = (id(a), id(b))
            same = compared.get(pair)
            if same is None:
                same = a.shape == b.shape and bool(torch.equal(a, b))
                compared[pair] = same
            if not same:
                return True
        self._mtp_installed_keys = new_keys
        return False

    def _mtp_decline_step(self, rows, k, live, reason, log=True, sessions=None):
        """Offer no row a draft and drop every retained proposal (the next proposal reseeds).

        ``log`` is off for the adaptive policy's plain steps, which
        ``_spec_policy_note`` logs on its own rate limit. ``sessions`` are the
        live rows' sessions when the caller walked them already (no second
        lookup per row); else they are looked up from ``live``.
        """
        if log and reason != self._mtp_decline_logged:
            logger.info(f"Gemma4MTP speculative contract: proposing nothing ({reason})")
            self._mtp_decline_logged = reason
        if sessions is None:
            sessions = (self._mtp_session_at(r) for r in live)
        for session in sessions:
            if session is not None:
                session.retained = None
        return self._spec_decline(rows, k)

    def _mtp_fast_decline_reason(self, first, live_n, top):
        """Why every row of the step declines before any row is walked, or None when the walk must run.

        The walk is needed by a trailing row (a released slot still listed
        live: it leaves the proposing set), by any retained proposal (a
        consumed one is observed, an unconsumed one dropped, in the walk), by
        a batch some table record covers (the policy decides) and by a batch
        whose highest row the largest record's block cannot hold (such rows
        leave the set, which can then fall under the cap). Otherwise the
        answer is the walk's own: the records are not captured, or the rows
        exceed the largest record.
        """
        if self._mtp_any_retained():
            return None
        if any(s < int(first.numel()) and bool(first[s] >= 0) for s in self._mtp_released_slots):
            return None
        if self._mtp_decoder is None or not self._mtp_captured:
            return "the verify records are not captured (warmup incomplete)"
        if self._mtp_shape_for(live_n) is not None:
            return None
        largest = self._mtp_shapes[-1]
        if self._spec_decline_reason(top, int(largest[1]) + 1, lambda pos, s=largest: self._mtp_width_for(s, pos)):
            return None
        return f"{live_n} proposing rows exceed the largest record {largest} ({SPEC_MAX_ROWS_ENV}={self._mtp_max_rows})"

    def _mtp_fast_decline(self, rows, k, live_n, top, reason):
        """The all-declined proposal without a per-row walk: O(1) host work beside the plain decode.

        No session is read or opened (a row prefilled above the cap opens its
        session, with the plain steps it has had, at the first proposal the
        walk serves: ``_mtp_slot_epoch``), the decline is logged once per
        reason, and the closed interval is accounted as the plain step of the
        live rows the way the walk's decline accounts it: with no proposal
        retained the step that just committed was the plain decode, in which
        every live row committed one token (``_spec_committed`` without
        counts), so the runner's count block is not read.
        """
        t_begin = self._spec_interval_begin(live_n, live_n, live_n)
        self._spec_mode_now = ("plain", self._spec_plain_bucket(live_n))
        self._spec_table_now = self._spec_plain_table_ms(live_n, top)
        try:
            self._mtp_plain_epoch += 1
            if reason != self._mtp_decline_logged:
                logger.info(f"Gemma4MTP speculative contract: proposing nothing ({reason})")
                self._mtp_decline_logged = reason
            return self._spec_decline(rows, k)
        finally:
            self._spec_interval_end("Gemma4MTP", t_begin, self._spec_mode_now, live_n)

    # -- propose: the acceptance-adaptive record choice ---------------------------------
    @staticmethod
    def _mtp_seed_continues(session, row, last):
        """Whether ``row``'s next replay seeds from the last replay's verify hidden (no reseed): its retained
        proposal was consumed by a verify, came from that replay and kept its row id (``propose_draft_tokens``)."""
        retained = session.retained
        return (
            retained is not None
            and retained.consumed
            and last is not None
            and retained.replay is last
            and retained.row_id == int(row)
        )

    def _mtp_choose_record(self, default, drafting, k, last, live_rows, overtaken=False):
        """``(shape, members)`` to replay for ``drafting`` rows, or ``(None, None)`` for a plain step
        (``spec_policy.SpecPolicy``).

        ``default`` is the smallest record covering the rows, whose draft
        length (clamped to the runner's ``k``) caps the choice: every
        captured record covering the rows with a draft length up to that
        cap is a candidate, costed at the rows' top position (the warmup
        line of its rung until serving calibrated it) and credited the
        measured tokens per row of its draft length at the rung once
        serving measured any record of that length there (else the
        acceptance model), with the reseed of its shape as the entry cost
        when a row reseeds this step. When rows
        continue from the last replay while others would reseed, the record
        over the continuing rows alone is a candidate too: the rows left out
        take one token per step from the plain decode the verify runs beside
        the replay, which the candidate's cost carries. A joining row that
        has no verify of its own yet and no trial behind it is ``fresh``:
        declined otherwise, the policy replays the cheapest record once as
        the request's trial (``GEMMA4_SPEC_TRIAL_GAIN``) -- in a batch with
        informed rows only when the new rows accepting every draft would
        lift the batch over the entry bar, since the trial reseeds every row
        (an (8, 3) reseed is ~230 ms against a 128-token request).
        Without the cost
        table (the fixed policy, or warmup measured nothing) the default
        serves for every row; a candidate whose rung has no measured cost is
        skipped, and when that leaves none the default serves too.
        ``overtaken``: the last proposal was never verified (an ordinary
        decode the runner had in flight committed past it, s.4d), so it does
        not age a trial.
        """
        if not self._spec_cost_table_ready("Gemma4MTP"):
            return default, None
        policy = self._spec_policy
        if overtaken:
            policy.note_overtaken()
        n = len(drafting)
        cap = min(int(default[1]), int(k))
        top = max(int(target[3]) for target in drafting)
        plain_ms, _bucket = self._spec_plain_ms_for(live_rows, top)
        continuing = tuple(
            i for i, (r, session, _a, _p, _n) in enumerate(drafting) if self._mtp_seed_continues(session, r, last)
        )
        joining = len(continuing) < n
        # The entry horizon is capped by the steps the joining rows are expected to live: the measured request
        # length less the steps a row has had.
        remaining = None
        if joining and policy.request_steps is not None:
            lived = [session.steps for i, (_r, session, *_) in enumerate(drafting) if i not in continuing]
            remaining = max(1, int(round(policy.request_steps)) - max(lived))
        candidates = []
        for shape in self._mtp_shapes:
            if int(shape[1]) > cap or int(shape[0]) < max(1, len(continuing)):
                continue
            width = self._mtp_width_for(shape, top)
            if width is None:
                continue
            step = self._spec_record_ms(shape, width, top)
            if step is None:
                continue
            measured = policy.calibration.tokens_per_row(("step", tuple(shape), int(width)))
            if int(shape[0]) >= n:
                entry = self._spec_reseed_ms(shape, top) if joining else 0.0
                if entry is not None:
                    candidates.append(RecordCandidate(shape, int(width), step, entry, tokens_per_row=measured))
            if joining and continuing:
                partial = self._spec_record_ms(shape, width, top, partial=True)
                candidates.append(
                    RecordCandidate(
                        shape, int(width), partial + plain_ms, 0.0, members=continuing, tokens_per_row=measured
                    )
                )
        if not candidates:
            return default, None
        previous = policy.last_k
        stats = [session.stats for _r, session, *_ in drafting]
        fresh_rows = [
            (i, session)
            for i, (_r, session, *_) in enumerate(drafting)
            if i not in continuing and session.stats.spec_steps == 0 and not session.trialed
        ]
        choice = policy.decide(
            candidates, stats, plain_ms, cap, remaining, fresh=tuple(i for i, _session in fresh_rows)
        )
        self._spec_policy_note("Gemma4MTP", choice, previous, n, str(default), stats, plain_ms)
        if choice is None:
            return None, None
        for _i, session in fresh_rows:
            session.trialed = True  # one trial per request, whatever reason the step replays for
        return choice.shape, choice.members

    def _mtp_check_result(self, result, shape, live):
        if tuple(result.shape) != tuple(shape):
            raise ValueError(
                f"Gemma4MTP speculative contract: serving_step replayed record {tuple(result.shape)}, asked {shape}"
            )
        if int(result.live) != int(live):
            raise ValueError(f"Gemma4MTP speculative contract: serving_step served {result.live} rows, asked {live}")
        k_v = int(shape[1])
        if tuple(result.drafts.shape) != (int(live), k_v):
            raise ValueError(
                f"Gemma4MTP speculative contract: drafts are {tuple(result.drafts.shape)}, expected {(live, k_v)}"
            )
        if tuple(result.argmax_ids.shape) != (int(live), k_v + 1):
            raise ValueError(
                f"Gemma4MTP speculative contract: argmax_ids are {tuple(result.argmax_ids.shape)}, "
                f"expected {(live, k_v + 1)}"
            )

    def propose_draft_tokens(self, num_drafts, committed, positions, counts, hidden=None):
        """``[rows, K]`` int32 drafts for the next step from one fused replay over the live rows.

        Per live row the runner's accepted count selects the new anchor and
        its position. A row whose retained proposal came from the latest
        replay, under the row id it still has, seeds the drafter from that
        replay's verify hidden (column ``count - 1``); every other row is
        reseeded by a target forward at its anchor. Rows the captured widths,
        ``max_seq_len`` or the bounded ring cannot hold are left out and
        decline; a step above ``GEMMA4_SPEC_MAX_ROWS`` or without a covering
        record declines every row.

        A step that declines as a whole before any row has to be looked at
        (``_mtp_fast_decline_reason``: the rows exceed the largest record, or
        the records are not captured, with nothing retained and no trailing
        row) is answered from the block's column-0 positions alone
        (``_mtp_fast_decline``): no row list, no block read, no session.
        """
        self._spec_boundary("propose_start")
        try:
            del hidden
            k = int(num_drafts)
            if k < 1 or k > self._SPEC_DRAFT_LEN:
                raise ValueError(
                    f"Gemma4MTP speculative contract: num_drafts {k} outside [1, {self._SPEC_DRAFT_LEN}] "
                    f"({SPEC_DRAFT_LEN_ENV})"
                )
            rows = int(committed.shape[0]) if committed.dim() > 1 else 1
            width = int(committed.shape[1]) if committed.dim() > 1 else 1
            try:
                first = positions.reshape(int(rows), -1)[:, 0]
            except Exception:
                first = None  # ``_spec_live_rows``' own fallback: every row is live
            if first is not None:
                live_mask = first >= 0
                live_n = int(live_mask.sum())
                if not live_n:
                    return self._spec_decline(rows, k)
                top = int(first.max())  # padding rows sit at -1
                reason = self._mtp_fast_decline_reason(first, live_n, top)
                if reason is not None:
                    return self._mtp_fast_decline(rows, k, live_n, top, reason)
                live = torch.nonzero(live_mask).reshape(-1).tolist()
            else:
                live = list(range(int(rows)))
            # The blocks are read to the host once per step; per-row tensor reads cost ~8 us each, three a row
            # (the 32-row all-declined step paid ~1 ms in them).
            counts_rows = counts.reshape(-1).tolist() if counts is not None else None
            pos_rows = positions.reshape(int(rows), -1).tolist()
            tokens, committing = self._spec_committed(counts_rows, live, width)
            t_begin = self._spec_interval_begin(len(live), tokens, committing)
            self._spec_mode_now = ("plain", self._spec_plain_bucket(len(live)))
            top = max(int(pos_rows[r][0]) for r in live)
            self._spec_table_now = self._spec_plain_table_ms(len(live), top)
            try:
                return self._mtp_propose(k, rows, live, committed, positions, counts, counts_rows, pos_rows)
            finally:
                self._spec_interval_end("Gemma4MTP", t_begin, self._spec_mode_now, len(live))
        finally:
            self._spec_boundary("propose_end")

    def _mtp_propose(self, k, rows, live, committed, positions, counts, counts_rows=None, pos_rows=None):
        """``propose_draft_tokens`` proper; ``_spec_mode_now`` ends as the mode the step serves. ``counts_rows``
        and ``pos_rows`` are the host lists of ``counts`` and ``positions`` (read once by the caller)."""
        if counts_rows is None:
            counts_rows = counts.reshape(-1).tolist() if counts is not None else None
        if pos_rows is None:
            pos_rows = positions.reshape(int(rows), -1).tolist()
        committed_rows = committed.reshape(int(rows), -1).tolist()
        targets = []  # (row, session, anchor, position)
        overtaken = 0  # live rows whose last proposal an in-flight ordinary decode committed past (s.4d)
        trailing = []  # rows whose request was released and no prefill followed: the runner's trailing proposal
        for r in live:
            block = [int(x) for x in committed_rows[r]]
            n = int(counts_rows[r]) if counts_rows is not None else 1
            n = max(1, min(n, len(block)))
            anchor = block[n - 1]
            position = int(pos_rows[r][n - 1])
            if position < 0:
                raise ValueError(
                    f"Gemma4MTP speculative contract: row {r} committed {n} tokens but column {n - 1} has position -1"
                )
            session = self._mtp_session_at(r)
            if session is None and r in self._mtp_released_slots:
                # The request on this slot was released and nothing prefilled into it since: the runner's
                # proposal carries the released request's last committed block, and no verify will follow.
                # Opened as a session it was a fresh row at its first proposal, which the policy entered on the
                # measured credit (a reseed and a replay for nothing, ~70 ms after every speculating prose
                # request on the 12B QB2) and booked as a failed run when the next prefill dropped it
                # (2026-10-30, the v50 run: three prose requests plain after the one that paid).
                trailing.append(r)
                continue
            session = session or self._mtp_open_session(r)
            self._mtp_sync_session(session)
            session.steps += 1
            session.committed += n
            retained = session.retained
            if retained is not None and retained.consumed and retained.verified > 0:
                # The count answers the verify that consumed the proposal: n - 1 of its ``verified``
                # drafts were accepted (the plugin's accept walk commits a prefix plus one token).
                self._spec_policy.observe(session.stats, n - 1, retained.verified)
            elif retained is not None and not retained.consumed:
                # Nothing answered from this proposal: the runner committed the row through a step it had
                # submitted before the proposal was published (an ordinary decode overlapped under
                # asynchronous scheduling, which the runner drains before asking again) and never sends its
                # drafts, so it is dropped rather than verified (contract s.4d). Seen here, before the policy
                # decides, so the decision knows the row was overtaken rather than finding no proposal. The
                # overlapped step committed one token after the anchor: that token against the first draft is
                # one position of acceptance evidence, and when they agree the replay's verify hidden at
                # column 1 seeds the next proposal (``overtaken``); otherwise the row reseeds at it.
                self._mtp_dropped_proposals += 1
                session.dropped += 1
                overtaken += 1
                one_past = n == 1 and position == retained.anchor_position + 1
                matched = one_past and bool(retained.drafts) and int(anchor) == int(retained.drafts[0])
                if one_past:
                    self._spec_policy.observe(session.stats, 1 if matched else 0, 1)
                if self._mtp_dropped_proposals <= 200:
                    logger.debug(
                        f"Gemma4MTP speculative contract: proposal dropped unverified for row {r} (anchor "
                        f"{retained.anchor_token} at {retained.anchor_position}, the runner committed {n} to "
                        f"{position}, first draft {'matched' if matched else 'missed'}); "
                        f"{self._mtp_dropped_proposals} so far"
                    )
                if matched:
                    retained.consumed = True
                    retained.overtaken = True
                else:
                    session.retained = None
            targets.append((r, session, anchor, position, n))
        if trailing and self._mtp_trailing_logged < 8:
            self._mtp_trailing_logged += 1
            logger.debug(
                f"Gemma4MTP speculative contract: rows {trailing} proposed after their release and before a "
                "prefill (the runner's trailing proposal): no session, no replay"
            )
        if not targets:
            return self._spec_decline(rows, k)
        sessions = [target[1] for target in targets]
        if self._mtp_decoder is None or not self._mtp_captured:
            return self._mtp_decline_step(
                rows, k, live, "the verify records are not captured (warmup incomplete)", sessions=sessions
            )
        # The record is chosen by the PROPOSING rows, never by the wire batch:
        # a row the captured widths, max_seq_len or the ring cannot hold is
        # left out first, and the smallest record covering what remains is
        # replayed with those rows as its first rows; every other record row
        # is padding. Fewer rows may select a smaller record with a longer
        # draft, whose taller verify block can exclude more rows, so this
        # iterates to a fixed point (the row set only shrinks). More rows than
        # the largest record holds are judged at that record's draft length
        # (the shortest block: a row failing there fails everywhere), and
        # when they all fit the step declines as a whole.
        drafting, declined = list(targets), {}
        largest = self._mtp_shapes[-1]
        while True:
            shape = self._mtp_shape_for(len(drafting))
            check = shape if shape is not None else largest
            p_v = int(check[1]) + 1
            width_for = lambda pos, s=check: self._mtp_width_for(s, pos)  # noqa: E731
            # Every reason grows with the position (a rung past the ladder, max_seq_len, the ring wrap): when
            # the highest row passes, every row does, and no row is judged on its own.
            if self._spec_decline_reason(max(target[3] for target in drafting), p_v, width_for) is None:
                break
            kept = []
            for target in drafting:
                reason = self._spec_decline_reason(target[3], p_v, width_for)
                if reason is None:
                    kept.append(target)
                else:
                    declined[target[0]] = reason
                    target[1].retained = None
            if len(kept) != len(drafting) and kept:
                drafting = kept
                continue
            drafting = kept
            break
        if declined:
            reasons = "; ".join(f"row {r}: {reason}" for r, reason in declined.items())
            if reasons != self._mtp_decline_logged:
                logger.info(f"Gemma4MTP speculative contract: rows left out of the proposal ({reasons})")
                self._mtp_decline_logged = reasons
        if not drafting:
            return self._spec_decline(rows, k)
        if shape is None:
            return self._mtp_decline_step(
                rows,
                k,
                live,
                f"{len(drafting)} proposing rows exceed the largest record {largest} "
                f"({SPEC_MAX_ROWS_ENV}={self._mtp_max_rows})",
                sessions=sessions,
            )
        last = self._mtp_last_replay
        shape, members = self._mtp_choose_record(shape, drafting, k, last, len(live), overtaken=overtaken > 0)
        if shape is None:
            return self._mtp_decline_step(
                rows, k, live, "the adaptive policy chose the plain step", log=False, sessions=sessions
            )
        if members is not None:
            # The rows left out join no replay this step: they commit one token each from the plain decode
            # the verify runs beside the replay, and reseed when the batch enters the full record.
            for i, target in enumerate(drafting):
                if i not in members:
                    target[1].retained = None
            drafting = [drafting[i] for i in members]
        k_v = int(shape[1])
        # Row ids are the propose rows: the runner settled every slot move
        # before it asked, so a row is its state slot here, and the decoder
        # keys the previous replay's seed hidden and the per-layer tables by it.
        row_ids = [target[0] for target in drafting]
        anchors, posits, seed_cols, reseed = [], [], [], []
        for r, session, anchor, position, n in drafting:
            retained = session.retained
            seed = None
            if retained is not None and not retained.consumed:
                raise RuntimeError(
                    f"Gemma4MTP speculative contract: row {r} reached the replay with an unconsumed proposal "
                    "(the propose loop above drops or marks every overtaken proposal)"
                )
            if retained is not None:
                expected = retained.anchor_position + n
                if position != expected:
                    raise RuntimeError(
                        f"Gemma4MTP speculative contract: row {r} committed {n} tokens after the proposal at "
                        f"position {retained.anchor_position}, so its anchor sits at {expected}, "
                        f"but the runner says {position}"
                    )
                # The decoder gathers the seed from the LAST replay's verify
                # hidden by row id, so only a row that kept its id since that
                # replay can take column n - 1 there (column 1 after an
                # overtaken proposal whose first draft the overlapped step
                # committed); a moved slot reseeds.
                if retained.replay is last and last is not None and retained.row_id == r:
                    seed = n if retained.overtaken else n - 1
            if seed is None:
                reseed.append(r)
            anchors.append(int(anchor))
            posits.append(int(position))
            seed_cols.append(seed)
        decoder = self._mtp_decoder
        full, layers = self._spec_timed("tables", self._mtp_step_page_tables)
        if self._mtp_tables_changed(layers):
            self._spec_timed("refresh", decoder.serving_refresh_page_tables, layers)
            self._mtp_remember_tables(layers)
        # Both calls below rewrite the decoder's shared verify-logits buffer.
        self._mtp_replay_serial += 1
        serial = self._mtp_replay_serial
        if reseed:
            # The reseed is its own decoder call over ITS rows: its table is
            # ordered by them and zero past them, like the replay's own.
            by_row = {r: (a, p) for r, a, p in zip(row_ids, anchors, posits)}
            self._spec_timed(
                "reseed",
                decoder.serving_reseed,
                shape,
                reseed,
                [by_row[r][0] for r in reseed],
                [by_row[r][1] for r in reseed],
                self._mtp_record_rows(full, shape[0], reseed),
            )
            self._spec_note_entry(shape, self._spec_last_phase_ms, self._spec_reseed_table_ms(shape, max(posits)))
        flat = self._mtp_record_rows(full, shape[0], row_ids)
        result = self._spec_timed("replay", decoder.serving_step, shape, row_ids, anchors, posits, flat, seed_cols)
        for name, ms in decoder.serving_step_phases().items():  # the replay's own sub-phases (inside "replay")
            self._spec_policy.calibration.note_phase(name, ms)
        self._mtp_check_result(result, shape, len(drafting))
        self._mtp_last_replay = result
        self._spec_mode_now = ("step", tuple(shape), int(result.width))
        self._spec_table_now = self._spec_record_table_ms(shape, result.width, max(posits))
        if not declined:
            self._mtp_decline_logged = None  # a full proposal: the next decline is a new event
        # A row that target argmax ids cannot answer is verified in logits
        # mode, possibly steps later (the scheduler keeps the drafts of a
        # request it did not schedule), after other rows' replays have rewritten
        # the shared buffer: its logits are read back now, once for the replay.
        # A replay of greedy-eligible rows only skips the readback.
        host_logits = None
        if any(self._mtp_row_needs_logits.get(r, True) for r in row_ids):
            host_logits = self._mtp_read_logits(result, self._mtp_vocab_size())
        drafts = result.drafts.to(torch.int64)
        argmax = result.argmax_ids.to(torch.int64)
        offer = min(k_v, k)
        out = torch.zeros((rows, k), dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        for i, (r, session, anchor, position, n) in enumerate(drafting):
            row_drafts = [int(t) for t in drafts[i].tolist()]
            session.anchor = int(anchor)
            session.position = int(position)
            session.retained = MTPRetainedRow(
                replay=result,
                shape_row=i,
                row_id=r,
                serial=serial,
                anchor_token=int(anchor),
                anchor_position=int(position),
                drafts=row_drafts,
                argmax_ids=[int(t) for t in argmax[i].tolist()],
                logits=None if host_logits is None else host_logits[i],
            )
            out[r, :offer] = torch.tensor(row_drafts[:offer], dtype=torch.int32)
            valid[r] = offer
        return self._spec_draft_output(out, valid)


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4VisionForConditionalGeneration(Gemma4ForCausalLM, SupportsMultiModal):
    """Gemma4 served as text + image.

    A SEPARATE class rather than making ``Gemma4ForCausalLM`` multimodal. The
    plugin registers that name for every Gemma4 arch and its text-only-ness is
    load-bearing: without ``SupportsMultiModal`` vLLM leaves ``multimodal_config``
    unpopulated and the request path stays text-only, which is what the dFlash,
    MTP and baseline profiles serve. Flipping it globally would put a multimodal
    processor in front of all of them.

    Selected per deployment through the generic ``TT_MODEL_CLASS_OVERRIDES``
    mechanism, exactly as the speculative profiles are.

    Registering the processor is what was missing for images to reach the tower
    at all: vLLM rejected image content with "Model class Gemma4ForCausalLM has
    no registered multimodal processor" (HTTP 400) even with the tower built and
    the weights loaded. The processor itself is upstream's
    (``vllm.model_executor.models.gemma4_mm``) -- the same arrangement
    tt_transformers uses for Gemma3 -- so image preprocessing and placeholder
    handling stay vLLM's, and only the encode stays ours.
    """


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4UnifiedProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4UnifiedVisionForConditionalGeneration(Gemma4VisionForConditionalGeneration):
    """Gemma4 "unified" (12B) served as text + image.

    Same serving path as ``Gemma4VisionForConditionalGeneration``; the only
    difference is upstream's ``Gemma4UnifiedProcessingInfo`` (unified config,
    ``num_soft_tokens`` instead of ``default_output_length``), which the
    processor registration is keyed on. The encoder-free embedder is picked by
    the checkpoint's weights (``model.vision_embedder.*``), not by this class.
    """


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4MTPContractVisionForConditionalGeneration(
    Gemma4MTPContractForCausalLM, Gemma4VisionForConditionalGeneration
):
    """Text + image serving with the assistant drafter on the speculative contract.

    A mixin composition: the contract class overrides ``decode_forward``,
    ``propose_draft_tokens``, the warmups and the lifecycle hooks, and its
    ``prefill_forward`` only closes the prefilled slots' generations before
    ``super().prefill_forward``, which resolves to
    ``Gemma4ForCausalLM.prefill_forward`` and its multimodal staging
    (``_stage_multimodal_prefill``). The drafter needs nothing modality
    specific: it reads the target's KV, RoPE caches and the anchor token's
    text embedding, and seeds from a target forward, never from prefill
    taps. The processor registration keys on this class, as it does for
    ``Gemma4VisionForConditionalGeneration``.
    """


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4UnifiedProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4MTPContractUnifiedVisionForConditionalGeneration(
    Gemma4MTPContractForCausalLM, Gemma4UnifiedVisionForConditionalGeneration
):
    """Gemma4 "unified" (12B) text + image serving with the assistant drafter on the contract.

    Same composition as ``Gemma4MTPContractVisionForConditionalGeneration``
    over ``Gemma4UnifiedVisionForConditionalGeneration``; the registration
    carries upstream's ``Gemma4UnifiedProcessingInfo``.
    """


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class Gemma4DFlashContractVisionForConditionalGeneration(
    Gemma4DFlashContractForCausalLM, Gemma4VisionForConditionalGeneration
):
    """Text + image/video serving with the dFlash drafter on the speculative contract (31B).

    The dFlash prefill override captures the residual taps around an explicit
    ``Gemma4ForCausalLM.prefill_forward`` call, which is where the multimodal
    staging lives (``_stage_multimodal_prefill`` replaces the embedding step
    by the merged ``[prompt_len, hidden]`` embeddings and forces the eager
    prefill the dFlash rail forces anyway), followed by the lane router and
    the chunk plan; the taps come from the decoder-layer loop and cover the
    prefilled rows whatever produced their embeddings. The override adds one
    multimodal rule: a continuation whose restart the base moves into a
    vision item (``_dflash_mm_restart_moves``) is prefilled without taps.
    The processor registration keys on this class, as it does for
    ``Gemma4VisionForConditionalGeneration``.
    """
