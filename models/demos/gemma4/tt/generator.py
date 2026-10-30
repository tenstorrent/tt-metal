# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.sampling import SamplingParams, slice_sampling_params
from models.common.sampling.tt_log_probs import LogProbsResult
from models.demos.gemma4.tt.common import GEMMA4_CP_PREFILL_CHUNK, create_tt_model, gemma4_cp_prefill_engaged
from models.demos.gemma4.tt.generator_trace import (
    apply_gemma4_prefill_trace_policy,
    chunked_prefill_trace_enabled,
    maybe_disable_pli_prefill_trace,
    patch_gemma4_trace_model_args,
    resolve_gemma4_prefill_chunk_size,
    resolve_gemma4_prefill_trace_enable,
    should_auto_enable_chunked_bounded,
    warmup_gemma4_model_prefill,
)
from models.demos.gemma4.tt.lane_prefill import lane_lmhead_gather_enabled, lane_prefill_table_upload_enabled
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.vision.embedders import (
    Gemma4MultimodalEmbedder,
    Gemma4UnifiedVisionEmbedder,
    has_unified_vision_weights,
)
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.demos.gemma4.tt.vision.vision_tower import VisionTower
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import (
    Mode,
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.generator import (
    MAX_BATCHED_PREFILL_SEQ_LEN,
    SUPPORTED_PREFILL_BATCH_SIZES,
    Generator,
    _pad_or_create_page_table,
    batched_prefill_padded_batch,
)
from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys_multimodal
from models.tt_transformers.tt.model_config import determine_device_name

# ── Vision tower weight conversion (HF -> Meta, per-block RoPE) ─────────────
# Replicated from models/demos/gemma4/tests/unit/test_vision_attention.py so
# the runtime does not import from tests/. Gemma-4 vision uses 2D (per-block)
# RoPE, so q/k weights and q/k norms are interleaved per RoPE block, not across
# the full head_dim.
_VISION_BLOCK_PROJ_RENAMES = [
    (".q_proj.", ".wq."),
    (".k_proj.", ".wk."),
    (".v_proj.", ".wv."),
    (".o_proj.", ".wo."),
    (".gate_proj.", ".w1."),
    (".up_proj.", ".w3."),
    (".down_proj.", ".w2."),
]


def _meta_permute_qk_weight(weight, n_heads, head_dim, ndim=2):
    blk = head_dim // ndim
    in_dim = weight.shape[-1]
    weight = weight.view(n_heads, ndim, 2, blk // 2, in_dim)
    weight = weight.transpose(2, 3)
    return weight.reshape(n_heads * head_dim, in_dim)


def _meta_permute_norm_weight(weight, head_dim, ndim=2):
    blk = head_dim // ndim
    weight = weight.view(ndim, 2, blk // 2)
    weight = weight.transpose(1, 2)
    return weight.reshape(head_dim)


def _convert_vision_block_hf_to_meta(state_dict, n_heads, n_kv_heads, head_dim, ndim=2):
    converted = {}
    for key, tensor in state_dict.items():
        if "q_proj.linear.weight" in key:
            tensor = _meta_permute_qk_weight(tensor, n_heads, head_dim, ndim)
        elif "k_proj.linear.weight" in key:
            tensor = _meta_permute_qk_weight(tensor, n_kv_heads, head_dim, ndim)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            tensor = _meta_permute_norm_weight(tensor, head_dim, ndim)
        new_key = key
        for old, new in _VISION_BLOCK_PROJ_RENAMES:
            new_key = new_key.replace(old, new)
        converted[new_key] = tensor
    # NOTE: deliberately NOT calling map_hf_to_meta_keys here — it would rename
    # self_attn->attention and the layernorms, which the Gemma-4 vision modules
    # do NOT expect (they keep self_attn / input_layernorm / post_*_layernorm).
    return converted


def _build_vision_state_dict(full_state_dict, vision_args):
    """Extract ``model.vision_tower.*`` from the full checkpoint and convert to
    the ``visual.*`` Meta-format keys ``VisionTower`` expects."""
    prefix = "model.vision_tower."
    # ``.contiguous()``: a cold load hands out views of the checkpoint's file mapping; take the
    # tower's ~1.1 GB into memory here rather than page by page during the device writes.
    vision_sd = {k[len(prefix) :]: v.contiguous() for k, v in full_state_dict.items() if k.startswith(prefix)}
    vision_sd = standardize_hf_keys_multimodal(vision_sd)
    vision_sd = _convert_vision_block_hf_to_meta(
        vision_sd, vision_args.n_heads, vision_args.n_kv_heads, vision_args.head_dim
    )
    return {f"visual.{k}": v for k, v in vision_sd.items()}


VISION_KEY_PREFIXES = ("model.vision_tower.", "model.vision_embedder.", "model.embed_vision.", "model.embed_audio.")


def load_vision_state_dict(model_path):
    """The vision-side tensors of a Gemma-4 checkpoint, read for real from safetensors.

    The text model's warm weight cache (models/common/weight_cache.py) replaces the HF
    state dict with a dataless placeholder (``torch.empty`` per manifest entry) once its
    tensorbins are complete. The vision encoders are not part of that cache, so on a
    warm run they must read their weights from the checkpoint themselves: the 31B
    SigLIP tower and the 12B unified embedder (~1.1 GB / ~80 MB) via the safetensors
    index, selecting only the vision keys. Hub ids resolve to the local snapshot.
    """
    import glob
    from pathlib import Path

    path = Path(model_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(model_path), local_files_only=True))
    files = sorted(glob.glob(str(path / "*.safetensors")))
    if not files:
        # The served checkpoint (tt-inference-server's MODEL_WEIGHTS_DIR / HF_MODEL
        # dir) when the hub snapshot vLLM resolved holds only config + tokenizer.
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        served = Gemma4ModelArgs.served_weights_dir()
        if served:
            logger.info(f"No safetensors under {path}; reading the vision tensors from {served}")
            path = Path(served)
            files = sorted(glob.glob(str(path / "*.safetensors")))
    from safetensors import safe_open

    out = {}
    t0 = time.time()
    for fn in files:
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if k.startswith(VISION_KEY_PREFIXES):
                    t = f.get_tensor(k)
                    # Materialise now: safetensors hands out views of the file mapping, and a
                    # checkpoint on a network mount would otherwise be paged in piecemeal
                    # while the tower's weights are written to the device (a silent stall).
                    out[k] = (t.to(torch.bfloat16) if t.dtype == torch.float32 else t).clone()
    if not out:
        raise FileNotFoundError(f"no vision tensors ({VISION_KEY_PREFIXES}) under {path}")
    total = sum(t.numel() * t.element_size() for t in out.values())
    logger.info(f"Read {len(out)} vision tensors ({total / 1e9:.2f} GB) from {path} in {time.time() - t0:.1f}s")
    return out


def vision_state_dict_for(state_dict, model_path):
    """``state_dict`` itself when it carries real vision tensors, else the real ones from disk."""
    if getattr(state_dict, "is_placeholder", False):
        logger.info("Warm text weight cache: loading the vision tensors from the checkpoint")
        return load_vision_state_dict(model_path)
    probe = [k for k in state_dict if k.startswith(VISION_KEY_PREFIXES)][:2]
    for k in probe:
        v = state_dict[k].float()
        if not torch.isfinite(v).all() or v.abs().max() > 1e6:
            logger.warning(f"vision tensor {k} is not real data; reloading the vision tensors from the checkpoint")
            return load_vision_state_dict(model_path)
    return state_dict


def _vision_encoder_seq_len(num_patches: int, max_qkv_mm_seq_len: int = 2048) -> int:
    """Padded encoder seq length >= num_patches that the vision blocks can run at.

    The attention chunks its qkv matmul by ``max_qkv_mm_seq_len`` rows above that
    length and its output projection by 1024, the MLP by 512, so a sequence above
    ``max_qkv_mm_seq_len`` must be a multiple of it; below, a multiple of 1024 is
    enough. That halves the rows a 630-patch video frame runs at (1024, not 2048);
    a 2520-patch image still pads to 4096.
    """
    if num_patches <= max_qkv_mm_seq_len:
        return max(1024, -(-num_patches // 1024) * 1024)
    return -(-num_patches // max_qkv_mm_seq_len) * max_qkv_mm_seq_len


# Same 128k batched-prefill token ceiling as the shared Generator
# (padded_batch × padded_prefill_seq_len).
GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN = MAX_BATCHED_PREFILL_SEQ_LEN

# Full-attention chunked SDPA (head_dim>=512) requires chunk_start_idx % 128 == 0.
# vLLM token-chunked continuations can land on unaligned start_pos (e.g. 48);
# align down and re-prefill the prefix (Galaxy SDPA_CHUNK_ALIGN pattern).
SDPA_CHUNK_ALIGN = 128


def align_num_cached_tokens_to_sdpa(num_cached_per_user: list[int]) -> list[int]:
    """Align cached-prefix lengths down to ``SDPA_CHUNK_ALIGN`` for chunked SDPA.

    When page_size < align (e.g. 64), rounding down re-computes boundary tokens
    (harmless: same KV is overwritten). Returns a new list.
    """
    aligned = []
    for idx, n in enumerate(num_cached_per_user):
        n = int(n)
        if n > 0:
            a = (n // SDPA_CHUNK_ALIGN) * SDPA_CHUNK_ALIGN
            if a != n:
                logger.info(
                    "SDPA chunk alignment: user {} cached {} -> {} (aligned to {})",
                    idx,
                    n,
                    a,
                    SDPA_CHUNK_ALIGN,
                )
                n = a
        aligned.append(n)
    return aligned


# ── Pure helpers of the vLLM bridge (host-tested without vLLM: generator_vllm imports vLLM at module scope) ──


def filter_warmup_sampling_params(configs, sampler):
    """Drop warmup sampling configs a sampler cannot run.

    ``sampler._allow_topk_sampling`` False: no non-greedy config at all (they all need the
    top-k/top-p program). ``sampler._allow_penalties_sampling`` False: no config with penalties.
    Greedy (temperature 0) and ``None`` always stay.
    """
    allow_penalties = bool(getattr(sampler, "_allow_penalties_sampling", True))
    allow_topk = bool(getattr(sampler, "_allow_topk_sampling", True))
    kept = []
    for cfg in configs:
        if cfg is None:
            kept.append(cfg)
            continue
        temperature = cfg.temperature
        first = temperature[0] if isinstance(temperature, (list, tuple)) else temperature
        if first == 0.0:
            kept.append(cfg)
            continue
        if not allow_topk:
            continue
        penalties = cfg.presence_penalty
        has_penalties = penalties is not None and (
            any(p not in (None, 0.0) for p in penalties) if isinstance(penalties, (list, tuple)) else penalties != 0.0
        )
        if not allow_penalties and has_penalties:
            continue
        kept.append(cfg)
    return kept


def merge_soft_tokens(input_ids: torch.Tensor, soft_tokens, pad_token_id: int, *, num_cached: int = 0):
    """Line up every modality's soft tokens with its placeholder slots in one prompt.

    ``soft_tokens`` is a list of ``(placeholder_token_id, embeds)`` with ``embeds`` a
    host ``[num_valid, text_hidden]`` tensor in prompt order for that modality.
    Returns ``(llm_ids, positions, merged_embeds)``: the prompt ids with every
    placeholder replaced by ``pad_token_id``, the placeholder positions sorted
    ascending across modalities (``int32 [num_valid]``) and the soft tokens in
    that same order (``bf16 [num_valid, text_hidden]``), ready for one scatter.

    ``num_cached`` (vLLM prefix caching): an item whose placeholders all sit inside
    the cached prefix arrives without its payload, so a modality may carry fewer
    soft tokens than placeholders. They then fill that modality's LAST placeholders
    (items are forwarded in prompt order, cached ones first) and every placeholder
    past the cached prefix must receive one. Any other count mismatch raises.
    """
    input_ids = input_ids.reshape(-1).to(torch.long)
    placeholder_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    positions_per_modality = []
    embeds_per_modality = []
    for token_id, embeds in soft_tokens:
        if token_id is None:
            raise ValueError("soft tokens for a modality this checkpoint has no placeholder id for")
        slot_mask = input_ids == token_id
        slots = torch.where(slot_mask)[0]
        n_slots, n_soft = int(slots.numel()), int(embeds.shape[0])
        if n_soft != n_slots:
            uncovered = slots[: max(n_slots - n_soft, 0)]
            if n_soft > n_slots or num_cached <= 0 or int(uncovered.max()) >= num_cached:
                raise ValueError(
                    f"Placeholder count for token id {token_id} ({n_slots}) != soft tokens ({n_soft}). "
                    "The processor's placeholder run lengths and the embedder's valid-token counts must agree "
                    "(a shortfall is only valid for items fully inside the cached prefix)."
                )
            slots = slots[n_slots - n_soft :]
        positions_per_modality.append(slots)
        embeds_per_modality.append(embeds.to(torch.bfloat16))
        placeholder_mask |= slot_mask
    if not positions_per_modality:
        raise ValueError("no soft tokens to merge")
    positions = torch.cat(positions_per_modality)
    order = torch.argsort(positions)
    positions = positions[order].to(torch.int32)
    merged_embeds = torch.cat(embeds_per_modality, dim=0)[order].contiguous()
    llm_ids = input_ids.clone()
    llm_ids[placeholder_mask] = pad_token_id
    return llm_ids, positions, merged_embeds


def items_within_prefill_window(offsets, lengths, window_end: int):
    """Indices of a request's multimodal items whose placeholders end inside ``window_end``.

    The plugin forwards every item of a scheduled request on every prefill step, while
    a chunked prefill only covers the prompt up to the chunk's end. An item whose
    placeholders start past that end has no slots yet and belongs to a later chunk;
    merging it now would fail the placeholder count. vLLM never ends a chunk inside an
    item (``disable_chunked_mm_input``), so an item is either fully inside the window or
    fully past it. ``offsets`` / ``lengths`` are the per-item ``mm_position`` lists the
    plugin forwards (``mm_placeholder_offsets`` / ``mm_placeholder_lengths``); an item
    without a range (older plugins) is kept.
    """
    n = max(len(offsets or ()), len(lengths or ()))
    kept = []
    for i in range(n):
        offset = offsets[i] if offsets is not None and i < len(offsets) else None
        length = lengths[i] if lengths is not None and i < len(lengths) else None
        if offset is not None and length is not None and int(offset) + int(length) > int(window_end):
            continue
        kept.append(i)
    return kept


# Max users in one true-batched prefill forward.
#
# Historically 4 everywhere: B>=8 was measured to wedge indefinitely after the
# first all_gather on P150x8 / 12B (eager or traced). That wedge no longer
# reproduces on Blackhole with the current stack -- B=32 true-batched prefill
# completes and produces coherent output on both 12B and 31B -- while the cap
# itself is expensive, forcing 32 users through 11 sequential prefill passes:
#
#   12B / P150x8 / batch-32   cap=4: TTFT 3879 ms   no cap: TTFT  850 ms
#   31B / P150x8 / batch-32   cap=4: TTFT 6578 ms   no cap: TTFT 1361 ms
#
# decode unchanged in both cases. The same cap also starved the vLLM server on
# a 4-chip P300x2 mesh (16 sequential prefills per scheduler step), which hung
# the 31B QB2 release until it was lifted.
#
# Raised on Blackhole only. Wormhole keeps 4: the original wedge has not been
# re-measured there, and WH carries 12 GB/ASIC vs BH's 32 GB, so a wider
# batched prefill has far less headroom. Override on any arch with
# GEMMA4_MAX_BATCHED_PREFILL_USERS (0 = no user cap).
#
# NOTE: the virtual-token ceiling (GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN) still
# applies on top of this, so long prompts stay microbatched regardless.
_DEFAULT_MAX_BATCHED_PREFILL_USERS = 4
_BLACKHOLE_MAX_BATCHED_PREFILL_USERS = 32


def _default_max_batched_prefill_users() -> int:
    """Arch-gated cap default; never raises if arch cannot be determined."""
    try:
        from models.common.utility_functions import is_blackhole

        if is_blackhole():
            return _BLACKHOLE_MAX_BATCHED_PREFILL_USERS
    except Exception:
        pass
    return _DEFAULT_MAX_BATCHED_PREFILL_USERS


def max_batched_prefill_users() -> int:
    raw = os.environ.get("GEMMA4_MAX_BATCHED_PREFILL_USERS")
    if raw is None:
        return _default_max_batched_prefill_users()
    val = int(raw)
    return val if val > 0 else 10**9


def resolve_batched_prefill_chunk_users(padded_batch: int, prefill_seq_len: int) -> int:
    """Largest user chunk that respects the 128k token ceiling and the B≤4 hang cap."""
    max_by_tokens = max(1, GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN // max(prefill_seq_len, 1))
    max_by_users = max_batched_prefill_users()
    chunk = min(padded_batch, max_by_tokens, max_by_users)
    while chunk > 1 and chunk * prefill_seq_len >= GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN:
        chunk //= 2
    # Prefer a supported padded batch size so each chunk stays on the fast path.
    supported = [b for b in SUPPORTED_PREFILL_BATCH_SIZES if b <= chunk]
    return supported[-1] if supported else 1


# ── Packed multi-prompt prefill (GEMMA4_PACKED_PREFILL=1) ────────────────────────────────────────────────────────
# N cold prompts of one padded bucket (e.g. 32 x 128 = 4096 tokens, the vLLM max_num_batched_tokens) run as ONE
# prefill forward instead of N per-user calls: the rows are concatenated into a [1, 1, N*S, H] token stream for the
# dense projections, the shared MLP, the router and the expert chunks (which see packed rows as plain tokens), and
# attention reshapes to [N, heads, S, D] so SDPA is causal per segment (the batch dim is the segment: no
# block-diagonal mask, sliding and full layers keep their own program configs, head_dim 256 / 512 unchanged). The
# K/V of every user is written through its own page-table row (``paged_fill_cache(batch_idx=slot)``) and the
# per-user last-token rows are gathered once for a single lm_head call. Eager only; cold prompts only
# (``num_cached == 0``); prompts longer than one prefill chunk stay per user.
_GEMMA4_PACKED_PREFILL_MM_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "input_features",
    "input_features_padded",
    "images",
    "image_embeds",
)


def gemma4_packed_prefill_enabled() -> bool:
    """``GEMMA4_PACKED_PREFILL=1``: cold prompts of one padded bucket prefill as one packed forward."""
    return os.environ.get("GEMMA4_PACKED_PREFILL", "0").lower() in ("1", "true", "yes")


def _packed_prefill_value_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(v is not None for v in value)
    return True


def packed_prefill_max_tokens(max_prefill_chunk_size: int) -> int:
    """Padded tokens one packed call may carry: ``GEMMA4_PACKED_PREFILL_MAX_TOKENS`` or the model's single-pass
    prefill budget (``max_prefill_chunk_size``, the bound the stock batched path applies too: the flattened
    ``[B*S, H]`` activations and the ``[B*S, E]`` routing are what the device has to hold at once)."""
    raw = int(os.environ.get("GEMMA4_PACKED_PREFILL_MAX_TOKENS", "0"))
    return raw if raw > 0 else int(max_prefill_chunk_size)


def plan_packed_prefill_groups(
    prompt_lens,
    num_cached_per_user,
    *,
    max_prefill_chunk_size: int,
    empty_slots=None,
    max_batch_size: int = 32,
    min_users: int = 2,
    max_tokens: int | None = None,
):
    """Host-only plan of the packed calls: ``[(padded_len, [request idx, ...]), ...]`` (possibly empty).

    Every cold request (``num_cached == 0``) whose padded length fits one prefill chunk joins its padded bucket;
    a bucket with at least ``min_users`` requests is split, in request order, into calls that respect the
    batched user / kernel caps (``resolve_batched_prefill_chunk_users``) and the per-call token budget
    (``max_tokens``, default ``packed_prefill_max_tokens``). Requests outside every group (cached
    continuations, long prompts, lone bucket members) stay on the stock per-user path.
    """
    if max_tokens is None:
        max_tokens = packed_prefill_max_tokens(max_prefill_chunk_size)
    buckets: dict[int, list[int]] = {}
    for i, (plen, cached) in enumerate(zip(prompt_lens, num_cached_per_user)):
        plen, cached = int(plen), int(cached)
        if cached > 0 or plen <= 0:
            continue
        padded = int(get_padded_prefill_len(plen))
        if padded > int(max_prefill_chunk_size):
            continue
        buckets.setdefault(padded, []).append(i)
    groups = []
    for padded in sorted(buckets):
        idxs = buckets[padded]
        if len(idxs) < min_users:
            continue
        slots = [int(empty_slots[i]) for i in idxs] if empty_slots is not None else idxs
        span = batched_prefill_padded_batch(len(idxs), slots, max_batch_size)
        cap = min(resolve_batched_prefill_chunk_users(min(span, max_batch_size), padded), int(max_tokens) // padded)
        if span > max_batch_size or cap < min_users:
            continue
        for start in range(0, len(idxs), cap):
            chunk = idxs[start : start + cap]
            if len(chunk) >= min_users:
                groups.append((padded, chunk))
    return groups


def mask_page_table_columns_past_allocation(page_tables, prompt_lens, block_sizes):
    """Zero every page-table column past each request's own allocation.

    Plugin rows keep earlier occupants' block ids past the live prefix and the padded K/V fill follows them,
    so route those writes to vLLM's null block 0. Returns ``(tables, zeroed_columns)``; inputs are not mutated.
    """
    if page_tables is None or prompt_lens is None:
        return page_tables, 0
    try:
        lens = [int(p) for p in (prompt_lens.tolist() if hasattr(prompt_lens, "tolist") else list(prompt_lens))]
    except (TypeError, ValueError):
        return page_tables, 0
    out = []
    zeroed = 0
    done = {}
    for i, pt in enumerate(page_tables):
        bs = block_sizes[i] if i < len(block_sizes) else None
        if not isinstance(pt, torch.Tensor) or bs is None or int(bs) <= 0:
            out.append(pt)
            continue
        key = (id(pt), int(bs))
        if key in done:
            out.append(done[key])
            continue
        table = pt if pt.dim() > 1 else pt.unsqueeze(0)
        masked = None
        for r in range(min(int(table.shape[0]), len(lens))):
            if lens[r] <= 0:
                continue
            first_free = -(-lens[r] // int(bs))
            if first_free >= int(table.shape[1]):
                continue
            stale = int((table[r, first_free:] != 0).sum())
            if stale == 0:
                continue
            if masked is None:
                masked = table.clone()
            masked[r, first_free:] = 0
            zeroed += stale
        result = pt if masked is None else (masked if pt.dim() > 1 else masked[0])
        done[key] = result
        out.append(result)
    return out, zeroed


def paged_block_size_for_view(cache_shape, head_dim, num_local_kv_heads):
    """Tokens per page-table column the paged ops address a ``[blocks, cache_heads, block, cache_hd]`` K/V buffer
    with from a layer whose per-chip view is ``(num_local_kv_heads, head_dim)``: the per-block element count is the
    invariant (``attention/operations.effective_block_size``), so this is the declared block size exactly when the
    two views agree and the page-table width must follow it when they do not."""
    _, cache_heads, block, cache_hd = (int(v) for v in list(cache_shape)[-4:])
    numerator = cache_heads * block * cache_hd
    denominator = max(1, int(num_local_kv_heads)) * int(head_dim)
    if numerator % denominator:
        raise ValueError(
            f"KV-cache view ({cache_heads} heads x {block} x {cache_hd}) is not addressable as "
            f"({num_local_kv_heads} heads x {head_dim})"
        )
    return numerator // denominator


def layer_local_kv_heads(attn, cfg):
    """KV heads an attention layer holds per chip, as ``attention/prefill.py`` derives them for its paged ops."""
    tp = max(1, int(getattr(getattr(attn, "mesh_config", None), "tp", 1) or 1))
    if getattr(getattr(attn, "weights", None), "kv_replicated", False):
        return 1
    return max(1, int(cfg.num_key_value_heads) // tp)


def layer_paged_block_size(attn, cfg, cache):
    """The block size ``attn`` addresses ``cache`` with (``paged_block_size_for_view`` on the cache's padded shape)."""
    shape = getattr(cache, "padded_shape", None) or cache.shape
    return paged_block_size_for_view(shape, int(cfg.head_dim), layer_local_kv_heads(attn, cfg))


def _load_text_tokenizer(model_path):
    # The 12B tokenizer config can advertise multimodal extra_special_tokens as
    # a list (for example ["<|video|>"]), while this transformers version expects
    # a dict. The text-only demo does not need those model-specific aliases.
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, extra_special_tokens={})
    except (ValueError, OSError, EnvironmentError) as e:
        # The whole Gemma4 family shares one identical tokenizer, but some
        # checkpoints (e.g. gemma-4-31B-it) ship without local tokenizer files.
        # On an offline box AutoTokenizer can't fetch them and raises a misleading
        # "need sentencepiece/tiktoken" error. Fall back to an explicit source
        # (GEMMA4_TOKENIZER) or the 12B tokenizer, which is byte-identical.
        fallback = os.environ.get("GEMMA4_TOKENIZER", "google/gemma-4-12B-it")
        if os.path.normpath(str(fallback)) == os.path.normpath(str(model_path)):
            raise
        logger.warning(
            f"Tokenizer load from '{model_path}' failed ({type(e).__name__}: {e}); "
            f"falling back to the shared Gemma4 tokenizer '{fallback}'. "
            f"Override with GEMMA4_TOKENIZER."
        )
        return AutoTokenizer.from_pretrained(fallback, trust_remote_code=True, extra_special_tokens={})


def _patch_model_args(
    model_args,
    mesh_device,
    max_batch_size,
    max_seq_len,
    model_path,
    tokenizer,
    has_per_layer_inputs=False,
    bounded_sliding=False,
):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.max_context_len = max_seq_len
    # Prefill chunking — default to policy multi-chunk (usually 4096).
    # P150x8 / 31B / 128k unbounded + chunk=4096 is fast (TTFT ~31s) but collapses
    # to "lapped…"; policy auto-bounds at 128k and selects chunk=2048 (QB2 path).
    # Overrides: GEMMA4_GEN_PREFILL_CHUNK=<n>, GEMMA4_DEMO_SINGLE_CHUNK=1 (legacy
    # full-ISL single chunk for A/B / correctness — avoid on long ISL).
    _chunk_override = int(os.environ.get("GEMMA4_GEN_PREFILL_CHUNK", "0"))
    if _chunk_override <= 0 and gemma4_cp_prefill_engaged(mesh_device):
        _chunk_override = GEMMA4_CP_PREFILL_CHUNK
    _force_single = os.environ.get("GEMMA4_DEMO_SINGLE_CHUNK", "0") != "0"
    _needs_chunk_for_dram = (not _force_single) and should_auto_enable_chunked_bounded(
        max_seq_len,
        mesh_device,
        model_path,
        bounded_sliding=bounded_sliding,
    )
    if _chunk_override > 0:
        model_args.max_prefill_chunk_size = _chunk_override
    elif _force_single:
        model_args.max_prefill_chunk_size = 1 << max(int(max_seq_len - 1).bit_length(), 11)
    else:
        from models.demos.gemma4.tt.generator_trace import GEMMA4_DEFAULT_PREFILL_CHUNK

        model_args.max_prefill_chunk_size = resolve_gemma4_prefill_chunk_size(
            max_seq_len,
            mesh_device=mesh_device,
            non_qb2_default=GEMMA4_DEFAULT_PREFILL_CHUNK,
            model_name_or_path=model_path,
            bounded_sliding=bounded_sliding,
        )
        if _needs_chunk_for_dram:
            logger.warning(
                f"Bounded long-context (model={Path(model_path).name}, max_seq_len={max_seq_len}): "
                f"multi-chunk prefill (chunk={model_args.max_prefill_chunk_size}) per policy to "
                f"avoid single-chunk DRAM OOM."
            )
        else:
            logger.info(
                f"Prefill multi-chunk default (chunk={model_args.max_prefill_chunk_size}, "
                f"max_seq_len={max_seq_len}). Set GEMMA4_DEMO_SINGLE_CHUNK=1 for full-ISL single chunk."
            )
    model_args.mesh_device = mesh_device
    model_args.device_name = determine_device_name(mesh_device)
    model_args.model_name = model_path
    model_args.base_model_name = Path(model_path).name
    model_args.tokenizer = tokenizer
    model_args.processor = None
    patch_gemma4_trace_model_args(model_args, prefill_trace_enabled=True)
    model_args.is_llama_vision = lambda: False

    def _encode_prompt(prompt, instruct=False):
        if instruct and getattr(tokenizer, "chat_template", None):
            # tokenize=True can return a BatchEncoding (dict) depending on the
            # transformers/tokenizer version; return_dict=False forces a plain
            # List[int], which is what preprocess_inputs_prefill / torch.tensor
            # expect.
            out = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            # Defensive: some versions still hand back a dict-like with input_ids.
            return out["input_ids"] if isinstance(out, dict) else out
        return tokenizer.encode(prompt, add_special_tokens=True)

    model_args.encode_prompt = _encode_prompt


def _g4_dram_bottom_up_bytes(view) -> int:
    """Bytes allocated below the largest free gap: the bottom-up region where trace intermediates live.

    Kernel binaries are allocated top-down (above the gap) and never collide with a replay, so they are
    left out; a persistent tensor placed into a hole below the gap counts, since that hole was free (and
    therefore usable by the trace) at capture time. Falls back to the allocated total without a block table.
    """
    try:
        table = list(view.block_table)
    except Exception:  # noqa: BLE001
        table = []
    if not table:
        return int(view.total_bytes_allocated_per_bank)
    free_start, free_size = 0, -1
    for b in table:
        if str(b.get("allocated", "")).lower() in ("no", "false", "0"):
            size = int(b.get("size", 0))
            if size > free_size:
                free_start, free_size = int(b.get("address", 0)), size
    total = 0
    for b in table:
        if str(b.get("allocated", "")).lower() in ("yes", "true", "1"):
            addr = int(b.get("address", 0))
            if free_size < 0 or addr < free_start:
                total += int(b.get("size", 0))
    return total


class ChunkedPrefillPageTableGuardMixin:
    """Gemma4 prefill guards + async decode continuity (no ``tt_transformers`` edits).

    - Trim over-wide page tables so pad width stays non-negative (vLLM hybrid).
    - Own the eager single-/multi-chunk loop (``_prefill_forward_single_user_text_eager``)
      for bounded last-token length, ring-aligned last-chunk expand, and
      intermediate ``get_last_token=-1``.
    - Eager on-device sampling writeback into padded decode tokens (#51186) so
      non-PLI async decode stays coherent without shared-generator changes.
    - Safe async-ahead token merge on ``decode_forward`` (bucket / OOB
      ``slot_remap`` host fallback) via :mod:`models.demos.gemma4.tt.async_decode`.
    - Batch-keyed decode traces via ``_decode_forward_trace_text`` keyed by
      ``(on_device_sampling, batch)`` so B=1 and B=max stay separate without
      changing shared ``Generator`` used by all models.
    - Sequential multi-user prefill: slice hybrid per-layer page tables to the
      active 1-row (tt_transformers forces ``user_id=0`` with a sliced legacy
      ``page_table``; full-attn must not keep reading batch row 0).
    - Mixed into demo (:class:`Gemma4Generator`) and vLLM (``Gemma4ForCausalLM``).
    """

    @staticmethod
    def _page_table_row_matches(page_table_1row, page_tables_per_layer, row) -> bool:
        """True when some multi-row per-layer table has ``row`` equal to the slice's prefix."""
        pt = page_table_1row if page_table_1row.dim() > 1 else page_table_1row.unsqueeze(0)
        pt32 = pt[0].to(dtype=torch.int32)
        for ref in page_tables_per_layer:
            if not (isinstance(ref, torch.Tensor) and ref.dim() > 1 and int(ref.shape[0]) > row):
                continue
            cols = min(int(pt32.shape[0]), int(ref.shape[1]))
            if cols > 0 and torch.equal(pt32[:cols], ref[row, :cols].to(dtype=torch.int32)):
                return True
        return False

    def _prepare_decode_trace_once(self, kv_cache, page_table, on_device_sampling):
        """Opt out of the hoisted decode-trace preparation (upstream #53551).

        ``Generator._prepare_decode_trace_once`` prepares the decode trace from
        inside ``prefill_forward_text`` and sizes it with
        ``batch_size = page_table.shape[0]`` -- the *prefill* page table. Gemma4
        keeps its per-layer page tables padded to ``max_batch_size`` on purpose
        (see ``_pad_page_tables_batch_to_max``: decode traces are captured against
        persistent buffers sized at max batch so addresses never grow), so any
        prefill narrower than max batch makes the hoisted decode compile run a
        batch-1 input against 32-row page tables::

            TT_FATAL: Batch size between page_table and input_tensor must match
                      (paged_update_cache_device_operation)

        That aborts EngineCore during prefill warmup, so the server never comes
        up whenever max_num_seqs > 1 (max_num_seqs=1 happens to match and is why
        the vLLM nightly leg still boots).

        Upstream already treats this hoist as optional -- ``_uses_prefetcher``
        returns early for models it does not suit, noting that "keeping these
        models on the original non-hoisted path costs this change nothing it
        targets". Gemma4 does its own decode warmup at the correct batches in
        ``warmup_model_decode``, so skipping the hoist restores exactly the
        behaviour it had before #53551 landed.
        """
        return

    def _activate_sequential_per_layer_row(self, page_table) -> None:
        """Slice the multi-row hybrid page-table stash to the active sequential user.

        The shared loop prefills users in row order with ``user_id=0``, so the row cursor identifies the request
        and the 1-row slice only confirms it (slice contents are not unique under lane rings).
        """
        if page_table is None or not isinstance(page_table, torch.Tensor):
            return
        pt = page_table if page_table.dim() > 1 else page_table.unsqueeze(0)
        if int(pt.shape[0]) != 1:
            return
        for m in self.model:
            if getattr(m, "_sequential_row_active", False):
                continue  # nested call inside this request's traced entry
            active = getattr(m, "_active_page_tables_per_layer", None)
            if not active:
                continue
            batch_host = getattr(m, "_sequential_batch_page_tables", None)
            if batch_host is None:
                if not any(isinstance(p, torch.Tensor) and p.dim() > 1 and int(p.shape[0]) > 1 for p in active):
                    continue
                m._sequential_batch_page_tables = active
                m._sequential_row_cursor = 0
                batch_host = active
            row = int(getattr(m, "_sequential_row_cursor", 0))
            rows = max(int(p.shape[0]) for p in batch_host if isinstance(p, torch.Tensor) and p.dim() > 1)
            if row >= rows:
                raise ValueError(f"sequential prefill activated row {row} of {rows}-row per-layer page tables")
            if int(pt.max()) > 0 and not self._page_table_row_matches(pt, batch_host, row):
                raise ValueError(
                    f"sequential prefill: legacy page-table slice {pt[0, :4].tolist()} does not match row {row} "
                    "of the per-layer tables"
                )
            m._sequential_row_cursor = row + 1
            sliced = []
            sliced_by_id = {}  # layers of one kv-cache group share the host table: one slice object per group
            for p in batch_host:
                if isinstance(p, torch.Tensor) and p.dim() > 1 and int(p.shape[0]) > 1:
                    view = sliced_by_id.get(id(p))
                    if view is None:
                        view = sliced_by_id[id(p)] = p[row : row + 1]
                    sliced.append(view)
                else:
                    sliced.append(p)
            m._active_page_tables_per_layer = sliced
            # Prefill installs full-batch host tables and H2D-copies them once
            # (batch_key=B). Sequential users then slice host `_active` to 1-row
            # but `ttnn_prefill_forward` only calls `_page_tables_to_ttnn`, which
            # returns existing B=1 device buffers *without* refreshing content.
            # Without this H2D, users after the first keep reading/writing user
            # 0's block IDs (full-attn cross-chunk SDPA + sliding ring fill).
            self._install_per_layer_page_tables(m, sliced, writer="prefill")

    def _clear_sequential_batch_page_tables(self) -> None:
        """Put the full-batch per-layer tables back after sequential prefill."""
        for m in self.model:
            batch_host = getattr(m, "_sequential_batch_page_tables", None)
            if batch_host is None:
                continue
            m._active_page_tables_per_layer = batch_host
            self._install_per_layer_page_tables(m, batch_host, writer="prefill")
            m._sequential_row_cursor = 0
            del m._sequential_batch_page_tables

    @staticmethod
    def _page_tables_rows(page_tables_per_layer):
        """Row count that keys a per-layer table list's persistent device buffers
        (the same rule as ``Gemma4Model._host_page_tables_batch``)."""
        for pt in page_tables_per_layer or []:
            if isinstance(pt, torch.Tensor):
                return int(pt.shape[0]) if pt.dim() > 1 else 1
        return None

    @classmethod
    def _install_per_layer_page_tables(cls, model, page_tables_per_layer, *, writer):
        """H2D the per-layer tables into the persistent buffers for their row count and record the writer.

        Prefill and decode share those buffers; decode must re-upload before replaying after a prefill write, and
        a prefill must re-upload after a decode wrote (its trace inputs rewrite the device table), so a prefill
        forces the upload unless the last writer for this row count was prefill: then the host-side "unchanged"
        shortcut of ``update_persistent_per_layer_page_tables`` may skip the H2D (the chunks of one long prompt,
        back-to-back prefills on the same table).
        """
        if not hasattr(model, "update_persistent_per_layer_page_tables"):
            return
        rows = cls._page_tables_rows(page_tables_per_layer)
        writers = getattr(model, "_g4_page_table_writer_by_rows", None)
        if writers is None:
            writers = {}
            model._g4_page_table_writer_by_rows = writers
        last_writer = writers.get(rows) if rows is not None else None
        force = writer == "prefill" and last_writer != "prefill"
        t0 = time.perf_counter()
        model.update_persistent_per_layer_page_tables(page_tables_per_layer, force=force)
        add = getattr(model, "_g4_pf_add", None)
        if add is not None:
            add("install", time.perf_counter() - t0)
            add("install_n", 0.001)  # counted in the same ms accumulator: 1 per install
        if rows is None:
            return
        writers[rows] = writer

    @classmethod
    def _page_tables_written_by_prefill(cls, models, per_submesh) -> bool:
        """Whether any submesh's persistent buffers for these tables' row count
        were last written by prefill (so decode must re-upload before replay)."""
        for model, tables in zip(models, per_submesh or []):
            rows = cls._page_tables_rows(tables)
            if rows is None:
                continue
            writers = getattr(model, "_g4_page_table_writer_by_rows", None) or {}
            if writers.get(rows) == "prefill":
                return True
        return False

    def _effective_paged_block_size(self, kv_cache):
        """Effective block_size the paged ops address this model's K/V cache with.

        Under vLLM hybrid kv-cache groups the K/V buffer is HMA-shared: a full-attention
        layer views a buffer whose declared head_dim belongs to a sliding layer
        (e.g. declared block 64 / head_dim 256 shared with full-attn head_dim 512 →
        eff_bs 64). ``paged_fill_cache`` / chunked SDPA address that view, so page-table
        math must use it too.

        When every layer owns a matching cache (Option A / non-hybrid), declared
        block_size is correct — do **not** scale by max(head_dim) just because sliding
        and full layers differ. That wrongly halves the block size, doubles the page
        table width, and makes later chunks' ``chunk_page_table`` slices land on the
        zero-pad (clobbering earlier chunks' KV).
        """
        block_size = get_block_size(kv_cache)
        # Groups can end up with different effective block sizes (e.g. 31B at TP=4: sliding 64,
        # full 128). Tables trimmed with the smallest one stay wide enough for every layer; the
        # paged ops only require ``seq_len <= cols * block_size``.
        # Every layer's size comes from the same per-block invariant the attention's paged ops use (an HMA-shared
        # buffer allocated at another layer's view, or a cache holding more or fewer heads per chip than the layer
        # writes), so the page-table width always matches what the fill addresses.
        sizes = []
        for i, layer in enumerate(getattr(self.model[0], "layers", [])):
            attn = getattr(layer, "self_attn", None)
            cfg = getattr(attn, "config", None)
            if cfg is None or i >= len(kv_cache) or kv_cache[i] is None:
                continue
            sizes.append(int(layer_paged_block_size(attn, cfg, kv_cache[i][0])))
        return min(sizes) if sizes else block_size

    def _paged_prefill_block_size(self, kv_cache):
        # Base Generator hook: chunked-prefill page-table padding/slicing uses this so
        # it matches the HMA effective block_size instead of the declared shape.
        return self._effective_paged_block_size(kv_cache)

    def _uses_bounded_sliding_kv(self, model_id=-1):
        model = self.model[model_id]
        return bool(getattr(model, "bounded_sliding_kv_cache", False))

    def _prefill_get_last_token(self, last_token_idx):
        """True last-token index; lm_head tile-aligns separately in the model.

        Always pass the real index so unbounded ``paged_fill_cache`` can cap at
        ``valid_seq_len = last+1`` and skip power-of-2 pad rows. Extra page-table
        columns pad with 0 (vLLM null block). Decode skip is position -1.
        """
        return int(last_token_idx)

    def _chunk_prefill_get_last_token(self, *, is_last_chunk, last_token_idx_in_chunk, chunk_size):
        """Per-chunk ``get_last_token`` for Gemma4 multi-chunk prefill.

        Intermediate → ``-1`` (skip lm_head; skip bounded ring stash — last chunk
        overwrites). Last chunk → true index (model tile-aligns for lm_head;
        fill uses ``valid_seq_len = last+1`` to exclude pad).
        """
        del chunk_size
        if not is_last_chunk:
            return -1
        return int(last_token_idx_in_chunk)

    def _adjust_last_prefill_chunk(
        self,
        *,
        last_chunk_start,
        last_token_idx_in_chunk,
        last_token_idx_in_seq,
        chunk_size,
        block_size,
        model_id=-1,
    ):
        """Expand last-chunk start so one full chunk covers ≥ sliding window.

        When bounded and the grid remnant is ``< cache_position_modulo``, pull
        the start earlier (align by ``max(block_size, 128)``). Do **not** merge
        the previous chunk or flush intermediates — that corrupted token-0.
        """
        if not self._uses_bounded_sliding_kv(model_id):
            return last_chunk_start, last_token_idx_in_chunk
        modulo = None
        window = None
        model = self.model[model_id]
        for layer in getattr(model, "layers", []):
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            if cfg is not None and getattr(cfg, "cache_position_modulo", None) is not None:
                modulo = int(cfg.cache_position_modulo)
                window = int(getattr(cfg, "sliding_window", 0) or 0) or modulo
                break
        if modulo is None:
            sw = getattr(getattr(model, "hf_config", None), "sliding_window", None)
            modulo = int(sw) if sw else None
            window = modulo
        if not modulo or last_chunk_start <= 0:
            return last_chunk_start, last_token_idx_in_chunk
        # Threshold on the WINDOW, not the ring. Attention only ever reads the
        # last ``window`` positions; ring slots beyond that are masked out, so
        # the last chunk only has to carry a full window of real K/V. Testing
        # against ``modulo`` was equivalent while ring == window, but with
        # speculative ring headroom (ring 2048, window 1024) it fired a needless
        # expansion for a perfectly adequate remnant -- and the expanded chunk
        # produced wrong K/V: bounded spec decode at 32k went from parity
        # (2.40/5 @ 42.24 tok/s/u, matching unbounded) to corrupt-from-token-1.
        remnant = int(last_token_idx_in_chunk) + 1
        if remnant >= window:
            return last_chunk_start, last_token_idx_in_chunk
        # paged_fill_cache has no start-position input: row r is written to
        # circular slot r % modulo. Keep the expanded chunk's absolute start
        # ring-aligned so local row indices map to the corresponding absolute
        # slots. Pulling back by one complete window gives the last chunk at
        # least one full window of real K/V without changing its ring origin.
        # The start must be a multiple of the RING (``modulo``), not merely of
        # 128: row r lands in slot r % modulo, so any other start rotates every
        # local row off its absolute slot. With the exact-window ring
        # (modulo == sliding_window) chunk starts were already multiples of it
        # and 128-alignment happened to work; with speculative ring headroom it
        # does not. Align DOWN so the expanded chunk still covers a full window.
        align = int(modulo)
        target_start = max(0, int(last_chunk_start) - int(window))
        new_start = (target_start // align) * align
        new_idx = int(last_token_idx_in_seq) - new_start
        if new_idx + 1 > int(chunk_size):
            raise ValueError(f"Expanded bounded last chunk has {new_idx + 1} rows, exceeding chunk_size={chunk_size}")
        if new_start % modulo != 0:
            raise ValueError(f"Expanded bounded last chunk start={new_start} must be aligned to modulo={modulo}")
        logger.info(
            "Gemma4 multi-chunk: expanded ring-aligned last chunk start {}→{} "
            "(remnant={}, expanded_rows={}, sliding_window={}, align={})",
            last_chunk_start,
            new_start,
            remnant,
            new_idx + 1,
            modulo,
            align,
        )
        return new_start, new_idx

    def _refresh_prefill_valid_seq_len(self, *, model_id=-1, last_token_idx=None, num_cached_tokens=0):
        """Refresh the persistent bounded-fill cap tensor out of any active trace.

        Traced prefill captures ``paged_fill_cache`` with ``get_last_token=-1``, so
        the host-side ``valid_seq_len`` slice is skipped. The writer kernel instead
        reads ``model.prefill_valid_len_dev``; this method copies the real
        (unpadded) length into that buffer before capture/replay.
        """
        model = self.model[model_id]
        update = getattr(model, "update_prefill_valid_seq_len", None)
        if update is None or getattr(model, "prefill_valid_len_dev", None) is None:
            return
        if last_token_idx is None:
            return
        # Batched traced prefill passes a per-slot list, but the fill-cap tensor
        # is a single scalar. Trace is disabled for batched+bounded upstream; for
        # eager batched+bounded the host-side K/V slice carries the real length,
        # so skip the scalar refresh instead of failing the request.
        if isinstance(last_token_idx, (list, tuple)):
            return
        valid_len = int(last_token_idx) - int(num_cached_tokens) + 1
        if valid_len > 0:
            update(valid_len)

    def _prepare_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        chunk_page_table=None,
        kv_cache=None,
        model_id=-1,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        start_pos=0,
    ):
        """Compile on persistent trace inputs, retaining Gemma4's starting tail state."""
        had_starting_tails = self._any_sliding_prefill_tails(model_id)
        had_persistent = self._any_sliding_prefill_persistent(model_id)
        if batch_size == 1 and int(start_pos) == 0:
            self._release_all_sliding_prefill_tails(model_id, clear_persistent=True)
        prepared = super()._prepare_trace_prefill(
            prefill_ids,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            kv_cache=kv_cache,
            model_id=model_id,
            global_user_id=global_user_id,
            batch_size=batch_size,
            user_id=user_id,
            start_pos=start_pos,
        )
        prepared["gemma4_tail_state"] = (batch_size, int(start_pos), had_starting_tails, had_persistent)
        # The bridge and input preparation route auxiliary inputs through model
        # attributes. Other buckets may change them before deferred recording.
        model = self.model[model_id]
        prepared["gemma4_prefill_context"] = {
            name: getattr(model, name, None)
            for name in (
                "_active_page_tables_per_layer",
                "_prefill_input_ids_torch",
                "_prefill_embeds_torch",
                "_prefill_batch_size",
                "_prefill_seq_len_per_user",
            )
        }
        return prepared

    def _record_trace_prefill(self, prepared):
        """Restore the compiled branch and record using the already allocated inputs."""
        model_id = prepared["model_id"]
        batch_size, start_pos, had_starting_tails, had_persistent = prepared["gemma4_tail_state"]
        if batch_size == 1:
            if start_pos == 0:
                # sp0 must take the no-tail branch in compile and capture. A
                # soft release would retain the ring and add an unwarmed copy.
                self._release_all_sliding_prefill_tails(model_id, clear_persistent=True)
            elif not had_starting_tails:
                self._release_all_sliding_prefill_tails(model_id, clear_persistent=not had_persistent)
        model = self.model[model_id]
        context = {**prepared["gemma4_prefill_context"], "_prefill_trace_mode": True}
        previous = {name: getattr(model, name, None) for name in context}
        try:
            for name, value in context.items():
                setattr(model, name, value)
            return super()._record_trace_prefill(prepared)
        finally:
            for name, value in previous.items():
                setattr(model, name, value)

    def _capture_trace_prefill_sampling(self, model_id, sampling_batch):
        """Gemma4 override: replicate the sampling trace input, do not column-shard it.

        The shared tt_transformers implementation builds this buffer with
        ``ShardTensorToMesh(dim=-1)``, i.e. ``[1, 1, sampling_batch, dim/TP]``,
        because tt_transformers activations are column-sharded. Gemma4 residuals
        are **full width on every device** (embed all-gather + per-layer
        all-reduces) — the same convention
        ``extract_last_tokens_batched_prefill`` relies on when it re-uploads the
        gathered last-token rows with ``ReplicateTensorToMesh``. Feeding a
        ``dim/TP``-wide tensor into Gemma4's ``_apply_lm_head`` therefore fails
        the matmul contract:

            The width of the first tensor must be equal to the height of the
            second tensor. Mismatch: width=480 height=3840   (12B, TP=8)

        Only the *traced* batched-prefill sampling path hit this; the eager
        fallback right below it in ``_row_sharded_batched_prefill`` passes the
        replicated ``user_hidden`` straight to ``_apply_norm_and_lm_head`` and
        has always been correct. Mirroring the eager layout here makes the two
        agree. This is mesh-shape driven, not arch driven — any TP>1 Gemma4 mesh
        that captures this trace needs it.
        """
        mesh_device = self.model_args[model_id].mesh_device
        full_dim = self.model_args[model_id].dim
        model = self.model[model_id]
        mesh_mapper = model._replicate_to_mesh_mapper()

        def _make_input():
            return ttnn.from_torch(
                torch.zeros(1, 1, sampling_batch, full_dim, dtype=torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        logits = model._apply_norm_and_lm_head(_make_input())
        model.sampling.sample(logits, enable_trace=False)
        ttnn.synchronize_device(mesh_device)
        logger.info("Gemma4: done compiling prefill sampling (replicated input)")

        trace_input = _make_input()
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        logits = model._apply_norm_and_lm_head(trace_input, deallocate_input=False)
        tt_tokens, tt_log_probs = model.sampling.sample(logits, enable_trace=False)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info("Gemma4: done capturing prefill sampling trace")

        return trace_id, (tt_tokens, tt_log_probs), trace_input

    def _easy_trace_prefill(self, *args, **kwargs):
        page_table = kwargs.get("page_table")
        if page_table is None and len(args) >= 2:
            page_table = args[1]
        self._activate_sequential_per_layer_row(page_table)
        # Refresh before capture *and* replay so the writer kernel sees the
        # current request's real length (trace binds the buffer address; this
        # updates its contents out-of-trace).
        self._refresh_prefill_valid_seq_len(
            model_id=kwargs.get("model_id", -1),
            last_token_idx=kwargs.get("last_token_idx"),
            num_cached_tokens=kwargs.get("num_cached_tokens", 0),
        )
        # Non-APC traced prefill: ``full_page_table`` is sized to the raw prompt
        # (e.g. 61 blocks for a 3896-token prompt) while the captured device
        # buffer is sized to the padded bucket (64 for 4096). Under bounded
        # sliding, layer-0's cache has only ``sliding_window/block_size`` blocks,
        # so ``_pad_or_create_page_table`` cannot grow the short table up to the
        # captured width and ``copy_host_to_device`` TT_FATALs. Drop
        # ``full_page_table`` so the already-padded ``page_table`` is used.
        # APC (num_cached_tokens > 0) still needs the full mapping for chunk
        # slicing — leave it alone there.
        # Traced multi-chunk always needs the full mapping so every chunk
        # (including sp0) builds a ``chunk_page_table`` for absolute fill.
        force_chunk_pt = bool(kwargs.pop("force_chunk_page_table", False))
        if not kwargs.get("num_cached_tokens") and kwargs.get("full_page_table") is not None and not force_chunk_pt:
            kwargs["full_page_table"] = None
        # Defer lm_head outside the trace: capture returns post-norm hidden
        # states; ``process_logits_after_prefill_trace`` runs lm_head on the
        # last-token tile. Must live on this mixin (not only Gemma4Generator)
        # so the vLLM ``Gemma4ForCausalLM`` path also sets the flag — otherwise
        # ``get_last_token=-1`` (trace default) hits the intermediate-chunk
        # ``return None`` and warmup crashes in process_logits_after_prefill_trace.
        for m in self.model:
            m._prefill_trace_mode = True
            m._sequential_row_active = True  # the capture below re-enters the eager path for this user
        # The row of the 32-row last-token tile this user's logits land on (the shared loop reads
        # ``(last_token_idx - num_cached_tokens) % 32``); ``_append_prefill_result`` slices it on device.
        self._g4_pf_row = self._g4_prefill_row_record(kwargs.get("last_token_idx"), kwargs.get("num_cached_tokens", 0))
        t0 = time.perf_counter()
        try:
            if force_chunk_pt:
                return self._easy_trace_prefill_with_chunk_page_table(*args, **kwargs)
            return super()._easy_trace_prefill(*args, **kwargs)
        finally:
            for m in self.model:
                m._prefill_trace_mode = False
                m._sequential_row_active = False
                add = getattr(m, "_g4_pf_add", None)
                if add is not None:
                    add("trace", time.perf_counter() - t0)
                    break

    @staticmethod
    def _g4_prefill_row_record(last_token_idx, num_cached_tokens):
        """``(last_token_idx, row)`` for a single-user traced prefill (``last_token_idx`` one int), ``None`` for the
        batched form (a list of per-slot indices: the shared batched consumption reads its own rows) and for no
        index: the one-row readback then stays off for that call."""
        if last_token_idx is None or isinstance(last_token_idx, (list, tuple)):
            return None
        try:
            idx = int(last_token_idx)
            cached = int(num_cached_tokens or 0)
        except (TypeError, ValueError):
            return None
        return (idx, (idx - cached) % 32)

    def _append_prefill_result(self, prefill_results, idx, model_id, last_token_idx, tail_out, sampling_enabled):
        """Queue one row of the logits for readback instead of the whole 32-row tile from every device.

        The shared loop reads the untilized ``[1, 1, 32, vocab]`` tile back from every TP device (4 x 16.8 MB for
        the 262k vocab) and keeps one row of one device. After a traced single-user prefill the row is known
        (``_easy_trace_prefill``), so slice it on device and read one 512 KB row per device; the model's
        ``process_output_prefill`` takes the one-row host tensor as it is. The eager, sampling, lane-sharded and
        hidden-states paths keep the shared readback. GEMMA4_PREFILL_ROW_READBACK=0 restores it everywhere.
        """
        pair = getattr(self, "_g4_pf_row", None)
        self._g4_pf_row = None
        # the row belongs to the traced call that recorded it: the same user (its absolute last-token index)
        row = (
            pair[1]
            if pair is not None and not isinstance(last_token_idx, (list, tuple)) and pair[0] == int(last_token_idx)
            else None
        )
        model = self.model[model_id]
        if getattr(model, "_lane_sharded", False) and (sampling_enabled or isinstance(tail_out, ttnn.Tensor)):
            return self._append_lane_prefill_result(
                prefill_results, idx, model_id, last_token_idx, tail_out, sampling_enabled
            )
        if (
            sampling_enabled
            or row is None
            or not isinstance(tail_out, ttnn.Tensor)
            or os.environ.get("GEMMA4_PREFILL_ROW_READBACK", "1") == "0"
            or len(tail_out.shape) != 4
            or int(tail_out.shape[-2]) <= row
            or tail_out.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            return super()._append_prefill_result(
                prefill_results, idx, model_id, last_token_idx, tail_out, sampling_enabled
            )
        vocab = int(tail_out.shape[-1])
        row_tensor = ttnn.slice(tail_out, [0, 0, row, 0], [1, 1, row + 1, vocab])
        queued = row_tensor.cpu(blocking=False)  # one row per device; the consumer reads the first shard
        prefill_results.append(
            {
                "idx": idx,
                "model_id": model_id,
                "last_token_idx": last_token_idx,
                "logits": queued,
                "sampling": sampling_enabled,
            }
        )

    def _append_lane_prefill_result(
        self, prefill_results, idx, model_id, last_token_idx, tail_out, sampling_enabled=False
    ):
        """Queue the OWNER lane's logits tile (or sampled tokens) of one serial prefill, chosen now, not at the readback.

        Under lanes every column runs the single-user prefill, but only the request's owner lane prefills against
        its real page tables (the other lanes compute on the scratch row: their attention over the cached prefix and
        the earlier chunks reads the null block, their MoE rows are the owner's partials). The shared loop reads every
        user's tile back after the whole batch and ``process_output_prefill`` picked the lane from the model's
        ``_g4_active_owner_lane``, which by then names the LAST user's lane: in a step with several users on
        different lanes (the vLLM chunked continuations and the cached-prefix prefills of a concurrent burst, which
        the lane router declines) every user off that lane was handed a non-owner lane's tile, a corrupted copy of
        its own logits, and a wrong first token followed by the right continuation (2026-10-25, the 26B Galaxy
        multidoc runs: 14-18 of 20 right; serial and staggered requests, which never share a step, 20/20). The owner
        is this user's while its forward is being queued, so take that lane's first chip here; ``process_output_prefill``
        then reads the one-shard host tensor as it is. One 32 x vocab tile per user comes back instead of one per
        chip (the Galaxy lane engine: 1 of 32). With on-device sampling (``tail_out`` = the sampler's (tokens, log
        probs)) the same lane choice applies: the shared loop reads the tokens and the top-k log probs from the FIRST
        shard of what was queued (lane 0's chip for a mesh tensor, wrong for a user on any other lane), so the owner's
        shards are queued here and that first shard is the owner's."""
        model = self.model[model_id]
        owner = int(getattr(model, "_g4_active_owner_lane", 0) or 0) % int(model.mesh_config.lanes)
        dev = model.lane_device_indices()[owner]

        def shard(t):
            return ttnn.get_device_tensors(t)[dev].cpu(blocking=False)

        if sampling_enabled:
            tt_tokens, tt_log_probs = tail_out
            if isinstance(tt_log_probs, LogProbsResult):
                log_probs = LogProbsResult(
                    topk_logprobs=None,
                    topk_indices=None,
                    topk_logprobs_host=shard(tt_log_probs.topk_logprobs),
                    topk_indices_host=shard(tt_log_probs.topk_indices),
                )
            elif isinstance(tt_log_probs, ttnn.Tensor):
                log_probs = shard(tt_log_probs)
            else:
                log_probs = None
            queued = [shard(tt_tokens), log_probs]
        else:
            queued = shard(tail_out)
        prefill_results.append(
            {
                "idx": idx,
                "model_id": model_id,
                "last_token_idx": last_token_idx,
                "logits": queued,
                "sampling": bool(sampling_enabled),
                "lane": owner,
            }
        )

    def _easy_trace_prefill_with_chunk_page_table(
        self,
        prefill_ids,
        page_table=None,
        full_page_table=None,
        user_id=0,
        last_token_idx=None,
        kv_cache=None,
        model_id=-1,
        prefill_seq_len=None,
        batch_size=1,
        num_cached_tokens=0,
        **kwargs,
    ):
        """Like ``Generator._easy_trace_prefill``, but always builds ``chunk_page_table``.

        Needed so the first 4k chunk of a multi-chunk prefill still takes the
        sliding-window tail path (``is_chunked=True``) and so absolute-block fill
        matches the eager multi-chunk loop. Trace keys use ``sp0_mc`` / ``sp1_mc``
        so they do not collide with cold single-chunk ``sp0``/`sp1`` captures.
        """
        del last_token_idx  # refresh already done by caller
        global_user_id = kwargs.get("global_user_id", None)
        use_start_pos = "sp1_mc" if num_cached_tokens > 0 else "sp0_mc"
        trace_key = f"{prefill_seq_len}_{model_id}_{batch_size}_{use_start_pos}"

        chunk_start_idx = num_cached_tokens
        block_size = get_block_size(kv_cache)

        if page_table is not None and batch_size == 1:
            page_table = page_table[user_id : user_id + 1, :]
        if full_page_table is not None and batch_size == 1:
            full_page_table = full_page_table[user_id : user_id + 1, :]

        from models.tt_transformers.tt.generator import _get_max_blocks_prefill

        max_blocks_prefill = _get_max_blocks_prefill(kv_cache)
        source_page_table = full_page_table if full_page_table is not None else page_table
        if source_page_table is None:
            raise ValueError("Traced multi-chunk prefill requires a page_table")
        # Bounded sliding: layer-0's paged cache is only ``sliding_window`` blocks,
        # so ``_get_max_blocks_prefill`` under-sizes the captured page-table buffer.
        # Long-ISL replay then hands a full-attention-wide host table (e.g. 4096
        # cols at 256k) into a short device buffer → shape TT_FATAL. Size to the
        # model's max_seq_len so capture (8k warmup) and replay share one width.
        max_seq_blocks = num_blocks_in_seq(int(self.model_args[model_id].max_seq_len), block_size)
        target_blocks = max(max_blocks_prefill, max_seq_blocks, int(source_page_table.shape[1]))
        page_table = _pad_or_create_page_table(source_page_table, target_blocks)
        chunk_page_table = None
        if batch_size == 1:
            chunk_start_block = num_cached_tokens // block_size
            chunk_end_block = num_blocks_in_seq(num_cached_tokens + prefill_seq_len, block_size)
            chunk_page_table = source_page_table[:, chunk_start_block:chunk_end_block]
            chunk_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
            chunk_page_table = _pad_or_create_page_table(chunk_page_table, chunk_blocks)

        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                prefill_ids,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                kv_cache=kv_cache,
                model_id=model_id,
                global_user_id=global_user_id,
                batch_size=batch_size,
                user_id=user_id,
                start_pos=chunk_start_idx,
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace

        return self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            prefill_ids,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            model_id=model_id,
            global_user_id=global_user_id,
            batch_size=batch_size,
            user_id=user_id,
            start_pos=chunk_start_idx,
        )

    def _any_sliding_prefill_tails(self, model_id=-1) -> bool:
        """True if any layer has a cross-chunk ``_sliding_prefill_tail`` stash."""
        for layer in getattr(self.model[model_id], "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if getattr(attn, "_tail_pool_map", None):
                return True  # pooled tails live in the boot pool, not the fallback dict
            if any(v is not None for v in (getattr(attn, "_sliding_tails_by_key", None) or {}).values()):
                return True
        return False

    def _any_sliding_prefill_persistent(self, model_id=-1) -> bool:
        """True if any layer has ``sliding_prefill_tail_persistent`` ring buffers."""
        for layer in getattr(self.model[model_id], "layers", []):
            attn = getattr(layer, "self_attn", None)
            cfg = getattr(attn, "config", None) if attn is not None else None
            if cfg is not None and getattr(cfg, "sliding_prefill_tail_persistent", None) is not None:
                return True
        return False

    def _release_all_sliding_prefill_tails(self, model_id=None, *, clear_persistent: bool = False, req_key=None):
        models = self.model if model_id is None else [self.model[model_id]]
        for model in models:
            for layer in getattr(model, "layers", []):
                attn = getattr(layer, "self_attn", None)
                if attn is None or not hasattr(attn, "_release_sliding_prefill_tail"):
                    continue
                if req_key is not None:
                    attn._release_sliding_prefill_tail(req_key=req_key)
                else:
                    attn._release_sliding_prefill_tail(clear_persistent=clear_persistent)

    def _bind_sliding_tail_key(self, req_key):
        for model in self.model:
            for layer in getattr(model, "layers", []):
                cfg = getattr(getattr(layer, "self_attn", None), "config", None)
                if cfg is not None:
                    cfg._g4_active_req_key = req_key

    def prefill_forward_single_user_text(
        self, tokens, page_table=None, *, kv_cache=None, num_cached_tokens=0, **kwargs
    ):
        # vLLM multimodal: the shared per-user loop hands every user its ``pixel_values``
        # entry; the bridge stages the user's host-merged prompt embeddings there
        # (``_stage_multimodal_prefill``) and text users carry None.
        mm_embeds = kwargs.pop("pixel_values", None)
        if torch.is_tensor(mm_embeds):
            return self._prefill_single_user_with_embeds(
                mm_embeds, tokens, page_table, kv_cache=kv_cache, num_cached_tokens=num_cached_tokens, **kwargs
            )
        # Lane-sharded page tables are stacked on the request's owner lane (_page_table_host_layout);
        # bind it from global_user_id before any per-layer staging in this forward.
        _gid = kwargs.get("global_user_id", None)
        for _m in self.model:
            _mc = getattr(_m, "mesh_config", None)
            if _mc is not None and getattr(_mc, "lane_sharded", False):
                _slots = int(getattr(_m, "lane_slots", 0) or 32)
                _m._g4_active_owner_lane = (int(_gid) // _slots) % _mc.lanes if _gid is not None else 0
        self._activate_sequential_per_layer_row(page_table)
        # Bind this request's stable identity (its first global block id — the
        # same keying _bounded_ring_slots uses) to every layer config so the
        # cross-chunk sliding-tail stash is consumed/produced PER REQUEST.
        # Interleaved multi-request continuations through a single per-layer
        # slot handed one request's window tail to another (conc3/9k fluent
        # nondeterministic corruption); scheduler order and row placement are
        # not stable across rounds (plugin PR #68), so only a request-owned
        # key is safe.
        req_key = None
        if page_table is not None and torch.is_tensor(page_table) and page_table.numel() > 0:
            pt2d = page_table if page_table.dim() > 1 else page_table.unsqueeze(0)
            if int(pt2d[0].max()) > 0:
                # All-zero row = vLLM null block / B=1 warmup mock tables —
                # never a real request; keep key None so trace-unsafe pool
                # copies cannot run during warmup capture. Key on the first
                # block id +1: under bounded the remapped sliding table's
                # slot 0 legitimately starts at block id 0, and a falsy key
                # would silently bypass the pool for that request.
                req_key = int(pt2d[0, 0]) + 1
        # Slot source: global_user_id (lane-sharded paths) or the per-call user_id kwarg. Batched prefill
        # passes a list and stays on the legacy key (it requires num_cached==0, so cold-start reset isolates it).
        _slot_src = _gid if isinstance(_gid, int) else kwargs.get("user_id")
        if req_key is not None and isinstance(_slot_src, int):
            # Prefix caching lets first-block ids collide across requests, so key the stash on the slot plus a
            # per-slot generation; a start offset other than the slot's expected continuation begins a new one.
            _slot = int(_slot_src)
            _start = int(num_cached_tokens or 0)
            _gens = getattr(self, "_g4_slot_tail_gen", None)
            if _gens is None:
                _gens = {}
                self._g4_slot_tail_gen = _gens
            _gen, _expected = _gens.get(_slot, (0, None))
            if _start == 0 or _expected is None or _start != _expected:
                if _gen:
                    # Release the previous generation's tail on every layer, or dead keys exhaust the pool
                    # and grow the spill dict without bound.
                    self._release_all_sliding_prefill_tails(req_key=((_slot + 1) << 24) + _gen)
                _gen += 1
            _gens[_slot] = (_gen, _start + int(tokens.shape[-1]))
            req_key = ((_slot + 1) << 24) + _gen
        self._bind_sliding_tail_key(req_key)
        if page_table is not None and kv_cache is not None:
            block_size = self._effective_paged_block_size(kv_cache)
            needed_blocks = num_blocks_in_seq(tokens.shape[-1] + num_cached_tokens, block_size)
            # Bounded sliding KV cache: sliding layers pass ``cache_position_modulo``
            # to ``paged_fill_cache``, which requires ``page_table`` to span the whole
            # window (``cols * block_size >= cache_position_modulo``) so the circular
            # wrap can address every slot — even for a short prompt whose own block
            # count is far smaller. Trimming to the prompt's ``needed_blocks`` (as the
            # over-wide guard does) would undo the widening applied upstream in
            # ``_get_prefill_user_page_table`` and TT_FATAL the fill. Floor
            # ``needed_blocks`` at the window's block count when any layer runs bounded
            # (read from the per-layer configs so this holds for both the demo and vLLM
            # generators without a generator-level flag).
            modulos = []
            for layer in getattr(self.model[0], "layers", []):
                cfg = getattr(getattr(layer, "self_attn", None), "config", None)
                modulo = getattr(cfg, "cache_position_modulo", None)
                if modulo:
                    modulos.append(modulo)
            if modulos and block_size:
                needed_blocks = max(needed_blocks, num_blocks_in_seq(max(modulos), block_size))
            if page_table.shape[1] > needed_blocks:
                page_table = page_table[:, :needed_blocks]
        # Eager single-chunk still prefers the host-side slice when get_last_token
        # is known; refreshing here keeps the persistent tensor current for any
        # path that falls through to the kernel cap (and for multi-chunk's last
        # chunk if a future hook uses it).
        self._refresh_prefill_valid_seq_len(
            model_id=kwargs.get("model_id", -1),
            last_token_idx=kwargs.get("last_token_idx"),
            num_cached_tokens=num_cached_tokens,
        )

        model_id = kwargs.get("model_id", -1)
        seq_len = tokens.shape[-1]
        max_chunk = self.model_args[model_id].max_prefill_chunk_size
        use_traced_chunks = (
            chunked_prefill_trace_enabled()
            and page_table is not None
            and kv_cache is not None
            and seq_len > max_chunk
            and max_chunk in (128, 512, 1024, 2048, 4096)
            and not bool(getattr(self.model[model_id], "hidden_size_per_layer_input", 0))
            # Bounded final-chunk K/V must be stashed eagerly and committed only
            # after lm_head. A captured chunk has get_last_token=-1 and cannot
            # perform the host-side boundary merge safely after the trace.
            and not self._uses_bounded_sliding_kv(model_id)
            # Lane-sharded KV stacks each chunk's page table on the owner column host-side; the traced replay's
            # chunk-table refresh cannot, so chunked prompts stay eager under lanes.
            and not bool(getattr(getattr(self.model[model_id], "mesh_config", None), "lane_sharded", False))
            # A multimodal prefill feeds merged embeddings (``_g4_mm_prefill_embeds``); the
            # captured chunk's input is the token ids, so it stays eager.
            and getattr(self.model[model_id], "_g4_mm_prefill_embeds", None) is None
        )
        if not use_traced_chunks:
            # Eager path stays in gemma4 (do not patch models/tt_transformers):
            # true last-token for bounded fill, ring-aligned last-chunk expand,
            # and intermediate get_last_token=-1.
            return self._prefill_forward_single_user_text_eager(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                num_cached_tokens=num_cached_tokens,
                **kwargs,
            )

        # ── Traced multi-chunk: replay 4k sp0/sp1 traces per generator chunk ──
        user_id = kwargs.get("user_id", 0)
        last_token_idx = kwargs["last_token_idx"]
        batch_size = kwargs.get("batch_size", 1)
        assert batch_size == 1, "Traced multi-chunk prefill supports batch_size=1 only"
        chunk_size = get_max_prefill_chunk_size(seq_len, max_chunk)
        last_token_idx_in_seq = last_token_idx - num_cached_tokens
        last_token_idx_in_chunk = last_token_idx_in_seq % chunk_size
        last_chunk_start = (last_token_idx_in_seq // chunk_size) * chunk_size

        chunk_source_page_table, block_size = self._chunk_prefill_page_table(
            page_table, user_id=user_id, model_id=model_id, kv_cache=kv_cache
        )
        page_table_user = chunk_source_page_table[user_id : user_id + 1, :]
        needed_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size)
        if page_table_user.shape[1] > needed_blocks:
            page_table_user = page_table_user[:, :needed_blocks]
        num_padding_blocks = needed_blocks - page_table_user.shape[1]
        # Extra columns pad with 0 (vLLM null block). Fill skip is valid_seq_len,
        # not page-table -1.
        if num_padding_blocks > 0:
            page_table_user_padded = torch.cat(
                [
                    page_table_user,
                    torch.zeros((1, num_padding_blocks), dtype=torch.int32),
                ],
                dim=-1,
            )
        else:
            page_table_user_padded = page_table_user
        CHUNK_USER_ID = 0

        logger.info(
            "Gemma4 traced multi-chunk prefill: seq_len={} chunk_size={} cached={}",
            seq_len,
            chunk_size,
            num_cached_tokens,
        )
        self._release_all_sliding_prefill_tails(model_id)

        for chunk_start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
            chunk_end = chunk_start + chunk_size
            chunk_start_relative = chunk_start - num_cached_tokens
            chunk_end_relative = chunk_end - num_cached_tokens
            chunk_tokens = tokens[:, chunk_start_relative:chunk_end_relative]
            is_last_chunk = chunk_start_relative == last_chunk_start

            # Absolute last-token index for valid_seq_len refresh: full chunk for
            # intermediate chunks; real prompt end for the last chunk.
            if is_last_chunk:
                chunk_last_token_idx = last_token_idx
            else:
                chunk_last_token_idx = chunk_start + chunk_size - 1

            tt_out = self._easy_trace_prefill(
                chunk_tokens,
                page_table=page_table_user_padded,
                full_page_table=page_table_user_padded,
                user_id=CHUNK_USER_ID,
                last_token_idx=chunk_last_token_idx,
                kv_cache=kv_cache,
                model_id=model_id,
                prefill_seq_len=chunk_size,
                batch_size=1,
                num_cached_tokens=chunk_start,
                force_chunk_page_table=True,
            )
            if is_last_chunk:
                last_token_idx_for_trace = last_token_idx_in_chunk
                return self.model[model_id].process_logits_after_prefill_trace(tt_out, last_token_idx_for_trace)
            del tt_out
        raise RuntimeError("Traced multi-chunk prefill produced no last-chunk logits")

    # ── Packed multi-prompt prefill (GEMMA4_PACKED_PREFILL=1) ──────────────────────────────────────────────────
    def _g4_try_packed_prefill(self, call_kwargs: dict, *, run_rest):
        """Serve a ``prefill_forward_text`` call as packed group forwards plus the stock path for the rest.

        ``call_kwargs`` are the call's keyword arguments (``tokens``, ``page_table``, ``kv_cache``,
        ``prompt_lens``, ``empty_slots``, ``start_pos``, ...); ``run_rest(**sub_kwargs)`` runs the stock path on
        the requests no group took. Returns the ``[B, 1, vocab]`` host logits, or None when the call is not
        packable (flag off, warmup, device sampling, hidden-state return, multimodal payload, cached prefixes
        only, lanes, PLI models, DP > 1, pending deferred trace recording, no page table / KV cache).
        """
        if not gemma4_packed_prefill_enabled():
            return None
        tokens = call_kwargs.get("tokens")
        page_table = call_kwargs.get("page_table")
        kv_cache = call_kwargs.get("kv_cache")
        if tokens is None or page_table is None or kv_cache is None or not torch.is_tensor(page_table):
            return None
        if call_kwargs.get("model_id_warmup") is not None or call_kwargs.get("sampling_params") is not None:
            return None
        if call_kwargs.get("return_hidden_states") or call_kwargs.get("page_tables_per_layer") is not None:
            return None
        if any(_packed_prefill_value_present(call_kwargs.get(k)) for k in _GEMMA4_PACKED_PREFILL_MM_KEYS):
            return None
        if getattr(self, "data_parallel", 1) != 1 or getattr(self, "_defer_trace_recording", False):
            return None
        # A traced first call owns the prefill warmup sweep and the traced decode preparation: pack after it.
        # Eager serving (``--disable-trace-capture``) leaves ``already_warmed_up_prefill`` False after the
        # plugin's two-phase warmup, and the stock eager sweep it would re-run only fills the program cache.
        if (
            call_kwargs.get("enable_trace", False)
            and call_kwargs.get("warmup_prefill", True)
            and not getattr(self, "already_warmed_up_prefill", False)
        ):
            return None
        model = self.model[0]
        args = self.model_args[0]
        if getattr(args, "disable_batched_prefill", False) or getattr(model, "hidden_size_per_layer_input", 0):
            return None
        if getattr(getattr(model, "mesh_config", None), "lane_sharded", False):
            return None
        B = int(tokens.shape[0])
        prompt_lens = call_kwargs.get("prompt_lens")
        prompt_lens = [int(tokens.shape[1])] * B if prompt_lens is None else [int(p) for p in prompt_lens]
        start_pos = call_kwargs.get("start_pos")
        num_cached = [0] * B if start_pos is None else [int(n) for n in start_pos]
        if len(prompt_lens) != B or len(num_cached) != B:
            return None
        empty_slots = call_kwargs.get("empty_slots")
        slots_all = list(range(B)) if empty_slots is None else [int(x) for x in empty_slots]
        groups = plan_packed_prefill_groups(
            prompt_lens,
            num_cached,
            max_prefill_chunk_size=int(args.max_prefill_chunk_size),
            empty_slots=slots_all,
            max_batch_size=int(args.max_batch_size),
        )
        if not groups:
            return None

        self.mode = Mode.PREFILL
        out = torch.zeros(B, 1, int(model.vocab_size))
        packed = set()
        for padded, idxs in groups:
            t0 = time.perf_counter()
            self._g4_prefill_packed_group(padded, idxs, tokens, page_table, kv_cache, prompt_lens, slots_all, out)
            logger.info(
                "[gemma4-packed-prefill] {} users x {} padded tokens in one forward: {:.1f} ms (slots {})",
                len(idxs),
                padded,
                (time.perf_counter() - t0) * 1e3,
                [slots_all[i] for i in idxs][:8],
            )
            packed.update(idxs)
        rest = [i for i in range(B) if i not in packed]
        if rest:
            sub = {}
            for key, value in call_kwargs.items():
                if key == "kv_cache":
                    pass  # per-model list, never per-request
                elif isinstance(value, (list, tuple)) and len(value) == B:
                    value = [value[i] for i in rest]
                elif torch.is_tensor(value) and value.dim() >= 1 and int(value.shape[0]) == B:
                    value = value[rest]
                sub[key] = value
            sub["prompt_lens"] = [prompt_lens[i] for i in rest]
            sub["empty_slots"] = [slots_all[i] for i in rest]
            sub["start_pos"] = None if start_pos is None else [num_cached[i] for i in rest]
            sub["warmup_prefill"] = False
            rest_out = run_rest(**sub)
            if isinstance(rest_out, tuple):
                raise RuntimeError("packed prefill remainder returned sampled tokens; host logits expected")
            out[rest] = rest_out.reshape(len(rest), 1, -1)[..., : out.shape[-1]].to(out.dtype)
        return out

    def _g4_prefill_packed_group(self, padded, idxs, tokens, page_table, kv_cache, prompt_lens, slots_all, out):
        """One packed forward: requests ``idxs`` at their physical slots, logits written into ``out`` rows."""
        model_id = 0
        model = self.model[model_id]
        args = self.model_args[model_id]
        model_kv = kv_cache[model_id]
        slots = [slots_all[i] for i in idxs]
        padded_batch = batched_prefill_padded_batch(len(idxs), slots, int(args.max_batch_size))
        block_size = self._effective_paged_block_size(model_kv)
        num_blocks = num_blocks_in_seq(padded, block_size)
        pt_src = page_table if page_table.dim() > 1 else page_table.unsqueeze(0)
        prefill_ids = torch.zeros(padded_batch, padded, dtype=torch.long)
        last_idx = [0] * padded_batch  # pad slots: never filled (``valid_slots`` lists only the real ones)
        pt = torch.zeros(padded_batch, num_blocks, dtype=torch.int32)
        cols = min(num_blocks, int(pt_src.shape[1]))
        for slot, i in zip(slots, idxs):
            plen = int(prompt_lens[i])
            prefill_ids[slot, :plen] = tokens[i, :plen].to(torch.long)
            last_idx[slot] = plen - 1
            pt[slot, :cols] = pt_src[i, :cols].to(torch.int32)
        hidden = self.prefill_forward_single_user_text(
            prefill_ids,
            page_table=pt,
            kv_cache=model_kv,
            num_cached_tokens=0,
            user_id=slots,
            last_token_idx=last_idx,
            model_id=model_id,
            batch_size=padded_batch,
        )
        # Post-norm hidden [1, 1, padded_batch * padded, H]: gather every user's last-token row and run ONE
        # lm_head on the [padded_batch, H] tile instead of a slice + lm_head + readback per slot.
        hidden_dim = int(hidden.shape[-1])
        hidden4 = ttnn.reshape(hidden, [padded_batch, 1, padded, hidden_dim])
        rows = model.extract_last_tokens_batched_prefill(
            hidden4, last_idx, padded_batch, padded, target_batch=max(padded_batch, 32)
        )
        for t in (hidden4, hidden):
            try:
                ttnn.deallocate(t, force=True)
            except Exception:  # a reshape view shares the buffer freed by the first dealloc
                pass
        logits = model._apply_lm_head(rows, is_decode=False)
        mesh_cfg = getattr(model, "mesh_config", None)
        if mesh_cfg is not None and getattr(mesh_cfg, "tp", 1) > 1:
            host = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            host = ttnn.to_torch(logits)
        ttnn.deallocate(logits, force=True)
        host = host.reshape(-1, host.shape[-1])[:, : int(model.vocab_size)]
        for slot, i in zip(slots, idxs):
            out[i, 0] = host[slot].to(out.dtype)

    def prefill_forward_lanes(self, tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        """Lane-parallel prefill; returns host logits ``[lanes * k, vocab]`` (lane-major: row ``lane * k + r``).

        One user per lane (``k == 1``): ``tokens`` is [lanes, S], ``page_tables`` [lanes, 1, blocks] and
        ``prompt_lens`` has one entry per lane; the lengths must be equal because the KV fill and the last-token
        slice are shared by every column. ``k`` users per lane: ``tokens`` [lanes, k, S], ``page_tables``
        [lanes, k, blocks], ``prompt_lens`` lanes * k entries (lane-major); each column prefills its k users as one
        batched stream of ``k * S`` tokens within the prefill chunk, and the lengths must agree across lanes for
        each ``r`` (round ``r`` shares one valid length). ``slot_ids`` (same order as ``prompt_lens``) gives the
        bounded sliding rings each user's ring-local rows; the vLLM router installs its own per-layer tables.
        """
        model = self.model[0]
        mesh_cfg = model.mesh_config
        assert mesh_cfg is not None and getattr(mesh_cfg, "lane_sharded", False), "lanes gate is off"
        lanes = mesh_cfg.lanes
        if tokens.dim() == 3:
            if int(tokens.shape[1]) > 1:
                return self._prefill_forward_lanes_batched(tokens, page_tables, kv_cache, prompt_lens, slot_ids)
            tokens = tokens[:, 0]
        assert tokens.dim() == 2 and tokens.shape[0] == lanes, f"tokens must be [lanes={lanes}, seq]"
        seq_len = int(tokens.shape[-1])
        last_idx = [int(p) - 1 for p in prompt_lens]
        assert len(last_idx) == lanes and all(0 <= i < seq_len for i in last_idx)
        if len(set(last_idx)) != 1:
            # Every lane's KV fill runs with one shared valid length; a shorter
            # lane would have still-needed window history padded over.
            raise ValueError(f"lane prefill needs equal prompt lengths, got {[i + 1 for i in last_idx]}")

        # Callers pass the per-model kv_cache list (as prefill_forward_text
        # takes); lanes always run the single model.
        model_kv = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache

        block_size = self._effective_paged_block_size(model_kv)
        page_tables = self._lane_prefill_table_width(page_tables, seq_len, block_size)

        # Bounded sliding pools are per-layer rings addressed by ring-local block ids (slot s owns rows
        # [s*rb, (s+1)*rb)), so feed sliding layers ring-local tables and full-attention layers the global one.
        if slot_ids is not None:
            slots_l = [int(slot_ids)] * lanes if isinstance(slot_ids, int) else [int(x) for x in slot_ids]
            g_host = page_tables.reshape(lanes, -1).to(torch.int32)
            per_layer = self._lane_prefill_ring_tables(model, g_host, slots_l, block_size)
            if per_layer is not None:
                model._active_page_tables_per_layer = per_layer
                if lane_prefill_table_upload_enabled():
                    self._install_per_layer_page_tables(model, per_layer, writer="prefill")

        max_chunk = int(self.model_args[0].max_prefill_chunk_size)
        if seq_len <= max_chunk:
            inputs = model.prepare_inputs_prefill(
                tokens,
                page_table=page_tables,
                batch_size=1,
                user_id=0,
                lane_parallel=True,
            )
            prefill_input, rot_mats_global, rot_mats_local, page_table_tt, *_ = inputs
            tt_logits = model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                page_table=page_table_tt,
                get_last_token=self._prefill_get_last_token(max(last_idx)),
                kv_cache=model_kv,
                batch_size=1,
            )
            last_in_chunk = last_idx
        else:
            # Lane-parallel eager chunk loop. Bind a call-unique request key so the bounded sliding cross-chunk
            # tail stash engages (key None bypasses it) without colliding with other requests' block ids.
            self._g4_lane_prefill_calls = int(getattr(self, "_g4_lane_prefill_calls", 0)) + 1
            req_key = (self._g4_lane_prefill_calls << 20) + int(page_tables[0, 0, 0]) + 1
            self._bind_sliding_tail_key(req_key)
            chunk_size = get_max_prefill_chunk_size(seq_len, max_chunk)
            last_abs = max(last_idx)
            last_chunk_start = (last_abs // chunk_size) * chunk_size
            assert all(
                i >= last_chunk_start for i in last_idx
            ), f"lane prefill needs all last tokens in the final chunk (starts {last_chunk_start})"
            tt_logits = None
            for chunk_start in range(0, last_chunk_start + 1, chunk_size):
                is_last = chunk_start == last_chunk_start
                chunk_tokens = tokens[:, chunk_start : chunk_start + chunk_size]
                chunk_pt = page_tables[..., chunk_start // block_size : (chunk_start + chunk_size) // block_size]
                inputs = model.prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_tables,
                    chunk_page_table=chunk_pt,
                    batch_size=1,
                    user_id=0,
                    lane_parallel=True,
                )
                prefill_input, rot_mats_global, rot_mats_local, page_table_tt, chunk_pt_tt, *_ = inputs
                step_logits = model.ttnn_prefill_forward(
                    prefill_input,
                    rot_mats_global=rot_mats_global,
                    rot_mats_local=rot_mats_local,
                    user_id=0,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_pt_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=self._prefill_get_last_token(
                        (last_abs - last_chunk_start) if is_last else (chunk_size - 1)
                    ),
                    kv_cache=model_kv,
                    batch_size=1,
                )
                if is_last:
                    tt_logits = step_logits
                elif step_logits is not None:
                    step_logits.deallocate(True)
            last_in_chunk = [i - last_chunk_start for i in last_idx]

        # Logits are tp-gathered inside the forward, so each lane's first chip
        # holds that lane's full-vocab logits.
        shards = ttnn.get_device_tensors(tt_logits)
        out = torch.zeros(lanes, model.vocab_size, dtype=torch.float32)
        for lane, dev in enumerate(model.lane_device_indices()):
            host = ttnn.to_torch(shards[dev]).float()
            out[lane] = host[0, 0, last_in_chunk[lane] % 32, : model.vocab_size]
        tt_logits.deallocate(True)
        return out

    @staticmethod
    def _lane_prefill_table_width(page_tables, seq_len, block_size):
        """Trim a lane page table to this prefill's block grid (wider tables TT_FATAL the fill); vLLM null block 0
        pads a narrower one."""
        needed_blocks = num_blocks_in_seq(seq_len, block_size)
        if page_tables.shape[-1] > needed_blocks:
            return page_tables[..., :needed_blocks]
        if page_tables.shape[-1] < needed_blocks:
            pad = torch.zeros(*page_tables.shape[:-1], needed_blocks - page_tables.shape[-1], dtype=page_tables.dtype)
            return torch.cat([page_tables, pad], dim=-1)
        return page_tables

    @staticmethod
    def _lane_prefill_ring_tables(model, global_rows, slots, block_size):
        """Per-layer host tables for a lane prefill whose users sit at ``slots``: bounded sliding layers get each
        slot's ring-local rows, the other layers the global rows. ``global_rows`` is [users, blocks] in the lane
        layout the caller installs (one row per lane, or the lanes' decode frame rows ``lane * lane_slots + r``);
        ``slots`` names the slot of each user row in that layout. None when no layer is a bounded ring."""
        per_layer, ring_cache = [], {}
        for layer in getattr(model, "layers", []):
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            m = getattr(cfg, "cache_position_modulo", None) if cfg is not None else None
            if not m:
                per_layer.append(global_rows)
                continue
            rb = int(m) // block_size
            if rb not in ring_cache:
                ring_cache[rb] = torch.stack([torch.arange(s * rb, (s + 1) * rb, dtype=torch.int32) for s in slots])
            per_layer.append(ring_cache[rb])
        if not any(t is not global_rows for t in per_layer):
            return None
        return per_layer

    def _prefill_forward_lanes_batched(self, tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        """``k`` users per lane in one eager walk (see ``prefill_forward_lanes``).

        Each column runs the batched prefill form the serial engine uses for a true batch: its ``k`` users as one
        stream of ``k * S`` tokens (``batch_size=k``, rows ``0..k-1`` of the column's page table, per-user valid
        lengths capping the KV fill, lm_head deferred), then every user's last-token tile is projected on its own
        and read from the lane's first chip. The tables ``[lanes, k, blocks]`` lane-shard to ``[k, blocks]`` per
        column; per-layer tables, when this call installs them (``slot_ids``), use the lanes' decode frame.
        """
        model = self.model[0]
        mesh_cfg = model.mesh_config
        lanes = int(mesh_cfg.lanes)
        assert tokens.dim() == 3 and int(tokens.shape[0]) == lanes, f"tokens must be [lanes={lanes}, k, seq]"
        k, seq_len = int(tokens.shape[1]), int(tokens.shape[2])
        assert page_tables.dim() == 3 and tuple(int(x) for x in page_tables.shape[:2]) == (
            lanes,
            k,
        ), f"page_tables must be [lanes={lanes}, k={k}, blocks], got {tuple(page_tables.shape)}"
        plens = [int(p) for p in prompt_lens]
        if len(plens) != lanes * k:
            raise ValueError(f"lane-batched prefill needs {lanes * k} prompt lengths (lane-major), got {len(plens)}")
        last_r = self._lane_round_last_indices(plens, lanes, k, seq_len)
        max_chunk = int(self.model_args[0].max_prefill_chunk_size)
        if k * seq_len > max_chunk:
            raise ValueError(
                f"lane-batched prefill stream {k} x {seq_len} = {k * seq_len} tokens exceeds the prefill chunk {max_chunk}"
            )
        lane_slots = int(getattr(model, "lane_slots", 0) or 32)
        if k > lane_slots:
            raise ValueError(f"lane-batched prefill of {k} users per lane exceeds the lane's {lane_slots} slots")

        model_kv = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
        block_size = self._effective_paged_block_size(model_kv)
        page_tables = self._lane_prefill_table_width(page_tables.to(torch.int32), seq_len, block_size)

        if slot_ids is not None:
            slots_l = [int(x) for x in slot_ids]
            if len(slots_l) != lanes * k:
                raise ValueError(f"lane-batched prefill needs {lanes * k} slot ids (lane-major), got {len(slots_l)}")
            # The lanes' decode frame: lane ln's user r at row ln * lane_slots + r (the batched walk fills per-column
            # rows 0..k-1); the other rows are null and never filled.
            frame_slots = [0] * (lanes * lane_slots)
            g_frame = torch.zeros(lanes * lane_slots, int(page_tables.shape[-1]), dtype=torch.int32)
            for ln in range(lanes):
                for r in range(k):
                    g_frame[ln * lane_slots + r] = page_tables[ln, r]
                    frame_slots[ln * lane_slots + r] = slots_l[ln * k + r]
            per_layer = self._lane_prefill_ring_tables(model, g_frame, frame_slots, block_size)
            if per_layer is None:
                per_layer = [g_frame] * len(getattr(model, "layers", []))
            model._active_page_tables_per_layer = per_layer
            if lane_prefill_table_upload_enabled():
                self._install_per_layer_page_tables(model, per_layer, writer="prefill")

        inputs = model.prepare_inputs_prefill(
            tokens.reshape(lanes, k * seq_len),
            page_table=page_tables,
            batch_size=k,
            user_id=0,
            lane_parallel=True,
        )
        prefill_input, rot_mats_global, rot_mats_local, page_table_tt, *_ = inputs
        hidden = model.ttnn_prefill_forward(
            prefill_input,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=0,
            page_table=page_table_tt,
            get_last_token=-1,
            kv_cache=model_kv,
            batch_size=k,
            valid_seq_lens=[i + 1 for i in last_r],
        )
        # The batched form returns post-norm hidden states [1, 1, k * S, hidden] per column (lm_head deferred):
        # project each user's last-token tile and read the lane's row from its first chip.
        return self._lane_logits_from_hidden(model, hidden, k, seq_len, last_r, deallocate=True)

    @staticmethod
    def _lane_logits_from_hidden(model, hidden, k, seq_len, last_r, *, deallocate, rounds=None, lanes=None):
        """Host logits ``[lanes * k, vocab]`` (lane-major) from a lane frame's post-norm hidden states
        ``[1, 1, k * S, hidden]`` per column, for the ``rounds`` that carry prompts (all ``k`` by default; a padded
        replay's scratch rounds are skipped, their rows stay zero) on the ``lanes`` that carry prompts (all by
        default; a scratch lane's shard is never read). The real rounds' last-token rows are gathered on the
        device into one 32-row tile, projected by lm_head once and read back once per lane (only those rows);
        the per-round form (GEMMA4_LANE_LMHEAD_GATHER=0, and the fallback when the gather raises) projects each
        round's tile and reads the whole tile from each lane. ``deallocate`` frees the hidden states afterwards
        (never for a trace's persistent output)."""
        n_lanes = int(model.mesh_config.lanes)
        out = torch.zeros(n_lanes * k, model.vocab_size, dtype=torch.float32)
        devs = model.lane_device_indices()
        lane_ids = list(range(n_lanes)) if lanes is None else sorted({int(x) for x in lanes if 0 <= int(x) < n_lanes})
        lane_devs = {ln: devs[ln] for ln in lane_ids}
        real = list(range(k)) if rounds is None else sorted({int(r) for r in rounds if 0 <= int(r) < k})
        if real and len(real) <= 32 and lane_lmhead_gather_enabled():
            try:
                Gemma4Generator._lane_logits_gathered(model, hidden, seq_len, last_r, real, lane_devs, k, out)
                if deallocate:
                    hidden.deallocate(True)
                return out
            except Exception as e:  # noqa: BLE001 - the per-round form below is the proven path
                if not getattr(model, "_g4_lane_lmhead_gather_warned", False):
                    model._g4_lane_lmhead_gather_warned = True
                    logger.warning("Gemma4: gathered lane lm_head failed ({!r}); the per-round form from here", e)
        for r in real:
            tile_start = r * seq_len + (last_r[r] // 32) * 32
            sliced = ttnn.slice(hidden, (0, 0, tile_start, 0), (1, 1, tile_start + 32, int(hidden.shape[-1])))
            logits = model._apply_lm_head(sliced, is_decode=False)
            shards = ttnn.get_device_tensors(logits)
            for ln, dev in lane_devs.items():
                host = ttnn.to_torch(shards[dev]).float()
                out[ln * k + r] = host[0, 0, last_r[r] % 32, : model.vocab_size]
            logits.deallocate(True)
        if deallocate:
            hidden.deallocate(True)
        return out

    @staticmethod
    def _lane_logits_gathered(model, hidden, seq_len, last_r, real, lane_devs, k, out):
        """The gathered form: row ``r * S + last_r[r]`` of every real round sliced from the row-major hidden states,
        stacked (rows ``0..n-1`` of a zero-padded 32-row tile), projected once, and rows ``0..n-1`` of the logits read
        from the first chip of each lane in ``lane_devs`` (``lane -> device index``) into ``out[lane * k + r]``."""
        H = int(hidden.shape[-1])
        n = len(real)
        src = hidden
        if getattr(hidden, "dtype", ttnn.bfloat16) != ttnn.bfloat16:
            src = ttnn.typecast(hidden, ttnn.bfloat16)
        rm = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if src is not hidden and rm is not src:
            src.deallocate(True)
        pieces = []
        for r in real:
            row = r * seq_len + int(last_r[r])
            pieces.append(ttnn.slice(rm, (0, 0, row, 0), (1, 1, row + 1, H)))
        if rm is not hidden:
            rm.deallocate(True)
        stacked = pieces[0] if n == 1 else ttnn.concat(pieces, dim=2)
        if n > 1:
            for t in pieces:
                t.deallocate(True)
        if n < 32:
            padded = ttnn.pad(stacked, padding=[(0, 0), (0, 0), (0, 32 - n), (0, 0)], value=0.0)
            stacked.deallocate(True)
            stacked = padded
        tile = ttnn.to_layout(stacked, ttnn.TILE_LAYOUT)
        if tile is not stacked:
            stacked.deallocate(True)
        logits = model._apply_lm_head(tile, is_decode=False)
        V = int(logits.shape[-1])
        read = logits
        if n < 32:
            lg_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if lg_rm is not logits:
                logits.deallocate(True)
            read = ttnn.slice(lg_rm, (0, 0, 0, 0), (1, 1, n, V))
            if read is not lg_rm:
                lg_rm.deallocate(True)
        shards = ttnn.get_device_tensors(read)
        for ln, dev in lane_devs.items():
            host = ttnn.to_torch(shards[dev]).float()
            for j, r in enumerate(real):
                out[ln * k + r] = host[0, 0, j, : model.vocab_size]
        read.deallocate(True)

    @staticmethod
    def _lane_round_last_indices(plens, lanes, k, seq_len):
        """Per-round last-token indices of a lane frame (``plens`` lane-major, equal across lanes per round)."""
        last_r = []
        for r in range(k):
            row = [int(plens[ln * k + r]) - 1 for ln in range(lanes)]
            if len(set(row)) != 1:
                raise ValueError(
                    f"lane prefill round {r} needs equal prompt lengths across lanes, got {[i + 1 for i in row]}"
                )
            if not 0 <= row[0] < seq_len:
                raise ValueError(f"lane prefill round {r}: prompt length {row[0] + 1} outside [1, {seq_len}]")
            last_r.append(row[0])
        return last_r

    # ── the traced lane prefill: the lane frame captured at boot, replayed for every matching call ────────────
    # (tt/lane_prefill.py names the buckets and the replay decision)

    def _g4_lane_trace_registry(self):
        reg = getattr(self, "_g4_lane_prefill_traces", None)
        if reg is None:
            reg = self._g4_lane_prefill_traces = {}
        return reg

    def _g4_lane_frame_page_tables(self, model, page_tables, slots, block_size):
        """Per-layer host tables of a lane frame whose users sit at ``slots`` (lane-major), in the layout the
        frame's persistent buffers use: one row per lane for k == 1, the lanes' decode frame for k > 1; bounded
        sliding layers get ring-local rows, the others the global rows."""
        lanes, k = int(page_tables.shape[0]), int(page_tables.shape[1])
        lane_slots = int(getattr(model, "lane_slots", 0) or 32)
        if k == 1:
            g_rows = page_tables.reshape(lanes, -1).to(torch.int32)
            per_layer = self._lane_prefill_ring_tables(model, g_rows, [int(x) for x in slots], block_size)
            return per_layer if per_layer is not None else [g_rows] * len(getattr(model, "layers", []))
        g_frame = torch.zeros(lanes * lane_slots, int(page_tables.shape[-1]), dtype=torch.int32)
        frame_slots = [0] * (lanes * lane_slots)
        for ln in range(lanes):
            for r in range(k):
                g_frame[ln * lane_slots + r] = page_tables[ln, r]
                frame_slots[ln * lane_slots + r] = int(slots[ln * k + r])
        per_layer = self._lane_prefill_ring_tables(model, g_frame, frame_slots, block_size)
        return per_layer if per_layer is not None else [g_frame] * len(getattr(model, "layers", []))

    @staticmethod
    def _g4_lane_trace_forward(model, device_inputs, k, model_kv):
        """The lane frame's trace body, shared verbatim by the compile pass and the capture: the persistent inputs
        are the RAW token ids ``[1, k * S]`` per column and the legacy page table, so the embed runs inside the
        trace through ``transform_and_embed_prefill_inputs_device`` (as the traced single-user prefill does) and the
        forward sees ``[1, 1, k * S, hidden]`` embeddings -- handing it the token tensor reads ``shape[2]`` of a
        rank-2 tensor (window 9 stage 0: "ShapeBase[] index out of range. 2 not in [-4, 2)"). A third input, the
        ``[1, 32]`` row index (``lane_trace_gather_indices``), makes the trace gather the real rounds' last-token
        rows and project them itself (``_lane_trace_gather_lm_head``): the trace then returns row-major logits
        ``[1, 1, 32, vocab]``; without it, the post-norm hidden states for the eager gathered lm_head."""
        tt_embeds, page_table, chunk_pt, chunk_start = model.transform_and_embed_prefill_inputs_device(
            device_inputs[0], device_inputs[1], None
        )
        hidden = model.ttnn_prefill_forward(
            tt_embeds,
            rot_mats_global=None,
            rot_mats_local=None,
            user_id=0,
            page_table=page_table,
            chunk_page_table=chunk_pt,
            chunk_start_idx=chunk_start,
            get_last_token=-1,
            kv_cache=model_kv,
            batch_size=k,
            valid_seq_lens=None,
        )
        if len(device_inputs) < 3 or device_inputs[2] is None:
            return hidden
        return Gemma4Generator._lane_trace_gather_lm_head(model, hidden, device_inputs[2])

    @staticmethod
    def _lane_trace_gather_lm_head(model, hidden, index):
        """Inside the lane trace: the 32 rows ``index`` names, gathered from the frame's post-norm hidden states
        ``[1, 1, k * S, H]`` by ``ttnn.embedding`` over their row-major form (the hidden states are the embedding
        table, the index its ``[1, 32]`` uint32 input), projected once by lm_head (softcap, TP-gathered to the
        full vocab) and left row-major ``[1, 1, 32, vocab]``, so a replay reads back a row slice and runs no other
        program. Every shape here is fixed by the frame, whatever the call's real rounds (the eager gathered
        lm_head's concat / pad / slice shapes followed the real-round count: a program-cache miss per first-seen
        count on the mesh, window 12)."""
        src = hidden
        if getattr(hidden, "dtype", ttnn.bfloat16) != ttnn.bfloat16:
            src = ttnn.typecast(hidden, ttnn.bfloat16)
            hidden.deallocate(True)
        rm = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if rm is not src:
            src.deallocate(True)
        gathered = ttnn.embedding(index, rm, dtype=ttnn.bfloat16)  # [1, 32, H] row-major
        rm.deallocate(True)
        tile = ttnn.to_layout(ttnn.unsqueeze_to_4D(gathered), ttnn.TILE_LAYOUT)
        logits = model._apply_lm_head(tile, is_decode=False)  # frees ``tile``; [1, 1, 32, vocab]
        out = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if out is not logits:
            logits.deallocate(True)
        return out

    def _g4_warm_lane_trace_readback(self, model, logits):
        """Compile the row-slice program of every readback size (``LANE_TRACE_READBACK_ROWS``) on the trace's
        logits once per process, so no served call's real-round count compiles on the mesh; a failure here only
        costs that call's latency later."""
        from models.demos.gemma4.tt.lane_prefill import LANE_TRACE_READBACK_ROWS

        if getattr(self, "_g4_lane_readback_warmed", False):
            return
        self._g4_lane_readback_warmed = True
        rows, V = int(logits.shape[-2]), int(logits.shape[-1])
        for r in LANE_TRACE_READBACK_ROWS:
            if r >= rows:
                continue
            try:
                t = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, r, V))
                t.deallocate(True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Gemma4: warming the {}-row lane readback slice failed ({!r})", r, e)
                break

    def _g4_capture_lane_prefill_trace(self, bucket, kv_cache):
        """Capture the lane-frame prefill trace of ``bucket = (k, S)``: k rounds of S tokens per column, the mock
        users at slot r of every lane. The persistent trace inputs are the host-staged token ids ``[lanes, k * S]``
        and the legacy page table ``[lanes, k, blocks]`` that ``prepare_inputs_prefill(lane_parallel=True,
        trace_enabled=True)`` returns; the per-layer tables are the frame's persistent buffers (installed before the
        capture and refreshed before every replay; the key's buffers never grow: ``fit_table_width``); the trace
        embeds the tokens and returns post-norm hidden states (lm_head runs outside, per round). A compile pass over
        the persistent inputs precedes the capture (no program may load inside it). The capture is an allocation
        event for the decode-trace guard. Returns True when the bucket is (now) captured."""
        from models.demos.gemma4.tt.lane_prefill import (
            TRACE_SEQ_LENS,
            lane_kv_pool_blocks,
            lane_lmhead_in_trace_enabled,
            lane_trace_capture_width,
            lane_unique_block_rows,
        )
        from models.tt_transformers.tt.common import copy_host_to_device

        reg = self._g4_lane_trace_registry()
        k, S = int(bucket[0]), int(bucket[1])
        if (k, S) in reg:
            return True
        model = self.model[0]
        mesh = model.mesh_device
        mesh_cfg = model.mesh_config
        if mesh_cfg is None or not getattr(mesh_cfg, "lane_sharded", False):
            raise ValueError("the traced lane prefill needs the lane engine")
        lanes = int(mesh_cfg.lanes)
        lane_slots = int(getattr(model, "lane_slots", 0) or 32)
        max_chunk = int(self.model_args[0].max_prefill_chunk_size)
        if S not in TRACE_SEQ_LENS or k < 1 or k > lane_slots or k * S > max_chunk:
            raise ValueError(f"lane prefill trace bucket {k}x{S} outside the chunk {max_chunk} / slots {lane_slots}")
        model_kv = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
        block_size = self._effective_paged_block_size(model_kv)
        needed = num_blocks_in_seq(S, block_size)
        slots = [ln * lane_slots + r for ln in range(lanes) for r in range(k)]
        tokens = torch.zeros(lanes, k * S, dtype=torch.long)
        # Global rows: null block 0 (scratch writes), at the serving width so no later install grows the key's
        # buffers under the trace (the trace input keeps the frame's own ``needed`` blocks).
        width = lane_trace_capture_width(needed, getattr(self.model_args[0], "max_seq_len", 0), block_size)
        # The mock users get their own blocks (the batched fill's writers never share one), the rest null.
        page_tables = lane_unique_block_rows(lanes * k, needed, lane_kv_pool_blocks(model_kv), width).reshape(
            lanes, k, width
        )
        per_layer = self._g4_lane_frame_page_tables(model, page_tables, slots, block_size)
        widths = [None if t is None else int(t.shape[-1]) for t in per_layer]
        model._active_page_tables_per_layer = per_layer
        self._install_per_layer_page_tables(model, per_layer, writer="prefill")
        rows = self._page_tables_rows(per_layer)
        tables_fp = self._g4_lane_trace_table_fingerprint(model, rows)
        host = model.prepare_prefill_inputs_trace(
            tokens, page_table=page_tables[..., :needed].contiguous(), batch_size=k, user_id=0, lane_parallel=True
        )
        host_inputs = [host[0], host[3]]
        logits_in_trace = lane_lmhead_in_trace_enabled()
        if logits_in_trace:
            # the gather's row index (``lane_trace_gather_indices``), replicated: every lane reads the same rows
            host_inputs.append(self._lane_trace_index_host(model, [0] * 32))
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh)
        self._refresh_prefill_valid_seq_len(model_id=0, last_token_idx=S - 1)
        if k == 1:
            self._release_all_sliding_prefill_tails(0, clear_persistent=True)  # the sp0 no-tail branch, as captured

        def forward():
            return self._g4_lane_trace_forward(model, device_inputs, k, model_kv)

        model._prefill_trace_mode = True
        t0 = time.perf_counter()
        trace_id = None
        try:
            out = forward()  # compile pass over the persistent inputs
            ttnn.synchronize_device(mesh)
            out.deallocate(True)
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            out_trace = forward()
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
        except Exception:
            # A capture that failed midway is still recording: close it and release it before the error travels
            # (the boot goes on without this frame; an open capture would turn the next host write into a TT_FATAL).
            if trace_id is not None:
                self._g4_abort_trace_capture(mesh, trace_id)
            raise
        finally:
            model._prefill_trace_mode = False
            if hasattr(model, "_active_page_tables_per_layer"):
                del model._active_page_tables_per_layer
        reg[(k, S)] = {
            "trace_id": trace_id,
            "inputs": list(device_inputs),
            "output": out_trace,
            "widths": widths,
            "needed_blocks": needed,
            "rows": rows,
            "tables": tables_fp,
            "logits_in_trace": bool(logits_in_trace and len(device_inputs) >= 3),
        }
        if reg[(k, S)]["logits_in_trace"]:
            self._g4_warm_lane_trace_readback(model, out_trace)
        self._g4_arm_alloc_check("lane prefill trace capture")
        logger.info(
            "Gemma4: captured the lane prefill trace {}x{} ({} lanes, {} rounds of {} tokens per column) in {:.1f} s",
            k,
            S,
            lanes,
            k,
            S,
            time.perf_counter() - t0,
        )
        return True

    @staticmethod
    def _lane_trace_index_host(model, indices):
        """The ``[1, 32]`` uint32 host tensor of a lane trace's gather index, replicated over the mesh (the
        persistent input is written by ``copy_host_to_device`` on every replay)."""
        host = torch.tensor([list(int(i) for i in indices)], dtype=torch.int32)
        return ttnn.from_torch(
            host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
            mesh_mapper=model._replicate_to_mesh_mapper(),
        )

    @staticmethod
    def _lane_trace_logits_readback(model, logits, k, real, lane_ids, out):
        """``out[lane * k + r]`` for the real rounds ``real`` (in order) of the lanes ``lane_ids``, from the lane
        trace's row-major logits ``[1, 1, 32, vocab]`` whose rows ``0..n-1`` hold those rounds: one row slice of
        ``lane_trace_readback_rows(n)`` rows (a warmed shape; none at 32), one read per named lane (a scratch
        lane's rows are never read: the served c1 frame reads one shard, not four)."""
        from models.demos.gemma4.tt.lane_prefill import lane_trace_readback_rows

        n = len(real)
        rows, V = int(logits.shape[-2]), int(logits.shape[-1])
        want = lane_trace_readback_rows(n, rows)
        read = logits
        if want < rows:
            read = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, want, V))
        shards = ttnn.get_device_tensors(read)
        devs = model.lane_device_indices()
        for ln in lane_ids:
            host = ttnn.to_torch(shards[devs[int(ln)]]).float()
            for j, r in enumerate(real):
                out[int(ln) * k + int(r)] = host[0, 0, j, : model.vocab_size]
        if read is not logits:
            read.deallocate(True)

    @staticmethod
    def _g4_abort_trace_capture(mesh, trace_id):
        """Close and release a capture that raised midway; each step best-effort, the caller re-raises."""
        try:
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        except Exception as e:  # noqa: BLE001
            logger.warning("Gemma4: closing the failed lane trace capture {} raised {!r}", trace_id, e)
        try:
            ttnn.release_trace(mesh, trace_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Gemma4: releasing the failed lane trace capture {} raised {!r}", trace_id, e)

    @staticmethod
    def _g4_lane_trace_table_fingerprint(model, rows):
        """Identity of the persistent per-layer page-table buffers keyed by ``rows`` (object ids and device
        addresses): a trace reads the tables at the addresses it was captured with, so a replay compares this with
        the capture's. None when the model keeps no persistent tables for the key (host stubs, nothing installed)."""
        by_batch = getattr(model, "_persistent_pt_by_batch", None)
        if not by_batch or rows is None:
            return None
        persistent = by_batch.get(rows)
        if persistent is None:
            return None
        fp = []
        for t in persistent:
            addr = None
            if t is not None:
                try:
                    addr = int(t.buffer_address())
                except Exception:  # noqa: BLE001 - a mesh tensor may expose no single address
                    addr = None
            fp.append((id(t), addr))
        return tuple(fp)

    def _g4_lane_prefill_trace_replay(
        self, bucket, tokens, page_tables, per_layer, kv_cache, prompt_lens, rounds=None, lanes=None
    ):
        """Replay the captured lane-frame trace of ``bucket`` for a call: ``tokens`` [lanes, k, S] (or [lanes, S]),
        ``page_tables`` [lanes, k, blocks], ``per_layer`` the call's host per-layer tables (fitted to the captured
        widths and uploaded before the replay), ``prompt_lens`` lane-major, ``rounds`` the rounds that carry
        prompts (all by default; a padded call names its real rounds so the lm_head skips the scratch rows),
        ``lanes`` the lanes that carry prompts (all by default; only those are read back). Returns host logits
        ``[lanes * k, vocab]``, or None when the call must stay eager (a persistent buffer would have changed under
        the trace). Host phases of the call land in the model's ``_g4_pf_ms`` (install / stage / replay / gather /
        readback) for the bridge's perf line."""
        from models.demos.gemma4.tt.lane_prefill import (
            fit_table_width,
            lane_trace_gather_indices,
            lane_trace_sync_profile_enabled,
        )
        from models.tt_transformers.tt.common import copy_host_to_device

        entry = self._g4_lane_trace_registry().get((int(bucket[0]), int(bucket[1])))
        if entry is None:
            return None
        k, S = int(bucket[0]), int(bucket[1])
        model = self.model[0]
        mesh = model.mesh_device
        n_lanes = int(model.mesh_config.lanes)
        add = getattr(model, "_g4_pf_add", None) or (lambda key, seconds: None)
        if tokens.dim() == 2:
            tokens = tokens.reshape(n_lanes, 1, -1)
        assert tuple(int(x) for x in tokens.shape) == (
            n_lanes,
            k,
            S,
        ), f"tokens {tuple(tokens.shape)} vs bucket {k}x{S}"
        plens = [int(p) for p in prompt_lens]
        last_r = self._lane_round_last_indices(plens, n_lanes, k, S)
        real = list(range(k)) if rounds is None else sorted({int(r) for r in rounds if 0 <= int(r) < k})
        lane_ids = list(range(n_lanes)) if lanes is None else sorted({int(x) for x in lanes if 0 <= int(x) < n_lanes})
        fitted = [fit_table_width(pt, w) if w is not None else pt for pt, w in zip(per_layer, entry["widths"])]
        model._active_page_tables_per_layer = fitted
        try:
            model._g4_persistent_alloc_event = None
            self._install_per_layer_page_tables(model, fitted, writer="prefill")
            # The trace reads its per-layer tables at the addresses it was captured with: buffers that moved (grown by
            # a wider eager install between calls, or reallocated under this install) make the frame stale for good,
            # so the call runs eager and the frame is dropped (one warning, never a replay of freed memory).
            moved = None
            if getattr(model, "_g4_persistent_alloc_event", None):
                moved = (
                    f"the per-layer page-table buffers changed under this install ({model._g4_persistent_alloc_event})"
                )
            elif self._g4_lane_trace_table_fingerprint(model, entry.get("rows")) != entry.get("tables"):
                moved = "the key's persistent page-table buffers were reallocated since the capture"
            if moved:
                logger.warning(
                    "Gemma4: lane prefill trace {}x{} not replayed and dropped: {}; this call and the frame's later calls "
                    "run eager",
                    k,
                    S,
                    moved,
                )
                self._g4_lane_trace_registry().pop((k, S), None)
                return None
            t_stage = time.perf_counter()
            legacy = fit_table_width(page_tables.to(torch.int32), entry["needed_blocks"])
            host = model.prepare_prefill_inputs_trace(
                tokens.reshape(n_lanes, k * S), page_table=legacy, batch_size=k, user_id=0, lane_parallel=True
            )
            host_inputs = [host[0], host[3]]
            in_trace = bool(entry.get("logits_in_trace")) and len(entry["inputs"]) >= 3
            if in_trace:
                host_inputs.append(self._lane_trace_index_host(model, lane_trace_gather_indices(real, last_r, S)))
            copy_host_to_device(host_inputs, device_tensors=entry["inputs"][: len(host_inputs)], mesh_device=mesh)
            # the bounded ring fill's device cap: one length per device = the round's length (k > 1 buckets exist
            # only without bounded sliding)
            self._refresh_prefill_valid_seq_len(model_id=0, last_token_idx=last_r[0])
            t_replay = time.perf_counter()
            add("stage", t_replay - t_stage)
            ttnn.execute_trace(mesh, entry["trace_id"], cq_id=0, blocking=False)
            if lane_trace_sync_profile_enabled():
                ttnn.synchronize_device(mesh)
            t_read = time.perf_counter()
            add("replay", t_read - t_replay)
            if in_trace:
                out = torch.zeros(n_lanes * k, model.vocab_size, dtype=torch.float32)
                self._lane_trace_logits_readback(model, entry["output"], k, real, lane_ids, out)
                add("readback", time.perf_counter() - t_read)
                return out
            kw = {"lanes": lane_ids} if lanes is not None else {}
            out = self._lane_logits_from_hidden(
                model, entry["output"], k, S, last_r, deallocate=False, rounds=rounds, **kw
            )
            add("gather", time.perf_counter() - t_read)
            return out

        finally:
            if hasattr(model, "_active_page_tables_per_layer"):
                del model._active_page_tables_per_layer

    def _prefill_single_user_with_embeds(
        self, mm_embeds, tokens, page_table=None, *, kv_cache=None, num_cached_tokens=0, **kwargs
    ):
        """Prefill one user from precomputed prompt embeddings (multimodal).

        ``mm_embeds`` is the host ``[prompt_len, hidden]`` embedding of the FULL prompt:
        text rows embedded and scaled as ``embed_tokens`` does, soft tokens in their
        placeholder slots. The text prefill then runs unchanged -- page tables,
        per-layer tables, chunking, cached-prefix offsets, bounded rings -- except
        that ``prepare_inputs_prefill`` reads the installed override instead of
        embedding ``tokens`` (which still drive every shape). Eager only.
        """
        if int(kwargs.get("batch_size", 1) or 1) != 1 or isinstance(kwargs.get("user_id"), (list, tuple)):
            raise NotImplementedError("multimodal prefill is per user (batched prefill is off for multimodal batches)")
        model = self.model[kwargs.get("model_id", -1)]
        base = int(num_cached_tokens or 0)
        suffix = mm_embeds[base : base + int(tokens.shape[-1])]
        if int(suffix.shape[0]) == 0:
            raise ValueError(f"cached prefix {base} covers the whole merged prompt ({int(mm_embeds.shape[0])} rows)")
        model._g4_mm_prefill_embeds = (suffix.contiguous(), base)
        # Image-span bidirectional attention (HF ``use_bidirectional_attention="vision"``): the absolute vision
        # spans of the prompt (``tokens`` is the uncached suffix, cut at ``base``); the model keeps them
        # bidirectional in every chunk and continuation of this prefill (``_g4_mm_prefill_vision``).
        if getattr(self, "image_token_id", None) is not None:
            from models.demos.gemma4.tt.attention.mm_prefill_vision import vision_spans_of_prompt

            model._g4_mm_prefill_blocks = (
                vision_spans_of_prompt(tokens, self.image_token_id, getattr(self, "video_token_id", None), base=base)
                or None
            )
        try:
            return self.prefill_forward_single_user_text(
                tokens, page_table, kv_cache=kv_cache, num_cached_tokens=num_cached_tokens, **kwargs
            )
        finally:
            model._g4_mm_prefill_embeds = None
            model._g4_mm_prefill_blocks = None

    def _prefill_forward_single_user_text_eager(
        self,
        tokens,
        page_table=None,
        user_id=0,
        last_token_idx=None,
        kv_cache=None,
        model_id=-1,
        num_cached_tokens: int = 0,
        batch_size=1,
        **kwargs,
    ):
        """Gemma4-local eager prefill (single- or multi-chunk).

        Localized from ``Generator.prefill_forward_single_user_text`` so bounded
        ring fill / last-chunk expansion does not require shared-generator hooks.
        """
        seq_len = tokens.shape[-1]
        use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
        use_prefix_caching = num_cached_tokens > 0
        if use_chunked_prefill or use_prefix_caching:
            assert page_table is not None, "page_table must be provided for chunked prefill"
            assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
            assert last_token_idx is not None and last_token_idx < seq_len + num_cached_tokens, (
                f"last_token_idx must be provided and less than seq_len + num_cached_tokens: "
                f"last_token_idx={last_token_idx}, seq_len={seq_len}, num_cached_tokens={num_cached_tokens}"
            )

            if use_chunked_prefill:
                chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size)
            else:
                chunk_size = seq_len

            last_token_idx_in_seq = last_token_idx - num_cached_tokens
            last_token_idx_in_chunk = last_token_idx_in_seq % chunk_size
            last_chunk_start = (last_token_idx_in_seq // chunk_size) * chunk_size
            chunk_source_page_table, block_size = self._chunk_prefill_page_table(
                page_table, user_id=user_id, model_id=model_id, kv_cache=kv_cache
            )
            last_chunk_start, last_token_idx_in_chunk = self._adjust_last_prefill_chunk(
                last_chunk_start=last_chunk_start,
                last_token_idx_in_chunk=last_token_idx_in_chunk,
                last_token_idx_in_seq=last_token_idx_in_seq,
                chunk_size=chunk_size,
                block_size=block_size,
                model_id=model_id,
            )
            page_table_user = chunk_source_page_table[user_id : user_id + 1, :]
            # Cap page-table width to the *real* (unpadded) sequence so pad
            # tokens in the last power-of-2 chunk cannot address real blocks.
            # Extra columns required by the padded chunk grid use 0 (vLLM null
            # block). Fill skip is valid_seq_len, not page-table -1.
            real_seq_len = int(last_token_idx) + 1
            needed_blocks = num_blocks_in_seq(real_seq_len, block_size)
            chunk_grid_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size)
            if page_table_user.shape[1] > needed_blocks:
                page_table_user = page_table_user[:, :needed_blocks]
            num_padding_blocks = max(0, chunk_grid_blocks - page_table_user.shape[1])
            if num_padding_blocks > 0:
                page_table_user_padded = torch.cat(
                    [
                        page_table_user,
                        torch.zeros((1, num_padding_blocks), dtype=torch.int32),
                    ],
                    dim=-1,
                )
            else:
                page_table_user_padded = page_table_user
            CHUNK_USER_ID = 0

            # Inject an expanded last start when adjust moves it off the chunk grid, and keep every
            # boundary out of a vision span (a span cut in two would be prefilled without its later rows).
            last_abs = num_cached_tokens + last_chunk_start
            from models.demos.gemma4.tt.attention.mm_prefill_vision import plan_chunk_starts_outside_vision_spans

            chunk_starts = plan_chunk_starts_outside_vision_spans(
                num_cached_tokens,
                last_token_idx_in_seq + 1,  # the real suffix length (``seq_len`` is the padded one)
                chunk_size,
                last_abs,
                getattr(self.model[model_id], "_g4_mm_prefill_blocks", None) or (),
            )
            if chunk_starts[-1] != last_abs:
                # The last chunk starts earlier to hold a whole vision span: its last-token row moves with it.
                last_abs = chunk_starts[-1]
                last_token_idx_in_chunk = last_token_idx_in_seq - (last_abs - num_cached_tokens)

            # G4_CHUNK_PROFILE=1: per-chunk host wall timing (prepare vs
            # forward-submit). Submission is async, so 'fwd' includes device
            # backpressure once the pipeline fills — a flat fwd series means
            # device-bound at that rate; a prep-heavy series means host work
            # between submissions is the tax.
            _profile_chunks = os.environ.get("G4_CHUNK_PROFILE", "0") == "1"
            _t_prev = time.perf_counter() if _profile_chunks else 0.0

            for chunk_index, chunk_start in enumerate(chunk_starts):
                chunk_end = chunk_start + chunk_size
                chunk_start_relative = chunk_start - num_cached_tokens
                is_last_chunk = chunk_start == last_abs
                # A chunk shortened to keep a vision span whole ends where the next one starts; its
                # valid row count caps the KV fill and the sliding-tail stash like a short last chunk.
                next_start = chunk_starts[chunk_index + 1] if not is_last_chunk else chunk_end
                chunk_end_relative = min(chunk_end, next_start) - num_cached_tokens
                chunk_end_relative = min(chunk_end_relative, seq_len)
                chunk_valid_rows = chunk_end_relative - chunk_start_relative
                shortened_chunk = (not is_last_chunk) and chunk_valid_rows < chunk_size

                chunk_tokens = tokens[:, chunk_start_relative:chunk_end_relative]
                if chunk_tokens.shape[-1] < chunk_size:
                    chunk_tokens = torch.nn.functional.pad(chunk_tokens, (0, chunk_size - chunk_tokens.shape[-1]))

                chunk_page_table = page_table_user_padded[:, chunk_start // block_size : chunk_end // block_size]
                # Continuation chunks must see real block IDs (>0). All 0 / empty
                # means the source table was truncated to the first scheduler
                # chunk width (vLLM APC / #51186) — fill would only touch the
                # null block and full-attn KV past that point would be empty.
                if chunk_start > 0 and chunk_page_table.numel() > 0:
                    n_valid = int((chunk_page_table > 0).sum().item())
                    if n_valid == 0:
                        logger.warning(
                            "Gemma4 APC chunk_page_table has no real block ids "
                            "at chunk_start={} (page_table_cols={} needed≈{} "
                            "block_size={}). Continuation KV will not be written "
                            "— check _get_prefill_user_page_table full-prompt width.",
                            chunk_start,
                            int(page_table_user_padded.shape[1]),
                            needed_blocks,
                            block_size,
                        )
                _t_gap = (time.perf_counter() - _t_prev) if _profile_chunks else 0.0
                chunk_inputs = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                    batch_size=batch_size,
                    user_id=CHUNK_USER_ID,
                    **kwargs,
                )
                _t_prep = (time.perf_counter() - _t_prev - _t_gap) if _profile_chunks else 0.0
                (
                    chunk_prefill_input,
                    chunk_rot_mats_global_prefill,
                    chunk_rot_mats_local_prefill,
                    page_table_tt,
                    chunk_page_table_tt,
                    _chunk_start_idx_tt,
                ) = chunk_inputs
                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=chunk_rot_mats_global_prefill,
                    rot_mats_local=chunk_rot_mats_local_prefill,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=self._chunk_prefill_get_last_token(
                        is_last_chunk=is_last_chunk,
                        last_token_idx_in_chunk=last_token_idx_in_chunk if is_last_chunk else (chunk_size - 1),
                        chunk_size=chunk_size,
                    ),
                    kv_cache=kv_cache,
                    batch_size=batch_size,
                    valid_seq_lens=chunk_valid_rows if shortened_chunk else None,
                    **kwargs,
                )
                if _profile_chunks:
                    _t_now = time.perf_counter()
                    logger.info(
                        "[g4-chunkprof] chunk_start={} gap={:.1f}ms prep={:.1f}ms fwd={:.1f}ms total={:.1f}ms",
                        chunk_start,
                        _t_gap * 1e3,
                        _t_prep * 1e3,
                        (_t_now - _t_prev - _t_gap - _t_prep) * 1e3,
                        (_t_now - _t_prev) * 1e3,
                    )
                    _t_prev = _t_now
                if is_last_chunk:
                    return tt_logits
                del tt_logits
            raise RuntimeError("Gemma4 eager multi-chunk prefill produced no last-chunk logits")

        inputs = self.model[model_id].prepare_inputs_prefill(
            tokens,
            page_table=page_table,
            batch_size=batch_size,
            user_id=user_id,
            **kwargs,
        )
        prefill_input, rot_mats_global_prefill, rot_mats_local_prefill, page_table_tt, *_ = inputs
        # Batched: keep get_last_token=-1 (deferred lm_head / full hidden return)
        # but pass per-slot real lengths so pad rows are not written into KV.
        valid_seq_lens = None
        if batch_size > 1 and isinstance(last_token_idx, (list, tuple)):
            valid_seq_lens = [int(i) + 1 for i in last_token_idx]
        return self.model[model_id].ttnn_prefill_forward(
            prefill_input,
            rot_mats_global=rot_mats_global_prefill,
            rot_mats_local=rot_mats_local_prefill,
            user_id=user_id,
            page_table=page_table_tt,
            get_last_token=(-1 if batch_size > 1 else self._prefill_get_last_token(last_token_idx)),
            kv_cache=kv_cache,
            batch_size=batch_size,
            valid_seq_lens=valid_seq_lens,
        )

    def _gemma4_eager_token_feedback_buffer(self, model_id: int):
        """Padded decode-token buffer for async-safe sampling writeback (#51186).

        Non-PLI Gemma4 pads tokens to ``[1,1,1,_DECODE_TOKEN_FEEDBACK_WIDTH]`` so
        the next decode step can read the sampled id from the captured trace
        input. Returns None for PLI / missing traces / incompatible shapes.
        """
        model = self.model[model_id]
        if getattr(model, "_tt_vllm_always_refresh_decode_trace_inputs", False):
            return None
        pad_w = getattr(model, "_DECODE_TOKEN_FEEDBACK_WIDTH", None)
        if pad_w is None:
            return None
        # ``_prev_decode_batch`` is only assigned once a decode step has run
        # (see ``decode_forward`` / Gemma4Generator.__init__), but vLLM's decode
        # *warmup* reaches this buffer first — on a TP mesh with on-device
        # sampling that raised AttributeError and killed EngineCore before the
        # server came up. Match the defensive getattr used at the other read sites.
        prev_decode_batch = getattr(self, "_prev_decode_batch", None)
        sampling_trace_key = (True, prev_decode_batch) if prev_decode_batch is not None else None
        if sampling_trace_key is None or not self.trace_inputs_decode[sampling_trace_key]:
            # No decode trace captured yet (the two-phase warmup's traceless
            # pass). Hand back a persistent dummy of the real feedback buffer's
            # spec instead of None: with None the warmup-phase-1 eager sample
            # runs the no-output-tensor SamplingDeviceOperation variant, and
            # the with-output variant then FIRST-COMPILES during phase 2 — a
            # program-cache allocation while traces are live (#30187 class;
            # TT_METAL_TRACE_ALLOC_TRACKING=1 flags it at decode warmup).
            # Allocating this dummy pre-capture keeps its address safe and
            # compiles the same op variant the runtime path replays.
            dummy = getattr(self, "_g4_warmup_token_feedback_dummy", None)
            if dummy is None:
                dummy = ttnn.from_torch(
                    torch.zeros(1, 1, 1, int(pad_w), dtype=torch.int64),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    device=model.mesh_device,
                    mesh_mapper=(
                        ttnn.ReplicateTensorToMesh(model.mesh_device)
                        if model.mesh_device.get_num_devices() > 1
                        else None
                    ),
                )
                self._g4_warmup_token_feedback_dummy = dummy
            return dummy
        feedback = self._decode_token_feedback_buffer(model, self.trace_inputs_decode[sampling_trace_key][model_id])
        if feedback is None:
            return None
        fb_shape = list(feedback.shape)
        if len(fb_shape) == 4 and fb_shape[-1] >= int(pad_w):
            return feedback
        return None

    def _gemma4_commit_sampled_tokens_to_feedback(self, sampled_outputs) -> bool:
        """Host-roundtrip the sampled ids into the decode token feedback buffer.

        In-place ``tt_out_tok=`` writeback is unreliable across sampling paths:
        force-argmax emits ``[1,1,B]`` while the feedback buffer is ``[1,1,1,32]``,
        and under ``async_scheduling`` a missed writeback restages the previous
        token ("TheThe user user…"). Always commit via host so the next traced
        decode step sees the just-sampled ids.
        """
        wrote = False
        for i, sampled in enumerate(sampled_outputs):
            feedback = self._gemma4_eager_token_feedback_buffer(i)
            if feedback is None:
                continue
            tok = sampled[0] if isinstance(sampled, tuple) else sampled
            if tok is None:
                continue
            try:
                host = ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1).to(torch.int64)
            except Exception:
                continue
            pad_w = int(feedback.shape[-1])
            buf = torch.zeros(pad_w, dtype=torch.int64)
            n = min(pad_w, int(host.numel()))
            if n <= 0:
                continue
            buf[:n] = host[:n]
            mesh = self.model_args[i].mesh_device
            replicate = (
                ttnn.ReplicateTensorToMesh(mesh)
                if hasattr(mesh, "get_num_devices") and mesh.get_num_devices() > 1
                else None
            )
            host_tt = ttnn.from_torch(
                buf.reshape(1, 1, 1, pad_w),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            )
            ttnn.copy_host_to_device_tensor(host_tt, feedback)
            wrote = True
        return wrote

    def sample_decode_on_device(
        self,
        tt_logits,
        sampling_params,
        start_pos=None,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        enable_trace=False,
        *,
        reload_sampling_params: bool,
        reset_sampling_state: bool,
        reload_inputs: bool | None = None,
        skip_precompile: bool = False,
    ):
        """Eager ``tt_out_tok`` inject for padded decode feedback (#51186).

        Shared ``Generator.sample_decode_on_device`` only passes ``tt_out_tok`` when
        the sampling *trace* is enabled. Gemma4 skips sampling traces
        (``_tt_disable_sampling_trace``), so inject the padded feedback buffer for
        eager sample. Sync only after a real eager writeback — unconditional
        host-commit + double ``synchronize_device`` per token (~2× mesh sync)
        tanks decode tok/s on the metal demo / non-async path.

        Opt into host-roundtrip commit with ``GEMMA4_HOST_COMMIT_FEEDBACK=1``
        for async force-argmax shape mismatches (``[1,1,B]`` ↛ ``[1,1,1,32]``).
        """
        restores = []
        wrote_feedback = False
        host_commit = os.environ.get("GEMMA4_HOST_COMMIT_FEEDBACK", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            for i in range(self.data_parallel):
                sampling_module = getattr(self.model[i], "sampling", None)
                if sampling_module is None:
                    continue
                feedback = self._gemma4_eager_token_feedback_buffer(i)
                if feedback is None:
                    continue
                orig_sample = sampling_module.sample

                def _make_sample(orig, fb):
                    def _sample(logits, *, enable_trace=True, tt_out_tok=None, **kwargs):
                        nonlocal wrote_feedback
                        if tt_out_tok is None:
                            tt_out_tok = fb
                            if not enable_trace:
                                wrote_feedback = True
                        return orig(
                            logits,
                            enable_trace=enable_trace,
                            tt_out_tok=tt_out_tok,
                            **kwargs,
                        )

                    return _sample

                sampling_module.sample = _make_sample(orig_sample, feedback)
                restores.append((sampling_module, orig_sample))

            out = super().sample_decode_on_device(
                tt_logits,
                sampling_params,
                start_pos=start_pos,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                enable_trace=enable_trace,
                reload_sampling_params=reload_sampling_params,
                reset_sampling_state=reset_sampling_state,
                reload_inputs=reload_inputs,
                skip_precompile=skip_precompile,
            )
            if host_commit:
                try:
                    mesh = getattr(self.model_args[0], "mesh_device", None)
                    if mesh is not None:
                        ttnn.synchronize_device(mesh)
                except Exception:
                    pass
                self._gemma4_commit_sampled_tokens_to_feedback(out)
                wrote_feedback = True
            # Default: skip host sync after eager sample. Single-CQ metal demos
            # already order sample → next decode on the same queue; a full mesh
            # sync every token was costing ~4–8 tok/s on LB 12B. Re-enable with
            # GEMMA4_SAMPLE_FEEDBACK_SYNC=1 for multi-CQ / async races (#51186).
            need_sync = wrote_feedback and os.environ.get("GEMMA4_SAMPLE_FEEDBACK_SYNC", "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if need_sync:
                try:
                    mesh = getattr(self.model_args[0], "mesh_device", None)
                    if mesh is not None:
                        ttnn.synchronize_device(mesh)
                except Exception:
                    pass
            return out
        finally:
            for sampling_module, orig_sample in restores:
                sampling_module.sample = orig_sample

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params: SamplingParams = None,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap=None,
        defer_device_sampling: bool = False,
        *,
        reload_inputs: bool,
        reload_page_table: bool,
        reload_sampling_params: bool,
        reset_sampling_state: bool,
        skip_trace_precompile: bool = False,
        prepare_trace: bool = False,
        **kwargs,
    ):
        """Gemma4 decode with batch-keyed traces and explicit reload commands."""
        if "reset_batch" in kwargs:
            raise TypeError("reset_batch is not supported by decode input update contract v1")
        # Sequential prefill narrows per-layer tables to one row; decode needs
        # the full batch even when the prefill entry point did not restore it.
        self._clear_sequential_batch_page_tables()
        if self.mode != Mode.DECODE:
            self.mode = Mode.DECODE

        for i in range(len(self.model)):
            self.model[i].switch_mode(Mode.DECODE)

        on_device_sampling = (sampling_params is not None) or defer_device_sampling
        if not enable_trace and not reload_inputs:
            raise ValueError("Non-traced decode rebuilds all forward inputs and requires reload_inputs=True")
        self._decode_reload_inputs = reload_inputs

        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        # Gemma4 keeps a trace for each sampling mode and decode bucket. This
        # records the current trace key for its token-feedback buffer; it does
        # not decide whether any input is reloaded.
        self._prev_decode_batch = int(tokens[0].shape[0]) if tokens else 1

        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "on_device_sampling": on_device_sampling,
        }

        if enable_trace:
            tt_decode_output = self._decode_forward_trace_text(
                **decode_kwargs,
                reload_inputs=reload_inputs,
                reload_page_table=reload_page_table,
                skip_precompile=skip_trace_precompile,
            )
        elif prepare_trace:
            tt_decode_output = self._prepare_decode_trace_variant(**decode_kwargs)
        else:
            tt_decode_output = self._decode_forward_no_trace_text(
                **decode_kwargs,
            )

        if defer_device_sampling and on_device_sampling:
            return tt_decode_output
        if sampling_params is not None:
            tt_decode_output = self.sample_decode_on_device(
                tt_decode_output,
                sampling_params=sampling_params,
                start_pos=start_pos,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                enable_trace=enable_trace,
                reload_sampling_params=reload_sampling_params,
                reset_sampling_state=reset_sampling_state,
                reload_inputs=reload_inputs,
                skip_precompile=skip_trace_precompile,
            )
        if read_from_device:
            to_host = self.read_decode_output(tt_decode_output)
            output = self.process_decode_output_host(to_host, is_tokens=(sampling_params is not None))
            if sampling_params is None:
                self._apply_sampling_slot_remap(slot_remap)
            return output
        if sampling_params is None:
            self._apply_sampling_slot_remap(slot_remap)
        return tt_decode_output

    def _decode_trace_key(self, on_device_sampling, tokens):
        return on_device_sampling, int(tokens[0].shape[0]) if tokens else 1

    def _prepare_decode_trace_text(self, *args, **kwargs):
        prepared = super()._prepare_decode_trace_text(*args, **kwargs)
        # The vLLM bridge routes these through model attributes. Recording may
        # happen after another batch or prefill has changed that routing.
        prepared["gemma4_page_tables"] = [getattr(m, "_active_page_tables_per_layer", None) for m in self.model]
        return prepared

    def _record_decode_trace_text(self, prepared):
        previous = [getattr(m, "_active_page_tables_per_layer", None) for m in self.model]
        try:
            for model, tables in zip(self.model, prepared["gemma4_page_tables"]):
                model._active_page_tables_per_layer = tables
            return super()._record_decode_trace_text(prepared)
        finally:
            for model, tables in zip(self.model, previous):
                model._active_page_tables_per_layer = tables

    # ── Decode-trace allocation guard ──────────────────────────────────────
    # A trace replay writes its intermediates to the addresses they had at capture time. Anything
    # allocated after the capture and still alive at a replay (per-layer page tables for a new batch
    # key, a sliding-window tail stash, a program's semaphores, ...) can sit in that range and be
    # overwritten: the symptom is a request that is right until the first traced decode and wrong
    # afterwards. The allocator level right after a capture is the quiescent baseline; a higher level
    # at the next replay means such an allocation happened, so the traces are released and re-captured
    # against the new layout. Bounded by GEMMA4_TRACE_ALLOC_GUARD_MAX re-captures (default 8);
    # GEMMA4_TRACE_ALLOC_GUARD=0 disables the guard.
    _g4_trace_guard_recaptures: int = 0

    @property
    def _g4_trace_alloc_baseline(self) -> dict:
        """``{decode trace key: allocator levels at its capture}`` of this generator (created on first use). A
        class-level dict here was mutated in place by every generator of the process before any instance
        assignment, so two generators shared one set of baselines."""
        baselines = self.__dict__.get("_g4_trace_alloc_baselines")
        if baselines is None:
            baselines = self.__dict__["_g4_trace_alloc_baselines"] = {}
        return baselines

    @_g4_trace_alloc_baseline.setter
    def _g4_trace_alloc_baseline(self, value: dict) -> None:
        self.__dict__["_g4_trace_alloc_baselines"] = value

    _g4_alloc_dump_count: int = 0

    def _g4_alloc_dump(self, prefix: str) -> None:
        """GEMMA4_ALLOC_DUMPS=1: write the allocator block tables (generated/reports) at trace captures and
        the first prefills, to diff for allocations that outlive a capture."""
        if os.environ.get("GEMMA4_ALLOC_DUMPS", "0") != "1" or self._g4_alloc_dump_count >= 40:
            return
        self._g4_alloc_dump_count += 1
        try:
            mesh = self.model_args[0].mesh_device
            ttnn.synchronize_device(mesh)
            ttnn.dump_device_memory_state(mesh, f"{self._g4_alloc_dump_count:02d}_{prefix}")
            logger.info(
                "Gemma4: allocator dump {:02d}_{} (levels {})",
                self._g4_alloc_dump_count,
                prefix,
                self._g4_alloc_levels(),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logger.warning("Gemma4: allocator dump failed: {}", exc)

    def _g4_alloc_levels(self):
        """Per mesh: (DRAM bytes allocated bottom-up, worker-L1 bytes allocated).

        Trace intermediates are bottom-up in DRAM and top-down in L1, so a persistent allocation that
        can collide with a replay raises the DRAM bottom-up total (kernel binaries are top-down and
        excluded) or the L1 total.
        """
        levels = []
        for args in self.model_args:
            mesh = getattr(args, "mesh_device", None)
            try:
                view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
                levels.append(_g4_dram_bottom_up_bytes(view))
            except Exception:  # noqa: BLE001 - no view means nothing to guard
                levels.append(0)
            try:
                levels.append(int(ttnn.get_memory_view(mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank))
            except Exception:  # noqa: BLE001
                levels.append(0)
        return tuple(levels)

    # When the allocator walk runs (``_g4_alloc_levels``: two get_memory_view block tables, 15-20 ms of host on
    # the decode step that follows a prefill at the 26B's allocation count). GEMMA4_TRACE_ALLOC_GUARD_ARM:
    #   prefill           - every prefill arms the next replay's check (the rule before 2026-10-24);
    #   alloc (default)   - only allocation events arm it: a per-layer page-table buffer allocated or grown
    #                       (``Gemma4Model._page_tables_to_ttnn``), a prefill trace (re)captured, a chunked or
    #                       continuation prefill (its sliding-tail stash), a multimodal prefill (encoder
    #                       allocations), plus a periodic check every GEMMA4_TRACE_ALLOC_GUARD_PERIOD replays
    #                       (default 32; 0 = events only). The decision on the levels is the same in both modes.
    # Default flipped to ``alloc`` on 2026-10-24: QB2 26B c1 TTFT 103 -> 87 ms, c32 TTFT -84 ms (devq 0486 A/B);
    # the Galaxy lane engine's post-eval hang reproducer (6144/2048 x128) and c128 ran without a hang in that mode.
    def _g4_guard_arm_mode(self) -> str:
        mode = os.environ.get("GEMMA4_TRACE_ALLOC_GUARD_ARM", "alloc")
        return mode if mode in ("prefill", "alloc") else "alloc"

    def _g4_arm_alloc_check(self, reason: str) -> bool:
        """Arm the next decode replay's allocator check; ``prefill`` (a prefill ran, nothing known allocated) arms
        only in the ``prefill`` mode, every other reason is an allocation event and arms in both modes."""
        if reason == "prefill" and self._g4_guard_arm_mode() == "alloc":
            return False
        self._g4_alloc_check_pending = True
        self._g4_alloc_check_reason = reason
        return True

    @staticmethod
    def _g4_prefill_trace_ids(generator):
        ids = getattr(generator, "trace_id_prefill", None) or {}
        return {k: v for k, v in dict(ids).items() if v is not None}

    def _g4_arm_after_prefill(self, *, prefill_trace_ids_before, chunked: bool, multimodal: bool) -> list:
        """The allocation events of one prefill call (both modes): page-table buffers allocated or grown on any
        model, a prefill trace (re)captured, a chunked / continuation prefill, a multimodal prefill."""
        reasons = []
        for m in self.model:
            event = getattr(m, "_g4_persistent_alloc_event", None)
            if event:
                m._g4_persistent_alloc_event = None
                reasons.append(f"page tables {event}")
        if self._g4_prefill_trace_ids(self) != (prefill_trace_ids_before or {}):
            reasons.append("prefill trace capture")
        if chunked:
            reasons.append("chunked prefill")
        if multimodal:
            reasons.append("multimodal prefill")
        for reason in reasons:
            self._g4_arm_alloc_check(reason)
        return reasons

    def _g4_trace_guard_should_recapture(self, decode_trace_key) -> bool:
        if os.environ.get("GEMMA4_TRACE_ALLOC_GUARD", "1") == "0":
            return False
        baseline = self._g4_trace_alloc_baseline.get(decode_trace_key)
        if baseline is None:
            # Captured by a warmup path: the level at this first replay is the baseline.
            self._g4_trace_alloc_baseline[decode_trace_key] = self._g4_alloc_levels()
            return False
        if not getattr(self, "_g4_alloc_check_pending", True):
            # nothing but replays since the last check: no new persistent allocation (the ``alloc`` arming
            # mode still walks the allocator every GEMMA4_TRACE_ALLOC_GUARD_PERIOD replays)
            self._g4_replays_since_check = int(getattr(self, "_g4_replays_since_check", 0)) + 1
            period = int(os.environ.get("GEMMA4_TRACE_ALLOC_GUARD_PERIOD", "32"))
            if self._g4_guard_arm_mode() != "alloc" or period <= 0 or self._g4_replays_since_check < period:
                return False
            self._g4_alloc_check_reason = "periodic"
        self._g4_alloc_check_pending = False
        self._g4_replays_since_check = 0
        current = self._g4_alloc_levels()
        if all(c <= b for b, c in zip(baseline, current)):
            # The level right after the capture can still include tensors the warmup frees later; the
            # lowest level seen at a replay is the true quiescent baseline, so track it downwards.
            self._g4_trace_alloc_baseline[decode_trace_key] = tuple(min(b, c) for b, c in zip(baseline, current))
            return False
        limit = int(os.environ.get("GEMMA4_TRACE_ALLOC_GUARD_MAX", "8"))
        now = time.monotonic()
        if self._g4_trace_guard_recaptures >= limit:
            # Past the budget, keep re-capturing but no more than once per interval: a load ramp with
            # chunked prefills grows the allocator at nearly every step (buffers that live across a
            # replay), and a first-time path landing after a spent budget (the first lane group on the
            # Galaxy) was replayed over and hung the device. 0 = never re-capture past the budget.
            interval = float(os.environ.get("GEMMA4_TRACE_ALLOC_GUARD_MIN_INTERVAL_S", "30"))
            last = float(getattr(self, "_g4_trace_guard_last_recapture_t", 0.0))
            if interval <= 0 or now - last < interval:
                if not getattr(self, "_g4_trace_guard_limit_logged", False):
                    self._g4_trace_guard_limit_logged = True
                    logger.warning(
                        "Gemma4: device allocations grew again after a decode-trace capture (baseline {} -> {}) "
                        "and the re-capture budget ({}) is spent; further growth re-captures at most every {} s "
                        "(GEMMA4_TRACE_ALLOC_GUARD_MIN_INTERVAL_S)",
                        baseline,
                        current,
                        limit,
                        interval,
                    )
                return False
        self._g4_trace_guard_recaptures += 1
        self._g4_trace_guard_last_recapture_t = now
        logger.warning(
            "Gemma4: device allocations grew after the decode-trace capture (bytes per bank {} -> {}); "
            "re-capturing the decode traces ({}/{})",
            baseline,
            current,
            self._g4_trace_guard_recaptures,
            limit,
        )
        return True

    def _g4_release_decode_traces(self, reason: str) -> None:
        """Release every captured decode trace (and the sampling traces bound to its logits)."""
        released = 0
        for _key, tids in list(self.trace_ids_decode.items()):
            if not tids:
                continue
            for mid, tid in tids.items():
                try:
                    ttnn.release_trace(self.model_args[mid].mesh_device, tid)
                    released += 1
                except Exception as exc:  # noqa: BLE001 - keep going, the dict is reset below
                    logger.warning("Gemma4: decode trace release failed: {}", exc)
        for m in self.model:
            sampling = getattr(m, "sampling", None)
            if sampling is not None and hasattr(sampling, "reset_trace"):
                sampling.reset_trace()
        # The bound input/output tensors are only dropped, not freed: the sampler and the serving loop
        # may still hold the token/log-prob outputs of the last step. The old buffers stay allocated,
        # which also keeps the re-capture from landing on them.
        self.trace_ids_decode = defaultdict(lambda: None)
        self.trace_inputs_decode = defaultdict(lambda: None)
        self.trace_output_decode = defaultdict(lambda: None)
        # ``_prev_decode_batch`` is kept: the step re-captures the same batch key, and the sampled-token
        # feedback (``_gemma4_eager_token_feedback_buffer``) finds the new trace inputs through it; with
        # it cleared the token would go to the warmup dummy and the next step would replay the old token.
        self.prev_page_table = None
        self._g4_trace_alloc_baseline = {}
        self._g4_recapture_without_precompile = True
        logger.warning("Gemma4: released {} decode trace(s): {}", released, reason)

    def _decode_forward_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        on_device_sampling=False,
        *,
        reload_inputs: bool,
        reload_page_table: bool,
        skip_precompile: bool = False,
    ):
        """Gemma4 decode traces keyed by ``(on_device_sampling, batch)``.

        Shared ``Generator`` keys traces by sampling mode only. Gemma4 warms
        B=1 and B=max separately (vLLM nearest-bucket / P150x8), so the key
        must include the per-DP chunk batch or replay hits the wrong Metal
        graph. Reload decisions are supplied by the caller.
        """
        from models.tt_transformers.tt.common import copy_host_to_device
        from models.tt_transformers.tt.generator import DECODE_PAGE_TABLE_INPUT_IDX

        batch = int(tokens[0].shape[0]) if tokens else 1
        decode_trace_key = self._decode_trace_key(on_device_sampling, tokens)
        if self.trace_ids_decode[decode_trace_key] and self._g4_trace_guard_should_recapture(decode_trace_key):
            self._g4_release_decode_traces("device allocations grew after the decode-trace capture")
        if not self.trace_ids_decode[decode_trace_key]:
            # A re-capture after a guard release happens inside a live request: the programs are warm,
            # and the compile pass would run the sampler's warmup-only precompile (all configs, token
            # count update) against that request's device state, so only the graph is recorded.
            skip_precompile = skip_precompile or bool(getattr(self, "_g4_recapture_without_precompile", False))
            self._g4_recapture_without_precompile = False
            trace_ids, tt_out_trace, *device_inputs = self._capture_decode_trace_text(
                tokens,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                on_device_sampling=on_device_sampling,
                skip_precompile=skip_precompile,
            )
            self.trace_ids_decode[decode_trace_key] = trace_ids
            self.trace_inputs_decode[decode_trace_key] = device_inputs
            self.trace_output_decode[decode_trace_key] = tt_out_trace
            self._g4_trace_alloc_baseline[decode_trace_key] = self._g4_alloc_levels()
            self._g4_alloc_dump(f"cap_b{batch}_")

        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None

            if reload_inputs:
                host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)
                copy_host_to_device(
                    host_tensors=host_inputs_i,
                    device_tensors=self.trace_inputs_decode[decode_trace_key][i],
                )
            elif reload_page_table:
                host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)
                host_page_table = host_inputs_i[DECODE_PAGE_TABLE_INPUT_IDX]
                device_page_table = self.trace_inputs_decode[decode_trace_key][i][DECODE_PAGE_TABLE_INPUT_IDX]
                if host_page_table is not None:
                    ttnn.copy_host_to_device_tensor(host_page_table, device_page_table)

        for i, trace_id in self.trace_ids_decode[decode_trace_key].items():
            ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
        return self.trace_output_decode[decode_trace_key]

    # ── Multimodal (vision) helpers shared by the demo and the vLLM generators ──────────────────────────────────
    # Both generators set ``vision_tower`` / ``embed_vision`` / ``is_multimodal`` (``Gemma4Generator.__init__``,
    # ``Gemma4ForCausalLM.initialize_vllm_model``); only the prefill-side merge of the soft tokens differs.

    def _replicate_mapper(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(self.mesh_device, "shape") else None

    def encode_vision(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor):
        """Run the on-device vision tower + projector and return the VALID projected soft tokens on device as
        ``[num_valid, text_hidden]`` (bf16).

        ``pixel_values`` is ``[1, num_patches, in_dim]`` and ``image_position_ids`` ``[1, num_patches, 2]`` (host
        torch, from the HF Gemma4Processor). The SigLIP tower returns a padded ``[1, batch, output_length,
        vision_hidden]`` pooled tensor plus a boolean ``[batch, output_length]`` mask (True = valid soft token) that
        the projector (RMSNorm + Linear, ttnn) lifts to text-embedding space; the unified (encoder-free) embedder
        projects straight to text space. A ttnn ``gather`` then strips the padded slots, so the result is the
        compact set of soft tokens in order, ready to scatter into the placeholder slots of the text stream.
        """
        if not self.is_multimodal:
            raise RuntimeError(
                "encode_vision requires a multimodal Gemma-4 generator (vision tower / unified embedder)."
            )
        if getattr(self.vision_tower, "projects_to_text", False):
            projected_tt, mask = self.vision_tower(pixel_values, image_position_ids)
        else:
            num_patches = image_position_ids.shape[1]
            seq_len = _vision_encoder_seq_len(num_patches)
            pooled_tt, mask = self.vision_tower(pixel_values, image_position_ids, seq_len)
            projected_tt = self.embed_vision(pooled_tt)  # [1, batch, output_length, text_hidden]
            if hasattr(pooled_tt, "deallocate"):
                pooled_tt.deallocate(True)
        return self._gather_valid_soft_tokens(projected_tt, mask)

    def _gather_valid_soft_tokens(self, projected_tt, mask: torch.Tensor):
        """On-device gather of the VALID rows of ``projected_tt`` (``[1, 1, N_padded, text_hidden]``) per the host
        mask ``[1, N_padded]`` -> ttnn ``[num_valid, text_hidden]``. ``mask`` is host metadata (a small bool tensor):
        only the integer valid indices touch the host, and the 4D dim=2 gather with a uint16 TILE index is the
        tested ttnn.gather configuration."""
        valid_indices = torch.where(mask[0])[0].to(torch.uint16)  # [num_valid]
        num_valid = int(valid_indices.numel())
        text_hidden = projected_tt.shape[-1]
        gather_index = valid_indices.view(1, 1, num_valid, 1).expand(1, 1, num_valid, text_hidden).contiguous()
        gather_index_tt = ttnn.from_torch(
            gather_index,
            device=self.mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate_mapper(),
        )
        embeds_tt = ttnn.gather(projected_tt, 2, index=gather_index_tt)  # [1,1,num_valid,text_hidden]
        if hasattr(projected_tt, "deallocate"):
            projected_tt.deallocate(True)
        return ttnn.reshape(embeds_tt, (num_valid, text_hidden))

    def warmup_vision(self, pixel_values: torch.Tensor = None, image_position_ids: torch.Tensor = None):
        """Compile the vision tower + projector + on-device gather.

        With no arguments a tiny dummy image is synthesized from the tower's config, so the warmup works without
        a real image at server startup; pass real processor outputs to warm up on a representative shape.
        """
        if not self.is_multimodal:
            raise RuntimeError(
                "warmup_vision requires a multimodal Gemma-4 generator (vision tower / unified embedder)."
            )
        logger.info("Gemma4: warming up vision tower + projector...")
        if pixel_values is None or image_position_ids is None:
            if getattr(self.vision_tower, "projects_to_text", False):
                in_dim, num_patches = int(self.vision_tower.patch_dim), 280
            else:
                vcfg = self.vision_tower.args.hf_config.vision_config
                in_dim = int(getattr(vcfg, "patch_size", 16)) ** 2 * 3  # raw patch, not hidden_size
                num_patches = (
                    max(1, int(getattr(vcfg, "default_output_length", 280)))
                    * int(getattr(vcfg, "pooling_kernel_size", 3)) ** 2
                )
            pixel_values = torch.zeros(1, num_patches, in_dim, dtype=torch.bfloat16)
            image_position_ids = torch.zeros(1, num_patches, 2, dtype=torch.long)
        image_embeds_tt = self.encode_vision(pixel_values, image_position_ids)
        if hasattr(image_embeds_tt, "deallocate"):
            image_embeds_tt.deallocate(True)
        logger.info("Gemma4: vision warmup complete")


class Gemma4Generator(ChunkedPrefillPageTableGuardMixin, Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        # Gemma4ModelArgs exposes no get_attn_sdpa_program_config, so Generator
        # cannot derive the resume offset alignment and must be told it. Same pin
        # align_num_cached_tokens_to_sdpa applies locally.
        "resumed_prefill_token_alignment": SDPA_CHUNK_ALIGN,
    }

    def __init__(
        self,
        *args,
        vision_tower=None,
        embed_vision=None,
        image_token_id=None,
        pad_token_id=None,
        video_token_id=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Gemma4 decode already returns sampled tokens when on-device sampling is enabled.
        self.enable_split_sampling = False
        # Multimodal components (None for text-only builds). ``is_multimodal`` gates the
        # vision path (encode_vision / prefill_forward_multimodal / warmup_vision).
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.image_token_id = image_token_id
        # Video soft tokens carry their own placeholder id; both ids form bidirectional vision blocks.
        self.video_token_id = video_token_id
        self.pad_token_id = pad_token_id
        # The SigLIP path needs tower + projector; the unified (encoder-free)
        # embedder projects straight to text space and carries no projector.
        self.is_multimodal = vision_tower is not None and (
            embed_vision is not None or getattr(vision_tower, "projects_to_text", False)
        )
        # The batch key of the last decode step (batch-keyed decode traces / async-ahead feedback, mixin); None
        # until the first step. Was unreachable after ``_mock_tokens``'s return.
        self._prev_decode_batch = None

    def _prefill_forward_text_impl(self, *args, sampling_params=None, **kwargs):
        # Single-user eager and deferred traced lm_head normally gather for the
        # host consumer. The device sampler needs the same TP shards as decode.
        # Scope this to each call: nested warmup can request host logits, and
        # later host-sampling calls must retain their full-vocabulary output.
        previous = [getattr(model, "_prefill_keep_logits_sharded", False) for model in self.model]
        for model in self.model:
            model._prefill_keep_logits_sharded = (
                sampling_params is not None
                and getattr(model, "_supports_on_device_sampling", False)
                and model.sampling is not None
            )
        try:
            return super()._prefill_forward_text_impl(*args, sampling_params=sampling_params, **kwargs)
        finally:
            for model, value in zip(self.model, previous):
                model._prefill_keep_logits_sharded = value

    def _mock_tokens(self, batch_size, seq_len, kv_cache, model_id):
        """Warmup tokens with *unique* per-user page-table rows.

        Stock ``Generator._mock_tokens`` fills the page table with zeros, so every
        user maps to physical block 0. That is fine for B=1 but wedges batched
        prefill (B≥2) when concurrent ``paged_fill_cache`` writers collide —
        indefinite stall after the first all_gather on P150x8 (batch-32 demo).
        """
        ret = super()._mock_tokens(batch_size, seq_len, kv_cache, model_id)
        page_table = ret.get("page_table")
        if page_table is not None and batch_size > 1:
            num_blocks = int(page_table.shape[1])
            needed = batch_size * num_blocks
            # Paged K: [num_blocks_total, n_heads, block_size, head_dim] (see get_block_size).
            pool = needed
            if kv_cache is not None and kv_cache[model_id] is not None:
                pool = int(kv_cache[model_id][0][0].shape[0])
            flat = torch.arange(needed, dtype=torch.int32) % max(pool, 1)
            ret["page_table"] = flat.reshape(batch_size, num_blocks)
        return ret

    def _maybe_disable_pli_prefill_trace(self, enable_trace: bool, batch_size: int = 1) -> bool:
        return maybe_disable_pli_prefill_trace(enable_trace, self.model[0], batch_size=batch_size)

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device,
        greedy_only: bool = False,
        decode_page_table=None,
    ):
        # The prefill sweep uses narrower tables, which are unsuitable
        # for Gemma's batch-keyed decode traces and hybrid per-layer tables.
        prepared_decode = None
        if enable_trace and not self.already_warmed_up_prefill and decode_page_table is not None:
            prepared_decode = self._prepare_decode_trace_for_warmup(
                kv_cache=kv_cache,
                page_table=decode_page_table,
                on_device_sampling=can_sample_on_device,
            )
        warmup_gemma4_model_prefill(
            self,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
        )
        if prepared_decode is not None:
            # Complete trace setup before serving real requests. The eager
            # preparation above wrote dummy K/V; capture only records commands.
            previous_mode = self.mode
            self.mode = Mode.DECODE
            for model in self.model:
                model.switch_mode(Mode.DECODE)
            try:
                trace_ids, outputs, *inputs = self._record_decode_trace_text(prepared_decode)
            finally:
                self.mode = previous_mode
            key = (can_sample_on_device, int(decode_page_table.shape[0]) // self.data_parallel)
            self.trace_ids_decode[key] = trace_ids
            self.trace_inputs_decode[key] = inputs
            self.trace_output_decode[key] = outputs

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos: list[int] = None,
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        """Demo entry: arms the decode-trace guard (``_g4_arm_alloc_check``: every prefill in the default mode, the
        allocation events of the call in the ``alloc`` mode) around :meth:`_prefill_forward_text_demo`."""
        self._g4_arm_alloc_check("prefill")  # a prefill may leave persistent allocations behind
        ids_before = self._g4_prefill_trace_ids(self)
        try:
            return self._prefill_forward_text_demo(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                empty_slots=empty_slots,
                enable_trace=enable_trace,
                model_id_warmup=model_id_warmup,
                sampling_params=sampling_params,
                start_pos=start_pos,
                return_hidden_states=return_hidden_states,
                warmup_prefill=warmup_prefill,
                **kwargs,
            )
        finally:
            lens = prompt_lens if prompt_lens is not None else [int(tokens.shape[1])]
            lens = lens.tolist() if hasattr(lens, "tolist") else list(lens)
            max_chunk = int(getattr(self.model_args[0], "max_prefill_chunk_size", 1 << 30) or 1 << 30)
            chunked = any(int(n) > max_chunk for n in lens) or (
                start_pos is not None and any(int(p) > 0 for p in list(start_pos))
            )
            multimodal = any(
                kwargs.get(k) is not None for k in ("pixel_values", "mm_embeddings", "prompt_embeds", "stage_inputs")
            )
            self._g4_arm_after_prefill(prefill_trace_ids_before=ids_before, chunked=chunked, multimodal=multimodal)

    def _prefill_forward_text_demo(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos: list[int] = None,
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        if model_id_warmup is not None:
            warmup_prefill = False

        batch_size, batch_seq_len = tokens.shape
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace, batch_size=batch_size)

        prompt_lens_list = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)
        if not isinstance(prompt_lens_list, list):
            prompt_lens_list = prompt_lens_list.tolist()
        num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens_list)
        if start_pos is not None:
            num_cached_per_user = align_num_cached_tokens_to_sdpa(num_cached_per_user)
            start_pos = num_cached_per_user
        prefill_seq_lens = [
            get_padded_prefill_len(seq_len - num_cached)
            for seq_len, num_cached in zip(prompt_lens_list, num_cached_per_user)
        ]
        is_harmony = tokens.shape[1] > 0 and int(tokens[0, 0]) == 200006
        # GEMMA4_PACKED_PREFILL=1: cold prompts of one padded bucket run as one packed forward (per-segment SDPA,
        # per-slot KV, one lm_head); everything else (cached continuations, long prompts, singles) keeps the
        # stock path below.
        packed_out = self._g4_try_packed_prefill(
            dict(
                tokens=tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens_list,
                empty_slots=empty_slots,
                enable_trace=enable_trace,
                model_id_warmup=model_id_warmup,
                sampling_params=sampling_params,
                start_pos=start_pos,
                return_hidden_states=return_hidden_states,
                warmup_prefill=warmup_prefill,
                **kwargs,
            ),
            run_rest=lambda **sub: super(Gemma4Generator, self).prefill_forward_text(**sub),
        )
        if packed_out is not None:
            return packed_out
        # Batched prefill is eligible on identical *padded* buckets. Hetero
        # *actual* lengths are OK once per-slot ``valid_seq_lens`` caps KV fill
        # (see attention/prefill.py batched path) — do not require actual_lens_equal.
        can_batch_prefill = (
            page_table is not None
            and batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
            and all(n == 0 for n in num_cached_per_user)
            and not (getattr(self.model[0], "users_row_sharded", False) and sampling_params is not None and is_harmony)
        )
        if sampling_params is not None and can_batch_prefill:
            sampling_module, sampling_dp, _, _ = self._get_sampling_contract(0)
            if sampling_module is not None and sampling_dp > 1:
                can_batch_prefill = False

        if can_batch_prefill:
            # Span highest physical slot (#52808), then apply Gemma4 user/token caps.
            padded_batch = batched_prefill_padded_batch(batch_size, empty_slots, self.model_args[0].max_batch_size)
            max_users_per_chunk = resolve_batched_prefill_chunk_users(padded_batch, prefill_seq_lens[0])
            # True-batched B>user_cap hangs on P150x8 after the first all_gather.
            # Micro-batch at ≤user_cap with remapped local slots; per-slot
            # valid_seq_lens keep pad rows out of KV (decode stays coherent).
            if batch_size > max_users_per_chunk and padded_batch <= self.model_args[0].max_batch_size:
                logger.info(
                    "Chunking Gemma4 batched prefill: batch_size={} padded_batch={} "
                    "prefill_seq_len={} chunk_size={} (token_cap={} user_cap={})",
                    batch_size,
                    padded_batch,
                    prefill_seq_lens[0],
                    max_users_per_chunk,
                    GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN,
                    max_batched_prefill_users(),
                )

                merged_output = None
                merged_tokens = None
                merged_log_probs = None
                for chunk_start in range(0, batch_size, max_users_per_chunk):
                    chunk_end = min(chunk_start + max_users_per_chunk, batch_size)
                    chunk_size = chunk_end - chunk_start
                    # The chunk's requests keep the slots the caller assigned:
                    # per-slot device state (seed RNG, row-sharded user id) is
                    # addressed by slot, so renumbering the chunk would write it
                    # to slots other requests own.
                    chunk_slots = (
                        list(empty_slots[chunk_start:chunk_end])
                        if empty_slots is not None
                        else list(range(chunk_start, chunk_end))
                    )
                    chunk_enable_trace = apply_gemma4_prefill_trace_policy(
                        enable_trace,
                        prefill_seq_lens[0],
                        # Keeping the real slots means the chunk runs as many rows as
                        # its highest slot needs, so the policy has to weigh that span
                        # and not the request count, or it captures a trace whose real
                        # token count is past the documented OOM-risk limit.
                        batched_prefill_padded_batch(chunk_size, chunk_slots, self.model_args[0].max_batch_size),
                        self.model[0],
                    )
                    chunk_result = super().prefill_forward_text(
                        tokens=tokens[chunk_start:chunk_end],
                        page_table=page_table[chunk_start:chunk_end] if page_table is not None else None,
                        kv_cache=kv_cache,
                        prompt_lens=prompt_lens_list[chunk_start:chunk_end],
                        empty_slots=chunk_slots,
                        enable_trace=chunk_enable_trace,
                        model_id_warmup=model_id_warmup,
                        # Each chunk carries its own requests' params. Passing the whole
                        # prefill-ordered set gave every chunk the first ``chunk_size``
                        # temperatures, penalties and seeds instead of its own.
                        sampling_params=slice_sampling_params(sampling_params, chunk_start, chunk_end),
                        start_pos=num_cached_per_user[chunk_start:chunk_end] if start_pos is not None else None,
                        return_hidden_states=return_hidden_states,
                        warmup_prefill=warmup_prefill and chunk_start == 0,
                        **kwargs,
                    )

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

        enable_trace = resolve_gemma4_prefill_trace_enable(
            enable_trace,
            self.model[0],
            self.model_args[0],
            batch_size=batch_size,
            prefill_seq_lens=prefill_seq_lens,
            can_batch_prefill=can_batch_prefill,
            empty_slots=empty_slots,
        )

        return super().prefill_forward_text(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        model_path,
        max_batch_size=1,
        max_seq_len=4096,
        num_layers=None,
        paged_attention_config=None,
        bounded_sliding_kv_cache=False,
        multimodal: bool = True,
        vision_dtype=ttnn.bfloat8_b,
        load_tokenizer: bool = True,
    ):
        """``load_tokenizer=False``: no tokenizer at all (``generator.tokenizer`` is None) for callers that feed
        token ids themselves (the prefill unit tests on a weights-only cache: no tokenizer backend needed)."""
        tokenizer = _load_text_tokenizer(model_path) if load_tokenizer else None
        if tokenizer is not None and not hasattr(tokenizer, "stop_tokens"):
            tokenizer.stop_tokens = [tokenizer.eos_token_id]

        model_args, model, tt_kv_cache, state_dict = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            model_path=model_path,
            create_kv_cache=True,
            paged_attention_config=paged_attention_config,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        # Lane-sharded serving: the generator sees the global slot space (lanes x per-column batch) while the
        # model/KV stay sized per column; lane_slots gives the owner lane as slot // lane_slots.
        _lanes = getattr(getattr(model, "mesh_config", None), "lane_sharded", False) and model.mesh_config.lanes
        generator_batch = max_batch_size * _lanes if _lanes else max_batch_size
        if _lanes:
            model.lane_slots = max_batch_size

        _patch_model_args(
            model_args,
            mesh_device=mesh_device,
            max_batch_size=generator_batch,
            max_seq_len=max_seq_len,
            model_path=model_path,
            tokenizer=tokenizer,
            has_per_layer_inputs=bool(getattr(model, "hidden_size_per_layer_input", 0)),
            bounded_sliding=bounded_sliding_kv_cache,
        )

        vision_tower = None
        embed_vision = None
        image_token_id = None
        video_token_id = None
        pad_token_id = None
        if multimodal:
            hf_config = Gemma4ModelArgs.load_hf_config(model_path)
            image_token_id = getattr(hf_config, "image_token_id", None)
            if image_token_id is None:
                raise ValueError("HF config has no image_token_id — not a multimodal Gemma-4 checkpoint?")
            video_token_id = getattr(hf_config, "video_token_id", None)
            pad_token_id = getattr(getattr(hf_config, "text_config", hf_config), "pad_token_id", tokenizer.pad_token_id)

            # VisionModelArgs reads HF_MODEL env for the checkpoint path (like ModelArgs).
            os.environ.setdefault("HF_MODEL", model_path)
            # Share the text model's mesh/CCL setup so the vision tower's TP
            # all-reduces reuse the same semaphores and link/topology tuning.
            # The text model's warm weight cache hands over a dataless placeholder
            # dict; the vision encoders are not in that cache, so take their tensors
            # from the checkpoint (measured 2026-10-07: NaN / inf / 1e34 otherwise).
            state_dict = vision_state_dict_for(state_dict, model_path)
            if has_unified_vision_weights(state_dict):
                # 12B "unified": no SigLIP tower; raw patches project straight to text space.
                vision_tower = Gemma4UnifiedVisionEmbedder(
                    mesh_device,
                    state_dict,
                    text_hidden_size=model.hidden_size,
                    weight_cache_path=model_args.weight_cache_path(ttnn.bfloat16),
                )
                embed_vision = None
            else:
                vision_args = VisionModelArgs(
                    mesh_device,
                    dummy_weights=True,
                    max_batch_size=1,
                    max_seq_len=8192,
                    mesh_config=model.mesh_config,
                    ccl_manager=model.ccl_manager,
                )
                vision_state_dict = _build_vision_state_dict(state_dict, vision_args)
                vision_tower = VisionTower(
                    args=vision_args,
                    dtype=vision_dtype,
                    state_dict=vision_state_dict,
                    tt_ccl=TT_CCL(mesh_device),
                    weight_cache_path=vision_args.weight_cache_path(vision_dtype),
                )
                embed_vision = Gemma4MultimodalEmbedder.from_state_dict(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    vision_args=vision_args,
                    text_hidden_size=model.hidden_size,
                    dtype=ttnn.bfloat16,
                    weight_cache_path=vision_args.weight_cache_path(ttnn.bfloat16),
                )

        generator = cls(
            [model],
            [model_args],
            mesh_device,
            processor=None,
            tokenizer=tokenizer,
            vision_tower=vision_tower,
            embed_vision=embed_vision,
            image_token_id=image_token_id,
            pad_token_id=pad_token_id,
            video_token_id=video_token_id,
        )
        return generator, [tt_kv_cache], tokenizer

    # ── Multimodal (vision) path ───────────────────────────────────────────
    # Only used when the generator was built with ``multimodal=True``; ``encode_vision`` / ``warmup_vision`` are the
    # mixin's. The text-only path (prefill_forward_text / decode_forward) is unchanged.

    def encode_vision_batch(self, pixel_values_batch: torch.Tensor, image_position_ids_batch: torch.Tensor):
        """Run the vision tower + embedder for a batch of images, one image at a time,
        and return the per-user valid projected soft tokens as a list of HOST tensors.

        The tower is tensor-parallel across the whole mesh, so a single image already
        uses every device; batching is a plain loop over users.

        Args:
            pixel_values_batch: ``[B, num_patches, in_dim]`` torch.
            image_position_ids_batch: ``[B, num_patches, 2]`` torch.

        Returns:
            list of ``B`` host tensors, each ``[num_valid_u, text_hidden]`` (bf16), in user
            order.
        """
        if not self.is_multimodal:
            raise RuntimeError("encode_vision_batch requires from_pretrained(..., multimodal=True).")
        is_mesh = hasattr(self.mesh_device, "shape")
        batch_size = int(pixel_values_batch.shape[0])

        per_user_embeds: list[torch.Tensor] = []
        _t0 = time.perf_counter()
        for u in range(batch_size):
            # encode_vision returns the compact, valid-only soft tokens, replicated
            # across the mesh; take one device's copy back to host.
            embeds_tt = self.encode_vision(pixel_values_batch[u : u + 1], image_position_ids_batch[u : u + 1])
            ttnn.synchronize_device(self.mesh_device)
            if is_mesh:
                user_embeds = ttnn.to_torch(ttnn.get_device_tensors(embeds_tt)[0])
            else:
                user_embeds = ttnn.to_torch(embeds_tt)
            if hasattr(embeds_tt, "deallocate"):
                embeds_tt.deallocate(True)
            per_user_embeds.append(user_embeds.contiguous().to(torch.bfloat16))
        logger.info(
            f"Vision batch encode: {batch_size} image(s), one per iteration (TP) "
            f"in {(time.perf_counter() - _t0) * 1000.0:.1f} ms"
        )
        return per_user_embeds

    def prefill_forward_multimodal(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor = None,
        image_position_ids: torch.Tensor = None,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        enable_trace: bool = False,
        image_embeds_tt=None,
    ) -> torch.Tensor:
        """Prefill an image+text prompt and return next-token logits on host.

        Feeds precomputed merged embeddings straight into
        ``Gemma4Model.ttnn_prefill_forward`` (which accepts hidden states
        directly), so the only deviation from the text path is the embedding
        step — the decoder body, RoPE, paged attention and lm_head are
        unchanged. Decode reuses the inherited ``decode_forward``.

        Args:
            tokens: ``[1, prompt_len]`` long ids (image_token_id placeholders included).
            pixel_values / image_position_ids: from Gemma4Processor. Optional if
                ``image_embeds_tt`` is supplied (precomputed vision embeds).
            page_table: ``[1, num_blocks]`` int32 identity page table (or None).
            kv_cache: ``[per_model_cache]`` list as returned by ``from_pretrained``
                (unwrapped to the per-layer ``[k, v]`` list internally).
            prompt_lens: list[int] of real (unpadded) prompt lengths.
            image_embeds_tt: optional precomputed vision soft tokens on device
                (``[num_valid, text_hidden]``, replicated) — skips the vision
                encoder. Used by the batched demo, which runs the tower 4-at-a-time
                (DP) once for all users and reuses the per-user embeds here.

        Returns:
            torch logits ``[1, vocab]`` for the last prompt token.
        """
        del enable_trace  # Multimodal prefill is not traced (vision encoder + on-device merge).
        if not self.is_multimodal:
            raise RuntimeError("prefill_forward_multimodal requires from_pretrained(..., multimodal=True).")
        model = self.model[0]
        if getattr(model, "hidden_size_per_layer_input", 0):
            raise NotImplementedError(
                "Multimodal prefill + per-layer inputs (PLI) is not supported. "
                "Use a non-PLI multimodal checkpoint (gemma-4 12B / 26B-A4B / 31B IT)."
            )
        prompt_len = int(tokens.shape[1])
        last_token_idx = (prompt_lens[0] if prompt_lens is not None else prompt_len) - 1
        prefill_seq_len = get_padded_prefill_len(prompt_len)

        # 1. Vision -> valid projected soft tokens on device: [num_valid, text_hidden].
        _vision_t0 = time.perf_counter()
        if image_embeds_tt is None:
            if pixel_values is None or image_position_ids is None:
                raise ValueError(
                    "prefill_forward_multimodal needs either pixel_values+image_position_ids "
                    "or a precomputed image_embeds_tt."
                )
            image_embeds_tt = self.encode_vision(pixel_values, image_position_ids)
            vision_prefill_ms = (time.perf_counter() - _vision_t0) * 1000.0
            logger.info(f"Vision encoder prefill: {vision_prefill_ms:.1f} ms")
        else:
            # Precomputed embeds (batched demo): vision already timed by encode_vision_batch.
            vision_prefill_ms = time.perf_counter() - _vision_t0
        num_valid = int(image_embeds_tt.shape[0])
        text_hidden = int(image_embeds_tt.shape[1])

        # 2. Build llm_ids on host (cheap int ops): replace image_token_id with pad,
        #    then pad the prompt out to the prefill bucket with pad_token_id. Embedding
        #    this on device yields the text embeddings with pad-token rows already in
        #    place, so only the image_token slots need filling.
        input_ids = tokens[0].to(torch.long)
        image_mask = input_ids == self.image_token_id
        n_image = int(image_mask.sum().item())
        if n_image != num_valid:
            raise ValueError(
                f"Image placeholder count ({n_image}) != vision soft tokens ({num_valid}). "
                "The processor's image_seq_length and the image processor's max_soft_tokens must agree."
            )
        image_positions = torch.where(image_mask)[0].to(torch.int32)  # [num_valid], within prompt_len
        llm_ids = input_ids.clone()
        llm_ids[image_mask] = self.pad_token_id
        pad = prefill_seq_len - prompt_len
        if pad > 0:
            llm_ids = torch.cat([llm_ids, torch.full((pad,), self.pad_token_id, dtype=torch.long)], dim=0)
        llm_ids_tt = ttnn.from_torch(
            llm_ids.view(1, prefill_seq_len),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
        )

        # 3. Text embeddings on device (lookup + sqrt(hidden) scale + all-gather).
        text_embeds_tt = model.embed_tokens(llm_ids_tt)  # [1, prefill_seq, text_hidden] (or 4D under TP)
        text_embeds_tt = ttnn.reshape(text_embeds_tt, (prefill_seq_len, text_hidden))
        # scatter wants interleaved DRAM; force it regardless of the embed output layout.
        text_embeds_tt = ttnn.to_layout(text_embeds_tt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # 4. Scatter the projected vision tokens into the image_token slots (on device).
        scatter_index = image_positions.view(num_valid, 1).expand(num_valid, text_hidden).contiguous()
        scatter_index_tt = ttnn.from_torch(
            scatter_index,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate_mapper(),
        )
        merged_tt = ttnn.scatter(text_embeds_tt, 0, scatter_index_tt, image_embeds_tt)  # [prefill_seq, text_hidden]
        if hasattr(text_embeds_tt, "deallocate"):
            text_embeds_tt.deallocate(True)
        if hasattr(image_embeds_tt, "deallocate"):
            image_embeds_tt.deallocate(True)
        merged_tt = ttnn.reshape(merged_tt, (1, 1, prefill_seq_len, text_hidden))

        # 5. Page table -> device.
        tt_page_table = None
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table.to(torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._replicate_mapper(),
            )

        # 6. Prefill: feed merged embeddings directly (bypass embed_tokens).
        # ``from_pretrained`` returns ``[tt_kv_cache]`` (a list indexed by model_id,
        # matching the base Generator contract), so unwrap once to get the per-layer
        # cache the model expects (``caches[i]`` -> ``[k_cache, v_cache]``).
        model_id = 0
        model_kv_cache = kv_cache[model_id] if kv_cache is not None else None
        logger.info(
            f"Multimodal prefill: prompt={prompt_len} (image tokens={n_image}), "
            f"padded={prefill_seq_len}, last_token={last_token_idx}"
        )
        get_last_token = (last_token_idx // 32) * 32
        _llm_t0 = time.perf_counter()
        # Image-span bidirectional attention for this whole-prompt prefill (GEMMA4_MM_BIDIR_MASK=1).
        from models.demos.gemma4.tt.attention.mm_prefill_vision import vision_spans_of_prompt

        model._g4_mm_prefill_blocks = (
            vision_spans_of_prompt(input_ids, self.image_token_id, getattr(self, "video_token_id", None)) or None
        )
        try:
            logits_tt = model.ttnn_prefill_forward(
                x=merged_tt,
                page_table=tt_page_table,
                kv_cache=model_kv_cache,
                get_last_token=get_last_token,
                batch_size=1,
                input_ids_torch=None,
                embeds_torch=None,
            )
        finally:
            model._g4_mm_prefill_blocks = None
        ttnn.synchronize_device(self.mesh_device)
        llm_prefill_ms = (time.perf_counter() - _llm_t0) * 1000.0
        logger.info(
            f"LLM prefill: {llm_prefill_ms:.1f} ms "
            f"(vision+LLM prefill total ≈ {vision_prefill_ms + llm_prefill_ms:.1f} ms)"
        )

        # 7. Read last-token logits to host.
        if model.mesh_config is not None and model.mesh_config.tp > 1:
            torch_logits = ttnn.to_torch(ttnn.get_device_tensors(logits_tt)[0])
        else:
            torch_logits = ttnn.to_torch(logits_tt)
        return torch_logits[..., last_token_idx % 32, : model.vocab_size]
