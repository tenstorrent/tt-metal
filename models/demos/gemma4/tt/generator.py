# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.sampling import SamplingParams, slice_sampling_params
from models.demos.gemma4.tt.async_decode import merge_async_ahead_decode_tokens
from models.demos.gemma4.tt.common import create_tt_model
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
from models.tt_transformers.tt.model_config import determine_device_name

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
    def _match_page_table_row(page_table_1row, page_tables_per_layer) -> int | None:
        """Return the batch row whose page-table prefix matches ``page_table_1row``."""
        if page_table_1row is None or not page_tables_per_layer:
            return None
        if not isinstance(page_table_1row, torch.Tensor):
            return None
        pt = page_table_1row if page_table_1row.dim() > 1 else page_table_1row.unsqueeze(0)
        if int(pt.shape[0]) != 1:
            return None
        candidates = [
            p for p in page_tables_per_layer if isinstance(p, torch.Tensor) and p.dim() > 1 and int(p.shape[0]) > 1
        ]
        if not candidates:
            return 0
        pt32 = pt[0].to(dtype=torch.int32)
        for ref in candidates:
            cols = min(int(pt32.shape[0]), int(ref.shape[1]))
            if cols <= 0:
                continue
            ref32 = ref.to(dtype=torch.int32)
            for r in range(int(ref32.shape[0])):
                if torch.equal(pt32[:cols], ref32[r, :cols]):
                    return r
        return None

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
        """Slice multi-row hybrid page-table stash to the active sequential user.

        Under bounded sliding the bridge keeps per-layer tables (full vs sliding)
        and also stuffs the remapped *sliding* table into legacy ``page_table``.
        Sequential tt_transformers then passes ``user_id=0`` with a 1-row slice —
        without this, every user writes/reads batch row 0 of the per-layer stash
        (and chunked full-attn fill uses ``full_pt[0]``).
        """
        if page_table is None or not isinstance(page_table, torch.Tensor):
            return
        pt = page_table if page_table.dim() > 1 else page_table.unsqueeze(0)
        if int(pt.shape[0]) != 1:
            return
        for m in self.model:
            active = getattr(m, "_active_page_tables_per_layer", None)
            if not active:
                continue
            batch_host = getattr(m, "_sequential_batch_page_tables", None)
            if batch_host is None:
                if not any(isinstance(p, torch.Tensor) and p.dim() > 1 and int(p.shape[0]) > 1 for p in active):
                    continue
                m._sequential_batch_page_tables = active
                batch_host = active
            row = self._match_page_table_row(pt, batch_host)
            if row is None:
                continue
            sliced = []
            for p in batch_host:
                if isinstance(p, torch.Tensor) and p.dim() > 1 and int(p.shape[0]) > 1:
                    sliced.append(p[row : row + 1])
                else:
                    sliced.append(p)
            m._active_page_tables_per_layer = sliced
            # Prefill installs full-batch host tables and H2D-copies them once
            # (batch_key=B). Sequential users then slice host `_active` to 1-row
            # but `ttnn_prefill_forward` only calls `_page_tables_to_ttnn`, which
            # returns existing B=1 device buffers *without* refreshing content.
            # Without this H2D, users after the first keep reading/writing user
            # 0's block IDs (full-attn cross-chunk SDPA + sliding ring fill).
            if hasattr(m, "update_persistent_per_layer_page_tables"):
                m.update_persistent_per_layer_page_tables(sliced)

    def _clear_sequential_batch_page_tables(self) -> None:
        """Put the full-batch per-layer tables back after sequential prefill."""
        for m in self.model:
            batch_host = getattr(m, "_sequential_batch_page_tables", None)
            if batch_host is None:
                continue
            m._active_page_tables_per_layer = batch_host
            if hasattr(m, "update_persistent_per_layer_page_tables"):
                m.update_persistent_per_layer_page_tables(batch_host)
            del m._sequential_batch_page_tables

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
        for i, layer in enumerate(getattr(self.model[0], "layers", [])):
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            if cfg is None or i >= len(kv_cache) or kv_cache[i] is None:
                continue
            cache = kv_cache[i][0]
            cache_hd = int(cache.shape[-1])
            if cache_hd != int(cfg.head_dim) and cache_hd > 0:
                # HMA-shared buffer: byte-invariant reinterpret for this layer's head_dim.
                return int(cache.shape[2]) * cache_hd // int(cfg.head_dim)
        return block_size

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
        model = self.model[model_id]
        for layer in getattr(model, "layers", []):
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            if cfg is not None and getattr(cfg, "cache_position_modulo", None) is not None:
                modulo = int(cfg.cache_position_modulo)
                break
        if modulo is None:
            sw = getattr(getattr(model, "hf_config", None), "sliding_window", None)
            modulo = int(sw) if sw else None
        if not modulo or last_chunk_start <= 0:
            return last_chunk_start, last_token_idx_in_chunk
        remnant = int(last_token_idx_in_chunk) + 1
        if remnant >= modulo:
            return last_chunk_start, last_token_idx_in_chunk
        # paged_fill_cache has no start-position input: row r is written to
        # circular slot r % modulo. Keep the expanded chunk's absolute start
        # ring-aligned so local row indices map to the corresponding absolute
        # slots. Pulling back by one complete window gives the last chunk at
        # least one full window of real K/V without changing its ring origin.
        align = max(int(block_size), 128)
        target_start = max(0, int(last_chunk_start) - modulo)
        new_start = ((target_start + align - 1) // align) * align
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

    def _capture_trace_prefill(
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
        """Capture prefill trace; reset sliding tails between compile and capture.

        The compile forward (outside begin_trace) leaves per-layer sliding K/V
        tails on the attention modules. Without a reset, the capture forward
        takes the middle-chunk branch and loads new programs mid-capture
        (TT_FATAL). sp0 must compile+capture with ``sliding_tail_in is None``;
        sp1 must start capture from the *same* tail state compile started with.
        APC / vLLM chunked continuations often JIT-capture sp1 with no prior
        stash — compile takes the no-tail SDPA path then stashes a tail, and
        capture would otherwise hit ``q_pad`` concat (program not in cache).
        """
        import ttnn
        from models.tt_transformers.tt.common import copy_host_to_device

        if batch_size > 1:
            return super()._capture_trace_prefill(
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

        prefill_kwargs = {
            "page_table": page_table,
            "chunk_page_table": chunk_page_table,
            "chunk_start_idx": start_pos,
            "user_id": user_id,
        }
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        tt_rot_mats_prefill_global = host_inputs[1]
        tt_rot_mats_prefill_local = host_inputs[2]
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4], host_inputs[5])

        # Snapshot pre-compile sliding state so capture can restore it. Compile
        # always mutates `_sliding_prefill_tail` (and may first-alloc or copy
        # into `sliding_prefill_tail_persistent`).
        had_starting_tails = self._any_sliding_prefill_tails(model_id)
        had_persistent = self._any_sliding_prefill_persistent(model_id)

        # Match Python graph for both compile and capture (sp0: first-alloc,
        # no ttnn.copy). Soft release leaves sliding_prefill_tail_persistent set
        # after compile, so capture would take the copy path and TT_FATAL
        # (program not in cache). Hard-clear persistent on both sides.
        if int(start_pos) == 0:
            self._release_all_sliding_prefill_tails(model_id, clear_persistent=True)

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            chunk_start_idx=transformed_inputs[3],
            kv_cache=kv_cache,
        )
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        logger.info("Done Compiling Model")

        # Restore the same starting tail state as compile before capture.
        if int(start_pos) == 0:
            self._release_all_sliding_prefill_tails(model_id, clear_persistent=True)
        elif not had_starting_tails:
            # sp1 JIT (APC / first middle chunk): compile ran no-tail SDPA then
            # stashed a new tail. Drop that stash so capture matches. Keep
            # persistent only when compile also started with it (end-of-forward
            # copy path); otherwise hard-clear so both passes first-alloc.
            self._release_all_sliding_prefill_tails(model_id, clear_persistent=not had_persistent)

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            chunk_start_idx=transformed_inputs[3],
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        logger.info("Done Capturing Prefill Trace")
        return trace_id, tt_out_trace, *device_inputs

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
        try:
            if force_chunk_pt:
                return self._easy_trace_prefill_with_chunk_page_table(*args, **kwargs)
            return super()._easy_trace_prefill(*args, **kwargs)
        finally:
            for m in self.model:
                m._prefill_trace_mode = False

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
            if attn is not None and any(
                v is not None for v in (getattr(attn, "_sliding_tails_by_key", None) or {}).values()
            ):
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

    def _release_all_sliding_prefill_tails(self, model_id=-1, *, clear_persistent: bool = False):
        for layer in getattr(self.model[model_id], "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_release_sliding_prefill_tail"):
                attn._release_sliding_prefill_tail(clear_persistent=clear_persistent)

    def prefill_forward_single_user_text(
        self, tokens, page_table=None, *, kv_cache=None, num_cached_tokens=0, **kwargs
    ):
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
            if int(pt2d[0, 0]) > 0:
                # First block id 0 = vLLM null block / warmup mock tables —
                # never a real request; keep key None so trace-unsafe pool
                # copies cannot run during warmup capture.
                req_key = int(pt2d[0, 0])
        for model in self.model:
            for layer in getattr(model, "layers", []):
                cfg = getattr(getattr(layer, "self_attn", None), "config", None)
                if cfg is not None:
                    cfg._g4_active_req_key = req_key
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

            # Inject an expanded last start when adjust moves it off the chunk grid.
            last_abs = num_cached_tokens + last_chunk_start
            chunk_starts = list(range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size))
            chunk_starts = [s for s in chunk_starts if s < last_abs]
            chunk_starts.append(last_abs)

            for chunk_start in chunk_starts:
                chunk_end = chunk_start + chunk_size
                chunk_start_relative = chunk_start - num_cached_tokens
                chunk_end_relative = min(chunk_end - num_cached_tokens, seq_len)
                is_last_chunk = chunk_start == last_abs

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
                chunk_inputs = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                    batch_size=batch_size,
                    user_id=CHUNK_USER_ID,
                    **kwargs,
                )
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
                    **kwargs,
                )
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
        reset_batch=False,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        enable_trace=False,
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
                    def _sample(logits, *, enable_trace=True, tt_out_tok=None, skip_precompile=False):
                        nonlocal wrote_feedback
                        if tt_out_tok is None:
                            tt_out_tok = fb
                            if not enable_trace:
                                wrote_feedback = True
                        return orig(
                            logits,
                            enable_trace=enable_trace,
                            tt_out_tok=tt_out_tok,
                            skip_precompile=skip_precompile,
                        )

                    return _sample

                sampling_module.sample = _make_sample(orig_sample, feedback)
                restores.append((sampling_module, orig_sample))

            out = super().sample_decode_on_device(
                tt_logits,
                sampling_params,
                start_pos=start_pos,
                reset_batch=reset_batch,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                enable_trace=enable_trace,
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
        reset_batch=False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap=None,
        defer_device_sampling: bool = False,
        **kwargs,
    ):
        """Gemma4 decode with safe async-ahead merge (no ``tt_transformers`` edits).

        Same control flow as ``Generator.decode_forward``, but merges host/device
        tokens via :func:`merge_async_ahead_decode_tokens` so bucket changes and
        OOB ``slot_remap`` fall back instead of ``IndexError``.
        """
        del kwargs  # Generator accepts extras; Gemma4 path ignores them.
        # Sequential per-user prefill narrows the per-layer tables to one row and
        # not every prefill entry point unwinds it; decode always needs the full
        # batch back or paged_update_cache TT_FATALs (1 row vs B users).
        self._clear_sequential_batch_page_tables()
        mode_switched = False
        if self.mode != Mode.DECODE:
            self.mode = Mode.DECODE
            mode_switched = True

        for i in range(len(self.model)):
            self.model[i].switch_mode(Mode.DECODE)

        on_device_sampling = (sampling_params is not None) or defer_device_sampling

        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        # Match _decode_forward_trace_text: (sampling, per-DP chunk batch).
        decode_trace_key = (
            on_device_sampling,
            int(tokens[0].shape[0]) if tokens else 1,
        )

        # Only merge device token/pos feedback when async decode is actually
        # enabled. ``model_capabilities["supports_async_decode"]`` is the single
        # source of truth (tt-inference-server llm.yaml -> GEMMA4_SUPPORTS_ASYNC_DECODE
        # -> capability, resolved once in Gemma4ForCausalLM); previously this path
        # ran regardless, so disabling async in config turned off vLLM's async
        # scheduler while Gemma4 kept doing async-ahead device feedback. With async
        # off the host tokens are authoritative and there is nothing to merge.
        async_decode_enabled = bool(getattr(self, "model_capabilities", {}).get("supports_async_decode", False))
        if (
            async_decode_enabled
            and on_device_sampling
            and (reset_batch or mode_switched)
            and enable_trace
            and self.trace_inputs_decode[decode_trace_key]
        ):
            new_tokens = []
            new_start_pos = []
            for i, tok_chunk in enumerate(tokens):
                trace_in = self.trace_inputs_decode[decode_trace_key][i]
                host_pos = start_pos[i].reshape(-1).to(torch.int64)
                host_toks = tok_chunk.reshape(-1)
                host_b = int(host_toks.shape[0])
                # Read the full device buffer before truncating. Nearest-bucket
                # B changes and mesh-row sharding can make shard-0 narrower than
                # host_b; the gemma4 helper falls back before any gather.
                dev_toks_full = ttnn.to_torch(ttnn.get_device_tensors(trace_in[0])[0]).reshape(-1).to(tok_chunk.dtype)
                dev_pos_full = ttnn.to_torch(ttnn.get_device_tensors(trace_in[1])[0]).reshape(-1).to(torch.int64)
                slot_remap_local = None
                if slot_remap is not None:
                    remap = slot_remap[i * host_b : (i + 1) * host_b]
                    remap_t = (remap if isinstance(remap, torch.Tensor) else torch.tensor(remap)).long()
                    slot_remap_local = remap_t - i * host_b
                prefilled = getattr(self, "_slots_prefilled_since_decode", None)
                prefilled_local = None
                if prefilled:
                    bs = tok_chunk.shape[0]
                    prefilled_local = {slot - i * bs for slot in prefilled if i * bs <= slot < (i + 1) * bs}
                merged, merged_pos, src = merge_async_ahead_decode_tokens(
                    host_toks,
                    host_pos,
                    dev_toks_full,
                    dev_pos_full,
                    slot_remap_local=slot_remap_local,
                    prefilled_local=prefilled_local,
                )
                if src != "merged" or int(dev_toks_full.shape[0]) != int(host_toks.shape[0]):
                    logger.info(
                        "async_ahead_merge src={} host_b={} dev_len={}",
                        src,
                        int(host_toks.shape[0]),
                        int(dev_toks_full.shape[0]),
                    )
                new_tokens.append(merged.view(tok_chunk.shape).to(tok_chunk.dtype))
                new_start_pos.append(merged_pos.view(start_pos[i].shape).to(start_pos[i].dtype))
            tokens = new_tokens
            start_pos = new_start_pos
        self._slots_prefilled_since_decode = set()

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
                reset_batch=reset_batch or mode_switched,
            )
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
                reset_batch=reset_batch,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                enable_trace=enable_trace,
            )
        if read_from_device:
            to_host = self.read_decode_output(tt_decode_output)
            return self.process_decode_output_host(to_host, is_tokens=(sampling_params is not None))
        return tt_decode_output

    def _decode_forward_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        on_device_sampling=False,
        reset_batch=False,
        skip_precompile=False,
    ):
        """Gemma4 decode traces keyed by ``(on_device_sampling, batch)``.

        Shared ``Generator`` keys traces by sampling mode only. Gemma4 warms
        B=1 and B=max separately (vLLM nearest-bucket / P150x8), so the key
        must include per-DP chunk batch or async-ahead keep and replay hit the
        wrong Metal graph. Kept local to this mixin — no ``tt_transformers`` edits.
        """
        from models.tt_transformers.tt.common import copy_host_to_device
        from models.tt_transformers.tt.generator import DECODE_PAGE_TABLE_INPUT_IDX

        batch = int(tokens[0].shape[0]) if tokens else 1
        decode_trace_key = (on_device_sampling, batch)
        if not self.trace_ids_decode[decode_trace_key]:
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

        prev_on_device_sampling = getattr(self, "_prev_on_device_sampling", None)
        self._prev_on_device_sampling = on_device_sampling
        prev_decode_batch = getattr(self, "_prev_decode_batch", None)
        self._prev_decode_batch = batch
        sampling_mode_changed = prev_on_device_sampling is not None and prev_on_device_sampling != on_device_sampling
        batch_changed = prev_decode_batch is not None and prev_decode_batch != batch
        reset_inputs = reset_batch or not on_device_sampling or sampling_mode_changed or batch_changed
        page_table_changed = page_table is not None and (
            self.prev_page_table is None
            or any(not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table))
        )

        for i in range(self.data_parallel):
            refresh_trace_inputs = reset_inputs or getattr(
                self.model[i], "_tt_vllm_always_refresh_decode_trace_inputs", False
            )
            user_page_table = page_table[i] if page_table is not None else None

            if refresh_trace_inputs:
                host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)
                copy_host_to_device(
                    host_tensors=host_inputs_i,
                    device_tensors=self.trace_inputs_decode[decode_trace_key][i],
                )
            elif page_table_changed:
                host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)
                host_page_table = host_inputs_i[DECODE_PAGE_TABLE_INPUT_IDX]
                device_page_table = self.trace_inputs_decode[decode_trace_key][i][DECODE_PAGE_TABLE_INPUT_IDX]
                if host_page_table is not None:
                    ttnn.copy_host_to_device_tensor(host_page_table, device_page_table)

        if page_table_changed:
            self.prev_page_table = tuple(pt.clone() for pt in page_table)
        for i, trace_id in self.trace_ids_decode[decode_trace_key].items():
            ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
        return self.trace_output_decode[decode_trace_key]


class Gemma4Generator(ChunkedPrefillPageTableGuardMixin, Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gemma4 decode already returns sampled tokens when on-device sampling is enabled.
        self.enable_split_sampling = False
        # Used by batch-keyed decode traces / async-ahead feedback (mixin).
        self._prev_decode_batch = None

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
    ):
        warmup_gemma4_model_prefill(
            self,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
        )

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
    ):
        tokenizer = _load_text_tokenizer(model_path)
        if not hasattr(tokenizer, "stop_tokens"):
            tokenizer.stop_tokens = [tokenizer.eos_token_id]

        model_args, model, tt_kv_cache, _ = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            model_path=model_path,
            create_kv_cache=True,
            paged_attention_config=paged_attention_config,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        _patch_model_args(
            model_args,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            model_path=model_path,
            tokenizer=tokenizer,
            has_per_layer_inputs=bool(getattr(model, "hidden_size_per_layer_input", 0)),
            bounded_sliding=bounded_sliding_kv_cache,
        )
        generator = cls([model], [model_args], mesh_device, processor=None, tokenizer=tokenizer)
        return generator, [tt_kv_cache], tokenizer
