# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLM Executor Engines — thick engines that own prefill/decode implementation.

EagerLLMExecutor  — direct execution (no tracing)
TracedLLMExecutor — traced execution (capture/replay)

These are reusable engines. Model-specific executors (e.g., EagerLlamaExecutor)
are thin wrappers that pass the model to these engines.

Design principle: Thick engine, thin model executor.
- Engine owns the implementation (prefill/decode forward, KV cache, trace capture/replay)
- Model executor owns the model (passes transformer to engine)
- Model-specific details come from model.model_args or model attributes
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.common.models.module_input_validation import suspend_module_input_validation, validate_module_input_configs
from models.common.sampling.sampling_params import SamplingParams
from models.common.tests.demos.cleanup_utils import cleanup_ttnn_value
from models.tt_transformers.tt.common import (
    Mode,
    copy_host_to_device,
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)

if TYPE_CHECKING:
    # Type-only import (annotations are lazy under ``from __future__ import annotations``),
    # so the optional perf-benchmark profiler adds no runtime import cost / no infra dep.
    from models.perf.benchmarking_utils import BenchmarkProfiler

# =============================================================================
# Page Table Helpers
# =============================================================================


def make_contiguous_page_table(
    batch_size: int,
    max_seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Create a simple contiguous page table for demos/tests.

    Returns page_table [batch_size, num_blocks] where each user gets contiguous
    blocks: user 0 -> [0,1,2,...], user 1 -> [N,N+1,...], etc.

    This is the simplest allocation strategy. For advanced use cases:
    - Shared prefix: multiple users can point to the same physical blocks
    - Pooled allocation: vLLM-style dynamic block allocation from a shared pool

    Args:
        batch_size: Number of users/sequences.
        max_seq_len: Maximum sequence length per user.
        block_size: KV cache block size (tokens per block).

    Returns:
        Page table tensor [batch_size, num_blocks], dtype int32.
    """
    num_blocks_per_user = (max_seq_len + block_size - 1) // block_size
    page_table = torch.zeros(batch_size, num_blocks_per_user, dtype=torch.int32)
    for user_id in range(batch_size):
        start_block = user_id * num_blocks_per_user
        page_table[user_id] = torch.arange(start_block, start_block + num_blocks_per_user, dtype=torch.int32)
    return page_table


# =============================================================================
# Batched prefill (partial B=8) — predicate + slot packing + per-bucket grouping
# =============================================================================
# TTTv2 normally prefills 32 users in a sequential per-user loop (32 forward passes). When a group of
# users shares the same padded prefill length and has no prefix cache, those passes can be fused into
# batched passes over a folded [1, 1, padded_batch*S, dim] tensor — a single-pass-per-group device
# workload that utilises the core grid far better than the 32 small S=128 passes. This closes the
# batch-32 TTFT gap vs TTTv1. Predicate ported from TTTv1 generator.py use_batched_prefill (L631-691),
# capped at `max_prefill_batch_size` (default 8) for partial batching — most of the win, far less DRAM
# risk than full B=32.
#
# Composition with the Phase-1 per-bucket TTFT fix: the Phase-1 fix restored per-bucket prefill (each
# user prefills at its OWN get_padded_prefill_len bucket, with a SHARED persistent cos/sin so ≥2
# bucket traces coexist correctly). Batched prefill therefore fires PER UNIFORM-LENGTH BUCKET GROUP —
# _group_slots_by_prefill_bucket splits empty_slots by bucket, and the predicate is applied per group.
# A mixed batch {128,1024} thus batches the 128-bucket users together and the 1024-bucket users
# together (≥2 coexisting BATCHED traces), rather than declining the whole batch. The batched trace
# body sources cos/sin from the SAME shared persistent tensor (never a fresh per-capture slice) so the
# mixed-bucket aliasing the Phase-1 fix cured cannot return under batching.

# Mirror of TTTv1 generator.py constants.
SUPPORTED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
MAX_BATCHED_PREFILL_SEQ_LEN = 128 * 1024

# A/B escape hatch: set TTT_BATCHED_EXTRACT_HOST_GATHER=1 to force the legacy host-gather last-token
# extraction (reads the whole folded [1,1,B*S,dim] hidden to host, gathers on host, re-uploads).
# Default (unset) uses the on-device selection-matmul gather, which keeps the hidden on device and
# reads back only the [1,1,32,vocab] logits — closing the batch-32 prefill-TTFT gap vs TTTv1. Kept
# only for before/after measurement; the on-device path is bit-identical (see _gather_last_tokens_*).
_BATCHED_EXTRACT_HOST_GATHER = bool(os.environ.get("TTT_BATCHED_EXTRACT_HOST_GATHER"))


def select_batched_prefill_padded_batch(model_args, batch_size, prefill_seq_lens, num_cached_per_user):
    """Return the per-group batch size for batched prefill, or ``None`` to use the sequential loop.

    Conditions (TTTv1 parity, design §4):
      - ``batch_size > 1`` and ``model_args`` present,
      - per-model opt-in (``supports_batched_prefill``) and not disabled (``disable_batched_prefill``),
      - all users share one padded prefill length (uniform ``prefill_seq_lens``),
      - no prefix caching (all ``num_cached_per_user == 0``; the batched path feeds full tokens),
      - ``data_parallel == 1``,
      - ``seq <= max_prefill_chunk_size`` (chunked prompts stay sequential — batching buys no
        single-pass win and re-introduces the DRAM pressure chunking relieves, tt-metal #45234),
      - ``padded_batch * seq < MAX_BATCHED_PREFILL_SEQ_LEN`` (DRAM / all-gather guard).

    ``padded_batch`` is the smallest supported batch size >= ``min(batch_size, max_prefill_batch_size)``;
    32 users with the default cap of 8 therefore run as 4 batched passes of 8. This predicate is applied
    per uniform-length bucket group (see _group_slots_by_prefill_bucket), not to the whole batch.
    """
    if model_args is None or batch_size <= 1:
        return None
    # Opt-in: only models whose prefill_forward threads `batch_size` (and that set this flag) may use
    # the batched path. Unmodified models fall through to the sequential loop — never crash.
    if not getattr(model_args, "supports_batched_prefill", False):
        return None
    if getattr(model_args, "disable_batched_prefill", False):
        return None
    if len(set(prefill_seq_lens)) != 1:
        return None
    if any(n != 0 for n in num_cached_per_user):
        return None
    if getattr(model_args, "data_parallel", 1) != 1:
        return None
    seq = int(prefill_seq_lens[0])
    max_chunk = getattr(model_args, "max_prefill_chunk_size", seq)
    if max_chunk is not None and seq > max_chunk:
        return None
    cap = getattr(model_args, "max_prefill_batch_size", 8) or 8
    group = min(batch_size, cap)
    padded_batch = next((b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= group), group)
    if padded_batch > batch_size:
        # Never clamp to a non-power-of-2 count. The folded QKV matmul reshapes [1,1,padded_batch*seq,·]
        # into MAX_QKV_MM_SEQ_LEN(2048)-length chunks (attention_1d.py); a fold longer than 2048 that is
        # not a multiple of 2048 raises. Clamping padded_batch straight to batch_size produced exactly
        # that for group sizes in (16,32) — e.g. a 30-user rotation folded to 3840 and crashed. Snap to
        # the largest SUPPORTED (power-of-2) bucket <= batch_size instead: the fold is then always
        # reshape-safe (power-of-2 * 128), and the sub-group loop prefills the trailing remainder as a
        # partial (eager) sub-group — the same partial path the default cap=8 already exercises. So 32
        # users fold in one pass while a 30-user group folds as 16 + 14. (TTTv1 instead pads UP to the
        # bucket and fills padding slots; this keeps TTTv2's no-pad-waste design and stays safe.)
        padded_batch = max((b for b in SUPPORTED_PREFILL_BATCH_SIZES if b <= batch_size), default=1)
    if padded_batch <= 1:
        return None
    if padded_batch * seq >= MAX_BATCHED_PREFILL_SEQ_LEN:
        return None
    return padded_batch


def _group_slots_by_prefill_bucket(empty_slots, prompt_lens, num_cached_per_user):
    """Group ``empty_slots`` (positions in the batch) by their padded prefill bucket.

    Returns an ordered dict ``{prefill_seq_len: [positions]}`` where each position ``i`` indexes
    ``empty_slots``/``prompt_lens``/``num_cached_per_user`` (NOT the physical slot). Positions whose
    ``prompt_lens[i] - num_cached_per_user[i] <= 0`` (nothing to prefill) are dropped — the caller
    handles those in the sequential skip path. Buckets are returned in first-seen order so the
    dispatch is deterministic across repeats (matches the sequential loop's slot order).
    """
    groups: dict[int, list[int]] = {}
    for i in range(len(empty_slots)):
        new_tokens = int(prompt_lens[i]) - num_cached_per_user[i]
        if new_tokens <= 0:
            continue
        bucket = get_padded_prefill_len(new_tokens)
        groups.setdefault(bucket, []).append(i)
    return groups


def _plan_batched_prefill(model_args, empty_slots, prompt_lens, num_cached_per_user):
    """Decide which positions batch (per bucket) and which stay sequential.

    Returns ``(batched_groups, sequential_positions)`` where:
      - ``batched_groups`` is a list of ``(prefill_seq_len, padded_batch, positions)`` — one entry per
        uniform-length bucket group that satisfies the predicate; ``positions`` index ``empty_slots``.
      - ``sequential_positions`` is the set of positions NOT covered by any batched group (buckets that
        declined batching, e.g. a lone user or a bucket the predicate rejected).

    When ``batched_groups`` is empty every position is sequential and the caller's original per-user
    loop runs unchanged — the ``DISABLE_BATCHED_PREFILL`` / non-opted-in / batch<=1 path is therefore
    byte-identical to the pre-feature code.
    """
    groups = _group_slots_by_prefill_bucket(empty_slots, prompt_lens, num_cached_per_user)
    batched_groups = []
    sequential_positions: set[int] = set()
    for bucket, positions in groups.items():
        group_seq_lens = [bucket] * len(positions)
        group_cached = [num_cached_per_user[i] for i in positions]
        padded_batch = select_batched_prefill_padded_batch(model_args, len(positions), group_seq_lens, group_cached)
        if padded_batch is not None:
            batched_groups.append((bucket, padded_batch, positions))
        else:
            sequential_positions.update(positions)
    return batched_groups, sequential_positions


def _build_batched_prefill_group(tokens, prompt_lens, page_table, positions, padded_batch, prefill_seq_len, block_size):
    """Pack one group of users into folded batched-prefill inputs.

    ``positions`` are the (up to ``padded_batch``) positional row indices into
    ``tokens``/``prompt_lens``/``page_table`` for the real users in this group — the same integers the
    plan uses to index ``empty_slots``, since those tensors are laid out in empty-slot order. Returns
    ``(folded_tokens [1, padded_batch*prefill_seq_len], group_page_table [padded_batch, num_blocks],
    group_idxs, last_token_idxs)``. ``group_idxs`` are those row indices (used to index
    ``tokens``/``prompt_lens``/the output); the page table is built with one LOCAL row per real user
    (row i = that user's physical blocks), so the attention KV-fill loop and the captured trace use
    local batch_idx 0..padded_batch-1 — identical across every group, which lets a single trace replay
    for all groups.
    """
    num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
    folded = torch.zeros(1, padded_batch * prefill_seq_len, dtype=tokens.dtype)
    group_pt = torch.zeros(padded_batch, num_blocks, dtype=torch.int32)
    group_idxs = list(positions[:padded_batch])
    last_token_idxs = []
    for local_i, idx in enumerate(group_idxs):
        sl = int(prompt_lens[idx])
        folded[0, local_i * prefill_seq_len : local_i * prefill_seq_len + sl] = tokens[idx, :sl]
        group_pt[local_i] = page_table[idx, :num_blocks]
        last_token_idxs.append(sl - 1)
    return folded, group_pt, group_idxs, last_token_idxs


def _build_decode_topk_param_tensors(mesh_device, sampling_params, batch_size, allow_force_argmax=True):
    """Build replicated (k, p, temp) device tensors for ``Sampling1D``'s top-k path.

    Reuses ``format_sampling_params`` (so temperature inversion, top-p clamping and the
    ``temp==0 -> k=1`` rewrite match TTTv1 / PERF.md exactly), then mirrors TTSampling's
    1D layout: ROW_MAJOR, uint32 k / bfloat16 p / bfloat16 temp, replicated across the mesh.

    Returns ``None`` **only when force-argmax is allowed** and the formatted params reduce to
    greedy (all k==1, p==0, temp==1) — the caller then uses the cheaper full-vocab argmax path.
    When ``allow_force_argmax`` is False the model has no argmax path, so greedy params must
    still build real tensors and route through the top-k op (k=1 top-k == argmax-via-topk),
    exactly as TTTv1 does for the ``temp=0`` PERF.md recipe — never returning ``None``.

    Note: ``format_sampling_params`` rewrites every ``temp==0`` row to ``(temp=1, k=1, p=0)``
    (generator.py:512-517) — top_p is zeroed too — so the PERF.md ``temp=0, top_k=32, top_p=0.08``
    recipe reduces to the greedy representation, not ``k=1/p=0.08/temp=1``.
    """
    from models.common.sampling import format_sampling_params

    # format_sampling_params asserts the batch is a multiple of 32; round up for the call (the
    # greedy params are uniform/broadcast) then slice back to batch_size. This lets the argmax
    # reduction below return None for batch_size < 32 (argmax works at any batch) instead of
    # tripping that assert before we ever reach the check.
    fmt_len = ((batch_size + 31) // 32) * 32
    fmt = format_sampling_params(sampling_params, fmt_len)
    k = list(fmt.top_k)[:batch_size]
    p = list(fmt.top_p)[:batch_size]
    temp = list(fmt.temperature)[:batch_size]

    if allow_force_argmax and all(kk == 1 for kk in k) and all(pp == 0 for pp in p) and all(tt == 1 for tt in temp):
        return None

    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    k_tt = ttnn.from_torch(
        torch.tensor(k, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    p_tt = ttnn.from_torch(
        torch.tensor(p, dtype=torch.float32),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    temp_tt = ttnn.from_torch(
        torch.tensor(temp, dtype=torch.float32),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    return k_tt, p_tt, temp_tt


# =============================================================================
# Protocol: LLMModel
# =============================================================================
# The engine expects a model with these attributes/methods. This is a duck-typed
# protocol, not a formal interface (no ABC). Any model that provides these can
# be used with the engine.
#
# Required attributes:
#   - vocab_size: int
#   - n_layers: int
#   - num_devices: int
#   - rope_setup: has .get_rot_idxs(), .get_rot_mats(), .cos_matrix, .sin_matrix, .load_device_weights()
#   - sampling: Sampling1D | None
#   - mesh_device: ttnn.MeshDevice
#
# Required methods:
#   - embed_prefill(tokens_tt) -> ttnn.Tensor
#   - embed_decode(tokens_tt) -> ttnn.Tensor
#   - prefill_forward(x_embed, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx, get_last_token) -> ttnn.Tensor
#   - post_process_prefill_output(hidden_states, last_token_idx) -> ttnn.Tensor
#   - decode_forward(x_embed, current_pos, rot_mats, page_table) -> ttnn.Tensor
#   - gather_and_untilize_logits(logits) -> ttnn.Tensor
#   - increment_positions(current_pos, rot_mat_idxs) -> None
#   - set_kv_cache(kv_cache) -> None


# =============================================================================
# Output Spec — captures device-level tensor spec from compile
# =============================================================================


@dataclass
class TensorSpec:
    """Captured tensor specification from compile warmup run.

    These specs describe what the executor produces. Useful for:
    - Multi-CQ output buffer pre-allocation
    - Exit-boundary validation
    - Downstream consumer contracts
    """

    shape: tuple[int, ...]
    dtype: ttnn.DataType
    layout: ttnn.Layout
    memory_config: ttnn.MemoryConfig | None = None

    @classmethod
    def from_tensor(cls, tensor: ttnn.Tensor) -> TensorSpec:
        """Capture spec from an actual device tensor."""
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            layout=tensor.layout,
            memory_config=tensor.memory_config() if tensor.is_allocated() else None,
        )


# =============================================================================
# EagerLLMExecutor — thick engine for direct execution
# =============================================================================


class EagerLLMExecutor:
    """Eager executor engine — owns prefill/decode implementation.

    Common LLM operations live here. Model-specific details come from
    the model object (model.model_args).

    This is a thick engine: it owns the full implementation of prefill/decode,
    including input prep, output processing, chunked prefill, and KV cache allocation.

    Attributes:
        model: Transformer model with prefill_forward(), decode_forward(), model_args.
        mesh_device: TT mesh device for execution.
        mode: Current execution mode (PREFILL or DECODE).
        prefill_output_spec: Captured output spec from compile_prefill().
        decode_output_spec: Captured output spec from compile_decode().
    """

    def __init__(self, model, mesh_device: ttnn.MeshDevice, iter_named_modules=None) -> None:
        """Initialize eager executor engine.

        Args:
            model: Transformer model with prefill_forward(), decode_forward(), model_args, etc.
            mesh_device: TT mesh device for execution.
        """
        self.model = model
        self.mesh_device = mesh_device
        self._iter_named_modules = iter_named_modules
        self.mode = None
        self._kv_cache: list | None = None

        # todo)) this could be a composed optional for debug!
        # Output specs captured during compile (for multi-CQ pre-allocation)
        self.prefill_output_spec: TensorSpec | None = None
        self.decode_output_spec: TensorSpec | None = None

    @property
    def model_args(self):
        """Model args come from the model object."""
        return getattr(self.model, "model_args", None)

    # =========================================================================
    # KV Cache
    # =========================================================================

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple[int, ...],  # [num_blocks, num_heads, block_size, head_dim]
        dtype: torch.dtype,
        num_layers: int,
    ) -> list[list[ttnn.Tensor]]:
        """Allocate paged KV cache on device. Returns list[list[ttnn.Tensor]]."""
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_cache = []
        cache_path = self.model_args.model_cache_path if self.model_args else None

        for layer_num in range(num_layers):
            kv_cache_dtype = ttnn.bfloat8_b
            ma = self.model_args
            if ma is not None:
                explicit = getattr(ma, "kv_cache_dtype", None)
                if explicit is not None:
                    kv_cache_dtype = explicit
                elif getattr(ma, "optimizations", None) is not None:
                    from models.tt_transformers.tt.model_config import TensorGroup

                    configured = ma.optimizations.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.KV_CACHE)
                    if configured is not None:
                        kv_cache_dtype = configured

            # todo)) this could be lazy weight?
            # todo)) use ttnn.zeros() directly instead of as_tensor?
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_cache_dtype,
                    cache_file_name=(
                        cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}" if cache_path else None
                    ),
                )
                for kv in ["k", "v"]
            ]
            kv_cache.append(kv_tt_i)

        self._kv_cache = kv_cache
        self.model.set_kv_cache(kv_cache)
        return kv_cache

    # =========================================================================
    # On-device sampling routing (argmax vs top-k op path)
    # =========================================================================

    def _sampling_decode_forward(self, logits, sampling_params, tt_out_tok=None):
        """Run ``model.sampling`` on device, choosing the path that matches PERF.md.

        Routing depends on the model's ``allow_force_argmax``:
          * ``allow_force_argmax=True``  — greedy params reducing to (k==1, p==0, temp==1) take
            the force-argmax full-vocab all-gather path (``decode_forward`` with no k/p/temp).
          * ``allow_force_argmax=False`` (the 1B/7B recipe) — there is no argmax path, so *every*
            recipe, greedy included, builds k/p/temp tensors and takes the **top-k op path**:
            per-device ``ttnn.topk`` → barrier-free all-gather of the ``[*,32]`` tuples →
            ``ttnn.sampling``. This mirrors TTTv1, whose ``temp=0`` PERF.md recipe always runs
            ``ttnn.topk`` (``format_sampling_params`` rewrites ``temp=0`` to the greedy k=1/p=0/temp=1
            representation, so k=1 top-k == argmax-via-topk).

        The k/p/temp tensors are built once and cached so the *same* persistent device
        tensors are referenced during trace warmup, capture and replay (greedy decode
        keeps them constant, so no per-step update is needed).
        """
        kpt = self._get_decode_sampling_kpt(sampling_params)
        if kpt is None:
            return self.model.sampling.decode_forward(logits, tt_out_tok=tt_out_tok)
        k_tt, p_tt, temp_tt = kpt
        return self.model.sampling.decode_forward(logits, k=k_tt, p=p_tt, temp=temp_tt, tt_out_tok=tt_out_tok)

    def _get_decode_sampling_kpt(self, sampling_params):
        """Lazily build + cache (k, p, temp) device tensors, or None for force-argmax."""
        if getattr(self, "_decode_sampling_kpt_built", False):
            return self._decode_sampling_kpt
        self._decode_sampling_kpt = _build_decode_topk_param_tensors(
            self.mesh_device,
            sampling_params,
            self.model.sampling.config.max_batch_size,
            allow_force_argmax=self.model.sampling.config.allow_force_argmax,
        )
        self._decode_sampling_kpt_built = True
        return self._decode_sampling_kpt

    def _assert_kv_cache_identity(self, kv_cache):
        """Verify kv_cache passed to forward is the same object bound at allocation."""
        if kv_cache is not None and self._kv_cache is not None:
            assert kv_cache is self._kv_cache, (
                "kv_cache passed to forward differs from the allocated cache. "
                "Call allocate_kv_cache() again after reallocating."
            )

    # =========================================================================
    # Input Preparation
    # =========================================================================

    def _prepare_prefill_device_inputs(
        self, tokens, page_table, start_pos=0, chunk_page_table=None, last_token_idx=None
    ):
        """Prepare eager prefill device inputs. Returns (tokens_embd, cos, sin, page_table_tt, chunk_page_table_tt).

        Args:
            tokens: Input tokens [1, seq_len].
            page_table: Page table for paged attention [1, num_blocks]. Required.
            start_pos: Starting position for prefix caching.
            chunk_page_table: Chunk page table for chunked prefill (optional, derived from page_table).
            last_token_idx: Index of last token for output extraction.
        """
        assert tokens.dim() == 2, "tokens must be 2D"
        tokens_reshaped = tokens.reshape(1, 1, 1, -1)
        S = tokens_reshaped.shape[-1]
        tokens_tt = ttnn.from_torch(
            tokens_reshaped,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        tokens_embd = self.model.embed_prefill(tokens_tt)

        # todo)) this rope code could be refactored?
        rope = self.model.rope_setup
        rope.load_device_weights()
        mat_len = rope.cos_matrix.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)

        prefill_start = start_pos
        slice_end = min(mat_len, required_end)

        cos_slice = rope.cos_matrix[:, :, prefill_start:slice_end, :]
        sin_slice = rope.sin_matrix[:, :, prefill_start:slice_end, :]

        if pad_len > 0:
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_page_table = ttnn.from_torch(
            page_table,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # NOTE: this is a dynamic feature because it depends on seq_len
        tt_chunk_page_table = None
        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return (
            tokens_embd,
            cos_slice,
            sin_slice,
            tt_page_table,
            tt_chunk_page_table,
        )

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table):
        """Prepare decode inputs as host tensors. Returns (tokens, current_pos, rope_idxs, page_table).

        Args:
            tokens: Input tokens [batch_size].
            current_pos: Current position per user [batch_size].
            page_table: Page table for paged attention [batch_size, max_blocks]. Required.
        """
        B = tokens.shape[0]
        max_batch = self.model_args.max_batch_size if self.model_args else B
        assert B == max_batch, f"Batch size {B} must equal max_batch_size {max_batch}"

        tokens_padded = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens)), "constant", 0)
        tokens_tt = ttnn.from_torch(
            tokens_padded,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_tt = ttnn.unsqueeze_to_4D(tokens_tt)

        rot_current_pos = torch.maximum(current_pos, torch.tensor(0, dtype=torch.int64))
        rope_idxs = self.model.rope_setup.get_rot_idxs(rot_current_pos, on_host=True)

        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, None),
                mesh_shape=cluster_shape,
            ),
        )

        tt_page_table = ttnn.from_torch(
            page_table,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, None),
                mesh_shape=cluster_shape,
            ),
        )

        return tokens_tt, current_pos_tt, rope_idxs, tt_page_table

    # todo)) inline this function
    def prepare_decode_inputs_device(self, tokens, current_pos, page_table):
        """Prepare decode inputs on device. Returns device tensors."""
        host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
        return copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

    # =========================================================================
    # Compile
    # =========================================================================

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,  # [batch_size, seq_len], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        prompt_lens: torch.Tensor | None = None,  # [batch_size], int64
        empty_slots: list[int] | None = None,
        start_pos: torch.Tensor | None = None,  # [batch_size], int64
        sampling_params: SamplingParams | None = None,  # accepted for parity; eager path never traces
    ) -> torch.Tensor:  # [batch_size, 1, vocab_size], float32
        """Compile prefill for specific inputs. Returns logits from warmup run.

        Also captures output spec for multi-CQ pre-allocation.

        Args:
            tokens: Input token IDs, shape [batch_size, seq_len].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache from allocate_kv_cache().
            prompt_lens: Actual prompt length per user, shape [batch_size].
            empty_slots: List of user IDs to prefill.
            start_pos: Starting position for prefix caching, shape [batch_size].

        Returns:
            Logits tensor, shape [batch_size, 1, vocab_size].
        """
        # Boundary assertions
        assert tokens.dim() == 2, f"tokens must be [batch_size, seq_len], got {tokens.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"
        if prompt_lens is not None:
            assert prompt_lens.dim() == 1, f"prompt_lens must be [batch_size], got {prompt_lens.dim()}D"
        if start_pos is not None:
            assert start_pos.dim() == 1, f"start_pos must be [batch_size], got {start_pos.dim()}D"

        logits = self.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
        )
        ttnn.synchronize_device(self.mesh_device)

        # Capture output spec (logits is host tensor here, so just record shape/dtype)
        # todo)) use TensorSpec.from_tensor() directly? logits is torch.tensor though so may need new function/method
        self.prefill_output_spec = TensorSpec(
            shape=tuple(logits.shape),
            dtype=ttnn.bfloat16,  # Model output dtype
            layout=ttnn.TILE_LAYOUT,
            memory_config=None,  # Host tensor
        )

        return logits

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,  # [batch_size], int64
        start_pos: torch.Tensor,  # [batch_size], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """Compile decode for specific inputs. One warmup run, discard output.

        Also captures output spec for multi-CQ pre-allocation.

        Args:
            tokens: Input token IDs, shape [batch_size].
            start_pos: Current position per user, shape [batch_size].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache from allocate_kv_cache().
            sampling_params: Sampling parameters for on-device sampling.
        """
        # Boundary assertions
        assert tokens.dim() == 1, f"tokens must be [batch_size], got {tokens.dim()}D"
        assert start_pos.dim() == 1, f"start_pos must be [batch_size], got {start_pos.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"

        output = self.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=False,
            sampling_params=sampling_params,
        )
        ttnn.synchronize_device(self.mesh_device)

        # Capture output spec from device tensor
        logits_or_tokens, _ = output
        if isinstance(logits_or_tokens, ttnn.Tensor) and logits_or_tokens.is_allocated():
            self.decode_output_spec = TensorSpec.from_tensor(logits_or_tokens)

    def compile(
        self,
        *,
        prefill_tokens: torch.Tensor,  # [batch_size, seq_len], int64
        prefill_page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        prompt_lens: torch.Tensor | None = None,  # [batch_size], int64
        empty_slots: list[int] | None = None,
        start_pos: torch.Tensor | None = None,  # [batch_size], int64
        sampling_params: SamplingParams | None = None,
        validate_configs: bool = False,
    ) -> None:
        """One-shot prefill + decode warmup compile.

        Convenience wrapper that compiles both prefill and decode for the given
        inputs (one warmup run each, output specs captured along the way). This is
        the per-instance counterpart to ``compile_prefill()`` / ``compile_decode()``
        and delegates to the same internal helper used by ``run_teacher_forcing()``.

        Args:
            prefill_tokens: Prefill token IDs, shape [batch_size, seq_len].
            prefill_page_table: Page table for paged attention, shape
                [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache from allocate_kv_cache().
            prompt_lens: Actual prompt length per user, shape [batch_size].
            empty_slots: List of user IDs to prefill.
            start_pos: Starting position for prefix caching, shape [batch_size].
            sampling_params: Sampling parameters for on-device decode sampling.
            validate_configs: When True, instrument the warmup passes to check that
                each module's actual input memory config matches its declared config
                (see ``_validate_module_configs``). Defaults to False.
        """
        _compile_prefill_and_decode(
            self,
            prefill_tokens=prefill_tokens,
            prefill_page_table=prefill_page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
            validate_configs=validate_configs,
        )

    def _validate_module_configs(self, *, mode: str):
        """Context manager that validates module input configs for one forward pass.

        Instruments the model's modules during a single ``mode`` ("prefill" or
        "decode") forward pass and checks that each module's actual input memory
        config matches the config it declares. Returns a no-op context when this
        executor has no ``iter_named_modules`` hook.

        Usage::

            with executor._validate_module_configs(mode="prefill"):
                executor.compile_prefill(...)
        """
        return _get_validation_context(self, mode=mode)

    # =========================================================================
    # Prefill Forward
    # =========================================================================

    def prefill_forward(
        self,
        tokens: torch.Tensor,  # [batch_size, seq_len], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        prompt_lens: torch.Tensor | None = None,  # [batch_size], int64
        empty_slots: list[int] | None = None,
        sampling_params: SamplingParams | None = None,
        start_pos: torch.Tensor | None = None,  # [batch_size], int64
    ) -> torch.Tensor:  # [batch_size, 1, vocab_size], float32
        """Per-user prefill loop with chunked prefill + prefix caching.

        Args:
            tokens: Input token IDs, shape [batch_size, seq_len].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache from allocate_kv_cache().
            prompt_lens: Actual prompt length per user, shape [batch_size].
            empty_slots: List of user IDs to prefill.
            sampling_params: Sampling parameters (not used in prefill).
            start_pos: Starting position for prefix caching, shape [batch_size].

        Returns:
            Logits tensor, shape [batch_size, 1, vocab_size].
        """
        # Boundary assertions
        assert tokens.dim() == 2, f"tokens must be [batch_size, seq_len], got {tokens.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"
        if prompt_lens is not None:
            assert prompt_lens.dim() == 1, f"prompt_lens must be [batch_size], got {prompt_lens.dim()}D"
        if start_pos is not None:
            assert start_pos.dim() == 1, f"start_pos must be [batch_size], got {start_pos.dim()}D"
        self.mode = Mode.PREFILL
        self._assert_kv_cache_identity(kv_cache)

        batch_size, batch_seq_len = tokens.shape
        vocab_size = self.model.vocab_size
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]

        # todo)) output_tensor is just overwritten later? why allocate it here then?
        output_tensor = torch.zeros(batch_size, 1, vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)
        if empty_slots is None:
            empty_slots = list(range(batch_size))

        prefill_results = []

        # Batched prefill fast path: fuse equal-length users into batched passes, PER uniform-length
        # bucket group (see _plan_batched_prefill / select_batched_prefill_padded_batch). Positions the
        # plan leaves sequential (or all positions when batching is off / declined) fall through to the
        # per-user loop below, which is byte-identical to the pre-feature path.
        num_cached_per_user = [int(start_pos[i]) if start_pos is not None else 0 for i in range(len(empty_slots))]
        batched_groups, sequential_positions = _plan_batched_prefill(
            self.model_args, empty_slots, prompt_lens, num_cached_per_user
        )
        seq_filter = None
        if batched_groups:
            seq_filter = sequential_positions
            for prefill_seq_len_g, padded_batch_g, positions_g in batched_groups:
                logger.info(
                    f"Batched prefill: {len(positions_g)} users in groups of {padded_batch_g} "
                    f"(seq={prefill_seq_len_g})"
                )
                prefill_results.extend(
                    self._prefill_forward_batched_group(
                        tokens, page_table, prompt_lens, positions_g, padded_batch_g, prefill_seq_len_g
                    )
                )

        for idx, user_id in enumerate(empty_slots):
            if seq_filter is not None and idx not in seq_filter:
                continue  # handled by the batched path above
            seq_len = int(prompt_lens[idx])
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            new_tokens = seq_len - num_cached_tokens
            if new_tokens <= 0:
                logger.info(
                    f"Skipping prefill for user_id={user_id}: seq_len={seq_len}, num_cached_tokens={num_cached_tokens}"
                )
                continue
            last_token_idx = seq_len - 1
            prefill_seq_len = get_padded_prefill_len(new_tokens)

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            prefill_ids = torch.cat(
                [
                    tokens[idx : idx + 1, num_cached_tokens:seq_len],
                    torch.zeros(1, prefill_seq_len - new_tokens).long(),
                ],
                dim=-1,
            )

            page_table_user = _get_prefill_user_page_table(
                page_table[idx : idx + 1],
                kv_cache,
                seq_len,
            )

            logits = self._prefill_single_user(
                prefill_ids,
                page_table=page_table_user,
                user_id=0,  # Always 0 with paged attention (page table handles user mapping)
                last_token_idx=last_token_idx,
                num_cached_tokens=num_cached_tokens,
            )

            logits = ttnn.untilize(logits, use_multicore=True)
            prefill_results.append(
                {
                    "idx": idx,
                    "last_token_idx": last_token_idx,
                    "logits": logits.cpu(blocking=False),
                }
            )

        # One device barrier drains every pending ``logits.cpu(blocking=False)`` transfer dispatched
        # above; ``_process_output_prefill`` then runs on the already-resident HOST tensors (no device
        # work). One barrier suffices (an idle ``synchronize_device`` is cheap), so the old per-user
        # barrier was redundant — kept hoisted as a cleanup. Batched extraction returns ONE shared host
        # logits tensor for a whole group (each user is a distinct row); concat its shards once per
        # unique tensor and slice each user's row instead of re-concatenating per user (32 host concats
        # -> 1 on batch-32). Cache keyed by tensor identity, so the sequential path is unchanged.
        if prefill_results:
            ttnn.synchronize_device(self.mesh_device)
        _concat_cache: dict = {}
        for res in prefill_results:
            key = id(res["logits"])
            full = _concat_cache.get(key)
            if full is None:
                assert res["logits"].storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
                full = _concat_host_output(res["logits"], cluster_shape)
                _concat_cache[key] = full
            last_relative = res["last_token_idx"] - (int(start_pos[res["idx"]]) if start_pos is not None else 0)
            output_tensor[res["idx"]] = full[0, 0, last_relative % 32, :vocab_size]

        return output_tensor

    def _prefill_single_user(self, tokens, page_table, user_id, last_token_idx, num_cached_tokens=0):
        """Prefill a single user with chunked prefill support.

        Args:
            tokens: Input tokens [1, seq_len].
            page_table: Page table for this user [1, num_blocks]. Required.
            user_id: User ID (always 0 with paged attention).
            last_token_idx: Index of last token for output extraction.
            num_cached_tokens: Number of tokens already in KV cache (prefix caching).
        """
        seq_len = tokens.shape[-1]
        max_chunk = self.model_args.max_prefill_chunk_size if self.model_args else seq_len
        use_chunked = seq_len > max_chunk
        use_prefix_caching = num_cached_tokens > 0

        # todo)) refactor this to be more readable?
        if use_chunked or use_prefix_caching:
            assert self._kv_cache is not None, "KV cache must be allocated for chunked prefill or prefix caching"
            chunk_size = get_max_prefill_chunk_size(seq_len, max_chunk) if use_chunked else seq_len

            last_token_in_seq = last_token_idx - num_cached_tokens
            block_size = get_block_size(self._kv_cache)
            last_token_in_chunk = last_token_in_seq % chunk_size
            last_chunk_start = (last_token_in_seq // chunk_size) * chunk_size

            page_table_user = page_table[user_id : user_id + 1, :]
            num_pad_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size) - page_table_user.shape[1]
            page_table_padded = torch.cat([page_table_user, torch.zeros(1, num_pad_blocks, dtype=torch.int32)], dim=-1)

            for chunk_start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
                chunk_end = chunk_start + chunk_size
                chunk_start_rel = chunk_start - num_cached_tokens
                chunk_end_rel = chunk_end - num_cached_tokens

                chunk_tokens = tokens[:, chunk_start_rel:chunk_end_rel]
                chunk_page_table = page_table_padded[:, chunk_start // block_size : chunk_end // block_size]

                prefill_input, cos, sin, page_table_tt, chunk_page_table_tt = self._prepare_prefill_device_inputs(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_padded,
                    chunk_page_table=chunk_page_table,
                    last_token_idx=last_token_idx,
                )

                get_last_token = (last_token_in_chunk // 32) * 32
                logits = self.model.prefill_forward(
                    prefill_input,
                    [cos, sin],
                    user_id=0,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=get_last_token,
                )

                if chunk_start_rel == last_chunk_start:
                    return logits
                else:
                    del logits
        else:
            prefill_input, cos, sin, page_table_tt, _ = self._prepare_prefill_device_inputs(
                tokens,
                page_table=page_table,
                last_token_idx=last_token_idx,
            )

            get_last_token = (last_token_idx // 32) * 32
            return self.model.prefill_forward(
                prefill_input,
                [cos, sin],
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=get_last_token,
            )

    # =========================================================================
    # Batched Prefill (partial B=padded_batch)
    # =========================================================================

    def _prepare_batched_prefill_device_inputs(self, folded_tokens, group_page_table, prefill_seq_len):
        """Prepare device inputs for one batched-prefill group (eager / trace-warmup compile).

        Returns ``(tokens_embd, cos, sin, page_table_tt)``:
          - ``folded_tokens`` [1, padded_batch*S] is embedded to a folded [1, 1, padded_batch*S, dim],
          - ``cos``/``sin`` are the per-user rope slices for positions 0..S-1 (all users start at 0,
            so the same slices broadcast over the batch axis inside attention),
          - ``group_page_table`` [padded_batch, num_blocks] maps each local row to its physical blocks.

        This eager path is used for direct eager execution and for the traced JIT warmup; the persistent
        TRACE inputs come from ``TracedLLMExecutor._prepare_batched_prefill_trace_inputs_host`` which
        instead sources cos/sin from the shared persistent tensor.
        """
        tokens_reshaped = folded_tokens.reshape(1, 1, 1, -1)
        tokens_tt = ttnn.from_torch(
            tokens_reshaped,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_embd = self.model.embed_prefill(tokens_tt)

        rope = self.model.rope_setup
        rope.load_device_weights()
        cos_slice = rope.cos_matrix[:, :, 0:prefill_seq_len, :]
        sin_slice = rope.sin_matrix[:, :, 0:prefill_seq_len, :]

        page_table_tt = ttnn.from_torch(
            group_page_table,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tokens_embd, cos_slice, sin_slice, page_table_tt

    def _extract_batched_prefill_logits(self, hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs):
        """Extract each user's last-token logits from a folded batched-prefill hidden state.

        ``hidden`` is the folded [1, 1, padded_batch*S, dim]. Two paths (gated by model_args
        ``batched_prefill_batched_extract``, default True):

        * **batched** (default): gather every user's last-token hidden row into one [1,1,32,dim]
          tensor and run norm + lm_head ONCE for the whole group (TTTv1
          ``extract_last_tokens_batched_prefill`` + ``_apply_norm_and_lm_head``). At padded_batch=32 this
          is a single lm_head over 32 rows == TTTv1's layout; the per-slot path instead runs 32 lm_heads
          (each wasting 31 padding rows of the tile) — the residual TTTv2→TTTv1 TTFT gap. Per-row math is
          identical (RMSNorm is per-row, lm_head rows are independent), so logits match the per-slot path.
        * **per-slot** (fallback): one ``post_process_prefill_output`` (norm+lm_head) per user — bit-
          identical to the sequential path, but 32× the lm_head work.

        Returns a list of per-user result dicts.
        """
        batched_extract = getattr(self.model_args, "batched_prefill_batched_extract", True)
        if batched_extract:
            return self._extract_batched_prefill_logits_gathered(
                hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs
            )

        dim = hidden.shape[-1]
        hidden = ttnn.reshape(hidden, [padded_batch, 1, prefill_seq_len, dim])
        results = []
        for local_i, idx in enumerate(group_idxs):
            slot_hidden = hidden[local_i : local_i + 1, :, :, :]
            logits = self.model.post_process_prefill_output(slot_hidden, last_token_idxs[local_i])
            logits = ttnn.untilize(logits, use_multicore=True)
            results.append(
                {"idx": idx, "last_token_idx": last_token_idxs[local_i], "logits": logits.cpu(blocking=False)}
            )
        return results

    def _extract_batched_prefill_logits_gathered(
        self, hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs
    ):
        """Batched last-token extraction: one norm + lm_head over all users in the group.

        Gathers each user's last-token hidden row (folded layout: user ``i`` occupies rows
        ``[i*S : i*S+S]``) into a single [1,1,32,dim] tensor — column-sharded across the mesh the same
        way the prefill residual is (dim=-1), so the model's distributed norm all-gathers correctly —
        then runs ``post_process_prefill_output`` once (slice [0:32] is a no-op on the 32-row tile).

        The gather is done ON DEVICE by default (``_gather_last_tokens_on_device``): a one-hot selection
        matmul keeps the folded hidden on device and reads back only the small [1,1,32,vocab] logits,
        instead of copying the whole [1,1,B*S,dim] hidden to host (~25 MB at B=32,S=128) to gather 32
        rows. This is the batched analogue of ``fast_prefill_last_token`` and closes the residual
        TTTv2-vs-TTTv1 batch-32 prefill-TTFT gap. Set ``TTT_BATCHED_EXTRACT_HOST_GATHER=1`` to force the
        legacy host gather (mirrors TTTv1 ``extract_last_tokens_batched_prefill``) for A/B measurement;
        both paths are bit-identical (deterministic gather of the same rows).
        """
        TILE = 32
        if _BATCHED_EXTRACT_HOST_GATHER:
            gathered = self._gather_last_tokens_host(hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs)
        else:
            gathered = self._gather_last_tokens_on_device(
                hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs
            )
        # last_token_idx = TILE-1 → post_process slices [0:TILE] (whole tile) then runs norm+lm_head once.
        logits = self.model.post_process_prefill_output(gathered, TILE - 1)
        logits = ttnn.untilize(logits, use_multicore=True)
        logits_host = logits.cpu(blocking=False)
        # Each user's logits sit at row local_i of the shared [.,.,32,vocab] output.
        return [
            {"idx": idx, "last_token_idx": local_i, "logits": logits_host} for local_i, idx in enumerate(group_idxs)
        ]

    def _gather_last_tokens_on_device(self, hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs):
        """Gather each user's last-token row from the folded hidden ON DEVICE (no host round-trip).

        ``hidden`` is folded ``[1, 1, padded_batch*S, dim]``, column-sharded on ``dim`` (dim=-1). User
        ``local_i``'s last token sits at fold row ``local_i*S + last_token_idxs[local_i]``. A one-hot
        selection matmul ``sel[1,1,32,B*S] @ hidden[1,1,B*S,dim] -> [1,1,32,dim]`` performs the gather:
        each device computes ``sel @ (its dim-shard of hidden)``, producing the dim-shard of the 32
        gathered rows — no collective, output stays column-sharded (dim=-1) TILE_LAYOUT, exactly what
        the distributed norm expects. ``sel`` has a single 1.0 per output row, so with HiFi4 + fp32
        accumulation the result is bit-identical to reading ``hidden[row]`` (``1.0*x + 0*... = x``; the
        bf16 value round-trips through fp32 unchanged). Only ``sel`` (tiny) is uploaded and only the
        final logits are read back — the ~B*S*dim host read (25 MB at B=32,S=128) is eliminated.
        Generalizes TTTv1's on-device slice (``all_same`` only) to arbitrary per-user last-token rows.
        """
        TILE = 32
        fold_len = padded_batch * prefill_seq_len
        sel = torch.zeros(1, 1, TILE, fold_len, dtype=torch.bfloat16)
        for local_i, _ in enumerate(group_idxs):
            sel[0, 0, local_i, local_i * prefill_seq_len + last_token_idxs[local_i]] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        gathered = ttnn.matmul(
            sel_tt,
            hidden,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(sel_tt)
        return gathered

    def _gather_last_tokens_host(self, hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs):
        """Legacy host gather (A/B baseline; TTTv1 ``extract_last_tokens_batched_prefill`` parity).

        Reads the whole folded ``[1,1,B*S,dim]`` hidden to host, gathers each user's last-token row, and
        re-uploads a ``[1,1,32,dim]`` column-sharded tile. Deterministic; bit-identical to the on-device
        path. Only used when ``TTT_BATCHED_EXTRACT_HOST_GATHER=1``.
        """
        ttnn.synchronize_device(self.mesh_device)
        shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(hidden)]
        host_full = torch.cat(shards, dim=-1) if len(shards) > 1 else shards[0]  # [1,1,B*S,full_dim]
        full_dim = host_full.shape[-1]
        host_full = host_full.reshape(padded_batch, prefill_seq_len, full_dim)
        TILE = 32
        combined = torch.zeros(1, 1, TILE, full_dim, dtype=host_full.dtype)
        for local_i, _ in enumerate(group_idxs):
            combined[0, 0, local_i, :] = host_full[local_i, last_token_idxs[local_i], :]
        return ttnn.from_torch(
            combined,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )

    def _prefill_forward_batched_group(self, tokens, page_table, prompt_lens, positions, padded_batch, prefill_seq_len):
        """Eager batched prefill for ONE uniform-length bucket group: process ``positions`` (row
        indices into ``tokens``/``prompt_lens``/``page_table``) in sub-groups of ``padded_batch``, one
        forward per sub-group, then extract each user's last-token logits. Returns a list of per-user
        result dicts."""
        block_size = get_block_size(self._kv_cache) if self._kv_cache is not None else 32
        prefill_results = []
        for g0 in range(0, len(positions), padded_batch):
            sub_positions = positions[g0 : g0 + padded_batch]
            folded, group_pt, group_idxs, last_token_idxs = _build_batched_prefill_group(
                tokens, prompt_lens, page_table, sub_positions, padded_batch, prefill_seq_len, block_size
            )
            tokens_embd, cos, sin, pt_tt = self._prepare_batched_prefill_device_inputs(
                folded, group_pt, prefill_seq_len
            )
            hidden = self.model.prefill_forward(
                tokens_embd,
                [cos, sin],
                user_id=list(range(len(group_idxs))),
                page_table=pt_tt,
                get_last_token=-1,
                batch_size=padded_batch,
            )
            prefill_results.extend(
                self._extract_batched_prefill_logits(hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs)
            )
        return prefill_results

    # =========================================================================
    # Decode Forward
    # =========================================================================

    def decode_forward(
        self,
        tokens: torch.Tensor,  # [batch_size], int64
        start_pos: torch.Tensor,  # [batch_size], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        read_from_device: bool = True,
        sampling_params: SamplingParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single decode step.

        Args:
            tokens: Input token IDs, shape [batch_size].
            start_pos: Current position per user, shape [batch_size].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache (for identity assertion).
            read_from_device: Whether to return host tensors.
            sampling_params: Sampling parameters for on-device sampling.

        Returns:
            (logits_or_tokens, log_probs) tuple.
            If sampling_params is None: logits [batch_size, 1, vocab_size], None
            If sampling_params provided: tokens [batch_size], None
        """
        # Boundary assertions
        assert tokens.dim() == 1, f"tokens must be [batch_size], got {tokens.dim()}D"
        assert start_pos.dim() == 1, f"start_pos must be [batch_size], got {start_pos.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"
        self.mode = Mode.DECODE
        self._assert_kv_cache_identity(kv_cache)
        B = tokens.shape[0]
        vocab_size = self.model.vocab_size
        num_devices = self.model.num_devices
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]

        sampling_on_device = sampling_params is not None

        if (
            sampling_on_device
            and self.model.sampling is not None
            and hasattr(self.model.sampling, "apply_decode_state")
        ):
            # TTTv1 SamplingGenerator pushes per-request k/p/temp into persistent device buffers
            # via apply_decode_state + seed_manager. The TTTv2 Sampling1D module holds no mutable
            # state (k/p/temp are per-call args; greedy/argmax needs none), so skip when absent.
            from models.common.sampling import broadcast_sampling_params, format_sampling_params

            per_request_params = format_sampling_params(broadcast_sampling_params(sampling_params, 0, slot_len=B), B)
            self.model.sampling.apply_decode_state(
                [per_request_params],
                reset_batch=False,
            )
            self.model.sampling.seed_manager.get_new_values()

        tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_page_table = self.prepare_decode_inputs_device(
            tokens, start_pos, page_table
        )

        # todo)) why is this here? it would be clearer to get_rot_mats right before its used?
        rot_mats = self.model.rope_setup.get_rot_mats(tt_rot_mat_idxs)
        x_embed = self.model.embed_decode(tt_tokens)

        logits = self.model.decode_forward(
            x_embed,
            tt_current_pos,
            rot_mats,
            page_table=tt_page_table,
        )

        if sampling_on_device and self.model.sampling is not None:
            # increment_positions omitted: positions are re-supplied from host each decode step
            # (see TracedLLMExecutor._capture_decode_trace for the trace-capture rationale).
            # tt_out_tok=None: let sampling allocate its own correctly-shaped output token tensor.
            # Reusing the [1,1,1,32] input-token buffer as the argmax output ([1,1,32]) is a shape
            # mismatch that forces a realloc — fatal during trace capture.
            tt_toks, tt_log_probs = self._sampling_decode_forward(logits, sampling_params, tt_out_tok=None)
            if read_from_device:
                ttnn.synchronize_device(self.mesh_device)
                toks = _process_output_decode_tokens(tt_toks.cpu(), B, cluster_shape)
                return toks, None
            return (tt_toks, tt_log_probs)

        logits = self.model.gather_and_untilize_logits(logits)

        if read_from_device:
            logits_host = logits.cpu()
            ttnn.synchronize_device(self.mesh_device)
            return _process_output_decode(logits_host, B, vocab_size, num_devices, cluster_shape), None

        return (logits, None)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """No-op for eager executor."""


# =============================================================================
# TracedLLMExecutor — thick engine with trace capture/replay
# =============================================================================


class TracedLLMExecutor:
    """Traced executor engine — adds trace capture/replay to eager execution.

    Same public API as EagerLLMExecutor. Compile methods capture traces,
    forward methods replay or fall back to eager.

    Attributes:
        model: Transformer model with prefill_forward(), decode_forward(), model_args.
        mesh_device: TT mesh device for execution.
        mode: Current execution mode (PREFILL or DECODE).
        prefill_output_spec: Captured output spec from compile_prefill().
        decode_output_spec: Captured output spec from compile_decode().
    """

    def __init__(
        self,
        model,
        mesh_device: ttnn.MeshDevice,
        iter_named_modules=None,
        ondevice_decode_loop: bool = False,
        fast_prefill_last_token: bool = False,
    ) -> None:
        """Initialize traced executor engine.

        Args:
            model: Transformer model with prefill_forward(), decode_forward(), model_args, etc.
            mesh_device: TT mesh device for execution.
            iter_named_modules: Optional callable yielding the model's named modules for config
                validation (model-specific; defaults to the generic walk in the eager engine).
            fast_prefill_last_token: Opt-in, default OFF. In the single-user (batch_size==1) sequential
                prefill path only the last-token ROW of the [1,1,32,vocab] logits tile is ever consumed
                (the assembly picks ``full[0,0,last%32,:]``). When True, that one row is sliced on device
                BEFORE the host readback, so the PCIe transfer + host ``torch.cat`` of the 8 device shards
                move [1,1,1,vocab] instead of the full 32-row tile (~32x less host concat + readback — the
                dominant non-forward term in the T3K batch-1 prefill TTFT gap vs TTTv1, which reads back
                only tokens). Correctness-identical (same row, sliced on device instead of host). Purely
                internal to prefill_forward (callers only ever see the row-reduced output_tensor), so it
                generalizes to every model's single-user prefill; kept opt-in to bound the change to the
                wired path. Inert for batched prefill (batch_size>1 shares a multi-row logits tile).
            ondevice_decode_loop: Opt-in, default OFF. When True, the captured decode trace advances
                position/rope on device (in-place ``ttnn.plus_one``) and feeds the sampled token back
                into the persistent token buffer (``ttnn.sampling(output_tensor=...)``), so steady-state
                steps replay with NO per-step host input staging — mirroring TTTv1's on-device sampling
                trace (generator ``refresh_trace_inputs=False`` + ``_increment_decode_positions_device``).
                Valid ONLY for free-running greedy/top-k generation, NOT teacher forcing (which injects
                host tokens every step), so accuracy/teacher-forcing runs must leave this OFF. Also inert
                on the force-argmax path (``_decode_loop_active`` gates it to the top-k op path).
        """
        self.model = model
        self.mesh_device = mesh_device
        self._eager = EagerLLMExecutor(model, mesh_device, iter_named_modules=iter_named_modules)
        self._cleaned_up = False
        self.ondevice_decode_loop = ondevice_decode_loop
        self.fast_prefill_last_token = fast_prefill_last_token

        # todo)) we cannot save many traces in memory! Gotta limit the number of traces! --> lru_cache?
        #        but the warmup_model_prefill() traces must be kept around forever!
        # Prefill traces: keyed by padded seq_len
        self.trace_id_prefill: dict[int, int | None] = defaultdict(lambda: None)
        self.trace_inputs_prefill: dict[int, tuple | None] = defaultdict(lambda: None)
        self.trace_output_prefill: dict[int, ttnn.Tensor | None] = defaultdict(lambda: None)

        # Batched prefill traces: keyed by (prefill_seq_len, padded_batch). One trace serves every
        # group of `padded_batch` users at that bucket — the captured KV-fill batch_idx values are
        # LOCAL (0..pb-1), identical across groups; only the tokens + page table differ and are copied
        # in on replay. Multiple (bucket, pb) traces coexist (mixed-length eval-32 -> {128,1024}); like
        # the single-user traces they share ONE persistent cos/sin (see _shared_prefill_cos_sin) so no
        # fresh per-capture cos/sin buffer lands in a coexisting trace's live range.
        self.trace_id_prefill_batched: dict[tuple, int | None] = defaultdict(lambda: None)
        self.trace_inputs_prefill_batched: dict[tuple, tuple | None] = defaultdict(lambda: None)
        self.trace_output_prefill_batched: dict[tuple, ttnn.Tensor | None] = defaultdict(lambda: None)

        # Shared, persistent full-range prefill cos/sin trace inputs (keyed by max_seq_len).
        # Materialised ONCE and reused by every per-bucket prefill trace so that no fresh per-capture
        # cos/sin slice buffer can land inside another coexisting bucket's live residual/activation
        # buffer and corrupt it (the ci-eval-32 mixed-bucket bug). See _shared_prefill_cos_sin.
        self._prefill_cos_sin_shared: dict[int, tuple] = {}

        # Decode traces: keyed by sampling_on_device (bool)
        self.trace_ids_decode: dict[bool, int | None] = defaultdict(lambda: None)
        self.trace_inputs_decode: dict[bool, tuple | None] = defaultdict(lambda: None)
        self.trace_output_decode: dict[bool, tuple | None] = defaultdict(lambda: None)

        # Per sampling-mode flag: the next decode step must re-seed the persistent buffers from host
        # (correct first token + start position). Set after every prefill and at init; cleared once
        # the device has been seeded, after which steady-state steps carry state on device.
        self._decode_needs_reseed: dict[bool, bool] = defaultdict(lambda: True)
        # Keyed per sampling-mode (True=on-device sampling, False=host), matching the per-key
        # decode trace + device page_table buffer in trace_inputs_decode. A single shared tensor
        # would let one mode's page-table update mask a needed copy on the other mode's trace
        # (whose device buffer is separate), leaving it stale.
        self._prev_decode_page_table: dict[bool, torch.Tensor | None] = defaultdict(lambda: None)

        self.mode = None
        self.already_warmed_up_prefill = False

        # Output specs captured during compile (for multi-CQ pre-allocation)
        self.prefill_output_spec: TensorSpec | None = None
        self.decode_output_spec: TensorSpec | None = None

    @property
    def model_args(self):
        """Model args come from the model object."""
        return getattr(self.model, "model_args", None)

    # =========================================================================
    # Delegate KV cache to eager engine
    # =========================================================================

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers) -> list:
        """Allocate paged KV cache on device. Delegates to eager engine."""
        return self._eager.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    @property
    def _kv_cache(self):
        return self._eager._kv_cache

    # =========================================================================
    # Warmup
    # =========================================================================

    def warmup_model_prefill(
        self,
        seq_lens: list[int],
        make_tokens,  # Callable[[int], torch.Tensor] — seq_len -> tokens [1, seq_len]
        make_page_table,  # Callable[[int], torch.Tensor] — seq_len -> page_table [1, num_blocks]
    ) -> None:
        """Compile prefill for multiple sequence lengths. Caller provides input factories.

        Args:
            seq_lens: List of sequence lengths to compile.
            make_tokens: Factory function: seq_len -> tokens tensor [1, seq_len].
            make_page_table: Factory function: seq_len -> page_table tensor [1, num_blocks].
        """
        for seq_len in seq_lens:
            tokens = make_tokens(seq_len)
            page_table = make_page_table(seq_len)
            self.compile_prefill(tokens=tokens, page_table=page_table)

    # =========================================================================
    # Trace Key Computation
    # =========================================================================

    def _get_prefill_trace_key(self, tokens: torch.Tensor) -> int:
        """Prefill trace key = padded seq_len."""
        return get_padded_prefill_len(tokens.shape[-1])

    def _get_decode_trace_key(self, sampling_params) -> bool:
        """Decode trace key = whether sampling is on device."""
        return sampling_params is not None

    def _decode_loop_active(self, sampling_params) -> bool:
        """Whether the on-device decode loop applies for this decode call.

        Requires: the opt-in flag, on-device sampling, AND the top-k op path (``ttnn.sampling``),
        whose output is uint32 ``[1,1,1,32]`` on Wormhole/Blackhole — an exact match for the
        persistent token buffer, so it can be fed back in place via ``output_tensor=`` with no
        realloc. The force-argmax path (``ttnn.argmax`` → uint32 ``[1,1,32]``, rank-3) does NOT
        match the rank-4 token buffer, so the loop stays off there and falls back to host resupply.
        """
        if not self.ondevice_decode_loop or sampling_params is None or self.model.sampling is None:
            return False
        # kpt is None only for the force-argmax path; non-None => top-k op path.
        return self._eager._get_decode_sampling_kpt(sampling_params) is not None

    def _prepare_prefill_trace_inputs_host(
        self, tokens, page_table, start_pos=0, chunk_page_table=None, last_token_idx=None
    ):
        """Prepare traced prefill host inputs for copy_host_to_device replay.

        Args:
            tokens: Input tokens [1, seq_len].
            page_table: Page table for paged attention [1, num_blocks]. Required.
            start_pos: Starting position for prefix caching.
            chunk_page_table: Chunk page table for chunked prefill (optional).
            last_token_idx: Index of last token for output extraction.
        """
        assert tokens.dim() == 2, "tokens must be 2D"
        tokens_reshaped = tokens.reshape(1, 1, 1, -1)
        S = tokens_reshaped.shape[-1]
        tokens_tt = ttnn.from_torch(
            tokens_reshaped,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        rope = self.model.rope_setup
        rope.load_device_weights()
        mat_len = rope.cos_matrix.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)
        max_seq_len = self.model_args.max_seq_len if self.model_args else mat_len

        if pad_len > 0:
            # Prefix-caching / out-of-range path: per-call padded slice (rare; not the batched-prefill
            # trace path). Kept as a fresh tensor since the shape is call-specific.
            cos_slice = rope.cos_matrix[:, :, 0:max_seq_len, :]
            sin_slice = rope.sin_matrix[:, :, 0:max_seq_len, :]
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)
        else:
            # Standard batched-prefill trace path: the cos/sin slice is the CONSTANT [0:max_seq_len]
            # range for EVERY bucket (same shape, same content, read-only, and never re-copied on
            # replay). Materialise it ONCE and share the SAME persistent device tensor across all
            # per-bucket prefill traces. This removes the fresh per-capture cos/sin buffer that
            # otherwise gets bottom-up-allocated into another coexisting bucket's live residual buffer
            # and corrupts it (device-verified: mixed-bucket ci-eval-32 corruption tracked exactly this
            # overlap). Mirrors TTTv1, which feeds prefill rot_mats from the persistent cos/sin matrix
            # rather than re-materialising a slice per bucket.
            cos_slice, sin_slice = self._shared_prefill_cos_sin(rope, max_seq_len)

        tt_page_table = ttnn.from_torch(
            page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        tt_chunk_page_table = None
        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return tokens_tt, cos_slice, sin_slice, tt_page_table, tt_chunk_page_table

    def _shared_prefill_cos_sin(self, rope, max_seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return the persistent, shared full-range ``[0:max_seq_len]`` prefill cos/sin tensors.

        Materialised once per ``max_seq_len`` and reused by every per-bucket prefill trace. The
        slice is content- and shape-identical for every bucket (the executor always slices the
        constant ``[0:max_seq_len]`` range, independent of the padded bucket length), cos/sin are
        read-only inside the trace body (``rotary_embedding_llama`` takes them by const ref), and
        they are never re-copied on replay. Sharing one device buffer therefore changes nothing
        functionally while eliminating the fresh per-capture slice buffer that could be allocated
        into another coexisting bucket trace's live residual buffer and corrupt it.
        """
        cached = self._prefill_cos_sin_shared.get(max_seq_len)
        if cached is not None:
            return cached
        cos_slice = rope.cos_matrix[:, :, 0:max_seq_len, :]
        sin_slice = rope.sin_matrix[:, :, 0:max_seq_len, :]
        self._prefill_cos_sin_shared[max_seq_len] = (cos_slice, sin_slice)
        return cos_slice, sin_slice

    # =========================================================================
    # Compile (delegates to eager, captures output specs)
    # =========================================================================

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,  # [batch_size, seq_len], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        prompt_lens: torch.Tensor | None = None,  # [batch_size], int64
        empty_slots: list[int] | None = None,
        start_pos: torch.Tensor | None = None,  # [batch_size], int64
        sampling_params: SamplingParams | None = None,
    ) -> torch.Tensor | None:  # [batch_size, 1, vocab_size] or None if already compiled
        """Compile prefill for specific inputs. Returns logits from warmup run.

        Returns None if trace for this seq_len already exists (no work needed).
        Also captures output spec for multi-CQ pre-allocation.

        Args:
            tokens: Input token IDs, shape [batch_size, seq_len].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache from allocate_kv_cache().
            prompt_lens: Actual prompt length per user, shape [batch_size].
            empty_slots: List of user IDs to prefill.
            start_pos: Starting position for prefix caching, shape [batch_size].
        """
        # Boundary assertions
        assert tokens.dim() == 2, f"tokens must be [batch_size, seq_len], got {tokens.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"

        # Skip if already compiled for this seq_len
        trace_key = self._get_prefill_trace_key(tokens)
        if self.trace_id_prefill[trace_key] is not None:
            return None

        # Allocate persistent on-device sampling buffers BEFORE the prefill trace is captured
        # (self.prefill_forward below). If they are first materialised during decode-trace capture
        # (while this prefill trace is live), tt-metal places them unsafely and a decode-trace buffer
        # clobbers the k tensor on replay -> corrupt k -> sampling-writer hang. See
        # _prealloc_sampling_buffers.
        self._prealloc_sampling_buffers(sampling_params)

        logits = self.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
        )
        ttnn.synchronize_device(self.mesh_device)

        # Capture output spec
        self.prefill_output_spec = TensorSpec(
            shape=tuple(logits.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=None,
        )

        return logits

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,  # [batch_size], int64
        start_pos: torch.Tensor,  # [batch_size], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """Compile decode for specific inputs. One warmup run, discard output.

        Skips if trace for this sampling mode already exists.
        Also captures output spec for multi-CQ pre-allocation.

        Args:
            tokens: Input token IDs, shape [batch_size].
            start_pos: Current position per user, shape [batch_size].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache from allocate_kv_cache().
            sampling_params: Sampling parameters for on-device sampling.
        """
        # Boundary assertions
        assert tokens.dim() == 1, f"tokens must be [batch_size], got {tokens.dim()}D"
        assert start_pos.dim() == 1, f"start_pos must be [batch_size], got {start_pos.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"

        # Skip if already compiled for this sampling mode
        trace_key = self._get_decode_trace_key(sampling_params)
        if self.trace_ids_decode[trace_key] is not None:
            return

        output = self.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=False,
            sampling_params=sampling_params,
        )
        ttnn.synchronize_device(self.mesh_device)

        # Capture output spec from device tensor
        logits_or_tokens, _ = output
        if isinstance(logits_or_tokens, ttnn.Tensor) and logits_or_tokens.is_allocated():
            self.decode_output_spec = TensorSpec.from_tensor(logits_or_tokens)

    # =========================================================================
    # Prefill (traced)
    # =========================================================================

    def prefill_forward(
        self,
        tokens: torch.Tensor,  # [batch_size, seq_len], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        prompt_lens: torch.Tensor | None = None,  # [batch_size], int64
        empty_slots: list[int] | None = None,
        # todo)) remove unnecessary sampling_params?
        sampling_params: SamplingParams | None = None,
        start_pos: torch.Tensor | None = None,  # [batch_size], int64
    ) -> torch.Tensor:  # [batch_size, 1, vocab_size], float32
        """Traced prefill: lazy capture on first call per seq_len, replay after.

        Args:
            tokens: Input token IDs, shape [batch_size, seq_len].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache from allocate_kv_cache().
            prompt_lens: Actual prompt length per user, shape [batch_size].
            empty_slots: List of user IDs to prefill.
            sampling_params: Sampling parameters (not used in prefill).
            start_pos: Starting position for prefix caching, shape [batch_size].

        Returns:
            Logits tensor, shape [batch_size, 1, vocab_size].
        """
        # Boundary assertions
        assert tokens.dim() == 2, f"tokens must be [batch_size, seq_len], got {tokens.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"
        if prompt_lens is not None:
            assert prompt_lens.dim() == 1, f"prompt_lens must be [batch_size], got {prompt_lens.dim()}D"
        if start_pos is not None:
            assert start_pos.dim() == 1, f"start_pos must be [batch_size], got {start_pos.dim()}D"

        self.mode = Mode.PREFILL
        self._eager._assert_kv_cache_identity(kv_cache)
        # A prefill breaks decode continuity: the next decode step must re-seed the persistent
        # on-device buffers from host (correct first token + start position) before the device
        # can carry state on its own. Mark all sampling modes stale, and drop the cached page
        # tables so the first post-prefill step always re-copies (device buffers are reused).
        self._decode_needs_reseed = defaultdict(lambda: True)
        self._prev_decode_page_table = defaultdict(lambda: None)

        batch_size, batch_seq_len = tokens.shape
        vocab_size = self.model.vocab_size
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]
        output_tensor = torch.zeros(batch_size, 1, vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)
        # todo)) empty_slots is only used when integrating with vLLM? If true, we should move this integration into generator.py
        if empty_slots is None:
            empty_slots = list(range(batch_size))

        prefill_results = []

        # Batched prefill fast path (traced): fuse equal-length users into batched passes, PER uniform-
        # length bucket group (see _plan_batched_prefill). Positions the plan leaves sequential (or all
        # positions when batching is off / declined) fall through to the per-user loop below.
        #
        # Per-user (sequential) prefill: each user is traced at its OWN get_padded_prefill_len bucket
        # (TTTv1-parity per-bucket TTFT). Multiple bucket traces (e.g. {128, 1024}) — single-user AND
        # batched — coexist correctly because every prefill trace's cos/sin input is the SAME shared
        # persistent tensor (_shared_prefill_cos_sin), never a fresh per-capture slice (a fresh slice
        # would be bottom-up-allocated into another coexisting trace's live residual buffer and corrupt
        # it — the ci-eval-32 mixed-bucket bug). No single-bucket collapse is needed.
        num_cached_per_user = [int(start_pos[i]) if start_pos is not None else 0 for i in range(len(empty_slots))]
        batched_groups, sequential_positions = _plan_batched_prefill(
            self.model_args, empty_slots, prompt_lens, num_cached_per_user
        )
        seq_filter = None
        if batched_groups:
            seq_filter = sequential_positions
            for prefill_seq_len_g, padded_batch_g, positions_g in batched_groups:
                logger.info(
                    f"Batched prefill (traced): {len(positions_g)} users in groups of {padded_batch_g} "
                    f"(seq={prefill_seq_len_g})"
                )
                prefill_results.extend(
                    self._prefill_forward_batched_group_traced(
                        tokens, page_table, prompt_lens, positions_g, padded_batch_g, prefill_seq_len_g
                    )
                )

        for idx, user_id in enumerate(empty_slots):
            if seq_filter is not None and idx not in seq_filter:
                continue  # handled by the batched path above
            seq_len = int(prompt_lens[idx])
            # todo)) prefix caching could be refactored into composable feature? --> shouldn't vLLM handle this not use? --> this could be a composable feature!
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            new_tokens = seq_len - num_cached_tokens
            if new_tokens <= 0:
                logger.info(
                    f"Skipping prefill for user_id={user_id}: seq_len={seq_len}, num_cached_tokens={num_cached_tokens}"
                )
                continue
            last_token_idx = seq_len - 1
            prefill_seq_len = get_padded_prefill_len(new_tokens)

            prefill_ids = torch.cat(
                [
                    tokens[idx : idx + 1, num_cached_tokens:seq_len],
                    torch.zeros(1, prefill_seq_len - new_tokens).long(),
                ],
                dim=-1,
            )

            can_trace = self.model_args and self.model_args.can_enable_trace(prefill_seq_len, num_cached_tokens)

            if can_trace:
                page_table_user = _get_prefill_trace_user_page_table(
                    page_table[idx : idx + 1],
                    kv_cache,
                    prefill_seq_len,
                )
            else:
                page_table_user = _get_prefill_user_page_table(
                    page_table[idx : idx + 1],
                    kv_cache,
                    seq_len,
                )

            # todo)) there should be warning message about prefill that cannot be traced!
            if can_trace:
                # todo)) what does it take to make the whole thing traceable?
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=0,
                    last_token_idx=last_token_idx,
                    prefill_seq_len=prefill_seq_len,
                )
                logits = self.model.post_process_prefill_output(logits, last_token_idx)
            else:
                logits = self._eager._prefill_single_user(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=0,  # Always 0 with paged attention
                    last_token_idx=last_token_idx,
                    num_cached_tokens=num_cached_tokens,
                )

            logits = ttnn.untilize(logits, use_multicore=True)
            # Single-user last-token slice ON DEVICE before readback (see fast_prefill_last_token):
            # only row (last_token_idx - num_cached)%32 of this [1,1,32,vocab] tile is consumed by the
            # assembly, so slicing it here shrinks the host readback + concat from the full tile to one
            # row. row_override tells the assembly the row already sits at index 0.
            row_override = None
            if self.fast_prefill_last_token and batch_size == 1:
                _row = (last_token_idx - num_cached_tokens) % 32
                logits = ttnn.slice(logits, (0, 0, _row, 0), (1, 1, _row + 1, logits.shape[-1]))
                row_override = 0
            prefill_results.append(
                {
                    "idx": idx,
                    "last_token_idx": last_token_idx,
                    "logits": logits.cpu(blocking=False),
                    "row_override": row_override,
                }
            )

        # One device barrier drains every pending ``logits.cpu(blocking=False)`` transfer dispatched
        # above (batched extraction + sequential loop); ``_process_output_prefill`` then runs on the
        # already-resident HOST tensors (it asserts HOST storage; no device work). One barrier suffices
        # (an idle ``synchronize_device`` is cheap), so the old per-user barrier was redundant — kept
        # hoisted purely as a cleanup. Batched extraction returns ONE shared host logits tensor for a
        # whole group (each user is a distinct row); concat its shards once per unique tensor and slice
        # each user's row instead of re-concatenating per user in ``_process_output_prefill`` (32 host
        # concats -> 1 on batch-32). On batch-32 this host work was ~52ms (fold-invariant), the largest
        # non-forward term in the TTTv2->TTTv1 batch-32 TTFT gap. Cache is keyed by tensor identity, so
        # the sequential path (a unique tensor per user) is unchanged (one concat each).
        if prefill_results:
            ttnn.synchronize_device(self.mesh_device)
        _concat_cache: dict = {}
        for res in prefill_results:
            key = id(res["logits"])
            full = _concat_cache.get(key)
            if full is None:
                assert res["logits"].storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
                full = _concat_host_output(res["logits"], cluster_shape)
                _concat_cache[key] = full
            last_relative = res["last_token_idx"] - (int(start_pos[res["idx"]]) if start_pos is not None else 0)
            row = res.get("row_override")
            output_tensor[res["idx"]] = full[0, 0, last_relative % 32 if row is None else row, :vocab_size]

        return output_tensor

    def _easy_trace_prefill(self, tokens, page_table, user_id, last_token_idx, prefill_seq_len):
        """Lazy trace capture for prefill. Captures on first call per seq_len."""
        if self.trace_id_prefill[prefill_seq_len] is None:
            return self._capture_and_run_prefill_trace(
                tokens,
                page_table,
                user_id,
                last_token_idx,
                prefill_seq_len,
            )

        host_inputs = self._prepare_prefill_trace_inputs_host(
            tokens,
            page_table=page_table,
            last_token_idx=last_token_idx,
        )
        trace_inputs = self.trace_inputs_prefill[prefill_seq_len]
        copy_host_to_device(
            host_tensors=(host_inputs[0], host_inputs[3], host_inputs[4]),
            device_tensors=(trace_inputs[0], trace_inputs[3], trace_inputs[4]),
        )

        ttnn.execute_trace(self.mesh_device, self.trace_id_prefill[prefill_seq_len], cq_id=0, blocking=False)
        return self.trace_output_prefill[prefill_seq_len]

    def _run_prefill_trace_body(self, device_inputs, user_id):
        tokens_embd = self.model.embed_prefill(device_inputs[0])
        rot_mats = [device_inputs[1], device_inputs[2]]
        tt_page_table = device_inputs[3]
        tt_chunk_page_table = device_inputs[4]

        return self.model.prefill_forward(
            tokens_embd,
            rot_mats,
            user_id=user_id,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            get_last_token=-1,
        )

    def _capture_and_run_prefill_trace(self, tokens, page_table, user_id, last_token_idx, prefill_seq_len):
        """Compile + capture trace for a specific prefill seq_len."""
        # todo)) should just assert the expected trace is already there (look up by prefill_seq_len)
        self._eager._prefill_single_user(
            tokens,
            page_table=page_table,
            user_id=user_id,
            last_token_idx=last_token_idx,
        )
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"Compiled prefill for seq_len={prefill_seq_len}")

        host_inputs = self._prepare_prefill_trace_inputs_host(
            tokens,
            page_table=page_table,
            last_token_idx=last_token_idx,
        )
        device_inputs = list(copy_host_to_device(host_inputs, mesh_device=self.mesh_device))
        # host_inputs[1],[2] are already the SHARED persistent cos/sin device tensors (from
        # _shared_prefill_cos_sin). copy_host_to_device is a no-op for already-on-device tensors, but
        # pin the shared object identity explicitly so every bucket's captured trace inputs reference
        # the SAME cos/sin buffer (never a fresh per-capture copy) — this is what prevents a second
        # bucket's cos/sin from being allocated into another bucket trace's live residual and
        # corrupting it. cos/sin are read-only in the trace and not re-copied on replay.
        device_inputs[1] = host_inputs[1]
        device_inputs[2] = host_inputs[2]
        device_inputs = tuple(device_inputs)

        with suspend_module_input_validation():
            trace_warmup_output = self._run_prefill_trace_body(device_inputs, user_id)
        ttnn.synchronize_device(self.mesh_device)
        cleanup_ttnn_value(trace_warmup_output)
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"Compiled trace-compatible prefill for seq_len={prefill_seq_len}")

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        with suspend_module_input_validation():
            logits = self._run_prefill_trace_body(device_inputs, user_id)

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

        self.trace_id_prefill[prefill_seq_len] = trace_id
        self.trace_inputs_prefill[prefill_seq_len] = device_inputs
        self.trace_output_prefill[prefill_seq_len] = logits

        logger.info(f"Captured prefill trace for seq_len={prefill_seq_len}")
        return logits

    # =========================================================================
    # Batched Prefill (traced, partial B=padded_batch)
    # =========================================================================

    def _prepare_batched_prefill_trace_inputs_host(self, folded_tokens, group_page_table):
        """Host-side inputs for a batched-prefill trace.

        Tokens and page table are host tensors (copied in per group on replay); cos/sin are the SHARED
        persistent full-range ``[0:max_seq_len]`` rope tensors (``_shared_prefill_cos_sin``), the SAME
        object every single-user AND batched trace uses. The rotary op accepts cos whose seq dim is
        longer than q's and broadcasts cos's batch dim (==1) over the unfolded [B,nh,S,hd] q
        (device-verified: rotary_embedding_llama prefill validate has no seq-dim check and requires
        cos.shape[0]==1). Reusing one buffer means NO fresh per-capture cos/sin slice can be
        bottom-up-allocated into a coexisting trace's live range — the Phase-1 mixed-bucket fix holds
        under batching. cos/sin are read-only in the trace body and never re-copied on replay.
        """
        tokens_reshaped = folded_tokens.reshape(1, 1, 1, -1)
        tokens_tt = ttnn.from_torch(
            tokens_reshaped,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        rope = self.model.rope_setup
        rope.load_device_weights()
        max_seq_len = self.model_args.max_seq_len if self.model_args else rope.cos_matrix.shape[2]
        cos_slice, sin_slice = self._shared_prefill_cos_sin(rope, max_seq_len)
        page_table_tt = ttnn.from_torch(
            group_page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tokens_tt, cos_slice, sin_slice, page_table_tt

    def _run_batched_prefill_trace_body(self, device_inputs, padded_batch, num_real):
        tokens_embd = self.model.embed_prefill(device_inputs[0])
        rot_mats = [device_inputs[1], device_inputs[2]]
        tt_page_table = device_inputs[3]
        return self.model.prefill_forward(
            tokens_embd,
            rot_mats,
            user_id=list(range(num_real)),
            page_table=tt_page_table,
            get_last_token=-1,
            batch_size=padded_batch,
        )

    def _easy_trace_batched_prefill(self, folded, group_pt, padded_batch, prefill_seq_len, num_real):
        """Lazy capture/replay of the batched-prefill trace keyed by (prefill_seq_len, padded_batch)."""
        key = (prefill_seq_len, padded_batch)
        if self.trace_id_prefill_batched[key] is None:
            return self._capture_and_run_batched_prefill_trace(
                folded, group_pt, padded_batch, prefill_seq_len, num_real
            )

        host_inputs = self._prepare_batched_prefill_trace_inputs_host(folded, group_pt)
        trace_inputs = self.trace_inputs_prefill_batched[key]
        # Only tokens (idx 0) and page table (idx 3) vary per group; cos/sin (idx 1,2) are the shared
        # persistent tensors, baked at capture and never re-copied.
        copy_host_to_device(
            host_tensors=(host_inputs[0], host_inputs[3]),
            device_tensors=(trace_inputs[0], trace_inputs[3]),
        )
        ttnn.execute_trace(self.mesh_device, self.trace_id_prefill_batched[key], cq_id=0, blocking=False)
        return self.trace_output_prefill_batched[key]

    def _capture_and_run_batched_prefill_trace(self, folded, group_pt, padded_batch, prefill_seq_len, num_real):
        """Compile + capture a batched-prefill trace for (prefill_seq_len, padded_batch)."""
        key = (prefill_seq_len, padded_batch)
        # Eager warmup (JIT-compile the batched ops outside trace capture). Throwaway cos/sin here.
        tokens_embd, cos, sin, pt_tt = self._eager._prepare_batched_prefill_device_inputs(
            folded, group_pt, prefill_seq_len
        )
        with suspend_module_input_validation():
            self.model.prefill_forward(
                tokens_embd,
                [cos, sin],
                user_id=list(range(num_real)),
                page_table=pt_tt,
                get_last_token=-1,
                batch_size=padded_batch,
            )
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"Compiled batched prefill for seq_len={prefill_seq_len}, padded_batch={padded_batch}")

        host_inputs = self._prepare_batched_prefill_trace_inputs_host(folded, group_pt)
        device_inputs = list(copy_host_to_device(host_inputs, mesh_device=self.mesh_device))
        # host_inputs[1],[2] are the SHARED persistent cos/sin device tensors (from _shared_prefill_cos_sin);
        # copy_host_to_device is a no-op for already-on-device tensors, but pin the shared object identity
        # so every batched trace references the SAME cos/sin buffer (never a fresh per-capture copy). This
        # is what keeps the Phase-1 mixed-bucket aliasing fix intact across coexisting batched traces.
        device_inputs[1] = host_inputs[1]
        device_inputs[2] = host_inputs[2]
        device_inputs = tuple(device_inputs)

        with suspend_module_input_validation():
            trace_warmup_output = self._run_batched_prefill_trace_body(device_inputs, padded_batch, num_real)
        ttnn.synchronize_device(self.mesh_device)
        cleanup_ttnn_value(trace_warmup_output)
        ttnn.synchronize_device(self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        with suspend_module_input_validation():
            hidden = self._run_batched_prefill_trace_body(device_inputs, padded_batch, num_real)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

        self.trace_id_prefill_batched[key] = trace_id
        self.trace_inputs_prefill_batched[key] = device_inputs
        self.trace_output_prefill_batched[key] = hidden

        logger.info(f"Captured batched prefill trace for seq_len={prefill_seq_len}, padded_batch={padded_batch}")
        return hidden

    def _prefill_forward_batched_group_traced(
        self, tokens, page_table, prompt_lens, positions, padded_batch, prefill_seq_len
    ):
        """Traced batched prefill for ONE uniform-length bucket group: one batched (traced when
        allowed) pass per sub-group of padded_batch users, then last-token extraction. Only full
        sub-groups are traced; a partial trailing sub-group runs eager (it would bake a different
        KV-fill loop length into the trace). Returns a list of per-user result dicts."""
        block_size = get_block_size(self._kv_cache) if self._kv_cache is not None else 32
        can_trace = bool(self.model_args and self.model_args.can_enable_trace(prefill_seq_len, 0))
        prefill_results = []

        for g0 in range(0, len(positions), padded_batch):
            sub_positions = positions[g0 : g0 + padded_batch]
            folded, group_pt, group_idxs, last_token_idxs = _build_batched_prefill_group(
                tokens, prompt_lens, page_table, sub_positions, padded_batch, prefill_seq_len, block_size
            )
            if can_trace and len(group_idxs) == padded_batch:
                hidden = self._easy_trace_batched_prefill(
                    folded, group_pt, padded_batch, prefill_seq_len, num_real=len(group_idxs)
                )
            else:
                tokens_embd, cos, sin, pt_tt = self._eager._prepare_batched_prefill_device_inputs(
                    folded, group_pt, prefill_seq_len
                )
                hidden = self.model.prefill_forward(
                    tokens_embd,
                    [cos, sin],
                    user_id=list(range(len(group_idxs))),
                    page_table=pt_tt,
                    get_last_token=-1,
                    batch_size=padded_batch,
                )
            prefill_results.extend(
                self._eager._extract_batched_prefill_logits(
                    hidden, padded_batch, prefill_seq_len, group_idxs, last_token_idxs
                )
            )
        return prefill_results

    # =========================================================================
    # Decode (traced)
    # =========================================================================

    def decode_forward(
        self,
        tokens: torch.Tensor,  # [batch_size], int64
        start_pos: torch.Tensor,  # [batch_size], int64
        page_table: torch.Tensor,  # [batch_size, max_blocks], int32
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        read_from_device: bool = True,
        sampling_params: SamplingParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Traced decode: lazy capture on first call, replay after.

        Args:
            tokens: Input token IDs, shape [batch_size].
            start_pos: Current position per user, shape [batch_size].
            page_table: Page table for paged attention, shape [batch_size, max_blocks]. Required.
            kv_cache: Per-layer KV cache (for identity assertion).
            read_from_device: Whether to return host tensors.
            sampling_params: Sampling parameters for on-device sampling.

        Returns:
            (logits_or_tokens, log_probs) tuple.
            If sampling_params is None: logits [batch_size, 1, vocab_size], None
            If sampling_params provided: tokens [batch_size], None
        """
        # Boundary assertions
        assert tokens.dim() == 1, f"tokens must be [batch_size], got {tokens.dim()}D"
        assert start_pos.dim() == 1, f"start_pos must be [batch_size], got {start_pos.dim()}D"
        assert page_table.dim() == 2, f"page_table must be [batch_size, max_blocks], got {page_table.dim()}D"

        self.mode = Mode.DECODE
        self._eager._assert_kv_cache_identity(kv_cache)
        sampling_on_device = sampling_params is not None
        B = tokens.shape[0]
        vocab_size = self.model.vocab_size
        num_devices = self.model.num_devices
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]

        if (
            sampling_on_device
            and self.model.sampling is not None
            and hasattr(self.model.sampling, "apply_decode_state")
        ):
            # TTTv1-only: push per-request k/p/temp into persistent device buffers before replay.
            # TTTv2 Sampling1D has no such state (greedy/argmax needs none) — skip when absent.
            from models.common.sampling import broadcast_sampling_params, format_sampling_params

            per_request_params = format_sampling_params(broadcast_sampling_params(sampling_params, 0, slot_len=B), B)
            self.model.sampling.apply_decode_state(
                [per_request_params],
                reset_batch=False,
            )
            self.model.sampling.seed_manager.get_new_values()

        if not self.trace_ids_decode[sampling_on_device]:
            self._capture_decode_trace(tokens, start_pos, page_table, kv_cache, sampling_on_device, sampling_params)

        loop_active = self._decode_loop_active(sampling_params)
        if loop_active and not self._decode_needs_reseed[sampling_on_device]:
            # Steady-state on-device loop: the captured trace advanced current_pos/rot_mat_idxs
            # (in-place plus_one) and wrote the sampled token back into the persistent token buffer
            # on the previous replay, so tokens/positions need NO host staging. Refresh only the
            # page table, and only when it actually changed (block boundaries crossed).
            prev_page_table = self._prev_decode_page_table[sampling_on_device]
            if prev_page_table is None or not torch.equal(prev_page_table, page_table):
                host_inputs = self._eager.prepare_decode_inputs_host(tokens, start_pos, page_table)
                device_inputs = self.trace_inputs_decode[sampling_on_device]
                ttnn.copy_host_to_device_tensor(host_inputs[3], device_inputs[3])  # page_table only
                self._prev_decode_page_table[sampling_on_device] = page_table.clone()
        else:
            # Reseed (first step after prefill, or loop disabled): full host refresh of all inputs.
            host_inputs = self._eager.prepare_decode_inputs_host(tokens, start_pos, page_table)
            copy_host_to_device(
                host_tensors=host_inputs,
                device_tensors=self.trace_inputs_decode[sampling_on_device],
            )
            self._prev_decode_page_table[sampling_on_device] = page_table.clone()
            if loop_active:
                self._decode_needs_reseed[sampling_on_device] = False

        ttnn.execute_trace(
            self.mesh_device,
            self.trace_ids_decode[sampling_on_device],
            cq_id=0,
            blocking=False,
        )
        tt_output = self.trace_output_decode[sampling_on_device]

        if read_from_device:
            if sampling_on_device:
                tt_toks, tt_log_probs = tt_output
                toks_host = tt_toks.cpu()
                return _process_output_decode_tokens(toks_host, B, cluster_shape), None
            else:
                logits, _ = tt_output
                logits_host = logits.cpu()
                return _process_output_decode(logits_host, B, vocab_size, num_devices, cluster_shape), None

        return tt_output

    def _prealloc_sampling_buffers(self, sampling_params) -> None:
        """Materialise the *persistent* on-device sampling buffers before ANY trace is captured.

        The k/p/temp param tensors and Sampling1D's device buffers were previously created lazily
        inside ``_capture_decode_trace`` — i.e. AFTER the prefill trace is already live. tt-metal
        flags this ("Allocating device buffers is unsafe due to the existence of an active trace",
        allocator.cpp) and the buffers can land where a decode-trace buffer overlaps them; on replay
        the trace clobbers the ``k`` tensor, the sampling writer reads a corrupt (huge) k and walks
        its top-p loop out of L1 -> hard device hang (seen on perf batch-32 + perf_decode_tuning
        "LoFi" on N300; LoFi shifts the decode-trace layout so the overlap lands on ``k``).
        Allocating them here, with no trace active, gives them stable addresses that the prefill and
        decode traces reserve around. Idempotent: the later capture-time calls reuse these caches.
        """
        if sampling_params is None or self.model.sampling is None:
            return
        # k/p/temp param tensors (cached on the eager engine; returns None for force-argmax models).
        self._eager._get_decode_sampling_kpt(sampling_params)
        # Sampling1D persistent device buffers (local indices / offsets / seeds / user ids).
        if hasattr(self.model.sampling, "load_device_buffers"):
            self.model.sampling.load_device_buffers()
        logger.info("Pre-allocated on-device sampling buffers before trace capture")

    def _capture_decode_trace(self, tokens, start_pos, page_table, kv_cache, sampling_on_device, sampling_params=None):
        """Compile + capture decode trace."""
        self._eager.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=False,
            sampling_params=None,
        )
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Compiled decode")

        host_inputs = self._eager.prepare_decode_inputs_host(tokens, start_pos, page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

        # On-device decode loop: when active, the sampled token is fed back in place into the
        # persistent token buffer (device_inputs[0]) and position/rope are advanced on device.
        loop_active = self._decode_loop_active(sampling_params)

        # On-device sampling: materialise sampling buffers AND JIT-compile the sampling program
        # BEFORE trace capture. Both buffer materialisation (Sampling1D LazyBuffers + the
        # LogProbsCalculator's ttnn.as_tensor scratch) and a first-run program compile issue device
        # writes, which are illegal inside begin/end_trace_capture
        # ("TT_FATAL: Writes are not supported during trace capture"). The eager warmup above runs
        # the non-sampling path (sampling_params=None), so the argmax/all-gather sampling op is not
        # yet compiled — warm it up here against real device logits.
        if (
            sampling_on_device
            and self.model.sampling is not None
            and hasattr(self.model.sampling, "load_device_buffers")
        ):
            self.model.sampling.load_device_buffers()
            ttnn.synchronize_device(self.mesh_device)
            with suspend_module_input_validation():
                wu_tokens, wu_current_pos, wu_rot_mat_idxs, wu_page_table = device_inputs
                wu_rot_mats = self.model.rope_setup.get_rot_mats(wu_rot_mat_idxs)
                wu_embed = self.model.embed_decode(wu_tokens)
                wu_logits = self.model.decode_forward(wu_embed, wu_current_pos, wu_rot_mats, page_table=wu_page_table)
                # Warm up with the SAME output_tensor the captured body uses, so the
                # ttnn.sampling(output_tensor=...) program variant is compiled before capture
                # (avoids a "compile during trace capture" fatal).
                wu_feedback = wu_tokens if loop_active else None
                wu_toks, _ = self._eager._sampling_decode_forward(wu_logits, sampling_params, tt_out_tok=wu_feedback)
            ttnn.synchronize_device(self.mesh_device)
            # Only free wu_toks if it is a fresh allocation; when loop_active it aliases the
            # persistent token buffer (device_inputs[0]) and MUST NOT be deallocated.
            if wu_feedback is None:
                cleanup_ttnn_value(wu_toks)
            ttnn.synchronize_device(self.mesh_device)
            if loop_active:
                # Warm up BOTH ttnn.plus_one program variants (skip_negative_entries True/False are
                # distinct compile-time programs) so they are in the program cache before capture —
                # otherwise the in-trace plus_one triggers "Cannot load new binaries during trace
                # capture". These increment wu_current_pos/wu_rot_mat_idxs once; harmless, since the
                # first decode step re-seeds the persistent buffers from host.
                ttnn.plus_one(wu_current_pos, skip_negative_entries=True)
                ttnn.plus_one(wu_rot_mat_idxs)
                ttnn.synchronize_device(self.mesh_device)
            logger.info("Compiled on-device sampling")

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        with suspend_module_input_validation():
            tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_page_table = device_inputs
            rot_mats = self.model.rope_setup.get_rot_mats(tt_rot_mat_idxs)
            x_embed = self.model.embed_decode(tt_tokens)

            logits = self.model.decode_forward(
                x_embed,
                tt_current_pos,
                rot_mats,
                page_table=tt_page_table,
            )

            if sampling_on_device and self.model.sampling is not None:
                if loop_active:
                    # On-device loop: feed the sampled token back into the persistent token buffer
                    # (device_inputs[0]) in place. On the top-k path ttnn.sampling emits uint32
                    # [1,1,1,32] on Wormhole/Blackhole — an exact match for the token buffer, so
                    # output_tensor= writes in place with no realloc (safe during capture). Then
                    # advance position + rope index on device (in-place plus_one), so the NEXT
                    # execute_trace sees the new token at the new position with zero host staging.
                    # These writes are AFTER every read of the buffers in this body, so there is no
                    # intra-trace hazard. plus_one is a pure device op (in-place, traceable); the
                    # earlier "host-assisted writes illegal during capture" note was mistaken.
                    tt_toks, tt_log_probs = self._eager._sampling_decode_forward(
                        logits, sampling_params, tt_out_tok=tt_tokens
                    )
                    ttnn.plus_one(tt_current_pos, skip_negative_entries=True)
                    ttnn.plus_one(tt_rot_mat_idxs)
                else:
                    # tt_out_tok=None: sampling allocates its own output; decode_forward re-supplies
                    # current_pos/rot_mat_idxs from host every step, so no in-trace increment.
                    tt_toks, tt_log_probs = self._eager._sampling_decode_forward(
                        logits, sampling_params, tt_out_tok=None
                    )
                output = (tt_toks, tt_log_probs)
            else:
                logits = self.model.gather_and_untilize_logits(logits)
                output = (logits, None)

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

        self.trace_ids_decode[sampling_on_device] = trace_id
        self.trace_inputs_decode[sampling_on_device] = device_inputs
        self.trace_output_decode[sampling_on_device] = output

        logger.info("Captured decode trace")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Release all captured traces and their pinned device tensors."""
        if self._cleaned_up:
            return

        # Shared prefill cos/sin tensors are referenced by EVERY per-bucket trace-input tuple
        # (indices 1,2). Deallocate them exactly once below (not per trace) to avoid a double-free.
        shared_cos_sin_ids = {
            id(t) for pair in self._prefill_cos_sin_shared.values() for t in pair if isinstance(t, ttnn.Tensor)
        }
        for key, trace_id in list(self.trace_id_prefill.items()):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
            inputs = self.trace_inputs_prefill[key]
            if inputs is not None:
                for t in inputs:
                    if isinstance(t, ttnn.Tensor) and id(t) in shared_cos_sin_ids:
                        continue  # shared cos/sin: freed once, after this loop
                    cleanup_ttnn_value(t)
            cleanup_ttnn_value(self.trace_output_prefill[key])
            self.trace_id_prefill[key] = None
            self.trace_inputs_prefill[key] = None
            self.trace_output_prefill[key] = None
        # Batched prefill traces: same shared-cos/sin id-skip (indices 1,2 are the shared tensors).
        for key, trace_id in list(self.trace_id_prefill_batched.items()):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
            inputs = self.trace_inputs_prefill_batched[key]
            if inputs is not None:
                for t in inputs:
                    if isinstance(t, ttnn.Tensor) and id(t) in shared_cos_sin_ids:
                        continue  # shared cos/sin: freed once, below
                    cleanup_ttnn_value(t)
            cleanup_ttnn_value(self.trace_output_prefill_batched[key])
            self.trace_id_prefill_batched[key] = None
            self.trace_inputs_prefill_batched[key] = None
            self.trace_output_prefill_batched[key] = None
        # Free the shared cos/sin once.
        for pair in self._prefill_cos_sin_shared.values():
            cleanup_ttnn_value(pair)
        self._prefill_cos_sin_shared.clear()
        for key, trace_id in list(self.trace_ids_decode.items()):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
            cleanup_ttnn_value(self.trace_inputs_decode[key])
            cleanup_ttnn_value(self.trace_output_decode[key])
            self.trace_ids_decode[key] = None
            self.trace_inputs_decode[key] = None
            self.trace_output_decode[key] = None

        self._eager._kv_cache = None
        self._cleaned_up = True


def _get_validation_context(executor, *, mode: str):
    """Return the config-validation context for an executor or thin wrapper."""
    engine = getattr(executor, "_engine", executor)
    if isinstance(engine, TracedLLMExecutor):
        engine = engine._eager
    if isinstance(engine, EagerLLMExecutor):
        return validate_module_input_configs(
            model=engine.model,
            iter_named_modules=engine._iter_named_modules,
            mode=mode,
        )
    return contextlib.nullcontext()


def _is_traced_executor(executor) -> bool:
    """Identify executors that expose trace-facing APIs."""
    return hasattr(executor, "trace_id_prefill") and hasattr(executor, "trace_ids_decode")


def _compile_prefill_and_decode(
    executor: EagerLLMExecutor | TracedLLMExecutor,
    *,
    prefill_tokens: torch.Tensor,
    prefill_page_table: torch.Tensor,  # Required
    kv_cache: list[list[ttnn.Tensor]] | None = None,
    prompt_lens: torch.Tensor | None = None,
    empty_slots: list[int] | None = None,
    start_pos: torch.Tensor | None = None,
    sampling_params: SamplingParams | None = None,
    validate_configs: bool = False,
) -> None:
    """Internal convenience helper for one-shot prefill+decode warmup."""
    assert prefill_tokens.dim() == 2, f"prefill_tokens must be [batch_size, seq_len], got {prefill_tokens.dim()}D"
    assert (
        prefill_page_table.dim() == 2
    ), f"prefill_page_table must be [batch_size, max_blocks], got {prefill_page_table.dim()}D"

    # Capture the DECODE trace BEFORE the prefill trace. The batched-prefill trace's folded buffers are
    # ~padded_batch× the single-user footprint; when the prefill trace is captured FIRST and the decode
    # trace is captured immediately after (while the prefill trace is live), the decode capture lands in
    # a layout that overlaps the large batched-prefill trace and clobbers its replay — every batched user
    # then decodes garbage from the very first token (single-user prefill survives only by its far smaller
    # footprint). Capturing decode first — so its buffers are reserved before the large prefill trace is
    # built, and nothing is captured after prefill to overlap it — removes the overlap. Correctness is
    # unaffected by the swap: the decode trace is warmed with placeholder tokens, but decode
    # tokens/positions are refreshed from host (or fed back on device) on every replay, so warmup values
    # never reach the output; only the capture ORDER changes. (This also removes the reason the sampling
    # buffers had to be pre-materialised while the prefill trace was live — decode now captures with no
    # prefill trace active — while _prealloc_sampling_buffers in compile_prefill still guards the prefill
    # capture.)
    prefill_context = (
        _get_validation_context(executor, mode="prefill") if validate_configs else contextlib.nullcontext()
    )
    decode_context = _get_validation_context(executor, mode="decode") if validate_configs else contextlib.nullcontext()
    batch_size = prefill_tokens.shape[0]

    decode_start_pos = torch.full(
        (batch_size,),
        prefill_tokens.shape[-1],
        dtype=torch.long,
        device=prefill_tokens.device,
    )
    with decode_context:
        executor.compile_decode(
            tokens=torch.zeros(batch_size, dtype=torch.long, device=prefill_tokens.device),
            start_pos=decode_start_pos,
            page_table=prefill_page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    with prefill_context:
        executor.compile_prefill(
            tokens=prefill_tokens,
            page_table=prefill_page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )


# =============================================================================
# Loop Policy Functions
# =============================================================================


@dataclass
class TeacherForceResult:
    """Result from a teacher-forcing evaluation run."""

    predicted_tokens: list[int]
    predicted_tokens_per_user: list[list[int]]
    reference_top5: torch.Tensor  # shape [num_tokens, 5]
    # Timing — populated by run_teacher_forcing so the token-accuracy leg can emit the SAME
    # benchmark measurement set as TTTv1's ci-token-matching leg (prefill_t/s, decode_t/s,
    # decode_t/s/u, TTFT). The decode loop feeds ground-truth tokens (teacher forcing), but the
    # device ops — and therefore the measured durations — are identical to a free-running decode,
    # so these are real perf numbers for the accuracy path. Defaults keep the dataclass
    # backward-compatible for any caller that ignores timing. Mirrors PerfBenchmarkResult.
    prefill_time_s: float = 0.0
    compile_decode_time_s: float = 0.0
    decode_times_s: list[float] = field(default_factory=list)
    batch_size: int = 1
    prefill_len: int = 0

    def top1_accuracy(self) -> float:
        matches = sum(1 for i, p in enumerate(self.predicted_tokens) if self.reference_top5[i, 0].item() == p)
        return matches / len(self.predicted_tokens)

    def top5_accuracy(self) -> float:
        matches = sum(1 for i, p in enumerate(self.predicted_tokens) if p in self.reference_top5[i, :])
        return matches / len(self.predicted_tokens)

    @property
    def ttft_ms(self) -> float:
        """Average time-to-first-token per user (ms)."""
        return self.prefill_time_s / self.batch_size * 1000 if self.batch_size else 0.0

    @property
    def prefill_time_to_token_s(self) -> float:
        """Average time-to-first-token per user (seconds) — TTTv1 ``prefill_time_to_token``."""
        return self.prefill_time_s / self.batch_size if self.batch_size else 0.0

    @property
    def prefill_tok_s(self) -> float:
        """Prefill throughput (tokens/s) over all users."""
        return (self.batch_size * self.prefill_len) / self.prefill_time_s if self.prefill_time_s > 0 else 0.0

    @property
    def decode_tok_s_u(self) -> float:
        """Steady-state decode tokens/s/user (compile/warmup step excluded)."""
        return len(self.decode_times_s) / sum(self.decode_times_s) if self.decode_times_s else 0.0

    @property
    def decode_tok_s(self) -> float:
        """Steady-state decode throughput (tokens/s) over all users."""
        return self.decode_tok_s_u * self.batch_size


def run_teacher_forcing(
    executor: EagerLLMExecutor | TracedLLMExecutor,
    *,
    prompt_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor,  # Required
    max_batch_size: int = 1,
    profiler: BenchmarkProfiler | None = None,
) -> TeacherForceResult:
    """Run teacher-forcing accuracy measurement.

    Teacher forcing feeds ground truth tokens at each step and measures
    prediction accuracy. This is the canonical accuracy measurement for LLMs.

    The prefill and per-step decode regions are timed (compile is excluded — it runs in a
    separate warmup pass, and the first timed decode step is treated as post-compile warmup like
    ``run_perf_benchmark``), so the returned ``TeacherForceResult`` also carries prefill/decode
    throughput. This lets the token-accuracy leg emit the same benchmark measurements as TTTv1's
    ci-token-matching leg. When ``profiler`` is supplied, the timed regions are also bracketed
    with ``inference_prefill`` / ``inference_decode`` steps for CI benchmark-data emission.

    Args:
        executor: Any executor with prefill_forward() and decode_forward() methods.
        prompt_tokens: Prompt token IDs, shape [batch_size, prompt_len].
        reference_tokens: Full reference sequence (prompt + target), shape [total_len].
        top5_tokens: Top-5 reference tokens per position, shape [num_target_tokens, 5].
        kv_cache: Per-layer KV cache from allocate_kv_cache().
        page_table: Page table for paged attention. Required.
        max_batch_size: Maximum batch size. Must match prompt_tokens.shape[0].
        profiler: Optional ``BenchmarkProfiler``; when supplied, brackets the timed prefill /
            decode regions ("inference_prefill" / "inference_decode"). Default ``None`` ⇒
            byte-inert for callers that don't emit benchmark data.

    Returns:
        TeacherForceResult with predicted tokens, accuracy metrics, and prefill/decode timings.
    """
    batch_size = prompt_tokens.shape[0]
    assert (
        batch_size == max_batch_size
    ), f"Teacher forcing expects active batch to match max_batch_size, got {batch_size} vs {max_batch_size}"
    prompt_len = prompt_tokens.shape[-1]
    total_len = len(reference_tokens)
    num_target = total_len - prompt_len

    # Compile prefill + decode with config validation
    _compile_prefill_and_decode(
        executor,
        prefill_tokens=prompt_tokens,
        prefill_page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=torch.tensor([prompt_len] * batch_size),
        empty_slots=list(range(batch_size)),
        validate_configs=True,
    )

    logger.info(f"Teacher forcing: prefilling {prompt_len} tokens with batch={batch_size}")
    prefill_kwargs = dict(
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=torch.tensor([prompt_len] * batch_size),
        empty_slots=list(range(batch_size)),
    )
    # Inference prefill: timed run with ops already compiled (this is TTFT). Mirror
    # run_perf_benchmark: synchronize inside the timed region for an accurate prefill duration.
    if profiler is not None:
        profiler.start("inference_prefill")
    t0 = time.perf_counter()
    prefill_output = executor.prefill_forward(prompt_tokens, **prefill_kwargs)
    if hasattr(executor, "mesh_device"):
        ttnn.synchronize_device(executor.mesh_device)
    prefill_time_s = time.perf_counter() - t0
    if profiler is not None:
        profiler.end("inference_prefill")

    first_tokens = torch.argmax(prefill_output, dim=-1).view(-1).tolist()
    predicted_tokens_per_user = [[int(tok)] for tok in first_tokens]

    logger.info(f"Teacher forcing: decoding {num_target - 1} tokens")
    decode_times_s: list[float] = []
    compile_decode_time_s = 0.0
    if profiler is not None:
        profiler.start("inference_decode")
    for step in range(1, num_target):
        gt_token = reference_tokens[prompt_len + step - 1]
        decode_token = torch.full((batch_size,), gt_token, dtype=torch.long)

        current_pos = torch.full((batch_size,), prompt_len + step - 1, dtype=torch.long)

        decode_kwargs = dict(
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=True,
        )
        t0 = time.perf_counter()
        logits, _ = executor.decode_forward(decode_token, current_pos, **decode_kwargs)
        elapsed = time.perf_counter() - t0
        # First timed decode step is post-compile warmup (excluded from the steady-state average),
        # mirroring run_perf_benchmark / TTTv1's iteration-0 handling.
        if step == 1:
            compile_decode_time_s = elapsed
        else:
            decode_times_s.append(elapsed)

        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).view(-1).tolist()
        for user_id, tok in enumerate(next_tokens):
            predicted_tokens_per_user[user_id].append(int(tok))
    if profiler is not None:
        profiler.end("inference_decode")

    return TeacherForceResult(
        predicted_tokens=predicted_tokens_per_user[0],
        predicted_tokens_per_user=predicted_tokens_per_user,
        reference_top5=top5_tokens[:num_target],
        prefill_time_s=prefill_time_s,
        compile_decode_time_s=compile_decode_time_s,
        decode_times_s=decode_times_s,
        batch_size=batch_size,
        prefill_len=prompt_len,
    )


@dataclass
class PerfBenchmarkResult:
    """Result from a performance benchmark run."""

    prefill_time_s: float
    compile_decode_time_s: float
    decode_times_s: list[float]
    batch_size: int
    num_decode_tokens: int
    generated_token_ids: list[list[int]]

    @property
    def ttft_ms(self) -> float:
        """Average time-to-first-token per user (ms)."""
        return self.prefill_time_s / self.batch_size * 1000

    @property
    def tok_s_u(self) -> float:
        """Tokens per second per user (steady-state decode)."""
        if not self.decode_times_s:
            return 0.0
        return len(self.decode_times_s) / sum(self.decode_times_s)

    @property
    def tok_s(self) -> float:
        """Total throughput."""
        return self.tok_s_u * self.batch_size

    @property
    def decode_latency_mean_ms(self) -> float:
        if not self.decode_times_s:
            return 0.0
        return (sum(self.decode_times_s) / len(self.decode_times_s)) * 1000

    def meets_target(self, expected: dict, tolerance: float = 0.05) -> dict[str, bool]:
        """Check against expected metrics. Returns {metric: passed}."""
        return {
            "tok_s_u": self.tok_s_u >= expected["tok_s_u"] * (1 - tolerance),
            "ttft_ms": self.ttft_ms <= expected["ttft_ms"] * (1 + tolerance),
        }


def run_perf_benchmark(
    executor: EagerLLMExecutor | TracedLLMExecutor,
    *,
    tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor,  # Required
    num_decode_tokens: int = 128,
    max_batch_size: int = 1,
    prompt_lens: torch.Tensor | None = None,
    start_pos: list[int] | None = None,
    sampling_params=None,
    pipeline_readback: bool = True,
    profiler: BenchmarkProfiler | None = None,
) -> PerfBenchmarkResult:
    """Run timed prefill + decode loop for performance measurement.

    Matches TTTv1 methodology: compile prefill is excluded from TTFT.
    Iteration 0 of decode is the compile iteration (timed separately).

    Args:
        executor: Any executor with prefill_forward() and decode_forward() methods.
        tokens: Input token IDs, shape [batch_size, seq_len].
        kv_cache: Per-layer KV cache from allocate_kv_cache().
        page_table: Page table for paged attention. Required.
        num_decode_tokens: Number of decode tokens to generate.
        max_batch_size: Maximum batch size.
        prompt_lens: Actual prompt length per user, shape [batch_size].
        start_pos: Starting position for prefix caching.
        sampling_params: Explicit sampling params (None = host argmax).
        pipeline_readback: Overlap each step's token readback with the next step's
            trace (host one step behind the device). Only engages when the executor's
            on-device decode loop is active on the top-k path; ignored otherwise. Set
            False to A/B against the blocking readback.
        profiler: Optional ``BenchmarkProfiler``. When supplied, the timed inference-prefill
            and inference-decode regions are bracketed with ``start``/``end`` steps
            ("inference_prefill" / "inference_decode") so callers can emit CI benchmark
            data. Default ``None`` ⇒ byte-inert for every other caller.

    Returns:
        PerfBenchmarkResult with raw timings + derived metrics.
    """
    assert _is_traced_executor(executor), "run_perf_benchmark() expects a traced executor during this transition"

    batch_size = tokens.shape[0]
    prompt_len = tokens.shape[1]
    max_batch_size = max(max_batch_size, batch_size)
    # todo)) prompt_lens should be always passed in!
    prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([prompt_len] * batch_size)

    prefill_kwargs = dict(
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        empty_slots=list(range(batch_size)),
        start_pos=start_pos,
    )

    compile_batch_size = max_batch_size
    compile_tokens = torch.zeros(compile_batch_size, prompt_len, dtype=tokens.dtype)
    compile_tokens[:batch_size] = tokens
    compile_prompt_lens = torch.zeros(compile_batch_size, dtype=prompt_lens.dtype)
    compile_prompt_lens[:batch_size] = prompt_lens

    # Compile prefill + decode: warmup run with config validation (excluded from TTFT)
    _compile_prefill_and_decode(
        executor,
        prefill_tokens=compile_tokens,
        prefill_page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=compile_prompt_lens,
        empty_slots=list(range(batch_size)),
        start_pos=start_pos,
        sampling_params=sampling_params,
        validate_configs=True,
    )

    # Inference prefill: timed run with ops already compiled (this is TTFT)
    if profiler is not None:
        profiler.start("inference_prefill")
    t0 = time.perf_counter()
    prefill_output = executor.prefill_forward(tokens, **prefill_kwargs, sampling_params=sampling_params)
    if hasattr(executor, "mesh_device"):
        ttnn.synchronize_device(executor.mesh_device)
    prefill_time = time.perf_counter() - t0
    if profiler is not None:
        profiler.end("inference_prefill")

    if isinstance(prefill_output, tuple):
        first_token = prefill_output[0]
    else:
        first_token = torch.argmax(prefill_output, dim=-1)
    first_token = first_token.view(-1)[:batch_size].detach().cpu()
    generated_token_ids = [[int(tok)] for tok in first_token.tolist()]

    current_tokens = torch.zeros(max_batch_size, dtype=torch.long)
    current_tokens[:batch_size] = first_token

    current_pos = torch.full((max_batch_size,), -1, dtype=torch.long)
    current_pos[:batch_size] = prompt_lens[:batch_size]

    compile_time = None
    decode_times = []

    # Pipelined non-blocking readback (on-device decode loop, top-k path only).
    # The captured trace feeds the sampled token back on device (ttnn.sampling
    # output_tensor=) and advances position/rope on device (ttnn.plus_one), so step
    # N+1's replay does NOT depend on step N's token reaching the host. We therefore
    # issue each step's token readback non-blocking, launch the next step's trace,
    # and resolve step N's token one iteration later — the host stays exactly one
    # step behind the device. This mirrors the deferred-readback idiom in
    # models/demos/llama3_70b_galaxy/demo/demo_decode.py (``.cpu(blocking=False)`` +
    # ``record_event`` ... ``event_synchronize``) and the vLLM async-read path; note
    # the single-box text demo instead reads back blocking. Overlapping the readback +
    # host bookkeeping with device compute removes the per-step host round-trip that
    # otherwise serializes with the very short batch-1 device step. The token stream is
    # unchanged: the device produces the same tokens; we only read them back deferred,
    # then drain the last one so generated_token_ids is byte-identical to the blocking
    # loop. Pass pipeline_readback=False to A/B back to the blocking readback.
    # Every other path (host resupply, force-argmax, host sampling) is unaffected —
    # only _decode_loop_active (executor.ondevice_decode_loop + top-k) qualifies.
    _engine = getattr(executor, "_engine", executor)
    _pipeline_readback = (
        isinstance(_engine, TracedLLMExecutor) and _engine._decode_loop_active(sampling_params) and pipeline_readback
    )

    if profiler is not None:
        profiler.start("inference_decode")
    if _pipeline_readback:
        cluster_shape = _engine.model_args.cluster_shape if getattr(_engine, "model_args", None) else [1, 1]
        mesh_device = executor.mesh_device

        def _consume(host_tok):
            toks = _process_output_decode_tokens(host_tok, batch_size, cluster_shape)
            for user_id, tok in enumerate(toks.view(-1)[:batch_size].tolist()):
                generated_token_ids[user_id].append(int(tok))

        prev_event = None
        prev_host_tok = None
        for i in range(num_decode_tokens):
            t0 = time.perf_counter()
            # read_from_device=False: get the persistent device token buffer without
            # a blocking readback. current_tokens is intentionally not refreshed — the
            # device owns the token via in-trace feedback after the first (reseed)
            # step; the steady-state path ignores it (and a page-table refresh copies
            # only the page table, never the token buffer).
            tt_output = executor.decode_forward(
                current_tokens,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                read_from_device=False,
                sampling_params=sampling_params,
            )
            tt_toks = tt_output[0]
            host_tok = tt_toks.cpu(blocking=False)  # issue read; do not wait
            read_event = ttnn.record_event(mesh_device, 0)  # retires when this read lands
            if prev_event is not None:
                # Wait for the PREVIOUS step's read, which retired in the background
                # while this step's trace ran — this is the loop's device-paced beat.
                ttnn.event_synchronize(prev_event)
            elapsed = time.perf_counter() - t0

            if prev_host_tok is not None:
                _consume(prev_host_tok)
            prev_event, prev_host_tok = read_event, host_tok
            current_pos[:batch_size] += 1

            if i == 0:
                compile_time = elapsed
            else:
                decode_times.append(elapsed)

        # Drain the final in-flight token so the generated stream matches exactly.
        if prev_host_tok is not None:
            ttnn.event_synchronize(prev_event)
            _consume(prev_host_tok)
    else:
        for i in range(num_decode_tokens):
            t0 = time.perf_counter()
            logits, _ = executor.decode_forward(
                current_tokens,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                read_from_device=True,
                sampling_params=sampling_params,
            )
            elapsed = time.perf_counter() - t0

            if i == 0:
                compile_time = elapsed
            else:
                decode_times.append(elapsed)

            if isinstance(logits, torch.Tensor) and logits.dim() >= 2:
                next_tok = torch.argmax(logits[:, -1, :], dim=-1)
            else:
                next_tok = logits
            next_tok = next_tok.view(-1)[:batch_size].detach().cpu()
            for user_id, tok in enumerate(next_tok.tolist()):
                generated_token_ids[user_id].append(int(tok))
            current_tokens[:batch_size] = next_tok
            current_pos[:batch_size] += 1

    if profiler is not None:
        profiler.end("inference_decode")

    return PerfBenchmarkResult(
        prefill_time_s=prefill_time,
        compile_decode_time_s=compile_time or 0.0,
        decode_times_s=decode_times,
        batch_size=batch_size,
        num_decode_tokens=num_decode_tokens,
        generated_token_ids=generated_token_ids,
    )


# =============================================================================
# ci-eval-32: 32-user cross-batch determinism (self-consistency) helpers
# =============================================================================
#
# TTTv2 equivalent of the TTTv1 ``ci-eval-32`` case: run the batch-32 prefill+decode
# loop ``repeat_batches`` times, rotating the prompt->slot assignment by one each
# repeat, and assert that undoing the rotation lines up per-user outputs. There is no
# external golden — the target is the model's own output under prompt rotation.
#
# These helpers operate on the per-user token-id streams already returned by
# ``run_perf_benchmark`` (``PerfBenchmarkResult.generated_token_ids``) — no new device
# plumbing. Comparing token-id lists (not decoded text) is stricter than TTTv1's
# decoded-string compare and avoids tokenizer-decode ambiguity.
#
# Prompt parity with TTTv1: the demos feed ``eval_repeat_prompts_batch32.json`` — the same
# numeric sequence-continuation prompts TTTv1's ci-eval-32 case uses (simple_text_demo.py).
# Caveat: the self-consistency assert requires bit-exact reproduction of a prompt's greedy
# output across batch slots. With the CI default (host argmax) decoding is slot-invariant and
# deterministic, so the case passes and is gated in CI with no xfail. The residual risk is
# on-device sampling on DEGENERATE repetitive output (a small model looping on one token): such
# output sits on argmax near-ties whose resolution can shift with batch position (batched-matmul
# / all-gather FP non-associativity) and would not be slot-invariant. This has NOT been observed
# for TTTv2-1B — the case is green on N300 under both host and on_device_topk (batched prefill on
# and off). If a future model's prompts degenerate under on-device sampling, prefer the host-argmax
# default for this case (or add per-model prompts) rather than treating it as a harness bug.


def load_eval_repeat_prompts_batch32() -> list[str]:
    """The 32 numeric sequence-continuation prompts TTTv1's ci-eval-32 uses (parity)."""
    path = Path("models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch32.json")
    with open(path) as f:
        data = json.load(f)
    return [entry["prompt"] for entry in data]


def rotate_prompts(all_prompts: list[str], repeat: int) -> list[str]:
    """Rotate the prompt->slot assignment by ``repeat``: slot j holds prompt (j+repeat)%N."""
    n = len(all_prompts)
    return [all_prompts[(j + repeat) % n] for j in range(n)]


def truncate_at_stop(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Prefix of ``ids`` up to (excluding) the first id in ``stop_ids``."""
    out: list[int] = []
    for t in ids:
        if t in stop_ids:
            break
        out.append(t)
    return out


def hf_stop_ids(tokenizer, hf_model_id: str | None = None) -> set[int]:
    """Best-effort stop-token id set for an HF ``AutoTokenizer``.

    Raw HF tokenizers have no ``.stop_tokens`` (that only exists on the TTTv1 wrapped
    tokenizer). Build the set from ``eos_token_id`` (int|list|None), and — when an
    ``hf_model_id`` is supplied — also fold in the model's ``generation_config`` eos ids,
    since chat models (e.g. Llama-3 Instruct) often carry extra eot ids there rather than
    on ``eos_token_id``. Missing/empty -> empty set (truncation simply runs full length).
    """
    stop: set[int] = set()

    def _add(value) -> None:
        if value is None:
            return
        if isinstance(value, bool):  # guard: bool is an int subclass
            return
        if isinstance(value, int):
            stop.add(int(value))
        elif isinstance(value, (list, tuple, set)):
            for e in value:
                _add(e)

    _add(getattr(tokenizer, "eos_token_id", None))
    # tt_transformers ModelArgs.tokenizer augments the HF tokenizer with ``stop_tokens``
    # (eos + any extra eot ids); raw HF AutoTokenizers don't have it (getattr -> None).
    _add(getattr(tokenizer, "stop_tokens", None))
    if hf_model_id is not None:
        try:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig.from_pretrained(hf_model_id)
            _add(getattr(gen_cfg, "eos_token_id", None))
        except Exception as e:  # generation_config absent / unreadable — eos_token_id is enough
            logger.debug(f"ci-eval-32: could not read generation_config eos ids for {hf_model_id}: {e}")
    return stop


def assert_cross_batch_consistency(per_repeat_outputs: list[list[list[int]]]) -> None:
    """Assert prompt-position invariance across repeats.

    ``per_repeat_outputs[b][u]`` = truncated token-id list for slot ``u`` of repeat ``b``.
    With slot j of repeat b holding prompt (j+b)%N (see ``rotate_prompts``), the same prompt
    sits at slot (offset+1)%N of repeat b and slot offset of repeat b+1 — so those two
    outputs must be identical if no per-user state leaks.
    """
    num_batches = len(per_repeat_outputs)
    assert num_batches >= 2, "cross-batch consistency needs >=2 repeats"
    n = len(per_repeat_outputs[0])
    failed, total = 0, 0
    first_failure = None
    for b in range(num_batches - 1):
        cur, nxt = per_repeat_outputs[b], per_repeat_outputs[b + 1]
        for offset in range(n):
            total += 1
            if cur[(offset + 1) % n] != nxt[offset]:
                failed += 1
                if first_failure is None:
                    first_failure = (b, offset)
    assert failed == 0, (
        f"ci-eval-32: {failed}/{total} cross-batch consistency checks failed "
        f"(first at repeat {first_failure[0]}->{first_failure[0] + 1}, offset {first_failure[1]})"
    )


def run_eval_repeat_batch32(
    *,
    make_executor,
    allocate_kv_cache,
    page_table: torch.Tensor,
    prompts: list[str],
    tokenizer,
    tokenize_fn,
    num_decode_tokens: int,
    max_batch_size: int,
    sampling_params=None,
    repeat_batches: int = 3,
    hf_model_id: str | None = None,
) -> None:
    """Drive the ci-eval-32 determinism case, building a fresh traced executor per repeat.

    Each repeat builds its own traced executor (``make_executor()``) and its own zeroed KV
    cache (``allocate_kv_cache(executor)``), so the rotated batches are fully independent —
    no shared device or host state can leak across repeats. The executor is cleaned up after
    each repeat. (The model is bit-deterministic across repeats either way; fresh-per-repeat
    is simply the cleanest independence guarantee for a determinism test, and the trace
    recapture cost is negligible at batch-32.)

    Args:
        make_executor: Zero-arg callable returning a fresh traced executor
            (``run_perf_benchmark`` requires traced). Called once per repeat.
        allocate_kv_cache: Callable(executor) -> fresh zeroed kv_cache bound on that executor.
        page_table: Fixed contiguous page table (shared across repeats).
        prompts: The N (=max_batch_size) prompts to rotate (TTTv1 ci-eval-32 numeric prompts;
            see the module note above re: degenerate-output sensitivity on small models).
        tokenizer: HF tokenizer (for stop / special ids).
        tokenize_fn: Callable(list[str]) -> (tokens, prompt_lens).
        num_decode_tokens: Decode steps per repeat.
        max_batch_size: Padded batch (== len(prompts) for this fixed-32 case).
        sampling_params: None -> host argmax (deterministic, mesh-agnostic default).
        repeat_batches: Number of rotated repeats (TTTv1 uses 3).
        hf_model_id: Optional, to enrich stop ids from generation_config.
    """
    assert (
        len(prompts) == max_batch_size
    ), f"ci-eval-32 expects len(prompts)==max_batch_size; got {len(prompts)} vs {max_batch_size}"
    stop_ids = hf_stop_ids(tokenizer, hf_model_id)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    # Garbage guard targets only special tokens that are NOT recognized stops: a legitimate
    # stop is removed by truncation, so anything special left in the body is degenerate output.
    garbage_ids = special_ids - stop_ids
    logger.info(
        f"ci-eval-32: repeat_batches={repeat_batches}, N={len(prompts)}, "
        f"stop_ids={sorted(stop_ids)}, |special_ids|={len(special_ids)}, sampling_params={sampling_params}"
    )

    per_repeat: list[list[list[int]]] = []
    for i in range(repeat_batches):
        traced_executor = make_executor()
        try:
            kv_cache = allocate_kv_cache(traced_executor)
            rotated = rotate_prompts(prompts, i)
            tokens, prompt_lens = tokenize_fn(rotated)
            result = run_perf_benchmark(
                traced_executor,
                tokens=tokens,
                kv_cache=kv_cache,
                page_table=page_table,
                num_decode_tokens=num_decode_tokens,
                max_batch_size=max_batch_size,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
            )
        finally:
            traced_executor.cleanup()
        truncated = [truncate_at_stop(ids, stop_ids) for ids in result.generated_token_ids]
        for u, ids in enumerate(truncated):
            bad = set(ids) & garbage_ids
            assert not bad, f"ci-eval-32: user {u} produced special token(s) {sorted(bad)} mid-stream"
        per_repeat.append(truncated)
        logger.info(f"ci-eval-32 repeat {i}: truncated lengths = {[len(t) for t in truncated]}")

    assert_cross_batch_consistency(per_repeat)
    logger.info(f"ci-eval-32: all {(repeat_batches - 1) * len(prompts)} cross-batch consistency checks passed")


# =============================================================================
# Shared helpers
# =============================================================================


def _concat_host_output(tt_out, cluster_shape, is_galaxy=False):
    """Concatenate multi-device output into a single host tensor."""
    torch_out_tensors = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tt_out)]
    row_dim, col_dim = (1, -1)

    rows, cols = cluster_shape
    mesh_shape = [torch_out_tensors[i : i + cols] for i in range(0, len(torch_out_tensors), cols)]
    row_concatenated = [torch.cat(row, dim=col_dim) for row in mesh_shape]
    return torch.cat(row_concatenated, dim=row_dim)


def _process_output_prefill(tt_out, last_token_idx, vocab_size, cluster_shape):
    """Device→host for prefill. Returns logits for the last token."""
    assert tt_out.storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
    return _concat_host_output(tt_out, cluster_shape)[0, 0, last_token_idx, :vocab_size]


def _process_output_decode(tt_out, B, vocab_size, num_devices, cluster_shape):
    """Device→host for decode. Returns logits [B, 1, vocab_size]."""
    if num_devices > 1:
        # Decode logits are vocab-sharded across devices; stitch shards back before argmax.
        tt_out = _concat_host_output(tt_out, cluster_shape).float()
    else:
        tt_out = ttnn.to_torch(tt_out).float()
    return tt_out[:, :, :B, :vocab_size].contiguous().view(B, 1, -1)


def _process_output_decode_tokens(tt_out, B, cluster_shape):
    """Device→host for decode when sampling on device. Returns token ids [B]."""
    padded_batch_size = 32
    tt_out = ttnn.reshape(tt_out, ttnn.Shape([1, 1, padded_batch_size, 1]))
    return _concat_host_output(tt_out, cluster_shape)[0, 0, :B, 0]


def _get_prefill_user_page_table(page_table, kv_cache, prefill_len):
    """Slice and pad page table for a single prefill user."""
    block_size = get_block_size(kv_cache)
    num_blocks = num_blocks_in_seq(prefill_len, block_size)
    return page_table[:, :num_blocks]


def _get_prefill_trace_user_page_table(page_table, kv_cache, prefill_seq_len):
    """Slice page table for traced prefill's padded sequence length."""
    block_size = get_block_size(kv_cache)
    num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
    return page_table[:, :num_blocks]
