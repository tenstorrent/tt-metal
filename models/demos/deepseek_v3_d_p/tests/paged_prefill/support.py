# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test-only contracts and measurements for GLM-5.2 paged prefill.

Nothing in this module is a serving allocator.  ``ReferencePageAllocator`` is a
small executable specification used to pin down ownership and reuse semantics
before a production implementation is connected to these tests.
"""

from __future__ import annotations

import importlib
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

PREFILL_PAGE_TOKENS = 5 * 1024
KVPE_HEAD_DIM = 512 + 64
INDEX_HEAD_DIM = 128
GLM52_PRIMARY_LAYERS = 78
UNMAPPED_BUNDLE = -1
SPARSE_SENTINEL = 0xFFFFFFFF


def paged_kv_location(
    logical_token: int,
    physical_bundle: int,
    *,
    sp: int,
    layer: int,
    num_layers: int = GLM52_PRIMARY_LAYERS,
) -> tuple[int, int, int]:
    """Reference for the sparse reader's direct owner/pool-page mapping.

    Returns ``(owner_sp, folded_pool_batch, owner_local_row)``.  Bundle
    selection is intentionally external: it comes from the compact page table,
    while the 32-token physical subpages are deterministic row offsets.
    """

    if sp <= 0 or PREFILL_PAGE_TOKENS % sp:
        raise ValueError("SP must be a positive divisor of 5120")
    if logical_token < 0 or physical_bundle < 0:
        raise ValueError("token and physical bundle must be non-negative")
    if not 0 <= layer < num_layers:
        raise ValueError("layer outside folded pool")
    chunk_local = PREFILL_PAGE_TOKENS // sp
    in_bundle = logical_token % PREFILL_PAGE_TOKENS
    return (
        in_bundle // chunk_local,
        physical_bundle * num_layers + layer,
        in_bundle % chunk_local,
    )


def gather_paged_kv_reference(
    pool_shards: Sequence[torch.Tensor],
    page_table: torch.Tensor,
    indices: torch.Tensor,
    *,
    slot: int,
    layer: int,
    num_layers: int = GLM52_PRIMARY_LAYERS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather global top-k rows from fragmented SP-local paged pool shards.

    Sentinels are never dereferenced.  The returned boolean mask identifies
    real rows; sentinel output rows are zeros only to make comparisons simple.
    """

    sp = len(pool_shards)
    if sp == 0:
        raise ValueError("at least one SP shard is required")
    flat_indices = indices.to(torch.int64).reshape(-1)
    width = pool_shards[0].shape[-1]
    gathered = torch.zeros((flat_indices.numel(), width), dtype=pool_shards[0].dtype)
    valid = flat_indices != SPARSE_SENTINEL
    for output_row, logical_token in enumerate(flat_indices.tolist()):
        if logical_token == SPARSE_SENTINEL:
            continue
        logical_bundle = logical_token // PREFILL_PAGE_TOKENS
        if logical_bundle >= page_table.shape[1]:
            raise IndexError("logical token is outside the compact page table")
        physical_bundle = int(page_table[slot, logical_bundle])
        if physical_bundle == UNMAPPED_BUNDLE:
            raise RuntimeError("selected token names an unallocated bundle")
        owner, folded_batch, local_row = paged_kv_location(
            logical_token,
            physical_bundle,
            sp=sp,
            layer=layer,
            num_layers=num_layers,
        )
        gathered[output_row] = pool_shards[owner][folded_batch, 0, local_row]
    return gathered.reshape(*indices.shape, width), valid.reshape(indices.shape)


def folded_cache_slot(physical_page: int, layer_slot: int, num_layer_slots: int) -> int:
    """User/page-major folding used by ``init_kvpe_cache``."""
    if physical_page < 0 or not 0 <= layer_slot < num_layer_slots:
        raise ValueError("invalid physical page or layer slot")
    return physical_page * num_layer_slots + layer_slot


def compact_full_layer_rank(indexer_types: Sequence[str], layer_idx: int) -> int:
    """Return compact index-cache slot for a full layer."""
    if not 0 <= layer_idx < len(indexer_types):
        raise ValueError("layer index out of range")
    if indexer_types[layer_idx] != "full":
        raise ValueError(f"shared layer {layer_idx} does not own an index-cache slot")
    return sum(mode == "full" for mode in indexer_types[:layer_idx])


def reconstruct_logical_pages(
    pool: torch.Tensor,
    physical_pages: Sequence[int],
    *,
    layer_slot: int,
    num_layer_slots: int,
    valid_tokens: int | None = None,
) -> torch.Tensor:
    """Reconstruct logical-token order from a folded ``[page*layer,1,T,D]`` pool."""
    pieces = [pool[folded_cache_slot(page, layer_slot, num_layer_slots), 0] for page in physical_pages]
    if not pieces:
        return pool.new_empty((0, pool.shape[-1]))
    result = torch.cat(pieces, dim=0)
    return result if valid_tokens is None else result[:valid_tokens]


@dataclass(frozen=True)
class DramMemorySample:
    label: str
    num_banks: int
    allocated_bytes: int
    free_bytes: int
    largest_contiguous_free_bytes_per_bank: int


def assert_no_device_allocation_for_page_table_update(
    pool_allocated: DramMemorySample,
    after_page_table_update: DramMemorySample,
) -> None:
    """Assert logical reserve/release did not grow the preallocated cache arena."""
    if pool_allocated.num_banks != after_page_table_update.num_banks:
        raise AssertionError("DRAM bank count changed during the measurement")
    if pool_allocated.allocated_bytes != after_page_table_update.allocated_bytes:
        raise AssertionError(
            "logical page-table update allocated device memory: "
            f"{pool_allocated.allocated_bytes} -> {after_page_table_update.allocated_bytes}"
        )


def sample_dram_memory(mesh_device, label: str) -> DramMemorySample:
    """Synchronize and capture the aggregate DRAM allocator view."""
    import ttnn

    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return DramMemorySample(
        label=label,
        num_banks=view.num_banks,
        allocated_bytes=view.num_banks * view.total_bytes_allocated_per_bank,
        free_bytes=view.num_banks * view.total_bytes_free_per_bank,
        largest_contiguous_free_bytes_per_bank=view.largest_contiguous_bytes_free_per_bank,
    )


def bank_page_counts(page_bank_ids: Sequence[int], num_banks: int) -> tuple[int, ...]:
    """Count allocator-selected pages per DRAM bank, rejecting invalid metadata."""
    if num_banks <= 0:
        raise ValueError("num_banks must be positive")
    counts = [0] * num_banks
    for bank_id in page_bank_ids:
        if not 0 <= bank_id < num_banks:
            raise ValueError(f"bank id {bank_id} outside [0, {num_banks})")
        counts[bank_id] += 1
    return tuple(counts)


def assert_bank_balance(page_bank_ids: Sequence[int], num_banks: int, tolerance_pages: int = 1) -> None:
    counts = bank_page_counts(page_bank_ids, num_banks)
    if counts and max(counts) - min(counts) > tolerance_pages:
        raise AssertionError(f"page placement is imbalanced across banks: {counts}")


@dataclass(frozen=True)
class PerfComparison:
    fixed_seconds: tuple[float, ...]
    paged_seconds: tuple[float, ...]

    @property
    def fixed_median(self) -> float:
        return statistics.median(self.fixed_seconds)

    @property
    def paged_median(self) -> float:
        return statistics.median(self.paged_seconds)

    @property
    def ratio(self) -> float:
        return self.paged_median / self.fixed_median


def alternate_hot_measurements(
    mesh_device,
    fixed_call: Callable[[], object],
    paged_call: Callable[[], object],
    *,
    warmups: int = 1,
    samples: int = 5,
) -> PerfComparison:
    """Measure synchronized hot fixed/paged calls in alternating order."""
    import ttnn

    if warmups < 1 or samples < 1:
        raise ValueError("at least one warmup and sample are required")

    def measure(call: Callable[[], object]) -> float:
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        call()
        ttnn.synchronize_device(mesh_device)
        return time.perf_counter() - start

    for _ in range(warmups):
        measure(fixed_call)
        measure(paged_call)

    fixed: list[float] = []
    paged: list[float] = []
    for iteration in range(samples):
        order = (("fixed", fixed_call), ("paged", paged_call))
        if iteration % 2:
            order = tuple(reversed(order))
        for name, call in order:
            (fixed if name == "fixed" else paged).append(measure(call))
    return PerfComparison(tuple(fixed), tuple(paged))


def load_sparse_mla_backend():
    """Load opt-in test glue without coupling tests to an unfinished production API.

    The module named by ``TT_PREFILL_PAGING_TEST_BACKEND`` must expose
    ``run_glm52_sparse_mla_parity(...)`` and return the artifact mapping described
    in ``test_glm52_sparse_mla_paged_qb.py``.
    """
    module_name = os.environ.get("TT_PREFILL_PAGING_TEST_BACKEND")
    if not module_name:
        return None
    return importlib.import_module(module_name)


def require_artifacts(artifacts: Mapping[str, object], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in artifacts]
    if missing:
        raise AssertionError(f"paged prefill backend omitted artifacts: {missing}")
