# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model-local persistent KV pool ownership for GLM-5.2 prefill paging.

This module deliberately stops at allocation and ownership.  It does not alter the
MLA/indexer kernels, the prefill scheduler, migration, or decode.  The existing
GLM-5.2 adapter remains the default; callers must explicitly construct this pool.

One physical bundle holds one 5,120-token prefill chunk for every local primary
KVPE layer and every compact ``full`` indexer layer.  The compact page table has
one physical-bundle id per logical bundle.  Each bundle contains 160 deterministic
32-token subpages; those offsets are derived rather than stored in the table.  The
same physical bundle id owns the corresponding storage in both cache tensors,
making allocation and release a single coordinated operation rather than two
independently fallible allocations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Callable

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches

GLM52_KV_PAGE_TOKENS = 32
GLM52_KV_BUNDLE_TOKENS = 5 * 1024
GLM52_KV_PAGES_PER_BUNDLE = GLM52_KV_BUNDLE_TOKENS // GLM52_KV_PAGE_TOKENS
GLM52_UNMAPPED_PAGE = -1


class Glm52PagedCacheExhausted(RuntimeError):
    """Raised when the fixed physical bundle pool has no free bundle."""


@dataclass(frozen=True)
class Glm52BundleAllocation:
    """Stable logical-to-physical ownership returned by :meth:`allocate`."""

    logical_slot: int
    logical_bundle: int
    physical_bundle: int

    @property
    def logical_page_start(self) -> int:
        return self.logical_bundle * GLM52_KV_PAGES_PER_BUNDLE

    @property
    def physical_page_start(self) -> int:
        return self.physical_bundle * GLM52_KV_PAGES_PER_BUNDLE


PageTableSync = Callable[[torch.Tensor, object], None]
ReleaseFence = Callable[[], None]


class Glm52PagedKvCachePool(MlaKvCaches):
    """GLM-5.2 KV tensors plus atomic bundle ownership and a persistent page table.

    ``kvpe`` is physically laid out as
    ``[capacity_bundles * num_primary_layers, 1, bundle_tokens / SP, 576]``
    in BF16 row-major format.  ``index`` has the corresponding bundle-major
    layout for compact full-indexer layers in BFP8 tile format.

    The host page table is the source of truth and has shape
    ``[num_logical_slots, max_bundles_per_slot]``.  Each entry maps one logical
    5,120-token bundle to one physical layer-bundle id; its 160 32-token subpages
    are deterministic offsets within that bundle.  The device tensor is allocated
    once; allocate/release copy new contents into that same tensor object.

    Updates are host-atomic: the candidate page table is copied to the persistent
    device tensor before ownership/free-list state is committed.  If the copy
    raises, host ownership remains unchanged.
    """

    def __init__(
        self,
        *,
        kvpe,
        index,
        device_page_table,
        num_logical_slots: int,
        max_bundles_per_slot: int,
        capacity_bundles: int,
        num_primary_layers: int,
        num_index_layers: int,
        sync_page_table: PageTableSync,
        synchronize_before_release: ReleaseFence | None = None,
    ):
        super().__init__(kvpe=kvpe, index=index)
        for name, value in (
            ("num_logical_slots", num_logical_slots),
            ("max_bundles_per_slot", max_bundles_per_slot),
            ("capacity_bundles", capacity_bundles),
            ("num_primary_layers", num_primary_layers),
            ("num_index_layers", num_index_layers),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if index is None:
            raise ValueError("GLM-5.2 paged cache requires the compact index cache")

        expected_primary_slots = capacity_bundles * num_primary_layers
        expected_index_slots = capacity_bundles * num_index_layers
        if kvpe.storage.shape[0] != expected_primary_slots:
            raise ValueError(
                f"KVPE pool batch dim {kvpe.storage.shape[0]} != "
                f"capacity({capacity_bundles}) * primary_layers({num_primary_layers})"
            )
        if index.shape[0] != expected_index_slots:
            raise ValueError(
                f"index pool batch dim {index.shape[0]} != "
                f"capacity({capacity_bundles}) * index_layers({num_index_layers})"
            )

        self.device_page_table = device_page_table
        self.num_logical_slots = num_logical_slots
        self.max_bundles_per_slot = max_bundles_per_slot
        self.capacity_bundles = capacity_bundles
        self.num_primary_layers = num_primary_layers
        self.num_index_layers = num_index_layers
        self._sync_page_table = sync_page_table
        self._synchronize_before_release = synchronize_before_release or (lambda: None)

        self._host_page_table = torch.full(
            (num_logical_slots, max_bundles_per_slot),
            GLM52_UNMAPPED_PAGE,
            dtype=torch.int32,
        )
        self._allocations: dict[tuple[int, int], int] = {}
        self._free_bundles = deque(range(capacity_bundles))
        # Exact logical prefix length is independent of the 5,120-token physical
        # compute/write granule.  Kernels may write the padded tail, but future
        # migration/session consumers must never infer validity from allocation.
        self._valid_ends = [0] * num_logical_slots
        self._lock = RLock()

    @property
    def host_page_table(self) -> torch.Tensor:
        """A defensive snapshot; callers cannot mutate allocator ownership."""

        with self._lock:
            return self._host_page_table.clone()

    @property
    def num_free_bundles(self) -> int:
        with self._lock:
            return len(self._free_bundles)

    def valid_end(self, logical_slot: int) -> int:
        """Return the exact end of the slot's real-token prefix."""

        self._validate_slot(logical_slot)
        with self._lock:
            return self._valid_ends[logical_slot]

    def allocation(self, logical_slot: int, logical_bundle: int) -> Glm52BundleAllocation | None:
        """Return an existing allocation without creating one."""

        self._validate_logical_address(logical_slot, logical_bundle)
        with self._lock:
            physical = self._allocations.get((logical_slot, logical_bundle))
            if physical is None:
                return None
            return Glm52BundleAllocation(logical_slot, logical_bundle, physical)

    def allocate(self, logical_slot: int, logical_bundle: int) -> Glm52BundleAllocation:
        """Atomically acquire one coordinated KVPE/index bundle.

        Repeating an already-owned logical address is idempotent.  Exhaustion is
        explicit; this component performs no eviction or implicit release.
        """

        self._validate_logical_address(logical_slot, logical_bundle)
        key = (logical_slot, logical_bundle)
        with self._lock:
            existing = self._allocations.get(key)
            if existing is not None:
                return Glm52BundleAllocation(logical_slot, logical_bundle, existing)
            if not self._free_bundles:
                raise Glm52PagedCacheExhausted(
                    f"GLM-5.2 paged KV pool exhausted while allocating " f"slot={logical_slot}, bundle={logical_bundle}"
                )

            physical = self._free_bundles[0]
            candidate = self._host_page_table.clone()
            candidate[logical_slot, logical_bundle] = physical

            self._sync_page_table(candidate, self.device_page_table)
            self._free_bundles.popleft()
            self._allocations[key] = physical
            self._host_page_table = candidate
            return Glm52BundleAllocation(logical_slot, logical_bundle, physical)

    def allocate_chunk(
        self,
        logical_slot: int,
        start_token: int,
        actual_end: int | None = None,
    ) -> tuple[Glm52BundleAllocation, ...]:
        """Reserve the one bundle addressed by an aligned 5,120-token compute chunk.

        Position-dependent SP ownership is part of the physical format: each input
        SP shard already contains exactly its owner slab only when the compute chunk
        begins on a bundle boundary. Actual sequence length may end inside the chunk,
        but the next compute chunk still starts at the next 5,120-token boundary.

        ``actual_end`` records the exact real-token boundary separately from the
        physical allocation. Omitting it performs allocation only, preserving the
        low-level API for callers that manage non-prefix bundles directly.
        """

        self._validate_slot(logical_slot)
        if not isinstance(start_token, int) or isinstance(start_token, bool):
            raise TypeError(f"start_token must be an integer, got {start_token!r}")
        if start_token < 0:
            raise IndexError(f"start_token must be non-negative, got {start_token}")
        if start_token % GLM52_KV_BUNDLE_TOKENS != 0:
            raise ValueError(
                f"GLM-5.2 paged prefill requires bundle-aligned compute starts; got start_token={start_token}"
            )

        logical_bundle = start_token // GLM52_KV_BUNDLE_TOKENS
        if logical_bundle >= self.max_bundles_per_slot:
            raise IndexError(
                f"chunk [{start_token}, {start_token + GLM52_KV_BUNDLE_TOKENS}) exceeds "
                f"{self.max_bundles_per_slot} logical bundles"
            )
        if actual_end is None:
            return (self.allocate(logical_slot, logical_bundle),)
        if not isinstance(actual_end, int) or isinstance(actual_end, bool):
            raise TypeError(f"actual_end must be an integer, got {actual_end!r}")
        if not start_token <= actual_end <= start_token + GLM52_KV_BUNDLE_TOKENS:
            raise ValueError(
                f"actual_end={actual_end} must be inside compute chunk "
                f"[{start_token}, {start_token + GLM52_KV_BUNDLE_TOKENS}]"
            )

        with self._lock:
            current_end = self._valid_ends[logical_slot]
            existing = self._allocations.get((logical_slot, logical_bundle))
            if current_end == actual_end and existing is not None:
                return (Glm52BundleAllocation(logical_slot, logical_bundle, existing),)
            if start_token != current_end:
                raise ValueError(
                    f"paged prefill chunks must extend the exact valid prefix: "
                    f"slot={logical_slot} starts at {start_token}, current valid_end={current_end}"
                )
            allocation = self.allocate(logical_slot, logical_bundle)
            self._valid_ends[logical_slot] = actual_end
            return (allocation,)

    def release(self, logical_slot: int, logical_bundle: int) -> Glm52BundleAllocation:
        """Release one bundle and invalidate its compact page-table entry."""

        self._validate_logical_address(logical_slot, logical_bundle)
        key = (logical_slot, logical_bundle)
        with self._lock:
            if key not in self._allocations:
                raise KeyError(f"no GLM-5.2 KV bundle allocated for slot={logical_slot}, bundle={logical_bundle}")
            # A table invalidation followed by reuse must not race an earlier layer
            # still fetching this mapping or its pool rows.
            self._synchronize_before_release()
            physical = self._allocations[key]
            candidate = self._host_page_table.clone()
            candidate[logical_slot, logical_bundle] = GLM52_UNMAPPED_PAGE

            self._sync_page_table(candidate, self.device_page_table)
            del self._allocations[key]
            self._free_bundles.append(physical)
            self._valid_ends[logical_slot] = min(
                self._valid_ends[logical_slot],
                logical_bundle * GLM52_KV_BUNDLE_TOKENS,
            )
            self._host_page_table = candidate
            return Glm52BundleAllocation(logical_slot, logical_bundle, physical)

    def release_slot(self, logical_slot: int) -> tuple[Glm52BundleAllocation, ...]:
        """Release every bundle owned by one logical slot in logical order.

        The page table is uploaded once and ownership is committed only after
        that upload succeeds.
        """

        self._validate_slot(logical_slot)
        with self._lock:
            owned = sorted(
                (
                    Glm52BundleAllocation(slot, bundle, physical)
                    for (slot, bundle), physical in self._allocations.items()
                    if slot == logical_slot
                ),
                key=lambda allocation: allocation.logical_bundle,
            )
            if not owned:
                return ()

            self._synchronize_before_release()
            candidate = self._host_page_table.clone()
            candidate[logical_slot, :] = GLM52_UNMAPPED_PAGE
            self._sync_page_table(candidate, self.device_page_table)
            for allocation in owned:
                del self._allocations[(allocation.logical_slot, allocation.logical_bundle)]
                self._free_bundles.append(allocation.physical_bundle)
            self._valid_ends[logical_slot] = 0
            self._host_page_table = candidate
            return tuple(owned)

    def allocated_bundles(self, logical_slot: int) -> tuple[Glm52BundleAllocation, ...]:
        """Return one slot's allocations ordered by logical bundle."""

        self._validate_slot(logical_slot)
        with self._lock:
            return tuple(
                sorted(
                    (
                        Glm52BundleAllocation(slot, bundle, physical)
                        for (slot, bundle), physical in self._allocations.items()
                        if slot == logical_slot
                    ),
                    key=lambda allocation: allocation.logical_bundle,
                )
            )

    def _validate_slot(self, logical_slot: int) -> None:
        if not isinstance(logical_slot, int) or isinstance(logical_slot, bool):
            raise TypeError(f"logical_slot must be an integer, got {logical_slot!r}")
        if not 0 <= logical_slot < self.num_logical_slots:
            raise IndexError(f"logical_slot {logical_slot} outside [0, {self.num_logical_slots})")

    def _validate_logical_address(self, logical_slot: int, logical_bundle: int) -> None:
        self._validate_slot(logical_slot)
        if not isinstance(logical_bundle, int) or isinstance(logical_bundle, bool):
            raise TypeError(f"logical_bundle must be an integer, got {logical_bundle!r}")
        if not 0 <= logical_bundle < self.max_bundles_per_slot:
            raise IndexError(f"logical_bundle {logical_bundle} outside [0, {self.max_bundles_per_slot})")


def allocate_glm52_paged_kv_cache_pool(
    *,
    mesh_device,
    hf_config,
    mesh_shape,
    sp_axis: int,
    num_primary_layers: int,
    num_logical_slots: int,
    max_sequence_length: int,
    capacity_bundles: int,
) -> Glm52PagedKvCachePool:
    """Explicitly allocate the default-off GLM-5.2 paged prefill cache pool."""

    for name, value in (
        ("num_primary_layers", num_primary_layers),
        ("num_logical_slots", num_logical_slots),
        ("max_sequence_length", max_sequence_length),
        ("capacity_bundles", capacity_bundles),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    max_bundles_per_slot = (max_sequence_length + GLM52_KV_BUNDLE_TOKENS - 1) // GLM52_KV_BUNDLE_TOKENS

    from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

    num_index_layers = num_full_indexer_layers(hf_config)
    if not num_index_layers:
        raise ValueError("GLM-5.2 paged cache requires a non-empty full/shared indexer map")

    kvpe = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=hf_config,
        mesh_device=mesh_device,
        seq_len=GLM52_KV_BUNDLE_TOKENS,
        mesh_shape=list(mesh_shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_primary_layers,
        num_users=capacity_bundles,
    )
    index = init_kvpe_cache(
        kvpe_cache_head_dim=hf_config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=GLM52_KV_BUNDLE_TOKENS,
        mesh_shape=list(mesh_shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_index_layers,
        num_users=capacity_bundles,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )

    initial_table = torch.full(
        (num_logical_slots, max_bundles_per_slot),
        GLM52_UNMAPPED_PAGE,
        dtype=torch.int32,
    )
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    device_page_table = ttnn.from_torch(
        initial_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    def sync_page_table(table: torch.Tensor, persistent_device_table) -> None:
        host_table = ttnn.from_torch(
            table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_table, persistent_device_table)

    def synchronize_before_release() -> None:
        ttnn.synchronize_device(mesh_device)

    return Glm52PagedKvCachePool(
        kvpe=kvpe,
        index=index,
        device_page_table=device_page_table,
        num_logical_slots=num_logical_slots,
        max_bundles_per_slot=max_bundles_per_slot,
        capacity_bundles=capacity_bundles,
        num_primary_layers=num_primary_layers,
        num_index_layers=num_index_layers,
        sync_page_table=sync_page_table,
        synchronize_before_release=synchronize_before_release,
    )
