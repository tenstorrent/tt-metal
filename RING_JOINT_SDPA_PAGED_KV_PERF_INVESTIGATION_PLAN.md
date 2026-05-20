# Ring Joint SDPA Paged K/V Implementation Notes

## Scope

This branch adds paged K/V support to multichip Ring Joint SDPA.

The supported paged K/V layout is:

- K/V tensors: `[physical_pages, heads, page_block_size, head_dim]`
- page table: row-major `INT32`, `[batch, logical_pages]`
- SP sharding: physical page dimension is sharded across ring ranks
- `chunk_start_idx`: logical token offset into the page table, tile-row aligned

The production performance path is rank-local paging:

- logical pages for a source rank map to physical pages owned by the same rank
- the reader can follow normal RingJoint ring synchronization
- K/V injection can still use the existing chain/multicast machinery

The arbitrary global page-table path is correctness-supported, but it is a more
expensive contract. A global physical page id also implies an owning SP rank, so
random global page placement can make an early logical chunk wait for future
ring owners and lose all-gather/compute overlap.

## Current Perf Findings

Rank-local and identity layouts are at parity when page size is chosen
reasonably:

| Family | Config | Rank-Local / Identity Result |
|--------|--------|------------------------------|
| WAN non-causal | `wan2_2_4xGLX`, `q224/k128` | page224+ is within about 0-2% of non-paged |
| MLA causal balanced | `mla_100k`, `q160/k160` | measured page sizes are within noise/parity |

Small WAN pages still have reader overhead:

| Page Size | WAN `q224/k128` Rank-Local Overhead |
|-----------|--------------------------------------|
| 64 | about 16-17% |
| 160 | about 7% |
| 224+ | negligible in the measured runs |

Global random cross-rank placement remains slower even at large pages because it
breaks ring-order ownership:

| Config | Best Global Result Observed |
|--------|-----------------------------|
| WAN `q224/k128` | about 41% overhead at page1120 |
| MLA `q160/k160` | about 22-24% overhead at page1600 |

The key point: placing pages in different DRAM locations is not the issue by
itself. The issue is moving a logical source-rank page to a physical page owned
by a different rank.

## Implementation Shape

The non-paged code path remains compile-time gated:

- reader compile-time arg `is_paged`
- page-table ownership arg `paged_kv_page_table_is_rank_local`
- paged reader helpers are called only under `if constexpr (is_paged)`
- rank-local/global owner differences are gated with `if constexpr`

Core pieces:

- API adds optional `page_table`, `chunk_start_idx`, and
  `paged_kv_page_table_is_rank_local`
- validation checks page-table dtype/layout/rank/capacity and paged K/V shape
- all-gather supports dim-0 sharding for physical pages
- reader resolves page-table entries and reads paged K/V chunks
- rank-local path uses local/gathered readers according to logical ring owner
- global path derives physical owner from page id and waits only for required
  owners
- K batch multicast and partial V multicast are enabled for supported paged MLA
  cases

## Preserved Tests

The branch keeps focused correctness coverage for:

- paged cache round trip
- identity, rank-local, and global page tables
- non-causal and causal balanced smoke tests
- non-zero `chunk_start_idx`
- joint sequence support
- validation failures for bad page-table layout/dtype/rank/capacity and invalid
  rank-local offsets
- program-cache permutation behavior
- accuracy and determinism parametrized over WAN and MLA shapes
- perf comparison tables for non-paged vs paged and page-size sensitivity

Required quick gate after touching production reader/factory code:

```bash
scripts/run_safe_pytest.sh \
  'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_paged_kv_accuracy[wan2_2_4xGLX-q224-k128-page64]' \
  'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_paged_kv_determinism[wan2_2_4xGLX-q224-k128-page64]' \
  'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_paged_kv_accuracy[mla_100k-q160-k160-page64]' \
  'tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_paged_kv_determinism[mla_100k-q160-k160-page64]' \
  -s --tb=short
```

Latest run before cleanup: PASS, 4 passed.

## Remaining Risk

Global random page ownership is not a zero-overhead mode. Making it reach parity
requires a larger design change, such as:

- constrain global placement to preserve owner order
- precompute owner dependency metadata and schedule/prefetch before the chunk is
  compute-critical
- gather/scatter global pages into ring-order staging before SDPA
- replicate or locally cache K/V pages

Until then, rank-local should be treated as the performance contract and global
as the flexible correctness contract.
