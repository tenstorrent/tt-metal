# CCL helpers — first real-silicon validation (4-chip Blackhole QuietBox)

Branch: `wransom/ccl_hw_4chip_review` = `wransom/ccl_help_review` + the test-only
adaptations below. Every prior multi-device verification of this work ran on the
functional craq-sim; this is the first run on real fabric.

## Hardware

4x Blackhole **p150a**, fw bundle 19.5.0.0 (uniform), `FABRIC_1D`, auto-discovered
mesh graph. Openable mesh shapes: `(1,4)`, `(2,2)`, `(4,1)`, `(1,2)`.
`(2,4)` / `(1,8)` cannot open (`system_mesh.cpp:159 requested_size <= system_size`) —
the box has 4 chips, not the 8 the suites pin.

## Test adaptation

`tests/nightly/t3000/ccl/test_point_to_point.py` pins `MESH_SHAPE = (2, 4)`.
Made it `CCL_HW_MESH_SHAPE`-overridable, defaulting to the `(1,4)` line here.
This is the only change; no op or helper code was touched for the runs below.

## Results — `point_to_point` nightly

| mesh | passed | failed |
|------|--------|--------|
| `(1,4)` | 18 | 1 |
| `(2,2)` | 16 | 1 |

Both meshes fail **only** the one pre-existing bug below. Everything else is green,
including:

- `test_point_to_point_with_device_delay` (both `row_major` and `tile`) — **new
  coverage**. These busy-wait on device clock cycles and can never terminate on the
  simulator, so they had never been executed. They pass on silicon and stress chip
  desync.
- `test_point_to_point_cache_hit_with_output_tensor` — validates the conn-block-first
  `Buffer*` binding positions across program-cache hits, and the cache-reuse
  semaphore re-arm. Green on real fabric.
- `(2,2)` additionally exercises the **column-direction** fabric route (the
  `ccl_dm_route` fwd/bwd sign reversal on `dim=0`), which a `(1,4)` line cannot.

Verdict: the review-branch helper API consolidation (`AtomicIncChannel` /
`arm_inc` / `inc`), the conn-block-first runtime-arg layout in both factories and
both kernels, and `signal`/`signal_once` all work on real silicon.

## Pre-existing bug found: unaligned ROW_MAJOR DRAM pages corrupt on Blackhole

Not a helper bug and not a regression — see "provenance" below.

`ttnn.point_to_point` silently corrupts data whenever the input is an **interleaved
DRAM** tensor whose page size is **not a multiple of the 64-byte Blackhole DRAM
alignment**. Exactly the odd-indexed pages arrive as garbage/zero; even pages are
correct.

Characterization (`tests/nightly/t3000/ccl/test_char_p2p.py`, bf16, 1 hop, `(1,4)`):

| page bytes | pages | memory | bad pages |
|-----------:|------:|--------|-----------|
| 32  |  8 | DRAM | 1,3,5,7  **FAIL** |
| 32  |  2 | DRAM | 1        **FAIL** |
| 32  |  4 | DRAM | 1,3      **FAIL** |
| 32  | 16 | DRAM | all odd  **FAIL** |
| 48  |  8 | DRAM | 1,3,5,7  **FAIL** |
| 64  |  8 | DRAM | none     PASS |
| 256 |  3 | DRAM | none     PASS |
| 32  |  8 | **L1** | none   PASS |

So the trigger is exactly `page_size % dram_alignment != 0`, at any page count >= 2.
The identical shape in **L1 passes**, which is what pins it to the DRAM alignment
(64 B on Blackhole, 16 B on L1).

### Why it was never seen

Blackhole's DRAM alignment is **64 B**; Wormhole's is **32 B**. A 32-byte RM page is
already aligned on Wormhole, so the whole class is invisible there — and the shipped
`point_to_point` tests ran on Wormhole/T3K and on the simulator. The one nightly case
that trips it, `(1,1,8,16)` ROW_MAJOR, is the suite's only case that is both
multi-page and sub-alignment; every other RM case is single-page or 64B-aligned.

### Contributing (confirmed) defect

Both program factories derive their page framing from the **L1** alignment while the
tensors are in DRAM:

- `ttnn/cpp/ttnn/operations/point_to_point/device/host/send_program_factory.cpp:31,35,51`
- `ttnn/cpp/ttnn/operations/point_to_point/device/host/receive_program_factory.cpp:30,34`

`hal::get_l1_alignment()` is 16 on both Blackhole and Wormhole; the measured buffer
here is `page_size=32`, `aligned_page_size=64`. The codebase idiom is the buffer's own
alignment (`buffer()->alignment()`, or `buffer_type()==DRAM ? get_dram_alignment() :
get_l1_alignment()`), as used throughout `data_movement/` (e.g.
`permute_rm_program_factory.cpp:22-23`). This is a real latent bug and should be
fixed regardless.

**However**: forcing the correct 64 B alignment through the accessor left the
computed NOC addresses byte-identical and the corruption unchanged, so the L1/DRAM
alignment misuse alone is **not** the complete mechanism. The remaining divergence is
inside `point_to_point`'s own reader kernel
(`device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp`), whose
`TensorAccessor` yields addresses that read zero for odd page indices even though
`ttnn.to_torch` reads the same buffer back correctly. That last step is unresolved.

The shared helper `ccl_packet_dims` is **innocent** — it applies whatever alignment it
is handed; the wrong value is chosen at the two call sites.

### Provenance — predates all of the helper work

`git show ece9e2fb6c8 -- .../reader_unary_interleaved_start_id_gen.cpp` (original
`point_to_point` PR #22880, Aug 2025) has the identical raw-page-size
`InterleavedAddrGen` pattern from day one, and that PR's own TODO list reads
*"fix issues with unaligned RM pages."* `git show fcc68c86a6b` shows `ccl_packet_dims`
was a byte-for-byte move of `point_to_point`'s own `detail::compute_aligned_packet_dims`
into the shared helper — the `get_l1_alignment()` call site did not change. A later
audit (PR #41902) targeted exactly this class of bug but missed these two kernels.

So this reproduces on `wransom/ccl_help` and on the pre-helper upstream implementation.
`wransom/ccl_help_review` only relocated the code.
