# Fabric 2D logical-ring bandwidth parity work log

This is the experiment log for
[`ALL_GATHER_FABRIC_2D_PARITY_PLAN.md`](ALL_GATHER_FABRIC_2D_PARITY_PLAN.md).
Record measurements before drawing conclusions, change one mechanism at a
time, and keep failed experiments so they are not repeated.

## 2026-07-21: Phase 0 matched benchmark fixture

### Change

Converted the full-size isolated sparse-MLA correctness and performance tests
from Fabric2D-only coverage into a matched matrix:

- `FABRIC_1D_RING` and `FABRIC_2D`;
- BF16 `1,152 B` rows and scaled-FP8 `704 B` rows;
- `65,536` rows per device on an `8x1` logical ring;
- persistent interleaved DRAM output and the automatic receiver-L1 path;
- exactly two CCL links and a `14,336 B` Fabric packet payload;
- seven measured samples after compile, warmup, and profiler-drain runs.

The test now asserts the active packet payload and logs physical direction
sequences, hop counts, direction changes, explicit-path selection, and escape
VC hop for every route variant used by the bidirectional multicast schedule.

### Command

```bash
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'matched_large_single_axis_correctness or matched_sparse_mla_row_perf' \
  -x -s
```

Result: `8 passed, 43 deselected` in `144.64 s`; safe-pytest result `PASS`.

### Measurements

| Format | Fabric | Median | Minimum | p90 | Effective receive BW | Samples (ms) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BF16 | Fabric1D | 5.720 ms | 5.717 ms | 5.738 ms | 92.392 GB/s | 5.717, 5.738, 5.720, 5.718, 5.727, 5.724, 5.720 |
| BF16 | Fabric2D | 26.909 ms | 26.791 ms | 27.166 ms | 19.640 GB/s | 26.909, 26.791, 27.166, 26.886, 27.070, 26.891, 26.942 |
| scaled FP8 | Fabric1D | 3.512 ms | 3.505 ms | 3.538 ms | 91.949 GB/s | 3.522, 3.511, 3.538, 3.520, 3.512, 3.505, 3.506 |
| scaled FP8 | Fabric2D | 16.327 ms | 16.186 ms | 16.503 ms | 19.781 GB/s | 16.400, 16.232, 16.327, 16.503, 16.259, 16.186, 16.383 |

### Evidence and conclusion

- Payload, layout, CCL policy, sample count, and hardware are now matched in a
  single permanent fixture. The previous risk of accidentally measuring
  Fabric1D with its default `4,352 B` packet is removed.
- Fabric1D remains above 91 GB/s for both formats, proving that the source
  reader, receiver-L1 fan-out, DRAM drain, persistent output, and current
  Galaxy-compatible Ethernet firmware can sustain the target rate.
- Fabric2D remains nearly dtype-independent at 19.6--19.8 GB/s. The first
  attribution target remains the Fabric2D transport path.
- The live physical ring contains straight three-hop paths, one- and two-turn
  paths, and explicit paths which switch to the escape VC after hops 1, 2, or
  3. Phase 1 must separate these cases.

## 2026-07-21: Phase 1 attribution preparation

### Code-path inspection

The native Fabric1D receiver forwarding path uses stateful NOC APIs for the
ERISC-to-downstream-ERISC packet copy. The Fabric2D explicit-path helper uses
the stateless form because a receiving router may select any of three
downstream directions dynamically while the existing stateful command buffer
contains only one destination state.

This is a plausible source of forwarding overhead, but it is not yet proven to
be the first bandwidth collapse. Enabling the stateful template blindly would
send to whichever downstream direction last programmed the shared command
buffer and is therefore unsafe.

The existing `test_tt_fabric` infrastructure can generate one-hop, straight
multi-hop, and corner routes with a fixed packet size and reports per-direction
bandwidth. The next step is to add a focused `14,336 B`, two-link Fabric2D YAML
ladder and run it under the cooperative device lock. If standard Fabric2D
routing remains fast, extend its traffic generator with the explicit linear
route encoding used by all-gather; if it already collapses, optimize the common
receive-to-forward path first.

### Safety notes

- Hardware tests use `scripts/run_safe_pytest.sh`; do not use `--dev` because
  watcher and LLK assertions increase firmware code size.
- Diagnose any hang with `tt-triage` before changing synchronization.
- Do not reintroduce receiver-channel trimming or burst channel scheduling;
  both previously hung this workload.

## 2026-07-22: Native one-hop Fabric2D implementation and attribution

Development continued under
[`ALL_GATHER_FABRIC_2D_ONE_HOP_NEIGHBOR_PLAN.md`](ALL_GATHER_FABRIC_2D_ONE_HOP_NEIGHBOR_PLAN.md).
The published handoff branch is
`pjosipovic/tmp-fabric2d-all-gather-handoff-20260722`; commit `be9b318dd40`
is the Fabric2D correctness prerequisite and commit `790282e1fb0` introduces
the native one-hop neighbor prototype.

### Functional result before channel-depth tuning

The host proves that every logical edge, including the 8x1 wrap edge, is a
direct physical Fabric neighbor. Eligible large one-dimensional gathers select
the existing native store-and-forward schedule under Fabric2D. Each emitted
packet terminates after one Ethernet hop; the receiving Tensix worker reads the
completed stripe from output DRAM and emits a new terminal packet for the next
neighbor. Non-direct axes retain the multicast fallback.

Full-size correctness passed for BF16 and scaled FP8 under Fabric1D and
Fabric2D. The first matched performance result was:

| Format | Fabric | Median | Effective receive BW |
| --- | --- | ---: | ---: |
| BF16 | Fabric1D | 5.565 ms | 94.963 GB/s |
| BF16 | Fabric2D | 12.068 ms | 43.792 GB/s |
| scaled FP8 | Fabric1D | 3.425 ms | 94.307 GB/s |
| scaled FP8 | Fabric2D | 7.300 ms | 44.239 GB/s |

This was already more than twice the old Fabric2D multicast bandwidth, but the
nearly dtype-independent 44 GB/s ceiling remained about half of Fabric1D.

### Terminal-only isolation

A matched two-rank line on the full `(4, 2)` mesh removes store-and-forward
relay iterations and DRAM rereads. It sends the same 65,536 rows/device through
two links and measures aggregate payload received by one rank. Before the
channel-depth change:

| Format | Fabric | Median | Aggregate terminal BW | Per-link BW |
| --- | --- | ---: | ---: | ---: |
| BF16 | Fabric1D | 1.589 ms | 47.511 GB/s | 23.756 GB/s |
| BF16 | Fabric2D | 3.271 ms | 23.080 GB/s | 11.540 GB/s |
| scaled FP8 | Fabric1D | 0.978 ms | 47.168 GB/s | 23.584 GB/s |
| scaled FP8 | Fabric2D | 1.978 ms | 23.325 GB/s | 11.663 GB/s |

One-direction/two-link, one-direction/one-link, and single-worker diagnostics
all preserved the same factor-of-two Fabric2D loss. Link scaling, bidirectional
contention, the worker mux, the relay loop, and DRAM reread were therefore not
the first bottleneck.

A combined payload/header slot submission experiment passed BF16 and FP8
correctness but changed terminal throughput by less than measurement noise. It
was removed under the 3% retention rule.

### Root cause: stranded Fabric2D sender slots

The static Fabric channel allocator selected these exact Blackhole geometries
for the release firmware on this machine:

```text
Fabric1D Ring:
  senders=[2,0,0] receivers=[1,0,0]
  sender_slots=[8,0,0] receiver_slots=[8,0,0]
  available=366656 B packet_slot=14384 B

Fabric2D Mesh:
  senders=[4,3,0] receivers=[1,1,0]
  sender_slots=[2,2,0] receiver_slots=[4,2,0]
  available=365248 B packet_slot=14432 B
```

Every live worker payload enters VC0 sender channel 0. Fabric1D gave that
channel eight packet slots, while the Fabric2D table gave every VC0 sender only
two. The Fabric2D configuration used 20 slots even though the L1 region could
hold 25, because a uniform per-VC table row could not use the five-slot
remainder. Live injection repeatedly stalled on the two-slot credit window.

The retained generic change gives otherwise unused whole packet slots to VC0's
local-worker injection channel for Mesh/Torus configurations without a Fabric
Tensix extension. It does not trim or reduce any forwarding or receiver
channel. On this geometry channel 0 grows from two to seven slots while all
other channel depths remain unchanged. Local and remote layout descriptions
are updated symmetrically.

### Performance after consuming stranded slots

Terminal-only A/B:

| Format | Fabric | Median | Aggregate terminal BW | Fabric2D vs prior |
| --- | --- | ---: | ---: | ---: |
| BF16 | Fabric1D | 1.589 ms | 47.508 GB/s | unchanged control |
| BF16 | Fabric2D | 1.649 ms | 45.795 GB/s | +98.5% |
| scaled FP8 | Fabric1D | 0.978 ms | 47.199 GB/s | unchanged control |
| scaled FP8 | Fabric2D | 0.999 ms | 46.170 GB/s | +97.9% |

Full 8x1, 512K-row gather:

| Format | Fabric | Median | Minimum | p90 | Effective receive BW | Samples (ms) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BF16 | Fabric1D | 5.564 ms | 5.558 ms | 5.566 ms | 94.982 GB/s | 5.564, 5.558, 5.564, 5.565, 5.561, 5.566, 5.564 |
| BF16 | Fabric2D | 5.820 ms | 5.809 ms | 5.827 ms | 90.803 GB/s | 5.823, 5.820, 5.810, 5.823, 5.809, 5.820, 5.827 |
| scaled FP8 | Fabric1D | 3.430 ms | 3.426 ms | 3.459 ms | 94.144 GB/s | 3.430, 3.428, 3.432, 3.459, 3.426, 3.431, 3.429 |
| scaled FP8 | Fabric2D | 3.528 ms | 3.522 ms | 3.550 ms | 91.530 GB/s | 3.550, 3.523, 3.528, 3.529, 3.526, 3.531, 3.522 |

Fabric2D now clears the 90 GB/s stretch gate for both formats and is within
4.4% (BF16) and 2.8% (FP8) of its matched Fabric1D control. All eight matched
correctness/performance cases passed through `scripts/run_safe_pytest.sh`.

### Cached stability and output lifetime

Ten consecutive cached full-size Fabric2D runs per format passed with no new
program-cache entry, hang, corruption, or timeout:

| Format | Median | p90 | Effective receive BW | Ten samples (ms) |
| --- | ---: | ---: | ---: | --- |
| BF16 | 5.825 ms | 5.831 ms | 90.728 GB/s | 5.830, 5.820, 5.843, 5.831, 5.828, 5.823, 5.823, 5.825, 5.816, 5.825 |
| scaled FP8 | 3.526 ms | 3.527 ms | 91.592 GB/s | 3.518, 3.526, 3.525, 3.526, 3.519, 3.521, 3.531, 3.527, 3.527, 3.527 |

The p90/median ratios are 1.0010 and 1.0003, comfortably inside the 1.05
stability gate. Final BF16 replicas were exact; scaled-FP8 replicas passed the
bounded conversion-noise and PCC checks. Full-size fresh-output BF16 and FP8
cases also passed and selected the native unicast writer.

### Commands

```bash
# Full matched correctness/performance matrix
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'matched_large_single_axis_correctness or matched_sparse_mla_row_perf' \
  -x -s -v

# Ten-run cached Fabric2D stability gate
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'matched_large_cached_stability' \
  -x -s -v

# Host allocator invariants
build_Release/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter='FabricStaticSizedChannelsAllocatorTest.*'
```

The two allocator tests verify the seven-slot Mesh/Torus worker channel, the
unchanged relay/receiver depths, complete use of whole-slot capacity, and the
unchanged eight-slot Fabric1D Ring allocation.

### Open correctness issue in the small-message scatter path

The added four-rank, 128-row/device scaled-FP8 case currently exposes sparse,
run-to-run corruption with both fresh and persistent output buffers. The BF16
case with the same topology and worker schedule passes exactly, and the full
512K-row scaled-FP8 case above remains correct and stable. Diagnostics rule out
an incorrect gather group and confirm that the native unicast writer is used.
The failure is currently isolated to the small-message, two-worker schedule,
which emits four 704-byte chunks through the Fabric scatter-write path. This is
left as an explicit failing regression test for the next debugging step; no
tolerance was weakened and the unsuccessful CB write-barrier experiment was
removed.
