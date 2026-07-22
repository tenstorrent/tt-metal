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
stability gate. Final BF16 and scaled-FP8 replicas were exact. FP8 validation
compares the opaque device payload bytes directly, avoiding a second device
typecast during validation. Full-size fresh-output BF16 and FP8 cases also
passed and selected the native unicast writer.

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

### Small-message FP8 validation resolution

The apparent four-rank, 128-row/device scaled-FP8 corruption was a validation
artifact rather than a Fabric scatter-write defect. Two independent device
`FP8 -> BF16` typecasts of the same payload can differ at rounding boundaries,
so converting the input and output separately made the comparison
nondeterministic.

The regression matrix now reads FP8 tensors as opaque host bytes and compares
the carried payload exactly. Both axes, BF16 and FP8, and fresh and persistent
outputs pass for the four-rank case (`8/8`). The two-rank matrix also passes
`8/8`, and the full 8x1 matrix passes `8/8` across Fabric1D and Fabric2D. The
diagnostic scatter bypass and CB write-barrier experiments were removed because
neither addressed a collective defect.

## 2026-07-22: explicit route plan and cache-key qualification

The automatic Fabric2D selection now materializes the complete directed
neighbor route plan on the host. Every forward and backward edge records its
source/destination Fabric node, logical direction, physical Ethernet direction,
available link indices, and terminal packet hop count. Selection requires every
edge to be direct and every requested link index (`0..num_links-1`) to exist.
The complete eligible plan is hashed into the all-gather program-cache key;
the previous eligibility boolean alone could not distinguish two different
physical route plans that both happened to be eligible.

Seven pure host tests pass without opening a device. They cover two-rank line
endpoints, four- and eight-rank lines, an eight-rank direct cycle including the
wrap edge, non-direct-wrap fallback, missing-link fallback, a cycle whose
successive edges turn in the physical topology, and cache-key inequality for
Fabric configuration, topology, axis, link count, and neighbor-plan changes:

```bash
build_Release/test/ttnn/unit_tests_ttnn_ccl \
  --gtest_filter='AllGatherNeighborRoutePlan.*'
```

Result: `7/7` passed. The generic Fabric allocator invariants also remain
green (`2/2` passed), and `./build_metal.sh --release` passes with no `--dev`.

Post-change device qualification used only `scripts/run_safe_pytest.sh`:

- full 8x1 BF16/scaled-FP8, Fabric1D/Fabric2D, fresh/persistent exact matrix:
  `8/8` passed;
- two-rank and four-rank line matrices, both formats and output lifetimes:
  `16/16` passed;
- cached full-size Fabric2D stability: `2/2` passed with exact payloads and no
  program-cache growth.

Fresh matched performance after route-plan hardening is:

| Format | Fabric | Median | Minimum | p90 | Effective receive BW | Samples (ms) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BF16 | Fabric1D | 5.565 ms | 5.563 ms | 5.577 ms | 94.971 GB/s | 5.563, 5.577, 5.564, 5.565, 5.565, 5.573, 5.564 |
| BF16 | Fabric2D | 5.812 ms | 5.809 ms | 5.828 ms | 90.932 GB/s | 5.828, 5.809, 5.811, 5.811, 5.826, 5.812, 5.820 |
| scaled FP8 | Fabric1D | 3.426 ms | 3.424 ms | 3.430 ms | 94.270 GB/s | 3.429, 3.426, 3.430, 3.426, 3.425, 3.424, 3.426 |
| scaled FP8 | Fabric2D | 3.535 ms | 3.525 ms | 3.565 ms | 91.370 GB/s | 3.543, 3.535, 3.548, 3.565, 3.527, 3.534, 3.525 |

The fresh ten-run Fabric2D stability measurements are:

| Format | Median | p90 | Effective receive BW | Ten samples (ms) |
| --- | ---: | ---: | ---: | --- |
| BF16 | 5.820 ms | 5.834 ms | 90.802 GB/s | 5.834, 5.819, 5.837, 5.827, 5.819, 5.823, 5.815, 5.820, 5.820, 5.812 |
| scaled FP8 | 3.533 ms | 3.539 ms | 91.405 GB/s | 3.526, 3.534, 3.540, 3.535, 3.530, 3.539, 3.539, 3.526, 3.532, 3.532 |

Fabric2D remains above the 90 GB/s stretch goal for both formats, stays within
4.3% (BF16) and 3.1% (scaled FP8) of its matched Fabric1D control, and keeps
p90 less than 1% above median.

## 2026-07-22: production ND-sharded packet coalescing

### Reproducing the remaining sparse-MLA long regression

The eight-rank interleaved fixture remained above 90 GB/s, but the production
4x2 sparse-MLA long proxy still showed an approximately 11.27 ms main gather.
An exact isolated fixture was added for the two concurrent four-rank SP lines:

- `64,640` rows per device, matching one SP rank at the long-cache point;
- BF16 `1,152 B` and scaled-FP8 `704 B` rows;
- persistent interleaved-DRAM output;
- either interleaved input or the production 32-row, block-cyclic ND-sharded
  DRAM cache layout;
- seven profiled samples after compile and warmup;
- native-unicast kernel-source assertion and a `40 GB/s` regression floor.

Before the fix, scaled FP8 measured:

| Input storage | Median | Effective receive BW |
| --- | ---: | ---: |
| interleaved DRAM | 2.864 ms | 47.659 GB/s |
| production ND-sharded DRAM | 11.275 ms | 12.108 GB/s |

The ND-sharded result reproduced the model's main gather and showed that the
transport and four-rank line algorithm were not the bottleneck.

### Root cause and generic fix

The native unicast factory enabled its destination-bank-owned schedule only
when both the input and output were interleaved. A production ND-sharded cache
therefore used the generic contiguous worker partition. Its writer could emit
at most the four entries supported by one NOC scatter command, producing only
`2,816 B` scaled-FP8 or `4,608 B` BF16 payloads.

The source reader already uses `TensorAccessor`, which resolves arbitrary
logical pages correctly for ND-sharded inputs. Bank ownership is required only
for the interleaved destination. The retained fix therefore removes the input
interleaving requirement while preserving all output geometry, DRAM, row-major,
slice-divisibility, and resource checks. This allows destination-bank-local
workers to pack twenty FP8 rows (`14,080 B`) or twelve BF16 rows (`13,824 B`)
under the `14,336 B` Fabric payload ceiling.

There is no dtype, model, exact-shape, environment-variable, or public API gate.
Non-interleaved outputs and unsupported geometries retain the generic schedule.

Post-fix production-layout results are:

| Format | Median | Effective receive BW |
| --- | ---: | ---: |
| BF16 | 4.707 ms | 47.458 GB/s |
| scaled FP8 | 2.864 ms | 47.663 GB/s |

Scaled FP8 improves by `3.94x`; interleaved controls remain unchanged. Exact
four-rank correctness passes on both mesh axes for BF16/scaled FP8, fresh and
persistent outputs, and interleaved/ND-sharded inputs (`16/16` total, including
the pre-existing interleaved cases).

### Product A/B after the fix

The complete sparse-MLA performance matrix passed (`12/12`). Device-program
critical-path time versus the original `6c5573df7a1` baseline is:

| Model | Cache | Warm | Cold | Long |
| --- | --- | ---: | ---: | ---: |
| DeepSeek V3.2 | BF16 | 12.570 -> 11.241 ms (`+10.6%`) | 118.699 -> 112.836 ms (`+4.9%`) | 32.810 -> 26.496 ms (`+19.2%`) |
| DeepSeek V3.2 | scaled FP8 | 11.187 -> 10.074 ms (`+9.9%`) | 104.868 -> 98.928 ms (`+5.7%`) | 27.964 -> 23.826 ms (`+14.8%`) |
| GLM 5.1 | BF16 | 10.486 -> 9.504 ms (`+9.4%`) | 96.245 -> 90.427 ms (`+6.0%`) | 27.601 -> 21.353 ms (`+22.6%`) |
| GLM 5.1 | scaled FP8 | 8.915 -> 7.984 ms (`+10.4%`) | 83.032 -> 77.126 ms (`+7.1%`) | 22.464 -> 18.280 ms (`+18.6%`) |

Every scenario is faster than the original baseline. In the DeepSeek scaled-FP8
long case, the five CCL programs total 4.479 ms instead of 8.311 ms at baseline;
the main persistent gather is 2.864 ms instead of 5.752 ms.

### Final release qualification on this box

All hardware pytest runs used `scripts/run_safe_pytest.sh`, release firmware,
and no `--dev`:

- `./build_metal.sh --release`: passed with warnings as errors;
- route-plan host tests: `7/7` passed;
- Fabric channel allocator host tests: `2/2` passed;
- matched eight-rank Fabric1D/Fabric2D correctness/performance matrix: `12/12`
  passed;
- ten-run cached Fabric2D stability: `2/2` passed;
- exact four-rank interleaved/ND-sharded perf matrix: `4/4` passed;
- complete sparse-MLA correctness suite: `39/39` passed in 918.05 seconds;
- complete sparse-MLA performance suite (plus dense controls): `27/27` passed.

The final matched 8x1 performance controls are:

| Format | Fabric | Median | Minimum | p90 | Effective receive BW |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | Fabric1D | 5.565 ms | 5.563 ms | 5.577 ms | 94.964 GB/s |
| BF16 | Fabric2D | 5.821 ms | 5.803 ms | 5.839 ms | 90.791 GB/s |
| scaled FP8 | Fabric1D | 3.432 ms | 3.423 ms | 3.435 ms | 94.102 GB/s |
| scaled FP8 | Fabric2D | 3.526 ms | 3.522 ms | 3.539 ms | 91.583 GB/s |

Fabric2D remains above the 90 GB/s stretch goal, within 4.4% (BF16) and 2.7%
(scaled FP8) of Fabric1D, while the production ND-sharded fix removes the last
observed sparse-MLA long regression.

## 2026-07-22: topology qualification is automatic and hardware-safe

The sparse-MLA CCL benchmark still carried experimental environment variables
for packet size, cache-row coalescing, link bandwidth, and opting the LoudBox
proxy into a ring. Its default SP path therefore measured Fabric1D even though
the production target and this project use a direct physical ring embedded in
Fabric2D.

Those experiment knobs are now removed. The benchmark automatically selects:

- Fabric2D on the local 8x1 SP ring;
- Fabric2D on Galaxy 8x4, where one mesh all-gather dispatch represents four
  concurrent SP=8 rings;
- the 200 Gbps/link/direction Galaxy roofline exposed by both Galaxy and the
  Galaxy-compatible Ethernet firmware installed on this LoudBox;
- the native neighbor-unicast backend, verified from realtime-profiler kernel
  sources. The benchmark fails if `unicast_writer.cpp` is absent or the
  multicast receiver is present.

The production-shaped BF16 KVPE benchmark passed through
`scripts/run_safe_pytest.sh` without an environment selector:

| Scenario | Local rows | Device-program time | Effective receive BW | 100 GB/s roofline utilization |
| --- | ---: | ---: | ---: | ---: |
| warm, 50K cache | 7,040 | 0.709 ms | 80.043 GB/s | 80.0% |
| long, 500K cache | 64,640 | 5.729 ms | 90.990 GB/s | 91.0% |

Both records contain only `tt_fabric_mux.cpp`, `unicast_reader.cpp`, and
`unicast_writer.cpp`. The long case moves 521.257 MB of critical-path receive
traffic and the full gathered tensor is 517,120 rows.

An exact physical QuietBox test was added for SP=2/TP=2. It covers two
concurrent SP lines, BF16/scaled FP8, fresh/persistent outputs, exact payloads,
and native-kernel selection. It requires exactly four physical devices. This
is necessary because opening a 2x2 submesh on the current eight-device box
left live Fabric2D routers without their physical peers and failed during
router initialization, before all-gather dispatch. On this box the corrected
test collects and skips all four cases cleanly; it must be run on a real
QuietBox for pass evidence. The existing full-mesh two-rank matrix remains the
local proxy and passed `8/8` across Fabric1D/Fabric2D, BF16/scaled FP8, and
fresh/persistent outputs.

```bash
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'quietbox_sized_concurrent_sp_lines' -x -s -v
```

The same sparse-MLA CCL test automatically expands to `(8, 4)` on a 32-device
Galaxy. Its SP-axis all-gather launches all four concurrent SP=8 rings with the
production long shape and persistent output. The command requiring Galaxy is:

```bash
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_ccl_perf.py \
  -k 'kvpe_all_gather_perf and long' -x -s -v
```

### G0-G8 completion audit

| Goal | Current evidence | Status |
| --- | --- | --- |
| G0 | Release build passed; prerequisite and handoff commits and matched commands are recorded | proven |
| G1 | Seven host route-plan cases cover direct line/ring edges, wrap, rejection, and plan hashing | proven |
| G2 | Exact BF16/FP8 fresh/persistent correctness passes at two, four, and eight ranks; profiler asserts native unicast | proven on this box |
| G3 | Ten cached full-size Fabric2D executions per format pass; p90 is below 1% over median | proven on this box |
| G4 | 90.791 GB/s BF16 and 91.583 GB/s scaled FP8 exceed 70 GB/s | proven |
| G5 | Both exceed 85 GB/s and are within 10% of matched Fabric1D | proven |
| G6 | Both Fabric modes exceed 90 GB/s for both formats | proven |
| G7 | Sparse-MLA correctness passes 39/39; perf passes 27/27; every DSA/GLM warm/cold/long result improves | proven on this box |
| G8 | No implementation selector or model/shape gate; automatic 8x1 and 8x4 test selection exists; QuietBox and four-ring Galaxy tests are hardware-safe | implementation/coverage ready; actual QuietBox and Galaxy runs still missing |

The project cannot yet be called product-ready under the plan's literal G8
exit condition: this host has eight devices, so it cannot supply real
four-device QuietBox or 32-device four-ring Galaxy pass evidence. No software
workaround or smaller submesh is treated as equivalent to that hardware proof.
