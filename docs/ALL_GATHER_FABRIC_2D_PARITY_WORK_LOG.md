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
