# Unary SFPU test alignment across Wormhole, Blackhole and Quasar

Branch: `ndivnic/align_sfpu_unary`

| Commit | Change |
|---|---|
| `e5afb420ec` | Share the BH unary SFPU kernel across functional, perf, and accuracy |
| `f40d4beb90` | Let the unary SFPU driver take explicit tolerance overrides (polygamma order sweep) |
| `6938e79047` | Move the remaining unary SFPU kernel callers onto the shared config builder |
| `6805697433` | Give the batched fp32 Reciprocal stress sweep its own timeout |

## What changed

- **One kernel for Wormhole/Blackhole.** `sources/eltwise_unary_sfpu_perf.cpp` is gone.
  `sources/eltwise_unary_sfpu_test.cpp` serves functional, perf and accuracy runs: it reads
  `TILE_CNT`, `LOOP_FACTOR`, `num_faces`, the two `UNPACK_TRANSPOSE_*` runtimes and
  `PERF_RUN_TYPE`, and splits the tile loop at `MAX_TILES_DEST` (4 with fp32 Dest, 8
  otherwise). `NUM_BLOCKS`/`NUM_TILES_IN_BLOCK` are no longer read.
- **One config for the kernel.** `build_eltwise_unary_sfpu_config` in
  `test_eltwise_unary_sfpu.py` is the only place that lists the kernel's templates,
  runtimes and stimuli. It goes through `create_test_or_perf_config`, so the functional
  and perf paths cannot drift apart. `eltwise_unary_sfpu` (the driver) uses it, and so do
  the callers that need the config object itself (`test_gathering_config`: build only;
  `test_unary_sfpu_ulp`: asserts `configuration.dest_acc`). `test_tanh_specials`,
  `test_sfpu_plot`, `test_vif_equiv_sweep`, `accuracy/accuracy_harness.py`,
  `perf_vif_targets.py` and `perf_sfpu_ema.py` go through the driver.
- **Quasar** is split into families that match WH/BH (main, comp, qsr_extra,
  signbit_fpu_route), and typecast is split out into
  `quasar/test_eltwise_unary_typecast_quasar.py` / `perf_eltwise_unary_typecast_quasar.py`.

## Coverage audit

Compared with `compare_test_and_perf.py` on the parent (`3c202f9944`) and on the branch.
Variant sets were enumerated per architecture and diffed as sets of parameter
combinations.

| Architecture | Parent | Branch | Lost | Added |
|---|---|---|---|---|
| Blackhole (functional + perf) | 10014 | 13150 | 0 | +3136 |
| Wormhole (functional + perf) | 10014 | 13150 | 0 | +3136 |
| Quasar (functional + perf) | 1364 | 2108 | 0 | +744 |

**No variant was lost on any architecture.** Comparer verdicts on the branch:

- WH/BH unary, functional vs perf: `approx_mode`, `fast_mode`, `dest_acc` and `dest_sync`
  are identical. Perf `input_dimensions` is now a subset of functional; on the parent the
  two sets were disjoint. Perf formats are a subset of functional (16/20). `stable_sort`
  and `fused_sort` are perf-only axes (TopK).
- Quasar unary, functional vs perf: `mathop`, `formats`, `dest_acc` and `approx_mode` are
  identical in main, comp and qsr_extra. Perf `dest_sync` and `input_dimensions` are a
  subset of functional. Before the split, the parent had a single packed
  `mathop_formats_dest_acc_sync_implied_math_input_dims` axis.
- Quasar typecast, functional vs perf: `formats`, `dest_acc` and `approx_mode` are
  identical. Perf `input_dimensions` is a subset of functional.

### Remaining cross-architecture differences

These are deliberate or hardware-bound, and listed so they are not mistaken for gaps:

| Difference | WH/BH | Quasar | Why |
|---|---|---|---|
| Input dimensions | functional `[64,64]`, `[128,64]`, `[128,256]`; perf `[128,64]` | functional `[32,32]`, `[64,64]`; perf `[64,64]` | Quasar Dest capacity |
| Op set | 93 functional / 97 perf ops | 33 ops (a subset) | Ops not yet implemented on Quasar |
| `fast_mode`, `stable_sort`, `fused_sort` | yes | no | No Quasar kernel variant |
| Typecast perf axes | one packed `typecast_case` (formats + dest_acc) | separate `formats`, `dest_acc` | Not aligned yet (follow-up) |

Perf-only on WH/BH, with no functional counterpart: `int32`, `comp_uint16`, `comp_uint32`,
`erfinv_fp32` and TopK. Functional-only (edge, regression and parameter sweeps): `edges`,
`sqrt_family_negative_zero_regression`, `int`, `relu_{min_int,max}_threshold`,
`int_shift`, `polygamma_order`, `signbit`, `isinf_isnan`, `threshold`,
`exponential_clamp_negative`.

## Validation

### Host-only

- `pyflakes` is clean on every touched file.
- Collection is clean under `CHIP_ARCH=wormhole` and `blackhole` (16154 variants across
  the unary, tanh, gathering and ULP modules).
- `test_compare_test_and_perf.py`, `test_sfpu_accuracy_budget.py` and `test_ulp_sweep.py`:
  290 passed.

### Local Wormhole n150 (`.claude/scripts/run_test.sh`)

| Run | Result |
|---|---|
| `test_tanh_specials.py` (all) | 4 passed |
| `test_gathering_config.py` (Blackhole build-only, `compile` on this host) | 3 passed |
| `test_unary_sfpu_ulp.py`, Exp/Tanh/Sqrt/Reciprocal bf16 slice | 42 passed, 18 skipped (unswept cells) |
| `test_eltwise_unary_sfpu.py`, Float16_b Exp/Polygamma/Relu slice | 538 passed, 20 skipped |
| `perf_eltwise_unary_sfpu.py`, Exp Float16_b slice | 42 passed |
| `test_sfpu_plot.py::test_sfpu_stress[Reciprocal-fp32-batched]` under `--timeout 60` | passed (144 s, per-case 600 s ceiling) |

### CI

| Workflow | Run | Result |
|---|---|---|
| LLK e2e, default markers | [38543048226](https://github.com/tenstorrent/tt-metal/actions/runs/38543048226) | 0 failed in all four jobs (WH 127711 + 130425 passed, BH 129570 + 130924 passed) |
| LLK e2e, `accuracy` | [38543060875](https://github.com/tenstorrent/tt-metal/actions/runs/38543060875) | 1 failed per arch: the batched stress timeout, fixed in `6805697433` |
| LLK e2e, `accuracy` (re-run) | [38545136863](https://github.com/tenstorrent/tt-metal/actions/runs/38545136863) | green, 0 failed in all four jobs (WH 2349 + 1919 passed, BH 441 passed; BH group 2's split was all skipped) |
| LLK perf (WH + BH) | [38530575647](https://github.com/tenstorrent/tt-metal/actions/runs/38530575647) | green |
| LLK ttsim weekly (WH, BH, Quasar) | [38530578952](https://github.com/tenstorrent/tt-metal/actions/runs/38530578952) | green |

LLK e2e jobs report "failure" even with 0 failed tests, because the JUnit publishing
step dies with `Maximum call stack size exceeded`. The same happens on main's nightly; it
is a pre-existing infrastructure issue. The e2e runs are judged on their pytest totals.

The `Reciprocal-fp32-batched` timeout also predates this branch: on the parent the case
takes 121 s on a Wormhole n150, against the workflow's 60 s `--timeout`. No workflow ran
`-m accuracy` before, so nobody hit it.

## Perf

**The alignment does not change kernel cost.** Same-board A/B on the local Wormhole n150,
branch against parent `3c202f9944`, on the same 248-variant `perf_eltwise_unary_sfpu`
slice and the full `perf_eltwise_unary_typecast`:

- 744 unary rows and 18 typecast rows match one-to-one, with identical text sizes per
  run type.
- The median branch/parent ratio is 1.000 for every metric (`L1_TO_L1`, `UNPACK_`,
  `MATH_`, `PACK_ISOLATE`, both `L1_CONGESTION` modes) at every marker (`INIT`, `KERNEL`,
  `TILE_LOOP`).
- The only rows outside ±5% are `PACK_ISOLATE`, which flips between about 5160 and 5924
  cycles in both directions (6 up, 2 down). That is the known two-state pack-isolate
  behaviour (`tests/PACK_ISOLATE_FLAKINESS_INVESTIGATION.md`), not a direction tied to
  the change.

CI perf run 38530575647 against main's last green LLK perf nightly
([37721569514](https://github.com/tenstorrent/tt-metal/actions/runs/37721569514),
`5fba608b4f`, 2026-10-08) shows branch numbers higher across the board (unary
`L1_TO_L1` median +14% WH / +10% BH, typecast +53% / +40%, ema about +2%). That shift
comes from main, not from this branch:

- the branch's only edits to `eltwise_unary_typecast_perf.cpp` and `sfpu_ema_perf.cpp`
  are comments, yet typecast moves the most;
- main picked up LLK changes between that nightly and the branch base, including the
  perf-counter infrastructure move (#56320, #56322), FP32 dest-acc drain-before-release
  ordering (#55807), the `_llk_unpack_A_` unpacker check (#59921) and pack/BH stride
  cleanups (#59512, #59486);
- the same-board A/B above, which isolates the commit, is flat.

Main's nightly LLK perf has not been green since 2026-10-08 (cancelled or failed every
night since), so there is no newer main baseline to compare with.

## Follow-ups

1. **Quasar on hardware/emulator.** The Quasar families were checked for collection and
   variant counts, and ran on ttsim in CI. Run them on the Quasar emulator before relying
   on the new comp/qsr_extra/typecast perf numbers.
2. **Unbundle BH/WH typecast perf.** Replace the packed `typecast_case` axis in
   `perf_eltwise_unary_typecast.py` with separate `formats` and `dest_acc` axes, as on
   Quasar, so the comparer can align the two.
3. **Functional counterparts for perf-only families.** `int32`, `comp_uint16`,
   `comp_uint32` and `erfinv_fp32` have perf coverage only on WH/BH.
4. **JUnit publishing.** The `Maximum call stack size exceeded` step error marks every
   LLK e2e job failed, on main as well; it needs an infrastructure fix.
5. **Perf baseline.** Re-check the CI perf comparison once main's LLK perf nightly is
   green again.
