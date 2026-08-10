# moe_fused_swiglu performance — 8×11 vs 8×12

Device-kernel duration measured on a Blackhole p150a. Each value is the median of seven Tracy-profiled dispatches after one warmup. The target is BF16 row-major activations, BFP4 tiled ND-sharded weights, BFP8 tiled output, K=7168, N=2048, and capacity 5120. `M` is read from a device-resident expert-count tensor.

The TTNN grid tuple is `(columns, rows)`, so the requested 8-row × 11-column and 8-row × 12-column grids are written `11x8` and `12x8` below. Both runs use the retained Python `ProgramDescriptor` through `ttnn.generic_op`:

- `11x8`: 88 workers, column dispatch; the device exposes an `11x10` compute-with-storage grid.
- `12x8`: 96 workers, row dispatch with `FABRIC_1D` and `MUX`; the device exposes a `12x9` compute-with-storage grid.

Delta is `(12x8 / 11x8 - 1)`. Negative means the 12-column grid is faster.

| M | 11x8 (µs) | 12x8 (µs) | delta |
|---:|---:|---:|---:|
| 0 | 3.553 | 3.642 | +2.505% |
| 64 | 78.226 | 76.817 | -1.801% |
| 128 | 85.496 | 90.065 | +5.344% |
| 256 | 109.204 | 110.136 | +0.853% |
| 512 | 186.028 | 186.608 | +0.312% |
| 1024 | 331.859 | 329.727 | -0.642% |
| 2048 | 624.076 | 624.224 | +0.024% |
| 4096 | 1206.698 | 1203.583 | -0.258% |
| 5120 | 1498.111 | 1494.799 | -0.221% |

Across all nonzero M values, the aggregate duration changes by -0.091%; across M=1024–5120 it changes by -0.230%. The additional eight workers therefore provide no material overall speedup for this shape. M=128 regresses by 5.3%, and the larger-M differences are below 0.7%. The 12x8 run was noisier at M=256 and M=512 (min-to-max spreads of 9.56% and 5.41%, respectively), so those two small deltas should not be treated as significant.

## Reproduction

The focused benchmark is `tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_grid_perf.py`. From the repository root, use the checkout's virtual environment and native build:

```bash
env \
  PATH="$PWD/python_env/bin:$PATH" \
  VIRTUAL_ENV="$PWD/python_env" \
  TT_METAL_HOME="$PWD" \
  PYTHONPATH="$PWD" \
  LD_LIBRARY_PATH="$PWD/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  MOE_GRID=11x8 \
  MOE_DISPATCH_AXIS=col \
  MOE_GRID_REPS=7 \
  MOE_GRID_MANIFEST=/tmp/moe_grid_11x8_manifest_7rep.json \
  ./scripts/run_safe_pytest.sh --profile --no-precompile \
    tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_grid_perf.py -s

env \
  PATH="$PWD/python_env/bin:$PATH" \
  VIRTUAL_ENV="$PWD/python_env" \
  TT_METAL_HOME="$PWD" \
  PYTHONPATH="$PWD" \
  LD_LIBRARY_PATH="$PWD/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  MOE_GRID=12x8 \
  MOE_DISPATCH_AXIS=row \
  MOE_GRID_REPS=7 \
  MOE_GRID_MANIFEST=/tmp/moe_grid_12x8_manifest_7rep.json \
  ./scripts/run_safe_pytest.sh --profile --no-precompile \
    tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_grid_perf.py -s
```

The raw profiler CSVs from this run are:

- `generated/profiler/reports/2026_08_10_07_53_33/ops_perf_results_2026_08_10_07_53_33.csv` (11x8)
- `generated/profiler/reports/2026_08_10_07_54_02/ops_perf_results_2026_08_10_07_54_02.csv` (12x8)

The 12x8 row-dispatch suite also passes its grid probe, numerical correctness and tail cases. A direct generic-op/public-C++ comparison is bitwise equal for the real output rows at the target K=7168, N=2048 and M=288.

## Geometry follow-up: logical 8H×12K transpose

The physical 12x8 rectangle can also be interpreted as eight hidden groups by twelve K-reduction groups, mapping logical `(h, k)` to physical `(k, h)`. The generic descriptor supports this experiment with `transpose_grid_axes=True`, exposed by the benchmark as `MOE_LOGICAL_TRANSPOSE=1`. This rotates the activation multicast and remaps all collective peers while leaving the production C++ operation unchanged.

The transpose does create more L1 headroom, but it changes two more important properties: the reduction grows from eight to twelve contributors, and `KGROUPS != M_BLOCK` disables the full-M W-down row schedule.

| logical geometry | `KR_PAD` | `HN_PAD` | GU chunks | gather pages | full-M row schedule | CB bytes/core | free CB budget |
|---|---:|---:|---:|---:|:---:|---:|---:|
| 11H×8K | 28 | 6 | 3 | 48 | yes | 1,389,120 | 72,256 |
| 12H×8K | 28 | 6 | 3 | 48 | yes | 1,399,488 | 61,888 |
| 8H×12K, transposed | 19 | 8 | 2 | 96 | no | 1,364,160 | 97,216 |

Seven-repetition medians show that the transposed geometry is not competitive:

| M | 12H×8K (µs) | 8H×12K (µs) | transposed delta |
|---:|---:|---:|---:|
| 0 | 3.642 | 3.596 | -1.3% |
| 64 | 76.817 | 98.504 | +28.2% |
| 128 | 90.065 | 109.893 | +22.0% |
| 256 | 110.136 | 151.170 | +37.3% |
| 512 | 186.608 | 230.986 | +23.8% |
| 1024 | 329.727 | 381.473 | +15.7% |
| 2048 | 624.224 | 691.190 | +10.7% |
| 4096 | 1203.583 | 1302.662 | +8.2% |
| 5120 | 1494.799 | 1608.079 | +7.6% |

The aggregate regression is +11.1% across nonzero M and +9.1% across M=1024–5120. A focused M=256 stage profile also found the transposed slowest-core gate/up stage about 7.5% longer, reduction about 26.5% longer, and down/phase-2 stages over 2x longer. A hybrid that uses twelve-way K only for phase 1 therefore has no measured phase-1 win to preserve.

The other factorizations of 96 workers are less natural on a physical 12x8 rectangle and also lose the tuned eight-row schedule. The only moderately plausible one is 16H×6K: its modeled CB footprint is 1,420,480 B/core (40,896 B free) and W-down remains resident, but its 16-wide and 6-deep collectives require a non-rectangular remap and `KGROUPS != M_BLOCK` still disables the fast full-M phase 2. Wider hidden factorizations such as 24H×4K and 32H×3K create 341 KB and 207 KB of nominal headroom only by dropping W-down residency, which forces the weights to be reread for every M block. The eight-way K reduction therefore fixes the useful 96-core factorization at 12H×8K unless phase 2 itself is redesigned.

The transposed sweep is in `generated/profiler/reports/2026_08_10_08_59_14/ops_perf_results_2026_08_10_08_59_14.csv`; the focused current/transposed M=256 stage captures are `generated/profiler/reports/2026_08_10_09_02_19` and `generated/profiler/reports/2026_08_10_09_02_36`.

The extra L1 does not fund a known high-value optimization. It admits a fourth `cb_h` slot (+69,632 B), previously measured as neutral, and another output slot (+26,112 B), while double-buffered gate/up weights, a third x slot and `M_BLOCK=16` remain far over budget. The BF16 gate/up accumulation region needs about 98 KB, just above the 97,216 B available, and its associated K-chunking path was already a regression.

For the retained 12H×8K geometry, the balanced hidden split is `6,6,6,6,5,5,5,5,5,5,5,5`, but resident W-down flow control allocates twelve fixed six-row slots. Publishing the resident payload as 64 real hidden rows instead of 72 padded rows would save `8 * 3 * 576 = 13,824` B/core and raise free CB budget to 75,712 B without changing the useful geometry. That compact-resident protocol is the most defensible L1 follow-up, although it does not by itself identify a profitable consumer for the reclaimed space.
