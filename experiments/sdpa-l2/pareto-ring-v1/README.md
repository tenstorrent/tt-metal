# Compressed-KV comparison with the four canonical SDPA choices

This is a read-only analysis of recorded device measurements, not a new benchmark or kernel change. `plot.py` reads 22 explicitly selected records, checks their shared geometry/input parameters, pins each record's SHA256, and emits `measurements.json` and an SVG. No numerical producer or historical evidence is modified. The static plot was rendered and visually checked locally.

## Measurement contract

H10, D128, square noncausal normal BF16 inputs, seed1240, Q256/K512, 110 cores, common per-head forwarding-chain reader. Accuracy is relative L2 against original-input FP64 attention, using 128 recorded Q rows/head and every K/V row. Output is BF16. Throughput is useful attention FLOPs divided by measured combined trace time, including actual device preprocessing but excluding initial upload. BF16 retains two KV slots and FP32 one.

These are separate historical full-chip runs, not the older resident/no-data-movement Pareto measurements, confidence intervals, production dispatch timing, or ring scaling results. Clock variation is a confound for small sequential differences. Some original full-chip timing runs disabled expensive exact preprocessing checks; separate primitive/small qualification does not retroactively supply missing gates. The final research report documents qualification boundaries.

Only the four canonical BF16-input numerical choices are included, A–D. B adds the private Q256 correctness reset and is explicitly not byte-identical frozen FAST code. E/F are the two B8 candidates from the research handoff. G–I show three single-component B4 tradeoffs. J/K show implemented residual alternatives with larger payloads.

The dashed frontier is strict dominance in **normal L2 versus single-chip useful throughput only**. It is not a qualification frontier. In particular I has known coherent-numerator stress failures. Payload is a third axis; a point dominated on the first two may remain valuable for a bandwidth-limited ring. Connecting discrete points does not imply an implementable interpolation.

## BFP4 options

Each K and each V uses the listed encoding; these are not K8/V4 hybrids. Values below are selected normal-input cases, not cross-distribution guarantees.

| ID / encoding and state | 32K L2 % | 32K TFLOP/s | 256K L2 % | 256K TFLOP/s | KV bytes / BF16 |
|---|---:|---:|---:|---:|---:|
| G: native RNE B4, full BF16 compensation | 16.938 | 191.28 | 17.168 | 200.04 | 28.125% |
| H: native RNE B4, FP32 state/native exp | 16.857 | 155.04 | 16.937 | 166.38 | 28.125% |
| I: adaptive B4 E−1/E, denominator-only BF16 | 16.021 | 216.35 | 16.421 | 225.86 | 28.125% |
| J: two B4 residual components, FP32 | 1.513 | 69.65 | 1.494 | 73.00 | 56.250% |
| K: B4+B8 residual components, FP32 | 0.518 | 67.51 | 0.515 | 70.53 | 81.250% |

There is no universal single-component winner: I has the best normal error/throughput tradeoff in this selection but retains known stress failures; G is the fully compensated fast control; H is the FP32-state starting point for further numerical work and avoids BF16 recurrence drift. The wider E−1/E/E+1 search gives16.023%/208.77TF at32K and16.419%/225.34TF at256K; its tiny long-context L2 difference does not establish a meaningful advantage over the simpler search. Native FP32 B4 still has severe outlier/common-mode error (see the native stress suite). We have not combined every repair into a fully qualified single-component scheme. Two B4 components have much better measured normal accuracy but are effectively nine stored bits/value including shared exponents, not a four-bit payload. B4+B8 is thirteen bits/value and not a strong compression ratio.

J/K use Q prescale1.0028, an inverse attention-scale adjustment, and two products per QK and per PV. All preprocessing is included in their combined timing. They reconstruct K/V as a sum of two components; both must be communicated to preserve these results. The reported useful FLOPs do not count the extra products as extra useful attention work.

## Ring payload accounting

For a standard1024-value tile, BF16 is2048B; B8 is1024+64=1088B; B4 is512+64=576B, including native shared exponents. Two B4 components occupy1152B; B4+B8 occupy1664B. Local implementation: `tt_metal/impl/data_format/tile.cpp`, `Tile::get_tile_size()`.

Single-component B4 therefore reduces native KV payload by71.875%, or3.556× versus BF16. It is47.059% smaller than B8. The adaptive exponent search chooses an exponent already carried in the native tile and does not require a second per-tile scale stream.

For H10/K512/D128, combined K+V per chunk is2.5MiB in BF16 versus720KiB in single-component B4. This is payload arithmetic, not measured Ethernet packet size or ring speedup. It assumes compressed components are sent and forwarded without BF16 expansion/requantization; protocol framing, padding, optional transformation metadata, overlap and synchronization may change effective savings. Ideally quantize once at the owning rank and forward unchanged bytes. That is a proposed ring integration contract, not an implemented or benchmarked ring path.

## Q256 fix

`../bfp4-lofi-v2/fast_correction.hpp` resets `ADDR_MOD_7` source/destination increments to zero before calling the frozen maximum-change correction exponential. A preceding compensation replay leaves automatic destination increment2 active, while that exponential explicitly advances its destination. Q256 interleaving then combines the increments incorrectly. Q128's ordering and repeated-KV tests can hide this defect. This does not change chunk sizes, buffering, or the compensation algorithm.

The original-BF16 N4096/H2/C4/Q256/K512 sampled-reference reproducer changes from1206.867% L2 to2.482314%, PCC0.999697080. The private fix also passes recorded replay checks. It is a correctness repair in the experimental harness, not a claim that every production configuration exercises this path; the dedicated integration regression matrix remains a next step.

## Reproduce the figure on macOS

```sh
python3 -B experiments/sdpa-l2/pareto-ring-v1/plot.py
qlmanage -t -s 1440 -o experiments/sdpa-l2/pareto-ring-v1 experiments/sdpa-l2/pareto-ring-v1/sdpa-ring-pareto.svg
python3 -B experiments/sdpa-l2/pareto-ring-v1/export_static.py
```

The square SVG canvas accommodates Quick Look; the crop removes only blank canvas below the plotted figure. No device reservation or Python plotting dependency is required.
