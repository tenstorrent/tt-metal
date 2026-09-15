# V-axis BFP4: standalone Blackhole experiment

Full-chip hardware results now confirm the mapping and targeted quiet-channel benefit; measured results and limitations are below. The four locked variants remain unchanged.

## Mapping

For original BF16 `V[1,H,N,128]`, make an exact device BF16 transpose `VT[1,H,128,N]`, then run the existing native-group RNE BFP4 preprocessor. Shared exponents now span 16 consecutive tokens at one value channel, rather than 16 value channels at one token. The quantized mathematical operand is `decode(quantize(VT)).transpose(-2,-1)`; the acceptance reference remains attention on the **original BF16 Q/K/V**.

Let `Nt=N/32`, chunk `c`, local token tile `k∈[0,16)`, channel tile `d∈[0,4)`. Gather global VT page `head*4*Nt + d*Nt + c*16 + k` into V CB tile `k*4+d`. This deliberately preserves the original **N-major CB tile grid**, while each individual tile contains transposed values. A D-major CB would be wrong with the unchanged PV block loop: the LLK advances the second operand by one page across output-column tiles, regardless of its transpose flag.

Q256/K512/D128, two independent 64-tile K slots and two 64-tile V slots, chain forwarding, barriers, PV subblocks and all CB data formats remain unchanged. V DRAM reads become four contiguous 16-page runs, scattered into the existing CB grid, rather than one contiguous 64-page run. That access-order change and the real transpose/quantization preprocessing must be measured; equal byte counts do not establish equal DM performance.

## Compute change and evidence

- [`matmul_custom.h`](../../../tt_metal/hw/inc/api/compute/experimental/matmul_custom.h): `mm_no_mop_init_short` / `mm_no_mop_reinit_short` pass transpose to **both** unpack and math. The execute function `matmul_block_no_mop` does not consume its transpose argument. Changing only that argument does nothing.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`: init sets within-face transpose in `THCON_SEC0_REG2_Haloize_mode`; execute addresses logical operand B with consecutive output-column tile offsets, independent of transpose. Logical in1 is physical SrcA.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h`: transpose changes face-order address modifiers; full-tile LoFi replay already supports this, as used by QK.
- Frozen MAIN and FAST streaming materialized-PV call sites all use `mm_no_mop_reinit_short(P, V, false, ...)`. An isolated wrapper can force `transpose=true` only when `in1_cb_id==2`, retaining QK and every existing index/stride. Include the API and selected frozen common header **before** defining call-renaming macros; include the selected streaming header under those macros, then undefine them.
- Existing mixed K/V format reconfiguration remains mandatory and unchanged: transpose does not replace unpacker format reconfiguration. FAST's separate direct math restoration is under `SDPA_FP32_STREAMING`, excluded by this BF16-only prototype. Do not extend the wrapper to FP32 without also auditing that restoration.

## Prototype and validation plan

Use a new private full-chip driver, reader and compute wrapper. Compare ordinary-D-group V against token-group V behind an explicit flag; keep native/grid7 exp and MAIN/FAST/denominator-only choices orthogonal. A standalone stable-address BF16 transpose uses the standard TTNN transpose compute kernel, a source-linear reader and transposed-page writer. This avoids temporary-output allocations during trace capture. Preprocessing recomputes transpose then Q/K/V quantization on every combined replay; report transpose and quantization separately without adding them twice.

First smoke: N1024/H2, original-reference all Q rows, normal and constant V, independent K8/V4 and K4/V4 controls, then common V/outliers. Check exact BF16 transpose against the original input and exact quantizer output against the **actual** transposed BF16 tensor. Verify all output finite, two trace replays bit-identical, original inputs immutable, source hashes unchanged before/after. Only then time N32K/256K with fixed Q256/K512 and two slots. Include both mathematical quantized-V error and total operator L2/PCC; no model-quality claims.

This changes grouping direction, not BFP4's three magnitude bits, power-of-two scale restriction, clipping behavior, LoFi arithmetic or BF16 recurrence. It does not make the format identical to NVFP4. Any Sage/NVFP4 precision comparison still needs matched quantizers and accumulation semantics.

## Implemented isolated prototype

`Vtransposed_fullchip.py` and `vtransposed/{compute.cpp,pv_transpose.hpp,reader_chain.cpp,transpose_writer.cpp}` implement the plan without copied or modified frozen headers. `--v-transposed` enables token grouping; omit it for ordinary channel grouping. `--grid7-exp` is independent. MAIN BF16, full FAST BF16, and FAST `--denom-only` retain exactly two KV slots; K8/V4 is the default. V8 is also supported as an axis/control ablation.

The optional `channel_v` distribution multiplies every sixteenth original BF16 V channel by 32, matching the CPU codec study. It reports quiet-channel operator error separately so the large channels do not hide small-channel loss. `--check-preprocess` verifies exact transpose and quantizer equality and reports decoded packed-V L2 against original BF16 V. Source hashes include the selected frozen `.h`/`.hpp`, reference, new files and critical transpose/no-MOP LLK implementations.

Static validation: Python compilation and three stdlib unit tests passed. Tests compare all 24 format/compensation/exp configurations with the prior builder, with/without V transpose, and confirm identical CB allocation, two slots, job assignment, Q256/K512/D128, and only the intended compute define. Symbolic page/index tests cover N1024/32768/262144 and multiple heads/chunks. These are **not** C++ compilation or hardware correctness/performance results.

Suggested first device command (then repeat without `--v-transposed` using a fresh label):

```sh
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v2/Vtransposed_fullchip.py \
  --label vt-k8v4-main-1024-v1 --destination main_bf16 --kv-formats b8_b4 \
  --length 1024 --heads 2 --cores 4 --v-transposed --check-preprocess \
  --distributions normal constant_v channel_v --iters 0
```

N1024 checks every output row against original-input FP64 attention and performs two bit-exact combined trace replays even with `--iters 0`. No prototype device job has been launched by this agent.

Audit follow-up: the source ledger also pins the transitively imported `frontier-accuracy-v1/run.py`. Transpose, original-input immutability and replay checks compare BF16 `uint16` storage bits, including signed zero. Quantizer checks deliberately compare exact decoded numeric values, not encoded BFP bytes, and label that distinction.

## Measured full-chip results (native exp, not grid7)

Hardware records: [32K D](vt-full-32768-D-v1.jsonl), [32K N](vt-full-32768-N-v1.jsonl), [256K D](vt-full-262144-D-v1.jsonl), [256K N](vt-full-262144-N-v1.jsonl), and seed1241 [D](vt-full-32768-D-seed1241-v1.jsonl)/[N](vt-full-32768-N-seed1241-v1.jsonl). Tables below are generated from these JSON ledgers; raw JSON is authoritative.

Configuration: H10/D128, 110 cores, noncausal, Q256/K512, two KV slots, K8-RNE5/V4-RNE, LoFi, **full BF16 numerator and denominator compensation**, native exp (`grid7_exp=false`), read barrier 2. References use original BF16 inputs, all KV and 128 recorded Q rows per head; every output is checked finite. All 16 result rows pass exact preprocessing, original-input immutability, bitwise combined replay, and unchanged-source gates; all six complete ledgers share the same 48 source hashes.

| N | Seed | Distribution | D-group L2 % | N-group L2 % | D PCC | N PCC |
|---:|---:|---|---:|---:|---:|---:|
| 32,768 | 1240 | normal | 12.007 | 12.038 | 0.992816 | 0.992791 |
| 32,768 | 1240 | channel_v | 11.965 | 12.444 | 0.992884 | 0.992266 |
| 32,768 | 1240 | outliers | 16.570 | 16.802 | 0.986282 | 0.985861 |
| 32,768 | 1240 | common_v | 0.513 | 0.513 | -0.002252 | -0.002242 |
| 32,768 | 1241 | normal | 12.067 | 12.100 | 0.992776 | 0.992735 |
| 32,768 | 1241 | channel_v | 11.363 | 11.425 | 0.993545 | 0.993494 |
| 262,144 | 1240 | normal | 12.293 | 12.396 | 0.992753 | 0.992705 |
| 262,144 | 1240 | channel_v | 11.741 | 12.263 | 0.993129 | 0.992576 |

### Quiet channels reveal the benefit

| N | Seed | Quiet L2 D % | Quiet L2 N % | Quiet PCC D | Quiet PCC N | Quiet gain D | Quiet gain N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 1240 | 79.662 | 12.012 | 0.628216 | 0.992818 | 0.502746 | 0.993237 |
| 32,768 | 1241 | 81.500 | 12.148 | 0.615231 | 0.992685 | 0.505699 | 0.996358 |
| 262,144 | 1240 | 80.582 | 12.405 | 0.622958 | 0.992714 | 0.506890 | 0.994842 |

These are the 120 unamplified channels, not a fitted or shifted reference. Token grouping reduces their error by about 85% and removes the roughly 0.5 gain attenuation, even though **global channel-outlier L2 slightly worsens**. Large channels dominate the global norm; the quiet-channel metric is essential. Normal/random-outlier errors do not improve, so this is a targeted channel-imbalance repair, not a universal precision upgrade.

Common V is not repaired: centered-residual L2 is 2242.717% / 2242.667% (same original-V mean subtracted from both actual/reference), versus BF16 output-floor L2 0.029% in the original metric. Its near-zero PCC and 0.513% original L2 must not be presented as good preservation of the small signal. This native-exp experiment does not establish what grid7 or explicit V centering would achieve.

### Real device timing

Milliseconds, seven measured trace replays after three warmups. Stage timings and totals are separate measurements; do not add the transpose timer to a combined total that already includes it.

| N | Seed/input | Attention D | Attention N | Prep D | Prep N | N transpose | Combined D | Combined N |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 1240/normal | 27.642 | 27.649 | 1.172 | 1.635 | 0.518 | 28.777 | 29.221 |
| 32,768 | 1240/channel_v | 27.647 | 27.647 | 1.173 | 1.634 | 0.519 | 28.774 | 29.226 |
| 32,768 | 1240/outliers | 27.660 | 27.654 | 1.176 | 1.633 | 0.523 | 28.776 | 29.229 |
| 32,768 | 1240/common_v | 27.646 | 27.673 | 1.174 | 1.631 | 0.520 | 28.775 | 29.226 |
| 32,768 | 1241/normal | 27.684 | 27.656 | 1.185 | 1.635 | 0.517 | 28.795 | 29.250 |
| 32,768 | 1241/channel_v | 27.657 | 27.653 | 1.171 | 1.634 | 0.517 | 28.772 | 29.226 |
| 262,144 | 1240/normal | 1787.207 | 1775.112 | 8.972 | 12.641 | 3.729 | 1807.017 | 1809.630 |
| 262,144 | 1240/channel_v | 1735.009 | 1776.420 | 8.951 | 12.629 | 3.722 | 1763.790 | 1810.381 |

Normal N32,768: measured combined delta 0.444ms (+1.544%); useful attention throughput 198.9→198.8 TFLOP/s, combined 191.0→188.1 TFLOP/s.

Normal N262,144: measured combined delta 2.614ms (+0.145%); useful attention throughput 196.9→198.2 TFLOP/s, combined 194.7→194.4 TFLOP/s.

At 32K the attention times are essentially unchanged and most incremental cost is the real ~0.52 ms transpose. At 256K the transpose is ~3.73 ms, but sequential end-to-end medians vary more: D-group channel V attention is 52.2 ms faster than D-group normal, despite unchanged geometry. Consequently the larger channel-case N-minus-D difference is not isolated evidence of gather overhead. Interleaved same-input repeats are needed before assigning those differences to layout. No claim of improved model quality is made.

## Bounded barrier follow-up

The existing `--read-barrier-tiles` setting is a reasonable bounded tuning knob, but **it applies to both K and V DRAM reads at the chain root**, not just the four-run V gather. With 64 pages per component/chunk, barrier2 waits after 32 pairs; barrier8/16 permit 8/4 batches; barrier0 leaves only the unconditional final barrier. Every setting retains the final read barrier before chain forwarding/publish, so reserved CB ownership, tile count, Q/K chunks and two-slot buffering remain unchanged. The asynchronous read API handles injection readiness; correctness must still be requalified at each setting.

Compare paired D/N at barriers 2, 8 and 16 on N32768 normal, then test only a promising setting at N262144. Interleave/reverse candidate order and require bit-identical outputs across barrier choices within each axis: this changes scheduling, not arithmetic. Record every combined trace sample and pin the same sources. Do not call an improvement V-specific unless the D/N comparison isolates it. The setting cannot optimize the separate transpose preprocessor, whose ~0.52/3.73 ms cost is not controlled by this reader macro.

Variable device clocks are a plausible confound, not proven causal attribution: a separate later live snapshot during the HiFi2 LUT256K workload showed AICLK 1306 MHz versus a 1350 MHz maximum/earlier active snapshot, with 168 W and 74.2°C. It was not simultaneous telemetry for these V-axis records. Use interleaved paired timing, no clock/power overrides, and no continuous host telemetry process inside timed intervals. The new paired timing experiment will measure, not assume, any benefit.

### Paired timing driver

`paired_vaxis_timing.py` reuses unchanged `Vtransposed_fullchip.build` for D/N × barriers 2/8/16. It qualifies all six candidates against the same original inputs before capturing any benchmark traces, requires identical output hashes across barriers within each axis, and retains every candidate buffer/trace. Each pair of rounds reverses the same order, then rotates the next pair's start; sample records include actual order and individual combined latencies. No logging or telemetry occurs between candidates within a round. Final replay/immutability/source checks precede the summary; trace cleanup attempts every retained trace even after a failure.

Default N32768/H10/C110 tensor storage is approximately 2.96 GiB for six independent builders. N262144 is approximately 23.67 GiB, plus trace and program reserve; the default hard budget is 24 GiB, and a live DRAM-free check additionally requires 10% headroom. `--barriers 2 8` reduces the conditional long-context test to four candidates. The prototype does not share or retarget source buffers behind the unchanged builder.

```sh
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v2/paired_vaxis_timing.py \
  --label vt-paired-32k-v1 --length 32768 --heads 10 --cores 110 \
  --barriers 2 8 16 --iters 7 --warmup 3
```

Python compilation and three stdlib tests pass: bounded candidate/memory accounting, complete/reversed round ordering with balanced positions over a full cycle, and all-trace cleanup under injected failures. The completed parent-run paired hardware results follow.

## Completed interleaved barrier result

[Paired 32K ledger](vt-paired-32k-v1.jsonl): same original normal input, N32768/H10/C110, K8/V4, full BF16 compensation, native exp, two KV slots. All six candidates were qualified before capture. The run records seven measured complete interleaved rounds after three warmup rounds; adjacent round pairs reverse order and successive pairs rotate their start. All outputs are bit-identical across barrier choices **within each V axis**, unchanged versus initial qualification, and match the earlier single-candidate output hashes. Exact preprocessing, original-input immutability, 56 source pins and all six trace releases pass.

| Read barrier tiles | D combined ms | N combined ms | N−D ms | N overhead % |
|---:|---:|---:|---:|---:|
| 2 | 28.825 | 29.286 | 0.461 | 1.598 |
| 8 | 28.827 | 29.269 | 0.442 | 1.535 |
| 16 | 28.829 | 29.270 | 0.440 | 1.527 |

These are medians of **combined** traces, including all device preprocessing. N grouping costs about 1.5–1.6% here. Barriers 8/16 do not provide a meaningful demonstrated win: N medians differ from barrier 2 by only 0.017/0.016 ms (under 0.06%), while D does not improve. No production tuning change is justified by this bounded run. The existing barrier 2 is a reasonable retained control; the axis overhead largely reflects the separately measured real transpose. This establishes neither a universal barrier optimum nor a B8/B4 format comparison.

## BFP8 controls through N262144

Completed [32K D](vt-b8-full-32768-D-v1.jsonl), [32K N](vt-b8-full-32768-N-v1.jsonl), [256K D](vt-b8-full-262144-D-v1.jsonl), and [256K N](vt-b8-full-262144-N-v1.jsonl) controls change V from B4 RNE to B8-RNE5, retaining K8, full BF16 compensation, native exp, barrier 2, Q256/K512, two slots and H10/C110. All eight results pass exact preprocessing/replay/immutability/finite/source gates; their 48 source hashes match the earlier B4 V-axis runs. Original-input FP64 reference uses all KV and 128 sampled Q rows per head. Five timed replays after three warmups; these format comparisons are sequential, **not** the interleaved barrier experiment above.

| N | Input | B8 D L2 % | B8 N L2 % | B8 D PCC | B8 N PCC |
|---:|---|---:|---:|---:|---:|
| 32,768 | normal | 3.151 | 3.148 | 0.999533 | 0.999534 |
| 32,768 | channel_v | 3.478 | 3.244 | 0.999401 | 0.999479 |
| 262,144 | normal | 3.735 | 3.730 | 0.999538 | 0.999538 |
| 262,144 | channel_v | 3.921 | 3.727 | 0.999267 | 0.999340 |

| N | Quiet B8 D L2 % | Quiet B8 N L2 % | Quiet D PCC | Quiet N PCC |
|---:|---:|---:|---:|---:|
| 32,768 | 11.454 | 3.142 | 0.993935 | 0.999536 |
| 262,144 | 11.683 | 3.731 | 0.993922 | 0.999538 |

Thus token-axis grouping also repairs the B8 channel-imbalance penalty: quiet channels return close to the normal-input accuracy band, rather than remaining around 11–12% error. Normal L2 hardly changes with axis. This does not resolve the remaining native-exp/LoFi/recurrence error; the N1024 constant-V control previously remained 0.633% for both axes.

### Real B8 timing and comparison with B4

| N | Input | Attention D ms | Attention N ms | Prep D ms | Prep N ms | N transpose ms | Combined D ms | Combined N ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | normal | 27.754 | 27.709 | 1.189 | 1.653 | 0.518 | 28.850 | 29.300 |
| 32,768 | channel_v | 27.750 | 27.709 | 1.190 | 1.652 | 0.520 | 28.849 | 29.314 |
| 262,144 | normal | 1814.849 | 1813.531 | 9.127 | 12.787 | 3.749 | 1845.329 | 1846.972 |
| 262,144 | channel_v | 1799.167 | 1808.254 | 9.117 | 12.766 | 3.733 | 1828.294 | 1844.252 |

| N | V format, N-axis | Normal L2 % | Channel quiet L2 % | Normal combined ms |
|---:|---|---:|---:|---:|
| 32,768 | B4 | 12.038 | 12.012 | 29.221 |
| 32,768 | B8 | 3.148 | 3.142 | 29.300 |
| 262,144 | B4 | 12.396 | 12.405 | 1809.630 |
| 262,144 | B8 | 3.730 | 3.731 | 1846.972 |

B8 N gives substantially lower error at similar observed speed: sequential normal combined medians are about 0.27% higher at 32K and 2.06% higher at 256K than B4 N. Variable clocks and run ordering prevent treating these small differences as an isolated format cost. Neither format gets a different LoFi matmul replay count in this kernel.

B4 still has a concrete capacity/traffic advantage: 576 versus 1088 bytes per V tile, 47.1% less V storage, 23.5% less total K8+V traffic, and 64 KiB less V-CB allocation per core with two 64-tile slots. Prefer B8 N as the numerical reference candidate for this measured geometry; justify B4 with a demonstrated memory/bandwidth constraint or real-model tolerance, not an assumed compute-speed multiplier. Do not claim universal dominance: matched broad outlier/common-mode qualification, more seeds and paired B8/B4 timing remain missing, and these operator tests do not establish model quality.
