# TTGenerator device-time optimization plan

Source profile: `generated/profiler/reports/2026_07_01_05_54_45/ops_perf_results_2026_07_01_05_54_45.csv`
(og generator, `use_torch_phase_fallback=True`). Config: `upsample_rates=[10,6]`,
`upsample_initial_channel=512`, `resblock_kernel_sizes=[3,7,11]`, `gen_istft_hop_size=5`,
`T_x=5` mel frames → `har_time_len=1500`. Two resblock resolutions: **(T=50, C=256)** stage 0 and
**(T=301, C=128)** stage 1.

**Goal:** lowest device time, **no PCC drop** (pipeline PCC must stay > 0.99; demo audio must stay
clear), no regression to the fallbacks the demo relies on.

## Where the 4,675 µs of device-kernel time goes

| Bucket | µs | % | Note |
|---|---:|---:|---|
| **FP32 elementwise** (Binary+Ternary+Unary+Reduce) | 1,682 | 36 % | dominant |
| — of which: **resblock/AdaIN main path (does NOT need fp32)** | **1,466** | **31 %** | the prize |
| — of which: harmonic/source/STFT path (genuinely fp32) | 216 | 5 % | leave alone |
| Conv2d (conv1d/convT via conv2d) | 876 | 19 % | one HiFi4 conv = 143 µs on 16 cores |
| FillPad (tile-pad fill from non-32-aligned T=50/301) | 320 | 7 % | scales with op count |
| Matmul (24× fp32 AdaIN style proj + 2× STFT) | 254 | 5 % | style proj M=1, BW-bound |
| Reduce (InstanceNorm mean/var) | 223 | 5 % | keep fp32 accum |
| Slice / Halo / Reshape / Typecast / misc | ~314 | 7 % | reshape churn 75 µs |

FP32 elementwise by shape (Y=length, X=channels):

| shape | cnt | µs | what it is |
|---|---:|---:|---|
| `50 × 256` | 66 | 744 | stage-0 resblock activations (snake, normalize, adds, addcmul) |
| `1 × 256` | 228 | 462 | **per-channel AdaIN coef/shift fold** (`_fold_adain_coef_shift_fp32`) |
| `301 × 128` | 12 | 195 | stage-1 resblock activations |
| `1500 × 32` (×1 / ×9) | 84 | 192 | SineGen / harmonic source — **needs fp32** |
| `1 × 128` | 12 | 65 | AdaIN coef/shift fold at C=128 |

Root cause of the fp32 storm: `tt_generator.py:528` `target_dtype = har_nlc.dtype`. `har_nlc` is the
STFT output (fp32), so the **entire** upsample/resblock decoder path is cast to fp32 even though its
weights are bf16 and only the InstanceNorm statistics need fp32.

Note this is a **short** input (T_x=5); fixed per-op overhead (the 228 tiny `1×256` ops, FillPad,
dispatch) is relatively larger here than on production-length utterances. Op-count reductions win
most at short length; the dtype change wins at every length.

---

## Tier 1 — numerically exact / PCC-neutral (do first, no validation risk)

### 1. Pre-fold InstanceNorm affine + `(1+γ)` into the style-projection (`fc`) weights — biggest clean win  ✅ DONE
> **Implemented** in `tt_adain_1d.py` (`preprocess_tt_adain_1d` folds `w`,`b`,`+1` into `fc'` in fp64;
> `TTAdaIN1d.forward` `affine_folded` path = one `addcmul`). **Measured (T_x=5, 1 cold forward):**
> device kernel **4,675 → 4,450 µs (−225 µs, −4.8 %)**; BinaryNg ops **605 → 413 (−192)**, every other
> op-code count identical; fp32 elementwise −369 µs. **PCC unchanged:** pipeline 0.9959 (gate 0.99),
> resblocks 0.9999, noise_res 0.99999. Note the fold-op count is *T-independent*, so this is a
> roughly-fixed ~0.22 ms saving — proportionally largest at short length; longer utterances need #6.

`_fold_adain_coef_shift_fp32` recomputes, per AdaIN call, `coef = (1+γ)·w_in` and
`shift = β + (1+γ)·b_in` — 4 tiny fp32 elementwise ops on `[1,C]`. That is the ~527 µs of `1×256` +
`1×128` ops. Both `coef` and `shift` are **affine in the style vector `s`** (γ, β are linear in `s`;
`w_in`, `b_in` are static), so absorb `w_in`, `b_in` and the `+1` into modified `fc'` weight/bias at
preprocess (in fp32 on host):
```
coef  = W_g'·s + c_g'   with  W_g' = w_in ⊙ W_γ ,  c_g' = w_in·(1+b_γ)
shift = W_b'·s + c_b'   with  W_b' = W_β + b_in⊙W_γ , c_b' = b_β + b_in + b_in·b_γ
```
Per-forward AdaIN then does `fc'(s) → (coef, shift)` (one matmul, already present) + one `addcmul`.
Removes the whole per-channel fold chain. Algebraically identical → **zero PCC risk.**
Est. save ~400–500 µs.

### 2. `activations_in_l1` made safe-by-construction + measured  ✅ DONE
> **Implemented** a per-forward footprint guard (`TTGenerator._loop_l1_safe`): the upsample/resblock
> loop stays L1-resident only when a conservative estimate of its peak activation footprint fits the
> device's usable interleaved L1 (`get_max_worker_l1_unreserved_size() × cores × 0.35`, `10×`
> concurrency); otherwise it falls back to DRAM. This removes the OOM hazard so the flag is safe on any
> input. Measured (p150: 168.6 MB usable L1): analytic L1 cutoff **≈ T_x 192 (~2.4 s audio, fp32)**;
> **T_x=5 → L1**, **T_x=600 → DRAM**, both run cleanly. **Device time (T_x=5, on top of #1):
> 4,450 → 4,288 µs (−162 µs, −3.6 %)**; loop activations now L1-interleaved (1,036 ops), only the 73
> harmonic/STFT ops stay in DRAM as intended. **PCC:** L1's effect on ref-PCC = **+0.0036** (neutral;
> L1 is *not* bit-identical to DRAM — some block-sharded convs / instance-norm repartition for L1
> inputs — but does not degrade accuracy). Demo default kept **opt-in** (`--l1-activations`); the guard
> protects it when enabled. Cumulative **#1+#2 = −387 µs (−8.3 %)** from the 4,675 µs baseline.

### 3. Fuse the `xs / num_kernels` scale and the pre-`ups` leaky_relu  ⚠️ LOW VALUE — deferred
Profiled cost is ~5 µs: only 2 `multiply(xs, 1/num_kernels)` ops/forward and they run on the small
bf16 stage tensors. Folding `1/num_kernels` into the next op's weights (conv_post / `ups[i+1]`, using
`leaky_relu(c·x)=c·leaky_relu(x)` for `c>0`) is exact but needs a preprocess weight change for ~5 µs.
Not worth the invasiveness now; revisit alongside #6.

### 4. Cut reshape/permute churn in fused InstanceNorm  ❌ NOT APPLICABLE
The fused (`ttnn.layer_norm`) InstanceNorm path is gated to `L % 32 == 0`. The generator's lengths
(50, 301) are **not** `%32==0`, so it uses the **legacy** reduce-based NLC path (`mean/sub/pow/mean/
rsqrt/mul`) — which has **no** permutes. Profiled Transpose is 5 µs total and lives in the STFT path,
not the norm. There is no permute churn to remove here.

### 5. Guard redundant typecasts  ⚠️ MAIN TARGET BLOCKED BY PCC — not taken
Profiled: 74 Typecast ops (64 µs), of which **72 are the coef/shift `fp32→bf16` casts** (36 at C=256,
36 at C=128) in the bf16 resblock AdaINs (`x` becomes bf16 after `ups`, so the fp32 style-linear's
coef/shift are cast down). The snake `alpha` casts I expected don't exist (resblock `alpha` is bf16,
matching bf16 activations). Removing the 72 casts by packing the style matmul straight to bf16
(`ttnn.linear(dtype=...)`) was **tried and reverted**: the matmul packer's fp32→bf16 rounding differs
from `ttnn.typecast` and dropped pipeline PCC **0.99592 → 0.99530** (still > 0.99, but a real
regression vs the no-drop constraint). The only genuinely-exact typecast left (`f0 → fp32` when
already fp32, in `_harmonic_source_path`) is ~1.8 µs and entangled with the unsqueeze/dealloc logic —
negligible. **These casts are a symptom of the accidental fp32→bf16 mix and disappear for free under
#6** (a consistently-bf16 main path has no rounding-mismatch to lose PCC to). Deferred to #6.

---

## Tier 2 — real device-time wins, require PCC re-validation

> Validate against `test_tt_generator_pipeline_pcc` (ref `har` injected, must stay > 0.99) **and**
> listen to the demo `.wav`. Do #1 first, then re-measure — the memory note that "bf16 main path is
> capped at ~2 %" is most likely explained by the affine fold (#1) having stayed fp32 and dominating.

### 6. Run the main decoder path in bf16  ❌ INVESTIGATED — NOT VIABLE
Investigated with a deterministic injected-har harness (mirrors `test_tt_generator_pipeline_pcc`,
toggling the noise-path and main-`x` dtypes). Findings:
- The **main resblocks are already bf16** (`x` becomes bf16 after `ups`), so there's no gain there.
- The remaining ~940 µs of fp32 (`50×256` + `301×128`) is the **noise_res / harmonic path**, which is
  **precision-load-bearing**. Converting it to bf16 drops pipeline PCC **0.99592 → 0.98836** (below
  the 0.99 gate): the InstanceNorm normalizes the small-amplitude harmonic signal (RMS ≈ 0.06) and
  amplifies its relative bf16 error before the residual add — the loss is *not* truncated away.
- Starting `x` in bf16 adds nothing (0.98836 → 0.98797).

Conclusion: the generator's fp32 floor is set by the fp32 noise_res + InstanceNorm and cannot be
lowered by dtype without breaking the PCC gate. Matches the prior `[[project_kokoro_generator_perf]]`
note. **Do not pursue bf16 main path.** Remaining device-time levers are non-dtype (see #7/#8).

### 7. Lower conv math fidelity  ❌ INVESTIGATED — NO BENEFIT (BW-bound)
The two `ups` ConvTransposes dominate the conv bucket: `ups[0]` (512→256, stride 10) = 143.5 µs on
16 cores, `ups[1]` (256→128, stride 6) = 77.2 µs on 80 cores. PCC sweep found **ups=HiFi3 is
PCC-neutral (0.99592 → 0.99606, no drop)** — but profiling showed **identical timing** (ups[0] still
143.5 µs): these convs are **weight-BW-bound** (reading the 10.5 MB fp32 `ups[0]` weight across 16
cores), not compute-bound, so dropping fidelity passes does nothing. HiFi2/LoFi drop PCC and still
don't help timing. Reverted. conv_post fidelity is likewise neutral but conv_post is only 6.7 µs.

### 8. Conv weight/core levers  ❌ INVESTIGATED — NOT WORTH IT (not actually BW-bound)
bf16 `ups` weights were **measured** (not just estimated): they save only **−11 µs** (ups convs
220.7 → 208.8 µs), not the ~110 µs a BW-bound halving would predict. So the `ups` convs are **not**
weight-BW-bound — they're dominated by ConvTranspose staging/halo overhead and the fixed cost of
16-core serialization on the 50-row `ups[0]` output. bf16 weights barely help and cost PCC (0.99592 →
0.99404). Not worth it; reverted. Core count: `ups[0]` is pinned at 16/110 cores by its 50-row output
(a ConvTranspose can't spread 50 rows further, no exact re-shard knob found). The conv bucket has no
worthwhile win at any PCC bar.

### Conclusion — generator device-kernel time is at its practical floor
After #1 + #2 (−387 µs / −8.3 %, both PCC-neutral), every further device-kernel lever tested fails:
bf16 activations (#6) break the 0.99 gate (0.98836); conv fidelity (#7) is BW-bound (no benefit);
bf16 conv weights barely help (−11 µs) because the convs are ConvTranspose-overhead-bound, not
precision-bound. **Relaxing the PCC bar does *not* unlock a meaningful device-kernel win.** #1 + #2 is
the practical ceiling for generator device time.

The real remaining opportunity is **wall-clock, not device-kernel** (PCC-safe, 0 device change):
- **#10 trace / program cache** — all 1,570 ops are `PROGRAM CACHE HIT=False`, no trace; the op-to-op
  gap was ~13.8 ms vs 4.7 ms of device kernel. This is the biggest real demo speedup.
- **#9 device-side SineGen** — drop the CPU phase-fallback device↔host round-trip (memory says the
  cumsum path landed; re-validate the `sine_merge > 0.98` gate).

### 9. Drop `use_torch_phase_fallback` (device-side SineGen) — wall-clock, not device-kernel
The CPU phase fallback forces device→host→device round-trips (the ~13.8 ms op-to-op gap in this
run). The memory note says the device SineGen cumsum path landed; if it holds the `sine_merge > 0.98`
gate, disabling the fallback removes the serialization stall and speeds the demo end-to-end. Keep on
only if audio degrades.

---

## Tier 3 — dispatch / wall-clock (0 device-kernel, 0 PCC impact)

### 10. Program cache / trace capture
**All 1,570 ops report `PROGRAM CACHE HIT = False`** and there is no trace — every op is dispatched
cold. Capture a `ttnn` trace for the generator (or at least reuse the program cache across the
utterance loop). Large host-time win, no device-kernel or PCC impact.

### 11. bf16 style-projection weights
The 24 fp32 style matmuls (`[1,128]→[1,512]`, 86 µs) are M=1 and weight-BW-bound; bf16 `fc` weights
halve the bytes. Fold into #1's pre-computed `fc'`. Validate PCC.

---

## Suggested order

1. #1 (affine pre-fold) — biggest exact win, and it de-risks #6.
2. #2 (`activations_in_l1`) + #5 (typecast guards) — trivial, exact.
3. #3, #4 — exact op-count/layout cleanups.
4. Re-profile. Then #6 (bf16 main path) with PCC gate, then #7/#8 (conv), then #9.
5. #10 (trace) + #11 for the end-to-end demo.

Rough envelope: Tier 1 alone ≈ −0.9–1.1 ms (→ ~3.6 ms). Adding #6 ≈ −0.75–0.9 ms more (→ ~2.7–2.9 ms),
i.e. **~35–40 % device-time reduction** if #6 holds PCC.
