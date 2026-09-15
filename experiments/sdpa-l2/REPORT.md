BF16 SDPA relative-L2 investigation — 2026-09-08

Latest performance study (2026-09-11): [FP32 subtraction block tuning and headroom](fp32-block-perf/REPORT.md).
The selected HiFi4/approximate-exp/full-FP32-subtraction algorithm sustains about
55 useful TFLOP/s at 32K/64K/128K/256K; 128/1024 remains the best-tested block size.
Normal L2 is 0.179–0.180%. Activity profiles identify limited FPU/SFPU overlap
as a performance opportunity. The tested patch is saved; the retained build is restored.

Latest investigation (2026-09-11): [Full-HiFi4 accuracy and performance](accuracy-investigation/REPORT.md).
HiFi4 plus FP32 subtraction substantially improves accuracy, but residual outlier
failures are reproduced by standalone main HiFi4 matmul. Expanded qualification
passes 162/170 supported cases; fresh-seed holdout passes 11/12. Fresh main/improved
timings, diagnostic patches, and restoration/regression evidence are included.
The retained improved implementations, not the diagnostic patches, remain built.

Previous numerical qualification (2026-09-10): [qualification-v1/REPORT.md](qualification-v1/REPORT.md).
All 1320 synthetic mode cases are accounted for; neither improved candidate passes
the agreed numerical/structural contract. Performance and model-score evaluation
were excluded. The retained improved build was restored and verified afterward.

Historical baseline report. The latest BF16 loop optimization and currently built
candidate are documented in [bf16-perf-v1/REPORT.md](bf16-perf-v1/REPORT.md).
The unchanged improved FP32 implementation is documented in [perf-v4/REPORT.md](perf-v4/REPORT.md).

Reproduced. On unmodified main, BF16/HiFi2 streaming SDPA produces 2.51% relative L2 at 4K keys and 2.84% at 32K, despite PCC above 0.9996. The tested 256K configuration is substantially worse than the reported ~3%: 18.94% for short-Q attention, and 18.21% on the tail of full causal attention. These are synthetic, reproducible Blackhole results, not a reproduction of an unspecified Galaxy model workload.

The executable repro is [repro_sdpa_l2.py](../../tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py). The operator sources and remote build have been restored to main. Experimental changes are saved as patches, not installed fixes.

Environment and definition

- Commit: `2ba6fc2339d53300ae87c5202f335ef56492cfb3`.
- Machine: `yyzo-bh-08`, IRD job `212869`, one Blackhole P100A, firmware 19.12.0, KMD 2.9.0. Release build, Clang 20.1.8; PyTorch 2.11.0+cpu.
- BF16 Q, K, V and output; batch 1, head dimension 128, query chunk 128, KV chunk 512 unless noted. Standard normal inputs, seed 1234; seed 1235 corroborates the trend. Dropout/masks/sinks are absent in the rectangular test.
- HiFi2, `math_approx_mode=True`, `fp32_dest_acc_en=False`, `packer_l1_acc=False`, `exp_approx_mode=True`.
- Relative L2 means `100 * ||actual-reference||_2 / ||reference||_2`, not squared L2, absolute RMSE, or elementwise relative error.
- Reference uses the identical already-rounded BF16 inputs, then FP64 blockwise online softmax. Dense FP64 softmax cross-checks pass at `rtol=atol=1e-12` for rectangular and causal cases. Thus input-quantization disagreement is excluded.
- The BF16 output-rounding-only floor is 0.165–0.168% for the normal rectangular cases. This is a measured rounding baseline, not an NVIDIA measurement. No NVIDIA or multi-chip Galaxy tests were run; LayerNorm was not tested.

Streaming selection is explicit in [sdpa_program_factory.cpp](../../ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp): `can_use_streaming_compute(fp32_dest_acc_en)` returns `!fp32_dest_acc_en`. Compile-time argument 30 selects the streaming branch in [sdpa.cpp](../../ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp). Setting FP32 destination accumulation switches to the standard path; those comparisons must not be described as improved streaming results.

Measured baseline

Rectangular attention computes all 128 query rows against all keys, with one head. Full causal attention computes the entire square operator with four heads, then compares its last 128 query rows per head against FP64. The latter is a tail metric, not L2 over every output element.

| KV context | Rectangular streaming L2 | PCC | Full causal tail L2 |
|---:|---:|---:|---:|
| 4,096 | 2.508% | 0.999689 | 2.515% |
| 32,768 | 2.836% | 0.999630 | 2.761% |
| 131,072 | 5.014% | 0.999319 | Not run |
| 262,144 | 18.942% | 0.997510 | 18.209% |

All these rectangular cases pass a PCC threshold of 0.994, used by an existing SDPA prefill helper. Seed 1235 gives 2.453%, 2.737%, and 18.028% at 4K, 32K, and 256K. An existing-test-style mixture of normal inputs plus 0.1% large outliers gives 1.773%, 1.294%, and 4.147%; error is distribution-dependent. Raw evidence: [baseline.jsonl](baseline.jsonl), [causal-tail.jsonl](causal-tail.jsonl), [seed1235.jsonl](seed1235.jsonl), [outliers.jsonl](outliers.jsonl).

Sampling matters. Uniformly spreading 128 reference queries over the full 256K causal sequence, including token zero, gave only 0.893% aggregate L2, whereas the final 128 queries gave 18.209%. Early outputs have much larger norms and dominate the aggregate denominator. The repro now defaults to tail sampling and also reports median/p95 per-query L2. The initial spread-sampling results are retained in [causal.jsonl](causal.jsonl); that older file predates the explicit `query_sampling` field.

Sources of error and controlled experiments

1. The main exponential is approximate regardless of the public flag. Both `sub_exp_block_bcast_cols` in [compute_streaming.hpp](../../ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp) and `sub_exp_block_bcast_cols_inplace` in [compute_common.hpp](../../ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp) instantiate `exp_*<true,...>`. On Blackhole this selects the fast Schraudolph-style approximation documented in `ckernel_sfpu_exp.h`. The flag still affects max-correction exponentials, so it is not globally unused. `math_approx_mode=False` also does not override an explicit template `true`.

   An experimental patch makes those main-exp call sites honor `EXP_APPROX_MODE`, supplying the scale explicitly in accurate mode. At 4K, streaming with the false flag improves from 2.519% to 1.822%. The standard FP32-destination/HiFi4 comparison improves from 2.073% to 0.490%. A patched HiFi2/approximate run reproduces the original numbers exactly. This confirms an exponential contribution, but not that it explains all error. [honor-exp-flag.patch](honor-exp-flag.patch), [accurate-exp.jsonl](accurate-exp.jsonl).

2. BF16 state accumulates drift across KV chunks. Streaming `sum_A/B`, `out_im_A/B`, `max_A/B`, and `exp_max_diff` are BF16. The softmax denominator is accumulated into partial-sum tiles via packer L1 accumulation; `salad_correct_fused` repeatedly rescales and merges previous sum/output state. These are additional rounding sites beyond the QK and PV matmuls. The kernel explicitly enables L1 accumulation internally even when the user compute configuration says `packer_l1_acc=False`.

   With Q=0, exact attention is simply the mean of V. At 256K, streaming gives 97.029% L2 with fitted gain 1.968 and PCC 0.999574. An idealized BF16 recurrence for the 32 partial-sum lanes stops at denominator 131,072 instead of 262,144, predicting a factor of two. This agrees closely with the device control, although it is not a bit-exact emulation. With V=1 and random Q/K, correct output is identically one, yet streaming L2 grows from 0.645% at 4K to 15.129% at 256K. These controls exclude QK matmul error as the sole cause. PCC in the older constant-V log is numerically meaningless because the reference variance is effectively zero; the final script returns null for that case. [uniform.jsonl](uniform.jsonl), [constant-v.jsonl](constant-v.jsonl), [cpu-rounding.jsonl](cpu-rounding.jsonl).

   Chunk sensitivity is large at 256K: 133.658% L2 with Kchunk=128, 67.852% with 256, and 18.942% with 512 (Qchunk=128). At fixed Qchunk=64, Kchunk=512 gives 18.494%, while 1024 gives 6.335%. Larger chunks reduce the number of updates, but are not a sufficient fix. Qchunk=128/Kchunk=1024 exceeded this card's L1 allocation limit (1,600,512 > 1,572,864 bytes); the initial partial [chunks.jsonl](chunks.jsonl) records only completed cases before that allocation error. Completed sweeps: [chunks-small.jsonl](chunks-small.jsonl), [chunks-large.jsonl](chunks-large.jsonl).

3. Higher matmul fidelity alone does not solve the problem. HiFi4 streaming gives 2.575% at 4K and 19.646% at 256K, versus HiFi2's 2.508% and 18.942%. Accurate-exp plus HiFi4 streaming also remains high: 1.966% and 19.920%. The precision of the softmax and its state updates is essential; the fidelity label alone is not an accuracy bound. [accurate-exp-hifi4-streaming.jsonl](accurate-exp-hifi4-streaming.jsonl).

4. FP32 storage alone is also insufficient. Default FP32-destination mode already promotes QK and sum CBs, but leaves output/max/correction state BF16. A second experiment additionally promotes those intermediate CBs while using accurate exp. It improves 4K L2 to 0.356%, but worsens 256K to 12.392% (accurate-exp with the original mixed state was 9.632%). This is a rejected standalone fix, not a successful optimization. The standard update helpers still use FPU `add_tiles` and `mul_tiles_bcast_cols`; FP32 data routed through SrcA/SrcB is reduced to TF32. The repository's [accuracy guide](../../tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md) explicitly warns about this loss during recurrent reductions. This is a strong follow-up suspect, not an experimentally isolated explanation of every remaining percent. [fp32-state.patch](fp32-state.patch), [fp32-state.jsonl](fp32-state.jsonl), [fp32-state-uniform.jsonl](fp32-state-uniform.jsonl).

5. Intermediate rounding consumes the remaining accuracy budget. A CPU ablation with exact accumulation gives 0.391% L2 after rounding QK to BF16, 0.487% after also rounding the max-subtraction result, and 0.511% after rounding exponentials. This is an illustrative rounding model, not hardware emulation and not an additive decomposition of device error. It indicates how tight a 0.5% target is when several intermediates are BF16. [analyze_rounding.py](analyze_rounding.py) reproduces this and the denominator-stagnation calculation.

Recommended next work

- Make the accurate-exp option real, with correct scale handling and architecture-specific tests. The saved patch is an experiment tested on Blackhole; it is not ready for general deployment, and performance was not benchmarked.
- Preserve online sum/output state through genuinely FP32 arithmetic: direct unpack-to-destination and SFPU operations where appropriate, or bounded KV partitions with a stable higher-precision/tree merge. Avoid repeatedly routing the running state through BF16/TF32. For the streaming path, this requires redesign of its fused update, not just changing CB formats; the factory currently requires matching BF16 sum/output formats.
- Retain extra precision through QK subtraction and softmax, and round the final output once where feasible. Measure each precision change independently.
- Gate accuracy with relative L2 plus PCC, including per-head/per-query statistics, late causal queries, context/chunk sweeps, Q=0 and V=1 invariants, and both normal and outlier distributions. The statement that TT has *only* PCC checks is too broad: the current prefill tests optionally check absolute RMSE, and the repo contains relative-Frobenius helpers. The relevant gap is that PCC-only/default checks can accept these cases and absolute RMSE is not scale-invariant.

Reproduction on the reserved container

```bash
cd /localdev/cglagovich/tt-metal-blackhole-20260908
export TT_METAL_HOME="$PWD" ARCH_NAME=blackhole
export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages

# Fast test; deliberately fails the requested 0.5% criterion on main.
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --kv-lens 4096 --variants hifi2 --max-l2-pct 0.5 --output sdpa-repro.jsonl

# Full causal operator, measuring last 128 rows of each of four heads.
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --causal --heads 4 --kv-lens 4096 32768 262144 --variants hifi2 \
  --output sdpa-causal-tail.jsonl
```

The local `python_env` overlays the container's `/opt/venv` packages and adds graphviz; another correctly installed tt-metal environment can use its normal Python interpreter and PYTHONPATH. The script also supports `--distribution`, `--k-chunks`, `--q-chunk`, `--seed`, `--query-sampling spread`, and `--check-reference-only`. Each invocation writes a fresh output file; use a new filename to preserve prior runs. The `exact_exp` variant means *request* the false flag, not a promise that unmodified main honors it. `fp32` means HiFi4, false approximation flags, and FP32 destination accumulation; `fp32_hifi2` retains HiFi2 and true approximation flags.

To repeat the exploratory modifications, apply `honor-exp-flag.patch` to main for the first ablation; the compute headers compile through device JIT. Apply `fp32-state.patch` additionally for the second and rebuild with `CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install`. Set a descriptive `--label` and separate output file. Both patches pass `git apply --check`; both were removed after measurement and main was rebuilt successfully.

Verification: FP64-reference and metric self-checks passed; all recorded successful runs completed on hardware; experimental host changes compiled and linked; the restored-main repro raised the expected assertion for 2.508% > 0.5%. Timings in JSON include JIT compilation/cache effects and are not steady-state performance measurements. We have not achieved or validated a 0.5% result at 256K, nor established the exact configuration behind the original Galaxy report.
