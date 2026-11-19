# Inputnorm gamma folding: FP32 coefficient follow-up

One isolated follow-up is prepared: `v5_inputnorm_weight_fold_fp32.py.snapshot`, SHA256 `87266809b3be76e4d0108da555036dfc86a242d8aebad00426c12f9b912d285b`. The adjacent patch is relative to the preserved BF16-fold snapshot. Only setup changes relative to that candidate; every runtime method, including the gamma-free norm and projection accumulation order, is AST-identical. No production/native/core-harness edit or hardware execution was performed.

## Actual BF16-fold evidence

The parent-run `v5_inputnorm_weight_fold_s256.json` passed short correctness and recorded decode1.279687276 ms versus matched selected1.285217237 ms, with prefill108.555185376 versus107.813201147 ms. However, `v5_inputnorm_weight_fold_stress_0_seed154.json` failed the unchanged long-stream quality gate at5991,6062,6425. Parent-reported output/contribution PCC pairs are .994640667/.993949038, .989651616/.988659694 and .994903838/.994396030. Device close was clean. These are numerical failures of that candidate, not an API blocker. The new provenance hashes both artifacts and sidecars; none was overwritten.

## Minimal adaptation and source support

The setup still computes `gamma = input_layernorm.weight.float() + 1`. It now retains `original.float() * gamma` in FP32 instead of rounding the folded coefficients to BF16. All four generic input-projection copies, native packed QZ, ordered packed KVAB, and the subsequently created DRAM-sharded decode QZ copy receive FP32 coefficients. No unchanged output/MLP/full-attention weight is widened. Runtime still returns BF16 unweighted normalized input and omits the final gamma multiply; no compensating operator is added.

The ordered path is applicable without a new kernel or configuration: BinaryNG multiply already specifies `dtype=float32, fast_and_approximate_mode=False`. `binary_ng_program_factory.cpp:1371–1400` installs direct-to-Dest unpack for SFPU operands, preserving the incoming FP32 folded coefficients for this product. Product/cumsum shapes and FP32 L1 allocations do not change; only the resident DRAM weight and its read traffic grow.

Native QZ remains the existing matmul path. `matmul_device_operation.cpp:58–60` accepts floating-point inputs independently; it does not require BF16 weights matching the activation dtype. The DRAM-sharded factory derives input tile/page/CB bytes from actual data formats (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`, including its multicore-pipeline implementation), rather than assuming two-byte coefficients. FP32 weights therefore have a source-supported path with the existing geometry. This is not a promise of full FP32 coefficient precision inside the FPU: native matmul operand/fidelity semantics still apply, and hardware allocation/performance remain untested. The adaptation isolates removal of BF16 coefficient storage rounding, especially on recurrent KVAB, without substituting another QZ arithmetic algorithm.

Additional resident coefficient payload from changing the existing copies is approximately126353408 bytes before shard padding:50593792 for original per-key projections,25165824 for interleaved packed QZ,25165824 for its sharded copy, and25427968 for ordered KVAB. Native QZ shard padding can increase that total. Ordered weights remain DRAM, so this does not double the large product/prefix L1 buffers. Extra read bandwidth may erase the small short-run gain; complete-path timing is necessary.

## CPU evidence and remaining question

`v5_inputnorm_weight_fold_fp32_cpu.py` executes the actual setup AST against real layer0 weights. It verifies caller tensor immutability, unaffected full-attention/nonprojection weights, FP32 folded dtypes, and exact native-QZ/ordered-KVAB packing coverage. Runtime method AST equality was independently checked while generating the snapshot.

For sampled real coefficient rows, FP32 folding differs from the FP64 product by at most1.862645149230957e-9; BF16 folded storage differed by up to0.0002384185791015625. This demonstrates that the intended additional coefficient-rounding source is removed. It does not show that the overall projection becomes canonical: the missing BF16 boundary on `normalized * gamma` remains, and synthetic projected outputs are not uniformly improved. The CPU script's ordinary float linear diagnostic is also not a native QZ/ordered-SFPU emulator. No synthetic-bit result is used as a model-quality verdict.

The bounded discriminator is the existing complete snapshot benchmark followed, if useful, by the same real cached seed1548192+33 stream with unchanged thresholds. One such candidate is prepared, not a sweep. If it still fails, retaining FP32 folded weights does not resolve this particular gamma-fold transformation; acceptance requires the actual decoder evidence.

Reproduce the CPU setup/packing proof without TTNN imports:

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_inputnorm_weight_fold_fp32_cpu.py
```

## Measured closure

The FP32-fold short complete-path benchmark `v5_inputnorm_weight_fold_fp32_s256.json` passed short PCC and deterministic replay: prefill PCC0.999933442088, decode PCC0.999939257990. It recorded prefill116.623264970 ms / decode1.380870817 ms versus matched selected107.813201146 / 1.285217237 ms (`v5_prefill_pair_blocks_baseline_s256.json`). That is 8.17% slower prefill and 7.44% slower decode. **Reject this FP32 adaptation on complete-path speed. No8192 FP32 quality run or quality verdict is claimed.** The BF16 fold separately remains rejected on its actual seed154 long-stream failures.

`v5_inputnorm_weight_fold.decision.json` binds the matched baseline and all three candidate evidence artifacts/sidecars by SHA256. FP32 short artifact SHA256:`fa26fb01ca69829b201a322d1cff87ce70d2f936ec741a0ccb784b27373fc92a`. No new candidate or runtime change is proposed.
