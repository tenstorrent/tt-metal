# Native GDN tail fusion candidates

The current device report `artifacts/native_linear_perf.csv` contains a100.967µs `ReshapeViewDeviceOperation` at row1194 after the tail's head permutation. The input `[B,12,S,128]` is multiplied by SiLU(z), permuted to `[B,S,12,128]`, then reshaped to `[1,B,S,1536]`. Putting12 heads in a tile dimension introduces physical head padding and a device copy. Dedicated head concatenation can express the shuffle directly.

## Prepared experiment

`tail_followup.py` reuses the real-weight `probe.py` CLI and applies one of three isolated modes through `TAIL_FUSION=concat|activation|both`:

- `concat`: replace the final permute with `ttnn.experimental.nlp_concat_heads(..., memory_config=ttnn.DRAM_MEMORY_CONFIG)` and retain leading-dimension reshape. Native contract: input rank4 `[B,H,S,D]`, TILE BF16/FP32/BFP8; output `[B,1,S,H*D]` with the same dtype. HereH12 andD128 are legal. The leading reshape to `[1,B,S,H*D]` preserves last-two-dimension packing. This should preserve bits. Hypothesis: remove roughly0.10ms of device-backed reshape plus the preceding transpose, not the entire0.435ms reshape category. Full-stack saving would be roughly5ms across48 linear layers if that local estimate holds.
- `activation`: replace `multiply(output, silu(z))` with `multiply(output,z,input_tensor_b_activations=[ttnn.UnaryOpType.SILU])`. The binary binding explicitly supports input-b activation lists. This removes a unary operation, measured8.519µs in the current profile. It can change BF16 intermediate rounding; bit-exact equivalence is not presumed.
- `both`: apply both replacements, available for a cumulative check after the isolated results.

The public `ttnn.transformer.concatenate_heads` also accepts `[B,H,S,D]` and returns `[B,S,H*D]`; it invokes the same native op then squeezes axis1. The experimental entry avoids that redundant wrapper step and already appears in this decoder's full-attention path.

No production code was modified for these candidates. The wrapper was Black-formatted and AST-parsed; hardware execution is parent-owned because full-model validation has exclusive device access. Results must be recorded before accepting either rewrite.

## Other source-backed alternatives

`output_head_major=False` on the native GDN op is not a free token-major output. The C++ wrapper untilizes the kernel's head-major result, reshapes it, and permutes to row-major `[B,T,H,V]`. The current head-major return with aligned time is a metadata-only reshape. Switching the flag would add movement before the tail and is not a substitute for direct concatenation.

`ttnn.experimental.kda.sigmoid_gated_rms_norm` is another existing dedicated op. It consumes native `[B*H,T,V]`, flat gate `[B,T,H*V]`, and BF16 weight `[V]`, and emits flat `[B,T,H*V]`. Its gate is sigmoid, while Qwen needs SiLU. Multiplying the flat result by z gives the correct algebra, `(norm(x)*weight*sigmoid(z))*z`, but changes rounding. Integrating it before the initial attended rank4 reshape could remove the norm/head-layout block and leave one flat multiply. It needs an independent numerical check and latency test; passing the unmodified SiLU requirement to the sigmoid-only op would be incorrect.

The parent already tested eliminating the initial attended reshape in isolation: it was bit-exact but slowed the reduced run (3.55ms versus3.36ms) and was rejected. That evidence does not establish the result of a dedicated full-tail replacement, which changes the consuming graph.

The wrapper now also provides `TAIL_FUSION=gated` for that dedicated norm experiment. It removes the native caller's padded rank4 attended detour, passes its existing `[B*12,T,128]` output directly into `sigmoid_gated_rms_norm`, reshapes the replicated norm weight to rank1 `[128]`, requests BF16 output, and multiplies the flat result by z. The op's default compute config is HiFi4, FP32 destination accumulation, approximate mode enabled, and packer L1 accumulation disabled. Native source validation requires TILE-alignedT/V, BF16 gate/weight, interleaved memory, and rank3 input/gate; the wrapper satisfies these contracts for padded served prefill. It preserves the existing output projection and cache-update boundaries. The change is algebraically correct but changes intermediate rounding and requires real-weight correctness measurement before adoption.

## Binary SiLU failure localization

The parent's unsynchronized `both` run produced nonfinite layer outputs while convolution and recurrent caches remained exact. This localizes the observed difference after recurrence but does not prove that SiLU itself lacks support. Three bounded controls tested that inference:

1. `binary_silu_repro.py`: BF16 `[1,12,128,128]`, gate linearly spanning[-20,20], normally distributed multiplicand spanning[-9.1875,9.25], TP4. RHS fused SiLU and commuted LHS fused SiLU were both finite and **bit-exact** to standalone SiLU followed by multiplication. FP32 output also remained finite (0.165% relative L2 vs rounded BF16 baseline).
2. `binary_silu_padded_repro.py`: construct the gate through the model-like head permutation before the same calls. The final padded shapes were still `[1,12,128,128]`, and BF16 fused results remained bit-exact. A hidden padded-head-stride explanation was not reproduced.
3. `binary_silu_model_repro.py`: instrument the actual real-weight reduced graph at the fused multiply boundary, synchronously read its inputs and result, and compare against separate SiLU+multiply. On both observed calls, actual norm range[-0.875,10.1875] and gate range[-16.25,15.625] were finite BF16; padded shapes matched. Fused versus separate output was **bit-exact and finite**.

Artifacts: `binary_silu_repro.json`, `binary_silu_padded_repro.json`, and `binary_silu_real_inputs.json` under `artifacts/`. Logs are `/tmp/qwen_binary_silu_repro.log`, `/tmp/qwen_binary_silu_padded_repro.log`, and `/tmp/qwen_binary_silu_model_repro.log`. All three device jobs exited0 and closed the mesh, without reset.

These controls refute an unsupported-SiLU claim and a simple real-input-range or BF16-rounding explanation. The actual-input instrumentation changes synchronization and tensor lifetimes, so it does not validate the unsynchronized fused graph or establish a runtime root cause. A graph scheduling/lifetime interaction remains a hypothesis, not a proven explanation. The combined binary-fusion candidate remains unverified; the independently valid concatenation and dedicated gated-norm candidates can be assessed on their own merits.

Reproduction uses `timeout 120 env PYTHONPATH=. python_env/bin/python <script>` for the two synthetic scripts. For the model control, use the pinned model environment from the main report, outer chunk128, and:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/binary_silu_model_repro.py \
  --candidate baseline --sequence 128 --iterations 1 \
  --result models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/binary_silu_instrumented_model.json
```

The instrumented model's timings are not benchmark results. This control records the tail multiplication only, and does not claim finite final-layer outputs without an additional captured-output comparison.
