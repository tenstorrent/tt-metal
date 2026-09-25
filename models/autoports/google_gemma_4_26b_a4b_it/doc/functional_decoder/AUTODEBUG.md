# AutoDebug: Gemma 4 26B-A4B decoder PCC

## Evidence and scope

Inspection-only fresh diagnosis, 2026-09-25. No hardware executions, CPU model experiments, or implementation edits were performed. The harness was being extended independently during inspection; this report concerns the recorded synthetic prefill failure.

Recorded command:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/synthetic_sliding_32.json
```

`synthetic_sliding_32.json` records sliding layer 0, S=32, synthetic weights, PCC **0.9713391913952353**, required **0.995**. The log reaches output comparison and closes the device; this is numerical disagreement, not a hang. Effective model dimensions are H=2816, 16 Q heads, 8 KV heads, D=256, 128 experts, top-8, expert intermediate 704, shared intermediate 2112, TP=1.

The main agent subsequently produced `real_sliding_32.json`: real-weight prefill PCC **0.9988936448679125**, traced decode PCC **0.9994597686704707**, repeated replay equal. I inspected that artifact but did not run it. This passing contrast supports investigating the provisional synthetic distribution first; it does not by itself prove the router mechanism. Preserve the failing synthetic case as diagnostic evidence while deriving representative synthetic weight statistics from the checkpoint.

## Leading hypothesis: BF16 softmax collapses router ranks before top-k

**Strong static evidence; causal attribution remains unverified.**

1. The harness initializes projection weights with `randn * 0.02`, leaves router scales at one, and runs the HF reference in FP32 (`tests/run_decoder.py:35-42,61`). The TT layer uploads BF16 weights and activations (`tt/functional_decoder.py:47-57`).
2. The router RMS-normalizes each token, multiplies by H^-0.5 and the learned scale, projects, computes softmax, and only then selects top-k (`models/demos/gemma4/tt/router.py:99-122`). For the synthetic projection distribution this predicts score standard deviation approximately 0.02; probabilities cluster near 1/128.
3. The lowered matmul defaults to **HiFi2**, not LoFi, for these BF16 inputs without a program/grid override, and returns the activation dtype (`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2809-2810,2847-2850`). Thus router scores are BF16. Rank-4 last-axis softmax selects `SoftmaxProgramFactoryAttentionOptimized`, and its output also inherits the input dtype (`normalization/softmax/device/softmax_device_operation.cpp:141-153,340-354`). The factory packs BF16 output (`softmax_program_factory_attention_optimized.cpp:78`). Top-k therefore consumes already-rounded BF16 probabilities.
4. Near 1/128, the BF16 step above that value is 2^-14, approximately 0.000061. A score separation of 0.001 produces a probability separation around 0.000008. Distinct FP32 rankings can collapse into equal BF16 probabilities before top-k. The exact selected-ID differences in this failure have not been measured.
5. Existing `models/demos/gemma4/tests/unit/test_router.py:35-42` explicitly raises synthetic router projection standard deviation from 0.02 to 1.0 to avoid this near-uniform BF16-vs-FP32 top-k disagreement. That test uses 8 experts/top-4, whereas the failing contract uses 128/top-8. This is evidence that the mechanism is known, not proof that changing this test distribution would satisfy the current contract.
6. A changed expert ID replaces a whole independently initialized expert contribution. The routed result receives its own RMSNorm before mixing with the shared branch (`functional_decoder.py:73-76`), so a small router-score perturbation can become a substantial decoder-output difference. Good probability PCC alone does not establish route correctness.

### Discriminating experiments

Run each control independently and retain the original seed, full expert count, top-k, weights, inputs and 0.995 threshold.

1. **Localize and hold inputs fixed.** Record reference and TT outputs after input norm, attention, attention residual, shared MLP post norm, raw routed experts, routed post norm, and final layer. Feed the *same captured TT attention residual* to both routers. Compare unordered selected expert sets, their intersection counts per token, probability ties around ranks 8/9, and selected weights. Do not compare sorted order alone: the expert sum is permutation invariant when IDs and weights remain paired.
2. **Oracle routing control.** On that same TT residual, compute HF router IDs/weights on CPU in the test harness, upload the dense routing tensor, and use the existing TT experts and remaining decoder operations. A large PCC recovery with unchanged expert computation verifies routing as a material contributor. If attention already diverges or oracle routing does not recover the result, continue localization rather than attributing everything to routing.
3. **Probability-rounding control.** On fixed HF router logits, compare FP32 top-k against top-k of BF16-rounded full-softmax probabilities. This CPU-only control tests whether probability storage alone changes the selected set. Separately compare top-k of TT logits against TT probability top-k to distinguish projection/input drift from softmax rank collapse.
4. **Minimal device candidate.** Select top-k on logits and compute softmax over the selected logits; then apply the existing per-expert scale and scatter. In exact arithmetic, `softmax(all_logits)[selected] / sum(selected_probs)` equals `softmax(selected_logits)`. This is **not** softmax of already-softmaxed probabilities. Verify that top-8 padded lanes are excluded from the selected softmax/reduction. Keep BF16 weights and the existing projection first, so this tests ordering alone. If needed, test FP32 router score storage/compute as a separate control, not a bundled change.
5. Rerun the original failing command after each verified candidate, then real-weight sliding/full attention and decode. Higher synthetic router variance is useful only as a contrast; it is not a fix for the original case.

Any candidate implementation or diagnostic harness must stay inside `models/autoports/google_gemma_4_26b_a4b_it/`.

## Secondary candidates if routing does not account for the error

- **Attention numerical drift.** Q/K are normalized but SDPA uses scale=1.0, matching HF; consequently relatively small Q/K errors can affect sharp attention scores. QKV and output projections use default HiFi2 BF16 computation; prefill SDPA already explicitly uses HiFi4 and FP32 destination accumulation (`attention/operations.py:75-82,579-586`; `attention/prefill.py:559-567,640-650`). Compare QKV, per-head norms, RoPE and SDPA boundaries before changing compute modes. The imported test thresholds include 0.97 for some 26B sliding layer cases, so passing those tests does not establish this stricter 0.995 contract.
- **Short-height RMSNorm path.** S=32 enters the imported width-sharded normalization path for every learned decoder norm, despite being a prefill call (`tt/rms_norm.py:135-151`). S>32 uses a different path. A same-input comparison against plain interleaved RMSNorm is a narrow control if the first divergent boundary is a norm. No source-level defect in this path was established.
- **Expert accumulation/shape or precision.** Prefill computes every expert, applies routing weights after the down projection, then reduces experts (`tt/experts/prefill.py:93-165`). Its matmuls already use HiFi4, BF16 output, and FP32 destination accumulation disabled. Use identical expert inputs and oracle routing to isolate this path before toggling precision. The sparse decode path has different compute defaults and is not covered by this S32 prefill failure.

## Checked and demoted

- Residual, router-input, separate expert normalization, shared/routed post norms and layer scalar order match installed HF `Gemma4TextDecoderLayer` (`python_env/lib/python3.10/site-packages/transformers/models/gemma4/modeling_gemma4.py:1405-1455`).
- HF RMSNorm directly multiplies its weight; the TT loader correctly does not add one (`modeling_gemma4.py:193-211`).
- Expert gate/up halves and weight transposes match HF (`tt/experts/weights.py:61-73`; HF `modeling_gemma4.py:1324-1328`).
- S32 is one chunk starting at zero, with no logical padding and no sliding-window boundary. Prefill attention reads the freshly projected K/V directly; the reversed page-table mapping does not govern the SDPA input for this case (`attention/prefill.py:549-650`). Cache mapping is therefore not a leading explanation for the recorded prefill PCC.
- The attention wrapper retains and forwards sliding tails across later chunks (`attention/__init__.py:220-253`); the autoport is not simply dropping them. Later-chunk correctness still needs tests.
- `Accurate` GELU differs from HF's tanh approximation, but no evidence connects that small analytic difference to this 0.971 failure. Do not promote it before component measurements.

## Conclusion

No root cause is proved by this static pass. BF16 probability rounding before top-k is the best-supported, readily falsifiable explanation for the synthetic 128-expert failure. The first useful experiment is same-input route-set comparison plus oracle-routing substitution, followed by an isolated logits-top-k candidate. Keep any remaining attention or expert drift separate in the evidence.
