# AutoDebug: functional-decoder stage-review findings

Source-only review; no TT device was opened and no implementation/test file was changed. The fresh AutoDebug runner was attempted twice, but its nested workspace sandbox could not create an unprivileged `bubblewrap` namespace. The findings below were therefore checked directly against the focused autoport sources and the functional-decoder/stage-review contracts.

## Headline findings

1. **Trace, watcher, and context smokes are structurally real but numerically vacuous.** Both smoke programs set every projection/MLP weight to zero (`tests/full_attention_decode_smoke.py:33-94`, `tests/linear_attention_decode_smoke.py:40-72`). Their expected full-layer result is therefore the residual input and token-mixer result is zero (`full_attention_decode_smoke.py:220-238`, `linear_attention_decode_smoke.py:149-166`). This cannot detect stale traced inputs, stale positions, wrong page routing, or broken recurrent/KV state.
2. **The trace harness never updates stable input or position buffers between replays.** It captures `decode()` over fixed tensors and repeatedly calls only `execute_trace` (`full_attention_decode_smoke.py:223-238`, `linear_attention_decode_smoke.py:152-166`). Stateful cache writes occur in both implementations (`tt/functional_decoder.py:367,395-401,568-589`), but zero weights make every replay invariant. Reported replay PCC is therefore determinism/identity evidence, not decode correctness.
3. **There is no prefill-to-decode cache-dependence test.** Prefill and decode are mutually exclusive branches/process invocations (`full_attention_decode_smoke.py:188-213`; decode starts at 215). Permuted pages only prove that a zero-valued decode does not crash. Nothing proves that prefill writes are read by decode, that logical-to-physical pages are honored, or that rows do not alias.
4. **Long linear prefill is an implementation-capability risk, not merely missing evidence.** `_linear_attention_prefill` builds a Python loop with one decode graph and one retained output tensor per token, then concatenates all outputs (`tt/functional_decoder.py:331-338`). At a near-context non-divisible length this implies about 262K sequential stateful op chains plus a 262K-element Python list. Small 32/33/64/65 runs do not establish the advertised context. This should be treated as a likely implementation limitation until a bounded-memory chunked/scan path is demonstrated.
5. **Watcher evidence exercises the same zero-weight path.** A watcher-clean zero path is useful kernel-liveness evidence, but it does not satisfy the meaningful full/linear, cache/state-mutating path required by the stage contract.

No separate arithmetic error in the decoder equations is proven by source inspection. Items 1-3 and 5 are evidence/harness defects; item 4 is the one likely production implementation defect.

## Focused verify/refute experiments

### 1. Nonzero traced replay PCC, batch 1 and 32, full and linear

Smallest durable harness change:

- Reuse the deterministic nonzero synthetic state builders in `full_attention_synthetic_pcc.py` and `linear_attention_synthetic_pcc.py`; do not maintain a third weight generator.
- Allocate `hidden_states`, `current_positions`, and page table once before capture. Between replays, copy new deterministic host values into those same device buffers without reallocating them.
- Snapshot/restore KV or convolution/recurrent state before each eager/reference/replay comparison. Otherwise capture and prior replays advance state and comparisons are not like-for-like.
- Run two distinct decode steps and require each traced output to match the corresponding eager/PyTorch oracle at PCC >= 0.995; also require output A != output B and, for batch 32, distinct per-row outputs.
- Parameterize `{full, linear} x {token_mixer, full_layer} x {batch 1, 32}`.

Decisive failure signatures:

- Output unchanged after hidden input changes: stale input address/update.
- Full-attention output unchanged after position changes: stale position/RoPE/cache index.
- Linear output diverges only on the second stateful step: recurrent/conv state capture or restore bug.
- Eager passes but replay fails with identical initial state: trace-lifecycle implementation bug.

### 2. Cache-dependent prefill-to-decode

Add one end-to-end test in a single decoder instance:

- Use nonzero weights and nonzero, row-distinct prompt activations.
- Batch rows use different prompt lengths/current positions and a non-identity page table; permute pages independently per row, not one common `flip`.
- Cover positions 63, 64, 65 and the final supported position.
- Prefill, then decode one nonzero token. Compare to an oracle carrying the same per-row history.
- Run controls: zero the filled cache or swap two rows' page-table entries before decode. The output must change in the predicted row only.
- Read back selected physical cache slots after prefill/update and compare them with expected K/V values. This localizes routing failures before SDPA.

This test directly verifies the production calls at `functional_decoder.py:444-455` and `568-589`. The current permuted-page smoke cannot refute cache aliasing because its K/V projections are zero.

### 3. Practical batch-32 numerical reference

Do not instantiate a batch-32 full HF causal model/cache. Use a layer-only, row-microbatched oracle:

- Construct one HF decoder layer from the existing deterministic synthetic state.
- Evaluate 32 independent batch-1 rows sequentially, each with its own small HF cache/state, and stack outputs. TTNN still runs the real batch-32 tensor once.
- For full attention, retain only the short prompt required by the boundary/cache test; the advertised-context test should validate TT cache addressing/state, not allocate a 262K HF attention matrix.
- For linear attention, use the existing layer-only HF recurrence per row, or a direct Torch recurrence oracle, retaining only conv and recurrent state.

This keeps reference memory approximately batch-1 layer memory plus stacked outputs and tests row independence. Refute it only if stacked batch-1 HF differs from a feasible small batch-32 HF control.

### 4. Long non-divisible linear-attention state near context

First run a source-level/instrumented host construction probe (no numerical HF required): count live outputs and enqueued ops for lengths 65, 1025, and a larger geometric step. The current loop predicts linear growth and retained outputs until final concat.

The durable implementation/harness boundary is:

- Replace per-token retained-output construction with a bounded chunk/scan implementation that carries only conv and recurrent state across chunks.
- Validate a long non-divisible length near 262,144 with a compact oracle that checks final conv state, recurrent-state checksums/slices, and selected output tokens at chunk boundaries plus the last token.
- Compare one short sequence exhaustively against HF, then compare chunked versus token-by-token Torch recurrence for the long sequence.
- Include lengths immediately below/at/above the chosen chunk boundary.

A mere shape pass with zero weights does not refute this finding; evidence must show bounded host/device live memory and nonzero state agreement.

### 5. Meaningful `TT_METAL_WATCHER=10` paths

Rerun watcher separately from profiling with the same nonzero acceptance harness:

- full attention: prefill then two traced decodes, permuted per-row pages, positions crossing 63/64/65, batch 32;
- linear attention: nonzero prefill/state initialization then two traced decodes, batch 32;
- include one full-layer case for each kind so RMSNorm/MLP/residual kernels are covered.

Require numerical assertions to pass in the watcher process and archive the exact command/log. A clean log without a nonzero PCC/cache/state assertion remains only a liveness smoke.

## Priority order

1. Share nonzero synthetic fixtures and add state snapshot/restore plus stable-buffer updates to traced batch-1/32 tests.
2. Add the single-instance, controlled prefill-to-decode page-routing test.
3. Use row-microbatched layer-only references for batch 32.
4. Implement and validate bounded-memory chunked linear prefill near context.
5. Point watcher commands at these meaningful tests and replace the zero-weight acceptance claims.
