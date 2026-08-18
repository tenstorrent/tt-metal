# AutoDebug Linear Context Report

## Scope

Read-only diagnostic for the Qwen/Qwen3.6-35B-A3B functional decoder linear-attention context probe. I did not run hardware-facing TT commands. I did not invoke `.agents/scripts/autodebug.sh` because that runner writes `./AUTODEBUG.md`, while this task allows writing only this report.

## Starting Evidence

- Failing target: `test_context_probe_linear_attention_prefill_decode_non_aligned` in `models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py:801-819`.
- The test creates synthetic layer 0, runs linear prefill, runs one decode at `current_pos=seq_len`, synchronizes, asserts shapes, then prints `linear_attention non_aligned prefill/decode seq_len=...`. The 262,144 logs never reach that print, PASS, FAIL, pytest summary, or device-close lines.
- Passing linear probes:
  - `context_probe_linear_prefill_65537_sparse.log`: `seq_len=65537`, passed, `209.08s call`.
  - `context_probe_linear_prefill_131073_sparse.log`: `seq_len=131073`, passed, `742.26s call`.
- Timeout linear probes:
  - `context_probe_linear_prefill_262144_sparse.log`: pytest timeout configured as `1800.0s`; log stops after device open/profiler sync.
  - `context_probe_linear_prefill_262144_retry.log`: pytest timeout configured as `2400.0s`; same stop point.
  - `context_probe_linear_prefill_262144_long.log`: pytest timeout configured as `5400.0s`; same stop point.
- No `FATAL`, `TT_FATAL`, traceback, FAIL, PASS, or pytest timeout stack appears in the three 262,144 logs. `tt_smi_list_after_linear_long_timeout.log` still lists the four Blackhole p300c boards as available.
- Full-attention control with the same functional decoder wrapper and MoE path completes advertised prefill context: `context_probe_full_prefill_262144_sparse.log` prints `full_attention non_aligned prefill seq_len=262144`, passes, and records `358.73s call`.

## Finding 1: Likely Reason For The 262,144 Timeout

The likely reason is algorithmic/dispatch scaling in the current linear-attention prefill implementation, not an observed TT hardware failure.

The TTNN linear prefill path in `tt/functional_decoder.py:568-584` loops over every token:

```python
for idx in range(seq_len):
    token = _slice(...)
    out, next_state = self._step(token, next_state)
    ...
```

Each `_step` in `tt/functional_decoder.py:513-563` performs the decode-style gated-delta update for one token: projections, causal-conv tap update, sigmoid/softplus/exp, Q/K normalization, recurrent-state decay, key/state matmul, rank-one update, query/state matmul, output norm/gate, and output projection. At 262,144 tokens this becomes millions of small TTNN operations submitted from Python, plus thousands of 32-token concats in `prefill_forward`.

The HF reference does not use this per-token recurrent path for prefill. In `modeling_qwen3_5_moe.py:520-543`, single-token cached decode calls `recurrent_gated_delta_rule`, but multi-token prefill calls `chunk_gated_delta_rule`. The fallback chunked implementation at `modeling_qwen3_5_moe.py:243-321` computes 64-token chunks with chunk-level triangular matmuls and only loops by chunk, not by token.

The scaling evidence matches this diagnosis. Full attention, which exercises the same post-attention MoE path, goes from 65,537 to 131,073 to 262,144 tokens in 86.12s, 171.46s, and 358.73s. Linear attention goes from 65,537 to 131,073 in 209.08s and 742.26s, a much sharper increase. Since full attention completes 262,144 with the same large hidden-state shape and MoE path, the 262,144 linear timeout is concentrated in the linear mixer recurrence path.

Because the 262,144 logs have no TT failure line and no device-health evidence of a stall, the honest description is: the current implementation does not complete the advertised linear-attention prefill/decode probe within the 90-minute run budget; it is not proven to hit DRAM, L1, NoC, watcher, or firmware failure.

## Finding 2: Vectorization Primitive Or Pattern

There is no obvious repo-local TTNN primitive or existing pattern that can be dropped in to vectorize this Qwen gated-delta recurrence correctly at this stage.

The closest primitive is `ttnn.experimental.prefix_scan`. Its binding documents an SSM scan over tensors shaped `[1, 1, L, 2EN]` (`ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_nanobind.cpp:18-20`), and the Wormhole Mamba demo uses it for the elementwise recurrence `hidden = a * hidden + bx` (`models/demos/wormhole/mamba/tt/mamba_ssm.py:258-265`).

Qwen gated-delta is different. The state is `[heads, key_dim, value_dim]`, and every token applies a key-dependent rank-one correction:

```text
state_t = g_t * state_{t-1}
delta_t = beta_t * (value_t - key_t @ state_t)
state_t = state_t + key_t.T @ delta_t
out_t = query_t @ state_t
```

This couples the key dimension through `key_t @ state_t` before writing the next state. Scalar `cumsum`, `cumprod`, and the existing SSM `prefix_scan` do not represent that matrix-valued affine transform. A correct vectorized prefill would need either a TTNN implementation of the HF chunked gated-delta rule or a new/custom fused scan kernel for this recurrence.

## Finding 3: Small Stage-Scoped Fix

I do not see a small, correctness-preserving stage-scoped fix that is likely to materially change the 262,144 outcome.

- Changing `current_pos` handling is not relevant to the timeout. Linear decode ignores page tables and uses the carried `linear_state`; the decode step is one `_step` and is constant-size after prefill.
- Page-table handling is not involved for `linear_attention`.
- The MoE path is unlikely to be the blocker because full-attention 262,144 prefill uses the same functional decoder wrapper and MoE path and passes in 358.73s.
- Adding more `synchronize_device` calls or progress prints could localize the exact point of slow progress, but it would not remove the per-token recurrence or the millions of dispatches. It may also slow the run further.
- Increasing the timeout would only change the evidence window, not the implementation capability. The 90-minute timeout already records the relevant stage risk.

The smallest fix that would plausibly change the outcome is not small: implement a chunked TTNN prefill path for Qwen gated-delta, matching HF `chunk_gated_delta_rule`, with vectorized QKV/Z/A/B projections, vectorized causal conv, chunk-local triangular computations, and recurrent-state handoff per chunk. That is new functional work, not a diagnostic-only or low-risk patch.

Predicted verification command after such a fix:

```bash
RUN_QWEN36_CONTEXT_PROBE=1 QWEN36_CONTEXT_LINEAR_SEQ=262144 \
pytest -q --timeout=5400 \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -k 'test_context_probe_linear_attention_prefill_decode_non_aligned' -s
```

Nearby verification would also need HF-vs-TTNN PCC for small and real-weight linear layers, fallback audit with `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`, and a traced decode check after the new prefill state.

## Final Status

Limitation, not fixed in this read-only diagnostic.

The largest completed linear-attention context evidence is the 131,073-token pass. The 262,144-token probe timed out three times, including a 5,400-second run, without a TT failure line. That is honest evidence of an execution-time limitation in the current token-by-token functional prefill path, not evidence that linear attention is functionally correct at the full advertised 262,144-token prefill context.

For the functional-decoder stage, record linear-attention prefill as `largest_passing=131073` and `advertised_context_262144=timeout/no TT failure line` until a correct chunked gated-delta TTNN prefill path exists and passes the command above.
