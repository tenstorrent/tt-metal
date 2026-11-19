# V6 prefill-state discrepancy: chunk64 versus sequential FP32 arithmetic

**The saved 128-token prefix discrepancy is entirely reproduced by replacing HF chunk64 arithmetic with the selected decoder's sequential recurrence on identical canonical inputs.** It is not evidence of an earlier projection error for this prefix. This is a CPU counterfactual and does not yet prove that fixing prefill state repairs the remaining V6 decoder failures.

The [CPU analysis](../../../tests/v6_prefill_state_hypothesis.py) recreates the canonical layer34/seed154 prefix using frozen oracle inputs, captures the actual HF chunk-rule operands, and applies the installed HF sequential recurrence to those same operands. The recreated HF chunk64 state, rounded to BF16, is bit-identical to the canonical incoming cache saved at position128. The CPU sequential **FP32** final state is bit-identical to the saved TT incoming state in `v5_combined_layer34_stream_probe.json.pos128.pt`.

| Comparison | BF16 differing elements | FP32 max absolute difference |
| --- | ---: | ---: |
| Same-input HF chunk64 versus HF sequential | 242 | 5.364418029785156e-7 |
| Same-input HF chunk64 versus HF chunk32 | 243 | 2.2351741790771484e-7 |
| HF saved cache versus TT cache rounded to BF16 | 242 | 0.00048828125 after BF16 rounding |
| Same-input sequential versus saved TT FP32 state | **0** | **0** |

The [result JSON](v6_prefill_state_hypothesis.json) records every one of the 242 differing flat indices and its HF, sequential and TT BF16 codes. All sequential and TT codes agree. It also records the analysis source, installed HF source, measured decoder snapshot, capture and input-prefix hashes. No device call was made. The very small FP32 reordering difference crosses BF16 rounding boundaries and therefore changes the next public decode call's effective cache.

The source distinction is concrete. Installed `torch_chunk_gated_delta_rule` defaults to chunk size64; it cumulatively sums log decay, builds a triangular correction matrix, transforms `v_beta` and `k_beta`, and updates the final state using matrix products. `FusedDecoder._linear_attention` calls the native chunk32 operation for outputs but **discards that final state**, rebuilding it token by token with SFPU recurrence. These are mathematically equivalent formulations with different FP32 rounding. Merely retaining the native chunk32 state is not an exact repair: the CPU chunk32 control already differs from chunk64 at243 BF16 cache elements, before considering native FPU narrowing.

A concrete device-graph candidate is to retain the existing output path and replace only the sequential final-state repair with an explicit FP32 **chunk64 state update** built in `fused_decoder.py`:

1. Use the existing normalized key, value, beta and FP32 log-decay tensors. Repeat the16 key heads to32, group logical tokens in64-token blocks, and form `k_beta`, `v_beta`, cumulative log decay and lower-triangular decay masks.
2. Build `A = -((k_beta @ key.T) * decay_mask)` with diagonal/upper triangle zero. For rows1..63, apply the same `A[i,:i] += sum(A[i,:i,None] * A[:i,:i], axis=-2)` recurrence; then add identity. Form `V = A @ v_beta` and `K = A @ (k_beta * exp(cumulative_g))`.
3. For each64-token block, calculate `V_new = V - K @ state`, then `state = state * exp(g_last) + (key * exp(g_last-g)).T @ V_new`. Copy that FP32 state to the existing persistent cache. Use full-FP32 products and explicit reduction graphs where native matrix multiplication narrows operands. Exact agreement still requires measurement of reduction/FMA ordering; no claim of a proven exact device implementation is made here.
4. Preserve arbitrary logical lengths: pad only the final internal64-token block with zero key/value/beta/log-decay and crop outputs. Padding must not create extra cache rounding boundaries. Internal chunks must preserve64-token grouping across the public call; the current32-token/interior-plus-decode-tail planning cannot establish exact HF chunk64 arithmetic for every length.

Before building this potentially costly graph, the [prefix transplant diagnostic](../../../tests/v6_prefix_state_control.py) isolates its potential benefit. It verifies immutable oracle/source hashes, regenerates fresh HF prefix outputs and canonical cache, and compares both bit-for-bit to saved evidence. A canonical cache tensor is uploaded once during setup; after the candidate's first prefill, a device copy replaces only its recurrent state. The existing stress harness, conv state, actual decode implementation and thresholds stay unchanged. The initial saved state used for all repetitions includes this transplant. Continuation uses the candidate's ordinary prefill with no second transplant.

This diagnostic is explicitly **oracle-assisted and ineligible for acceptance or production**. There is no host tensor work or conversion inside measured decode, but it intentionally injects a reference cache between prefill and decode. The [CPU setup result](v6_prefix_state_control_cpu_setup.json) passed; no hardware run was performed by this investigation.

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_prefill_state_hypothesis
# Root owns device scheduling and the selected candidate path:
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf QWEN_DECODER=fused TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_prefix_state_control --snapshot CANDIDATE.py.snapshot --output FRESH_DIAGNOSTIC.json
```

A passing transplant would support initial-state drift as sufficient to explain the tested remaining failures. A failing transplant would require examining later rounding/operand differences rather than claiming the chunk64 graph alone solves the model. Neither outcome changes the original PCC or performance gates.
