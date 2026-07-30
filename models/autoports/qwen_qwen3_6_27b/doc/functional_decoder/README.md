# Qwen3.6-27B functional decoder

Status: functional-decoder complete; independent stage review `clean-pass`.

## Target contract

The immutable target is `Qwen/Qwen3.6-27B` revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Transformers resolves it to a
`Qwen3_5TextConfig` with hidden size 5120, 64 layers, and advertised context
262,144.

There are two meaningful decoder kinds:

| Kind | Layers | Representative | Token mixer and cache |
|---|---:|---:|---|
| `linear_attention` | 48 | 0 | gated delta net; four-token depthwise-convolution state and 48x128x128 recurrent state per batch row |
| `full_attention` | 16 | 3 | 24 query heads, 4 KV heads, head dim 256, gated query output, Q/K norm, partial MRoPE, paged KV cache |

Both kinds use pre-token-mixer RMSNorm, residual addition, post-token-mixer
RMSNorm, and a dense SwiGLU MLP with intermediate size 17,408.

The public runtime API is keyword-friendly:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    hf_config=text_config,
    layer_idx=...,
    mesh_device=one_by_one_mesh,
    batch=...,
    max_context=...,
    page_size=...,
)

decoder.prefill_forward(
    hidden_states=...,
    page_table=...,
    current_positions=...,
)

decoder.decode_forward(
    hidden_states=...,
    page_table=...,
    current_positions=...,
)
```

`hidden_states`, `page_table`, and `current_positions` are device tensors.
Full-attention prefill fills the paged KV cache and decode updates it at the
device-resident current positions. Linear-attention prefill/decode updates its
convolution and recurrent states. Runtime forward methods must not call Torch,
`ttnn.from_torch`, or `ttnn.to_torch`.

## Evidence

| Gate | Result | Artifact |
|---|---|---|
| 1x1 Blackhole mesh open/close | pass, 2026-07-29 | `work_log.md` |
| Linear-attention synthetic decode PCC | 0.999997956 | `tests/linear_attention_synthetic_pcc.py` |
| Linear-attention synthetic prefill PCC, seq 5 | 0.999998057 | `tests/linear_attention_synthetic_pcc.py` |
| Full-attention synthetic decode PCC | 0.999960815 | `tests/full_attention_synthetic_pcc.py` |
| Full-attention synthetic paged prefill PCC, seq 33 | 0.999729385 | `tests/full_attention_synthetic_pcc.py` |
| Official-weight layer-0 decode PCC | 0.999921858 | `tests/linear_attention_real_pcc.py`, `real_weight_decode.log` |
| Nonzero traced decode, both kinds, batch 1/32 | pass; two sequential state/cache-mutating steps, stable input/position updates, PCC 0.998809–0.999991 | `autofix_trace_cache/*.log`, `tracy_nonzero/*` |
| Traced decode latency, full attention b1/b32 | 2.4741 / 2.6707 ms | `tracy_nonzero/full_b*/` |
| Traced decode latency, linear attention b1/b32 | 3.1645 / 21.5003 ms | `tracy_nonzero/linear_b*/` |
| Cache-dependent page routing | seq65 prefill → pos65 decode, page table `[1,0]`, PCC 0.999905 plus physical cache assertions | `autofix_trace_cache/full_cache_prefill_decode.log` |
| Advertised-context decode | pass at position 262,143, batch 1, full paged allocation | `full_attention_context_262144.log`, `linear_attention_context_262144.log` |
| Long non-divisible prefill | 192,511 pass; 194,559 hard DRAM OOM | `full_attention_prefill_192511.log`, `full_attention_prefill_194559.log` |
| Tile/page/chunk boundaries | 32/33, 64/65, 32,769 pass; permuted page table at position 65 passes | boundary/context logs |
| Repeated-input determinism | exact equality across trace replays | trace logs |
| Runtime fallback audit | clean for public and helper runtime methods | `tests/test_functional_decoder.py` |
| Linear near-context prefill | public full layer seq192511 pass in 474.957 s; seq262143 hard MLP DRAM OOM | `linear_prefill_target_192511_chunked.log`, `linear_prefill_target_262143_chunked.log` |
| Runtime fallback dynamic audit | pass with `throw_exception_on_fallback=true` | `fallback_audit_*.log` |
| Watcher-clean run | pass, meaningful nonzero traced paths, batch32, interval 10 | `watcher10/*` |
| Warmed prefill/decode perf reports | pass; nonzero decode raw ops CSV + filtered CSV + human table | `tracy_nonzero/*`, `tracy/*prefill*` |
| Stage review clean-pass | pass, no required work | `stage_review.md` |

The functional contract remains 262,144 tokens for decode. A complete
single-device full-attention layer prefill is physically limited to 192,511
tokens in this untuned functional implementation: 194,559 fails because a
2,390,753,280-byte allocation needs 298,844,160 bytes per bank while the
largest free block is 283,529,088 bytes per bank. Batch-32 KV at the advertised
context is exactly 32 GiB before weights and workspace, so serving-batch trace
and latency use context 64. These are recorded reductions backed by allocator
evidence, not smaller model configuration values.

Linear attention uses 64-token vectorized causal convolution plus a logarithmic
affine gated-delta scan. Full-layer HF PCC is 0.999998050 at sequence 5 and
0.999997842 at the 64+1 chunk boundary. Its public target-shape prefill also
passes at 192,511 tokens; 262,143 reaches a hard MLP DRAM allocation failure
(9,126,805,504 bytes requested, 1,140,850,688 per bank versus an 856,953,216
largest free block). Thus both layer kinds share the evidence-backed 192,511
single-pass prefill limit while retaining advertised 262,144-token decode.

Warmed full-layer prefill measured 3.730 ms at full-attention sequence 33 and
11.629 ms at linear-attention sequence 5. The human-readable reports and
filtered CSVs use `Device Time` in microseconds. Decode reports cover ten
traced replays in the original structural harness; the acceptance reports under
`tracy_nonzero/` cover two sequential nonzero replay steps between
`PERF_DECODE` signposts.
