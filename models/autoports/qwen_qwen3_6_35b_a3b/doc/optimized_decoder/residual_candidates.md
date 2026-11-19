# Residual layout candidates (unmeasured)

`tests/optimized_residual_candidates.py` defines `ResidualCandidateDecoder`, a
subclass of `ProjectionCandidateDecoder`, and a CLI using the same real-weight
`run_decoder.run` correctness, traced replay, timing, and profiler harness.
No production module or existing candidate module changed. No device was opened
for this CPU-only implementation task.

## Candidate path

With `residual_mode="sharded"`, full-attention public decode carries its BF16
residual in the inherited `norm_memory`: 32 width-sharded L1 cores, each shard
32 × 64. Its attention output projection writes directly into that layout with
`per_core_N=2`. Both residual additions and both norms consume the same layout.
The post-attention normalized activation still converts once to DRAM for the
unchanged router and sparse/shared experts. The final add accepts the sharded
residual and DRAM MoE result and writes a sharded output. The returned logical
shape is still `[1,1,2048]`; inherited `decode_batch` returns a tuple of rows.

A next full-attention decoder can accept that output directly: its first
`to_memory_config(..., norm_memory)` is already satisfied, and its input RMSNorm
consumes this exact layout. This is a proposed compatible contract, not a
measured stack result. Selection needs a device check feeding the sharded result
into the next input norm against a matched interleaved control, plus ordinary
whole-layer HF PCC and repeated traced replay.

The internal dispatch flag is set only around this Python decode call and
restored with `finally`. It selects whether attention output retains its shard;
it is not tensor data, a callback, a Torch operation, or a host synchronization.
All trace commands operate only on TT tensors.

Prefill, including singleton tails marked `cache_boundary=False`, and all
linear-attention paths continue through the parent `_forward`. Prefill does not
receive sharded singleton returns that would change its concat contract. Full
attention output geometry is still a setup-time policy shared by decode and any
singleton prefill calls; those singleton calls use the parent's DRAM return.

## Proposed comparisons

| Candidate | Extra constructor kwargs | Meaning |
|---|---|---|
| Control | `residual_mode="control"` | Parent projection candidate, inherited residual movement |
| Full sharded residual | `residual_mode="sharded"` | Attention output, both adds, norms and public result keep L1 shards |
| Output adaptation control | Above plus `residual_return_dram=true` | Quantifies final interface conversion; cannot reject the fully sharded family by itself |
| Explicit MoE reshard | Above plus `residual_moe_reshard=true` | One extra DRAM→L1 conversion if mixed-input binary has a runtime issue or is slower |

All rows must use the same selected precision, attention packing, and expert
geometry policy. The sharded row overrides `attention_out.per_core_n` to 2;
record this as an integrated geometry/layout candidate. Compare it to a tuned
control with `attention_out.per_core_n=2` as well as the best control geometry.
The complete effective projection geometry and residual policy appear in the
normal result JSON.

Example (owner supplies current best projection/expert kwargs and serializes):

```bash
timeout 600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.optimized_residual_candidates \
  --layer 3 --length 256 --context 512 --output <artifact>.json \
  --build-kwargs '{"residual_mode":"sharded","residual_return_dram":false,"residual_moe_reshard":false}'
```

## Mixed-memory binary source evidence

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`
validates operand A, operand B and output independently as sharded or
interleaved. It imposes no both-inputs-sharded requirement on this add.
`tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py`,
`test_binary_bcast_sharded_output_half_mem_config`, exercises sharded plus
interleaved inputs in both orders with sharded output. Thus the default tests
mixed-memory final addition directly, avoiding an assumed mandatory reshard.
The optional explicit MoE reshard is an available adapted experiment if needed;
no API rejection or performance conclusion has been recorded.

Inherited full-attention `norm` returns `x.memory_config()` for one-row decode.
The inherited sharding spec and the forced projection output spec coincide;
this removes projection→DRAM and residual→norm-shard conversions at that
boundary without changing the norm arithmetic or cache/state formats.

Remaining device evidence: matched whole-output PCC, final memory config and
next-norm equality, non-aligned prefill including singleton-tail concat,
repeated replay with allocation tracking, warmed latency against best correct
control, and profiler rows. Layout, trace allocation lifetime, and actual
binary-kernel movement remain unverified until those runs. Python compilation
and Black formatting passed.
