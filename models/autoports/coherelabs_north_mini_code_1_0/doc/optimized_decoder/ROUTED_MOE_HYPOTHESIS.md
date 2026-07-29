# Batch-32 Routed-MoE Hypothesis

## Conclusion

No current model-local TTNN composition provides a complete, compact,
fabric-free batch-32 routed output. The selected dense all-128-expert branch
therefore remains a real optimize-contract blocker, not an optimization that
can be signed off.

The failure is at the output/combine contract:

- `ttnn.sparse_matmul` retains dense batch dimensions in its output shape and
  zero-fills the complete output before every launch
  (`sparse_matmul_device_operation.cpp`). For North-Mini decode this is the
  `32 * 128` token/expert surface for gate, up, and down. Exact
  `nnz=32*8=256` reduces kernel loops but not allocation, zero-fill,
  elementwise, or reduction volume; the retained correct result is
  17.831 ms versus the selected dense path's 3.330 ms.
- `ttnn.experimental.moe_compute(compute_only=True)` is the only current
  compact active-token compute candidate at these exact dimensions. It
  measured 1.642 ms traced, but output slot 4 is intentionally a rolling
  two-buffer tensor shaped `[cores, 2, 32, 2048]`. Earlier expert outputs are
  consumed/reused in-kernel and are not persistent at the API boundary.
  Launching ordinary TTNN reductions afterward can see only the final buffer
  contents, not all routed contributions.
- `moe_compute(compute_only=False)` is the only existing consumer that drains
  those buffers while experts run. The public API requires `cluster_axis`;
  `SelectiveReduceCombine` requires `num_links > 0`, and its factory
  unconditionally constructs fabric neighbors/mux workers. The writer always
  opens fabric connections and performs fabric completion synchronization.
  There is no one-device/local-only branch. The retained 1x1 attempts failed
  first without a fabric context and then waiting for a remote handshake.
- Calling standalone `ttnn.experimental.selective_reduce_combine` after
  `compute_only=True` cannot repair this: compute-only explicitly disables
  the producer/consumer semaphore protocol, and the shared double buffer has
  already been recycled when the op completes.

## Other model-local families checked

| Family | Result |
|---|---|
| Dynamic `sparse_matmul` | Correct full output, 20.535–21.896 ms; full surface remains. |
| Binary mask + exact static `nnz` | Correct, 17.831 ms; required DRAM intermediates after a 201,326,592-byte sparse output exceeded L1. |
| Packed static sparse gate/up | Correct, best 19.584 ms; wider full-surface projection loses. |
| `moe_gpt` | Same rolling two-buffer output contract and requires `selective_reduce_combine` for correctness. |
| DeepSeek `unified_routed_expert_moe` | Not a practical decode escape hatch: it requires a pre-compacted dispatched buffer, BFLOAT8_B DRAM input, Blackhole 11x8, launches once per expert, and its minimum internal chunk is 16 tiles (512 rows) even when an expert has at most 32 batch-32 tokens. The available dispatcher is itself fabric-oriented and calls neighbor discovery; it has no supported one-device/no-link path. |
| Per-expert/grouped `moe_compute` calls | Could retain at most two experts per call, implying up to 64 fused launches and dynamic expert-weight selection/repacking not expressible by the current on-device APIs. It is not a trace-stable no-regression candidate. |
| Host route extraction/combine | Semantically possible but violates the device-resident traced runtime contract. |

## Exact blocker

A shared TTNN capability is required in one of these forms:

1. `sparse_matmul` compact output indexed by live route, e.g.
   `[token, top_k, N]`, without full token-by-expert zero-fill;
2. persistent complete routed outputs from `moe_compute(compute_only=True)`;
3. a local-only consumer fused to `moe_compute` that drains the rolling buffer,
   applies the supplied expert scores, and reduces to `[tokens, hidden]`
   without fabric neighbors, muxes, or cross-device semaphores.

The third is the narrowest likely change: keep the existing fast tilize and
expert kernels, add a `LocalCombine` path selected when the logical mesh is
1x1, write/accumulate directly to a preallocated `[32, 2048]` output, and
skip all `SelectiveReduceCombine` fabric setup. This cannot be implemented
from `optimized_decoder.py` because the needed data disappears inside the
device program before any model-local TTNN op can consume it. Shared TTNN
edits are excluded by this task's scope.

## Focused follow-up commands

After the shared API exists, keep the integration model-local and first add a
branch guard proving batch 32 cannot call `_dense_expert_moe_chunk`. Run only
the focused real-weight decode test under watcher:

```bash
TT_METAL_WATCHER=10 timeout 300 pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  -k 'routed_moe_b32_compact and decode'
```

Then compare the complete traced layer against the selected 3.330-ms control,
including router, compact compute, score weighting, reduction, and residual:

```bash
timeout 300 python -m tracy -r -p -v -m pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  -k 'routed_moe_b32_compact and decode'
```

Acceptance requires authentic-weight PCC at the current bar, ten stable trace
replays, no dense-expert call in source/runtime audit, no host readback, clean
watcher, and whole-layer batch-32 latency no worse than the selected default.
Until that shared capability lands, the honest stage result is
`more-work-needed (shared TTNN routed-output blocker)`.
