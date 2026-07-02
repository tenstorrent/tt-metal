# Ring Joint SDPA Causal GQA Plan

## Scope

Add causal grouped-query attention support to `ttnn.transformer.ring_joint_scaled_dot_product_attention` for the existing tensor-K/tensor-V path. The first production target is Minimax3-style chunked prefill:

- Full model heads: `NQH=64`, `NKH=4`, `NVH=4`
- Head dims: `Dq=128`, `Dk=128`, `Dv=128`
- Deployment of interest: `SP=8`, `TP=4`
- Per-chip head shard at `TP=4`: `16` Q heads and `1` K/V head
- Long-context perf target: about `5k` Q tokens attending to about `55k` K/V tokens
- Supported GQA relationship: `NKH == NVH < NQH` and `NQH % NKH == 0`; `NKH == NVH == 1 < NQH` is in scope because it is the per-chip Minimax3 shape
- Joint tensors are out of scope and must remain rejected for GQA shapes
- `ring_mla` latent-V behavior is out of scope
- No new Python API argument; infer GQA from tensor head dimensions

Non-causal GQA is not a primary target for this change. Keep existing non-causal WAN/MHA coverage as regression protection, but choose new GQA tests and tuning around causal prefill.

## Implementation Plan

1. Add explicit head-mode classification in the program factory:

- `MHA`: `NQH == NKH == NVH`
- `SEPARATE_V_SHARED_K`: existing tensor-V shared-K path, `NKH == 1 && NVH == NQH`
- `GQA_GROUPED_KV`: `NKH == NVH < NQH && NQH % NKH == 0`

2. Preserve existing fast paths:

- Keep MHA query-head chains unchanged.
- Keep the existing shared-K batch chain for `SEPARATE_V_SHARED_K`.
- Do not route Minimax per-chip `NKH == NVH == 1 < NQH` through the shared-K separate-V path; it is grouped-KV GQA.
- Do not change `ring_mla` or latent-V materialization.

3. Relax validation only for tensor-K/tensor-V GQA:

- Allow `NKH == NVH < NQH` with divisibility.
- Reject joint-tensor GQA before kernel launch with a clear message.
- Keep latent-V constraints unchanged.
- Keep mode-specific validation so unsupported head relationships fail before kernel launch.

4. Build grouped K/V transport from the first implementation:

- Build GQA chains scoped by `(batch, kv_head)`, not query head.
- Move both K and V through the same grouped chain because `NKH == NVH`.
- Map `q_head -> kv_head` with `q_heads_per_kv = NQH / NKH`.
- For Minimax3 on `SP=8, TP=4`, each chip has one K/V head, so the grouped chain should transport one K/V stream for all 16 local Q heads.
- For the production local shape `B=1, NKH=NVH=1`, use row-wide grouped K/V multicast so the single K/V stream is read by one injector per active row and fanned out to the row's Q-head workers.
- If a core owns work from multiple KV groups, use a correctness-preserving local-read fallback unless scheduler changes can keep the target perf shape single-group per core without regressing MHA/MLA.

5. Update reader chain usage:

- Keep tensor addressing as `nk = nq / q_heads_per_k` and `nv = nq / q_heads_per_v`.
- In GQA mode, match chain receive/forward on `kv_head`, not `q_head`.
- Use a group-local Q iteration ordinal for forwarding, because a core's flat Q iteration order may include multiple query heads.
- Preserve existing reader behavior for MHA and separate-V shared-K modes.

6. Keep diagnostics useful:

- Log the selected head mode.
- Include `NQH`, `NKH`, and `NVH` in validation errors.
- Log grouped-chain counts and any local-read fallback cores for GQA.

## Test Plan

Extend `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`; do not add joint tensors.

### Build And Harness

Use the release build:

```bash
./build_metal.sh --release
```

Use the safe wrapper for accuracy and determinism:

```bash
scripts/run_safe_pytest.sh <pytest-target-and-args>
```

`scripts/run_safe_pytest.sh` is the default for long-running validation because it triggers `tt-triage` diagnostics on hangs. If an accuracy or determinism run stalls, use the emitted `tt-triage` artifact as the first diagnostic before rerunning manually.

### Reference Correctness

Update `torch_sdpa_reference` to support grouped K/V without materializing repeated heads:

- Slice normally when `KV heads == Q heads`.
- Otherwise require `Q heads % KV heads == 0`.
- Select K/V heads with `kv_indices = q_head_indices // (Q heads / KV heads)`.
- Compare output shape `[B, NQH, S, Dv]`.

### Accuracy

Primary GQA smoke config:

- Name: `minimax3_gqa_smoke`
- Per-ring heads: `nhq=16`, `nhk=1`, `nhv=1`
- Dims: `d_q=d_k=d_v=128`
- Causal and balanced
- Small sequence length for fast correctness
- Chunk choice: `q=128`, `k=512`

Required commands:

```bash
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy -k minimax3_gqa_smoke -svv
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy -k "wan2_2_1xGLX or mla_100k" -svv
```

The second command is the non-GQA regression check.

### Determinism

Required coverage:

- Existing repeated-run determinism for `minimax3_gqa_smoke`.
- Chunked final-chunk determinism for Minimax3 GQA.
- Three or more repeated chunked runs must be bit-exact.
- Use deterministic input tensors; accuracy is covered separately.

Required commands:

```bash
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism -k minimax3_gqa_smoke -svv
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_minimax3_gqa_chunked_determinism -svv
```

### Causal Performance

Primary perf shape:

- Name: `minimax3_55k`
- Per-ring heads: `nhq=16`, `nhk=1`, `nhv=1`
- Dims: `d_q=d_k=d_v=128`
- Causal chunked prefill
- On Galaxy `SP=8, TP=4`: total `64Q/4KV`, one K/V head per chip
- On QuietBox `SP=4, TP=1`: exercises the same local `16Q/1KV` GQA transport shape
- Use final chunk for perf and determinism because it is closest to production `5k` Q attending to the full K/V cache

Perf table goals:

- Use `q_chunk_size=128`, `k_chunk_size=512` for the production Minimax3 GQA target.
- The local QuietBox exploration covered `q={32,64,128,256}` and `k={512,640}`; `q=128,k=512` was selected because it matched the best measured duration/utilization, while Kimi-like `q=32,k=640` was significantly slower for Minimax's `D=128` shape.
- Report duration, effective cores, chunk FLOPs, math utilization, and Tracy FPU util.
- Prefer the fastest/highest-util chunk pair for the perf gate.
- Confirm grouped K/V multicast is used for the one-KV-head production local shape, not per-query-head duplicated K/V chains.
- Local QuietBox result after GQA K/V multicast and chunked causal perf-model correction: final chunk `q=128,k=512` ran in `1.053 ms` at `48.4%` math util, with Tracy `PM FPU UTIL` now reporting `48.4%-49.0%`; runtime improved from the pre-multicast local baseline of about `1.20 ms` at `42.4%`.

Expected local perf command:

```bash
CI=false RING_JOINT_CHUNKED_CHUNK_ID=10 pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_minimax3_gqa_create_chunked_perf_table -svv
```

Perf-check goal:

- Add a CI-gated Minimax3 GQA chunked perf check after local baselining.
- Use the same `SDPA_PERF_CHECKS=1` and symmetric margin pattern as the existing Kimi chunked perf gate.
- Gate the final chunk only.
- Baseline QuietBox and Galaxy separately; do not invent the Galaxy expected value from the QuietBox run.

Expected perf-check command after baselining:

```bash
SDPA_PERF_CHECKS=1 pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_minimax3_gqa_chunked_perf_check -svv
```

## Success Criteria

The change is successful only if all gates below pass.

### Functional

- Tensor-K/tensor-V causal GQA supports `NKH == NVH < NQH`, including `NKH == NVH == 1 < NQH`.
- Existing MHA and `NKH == 1, NVH == NQH` separate-V behavior is preserved.
- Joint-tensor GQA fails before kernel launch with a clear unsupported validation message.
- `ring_mla` latent-V behavior is unchanged.
- Program-cache reuse does not alias GQA and non-GQA programs.

### Accuracy

- `./build_metal.sh --release` completes.
- `minimax3_gqa_smoke` passes against the grouped PyTorch reference through `scripts/run_safe_pytest.sh`.
- Existing WAN and MLA accuracy smoke tests still pass through `scripts/run_safe_pytest.sh`.
- Persistent K/V all-gather buffers and tensor addressing use `NKH/NVH`, not expanded `NQH`.

### Determinism

- `minimax3_gqa_smoke` repeated-run determinism passes through `scripts/run_safe_pytest.sh`.
- Minimax3 final chunk determinism passes through `scripts/run_safe_pytest.sh`.
- Repeated GQA outputs are bit-exact under the same conditions as existing determinism coverage.

### Performance

- GQA uses grouped `(batch, kv_head)` K/V transport from the first implementation.
- Production local `B=1, NKH=NVH=1` GQA uses row-wide grouped K/V multicast; broader multi-KV-head GQA may remain on grouped unicast unless separately optimized.
- Debug logs or profiler behavior confirm K/V traffic is not duplicated once per query head in the GQA group.
- Minimax3 final-chunk perf table runs to completion and reports valid `RingJointSDPADeviceOperation` durations.
- For chunked causal prefill with `kv_actual_isl`, the internal op perf model reports the prefix rectangle plus current-chunk triangle instead of the generic `Sq*Sk/2` causal approximation.
- On the local QuietBox baseline, `q=128,k=512` is the selected perf point unless repeated profiling on the target topology shows a better causal chunk choice.
- Existing MHA, MLA, and Kimi chunked perf checks remain within their current expected bands.
- A Minimax3 GQA CI perf gate is added only for topologies with measured baselines.

### Hang Debugging

- Accuracy and determinism runs use `scripts/run_safe_pytest.sh`.
- Any hang is diagnosed first from the `tt-triage` artifact emitted by the safe wrapper.

## Definition Of Done

- Release build passes.
- Accuracy passes for `minimax3_gqa_smoke`.
- Non-GQA accuracy smoke passes.
- Determinism passes for `minimax3_gqa_smoke`.
- Final-chunk Minimax3 GQA chunked determinism passes.
- Final-chunk Minimax3 GQA chunked perf table passes for `q=128,k=512`.
- Existing perf gates pass or are explicitly left unchanged with no evidence of regression.
- Plan and tests document that the production target is causal Minimax3 with one K/V head per chip at `SP=8, TP=4`.

## Non-Goals

- Joint-token or joint-tensor GQA.
- `NKH != NVH` GQA.
- Non-causal GQA optimization.
- Broad scheduler refactors outside `GQA_GROUPED_KV`.
- Inferring Galaxy perf thresholds from QuietBox-only profiler data.
