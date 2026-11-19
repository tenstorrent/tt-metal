# Compact local Q/K partial-RoPE fusion proposal

CPU source audit, 2026-11-13. No device import, build, timing, or numerical claim.

The current chain in `tt/multichip_decoder.py:695–739` first normalizes six Q
heads and one K head per user, then permutes and reshards both trig tensors,
slices/reshards each 64-channel prefix, calls native rotary twice, moves both
prefixes back to DRAM, slices both tails, concatenates, and finally reshards K
to cache-update owners. These operations can be replaced by one local Q/K
norm-plus-partial-RoPE program while retaining the existing cache update and SDPA.

Proposed contract: Q/K inputs retain `[1,B,6/1,256]` BF16 TILE height shards on
the first B eight-column row-order cores. Original trig is BF16 TILE DRAM
`[B,1,1,64]`, already evaluated at each user's position. A user owns two trig
tiles at `user*2 + tile`; row0 broadcasts across that user's heads. Read these
original tiles directly, avoiding the current permute and rotary-specific
rectangular sharding. Positions are never inferred from batch or shared across
users/ranks. `current_pos` still goes unchanged to cache update and SDPA.

Use Q workers on logical core index `user` and K workers on index `32+user`,
each interpreted as `(index%8,index//8)`. K reads its immutable input remotely
from owner `user`, as the existing parallel norm already does. Q writes its
eight output tiles to DRAM pages `user*8+tile`; K writes its eight tiles directly
to its owning `[32,256]` L1 shard. Return Q `[1,B,6,256]` DRAM and K
`[1,B,1,256]` HEIGHT_SHARDED on offset32 owners. This exactly matches the current
`k_update` destination and leaves V on the first B cores. Native fused cache
update requires disjoint equal-sized input core sets
(`experimental/paged_cache/device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:237–246`).
No collective or inter-layer residual layout changes are involved.

Preserve arithmetic boundaries. Existing parallel norm explicitly casts and
packs its FP32 affine result into BF16 (`parallel_qk_norm_candidate.py:76–88`).
Keep that CB boundary before rotary, including for the six untouched tail tiles.
For prefix tiles0/1, implement `rotate_half(x)=[-x1,x0]`, then sin product,
cos product, then addition. The actual native decode path for width64 selects
`create_multi_tile_decode_descriptor`, not the width32 single-tile transform.
Its factory assigns rotated intermediate the input dtype, sin intermediate the
sin dtype, cos intermediate the cos dtype, and output the input dtype
(`rotary_embedding_hf_sharded_program_factory.cpp:220–377`). Its compute kernel
packs each product before adding
(`kernels/compute/rotary_embedding_hf_sharded.cpp:131–203`). With current BF16
trig this is two BF16 product boundaries and a BF16 sum, in addition to BF16
normalization. Reuse the native `ckl::BinaryFpu` and pack sequence with HiFi4,
FP32 destination. A fused FP32 multiply-add would change these boundaries.
Copy normalized tiles2..7 unchanged; do not renormalize or recompute their bits.

Sparse exact core sets support every logical B1..32, including B3/17/31, without
the old rotary rectangle/divisor constraint. Each user still owns a padded
32-head tile; valid six/one heads must never reduce with padded head rows. The
candidate owns all tile movement, so it introduces no public alignment rule.

Make output buffers optional caller-owned arguments. Validate their complete
shape/dtype/layout/mesh ownership and nonaliasing, and require stable owners
across capture/replay. All input/output/trig/weight addresses must be explicit
runtime args and descriptor tensor operands. Cache replacement tests must keep
old allocations live and verify addresses differ, then compare the new output
against the matched native chain. Trace tests update Q/K and per-user trig in
place; never specialize a program on a position value. Account persistent Q/K
buffers if enabled in decoder setup; do not share them between concurrent owners.

Validation before integration: compare against **selected parallel norm plus
the existing native rotary/layout chain**, also report original native RMSNorm
chain as a control. Cover B1/3/17/32 then all B1..32, independently varying rank,
user, position and generation, zero heads and poisoned padded rows. Require
unchanged .995 PCC gates, finite logical outputs, exact untouched tails, and
bit-identical changed-input trace/eager and rebind/eager outputs. Report bit
differences against native rotary rather than hiding them behind PCC; investigate
any mismatch in the intended exact rotary sequence. Run watcher/allocation
tracking separately from matched warmed trace timings. Whole decoder/state and
stress gates follow any candidate integration. No speedup is asserted here.

All unqualified paths above are under the model autoport; native rotary paths
are under `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/`.

## Opt-in implementation handoff

Added only `tt/parallel_qk_rope_candidate.py` and
`tests/probe_parallel_qk_rope.py`; existing helpers, native kernels and decoder
selection are unchanged. The candidate has explicit preallocated outputs,
per-user trig access, exact native FPU intermediate formats, and a fixed K
destination owner. Its writer waits for the rotated two-tile prefix, then writes
that prefix and the six untouched normalized tiles before acknowledging NoC
completion. The normalized CB remains live until both writes complete.

The probe uses `Qwen3_5TextRotaryEmbedding` with the repository HF config,
different positions per rank/user/generation, zero valid heads, and NaN padding
whose presence is verified from actual device backing tiles. It compares three
whole chains: original native RMSNorm plus rotary, selected parallel RMSNorm
plus rotary, and the candidate. It requires .995 PCC to both controls and a
rounded torch oracle, exact bits versus the selected chain (including tails),
exact changed-input trace/eager, and cache rebinding with changed input/trig and
weights while retaining the old buffers. It reports distinct addresses and cache
entry counts. Allocation tracking includes a positive rejection control.
Uninstrumented timing uses both forward and reverse path order; watcher,
profiler or allocation-tracked runs do not publish timing results.

CPU validation: Python AST parsing and `--cpu-check` PASS, covering ownership
for all32 logical batches without torch/TTNN imports. Black formatted the two
new files. No native compilation, device execution, arithmetic equality or
speedup is claimed yet. A first compilation/API/layout failure must be repaired
before evaluating this material conversion-removal candidate.

Suggested parent-serialized commands from repository root (use fresh artifact
tags if retrying):

```sh
M=models/autoports/qwen_qwen3_6_27b
D=$M/doc/optimized_multichip_decoder
TT_METAL_WATCHER=10 MULTICHIP_JOB_TIMEOUT=900 bash "$D/run_job.sh" qkrope_smoke_v1 env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python "$M/tests/probe_parallel_qk_rope.py" --batches 1 --steps 3 --output "$D/qkrope_smoke_v1.json"
TT_METAL_WATCHER=10 MULTICHIP_JOB_TIMEOUT=1800 bash "$D/run_job.sh" qkrope_allb_watcher_v1 env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python "$M/tests/probe_parallel_qk_rope.py" --all-batches --steps 3 --output "$D/qkrope_allb_watcher_v1.json"
MULTICHIP_JOB_TIMEOUT=900 bash "$D/run_job.sh" qkrope_perf_v1 env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=0 python "$M/tests/probe_parallel_qk_rope.py" --batches 1 3 17 32 --steps 3 --samples 7 --replays 50 --output "$D/qkrope_perf_v1.json"
```

Initial source SHA256:

- Candidate: `23d22234e9eb4dbfcd79c92476575d19e1fe12d0974cf28f942dedeb565d0f43`.
- Probe: `6fadc13ca931286472f8319801be974805e5ac320f5a56569724bf2d054718b0`.
