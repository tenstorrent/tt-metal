# AutoDebug: batch-32 packed gate/up L1 collision

## Scope and verdict

This was an inspection-only analysis of the selected Linear TP4 decoder. No
source was changed and no hardware was used.

The failure is a late program-cache compile of the batch-32 packed gate/up
matmul after the shared persistent `attention_o` and `mlp_down` all-reduce
scratch has become resident. The batch-1 and batch-32 tensors have the same
one-tile padded M extent, but their different logical shapes produce different
matmul cache keys. Batch 1 compiles its packed program before `mlp_down`
scratch exists; batch 32 must compile a new entry while both scratch roles are
live. Its static circular buffers end at 1,163,136 while a retained global L1
buffer begins at 854,272, an overlap of 308,864 bytes.

The smallest behavior-preserving fix is a static logical-M branch inside the
decode MLP: retain the selected packed BFP8 gate/up projection exactly for
logical M=1, and for M>1 use the already-loaded TP-local `gate_decode` and
`up_decode` weights as two N=5,376 BFP8 projections. Spill the first projection
to DRAM before compiling/running the second, as the single-chip optimized
implementation already does, then perform GELU, reshard, multiply, down
projection, and the existing persistent reduction entirely on device. This
halves the dominant N-dependent packed-matmul CB family at the failing compile
boundary without changing batch-1 latency, persistent addresses, TP4
placement, or the public non-aligned contract.

## Evidence-ranked findings

### 1. Confirmed: batch 32 hits a new matmul compile under resident scratch

Confidence: high.

- The test maps host `[32, 1, 5376]` to logical device
  `[1, 1, 32, 5376]`, then calls a warm decode before trace capture
  (`test_multichip_decoder.py:55-68,588-632`). The failure is therefore not a
  replay-only or asynchronous teardown symptom.
- The stack reaches `_TPOptimizedSharedMLP.__call__` and fails at its packed
  `ttnn.linear` (`multichip_decoder.py:419-434`), while
  `create_and_cache_mesh_workload` is compiling program 1238
  (`final_suite.log:90-123`). Attention and non-aligned prefill have already
  completed.
- The validator reports static-CB end 1,163,136 and global-L1 start 854,272 on
  the worker range, a 308,864-byte overlap (`final_suite.log:90-110`). This is
  a deterministic admission/capacity failure, not a PCC failure.
- The earlier batch-1 traced tests pass and materialize a shared pool with
  exactly the `attention_o` and `mlp_down` roles
  (`test_multichip_decoder.py:455-503`). The pool is module-global by mesh
  identity and retains semaphores and buffers for the mesh lifetime
  (`multichip_decoder.py:63,556-570`). `_tp_allreduce` retains each buffer in
  `pool["buffers"]`; it deallocates only transient input/output tensors
  (`multichip_decoder.py:124-201`).
- Matmul program hashing includes the input tensors and therefore their tensor
  specs (`matmul_device_operation.cpp:2127-2149`); TensorSpec hashing includes
  logical shape. Thus logical M=1 and M=32 require distinct cache entries even
  though both pad to one 32-row tile. The batch-1 entry was compiled before
  the first MLP-down reduction allocated its scratch; the batch-32 entry is
  compiled later with both scratch roles resident. This explains the reported
  passing/failing boundary.

### 2. Confirmed: the selected packed projection creates the largest local CB family

Confidence: high for the source-level size relationship; hardware is required
to prove the final allocator fit.

- Gemma 4 31B has hidden width 5,376 and intermediate width 21,504. TP4 makes
  the local intermediate width equal to 5,376
  (`multichip_decoder.py:232-238`; `real_weight_stats.json:9-11`).
- The selected policy is packed gate/up with BFP8 output, 14 cores, and block
  width 12 (`multichip_decoder.py:54,57-62,589-595`). The failing projection is
  therefore M=32, K=5,376, N=10,752 with BFP4 weights. Its width-sharded output
  is `[32, 768]` per core (`multichip_decoder.py:316-340,419-433`).
- The DRAM-sharded matmul factory sizes its in0, triple-buffered in1, output,
  intermediate, and output-reshard CBs from block width and per-core N
  (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:181-229`).
  Splitting packed N=10,752 into two N=5,376 projections halves every material
  N-dependent member, including the dominant triple-buffered BFP4 weight CB.
  Source-level sizing predicts a reduction greater than the observed 308,864
  byte deficit.
- A block-width-only branch shrinks only K-block-dependent CBs, leaves the
  output/intermediate/reshard N family intact, and has no evidence that it
  clears this large deficit. The two separate weights are already resident
  and are the supported topology alternative (`multichip_decoder.py:250-285,
  451-466`).

### 3. Secondary pressure: the persistent key distinguishes logical rows

Confidence: high as a capacity issue, but not sufficient alone to fix the
observed late compile.

The persistent key includes logical `rows` and `width` in addition to dtype,
shard shape, grid, orientation, role, and slot
(`multichip_decoder.py:138-155`). Batch 1 and batch 32 therefore cannot reuse a
role/slot buffer even when their physical one-tile shard capacity is the same.
The all-reduce validator itself requires only compatible WIDTH_SHARDED memory,
grid containment, and sufficient shard volume
(`all_reduce_async_device_operation.cpp:18-72`), and the program factory uses
the buffer only as a globally allocated CB address
(`all_reduce_async_program_factory.cpp:356-377,600-630`). Canonicalizing that
key to physical shard capacity is a valid follow-up capacity improvement.

It is not the smallest complete fix for this failure: even without allocating
a new batch-32 scratch tensor, the already-resident batch-1 `mlp_down` scratch
still distinguishes the late M=32 compile from the earlier M=1 compile. The
projection CB family must fit while the existing two-buffer pool remains live.

## Recommended intervention boundary

Branch only on the static logical row extent available during graph
construction:

1. `hidden_states.shape[-2] == 1`: keep the current packed BFP8 path byte-for-
   byte so the measured default trace and latency do not change.
2. `hidden_states.shape[-2] > 1`: use `up_decode` and `gate_decode` separately,
   request `policy.mlp_packed_output_dtype` for both outputs, spill the first
   result to DRAM before the second projection, apply GELU and restore both to
   the existing 14-core local memory config, then continue through the existing
   multiply/down/reduction path.

This is not a host fallback: the conditional reads immutable tensor metadata,
and every data operation remains TTNN/device-side. Do not clear, deallocate, or
replace `pool["buffers"]`; captured traces depend on stable device addresses.
Do not change the public batch or sequence alignment requirement. Any logical
M>1 must select the safe path, including non-power-of-two batches.

If BFP8 separate outputs fail the accepted PCC threshold, retry the same
separate branch with BF16 outputs and record both attempts. Do not change the
batch-1 packed default as part of that adaptation.

## Focused verification

First run the exact ordered contrast that populates persistent scratch with
batch 1 and then compiles batch 32:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} \
PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools \
timeout 900 pytest -q -s -x \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k 'paged_decode_trace_matches_optimized_baseline or batch32_nonaligned_prefill_and_traced_decode'
```

Acceptance requires both layer kinds in both test families, accepted PCC on
all four replicas, deterministic repeated trace replay, and no increase or
address change in the two existing persistent role buffers across the M>1
tests. Then run the following risk-matched matrix:

- isolated and resident-pool batches 2, 3, 31, and 32 for sliding and full
  attention;
- non-aligned prefill-33 followed by decode for batch 32 and one non-power-of-
  two batch;
- before/after batch-1 traced warmed latency and an assertion that its MLP
  topology remains packed BFP8;
- watcher-clean batch-32 focused tests in a separate non-profiler run; and
- the full standard suite in its normal ordering.

Record the pool object, role/key count, and device buffer addresses before and
after M>1 calls. A fresh-process batch-32 pass alone is insufficient because it
does not reproduce the late-compile coexistence boundary.

## Claim review and remaining uncertainty

The source and log evidence account for the exact failing operation, byte
overlap, suite-order dependency, cache miss, persistent lifetime, and why
packed N is material. They do not prove allocator fit, PCC, or latency after
the proposed branch; those require the target 1x4 Blackhole mesh. The report
therefore recommends a focused hardware verification rather than claiming the
fix is already proven.
