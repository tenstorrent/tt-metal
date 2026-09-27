# Full-model serving work accounting

`tools/benchmark_work.py` is a standard-library-only source-derived estimate. It
provides no times and imports no TT device code. `WorkAccounting` rejects reduced
layers, reduced context, another mesh, and policies different from the checked-in
selected policy. The collector must independently bind runtime geometry and policy
to these inputs. Source hashes accompany each accountant instance. Runtime events,
not HTTP concurrency or configured server slots, determine execution membership.

The participating hardware is four Blackhole ASICs (two P300 cards), TP4 decode
and EP4 prefill. Aggregate theoretical DRAM bandwidth is 4 × 512 GB/s. Aggregate
LoFi arithmetic peak is 4 × 120 physical cores × 4096 FLOP/core/cycle × clock Hz,
or 2.654208 PFLOP/s at nominal 1.35 GHz. The [P300 specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html)
provides card core count and bandwidth; the existing
`doc/optimized_decoder/final_roofline_audit.md` records the arithmetic-cycle basis.
The supplied clock must be labeled nominal or observed by the collector. The
available worker grid does not reduce the physical ASIC theoretical denominator.

This is explicitly a **useful work / LoFi peak upper-envelope ratio**, not measured
FPU utilization or a precision-weighted attainable throughput. The selected graph
mixes LoFi, HiFi2, HiFi4, library-default shared-prefill matmuls, SFPU normalization
and transcendental work. HiFi paths cannot attain the common LoFi denominator.
There is no invented effective fidelity or inferred device time. Full host phase
time, including host work, communication and gaps, belongs in the denominator;
these estimates must never enter telemetry's device-time fields.

## Prefill numerator

For each real initial input of length S, sum all 30 layers (25 sliding, five full)
plus one last-token vocabulary projection. The last-token restriction comes from
`Gemma4Model.prefill_forward`; continuation and all-token logits are unsupported.
For H=2816, Q=16, shared width=2112, expert width=704, top-k=8, E=128:

- Projection FLOPs: `2*S*H*(2*Q*D + KV*D*(2 if sliding else 1))`.
  Sliding has D=256, KV=8; full has D=512, KV=2 and tied K=V.
- Shared MLP: `6*S*H*2112`; active experts: `6*S*H*704*8`.
- Router: `2*S*H*128`.
- Causal QK and PV: `4*Q*D*pairs`; full pairs `S*(S+1)/2`, sliding
  pairs `S*min(S,1024)-min(S,1024)*(min(S,1024)-1)/2`.
- Last-token head: `2*H*262144` per request.

Multiply-add counts as two FLOPs. Replicated global KV projections, physical TP
width padding, masked attention, EP union experts not selected by a token,
normalization and transcendental operations are excluded from useful FLOPs.
Their time is retained in the complete phase denominator. At S4096, the layer
terms independently reproduce the accepted single-layer useful-work audit.

## Decode numerator

`decode(positions, batch_slots=...)` describes one actual invocation. `positions`
contains active executed rows. `batch_slots` is the model head's logical batch
span, including inactive zero rows, and is distinct from server capacity. Actual
recorded invocations are summed, including extra asynchronous submissions if any;
never multiply one B1 model total by HTTP concurrency to approximate a batch.
The headline128 emitted tokens ordinarily uses one prefill result plus127 decode
positions4096–4222, but observed event counts take precedence.

`OptimizedDecoder.decode_forward` loops independently over every row. Thus each
active row rereads each layer's projections, shared MLP, router and eight indexed
experts. There is **no assumed expert union reuse across requests**. Conversely,
`Gemma4Model.logits` projects the whole model batch at once: its sharded vocabulary
weights are counted once per invocation, not once per row. TP local expert width
is192 (704/4 padded to32), shared width544 (2112/4 padded to32), local query heads4,
local KV heads2 sliding or1 full. Global KV is replicated across ASIC pairs.
Full runtime QKV includes both stored K and V columns even though useful FLOPs
exclude the redundant tied projection. Router weights are replicated per ASIC.

Every tiled operand uses32×32 tiles with payloads BF16=2048, FP32=4096,
BFP8=1088, BFP4=576 bytes. BFP shared-exponent overhead and physical width padding
are therefore included. Selected per-layer overrides are applied before sizing.
Indexed experts count8/128 resident expert matrices per row, not all128 and not
all retained unused EP-prefill weight copies. The LM head uses selected BFP4.

Native paged SDPA has FP32 destination and dynamic K chunks capped at128 tokens.
Both read endpoints round to that chunk. Sliding's logical start is
`max(0,position+1-1024)`, full's is0, logical end is`position+1`.
At position4096 this reads1152 sliding or4224 full tokens; at4223 sliding falls
to1024. This follows `tests/summarize_perf.py:native_sdpa_cache_reads` and native
SDPA runtime args. Two KV caches are counted on all ASICs. Updating each cache
row is modeled as one read and one write of the affected32-token tile.

The following explicit approximations account for smaller material traffic and
are exposed as separate terms, so their contribution can be inspected:

- Norm weights: `(8*H+2*D)*4` bytes per ASIC/layer/active row; repeated scalar
  weights and compiler/core reuse are approximated.
- Other layer activations:24 BF16 hidden-vector tile transfers,12 FP32 hidden-vector
  tile transfers, and four FP32 local-QKV tile transfers per ASIC/layer/active row.
  This is a coarse allowance for normalization, casts, residuals, head splitting,
  RoPE, gathers and epilogues. Some operations remain entirely in L1; the
  allowance may overestimate those and underestimate other intermediates. It is
  not an operation-level measured DRAM counter or a rigorous upper/lower bound.
- Metadata: one logical page-table prefix, eight UINT16 indices,128 BF16 route
  weights and64 bytes scalar state per ASIC/layer/row; actual kernel table rereads
  and tile padding of metadata can differ.
- Final norm: one replicated FP32 H-vector plus three FP32 batched activation
  transfers. Embeddings read one BF16 logical H row per model slot.
- Logits: one BF16 vocabulary tensor write, three softcap unary read/write passes,
  and one sampler read (eight transfers total). Physical batch rounds to32.

Persistent collective scratch and most expert intermediates reside in L1; L1 and
NoC/Ethernet bytes are not DRAM bytes. Their elapsed time remains in the denominator.
Additional per-core rereads, allocator effects, host PCIe traffic, trace commands,
and exact sampler/CCL scratch traffic are unmeasured. These limitations are reasons
to call the result an estimate, not reasons to drop their execution time. Estimates
must not be clamped at100%; suspicious ratios require explaining accounting bias.

Validation: `python models/autoports/google_gemma_4_26b_a4b_it/tools/test_benchmark_work.py`.
Tests cover independent prefill reconciliation, BFP exponents, actual per-row versus
batched weight sharing, partial batches, KV window boundaries,127 decode steps,
unsupported configurations and four-ASIC peaks. No hardware or profiler is used.

Source audit during the run: the non-DP runner compacts live request slots and
pads wire positions with trailing -1. On every membership reset, the autoport
trims to the last nonnegative position plus one (generator_vllm.py, decode_forward).
Thus31 active requests have32 wire rows but31 logical model slots. The observer's
len(request_ids) is the correct logical batch; physical head tiles still round
to32 in WorkAccounting.payload. Substituting wire_rows was investigated and
refuted; no runtime or accountant change was made.

## Trace rebinding work correction

Full-phase decode work includes the extra warm model execution when a trace is
rebound. `generator_vllm.prefill_forward` calls `configure_sampling`, which releases
the existing trace (`generator.py:142–152`). The next decode therefore binds and
calls `_capture`. `_capture` executes one warm `_forward`, records the graph, and
`decode_forward` subsequently calls `_replay` (`generator.py:377–405,423–440,462`).
Recording is **not** another device model execution: `FDMeshCommandQueue::record_begin`
enables bypass (`tt_metal/distributed/fd_mesh_command_queue.cpp:1315`) and
`SystemMemoryManager::issue_queue_reserve` redirects commands to host storage
(`tt_metal/impl/dispatch/system_memory_manager.cpp:520`). `record_end` restores
normal submission without replay (`fd_mesh_command_queue.cpp:1640–1666`).

For this compact DP1 greedy workload the collector reconstructs binding state from
the entire recorded dispatch sequence, anchored by an observed initial prefill.
The first following decode and each changed logical-batch size have two executed
model passes; subsequent unchanged decode batches have one. A later prefill resets
the anchor even at the same batch size. Both passes use the same positions because
capture setup restores token and position buffers. Plugin compaction marks layout
changes (`vllm_tt_plugin/model_runner.py:718–723`); trailing wire padding is trimmed
by `generator_vllm._decode_inputs:158–168`. No arbitrary holes, non-DP assumptions,
or unanchored streams are silently accepted. Runtime measurement time is unchanged.

`model_executions`, `warm_model_executions`, and `dram_bytes_per_execution` are
retained for every decode submission. Actual executed work rather than merely
emitted tokens determines DRAM totals. Sampler precompile and trace upload bytes
remain approximate/excluded as described above; warm execution duplicates the
model estimate, not a measured memory-controller count.
