# AutoDebug: persistent CCL pool for serial decoder layers

## Verdict and scope

The maximum-context failure is an L1 lifetime/capacity problem. Each layer
currently owns an independent set of persistent decode collective buffers, and
the capacity test reserves all 30 sets before prefill. Sharing these tensors
between serial layers is supported by the selected decoder's source-level
lifetime contract. Keep attention and grouped-MoE roles separate and retain
each layer's own CCL semaphores. Hardware validation is still required.

This is a delegated fresh-context, source-only AutoFix/AutoDebug investigation.
No device execution, TTNN imports, implementation changes, build, or additional
agents were used. The parent owns integration and all hardware experiments.
The auxiliary AutoDebug CLI was not launched because this task explicitly
prohibited further agents. Only this report and a stage work-log entry were
written.

## Starting evidence

- `final_max_262143_sliding.command.json` records layer 0, length 262143,
  one traced decode step, cache checks, repeated input, full-stack capacity
  reservation, and one prefill timing sample.
- `final_max_262143_sliding.log` reserves 37,614,720 bytes/device of persistent
  CCL L1 payload and 19,864,223,744 bytes/device of other resident DRAM. SDPA
  then fails during prefill: static CB end 1,315,840 overlaps an L1 buffer
  starting at 1,074,304 on cores `[0,0]-[7,7]` (241,536 bytes of overlap).
- `final_memory_plan.json` attributes 1,208,064 bytes to each sliding layer,
  1,482,624 to each full layer, and counts 25 sliding plus 5 full layers.
- This failure precedes the decode warmup and trace. It is not evidence of
  corrupt collective data, a semaphore hang, or insufficient maximum-context
  KV-cache allocation.
- Parent follow-up: sliding BFP8 attention CCL passed headline 4096/128 and
  real batch 32, so both attention kinds will use BFP8. Full grouped MoE stays
  BF16 and sliding grouped MoE stays BFP8. The payload calculations below
  distinguish this accepted precision change from the original failure.

Inspected source SHA256:

```text
tt/multichip_decoder.py     1fc2947d295412c54e9cc4809a7c3a9e5d0dc2ed85744d7abc3d62753158b99e
tests/run_multichip_decoder.py e1ec90a4803151cae2f3d944a08e743909c3a4811a871e730ce9a897cc75df57
tests/test_multichip_stack.py 2c29bc5eaf19d28c067e6e16ff51732294f612627437d616dd14a1285ea6f453
```

Line references below refer to this inspected state and may move during parent
integration.

## Capacity reconciliation

`MultichipDecoder.from_state_dict` creates a new `_collective_buffers` dict
for every instance (`tt/multichip_decoder.py:699-703`). Its decode-only
`allreduce` caches a three-tensor bundle by `(role, logical_shape, dtype)`
at lines 1095-1163. Prefill collectives use DRAM and do not need these L1
bundles. However, warmed bundles remain resident during subsequent prefill.

On the selected Linear topology and TP4, each plane of `[1,P,1,2816]`
requires 176 intermediate tiles (`[2,P,32,2816]`), 22 reduce-scatter output
tiles (`[1,P,1,704]`), and 88 gather-output tiles (`[1,P,1,2816]`): 286 tiles
per plane. BF16 tiles are 2048 bytes; BFP8 tiles are 1088 bytes.

| Pool key | RS intermediate | RS output | AG output | Total bytes/device |
| --- | ---: | ---: | ---: | ---: |
| attention, P1, BFP8 | 191,488 | 23,936 | 95,744 | 311,168 |
| moe_pair, P2, BFP8 | 382,976 | 47,872 | 191,488 | 622,336 |
| moe_pair, P2, BF16 | 720,896 | 90,112 | 360,448 | 1,171,456 |
| **All selected keys** | **1,295,360** | **161,920** | **647,680** | **2,104,960** |

The original precision policy's shared union is 2,690,688 bytes/device because
it additionally needs the BF16 attention bundle (585,728 bytes). The accepted
BFP8-attention policy with private per-layer bundles would still need
30,750,720 bytes/device. Thus the proposed pool changes tensor lifetime and
ownership; the precision change alone does not establish capacity.

A cheap local `python3` arithmetic check verified the original per-layer and
30-layer totals against `final_memory_plan.json`, and verified the selected
three-key union above. These are payload bytes, not allocator high-water
marks: bank rounding, semaphores, and static CBs still need the actual capacity
run. Do not infer a passing maximum context from the arithmetic alone.

The runner's current reservation at `tests/run_multichip_decoder.py:838-858`
also intentionally adds an extra copy for the current layer. Replace that
model with the actual pool; merely reducing a documented byte count while
leaving per-layer allocation unchanged would not repair the ownership issue.

## Lifetime and asynchronous completion audit

### Selected device sequence

For each logical decode row, the selected replicated path is:

```text
attention RS -> attention AG -> attention norm/residual -> router/experts/shared
             -> moe_pair RS -> moe_pair AG -> tail norms -> fresh BF16 output
```

`_reduce_attention` chooses the attention role (line 1093).
`_reduce_moe_pair` concatenates shared and routed results and chooses the
distinct `moe_pair` role (lines 1074-1090). These roles must not be collapsed
even when shapes or dtypes become compatible.

The attention gather buffer is consumed by post-attention normalization before
MoE inputs are constructed (`_forward`, lines 1283-1300). The MoE gather buffer
is consumed by both tail norms before the layer's final add. Slices of the
MoE result may alias their source; the proof does not require slices to copy.
For sliding MoE, the BFP8-to-BF16 typecast provides another fresh intermediate,
but the shared pool must also be safe for the full BF16 case without it.

`_fused_tail` ends with an ordinary `ttnn.add`, BF16 dtype and DRAM output
memory in the selected default (lines 1347-1354). It passes no output tensor.
`BinaryNgDeviceOperation::create_output_tensors` in
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:556-564`
therefore allocates a new device tensor. The returned layer boundary does not
alias any pool entry. The caller may retain earlier layer outputs for checks.

Batch 32 uses serial single-row forwards and concatenates their fresh outputs
(`tt/optimized_decoder.py:737-765`), so it uses the same role-lifetime sequence
repeated 32 times. Prefill chunks likewise return fresh outputs; persistent
decode pool allocation is not used for ordinary multi-row prefill collectives.

### Host order alone is insufficient

TTNN enqueues mesh workloads nonblocking
(`ttnn/api/ttnn/device_operation.hpp:178-213`). A Python function returning
does not prove a collective or its consumers have completed on every rank.

Both selected persistent kernels deliberately disable their startup barrier:

- Linear RS program factory:
  `reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:1575`.
- AG default program factory:
  `all_gather_async/device/all_gather_async_default_program_factory.cpp:752`.

Both use `barrier_semaphore.has_value() && !using_persistent_buffers`.
Passing a barrier semaphore does **not** restore this barrier when persistent
output tensors are supplied. Per-layer semaphores isolate completion counters;
they do not by themselves make arbitrary shared output storage safe.

### Why this two-role sequence is safe at the source level

Within one mesh command queue, local consumers complete before later local
programs. Completion of an allgather requires contribution data from every
rank, whose local reduce-scatter precedes its allgather. The next opposite-role
allreduce therefore gives the required cross-rank progress condition:

1. Before any rank starts the next **attention** RS, it has completed the
   previous MoE allreduce. Every other rank must already have produced its
   MoE RS output, after consuming its previous attention gather buffer.
   The previous attention RS scratch, RS output, and AG output are all dead.
2. Before any rank starts the next **MoE** RS, it has completed the next
   attention allreduce. Every other rank must already have produced its
   next attention RS output, after the previous tail consumed its MoE gather
   buffer and returned a fresh output. The previous MoE bundle is dead.

This also covers two same-kind layers, mixed layer kinds, the next decode row,
and the next serial trace replay. It does not require equal rank execution
speed. Different worker counts between sliding and full attention do not
change the tensor lifetime argument.

The relevant completion behavior is explicit in the selected kernels:

- RS reader waits for remote ready semaphores before reading intermediate
  tiles (`line_reduce_scatter_minimal_async_reader.cpp:261-283,396-418`).
  The writer flushes writes before signaling, drains local final writes,
  and terminates its fabric mux before exiting
  (`line_reduce_scatter_minimal_async_writer.cpp:356-378,389-447`).
- AG reader waits for every expected received slice, including the
  non-forwarding terminal case, and resets its own ready semaphore only at
  the end (`minimal_default_reader.cpp:146-165,295-333,356-407`). The writer
  flushes payload before ready signaling and drains its connection
  (`minimal_default_writer.cpp:614-624,678-701`).
- These file names are under their respective
  `ttnn/cpp/ttnn/operations/experimental/ccl/*/device/kernels` directories.
  The selected tiled, interleaved, width-divisible inputs use Linear RS and
  the ordinary minimal AG implementation, not the sharded or fused variants.

An arbitrary caller doing consecutive same-role allreduces and retaining the
previous result would violate this proof: a faster rank could overwrite a
slower rank's still-live AG output. The pool is decoder scratch for this
serial phase schedule, not a general-purpose allreduce result cache.

## Minimal proposed API and safeguards

Add caller-owned `CollectiveBufferPool(mesh_device)` with a buffer dictionary,
and an optional `collective_buffer_pool=` argument to `from_state_dict`.
Without an external pool, preserve private instance ownership. Bind the
instance's `_collective_buffers` to the supplied pool only after validation.
The caller creates one pool for one serial stack on one live mesh.

- Require the same mesh object, not only equal mesh shape. Never reuse a
  pool after closing/reopening a mesh.
- Initially restrict external pooling to the audited Linear, replicated,
  grouped-MoE selected path. Keep the fresh fused-tail output boundary. Ring,
  sharded residual, fused AGMM/MMRS, arbitrary concurrency, and custom direct
  calls to `allreduce` need separate lifetime reasoning.
- Retain the role in each key. Include logical/padded shape, dtype, and memory
  config, or assert a fixed tile/page/padding contract sufficient to derive
  them. A memory-config change must not return an existing tensor in a
  different memory tier. Topology is fixed by the pool's supported contract.
- Pool only tensors. Create and retain each layer's own `CCLManager`, its
  RS/AG ping-pong indices, and its global semaphores exactly as today.
- Keep strong references to the pool and all entries through warmup, capture,
  every replay, and pending queued work. Do not clear, deallocate, replace, or
  resize entries while a trace can reference them. Populate the union of
  exact shapes/dtypes before capture.
- Execute one serial phase schedule on the same command queue. Separate
  concurrent requests/queues require separate pools or an explicit device
  synchronization protocol. A host lock around enqueue calls is insufficient.
- Public decoder outputs remain newly allocated BF16 tensors. No return of
  a persistent AG buffer as the layer boundary is permitted.

No device-side barrier or collective-kernel change is required by this
design. Hardware evidence must still verify the source-level prediction.

## Focused verify/refute experiments for the parent

1. **Pool ownership and exact output A/B.** Extend the existing direct-stack
   runner with an explicit pooled/private option and selectable layer pair.
   Run both `(0,5)` and `(0,1)` with identical checkpoint data, input, cache,
   precision and trace settings. `(0,5)` shares BFP8 attention but has
   different MoE dtypes; `(0,1)` is essential to exercise shared MoE entries.
   Check actual buffer addresses/identities and unique bytes, not just equal
   dictionaries. Different layer semaphore addresses must remain distinct.
   Compare all ranks and every layer boundary exactly against the private
   TP4 run, plus the existing TP1 PCC >= 0.995 gate.
2. **Retained output and async stress.** Queue both layers without host
   conversion between them. Retain the first layer's output, read it after
   the next layer finishes, and compare it with the private control. Warm
   exact signatures, capture both layers in one trace, advance positions,
   and replay repeatedly (128 iterations is a useful short stress). Check
   finite output, exact replicas, duplicate replay equality, independent
   caches, and unchanged earlier outputs. A watcher run is appropriate for
   this async/CCL change, separate from profiler evidence.
3. **Real reservation before maximum prefill.** Prime the actual caller pool
   with zero `[1,1,1,2816]` BFP8 attention and `[1,2,1,2816]` BFP8/BF16
   MoE inputs, invoking the same decoder's `allreduce` in supported role
   order and discarding these dummy results. Keep that pool alive through
   prefill and traced decode. Replace the anonymous 30-layer L1 reservation
   with this actual union; preserve the existing DRAM full-stack reserve.
   Report payload plus observed allocation/CB outcome. Ensure warmup does
   not create a second private pool after reservation.
4. **Original failing workload.** Re-run the command below after the runner
   uses the actual pool. Passing prefill with the complete selected union
   proves the specific static-CB overlap is resolved for this case; the
   one final decode step verifies coexistence with live cache and scratch.
   Keep broader context and final-default validation under the parent stage.

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --layer 0 --length 262143 --steps 1 --trace --check-cache --repeat-input --reserve-full-stack --prefill-timing-samples 1 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/pooled_max_262143_sliding.json
```

These hardware experiments were **not run by this diagnosis agent**. The
parent subsequently integrated `--pool-ccl`/`--no-pool-ccl`, `--layers` and
`--steps` in the stack runner; the follow-up below records evidence received.
Do not call the pool proven solely because a mixed-layer test passes without
verifying that the intended tensors actually share storage.

## Adaptation if pooling fails its hardware control

First distinguish incorrect key/mesh/lifetime integration from a refutation of
the selected schedule. Preserve same precision and compare against private
TP4 buffers. Localize the first differing attention/MoE gather or tail output.

A smaller capacity alternative is to put only the Linear RS intermediate in
DRAM, retaining its RS result and AG output in L1. The public RS API exposes
`intermediate_memory_config`; allocate the persistent intermediate in DRAM
and pass that config explicitly. Validation permits it
(`reduce_scatter_common/reduce_scatter_validate_utils.cpp:110-135`), and the
Linear factory builds tensor accessors from the actual intermediate buffer
(`reduce_scatter_minimal_async_program.cpp:1344-1347,1410-1413`). This is
source-supported, not a guessed memory-config change. Measure accuracy,
capacity and warmed latency before selecting it. It removes 176 of 286
tiles per role-plane from L1; allocator/CB fit is still not guaranteed.

If that still fails capacity or costs too much, persistent DRAM for the
whole private bundle preserves existing ownership while removing these
payloads from L1. Set the collective memory policy consistently for inputs,
intermediate, RS output and AG output; remeasure the final decoder. This
does not repair arbitrary shared-buffer lifetime races, so use private
bundles if such a race actually refutes pooling. Do not reject the entire
persistent-CCL family on the first failed pooled experiment.

## Final status

Source diagnosis and minimal design complete. The original L1 capacity cause
is established by the failing allocation and reproduced arithmetic. Pooling
is supported under the stated two-role, serial, same-mesh contract, with
explicit async evidence and preserved output ownership. Implementation,
same-kind/private A/B, watcher, maximum-context and final latency validation
remain with the parent. No performance improvement is claimed by this report.

## Parent experiment received during report review

`pool_shared_kinds.failure.json` / `.log` record the proposed `(0,1)` stack
with 128 steps failing **before decode**, at `prefill/layer1`. All ranks are
finite; ranks 2 and 3 each differ from rank 0 in 2802 elements, while ranks 0
and 1 match. Runtime SHA256 is
`b7d00ca94ef5f73f5febed4b767c8f52dd47a9d89868863d502123a612077e8f`;
stack runner SHA256 is
`88def04a74dbfe6481b428ced76478dac06d1543e70f28f0e6da18dfe6f85d08`.
This uses the parent's newly integrated pool, not the earlier inspected hash.

The prompt length 33 does **not** become a 32-row chunk and a 1-row tail.
`MultichipDecoder.prefill_forward` delegates lengths <=1024 to
`OptimizedDecoder.prefill_forward`; the latter makes one chunk with valid
length 33, pads it to physical length 64, and calls `_forward` once with
`is_decode=False` (`tt/optimized_decoder.py:701-733`). Attention and grouped
MoE collectives should therefore both have 64 logical rows, keeping the
`value.shape[-2] == 1` persistence guard false. No decode warmup has happened.

The parent is running the private-buffer control. Before attributing this
failure to pooling, record the pool's entry count immediately before the
failing read (predicted zero) and mismatch counts by token row. The element
count alone does not prove the mismatch is confined to the last row. If the
private control reproduces the same prefill failure with an empty shared pool,
that refutes pooling as the cause of this symptom and identifies a separate
same-kind prefill regression to localize. If the pool is unexpectedly populated,
capture actual collective shapes/roles and investigate the violated phase
contract. Pooling hardware validation remains incomplete; this failure is
not silently counted as a passing test.

## Resolved follow-up: independent semaphore-grid defect

The private control reproduced the same prefill failure, and BF16 attention also failed. Boundary instrumentation found correct padding and an empty pool, then the first divergence at layer1 attention AG output. `AUTODEBUG_ccl_semaphore_grid.md` records the concrete 8x8 semaphore-versus-11x10 worker mismatch and parent whole-stack full-grid controls that repair replica/replay equality and the long-prefill loss. The remaining cumulative decode PCC failure is separate and still under parent investigation. This follow-up refutes pooling as the cause of that prefill symptom; it does not by itself complete the pooled maximum-context gate.
