# AutoDebug: shared-batch decode and persistent CCL buffer reuse

## Verdict

**No persistent-buffer ordering race was found in the new `_decode_shared_batch` schedule for its actual supported path: TP4, a 1x4 mesh, Linear collectives, one mesh command queue/sub-device, and the current non-fused `reduce_scatter_minimal_async` followed by `all_gather_async`.**

The next attention allreduce's leading reduce-scatter is itself the required cross-rank progress point. It prevents the following all-gather from overwriting rank `r`'s persistent gathered attention buffer until rank `r` has completed the consumer of the previous attention result. An intervening paired-MoE allreduce is not required for that ordering.

The obvious discrepancy is the `CollectiveBufferPool` comment at `models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py:49-55`. It says the *intervening* opposite-role allreduce is what makes same-role reuse safe. That is overly restrictive for the implementation inspected here. Keeping attention and MoE bundles distinct is a valid conservative ownership policy, but it is not the reason repeated attention reuse is safe.

No captured-temporary ownership defect was found in `_decode_shared_batch`. The per-slot tensors that must survive the loop are retained until their concatenations, persistent tensors are owned by the model's `CollectiveBufferPool`, and the returned output owns fresh storage. Trace scratch lifetime remains subject to the normal TT-Metal trace-allocation contract; this path does not introduce a unique lost-owner case.

Confidence is high for this exact source path. This was an inspection-only investigation: no hardware or tests were run. Ring topology, multiple command queues/sub-devices, concurrent callers, mismatched collective order, and a future fused/streamed RS-to-AG implementation are outside the proof.

## Direct observations versus interpretation

Direct code observations:

- The opt-in path only accepts batches 8 through 32, replicated residuals, grouped MoE reduction, the fused tail, and the exact `_SharedMLP` decode implementation (`multichip_decoder.py:1497-1515`).
- Each per-slot attention reduction has logical sequence extent one and therefore selects persistent CCL (`multichip_decoder.py:1317-1384, 1530-1544`).
- The delayed paired-MoE reduction has logical sequence extent `B`, where `8 <= B <= 32`. It fails the `value.shape[-2] == 1` persistence guard and uses `MeshConfig.allreduce`, which allocates ordinary RS/AG outputs (`multichip_decoder.py:1296-1312, 1324, 1385`; `models/demos/gemma4/config.py:96-127`). It does not touch a persistent `moe_pair` bundle.
- A persistent Linear allreduce is two separate mesh workloads: persistent Linear reduce-scatter followed by persistent all-gather (`multichip_decoder.py:1339-1384`).
- Mesh dispatch patches every workload's GO signal with the prior workload's completion count, and the dispatcher waits for that count before issuing the next GO (`tt_metal/distributed/fd_mesh_command_queue.cpp:443-465, 516-543`; `tt_metal/impl/program/dispatch.cpp:2975-2982, 3197-3204`; `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1181-1211`). Trace assembly applies the same count progression to every recorded node (`fd_mesh_command_queue.cpp:1575-1605`). This gives per-rank program completion order, not merely Python enqueue order.

Interpretation proved below:

- The program sequence on each rank is `AG_k -> C_k -> RS_(k+1) -> AG_(k+1)`, where `C_k` includes the operation that consumes the gathered attention output.
- Any rank beginning the write phase of `AG_(k+1)` has first completed an `RS_(k+1)` output that depends on every rank's contribution.
- Every such contribution can be sent only after that contributing rank completed `C_k`.
- Therefore `AG_(k+1)` cannot overwrite any rank's `G_k` before that rank is finished consuming `G_k`.

The requester-provided exact KV/logit and latency results are consistent with this conclusion, but they are supporting finite observations rather than the basis of the ordering proof.

## Exact lowered path

For an attention allreduce, `MultichipDecoder.allreduce` allocates and then reuses three logical buffers under a key containing role, logical and padded shapes, dtype, and memory configuration (`multichip_decoder.py:1324-1354`):

- `I`: the Linear reduce-scatter intermediate. Its first dimension is doubled to provide independent forward/backward regions (`multichip_decoder.py:1350-1352`; `reduce_scatter_minimal_async_op_device_operation.cpp:213-244`).
- `S`: the reduce-scatter result (`multichip_decoder.py:1339, 1355-1366`).
- `G`: the all-gather result returned to attention (`multichip_decoder.py:1340, 1368-1384`).

The native wrapper maps persistent buffer indices 0 and 1 to the optional intermediate and optional output tensors (`reduce_scatter_minimal_async.cpp:73-119`), and the device operation returns those exact caller buffers rather than allocating replacements (`reduce_scatter_minimal_async_op_device_operation.cpp:247-270`). The all-gather device operation likewise returns its caller-provided persistent output (`all_gather_async_device_operation.cpp:218-224, 251-328`). The default minimal all-gather factory is selected for this call (`all_gather_async_device_operation.cpp:44-70`).

Both persistent factories disable only their explicit startup barrier: RS passes `use_barrier_sem = false` at `reduce_scatter_minimal_async_program.cpp:1566-1577`, and AG does so at `all_gather_async_default_program_factory.cpp:747-763`. The corresponding barrier bodies are startup rendezvous code (`line_reduce_scatter_minimal_async_writer.cpp:223-248`; `minimal_default_writer.cpp:237-287`). Their omission does not remove the payload-ready handshakes inside either collective.

## Buffer and semaphore ledger

| State | Producer and consumer | End-of-operation state | Why the next attention call is safe |
| --- | --- | --- | --- |
| RS intermediate `I_k` | Remote RS writers write it, flush payload writes, then increment the receiver's ready semaphore (`line_reduce_scatter_minimal_async_writer.cpp:278-379`). RS readers wait for that count, read `I_k`, and issue a read barrier before exposing data to compute (`line_reduce_scatter_minimal_async_reader.cpp:227-285, 395-420`). | Every expected remote write has been observed and consumed before that rank's RS reader exits. | AG never reads `I_k`. Reuse by the next RS therefore cannot destroy an AG input or a model consumer input. |
| RS result `S_k` | The RS writer writes the final local reduction only after its reduction CB becomes ready (`line_reduce_scatter_minimal_async_writer.cpp:389-425`). AG's reader then reads the local slice into its output CB (`minimal_default_reader.cpp:118-144`). | The local queue cannot launch the next RS until the local AG and its later model consumers complete. | Only the local next RS overwrites `S`; same-rank queue ordering puts that overwrite after AG consumed `S_k`. |
| AG result `G_k` | AG writers copy slices into the local/remote gathered tensor, flushing packet payloads before their ready increments and draining writes/atomics before exit (`minimal_default_writer.cpp:299-455, 681-703`). AG readers wait on every expected ready count and reset the ready semaphore (`minimal_default_reader.cpp:295-407`). | `G_k` is complete when AG finishes. It remains live through the attention normalization/add consumer. | The next RS does not touch `G`. The following AG is the first operation that can overwrite it, and the cross-rank proof below orders that write after every rank's consumer. |
| External RS/AG ready semaphores | The CCL manager supplies two ping-pong sets for each collective (`models/demos/gpt_oss/tt/ccl.py:39-80`). RS and AG readers reset their local ready semaphore at exit; writers drain outstanding writes and atomics (`line_reduce_scatter_minimal_async_reader.cpp:427-428`; `line_reduce_scatter_minimal_async_writer.cpp:428-447`; `minimal_default_reader.cpp:400-407`; `minimal_default_writer.cpp:681-703`). | A set is locally zeroed after all expected arrivals, and all sender-side transactions are drained. | The other ping-pong set is used by the intervening call. Reuse two calls later occurs only after a complete intervening RS/AG progress chain. |
| RS local forward/backward semaphore | Interior Linear ranks use a program-owned semaphore created at zero (`reduce_scatter_minimal_async_program.cpp:1282-1285`). FWD signals only after its final output write barrier; BWD waits before reading that output (`line_reduce_scatter_minimal_async_writer.cpp:411-420`; `line_reduce_scatter_minimal_async_reader.cpp:365-384`). | Dispatch initializes program semaphores for every launch, including recorded mesh-trace launches; details are in the trace section below. | It is not a persistent global semaphore and does not accumulate across trace replays on this mesh-dispatch path. |

The RS compute kernel performs the actual tile additions for each configured reduction step (`device/kernels/line_reduction.cpp:30-64`). The host setup documents that endpoint and interior ranks perform the final reduction, with interior forward/backward readers synchronized (`reduce_scatter_minimal_async_program.cpp:1494-1508`). Thus each rank's final `S` slice contains all TP4 contributions; it is not a partial result that can become visible before a lagging rank enters the operation.

## Cross-rank happens-before proof

Let `G_k(r)` be the persistent attention gather buffer on rank `r`, `C_k(r)` its model consumer, and `RS_(k+1)(q)` / `AG_(k+1)(q)` the next attention collective phases on rank `q`.

For every destination rank `r` and every source-slice rank `q`, the earliest overwrite of q's slice in `G_k(r)` by the next call has this chain:

```text
C_k(r) completes
  -> RS_(k+1) launches on r
  -> r's contribution reaches the Linear RS
  -> S_(k+1)(q), which includes r's contribution, becomes complete
  -> AG_(k+1) launches on q
  -> q writes its slice into G_(k+1)(r)
```

The first edge is enforced by the command queue's prior-worker completion count. The middle two edges are the Linear reduce-scatter data dependency: writers send partials through the line (`reduce_scatter_minimal_async_program.cpp:1047-1078`; `line_reduce_scatter_minimal_async_writer.cpp:278-387`), readers wait for the associated ready counts before reducing (`line_reduce_scatter_minimal_async_reader.cpp:227-298, 395-420`), and every rank produces a final reduced slice (`reduce_scatter_minimal_async_program.cpp:1494-1508`). The last two edges follow from RS and AG being distinct, ordered mesh workloads and from AG reading the completed local `S` before sending it.

Because `r` and `q` are arbitrary, no part of the next gathered output can overwrite `G_k(r)` until `C_k(r)` has completed. This remains true if ranks progress at very different speeds and is the missing ordering argument that finite exact-output runs cannot provide.

The same reasoning also protects the two-call reuse interval of the ping-pong global semaphore sets. What matters is a complete intervening allreduce, not that it has the opposite semantic role.

## `_decode_shared_batch` ownership audit

The persistent attention result is consumed inside the loop by normalization and a fresh `ttnn.add` result before the next iteration (`multichip_decoder.py:1545-1550`). Dispatch completion ordering ensures that this is an actual device-completion boundary before the next RS launch, not just a host-language source-order assertion.

The tensors needed after the loop have explicit Python owners:

- `residuals` retains every fresh residual result.
- `routed_rows` retains every expert output.
- `shared_inputs` retains every normalized shared-MLP input.
- Each list remains live through its `ttnn.concat`, and the concat results feed the batched shared MLP, paired-MoE reduction, and fused tail (`multichip_decoder.py:1529-1565`).
- `_fused_tail` returns a new `ttnn.add` output; it does not return a view of `G`, `S`, or `I` (`multichip_decoder.py:1639-1652`).

Persistent collective buffers have a longer-lived owner: the model owns one `CollectiveBufferPool` and passes it to all serial decoder layers (`models/autoports/google_gemma_4_26b_a4b_it/tt/model.py:66-79`), while each decoder retains the shared dictionary (`multichip_decoder.py:807-814`). The CCL manager similarly owns the global semaphores.

During decode trace capture, the generator warms the path, restores its mutable inputs, synchronizes, captures `_forward`, and retains the final `trace_logits` (`generator.py:564-594`). TT-Metal records allocation/deallocation high-water marks while capture is active and registers the trace after capture (`tt_metal/distributed/mesh_device.cpp:1387-1457`); the allocator explicitly treats new allocations made while a trace is live as potentially unsafe (`tt_metal/impl/allocator/allocator.cpp:118-143`). The generator does not retain every ordinary intermediate for the trace's lifetime, so captured-address safety necessarily comes from this framework-level allocation discipline rather than from long-lived Python references. Callers must still avoid placing conflicting allocations over captured addresses. Nothing in `_decode_shared_batch` weakens or bypasses that framework contract.

## Trace replay semaphore adjudication

A plausible-looking alternative diagnosis was that Linear RS's local `fwd_bwd` semaphore is incremented and waited with monotonic `wait_min` targets but not reset in the kernel. That would be a replay race if trace replay skipped program semaphore initialization. It does not on the inspected mesh trace path:

1. `CreateSemaphore(..., 0)` makes `fwd_bwd` a program semaphore (`reduce_scatter_minimal_async_program.cpp:1282-1285`).
2. Dispatch gathers every program semaphore's initial value into the program-config transfers (`tt_metal/impl/program/dispatch.cpp:1259-1321`) and assembles those transfers into `program_config_buffer_command_sequence` (`dispatch.cpp:2507-2545`).
3. Mesh trace assembly visits every recorded program node, emits its program command sequence, and stores the resulting bypass command bytes in the trace descriptor (`tt_metal/distributed/fd_mesh_command_queue.cpp:1527-1605, 1619-1624`). Replay dispatches the trace buffer containing those bytes (`fd_mesh_command_queue.cpp:1258-1284`).
4. `write_program_command_sequence` unconditionally writes the program-config sequence before launch, regardless of whether the binary itself must be resent (`dispatch.cpp:3279-3287`).

Therefore `fwd_bwd` starts at zero on each captured invocation and each replay. Adding a kernel-end reset is not justified by this report and could obscure the actual distinction between program-owned and external persistent semaphores.

## Discrepancy and bounded recommendations

1. **Correct the `CollectiveBufferPool` explanation.** State that role separation prevents accidental cross-role aliasing and constrains ownership, but that serial reuse of one role is ordered by the next allreduce's leading Linear RS plus one-CQ dispatch ordering. The current wording incorrectly makes an opposite-role collective sound necessary.
2. **Keep the path restrictions explicit.** The proof relies on Linear TP4, equal collective order/count on every rank, the same mesh CQ/sub-device, and the current separate RS then AG workloads. Do not generalize it to direct repeated all-gathers, Ring, concurrent callers, different CQs/sub-devices, or a fused/streaming collective without a new audit.
3. **Retain a timing-skew regression on hardware.** The most discriminating follow-up is repeated traced B8/B32 execution while deliberately delaying one rank between `C_k` and `RS_(k+1)`, checking exact per-rank outputs/KV state. A delay should reduce performance but cannot change results under the proved protocol. This is a guard against future dispatch or collective changes, not evidence needed to rescue the current schedule.
4. **Treat the reported exact tests as corroboration.** The reported real-activation layer 0/5 KV hashes, full-30-layer B32 exact logits, and 677.44 to 641.01 ms timing are consistent with the source proof. No performance claim was independently measured here.

## Proof boundaries / other potential issues

- A direct `AG_k -> consumer -> AG_(k+1)` reuse has no leading all-rank reduction and is not covered.
- A future RS implementation that exposes a rank's final slice before incorporating every rank, or a fused RS/AG implementation that starts remote `G` writes before full RS completion, breaks the central implication and must add an explicit consumption/rendezvous mechanism.
- Different per-rank control flow or collective counts can deadlock or corrupt independently of buffer ownership. `_decode_shared_batch` currently uses the same fixed slot loop on every rank.
- New allocations after capture can conflict with any trace's captured scratch addresses under the general trace contract. This is not specific to `_decode_shared_batch`; allocator diagnostics should remain enabled when changing post-capture allocation behavior.

## Verification performed

Source inspection only. No implementation files were modified, no hardware-dependent command was attempted, and no tests were run. The only created file is this report.
