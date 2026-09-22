# AutoFix: shared direct all-reduce workspace

Source audit dated 2026-09-12. This investigator read native/model source and saved JSON/command artifacts; no hardware was opened and no implementation was edited. Regression and stress runs are owned by the parent.

## Finding and scope

One model-scoped `TT_CCL` can own the direct all-reduce workspace without aliasing returned outputs. This removes the full-stack cost of independently allocating identical persistent L1 scratch for every decoder. The inspected usage contract is **one ordered CQ0 layer stack, with no concurrently executing models or overlapping independent traces sharing this context**. Source alone does not prove safe scratch reuse under arbitrary cross-rank scheduling skew; the dependent decoder schedule requires its own replay validation.

`tt/multichip_decoder.py:23-29` accepts optional `ccl` and rejects a context whose `mesh_device` is not the identical mesh object. Lines 211-224 lazily allocate `_qwen_tp4_allreduce_buffer` on that context and retain a layer reference. The stack fixture passes `decoder.ccl` to its second decoder (`tests/run_multichip_decoder.py:102`); the module documentation states the ordered-CQ0/lifetime/concurrency contract at lines 8-9.

## Native contracts and size

Native paths below are relative to `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/`.

| Property | Source evidence and implication |
| --- | --- |
| Input/output layout | `device/all_reduce_async_device_operation.cpp:24-72` requires width-sharded input, workspace and output, rejects Blackhole DRAM input, requires the output grid to be contained in the workspace grid, and checks `output_shard_volume * ring_size <= workspace_shard_volume`. |
| Workspace is mutable scratch | `device/all_reduce_async_program_factory.cpp:358-368` binds reduction CB `c_1` directly to `buffer_tensor.buffer()`, using the input data format. `device/kernels/dataflow/worker_writer.cpp:123-152` writes each rank's contribution into a rank-specific workspace slot. Keep workspace/input dtypes consistent; current input and workspace are BF16. |
| Output does not alias workspace | `device/all_reduce_async_device_operation.cpp:85-88` allocates a new output. Factory lines 370-381 bind a separate output CB to that allocation. Returning or retaining the result therefore does not retain a view into scratch subsequently overwritten by another reduction. Input and scratch must remain distinct allocations. |
| Cache hits reuse addresses correctly | Factory lines 618-651 update the input pointer, output and workspace dynamic CB addresses, and writer/receiver semaphore addresses. Program-cache reuse does not fix the first layer's original pointers permanently. |
| No host initialization is required | Writers replace every participating rank's active shard contribution before the receiver exposes the reduction CB. Old scratch contents are not persistent model state. The model can allocate `ttnn.empty` once without clearing it between forwards. |

The model direct path (`tt/multichip_decoder.py:332-346`) packs B1-B32 into `[1,1,B,5120]`, tiled to 32 rows, with 80 width shards `[32,64]`. The BF16 workspace `[1,1,32,20480]` has 80 shards `[32,256]`, exactly four output shards per core:

- Output/input shard: `32 * 64 * 2 = 4,096` bytes/core.
- Workspace shard: `32 * 256 * 2 = 16,384` bytes/core.
- One workspace: **1,310,720 bytes/device = 1.25 MiB**.
- 64 independent workspaces: **83,886,080 bytes/device = 80 MiB**, or **1 MiB on each of the 80 participating cores**, before other persistent tensors and native circular buffers.

This fixed workspace covers the inspected B1-B32 contract. It is not a general allocation for larger batches, different hidden widths, other mesh sizes, or changed input dtype/grid. Such extensions need matching scratch geometry or a different path.

## Ordering, trace lifetime and remaining risk

`models/common/modules/tt_ccl.py:91-111` creates two zero-initialized barrier semaphore handles per axis and cycles them in Python. Shared ownership makes layers use the same sequence. Trace capture records the selected addresses; replay executes that captured sequence without calling the Python counter again. Preserve the context, workspace and semaphore allocations through capture, every replay, and completion of all queued users. Do not reallocate, deallocate, replace, or externally write the workspace while a captured trace can still use it.

The native receiver waits for all rank contributions, pushes the workspace into the compute CB, and resets the ready semaphore (`device/kernels/dataflow/reduction_receiver.cpp:29-34`). Compute consumes those tiles afterward (`device/kernels/compute/reduction.cpp:29-67`). The writer's flush/close/barrier sequence (`device/kernels/dataflow/worker_writer.cpp:131-177`) drains its writes; it does not acknowledge that every remote peer has finished computing from the workspace. Thus a sufficiently faster rank could write a subsequent reduction into a slower peer's scratch while that peer still consumes the prior reduction. Alternating semaphore addresses alone does not prevent this data overwrite. The inspected kernel has no explicit cross-rank consumption barrier.

The model's dependent attention-output and MLP-down reductions, with local compute between calls, provide a different schedule from unrelated back-to-back collectives. Existing precedent supports this design: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:875-908` selects one `persistent_buffers[cluster_axis]` for successive decode all-reduces while using cycled semaphore handles. That is a model usage precedent, not an unrestricted concurrency guarantee. Native `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:100,157,192-211` instead rotates eight workspace/semaphore pairs, including the loopback trace path, so those tests do not prove single-workspace reuse.

Use the shared context only in the documented ordered CQ0 stack. Do not submit concurrent models, independent queues, or overlapping independent traces against it. If the dependent model stress exposes corruption, first compare two rotating shared workspace/semaphore pairs; this remains small in L1. Explicit device synchronization is another possible investigation, but no additional synchronization or buffer pool is established as necessary by current results.

## Evidence and pending verification

Exact parent commands are recorded in `commands.log`:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh direct_ar_packed --policy-file models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder/direct_ar_packed_policy.json --prefill-repeats 20 --repeats 30
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh ar_batch32_full --layer 3 --batch 32 --length 257 --repeats 20
```

`direct_ar_packed.json` reports linear-layer B1/S128 decode **0.4729908 ms**, decode PCC **0.99999928**, changed decode PCC **0.99999982**, finite outputs and exact eager/replay state/output checks. This validates the original single-layer direct path, not cross-layer sharing.

`ar_batch32_full.json` reports full-attention B32/S257 with the shared-context API present: decode **1.7694724 ms**, decode PCC **0.99999535**, changed decode PCC **0.99999881**, finite outputs, exact eager/replay state/output checks and valid cache write ownership. Its captured source SHA is `2e4669aa88ec44aa9190f266dbb85fe65a46cd260c8046bf02759b1561296690`; that snapshot includes shared workspace ownership but predates the added mesh-identity guard. A single-layer case still does not exercise cross-layer reuse.

The parent is running the 28-case regression matrix, then B32 linear/full stacked **100 replay** stress. These remain pending at report time. The existing runner uses CQ0 and synchronizes between timed replays (`tests/run_multichip_decoder.py:359-380`); 100 repeats therefore exercise repeated dependent stacks, not concurrent or unsynchronized cross-trace execution. Preserve changed-input checks, raw state equality, cache ownership, and finite/PCC checks when assessing the result.

**Status:** sizing, fresh-output allocation, cache rebinding, ownership, and existing model precedent support the model-local repair. No host transfer or decoder capability reduction is required. Cross-layer reliability is pending the parent regression/stress evidence; arbitrary concurrency is outside the shared-context contract.

## Hardware resolution

`stress_stack_batch32_dram.json` passes with public_dram_batch=2, including
100 evolving eager calls and100 queued trace replays without host barriers
between calls. Shared workspace identity is asserted. Full raw state and
outputs are bitwise equal; final trajectory output PCC0.99998122 and recurrent
PCC0.99982654 versus the frozen optimized single-chip baseline. This supersedes
the pending stress status above. Final default adopts the batched DRAM boundary;
expanded B2/B3/B8/B16 stack regressions and final default stress are rerun.
