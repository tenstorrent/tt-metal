# Retained decode bucket CCL hypothesis audit

Source-only independent AutoFix hypothesis audit, 2026-11-16. No device execution, reset, process termination, or implementation edit was performed. Evidence is `triage/bucket_hang.txt` plus current source. `M` below denotes `models/autoports/qwen_qwen3_6_35b_a3b`.

## Verdict

The observed all-reduce wait is downstream synchronization fanout. The evidence does **not** support a global-semaphore counter wrap, a prematurely freed collective workspace, or two bucket traces concurrently overwriting that workspace as the initiating failure. The earliest visible stopped producer is device 1's deepseek combine reader, waiting for local NoC DRAM-zeroing write completion. This narrows the failure; it does not establish why that write barrier fails to complete or prove the optional bucket change harmless.

The caller reports that the full 40-layer candidate stalls at decode token 114 after an 18-token prompt, whereas a two-layer B2/B32 switching probe completed 76 replays and the fixed-B32 full server completed nine 96-output-token requests. Those observations alone cannot distinguish a shape-sensitive combine issue, allocator/address sensitivity, or a cumulative local NoC accounting issue. The preserved snapshot supplies the stronger ordering evidence below.

## Evidence ledger

| Observation | Exact source or artifact | Implication |
|---|---|---|
| Device 1 is executing operation 320374; devices 0, 2, and 3 are at 320388. | `triage/bucket_hang.txt:255`, `dump_running_operations.py` | The devices are not all blocked in the same collective. Kernel ID 24676 is a reduction kernel ID, not this operation's global operation ID. |
| Device 1 NCRISC on physical `1-2 (0,0)` is in `reader_combine`, through `zero_pages` and `Noc::write_zeros_dram_barrier`, then `noc_async_write_barrier`. | `dump_callstacks.py`; `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp:149` | Its local output-zeroing writes have not completed. It has not reached the output-init-done wait that follows zeroing. |
| The same core's BRISC is in `writer_combine.cpp:158`, waiting for output-init-complete; four untilizer readers wait for their counter-ready semaphore. | `dump_callstacks.py`; combine dataflow `writer_combine.cpp:155–159`, `reader_untilize.cpp:129` | These are downstream dependents of combine initialization, not evidence that direct all-reduce caused the stop. |
| Direct-AR global semaphore `0x179480` equals **3** at all 128 sampled locations across four devices. Devices 0/2/3's reduction receivers wait for the fourth contribution. | `dump_semaphores.py` in the snapshot; reduction receiver stacks | Three devices progressed and contributed; device 1's contribution is missing. The value is neither a wrapped counter nor an overshot expected count. Device 1 can receive contributions even while it has not launched the next collective. |
| AR receiver waits for `ring_size`, then pushes the reduction CB and resets its global semaphore to zero. | `all_reduce_async_program_factory.cpp:612,660–662`; `kernels/dataflow/reduction_receiver.cpp:23–36` | This is a per-invocation 0→4→0 counter, not a monotonically increasing token/replay counter. |
| AR writer sends one remote atomic after each complete destination shard and one local increment per destination core. | `all_reduce_async/.../kernels/dataflow/worker_writer.cpp:132–146,166–170` | A value of three is precisely compatible with one missing chip's sender. |

`check_binary_integrity.py` and `check_core_magic.py` passed. The snapshot also contains NoC status warnings and cores that resumed during triage. Do not reinterpret those warnings as proof of hardware health or a root cause. The stack/operation/semaphore agreement is the relevant evidence for this hypothesis.

## Ownership and replay ledger

1. `M/tt/model.py:90–102` constructs one `CollectiveWorkspace` and passes it to every decoder. `M/tt/multichip_decoder.py:813–827` explicitly gives its tensors and semaphores model lifetime and restricts its use to serial CQ0 forwards. The bucket patch does not replace this object.
2. Direct-AR scratch tensors are allocated once per dtype into `workspace.direct_ar_buffers`; its one direct-AR global semaphore is created only if absent (`multichip_decoder.py:1169–1191`). Layer references retain both. B2 and B32 use the same 32-padded direct-AR geometry (`1427–1455`); no per-replay deletion or semaphore rotation is introduced here.
3. Generic gather/reduce-scatter use `CCLManager`'s fixed ping-pong semaphore arrays (`models/demos/gpt_oss/tt/ccl.py:40–96`). Those Python indices select addresses at capture/eager construction; they are not counters advanced on device replay. The direct-AR semaphore responsible for this visible wait is separate from those arrays.
4. Both retained model traces are captured on CQ0 (`generator.py:534–545`). A decode submits exactly one selected model trace, followed by the existing sampling trace, on the same CQ0 (`serving.py:378–391`). Nonblocking host submission does not create an additional command queue. There is no new simultaneous B2/B32 execution API.
5. `generator.py:861–871` releases traces only in `close()`. The candidate retains both in `self.traces`; it introduces no retirement during request decode. The workspace outlives those references. No source-level use-after-free is visible in this path.
6. All-reduce program-cache override refreshes input address, both sender and receiver global-semaphore addresses, scratch CB address, and output CB address (`all_reduce_async_program_factory.cpp:678–706`). The extra variant does not leave an unpatched semaphore address in this implementation.
7. B2 is padded to 32 inside `_moe` before routing/dispatch/combine (`multichip_decoder.py:1541–1552`). B2 and B32 therefore use the same MoE logical token count, dispatch capacity, persistent EP output entry, and combine `seq_len_per_chip=32`. `ep_outputs` is keyed by capacity and allocated only if missing (`1699–1710`). This reduces the plausibility of a simple combine capacity mismatch caused solely by B2; different surrounding allocations, addresses, routing values, and launch history remain possible discriminants.

The fixed-B32 baseline uses this same workspace sharing and per-op global semaphore reset. Its substantially greater completed aggregate decode work also contradicts a simple low replay-count wrap in that semaphore. It does not rule out counters in the **NoC zeroing path**, which is a separate hypothesis: the observed barrier ultimately compares `NIU_MST_WR_ACK_RECEIVED` to `noc_nonposted_writes_acked[noc]` (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:750–752`).

## Discriminating next check

Preserve and inspect device 1 combine reader's actual NoC write issued/acknowledged counters and captured zeroing runtime arguments: output buffer address, page start/end, page size, and accessor distribution. Compare those values against the allocated combine output and the passing B32 program. `zero_init_common.hpp:17–26` zeroes an L1 scratch, loops over the exact output page interval, and waits for all zeroing writes. Establish whether the barrier is awaiting a real missing acknowledgement, has inconsistent software accounting, or was given an invalid destination before changing CCL semaphores.

A fresh bounded full-layer B2-only trace run (same capacity/cache, without retaining B32) versus the retained-B2/B32 candidate can distinguish retention/address pressure from ordinary B2 execution, after the owner safely handles the live failure. Adding a collective barrier or resetting the AR global semaphore would target a downstream waiter and is not justified by this snapshot.
