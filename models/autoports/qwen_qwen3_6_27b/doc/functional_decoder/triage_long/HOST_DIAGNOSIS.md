# AutoTriage: host fetch-queue reservation stall

## Owner correction: capture interference prevents a root-cause claim

The queue/host snapshots below were collected **after** full tt-triage. The
checkout's `tools/triage/triage.py:959` explicitly halts Blackhole cores and
suppresses resume because HALT → READ/WRITE → CONTINUE can break cores.
Before that capture, the only observations were log silence and process
CPU/wait state, which cannot distinguish slow execution from a hang. Thus the
captured queue state may have been created or perturbed by triage. It does
**not** establish a preexisting runtime consumer/visibility defect. The raw
values remain useful forensic evidence, with this causal limitation.

The proposed standalone private-memory halt/resume helper was never run and
was removed after discovering this documented hardware issue. Normal
open/add/read/close after the original1800-second timeout passed in2 seconds
(`../recovery_smoke.log`), without operator reset. A fresh full-context run
with progress markers is required. The diagnostic drain idea is unverified.

## Initial diagnosis (qualified by the correction above)

The captured stop is a **prefetch consumer visibility/cursor failure with a legitimately full FetchQ**, not an initial >2 GiB tensor upload or allocation wait. The later raw-L1 snapshot establishes 1,535 nonzero slots out of 1,536, with only the host's required guard slot zero. The host is correctly waiting for space. The device prefetcher nevertheless loops as though its next slot were zero. The actual private-memory consumer pointer is still unknown; distinguishing a wrong pointer from a correctly addressed stale read is the next decisive check. A cumulative 32-bit command-count overflow is not supported by the inspected queue implementation: its producer and consumer cursors are bounded L1 slot addresses and wrap at the queue boundary.

### Later raw-L1 capture: supersedes an empty-queue interpretation

Parent ran `dump_qwen_fetchq.py`; `fetchq_snapshot.json` and `fetchq_capture.log` record:

| Field | Observed value |
|---|---:|
| Queue base / exclusive limit | `0x1bd40` / `0x1d540` |
| Queue bytes / four-byte slots | 6,144 / 1,536 |
| Published last-consumed fence, before and after snapshot | `0x1d084` (index 1,233) |
| Host next write and cached fence, parent debugger capture | `0x1d084` / `0x1d084` |
| Sole zero slot | `0x1d084` |
| Next consumer inferred from source transition | `0x1d088` (index 1,234) |
| Value at inferred next slot | `2824`, encoding 45,184 bytes, no stall flag |
| Nonzero slots | 1,535 |

Thus the host's full-ring guard condition is corroborated by raw SRAM, and a stale **host** fence read is demoted. The no-work observation applies to what the prefetch RISC **sees**, not what the queue physically contains. There is no missing publication at the inferred next slot. The inference `next=last_consumed+4` is not a substitute for reading the real private pointer; loss/corruption of that state is now the leading concrete discriminator.

Prioritized mechanisms:

1. **Consumer private cursor differs from `0x1d088`**, especially if it equals the sole zero guard `0x1d084`. This immediately explains the persistent loop despite full SRAM. Read the static pointer at BRISC private address `0xffb00d04` and verify `gp`/load address if possible. If wrong, then investigate the first writer/corruption of that private datum; source already performs the required increment and wrap, so merely adding an increment is not a supported fix.
2. **Consumer pointer is correct but RISC reads stale zero from L1.** The source already invalidates Blackhole L1 inside every empty-slot poll, and compiled machine code contains that invalidation. Therefore this would be an actual visibility/hardware/runtime condition, not a forgotten fence in the present source.
3. **Captured runtime differs from the expected code/state, or triage altered its progress.** The captures are sequential, and the original detailed triage warned that some halted cores resumed. A fresh PC and pointer snapshot can separate this from a persistently executing expected wait loop. There is no evidence presently tying those warnings specifically to prefetch BRISC.

Compiled candidate ELF verification (host-only `riscv-tt-elf-objdump -dl`): at ELF `0xc08c` the loop executes `fence`; at `0xc090` it loads `prefetch_q_rd_ptr` from `gp+1300`, annotated as `0xffb00d04`; at `0xc094` it loads the slot through that pointer; at `0xc098` it branches to `0xc08c` on zero. These are ELF addresses before runtime relocation. `tt_metal/hw/inc/internal/tt-1xx/cache.h:8-16` documents Blackhole's four-line write-through L1 cache and defines `invalidate_l1_cache()` as the emitted `fence`. The volatile slot load is not hoisted out of the loop. Quasar alias/phase changes do not apply (`cq_common.hpp:69-90`, `dispatch_mem_map.cpp:25-30`).

ELF identity qualification: `generated/inspector/kernels.yaml` maps captured watcher kernel ID 1 to cache variant `7334400216326404496`, which supplied the inspected ELF. Its `.text` spans `0x5110..0xcb30`, whereas captured runtime PC is `0x11368`. Runtime relocation has not yet been checked from current mailbox `kernel_config_base/kernel_text_offset` or a verbose callstack record. The raw queue geometry and fence address match the independent host capture, but that alone does not prove the instruction/private-symbol identity. Parent should verify the runtime kernel offset/path with targeted verbose dispatcher/callstack evidence before interpreting the private pointer under that ELF. If corresponding stop is ELF `0xc098`, the runtime displacement would be `0x52d0`; that is a consistency check, not a measured relocation.

The parent may capture the BRISC private cursor with the debugger's supported private-memory API **only as an explicit evidence-preserving parent recovery action**. Unlike the completed raw L1 script, that operation can involve debug-module state or halt handling. Existing `RiscDebugMemoryAccess(..., ensure_halted_access=False)` is not a guarantee of zero halt/state changes: `memory_access.py` still enters `ensure_private_memory_access()` when `can_debug()` is true. Do not extend the raw script by blindly NoC-reading `0xffb00d04`; it is a private address, not an L1 address. A correctly captured value at that address, plus the expected loop's current PC, is the most useful next evidence.

No device command, TTNN import, reset, target run, or implementation edit was performed for this investigation. This report follows the installed AutoTriage skill and its `references/AUTOTRIAGE_PROMPT.md`, using the supplied captures as primary evidence.

## Triage evidence

- `host_stacks.log` thread 1 and `main_stack.log` show `SystemMemoryManager::fetch_queue_reserve_back` -> `program_dispatch::write_program_command_sequence` -> `FDMeshCommandQueue::enqueue_mesh_workload` -> `BinaryNgDeviceOperation`; the fuller stack resolves the pending operation as `ttnn::add`.
- This is program enqueue, not `from_torch`, write-buffer upload, allocator wait, or a kernel currently computing the add. `py-bt` was unavailable, so the exact Python add and outer token chunk are not established.
- `tt-triage.txt` has no running worker operation. Prefetch BRISC at physical `(16,2)` is in `fetch_q_get_cmds`, `cq_prefetch.cpp:943`, waiting for `prefetch_q_read_slot(...) != 0`. Dispatch BRISC `(16,3)` waits for upstream pages. Dispatch waits are downstream of absent prefetch work, not evidence of an attention or MLP kernel deadlock.
- Runtime configuration shows `timeout_duration_for_operations=0s`; the host reserve wait has no configured dispatch timeout.
- Do not treat `triage-summary.txt` as a clean hardware report. The detailed capture reports ERISC counter mismatches under `check_noc_status.py` and several cores that resumed after triage halted them under `check_broken_components.py`. Those observations are not enough to attribute this host/prefetch mismatch to NoC or damaged hardware, but they remain unclassified and are not silently waived here.

## Source evidence and cursor ledger

| Owner/state | Source and transition | Stalled implication |
|---|---|---|
| Host next FetchQ slot `prefetch_q_dev_ptrs[cq]` | `tt_metal/impl/dispatch/system_memory_manager.cpp:309,681-749,847-897`: starts at base; writes one encoded size; advances by entry width; wraps at limit | Reserve polls only when host pointer equals its last observed device fence |
| Device last-consumed fence | `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:600-612`: clear consumed slot, then publish **consumed slot address** to `PREFETCH_Q_RD` | A fresh published fence equal to host pointer means producer must leave that guard slot unavailable |
| Device next-consume slot | Same file `614-619`: increment entry pointer and wrap to base; `935-943`: poll current slot for nonzero | No work is visible in the next consumer slot |
| FetchQ entry publication | `system_memory_manager.cpp:872,892-897`: host store fence, then 32-bit entry write on Blackhole, then pointer increment | Command bodies precede their size publication; source does not omit publication on the normal path |
| Host issue buffer | `system_memory_manager.cpp:518-548,589-616`: address within bounded issue ring, wrap at limit | Large aggregate traffic does not monotonically advance a byte cursor past 2 GiB |
| Program sequence | `tt_metal/impl/program/dispatch.cpp:3183-3244`: write body, push issue pointer, reserve FetchQ, publish FetchQ size; one-shot and split paths both do so | Current unpublished add can be held by the broken handshake, rather than causing its device-side stall |

Blackhole uses four-byte FetchQ entries (`dispatch_mem_map.cpp:64-70`); cached/phase-tagged FetchQ is enabled only for Quasar RTL simulation (`dispatch_mem_map.cpp:25-30`). Consequently a phase-bit/cache-lap explanation from Quasar does not apply to this Blackhole run. Host and device use the same queue size and entry width from the dispatch configuration. The uncached fence publication already exists in source; adding that same fix would not repair a missing behavior.

With a consistent ring and visible stores, host-full and device-empty cannot persist together: the host stops one consumed slot behind the consumer, while the slots leading to that guard should already contain pending entries. The later raw snapshot confirms exactly those pending entries; the next missing datum is the actual consumer cursor/read, not total uploaded bytes. The device stop is also outside the in-flight PCIe-read retirement branch; it indicates that the RISC considers nothing available or pending to fetch.

The long runner uploads an approximately 2.5 GiB input (`262143 * 5120 * 2 = 2684344320` bytes before tile padding), but reaches public `prefill_forward` and a binary add enqueue. There is no evidence of a signed-size conversion at this stop. The first upload and setup synchronization necessarily precede prefill in `tests/run_long_context.py:128-133`. The decoder's `prefill_forward` issues all 512-token chunks without an intervening device synchronization, retaining outputs until concat (`tt/functional_decoder.py:218-243`). That provides sustained command traffic and many queue wraps; it does not prove a cursor-wrap bug by itself.

## Proposed minimal target-local discriminator/workaround

After parent-owned recovery and a successful trivial smoke, add **bounded draining between completed physical prefill chunks** in the target decoder, for example `ttnn.synchronize_device(self.mesh)` after each chunk during the discriminator. This changes neither supported context nor device math and introduces no torch/host numerical fallback. Log the absolute chunk offset immediately before and after each drain in the harness or a temporary diagnostic wrapper. If that passes, test a less frequent drain interval and record both correctness and measured cost before retaining a target-local workaround.

This is a hypothesis test for sustained outstanding enqueue pressure, not a verified fix. It will not repair a deterministic lost queue entry caused by every ring lap, and a drain itself can block if the inconsistency recurs. Do not describe it as eliminating a 2 GiB limit or proving an overflow. Splitting the setup upload alone does not target the observed pending-add reservation, so it is not the first causal repair to test.

The raw-L1 and parent host debugger captures now supply host cursor/fence, published `PREFETCH_Q_RD`, queue geometry and all slots. The remaining decisive datum is device-private `prefetch_q_rd_ptr` (and current loop PC). The drain workaround remains only a recovery experiment; a full queue is normal backpressure and the new evidence does not establish it as the root cause.

## Remaining uncertainty

- No exact queue-index arithmetic defect is established from source; changing core runtime based on the present report would be premature.
- The precise failing Python add, completed token count, and number of queue wraps are not available in the capture. The add may be residual addition or RoPE addition; neither is itself proven bad.
- A visible host stack polling device registers can consume CPU, so the separately reported frozen CPU-time estimate is not used as causal proof.
- Broken-core and ERISC triage warnings require parent classification during recovery. This report does not authorize/reset hardware or claim it healthy.
- The proposed drain workaround and full-context rerun remain unverified hardware work for the parent.
