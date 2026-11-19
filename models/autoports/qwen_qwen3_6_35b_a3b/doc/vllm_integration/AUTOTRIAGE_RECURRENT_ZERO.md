# AutoTriage: recurrent combine zero-write completion hang

Source-only independent investigation, 2026-11-16. No device access, MMIO reads, target execution, process mutation, or implementation changes were made. This report follows the installed `tt-autodebug:autotriage` skill and its detailed evidence method.

## Diagnosis

The recurrent first observed failure is again device 1, physical worker (1,2), logical (0,0), NCRISC, waiting for combine output-zeroing write acknowledgements. **There is a concrete earlier completion gap worth testing:** `moe_fused_swiglu_reader.cpp` can return after nonposted multicast writes have departed, without waiting for their acknowledgements. The next kernel seeds its expected acknowledgement count from the current hardware acknowledgement count. Late acknowledgements from the previous kernel can then make the next kernel's exact-equality write barrier wait forever.

The source establishes that this boundary is not drained on every path. The available capture does **not** establish that late acknowledgements actually occurred in this run. The hypothesis is more specific than a generic zero-helper or CCL issue and admits a small mixed-program experiment. Do not call it a verified root-cause repair until that contrast and the original workload pass.

## Direct evidence and limits

- `logs/tt_controls_post_reboot.log:2408–2409` records q01 greedy and sampled, 256 tokens each, completing before the stall. `logs/tt_controls_post_reboot_stack.txt` places the host at `generate` step 159, q02 greedy, prompt length 25, previous token 22238, blocked in `_read_tokens`/`ttnn.to_torch`.
- `triage/post_reboot_controls_hang.txt:149` places device-1 NCRISC at `reader_combine.cpp:149` → `zero_pages:26` → `Noc::write_zeros_dram_barrier:51` → `noc_async_write_barrier:1960`. Its BRISC is waiting at `writer_combine.cpp:158`, before initialization publication. Untilizer readers wait for counter-ready; other devices wait in reduction receivers.
- `dump_running_operations` in that artifact records device 1 at op **320374**, previous 320373, and devices 0/2/3 at op **320388**, previous 320387. These are also the op IDs in the pre-reboot failure. Because traces retain launch metadata, this is consistent with recurrence at the same captured program occurrence, not a newly increasing replay counter. It does not by itself identify the model layer; Inspector cannot map these IDs to operation parameters.
- The raw artifact inspected contains call stacks, running operations, and Ethernet status. Its companion summary lists more tools as passing, but no blocked NCRISC hardware/software acknowledgement pair is present in the raw evidence. Do not infer a healthy completion counter from the summary.
- The previous isolated `zero_pages` test passed 5,000 retained replays. It executes an isolated repeated zeroing program, without the preceding expert multicast program or its kernel-boundary accounting. It therefore does not test the transition identified below.

## Native zeroing and acknowledgement ledger

Paths here are repository-relative. `BH` means `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h`; `MOE` means `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/`.

| Boundary | Source contract | Result |
|---|---|---|
| Zero extent | Combine output is BF16 `[1,1,32,8,2048]`; sender takes pages `[0,52)`, four untilizers each take 51 remaining pages. | Sender issues 52 × 4096-byte writes; intervals are disjoint and cover 256 pages. This remains source-derived, not a live runtime-argument dump. |
| Local scratch | `zero_init_common.hpp:21–26` initializes a 16384-byte scratch through local zero reads and completes their read barrier before DRAM writes. | Scratch size covers every 4096-byte page. A read/write barrier is already present; adding a supposedly missing barrier here is unjustified. |
| Write issue | `internal/noc_zero_dram.inl:40–46` calls `noc_async_write_impl` with the accessor-mapped address. `dataflow_api_raw.h:148–175` selects `ncrisc_noc_fast_write_any_len`; BH:986–1033 emits one command for 4096 bytes. | Nonposted unicast, `num_dests=1`, exactly one command and one expected ACK per page. No local double increment found. |
| Counter update | BH:654–659 increments dedicated-mode issued by one and expected acknowledgements by `num_dests`. | Sender adds 52 to its current expected ACK baseline. |
| Completion | BH:750–752 compares `NIU_MST_WR_ACK_RECEIVED == noc_nonposted_writes_acked[noc]`. | Equality can fail indefinitely for an overshoot as well as an undershoot. The capture does not distinguish these. |
| Kernel entry | `ncrisck.cc:58–62` calls `noc_local_state_init(NOC_INDEX)` for dedicated mode; BH:873–886 reads current hardware counters and assigns the software baseline. | The new kernel does not inherit the previous kernel's software expected ACK count. Outstanding old ACKs arriving after this sample are unaccounted in its new expectation. |
| Kernel exit | `ncrisck.cc:84–93` checks reads complete, nonposted writes **sent**, atomics complete, posted writes sent, packet tags and MID; no write-ACK equality assertion. BRISC has the analogous checks in `brisck.cc`. | A sent-only writer can pass the write-related exit check while ACKs remain in flight. Assertions are also disabled in this workload. |

For an old hardware count `H0`, `P > 0` old acknowledgements still in flight at new-kernel initialization, and 52 new zero writes: new software expectation is `H0 + 52`; the eventual hardware count is `H0 + P + 52`. The exact barrier cannot become equal after overshoot. This is an accounting-boundary mechanism, not a claim of counter wrap.

## Concrete predecessor gap

`multichip_decoder.py:1659–1696` executes native fused SwiGLU and then native combine. The intervening reshapes are tensor-view operations at this call level; the captured preceding on-core kernel identity remains unavailable and should be recorded in the experiment.

1. The fused reader broadcasts indices, counts, and optional starts on the grid's root `(0,0)` (`MOE/kernels/moe_fused_swiglu_reader.cpp:450–486`). Those payload multicasts are nonposted. It calls `noc.async_writes_flushed()`, multicasts a nonposted validity semaphore, calls the same sent-only flush, and resets the local semaphore.
2. `noc_async_writes_flushed` explicitly waits for departure, not completion (`dataflow_api.h:1971–1993`). A receiver seeing VALID proves arrival at that receiver; it does not prove the sender has received every hardware write acknowledgement.
3. A zero count produces `m_blocks == 0`, and the entire expert compute/collective loop is skipped (`reader.cpp:611,671–673`). The function ends at line 1365 with **no unconditional write acknowledgement barrier**. Thus the all-local-experts-empty path never reaches `scatter_payload`'s write barrier at `moe_fused_swiglu_dataflow.hpp:181`.
4. This path is legal for local expert dispatch. The model masks counts to one rank's 64 experts (`multichip_decoder.py:1031–1045,1623–1626`). A particular rank may receive no routed tokens. No captured counts establish that this happened here; it is a targeted fixture, not an asserted fact about q02.
5. Even for nonempty experts, phase-2 H multicast occurs **after** the phase-1 scatter barrier. `h_slot_send_posted` (`reader.cpp:212–266`, called at 1161/1275) ends with the same sent-only flush. The actual default `H_MCAST_POSTED=false` and `H_ROUND_NOC1_MASK=0` (`moe_fused_swiglu_geometry.hpp:34,42`) mean the reader owns these nonposted payload and flag multicasts. The final reader epilogue drains pending reads where necessary but contains no final write-ACK drain. Compute and output writes usually add time; they are not an explicit acknowledgement contract for reader NoC0.
6. The fused writer's conditional output drain (`writer.cpp:642–646`) operates on **its own NoC1**, so it does not discharge the reader's NoC0 ACK accounting. Adding another writer barrier would target the wrong owner.

The earlier report's observation that the fused **writer** already drains output is correct, but insufficient: the reader has distinct multicast transactions and no corresponding unconditional epilogue drain.

## NoC ownership, mode and init ledger

- Combine reader is RISCV_1 with the preferred DRAM-read NoC0; writer is RISCV_0 with preferred DRAM-write NoC1 (`combine_program_factory.cpp:1053–1071`, `kernel_types.hpp:141–153`). The descriptor's omitted mode defaults to `DM_DEDICATED_NOC` (`program_descriptors.hpp:98–102`). `Noc()` captures the calling kernel's `noc_index`; its destination mapping, issue and barrier use that same ID.
- Fused SwiGLU uses `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (`moe_fused_swiglu_program_factory.cpp:574–590`). Its writer explicitly documents and uses one NoC1 for the entire payload/atomic path (`writer.cpp:193–194`). Reviewed raw transaction-ID calls use the default selected NoC. No explicit writer use of the reader's NoC0 was found in this predecessor.
- Dedicated mode gives the two RISCs independent selected NoCs; their command-buffer numbers may coincide, but they address different NoC register banks. Dynamic mode instead partitions command buffers by RISC and adds both RISCs' shared software counters at barriers. No dynamic mode is configured for these two operations. The isolated helper uses the same dedicated split, so its passing result does not implicate mode selection itself.
- BRISC firmware reads each launch's NoC ID/mode, reruns `noc_init` only when mode changes, initializes local counters, and starts NCRISC after its setup (`brisc.cc:598–629`). `ncrisc.cc:185–190` waits for the GO notification before entering the kernel; kernel entry then samples its local counter baseline. No source evidence shows these two kernel entries racing a simultaneous NoC0 issuer inside the same combine operation.
- Ordinary writes intentionally inherit MID=0 and clean transaction tags. Mode-change init clears MID; same-mode launches do not universally scrub inherited command state. That remains an alternative, but no dirty MID/tag is captured and no preceding PCIe operation has been identified. It is weaker than the explicit undrained-reader path above.
- The blocked combine writer has not reached fabric connection setup at all. Its later path has a write barrier and final full barrier (`writer_combine.cpp:347,388`). Prior dispatch writers also contain explicit write/full-barrier drains (`writer_sender_dispatch.cpp:503,548`; `writer_worker_dispatch.cpp:603–608`). The reviewed native writer paths do not establish an alternate-NoC BRISC issuer corrupting this sender's dedicated NoC0 count. This is a scoped audit, not a claim that every earlier model kernel was exhaustively examined.

## Smallest discriminating experiment

After the owner handles the live failure under the original timeout/device policy, run a **mixed-program** retained trace, keeping the current isolated-zero test intact as its passing control:

1. Program A is the native fused SwiGLU with the same 8×8 worker grid and selected precision, using a legal all-zero local-count fixture; also test a one-nonempty-expert fixture. Its weights/input can be a single decoder's valid test tensors. Program B immediately runs the existing exact native `zero_pages` probe with sender NoC0 and four untilizer NoC1 roles. Capture A→B together, rather than repeatedly running B alone. Use the same 1×4 mesh, bounded timeout and poisoned-output checks as the existing zero probe. No server or other device owner may overlap it.
2. If a lighter synthetic A is needed, copy only the fused reader's exact root multicast/count-valid sequence and its sent-only flush onto an 8×8 grid with receiver VALID waits. Retain correct multicast bounds/63 destinations and both dataflow kernels. Label this an accounting mechanism probe, not a full-native-MoE reproduction.
3. Record **on-device into a reserved ordinary L1 diagnostic buffer**, at A entry/exit and B entry/pre-barrier, the selected NoC, software issued/expected ACK values, and the hardware sent/ACK values already read by the normal kernel/barrier API. Record page interval and output address too. Export these ordinary-buffer values through the established safe evidence path. Do not repeat host reads of private NCRISC or NoC MMIO registers. A bounded diagnostic loop may update that reserved buffer while retaining the original wait condition; it must never waive completion or publish output after timeout.
4. Controlled variant: change only A's reader epilogue to `noc.async_write_barrier()` immediately before return. Keep B, page mapping, precision, trace count and fixtures identical. Baseline ACK overshoot at B plus the epilogue variant's clean completion would establish the missing cross-kernel drain. An undershoot instead favors address/transport/overcount and rejects that explanation. The epilogue barrier is absent from the current reader source; this is not a proposal to add an already-existing behavior.

The strongest follow-up is the same original 12-case full-40 qualitative control and the serving lifecycle after a demonstrated repair. A reduced mixed-program pass alone cannot accept the original full-model gate. Do not weaken zeroing, semaphore thresholds, or the canonical sampler to make this test finish. No proposed experiment or source variant was executed by this reviewer.

## Uncertainty

No live NCRISC expected/actual ACK pair, expert counts, output runtime address, or exact previous on-core kernel identity was captured. Consequently the source gap is concrete, but attribution of this particular hang remains a hypothesis. The same physical worker and retained op ID across both failures make a fixed program occurrence/transition especially useful to isolate. They do not establish a defective device, fabric counter wrap, or numerical/token failure.
