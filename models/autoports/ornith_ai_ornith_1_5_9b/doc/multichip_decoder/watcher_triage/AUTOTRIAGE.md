# AUTOTRIAGE

## Diagnosis

- The fused prefill operation violates its **dedicated-NoC accounting contract**: its NCRISC reader issues weight reads on NoC1, while its BRISC writer owns NoC1 and checks that NoC's hardware read-response total against BRISC's private issued-read counter at kernel exit. The writer issues no reads, so completed NCRISC reads make that equality fail. This explains the BRISC `NKFW` assertion without requiring any unfinished read. The operation's per-step transaction-ID barriers already wait for those reads; they do not reconcile the two RISCs' counters. This is an operation-side ownership/accounting defect, not evidence that watcher should be disabled.

A model-local public-API workaround exists: cap this operation's planning budget at 64 KiB using `l1_headroom`, forcing the streaming regime, which keeps both operands' reads on NCRISC's NoC0. The source proof and a conservative fit bound are below. Hardware correctness and performance of that workaround remain for the main agent to validate. No source was changed and no TTNN, device, recovery, or AutoDebug launcher was run during this investigation.

## Triage Evidence

1. **Evidence scope.** The required `../AUTOTRIAGE_INPUT.md` was read first. It describes the earlier isolated DRAM-sharded matmul geometry failure, including a superseded candidate description; it does not describe this watcher failure. That report and its `../AUTOTRIAGE.md` were not modified. This report uses the watcher artifacts specified in the current request.

2. **Direct stop-site evidence:** `../watcher_0.log:291-300` records, at **2026-11-12 20:04:19.474 UTC**, device 0, logical worker **(1,1)**, virtual **(2,3)**:

   - BRISC: `matmul_reduce_scatter_prefill_writer.cpp`.
   - Reported assert: kernel completing with pending NoC transactions, “missing NOC reads flushed barrier.”
   - Waypoints: **`NKFW, W, W, W, W`**.
   - NCRISC kernel metadata names `matmul_reduce_scatter_prefill_reader.cpp`; all three TRISC entries name `matmul_reduce_scatter_prefill_compute.cpp`.
   - The watcher thread throws and the host process aborts. This is not an LLK unpack/pack assertion, nor a stack trace identifying a fabric send stall.

   Firmware establishes that `NKFW` is after `kernel_main()` returns. The first check there is the read-counter equality, before write/atomic/tag checks. The diagnostic wording alone does **not** establish the number or existence of outstanding reads.

3. **Retained tt-triage is incomplete.** `tt-triage-retained.txt` contains health output, not an operation/counter dump. `triage-retained-summary.txt:1` and `console-retained.log` say the inspector was unavailable, with no live RPC or usable serialized logs at the specified inspector path. Consequently running-operation, previous-operation, call-stack, NoC-status, binary-integrity, CB, semaphore, and lightweight-assert inspections were skipped. A passing `check_cb_inactive` is not proof that the failing operation's CB ledger was inspected. The available checks report DRAM training/BIST success, zero uncorrected DRAM errors, live Ethernet links, responsive ARC, and passing broken-component/location checks. These health observations do not prove perfect hardware or explain the assertion.

4. **Retained watcher snapshots do not contain the failing register values.** The original artifact copies under `original_run/generated/` are preferable to the reusable `../watcher/0/generated/` paths. The copied watcher log ends partway through device 0's scan, at logical (0,1), immediately before the reported asserting core. It provides no response/issued-count pair or outstanding-TRID values for (1,1). The host log preserves the actual assertion. `mesh_workloads_log.yaml` is empty; compilation records are not dispatch records.

5. **Passing/failing contrast:** the original host log completes prefills through length 33 and subsequent decode/refresh checks. The next length in `../final_contracts_0.json` is 1023, padded to 1024 for the fused call. The no-watcher report records all 11 lengths (1, 2, 3, 4, 31, 32, 33, 1023, 1024, 1025, 2051), and `../final_long_0.json` records maximum-context checks through 262144. These are retained numerical results, not fresh runs or proof that watcher contracts hold.

6. **Compiled plan evidence strengthens the contrast.** The original `inspector/kernels.yaml` identifies cached reader ELFs. Offline `readelf --debug-dump=info` exposes their `kernel_compile_time_args` as a little-endian uint32 array; decoding it with `reader_ct` in `matmul_reduce_scatter_prefill_args.hpp:63-100` gives:

   | Compiled reader hash | Inspector program IDs | M tiles / K tiles / N tiles | Core M × N; K block | Resident A; A/W read NoCs |
   |---|---|---|---|---|
   | `8219329228262968830` | 168, 170, 172, 174 | 1 / 32 / 128 | 1 × 3; 16 | yes; **0/0** |
   | `8079777566301672127` | 188, 190, 192, 194 | 1 / 96 / 128 | 1 × 3; 24 | yes; **0/0** |
   | `8128803472369341988` | 330, 332, 334, 336 | 2 / 32 / 128 | 1 × 3; 16 | yes; **0/0** |
   | `1241370424416735290` | 348, 350, 352, 354 | 2 / 96 / 128 | 1 × 3; 24 | yes; **0/0** |
   | `17759633515502528666` | 394, 396, 398, 400 | 32 / 32 / 128 | 3 × 4; 16 | yes; **0/1** |
   | `2520488003482403501` | 404, 406, 408, 410 | 32 / 96 / 128 | 3 × 4; 16 | yes; **0/1** |

   Both last plans have four entries, one wave, one chunk, `P=12`, A split 2 rows, W split 8 K rows, and one-step prefetch. They correspond to 1024-row attention-output and down projections. The files are under `/home/ttuser/ornith-9b/tt-metal-cache/tt-metal-cache15744046203423885121/kernels/matmul_reduce_scatter_prefill_reader/<hash>/ncrisc/ncrisc.elf`. Their DWARF also records `noc_index=0`, `noc_mode=0` (dedicated), and read command buffer 1. These are compiled configurations, not captured device counters. The retained evidence cannot conclusively assign the assertion to program 394 versus 404; both contain the same defect.

## Source Evidence

Paths in this section are relative to the repository root. The operation's shared directory is `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/`.

### First source-side contract violation

- `device/matmul_reduce_scatter_prefill_program_factory.cpp:663-687` builds NCRISC/RISCV_1 on `RISCV_0_default` = NoC0 and BRISC/RISCV_0 on `RISCV_1_default` = NoC1. `tt_metal/api/tt-metalium/program_descriptors.hpp:98-102` defaults these descriptors to **DM_DEDICATED_NOC**.
- The same factory, lines 525-536, sets A reads to NoC0 and W reads to NoC1 whenever the plan is resident and outside the thin-wide exception. The exception requires scatter on N, N along x, at most four M lines and at least ten N lines. Short prefills meet that exception; the compiled 1024-row plans do not.
- `device/kernels/matmul_reduce_scatter_prefill_reader.cpp:110-114` explicitly assumes that because the writer only writes, the reader can own reads on both NoCs. `issue_w`, lines 314-335, then passes `w_read_noc=1` into `noc_async_read` from NCRISC. **The first such step-0 read is the first accounting ownership violation.** No missing writer barrier is needed to produce it.

The private-versus-shared distinction is concrete:

| Resource | Producer / owner | Accounting and completion contract |
|---|---|---|
| NoC0 A reads, multicast writes, reader atomics | NCRISC | NCRISC owns the NoC and initializes its local counters from hardware at entry. |
| NoC1 W reads | **NCRISC**, on the resident split-NoC plan | Read API increments NCRISC's `noc_reads_num_issued[1]`; transaction completion increases the core's shared NoC1 hardware response counter. |
| NoC1 output writes and mailbox atomics | BRISC | Writer performs its own write/atomic completion checks; it never increments BRISC's read-issued total. |
| NoC1 exit read check | **BRISC** | Compares the shared hardware response total with **BRISC's** read-issued baseline; NCRISC's increments are invisible to this comparison. |

`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:73` and `ncrisc.cc:36` define separate RISC-local arrays. Kernel wrappers `brisck.cc:66-90` and `ncrisck.cc:60-85` initialize only their assigned NoC. In Blackhole's `noc_nonblocking_api.h:872-885`, initialization snapshots `NIU_MST_RD_RESP_RECEIVED` into the calling RISC's counter. Read issue increments the caller's counter (for example lines 587-588); lines 599-605 distinguish two different predicates:

```text
dedicated global read barrier: HW_RD_RESP_RECEIVED[noc] == this_RISC_reads_num_issued[noc]
transaction-ID read barrier:   HW_REQS_OUTSTANDING_ID[noc, trid] == 0
```

Let `H1` be BRISC's initial NoC1 hardware response snapshot and `r > 0` the NCRISC W-read responses completed after it. At writer exit:

```text
BRISC expected read responses = H1       (writer issued zero reads)
hardware read responses       = H1 + r   (NCRISC's completed reads count here)
```

Thus the exit equality fails even when **every outstanding-read count is zero**. More waiting cannot remove completed responses. NCRISC's unowned-NoC software total also lacks that NoC's normal entry baseline, so a plain NCRISC global barrier on NoC1 is not a substitute for correcting ownership.

### Why logical core (1,1) is predicted

For the recorded 32 × 32-tile output block on the model's 11 × 10 worker grid, the planner/factory place an M-on-x **11 × 8** matmul grid at logical origin **(0,1)** (`planner.cpp:261-292`, `program_factory.cpp:414-442`). Consequently logical (1,1) is `(gm,gn)=(1,0)`.

Replaying `mrs_plan_mcast_senders` (`program_factory.cpp:198-260`) gives starts **a=0, a2=4, b=1, b2=5**. Logical (1,1) sends **W part 0** and neither A part. W part 0 is eight K rows by four valid N columns: **32 tile reads per step** on NoC1. Its predicted completed-response increase is 256 for the attention plan (4 entries × 2 steps × 32) or 768 for the down plan (4 × 6 × 32). These are calculated counts, not register measurements. Logical (0,1), immediately preceding it in the retained scan, sends no W part and has no such NoC1 read-count increase. This explains the reported first core more specifically than a general CCL or hardware theory. Other W senders are also exposed; the report does not prove that they all asserted.

### Transaction IDs and CB/semaphore ledger

The predicate `two_read_nocs = (a_read_noc != w_read_noc || a_read_noc != noc_index)` is **true** for the failing 0/1 reader. It does not accidentally omit the second barrier.

| Resource / sequence | Producer | Consumer / required count |
|---|---|---|
| Read transaction IDs | NCRISC tags step `s` with `1 + s % 8` on NoC0 and, when `two_read_nocs`, NoC1 (`reader.cpp:347-353`). | For every moving sender step, barriers wait on that ID on NoC0 **and NoC1** before multicast/CB credit (`468-472`). |
| Look-ahead read windows | `reserve_current` / `look_ahead` reserve and issue only steps below `num_steps`; prefetch is 1. | Every flat step is visited; at most the next step is prefetched, far below the eight-ID reuse distance. The final step cannot leave an out-of-range prefetched read. |
| CB0 A, CB1 W | Reader pushes `core_m*kb` and `kb*core_n` tiles per flat step, after sender read barriers and receiver multicast-ready waits (`548-549`). | Compute waits/consumes the same windows (`compute.cpp:218-265` and its window-pop logic). For the recorded down plan: 24 steps, 48 A tiles and 64 W tiles per step. Resident A is physically read only for the initial six steps, then re-credited after ring wrap; W moves every step. |
| A/A2/B/B2 multicast handshakes | Each receiver reserves a window and signals consumer-ready once per moving family; each sender waits for its line's receiver count, multicasts, then signals data-ready. | Receiver waits/reset occur in the same A, A2, B, B2 order. Reader flushes earlier multicast sources before reuse (`438`) and barriers the last writes before exit (`564`). This operates on NoC0. |
| CB3 finished output | Compute packs `P` tiles per entry. | BRISC handles exactly `num_chunks*num_entries` entries, then drains its queue (`writer.cpp:95-130`). Recorded plan: four entries × 12 tiles = 48 tiles, handoff depth 2. |
| FWD/BWD mailbox and CONSUMED semaphore | BRISC increments the selected port mailbox once per non-FIN entry. Port NCRISC gathers that entry's finished tiles. | Port waits for successive mailbox counts, barriers reads, then acknowledges the producing core once; BRISC waits for that acknowledgement before popping/reusing the finished slot (`port_reader.cpp:60-68,92-108`; `writer.cpp:83-91`). Remote reads increment the **port core's** counters, not the source matmul core's counters. |
| FIN output | BRISC writes valid output tiles only after compute credit. | It waits for the full entry and executes `noc_async_write_barrier()` before releasing it (`writer.cpp:99-117`); final mailbox atomics drain at line 131. No read is issued by this writer. |
| Arrival CB4/CB5, arrival counters and FREED reports | Upstream fabric delivers data plus increments; reader credits `P` tiles once an entry threshold is met. | Compute consumes the credited arrivals. Reader reports one freed chunk per direction after the corresponding CB acknowledgements, then subtracts exactly `num_chunks*total_{fwd,bwd}` from global arrival counters and barriers atomics (`reader.cpp:195-245,557-574`). |
| CB2 partial hold / CB7 zero | Compute owns the fp32 partial staging; BRISC publishes one zero tile. | These are local compute state; neither adds matmul-core NoC1 reads. No observed LLK/format assertion implicates them. |

The recorded down reader has 6 A-window slots and 4 W-window slots. The activation's resident wrap and all operand windows have explicit producers/consumers; the evidence supplies no CB underproduction signal.

`tt_metal/hw/inc/api/dataflow/dataflow_api.h:2602-2612` already waits for the read command buffer to become ready **before** polling the transaction-ID outstanding count, explicitly addressing Blackhole issue/poll ordering, then invalidates L1 cache. Adding that ordering is not a missing fix in this snapshot. Transaction-ID waits do not write BRISC's private counters. Read tags are also distinct from the exit check for cleared write/atomic packet tags (`noc_nonblocking_api.h:260-265`); the observed assertion happens before that check.

These ledgers provide a source explanation for **completed but differently accounted reads**. They do not constitute a captured all-core proof that no reads were in flight anywhere when the watcher sampled.

## Downstream Effects

- BRISC's assert occurs after its writer work, not in a mailbox wait, CB wait, fabric close, or read issue. A source-local counter inequality is sufficient to halt it. The other four `W` waypoints are firmware wait states; they should not be presented as four identified deadlocks.
- Once BRISC cannot return to dispatch, later work and host synchronization can stall and watcher aborts the host. The retained evidence does not identify a live fabric send-slot, credit, route, or teardown stop, so a fabric-route or reset-recovery diagnosis would be unsupported.
- The earlier 64-core DRAM-sharded geometry bug used a different operation and kernels. Memory-config mismatch warnings in this run do not establish that earlier bug as this failure's cause.
- With watcher assertions absent, the operation can finish numerically because the TRID barriers still wait for its operand data. This agrees with the passing real-weight runs while preserving the significance of the violated ownership contract.

## Proposed Fix

### Within this stage: use the public L1-budget argument to select the streaming regime

The public signature and binding expose **`l1_headroom`**, but no NoC, NoC-mode, read-NoC, prefetch, or resident-regime override. `ComputeConfigDescriptor` controls compute settings, not the hidden data-movement descriptors. A direct Python `two_read_nocs=False` or `noc_mode=dynamic` fix is therefore unavailable.

There is nevertheless a concrete model-local configuration remedy. In `models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py`, obtain the L1 bank size once during construction and pass:

```python
# Proposed change only; not executed or applied in this investigation.
bank_bytes = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_per_bank
assert bank_bytes > 64 * 1024
self.mrs_l1_headroom = bank_bytes - 64 * 1024

# Add to the existing ttnn.experimental.matmul_reduce_scatter_prefill(...) call:
l1_headroom=self.mrs_l1_headroom,
```

Use **allocator bank bytes**, not the physical L1 capacity or currently free bytes. `ttnn/ttnn/device.py:232-233`, `ttnn/cpp/ttnn-nanobind/device.cpp:602-625`, `tt_metal/detail/reports/memory_reporter.cpp:172-182`, and `tt_metal/impl/allocator/bank_manager.cpp:201-204` connect this accessor to the allocator's bank size. Query it at setup, outside timed/trace execution.

Why this removes the faulty source path:

1. `matmul_reduce_scatter_prefill.cpp:200-236` sets the budget to at most `bank_bytes-l1_headroom`; live-L1 constraints and canonicalization only preserve or reduce the budget. The proposed argument therefore bounds it by **65536 bytes**.
2. For these scatter-on-N calls, a resident plan holds `tA*core_m*Kt` activation bytes (`planner.cpp:209-219`). Activations are BF16, `tA=2048`, `core_m>=1`, and the row projections have `Kt=32` or `96`. The resident A allocation alone is at least **65536 bytes**, with positive W/staging/zero allocations in addition. **No resident plan can fit**, even after rechunking or reducing the K block.
3. `choose_k_block` tries streaming after resident candidates fail (`planner.cpp:243-258`). Every streaming plan sets **A_READ_NOC=0, W_READ_NOC=0** (`program_factory.cpp:465-468,525-536`). Thus `two_read_nocs=false`, NCRISC uses only NoC0, and BRISC's NoC1 read count remains its valid baseline. Watcher stays enabled and checks the original invariant.
4. This is not merely a budget guaranteed to make the planner throw. For the default 1024-token chunk and TP4 width, the largest block is 32 × 32 tiles. A permitted `(n_m,n_n)=(4,4)` split yields 8 × 8-tile chunks, a grid with `core_m=core_n=P=1`, and 16 chunks, within the 256-chunk search limit. With `kb=1`, streaming depths 3, hold depth at most 2, two zone sets, and at most three arrival slots per direction, the planner's footprint is bounded by:

   ```text
   A + W + hold + staging + zero
     <= 3*2048 + 3*1088 + 2*4096 + (2 + 2*(3+3))*2048 + 2048
      = 48320 bytes
   port footprint <= (2 + 2*(3+3) + 2*4)*2048 = 45056 bytes
   ```

   Both fit 65536 bytes. Smaller logical lengths have an available comparable split. The solver may select a faster streamed candidate; this bound establishes an available candidate, not its exact chosen plan. Retained max-context execution uses model chunking, so the bound concerns each fused call, not the total sequence length. Larger configured chunk sizes or different operand types need a new fit bound.

This workaround deliberately trades resident-operand reuse and possibly substantial performance for valid exclusive NoC ownership. It is compatible with the permitted model-file/test/doc scope, and **the current inspected fused call does not already pass this argument**. It must be measured before making any performance claim.

### Simpler scope-compatible alternative

Set the existing `fused_prefill=False` model option for ordinary and watcher runs. `MultichipDecoder.linear()` then uses the existing `minimal_matmul` path; `reduce()` applies `reduce_scatter_minimal_async` and, for replicated residuals, `all_gather_async`. This avoids the defective fused implementation without changing shared kernels or suppressing watcher. Changing `collective`, decode reader/core counts, or inserting host synchronization alone does not redirect the fused reader's hidden NoC policy. The separate path also requires correctness/performance validation; it is an alternative, not a demonstrated performance-equivalent repair.

### Shared-operation repair, outside current authorization

The straightforward upstream correction is to keep W reads on the reader's NoC0 for all plans, extending behavior already used for streamed/thin-wide plans. Retaining split-NoC reads instead requires an explicit disjoint-traffic ownership protocol, per-RISC accounting reconciliation after proven quiescence, and suitable completion checking. Current Metal documentation describes those obligations in `tt_metal/api/tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp:59-82`; the factory's legacy descriptors do not implement that protocol. Dynamic mode would require a coordinated shared-kernel/config audit, not a model compute-config flag.

Do not add a plain NoC1 read barrier to the writer: it compares the same mismatched counts and can wait forever. Do not resnapshot counters while reads may still be outstanding, reset counters to hide the discrepancy, disable watcher assertions, or simply omit the exit check. Those approaches do not restore the required contract.

### Small discriminating checks for the main agent

1. Preserve inspector dispatch/runtime data and both RISCs' counter values on a minimal 1023-token reproduction. Record NoC1 response baseline/delta and outstanding IDs 1-8 at logical (1,1). The accounting diagnosis predicts response growth despite zero outstanding IDs at writer exit; a nonzero ID would require investigating an additional completion bug.
2. Run that case with the proposed `l1_headroom` argument and watcher enabled. Confirm the actual compiled reader has `RESIDENT_A=RESIDENT_W=0`, `A_READ_NOC=W_READ_NOC=0`; check both attention-output and down projections. A completed numerical run without checking the selected plan would leave the proposed mechanism unverified.
3. If it passes, cover both attention kinds, all 11 lengths, state-refresh/decode checks, and chunked maximum context under watcher, then measure latency with the same configuration used in production. If the budget workaround is too slow, compare the separate API path before choosing the model default. Do not claim the resident fused implementation was repaired.

## Uncertainty

- **High confidence:** the compiled split-NoC reader plus the actual exit predicate deterministically violates per-RISC accounting, explains the short/long-plan contrast, and predicts the reported core. The exact first failing projection/program and the numerical counter pair were not retained.
- **Not directly measured:** zero outstanding transactions at the assertion. Per-step TRID waits, bounded look-ahead and the finished-CB dependency explain why unfinished local operand reads are unnecessary and unlikely at writer exit; inspector's missing register data prevents an unconditional hardware-state claim.
- **Source-proven workaround selection, hardware-unverified result:** the 64 KiB argument excludes resident plans and permits a streamed plan for the stated BF16/BFP8 TP4/default-chunk geometry. No hardware correctness, trace, full-length, or latency check of this change was performed here. Live L1 below the required streamed footprint could still cause a planner refusal.
- Cached compilation metadata and offline ELF constants are retained artifacts, not a live dispatch timeline. The original copies should be retained for follow-up, as the standard generated-log paths can be overwritten by another run.
