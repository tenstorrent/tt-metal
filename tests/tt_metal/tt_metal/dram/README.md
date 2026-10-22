# mimir_tests

`mimir_tests` covers Mimir bring-up: host access to DRAM and TL1, firmware and kernel load, and CCE copies to local and remote DRAM and TL1. The gtest prefixes `MM0` to `MM6` number these bring-up stages in order. It is built only when `TT_UMD_BUILD_GRENDEL_JTAG=ON`. The binary is `build/test/tt_metal/mimir_tests`. Each row of the run order is a separate invocation, and the order matters.

## Environment

Set:
- `TT_METAL_EMU_SERVER`
- `TT_METAL_EMU_SOC_DESC`, set to the descriptor for the run (see Descriptors)
- `TT_METAL_SLOW_DISPATCH_MODE=1`

Unset:
- `TT_METAL_WATCHER`
- `TT_METAL_CLEAR_DRAM`
- `TT_METAL_CLEAR_L1`
- `TT_METAL_KERNEL_READBACK_ENABLE`
- `TT_METAL_SKIP_LOADING_FW`
- `TT_METAL_FORCE_REINIT`
- `TT_METAL_SIMULATOR`

Rows 2c and 2d set the variables named in their rows. Every other row leaves them unset.

From row 2b onward, prefix each command with `TT_METAL_OPERATION_TIMEOUT_SECONDS=600 timeout 3600` until a passing 32 KiB local run gives calibrated values. Then set both from that run.

Example:

```
TT_METAL_OPERATION_TIMEOUT_SECONDS=600 timeout 3600 ./build/test/tt_metal/mimir_tests --gtest_filter='<filter>'
```

## Descriptors

Descriptors are in `tt_metal/third_party/umd/tests/soc_descs/`:
- `mimir_1x1.yaml`
- `mimir_2x_package.yaml`

Run the tests once per descriptor:
- `mimir_1x1` with `models/mimir` and `bringup_mimir_1x1.py`.
- `mimir_2x_package` with `models/mmk` and `bringup_two_mimir.py` or `bringup_mmk.py`.

## Run order

| # | Filter | Covers | Descriptors |
|---|---|---|---|
| 1 | `*MimirHost*` | Host DRAM and host TL1 access | all |
| 2a | `*MimirFwPreflight*` | Firmware load preflight | all |
| 2b | `MimirFwLoad.*:*MimirFwLaunch*` | Firmware load and init, image manifest, CCE isolation, no-op launch per CCE | all |
| 2c | `TT_METAL_SKIP_LOADING_FW=1` prefix, filter `*MimirFwLoadNegative*` | Firmware never reports done when loading is skipped. The test zeroes the firmware window in TL1 itself | all |
| 2d | `TT_METAL_FORCE_REINIT=1` prefix, filter `*MimirFwReinit*` | Firmware reset, reload and relaunch on every device open, three cycles | all |
| 3 | `*MimirKernelLoad*:*MimirLocalDram*:*MimirLocalTl1*` | Kernel load, local DRAM copies, same-chiplet TL1 | all |
| 4 | `*MimirRemoteTl1Direct*` | Remote TL1 write/read, host-initiated remote read, size and alignment; direct pairs | 2x only |
| 5 | `*MimirRemoteTl1Bidir*` | Concurrent remote TL1 in both directions, direct pairs | 2x only |
| 6 | `*MimirRemoteTl1Routed*/M0Cce0ToM1Cce0` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 7 | `*MimirRemoteTl1Routed*/M0Cce1ToM1Cce1` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 8 | `*MimirRemoteTl1Routed*/M1Cce0ToM0Cce0` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 9 | `*MimirRemoteTl1Routed*/M1Cce1ToM0Cce1` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 10 | `*MimirRemoteTl1Routed*/M0Cce0ToM1Cce1` | Remote TL1 as in row 4, two-hop pair | 2x only |
| 11 | `*MimirRemoteTl1Routed*/M1Cce1ToM0Cce0` | Remote TL1 as in row 4, two-hop pair | 2x only |
| 12 | `*MimirRemoteDram*/SrcM0Cce1*` | Remote DRAM write, read, round trip and traffic matrix; one-hop source | 2x only |
| 13 | `*MimirRemoteDram*/SrcM1Cce0*` | Remote DRAM write, read, round trip and traffic matrix; one-hop source | 2x only |
| 14 | `*MimirRemoteDram*/SrcM0Cce0*` | Remote DRAM write, read, round trip and traffic matrix; two-hop source | 2x only |
| 15 | `*MimirRemoteDram*/SrcM1Cce1*` | Remote DRAM write, read, round trip and traffic matrix; two-hop source | 2x only |
| 16 | `*MimirRemoteMixed*` | Remote TL1 one-to-many and many-to-one; concurrent local DRAM, remote DRAM and remote TL1 | 2x only |

Notes on the rows:
- Row 1 starts with the identity check. If it fails, every test in row 1 skips.
- Rows 4 to 16 each start with a preflight readback of SS_BAR, ATT and interleave.
- The CCE DMRISC remap is compared with UMD's values once per process, right after the first CreateDevice. CreateDevice is where UMD releases each CCE uncore and programs the remap before it loads firmware, so the comparison runs after it.
- Rows 4 to 16 are 2x only. Each `TEST_P` also skips at runtime from the descriptor's topology.
- Rows 6 to 15 run one route each, so a hang in one route does not lose the others.
- Row 16 runs last because it includes routed destinations.

## MM-7 Tensor Prefetcher

Rows 17 onward cover the Tensor Prefetcher on CCEs. They use the same environment and timeout prefix as rows 2b onward. Run them after a fresh session and bring-up (see Fresh-session rule), one test per invocation, in this order.

| # | Filter | Covers | Descriptors |
|---|---|---|---|
| 17 | `MimirPrefetcher.MM7GatePreconditionsHold` | Prefetcher preconditions hold and the port reports supported; one sender per DRAM view | all |
| 18 | `MimirPrefetcher.MM7GateUninitializedSenderFails` | A sender CCE whose GO_MSG reads INIT fails the preconditions, naming that CCE | all |
| 19 | `MimirPrefetcher.MM7GateStartStopWorks` | Start and Stop with no requests | all |
| 20 | `MimirPrefetcher.MM7GcbSmokePreconditionsWithReceiverRunning` | Preconditions stay ok while a receiver Program runs on a sender CCE | all |
| 21 | `MimirPrefetcher.MM7GcbLayoutLocalReceiver` | DRAM-receiver GCB state block, config page and credit lines; same-Mimir receiver | all |
| 22 | `MimirPrefetcher.MM7GcbLayoutRemoteReceiver` | The same with a receiver on the other Mimir, receiver order kept | 2x only |
| 23 | `MimirPrefetcher.MM7GcbLayoutTwoSenders` | Two senders, one receiver each; a second such GCB takes disjoint ranges | 2x only |
| 24 | `MimirPrefetcher.MM7GcbLayoutRejectsConflictingReceivers` | Rejects aliased, sender-as-receiver and duplicate receivers | all |
| 25 | `MimirPrefetcher.MM7GcbSmokeDiscardDrain` | Test sender into a discarding receiver, without the manager; both ends' counters agree | all |
| 26 | `MimirPrefetcher.MM7GcbSmokeValidatorLocal` | Test sender into a validating receiver on the same Mimir | all |
| 27 | `MimirPrefetcher.MM7GcbSmokeValidatorWrongSeed` | The validator reports a mismatch for a wrong seed | all |
| 28 | `MimirPrefetcher.MM7GcbSmokeResize` | Three page sizes in one GCB, with resize and wrap padding | all |
| 29 | `MimirPrefetcher.MM7GcbSmokeValidatorD2d` | Validating receiver on the other Mimir | 2x only |
| 30 | `MimirPrefetcher.MM7GcbSmokeDrainUntilStopped` | Receiver drains until the host stops it | all |
| 31 | `MimirPrefetcher.MM7GcbSmokeStallUntilReleased` | The receiver holds a full FIFO, so the sender gives CREDIT_TIMEOUT; then the host releases it | all |
| 32 | `MimirPrefetcher.MM7GcbSmokeDrainTimeout` | The receiver holds every page, so the sender's final barrier gives DRAIN_TIMEOUT | all |
| 33 | `MimirPrefetcher.MM7GcbSmokeReceiverWaitTimeout` | No sender: the receiver drains host-credited pages, then its wait gives CREDIT_TIMEOUT | all |
| 34 | `MimirPrefetcher.MM7GcbSmokeRelaunchResumes` | A second sender and validator pair resumes from the pointers the first left in TL1 | all |
| 35 | `MimirPrefetcher.MM7LifecycleStartStop` | Prefetcher Start and Stop; the sender reports Stopped | all |
| 36 | `MimirPrefetcher.MM7LifecycleStartStopTwice` | A second Start zeroes the control block the first session left; an unknown command left in ring slot 0 is never read | all |
| 37 | `MimirPrefetcher.MM7LifecycleBusySenderFailsStart` | Start fails on a sender CCE that runs a receiver Program, before it launches or zeroes anything | all |
| 38 | `MimirPrefetcher.MM7LifecycleQueueRejectsReceiverOnSender` | Queue rejects a GCB whose receiver is a sender CCE | all |
| 39 | `MimirPrefetcher.MM7LifecycleWaitCq` | A WAIT_CQ holds the sender until the command queue writes its slot | all |
| 40 | `MimirPrefetcher.MM7LifecycleDrainAtStop` | Stop's DRAIN holds the sender until an unacked credit is acked, then STOP | all |
| 41 | `MimirPrefetcher.MM7T1Local0To1` | One prefetch, CCE 0 to CCE 1, then Start, Queue and Stop again | all |
| 42 | `MimirPrefetcher.MM7T1Local3To2` | One prefetch, CCE 3 to CCE 2, on the second Mimir | 2x only |
| 43 | `MimirPrefetcher.MM7T1D2d0To2` | One prefetch, CCE 0 to CCE 2, one D2D hop | 2x only |
| 44 | `MimirPrefetcher.MM7T1D2d3To1` | One prefetch, CCE 3 to CCE 1, one D2D hop | 2x only |
| 45 | `MimirPrefetcher.MM7T2FanOut0To123` | Fan-out from CCE 0 to CCEs 1, 2 and 3 | 2x only |
| 46 | `MimirPrefetcher.MM7T2FanOut3To012` | Fan-out from CCE 3 to CCEs 0, 1 and 2 | 2x only |
| 47 | `MimirPrefetcher.MM7T2TwoSenders01To2` | Two senders, CCEs 0 and 1, into two GCBs on CCE 2 | 2x only |
| 48 | `MimirPrefetcher.MM7T2TwoSenders02To3` | Two senders, CCEs 0 and 2, into two GCBs on CCE 3 | 2x only |
| 49 | `MimirPrefetcher.MM7T3SubChunkLocal0To1` | A block larger than a stage third, in sub-chunks through two stage slots | all |
| 50 | `MimirPrefetcher.MM7T3SubChunkD2d0To2` | The same across D2D | 2x only |
| 51 | `MimirPrefetcher.MM7T3BelowStageThirdLocal0To1` | Blocks smaller than a stage third, two per stage slot | all |
| 52 | `MimirPrefetcher.MM7T3BelowStageThirdD2d0To2` | The same across D2D | 2x only |
| 53 | `MimirPrefetcher.MM7T3PartialStageChunkLocal0To1` | A round that ends in a partial stage chunk | all |
| 54 | `MimirPrefetcher.MM7T3AboveStageThirdLocal0To1` | A block above the default stage third, in one-tile sub-chunks | all |
| 55 | `MimirPrefetcher.MM7T3AboveStageThirdD2d0To2` | The same across D2D | 2x only |
| 56 | `MimirPrefetcher.MM7T3PageSplitLocal0To1` | One Queue call split into two request pages; local only, since the split is on the host | all |
| 57 | `MimirPrefetcher.MM7T3AbaLocal0To1` | Tensors A, B, A with different page sizes; resize and wrap | all |
| 58 | `MimirPrefetcher.MM7T3AbaD2d0To2` | The same across D2D | 2x only |
| 59 | `MimirPrefetcher.MM7T3WrapAndResizeLocal0To1` | Resize padding and wraps in a FIFO the page size does not divide | all |
| 60 | `MimirPrefetcher.MM7T3WrapAndResizeD2d0To2` | The same across D2D | 2x only |
| 61 | `MimirPrefetcher.MM7T3GcbSwitchLocal0To1` | GCB 1, then 2, then 1, resuming mid-FIFO | all |
| 62 | `MimirPrefetcher.MM7T3GcbSwitchD2d0To2` | The same across D2D | 2x only |
| 62a | `MimirPrefetcher.MM7T3ReceiverSwitch0To1Then2` | GCB 1 to CCE 1, then GCB 2 to CCE 2 across D2D, then GCB 1 again | 2x only |
| 63 | `MimirPrefetcher.MM7T3FenceLocal0To1` | A WAIT_CQ fence holds delivery until the slot is written | all |
| 64 | `MimirPrefetcher.MM7T5TraceCaptureReplay` | Skipped: trace capture needs fast dispatch | all |
| 65 | `MimirPrefetcher.MM7RejectKRowMajorTensor` | Queue rejects a K-row-major tensor | all |
| 66 | `MimirPrefetcher.MM7RejectStreamingTensor` | Queue rejects a streaming tensor | all |
| 67 | `MimirPrefetcher.MM7RejectRangePastTensor` | Queue rejects a request past the tensor | all |
| 68 | `MimirPrefetcher.MM7RejectRangePastView` | Queue rejects a request past the DRAM view | all |
| 69 | `MimirPrefetcher.MM7T4MismatchD2d2To0` | Wrong-seed validator reports MISMATCH on every word of the tensor's page; the sender stops OK | 2x only |
| 70 | `MimirPrefetcher.MM7T4CreditTimeoutD2d1To3` | A request larger than the FIFO gives CREDIT_TIMEOUT; the host drops the requests after it and Stop throws, naming the sender | 2x only |
| 71 | `MimirPrefetcher.MM7T4DrainTimeoutD2d1To3` | A receiver holds a request, so Stop's DRAIN gives DRAIN_TIMEOUT and Stop throws | 2x only |

Notes on the rows:
- Rows that need a second Mimir skip at runtime on `mimir_1x1`.
- Row 64 always skips: trace capture needs fast dispatch.
- Every row that checks a receiver's FIFO state also checks the 256 B after the FIFO end against a pattern written before the run, so a write past the FIFO end fails the row.
- The T4 rows run last: each ends with a sender or receiver timeout.
- For a JUnit report, add `--gtest_output=xml:<file>` to each command.

## MM-8 DM-suite dispositions

Every source in the existing DM suite gets exactly one disposition, picked by the first value in the rule below that fits. The MM8 ports are `TEST_P` suites instantiated as `MM8`, so their filters are written `*MimirDm<Name>*`. The 44 cases of the first port all passed on `mimir_2x_package` on 2026-10-21 (soc-c-26) and on `mimir_1x1` on 2026-10-21. On 2026-10-20 (soc-zebu-03), `MimirDmIdma.Copy/RemoteTl1` and `Copy/RemoteGddr` failed on `mimir_2x_package` with `ack_timeout`: the data never landed, on the write and the read legs, and the cause is not known. The MM-8 follow-up then changed every MM8 suite and added seven cases (51 in all). On `mimir_2x_package` all 51 and the MM3 `MimirKernelLoad` cases passed on 2026-10-22 (soc-zebu-04, one fresh session). An earlier session on 2026-10-21 (soc-zebu-01) passed 23 cases, then `Copy/OwnTl1` failed (acked never came, the data never landed) and `DramUnary.ReaderWriterHandshake/Cce2` hit the host deadline; both passed on soc-zebu-04, so they are recorded as that session's failures, cause not known. The follow-up has not run on `mimir_1x1`.

| Value | Rule |
|---|---|
| `B-port` | MM8 builds a Mimir analogue |
| `covered` | The upstream test cannot run, but an existing MM3-7 test exercises its intent on that descriptor |
| `B-blocked` | An analogue is possible but needs a named capability or runtime change, or another stage (4.x) |
| `C` | Mimir has no analogue at all: no multicast, no NoC atomics, no PCIe, Eth, compute, L1 banks or worker cores |
| `out-of-scope` | Not part of MM-8 |

Source paths are relative to `tests/tt_metal/tt_metal/`. A `gap` in a covered-by column means no existing test covers that descriptor. A `—` means the row has no evidence column. Covered-by filters are glob patterns that allow for the instantiation prefix and parameter suffix. A 2x-only test is never listed under 1x1.

**`data_movement/sources.cmake` (38 tests plus `dm_common.cpp`)**

| Source | Disposition | Covered by: 2x | Covered by: 1x1 | Reason / prerequisite | Case overrides |
|---|---|---|---|---|---|
| `data_movement/dram_unary/test_unary_dram.cpp` | B-port | `*MimirDmDramUnary*`; interim: `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*`, `*MimirLocalDram.LoopbackChecksIntermediate*` | `*MimirDmDramUnary*`; interim: `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*`, `*MimirLocalDram.LoopbackChecksIntermediate*` | Two harts of one CCE, SPA data path, DRAM semaphore (D2) | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/one_to_all/test_one_to_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.MM7T2FanOut0To123`, `MimirPrefetcher.MM7T2FanOut3To012` | gap | Unicast cases only; 1x1 gap: one_to_all unicast fan-out (D4) | Multicast* and *Semaphore* cases: C (no multicast, no NoC semaphore atomics); `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/one_to_all/test_multicast_schemes.cpp` | C | — | — | No multicast on Mimir | — |
| `data_movement/loopback/test_loopback.cpp` | covered | `*MimirLocalTl1.SameChipletControl*` | `*MimirLocalTl1.SameChipletControl*` | No CCE NoC; same-chiplet TL1 copy is the intent | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/reshard_hardcoded/test_reshard_hardcoded.cpp` | C | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/deinterleave_hardcoded/test_deinterleave_hardcoded.cpp` | C | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/conv_hardcoded/test_conv_hardcoded.cpp` | C | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/one_to_one/test_one_to_one.cpp` | covered | `*MimirLocalTl1*`, `*MimirRemoteTl1Direct.WriteRead*`, `*MimirRemoteTl1Routed.WriteRead*`, `MimirPrefetcher.MM7T1D2d0To2`, `MimirPrefetcher.MM7T1D2d3To1`, `MimirPrefetcher.MM7T1Local3To2` | `*MimirLocalTl1*`, `MimirPrefetcher.MM7T1Local0To1` | No CCE NoC; TL1-to-TL1 copy and prefetch cover the intent | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/one_from_one/test_one_from_one.cpp` | covered | `*MimirLocalTl1*`, `*MimirRemoteTl1Direct.HostInitRemoteRead*`, `*MimirRemoteTl1Routed.HostInitRemoteRead*` | `*MimirLocalTl1*`; gap: host-initiated remote read | 1x1 gap: host-initiated remote read (D4) | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/core_bidirectional/test_core_bidirectional.cpp` | covered | `*MimirRemoteTl1Bidir*.Concurrent*` | gap | PacketSizes only (7 of 9 TEST_F cases skip upstream); 1x1 gap: same-chiplet instances are separate launches, not concurrent (D4) | — |
| `data_movement/one_from_all/test_one_from_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.MM7T2TwoSenders01To2`, `MimirPrefetcher.MM7T2TwoSenders02To3` | gap | 1x1 gap: fan-in/all-to-all on mimir_1x1 (D4) | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/all_to_all/test_all_to_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.MM7T2TwoSenders01To2`, `MimirPrefetcher.MM7T2TwoSenders02To3` | gap | 1x1 gap: fan-in/all-to-all on mimir_1x1 (D4) | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/all_from_all/test_all_from_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.MM7T2TwoSenders01To2`, `MimirPrefetcher.MM7T2TwoSenders02To3` | gap | 1x1 gap: fan-in/all-to-all on mimir_1x1 (D4) | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/interleaved/test_interleaved.cpp` | B-blocked | — | — | Interleaved bank addressing from a CCE (int64 bank offsets, D3) | Sync cases: B-blocked (CBs); L1 cases: C (no L1 banks) |
| `data_movement/one_packet/test_one_packet.cpp` | covered | `*MimirRemoteTl1Direct.SizeAlignment*` | gap | SizeAlignment runs with read_back=false; 1x1 gap: sub-word and odd-size TL1 transfers (D4); iDMA max packet size is in iDMA probe case 4 | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/interleaved_to_sharded_hardcoded/test_interleaved_to_sharded_hardcoded.cpp` | C | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/multi_interleaved/test_multi_interleaved.cpp` | B-blocked | — | — | Interleaved bank addressing from a CCE (D3); CBs | — |
| `data_movement/dram_sharded/test_dram_sharded.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirRemoteDram.ReadToTl1*`, `MimirPrefetcher.MM7T1Local0To1`, `MimirPrefetcher.MM7T1Local3To2`, `MimirPrefetcher.MM7T1D2d0To2`, `MimirPrefetcher.MM7T1D2d3To1` | `*MimirLocalDram.CceReadsDramToTl1*`, `MimirPrefetcher.MM7T1Local0To1` | Intent: read DRAM shards into TL1 | *Trid* cases: B-blocked on 4.x MM43; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/transaction_id/test_transaction_id.cpp` | B-blocked | — | — | 4.x MM43 (iDMA command-buffer trids) | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/direct_write/test_direct_write.cpp` | covered | `*MimirRemoteTl1Direct.SizeAlignment*` | gap | 4 B case of SizeAlignment; upstream tests inline dword writes, which the CCE does not have, so the 4 B CPU store is a stand-in; 1x1 gap: sub-word direct write (D4) | Mcast cases: C |
| `data_movement/pcie_read_bw/test_pcie_read_bw.cpp` | C | — | — | No PCIe | — |
| `data_movement/pcie_write_bw/test_pcie_write_bw.cpp` | C | — | — | No PCIe | — |
| `data_movement/device_pcie_loopback/test_device_pcie_loopback.cpp` | C | — | — | No PCIe | — |
| `data_movement/pcie_mid_cleanup/test_pcie_mid_cleanup.cpp` | C | — | — | No PCIe | — |
| `data_movement/atomics/test_atomic_semaphore_bandwidth.cpp` | B-blocked | — | — | 4.x MM42 (local AMO, iDMA accumulate); the NoC-atomic form has no ITN equivalent | — |
| `data_movement/multicast_atomics/test_multicast_atomic_semaphore.cpp` | C | — | — | No multicast, no NoC atomics | — |
| `data_movement/noc_api_latency/test_noc_api_latency.cpp` | C | — | — | NoC API latency; no CCE NoC | — |
| `data_movement/noc_estimator_tests/test_noc_estimator.cpp` | C | — | — | NoC estimator; no CCE NoC | — |
| `data_movement/dram_neighbour/test_dram_neighbour.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirRemoteDram.ReadToTl1*` | `*MimirLocalDram.CceReadsDramToTl1*` | Intent: read the near DRAM bank | — |
| `data_movement/matmul/test_matmul_1d.cpp` | C | — | — | DM multicast and NoC semaphores; no CCE NoC | — |
| `data_movement/matmul/test_matmul_1d_v2.cpp` | C | — | — | DM multicast and NoC semaphores; no CCE NoC | — |
| `data_movement/matmul/test_matmul_2d.cpp` | C | — | — | DM multicast and NoC semaphores; no CCE NoC | — |
| `data_movement/quasar_cache/test_quasar_cache.cpp` | B-port | `*MimirDmCache*` | `*MimirDmCache*` | cce_cache_ops on TL1; cache probes done. `QuasarL1DCacheOps.InvalidateFull` is `L1InvalidateFull`; `L2InvalidateFreshRead` is the L2-only fresh read, checked by a cached hart read where upstream reads back on the host | — |
| `data_movement/quasar_cache_perf/test_quasar_cache_perf.cpp` | B-port | `*MimirDmCachePerf*` | `*MimirDmCachePerf*` | Functional only, nightly; no timing asserts; cache probes done | — |
| `data_movement/quasar_examples/quasar_idma/test_idma_example.cpp` | B-port | `*MimirDmIdma*` | `*MimirDmIdma*` | cce_idma_copy; iDMA probes done | IDMA_1D_Strided: B-port as `Strided1D` (own TL1) and `Strided1DLocalGddr` (local GDDR source), with 64 B elements, and as `Strided1DSubLine` and `Strided1DSubLineLocalGddr` with upstream's 8 B elements and 16 B source stride; remote strided endpoints not ported |
| `data_movement/quasar_examples/quasar_addrgen/test_addrgen_example.cpp` | B-port | `*MimirDmAddrgen*` | `*MimirDmAddrgen*` | Pop addresses into TL1 and compare on host; `addrgen_reset` probe done; explicit-count pops (`pop_*_addrgen<ADDRGEN_0>(1)`) work around tt-metal #59983 (the count-less pop encodes rs1 = x1); the banking registers stay at reset, as upstream: a one-bank banking loop from a probe run that also changed the pops was removed after every addrgen, im2col and strided iDMA case passed without it on 2026-10-22 | Interleaved_Banking: C (encodes ATT endpoint IDs 20..23) |
| `data_movement/quasar_examples/quasar_im2col/test_im2col_example.cpp` | B-port | `*MimirDmIm2col*` | `*MimirDmIm2col*` | Pop addresses into TL1 and compare on host; `addrgen_reset` probe done; the loop index resets to 0 at loop end, as the RTL notes describe (neosim overlay-addrgen-rtl-semantics.md; tensix-ip_docs overlay/rocc/address_generator.md); `addrgen_api.hpp:38-39` says modulo and is out of date | — |
| `data_movement/quasar_narrow_row/test_pack_untilize_narrow_row.cpp` | C | — | — | Compute; no Tensix | — |
| `data_movement/dm_common.cpp` | out-of-scope | — | — | Helper, not a test | — |

**Plan-named**

| Source | Disposition | Covered by: 2x | Covered by: 1x1 | Reason / prerequisite | Case overrides |
|---|---|---|---|---|---|
| `api/test_direct.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | ReaderOnly and WriterOnly only | ReaderWriter: B-blocked (DFB); Datacopy cases: C; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `api/test_runtime_args.cpp` | B-port | `*MimirKernelLoad.ArgsReplaceStale*`, `*MimirDmRuntimeArgs*` | `*MimirKernelLoad.ArgsReplaceStale*`, `*MimirDmRuntimeArgs*` | TensixIllegalTooManyRuntimeArgs and TensixIllegallyModifyRTArgs ported as MM3 analogues; ArgsReplaceStale keeps the arg count, so it does not cover the second | Legacy RTA/CRTA: covered by MM3 (ArgsReplaceStale, cce_args_echo); Quasar CRTA: B-blocked (KernelSpec is TENSIX-only); Compute and Eth cases: C; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_alias_dataflow_buffer.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C |
| `api/dataflow_buffer/test_borrowed_memory_dataflow_buffer.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C |
| `api/dataflow_buffer/test_dataflow_buffer_apis.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C |
| `api/dataflow_buffer/test_dataflow_buffer_base.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_configs.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_disjoint_slots.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C |
| `api/dataflow_buffer/test_dataflow_buffer_edge_cases.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_intra.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_multinode.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_overrides.cpp` | B-blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: C |
| `api/dataflow_buffer/dfb_init_timing_bench.cpp` | out-of-scope | — | — | Separate executable (`api/CMakeLists.txt:107`) | — |
| `api/test_scoped_lock_cache.cpp` | B-port | `*MimirDmScopedLockClm*` | `*MimirDmScopedLockClm*` | CLM cases (`:540-555`) on two harts of one CCE; cache probes done | DFB and handshake cases: B-blocked; Scratchpad and LTA cases: C (Tensix-DM abstractions) |
| `test_quasar_trace.cpp` | B-blocked | — | — | Trace needs fast dispatch; same as `MM7T5TraceCaptureReplay`, a bare skip | — |
| `test_quasar_mesh_workloads.cpp` | B-blocked | — | — | KernelSpec is TENSIX-only (`program_spec.cpp:4047`); every workload pairs DM kernels with a compute kernel | Compute-output checks: C; TestInterleavedWorkloadsAcrossTwoCQs also needs a second CQ |
| `test_quasar_compute_kernels.cpp` | C | — | — | Compute; no Tensix | — |
| `llk/test_quasar_mailboxes.cpp` | C | — | — | Compute/Tensix mailboxes | — |
| `api/metal2_host_api/unit_tests/*` | out-of-scope | — | — | Metal 2.0 mock CPU tests; with the two rows below they replace the plan's test_program_spec / test_program_run_args | — |
| `api/metal2_host_api/integration_tests/{binding_loopbacks,kernel_args_loopbacks,scratchpad,mesh_workload_factories}.cpp` | B-blocked | — | — | Device tests built on KernelSpec, which is TENSIX-only (`program_spec.cpp:4047`) | — |
| `api/metal2_host_api/integration_tests/scratchpad_fast_dispatch.cpp` | B-blocked | — | — | Fast dispatch, and KernelSpec is TENSIX-only | — |
| `api/metal2_host_api/integration_tests/{compute_semaphore,llk_operand_mul}.cpp` | C | — | — | Compute; no Tensix | — |
| `api/metal2_host_api/kernel_compilation_tests/*` | out-of-scope | — | — | Kernel compilation against a mock device; nothing runs on a device | — |

**Extra**

| Source | Disposition | Covered by: 2x | Covered by: 1x1 | Reason / prerequisite | Case overrides |
|---|---|---|---|---|---|
| `test_dram_loopback_single_core.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | Tensix loopback; Mimir DRAM-to-TL1-to-DRAM copies cover the intent | Eth cases: C |
| `api/test_dram.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | Tensix loopback x2 and double-buffered | Eth cases: C |
| `test_single_dm_l1_write.cpp` | covered | `MM3/MimirKernelLoad.*`, `MimirKernelLoad.SameBinaryAllCcesUnicast` | `MM3/MimirKernelLoad.*`, `MimirKernelLoad.SameBinaryAllCcesUnicast` | SingleDmL1Write by the MM3 cases | GridProbeStep0 and FullGridDmL1Write_L1a: covered by `MimirKernelLoad.SameBinaryAllCcesUnicast` (every CCE); FullGridCompute_L1c and GridMulticastFanOut: C |
| `test_quasar_mesh_buffers.cpp` | B-port | `*MimirDmMeshBufferDram*` | `*MimirDmMeshBufferDram*` | MeshBufferWriteReadDRAM and MeshBufferMultipleWriteReadRoundsDRAM (`:67,88`); host DRAM MeshBuffer placement probe done (D3) | L1 and sharded L1: C; MeshBufferCrossCQWriteReadRoundsDRAM: B-blocked (multi-CQ) |
| `test_quasar_semaphores.cpp` | C | — | — | NoC semaphore increments and NoC reads; no NoC semaphore atomic on Mimir | — |
| `test_quasar_events.cpp` | B-blocked | — | — | KernelSpec is TENSIX-only (`program_spec.cpp:4047`); single-CQ events work in slow dispatch | CrossCQEventHandoff* and RecordEventToHostFromBothCQs: also need a second CQ |
| `test_quasar_fds.cpp` | C | — | — | No Tensix FDS | — |
| `test_quasar_sub_devices.cpp` | C | — | — | No worker cores | — |
| `llk/test_quasar_bfd_datacopy.cpp` | C | — | — | Compute; no Tensix | — |
| `noc/test_quasar_att_address.cpp` | out-of-scope | — | — | Host CPU test | — |
| `noc/test_quasar_att_config.cpp` | out-of-scope | — | — | Host CPU test | — |
| `noc/test_quasar_att_image.cpp` | out-of-scope | — | — | Host CPU test | — |
| `api/test_dram_sender_global_cb.cpp` | C | — | — | Receivers are worker cores; `build_dram_sender_mapping` throws on Quasar (`dram_sender_topology.cpp:36-40`); Mimir DRAM-receiver GCB is covered by MM7 Gcb* and T1-T4 | — |
| `api/test_prefetcher_pipe.cpp` | C | — | — | Metal 2.0 KernelSpec worker kernels; no worker cores | — |
| `api/test_prefetcher_pipe_dram_sender.cpp` | C | — | — | Same throw as above; the CCE sender returns BadCmd for a PrefetcherPipe target (`tensor_prefetcher_cce.cpp:74-75` for DRAIN, `:242-244` for PREFETCH) | — |

### Probe results

Each probe was run on the `mimir_2x_package` emulator (soc-c-30 and soc-zebu-03). Columns: probe, result (yes/no/hang), date, commit and run log (paths are relative to the session scratchpad). The probes ran on 2x only; the ports that replaced them passed on `mimir_1x1` on 2026-10-21.

| Probe | Result | Date | Run log and commit | Rows that depend on it |
|---|---|---|---|---|
| Cache host-visibility: are cached TL1 stores visible to the host without a flush? | no (host saw 0 of 64 cached stores without a flush; after a flush it sees them) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_probes/mimir_2x_package_HostSeesCachedStores.log` | quasar_cache, quasar_cache_perf, test_scoped_lock_cache |
| `tt.cache.*` trap: do the instructions trap (a trap is a hang)? | no (no trap; all four forms returned) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_probes/mimir_2x_package_CacheInstrTrap.log` | quasar_cache, quasar_cache_perf, test_scoped_lock_cache |
| iDMA case 1: own TL1 to own TL1 (VC 5), package-global SPA operands | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_OwnTl1.log` | quasar_idma |
| iDMA case 2: NoC-mode command-buffer request (`idma_en=0`) | no (never showed free space, nothing issued, no hang); Branch B stands, no CCE NoC | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_NocCmdbuf.log` | quasar_idma and every `covered` NoC movement row |
| iDMA case 3a: sibling TL1 (write VC 2, read VC 5) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_SiblingTl1.log` | quasar_idma |
| iDMA case 3b: local GDDR0 (write VC 0, read VC 5) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_LocalGddr.log` | quasar_idma |
| iDMA case 3c: remote TL1, kernel loading the remote destination | hang (no status in 400 s; session wedged) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_RemoteTl1.log` | quasar_idma |
| iDMA case 3c rerun: remote TL1, host-verified (write VC 3, read VC 5) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_RemoteTl1_run2.log` | quasar_idma |
| iDMA case 3d: remote GDDR, host-verified | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_RemoteGddr.log` | quasar_idma |
| iDMA case 4: maximum packet size (8 KiB) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_idma/mimir_2x_package_MaxPacket.log` | quasar_idma, one_packet |
| `addrgen_reset` hang | no hang (returns); pops advanced once both the explicit count and the one-bank loop were in place (one run changed both) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_probes/mimir_2x_package_AddrgenReset.log, ..._AddrgenDiag.log, ..._AddrgenReset_fix.log` | quasar_addrgen, quasar_im2col |
| Host DRAM MeshBuffer placement | yes (banks 0-3 at the expected address in their own view; round trip matched) | 2026-10-20 | `3ef0a46e42a` + uncommitted MM-8 tree; `scratchpad mm8_probes/mimir_2x_package_InterleavedPlacement.log` | test_quasar_mesh_buffers |

### Hand checks

The table is kept by hand, with no checker script. These three checks are redone whenever the table changes:
1. Every `data_movement/sources.cmake` source has exactly one row.
2. Every covered-by filter that is not `gap` or `—` matches at least one test in `mimir_tests --gtest_list_tests`. A 2x-only test also lists on 1x1, so the 1x1 column names only tests that run there.
3. Every `MM8/` test listed by `--gtest_list_tests --gtest_filter='MM8/*'` is matched by some covered-by filter and by some run row.

## MM-8 DM regression

These rows are the plan's `mm_data_movement_regression`. They continue from row 71, with the same environment and timeout prefix as rows 2b onward. Each case runs as its own invocation (list them with `--gtest_list_tests --gtest_filter='MM8/*'`). Add `--gtest_output=xml:<dir>/<descriptor>_<row>.xml` to each command; it is the Gate 6 evidence, read against the covered-by columns above. On `mimir_1x1` (2026-10-21), 80 cases ran (the 44 `MM8/*` cases of the first port and the 36 MM-7 rows marked "all"): 75 passed, 5 skipped, 0 failed. The skips are `MimirDmIdma.Copy/RemoteTl1` and `Copy/RemoteGddr` (need two Mimirs), `MimirDmDramUnary.ReaderWriterHandshake/Cce2` and `Cce3` (1x1 has 2 CCEs), and `MM7T5TraceCaptureReplay` (always skips). The follow-up's 51 `MM8/*` cases have not run on `mimir_1x1`. Rerun MM3 KernelLoad (row 3) with them: the follow-up changed its echo kernel.

CI: `mimir_tests` is built only with `TT_UMD_BUILD_GRENDEL_JTAG=ON`, which is off by default, and no CI runner has a Mimir emulator. So MM-8-T5 CI placement is not done, and these rows are run by hand.

These follow-up cases test a contract upstream assumes; if one fails on a correct emulator, record it as a finding, not a test bug: `MimirDmCache.Op/HartToHartCoherent` (a consumer's cached copy sees the producer's flushed stores without an invalidate), `MimirDmCachePerf.WritePath/CachedFastFlush` (no fence before the FLUSH64 writes), `MimirDmScopedLockClm.CacheOps/FlushOnRelease` (the release flush is in TL1 at the first read) and `MimirDmIdma.Copy/RemoteTl1KernelVerified` (the kernel loads a remote TL1 destination). All four passed on `mimir_2x_package` on 2026-10-22.

New rule: after any row that ends on a host deadline, start a fresh emulator session and run bring-up before the next row (MIM-601).

| # | Filter | Covers | Descriptors |
|---|---|---|---|
| 72 | `MM8/MimirDmRuntimeArgs.*` | Too many runtime args and illegal modification are rejected, the stored args are unchanged after a rejection, and a kernel with args at the ceiling launches | all |
| 73 | `MM8/MimirDmMeshBufferDram.*` | Host DRAM MeshBuffer: interleaved placement per bank, write/read with a per-run pattern, repeated rounds | all |
| 74 | `MM8/MimirDmIdma.*` | iDMA copies: own TL1, sibling TL1, local GDDR0 and GDDR1, max packet, strided gather of 64 B and of 8 B elements from own TL1 and from local GDDR; remote TL1 (host-verified and kernel-verified) and remote GDDR on 2x only; guards before and after each destination. RemoteTl1 and RemoteGddr failed with `ack_timeout` on 2026-10-20 (soc-zebu-03) and passed on 2026-10-21 (soc-c-26) and 2026-10-22 (soc-zebu-04); OwnTl1 failed the same way on 2026-10-21 (soc-zebu-01) and passed on 2026-10-22; the cause is not known. RemoteTl1KernelVerified hung with no status on its first run (MM-8 probe, 2026-10-20, soc-zebu-03) and passed six times on 2026-10-22, all in one soc-zebu-04 session; it was disabled until then | all (Remote* 2x only) |
| 75 | `MM8/MimirDmCache.*` | Cache ops on TL1: flush, invalidate (line, whole L1 D$, L2 line), fresh read (L1 and L2, L2 only), coherent hart-to-hart visibility; every store case checks the hart's cached view of its stores. Run `Op/L1InvalidateFull` last: the whole-L1 D$ discard can drop the kernel's own stack | all |
| 76 | `MM8/MimirDmCachePerf.*` | Write paths, functional only; CachedFastFlush has no fence before the FLUSH64 writes, as upstream | all |
| 77 | `MM8/MimirDmAddrgen.*` | Addrgen strided 1D, 2D and face patterns, popped into TL1 and compared on the host, with the banking registers at reset | all |
| 78 | `MM8/MimirDmIm2col.*` | Im2col full image and dilation 1, compared on the host | all |
| 79 | `MM8/MimirDmDramUnary.*` | Reader and writer harts on one CCE, GDDR to TL1 to GDDR with a DRAM semaphore; the semaphores are at the runtime's addresses, initialised at launch and end at the chunk count; guards before and after the destination and the staging ring | all |
| 80 | `MM8/MimirDmScopedLockClm.*` | Scoped-lock CLM cases: invalidate on acquire, flush on release (in TL1 at the first read), locked range only; unlocked lines stay cached in both invalidate cases | all |

## Fresh-session rule

Before a full run, start a fresh emulator session and run chippy bring-up.

Run the host suite (row 1) right after bring-up. The host suites skip when any CCE hart is out of reset. A process that opened a device resets the CCE harts on clean exit and leaves the uncore released, as bring-up does, so row 1 also runs after such a process. After a killed process the harts may still be running, and row 1 skips.

If a run is killed mid-stall, the session is probably wedged. Use the same rule: a fresh session plus bring-up.

## Rules

- Never set `TT_METAL_CLEAR_DRAM`. It writes every view in full, 1 GiB per Mimir, which is far too slow on the emulator.

## Manual D2D AXI monitor step

The emulator's D2D AXI monitor is a manual debug step. It is a DPI bind (`models/mimir/emu_hdl/axi_mon_d2d_bind.sv`, `top.utf:22`). To use it, relaunch the SiVal server without `--disable-dpi`, which the server command in the repo-root `mimir_setup.md` passes. With DPI on, the SMC AXI monitor also logs every host beat.
