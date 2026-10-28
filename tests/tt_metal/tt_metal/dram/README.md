# mimir_tests

`mimir_tests` covers Mimir bring-up: host access to DRAM and TL1, firmware and kernel load, and CCE copies to local and remote DRAM and TL1. The gtest prefixes `MM0` to `MM6` number these bring-up stages in order. It is built only when `TT_UMD_BUILD_GRENDEL_JTAG=ON`. The binary is `build/test/tt_metal/mimir_tests`. Each row of the run order is a separate invocation, and the order matters.

## Environment

Every case starts with `skip_unless_programmable_dram_package()` (`tests/tt_metal/tt_metal/context/mimir_contract.hpp`) and runs when it passes: the selected package declares programmable DRAM cores and a transport is configured. `TT_METAL_EMU_SERVER` is not required.

Select the package, set to the descriptor for the run (see Descriptors), with one of:
- a chippy server (`TT_METAL_EMU_SERVER` or `TT_METAL_GRENDEL_JTAG_SERVER`) and `TT_METAL_EMU_SOC_DESC` or `TT_METAL_GRENDEL_JTAG_SOC_DESC` to match
- `TT_METAL_SOC_DESC`, with no server

Set:
- `TT_METAL_SLOW_DISPATCH_MODE=1`. A Mimir-only package has no Tensix cores for fast dispatch, so every case skips without it.

Cases that read Mimir-local addresses (`Cluster` TL1 and register reads, D2D state, the remote preflight, Mimir-local address records) also call `skip_unless_mimir_local_package()` and skip on a `full_package` descriptor.

Unset:
- `TT_METAL_WATCHER`
- `TT_METAL_CLEAR_DRAM`
- `TT_METAL_CLEAR_L1`
- `TT_METAL_KERNEL_READBACK_ENABLE`
- `TT_METAL_SKIP_LOADING_FW`
- `TT_METAL_FORCE_REINIT`
- `TT_METAL_SIMULATOR`
- `TT_METAL_QUASAR_VARIANT` (the L2 cache-controller addresses the MM4x kernels use are those of the unset or 2.0.2 variant)

Rows 2c, 2d and 125 set the variables named in their rows. Every other row leaves them unset.

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

Every source in the existing DM suite gets exactly one disposition, picked by the first value in the rule below that fits. The MM8 ports are `TEST_P` suites instantiated as `MM8`, so their filters are written `*MimirDm<Name>*`. The 44 cases of the first port all passed on `mimir_2x_package` on 2026-10-21 (soc-c-26) and on `mimir_1x1` on 2026-10-21. On 2026-10-20 (soc-zebu-03), `MimirDmIdma.Copy/RemoteTl1` and `Copy/RemoteGddr` failed on `mimir_2x_package` with `ack_timeout`: the data never landed, on the write and the read legs, and the cause is not known. The MM-8 follow-up then changed every MM8 suite and added seven cases (51 in all). On `mimir_2x_package` all 51 and the MM3 `MimirKernelLoad` cases passed on 2026-10-22 (soc-zebu-04, one fresh session). An earlier session on 2026-10-21 (soc-zebu-01) passed 23 cases, then `Copy/OwnTl1` failed (acked never came, the data never landed) and `DramUnary.ReaderWriterHandshake/Cce2` hit the host deadline; both passed on soc-zebu-04, so they are recorded as that session's failures, cause not known. On `mimir_1x1` the 46 follow-up cases that run there and the MM3 `MimirKernelLoad` cases passed on 2026-10-23 (soc-zebu-03, one fresh session); the other 5 need two Mimirs or CCEs 2 and 3.

| Value | Rule |
|---|---|
| `B-port` | MM8 builds a Mimir analogue |
| `covered` | The upstream test cannot run, but an existing MM3-7 or MM4x test exercises its intent on that descriptor |
| `B-blocked` | An analogue is possible but needs a named capability or runtime change |
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
| `data_movement/dram_sharded/test_dram_sharded.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirRemoteDram.ReadToTl1*`, `MimirPrefetcher.MM7T1Local0To1`, `MimirPrefetcher.MM7T1Local3To2`, `MimirPrefetcher.MM7T1D2d0To2`, `MimirPrefetcher.MM7T1D2d3To1` | `*MimirLocalDram.CceReadsDramToTl1*`, `MimirPrefetcher.MM7T1Local0To1` | Intent: read DRAM shards into TL1 | *Trid* cases: covered by `MM43/MimirTridIdma.UniqueIdsLocalGddr*`; `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/transaction_id/test_transaction_id.cpp` | covered | `MM43/MimirTridIdma.BarrierThenDependent*`, `MM43/MimirTridIdma.UniqueIds*` | `MM43/MimirTridIdma.BarrierThenDependent*`, `MM43/MimirTridIdma.UniqueIds*` | iDMA command-buffer trids: unique trids, per-trid ack wait, then the dependent transfer; final data gates, stale data right after the ack is recorded (MIM-628) | `_2_0` variants: B-blocked (KernelSpec is TENSIX-only) |
| `data_movement/direct_write/test_direct_write.cpp` | covered | `*MimirRemoteTl1Direct.SizeAlignment*` | gap | 4 B case of SizeAlignment; upstream tests inline dword writes, which the CCE does not have, so the 4 B CPU store is a stand-in; 1x1 gap: sub-word direct write (D4) | Mcast cases: C |
| `data_movement/pcie_read_bw/test_pcie_read_bw.cpp` | C | — | — | No PCIe | — |
| `data_movement/pcie_write_bw/test_pcie_write_bw.cpp` | C | — | — | No PCIe | — |
| `data_movement/device_pcie_loopback/test_device_pcie_loopback.cpp` | C | — | — | No PCIe | — |
| `data_movement/pcie_mid_cleanup/test_pcie_mid_cleanup.cpp` | C | — | — | No PCIe | — |
| `data_movement/atomics/test_atomic_semaphore_bandwidth.cpp` | covered | `MM42/MimirAtomicLocal.Hotspot*` | `MM42/MimirAtomicLocal.Hotspot*` | The value and count sweep as amoadd.w on a cached own-TL1 word with one, two and eight harts; the NoC-atomic form has no ITN equivalent | — |
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
4. Every test listed by `--gtest_list_tests --gtest_filter='MM41/*:MM42/*:MM43/*:MM42Detector/*'` is matched by exactly one row from 81 to 130.

## MM-8 DM regression

These rows are the plan's `mm_data_movement_regression`. They continue from row 71, with the same environment and timeout prefix as rows 2b onward. Each case runs as its own invocation (list them with `--gtest_list_tests --gtest_filter='MM8/*'`, with `TT_METAL_EMU_SOC_DESC` set: the `MimirDmMeshBufferDram` case names end in the descriptor name, and without it they end in `Unconfigured` and match nothing). Add `--gtest_output=xml:<dir>/<descriptor>_<row>.xml` to each command; it is the Gate 6 evidence, read against the covered-by columns above. On `mimir_1x1` (2026-10-21), 80 cases ran (the 44 `MM8/*` cases of the first port and the 36 MM-7 rows marked "all"): 75 passed, 5 skipped, 0 failed. The skips are `MimirDmIdma.Copy/RemoteTl1` and `Copy/RemoteGddr` (need two Mimirs), `MimirDmDramUnary.ReaderWriterHandshake/Cce2` and `Cce3` (1x1 has 2 CCEs), and `MM7T5TraceCaptureReplay` (always skips). On `mimir_1x1` (2026-10-23, soc-zebu-03, at `3a3d04f9bd2`), the follow-up's 46 `MM8/*` cases that run there and the 9 `MimirKernelLoad` cases of row 3 that run there all passed (55 of 55). The 5 `MM8/*` cases that skip there (`MimirDmIdma.Copy/RemoteTl1`, `Copy/RemoteGddr` and `Copy/RemoteTl1KernelVerified`, and `MimirDmDramUnary.ReaderWriterHandshake/Cce2` and `Cce3`) were not run. Rerun MM3 KernelLoad (row 3) with these rows: the follow-up changed its echo kernel.

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

## MM4x: test plan Sections 4.1-4.3

`MM41` covers Section 4.1 (CCE caches), `MM42` Section 4.2 (atomics) and `MM43` Section 4.3 (transaction IDs). `MM42Detector` holds the non-gating MIM-560 detector, so `MM42/*` does not run it. The plan is `~/.claude/plans/the-changes-in-this-cozy-lollipop.md`.

### MM4x probe results

Each probe ran on the `mimir_2x_package` emulator (soc-zebu-04) on 2026-10-22, on all four CCEs, one case per invocation, against `61d03f6e674` plus the uncommitted MM4x tree. Kernels were built from a frozen copy of `test_kernels/cce`. Logs are in `/localdev/sagarwal/mm4x_runs_2026-10-22/mm4x/probe_logs_final/`.

| Probe | Result | Rows that depend on it |
|---|---|---|
| `MM41/MimirCacheProbe.EightHartLaunch/*` | yes: all eight harts of every CCE ran a kernel and posted status | Every 8-hart MM4x case |
| `MM41/MimirCacheProbe.SnoopConfig/*` | `NOC_SNOOP_TL_MASTER_CFG` = 0x6 (default) on every CCE. L2 CONFIGURATION = 0x06060804: 4 banks, 8 ways, 64 sets, 64 B lines (128 KiB). Chippy's reset value (8 banks, 32 sets) and the "4-way" comment in `overlay_addresses.h` do not match it | `MimirCacheCoherence.DirtyLineRace` starting value; MM41 stress sizes |
| `MM41/MimirCacheProbe.L2Geometry/*` | Masks identical on every CCE. `cflush.d.l1 x0` alone never reached TL1 (bit 0 never set), so it stops in L2. Replacement is not LRU: at strides of 16 KiB and more the dirty target was evicted after as few as 3 same-set reads and survived some counts above 8, the same way on every CCE. 8 KiB strides (two sets) evicted it at every count from 17. Conflict stride 16 KiB | `MimirCacheStress.ReuseThrash` strides; `DirtyEvictionPartialLine` and `DirtyLineRace` eviction sweeps (24 same-set reads) |
| `MM42/MimirAtomicProbe.TwoHartAmo/*` | yes: 2 harts x 64 amoadd.w and 64 LR/SC adds; totals, return-value sums and the host view all correct. 0 SC failures, so the two harts may barely have overlapped | Every multi-hart AMO case |
| `MM43/MimirTridProbe.Width/*` | TR_ID written as 17 reads back 17 (the register keeps bit 4), but the command counts on trid 1: pending 1 on trid 1, 0 on trid 17. Consistent with MIM-539 (TRID[4] dropped); an out-of-range counter read would look the same | Records only; range tests use trids 0-15 |
| `MM43/MimirTridProbe.PendingVisible/*` | 12 pending on one trid during a 16 x 8 KiB burst to local GDDR, on every CCE | NominalDepth and Saturate high-water asserts |
| `MM43/MimirTridProbe.Scope/*` | shared: while hart 0 had 11 commands pending on trid 5, hart 1 of the same CCE read 11 on its own command buffer. Pending counters are per CCE, not per hart. Whether CCEs share them is not tested | `MimirTridIdma.MultiHart` gives each hart its own trids |
| `MM43/MimirTridProbe.DepthBeyond16/*` | 32 x 2 KiB over trids 0-15 twice: landed, 404 VC-space stall cycles in all, at most 1 pending per trid, so earlier commands had finished before their trid was reused | Non-gating |

A first pass of these probes compiled the live kernel sources while they were being edited (a relative kernel path resolves against the working directory first, `kernel.cpp` `resolve_path`). That pass was discarded. Its Scope runs failed because the observer kernel was mid-edit.

### MM4x run order

Rows 81 onward continue from row 80. They use the same environment and timeout prefix as rows 2b onward, plus a fresh session and bring-up before row 81. Run each case as its own invocation (`--gtest_list_tests` with the row's filter lists them) and add `--gtest_output=xml:<dir>/<descriptor>_<row>.xml`. Every `MM4x` suite is a `TEST_P` over the CCE index (`/Cce0` to `/Cce3`), except where the row says otherwise; 2x-only cases skip at runtime on `mimir_1x1`. "Own row" marks a hang-prone case: run it alone, and after any host deadline or any kernel timeout code start a fresh session and run bring-up before the next row (MIM-601).

| # | Filter | Covers | Own row | Descriptors |
|---|---|---|---|---|
| 81 | `MM41/MimirMm4xCompile.*` | Compile only: every MM4x kernel builds for all eight harts of CCE 0; nothing launches | — | all |
| 82 | `MM41/MimirCacheProbe.EightHartLaunch/*` | Probe: all eight harts of a CCE run a kernel | yes | all |
| 83 | `MM41/MimirCacheProbe.SnoopConfig/*` | Probe: snoop-master config and L2 CONFIGURATION | yes | all |
| 84 | `MM42/MimirAtomicProbe.TwoHartAmo/*` | Probe: two-hart amoadd.w and LR/SC totals | yes | all |
| 85 | `MM41/MimirCacheProbe.L2Geometry/*` | Probe: dirty-line eviction against stride and count | yes | all |
| 86 | `MM43/MimirTridProbe.Width/*` | Probe: TR_ID 17 | yes | all |
| 87 | `MM43/MimirTridProbe.PendingVisible/*` | Probe: highest pending count on one trid | yes | all |
| 88 | `MM43/MimirTridProbe.Scope/*` | Probe: pending counters per hart or shared | yes | all |
| 89 | `MM43/MimirTridProbe.DepthBeyond16/*` | Probe: 32 commands over 16 trids | yes | all |

Stop after row 89: the gated cases below take their constants from these results (see MM4x probe results). Then rerun the MM-8 rows whose kernels MM4x changed (75, 76 and 80, with `Op/L1InvalidateFull` last) before row 90.

| # | Filter | Covers | Own row | Descriptors |
|---|---|---|---|---|
| 90 | `MM41/MimirCacheOps.L1DcacheFlushLine/*:MM41/MimirCacheOps.L1DcacheFlushFull/*` | cflush line and whole L1 D$, then the L2 flush; records whether cflush alone reached TL1 | — | all |
| 91 | `MM41/MimirCacheOps.DirtyEvictionPartialLine/*` | Partial-line cached writes evicted by same-set reads; untouched bytes intact | — | all |
| 92 | `MM41/MimirCacheSharedL2.*` | Same line and offset, same line other offset, adjacent line: one writer and one reader (the no-flush read records, the read after the flush and invalidate gates), and eight concurrent writers | — | all |
| 93 | `MM41/MimirCacheScopedLock.MultiConsumer/*:MM41/MimirCacheScopedLock.RingWrap/*` | Seven consumers of one producer; a four-slot ring over twelve rounds | — | all |
| 94 | — | Removed on 2026-10-23 with R1: section 4.1 does not name INVALIDATEFLUSHDIRTY, and `flush_l2_cache_full` (row 75) covers the L2 full flush | — | — |
| 95 | `MM41/MimirCacheOps.FlushFenceOrdering/*` | Flush without fence, fence without flush (both record), L1 then L2 flush and the full hierarchy (`cflush.d.l1 x0` then `flush_l2_cache_full` each pass; both gate), producer racing a consumer hart; the full-hierarchy leg runs last | yes | all |
| 96 | `MM41/MimirCacheOps.FullInvalidate/*` | Whole-L1 D$ discard after a full clean (`cflush.d.l1 x0`, then FLUSH64 of every cacheable TL1 line, in the same asm block), then a FULLINVALIDATE wipe with a bounded poll. A wipe that never reads 0 needs a fresh session, so run this row last | yes | all |
| 97 | `MM41/MimirCacheWriterProbe.ExternalWriters/*_Host_*:*_RemoteCce_*:*_SiblingSpa_SnoopOffInvalidate:*_SiblingSpa_InvalidateOnly` | External-writer probe: host, remote CCE and sibling SPA writers, snoop off with INVALIDATE64 and INVALIDATE64 only | — | all (RemoteCce 2x only) |
| 98 | `MM41/MimirCacheWriterProbe.ExternalWriters/*_SiblingSpa_SnoopOn` | Sibling SPA writer with RegionAttrs bit 30 set on its remap entry 1 (R2); asserts the cached re-read is fresh with no invalidate | yes | all |
| 99 | `MM41/MimirCacheWriterProbe.ExternalWriters/*_LocalIdma_*` | Same-CCE iDMA writer, three modes. Snoop on uses the iDMA snoop packet tag and asserts the reader is refreshed, as GRENSV-2290 expects (chippy `cce_xcce_snoop_dma` sets the tag the same way). Rows 100-101 use the same snoop-on rule | yes, per case | all |
| 100 | `MM41/MimirCacheWriterProbe.ExternalWriters/*_LocalGddrIdma_*` | iDMA from local GDDR, three modes | yes, per case | all |
| 101 | `MM41/MimirCacheWriterProbe.ExternalWriters/*_RemoteGddrIdma_*` | iDMA from remote GDDR, three modes | yes, per case | 2x only |
| 102 | `MM41/MimirCacheScopedLock.NonSnoopingProducer/*` | Sibling-CCE SPA producer, consumer under the scoped lock | — | all |
| 103 | `MM41/MimirCacheCoherence.CrossCce/*` | Cached stores flushed and read by the sibling over SPA, and the reverse | — | all |
| 104 | `MM41/MimirCacheCoherence.CrossCceSnoopOn/*` | Sibling iDMA with the snoop packet tag into a warm cached reader; asserts no stale word (GRENSV-2290) | yes | all |
| 105 | `MM41/MimirCacheCoherence.CrossD2d/*` | CrossCce with the CCE at the same location on the other Mimir; one route per case | yes, per case | 2x only |
| 106 | `MM41/MimirCacheCoherenceRace.DirtyLineRace/*` | One case per CCE and snoop-master config (`Cce{n}_Cfg{c}`, the default 0x6 first): neighbour or same words x evict, FLUSH64 or INVALIDATE64 against a sibling iDMA writer; 0x6 gates (it relies on the iDMA snoop tag, GRENSV-2290), the other configs record; each case restores the config | yes, per case | all |
| 107 | `MM41/MimirCacheStress.ReuseThrash/*` | Sequential, 16 KiB conflict-stride and seeded random patterns, 2 KiB to 192 KiB; cycles recorded | — | all |
| 108 | `MM41/MimirCacheStress.PerfSweep/*` | The four MM-8 write paths at 64, 128 and 192 KiB; cycles recorded, no timing asserts | — | all |
| 109 | `MM41/MimirCacheStress.PreloadChurn/*` | Sibling iDMA preload with and without the snoop tag, then the hot-set re-read; both legs gate (the snoop leg per GRENSV-2290). Records how many stale words the hart's cache still held from earlier launches before each leg drops them | yes | all |
| 110 | `MM42/MimirAtomicLocal.OpMatrix/*:MM42/MimirAtomicLocal.Hotspot/*:MM42/MimirAtomicLocal.Distributed/*:MM42/MimirAtomicLocal.LrScCas/*:MM42/MimirAtomicLocal.Wrap/*:MM42/MimirAtomicLocal.Ordering/*:MM42/MimirAtomicLocal.CleanDirtyLine/*:MM42/MimirAtomicLocal.Storm/*:MM42/MimirAtomicLocal.NeighbourWriteRace/*:MM42/MimirAtomicLocal.Soak/*` | Tier A: every amo*.w/.d and LR/SC CAS; one, two and eight senders (atomic_semaphore_bandwidth sweep); distributed words; wrap; ordering; clean and dirty lines; mixed-op storm; hart stores beside AMOs; seeded soak. Contention is recorded per leg | — | all |
| 111 | `MM42/MimirAtomicItnControl.*` | Sibling CCE writes a word and a flag over SPA; this CCE waits with uncached loads, invalidates, then AMOs | — | all |
| 112 | `MM42/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/*` | AMO after a sibling iDMA with the snoop tag, no invalidate; asserts the AMO sees the iDMA value (GRENSV-2290) | yes | all |
| 113 | `MM8/MimirDmIdma.*` | Row 74 again: the single-trid baseline before the MM43 rows | as row 74 | as row 74 |
| 114 | `MM43/MimirTridIdma.UniqueIdsOwnTl1/*:MM43/MimirTridIdma.UniqueIdsSiblingTl1/*:MM43/MimirTridIdma.ReuseWrap/*:MM43/MimirTridIdma.DuplicateId/*:MM43/MimirTridIdma.BarrierThenDependent*` | Per-command trid over 0-15 to own and sibling TL1; reuse and wrap; duplicate IDs; barrier then dependent read or write (stale-at-ack counts record) | — | all |
| 115 | `MM43/MimirTridIdma.UniqueIdsLocalGddr/*` | GDDR0 and GDDR1 writes, reads from both on VC 5 | yes | all |
| 116 | `MM43/MimirTridIdma.NominalDepth/*` | 16 trids back to back, mixed targets, the VC rewritten per command | yes | all |
| 117 | `MM43/MimirTridIdma.MixedTargets/*` | One table batch over own TL1, sibling TL1, GDDR0/1 and (2x) remote, host-verified remote | yes | all |
| 118 | `MM43/MimirTridIdma.Saturate/*` | Maximum outstanding per backend VC with 8 KiB commands, then combined (total pending across the table's trids) | yes | all |
| 119 | `MM43/MimirTridIdma.MultiHart/*` | Eight harts, each with its own trids (Scope: counters are per CCE) | yes | all |
| 120 | `MM43/MimirTridIdma.Fairness/*` | One CCE at depth 16 while other CCEs issue table traffic | yes | all |
| 121 | `MM43/MimirTridIdma.Soak/*` | Seeded random targets, sizes and trid order with repeated wrap | yes | all |
| 122 | `MM43/MimirTridRemote.RemoteTl1/*` | Remote TL1, both directions, host-verified; one route per case | yes, per case | 2x only |
| 123 | `MM43/MimirTridRemote.RemoteGddr/*` | Remote GDDR, both directions, host-verified; one route per case | yes, per case | 2x only |
| 124 | `MM43/MimirTridConcurrent.WithPrefetcher/*` | Trid stream on CCE 1 while CCE 0 prefetches to CCE 2; overlap asserted, otherwise a skip | yes | 2x only |
| 125 | `TT_METAL_FORCE_REINIT=1` prefix, filter `MM43/MimirTridReset.NoStaleIdAfterReinit/*` | All 16 pending counts read 0 after a hart reinit, then every trid completes a command | yes | all |
| 126 | `MM42/MimirAccumProbe.FmtOpMatrix/*` | iDMA L1 accumulate per (format, op) against the host model; stops at the first point that does not ack | yes | all |
| 127 | `MM42/MimirAccumProbe.DirtyLine/*` | Accumulate onto a dirty cached line; records whether the write-back replaced the result | — | all |
| 128 | `MM42/MimirAtomicIdmaAccum.SingleHart/*:MM42/MimirAtomicIdmaAccum.AllHartsShared/*:MM42/MimirAtomicIdmaAccum.AllHartsDisjoint/*:MM42/MimirAtomicIdmaAccum.BackToBack/*:MM42/MimirAtomicIdmaAccum.MixedPathControl/*` | Tier M: INT32 accumulate, op 0x9, VC 5, one and eight harts, shared and disjoint, bursts on both command buffers, AMO then accumulate then AMO | — | all |
| 129 | `MM42/MimirAtomicIdmaAccum.MaxOutstanding/*` | 64 posted accumulates over trids 0-15 | yes | all |
| 130 | `MM42Detector/MimirKnownIssue.Mim560Detector/*` | MIM-560 detector: AMO, accumulate and uncached partial stores on one line while the sibling CCE streams SPA stores (ITN writes) into it, released by the host once the AMO loop has started. Non-gating; 0 corruption is inconclusive. Runs last | yes | all |

Notes on the rows:
- Rows 82 to 89 are the probes; their results are in MM4x probe results.
- `MimirTridIdma` cases with a table (NominalDepth, MixedTargets, Saturate's combined leg, Fairness, Soak) rewrite the command buffer's VC per command while earlier trids are pending. Nothing showed that safe before these rows.
- Fairness runs one hart on several CCEs; the Scope probe tested only harts of one CCE, so whether CCEs share pending counters is not known.
- MM-8 `Op/L1InvalidateLine` passes only while nothing touches the stack between the stores and the per-line discard. A 4.x kernel change that grew `kernel_main`'s frame broke it once (2 of 4 dirty lines were evicted to L2 before the discard); MM4x modes now run in a separate `noinline` function.
- Rows 98 and 106 write CCE uncore registers from the host through the SMC (RegionAttrs bit 30 on remap entry 1; the snoop-master config). Each guard restores the value it found.

### MM4x results

On `mimir_2x_package` (rv-zebu-08, one session after bring-up, 2026-10-22), every MM4x case ran once on CCE 0, or on its first route for the per-route rows, against `61d03f6e674` plus the uncommitted MM4x tree. Kernels were built from a copy of `test_kernels/cce` that was refreshed only between cases. Each case ran against the tree as it stood when it ran: fixes that landed during the run are noted in their rows, and the final tree has not run end to end. The other CCEs and routes, row 96 and `mimir_1x1` have not run. Logs, XML and the run scripts are in `/localdev/sagarwal/mm4x_runs_2026-10-22/` (`mm4x/runs/`, `mm4x/xml/`, `mm4x/pass2.list`).

| Rows | Result |
|---|---|
| 81 | pass (5 kernels) |
| 75, 76, 80 (MM-8 rerun) | pass, all 19 cases. `Op/L1InvalidateLine` first failed with a 4.x kernel whose larger `kernel_main` frame evicted 2 of 4 dirty lines before the discard; it passed after the MM4x modes moved to a `noinline` function |
| 90-93, 97-105, 107, 108, 110-117, 119-130 | pass. Rows 104 and 112 ran in their record-only versions; the current versions assert the snoop (see rows 99-101) |
| 94 | fail, finding: INVALIDATEFLUSHDIRTY is a no-op on this emulator. After `cflush x0` and a write of 1 (as chippy's `mimir_cce_mem_sweep_cached` does), the cached stores never reached TL1; a cached read still returned them and a FLUSH64 then wrote them. Writing 0xFF (AllHarts) behaves the same, and the register reads 0 at once. R1 (`invalidate_flush_dirty_l2_full`) therefore did not clean. R1 and these cases were removed on 2026-10-23; the finding is for the hardware owners |
| 95 | fail on the full-hierarchy leg only, in the version where that leg used R1 (complete at the consumer's first read in 1 of 8 passes, as for fence without flush). Flush without fence 8/8, fence without flush 1/8, L1 then L2 8/8. The leg now uses `flush_l2_cache_full` and has not run |
| 96 | not run: it was held while its clean used R1. It now cleans with FLUSH64 and has not run |
| 98, 99-101 | Ran in the version whose snoop-on legs only recorded. External writers: no writer is snooped by default; INVALIDATE64 alone gives a fresh read (it also clears the hart's L1 D$ line). A sibling SPA writer with RegionAttrs bit 30 (R2) is snooped. The iDMA snoop packet tag is not honoured on these paths: the reader stayed stale for local, local-GDDR and remote-GDDR iDMA writers. It is honoured for a sibling CCE's TL1-to-TL1 iDMA (backend VC 2): rows 104 and 112 read fresh in both versions. The current version asserts the GRENSV-2290 behaviour, so the snoop-on legs of rows 99-101 and PreloadChurn's snoop leg (a GDDR-sourced copy on VC 5) are expected to fail on this emulator until the tag is honoured |
| 106 | On CCE 0: `Cfg6` (default 0x6, in the version where it did not yet gate), `Cfg0` and `Cfg2` pass. `Cfg1` (0x1, first in the earlier all-config form) and `Cfg3` (0x3), the two configs with `disable_inline_writes`=1 that ran: a sibling iDMA write into a line the hart held dirty was acked but never became visible in TL1 within the budget, and the case stopped for a fresh session. iDMA health checks and SnoopConfig (0x6 restored) passed after the first stop. `Cfg4`, `Cfg5` and `Cfg7` have not run |
| 109 | pass after a fix (the snoop leg first read lines the invalidate leg had left cached; the host's re-seed of TL1 is not snooped). That version recorded the snoop leg; the current version gates it and is expected to fail (see rows 99-101) |
| 110 | pass after a fix: 8-hart CAS increments on one word needed up to 3059 (.w) and 3571 (.d) tries for one increment, past the first 1024-try cap; LR/SC adds (0 SC failures even with 8 harts, which interleaved 4088 times) and AMOs never starved |
| 118 | per-backend legs pass with 8 KiB commands (pending high water VC0 13, VC1 13, VC2 10, VC3 14, VC5 2). The combined table leg reached only 1 pending per trid, which cannot show combined saturation; it now asserts the total pending across its trids instead. That version has not run |
| 124 | pass: overlap seen in the window (sender pages 0 to 256, stream rounds 0 to 1) |
| 125 | pass with `TT_METAL_FORCE_REINIT=1` |
| 130 | inconclusive: 0 corrupted words, but in that version (sibling iDMA as the ITN source) the ITN writes did not overlap the AMO rounds; the accumulate did. The host-released SPA stream has not run |

On `mimir_2x_package` (2026-10-26, at `3a3d04f9bd2`), a run of the current tree on all four CCEs and every route started: row 81, the MM-8 rerun of rows 75, 76 and 80 (`Op/L1InvalidateFull` last), then rows 90-130 in order with row 96 last. The probe rows 82-89 were left out: they ran on all four CCEs on 2026-10-22 and only record. It ran in six emulator sessions, each after bring-up: soc-c-24 (40 cases), soc-zebu-01 (24 cases), soc-c-23 (89 cases), soc-zebu-01 again (80 cases, 2026-10-26 to 2026-10-27), rv-zebu-08 (48 cases, 2026-10-27) and soc-c-31 (13 cases, 2026-10-27). Each ended when the session ended. The fourth ended when the emulator stopped serving during `MimirAtomicLocal.Soak/Cce1`, which is not counted and ran again in the fifth. The sixth ended with `MimirTridIdma.Saturate/Cce2` killed mid-run, which is not counted. Sessions 1-5 ran at `3a3d04f9bd2` (before the branch was rebased onto main) with UMD `d7547af48`; the sixth ran at `86924f535c9`, after the rebase and the port of `mimir_tests` to a unit MeshDevice, with UMD `f0d497f33`. 294 of the 361 cases have run: 272 pass and 22 fail. Only row 106's failures needed a fresh session; each was followed by bring-up, and `SnoopConfig` and an iDMA case passed before the next case.

| Rows | Result on `mimir_2x_package`, 2026-10-26 |
|---|---|
| 81; 75, 76, 80 (MM-8 rerun) | pass (5; all 19) |
| 90-93, 97, 102, 103, 105 | pass on all four CCEs (and every route for row 105) |
| 95 | pass on all four CCEs; the first 2x run of the version whose full-hierarchy leg uses `flush_l2_cache_full` |
| 98, 104 | pass on all four CCEs; the first 2x runs of the versions that assert the snoop (RegionAttrs bit 30 for row 98, the iDMA snoop tag on a sibling CCE's TL1-to-TL1 copy for row 104) |
| 99, 100, 101 | the `SnoopOn` leg fails on all four CCEs, as expected: after an iDMA with the snoop tag (own TL1, local GDDR, remote GDDR) the cached reader stayed stale on all 64 words. The other legs pass |
| 106 | 26 of 32 pass, the first 2x runs of the version where the default config gates. `Cfg0`, `Cfg2`, `Cfg4` and `Cfg6` pass on all four CCEs. The configs with bit 0 set fail intermittently: CCE 0 fails `Cfg1` and `Cfg5`, CCE 1 fails `Cfg3`, CCE 2 fails `Cfg1` and `Cfg5`, CCE 3 fails `Cfg1`; the other bit-0 cases pass. As on `mimir_1x1`, each failure is on a neighbour-line leg (`Neighbour_Flush64` for CCE 0 and CCE 3, `Neighbour_Evict` for CCE 1 and CCE 2): a sibling iDMA write was acked but never became visible in TL1, and the case stopped for a fresh session |
| 107, 108 | pass on all four CCEs (`ReuseThrash` about 16 min a case, `PerfSweep` about 10 min) |
| 109 | fail on the snoop leg on all four CCEs, as expected: a sibling CCE's iDMA from GDDR (VC 5) with the snoop tag left the hot set stale (1664 or 1696 stale words at the first read, 0 of 4 rounds complete). The invalidate leg passes, 4 of 4 rounds |
| 110, 111, 112 | pass on all four CCEs |
| 113 (MM-8 `MimirDmIdma.Copy`) | pass on all 12 routes |
| 114 | pass on all four CCEs |
| 115, 116, 117 | pass on all four CCEs |
| 118 | `Saturate` passes on CCE 0 and CCE 1 (about 15 min a case); CCE 2 and CCE 3 have not run |

The rest, from row 118's `MimirTridIdma.Saturate/Cce2` on (67 cases: rows 118-130, then row 96), has not run. The run directory is `/localdev/sagarwal/mm4x_runs_2026-10-26/mimir_2x_package/` (`cases.tsv` is the case order, `results.tsv` the results, `run.sh` resumes the run and skips recorded cases).

On `mimir_1x1` (soc-c-23, 2026-10-23), every MM4x case that runs on 1x1 ran once on CCE 0 and CCE 1 against the current tree (R1 removed): 167 cases, including the 5 compile rows and row 125. The cases that skip on 1x1 (CCE 2 and 3, rows marked 2x only, the remote-CCE legs) were not run. After each case that stopped for a fresh session, bring-up was rerun and `SnoopConfig` and an iDMA case passed before the next case. 154 pass and 13 fail; after the `kModeTridTable` change (row 118) Saturate passes, so 156 pass and 11 fail, all in rows 99, 100, 106 and 109.

| Rows | Result on `mimir_1x1` |
|---|---|
| Every row 81-130 not listed below, apart from the rows marked 2x only | pass. Row 95's full-hierarchy leg is complete at the flag in 8 of 8 passes. Row 96: the discard leaves TL1 old (0 of 64 new words) and the wipe turns 64 stale cached words fresh. Rows 104 and 112 read fresh: a sibling CCE's TL1-to-TL1 iDMA (VC 2) with the snoop tag is snooped. Row 130 saw the ITN writes and the accumulate overlap the AMO rounds, with 0 corrupted words (recorded as inconclusive, not evidence against MIM-560) |
| 99, 100 | fail on the `SnoopOn` legs on both CCEs, as expected: after an iDMA with the snoop tag from the reader's own CCE (own TL1, local GDDR), the cached reader stayed stale on all 64 words. The snoop-off-with-invalidate and invalidate-only legs pass |
| 106 | `Cfg0`, `Cfg2`, `Cfg4` and `Cfg6` pass on both CCEs. The configs with bit 0 set fail intermittently: CCE 0 fails `Cfg1`, `Cfg3` and `Cfg7` and passes `Cfg5`; CCE 1 fails `Cfg1` and `Cfg5` and passes `Cfg3` and `Cfg7`. Each failure is on a neighbour-line leg (`Neighbour_Flush64`, or `Neighbour_Evict` for CCE 0 `Cfg7`): a sibling iDMA write was acked but never became visible in TL1, and the case stopped for a fresh session |
| 109 | fail on the snoop leg on both CCEs, as expected: a sibling CCE's iDMA from GDDR (VC 5) with the snoop tag left the hot set stale (1664 stale words at the first read, 0 of 4 rounds complete). The invalidate leg passes, 4 of 4 rounds |
| 118 | pass on both CCEs after a kernel change. In the first run the combined leg failed: the total pending across the table's trids never exceeded 1, while every trid's count cleared 366 cycles after the last issue and the data took about 4.9M cycles to land, so each count cleared on accept (MIM-628) before the table loop issued the next entry. `kModeTridTable` now resolves every entry before the first issue and records the largest gap between issues. With it the total reached 2 with a largest gap of 263 cycles, so commands to different backends were pending together. The per-backend legs reached VC0 13, VC1 13, VC2 10, VC5 2 in both runs. MixedTargets, NominalDepth, Fairness and Soak, the other table cases, passed again on both CCEs with the change |

## Fresh-session rule

Before a full run, start a fresh emulator session and run chippy bring-up.

Run the host suite (row 1) right after bring-up. The host suites skip when any CCE hart is out of reset. A process that opened a device resets the CCE harts on clean exit and leaves the uncore released, as bring-up does, so row 1 also runs after such a process. After a killed process the harts may still be running, and row 1 skips.

If a run is killed mid-stall, the session is probably wedged. Use the same rule: a fresh session plus bring-up.

## Rules

- Never set `TT_METAL_CLEAR_DRAM`. It writes every view in full, 1 GiB per Mimir, which is far too slow on the emulator.
- Tests that run kernels reach host DRAM through `mimir_contract::write_gddr_view` / `read_gddr_view` on the unit mesh (`ScopedDevice::mesh()`), which use the slow-dispatch DRAM channel API. They reach CCE TL1 through the `IDevice*` forms of `write_cce_tl1` / `read_cce_tl1`, which use the contract suite's `write_cce_sram` / `read_cce_sram`. A DRAM write below the allocator's DRAM base is refused.
- The host suites (`MimirHost*`), the firmware-load and reset-state checks, and the register reads (`mimir_hw_state`, `mimir_snoop_ctl`, debug dumps) stay on `Cluster`. They test the backend boundary itself, and no contract helper covers register reads.

## Manual D2D AXI monitor step

The emulator's D2D AXI monitor is a manual debug step. It is a DPI bind (`models/mimir/emu_hdl/axi_mon_d2d_bind.sv`, `top.utf:22`). To use it, relaunch the SiVal server without `--disable-dpi`, which the server command in the repo-root `mimir_setup.md` passes. With DPI on, the SMC AXI monitor also logs every host beat.
