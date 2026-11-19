# mimir_tests

`mimir_tests` covers Mimir bring-up: host access to DRAM and TL1, firmware and kernel load, and CCE copies to local and remote DRAM and TL1. Run them in the order of the run-order table below. It is built only when `TT_UMD_BUILD_GRENDEL_JTAG=ON`. The binary is `build/test/tt_metal/mimir_tests`. Each row of the run order is a separate invocation, and the order matters.

## Environment

Every case starts with `skip_unless_programmable_dram_package()` (`tests/tt_metal/tt_metal/dram/mimir_contract.hpp`) and runs when it passes: the selected package declares programmable DRAM cores and a transport is configured. `TT_METAL_EMU_SERVER` is not required.

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
- `TT_METAL_QUASAR_VARIANT` (the L2 cache-controller addresses the kernels of the cache, atomics and transaction-ID tests use are those of the unset or 2.0.2 variant)

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
| 3 | `*MimirKernelLoad*:*MimirLocalTl1*` | Kernel load, same-chiplet TL1 | all |
| 3a | `DmSuitePort/MimirDmIdma.Copy/OwnTl1:DmSuitePort/MimirDmIdma.Copy/LocalGddr*` | iDMA smoke before row 3b: one copy to own TL1, and a write and a read of local GDDR0 and of GDDR1, on CCE 0 hart 0 (the single-command modes) | all |
| 3b | `LocalDram/MimirLocalDram.*` | Local DRAM copies by iDMA (`cce_idma_copy` `kModeTridBatch`): every CCE, all 8 harts, every local view and anchor; DRAM to TL1, TL1 to DRAM and the loopback through TL1; the size and alignment matrix (write direction); repeated overlapping copies, one launch per copy. Own row (see the note) | all |
| 4 | `D2dDirect/MimirRemoteTl1.*` | Remote TL1 write/read, host-initiated remote read, size and alignment; direct pairs | 2x only |
| 5 | `*MimirRemoteTl1Bidir*` | Concurrent remote TL1 in both directions, direct pairs | 2x only |
| 6 | `D2dRouted/MimirRemoteTl1.*/M0Cce0ToM1Cce0` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 7 | `D2dRouted/MimirRemoteTl1.*/M0Cce1ToM1Cce1` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 8 | `D2dRouted/MimirRemoteTl1.*/M1Cce0ToM0Cce0` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 9 | `D2dRouted/MimirRemoteTl1.*/M1Cce1ToM0Cce1` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 10 | `D2dRouted/MimirRemoteTl1.*/M0Cce0ToM1Cce1` | Remote TL1 as in row 4, two-hop pair | 2x only |
| 11 | `D2dRouted/MimirRemoteTl1.*/M1Cce1ToM0Cce0` | Remote TL1 as in row 4, two-hop pair | 2x only |
| 12 | `*MimirRemoteDram*/SrcM0Cce1*` | Remote DRAM write, read, round trip and traffic matrix; one-hop source | 2x only |
| 13 | `*MimirRemoteDram*/SrcM1Cce0*` | Remote DRAM write, read, round trip and traffic matrix; one-hop source | 2x only |
| 14 | `*MimirRemoteDram*/SrcM0Cce0*` | Remote DRAM write, read, round trip and traffic matrix; two-hop source | 2x only |
| 15 | `*MimirRemoteDram*/SrcM1Cce1*` | Remote DRAM write, read, round trip and traffic matrix; two-hop source | 2x only |
| 16 | `*MimirRemoteMixed*` | Remote TL1 one-to-many and many-to-one; concurrent local DRAM, remote DRAM and remote TL1 | 2x only |

Notes on the rows:
- Row 1 starts with the identity check. If it fails, every test in row 1 fails with the mismatch before any device is opened.
- Row 3b is hang-prone, like the cache, atomics and transaction-ID cases marked "own row": run each case as its own invocation (`--gtest_list_tests --gtest_filter='LocalDram/*'` lists them), and after any host deadline or any kernel timeout code (`vc_space_timeout`, `ack_timeout`, `data_timeout`) start a fresh session and run bring-up before the next case (MIM-601). Every launch can end in a timeout code, so a failure early in `SizeAlignmentMatrix` costs the rest of the session.
- Rows 4 to 16 each start with a preflight readback of SS_BAR, ATT and interleave. A mismatch fails the test before CreateDevice.
- The CCE DMRISC remap is compared with UMD's values once per process, right after the first CreateDevice. Chippy bring-up programs the remap and releases the uncore; nothing in Metal or UMD rewrites the remap at CreateDevice (startup and exit reset only the harts), so the comparison checks bring-up's values.
- Rows 4 to 16 are 2x only. Each `TEST_P` also skips at runtime from the descriptor's topology.
- Rows 6 to 15 run one route each, so a hang in one route does not lose the others.
- Row 16 runs last because it includes routed destinations.

## Contract tests

The `MimirEmu*` cases check the runtime through its public API only: they open a unit `MeshDevice`, launch Programs through the mesh command queue, and reach GDDR and TL1 through the `mimir_contract.hpp` helpers. They have no row in the run order, no preflight and no hardware-state checks, and run in one invocation:

```
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/mimir_tests --gtest_filter='*MimirEmu*'
```

| File | Cases |
|---|---|
| `test_mimir_host_tl1.cpp` | `CceSramRoundTripThroughUnitMesh`; `CceSramRoundTripThroughMetalCluster` and `MimirEmu.CceSramChannelsDoNotAliasThroughMetalCluster`, which open a `Cluster` directly and skip without a chippy server |
| `test_mimir_host_dram.cpp` | `AllocatedDramBufferHostLoopback`, `DramChannelsDoNotAliasThroughPublicDeviceApi` |
| `test_mimir_fw_load.cpp` | `MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi` |
| `test_mimir_kernel_load.cpp` | `HartZeroRunsDramKernel`, `AllHartsRunDramKernel` |
| `test_mimir_local_dram.cpp` | `MimirEmuSameMimirDirect/Idma*` and `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr`: local GDDR by iDMA; `MimirEmu.AllCcesCopyLocalMimirBlock` |
| `test_mimir_remote_dram.cpp` | `MimirEmuCrossMimirNoc/CopiesGddr*` and `MimirEmuCrossMimirNoc/AllHartsWriteGddr`: remote GDDR by hart loads and stores; `CopiesEveryMimirBlock` |
| `test_mimir_tl1.cpp` | `WritesCrossMimirCceSram`, `PrefetchesLocalGddrToRemoteCceSram` |

`is_cross_mimir_case()` picks local or remote partitions from the instantiation prefix, so keep `CrossMimir` in the prefix of any remote-partition instantiation.

## Tensor Prefetcher

Rows 17 onward cover the Tensor Prefetcher on CCEs. They use the same environment and timeout prefix as rows 2b onward. Run them after a fresh session and bring-up (see Fresh-session rule), one test per invocation, in this order.

| # | Filter | Covers | Descriptors |
|---|---|---|---|
| 17 | `MimirPrefetcher.SupportPreconditionsHold` | Prefetcher preconditions hold and the port reports supported; one sender per DRAM view | all |
| 18 | `MimirPrefetcher.SupportPreconditionsFailOnUninitializedSender` | A sender CCE whose GO_MSG reads INIT fails the preconditions, naming that CCE | all |
| 19 | `MimirPrefetcher.SupportStartStopSucceeds` | Start and Stop with no requests | all |
| 20 | `MimirPrefetcher.SupportPreconditionsHoldWhileReceiverRuns` | Preconditions stay ok while a receiver Program runs on a sender CCE | all |
| 21 | `MimirPrefetcher.GcbLayoutLocalReceiver` | DRAM-receiver GCB state block, config page and credit lines; same-Mimir receiver | all |
| 22 | `MimirPrefetcher.GcbLayoutRemoteReceiver` | The same with a receiver on the other Mimir, receiver order kept | 2x only |
| 23 | `MimirPrefetcher.GcbLayoutTwoSenders` | Two senders, one receiver each; a second such GCB takes disjoint ranges | 2x only |
| 24 | `MimirPrefetcher.GcbLayoutRejectsConflictingReceivers` | Rejects aliased, sender-as-receiver and duplicate receivers | all |
| 25 | `MimirPrefetcher.GcbSmokeDiscardDrain` | Test sender into a discarding receiver, without the manager; both ends' counters agree | all |
| 26 | `MimirPrefetcher.GcbSmokeValidatorLocal` | Test sender into a validating receiver on the same Mimir | all |
| 27 | `MimirPrefetcher.GcbSmokeValidatorWrongSeed` | The validator reports a mismatch for a wrong seed | all |
| 28 | `MimirPrefetcher.GcbSmokeResize` | Three page sizes in one GCB, with resize and wrap padding | all |
| 29 | `MimirPrefetcher.GcbSmokeValidatorD2d` | Validating receiver on the other Mimir | 2x only |
| 30 | `MimirPrefetcher.GcbSmokeDrainUntilStopped` | Receiver drains until the host stops it | all |
| 31 | `MimirPrefetcher.GcbSmokeStallUntilReleased` | The receiver holds a full FIFO, so the sender gives CREDIT_TIMEOUT; then the host releases it | all |
| 32 | `MimirPrefetcher.GcbSmokeDrainTimeout` | The receiver holds every page, so the sender's final barrier gives DRAIN_TIMEOUT | all |
| 33 | `MimirPrefetcher.GcbSmokeReceiverWaitTimeout` | No sender: the receiver drains host-credited pages, then its wait gives CREDIT_TIMEOUT | all |
| 34 | `MimirPrefetcher.GcbSmokeRelaunchResumes` | A second sender and validator pair resumes from the pointers the first left in TL1 | all |
| 35 | `MimirPrefetcher.LifecycleStartStop` | Prefetcher Start and Stop; the sender reports Stopped | all |
| 36 | `MimirPrefetcher.LifecycleStartStopTwice` | A second Start zeroes the control block the first session left; an unknown command left in ring slot 0 is never read | all |
| 37 | `MimirPrefetcher.LifecycleBusySenderFailsStart` | Start fails on a sender CCE that runs a receiver Program, before it launches or zeroes anything | all |
| 38 | `MimirPrefetcher.LifecycleQueueRejectsReceiverOnSender` | Queue rejects a GCB whose receiver is a sender CCE | all |
| 39 | `MimirPrefetcher.LifecycleWaitCq` | A WAIT_CQ holds the sender until the command queue writes its slot | all |
| 40 | `MimirPrefetcher.LifecycleDrainAtStop` | Stop's DRAIN holds the sender until an unacked credit is acked, then STOP | all |
| 41 | `MimirPrefetcher.PrefetchOnePagePerSessionLocal0To1` | One prefetch, CCE 0 to CCE 1, then Start, Queue and Stop again | all |
| 42 | `MimirPrefetcher.PrefetchOnePagePerSessionLocal3To2` | One prefetch, CCE 3 to CCE 2, on the second Mimir | 2x only |
| 43 | `MimirPrefetcher.PrefetchOnePagePerSessionD2d0To2` | One prefetch, CCE 0 to CCE 2, one D2D hop | 2x only |
| 44 | `MimirPrefetcher.PrefetchOnePagePerSessionD2d3To1` | One prefetch, CCE 3 to CCE 1, one D2D hop | 2x only |
| 45 | `MimirPrefetcher.PrefetchFanOut0To123` | Fan-out from CCE 0 to CCEs 1, 2 and 3 | 2x only |
| 46 | `MimirPrefetcher.PrefetchFanOut3To012` | Fan-out from CCE 3 to CCEs 0, 1 and 2 | 2x only |
| 47 | `MimirPrefetcher.PrefetchTwoSenders01To2` | Two senders, CCEs 0 and 1, into two GCBs on CCE 2 | 2x only |
| 48 | `MimirPrefetcher.PrefetchTwoSenders02To3` | Two senders, CCEs 0 and 2, into two GCBs on CCE 3 | 2x only |
| 49 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/SubChunkLocal0To1` | A block larger than a stage third, in sub-chunks through two stage slots | all |
| 50 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/SubChunkD2d0To2` | The same across D2D | 2x only |
| 51 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/BelowStageThirdLocal0To1` | Blocks smaller than a stage third, two per stage slot | all |
| 52 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/BelowStageThirdD2d0To2` | The same across D2D | 2x only |
| 53 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/PartialStageChunkLocal0To1` | A round that ends in a partial stage chunk | all |
| 54 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AboveStageThirdLocal0To1` | A block above the default stage third, in one-tile sub-chunks | all |
| 55 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AboveStageThirdD2d0To2` | The same across D2D | 2x only |
| 56 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/PageSplitLocal0To1` | One Queue call split into two request pages; local only, since the split is on the host | all |
| 57 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AbaLocal0To1` | Tensors A, B, A with different page sizes; resize and wrap | all |
| 58 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AbaD2d0To2` | The same across D2D | 2x only |
| 59 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/WrapAndResizeLocal0To1` | Resize padding and wraps in a FIFO the page size does not divide | all |
| 60 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/WrapAndResizeD2d0To2` | The same across D2D | 2x only |
| 61 | `TensorPrefetcher/MimirPrefetcherGcbSwitch.Delivers/GcbSwitchLocal0To1` | GCB 1, then 2, then 1, resuming mid-FIFO | all |
| 62 | `TensorPrefetcher/MimirPrefetcherGcbSwitch.Delivers/GcbSwitchD2d0To2` | The same across D2D | 2x only |
| 62a | `TensorPrefetcher/MimirPrefetcherGcbSwitch.Delivers/ReceiverSwitch0To1Then2` | GCB 1 to CCE 1, then GCB 2 to CCE 2 across D2D, then GCB 1 again | 2x only |
| 63 | `MimirPrefetcher.PrefetchFenceHoldsDeliveryLocal0To1` | A WAIT_CQ fence holds delivery until the slot is written | all |
| 64 | — | Removed on 2026-10-29: trace capture needs fast dispatch, and Mimir runs slow dispatch only | — |
| 65 | `MimirPrefetcher.QueueRejectsKRowMajorTensor` | Queue rejects a K-row-major tensor | all |
| 66 | `MimirPrefetcher.QueueRejectsStreamingTensor` | Queue rejects a streaming tensor | all |
| 67 | `MimirPrefetcher.QueueRejectsRangePastTensor` | Queue rejects a request past the tensor | all |
| 68 | `MimirPrefetcher.QueueRejectsRangePastView` | Queue rejects a request past the DRAM view | all |
| 69 | `MimirPrefetcher.PrefetchRecoversAfterMismatchD2d2To0` | Wrong-seed validator reports MISMATCH on every word of the tensor's page; the sender stops OK | 2x only |
| 70 | `MimirPrefetcher.PrefetchRecoversAfterCreditTimeoutD2d1To3` | A request larger than the FIFO gives CREDIT_TIMEOUT; the host drops the requests after it and Stop throws, naming the sender | 2x only |
| 71 | `MimirPrefetcher.PrefetchRecoversAfterDrainTimeoutD2d1To3` | A receiver holds a request, so Stop's DRAIN gives DRAIN_TIMEOUT and Stop throws | 2x only |

Notes on the rows:
- Rows that need a second Mimir skip at runtime on `mimir_1x1`.
- Every row that checks a receiver's FIFO state also checks the 256 B after the FIFO end against a pattern written before the run, so a write past the FIFO end fails the row.
- The `PrefetchRecoversAfter*` rows (69-71) run last. Rows 70-71 end with a sender timeout and a stalled receiver; row 69 with a validator mismatch.
- For a JUnit report, add `--gtest_output=xml:<file>` to each command.

## Upstream data-movement tests on Mimir: dispositions

Every source in the existing DM suite gets exactly one disposition, picked by the first value in the rule below that fits. The ports are `TEST_P` suites instantiated with the `DmSuitePort` prefix, and every port fixture starts with `MimirDm`, so their filters are written `*MimirDm<Name>*`. The 44 cases of the first port all passed on `mimir_2x_package` on 2026-10-21 (soc-c-26) and on `mimir_1x1` on 2026-10-21. On 2026-10-20 (soc-zebu-03), `MimirDmIdma.Copy/RemoteTl1` and `Copy/RemoteGddr` failed on `mimir_2x_package` with `ack_timeout`: the data never landed, on the write and the read legs, and the cause is not known. A later revision then changed every port suite and added seven cases (51 in all). On `mimir_2x_package` all 51 and the `MimirKernelLoad` cases passed on 2026-10-22 (soc-zebu-04, one fresh session). An earlier session on 2026-10-21 (soc-zebu-01) passed 23 cases, then `Copy/OwnTl1` failed (acked never came, the data never landed) and `DramUnary.ReaderWriterHandshake/Cce2` hit the host deadline; both passed on soc-zebu-04, so they are recorded as that session's failures, cause not known. On `mimir_1x1` the 46 follow-up cases that run there and the `MimirKernelLoad` cases passed on 2026-10-23 (soc-zebu-03, one fresh session); the other 5 need two Mimirs or CCEs 2 and 3.

| Value | Rule |
|---|---|
| `port` | The upstream test is ported to a Mimir analogue |
| `covered` | The upstream test cannot run, but a Mimir test written for another purpose (kernel load, local or remote DRAM or TL1, Tensor Prefetcher, cache, atomics or transaction IDs) exercises its intent on that descriptor |
| `blocked` | An analogue is possible but needs a named capability or runtime change |
| `no-analogue` | Mimir has no analogue at all: no multicast, no NoC atomics, no PCIe, Eth, compute, L1 banks or worker cores |
| `out-of-scope` | Not a device data-movement test: a helper, host-CPU or mock-device test, or a separate executable |

Source paths are relative to `tests/tt_metal/tt_metal/`. A `gap` in a covered-by column means no existing test covers that descriptor. A `—` means the row has no evidence column. Covered-by filters are glob patterns that allow for the instantiation prefix and parameter suffix. A 2x-only test is never listed under 1x1.

**`data_movement/sources.cmake` (39 tests plus `dm_common.cpp`)**

| Source | Disposition | Covered by: 2x | Covered by: 1x1 | Reason / prerequisite | Case overrides |
|---|---|---|---|---|---|
| `data_movement/dram_unary/test_unary_dram.cpp` | port | `*MimirDmDramUnary*`; interim, by iDMA: `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*`, `*MimirLocalDram.LoopbackChecksIntermediate*` | `*MimirDmDramUnary*`; interim, by iDMA: `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*`, `*MimirLocalDram.LoopbackChecksIntermediate*` | Two harts of one CCE, SPA data path, DRAM semaphore; the interim MimirLocalDram cases move the data by iDMA, not by hart loads and stores | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/one_to_all/test_one_to_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.PrefetchFanOut0To123`, `MimirPrefetcher.PrefetchFanOut3To012` | gap | Unicast cases only; 1x1 gap: one_to_all unicast fan-out | Multicast* and *Semaphore* cases: no-analogue (no multicast, no NoC semaphore atomics); `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/one_to_all/test_multicast_schemes.cpp` | no-analogue | — | — | No multicast on Mimir | — |
| `data_movement/loopback/test_loopback.cpp` | covered | `*MimirLocalTl1.SameMimirCopy*` | `*MimirLocalTl1.SameMimirCopy*` | No CCE NoC; same-chiplet TL1 copy is the intent | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/reshard_hardcoded/test_reshard_hardcoded.cpp` | no-analogue | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/deinterleave_hardcoded/test_deinterleave_hardcoded.cpp` | no-analogue | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/conv_hardcoded/test_conv_hardcoded.cpp` | no-analogue | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/one_to_one/test_one_to_one.cpp` | covered | `*MimirLocalTl1*`, `D2dDirect/MimirRemoteTl1.WriteRead/*`, `D2dRouted/MimirRemoteTl1.WriteRead/*`, `MimirPrefetcher.PrefetchOnePagePerSessionD2d0To2`, `MimirPrefetcher.PrefetchOnePagePerSessionD2d3To1`, `MimirPrefetcher.PrefetchOnePagePerSessionLocal3To2` | `*MimirLocalTl1*`, `MimirPrefetcher.PrefetchOnePagePerSessionLocal0To1` | No CCE NoC; TL1-to-TL1 copy and prefetch cover the intent | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/one_from_one/test_one_from_one.cpp` | covered | `*MimirLocalTl1*`, `D2dDirect/MimirRemoteTl1.HostInitRemoteRead/*`, `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/*` | `*MimirLocalTl1*`; gap: host-initiated remote read | 1x1 gap: host-initiated remote read | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/core_bidirectional/test_core_bidirectional.cpp` | covered | `*MimirRemoteTl1Bidir*.Concurrent*` | gap | PacketSizes only (7 of 9 TEST_F cases skip upstream); 1x1 gap: same-chiplet instances are separate launches, not concurrent | — |
| `data_movement/one_from_all/test_one_from_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.PrefetchTwoSenders01To2`, `MimirPrefetcher.PrefetchTwoSenders02To3` | gap | 1x1 gap: fan-in/all-to-all on mimir_1x1 | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/all_to_all/test_all_to_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.PrefetchTwoSenders01To2`, `MimirPrefetcher.PrefetchTwoSenders02To3` | gap | 1x1 gap: fan-in/all-to-all on mimir_1x1 | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/all_from_all/test_all_from_all.cpp` | covered | `*MimirRemoteMixed.FanOutFanIn*`, `MimirPrefetcher.PrefetchTwoSenders01To2`, `MimirPrefetcher.PrefetchTwoSenders02To3` | gap | 1x1 gap: fan-in/all-to-all on mimir_1x1 | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/interleaved/test_interleaved.cpp` | blocked | — | — | Interleaved bank addressing from a CCE (needs a runtime change: int64 device bank offsets or 64-bit per-view args) | Sync cases: blocked (CBs); L1 cases: no-analogue (no L1 banks) |
| `data_movement/one_packet/test_one_packet.cpp` | covered | `D2dDirect/MimirRemoteTl1.SizeAlignment/*` | gap | SizeAlignment runs with read_back=false; 1x1 gap: sub-word and odd-size TL1 transfers; iDMA max packet size is in iDMA probe case 4 | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/interleaved_to_sharded_hardcoded/test_interleaved_to_sharded_hardcoded.cpp` | no-analogue | — | — | Hardcoded Tensix/CB kernel; no worker cores | — |
| `data_movement/multi_interleaved/test_multi_interleaved.cpp` | blocked | — | — | Interleaved bank addressing from a CCE (needs a runtime change: int64 device bank offsets or 64-bit per-view args); CBs | — |
| `data_movement/dram_sharded/test_dram_sharded.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirRemoteDram.ReadToTl1*`, `MimirPrefetcher.PrefetchOnePagePerSessionLocal0To1`, `MimirPrefetcher.PrefetchOnePagePerSessionLocal3To2`, `MimirPrefetcher.PrefetchOnePagePerSessionD2d0To2`, `MimirPrefetcher.PrefetchOnePagePerSessionD2d3To1` | `*MimirLocalDram.CceReadsDramToTl1*`, `MimirPrefetcher.PrefetchOnePagePerSessionLocal0To1` | Intent: read DRAM shards into TL1 | *Trid* cases: covered by `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr*`; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/transaction_id/test_transaction_id.cpp` | covered | `IdmaTransactionId/MimirTridIdma.BarrierThenDependent*`, `IdmaTransactionId/MimirTridIdma.UniqueIds*` | `IdmaTransactionId/MimirTridIdma.BarrierThenDependent*`, `IdmaTransactionId/MimirTridIdma.UniqueIds*` | iDMA command-buffer trids: unique trids, per-trid ack wait, then the dependent transfer; final data gates; the dependent transfer's data gates too (the read-after-write destination, and the write-after-read read checked at ack) | `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `data_movement/direct_write/test_direct_write.cpp` | covered | `D2dDirect/MimirRemoteTl1.SizeAlignment/*` | gap | 4 B case of SizeAlignment; upstream tests inline dword writes, which the CCE does not have, so the 4 B CPU store is a stand-in; 1x1 gap: sub-word direct write | Mcast cases: no-analogue |
| `data_movement/pcie_read_bw/test_pcie_read_bw.cpp` | no-analogue | — | — | No PCIe | — |
| `data_movement/pcie_write_bw/test_pcie_write_bw.cpp` | no-analogue | — | — | No PCIe | — |
| `data_movement/device_pcie_loopback/test_device_pcie_loopback.cpp` | no-analogue | — | — | No PCIe | — |
| `data_movement/pcie_mid_cleanup/test_pcie_mid_cleanup.cpp` | no-analogue | — | — | No PCIe | — |
| `data_movement/atomics/test_atomic_semaphore_bandwidth.cpp` | covered | `CceAtomics/MimirAtomicLocal.Hotspot*` | `CceAtomics/MimirAtomicLocal.Hotspot*` | The value and count sweep as amoadd.w on a cached own-TL1 word with one, two and eight harts; the NoC-atomic form has no ITN equivalent | — |
| `data_movement/multicast_atomics/test_multicast_atomic_semaphore.cpp` | no-analogue | — | — | No multicast, no NoC atomics | — |
| `data_movement/noc_api_latency/test_noc_api_latency.cpp` | no-analogue | — | — | NoC API latency; no CCE NoC | — |
| `data_movement/noc_estimator_tests/test_noc_estimator.cpp` | no-analogue | — | — | NoC estimator; no CCE NoC | — |
| `data_movement/dram_neighbour/test_dram_neighbour.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirRemoteDram.ReadToTl1*` | `*MimirLocalDram.CceReadsDramToTl1*` | Intent: read the near DRAM bank | — |
| `data_movement/matmul/test_matmul_1d.cpp` | no-analogue | — | — | DM multicast and NoC semaphores; no CCE NoC | — |
| `data_movement/matmul/test_matmul_1d_v2.cpp` | no-analogue | — | — | DM multicast and NoC semaphores; no CCE NoC | — |
| `data_movement/matmul/test_matmul_2d.cpp` | no-analogue | — | — | DM multicast and NoC semaphores; no CCE NoC | — |
| `data_movement/quasar_cache/test_quasar_cache.cpp` | port | `*MimirDmCache*` | `*MimirDmCache*` | cce_cache_ops on TL1; cache probes done. `QuasarL1DCacheOps.InvalidateFull` is `L1InvalidateFull`; `L2InvalidateFreshRead` is the L2-only fresh read, checked by a cached hart read where upstream reads back on the host | — |
| `data_movement/quasar_cache_perf/test_quasar_cache_perf.cpp` | port | `*MimirDmCachePerf*` | `*MimirDmCachePerf*` | Functional only, nightly; no timing asserts; cache probes done | — |
| `data_movement/quasar_examples/quasar_idma/test_idma_example.cpp` | port | `*MimirDmIdma*`, `*MimirLocalDram*` | `*MimirDmIdma*`, `*MimirLocalDram*` | cce_idma_copy; iDMA probes done. `*MimirLocalDram*` is the local-GDDR matrix (every CCE, hart, view, size and alignment) on `kModeTridBatch`; `MimirDmIdma` uses the single-command modes | IDMA_1D_Strided: port as `Strided1D` (own TL1) and `Strided1DLocalGddr` (local GDDR source), with 64 B elements, and as `Strided1DSubLine` and `Strided1DSubLineLocalGddr` with upstream's 8 B elements and 16 B source stride; remote strided endpoints not ported |
| `data_movement/quasar_examples/quasar_addrgen/test_addrgen_example.cpp` | port | `*MimirDmAddrgen.Pattern/Strided*:*MimirDmAddrgen.Pattern/Face_*` | `*MimirDmAddrgen.Pattern/Strided*:*MimirDmAddrgen.Pattern/Face_*` | Pop addresses into TL1 and compare on host; `addrgen_reset` probe done; explicit-count pops (`pop_*_addrgen<ADDRGEN_0>(1)`) work around tt-metal #59983 (the count-less pop encodes rs1 = x1); the banking registers stay at reset, as upstream: a one-bank banking loop from a probe run that also changed the pops was removed after every addrgen, im2col and strided iDMA case passed without it on 2026-10-22 | Interleaved_Banking: no-analogue (encodes ATT endpoint IDs 20..23) |
| `data_movement/quasar_examples/quasar_addrgen/test_tensor_accessor_addrgen.cpp` | covered | `*MimirDmAddrgen.Pattern/Strided*:*MimirDmAddrgen.Pattern/Face_*`, `*MimirDmMeshBufferDram*` | `*MimirDmAddrgen.Pattern/Strided*:*MimirDmAddrgen.Pattern/Face_*`, `*MimirDmMeshBufferDram*` | Needs TensorAccessor buffers, dataflow buffers and Tensix workers; the HW addrgen walks are covered by `MimirDmAddrgen` and interleaved DRAM pages by `MimirDmMeshBufferDram` | — |
| `data_movement/quasar_examples/quasar_im2col/test_im2col_example.cpp` | port | `*MimirDmAddrgen.Pattern/Im2Col_*` | `*MimirDmAddrgen.Pattern/Im2Col_*` | Pop addresses into TL1 and compare on host; `addrgen_reset` probe done; the loop index resets to 0 at loop end, as the RTL notes describe (neosim overlay-addrgen-rtl-semantics.md; tensix-ip_docs overlay/rocc/address_generator.md); `addrgen_api.hpp:38-39` says modulo and is out of date | — |
| `data_movement/quasar_narrow_row/test_pack_untilize_narrow_row.cpp` | no-analogue | — | — | Compute; no Tensix | — |
| `data_movement/dm_common.cpp` | out-of-scope | — | — | Helper, not a test | — |

**Sources outside `data_movement/`**

| Source | Disposition | Covered by: 2x | Covered by: 1x1 | Reason / prerequisite | Case overrides |
|---|---|---|---|---|---|
| `api/test_direct.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | ReaderOnly and WriterOnly only | ReaderWriter: blocked (DFB); Datacopy cases: no-analogue; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `api/test_runtime_args.cpp` | port | `*MimirKernelLoad.ArgsReplaceStale*`, `*MimirDmRuntimeArgs*` | `*MimirKernelLoad.ArgsReplaceStale*`, `*MimirDmRuntimeArgs*` | TensixIllegalTooManyRuntimeArgs and TensixIllegallyModifyRTArgs ported as `MimirDmRuntimeArgs` cases; ArgsReplaceStale keeps the arg count, so it does not cover the second | Legacy RTA/CRTA: covered by `MimirKernelLoad.ArgsReplaceStale` (cce_args_echo); Quasar CRTA: blocked (KernelSpec is TENSIX-only); Compute and Eth cases: no-analogue; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_alias_dataflow_buffer.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue |
| `api/dataflow_buffer/test_borrowed_memory_dataflow_buffer.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue |
| `api/dataflow_buffer/test_dataflow_buffer_apis.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue |
| `api/dataflow_buffer/test_dataflow_buffer_base.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_configs.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_disjoint_slots.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue |
| `api/dataflow_buffer/test_dataflow_buffer_edge_cases.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_intra.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_multinode.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue; `_2_0` variants: blocked (KernelSpec is TENSIX-only) |
| `api/dataflow_buffer/test_dataflow_buffer_overrides.cpp` | blocked | — | — | `supports_dfbs=false` on DRAM cores; DFB spec is TENSIX-only | DM-Tensix and INTRA cases: no-analogue |
| `api/dataflow_buffer/dfb_init_timing_bench.cpp` | out-of-scope | — | — | Separate executable (`api/CMakeLists.txt:107`) | — |
| `api/test_scoped_lock_cache.cpp` | port | `*MimirDmScopedLockClm*` | `*MimirDmScopedLockClm*` | CLM cases (`:540-555`) on two harts of one CCE; cache probes done | DFB and handshake cases: blocked; Scratchpad and LTA cases: no-analogue (Tensix-DM abstractions) |
| `test_quasar_trace.cpp` | blocked | — | — | Trace needs fast dispatch | — |
| `test_quasar_mesh_workloads.cpp` | blocked | — | — | KernelSpec is TENSIX-only (`program_spec.cpp:4047`); every workload pairs DM kernels with a compute kernel | Compute-output checks: no-analogue; TestInterleavedWorkloadsAcrossTwoCQs also needs a second CQ |
| `test_quasar_compute_kernels.cpp` | no-analogue | — | — | Compute; no Tensix | — |
| `llk/test_quasar_mailboxes.cpp` | no-analogue | — | — | Compute/Tensix mailboxes | — |
| `api/metal2_host_api/unit_tests/*` | out-of-scope | — | — | Metal 2.0 mock CPU tests; with the two rows below they are the ProgramSpec and ProgramRunArgs tests (no test_program_spec.cpp / test_program_run_args.cpp exists) | — |
| `api/metal2_host_api/integration_tests/{binding_loopbacks,kernel_args_loopbacks,scratchpad,mesh_workload_factories}.cpp` | blocked | — | — | Device tests built on KernelSpec, which is TENSIX-only (`program_spec.cpp:4047`) | — |
| `api/metal2_host_api/integration_tests/scratchpad_fast_dispatch.cpp` | blocked | — | — | Fast dispatch, and KernelSpec is TENSIX-only | — |
| `api/metal2_host_api/integration_tests/{compute_semaphore,llk_operand_mul}.cpp` | no-analogue | — | — | Compute; no Tensix | — |
| `api/metal2_host_api/kernel_compilation_tests/*` | out-of-scope | — | — | Kernel compilation against a mock device; nothing runs on a device | — |
| `test_dram_loopback_single_core.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | Tensix loopback; Mimir DRAM-to-TL1-to-DRAM copies cover the intent | Eth cases: no-analogue |
| `api/test_dram.cpp` | covered | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | `*MimirLocalDram.CceReadsDramToTl1*`, `*MimirLocalDram.CceWritesTl1ToDram*` | Tensix loopback x2 and double-buffered | Eth cases: no-analogue |
| `test_single_dm_l1_write.cpp` | covered | `KernelLoad/MimirKernelLoad.*`, `MimirKernelLoad.SameBinaryAllCcesUnicast` | `KernelLoad/MimirKernelLoad.*`, `MimirKernelLoad.SameBinaryAllCcesUnicast` | SingleDmL1Write by the per-CCE `MimirKernelLoad` cases | GridProbeStep0 and FullGridDmL1Write_L1a: covered by `MimirKernelLoad.SameBinaryAllCcesUnicast` (every CCE); FullGridCompute_L1c and GridMulticastFanOut: no-analogue |
| `test_quasar_mesh_buffers.cpp` | port | `*MimirDmMeshBufferDram*` | `*MimirDmMeshBufferDram*` | MeshBufferWriteReadDRAM and MeshBufferMultipleWriteReadRoundsDRAM (`:67,88`); host DRAM MeshBuffer placement probe done (see Data-movement port probe results) | L1 and sharded L1: no-analogue; MeshBufferCrossCQWriteReadRoundsDRAM: blocked (multi-CQ) |
| `test_quasar_semaphores.cpp` | no-analogue | — | — | NoC semaphore increments and NoC reads; no NoC semaphore atomic on Mimir | — |
| `test_quasar_events.cpp` | blocked | — | — | KernelSpec is TENSIX-only (`program_spec.cpp:4047`); single-CQ events work in slow dispatch | CrossCQEventHandoff* and RecordEventToHostFromBothCQs: also need a second CQ |
| `test_quasar_fds.cpp` | no-analogue | — | — | No Tensix FDS | — |
| `test_quasar_sub_devices.cpp` | no-analogue | — | — | No worker cores | — |
| `llk/test_quasar_bfd_datacopy.cpp` | no-analogue | — | — | Compute; no Tensix | — |
| `noc/test_quasar_att_address.cpp` | out-of-scope | — | — | Host CPU test | — |
| `noc/test_quasar_att_config.cpp` | out-of-scope | — | — | Host CPU test | — |
| `noc/test_quasar_att_image.cpp` | out-of-scope | — | — | Host CPU test | — |
| `api/test_dram_sender_global_cb.cpp` | no-analogue | — | — | Receivers are worker cores; `build_dram_sender_mapping` throws on Quasar (`dram_sender_topology.cpp:36-40`); Mimir DRAM-receiver GCB is covered by the `MimirPrefetcher` `GcbLayout*`, `GcbSmoke*`, `Prefetch*` and `PrefetchRecoversAfter*` tests | — |
| `api/test_prefetcher_pipe.cpp` | no-analogue | — | — | Metal 2.0 KernelSpec worker kernels; no worker cores | — |
| `api/test_prefetcher_pipe_dram_sender.cpp` | no-analogue | — | — | Same throw as above; the CCE sender returns BadCmd for a PrefetcherPipe target (`tensor_prefetcher_cce.cpp:74-75` for DRAIN, `:242-244` for PREFETCH) | — |

### Data-movement port probe results

Each probe was run on the `mimir_2x_package` emulator (soc-c-30 and soc-zebu-03). Columns: probe, result (yes/no/hang), date, commit and run log (paths are relative to the session scratchpad). The probes ran on 2x only; the ports that replaced them passed on `mimir_1x1` on 2026-10-21.

| Probe | Result | Date | Run log and commit | Rows that depend on it |
|---|---|---|---|---|
| Cache host-visibility: are cached TL1 stores visible to the host without a flush? | no (host saw 0 of 64 cached stores without a flush; after a flush it sees them) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_probes/mimir_2x_package_HostSeesCachedStores.log` | quasar_cache, quasar_cache_perf, test_scoped_lock_cache |
| `tt.cache.*` trap: do the instructions trap (a trap is a hang)? | no (no trap; all four forms returned) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_probes/mimir_2x_package_CacheInstrTrap.log` | quasar_cache, quasar_cache_perf, test_scoped_lock_cache |
| iDMA case 1: own TL1 to own TL1 (VC 5), package-global SPA operands | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_OwnTl1.log` | quasar_idma |
| iDMA case 2: NoC-mode command-buffer request (`idma_en=0`) | no (never showed free space, nothing issued, no hang), so a CCE has no usable NoC interface: its command buffers' NoC mode, which the NoC dataflow API uses, issues nothing. NoC-based upstream tests are covered by CCE-native tests or have no analogue | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_NocCmdbuf.log` | quasar_idma and every `covered` NoC movement row |
| iDMA case 3a: sibling TL1 (write VC 2, read VC 5) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_SiblingTl1.log` | quasar_idma |
| iDMA case 3b: local GDDR0 (write VC 0, read VC 5) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_LocalGddr.log` | quasar_idma |
| iDMA case 3c: remote TL1, kernel loading the remote destination | hang (no status in 400 s; session wedged) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_RemoteTl1.log` | quasar_idma |
| iDMA case 3c rerun: remote TL1, host-verified (write VC 3, read VC 5) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_RemoteTl1_run2.log` | quasar_idma |
| iDMA case 3d: remote GDDR, host-verified | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_RemoteGddr.log` | quasar_idma |
| iDMA case 4: maximum packet size (8 KiB) | yes (landed, acked) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_idma/mimir_2x_package_MaxPacket.log` | quasar_idma, one_packet |
| `addrgen_reset` hang | no hang (returns); pops advanced once both the explicit count and the one-bank loop were in place (one run changed both) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_probes/mimir_2x_package_AddrgenReset.log, ..._AddrgenDiag.log, ..._AddrgenReset_fix.log` | quasar_addrgen, quasar_im2col |
| Host DRAM MeshBuffer placement | yes (banks 0-3 at the expected address in their own view; round trip matched) | 2026-10-20 | `3ef0a46e42a` + uncommitted DM-suite port changes; `scratchpad mm8_probes/mimir_2x_package_InterleavedPlacement.log` | test_quasar_mesh_buffers |

### Hand checks

The table is kept by hand, with no checker script. These four checks are redone whenever the table changes:
1. Every `data_movement/sources.cmake` source has exactly one row.
2. Every covered-by filter that is not `gap` or `—` matches at least one test in `mimir_tests --gtest_list_tests`. A 2x-only test also lists on 1x1, so the 1x1 column names only tests that run there.
3. Every `DmSuitePort/` test listed by `--gtest_list_tests --gtest_filter='DmSuitePort/*'` is matched by some covered-by filter and by some run row.
4. Every test listed by `--gtest_list_tests --gtest_filter='CceCache/*:CceAtomics/*:IdmaTransactionId/*:CceKernelCompile/*'` is matched by exactly one row from 81 to 130.

## Running the data-movement ports

These rows run the `DmSuitePort` suites by hand. They continue from row 71, with the same environment and timeout prefix as rows 2b onward. Each case runs as its own invocation (list them with `--gtest_list_tests --gtest_filter='DmSuitePort/*'`, with the descriptor selected by `TT_METAL_SOC_DESC`, or by `TT_METAL_EMU_SOC_DESC` with `TT_METAL_EMU_SERVER` also set: the `MimirDmMeshBufferDram` case names end in the descriptor name, and without a selected descriptor they end in `Unconfigured` and match nothing). Add `--gtest_output=xml:<dir>/<descriptor>_<row>.xml` to each command; read the reports against the covered-by columns above. On `mimir_1x1` (2026-10-21), 80 cases ran (the 44 `DmSuitePort/*` cases of the first port and the 36 Tensor Prefetcher rows marked "all"): 75 passed, 5 skipped, 0 failed. The skips are `MimirDmIdma.Copy/RemoteTl1` and `Copy/RemoteGddr` (need two Mimirs), `MimirDmDramUnary.ReaderWriterHandshake/Cce2` and `Cce3` (1x1 has 2 CCEs), and `TraceCaptureReplay` (always skips). On `mimir_1x1` (2026-10-23, soc-zebu-03, at `3a3d04f9bd2`), the later revision's 46 `DmSuitePort/*` cases that run there and the 9 `MimirKernelLoad` cases of row 3 that run there all passed (55 of 55). The 5 `DmSuitePort/*` cases that skip there (`MimirDmIdma.Copy/RemoteTl1`, `Copy/RemoteGddr` and `Copy/RemoteTl1KernelVerified`, and `MimirDmDramUnary.ReaderWriterHandshake/Cce2` and `Cce3`) were not run. Rerun the `MimirKernelLoad` cases (row 3) with these rows: the later revision changed its echo kernel.

CI: `mimir_tests` is built only with `TT_UMD_BUILD_GRENDEL_JTAG=ON`, which is off by default, and no CI runner has a Mimir emulator. So no CI job runs them; these rows are run by hand.

These four cases test a contract upstream assumes; if one fails on a correct emulator, record it as a finding, not a test bug: `MimirDmCache.Op/HartToHartCoherent` (a consumer's cached copy sees the producer's flushed stores without an invalidate), `MimirDmCachePerf.WritePath/CachedFastFlush` (no fence before the FLUSH64 writes), `MimirDmScopedLockClm.CacheOps/FlushOnRelease` (the release flush is in TL1 at the first read) and `MimirDmIdma.Copy/RemoteTl1KernelVerified` (the kernel loads a remote TL1 destination). All four passed on `mimir_2x_package` on 2026-10-22.

New rule: after any row that ends on a host deadline, start a fresh emulator session and run bring-up before the next row (MIM-601).

| # | Filter | Covers | Descriptors |
|---|---|---|---|
| 72 | `DmSuitePort/MimirDmRuntimeArgs.*` | Too many runtime args and illegal modification are rejected, the stored args are unchanged after a rejection, and a kernel with args at the ceiling launches | all |
| 73 | `DmSuitePort/MimirDmMeshBufferDram.*` | Host DRAM MeshBuffer: interleaved placement per bank, write/read with a per-run pattern, repeated rounds | all |
| 74 | `DmSuitePort/MimirDmIdma.*` | iDMA copies: own TL1, sibling TL1, local GDDR0 and GDDR1, max packet, strided gather of 64 B and of 8 B elements from own TL1 and from local GDDR; remote TL1 (host-verified and kernel-verified) and remote GDDR on 2x only; guards before and after each destination. RemoteTl1 and RemoteGddr failed with `ack_timeout` on 2026-10-20 (soc-zebu-03) and passed on 2026-10-21 (soc-c-26) and 2026-10-22 (soc-zebu-04); OwnTl1 failed the same way on 2026-10-21 (soc-zebu-01) and passed on 2026-10-22; the cause is not known. RemoteTl1KernelVerified hung with no status on its first run (an early probe, 2026-10-20, soc-zebu-03) and passed six times on 2026-10-22, all in one soc-zebu-04 session; it was disabled until then | all (Remote* 2x only) |
| 75 | `DmSuitePort/MimirDmCache.*` | Cache ops on TL1: flush, invalidate (line, whole L1 D$, L2 line), fresh read (L1 and L2, L2 only), coherent hart-to-hart visibility; every store case checks the hart's cached view of its stores. Run `Op/L1InvalidateFull` last: the whole-L1 D$ discard can drop the kernel's own stack | all |
| 76 | `DmSuitePort/MimirDmCachePerf.*` | Write paths, functional only; CachedFastFlush has no fence before the FLUSH64 writes, as upstream | all |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided*:DmSuitePort/MimirDmAddrgen.Pattern/Face_*` | Addrgen strided 1D, 2D and face patterns, popped into TL1 and compared on the host, with the banking registers at reset | all |
| 78 | `DmSuitePort/MimirDmAddrgen.Pattern/Im2Col_*` | Im2col full image and dilation 1, compared on the host | all |
| 79 | `DmSuitePort/MimirDmDramUnary.*` | Reader and writer harts on one CCE, GDDR to TL1 to GDDR with a DRAM semaphore; the semaphores are at the runtime's addresses, initialised at launch and end at the chunk count; guards before and after the destination and the staging ring | all |
| 80 | `DmSuitePort/MimirDmScopedLockClm.*` | Scoped-lock CLM cases: invalidate on acquire, flush on release (in TL1 at the first read), locked range only; unlocked lines stay cached in both invalidate cases | all |

## CCE cache, atomics and transaction-ID tests

The `CceCache` suites cover the CCE caches, `CceAtomics` covers atomics and `IdmaTransactionId` covers transaction IDs; `CceKernelCompile` only compiles the kernels they add or extend.

### Cache, atomics and transaction-ID probe results

Each probe ran on the `mimir_2x_package` emulator (soc-zebu-04) on 2026-10-22, on all four CCEs, one case per invocation, against `61d03f6e674` plus the uncommitted cache, atomics and transaction-ID test changes. Kernels were built from a frozen copy of `test_kernels/cce`. Logs are in `/localdev/sagarwal/mm4x_runs_2026-10-22/mm4x/probe_logs_final/`.

| Probe | Result | Rows that depend on it |
|---|---|---|
| `CceCache/MimirCacheProbe.EightHartLaunch/*` | yes: all eight harts of every CCE ran a kernel and posted status | Every 8-hart case in these suites |
| `CceCache/MimirCacheProbe.SnoopConfig/*` | `NOC_SNOOP_TL_MASTER_CFG` = 0x6 (default) on every CCE. L2 CONFIGURATION = 0x06060804: 4 banks, 8 ways, 64 sets, 64 B lines (128 KiB). Chippy's reset value (8 banks, 32 sets) and the "4-way" comment in `overlay_addresses.h` do not match it | `MimirCacheCoherenceRace.DirtyLineRace` starting value; `MimirCacheStress` sizes |
| `CceCache/MimirCacheProbe.L2Geometry/*` | Masks identical on every CCE. `cflush.d.l1 x0` alone never reached TL1 (bit 0 never set), so it stops in L2. Replacement is not LRU: at strides of 16 KiB and more the dirty target was evicted after as few as 3 same-set reads and survived some counts above 8, the same way on every CCE. 8 KiB strides (two sets) evicted it at every count from 17. Conflict stride 16 KiB | `MimirCacheStress.ReuseThrash` strides; `DirtyEvictionPartialLine` and `DirtyLineRace` eviction sweeps (24 same-set reads). Probe removed on 2026-10-30; the sizes are kept as constants |
| `CceAtomics/MimirAtomicProbe.TwoHartAmo/*` | yes: 2 harts x 64 amoadd.w and 64 LR/SC adds; totals, return-value sums and the host view all correct. 0 SC failures, so the two harts may barely have overlapped | Every multi-hart AMO case |
| `IdmaTransactionId/MimirTridProbe.Width/*` | TR_ID written as 17 reads back 17 (the register keeps bit 4), but the command counts on trid 1: pending 1 on trid 1, 0 on trid 17. Consistent with MIM-539 (TRID[4] dropped); an out-of-range counter read would look the same | Probe removed on 2026-10-30: MIM-539 is Future Fix, so the 16-ID limit stays; range tests use trids 0-15 |
| `IdmaTransactionId/MimirTridProbe.Scope/*` | shared: while hart 0 had 11 commands pending on trid 5, hart 1 of the same CCE read 11 on its own command buffer. Pending counters are per CCE, not per hart. Whether CCEs share them is not tested | Probe removed on 2026-10-30: no test depends on the scope. `MimirTridIdma.MultiHart` gives each hart its own trids, which is safe either way |

A first pass of these probes compiled the live kernel sources while they were being edited (a relative kernel path resolves against the working directory first, `kernel.cpp` `resolve_path`). That pass was discarded. Its Scope runs failed because the observer kernel was mid-edit.

### Cache, atomics and transaction-ID run order

Rows 81 onward continue from row 80. They use the same environment and timeout prefix as rows 2b onward, plus a fresh session and bring-up before row 81. Run each case as its own invocation (`--gtest_list_tests` with the row's filter lists them) and add `--gtest_output=xml:<dir>/<descriptor>_<row>.xml`. Every suite here is a `TEST_P` over the CCE index (`/Cce0` to `/Cce3`), except where the row says otherwise; 2x-only cases skip at runtime on `mimir_1x1`. "Own row" marks a hang-prone case: run it alone, and after any host deadline or any kernel timeout code start a fresh session and run bring-up before the next row (MIM-601).

| # | Filter | Covers | Own row | Descriptors |
|---|---|---|---|---|
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.*` | Compile only: every kernel the cache, atomics and transaction-ID tests add or extend builds for all eight harts of CCE 0; nothing launches | — | all |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/*` | Probe: all eight harts of a CCE run a kernel | yes | all |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/*` | Snoop-master config read by the hart equals the host read and the default 0x6; L2 CONFIGURATION gives 64 B lines and 128 KiB in all | yes | all |
| 84 | `CceAtomics/MimirAtomicProbe.TwoHartAmo/*` | Probe: two-hart amoadd.w and LR/SC totals | yes | all |
| 85 | — | Removed on 2026-10-30: record only; its results are kept as the L2 constants | — | — |
| 86 | — | Removed on 2026-10-30: record only, and MIM-539 (16 trids) is Future Fix, so there is no change to catch | — | — |
| 87 | — | Removed on 2026-10-29: DuplicateId asserts a pending high water above 1 | — | — |
| 88 | — | Removed on 2026-10-30: record only, and no test depends on the counter scope | — | — |
| 89 | — | Removed on 2026-10-29: record only, and only a per-trid maximum is observable | — | — |

Stop after row 89: the gated cases below take their constants from these results (see Cache, atomics and transaction-ID probe results). Then rerun rows 75, 76 and 80, whose kernels these suites extend (with `Op/L1InvalidateFull` last), before row 90.

`MimirLocalDram` (row 3b) launches `cce_idma_copy`, so any later edit to that kernel adds row 3b to the rerun list. On 2026-10-29 the kernel gained a poll region in `kModeTridBatch` (MIM-560), and every mode but `kModeTridBatch` and `kModeAccumulate` now rejects a nonzero poll region. Rerun every row that launches it: 3a, 3b, 74, 81 (compile), 99-101, 104, 106, 109, 112, 113, 114-125 and 126-130.

| # | Filter | Covers | Own row | Descriptors |
|---|---|---|---|---|
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/*:CceCache/MimirCacheOps.L1DcacheFlushFull/*` | cflush line and whole L1 D$, then the L2 flush; records whether cflush alone reached TL1 | — | all |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/*` | Partial-line cached writes evicted by same-set reads; untouched bytes intact | — | all |
| 92 | `CceCache/MimirCacheSharedL2.*` | Same line and offset, same line other offset, adjacent line: one writer and one reader (the no-flush read records; a shared-L2 read, after the writer's `cflush` and the reader's L1 invalidate only, gates; the read after the full flush and invalidate gates), and eight concurrent writers | — | all |
| 93 | `CceCache/MimirCacheScopedLock.MultiConsumer/*:CceCache/MimirCacheScopedLock.RingWrap/*` | Seven consumers of one producer; a four-slot ring over twelve rounds | — | all |
| 94 | — | Removed on 2026-10-23 together with the since-removed `invalidate_flush_dirty_l2_full()` helper: no case needs INVALIDATEFLUSHDIRTY, and `flush_l2_cache_full` (row 75) covers the L2 full flush | — | — |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/*` | Flush without fence (gates that the data reaches TL1), L1 then L2 flush and the full hierarchy (`cflush.d.l1 x0` then `flush_l2_cache_full` each pass; the data is discarded from L1 between the two flushes, so a no-op `cflush` fails; both gate), producer racing a consumer hart; the full-hierarchy leg runs last | yes | all |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/*` | Whole-L1 D$ discard after a full clean (`cflush.d.l1 x0`, then FLUSH64 of every cacheable TL1 line, in the same asm block), then a FULLINVALIDATE wipe with a bounded poll. A wipe that never reads 0 needs a fresh session, so run this row last | yes | all |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/*_Host_*:*_RemoteCce_*:*_SiblingSpa_SnoopOffInvalidate:*_SiblingSpa_InvalidateOnly` | External-writer probe: host, remote CCE and sibling SPA writers, snoop off with INVALIDATE64 and INVALIDATE64 only; both gate that the reader first held the old value and then reads the new one | yes | all (RemoteCce 2x only) |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/*_SiblingSpa_SnoopOn` | Sibling SPA writer with RegionAttrs bit 30 set on its remap entry 1 (set by ScopedSramRemapSnoop); asserts the cached re-read is fresh with no invalidate | yes | all |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/*_LocalIdma_*` | Same-CCE iDMA writer, snoop off with INVALIDATE64 and INVALIDATE64 only. The writer starts after the reader's warm read. No snoop-on leg: the hardware never snoops a CCE's iDMA writes into its own TL1 (GRENSV-3774, GRENSV-1577). Rows 100-101 leave it out for the same reason | yes, per case | all |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/*_LocalGddrIdma_*` | iDMA from local GDDR, two modes | yes, per case | all |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/*_RemoteGddrIdma_*` | iDMA from remote GDDR, two modes | yes, per case | 2x only |
| 102 | `CceCache/MimirCacheScopedLock.NonSnoopingProducer/*` | Sibling-CCE SPA producer, consumer under the scoped lock | — | all |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/*` | Cached stores flushed and read by the sibling over SPA, and the reverse | — | all |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/*` | Sibling iDMA with the snoop packet tag into a warm cached reader; asserts no stale word (GRENSV-3774) | yes | all |
| 105 | `CceCache/MimirCacheCoherence.CrossD2d/*` | CrossCce with the CCE at the same location on the other Mimir; one route per case | yes, per case | 2x only |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/*` | One case per CCE and snoop-master config (`Cce{n}_Cfg{c}`, the default 0x6 first): neighbour or same words x evict, FLUSH64 or INVALIDATE64 against a sibling iDMA writer; every config gates that the external words survive, and 0x0, 0x2, 0x4 and 0x6 also gate the hart's neighbour write on the evict and FLUSH64 actions; each case restores the config | yes, per case | all |
| 107 | `CceCache/MimirCacheStress.ReuseThrash/*` | Sequential, 16 KiB conflict-stride and seeded random patterns, 2 KiB to 192 KiB; cycles recorded | — | all |
| 108 | `CceCache/MimirCacheStress.PerfSweep/*` | The four `MimirDmCachePerf` write paths at 64, 128 and 192 KiB; cycles recorded, no timing asserts | — | all |
| 109 | `CceCache/MimirCacheStress.PreloadChurn/*` | Sibling iDMA preload with and without the snoop tag, then the hot-set re-read; both legs gate (the snoop leg per GRENSV-3774). Records how many stale words the hart's cache still held from earlier launches before each leg drops them | yes | all |
| 110 | `CceAtomics/MimirAtomicLocal.OpMatrix/*:CceAtomics/MimirAtomicLocal.Hotspot/*:CceAtomics/MimirAtomicLocal.Distributed/*:CceAtomics/MimirAtomicLocal.LrScCas/*:CceAtomics/MimirAtomicLocal.Wrap/*:CceAtomics/MimirAtomicLocal.Ordering/*:CceAtomics/MimirAtomicLocal.CleanDirtyLine/*:CceAtomics/MimirAtomicLocal.Storm/*:CceAtomics/MimirAtomicLocal.NeighbourWriteRace/*:CceAtomics/MimirAtomicLocal.Soak/*` | Hart AMOs: every amo*.w/.d and LR/SC CAS; one, two and eight senders (atomic_semaphore_bandwidth sweep); distributed words; wrap; ordering; clean and dirty lines; mixed-op storm; hart stores beside AMOs; seeded soak. Contention is recorded per leg | — | all |
| 111 | `CceAtomics/MimirAtomicItnControl.*` | Sibling CCE writes a word and a flag over SPA; this CCE waits with uncached loads, invalidates, then AMOs | — | all |
| 112 | `CceAtomics/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/*` | AMO after a sibling iDMA with the snoop tag, no invalidate; asserts the AMO sees the iDMA value (GRENSV-3774) | yes | all |
| 113 | `DmSuitePort/MimirDmIdma.*` | Row 74 again: the single-trid baseline before the transaction-ID rows | as row 74 | as row 74 |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsOwnTl1/*:IdmaTransactionId/MimirTridIdma.UniqueIdsSiblingTl1/*:IdmaTransactionId/MimirTridIdma.ReuseWrap/*:IdmaTransactionId/MimirTridIdma.DuplicateId/*:IdmaTransactionId/MimirTridIdma.BarrierThenDependent*` | Per-command trid over 0-15 to own and sibling TL1; reuse and wrap; duplicate IDs; barrier then dependent read or write (the read-after-write destination gates; the write-after-read read is checked at ack) | — | all |
| 115 | `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr/*` | GDDR0 and GDDR1 writes, reads from both on VC 5 | yes | all |
| 116 | `IdmaTransactionId/MimirTridIdma.NominalDepth/*` | 16 trids back to back, mixed targets, the VC rewritten per command | yes | all |
| 117 | `IdmaTransactionId/MimirTridIdma.MixedTargets/*` | One table batch over own TL1, sibling TL1, GDDR0/1 and (2x) remote, host-verified remote | yes | all |
| 118 | `IdmaTransactionId/MimirTridIdma.Saturate/*` | Maximum outstanding per backend VC with 8 KiB commands, then combined (total pending across the table's trids) | yes | all |
| 119 | `IdmaTransactionId/MimirTridIdma.MultiHart/*` | Eight harts, each with its own trid, since the counters may be shared by the CCE | yes | all |
| 120 | `IdmaTransactionId/MimirTridIdma.Fairness/*` | One CCE at depth 16 while other CCEs issue table traffic | yes | all |
| 121 | `IdmaTransactionId/MimirTridIdma.Soak/*` | Seeded random targets, sizes and trid order with repeated wrap | yes | all |
| 122 | `IdmaTransactionId/MimirTridRemote.RemoteTl1/*` | Remote TL1, both directions, host-verified; one route per case | yes, per case | 2x only |
| 123 | `IdmaTransactionId/MimirTridRemote.RemoteGddr/*` | Remote GDDR, both directions, host-verified; one route per case | yes, per case | 2x only |
| 124 | `IdmaTransactionId/MimirTridConcurrent.WithPrefetcher/*` | Trid stream on CCE 1 while CCE 0 prefetches to CCE 2; the prefetch is queued once the stream has started, and the stream must advance between the first and last page sent | yes | 2x only |
| 125 | `TT_METAL_FORCE_REINIT=1` prefix, filter `IdmaTransactionId/MimirTridReset.NoStaleIdAfterReinit/*` | All 16 pending counts read 0 after a hart reinit, then every trid completes a command | yes | all |
| 126 | `CceAtomics/MimirAccumProbe.FmtOpMatrix/*` | iDMA L1 accumulate per (format, op) against the host model; the modelled INT32, INT8 and UINT8 add points gate; stops at the first point that does not ack | yes | all |
| 127 | — | Removed on 2026-10-30: record only, and it describes a software rule (flush before an own-TL1 iDMA), not hardware | — | — |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SingleHart/*:CceAtomics/MimirAtomicIdmaAccum.AllHartsShared/*:CceAtomics/MimirAtomicIdmaAccum.AllHartsDisjoint/*:CceAtomics/MimirAtomicIdmaAccum.BackToBack/*:CceAtomics/MimirAtomicIdmaAccum.SequentialAmoAccumulateAmo/*` | iDMA accumulate: INT32 accumulate, op 0x9, VC 5, one and eight harts, shared and disjoint, bursts on both command buffers, AMO then accumulate then AMO | — | all |
| 129 | `CceAtomics/MimirAtomicIdmaAccum.MaxOutstanding/*` | 64 posted accumulates over trids 0-15 | yes | all |
| 130 | — | Removed on 2026-10-30: record only, and MIM-560 is Future Fix, so asserting no corruption would test for a fix that is not coming; the suite avoids TL1 stores during iDMA instead | — | — |

Notes on the rows:
- Rows 82 to 89 are the probes; their results are in Cache, atomics and transaction-ID probe results.
- `MimirTridIdma` cases with a table (NominalDepth, MixedTargets, Saturate's combined leg, Fairness, Soak) rewrite the command buffer's VC per command while earlier trids are pending. Nothing showed that safe before these rows.
- Fairness runs one hart on several CCEs; the removed Scope probe tested only harts of one CCE, so whether CCEs share pending counters is not known.
- `MimirDmCache.Op/L1InvalidateLine` passes only while nothing touches the stack between the stores and the per-line discard. A kernel change that grew `kernel_main`'s frame broke it once (2 of 4 dirty lines were evicted to L2 before the discard); the extended `cache_ops` modes now run in a separate `noinline` function.
- Rows 98 and 106 write CCE uncore registers from the host through the SMC (RegionAttrs bit 30 on remap entry 1; the snoop-master config). Each guard restores the value it found.

### Cache, atomics and transaction-ID results

On `mimir_2x_package` (rv-zebu-08, one session after bring-up, 2026-10-22), every case in these suites ran once on CCE 0, or on its first route for the per-route rows, against `61d03f6e674` plus the uncommitted cache, atomics and transaction-ID test changes. Kernels were built from a copy of `test_kernels/cce` that was refreshed only between cases. Each case ran against the tree as it stood when it ran: fixes that landed during the run are noted in their rows, and the final tree has not run end to end. The other CCEs and routes, row 96 and `mimir_1x1` have not run. Logs, XML and the run scripts are in `/localdev/sagarwal/mm4x_runs_2026-10-22/` (`mm4x/runs/`, `mm4x/xml/`, `mm4x/pass2.list`).

| Rows | Result |
|---|---|
| 81 | pass (5 kernels) |
| 75, 76, 80 (rerun) | pass, all 19 cases. `Op/L1InvalidateLine` first failed with a kernel whose larger `kernel_main` frame evicted 2 of 4 dirty lines before the discard; it passed after the extended `cache_ops` modes moved to a `noinline` function |
| 90-93, 97-105, 107, 108, 110-117, 119-130 | pass. Rows 104 and 112 ran in their record-only versions; the current versions assert the snoop (see rows 99-101) |
| 94 | fail, finding: INVALIDATEFLUSHDIRTY is a no-op on this emulator. After `cflush x0` and a write of 1 (as chippy's `mimir_cce_mem_sweep_cached` does), the cached stores never reached TL1; a cached read still returned them and a FLUSH64 then wrote them. Writing 0xFF (AllHarts) behaves the same, and the register reads 0 at once. The `invalidate_flush_dirty_l2_full()` helper therefore did not clean. The helper and these cases were removed on 2026-10-23; the finding is for the hardware owners |
| 95 | fail on the full-hierarchy leg only, in the version where that leg used the since-removed `invalidate_flush_dirty_l2_full()` helper (complete at the consumer's first read in 1 of 8 passes, as for fence without flush). Flush without fence 8/8, fence without flush 1/8, L1 then L2 8/8. The leg now uses `flush_l2_cache_full` and has not run |
| 96 | not run: it was held while its clean used that helper. It now cleans with FLUSH64 and has not run |
| 98, 99-101 | Ran in the version whose snoop-on legs only recorded. External writers: no writer is snooped by default; INVALIDATE64 alone gives a fresh read (it also clears the hart's L1 D$ line). A sibling SPA writer with RegionAttrs bit 30 (set by ScopedSramRemapSnoop) is snooped. The iDMA snoop packet tag is not honoured on these paths: the reader stayed stale for local, local-GDDR and remote-GDDR iDMA writers. It is honoured for a sibling CCE's TL1-to-TL1 iDMA (backend VC 2): rows 104 and 112 read fresh in both versions. The 2026-10-26 run asserted the snoop on these legs. The `SnoopOn` legs of rows 99-101 have since been removed; rows 104 and 112 still assert it. In the CCE RTL an iDMA write into the issuing CCE's own TL1 takes a direct TL1 port that bypasses the snoop master (soc/cce `tt_cce_mimir_wrap.sv:965-969`), so the snoop-on legs of rows 99-101, whose writer is the reader's own CCE, fail by design rather than because of the emulator. PreloadChurn's snoop leg (a sibling CCE's GDDR-sourced copy on VC 5) passes through the target's snoop master in the RTL, so its failure is not explained |
| 106 | On CCE 0: `Cfg6` (default 0x6, in the version where it did not yet gate), `Cfg0` and `Cfg2` pass. `Cfg1` (0x1, first in the earlier all-config form) and `Cfg3` (0x3), the two configs with `disable_inline_writes`=1 that ran: a sibling iDMA write into a line the hart held dirty was acked but never became visible in TL1 within the budget, and the case stopped for a fresh session. iDMA health checks and SnoopConfig (0x6 restored) passed after the first stop. `Cfg4`, `Cfg5` and `Cfg7` have not run |
| 109 | pass after a fix (the snoop leg first read lines the invalidate leg had left cached; the host's re-seed of TL1 is not snooped). That version recorded the snoop leg; the current version gates it and is expected to fail (see rows 99-101) |
| 110 | pass after a fix: 8-hart CAS increments on one word needed up to 3059 (.w) and 3571 (.d) tries for one increment, past the first 1024-try cap; LR/SC adds (0 SC failures even with 8 harts, which interleaved 4088 times) and AMOs never starved |
| 118 | per-backend legs pass with 8 KiB commands (pending high water VC0 13, VC1 13, VC2 10, VC3 14, VC5 2). The combined table leg reached only 1 pending per trid, which cannot show combined saturation; it now asserts the total pending across its trids instead. That version has not run |
| 124 | pass, in the version that queued the prefetch before the stream: overlap seen in the window (sender credit units 0 to 256, stream rounds 0 to 1) |
| 125 | pass with `TT_METAL_FORCE_REINIT=1` |
| 130 | inconclusive: 0 corrupted words, but in that version (sibling iDMA as the ITN source) the ITN writes did not overlap the AMO rounds; the accumulate did. The host-released SPA stream has not run |

On `mimir_2x_package` (2026-10-26, at `3a3d04f9bd2`), a run of the current tree on all four CCEs and every route started: row 81, the rerun of rows 75, 76 and 80 (`Op/L1InvalidateFull` last), then rows 90-130 in order with row 96 last. The probe rows 82-89 were left out: they ran on all four CCEs on 2026-10-22 and only record. It ran in six emulator sessions, each after bring-up: soc-c-24 (40 cases), soc-zebu-01 (24 cases), soc-c-23 (89 cases), soc-zebu-01 again (80 cases, 2026-10-26 to 2026-10-27), rv-zebu-08 (48 cases, 2026-10-27) and soc-c-31 (13 cases, 2026-10-27). Each ended when the session ended. The fourth ended when the emulator stopped serving during `MimirAtomicLocal.Soak/Cce1`, which is not counted and ran again in the fifth. The sixth ended with `MimirTridIdma.Saturate/Cce2` killed mid-run, which is not counted. Sessions 1-5 ran at `3a3d04f9bd2` (before the branch was rebased onto main) with UMD `d7547af48`; the sixth ran at `86924f535c9`, after the rebase and the port of `mimir_tests` to a unit MeshDevice, with UMD `f0d497f33`. 294 of the 361 cases have run: 272 pass and 22 fail. Only row 106's failures needed a fresh session; each was followed by bring-up, and `SnoopConfig` and an iDMA case passed before the next case.

| Rows | Result on `mimir_2x_package`, 2026-10-26 |
|---|---|
| 81; 75, 76, 80 (rerun) | pass (5; all 19) |
| 90-93, 97, 102, 103, 105 | pass on all four CCEs (and every route for row 105) |
| 95 | pass on all four CCEs; the first 2x run of the version whose full-hierarchy leg uses `flush_l2_cache_full` |
| 98, 104 | pass on all four CCEs; the first 2x runs of the versions that assert the snoop (RegionAttrs bit 30 for row 98, the iDMA snoop tag on a sibling CCE's TL1-to-TL1 copy for row 104) |
| 99, 100, 101 | the `SnoopOn` leg fails on all four CCEs, as expected: after an iDMA with the snoop tag (own TL1, local GDDR, remote GDDR) the cached reader stayed stale on all 64 words. The other legs pass. The `SnoopOn` legs have since been removed |
| 106 | 26 of 32 pass, the first 2x runs of the version where the default config gates. `Cfg0`, `Cfg2`, `Cfg4` and `Cfg6` pass on all four CCEs. The configs with bit 0 set fail intermittently: CCE 0 fails `Cfg1` and `Cfg5`, CCE 1 fails `Cfg3`, CCE 2 fails `Cfg1` and `Cfg5`, CCE 3 fails `Cfg1`; the other bit-0 cases pass. As on `mimir_1x1`, each failure is on a neighbour-line leg (`Neighbour_Flush64` for CCE 0 and CCE 3, `Neighbour_Evict` for CCE 1 and CCE 2): a sibling iDMA write was acked but never became visible in TL1, and the case stopped for a fresh session |
| 107, 108 | pass on all four CCEs (`ReuseThrash` about 16 min a case, `PerfSweep` about 10 min) |
| 109 | fail on the snoop leg on all four CCEs, as expected: a sibling CCE's iDMA from GDDR (VC 5) with the snoop tag left the hot set stale (1664 or 1696 stale words at the first read, 0 of 4 rounds complete). The invalidate leg passes, 4 of 4 rounds |
| 110, 111, 112 | pass on all four CCEs |
| 113 (`MimirDmIdma.Copy`) | pass on all 12 routes |
| 114 | pass on all four CCEs |
| 115, 116, 117 | pass on all four CCEs |
| 118 | `Saturate` passes on CCE 0 and CCE 1 (about 15 min a case); CCE 2 and CCE 3 have not run |

The rest, from row 118's `MimirTridIdma.Saturate/Cce2` on (67 cases: rows 118-130, then row 96), has not run. The run directory is `/localdev/sagarwal/mm4x_runs_2026-10-26/mimir_2x_package/` (`cases.tsv` is the case order, `results.tsv` the results, `run.sh` resumes the run and skips recorded cases).

On `mimir_1x1` (soc-c-23, 2026-10-23), every case in these suites that runs on 1x1 ran once on CCE 0 and CCE 1 against the current tree (the `invalidate_flush_dirty_l2_full()` helper removed): 167 cases, including the 5 compile rows and row 125. The cases that skip on 1x1 (CCE 2 and 3, rows marked 2x only, the remote-CCE legs) were not run. After each case that stopped for a fresh session, bring-up was rerun and `SnoopConfig` and an iDMA case passed before the next case. 154 pass and 13 fail; after the `kModeTridTable` change (row 118) Saturate passes, so 156 pass and 11 fail, all in rows 99, 100, 106 and 109.

| Rows | Result on `mimir_1x1` |
|---|---|
| Every row 81-130 not listed below, apart from the rows marked 2x only | pass. Row 95's full-hierarchy leg is complete at the flag in 8 of 8 passes. Row 96: the discard leaves TL1 old (0 of 64 new words) and the wipe turns 64 stale cached words fresh. Rows 104 and 112 read fresh: a sibling CCE's TL1-to-TL1 iDMA (VC 2) with the snoop tag is snooped. Row 130 saw the ITN writes and the accumulate overlap the AMO rounds, with 0 corrupted words (recorded as inconclusive, not evidence against MIM-560) |
| 99, 100 | fail on the `SnoopOn` legs on both CCEs, as expected: after an iDMA with the snoop tag from the reader's own CCE (own TL1, local GDDR), the cached reader stayed stale on all 64 words. The snoop-off-with-invalidate and invalidate-only legs pass. The `SnoopOn` legs have since been removed |
| 106 | `Cfg0`, `Cfg2`, `Cfg4` and `Cfg6` pass on both CCEs. The configs with bit 0 set fail intermittently: CCE 0 fails `Cfg1`, `Cfg3` and `Cfg7` and passes `Cfg5`; CCE 1 fails `Cfg1` and `Cfg5` and passes `Cfg3` and `Cfg7`. Each failure is on a neighbour-line leg (`Neighbour_Flush64`, or `Neighbour_Evict` for CCE 0 `Cfg7`): a sibling iDMA write was acked but never became visible in TL1, and the case stopped for a fresh session |
| 109 | fail on the snoop leg on both CCEs, as expected: a sibling CCE's iDMA from GDDR (VC 5) with the snoop tag left the hot set stale (1664 stale words at the first read, 0 of 4 rounds complete). The invalidate leg passes, 4 of 4 rounds |
| 118 | pass on both CCEs after a kernel change. In the first run the combined leg failed: the total pending across the table's trids never exceeded 1, while every trid's count cleared 366 cycles after the last issue and the data took about 4.9M cycles to land, so each count cleared on accept before the table loop issued the next entry. `kModeTridTable` now resolves every entry before the first issue and records the largest gap between issues. With it the total reached 2 with a largest gap of 263 cycles, so commands to different backends were pending together. The per-backend legs reached VC0 13, VC1 13, VC2 10, VC5 2 in both runs. MixedTargets, NominalDepth, Fairness and Soak, the other table cases, passed again on both CCEs with the change |

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

## Results on `mimir_1x1`, 2026-11-03

One line per test case, from the gtest XML reports. Time is the case's own time; each invocation also spends about 15 s opening the device. Run against `6ea8f29c6e4` with UMD `f0d497f3` (`kstevens/grendel-emu-additions`), in a Debug build with one fix: `tt_metal/llrt/core_descriptor.cpp` no longer asserts on an empty dispatch core list when the package has no Tensix grid. Without that fix every device open fails in a Debug build. Emulator `models/mimir`, each session after `bringup_mimir_1x1.py`: soc-c-29 (rows 1 and 2a), soc-c-31 (the rest of rows 2b to 3b and the contract tests), soc-c-29 again (rows 17 to 80), soc-c-26 (rows 81 to 106, up to `Cce0_Cfg3`), soc-zebu-03 (`Cce0_Cfg4` to `Cce0_Cfg7`), soc-c-28 (the `Cce1` cases of row 106), and soc-c-29 again (rows 107 to 129).

Changes from the run order:
- Row 2c ran last in its session, after row 3b and the contract tests, because it leaves CCE1 failing firmware init until the emulator restarts. In the README order, rows 2d and 3 fail on CCE1 after it.
- `SizeAlignmentMatrix` (row 3b) ran with `timeout 14400`: each CCE takes about 2 h 20 min, past the `timeout 3600` prefix.
- The rows marked 2x only did not run.
- Row 106 restarted the emulator twice. A `DirtyLineRace` case that runs out of budget prints the MIM-601 message, which asks for a fresh session because the unfinished iDMA can corrupt later cases. The cases after such a failure were rerun in a new session, and the table shows the rerun. Results did not always repeat: `Cce1_Cfg1` failed after the first such failure and passed in a fresh session, and `Cce1_Cfg3` did the opposite.
- Rows 75, 76 and 80 were not rerun before row 90: the binary did not change between them.

| Row | Test | Result | Time |
|---|---|---|---|
| 1 | `MimirHostDram.ReconnectPersists` | pass | 2.9 s |
| 1 | `MimirHostDram.HostSideRejections` | fail: A 6-byte write clears the 2 unwritten bytes of its last word: read `0x00001615`, expected `0xFFFF1615`. Chippy zero-fills a partial trailing word (chippy `transport_interface.h:73-78`) | 0.1 s |
| 1 | `MimirHostTl1.Tl1MapLayout` | pass | 0.1 s |
| 1 | `MimirHostTl1.BoundaryAndIsolation` | pass | 0.5 s |
| 1 | `MimirHostTl1.HostSideRejections` | pass | 0.0 s |
| 1 | `HostDram/MimirHostDramView.PatternReadback/View0` | pass | 15 min 36 s |
| 1 | `HostDram/MimirHostDramView.PatternReadback/View1` | pass | 15 min 42 s |
| 1 | `HostDram/MimirHostDramView.BoundaryAndAlias/View0` | fail: Alias walk at view offset `0x2000000` read `0x00000000`, expected the guard `0x6aa18000`; View1 passes | 0.5 s |
| 1 | `HostDram/MimirHostDramView.BoundaryAndAlias/View1` | pass | 0.4 s |
| 1 | `HostTl1/MimirHostTl1Cce.PerCceVisibility/Cce0` | pass | 7 min 28 s |
| 1 | `HostTl1/MimirHostTl1Cce.PerCceVisibility/Cce1` | pass | 6 min 11 s |
| 2a | `MimirFwPreflight.FirmwareLoadPreflight` | fail: The 1-byte `fw_shared_globals_ready` write clears the other 3 bytes of its TL1 word on both CCEs (`0x00000019`, expected `0x5a5a5a19`); same chippy partial-word write | 19 s |
| 2b | `MimirFwLoad.LoadAndReachInitDone` | pass | 20 s |
| 2b | `MimirFwLoad.ImageManifest` | pass | 4.3 s |
| 2b | `MimirFwLoad.AllCcesIndependent` | pass | 44 s |
| 2b | `FirmwareLoad/MimirFwLaunch.NoOpLaunchCompletes/Cce0` | pass | 5.0 s |
| 2b | `FirmwareLoad/MimirFwLaunch.NoOpLaunchCompletes/Cce1` | pass | 5.0 s |
| 2c | `MimirFwLoadNegative.SkipLoadNeverDone` | fail: With both firmware windows zeroed, the device open succeeds: both CCEs report done (GO signal `0x00`, expected `RUN_MSG_INIT`). In an earlier session only CCE0 reported done, and CCE1 then failed firmware init in every later process until the emulator restarted | 17 s |
| 2d | `MimirFwReinit.ReloadsOnEveryOpen` | pass | 16 s |
| 3 | `MimirKernelLoad.SameBinaryAllCcesUnicast` | pass | 9.4 s |
| 3 | `KernelLoad/MimirKernelLoad.MinimalKernelExecutes/Cce0` | pass | 5.1 s |
| 3 | `KernelLoad/MimirKernelLoad.MinimalKernelExecutes/Cce1` | pass | 5.4 s |
| 3 | `KernelLoad/MimirKernelLoad.EachHartAlone/Cce0` | pass | 12 s |
| 3 | `KernelLoad/MimirKernelLoad.EachHartAlone/Cce1` | pass | 12 s |
| 3 | `KernelLoad/MimirKernelLoad.ArgsReplaceStale/Cce0` | pass | 9.2 s |
| 3 | `KernelLoad/MimirKernelLoad.ArgsReplaceStale/Cce1` | pass | 8.9 s |
| 3 | `KernelLoad/MimirKernelLoad.FirmwareKeepsRunningAcrossReopen/Cce0` | pass | 12 s |
| 3 | `KernelLoad/MimirKernelLoad.FirmwareKeepsRunningAcrossReopen/Cce1` | pass | 12 s |
| 3 | `LocalTl1/MimirLocalTl1.SameMimirCopy/Cce0To1` | pass | 13 s |
| 3 | `LocalTl1/MimirLocalTl1.SameMimirCopy/Cce1To0` | pass | 13 s |
| 3a | `DmSuitePort/MimirDmIdma.Copy/OwnTl1` | pass | 9.3 s |
| 3a | `DmSuitePort/MimirDmIdma.Copy/LocalGddr` | pass | 12 s |
| 3a | `DmSuitePort/MimirDmIdma.Copy/LocalGddr1` | pass | 13 s |
| 3b | `LocalDram/MimirLocalDram.CceWritesTl1ToDram/Cce0` | pass | 2 min 39 s |
| 3b | `LocalDram/MimirLocalDram.CceWritesTl1ToDram/Cce1` | pass | 2 min 38 s |
| 3b | `LocalDram/MimirLocalDram.CceReadsDramToTl1/Cce0` | pass | 2 min 36 s |
| 3b | `LocalDram/MimirLocalDram.CceReadsDramToTl1/Cce1` | pass | 2 min 38 s |
| 3b | `LocalDram/MimirLocalDram.LoopbackChecksIntermediate/Cce0` | pass | 5 min 31 s |
| 3b | `LocalDram/MimirLocalDram.LoopbackChecksIntermediate/Cce1` | pass | 5 min 28 s |
| 3b | `LocalDram/MimirLocalDram.SizeAlignmentMatrix/Cce0` | pass | 2 h 17 min |
| 3b | `LocalDram/MimirLocalDram.SizeAlignmentMatrix/Cce1` | pass | 2 h 21 min |
| 3b | `LocalDram/MimirLocalDram.RepeatedAndOverlappingPerHart/Cce0` | pass | 20 min 7 s |
| 3b | `LocalDram/MimirLocalDram.RepeatedAndOverlappingPerHart/Cce1` | pass | 20 min 25 s |
| contract | `MimirEmu.CceSramChannelsDoNotAliasThroughMetalCluster` | pass | 0.0 s |
| contract | `MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi` | pass | 4.1 s |
| contract | `MimirEmu.AllCcesCopyLocalMimirBlock` | pass | 16 s |
| contract | `MimirEmu/AllocatedDramBufferHostLoopback.RoundTrip/Partition0` | pass | 4.1 s |
| contract | `MimirEmu/AllocatedDramBufferHostLoopback.RoundTrip/Partition1` | pass | 4.1 s |
| contract | `MimirEmu/DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip/Partition0` | pass | 3.9 s |
| contract | `MimirEmu/DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip/Partition1` | pass | 4.0 s |
| contract | `MimirEmu/CceSramRoundTripThroughUnitMesh.RoundTrip/Cce0` | pass | 4.0 s |
| contract | `MimirEmu/CceSramRoundTripThroughUnitMesh.RoundTrip/Cce1` | pass | 4.1 s |
| contract | `MimirEmu/CceSramRoundTripThroughMetalCluster.RoundTrip/Cce0` | pass | 0.0 s |
| contract | `MimirEmu/CceSramRoundTripThroughMetalCluster.RoundTrip/Cce1` | pass | 0.0 s |
| contract | `MimirEmu/HartZeroRunsDramKernel.WritesMagic/Cce0` | pass | 4.7 s |
| contract | `MimirEmu/HartZeroRunsDramKernel.WritesMagic/Cce1` | pass | 5.0 s |
| contract | `MimirEmu/AllHartsRunDramKernel.WritesMagic/Cce0` | pass | 7.2 s |
| contract | `MimirEmu/AllHartsRunDramKernel.WritesMagic/Cce1` | pass | 8.6 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition0` | pass | 5.6 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition1` | pass | 5.3 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition0` | pass | 4.9 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition1` | pass | 4.9 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition0` | pass | 10 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition1` | pass | 10.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition0` | pass | 10.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition1` | pass | 10.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition0` | pass | 12 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition1` | pass | 11 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition0` | pass | 11 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition1` | pass | 12 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce0Partition0` | pass | 6.7 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce0Partition1` | pass | 6.9 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce1Partition0` | pass | 7.1 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce1Partition1` | pass | 6.9 s |
| contract | `MimirEmu/WritesCrossMimirCceSram.WritesMagic/Cce0` | skip: Configured descriptor does not expose a remote Mimir CCE. | 4.3 s |
| contract | `MimirEmu/WritesCrossMimirCceSram.WritesMagic/Cce1` | skip: Configured descriptor does not expose a remote Mimir CCE. | 4.0 s |
| contract | `MimirEmu/PrefetchesLocalGddrToRemoteCceSram.RoundTrip/Cce0` | skip: Configured descriptor does not expose a remote Mimir CCE. | 4.2 s |
| contract | `MimirEmu/PrefetchesLocalGddrToRemoteCceSram.RoundTrip/Cce1` | skip: Configured descriptor does not expose a remote Mimir CCE. | 4.5 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmu/CopiesEveryMimirBlock.RoundTrip/Cce0` | pass | 5.4 s |
| contract | `MimirEmu/CopiesEveryMimirBlock.RoundTrip/Cce1` | pass | 4.6 s |
| 17 | `MimirPrefetcher.SupportPreconditionsHold` | pass | 4.5 s |
| 18 | `MimirPrefetcher.SupportPreconditionsFailOnUninitializedSender` | pass | 4.5 s |
| 19 | `MimirPrefetcher.SupportStartStopSucceeds` | pass | 6.7 s |
| 20 | `MimirPrefetcher.SupportPreconditionsHoldWhileReceiverRuns` | pass | 6.0 s |
| 21 | `MimirPrefetcher.GcbLayoutLocalReceiver` | pass | 4.8 s |
| 24 | `MimirPrefetcher.GcbLayoutRejectsConflictingReceivers` | pass | 4.5 s |
| 25 | `MimirPrefetcher.GcbSmokeDiscardDrain` | pass | 7.0 s |
| 26 | `MimirPrefetcher.GcbSmokeValidatorLocal` | pass | 7.2 s |
| 27 | `MimirPrefetcher.GcbSmokeValidatorWrongSeed` | pass | 6.8 s |
| 28 | `MimirPrefetcher.GcbSmokeResize` | pass | 6.6 s |
| 30 | `MimirPrefetcher.GcbSmokeDrainUntilStopped` | pass | 6.6 s |
| 31 | `MimirPrefetcher.GcbSmokeStallUntilReleased` | pass | 9.0 s |
| 32 | `MimirPrefetcher.GcbSmokeDrainTimeout` | pass | 8.8 s |
| 33 | `MimirPrefetcher.GcbSmokeReceiverWaitTimeout` | pass | 8.2 s |
| 34 | `MimirPrefetcher.GcbSmokeRelaunchResumes` | pass | 7.9 s |
| 35 | `MimirPrefetcher.LifecycleStartStop` | pass | 5.7 s |
| 36 | `MimirPrefetcher.LifecycleStartStopTwice` | pass | 7.6 s |
| 37 | `MimirPrefetcher.LifecycleBusySenderFailsStart` | pass | 6.1 s |
| 38 | `MimirPrefetcher.LifecycleQueueRejectsReceiverOnSender` | pass | 7.6 s |
| 39 | `MimirPrefetcher.LifecycleWaitCq` | pass | 11 s |
| 40 | `MimirPrefetcher.LifecycleDrainAtStop` | pass | 11 s |
| 41 | `MimirPrefetcher.PrefetchOnePagePerSessionLocal0To1` | pass | 12 s |
| 49 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/SubChunkLocal0To1` | pass | 14 s |
| 51 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/BelowStageThirdLocal0To1` | pass | 13 s |
| 53 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/PartialStageChunkLocal0To1` | pass | 13 s |
| 54 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AboveStageThirdLocal0To1` | pass | 11 s |
| 56 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/PageSplitLocal0To1` | pass | 12 s |
| 57 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AbaLocal0To1` | pass | 14 s |
| 59 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/WrapAndResizeLocal0To1` | pass | 18 s |
| 61 | `TensorPrefetcher/MimirPrefetcherGcbSwitch.Delivers/GcbSwitchLocal0To1` | pass | 13 s |
| 63 | `MimirPrefetcher.PrefetchFenceHoldsDeliveryLocal0To1` | pass | 14 s |
| 65 | `MimirPrefetcher.QueueRejectsKRowMajorTensor` | pass | 9.0 s |
| 66 | `MimirPrefetcher.QueueRejectsStreamingTensor` | pass | 8.7 s |
| 67 | `MimirPrefetcher.QueueRejectsRangePastTensor` | pass | 10.0 s |
| 68 | `MimirPrefetcher.QueueRejectsRangePastView` | pass | 9.3 s |
| 72 | `DmSuitePort/MimirDmRuntimeArgs.IllegalTooManyRuntimeArgs/Cce0` | pass | 6.2 s |
| 72 | `DmSuitePort/MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs/Cce0` | pass | 6.2 s |
| 73 | `DmSuitePort/MimirDmMeshBufferDram.InterleavedPlacement/mimir_1x1` | pass | 13 s |
| 73 | `DmSuitePort/MimirDmMeshBufferDram.WriteRead/mimir_1x1` | pass | 6.7 s |
| 73 | `DmSuitePort/MimirDmMeshBufferDram.MultipleWriteReadRounds/mimir_1x1` | pass | 29 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/OwnTl1` | pass | 8.4 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/MaxPacket` | pass | 16 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/SiblingTl1` | pass | 15 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr` | pass | 13 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr1` | pass | 13 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1` | skip: Remote iDMA cases need a 2x descriptor (two Mimirs). | 0.1 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/RemoteGddr` | skip: Remote iDMA cases need a 2x descriptor (two Mimirs). | 0.1 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1D` | pass | 8.7 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1DLocalGddr` | pass | 8.4 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLine` | pass | 7.6 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLineLocalGddr` | pass | 7.7 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1KernelVerified` | skip: Remote iDMA cases need a 2x descriptor (two Mimirs). | 0.1 s |
| 75 | `DmSuitePort/MimirDmCache.Op/NoFlushNotVisible` | pass | 9.4 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1FlushLine` | pass | 8.3 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1FlushFull` | pass | 8.5 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1InvalidateLine` | pass | 8.3 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushLine` | pass | 8.5 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushRange` | pass | 9.8 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushFull` | pass | 9.5 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2InvalidateLine` | pass | 8.5 s |
| 75 | `DmSuitePort/MimirDmCache.Op/InvalidateFreshRead` | pass | 8.2 s |
| 75 | `DmSuitePort/MimirDmCache.Op/HartToHartCoherent` | fail: Hart 1 warm read before hart 0 stores returned the old value in 0 of 64 words, expected 64 (`test_mimir_dm_cache.cpp:238`, CCE0, TL1 `0x8b4c0`) | 13 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2InvalidateFreshRead` | fail: The cached re-read before the L2 invalidate saw 0 of 64 stale words, expected 64 (`test_mimir_dm_cache.cpp:282`, CCE0, TL1 `0x8b4c0`) | 8.3 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1InvalidateFull` | pass | 8.7 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/Uncached1B` | pass | 48 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/Uncached8B` | pass | 48 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/CachedRangeFlush` | pass | 45 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/CachedFastFlush` | pass | 46 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_SrcOnly` | pass | 5.8 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_DstOnly` | pass | 5.3 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_Both` | pass | 5.2 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_SrcOnly` | pass | 5.4 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_DstOnly` | pass | 5.4 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_Both` | pass | 5.2 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_SrcOnly` | pass | 5.5 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_DstOnly` | pass | 5.5 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_Both` | pass | 5.6 s |
| 78 | `DmSuitePort/MimirDmAddrgen.Pattern/Im2Col_FullImage` | pass | 8.2 s |
| 78 | `DmSuitePort/MimirDmAddrgen.Pattern/Im2Col_Dilation1` | pass | 6.8 s |
| 79 | `DmSuitePort/MimirDmDramUnary.ReaderWriterHandshake/Cce0` | pass | 11 s |
| 79 | `DmSuitePort/MimirDmDramUnary.ReaderWriterHandshake/Cce1` | pass | 10 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/InvalidateOnAcquire` | pass | 7.3 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/FlushOnRelease` | pass | 6.8 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/LockedRangeOnly` | pass | 6.8 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_cache_ops` | pass | 6.3 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_idma_copy` | pass | 5.0 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_amo_ops` | pass | 6.2 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_scoped_lock_producer` | pass | 5.3 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_scoped_lock_consumer` | pass | 5.4 s |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/Cce0` | pass | 37 s |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/Cce1` | pass | 36 s |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/Cce0` | pass | 8.7 s |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/Cce1` | pass | 8.3 s |
| 84 | `CceAtomics/MimirAtomicProbe.TwoHartAmo/Cce0` | pass | 11 s |
| 84 | `CceAtomics/MimirAtomicProbe.TwoHartAmo/Cce1` | pass | 10 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/Cce0` | pass | 8.9 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/Cce1` | pass | 9.0 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushFull/Cce0` | pass | 9.6 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushFull/Cce1` | pass | 9.0 s |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/Cce0` | pass | 12 s |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/Cce1` | pass | 12 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce0_WriterReader` | pass | 13 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce0_EightWriters` | pass | 36 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce1_WriterReader` | pass | 14 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce1_EightWriters` | pass | 35 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce0_WriterReader` | pass | 12 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce0_EightWriters` | pass | 35 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce1_WriterReader` | pass | 13 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce1_EightWriters` | pass | 34 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce0_WriterReader` | pass | 12 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce0_EightWriters` | pass | 35 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce1_WriterReader` | pass | 13 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce1_EightWriters` | pass | 35 s |
| 93 | `CceCache/MimirCacheScopedLock.MultiConsumer/Cce0` | pass | 12 s |
| 93 | `CceCache/MimirCacheScopedLock.MultiConsumer/Cce1` | pass | 12 s |
| 93 | `CceCache/MimirCacheScopedLock.RingWrap/Cce0` | pass | 8.0 s |
| 93 | `CceCache/MimirCacheScopedLock.RingWrap/Cce1` | pass | 6.7 s |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/Cce0` | pass | 37 s |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/Cce1` | pass | 40 s |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/Cce0` | pass | 16 s |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/Cce1` | pass | 15 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_Host_SnoopOffInvalidate` | pass | 9.0 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_Host_InvalidateOnly` | pass | 8.8 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_SnoopOffInvalidate` | pass | 14 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_InvalidateOnly` | pass | 12 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteCce_SnoopOffInvalidate` | skip: Descriptor exposes 1 Mimir; the case needs two. | 0.1 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteCce_InvalidateOnly` | skip: Descriptor exposes 1 Mimir; the case needs two. | 0.1 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_Host_SnoopOffInvalidate` | pass | 8.5 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_Host_InvalidateOnly` | pass | 8.2 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_SiblingSpa_SnoopOffInvalidate` | pass | 12 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_SiblingSpa_InvalidateOnly` | pass | 12 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_RemoteCce_SnoopOffInvalidate` | skip: Descriptor exposes 1 Mimir; the case needs two. | 0.1 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_RemoteCce_InvalidateOnly` | skip: Descriptor exposes 1 Mimir; the case needs two. | 0.1 s |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_SnoopOn` | pass | 12 s |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_SiblingSpa_SnoopOn` | pass | 12 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalIdma_SnoopOffInvalidate` | pass | 15 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalIdma_InvalidateOnly` | pass | 15 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalIdma_SnoopOffInvalidate` | pass | 15 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalIdma_InvalidateOnly` | pass | 16 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalGddrIdma_SnoopOffInvalidate` | pass | 17 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalGddrIdma_InvalidateOnly` | pass | 15 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalGddrIdma_SnoopOffInvalidate` | pass | 15 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalGddrIdma_InvalidateOnly` | pass | 15 s |
| 102 | `CceCache/MimirCacheScopedLock.NonSnoopingProducer/Cce0` | fail: Consumer warm reads before the lock returned 3 of 8 old sentinels, the rest other data such as `0xd2000000` (`test_mimir_dm_scoped_lock.cpp:465`); after acquire, 2 of the 4 cached words outside the locked range still held that data (`:471`) | 6.8 s |
| 102 | `CceCache/MimirCacheScopedLock.NonSnoopingProducer/Cce1` | fail: Consumer warm reads before the lock returned 5 of 8 old sentinels (`test_mimir_dm_scoped_lock.cpp:465`); after acquire, 2 of the 4 cached words outside the locked range still held other data (`:471`) | 6.5 s |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/Cce0` | pass | 20 s |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/Cce1` | pass | 21 s |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/Cce0` | pass | 11 s |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/Cce1` | pass | 11 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg6` | pass | 52 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg0` | pass | 52 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg1` | pass | 52 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg2` | pass | 51 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg3` | fail: cfg3 Neighbour_Flush64: the hart budget ran out before it read the external words; the writer iDMA was acked but its data never landed in TL1 (`test_mimir_dm_cache.cpp:1988`, the MIM-601 fresh-session message) | 29 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg4` | pass | 47 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg5` | pass | 47 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg7` | fail: cfg7 Neighbour_Evict: the writer iDMA was acked but its data never landed in TL1 before the hart budget ran out (`test_mimir_dm_cache.cpp:1988`). Failed the same way at the start of a fresh session | 23 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg6` | pass | 51 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg0` | pass | 50 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg1` | pass | 51 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg2` | pass | 51 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg3` | fail: cfg3 Neighbour_Evict: external write lost, 0 of 8 external words kept, all 8 hart words new (`test_mimir_dm_cache.cpp:1998`); Neighbour_Flush64 in the same case kept 8 of 8. It passed in the soc-c-26 session after the first MIM-601 failure | 52 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg4` | pass | 50 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg5` | pass | 52 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg7` | fail: cfg7 Neighbour_Evict: the writer iDMA was acked but its data never landed in TL1 before the hart budget ran out (`test_mimir_dm_cache.cpp:1988`). Failed the same way in two sessions | 23 s |
| 107 | `CceCache/MimirCacheStress.ReuseThrash/Cce0` | pass | 3 min 15 s |
| 107 | `CceCache/MimirCacheStress.ReuseThrash/Cce1` | pass | 3 min 14 s |
| 108 | `CceCache/MimirCacheStress.PerfSweep/Cce0` | pass | 2 min 1 s |
| 108 | `CceCache/MimirCacheStress.PerfSweep/Cce1` | pass | 2 min 2 s |
| 109 | `CceCache/MimirCacheStress.PreloadChurn/Cce0` | fail: Snoop leg (CCE1 iDMA from GDDR view 0 into hot set TL1 `0x8b4c0`): 1024 stale words consumed at the first read, expected 0, and 0 of 4 rounds matched (`test_mimir_dm_cache.cpp:2211`, `:2212`) | 1 min 10 s |
| 109 | `CceCache/MimirCacheStress.PreloadChurn/Cce1` | fail: Snoop leg: 1024 stale words consumed at the first read, expected 0, and 0 of 4 rounds matched (`test_mimir_dm_cache.cpp:2211`, `:2212`) | 1 min 12 s |
| 110 | `CceAtomics/MimirAtomicLocal.OpMatrix/Cce0` | pass | 9.0 s |
| 110 | `CceAtomics/MimirAtomicLocal.OpMatrix/Cce1` | pass | 9.3 s |
| 110 | `CceAtomics/MimirAtomicLocal.Hotspot/Cce0` | pass | 53 s |
| 110 | `CceAtomics/MimirAtomicLocal.Hotspot/Cce1` | pass | 52 s |
| 110 | `CceAtomics/MimirAtomicLocal.Distributed/Cce0` | pass | 1 min 10 s |
| 110 | `CceAtomics/MimirAtomicLocal.Distributed/Cce1` | pass | 1 min 14 s |
| 110 | `CceAtomics/MimirAtomicLocal.LrScCas/Cce0` | pass | 1 min 49 s |
| 110 | `CceAtomics/MimirAtomicLocal.LrScCas/Cce1` | pass | 1 min 53 s |
| 110 | `CceAtomics/MimirAtomicLocal.Wrap/Cce0` | pass | 1 min 43 s |
| 110 | `CceAtomics/MimirAtomicLocal.Wrap/Cce1` | pass | 1 min 46 s |
| 110 | `CceAtomics/MimirAtomicLocal.Ordering/Cce0` | pass | 16 s |
| 110 | `CceAtomics/MimirAtomicLocal.Ordering/Cce1` | pass | 15 s |
| 110 | `CceAtomics/MimirAtomicLocal.CleanDirtyLine/Cce0` | pass | 23 s |
| 110 | `CceAtomics/MimirAtomicLocal.CleanDirtyLine/Cce1` | pass | 24 s |
| 110 | `CceAtomics/MimirAtomicLocal.Storm/Cce0` | pass | 58 s |
| 110 | `CceAtomics/MimirAtomicLocal.Storm/Cce1` | pass | 54 s |
| 110 | `CceAtomics/MimirAtomicLocal.NeighbourWriteRace/Cce0` | pass | 26 s |
| 110 | `CceAtomics/MimirAtomicLocal.NeighbourWriteRace/Cce1` | pass | 26 s |
| 110 | `CceAtomics/MimirAtomicLocal.Soak/Cce0` | pass | 27 s |
| 110 | `CceAtomics/MimirAtomicLocal.Soak/Cce1` | pass | 24 s |
| 111 | `CceAtomics/MimirAtomicItnControl.SiblingWriteThenAmo/Cce0` | pass | 10 s |
| 111 | `CceAtomics/MimirAtomicItnControl.SiblingWriteThenAmo/Cce1` | pass | 10 s |
| 112 | `CceAtomics/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/Cce0` | pass | 13 s |
| 112 | `CceAtomics/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/Cce1` | pass | 13 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/OwnTl1` | pass | 8.3 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/MaxPacket` | pass | 15 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/SiblingTl1` | pass | 12 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr` | pass | 13 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr1` | pass | 12 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1` | skip: Remote iDMA cases need a 2x descriptor (two Mimirs). | 0.1 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteGddr` | skip: Remote iDMA cases need a 2x descriptor (two Mimirs). | 0.1 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1D` | pass | 8.2 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DLocalGddr` | pass | 8.1 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLine` | pass | 7.5 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLineLocalGddr` | pass | 7.4 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1KernelVerified` | skip: Remote iDMA cases need a 2x descriptor (two Mimirs). | 0.1 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsOwnTl1/Cce0` | pass | 15 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsOwnTl1/Cce1` | pass | 15 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsSiblingTl1/Cce0` | pass | 30 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsSiblingTl1/Cce1` | pass | 26 s |
| 114 | `IdmaTransactionId/MimirTridIdma.ReuseWrap/Cce0` | pass | 41 s |
| 114 | `IdmaTransactionId/MimirTridIdma.ReuseWrap/Cce1` | pass | 41 s |
| 114 | `IdmaTransactionId/MimirTridIdma.DuplicateId/Cce0` | fail: Pending high water stayed at 1 for both trid 9 (to GDDR) and trid 15 (to CCE1 TL1), expected more than 1 (`test_mimir_trid.cpp:907`, `:916`); all 16 commands landed. CCE1 passed | 27 s |
| 114 | `IdmaTransactionId/MimirTridIdma.DuplicateId/Cce1` | pass | 27 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentReadAfterWrite/Cce0` | pass | 23 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentReadAfterWrite/Cce1` | pass | 23 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentWriteAfterRead/Cce0` | pass | 23 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentWriteAfterRead/Cce1` | pass | 24 s |
| 115 | `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr/Cce0` | pass | 48 s |
| 115 | `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr/Cce1` | pass | 47 s |
| 116 | `IdmaTransactionId/MimirTridIdma.NominalDepth/Cce0` | pass | 26 s |
| 116 | `IdmaTransactionId/MimirTridIdma.NominalDepth/Cce1` | pass | 26 s |
| 117 | `IdmaTransactionId/MimirTridIdma.MixedTargets/Cce0` | pass | 12 s |
| 117 | `IdmaTransactionId/MimirTridIdma.MixedTargets/Cce1` | pass | 12 s |
| 118 | `IdmaTransactionId/MimirTridIdma.Saturate/Cce0` | pass | 2 min 32 s |
| 118 | `IdmaTransactionId/MimirTridIdma.Saturate/Cce1` | pass | 2 min 25 s |
| 119 | `IdmaTransactionId/MimirTridIdma.MultiHart/Cce0` | pass | 1 min 28 s |
| 119 | `IdmaTransactionId/MimirTridIdma.MultiHart/Cce1` | pass | 1 min 27 s |
| 120 | `IdmaTransactionId/MimirTridIdma.Fairness/Cce0` | pass | 51 s |
| 120 | `IdmaTransactionId/MimirTridIdma.Fairness/Cce1` | pass | 50 s |
| 121 | `IdmaTransactionId/MimirTridIdma.Soak/Cce0` | pass | 1 min 53 s |
| 121 | `IdmaTransactionId/MimirTridIdma.Soak/Cce1` | pass | 1 min 42 s |
| 125 | `IdmaTransactionId/MimirTridReset.NoStaleIdAfterReinit/Cce0` | pass | 1 min 20 s |
| 125 | `IdmaTransactionId/MimirTridReset.NoStaleIdAfterReinit/Cce1` | pass | 1 min 20 s |
| 126 | `CceAtomics/MimirAccumProbe.FmtOpMatrix/Cce0` | pass | 2 min 5 s |
| 126 | `CceAtomics/MimirAccumProbe.FmtOpMatrix/Cce1` | pass | 2 min 4 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SingleHart/Cce0` | pass | 8.1 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SingleHart/Cce1` | pass | 7.9 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsShared/Cce0` | pass | 31 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsShared/Cce1` | pass | 31 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsDisjoint/Cce0` | pass | 34 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsDisjoint/Cce1` | pass | 32 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.BackToBack/Cce0` | pass | 26 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.BackToBack/Cce1` | pass | 27 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SequentialAmoAccumulateAmo/Cce0` | pass | 14 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SequentialAmoAccumulateAmo/Cce1` | pass | 14 s |
| 129 | `CceAtomics/MimirAtomicIdmaAccum.MaxOutstanding/Cce0` | pass | 11 s |
| 129 | `CceAtomics/MimirAtomicIdmaAccum.MaxOutstanding/Cce1` | pass | 12 s |

### Root causes of the 2026-11-03 failures

From the logs, the source and git history; nothing below was rerun. Cache and snoop behaviour is as observed on the emulator. Paths are in tt-metal unless they name chippy, grendelemulation or the ZeBu Python package (`/proj_soc/zebu/Modules/collateral/venv/26ww33/lib/python3.12/site-packages/zebu`, readable from soc-l-12).

**Stale lines in the shared L2 from an earlier process** (`NonSnoopingProducer` Cce0 and Cce1, `HartToHartCoherent`, `L2InvalidateFreshRead`, `SkipLoadNeverDone`), a runtime gap. Since 3ecb73d1637 (2026-10-21), device open resets only the CCE harts (`ALL_NEO_DMS`, `risc_firmware_initializer.cpp:610`, `:1393`); before it, open asserted `RiscType::ALL`, which also reset the uncore. The commit moved uncore and remap ownership to bring-up and does not mention caches. By the RTL (soc/cce `ce574b3`; tensix/soc/chipyard `d82dfe27`, generated `TTCCEConfigOneClk`), a hart reset clears that hart's L1 I$ (`ICache.sv:210-218`) and its L1 D$, dropping dirty lines (`DCache.sv:1330-1332`). Only the uncore reset or the L2's full-invalidate register clears the shared L2 (`Directory.sv:144-154`, `:218-231`). TL1 below 4 MiB, firmware included, is cacheable (`PMAChecker.sv:132`), and host writes to TL1 are not snooped, so L2 lines from one process survive into the next and a hart refills from them instead of from what the host wrote. Every row runs as its own process, so this is how one case's data reaches a later case's warm read.
- `NonSnoopingProducer` (row 102), symptom of the L2 gap, not a test bug. Every foreign word in the consumer's warm read is data an earlier case left at TL1 `0x8b4c0`: `0xD2000000 | cce<<16` plus the word index is row 96's wipe value, `0x6a049000` and `0x6a04a000` are row 96's guard (`make_guard(0x49 + cce)`), and `0xaa000a10` is a row 93 `RingWrap` slot. Row 96 ran before row 102 in this session, although row 96 says to run it last. Rows 93 and 96 ran as earlier processes, so their values crossed a process boundary in the L2. The consumer's warm read is at `cce_scoped_lock_consumer.cpp:125`.
- `HartToHartCoherent` and `L2InvalidateFreshRead` (row 75), likely a symptom of the L2 gap. The warm-read values are not logged, so a shared-L2 defect is not excluded. The warm read (`cce_cache_ops.cpp:1185`) fails before the feature check (`test_mimir_dm_cache.cpp:239`). On 2026-11-13 `HartToHartCoherent`'s coherence check itself passed (stale before 0, 64 matches). In both 2x runs the warm read found 16 of 64 words not old, exactly one 64 B line. The case before `L2InvalidateFreshRead`, `L1InvalidateFull`, ends with a whole-L1 discard and an uncached check, not a cached read, so its stale line would have come from an earlier case. Not explained: on 2026-11-13 `L2InvalidateFreshRead` passed 37 s after `HartToHartCoherent` failed, with no restart. Under the rule for the four contract cases in Running the data-movement ports this is not a coherence finding, but the failure is still correct: the test must fail when the state it depends on is wrong.
- `SkipLoadNeverDone` (row 2c), stale cached firmware, medium-high confidence. Go signal 0 is `RUN_MSG_DONE` (`dev_msgs.h:108`), which only firmware writes (`tt-2xx/drisc.cc:146`). The zeroed window `[0x8080, 0xE080)` does not overlap the mailbox `[0x80, 0x8080)`, and the runtime writes `RUN_MSG_INIT` after the zeroing. Firmware text sits in the cacheable window and nothing invalidates it before the harts are released, so the harts ran firmware that was no longer in TL1. This also fits the earlier session, where CCE1 cached some zeroed lines and then failed firmware init in every later process until the emulator restarted. No earlier pass of row 2c is recorded. This is a runtime defect, not a test bug: Metal's firmware load does not guarantee the harts run what it wrote, because they refill from L2 lines holding the previous process's firmware. The 2026-11-04 pass on `mimir_2x_package` ran in its own session right after bring-up; bring-up never releases the harts, so the L2 was cold. Its 1 min 51 s against 17 s on `mimir_1x1` also fits a timeout against an early "done". The test cannot yet rule out a zeroing that never reached TL1 or a lost `RUN_MSG_INIT` write; it reads back neither.
- Rows 75 and 102 passed on 2026-10-22, 10-23 and 10-26 in the same case order, so the caches were clean between processes then. The test changes since 10-23 do not touch these paths. The likely change is 3ecb73d1637, which the 11-03 tree (`6ea8f29c6e4`) contains. Whether the 10-26 tree had it cannot be checked: `3a3d04f9bd2` no longer exists after the rebase.
- To confirm: in row 2c, read the zeroed window back after the open (all zeros with a successful open confirms stale firmware), and run 2c right after bring-up and again after 2b. Fix in the runtime, not the tests. At open, after the firmware and the other host writes to TL1 and before the harts are released, invalidate the L2: the full-invalidate register, or the uncore reset with the AXI-clock steps in Mimir spec v1.14 §7.4. On `mimir_1x1` row 96's FULLINVALIDATE wipe works; INVALIDATEFLUSHDIRTY (row 94) is a no-op on the emulator. Do not add a record-and-drop before the warm reads: the carried-over count is only recorded, never asserted, so the drop would hide the gap. Until the runtime fix lands, these four cases fail correctly. Add a test that catches the gap directly: one process leaves dirty lines at known offsets, the next writes a seed from the host and forces an eviction, and the host checks the seed survived. In row 2c, add a read-back of the zeroed window and check that `fw_shared_globals_ready` and `subordinate_sync` stay clear.

**Sub-word host writes** (`MimirHostDram.HostSideRejections`, `MimirFwPreflight.FirmwareLoadPreflight`). The tests expect something the transport does not do. A host write whose size is not a multiple of 4 goes `EmuTTDevice` → `GrendelTTDevice::write_to_device` → `MimirChippyMemoryMap` (UMD `mimir_chippy_memory.cpp:58-68`) → chippy `TransportInterface::_write<uint32_t>` (`transport_interface.h:73-78`), which builds the last word on a zero base, so the unwritten bytes read back 0. `GrendelJtagProtocol` is not on this path; its comment describes the same chippy behaviour. Neither test regressed: 2aeef6e883d (2026-10-30) changed `HostSideRejections` from expecting the zero fill to expecting the bytes kept, and 94024ee1a9c added the neighbour-byte check to the preflight. The runtime's own 1-byte writes store 0, so firmware init is not affected. Fix in UMD or chippy, with a read-modify-write of the last word. A read-modify-write in the test would hide the transport bug, so both tests keep failing until then.

**Too little overlap** (`MimirTridIdma.DuplicateId`: Cce0 on `mimir_1x1`, Cce0-2 on `mimir_2x_package`), a test bug in the stimulus, not the assert. 6ee2a27c9d0 (2026-10-30) asserts `EXPECT_GT(..., 1u)` on the pending high water (`test_mimir_trid.cpp:905`, `:914`). With 512 B commands, issuing a command can take longer than the previous one takes to ack, so the commands may never overlap and the high water reads 1 on correct hardware. The assert is still right: it is the only check that the counter counts past 1. A counter stuck at 0 or 1 also reads 1, and every other check passes without overlap. Relaxing the assert, or one shared destination slot, would let a broken counter pass; with a shared slot an early ack passes once the first command lands. Fix: commands large enough to always overlap (Saturate reached 10-13 pending with 8 KiB), a separate destination per command, and only the source shared. GDDR fits 15 separate 8 KiB slots and the sibling TL1 slot fits one, so the sibling leg needs smaller or fewer commands. Also record the largest gap between issues in batch mode's free `probe_second`; a high water of 1 with a gap well under the ack latency then points at the hardware. Do not move the case to `kModeTridTable`: the at-ack data check is batch-only. The ~190k data cycles in the log are one `data_landed` pass over 2048 words, not landing latency. Row 118's "cleared on accept" explanation may rest on the same reading; not checked.

**Snoop-config intermittents** (`DirtyLineRace` Cce0 and Cce1 `Cfg3` and `Cfg7`, `PreloadChurn` Cce0 and Cce1), known. Every `DirtyLineRace` failure has bit 0 (`disable_inline_writes`, `tensix_neo_reg.h:8155-8162`) set, as on 10-23 and 10-26. `Cce0_Cfg3`, `Cce0_Cfg7` and `Cce1_Cfg7` are the known "acked, never visible" mode. `Cce1_Cfg3` is new: the hart's uncached poll saw the external words, but the host found 0 of 8 after the hart's final `flush_l2_cache_line` (`cce_cache_ops.cpp:871`); the hart writing back its stale copy of the line is the likely cause, inferred from the kernel. `PreloadChurn`'s snoop leg is recorded as expected, though the row 99-101 results note says the RTL does not explain it. The `kIdmaSnoopNote` text appended to both tests' failure messages (`test_mimir_dm_cache.cpp:514-516`) blames an iDMA into the CCE's own TL1, but in both tests the writer is the sibling CCE.

**Not known** (`HostDram/MimirHostDramView.BoundaryAndAlias/View0`, offset `0x2000000` read 0).
- Host side ruled out. The write and the read take the same path to chiplet-local `0x802000000` (`tt_cluster.cpp:907-939`; UMD `simulation_chip.cpp:75-94`, `grendel_tt_device.cpp:74-86`, `mimir_chippy_memory.cpp`; chippy `Memory::bulk_write_bytes` and `bulk_read_bytes`), and the transactor carries 56 address bits (grendelemulation `models/mimir/emu_hdl/axi_sim_to_smc.sv:24-27`). `0x2000000` is the CCE0 config base in chippy's local map (`lib/arch/grendel/mimir.h:27`), but nothing on the host drops the GDDR base.
- The address is valid. Every bring-up programmed tile interleave with the 8 GB realign (`GDDR NoC config: interleave_mode=1, realign_mode=2`), which maps the address to tile 0, slice 0, byte `0x800000` (page 256 of 4096), per the shim model in grendelemulation `models/mimir/tests/mim-573/test_tile_intlv_realign.py:63-75`. No other walk offset maps there. MIM-573 affects only non-power-of-two sizes.
- Not page exhaustion, which is a fatal emulator error (grendelemulation `models/mimir/tests/test_gddr.py:189-206`), and not an AXI error response: the transactor raises on a non-OKAY response (zebu `axi4m.py:343-367`), the SiVal server then drops the connection (grendelemulation `py/syseng_common/server.py:334-338`) and the test would have aborted. The server logs of these sessions show no AXI failure. So the write and the read both completed OKAY, and the read returned 0.
- Next: rerun row 1 with DPI on (see Manual D2D AXI monitor step). The SMC AXI monitor logs both beats with their responses, and the slice-0 ddrmon log shows whether the write reached `ddr_model`.

## Results on `mimir_2x_package`, 2026-11-04

Finished: all 481 invocations ran. One line per test case, from the gtest XML reports. Time is the case's own time. The case time includes opening the device; each invocation adds about 0.8 s on top. A case with no report (killed by the time limit, or the emulator stopped) shows the filter it ran and the invocation's wall time. Run against `6ea8f29c6e4` with UMD `f0d497f3` (`kstevens/grendel-emu-additions`), in the same Debug build as the `mimir_1x1` run, with the same `core_descriptor.cpp` fix. Emulator `models/mmk`, each session after `bringup_two_mimir.py`: soc-c-30 (rows 1 to 3a); soc-c-27 (row 3b); soc-c-31 (row 3b); soc-c-27 (row 3b); soc-zebu-01 (rows 3b to 15); soc-c-28 (row 16); soc-c-27 (the contract tests); rv-zebu-08 (rows 17 to 48); soc-zebu-01 (rows 49 to 60); soc-zebu-01 (rows 61 to 62a); soc-zebu-03 (rows 63 to 71); soc-c-24 (row 72); soc-c-24 (row 73); soc-c-29 (row 74); soc-c-24 (row 75); soc-c-28 (row 76); soc-c-24 (rows 77 to 78); soc-zebu-03 (row 79); soc-zebu-03 (row 80); soc-c-27 (row 81); rv-zebu-07 (rows 82 to 83); rv-zebu-08 (row 84); soc-zebu-04 (rows 90 to 91); rv-zebu-07 (row 92); soc-zebu-03 (row 93); soc-c-23 (rows 95 to 96); soc-c-25 (rows 97 to 101); soc-zebu-03 (row 102); soc-c-25 (rows 103 to 105); soc-c-29 (row 106); soc-c-29 (row 106); rv-zebu-08 (row 106); soc-c-24 (row 106); rv-zebu-08 (row 106); soc-c-25 (row 106); soc-c-30 (row 106); soc-c-30 (no case finished yet); soc-c-27 (rows 107 to 109); soc-zebu-02 (row 110); rv-zebu-07 (row 111); soc-zebu-04 (row 112); soc-zebu-04 (row 113); soc-c-30 (rows 114 to 121); soc-c-31 (rows 122 to 123); soc-c-31 (row 124); soc-c-26 (row 125); soc-zebu-02 (row 126); soc-zebu-01 (rows 128 to 129); rv-zebu-08 (row 2c).

Changes from the run order:
- Row 2c runs last, as on `mimir_1x1`: it leaves CCE1 failing firmware init until the emulator restarts.
- Each invocation runs with `timeout 7200` instead of `timeout 3600`, because rows cover more views and CCEs on this package and the emulator is slower. Row 1 ran without the `timeout` prefix after a driver change, under an 8 h limit from the driver. Rows 14 to 16 run with `timeout 9900`, because row 12 took 6625 s; row 13 had already started and had its limit lifted to 9900 s while it ran.
- From row 3b, and from row 72 on, each case runs as its own invocation.
- `SizeAlignmentMatrix` (row 3b) runs with `timeout 43200` instead of `timeout 3600`: on `mimir_1x1` each CCE took about 2 h 20 min, and host access on `models/mmk` is several times slower.
- A fresh emulator session, with bring-up, starts before row 3b and before each `SizeAlignmentMatrix` case after the first, so no session reaches the emulator job's 24 h limit inside a `SizeAlignmentMatrix` case. Fresh sessions also start before rows 17 and 81, as the run order asks, and from row 16 on before each switch to a different test suite, so each suite starts from a clean emulator.
- After a case fails with a kernel timeout code, a host deadline, the MIM-601 message or "Got empty response from server", the emulator restarts, bring-up reruns and the run resumes at the next case. The emulator job has a 24 h limit, so the case running when it ends fails with "Got empty response from server". A case lost that way runs once more in the next session.

| Row | Test | Result | Time |
|---|---|---|---|
| 1 | `MimirHostDram.ReconnectPersists` | pass | 12 s |
| 1 | `MimirHostDram.HostSideRejections` | fail: As on `mimir_1x1`: a 6-byte write clears the 2 unwritten bytes of its last word, read `0x00001615`, expected `0xFFFF1615`. Chippy zero-fills a partial trailing word (chippy `transport_interface.h:73-78`) | 0.2 s |
| 1 | `MimirHostTl1.Tl1MapLayout` | pass | 0.6 s |
| 1 | `MimirHostTl1.BoundaryAndIsolation` | pass | 3.0 s |
| 1 | `MimirHostTl1.HostSideRejections` | pass | 0.2 s |
| 1 | `HostDram/MimirHostDramView.PatternReadback/View0` | pass | 48 min 26 s |
| 1 | `HostDram/MimirHostDramView.PatternReadback/View1` | pass | 50 min 3 s |
| 1 | `HostDram/MimirHostDramView.PatternReadback/View2` | pass | 49 min 45 s |
| 1 | `HostDram/MimirHostDramView.PatternReadback/View3` | pass | 49 min 43 s |
| 1 | `HostDram/MimirHostDramView.BoundaryAndAlias/View0` | fail: As View0 on `mimir_1x1`: the alias walk at view offset `0x2000000` read `0x00000000`, expected the guard `0x6aa18000`. View 2 fails the same way; views 1 and 3 pass, so on both packages it is the first view of each Mimir | 2.1 s |
| 1 | `HostDram/MimirHostDramView.BoundaryAndAlias/View1` | pass | 2.0 s |
| 1 | `HostDram/MimirHostDramView.BoundaryAndAlias/View2` | fail: Same as View0: the alias walk at view offset `0x2000000` read `0x00000000`, expected the guard `0x6aa18000`. View 2 is the first view of Mimir 1 | 2.0 s |
| 1 | `HostDram/MimirHostDramView.BoundaryAndAlias/View3` | pass | 2.0 s |
| 1 | `HostTl1/MimirHostTl1Cce.PerCceVisibility/Cce0` | pass | 17 min 56 s |
| 1 | `HostTl1/MimirHostTl1Cce.PerCceVisibility/Cce1` | pass | 18 min 3 s |
| 1 | `HostTl1/MimirHostTl1Cce.PerCceVisibility/Cce2` | pass | 18 min 28 s |
| 1 | `HostTl1/MimirHostTl1Cce.PerCceVisibility/Cce3` | pass | 18 min 9 s |
| 2a | `MimirFwPreflight.FirmwareLoadPreflight` | fail: As on `mimir_1x1`, now on all four CCEs: the 1-byte `fw_shared_globals_ready` write clears the other 3 bytes of its TL1 word (CCE0 `0x000000cd`, expected `0x5a5a5acd`); the same chippy partial-word write | 1 min 36 s |
| 2b | `MimirFwLoad.LoadAndReachInitDone` | pass | 1 min 57 s |
| 2b | `MimirFwLoad.ImageManifest` | pass | 25 s |
| 2b | `MimirFwLoad.AllCcesIndependent` | pass | 11 min 53 s |
| 2b | `FirmwareLoad/MimirFwLaunch.NoOpLaunchCompletes/Cce0` | pass | 28 s |
| 2b | `FirmwareLoad/MimirFwLaunch.NoOpLaunchCompletes/Cce1` | pass | 27 s |
| 2b | `FirmwareLoad/MimirFwLaunch.NoOpLaunchCompletes/Cce2` | pass | 28 s |
| 2b | `FirmwareLoad/MimirFwLaunch.NoOpLaunchCompletes/Cce3` | pass | 28 s |
| 2d | `MimirFwReinit.ReloadsOnEveryOpen` | pass | 1 min 36 s |
| 3 | `MimirKernelLoad.SameBinaryAllCcesUnicast` | pass | 54 s |
| 3 | `KernelLoad/MimirKernelLoad.MinimalKernelExecutes/Cce0` | pass | 28 s |
| 3 | `KernelLoad/MimirKernelLoad.MinimalKernelExecutes/Cce1` | pass | 28 s |
| 3 | `KernelLoad/MimirKernelLoad.MinimalKernelExecutes/Cce2` | pass | 28 s |
| 3 | `KernelLoad/MimirKernelLoad.MinimalKernelExecutes/Cce3` | pass | 28 s |
| 3 | `KernelLoad/MimirKernelLoad.EachHartAlone/Cce0` | pass | 1 min 6 s |
| 3 | `KernelLoad/MimirKernelLoad.EachHartAlone/Cce1` | pass | 1 min 3 s |
| 3 | `KernelLoad/MimirKernelLoad.EachHartAlone/Cce2` | pass | 1 min 4 s |
| 3 | `KernelLoad/MimirKernelLoad.EachHartAlone/Cce3` | pass | 1 min 4 s |
| 3 | `KernelLoad/MimirKernelLoad.ArgsReplaceStale/Cce0` | pass | 39 s |
| 3 | `KernelLoad/MimirKernelLoad.ArgsReplaceStale/Cce1` | pass | 39 s |
| 3 | `KernelLoad/MimirKernelLoad.ArgsReplaceStale/Cce2` | pass | 39 s |
| 3 | `KernelLoad/MimirKernelLoad.ArgsReplaceStale/Cce3` | pass | 39 s |
| 3 | `KernelLoad/MimirKernelLoad.FirmwareKeepsRunningAcrossReopen/Cce0` | pass | 1 min 22 s |
| 3 | `KernelLoad/MimirKernelLoad.FirmwareKeepsRunningAcrossReopen/Cce1` | pass | 1 min 8 s |
| 3 | `KernelLoad/MimirKernelLoad.FirmwareKeepsRunningAcrossReopen/Cce2` | pass | 1 min 1 s |
| 3 | `KernelLoad/MimirKernelLoad.FirmwareKeepsRunningAcrossReopen/Cce3` | pass | 1 min 0 s |
| 3 | `LocalTl1/MimirLocalTl1.SameMimirCopy/Cce0To1` | pass | 53 s |
| 3 | `LocalTl1/MimirLocalTl1.SameMimirCopy/Cce1To0` | pass | 52 s |
| 3 | `LocalTl1/MimirLocalTl1.SameMimirCopy/Cce2To3` | pass | 52 s |
| 3 | `LocalTl1/MimirLocalTl1.SameMimirCopy/Cce3To2` | pass | 52 s |
| 3a | `DmSuitePort/MimirDmIdma.Copy/OwnTl1` | pass | 38 s |
| 3a | `DmSuitePort/MimirDmIdma.Copy/LocalGddr` | pass | 50 s |
| 3a | `DmSuitePort/MimirDmIdma.Copy/LocalGddr1` | pass | 50 s |
| 3b | `LocalDram/MimirLocalDram.CceWritesTl1ToDram/Cce0` | pass | 8 min 19 s |
| 3b | `LocalDram/MimirLocalDram.CceWritesTl1ToDram/Cce1` | pass | 8 min 15 s |
| 3b | `LocalDram/MimirLocalDram.CceWritesTl1ToDram/Cce2` | pass | 8 min 21 s |
| 3b | `LocalDram/MimirLocalDram.CceWritesTl1ToDram/Cce3` | pass | 8 min 24 s |
| 3b | `LocalDram/MimirLocalDram.CceReadsDramToTl1/Cce0` | pass | 8 min 9 s |
| 3b | `LocalDram/MimirLocalDram.CceReadsDramToTl1/Cce1` | pass | 8 min 14 s |
| 3b | `LocalDram/MimirLocalDram.CceReadsDramToTl1/Cce2` | pass | 8 min 16 s |
| 3b | `LocalDram/MimirLocalDram.CceReadsDramToTl1/Cce3` | pass | 8 min 16 s |
| 3b | `LocalDram/MimirLocalDram.LoopbackChecksIntermediate/Cce0` | pass | 19 min 4 s |
| 3b | `LocalDram/MimirLocalDram.LoopbackChecksIntermediate/Cce1` | pass | 20 min 43 s |
| 3b | `LocalDram/MimirLocalDram.LoopbackChecksIntermediate/Cce2` | pass | 16 min 43 s |
| 3b | `LocalDram/MimirLocalDram.LoopbackChecksIntermediate/Cce3` | pass | 16 min 58 s |
| 3b | `LocalDram/MimirLocalDram.SizeAlignmentMatrix/Cce0` | pass | 7 h 7 min |
| 3b | `LocalDram/MimirLocalDram.SizeAlignmentMatrix/Cce1` | pass | 7 h 3 min |
| 3b | `LocalDram/MimirLocalDram.SizeAlignmentMatrix/Cce2` | pass | 7 h 4 min |
| 3b | `LocalDram/MimirLocalDram.SizeAlignmentMatrix/Cce3` | pass | 7 h 3 min |
| 3b | `LocalDram/MimirLocalDram.RepeatedAndOverlappingPerHart/Cce0` | pass | 1 h 1 min |
| 3b | `LocalDram/MimirLocalDram.RepeatedAndOverlappingPerHart/Cce1` | pass | 1 h 1 min |
| 3b | `LocalDram/MimirLocalDram.RepeatedAndOverlappingPerHart/Cce2` | pass | 1 h 0 min |
| 3b | `LocalDram/MimirLocalDram.RepeatedAndOverlappingPerHart/Cce3` | pass | 1 h 1 min |
| 4 | `D2dDirect/MimirRemoteTl1.WriteRead/M0Cce1ToM1Cce0` | pass | 1 min 41 s |
| 4 | `D2dDirect/MimirRemoteTl1.WriteRead/M1Cce0ToM0Cce1` | pass | 1 min 40 s |
| 4 | `D2dDirect/MimirRemoteTl1.HostInitRemoteRead/M0Cce1ToM1Cce0` | pass | 1 min 11 s |
| 4 | `D2dDirect/MimirRemoteTl1.HostInitRemoteRead/M1Cce0ToM0Cce1` | pass | 1 min 11 s |
| 4 | `D2dDirect/MimirRemoteTl1.SizeAlignment/M0Cce1ToM1Cce0` | pass | 2 min 36 s |
| 4 | `D2dDirect/MimirRemoteTl1.SizeAlignment/M1Cce0ToM0Cce1` | pass | 2 min 34 s |
| 5 | `MimirRemoteTl1Bidir.Concurrent` | pass | 2 min 53 s |
| 6 | `D2dRouted/MimirRemoteTl1.WriteRead/M0Cce0ToM1Cce0` | pass | 1 min 42 s |
| 6 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M0Cce0ToM1Cce0` | pass | 1 min 11 s |
| 6 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M0Cce0ToM1Cce0` | pass | 2 min 35 s |
| 7 | `D2dRouted/MimirRemoteTl1.WriteRead/M0Cce1ToM1Cce1` | pass | 1 min 42 s |
| 7 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M0Cce1ToM1Cce1` | pass | 1 min 13 s |
| 7 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M0Cce1ToM1Cce1` | pass | 2 min 34 s |
| 8 | `D2dRouted/MimirRemoteTl1.WriteRead/M1Cce0ToM0Cce0` | pass | 1 min 42 s |
| 8 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M1Cce0ToM0Cce0` | pass | 1 min 11 s |
| 8 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M1Cce0ToM0Cce0` | pass | 2 min 34 s |
| 9 | `D2dRouted/MimirRemoteTl1.WriteRead/M1Cce1ToM0Cce1` | pass | 1 min 42 s |
| 9 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M1Cce1ToM0Cce1` | pass | 1 min 10 s |
| 9 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M1Cce1ToM0Cce1` | pass | 2 min 35 s |
| 10 | `D2dRouted/MimirRemoteTl1.WriteRead/M0Cce0ToM1Cce1` | pass | 1 min 41 s |
| 10 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M0Cce0ToM1Cce1` | pass | 1 min 11 s |
| 10 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M0Cce0ToM1Cce1` | pass | 4 min 19 s |
| 11 | `D2dRouted/MimirRemoteTl1.WriteRead/M1Cce1ToM0Cce0` | pass | 1 min 42 s |
| 11 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M1Cce1ToM0Cce0` | pass | 1 min 10 s |
| 11 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M1Cce1ToM0Cce0` | pass | 3 min 24 s |
| 12 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce1View0` | skip: DRAM view 0 is on the source Mimir, not a remote view. | 0.0 s |
| 12 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce1View1` | skip: DRAM view 1 is on the source Mimir, not a remote view. | 0.0 s |
| 12 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce1View2` | pass | 7 min 36 s |
| 12 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce1View3` | pass | 7 min 36 s |
| 12 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce1View0` | skip: DRAM view 0 is on the source Mimir, not a remote view. | 0.0 s |
| 12 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce1View1` | skip: DRAM view 1 is on the source Mimir, not a remote view. | 0.0 s |
| 12 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce1View2` | pass | 7 min 49 s |
| 12 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce1View3` | pass | 7 min 47 s |
| 12 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce1View0` | skip: DRAM view 0 is on the source Mimir, not a remote view. | 0.0 s |
| 12 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce1View1` | skip: DRAM view 1 is on the source Mimir, not a remote view. | 0.0 s |
| 12 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce1View2` | pass | 11 min 6 s |
| 12 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce1View3` | pass | 11 min 8 s |
| 12 | `RemoteDram/MimirRemoteDramTrafficMatrix.TrafficMatrix/SrcM0Cce1` | pass | 57 min 21 s |
| 13 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce0View0` | pass | 7 min 43 s |
| 13 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce0View1` | pass | 7 min 50 s |
| 13 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce0View2` | skip: DRAM view 2 is on the source Mimir, not a remote view. | 0.0 s |
| 13 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce0View3` | skip: DRAM view 3 is on the source Mimir, not a remote view. | 0.0 s |
| 13 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce0View0` | pass | 8 min 4 s |
| 13 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce0View1` | pass | 7 min 51 s |
| 13 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce0View2` | skip: DRAM view 2 is on the source Mimir, not a remote view. | 0.0 s |
| 13 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce0View3` | skip: DRAM view 3 is on the source Mimir, not a remote view. | 0.0 s |
| 13 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce0View0` | pass | 12 min 4 s |
| 13 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce0View1` | pass | 11 min 31 s |
| 13 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce0View2` | skip: DRAM view 2 is on the source Mimir, not a remote view. | 0.0 s |
| 13 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce0View3` | skip: DRAM view 3 is on the source Mimir, not a remote view. | 0.0 s |
| 13 | `RemoteDram/MimirRemoteDramTrafficMatrix.TrafficMatrix/SrcM1Cce0` | pass | 1 h 3 min |
| 14 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce0View0` | skip: DRAM view 0 is on the source Mimir, not a remote view. | 0.0 s |
| 14 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce0View1` | skip: DRAM view 1 is on the source Mimir, not a remote view. | 0.0 s |
| 14 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce0View2` | pass | 8 min 21 s |
| 14 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM0Cce0View3` | pass | 9 min 1 s |
| 14 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce0View0` | skip: DRAM view 0 is on the source Mimir, not a remote view. | 0.0 s |
| 14 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce0View1` | skip: DRAM view 1 is on the source Mimir, not a remote view. | 0.0 s |
| 14 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce0View2` | pass | 8 min 55 s |
| 14 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM0Cce0View3` | pass | 8 min 32 s |
| 14 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce0View0` | skip: DRAM view 0 is on the source Mimir, not a remote view. | 0.0 s |
| 14 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce0View1` | skip: DRAM view 1 is on the source Mimir, not a remote view. | 0.0 s |
| 14 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce0View2` | pass | 12 min 37 s |
| 14 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM0Cce0View3` | pass | 12 min 5 s |
| 14 | `RemoteDram/MimirRemoteDramTrafficMatrix.TrafficMatrix/SrcM0Cce0` | pass | 1 h 3 min |
| 15 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce1View0` | pass | 8 min 42 s |
| 15 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce1View1` | pass | 8 min 4 s |
| 15 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce1View2` | skip: DRAM view 2 is on the source Mimir, not a remote view. | 0.0 s |
| 15 | `RemoteDram/MimirRemoteDram.WriteFromTl1/SrcM1Cce1View3` | skip: DRAM view 3 is on the source Mimir, not a remote view. | 0.0 s |
| 15 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce1View0` | pass | 9 min 9 s |
| 15 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce1View1` | pass | 8 min 43 s |
| 15 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce1View2` | skip: DRAM view 2 is on the source Mimir, not a remote view. | 0.0 s |
| 15 | `RemoteDram/MimirRemoteDram.ReadToTl1/SrcM1Cce1View3` | skip: DRAM view 3 is on the source Mimir, not a remote view. | 0.0 s |
| 15 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce1View0` | pass | 11 min 29 s |
| 15 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce1View1` | pass | 11 min 25 s |
| 15 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce1View2` | skip: DRAM view 2 is on the source Mimir, not a remote view. | 0.0 s |
| 15 | `RemoteDram/MimirRemoteDram.Tl1RoundTrip/SrcM1Cce1View3` | skip: DRAM view 3 is on the source Mimir, not a remote view. | 0.0 s |
| 15 | `RemoteDram/MimirRemoteDramTrafficMatrix.TrafficMatrix/SrcM1Cce1` | pass | 1 h 1 min |
| 16 | `MimirRemoteMixed.FanOutFanIn` | pass | 2 min 19 s |
| 16 | `MimirRemoteMixed.Concurrent` | pass | 6 min 7 s |
| contract | `MimirEmu.CceSramChannelsDoNotAliasThroughMetalCluster` | pass | 0.1 s |
| contract | `MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi` | pass | 32 s |
| contract | `MimirEmu.AllCcesCopyLocalMimirBlock` | pass | 1 min 58 s |
| contract | `MimirEmu/AllocatedDramBufferHostLoopback.RoundTrip/Partition0` | pass | 29 s |
| contract | `MimirEmu/AllocatedDramBufferHostLoopback.RoundTrip/Partition1` | pass | 29 s |
| contract | `MimirEmu/AllocatedDramBufferHostLoopback.RoundTrip/Partition2` | pass | 28 s |
| contract | `MimirEmu/AllocatedDramBufferHostLoopback.RoundTrip/Partition3` | pass | 29 s |
| contract | `MimirEmu/DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip/Partition0` | pass | 28 s |
| contract | `MimirEmu/DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip/Partition1` | pass | 28 s |
| contract | `MimirEmu/DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip/Partition2` | pass | 27 s |
| contract | `MimirEmu/DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip/Partition3` | pass | 28 s |
| contract | `MimirEmu/CceSramRoundTripThroughUnitMesh.RoundTrip/Cce0` | pass | 27 s |
| contract | `MimirEmu/CceSramRoundTripThroughUnitMesh.RoundTrip/Cce1` | pass | 27 s |
| contract | `MimirEmu/CceSramRoundTripThroughUnitMesh.RoundTrip/Cce2` | pass | 27 s |
| contract | `MimirEmu/CceSramRoundTripThroughUnitMesh.RoundTrip/Cce3` | pass | 27 s |
| contract | `MimirEmu/CceSramRoundTripThroughMetalCluster.RoundTrip/Cce0` | pass | 0.1 s |
| contract | `MimirEmu/CceSramRoundTripThroughMetalCluster.RoundTrip/Cce1` | pass | 0.0 s |
| contract | `MimirEmu/CceSramRoundTripThroughMetalCluster.RoundTrip/Cce2` | pass | 0.0 s |
| contract | `MimirEmu/CceSramRoundTripThroughMetalCluster.RoundTrip/Cce3` | pass | 0.0 s |
| contract | `MimirEmu/HartZeroRunsDramKernel.WritesMagic/Cce0` | pass | 29 s |
| contract | `MimirEmu/HartZeroRunsDramKernel.WritesMagic/Cce1` | pass | 29 s |
| contract | `MimirEmu/HartZeroRunsDramKernel.WritesMagic/Cce2` | pass | 30 s |
| contract | `MimirEmu/HartZeroRunsDramKernel.WritesMagic/Cce3` | pass | 29 s |
| contract | `MimirEmu/AllHartsRunDramKernel.WritesMagic/Cce0` | pass | 34 s |
| contract | `MimirEmu/AllHartsRunDramKernel.WritesMagic/Cce1` | pass | 34 s |
| contract | `MimirEmu/AllHartsRunDramKernel.WritesMagic/Cce2` | pass | 32 s |
| contract | `MimirEmu/AllHartsRunDramKernel.WritesMagic/Cce3` | pass | 35 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition0` | pass | 29 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition1` | pass | 29 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition2` | skip: Configured descriptor does not expose CCE 0 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition3` | skip: Configured descriptor does not expose CCE 0 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition0` | pass | 29 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition1` | pass | 28 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition2` | skip: Configured descriptor does not expose CCE 1 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition3` | skip: Configured descriptor does not expose CCE 1 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition0` | skip: Configured descriptor does not expose CCE 2 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition1` | skip: Configured descriptor does not expose CCE 2 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition2` | pass | 29 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition3` | pass | 29 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition0` | skip: Configured descriptor does not expose CCE 3 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition1` | skip: Configured descriptor does not expose CCE 3 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition2` | pass | 30 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition3` | pass | 29 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition0` | pass | 47 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition1` | pass | 47 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition2` | skip: Configured descriptor does not expose CCE 0 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition3` | skip: Configured descriptor does not expose CCE 0 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition0` | pass | 47 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition1` | pass | 47 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition2` | skip: Configured descriptor does not expose CCE 1 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition3` | skip: Configured descriptor does not expose CCE 1 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition0` | skip: Configured descriptor does not expose CCE 2 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition1` | skip: Configured descriptor does not expose CCE 2 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition2` | pass | 47 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition3` | pass | 46 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition0` | skip: Configured descriptor does not expose CCE 3 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition1` | skip: Configured descriptor does not expose CCE 3 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition2` | pass | 47 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition3` | pass | 46 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition0` | pass | 51 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition1` | pass | 51 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition2` | skip: Configured descriptor does not expose CCE 0 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition3` | skip: Configured descriptor does not expose CCE 0 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition0` | pass | 52 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition1` | pass | 56 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition2` | skip: Configured descriptor does not expose CCE 1 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition3` | skip: Configured descriptor does not expose CCE 1 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition0` | skip: Configured descriptor does not expose CCE 2 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition1` | skip: Configured descriptor does not expose CCE 2 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition2` | pass | 53 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition3` | pass | 54 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition0` | skip: Configured descriptor does not expose CCE 3 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition1` | skip: Configured descriptor does not expose CCE 3 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition2` | pass | 57 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition3` | pass | 55 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce0Partition0` | pass | 36 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce0Partition1` | pass | 34 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce0Partition2` | skip: Configured descriptor does not expose CCE 0 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce0Partition3` | skip: Configured descriptor does not expose CCE 0 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce1Partition0` | pass | 40 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce1Partition1` | pass | 37 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce1Partition2` | skip: Configured descriptor does not expose CCE 1 partition 2 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce1Partition3` | skip: Configured descriptor does not expose CCE 1 partition 3 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce2Partition0` | skip: Configured descriptor does not expose CCE 2 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce2Partition1` | skip: Configured descriptor does not expose CCE 2 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce2Partition2` | pass | 38 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce2Partition3` | pass | 36 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce3Partition0` | skip: Configured descriptor does not expose CCE 3 partition 0 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce3Partition1` | skip: Configured descriptor does not expose CCE 3 partition 1 as a local partition. | 0.0 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce3Partition2` | pass | 35 s |
| contract | `MimirEmuSameMimirDirect/AllHartsIdmaWriteGddr.WritesMagic/Cce3Partition3` | pass | 35 s |
| contract | `MimirEmu/WritesCrossMimirCceSram.WritesMagic/Cce0` | pass | 28 s |
| contract | `MimirEmu/WritesCrossMimirCceSram.WritesMagic/Cce1` | pass | 28 s |
| contract | `MimirEmu/WritesCrossMimirCceSram.WritesMagic/Cce2` | pass | 27 s |
| contract | `MimirEmu/WritesCrossMimirCceSram.WritesMagic/Cce3` | pass | 28 s |
| contract | `MimirEmu/PrefetchesLocalGddrToRemoteCceSram.RoundTrip/Cce0` | pass | 30 s |
| contract | `MimirEmu/PrefetchesLocalGddrToRemoteCceSram.RoundTrip/Cce1` | pass | 29 s |
| contract | `MimirEmu/PrefetchesLocalGddrToRemoteCceSram.RoundTrip/Cce2` | pass | 29 s |
| contract | `MimirEmu/PrefetchesLocalGddrToRemoteCceSram.RoundTrip/Cce3` | pass | 29 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition2` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition3` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition2` | pass | 28 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce1Partition3` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition0` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition1` | pass | 31 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition2` | skip: Configured descriptor does not expose CCE 2 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce2Partition3` | skip: Configured descriptor does not expose CCE 2 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition0` | pass | 28 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition1` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition2` | skip: Configured descriptor does not expose CCE 3 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip/Cce3Partition3` | skip: Configured descriptor does not expose CCE 3 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition2` | pass | 28 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition3` | pass | 30 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition2` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce1Partition3` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition0` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition1` | pass | 28 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition2` | skip: Configured descriptor does not expose CCE 2 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce2Partition3` | skip: Configured descriptor does not expose CCE 2 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition0` | pass | 28 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition1` | pass | 27 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition2` | skip: Configured descriptor does not expose CCE 3 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip/Cce3Partition3` | skip: Configured descriptor does not expose CCE 3 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition2` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition3` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition2` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce1Partition3` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition0` | pass | 34 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition1` | pass | 34 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition2` | skip: Configured descriptor does not expose CCE 2 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce2Partition3` | skip: Configured descriptor does not expose CCE 2 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition0` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition1` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition2` | skip: Configured descriptor does not expose CCE 3 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip/Cce3Partition3` | skip: Configured descriptor does not expose CCE 3 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce0Partition0` | skip: Configured descriptor does not expose CCE 0 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce0Partition1` | skip: Configured descriptor does not expose CCE 0 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce0Partition2` | pass | 31 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce0Partition3` | pass | 34 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce1Partition0` | skip: Configured descriptor does not expose CCE 1 partition 0 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce1Partition1` | skip: Configured descriptor does not expose CCE 1 partition 1 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce1Partition2` | pass | 31 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce1Partition3` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce2Partition0` | pass | 32 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce2Partition1` | pass | 32 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce2Partition2` | skip: Configured descriptor does not expose CCE 2 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce2Partition3` | skip: Configured descriptor does not expose CCE 2 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce3Partition0` | pass | 33 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce3Partition1` | pass | 32 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce3Partition2` | skip: Configured descriptor does not expose CCE 3 partition 2 as a remote partition. | 0.0 s |
| contract | `MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic/Cce3Partition3` | skip: Configured descriptor does not expose CCE 3 partition 3 as a remote partition. | 0.0 s |
| contract | `MimirEmu/CopiesEveryMimirBlock.RoundTrip/Cce0` | pass | 29 s |
| contract | `MimirEmu/CopiesEveryMimirBlock.RoundTrip/Cce1` | pass | 29 s |
| contract | `MimirEmu/CopiesEveryMimirBlock.RoundTrip/Cce2` | pass | 29 s |
| contract | `MimirEmu/CopiesEveryMimirBlock.RoundTrip/Cce3` | pass | 29 s |
| 17 | `MimirPrefetcher.SupportPreconditionsHold` | pass | 24 s |
| 18 | `MimirPrefetcher.SupportPreconditionsFailOnUninitializedSender` | pass | 24 s |
| 19 | `MimirPrefetcher.SupportStartStopSucceeds` | pass | 30 s |
| 20 | `MimirPrefetcher.SupportPreconditionsHoldWhileReceiverRuns` | pass | 29 s |
| 21 | `MimirPrefetcher.GcbLayoutLocalReceiver` | pass | 26 s |
| 22 | `MimirPrefetcher.GcbLayoutRemoteReceiver` | pass | 26 s |
| 23 | `MimirPrefetcher.GcbLayoutTwoSenders` | pass | 28 s |
| 24 | `MimirPrefetcher.GcbLayoutRejectsConflictingReceivers` | pass | 25 s |
| 25 | `MimirPrefetcher.GcbSmokeDiscardDrain` | pass | 31 s |
| 26 | `MimirPrefetcher.GcbSmokeValidatorLocal` | pass | 34 s |
| 27 | `MimirPrefetcher.GcbSmokeValidatorWrongSeed` | pass | 34 s |
| 28 | `MimirPrefetcher.GcbSmokeResize` | pass | 32 s |
| 29 | `MimirPrefetcher.GcbSmokeValidatorD2d` | pass | 34 s |
| 30 | `MimirPrefetcher.GcbSmokeDrainUntilStopped` | pass | 31 s |
| 31 | `MimirPrefetcher.GcbSmokeStallUntilReleased` | pass | 49 s |
| 32 | `MimirPrefetcher.GcbSmokeDrainTimeout` | pass | 49 s |
| 33 | `MimirPrefetcher.GcbSmokeReceiverWaitTimeout` | pass | 45 s |
| 34 | `MimirPrefetcher.GcbSmokeRelaunchResumes` | pass | 35 s |
| 35 | `MimirPrefetcher.LifecycleStartStop` | pass | 28 s |
| 36 | `MimirPrefetcher.LifecycleStartStopTwice` | pass | 34 s |
| 37 | `MimirPrefetcher.LifecycleBusySenderFailsStart` | pass | 28 s |
| 38 | `MimirPrefetcher.LifecycleQueueRejectsReceiverOnSender` | pass | 34 s |
| 39 | `MimirPrefetcher.LifecycleWaitCq` | pass | 33 s |
| 40 | `MimirPrefetcher.LifecycleDrainAtStop` | pass | 36 s |
| 41 | `MimirPrefetcher.PrefetchOnePagePerSessionLocal0To1` | pass | 49 s |
| 42 | `MimirPrefetcher.PrefetchOnePagePerSessionLocal3To2` | pass | 48 s |
| 43 | `MimirPrefetcher.PrefetchOnePagePerSessionD2d0To2` | pass | 49 s |
| 44 | `MimirPrefetcher.PrefetchOnePagePerSessionD2d3To1` | pass | 52 s |
| 45 | `MimirPrefetcher.PrefetchFanOut0To123` | pass | 1 min 17 s |
| 46 | `MimirPrefetcher.PrefetchFanOut3To012` | pass | 1 min 19 s |
| 47 | `MimirPrefetcher.PrefetchTwoSenders01To2` | pass | 60 s |
| 48 | `MimirPrefetcher.PrefetchTwoSenders02To3` | pass | 1 min 1 s |
| 49 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/SubChunkLocal0To1` | pass | 1 min 21 s |
| 50 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/SubChunkD2d0To2` | pass | 1 min 23 s |
| 51 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/BelowStageThirdLocal0To1` | pass | 1 min 18 s |
| 52 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/BelowStageThirdD2d0To2` | pass | 1 min 17 s |
| 53 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/PartialStageChunkLocal0To1` | pass | 1 min 12 s |
| 54 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AboveStageThirdLocal0To1` | pass | 56 s |
| 55 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AboveStageThirdD2d0To2` | pass | 1 min 0 s |
| 56 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/PageSplitLocal0To1` | pass | 1 min 3 s |
| 57 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AbaLocal0To1` | pass | 1 min 25 s |
| 58 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/AbaD2d0To2` | pass | 1 min 19 s |
| 59 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/WrapAndResizeLocal0To1` | pass | 1 min 38 s |
| 60 | `TensorPrefetcher/MimirPrefetcherSingleGcb.Delivers/WrapAndResizeD2d0To2` | pass | 1 min 33 s |
| 61 | `TensorPrefetcher/MimirPrefetcherGcbSwitch.Delivers/GcbSwitchLocal0To1` | pass | 1 min 1 s |
| 62 | `TensorPrefetcher/MimirPrefetcherGcbSwitch.Delivers/GcbSwitchD2d0To2` | pass | 1 min 2 s |
| 62a | `TensorPrefetcher/MimirPrefetcherGcbSwitch.Delivers/ReceiverSwitch0To1Then2` | pass | 1 min 3 s |
| 63 | `MimirPrefetcher.PrefetchFenceHoldsDeliveryLocal0To1` | pass | 45 s |
| 65 | `MimirPrefetcher.QueueRejectsKRowMajorTensor` | pass | 39 s |
| 66 | `MimirPrefetcher.QueueRejectsStreamingTensor` | pass | 38 s |
| 67 | `MimirPrefetcher.QueueRejectsRangePastTensor` | pass | 38 s |
| 68 | `MimirPrefetcher.QueueRejectsRangePastView` | pass | 39 s |
| 69 | `MimirPrefetcher.PrefetchRecoversAfterMismatchD2d2To0` | pass | 54 s |
| 70 | `MimirPrefetcher.PrefetchRecoversAfterCreditTimeoutD2d1To3` | pass | 1 min 29 s |
| 71 | `MimirPrefetcher.PrefetchRecoversAfterDrainTimeoutD2d1To3` | pass | 1 min 12 s |
| 72 | `DmSuitePort/MimirDmRuntimeArgs.IllegalTooManyRuntimeArgs/Cce0` | pass | 32 s |
| 72 | `DmSuitePort/MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs/Cce0` | pass | 29 s |
| 73 | `DmSuitePort/MimirDmMeshBufferDram.InterleavedPlacement/mimir_2x_package` | pass | 1 min 15 s |
| 73 | `DmSuitePort/MimirDmMeshBufferDram.WriteRead/mimir_2x_package` | pass | 35 s |
| 73 | `DmSuitePort/MimirDmMeshBufferDram.MultipleWriteReadRounds/mimir_2x_package` | pass | 1 min 46 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/OwnTl1` | pass | 39 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/MaxPacket` | pass | 1 min 0 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/SiblingTl1` | pass | 52 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr` | pass | 54 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr1` | pass | 51 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1` | pass | 51 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/RemoteGddr` | pass | 53 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1D` | pass | 39 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1DLocalGddr` | pass | 38 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLine` | pass | 37 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLineLocalGddr` | pass | 36 s |
| 74 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1KernelVerified` | pass | 55 s |
| 75 | `DmSuitePort/MimirDmCache.Op/NoFlushNotVisible` | pass | 40 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1FlushLine` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1FlushFull` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1InvalidateLine` | pass | 37 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushLine` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushRange` | pass | 37 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushFull` | pass | 45 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2InvalidateLine` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/InvalidateFreshRead` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/HartToHartCoherent` | fail: As on `mimir_1x1`: hart 1's warm read before hart 0's stores returned the old value in 16 of 64 words, expected 64 (`test_mimir_dm_cache.cpp:238`) | 51 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2InvalidateFreshRead` | fail: As on `mimir_1x1`: the cached re-read before the L2 invalidate saw 0 of 64 stale words, expected 64 (`test_mimir_dm_cache.cpp:282`) | 37 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1InvalidateFull` | pass | 38 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/Uncached1B` | pass | 3 min 3 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/Uncached8B` | pass | 2 min 38 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/CachedRangeFlush` | pass | 2 min 34 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/CachedFastFlush` | pass | 2 min 40 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_SrcOnly` | pass | 30 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_DstOnly` | pass | 31 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_Both` | pass | 30 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_SrcOnly` | pass | 30 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_DstOnly` | pass | 31 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_Both` | pass | 30 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_SrcOnly` | pass | 30 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_DstOnly` | pass | 31 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_Both` | pass | 31 s |
| 78 | `DmSuitePort/MimirDmAddrgen.Pattern/Im2Col_FullImage` | pass | 32 s |
| 78 | `DmSuitePort/MimirDmAddrgen.Pattern/Im2Col_Dilation1` | pass | 35 s |
| 79 | `DmSuitePort/MimirDmDramUnary.ReaderWriterHandshake/Cce0` | pass | 46 s |
| 79 | `DmSuitePort/MimirDmDramUnary.ReaderWriterHandshake/Cce1` | pass | 45 s |
| 79 | `DmSuitePort/MimirDmDramUnary.ReaderWriterHandshake/Cce2` | pass | 45 s |
| 79 | `DmSuitePort/MimirDmDramUnary.ReaderWriterHandshake/Cce3` | pass | 43 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/InvalidateOnAcquire` | pass | 32 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/FlushOnRelease` | pass | 32 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/LockedRangeOnly` | pass | 32 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_cache_ops` | pass | 27 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_idma_copy` | pass | 28 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_amo_ops` | pass | 30 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_scoped_lock_producer` | pass | 26 s |
| 81 | `CceKernelCompile/MimirCceTestKernelCompile.Kernel/cce_scoped_lock_consumer` | pass | 25 s |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/Cce0` | pass | 1 min 51 s |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/Cce1` | pass | 1 min 53 s |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/Cce2` | pass | 1 min 54 s |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/Cce3` | pass | 1 min 53 s |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/Cce0` | pass | 36 s |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/Cce1` | pass | 36 s |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/Cce2` | pass | 37 s |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/Cce3` | pass | 36 s |
| 84 | `CceAtomics/MimirAtomicProbe.TwoHartAmo/Cce0` | pass | 41 s |
| 84 | `CceAtomics/MimirAtomicProbe.TwoHartAmo/Cce1` | pass | 41 s |
| 84 | `CceAtomics/MimirAtomicProbe.TwoHartAmo/Cce2` | pass | 41 s |
| 84 | `CceAtomics/MimirAtomicProbe.TwoHartAmo/Cce3` | pass | 41 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/Cce0` | pass | 36 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/Cce1` | pass | 36 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/Cce2` | pass | 37 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/Cce3` | pass | 36 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushFull/Cce0` | pass | 35 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushFull/Cce1` | pass | 36 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushFull/Cce2` | pass | 37 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushFull/Cce3` | pass | 37 s |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/Cce0` | pass | 55 s |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/Cce1` | pass | 55 s |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/Cce2` | pass | 54 s |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/Cce3` | pass | 57 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce0_WriterReader` | pass | 49 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce0_EightWriters` | pass | 1 min 55 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce1_WriterReader` | pass | 49 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce1_EightWriters` | pass | 1 min 54 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce2_WriterReader` | pass | 48 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce2_EightWriters` | pass | 1 min 50 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce3_WriterReader` | pass | 50 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce3_EightWriters` | pass | 1 min 55 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce0_WriterReader` | pass | 48 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce0_EightWriters` | pass | 1 min 51 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce1_WriterReader` | pass | 49 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce1_EightWriters` | pass | 1 min 57 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce2_WriterReader` | pass | 48 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce2_EightWriters` | pass | 1 min 53 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce3_WriterReader` | pass | 47 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce3_EightWriters` | pass | 1 min 56 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce0_WriterReader` | pass | 48 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce0_EightWriters` | pass | 1 min 55 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce1_WriterReader` | pass | 48 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce1_EightWriters` | pass | 1 min 52 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce2_WriterReader` | pass | 48 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce2_EightWriters` | pass | 1 min 57 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce3_WriterReader` | pass | 48 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce3_EightWriters` | pass | 1 min 53 s |
| 93 | `CceCache/MimirCacheScopedLock.MultiConsumer/Cce0` | pass | 48 s |
| 93 | `CceCache/MimirCacheScopedLock.MultiConsumer/Cce1` | pass | 48 s |
| 93 | `CceCache/MimirCacheScopedLock.MultiConsumer/Cce2` | pass | 49 s |
| 93 | `CceCache/MimirCacheScopedLock.MultiConsumer/Cce3` | pass | 49 s |
| 93 | `CceCache/MimirCacheScopedLock.RingWrap/Cce0` | pass | 31 s |
| 93 | `CceCache/MimirCacheScopedLock.RingWrap/Cce1` | pass | 32 s |
| 93 | `CceCache/MimirCacheScopedLock.RingWrap/Cce2` | pass | 32 s |
| 93 | `CceCache/MimirCacheScopedLock.RingWrap/Cce3` | pass | 32 s |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/Cce0` | pass | 2 min 49 s |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/Cce1` | pass | 2 min 48 s |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/Cce2` | pass | 2 min 50 s |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/Cce3` | pass | 2 min 46 s |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/Cce0` | pass | 1 min 8 s |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/Cce1` | pass | 1 min 9 s |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/Cce2` | pass | 1 min 8 s |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/Cce3` | pass | 1 min 11 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_Host_SnoopOffInvalidate` | pass | 38 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_Host_InvalidateOnly` | pass | 38 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_SnoopOffInvalidate` | pass | 51 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_InvalidateOnly` | pass | 50 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteCce_SnoopOffInvalidate` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteCce_InvalidateOnly` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_Host_SnoopOffInvalidate` | pass | 37 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_Host_InvalidateOnly` | pass | 37 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_SiblingSpa_SnoopOffInvalidate` | pass | 49 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_SiblingSpa_InvalidateOnly` | pass | 47 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_RemoteCce_SnoopOffInvalidate` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_RemoteCce_InvalidateOnly` | pass | 49 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_Host_SnoopOffInvalidate` | pass | 37 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_Host_InvalidateOnly` | pass | 37 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_SiblingSpa_SnoopOffInvalidate` | pass | 49 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_SiblingSpa_InvalidateOnly` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_RemoteCce_SnoopOffInvalidate` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_RemoteCce_InvalidateOnly` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_Host_SnoopOffInvalidate` | pass | 38 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_Host_InvalidateOnly` | pass | 37 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_SiblingSpa_SnoopOffInvalidate` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_SiblingSpa_InvalidateOnly` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_RemoteCce_SnoopOffInvalidate` | pass | 48 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_RemoteCce_InvalidateOnly` | pass | 49 s |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_SnoopOn` | pass | 48 s |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_SiblingSpa_SnoopOn` | pass | 48 s |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_SiblingSpa_SnoopOn` | pass | 49 s |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_SiblingSpa_SnoopOn` | pass | 48 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalIdma_SnoopOffInvalidate` | pass | 1 min 3 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalIdma_InvalidateOnly` | pass | 1 min 6 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalIdma_SnoopOffInvalidate` | pass | 1 min 3 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalIdma_InvalidateOnly` | pass | 1 min 3 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_LocalIdma_SnoopOffInvalidate` | pass | 1 min 3 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_LocalIdma_InvalidateOnly` | pass | 1 min 3 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_LocalIdma_SnoopOffInvalidate` | pass | 1 min 3 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_LocalIdma_InvalidateOnly` | pass | 1 min 6 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalGddrIdma_SnoopOffInvalidate` | pass | 1 min 1 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalGddrIdma_InvalidateOnly` | pass | 1 min 0 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalGddrIdma_SnoopOffInvalidate` | pass | 1 min 1 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_LocalGddrIdma_InvalidateOnly` | pass | 1 min 1 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_LocalGddrIdma_SnoopOffInvalidate` | pass | 1 min 1 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_LocalGddrIdma_InvalidateOnly` | pass | 1 min 1 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_LocalGddrIdma_SnoopOffInvalidate` | pass | 59 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_LocalGddrIdma_InvalidateOnly` | pass | 1 min 1 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteGddrIdma_SnoopOffInvalidate` | pass | 1 min 2 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteGddrIdma_InvalidateOnly` | pass | 1 min 1 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_RemoteGddrIdma_SnoopOffInvalidate` | pass | 1 min 1 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce1_RemoteGddrIdma_InvalidateOnly` | pass | 1 min 0 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_RemoteGddrIdma_SnoopOffInvalidate` | pass | 1 min 0 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce2_RemoteGddrIdma_InvalidateOnly` | pass | 1 min 1 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_RemoteGddrIdma_SnoopOffInvalidate` | pass | 1 min 1 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce3_RemoteGddrIdma_InvalidateOnly` | pass | 1 min 0 s |
| 102 | `CceCache/MimirCacheScopedLock.NonSnoopingProducer/Cce0` | pass | 32 s |
| 102 | `CceCache/MimirCacheScopedLock.NonSnoopingProducer/Cce1` | pass | 32 s |
| 102 | `CceCache/MimirCacheScopedLock.NonSnoopingProducer/Cce2` | pass | 32 s |
| 102 | `CceCache/MimirCacheScopedLock.NonSnoopingProducer/Cce3` | pass | 32 s |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/Cce0` | pass | 1 min 46 s |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/Cce1` | pass | 1 min 22 s |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/Cce2` | pass | 1 min 21 s |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/Cce3` | pass | 1 min 23 s |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/Cce0` | pass | 1 min 6 s |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/Cce1` | pass | 56 s |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/Cce2` | pass | 1 min 3 s |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/Cce3` | pass | 1 min 0 s |
| 105 | `CceCache/MimirCacheCoherence.CrossD2d/Cce0` | pass | 1 min 25 s |
| 105 | `CceCache/MimirCacheCoherence.CrossD2d/Cce1` | pass | 1 min 21 s |
| 105 | `CceCache/MimirCacheCoherence.CrossD2d/Cce2` | pass | 1 min 21 s |
| 105 | `CceCache/MimirCacheCoherence.CrossD2d/Cce3` | pass | 1 min 19 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg6` | pass | 2 min 40 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg0` | pass | 2 min 42 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg1` | pass | 2 min 41 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg2` | pass | 2 min 45 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg3` | fail: As on `mimir_1x1`, a config with bit 0 set: on the `Neighbour_Evict` leg the writer iDMA was acked but its 8 words never became visible in TL1 before the hart budget ran out (MIM-601), so the emulator restarted (`test_mimir_dm_cache.cpp:1988`) | 2 min 5 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg4` | pass | 2 min 42 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg5` | pass | 2 min 45 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg7` | pass | 2 min 45 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg6` | pass | 2 min 43 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg0` | pass | 2 min 43 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg1` | pass | 2 min 46 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg2` | pass | 2 min 46 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg3` | pass | 3 min 0 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg4` | pass | 3 min 5 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg5` | fail: As on `mimir_1x1`, a config with bit 0 set: on the `Neighbour_Evict` leg the writer iDMA was acked but its 8 words never became visible in TL1 before the hart budget ran out (MIM-601), so the emulator restarted (`test_mimir_dm_cache.cpp:1988`) | 2 min 13 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce1_Cfg7` | fail: As on `mimir_1x1`, a config with bit 0 set: on the `Neighbour_Flush64` leg the writer iDMA was acked but its 8 words never became visible in TL1 before the hart budget ran out (MIM-601), so the emulator restarted (`test_mimir_dm_cache.cpp:1988`) | 2 min 24 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg6` | pass | 2 min 54 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg0` | pass | 2 min 54 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg1` | pass | 2 min 55 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg2` | pass | 32 min 15 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg3` | fail: As on `mimir_1x1`, a config with bit 0 set: on the `Neighbour_Flush64` leg the writer iDMA was acked but its 8 words never became visible in TL1 before the hart budget ran out (MIM-601), so the emulator restarted (`test_mimir_dm_cache.cpp:1988`) | 5 min 42 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg4` | pass | 2 min 45 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg5` | pass | 2 min 44 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce2_Cfg7` | pass | 2 min 45 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg6` | pass | 2 min 47 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg0` | pass | 2 min 46 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg1` | pass | 2 min 47 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg2` | pass | 2 min 45 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg3` | fail: As on `mimir_1x1`, a config with bit 0 set: on the `Neighbour_Evict` leg the writer iDMA was acked but its 8 words never became visible in TL1 before the hart budget ran out (MIM-601), so the emulator restarted (`test_mimir_dm_cache.cpp:1988`) | 2 min 6 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg4` | pass | 3 min 7 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg5` | fail: As on `mimir_1x1`, a config with bit 0 set: on the `Neighbour_Flush64` leg the writer iDMA was acked but its 8 words never became visible in TL1 before the hart budget ran out (MIM-601), so the emulator restarted (`test_mimir_dm_cache.cpp:1988`) | 2 min 43 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce3_Cfg7` | fail: As on `mimir_1x1`, a config with bit 0 set: on the `Neighbour_Evict` leg the writer iDMA was acked but its 8 words never became visible in TL1 before the hart budget ran out (MIM-601), so the emulator restarted (`test_mimir_dm_cache.cpp:1988`) | 2 min 15 s |
| 107 | `CceCache/MimirCacheStress.ReuseThrash/Cce0` | pass | 19 min 14 s |
| 107 | `CceCache/MimirCacheStress.ReuseThrash/Cce1` | pass | 18 min 35 s |
| 107 | `CceCache/MimirCacheStress.ReuseThrash/Cce2` | pass | 18 min 28 s |
| 107 | `CceCache/MimirCacheStress.ReuseThrash/Cce3` | pass | 18 min 22 s |
| 108 | `CceCache/MimirCacheStress.PerfSweep/Cce0` | pass | 11 min 6 s |
| 108 | `CceCache/MimirCacheStress.PerfSweep/Cce1` | pass | 11 min 24 s |
| 108 | `CceCache/MimirCacheStress.PerfSweep/Cce2` | pass | 11 min 19 s |
| 108 | `CceCache/MimirCacheStress.PerfSweep/Cce3` | pass | 11 min 4 s |
| 109 | `CceCache/MimirCacheStress.PreloadChurn/Cce0` | fail: As on `mimir_1x1`: snoop leg (CCE1 iDMA from GDDR view 0 into hot set TL1 `0x8b4c0`), 1008 stale words consumed at the first read, expected 0. The iDMA snoop tag applies only to a sibling CCE's iDMA writes into this CCE's TL1 (`test_mimir_dm_cache.cpp:2211`) | 3 min 58 s |
| 109 | `CceCache/MimirCacheStress.PreloadChurn/Cce1` | fail: As on `mimir_1x1`: snoop leg (CCE0 iDMA from GDDR view 1 into hot set TL1 `0x8b4c0`), 1024 stale words consumed at the first read, expected 0. The iDMA snoop tag applies only to a sibling CCE's iDMA writes into this CCE's TL1 (`test_mimir_dm_cache.cpp:2211`) | 3 min 53 s |
| 109 | `CceCache/MimirCacheStress.PreloadChurn/Cce2` | fail: As on `mimir_1x1`: snoop leg (CCE3 iDMA from GDDR view 2 into hot set TL1 `0x8b4c0`), 1008 stale words consumed at the first read, expected 0. The iDMA snoop tag applies only to a sibling CCE's iDMA writes into this CCE's TL1 (`test_mimir_dm_cache.cpp:2211`) | 3 min 49 s |
| 109 | `CceCache/MimirCacheStress.PreloadChurn/Cce3` | fail: As on `mimir_1x1`: snoop leg (CCE2 iDMA from GDDR view 3 into hot set TL1 `0x8b4c0`), 1024 stale words consumed at the first read, expected 0. The iDMA snoop tag applies only to a sibling CCE's iDMA writes into this CCE's TL1 (`test_mimir_dm_cache.cpp:2211`) | 3 min 52 s |
| 110 | `CceAtomics/MimirAtomicLocal.OpMatrix/Cce0` | pass | 39 s |
| 110 | `CceAtomics/MimirAtomicLocal.OpMatrix/Cce1` | pass | 39 s |
| 110 | `CceAtomics/MimirAtomicLocal.OpMatrix/Cce2` | pass | 39 s |
| 110 | `CceAtomics/MimirAtomicLocal.OpMatrix/Cce3` | pass | 40 s |
| 110 | `CceAtomics/MimirAtomicLocal.Hotspot/Cce0` | pass | 2 min 56 s |
| 110 | `CceAtomics/MimirAtomicLocal.Hotspot/Cce1` | pass | 2 min 58 s |
| 110 | `CceAtomics/MimirAtomicLocal.Hotspot/Cce2` | pass | 3 min 0 s |
| 110 | `CceAtomics/MimirAtomicLocal.Hotspot/Cce3` | pass | 2 min 58 s |
| 110 | `CceAtomics/MimirAtomicLocal.Distributed/Cce0` | pass | 3 min 44 s |
| 110 | `CceAtomics/MimirAtomicLocal.Distributed/Cce1` | pass | 3 min 45 s |
| 110 | `CceAtomics/MimirAtomicLocal.Distributed/Cce2` | pass | 3 min 43 s |
| 110 | `CceAtomics/MimirAtomicLocal.Distributed/Cce3` | pass | 3 min 48 s |
| 110 | `CceAtomics/MimirAtomicLocal.LrScCas/Cce0` | pass | 6 min 2 s |
| 110 | `CceAtomics/MimirAtomicLocal.LrScCas/Cce1` | pass | 6 min 8 s |
| 110 | `CceAtomics/MimirAtomicLocal.LrScCas/Cce2` | pass | 6 min 7 s |
| 110 | `CceAtomics/MimirAtomicLocal.LrScCas/Cce3` | pass | 6 min 7 s |
| 110 | `CceAtomics/MimirAtomicLocal.Wrap/Cce0` | pass | 5 min 29 s |
| 110 | `CceAtomics/MimirAtomicLocal.Wrap/Cce1` | pass | 5 min 30 s |
| 110 | `CceAtomics/MimirAtomicLocal.Wrap/Cce2` | pass | 5 min 26 s |
| 110 | `CceAtomics/MimirAtomicLocal.Wrap/Cce3` | pass | 5 min 33 s |
| 110 | `CceAtomics/MimirAtomicLocal.Ordering/Cce0` | pass | 59 s |
| 110 | `CceAtomics/MimirAtomicLocal.Ordering/Cce1` | pass | 60 s |
| 110 | `CceAtomics/MimirAtomicLocal.Ordering/Cce2` | pass | 59 s |
| 110 | `CceAtomics/MimirAtomicLocal.Ordering/Cce3` | pass | 60 s |
| 110 | `CceAtomics/MimirAtomicLocal.CleanDirtyLine/Cce0` | pass | 1 min 22 s |
| 110 | `CceAtomics/MimirAtomicLocal.CleanDirtyLine/Cce1` | pass | 1 min 23 s |
| 110 | `CceAtomics/MimirAtomicLocal.CleanDirtyLine/Cce2` | pass | 1 min 23 s |
| 110 | `CceAtomics/MimirAtomicLocal.CleanDirtyLine/Cce3` | pass | 1 min 23 s |
| 110 | `CceAtomics/MimirAtomicLocal.Storm/Cce0` | pass | 2 min 55 s |
| 110 | `CceAtomics/MimirAtomicLocal.Storm/Cce1` | pass | 2 min 58 s |
| 110 | `CceAtomics/MimirAtomicLocal.Storm/Cce2` | pass | 3 min 2 s |
| 110 | `CceAtomics/MimirAtomicLocal.Storm/Cce3` | pass | 2 min 56 s |
| 110 | `CceAtomics/MimirAtomicLocal.NeighbourWriteRace/Cce0` | pass | 1 min 30 s |
| 110 | `CceAtomics/MimirAtomicLocal.NeighbourWriteRace/Cce1` | pass | 1 min 33 s |
| 110 | `CceAtomics/MimirAtomicLocal.NeighbourWriteRace/Cce2` | pass | 1 min 31 s |
| 110 | `CceAtomics/MimirAtomicLocal.NeighbourWriteRace/Cce3` | pass | 1 min 31 s |
| 110 | `CceAtomics/MimirAtomicLocal.Soak/Cce0` | pass | 1 min 27 s |
| 110 | `CceAtomics/MimirAtomicLocal.Soak/Cce1` | pass | 56 s |
| 110 | `CceAtomics/MimirAtomicLocal.Soak/Cce2` | pass | 1 min 29 s |
| 110 | `CceAtomics/MimirAtomicLocal.Soak/Cce3` | pass | 60 s |
| 111 | `CceAtomics/MimirAtomicItnControl.SiblingWriteThenAmo/Cce0` | pass | 41 s |
| 111 | `CceAtomics/MimirAtomicItnControl.SiblingWriteThenAmo/Cce1` | pass | 42 s |
| 111 | `CceAtomics/MimirAtomicItnControl.SiblingWriteThenAmo/Cce2` | pass | 41 s |
| 111 | `CceAtomics/MimirAtomicItnControl.SiblingWriteThenAmo/Cce3` | pass | 41 s |
| 112 | `CceAtomics/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/Cce0` | pass | 50 s |
| 112 | `CceAtomics/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/Cce1` | pass | 49 s |
| 112 | `CceAtomics/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/Cce2` | pass | 50 s |
| 112 | `CceAtomics/MimirAtomicLocal.CleanDirtyLineSnoopedWrite/Cce3` | pass | 51 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/OwnTl1` | pass | 38 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/MaxPacket` | pass | 58 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/SiblingTl1` | pass | 50 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr` | pass | 49 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr1` | pass | 48 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1` | pass | 50 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteGddr` | pass | 49 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1D` | pass | 36 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DLocalGddr` | pass | 36 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLine` | pass | 35 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLineLocalGddr` | pass | 34 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1KernelVerified` | pass | 52 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsOwnTl1/Cce0` | pass | 57 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsOwnTl1/Cce1` | pass | 57 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsOwnTl1/Cce2` | pass | 56 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsOwnTl1/Cce3` | pass | 57 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsSiblingTl1/Cce0` | pass | 1 min 32 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsSiblingTl1/Cce1` | pass | 1 min 32 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsSiblingTl1/Cce2` | pass | 1 min 32 s |
| 114 | `IdmaTransactionId/MimirTridIdma.UniqueIdsSiblingTl1/Cce3` | pass | 1 min 32 s |
| 114 | `IdmaTransactionId/MimirTridIdma.ReuseWrap/Cce0` | pass | 2 min 25 s |
| 114 | `IdmaTransactionId/MimirTridIdma.ReuseWrap/Cce1` | pass | 2 min 26 s |
| 114 | `IdmaTransactionId/MimirTridIdma.ReuseWrap/Cce2` | pass | 2 min 27 s |
| 114 | `IdmaTransactionId/MimirTridIdma.ReuseWrap/Cce3` | pass | 2 min 27 s |
| 114 | `IdmaTransactionId/MimirTridIdma.DuplicateId/Cce0` | fail: As `Cce0` on `mimir_1x1`: pending high water for trid 9 stayed at 1, expected more than 1; all 16 commands landed (`test_mimir_trid.cpp:907`). `Cce3` passed | 1 min 40 s |
| 114 | `IdmaTransactionId/MimirTridIdma.DuplicateId/Cce1` | fail: As `Cce0` on `mimir_1x1`: pending high water for trid 9 stayed at 1, expected more than 1; all 16 commands landed (`test_mimir_trid.cpp:907`). `Cce3` passed | 1 min 41 s |
| 114 | `IdmaTransactionId/MimirTridIdma.DuplicateId/Cce2` | fail: As `Cce0` on `mimir_1x1`: pending high water for trid 9 stayed at 1, expected more than 1; all 16 commands landed (`test_mimir_trid.cpp:907`). `Cce3` passed | 1 min 39 s |
| 114 | `IdmaTransactionId/MimirTridIdma.DuplicateId/Cce3` | pass | 1 min 39 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentReadAfterWrite/Cce0` | pass | 1 min 20 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentReadAfterWrite/Cce1` | pass | 1 min 21 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentReadAfterWrite/Cce2` | pass | 1 min 24 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentReadAfterWrite/Cce3` | pass | 1 min 21 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentWriteAfterRead/Cce0` | pass | 1 min 26 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentWriteAfterRead/Cce1` | pass | 1 min 27 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentWriteAfterRead/Cce2` | pass | 1 min 27 s |
| 114 | `IdmaTransactionId/MimirTridIdma.BarrierThenDependentWriteAfterRead/Cce3` | pass | 1 min 31 s |
| 115 | `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr/Cce0` | pass | 2 min 43 s |
| 115 | `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr/Cce1` | pass | 2 min 44 s |
| 115 | `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr/Cce2` | pass | 2 min 42 s |
| 115 | `IdmaTransactionId/MimirTridIdma.UniqueIdsLocalGddr/Cce3` | pass | 2 min 48 s |
| 116 | `IdmaTransactionId/MimirTridIdma.NominalDepth/Cce0` | pass | 1 min 31 s |
| 116 | `IdmaTransactionId/MimirTridIdma.NominalDepth/Cce1` | pass | 1 min 31 s |
| 116 | `IdmaTransactionId/MimirTridIdma.NominalDepth/Cce2` | pass | 1 min 31 s |
| 116 | `IdmaTransactionId/MimirTridIdma.NominalDepth/Cce3` | pass | 1 min 30 s |
| 117 | `IdmaTransactionId/MimirTridIdma.MixedTargets/Cce0` | pass | 51 s |
| 117 | `IdmaTransactionId/MimirTridIdma.MixedTargets/Cce1` | pass | 52 s |
| 117 | `IdmaTransactionId/MimirTridIdma.MixedTargets/Cce2` | pass | 51 s |
| 117 | `IdmaTransactionId/MimirTridIdma.MixedTargets/Cce3` | pass | 51 s |
| 118 | `IdmaTransactionId/MimirTridIdma.Saturate/Cce0` | pass | 12 min 53 s |
| 118 | `IdmaTransactionId/MimirTridIdma.Saturate/Cce1` | pass | 12 min 58 s |
| 118 | `IdmaTransactionId/MimirTridIdma.Saturate/Cce2` | pass | 13 min 0 s |
| 118 | `IdmaTransactionId/MimirTridIdma.Saturate/Cce3` | pass | 12 min 54 s |
| 119 | `IdmaTransactionId/MimirTridIdma.MultiHart/Cce0` | pass | 4 min 37 s |
| 119 | `IdmaTransactionId/MimirTridIdma.MultiHart/Cce1` | pass | 4 min 50 s |
| 119 | `IdmaTransactionId/MimirTridIdma.MultiHart/Cce2` | pass | 4 min 39 s |
| 119 | `IdmaTransactionId/MimirTridIdma.MultiHart/Cce3` | pass | 4 min 45 s |
| 120 | `IdmaTransactionId/MimirTridIdma.Fairness/Cce0` | pass | 4 min 5 s |
| 120 | `IdmaTransactionId/MimirTridIdma.Fairness/Cce1` | pass | 4 min 6 s |
| 120 | `IdmaTransactionId/MimirTridIdma.Fairness/Cce2` | pass | 4 min 1 s |
| 120 | `IdmaTransactionId/MimirTridIdma.Fairness/Cce3` | pass | 4 min 2 s |
| 121 | `IdmaTransactionId/MimirTridIdma.Soak/Cce0` | pass | 5 min 26 s |
| 121 | `IdmaTransactionId/MimirTridIdma.Soak/Cce1` | pass | 5 min 24 s |
| 121 | `IdmaTransactionId/MimirTridIdma.Soak/Cce2` | pass | 5 min 26 s |
| 121 | `IdmaTransactionId/MimirTridIdma.Soak/Cce3` | pass | 5 min 29 s |
| 122 | `IdmaTransactionId/MimirTridRemote.RemoteTl1/Cce0ToCce2` | pass | 1 min 29 s |
| 122 | `IdmaTransactionId/MimirTridRemote.RemoteTl1/Cce1ToCce3` | pass | 1 min 30 s |
| 122 | `IdmaTransactionId/MimirTridRemote.RemoteTl1/Cce2ToCce0` | pass | 1 min 28 s |
| 122 | `IdmaTransactionId/MimirTridRemote.RemoteTl1/Cce3ToCce1` | pass | 1 min 30 s |
| 123 | `IdmaTransactionId/MimirTridRemote.RemoteGddr/Cce0ToCce2` | pass | 1 min 30 s |
| 123 | `IdmaTransactionId/MimirTridRemote.RemoteGddr/Cce1ToCce3` | pass | 1 min 30 s |
| 123 | `IdmaTransactionId/MimirTridRemote.RemoteGddr/Cce2ToCce0` | pass | 1 min 30 s |
| 123 | `IdmaTransactionId/MimirTridRemote.RemoteGddr/Cce3ToCce1` | pass | 1 min 31 s |
| 124 | `IdmaTransactionId/MimirTridConcurrent.WithPrefetcher/Sender0Receiver2Stream1` | pass | 2 min 47 s |
| 125 | `IdmaTransactionId/MimirTridReset.NoStaleIdAfterReinit/Cce0` | pass | 4 min 28 s |
| 125 | `IdmaTransactionId/MimirTridReset.NoStaleIdAfterReinit/Cce1` | pass | 4 min 39 s |
| 125 | `IdmaTransactionId/MimirTridReset.NoStaleIdAfterReinit/Cce2` | pass | 4 min 27 s |
| 125 | `IdmaTransactionId/MimirTridReset.NoStaleIdAfterReinit/Cce3` | pass | 4 min 32 s |
| 126 | `CceAtomics/MimirAccumProbe.FmtOpMatrix/Cce0` | pass | 8 min 33 s |
| 126 | `CceAtomics/MimirAccumProbe.FmtOpMatrix/Cce1` | pass | 8 min 31 s |
| 126 | `CceAtomics/MimirAccumProbe.FmtOpMatrix/Cce2` | pass | 8 min 37 s |
| 126 | `CceAtomics/MimirAccumProbe.FmtOpMatrix/Cce3` | pass | 8 min 30 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SingleHart/Cce0` | pass | 36 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SingleHart/Cce1` | pass | 35 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SingleHart/Cce2` | pass | 36 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SingleHart/Cce3` | pass | 36 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsShared/Cce0` | pass | 1 min 43 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsShared/Cce1` | pass | 1 min 42 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsShared/Cce2` | pass | 1 min 40 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsShared/Cce3` | pass | 1 min 43 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsDisjoint/Cce0` | pass | 1 min 46 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsDisjoint/Cce1` | pass | 1 min 47 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsDisjoint/Cce2` | pass | 1 min 47 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.AllHartsDisjoint/Cce3` | pass | 1 min 47 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.BackToBack/Cce0` | pass | 1 min 30 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.BackToBack/Cce1` | pass | 1 min 31 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.BackToBack/Cce2` | pass | 1 min 29 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.BackToBack/Cce3` | pass | 1 min 31 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SequentialAmoAccumulateAmo/Cce0` | pass | 54 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SequentialAmoAccumulateAmo/Cce1` | pass | 51 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SequentialAmoAccumulateAmo/Cce2` | pass | 50 s |
| 128 | `CceAtomics/MimirAtomicIdmaAccum.SequentialAmoAccumulateAmo/Cce3` | pass | 50 s |
| 129 | `CceAtomics/MimirAtomicIdmaAccum.MaxOutstanding/Cce0` | pass | 45 s |
| 129 | `CceAtomics/MimirAtomicIdmaAccum.MaxOutstanding/Cce1` | pass | 46 s |
| 129 | `CceAtomics/MimirAtomicIdmaAccum.MaxOutstanding/Cce2` | pass | 44 s |
| 129 | `CceAtomics/MimirAtomicIdmaAccum.MaxOutstanding/Cce3` | pass | 44 s |
| 2c | `MimirFwLoadNegative.SkipLoadNeverDone` | pass | 1 min 51 s |

## Results on `mimir_2x_package`, 2026-11-13

Ended early: the run started and ended on 2026-11-13. It was ended on the owner's request after `CceCache/MimirCacheCoherenceRace`, at 21:25 UTC, and the emulator was stopped; it did not finish its plan. It is a rerun of the tests that launch the kernels changed by the CCE kernel cleanups, not a full run. One line per test case. Run against `de3ae9179c8` plus the uncommitted CCE kernel cleanups, in the same Debug build (`build_Debug`). Emulator `models/mmk`, each session after `bringup_two_mimir.py`: soc-c-31, soc-c-24, soc-c-31, soc-zebu-03, soc-c-24, soc-c-23, rv-zebu-07, soc-c-28, rv-zebu-07, rv-zebu-08, soc-zebu-01, soc-c-24, soc-zebu-04. One case per invocation, with `TT_METAL_OPERATION_TIMEOUT_SECONDS=600` and `timeout 7200`. Results, logs and scripts are kept in `/localdev/sagarwal/mimir_reruns_2026-11-13/mimir_2x_package/`.

Scope differences from 2026-11-04:
- Only CCE 0 (and suites with no per-CCE variants) ran. The CCE 2 chain was queued and did not run; CCE 1 and CCE 3 were never queued.
- `LocalDram/MimirLocalDram` (including `SizeAlignmentMatrix`), `RemoteDram/MimirRemoteDram` and `RemoteDram/MimirRemoteDramTrafficMatrix` were not queued.
- Rows not listed in the table were not rerun.
- Queued and not run: first chain, 101 cases: `CceCache/MimirCacheStress` (3), `CceKernelCompile/MimirCceTestKernelCompile` (5), `CceCache/MimirCacheScopedLock` (3), `CceAtomics/MimirAtomicProbe` (1), `CceAtomics/MimirAtomicLocal` (11), `CceAtomics/MimirAtomicItnControl` (1), `CceAtomics/MimirAccumProbe` (1), `CceAtomics/MimirAtomicIdmaAccum` (6), `IdmaTransactionId/MimirTridIdma` (13), `IdmaTransactionId/MimirTridReset` (1), `IdmaTransactionId/MimirTridConcurrent` (1), `MimirPrefetcher` (40), `TensorPrefetcher/MimirPrefetcherSingleGcb` (12), `TensorPrefetcher/MimirPrefetcherGcbSwitch` (3); second chain (CCE 2), 96 cases: `KernelLoad/MimirKernelLoad` (4), `DmSuitePort/MimirDmDramUnary` (1), `FirmwareLoad/MimirFwLaunch` (1), `MimirEmu/AllHartsRunDramKernel` (1), `LocalTl1/MimirLocalTl1` (2), `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs` (2), `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs` (2), `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment` (2), `CceCache/MimirCacheProbe` (2), `CceCache/MimirCacheOps` (5), `CceCache/MimirCacheSharedL2` (6), `CceCache/MimirCacheWriterProbe` (13), `CceCache/MimirCacheCoherence` (3), `CceCache/MimirCacheCoherenceRace` (8), `CceCache/MimirCacheStress` (3), `CceCache/MimirCacheScopedLock` (3), `CceAtomics/MimirAtomicProbe` (1), `CceAtomics/MimirAtomicLocal` (11), `CceAtomics/MimirAtomicItnControl` (1), `CceAtomics/MimirAccumProbe` (1), `CceAtomics/MimirAtomicIdmaAccum` (6), `IdmaTransactionId/MimirTridIdma` (13), `IdmaTransactionId/MimirTridRemote` (4), `IdmaTransactionId/MimirTridReset` (1).
- The emulator restarted with bring-up at every other suite switch until 19:58 UTC and at every third suite switch from then on, always before the prefetcher rows and before row 81, and after any timeout or lost emulator.

Final: 122 cases ran, 120 pass, 2 fail. Results that differ from 2026-11-04: `DmSuitePort/MimirDmCache.Op/L2InvalidateFreshRead` (pass, 2026-11-04: fail); `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg1` (fail, 2026-11-04: pass); `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg3` (pass, 2026-11-04: fail).

| Row | Test | Result | Time |
|---|---|---|---|
| 2b | `FirmwareLoad/MimirFwLaunch.NoOpLaunchCompletes/Cce0` | pass | 29 s |
| 2b | `MimirFwLoad.AllCcesIndependent` | pass | 11 min 51 s |
| 2d | `MimirFwReinit.ReloadsOnEveryOpen` | pass | 1 min 37 s |
| 3 | `KernelLoad/MimirKernelLoad.ArgsReplaceStale/Cce0` | pass | 40 s |
| 3 | `KernelLoad/MimirKernelLoad.EachHartAlone/Cce0` | pass | 1 min 7 s |
| 3 | `KernelLoad/MimirKernelLoad.FirmwareKeepsRunningAcrossReopen/Cce0` | pass | 1 min 0 s |
| 3 | `KernelLoad/MimirKernelLoad.MinimalKernelExecutes/Cce0` | pass | 29 s |
| 3 | `MimirKernelLoad.SameBinaryAllCcesUnicast` | pass | 53 s |
| 4 | `D2dDirect/MimirRemoteTl1.HostInitRemoteRead/M0Cce1ToM1Cce0` | pass | 1 min 12 s |
| 4 | `D2dDirect/MimirRemoteTl1.HostInitRemoteRead/M1Cce0ToM0Cce1` | pass | 1 min 11 s |
| 4 | `D2dDirect/MimirRemoteTl1.SizeAlignment/M0Cce1ToM1Cce0` | pass | 2 min 34 s |
| 4 | `D2dDirect/MimirRemoteTl1.SizeAlignment/M1Cce0ToM0Cce1` | pass | 2 min 35 s |
| 4 | `D2dDirect/MimirRemoteTl1.WriteRead/M0Cce1ToM1Cce0` | pass | 1 min 41 s |
| 4 | `D2dDirect/MimirRemoteTl1.WriteRead/M1Cce0ToM0Cce1` | pass | 1 min 42 s |
| 5 | `MimirRemoteTl1Bidir.Concurrent` | pass | 2 min 53 s |
| 6 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M0Cce0ToM1Cce0` | pass | 1 min 11 s |
| 6 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M0Cce0ToM1Cce0` | pass | 2 min 30 s |
| 6 | `D2dRouted/MimirRemoteTl1.WriteRead/M0Cce0ToM1Cce0` | pass | 1 min 41 s |
| 7 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M0Cce1ToM1Cce1` | pass | 1 min 13 s |
| 7 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M0Cce1ToM1Cce1` | pass | 2 min 31 s |
| 7 | `D2dRouted/MimirRemoteTl1.WriteRead/M0Cce1ToM1Cce1` | pass | 1 min 41 s |
| 8 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M1Cce0ToM0Cce0` | pass | 1 min 12 s |
| 8 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M1Cce0ToM0Cce0` | pass | 2 min 32 s |
| 8 | `D2dRouted/MimirRemoteTl1.WriteRead/M1Cce0ToM0Cce0` | pass | 1 min 41 s |
| 9 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M1Cce1ToM0Cce1` | pass | 1 min 10 s |
| 9 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M1Cce1ToM0Cce1` | pass | 2 min 37 s |
| 9 | `D2dRouted/MimirRemoteTl1.WriteRead/M1Cce1ToM0Cce1` | pass | 1 min 40 s |
| 10 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M0Cce0ToM1Cce1` | pass | 1 min 10 s |
| 10 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M0Cce0ToM1Cce1` | pass | 2 min 35 s |
| 10 | `D2dRouted/MimirRemoteTl1.WriteRead/M0Cce0ToM1Cce1` | pass | 1 min 42 s |
| 11 | `D2dRouted/MimirRemoteTl1.HostInitRemoteRead/M1Cce1ToM0Cce0` | pass | 1 min 11 s |
| 11 | `D2dRouted/MimirRemoteTl1.SizeAlignment/M1Cce1ToM0Cce0` | pass | 2 min 29 s |
| 11 | `D2dRouted/MimirRemoteTl1.WriteRead/M1Cce1ToM0Cce0` | pass | 1 min 40 s |
| 16 | `MimirRemoteMixed.Concurrent` | pass | 5 min 34 s |
| 16 | `MimirRemoteMixed.FanOutFanIn` | pass | 2 min 0 s |
| contract | `MimirEmu.AllCcesCopyLocalMimirBlock` | pass | 1 min 39 s |
| contract | `MimirEmu/AllHartsRunDramKernel.WritesMagic/Cce0` | pass | 31 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrLargerThanAlignment.RoundTrip/Cce0Partition0` | pass | 55 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithCompileTimeArgs.RoundTrip/Cce0Partition0` | pass | 27 s |
| contract | `MimirEmuSameMimirDirect/IdmaCopiesGddrWithRuntimeArgs.RoundTrip/Cce0Partition0` | pass | 50 s |
| 72 | `DmSuitePort/MimirDmRuntimeArgs.IllegalTooManyRuntimeArgs/Cce0` | pass | 30 s |
| 72 | `DmSuitePort/MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs/Cce0` | pass | 27 s |
| 75 | `DmSuitePort/MimirDmCache.Op/HartToHartCoherent` | fail, as on 2026-11-04: consumer warm read before the stores: `consumer[abi::kStatusWarmOld]` was 16, expected 64 (`kCacheWords`) (`test_mimir_dm_cache.cpp:238`) | 49 s |
| 75 | `DmSuitePort/MimirDmCache.Op/InvalidateFreshRead` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1FlushFull` | pass | 40 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1FlushLine` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1InvalidateFull` | pass | 37 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L1InvalidateLine` | pass | 38 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushFull` | pass | 46 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushLine` | pass | 37 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2FlushRange` | pass | 36 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2InvalidateFreshRead` | pass (2026-11-04: fail) | 37 s |
| 75 | `DmSuitePort/MimirDmCache.Op/L2InvalidateLine` | pass | 37 s |
| 75 | `DmSuitePort/MimirDmCache.Op/NoFlushNotVisible` | pass | 40 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/CachedFastFlush` | pass | 2 min 32 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/CachedRangeFlush` | pass | 2 min 32 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/Uncached1B` | pass | 2 min 50 s |
| 76 | `DmSuitePort/MimirDmCachePerf.WritePath/Uncached8B` | pass | 2 min 31 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_Both` | pass | 28 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_DstOnly` | pass | 29 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Face_SrcOnly` | pass | 28 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_Both` | pass | 27 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_DstOnly` | pass | 28 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided1D_SrcOnly` | pass | 29 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_Both` | pass | 28 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_DstOnly` | pass | 28 s |
| 77 | `DmSuitePort/MimirDmAddrgen.Pattern/Strided2D_SrcOnly` | pass | 28 s |
| 78 | `DmSuitePort/MimirDmAddrgen.Pattern/Im2Col_Dilation1` | pass | 33 s |
| 78 | `DmSuitePort/MimirDmAddrgen.Pattern/Im2Col_FullImage` | pass | 30 s |
| 79 | `DmSuitePort/MimirDmDramUnary.ReaderWriterHandshake/Cce0` | pass | 46 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/FlushOnRelease` | pass | 32 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/InvalidateOnAcquire` | pass | 33 s |
| 80 | `DmSuitePort/MimirDmScopedLockClm.CacheOps/LockedRangeOnly` | pass | 32 s |
| 82 | `CceCache/MimirCacheProbe.EightHartLaunch/Cce0` | pass | 2 min 0 s |
| 83 | `CceCache/MimirCacheProbe.SnoopConfig/Cce0` | pass | 39 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushFull/Cce0` | pass | 41 s |
| 90 | `CceCache/MimirCacheOps.L1DcacheFlushLine/Cce0` | pass | 39 s |
| 91 | `CceCache/MimirCacheOps.DirtyEvictionPartialLine/Cce0` | pass | 57 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce0_EightWriters` | pass | 1 min 56 s |
| 92 | `CceCache/MimirCacheSharedL2.AdjacentLine/Cce0_WriterReader` | pass | 52 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce0_EightWriters` | pass | 2 min 2 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineDiffOffset/Cce0_WriterReader` | pass | 52 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce0_EightWriters` | pass | 2 min 0 s |
| 92 | `CceCache/MimirCacheSharedL2.SameLineSameOffset/Cce0_WriterReader` | pass | 52 s |
| 95 | `CceCache/MimirCacheOps.FlushFenceOrdering/Cce0` | pass | 2 min 48 s |
| 96 | `CceCache/MimirCacheOps.FullInvalidate/Cce0` | pass | 1 min 9 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_Host_InvalidateOnly` | pass | 37 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_Host_SnoopOffInvalidate` | pass | 39 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteCce_InvalidateOnly` | pass | 49 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteCce_SnoopOffInvalidate` | pass | 49 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_InvalidateOnly` | pass | 49 s |
| 97 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_SnoopOffInvalidate` | pass | 49 s |
| 98 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_SiblingSpa_SnoopOn` | pass | 49 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalIdma_InvalidateOnly` | pass | 1 min 5 s |
| 99 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalIdma_SnoopOffInvalidate` | pass | 1 min 5 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalGddrIdma_InvalidateOnly` | pass | 1 min 2 s |
| 100 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_LocalGddrIdma_SnoopOffInvalidate` | pass | 1 min 1 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteGddrIdma_InvalidateOnly` | pass | 1 min 1 s |
| 101 | `CceCache/MimirCacheWriterProbe.ExternalWriters/Cce0_RemoteGddrIdma_SnoopOffInvalidate` | pass | 1 min 4 s |
| 103 | `CceCache/MimirCacheCoherence.CrossCce/Cce0` | pass | 1 min 11 s |
| 104 | `CceCache/MimirCacheCoherence.CrossCceSnoopOn/Cce0` | pass | 47 s |
| 105 | `CceCache/MimirCacheCoherence.CrossD2d/Cce0` | pass | 1 min 12 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg0` | pass | 2 min 39 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg1` | fail: Failed DirtyLineRace on CCE0 cfg1_Neighbour_Flush64: the hart's budget ran out before it read the external words, so the iDMA may still be in flight; the writer's command was acked, so its words were accepted but are not visible in TL1 (kernel code acked flags [vc_space true acked true landed fal...; the same symptom as the intermittent bit-0 config failures in the 2026-10-26 run (Cache, atomics and transaction-ID results, row 106); not rerun (2026-11-04: pass) | 2 min 27 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg2` | pass | 2 min 38 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg3` | pass (2026-11-04: fail) | 2 min 35 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg4` | pass | 2 min 35 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg5` | pass | 2 min 33 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg6` | pass | 2 min 41 s |
| 106 | `CceCache/MimirCacheCoherenceRace.DirtyLineRace/Cce0_Cfg7` | pass | 2 min 38 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr` | pass | 51 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/LocalGddr1` | pass | 52 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/MaxPacket` | pass | 1 min 0 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/OwnTl1` | pass | 40 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteGddr` | pass | 52 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1` | pass | 56 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/RemoteTl1KernelVerified` | pass | 53 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/SiblingTl1` | pass | 51 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1D` | pass | 38 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DLocalGddr` | pass | 37 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLine` | pass | 37 s |
| 113 | `DmSuitePort/MimirDmIdma.Copy/Strided1DSubLineLocalGddr` | pass | 35 s |
