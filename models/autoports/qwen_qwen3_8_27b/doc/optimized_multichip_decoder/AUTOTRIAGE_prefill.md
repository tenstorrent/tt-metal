# AUTOTRIAGE

## Diagnosis

- **The fused 2D matmul reuses a single `MatmulOpReceiver` across eight output-row blocks, although all-gather publishes readiness only once for each gathered K slice.** With `per_core_M=16`, `out_block_h=2`, and TP4, the first K pass requests direction thresholds `1,1,2,2`; the second pass requests `3`, which neither direction can produce. The input0 sender stops at `worker_sync_utils.hpp:253`, starving its multicast receivers and compute cores. This is a receiver-state lifetime/loop-count defect, exposed by output blocking, rather than a native-weight descriptor or fabric teardown failure.

Investigation was limited to preserved evidence, source inspection, and host-only arithmetic. No target code, device access, reset, process termination, or implementation edit was performed. Only this report was written. Checkout HEAD: `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`. The current prefill adapter and preserved `prefill_source_provenance/adapter_v3.py` have the same SHA256: `21999833afd71d1d6fe936c47d437129b2a2ec1a911f6da4157c0bbb4437001a`.

## Triage Evidence

Primary evidence: [preserved tt-triage capture](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/triage_prefill_ccl_fused_smoke_ag_h2_l0/tt-triage.txt:1), containing 1,344 RISC-V stack rows and one running-operation row. The following are directly observed:

- Running operation **1408, `AllGatherMatmulAsyncDeviceOperation`**, spans devices **0–3 and 256 matmul cores**. Input0 is `[1,1,4096,1280]`, BF16, interleaved DRAM; input1 is `[1,1,5120,4608]`, BFP4, interleaved DRAM. Both have 32×32 tiles and no shard specification. The previous operation is **1407, `LayerNormPostAllGatherDeviceOperation`**. See capture line 1351.
- On **all eight logical `(0,y)` input0 sender cores on every device**, NCRISC is inside `noc_semaphore_wait_min` through `MatmulOpReceiver::update_current_block_start_tile_id` at `worker_sync_utils.hpp:253`, called from `reader_bmm_tile_layout_in0_sender_padding.cpp:267`. Representative logical `(0,0)` rows are capture lines **8, 344, 680, and 1016**; device 0 physical `(3,2)` has PC `0xe2c0`, kernel ID 123.
- There is no isolated slow rank: the same sender-side stop occurs on all four devices. The observed fanout per device is:

| RISC-V / kernel | Cores per device | Captured location |
|---|---:|---|
| NCRISC, input0 sender | 8 | Fused AG readiness `wait_min`, helper line 253 |
| NCRISC, input0 receiver | 56 | Multicast `receiver_sem.wait(VALID)`, receiver line 88 |
| TRISC0, `bmm_large_block_zm_fused_bias_activation` | 64 | Input0 `wait_front`, compute line 339 |
| BRISC, input1 sender/writer | 8 | Input1 `reserve_back`, sender/writer line 461 |
| BRISC, input1 receiver/writer | 56 | Input1 `reserve_back`, receiver/writer line 124 |
| TRISC1 / TRISC2, same compute kernel | 64 each | Matmul math / partial packing, compute lines 382 / 450 |

- Dispatch, prefetch, telemetry, and fabric-router stacks also appear. **No `minimal_default_reader` or `minimal_default_writer` AG worker appears in the captured running stacks.** This is consistent with completed AG work; absence from this capture alone is not a device completion certificate.
- There is **no captured LLK assertion or `ebreak`**, and no decoded CB occupancy, semaphore value, loop-local variable, or NoC acknowledgement-counter dump. Fabric `transaction_flushed` and credit-polling frames are sampled router locations, not evidence by themselves of stuck transactions.

The [original log](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/prefill_ccl_fused_smoke_ag_h2_l0.log:1) reaches baseline `PREFILL_DONE` and `DECODE_DONE`, then initializes the four-device ring without reaching multichip `PREFILL_DONE`. The supplied input identifies this as the first eager TP4 prefill, before trace capture. The [sweep record](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/prefill_ccl_fused_smoke_results.json) records `ag_h2` exit `-15` after watchdog intervention. That termination is subsequent handling, not the original failure.

Passing contrast is preserved: the layer-0 separate-op control exits 0 with prefill PCC `0.9999584868043154`; the RS-only `h2` candidate also exits 0. The [decode controls](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/column_family32_results.json) include passing interleaved-weight 2D AG cases `both_ag2d_k10` and `both_ag2d_k20` at M32. These controls support an AG-specific repeated-output-block defect; they are not a measurement of the proposed repair.

## Source Evidence

**Geometry and loop ledger.** The [prefill adapter](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_prefill_ccl_candidate.py:115) computes `pm=ceil(4096/32/8)=16`, `pn=ceil(4608/32/8)=18`, defaults `block_h` to 2, and explicitly sets `out_block_w=pn`. Its weight conversion to interleaved DRAM is already present at lines 24–27. The [2D fusion factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:145) derives the following counts and passes them to both readers and compute:

| Quantity | Value / consequence |
|---|---|
| M, gathered K, N in tiles | 128, 160, 144 |
| Core grid with work | `128/16 × 144/18 = 8 × 8` |
| Batch count | 1; sparsity disabled; input0 CB reuse disabled |
| Per-core outer loops | `bh=0..7`, `bw=0` |
| K blocks per outer block | `160/4 = 40` |
| K tiles per rank slice | `1280/32 = 40` |
| K blocks per rank slice | `40/4 = 10` |
| Slice-boundary helper calls | Four per K pass, 32 per sender over eight passes |
| Intended data-block transfers per core | `8 × 1 × 40 = 320` |

The [AG/MM factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:54) creates the matmul signaler, then supplies its receiver cores and semaphore IDs to AG. The [tensor slicer](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp:600) supplies a 40-tile A slice offset. B uses `40 × 144 = 5760` tiles per K slice. The [runtime-argument serializer](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp:473) already supplies both the slice offset and wrap offset, start rank, direction, and semaphore IDs. Missing runtime arguments are not the explanation here.

**AG readiness producer/consumer ledger.** Let `S0` and `S1` denote the two program-local fused-operation readiness semaphores on each of the eight input0 sender cores. They are distinct from AG's global transport-ready semaphores and from matmul multicast handshakes.

The [matmul factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:734) registers the input0 sender column. [Signaler initialization](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp:342) creates two zero-initialized semaphores. The AG factory creates separate local-writer, direction-0 reader, and direction-1 reader signaler groups; each group has **two workers** (`num_links=2 × workers_per_direction=1`). Its runtime arguments select semaphore 1 for local writers and direction-1 readers, and semaphore 0 for direction-0 readers: [factory lines 1005–1014, 1285–1298, 1357–1363](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp:1005).

| Resource | Producer → consumer | Exact relevant count |
|---|---|---|
| Local-writer group barrier | One slave → group master; master → slave release | Once after the local slice is written; master waits for 1, resets, and releases slave |
| Direction-0 reader group barrier | One slave → master; master → slave release | Twice, once for each received rank slice |
| Direction-1 reader group barrier | One slave → master; master → slave release | Once, for its received remote slice |
| `self_write_done` | Direction-1 writer → colocated reader | `+1` once; reader waits for 1 and clears it before publishing its first remote slice |
| `S1` | Local-writer master and direction-1 reader master → all eight A senders | `+1` local, `+1` remote: final value **2** |
| `S0` | Direction-0 reader master → all eight A senders | `+1` for each of two remote slices: final value **2** |
| `S0/S1` reads | A sender's `MatmulOpReceiver` | Non-consuming `wait_min(floor(curr_transfer_idx/2)+1)`; no decrement/reset |

The ring targets are `ceil((4−1)/2)=2` forward and `1` backward, with static alternation disabled: [target calculation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:1809), AG factory lines 867–868. The [AG reader](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp:147) consequently executes two direction-0 slice iterations and one direction-1 iteration. Split forwarding is explicitly disabled when `fuse_op` is true. It signals once **after each complete received slice**, at lines 389–397. The [AG writer](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp:481) signals local readiness exactly once and only on direction 1.

[OpSignaler](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp:68) makes only the group master increment each destination semaphore. Two links therefore do **not** double `S0/S1` production. `chunks_per_sync=10` controls lower-level packet-ready notifications, not these slice-ready counts. Those transport notifications are produced per ten packets plus a final remainder and consumed by AG readers before they publish slice readiness. AG's scratch CB0, on its separate worker cores, couples packet production and consumption; it is not the matmul CB0. Increasing channel buffers, links, or chunk notifications cannot create a third gathered slice in a direction.

**First impossible wait.** The [input0 sender](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp:116) constructs its receiver once, before every batch/M/N/K loop. At lines 259–268 it calls the same object for every K pass. [The helper](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp:209) initializes both rank cursors to local rank `r`, starts at direction 1, and sets `curr_transfer_idx=0`. It increments that index at each slice boundary, without resetting it at a new outer block.

| `bh`, K-block boundary | Transfer index before wait | Direction / selected rank slice | Requested readiness |
|---|---:|---|---:|
| 0, 0 | 0 | 1 / `r` | `S1 >= 1` |
| 0, 10 | 1 | 0 / `r−1 mod 4` | `S0 >= 1` |
| 0, 20 | 2 | 1 / `r+1 mod 4` | `S1 >= 2` |
| 0, 30 | 3 | 0 / `r−2 mod 4` | `S0 >= 2` |
| **1, 0** | **4** | **1 / `r+2 mod 4`** | **`S1 >= 3`, impossible** |

This sequence was checked with a host-only arithmetic replay for all four ranks. Even granting fully completed AG, the second output-row block cannot proceed. If the eight passes could continue, each direction would eventually request 16 from a producer that supplies only 2. These are readiness observations, not 32 consumable tokens that AG should manufacture.

The predicted `bh=1, block=0, S1=2, target=3` is a **source deduction**, not a value decoded from the capture. It exactly predicts the captured sender stop-site and symmetric fanout.

**Matmul buffer and multicast ledger.** Factory lines 152–189 use input buffering depth 2. The relevant physical buffers and intended per-core traffic are:

| Resource | Producer → consumer | Size / loop count |
|---|---|---|
| A CB0 | NCRISC A sender or receiver → compute unpack | 8 tiles per K block; 16-tile capacity; `320 × 8 = 2560` tiles intended |
| A multicast handshakes | Seven row receivers → sender ready count; sender → seven VALID flags | Each receiver reserves 8 tiles, sets INVALID, increments ready by 1, waits VALID, then pushes; sender waits for 7, clears, multicasts data and VALID; 320 rounds intended |
| B CB1 | BRISC B sender or receiver → compute unpack | 72 tiles per K block; 144-tile capacity; `320 × 72 = 23040` tiles intended |
| B multicast handshakes | Seven column receivers → sender ready count; sender → seven VALID flags | Same protocol, 72-tile blocks, 320 rounds intended |
| Partial CB5 | Compute pack → compute reload/partial-credit retirement | 36 tiles per output block, in 2-tile subblocks; 39 non-final K rounds per outer block |
| Output CB4 | Compute pack → BRISC DRAM writer | 36 tiles per outer block, `8 × 36 = 288` output tiles per core |

CB4 and CB5 share 36 BF16 tiles of storage in this configuration: output is interleaved, intermediate/output formats match, and output is tiled ([factory sharing branch](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:1012)). Packer L1 accumulation retires partial CB credits per subblock; the penultimate K block retains partials for the final reload. There is no captured packer assertion establishing an initialization problem.

The [A multicast receiver](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp:75) waits at line 88 after announcing readiness. The sender is blocked **before** its next A CB reservation, DRAM read, or multicast. [Compute](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp:339) waits for A before B and later pops both at lines 493–494. Consequently B buffers fill and both kinds of BRISC reader stop reserving, exactly as captured. The B sender uses its own `MatmulOpReceiver` with `wait_for_op_signal=false` ([initialization](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:154)); its observed CB backpressure does not indicate a missing AG signal to B.

**The same lifetime boundary also affects addressing.** Input0's fused update is passed `in0_tensor_start_tile_id`, although the enclosing loop advances `in0_tensor_current_h_dim_block_tile_id` by `out_block_h × Kt = 320` tiles per `bh`. The helper overwrites the initialized inner cursor at each slice boundary, discarding that row offset. A semaphore-only workaround would therefore read the wrong M rows. B likewise passes `in1_batch_tile_id` instead of its current width-block base at lines 360–363; that loses a `bw` offset when N is split. This trial has `bw=0`, so the B width-base issue is latent here. Both operands' rank/page cursors must also restart together for every K traversal.

The M32 decode controls have one M block and one N block, so they never cross this defective lifetime boundary. Separate AG plus matmul does not enable this fused receiver. RS does not use this AG readiness protocol. The observed passing/failing contrast follows from the same contract.

## Downstream Effects

- The earliest source-side unmet condition is the input0 sender's demand for another AG readiness generation. The 224 input0 multicast receiver waits and 256 compute input0 waits across the mesh are its direct fanout.
- Input1 CB reservation stalls, math/pack synchronization, and dispatch/host completion delays follow from missing A. They are not independent evidence of B-format corruption, multicast acknowledgement undercount, or a global fabric deadlock.
- The capture does not identify an AG worker in a blocked send, close, or credit-return phase. Router polling/transaction-check frames supply no measured credit or acknowledgement deficit. A teardown or route-repair theory explains neither the precise A-sender wait nor the M32/M4096 contrast as directly as the finite readiness ledger.
- Native DRAM-sharded weight parser/K-order defects from the earlier report are inapplicable to this observed B access path: triage explicitly reports interleaved B and the BRISC stop is inside its interleaved branch.

## Proposed Fix

**Smallest caller adaptation for the captured projection:** select one output block per core: `out_block_h=per_core_M=16`, retaining the already-present `out_block_w=per_core_N=18`. The existing policy can express this with `pf_fused_role_block_h: {"mix": 16}`. The source change, if made, should enforce that constraint for the prefill AG path when feasible. This removes repeated K traversals without changing decode. Raising `block_h` only to 4 or 8 still leaves repeated passes and the same impossible wait; the saved `ag_large` candidate's value 8 is insufficient.

For this mix configuration, A CB storage becomes `2×16×4×2048 = 262144` bytes, B remains `2×4×18×576 = 82944`, and shared output/partial storage becomes `16×18×2048 = 589824`: **934912 bytes (913 KiB) per matmul core**, before other reservations. This is a plausible capacity candidate under Blackhole's 1536 KiB physical L1, not an allocator or hardware validation. Apply capacity checks independently to other roles: a wider gate/up projection can exceed L1 at full-height blocking. A mix-only override is not a whole-decoder fix. For roles that cannot fit, use the passing separate AG/MM path, or explicitly split that projection's M rows into smaller independent fused invocations and concatenate outputs. The existing separate-op control is the evidenced fallback.

**Exact kernel repair for general M/N output blocking:**

1. Preserve a fully initialized receiver template, and copy/reset its traversal state at the beginning of **each `(batch, bh, bw)` K pass** in both the input0 and input1 sender kernels. Restore both `ring_idxs`, both `start_page_idxs`, initial `curr_dir`, and `curr_transfer_idx=0`; retain runtime configuration and semaphore IDs. A pass-local copy of the original initialized object avoids reparsing already-consumed runtime arguments. Do not zero the readiness semaphores: their published data remains valid for all output blocks of this invocation.
2. In input0's update call, use `in0_tensor_current_h_dim_block_tile_id` as the base. In input1's call, use `in1_tensor_current_w_dim_block_tile_id` as the base. Preserve the existing batch and outer-loop advances. This restores the complete traversal/addressing state, rather than merely clamping the wait threshold.
3. Keep A and B on the identical per-pass rank order `r, r−1, r+1, r−2` for this TP4 configuration. Each subsequent pass can re-observe thresholds `1,1,2,2`; it requires no additional AG traffic. Scope changes to the relevant callers or audit every shared-helper caller before changing helper semantics globally.
4. Until that support is implemented, reject unsupported multi-output-block fused configurations in host validation, with a useful error. Current [validation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp:46) accepts rank-4 inputs with dimensions 0 and 1 equal to 1 and does not constrain these M/N loop counts. M4096 itself is not the invalid condition; repeated output-block traversals with an unrestarted receiver are.

**Independent metadata defect to handle before watcher verification:** the AG/MM factory still hardcodes `num_transfers=4` at line 58. The receiver multiplies this by two and asserts `num_tensor_slices × num_blocks_per_slice == num_blocks`: here **`8×10 != 40`**. For this even four-rank ring, the corresponding per-direction transfer parameter must be 2 (derive it from the supported ring geometry); alternatively represent the actual slice count explicitly and validate that count. Odd rings need an explicit contract rather than blind integer division. This is an initialization/assertion problem, not the captured hang: all observed senders are already past initialization. Fixing this metadata alone does not reset the traversal and does not remove the second-pass wait.

Absence checks were performed before proposing these changes: neither sender resets/reconstructs the receiver inside its outer loops; the helper has no per-pass reset; the A/B calls still use the bases described above; the caller defaults to height 2; and host validation has no single-output-block guard. Conversely, weight interleaving, runtime slice offsets, local-slice signaling, and the reader's final atomic barrier already exist and are not proposed as missing fixes.

## Uncertainty

- Confidence is high in the finite-producer/ever-increasing-consumer contract violation. The capture does not expose the numeric semaphore contents or `bh/block/curr_transfer_idx`; the precise second-pass state is predicted from the prepared source and supplied configuration. No claim is made that these locals were measured.
- No AG completion-counter snapshot is preserved. The lack of running AG worker stacks supports the diagnosis but does not independently prove that every transport write completed. The mismatch remains sufficient to deadlock even if every AG producer completes successfully.
- The preserved adapter matches current source byte-for-byte. Compiled kernel argument dumps and binary/source hashes are not included in this capture. The extra hardcoded-transfer assertion is inconsistent with an active assertion-enabled execution reaching line 253; disabled assertions or a build/source difference must be accounted for in any future watcher run.
- Neither the configuration adaptation nor the kernel repair was executed. Future authorized verification should check eager completion and numerical agreement across **all** M blocks and ranks, then repeated launches/trace replay; include single-output-block controls and multiple M/N output blocks. A fix that merely completes can still be wrong if it reuses the first M rows or mismatches A/B slice order. Preserve the existing native decode path during that work.
