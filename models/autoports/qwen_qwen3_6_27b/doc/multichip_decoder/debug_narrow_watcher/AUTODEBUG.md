# AutoDebug: native MMRS prefill watcher assertion

## Findings

**The leading defect is a dedicated-NoC ownership/accounting mismatch in native MMRS, not a missing read barrier in its writer.** The NCRISC reader deliberately sends weight reads through NoC1 while the BRISC writer is configured as a dedicated NoC1 kernel. The writer issues no reads, but its exit assertion compares the core-wide NoC1 read-response count with its own private read count. Reader completions change the former without changing the latter. This check can fail after every read has completed.

The failing compilation selects precisely this combination, and logical worker `(1,1)` is a weight-read sender. The reader has tagged completion barriers for every issued step; the writer drains its writes and staging handoffs. Adding an ordinary read barrier to the writer would poll the same mismatched counters and can hang.

**Confidence:** high for the source-level ownership defect and its applicability to the recorded configuration. Source inspection establishes a false-positive mechanism for the diagnostic “missing NOC reads flushed barrier.” The artifacts do not contain the actual hardware/software counter values or outstanding-transaction counts at the assertion, so they do not independently prove the exact register state or exclude every additional hardware defect. This is a real native-op contract defect even if the diagnostic falsely implies a pending read; suppressing watcher is not a repair.

**Coverage discrepancy:** the passing “B1/B32 S65 64-owner” resource runs both use **B1/S65 for prefill**. Their batch option changes decode. They also record an earlier model source hash. They are not a same-source B32/S65 prefill contrast.

No target code, tests, compilation, or device operations were run. This investigation used source/evidence inspection and local static searches. Only this report was created.

## Evidence and revision boundaries

- Checkout HEAD: `0915d93157bd320c7875533fba09bfbffeb2ebeb`.
- Current `tt/multichip_decoder.py` SHA256 is exactly the reported `de734a896dc879ef5e4d3ebfb2f41f4734d53e31af1cae52c48d2d8775e42f3e`.
- [Failing command](../final_watcher_sharded_l0_b32_s33.command.json) records layer0, B32/S33, sharded residual, `--trace`, `TT_METAL_WATCHER=10`, allocation tracking, the same HEAD, and that model hash.
- [Parent log lines 45–53](../final_watcher_sharded_l0_b32_s33.log) explicitly identify Device0 logical worker `(1,1)` / virtual `(2,3)`, BRISC, native `matmul_reduce_scatter_prefill_writer.cpp`, and waypoint `NKFW,W,W,W,W`.
- [kernel_names.txt](kernel_names.txt) IDs 450/451/452 identify reader/compute/writer of a native MMRS group. [watcher.log](watcher.log) stops its second dump immediately after the preceding worker `(0,1)`, which has IDs 452/450/451. It does **not** include a complete failing-core register dump; the parent log supplies the assertion and failing-core identity. Do not infer missing counters from this truncated dump.
- The parent log contains two reader compiler commands at lines 24 and 35. Both explicitly encode `NOC_INDEX=0`, `NOC_MODE=0`, `A_READ_NOC=0`, `W_READ_NOC=1`. They cover the GDN output projection and MLP down projection, respectively. Compilation of both does not establish that both executed before the failure. The conclusion applies to either configuration.
- [Resource B1](../final_resources_b1.json) and [resource B32](../final_resources_b32.json), lines 6–13, record model hash `384f75b273279e13b64327bc5aeef3612244997f7ac1804c78131220b386155c`, not `de734a…`. Their native program-factory hash `f5ec0087d726ead9d4df4f43b3093507bcb86a1132bf3b8610f18ac5fd00bd78` matches the current factory. No historical model snapshot was needed or assumed in this diagnosis.

Numerical non-watcher passes are reported by the requester and supported by the nearby final correctness artifacts. They are consistent with completed, correct reads plus incorrect cumulative accounting; they do not validate watcher compatibility. The nearby `final_watcher_baseline_l0_b32_s33.json` is a **single-chip** baseline (`provenance.mesh=[1,1]`), not proof that a proposed TP4 fallback has already passed.

## Lowered operation and exact failing geometry

The relevant model chain is GDN output → row-parallel output projection → residual add → normalization → local gate/up/SwiGLU → row-parallel MLP down projection → residual add. GDN itself pads each sequence to 64, but restores logical S33 before the output projection ([model], lines 623,674–703). The fused wrapper uses the logical batch and sequence dimensions, not B32×S64.

[Model] lines 354–384 flatten row-parallel inputs, make them DRAM interleaved, explicitly pad total rows to a multiple of 32, and call native MMRS. B32×S33 is already 1056 rows, so no rows are added. The two native weights are made DRAM interleaved at setup (lines 224–230); default activations are BF16 and weights BFP8. The logged aligned tile sizes are 2048 and 1088 bytes. The op returns BF16, DRAM-interleaved width 1280 per rank, then the wrapper slices/reshapes back to `[32,33,1280]`.

The call passes `cluster_axis=1`, `scatter_dim=-1`, Linear topology, and the model's link count. The C++ wrapper canonicalizes the scatter axis without swapping operands, derives G4 from the 1×4 mesh, and plans from the logical, explicitly padded M ([entry], lines 163–177,200–249). It also considers live L1 capacity, so a shape is not by itself a complete plan description. Model knobs `prefill_grid`, `prefill_k`, `readers`, and `cores` are not passed to this native MMRS call.

These values come from decoding the captured compile arguments against [args] lines 64–99 and [factory] lines 499–544:

| Effective field | GDN output projection | MLP down projection |
|---|---:|---:|
| A shape per rank | `[1,1056,1536]` | `[1,1056,4352]` |
| W shape per rank | `[1536,5120]` | `[4352,5120]` |
| M/K/N tiles | 33 / 48 / 160 | 33 / 136 / 160 |
| Per-owner output block | 33×40 tiles | 33×40 tiles |
| Per-core block; P | 3×5; 15 | 3×5; 15 |
| K block; K blocks per entry | 16; 3 | 8; 17 |
| Resident A / resident W | 1 / 0 | 1 / 0 |
| A / W capacity in windows | 3 / 3 | 17 / 3 |
| A / W split | 2 / 8 | 2 / 4 |
| Chunks; waves; entries | 1; 1; 4 | 1; 1; 4 |
| Prefetch; lookahead after barrier | 1; true | 1; true |
| A / W read NoC | 0 / 1 | 0 / 1 |
| Reader steps in complete invocation | 12 | 68 |

The one-wave rule makes the recorded 3×5 window the entire per-core block ([planner], lines 177–184). With the 11×10 available worker grid, `choose_grid` (lines 261–299) yields row strip, M-on-X, grid 11×8. [Factory] lines 414–429 maps `(gm,gn)` to logical `(gm,1+gn)`. Thus failing worker `(1,1)` is `(gm,gn)=(1,0)`, core index 8, output origin `(row0,col0)=(3,0)`.

The resident multicast placement algorithm ([factory], lines 198–261,754–769) yields sender starts `a=0,a2=4,b=1,b2=5`. Family b selects `(gm,gn)=((1+gn)%11,gn)`, so `(1,0)` is a b weight sender. It actually issues NoC1 reads; merely being inside the matmul grid is not the evidence.

For this worker `n_valid_w=5`. Its b part reads 8×5=40 tiles per step for the K48 plan, and 4×5=20 for K136. Over the four entries this is 480 or 1360 `noc_async_read` calls, respectively ([reader], lines 314–337). These tile sizes are below the NoC burst limit.

## Writer and buffer completion audit

[Writer] has the following complete transaction inventory. Its CB/semaphore waits poll local registers/L1; TensorAccessor construction/address calculation adds no hidden NoC reads.

| Writer path | Transaction and completion contract |
|---|---|
| Zero tile, lines 64–70 | Local L1 stores and CB push only; no NoC transaction. |
| FIN, lines 99–117 | Wait for each finished row group, issue valid output tile writes, wait for the complete P-tile slot, then `noc_async_write_barrier()`. |
| FWD/BWD, lines 118–120 | Wait for complete P-tile slot, increment chosen port mailbox. |
| Slot reclamation, lines 83–92 | For FWD/BWD, wait for monotonically increasing consumed acknowledgement; then pop exactly P tiles. FIN needs no remote acknowledgement because its writes were barriered. |
| Exit, lines 128–131 | Drain all queued slots, then `noc_async_atomic_barrier()`. |

The schedule depends on mesh position `p=coord[cluster_axis]`, not a physical device ID ([factory], line 360). The runner requests automatic physical-device mapping ([runner], lines 95–99); this capture does not establish Device0's mesh position. [Planner] lines 74–82,113–150 gives these complete G4/Linear schedules, identical for both halves:

| Mesh position p | Block order | Writer kinds |
|---:|---|---|
| 0 | 3,2,1,0 | FWD,FWD,FWD,FIN |
| 1 | 3,2,0,1 | FWD,FWD,BWD,FIN |
| 2 | 0,3,1,2 | BWD,FWD,BWD,FIN |
| 3 | 0,1,2,3 | BWD,BWD,BWD,FIN |

All have four entries and a final FIN, so the read counts and completion argument above do not require a Device0-to-p mapping. As a concrete queue instantiation, the p=0 worker `(1,1)` ledger follows. Handoff depth is 2 ([planner-header], line 119); other positions use the same slot counts with the listed direction-specific acknowledgements.

| Entry | Slot | Publication and reclamation |
|---:|---:|---|
| 0, block 3 | 0 | Wait 15 finished tiles, signal FWD; outstanding becomes 1. |
| 1, block 2 | 1 | Wait 30 tiles, signal FWD; outstanding reaches 2, wait ack1 and pop entry0's 15 tiles. |
| 2, block 1 | 0 | Wait 30 tiles, signal FWD; wait ack2 and pop entry1's 15 tiles. |
| 3, block 0 | 1 | Stream 3×5 valid tiles to output, wait 30 tiles and write barrier; wait ack3 and pop entry2. |
| Drain | 1 | Pop FIN's 15 tiles; outstanding becomes 0; atomic barrier. |

The FIN output pages are 120–124,160–164,200–204, from `(row0+i)*40+col0+j`; each row has all 5 valid columns. There is no ragged wave or short final slot at this worker. The port reader's `flush()` executes its read barrier **before** emitting the consumed acknowledgements ([port-reader], lines 60–68). The writer therefore cannot overwrite/reclaim a handoff slot while that port still reads it.

The reader's own completion chain also matters:

1. Each issued step selects TRID `1+step%8` on the read NoCs ([reader], lines 347–353).
2. Current-step and lookahead reads target reserved windows; lookahead is bounded by `issued<num_steps` (lines 359–407).
3. Every moving sender step waits on that TRID on NoC0 and, with the recorded configuration, NoC1 (lines 468–472), before publishing either operand window (lines 548–549).
4. The loop visits every step through `num_steps-1` (lines 554–556). It never prefetches an unvisited extra step. Resident A is re-credited only when it does not move; that does not leave a new read unbarriered.
5. Compute waits for these operand windows and consumes the K blocks before producing the finished output ([compute], lines 214–284 and the subsequent pack/finish paths). Writer completion requires all finished entries, including the full final slot.

Thus the source contains no unawaited W read on the path to a completed writer. The tagged barrier waits for read-command acceptance and hardware outstanding count for the ID to become zero ([dataflow], lines 2602–2612; [BH-NoC], lines 603–605). It does not rely on the cumulative per-RISC count that watcher later compares.

## Why “reads flushed” fails anyway

[Factory] lines 663–687 configures reader as NCRISC/NoC0 and writer as BRISC/NoC1. `DataMovementConfigDescriptor` defaults to `DM_DEDICATED_NOC` ([descriptor], lines 98–101), confirmed by the recorded reader compiler flags. Yet factory lines 535–536 select W reads on NoC1 for resident plans except the thin/wide exception.

Dedicated BRISC startup snapshots NoC1 hardware counters into BRISC's private software counters ([BRISC-kernel], lines 66–68; [BH-NoC], lines 873–885). The reader's NoC1 read calls increment **NCRISC's** `noc_reads_num_issued[1]` ([BH-NoC], lines 587–588); they do not update BRISC's copy. Hardware response counters cover both RISCs on that core.

The exit predicate is simply:

```text
hardware.NIU_MST_RD_RESP_RECEIVED[1] == BRISC.noc_reads_num_issued[1]
```

([BH-NoC], lines 599–600; [BRISC-kernel], lines 85–96). If the startup baseline is B and the reader completes R later NoC1 reads, the comparison becomes `B+R == B`, although the outstanding count can be zero. The host's diagnostic wording does not distinguish hardware ahead of software from hardware behind software ([debug-message], line 238).

This case does not require all reader traffic to start after the writer snapshot. Writer startup snapshots counters **before** publishing its zero tile. Compute waits for that tile before consuming operands ([compute], line 142). With only 3 W windows and 12/68 total steps, the reader cannot complete the entire invocation before that publication/compute progress. Under the kernel/CB contracts, some W read completions necessarily follow the writer's baseline, so the counter discrepancy is not merely a hypothetical launch-timing story.

The repository explicitly documents this exact hazard in its dedicated/disjoint sharing contract: hardware counters count both kernels, full barriers and kernel exit checks compare each RISC's counts, and a safe sharing implementation must re-sync only once the peer has nothing in flight ([sharing-contract], lines 59–82). Native MMRS performs no such re-sync/peer completion protocol.

Dedicated checks occur immediately after `kernel_main()` in `brisck.cc`, before the outer BRISC firmware's `wait_ncrisc_trisc()` ([BRISC-firmware], line 685). Dynamic mode instead aggregates both RISCs' counters and checks after that wait (lines 693–716). This explains why the logged `NKFW` identifies an epilogue check rather than a kernel stalled inside a reader barrier.

An ordinary `noc_async_read_barrier(1)` in this writer would loop on that same equality ([dataflow], lines 1921–1933). More completed peer reads cannot restore it. A blind counter reset can also hide real in-flight work; it is only safe after proving peer quiescence. Neither is the recommended intervention.

## Passing contrasts and limits

[Resource-test] line 218 constructs `prefill_host` with shape `[1,65,5120]`; line 246 creates B1 prefill states. Both `--batch 1` and `--batch 32` use that input in lines 278–289. Their decode sequence uses `args.batch` in lines 270–275, and batch≤32 makes `_linear` bypass native MMRS. The resource test explicitly synchronizes after prefill, eager decode, and capture, and uses blocking trace replay (lines 289,297,324,336), so its passes are meaningful for those workloads.

B1/S65 flattens to 65 rows and pads to 96, i.e. block 3×40 tiles. The source-derived unchunked geometry is M-on-Y, core 1×4, grid 3×10. It satisfies the native thin/wide condition:

```text
resident && scatter_dim == -1 && !m_on_x && grid_m <= 4 && grid_n >= 10
```

This sends W reads to NoC0 instead ([factory], lines 525–536), avoiding NoC1 counter contamination. Stream plans also select W NoC0. The actual resource runs do not record `_last_plan`, and their model source differs; this geometry is a source-derived explanation consistent with their passes, not a recovered runtime descriptor or proof of a same-source controlled contrast.

| Observation | What the finding explains / leaves open |
|---|---|
| BRISC writer reports missing read flush | Shared hardware count advances from reader W reads; BRISC issues no reads. |
| First reported worker `(1,1)` | Exactly a W sender under the recorded geometry. Watcher stops on its first reported error; this does not make it the only affected core. |
| Numerical non-watcher passes | Tagged barriers can complete the data correctly while cumulative exit accounting is wrong. |
| Resource B1/B32 passes | Both have small B1/S65 prefill; decode avoids native MMRS. Revision caveat remains. |
| `--trace` on failing command | The runner performs eager prefill/readback before trace work; this is not evidence of a trace-only bug. |
| GDN versus full attention | The defect follows selected native NoCs/regime, not GDN arithmetic. Other row-parallel shapes can reach it; their failures were not observed here. |

The runner calls `prefill_forward` then converts output/state to host before printing `PREFILL_DONE` ([runner], lines 154–187). Its timed prefill also explicitly synchronizes (lines 320–327); trace capture is later and covers decode. The failing log lacks `PREFILL_DONE`. Host enqueue progress or compilation of later kernels therefore cannot establish successful completion of initial prefill.

## Repair options within the allowed scope

### Smallest robust model-local adaptation

Use the existing `fused_mmrs=False` setup policy, or change that default in `multichip_decoder.py` while preserving explicit user policy overrides. Setup then leaves `row_parallel_weights` empty (lines 224–230), `_linear` delegates to `super()._linear` (lines 354–356), and the existing `_reduce` performs reduce-scatter on its full-width 5120 partial (lines 386–390). Both attention and MLP already call `_reduce` at their consumers (lines 419–420,454).

This avoids the defective native descriptor without shared C++ changes or host synchronization in forward. Do not additionally reduce inside `_linear` unless the width contract is kept consistent. The inherited projection and subsequent collective change arithmetic grouping and materialize a wider intermediate; TP4 accuracy, state, watcher, and performance require validation. The single-chip baseline is not that validation. This report recommends the fallback as the simplest scoped repair candidate, not as a measured performance-preserving fix.

### If retaining fused MMRS is required

The smallest **native descriptor** change is to force reader `reader_ct::W_READ_NOC` (positional CTA index 25) to 0, retaining A NoC0, writer NoC1, and dedicated mode. The reader already supports that branch, and the thin/wide path uses it. It removes the disputed sharing rather than adding a barrier. A descriptor override must preserve all per-device programs, resources, runtime bindings, and cache keys.

**API limitation:** the current Python MMRS binding exposes no descriptor override, reader-NoC option, or descriptor-export function. [Binding] lines 69–121 exposes the op plus `_matmul_reduce_scatter_prefill_last_plan()` only. `generic_op` accepts descriptors, but does not export this native factory's semaphores, mailbox, ports, and multicast descriptor. Therefore the CTA change is a precise intervention target, not an existing one-line Python option. Adding a shared C++ binding is outside the stated scope; recreating the factory in the model is substantial duplication.

A model-local fused workaround worth testing is to partition **only these row-parallel linear projections** into small flattened row pieces, pad each piece to tiles, call MMRS, then concatenate/slice and restore the logical shape. This leaves attention/recurrence sequence processing unchanged. Full 96-row pieces reproduce the source-derived 3×40 thin/wide geometry at adequate L1 budget; shorter tile-padded tails give 1–3×40 blocks. M1056 becomes eleven 96-row calls per projection, without extra mathematical rows. At most 128 rows can also yield the thin/wide path (nine calls: eight 128-row pieces plus 32), but lacks the direct S65 geometry precedent. Both increase dispatch, repeated weight reads, and per-call transport setup/handshake overhead; the output payload need not increase. Neither has been tested here. Live-L1-driven native chunking can change geometry, so they must not be called universally safe from row size alone.

Changing math fidelity, output placement, link count, or `l1_headroom` does not directly repair ownership. Some choices can alter the plan and avoid the branch, but the effective selected regime/NoCs must be established. Merely switching both matmul DM descriptors to dynamic mode would require a broader mode/command-buffer/barrier audit and is not the minimal model-local option.

## Follow-up validation (not executed)

1. First reproduce the exact sharded layer0 B32/S33 path under watcher after the selected model-local intervention, including eager prefill completion/readback and warmed repeat calls. Record model/native hashes, physical-device-to-mesh-coordinate mapping, and effective native plans when used. Preserve separate results for attention output and MLP down; compiled kernel presence is not execution evidence.
2. Compare both projections, complete decoder outputs, and GDN states with existing numerical gates. Validate full-attention layer3 too; do not infer it from GDN. Re-run at the existing nearby lengths/batches, including the B1/S65 resource case and B32/S33, so different row padding and native plans are exercised.
3. For a retained native path, record `_matmul_reduce_scatter_prefill_last_plan()` after **each** projection rather than only after the last call; inspect relevant CTA values as well. A post-call plan query is evidence, not a pre-launch safety check. Run with normal and resource-pressure L1 states, program-cache misses/hits, changed inputs, and blocking completion.
4. Re-run 64-owner resource isolation and performance checks if the change is retained. Make prefill batch/length explicit in reports; add a true B32/S33 prefill case instead of relabeling the existing decode-batch test.
5. A decisive native investigation, if later permitted outside this inspection, is to compare the same descriptor with W-read CTA index 25 forced 0 and collect hardware response count, BRISC/NCRISC software counts, and outstanding IDs. This would confirm the precise runtime accounting state. No watcher disabling or counter-reset-only workaround is needed to test the leading hypothesis.

## Other potential issues, reviewed and not promoted

- **A real read-barrier omission in the writer:** contradicted by the transaction inventory and the reader/compute/finished-slot chain. No writer read exists to flush.
- **Unbounded speculative tail read / short final wave:** lookahead stops at `num_steps`; every step is visited; this configuration has one full 3×5 wave and the failing worker has full valid extents.
- **Read transaction tags not cleared:** tagged waits do leave the read tag set, but the Blackhole dedicated packet-tag exit check deliberately checks only write/atomic-capable buffers 0,2,3, excluding dedicated read buffer 1 ([BH-NoC], lines 64–72,260–265). This is not a second missing cleanup.
- **Transport reads mislabeled as writer reads:** the port reader runs on distinct strip cores. Its reads are barriered before acknowledgement; they do not increment the failing matmul core's initiator response counter.
- **Prior-kernel leftover traffic or general corruption:** not exhaustively excluded without runtime counters, but requires extra unobserved conditions while the current compiled configuration already violates the dedicated accounting contract. It is secondary, not an equally established root cause.
- **Treating S33 as S64 or resource B32 as prefill B32:** contradicted by logical-shape handling and resource-test construction, respectively.

The report was revisited against the code after drafting: the headline is restricted to the NoC ownership/accounting defect; passing-plan reconstruction, hardware counter state, fallback correctness/performance, and fused row-partition safety remain explicitly qualified.

[model]: /home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/multichip_decoder.py:354
[entry]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/matmul_reduce_scatter_prefill.cpp:149
[factory]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_program_factory.cpp:499
[args]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_args.hpp:64
[planner]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_planner.cpp:261
[planner-header]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_planner.hpp:119
[writer]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_writer.cpp:95
[reader]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_reader.cpp:347
[compute]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_compute.cpp:142
[port-reader]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_port_reader.cpp:60
[BH-NoC]: /home/ttuser/qwen36-27b/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:599
[BRISC-kernel]: /home/ttuser/qwen36-27b/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc:66
[BRISC-firmware]: /home/ttuser/qwen36-27b/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc:685
[dataflow]: /home/ttuser/qwen36-27b/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h:2602
[descriptor]: /home/ttuser/qwen36-27b/tt-metal/tt_metal/api/tt-metalium/program_descriptors.hpp:98
[sharing-contract]: /home/ttuser/qwen36-27b/tt-metal/tt_metal/api/tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp:59
[debug-message]: /home/ttuser/qwen36-27b/tt-metal/tt_metal/impl/debug/debug_helpers.hpp:238
[resource-test]: /home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/check_multichip_stack_resources.py:218
[runner]: /home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/run_multichip_decoder.py:154
[binding]: /home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/matmul_reduce_scatter_prefill_nanobind.cpp:69
