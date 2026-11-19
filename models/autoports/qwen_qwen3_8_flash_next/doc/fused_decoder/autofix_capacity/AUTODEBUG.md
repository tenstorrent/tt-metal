# AutoDebug: PLE public4768 capacity regression, final_v4

Investigation date: 2026-11-13. Inspection only: no device execution, resets, source edits, or AutoDebug relaunch. This report is the only file created by the investigation.

## Findings

**The fused path loses the functional implementation's contiguous-memory margin before public prefill, then adds two avoidable 1.25 MiB buffers at the first GR-down product allocation.** The allocator correctly rejects a 40 MiB tensor: each of eight banks needs 5,242,880 bytes, but its largest free block is 5,178,752 bytes. Aggregate free memory is sufficient. The exact live-byte ledger below reconciles the failure without a previous product tensor: this is the first output block of GR down, not a later iteration retaining an earlier 40 MiB product.

Two concrete source discrepancies matter:

1. **An already-FP32 GR weight is copied by `typecast(weight, float32)` on every block.** In this checkout same-dtype device typecast allocates; it is not an identity operation. This adds 1,310,720 bytes at the failing allocation. The functional GR path directly multiplies its FP32 resident weight.
2. **Grouping resident projection weights for decode changes their placement and requires a new prefill slice.** Grouping all ten GR-down blocks into one parent makes the first prefill slice allocate another 1,310,720 bytes. Setup has already fragmented the free pool: before all-tail preparation the fused largest hole is about 20.23 MB smaller per bank than functional, despite only 761,856 additional allocated bytes across the device.

**Separately, the accurate-sum branch retains the full product until the next multiply returns.** This is a real avoidable lifetime issue for later blocks, but no previous product is live at the failed allocation captured here. It should be repaired without presenting direct overlap with an old product as the demonstrated cause of this traceback. Its indirect effect on preparation-time allocation placement is unmeasured.

Confidence: high for the redundant allocations, lifetime discrepancy, fragmentation, and byte reconciliation; incomplete attribution of individual free holes to particular setup operations. Removing the cast or changing grouping is not a hardware-verified capacity fix.

## Direct observations and revision consistency

Repository: `/home/ttuser/qwen38-flash/tt-metal`, HEAD `70dc6f5dd24298741e2f108a281160677e986155`. Model paths below are relative to `models/autoports/qwen_qwen3_8_flash_next` unless explicitly rooted at `ttnn/` or `tt_metal/`.

- [Failure log](../logs/final_v4_ple_4768.log): line 91 records completed preparation for prefill32, prefill1..31, and decode1, with runtime audit and restored state. Lines 93 and 102–137 locate the failure at `BinaryNgDeviceOperation::create_output_tensors`, called by `fused_decoder.py:60` from the first public prefill's `attn_mix.precise_down`.
- [Gate plan](../final_v4_gate_plan.json) selects `run_fused_probe --probe probe_ple_decoder --variant final`, length4768, decode3, `bfloat8_b` experts, the fully resident group10 packed PLE table, and `--prepare-all-tails`. The flag implies program preparation in [probe_ple_decoder.py:175](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_ple_decoder.py:175).
- [Selection](../final_v4_ple_4768_selection.json) records `FusedDecoder`, `FusedSparseMoe`, ordinary `GatedResidual`, final policy `{}`, and `completed=false`. Final policy uses constructor defaults; the failing residual normalization is not `FusedGatedResidual`.
- [Failure result](../final_v4_ple_4768.json) has no completed prefill, repeated prefill, or decode metrics. This is an allocation failure, not a numerical gate failure.
- [Functional context contract](../../context_contract.json) specifies a prepared, resident, batch1 PLE single-call profile through 4768. It does not promise maximum-length batch5 or maximum prefill with a live decode trace. [Functional public4768 evidence](../../functional_decoder/ple_public_hf_4768_private_v1.json) passes prefill, decode warm/capture/replay, and repeated prefill with the same length/preparation/table/expert settings.
- [Gate progress](../final_v4_gate_plan_progress.json) records preceding short-HF, continuation, smoke, and batch gates passing. PLE short-HF uses 33 tokens; batch5 uses lengths `[1,2,3,1,2]`; continuation uses small resident prefixes and at most 129-row independent prefills. These are not matching 4768-row residency controls.

The current fused source SHA256 is `42dfe986a782ad369fc11515aaff115068345a6e6f768925a0012a97106293cb`, matching both selection and provenance. All files listed in the failed run's source inventory matched current files when inspected. The functional passing run's `probe_ple_decoder.py`, `functional_decoder.py`, `decoder_components.py`, and `precise_residual_projection.py` hashes also match current files. Other historical inventory entries differ; the functional [capacity closure](../../functional_decoder/final_capacity_qsa_closure_v4.json) records format/QSA source closures. The baseline is historical evidence, not a fresh functional run in this investigation.

## Resident allocation contrast

These are recorded memory snapshots, not estimates. Free bytes are device-wide; largest-block values are per bank.

| Boundary | Functional free bytes | Fused free bytes | Functional largest block/bank | Fused largest block/bank |
|---|---:|---:|---:|---:|
| Before all-tail preparation | 514,902,528 | 514,140,672 | 64,362,816 | 44,131,712 |
| After preparation and synchronized cleanup | 341,782,016 | 362,415,616 | 41,217,152 | 29,590,912 |
| Fused failed product allocation | — | 118,805,504 | — | 5,178,752 |

Before preparation, functional `free / 8 == largest`, whereas fused has 20,135,872 free bytes per bank outside its largest block. Thus the substantial fragmentation predates the accurate-sum warmup. After preparation, fused has **20,633,600 more total free bytes**, but its largest block remains **11,626,240 smaller per bank**. A claim that fused simply retains more program-cache bytes does not fit these observations: preparation consumes 151,725,056 bytes net in fused versus 173,120,512 in functional.

The earliest relevant source changes are setup packing in [fused_decoder.py:1574](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:1574), followed by projection regrouping at [line1641](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:1641):

- Shared MoE gate/up/route weights are concatenated before the originals are deleted (`:164–172`).
- GDN Z/A/B weights are concatenated before their original fields are deleted (`:1583–1599`).
- `projection_decode_group=64` concatenates original precise-projection blocks (`:32–46`). This relocates four GR matrices totaling 50 MiB, a 5 MiB router matrix, a 50 MiB BF16 QKV matrix, and a 30 MiB BF16 output matrix: **135 MiB of precise-projection weights** with unchanged total weight bytes.
- Fused MoE reduction indices/mapping are additionally resident (`:145–162`).

Concatenation requires its destination while source tensors remain live. Replacing/deleting the originals subsequently leaves free holes among still-resident allocations. There is no evidence that the constructor permanently retains duplicate complete projection matrices. The logged net setup increase is small; placement is the larger difference. Exact attribution of the final largest hole among shared packing, GDN packing, projection grouping, and backend program allocations requires an allocation-address trace or controlled runs.

Preparation also retains program storage: [ProgramImpl:3223](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/program/program.cpp:3223) stores per-device kernel binary buffers in DRAM and allocates them from the top. The allocator requires a single adequate interval and coalesces adjacent free intervals; sufficient total free memory cannot combine separated holes into one tensor allocation.

## Exact first-failure resource ledger

`4768 = 149 × 32`: no partial public-prefill tail exists in this case. [FunctionalDecoder.prefill_forward:144](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py:144) retains the complete input and allocates a complete BF16 output before entering its 32-row loop. Each occupies `4768 × 10240 × 2 = 97,648,640` bytes. The probe separately keeps prompt token words and one decode input/token buffer resident ([probe:345](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_ple_decoder.py:345)).

The first chunk runs packed lookup and PLE before GR down. `_chunk` keeps `embeddings`, replaces its temporary input hidden tensor with the PLE residual result, and calls the ordinary GR reader ([functional_decoder.py:123](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py:123)). That reader retains `grouped`, `normalized`, and `flattened` at the nested projection call ([decoder_components.py:77](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/decoder_components.py:77)).

| Additional live allocation after preparation | Physical bytes across 8 banks |
|---|---:|
| Full prompt BF16 `[1,1,4768,10240]` | 97,648,640 |
| Full prefill output, same shape/dtype | 97,648,640 |
| Prompt token words UINT32 `[1,4768,16]` | 305,152 |
| One decode hidden input, padded 32 rows | 655,360 |
| Decode token words, bank-rounded | 512 |
| Current 32-row token-word chunk | 2,048 |
| Packed lookup result BF16 `[1,1,32,2560]` | 163,840 |
| PLE-updated hidden BF16 | 655,360 |
| GR grouped BF16 | 655,360 |
| GR normalized BF16 | 655,360 |
| GR flattened BF16 | 655,360 |
| Expanded FP32 activation, physically `[1,32,32,10240]` | 41,943,040 |
| FP32 weight slice `[1,1,32,10240]` | 1,310,720 |
| Same-shape FP32-to-FP32 weight cast | 1,310,720 |
| **Total** | **243,610,112** |

**362,415,616 − 243,610,112 = 118,805,504 = 8 × 14,850,688**, exactly the allocator's logged remaining free memory before creating the product. The failed product is not included in this ledger.

The expanded activation is easy to undercount. Its logical shape is `[1,32,1,10240]`, but TILE pads the height 1 to 32 in each of 32 outer planes. It therefore occupies 40 MiB, not 1.25 MiB. It cannot be an alias of the small `[1,1,32,10240]` input: [reshape.cpp:760](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:760) rejects that view; tiled reshape supplies a padded output allocation. This expansion is shared with functional GR, so it is not itself the fused regression.

The product is `[1,32,32,10240]` FP32: 10,240 tiles of 4,096 bytes, evenly distributed as 1,280 tiles per bank. [Bank sizing](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/allocator/allocator.cpp:784) yields 5,242,880 bytes/bank, matching the log. [BankManager](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:481) asks the allocator for that contiguous extent. Nothing in the logical/physical/output-size calculations disagrees.

The exact live-byte match supports the **first output block of the failing GR-down invocation**. An additional previous 40 MiB product, or accumulated reduced columns, would not fit this accounting without an unobserved compensating release. The traceback alone does not supply a block index; the ledger supplies the additional evidence. It does not independently identify the public token-chunk index: the same live-byte total could recur on another chunk. The roughly six-millisecond interval from entering the first public prefill audit to the OOM is consistent with failing on its first chunk, but that index was not instrumented. No heap address map was captured.

Numerically, `29,590,912 − 2 × 12,206,080 = 5,178,752`, matching the final largest hole after charging two full-prompt buffers to one extent. This is consistent with such placement, **not proof of addresses or allocation order within that extent**. In particular, the expanded 40 MiB activation also successfully acquired another sufficiently large extent before the product failed.

## Why the FP32 cast and slice are avoidable

[PreciseResidualProjection:42](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/precise_residual_projection.py:42) uploads GR weights as FP32. Functional down retains ten `[1,1,32,10240]` blocks and multiplies each directly (`:48–51`). Fused grouping combines them into `[1,1,320,10240]`; prefill still uses 32 output rows, so [fused:59](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:59) materializes a slice. The slice device operation creates an output allocation absent a supplied output tensor ([slice_device_operation.cpp:413](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp:413)).

The subsequent cast is another actual allocation:

1. The Python binding directly calls `ttnn::typecast` with no preallocated output (`ttnn/cpp/ttnn-nanobind/operations/copy.cpp:59–74`).
2. [typecast.cpp:31](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp:31) routes device tensors to `prim::typecast`, including equal input/output dtypes. FP32 precision preservation is enabled.
3. [TypecastDeviceOperation:162](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp:162) allocates a new device tensor. Its skip condition is zero volume, not equal dtype.

Each extra buffer costs 163,840 bytes per bank, compared with the observed 64,128-byte contiguous deficit. The size comparison makes them worthwhile interventions; it does **not** guarantee either allocation occupied the particular hole needed by the product. Removing one also changes preparation programs and placement, so its final effect cannot be predicted solely by subtracting bytes from the failure snapshot.

For this failing GR path, bypassing the cast leaves **FP32 × FP32 → FP32**. No mixed-precision argument is needed to justify the narrow same-dtype repair.

## Mixed-precision removal was checked separately

The same fused helper also handles BF16-resident GDN weights. A broader removal would pass FP32 activation first and BF16 weight second. The current lowered path supports FP32 arithmetic for that order:

| Stage | Actual current behavior |
|---|---|
| Default multiply | `binary.cpp:1390–1416` uses `fast_and_approximate_mode=false` for FP32/BF16; only block-float operands force fast mode. |
| Canonical operands/output | `binary.cpp:772–783,809–852` preserves lhs/rhs and chooses FP32 output from the FP32 lhs. The old `preprocess_inputs` swap/repeat helper has no call site on this path. |
| Validation/SFPU | `binary/common/binary_op_utils.cpp:66–82` accepts mixed floating MUL; `binary_ng_device_operation.cpp:63–64` selects SFPU when fast mode is false. |
| Broadcast | `[1,Q,1,K] × [1,1,O,K]` selects `ROW_A`, with outer-dimension broadcasting. |
| Blackhole factory | `binary_ng_program_factory.cpp:1134–1146` enables FP32 DEST from either operand/output; `:1450–1466` disables the affected LLK row-broadcast path in favor of software broadcast. |
| Unpack and compute | `:1374–1381` requests FP32 unpack for both inputs. `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp:58–89` loads lhs, reconfigures for rhs, multiplies in DEST, restores lhs format, and packs FP32 output. |
| Primitive | `binary_ng_utils.cpp:453–459` selects `mul_binary_tile`; Blackhole `ckernel_sfpu_binary.h:220–242` multiplies FP32 values without its BF16 rounding branch when FP32 DEST is active. |

Thus BF16 weight widening can occur during unpack without a DRAM FP32 copy. This is source-level support, not a measured cast/no-cast equivalence result. Keep the FP32 activation first and preserve the default accurate multiply; do not accompany removal with fast mode or operand reversal. Existing `test_binary_ng_bcast_fp32_dest_acc.py` covers related mixed broadcast but uses nearly uniform values/tolerances, not this exact model intervention. A nonuniform exact-shape comparison and integrated routing checks are still needed for a broader change.

## Other potential issues: product retention and later allocations

In the non-balanced branch, [fused:78–85](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:78) computes `reduced = ttnn.sum(products, ...)` but leaves `products` bound. `sum` creates separate output storage ([reduce_op_device_operation.cpp:294](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_device_operation.cpp:294)); reshaping/appending its result does not consume the Python reference to the input.

At the next iteration Python evaluates the slice, cast, and new multiply before rebinding `products`. Consequently the old 40 MiB product remains live while the next 40 MiB output is requested. The last product also remains live through final concatenation and BF16 output casting. Functional GR instead repeatedly shrinks `products`, explicitly deleting the full previous level before allocating the next addition ([precise_residual_projection.py:53](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/precise_residual_projection.py:53)).

The narrow intervention is to drop the Python product reference after enqueueing its last sum consumer. Do not force-deallocate a tensor that aliases `reduced` in the balanced-tree branch. Existing functional code already uses ordinary `del` after queued consumers, providing the local lifetime precedent.

This avoids an unnecessary full-size live buffer, but it is not proof of a larger aggregate peak than functional: the functional first reduction briefly holds full 40 MiB plus two 20 MiB slices. Fused instead requests another single 40 MiB extent while the old 40 MiB remains live, changing the contiguous-allocation requirement. The exact failure ledger above contains no previous product, so releasing it supplies no direct 40 MiB saving at this particular call. Changing the lifetime during preparation could still change the final free-hole layout; that indirect effect cannot be determined from these snapshots.

## Synchronization, admission, and alternative explanations

Preparation synchronizes after each warm call, before releasing warm buffers, and after restoring state ([probe:298–341](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_ple_decoder.py:298)). Its completion therefore has a real device-completion boundary. Preparation runs with only small hidden/token inputs, before either 97,648,640-byte public-prefill tensor exists.

Public prefill fails synchronously in product output allocation, before that multiply dispatch and before the probe's post-prefill synchronization. Earlier PLE/GR operations may be queued; the report does not infer their completed numerical correctness from this traceback. The immediate failure is neither teardown nor a reduction-kernel stall.

The public API checks logical context limits and chunks 32 rows. It does not make a DRAM-capacity admission promise, reserve a full forward workspace, or check available contiguous memory. The 4768 boundary is an explicitly measured physical profile, not a universal allocator guarantee. The failing binary output uses fixed broadcast/padding rules; no free-capacity-based planner chooses a conflicting tensor size here. Changing the documented context limit or weakening gates would avoid rather than repair the regression.

The passing contrasts fit this diagnosis: all-tail setup finishes before the two large public tensors are allocated; short-HF and continuation keep much smaller full-call inputs/outputs; batch5 uses very short serial requests. Streamed context smoke likewise does not establish full resident public-prefill capacity. Neither those passes nor the functional pass falsifies the demonstrated fused allocation-layout difference.

No allocator defect, incorrect output-size calculation, persistent duplicate whole-weight leak, or BF16 rounding failure is demonstrated by these artifacts. The fused model as a whole has less allocated memory after preparation; these snapshots do not isolate the cache cost of accurate sum from other operations. Exact attribution of the fragmented holes and whether one minimal repair suffices remain unresolved.

## Smallest useful interventions and follow-up controls

These are recommendations for a subsequent authorized repair, not changes made here.

1. **Bypass same-dtype FP32 weight casts.** This removes a proven redundant allocation on the failing GR path while keeping its operand types and arithmetic. Start with the dtype-conditional change if limiting numerical scope matters; the broader mixed path has the source support described above.
2. **Release full products after the accurate sum's last use.** This removes the proven later-block lifetime defect without changing the summation order. Evaluate its indirect preparation-layout effect separately from the direct first-block cast/slice savings.
3. **Test PLE with `projection_decode_group=1` while retaining accurate reductions and other final settings.** This preserves original projection weight blocks, avoids their 135 MiB setup relocation, and removes the GR prefill slice. Compare setup/preparation free blocks and end-to-end gates; shared/GDN packing still differs, so this is a control, not a guaranteed solution or recommendation to disable every fusion.
4. If necessary, isolate shared projection packing and GDN Z/A/B packing separately, or redesign setup loading/allocation order to avoid repacking an already resident model. `pack_gdn=False` alone does not disable Z/A/B packing: the default exact-convolution/layout flags still enter the condition at line 1583 and execute the concatenation at line 1586. Reducing token chunk size is a broader workspace intervention with new program and performance consequences, not the first narrowly supported fix.

For a decisive hardware follow-up, retain the exact prepared all32tails4768 command, dtype/table settings, all correctness gates, and repeated-prefill lifecycle. Record allocator blocks at: post-construction/state setup; post-preparation; post-full-input/output allocation; post-expanded activation; post-weight slice/cast; and just before each GR-down multiply, including chunk/block index. Compare each intervention independently before combining them. Then rerun public4768, the functional-profile4767 neighbor, tail coverage, short-HF, continuation, batch5, and trace/reset gates. The historical4769 failure is useful boundary evidence but must not be silently substituted for the4768 regression target.

## Draft review and verification

Cheap checks performed: source/provenance SHA256 comparison, JSON inspection of exact commands/results, integer calculation of tile/bank sizes and the complete live-byte ledger, and Python/C++/kernel source tracing. No target modules were imported or executed.

The first-draft claims were challenged against source and observations: same-dtype typecast was checked through the allocating device operation rather than assumed to alias; mixed precision was checked after actual dispatch; the product-retention hypothesis was demoted from the observed-failure explanation after the exact first-block ledger; larger program-cache growth and an aggregate-memory shortage were rejected using recorded snapshots. Address-specific causality and a guaranteed repaired4768 pass are deliberately left unclaimed.

An independent xhigh report review passed after narrowing the token-chunk-index inference, distinguishing direct product overlap from indirect warmup placement effects, and correcting the reduction-output allocation citation. It independently checked the byte totals, setup regrouping, mixed-precision lowering, and report links. All 173 entries in the failed run's source inventory matched current files; no entries were missing.
