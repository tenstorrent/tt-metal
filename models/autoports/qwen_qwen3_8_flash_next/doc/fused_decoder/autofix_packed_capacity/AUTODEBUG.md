# AutoDebug: packed expert public4768 contiguous-allocation regression

Investigation date: 2026-11-13. Inspection only: no hardware execution, device queries, resets, runtime edits, or AutoDebug relaunch. This report is the investigation's only file change.

## Findings

**This is a placement-sensitive allocation regression, not an extra live packed/slice tensor or an incorrectly sized down output.** The packed-SiLU failure has 19,514,240 free bytes per bank, but its largest interval is 10,160,512 bytes; the correctly sized 80 MiB down output needs 10,485,760 contiguous bytes per bank. The deficit is **325,248 bytes/bank**, or 2,601,984 bytes across eight banks. The source-derived live-buffer ledger below exactly reconciles the reported free memory, with the 40 MiB packed result already released; it is not a captured address inventory.

**The strongest runtime explanation is the changed placement of the surviving 20 MiB activation.** The passing direct path retains its first gate allocation and computes SiLU/multiply into it. Both packed paths allocate their surviving activation while an earlier 40 MiB packed result is still live. Deleting the packed result afterward restores free *volume*, but cannot move the surviving activation or coalesce across it. This explains why removing the slice buffers can improve an earlier peak without fixing down's contiguous allocation. The allocation order is proven; the identity and addresses of the blocks separating the actual free intervals remain unrecorded.

**There is also a revision-specific setup ownership difference.** Historical direct/unpack retains the old MLP's shared gate/up/route buffers through fused-reduction metadata setup. Packed/current code replaces the MLP earlier, freeing these buffers before metadata allocation. Consequently, the reported direct pass is not a one-option comparison against current packed code. This changes persistent placement without changing final weight bytes. It should be isolated, not conflated with an expert-weight leak.

Concrete smaller opportunities are dead routing/decoder locals, redundant reduction-index storage, and an avoidable 1 MiB tiled score permutation. None is demonstrated to restore public4768 by itself. The first minimal repair experiment should change **activation allocation order**, preserving the packed matmul, 40-core `(10,4)` gate/up policy, arithmetic, exact QKV tree, resident PLE table, and public API capacity.

## Direct observations and revision boundaries

Repository HEAD in all three captures is `70dc6f5dd24298741e2f108a281160677e986155`; uncommitted source hashes distinguish them.

| Evidence | Fused source SHA256 | Result |
|---|---|---|
| [direct/unpack4768](../direct_unpack_v1_ple_4768.json), [selection](../direct_unpack_v1_ple_4768_selection.json) | `3dc6c4e64005ca22a489691e9b5b8a31948dc6a842aa3b4b801fa7d00805d25c` | Prefill, decode capture/replay, state reset, repeated prefill pass |
| [direct/unpack4767](../direct_unpack_v1_ple_4767.json) | Same direct source | Same gates pass |
| [packed40 log](../logs/packed40_v1_ple_4768.log), [selection](../packed40_v1_ple_4768_selection.json) | `0c6cfb1a5a108d2264e8d9ff488bd0ccd2d01f5dab676deb99afe1f83b24c47d` | Down output allocation fails, log lines 93 and 111–137 |
| [packed-SiLU log](../logs/packed_silu_v2_ple_4768.log), [selection](../packed_silu_v2_ple_4768_selection.json) | `803c8b8d1cd209fb5d2852ba6790464157f3f0ee58589aa860077560f0fbefe0` | Same down allocation fails; this is the current runtime source |

Frozen fused sources and launchers under `../sources/<capture>/` hash-match their provenance. Against the current model tree, direct provenance has four differing files (fused runtime and three probe/helper files), packed40 has two, and packed-SiLU has only `tests/run_fused_probe.py` differing. No inventoried files are missing. The current launcher adds policies; the captured selected policies are preserved. Common `functional_decoder.py`, `decoder_components.py`, and `probe_ple_decoder.py` match. Historical claims below use the frozen sources, not current line numbers retroactively.

All three public runs use layer1, BF8 expert weights, the fully resident group10 PLE table, length4768, decode3, and all32-tail preparation. Direct has compact experts/GDN I/O, direct grouped projection setup, unpack widening, and the default grouped64 exact QKV tree. Packed40 adds expert packing and gate/up grid `(10,4)`; packed-SiLU also enables the packed reader. `compact_experts=True` only selects compact execution for `tokens == 1`; the full 32-row chunks still use expanded experts ([current branch](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:410)).

[Packed40 short-HF](../packed40_v1_layer1_hf.json) and [packed-SiLU short-HF](../packed_silu_v3_layer1_hf.json) pass at length33. [Integrated packed-reader component evidence](../packed_silu_integrated_component_v2.json) reports exact outputs and replay for groups `{1,10,512}` × rows `{1,7,31,32}`. These are compatible with this allocation diagnosis; they do not exercise the two large resident public4768 tensors. No numerical failure is established by the OOM captures.

### Recorded allocator snapshots

Free bytes in the first two columns are device-wide. Largest blocks and failure values are per bank.

| Run | Free before preparation | Largest before/bank | Free after preparation | Largest after/bank | Free at failure/bank | Largest at failure/bank |
|---|---:|---:|---:|---:|---:|---:|
| Direct/unpack | 514,214,400 | 44,140,928 | 374,697,472 | 30,148,992 | Not captured | Not captured |
| Packed40 | 514,214,400 | 44,224,896 | 373,648,896 | 30,101,888 | 19,385,216 | 10,031,488 |
| Packed-SiLU | 514,214,400 | 44,224,896 | 374,681,088 | 30,230,912 | 19,514,240 | 10,160,512 |

Important constraints on the story:

- All three have identical free volume before preparation. A persistent additional expert matrix is incompatible with this observation.
- Packed-SiLU has only **16,384 bytes more resident storage after preparation than direct**, and its largest block is **81,920 bytes/bank larger**. Therefore “packed-SiLU already has a smaller largest hole before public prefill” is false.
- Packed40 versus packed-SiLU improves free memory by 1,032,192 bytes total and the largest interval by 129,024 bytes/bank, both after preparation and at failure. Both failing paths consume exactly **218,567,168 bytes** between post-preparation and the failed request, and reduce the largest-block statistic by exactly **20,070,400 bytes/bank**. The reader does not save another live 20/40 MiB at the point of failure.
- Program preparation leaves persistent storage. [ProgramImpl](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/program/program.cpp:3223) owns DRAM kernel binaries and allocates them top-down. The net preparation totals do not identify particular binary buffers or their addresses.

## Allocation dimensions and exact live-buffer ledger

Model dimensions are hidden2560, intermediate640, experts512, top-k10, four residual streams. `4768 = 149 × 32`. Public prefill retains its complete input and allocates a complete output before the chunk loop ([functional decoder](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py:144)); each is `4768 × 10240 × 2 = 97,648,640` bytes. The probe also retains one decode input/token buffer ([probe](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_ple_decoder.py:345)).

Sparse matmul's dense-A/sparse-B gate path combines the operands' batch axes. Its expanded logical gate output is rank6 `[1,1,1,512,Q,640]`, or width1280 when packed. Down selects sparse A and sparse B, so the shape helper drops A's batch dimensions and produces `[1,512,Q,2560]`. See [shape construction](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:49) and [output specification](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:610).

At Q32:

- Separate gate or activated: `512 × 32 × 640 × 2 = 20,971,520` bytes, 20 MiB.
- Packed gate/up: `512 × 32 × 1280 × 2 = 41,943,040` bytes, 40 MiB.
- Expanded down: `512 × 32 × 2560 × 2 = 83,886,080` bytes, 80 MiB.

Down has 40,960 BF16 tiles of 2,048 bytes. [Bank sizing](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/allocator/allocator.cpp:784) gives 5,120 tiles/bank = 10,485,760 bytes/bank. [BankManager](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:481) requests one contiguous interval. [FreeListOpt](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/allocator/algorithms/free_list_opt.cpp:135) cannot combine separated free blocks; deallocation coalesces only adjacent intervals. There is no demonstrated logical/physical size mismatch or incorrect extra expert factor.

The explicit [program builder](/home/ttuser/qwen38-flash/tt-metal/models/demos/gpt_oss/tt/experts/config.py:134) also reconciles at Q32. Direct gate/up use grid `(5,1)`, N20 tiles, four N tiles/core, K80 tiles; packed gate/up use `(10,4)`, N40 tiles, one N tile/core, K80 tiles. Both down paths use `(8,1)`, N80 tiles, ten N tiles/core, K20 tiles. All use one M tile/core, `in0_block_w=1`, and 1×1 output subblocks. Packing overrides only gate/up cores, not down's grid or output extent. A smaller gate core grid may change program storage/performance but does not directly reduce these tensor byte counts.

The following are additional live allocations after the preparation snapshot, immediately before down allocation. Counts include the bank-rounding of small tensors, not just logical payload size.

| Owner/buffer | Shape or purpose | Allocated bytes across 8 banks |
|---|---|---:|
| Probe full hidden input | BF16 `[1,1,4768,10240]` | 97,648,640 |
| Public prefill output | Same shape/dtype | 97,648,640 |
| Prompt token words | UINT32 RM `[1,4768,16]` | 305,152 |
| Decode hidden input | BF16, physical32 rows | 655,360 |
| Decode token words | UINT32 RM, bank-rounded | 512 |
| Current token-word chunk | UINT32 RM `[1,32,16]` | 2,048 |
| `_chunk.embeddings` | PLE lookup, BF16 `[1,1,32,2560]` | 163,840 |
| `_chunk.hidden_states` | Current residual, BF16 `[1,1,32,10240]` | 655,360 |
| `_chunk.mixed` / MoE `x` | Shared reference to BF16 `[1,1,32,2560]` | 163,840 |
| `_chunk.block` | Already-consumed attention result | 163,840 |
| `_chunk.injection` | BF16 `[1,4,32,1]`, four tiles, bank-rounded | 16,384 |
| MoE `activated` | Expanded BF16 expert activation | 20,971,520 |
| MoE `logits` | FP32 `[1,1,32,512]` | 65,536 |
| MoE `scores` | FP32 `[1,1,32,10]`, one tile, bank-rounded | 32,768 |
| MoE `indices` | UINT32, same padded shape | 32,768 |
| MoE `dense` | BF16 `[1,1,32,512]` | 32,768 |
| MoE `active` / `kwargs['sparsity']` | Shared reference to BF16 RM `[1,1,1,512]` | 8,192 |
| **Total** | | **218,567,168** |

**374,681,088 − 218,567,168 = 156,113,920 = 8 × 19,514,240**, exactly the packed-SiLU log. Likewise **373,648,896 − 218,567,168 = 8 × 19,385,216**, exactly packed40. The failed 80 MiB request is not included.

The enclosing decoder locals come directly from [the chunk body](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py:123); routing and activation locals from [FusedSparseMoe](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:403). Router output shapes are explicit in [router_topk.py](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/router_topk.py:133). The one-tile score/index tensors each reserve a tile per bank, explaining why counting only their 4 KiB payloads would miss 56 KiB.

This exact match supports a normal Q32 chunk with no extra packed/slice buffers, no previous down output, and no unaccounted weight copy. It does not identify the token-chunk ordinal; the same live ledger can recur later. No allocation-address map was captured.

## Leading causal explanation: activation allocation order

The relevant historical/current sequences are:

| Path | Allocation and last-use sequence before down | Surviving activation |
|---|---|---|
| Direct/unpack PLE | Allocate gate20; allocate up20; SiLU/multiply with `output_tensor=gate`; delete gate/up names | The **first** 20 MiB allocation |
| Packed40 PLE | Allocate packed40; materialize gate20 and up20 slices; delete packed; SiLU/multiply into gate; delete gate/up names | A 20 MiB allocation created **while packed40 was live** |
| Packed-SiLU PLE | Allocate packed40; helper allocates activated20 and writes it; delete packed | A 20 MiB allocation created **while packed40 was live** |

Evidence: [frozen direct source](../sources/direct_unpack_v1_ple_4768/fused_decoder.py), lines335–355; [frozen packed40](../sources/packed40_v1_ple_4768/fused_decoder.py), lines343–370; [current helper](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:313) and [call site](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:433). Slice outputs have their own allocation ([slice device operation](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp:413)); the packed helper's `ttnn.empty` likewise gives independent output storage. No view of the packed buffer is returned.

The packed reader cuts the gate-stage peak from packed40 + gate20 + up20 = 80 MiB to packed40 + activated20 = 60 MiB. But both paths reach down with the same20 MiB activation, and allocation cannot compact its address automatically. This predicts both failing packed variants, their identical incremental failure ledger, and the reader's inability to recover capacity merely by eliminating slices.

**Smallest implied intervention:** allocate the activation destination before producing packed gate/up, then give that destination to the custom reader/writer. The helper currently lacks an output argument; add a validated optional destination and keep the same output geometry and arithmetic. This schedules the necessary20 MiB allocation earlier, performs no additional device copy or numerical operation, and preserves the60 MiB gate-stage peak and100 MiB activation+down peak. It is the closest local analogue to direct's early retained gate allocation.

This is a testable placement hypothesis, not a guaranteed fix. An earlier activation could also make the subsequent40 MiB packed request harder to place. Record both addresses/free intervals and run the whole capacity lifecycle. Do not implement overlapping input/output as a shortcut: different cores read different packed source tiles, and safe in-place compaction has not been established.

## Setup ownership and packed weight placement

Current setup creates the functional MLP, shallow-copies its dictionary, and assigns `model.mlp = fused` **before** packing ([current setup](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:1966)). The historical direct version assigns it **after** packing and `prepare_fused_reduce` ([frozen direct](../sources/direct_unpack_v1_ple_4768/fused_decoder.py), lines1851–1859).

The distinction matters for shared weights: `pack_projections` concatenates shared gate/up/route then deletes their fields. Historical direct's old MLP still owns the three original buffers until replacement; current code releases them immediately. Hence the32 index tensors, mapping, and compact-sparsity tensor can be placed into different holes.

There is a useful exact numeric signature. Those metadata tensors allocate:

`Σ(n=1..32) ceil(n/8) × 1024 + 1024 + 1024 = 83,968 bytes/bank`.

This exactly equals the packed pre-preparation largest-block advantage: `44,224,896 − 44,140,928 = 83,968`. UINT16 RM512 rows have 1024-byte pages ([page layout](/home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/tensor/spec/layout/page_config.cpp:113)); the bank spread formula explains rounding. Moving metadata into newly freed shared-weight holes is therefore strongly consistent with the snapshot. This equality still does not establish allocation addresses. The changed setup actually increases the largest prefill-entry block in packed-SiLU, so it is a secondary placement variable, not proof of a worse setup peak.

The expert packing itself does **not** retain the original pair: gate/up were uploaded consecutively ([original loader](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/decoder_components.py:122)); current code obtains device/dtype, deletes both, and uploads checkpoint-transposed `[1,512,2560,1280]` ([replacement](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:386)). Both640 and1280 are tile-aligned; BF8 tile counts of the packed weight equal those of the two originals. Host `transpose/contiguous` storage does not allocate device DRAM. The original gate/up pair together and their replacement each occupy 1,782,579,200 device bytes at 1088 bytes/BF8 tile. Equal total setup free memory independently constrains any duplicate-weight theory.

**Minimal setup control:** retain only the three original shared-weight references until reduction metadata preparation finishes, while continuing to release original expert buffers before their packed upload. Do not restore the old MLP owner wholesale: that would also retain the very large expert pair during replacement. A native packed loader that avoids constructing separate device gate/up in the first place is a cleaner future setup design, but must be tested for placement; it is not needed to explain a nonexistent persistent duplicate.

## Concrete removable buffers and reduction-side opportunities

### Dead locals at the failed request

The following references have no remaining data consumer by down allocation:

- `logits`, after top-k and dense zero/scatter construction; `scores` and `indices`, after dense construction in expanded mode. Their bank-rounded allocation is **131,072 bytes total / 16,384 per bank**. Delete after their final queued consumer, separately in compact/expanded branches. `active` is still needed by down and is also retained through `kwargs`; `dense` is still needed by reduction.
- `_chunk.embeddings`, after PLE consumes it, and `_chunk.block`, after `attn_mix.write` consumes it. Together **327,680 bytes total / 40,960 per bank**. The current Python locals survive through the MoE call.

These five references total **458,752 bytes / 57,344 per bank**, less than the325,248-byte/bank deficit. Their removal is useful and adds no device arithmetic, but cannot be sold as sufficient by byte subtraction. It might join larger neighboring holes; only addresses can establish that. Do not delete `mixed`/`x`, residual, injection, or activated before their real last consumers, and do not force-deallocate aliased inputs. The local code already uses ordinary `del` after enqueuing the final consumer.

### The existing fused weighted reduction is already enabled

[Current lines475–490](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:475) call `deepseek_moe_fast_reduce_nc_fused` **after** the expanded down output is created. Thus “enable fused weighted reduction” is not a fix for this failure: it is active in both passing direct and failing packed runs, and does not fuse the down matmul itself.

There are nonetheless two concrete opportunities:

1. **Avoid the tiled score permutation.** Current `to_layout(permute(dense,(2,0,1,3)), ROW_MAJOR)` changes TILE `[1,1,32,512]` to logical `[32,1,1,512]`, physically `[32,1,32,512]`. That materializes **1 MiB**, then converts to a32 KiB RM tensor. Instead convert dense to RM first and reshape `[1,1,Q,512]` to `[Q,1,1,512]`. The row order is identical and the last dimension stays512, allowing a metadata RM reshape. TILE permute is not a no-op when moving H ([permute](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/permute/permute.cpp:170)); see [reshape view conditions](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:760). This removes an operation and transient allocation while preserving dense expert order. It helps a **later** peak and can change preparation footprint/placement; it does not directly remove bytes already live at the failed down request.
2. **Deduplicate reduction indices.** The32 `arange(512)` tensors cost655,360 bytes after bank rounding, although every token uses the same expert list. The existing reader derives token count from scores and can consume a maximum32-row index tensor for smaller Q; validation does not require matching index Q. Reusing that32-row allocation would save **622,592 bytes total / 77,824 per bank**. It also makes tails mirror more index metadata than before, so tail performance needs measurement. A dedicated broadcast-one-row reader would remove that overhead, but is a kernel change. This is a setup-memory optimization; its address effect is not predictable from aggregate statistics.

The index-saving arithmetic is `655,360 − 32,768 = 622,592`; mapping and compact-sparsity remain separate. Validate the reuse against all Q1..32 and compact decode rather than assuming every index API accepts broadcast tensors.

The shared table's contract was checked through the [factory](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp:128) and [reader](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/deepseek_moe_fast_reduce_nc_fused_reader.cpp:98): the buffer has32 actual1024-byte pages, all are mirrored, and only `t < num_tokens` accesses indices. The maximum address is element `31×512+511=16383`, within the32 KiB table. Keep compact decode's existing `compact_ids`; only expanded `self.reduce_indices[tokens]` is replaced.

**Do not replace the expanded reduction's512-wide inputs with top10 IDs/scores as a drop-in.** [Validation](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_device_operation.cpp:84) requires scores' final dimension to match the input reduction dimension. [The reader](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/deepseek_moe_fast_reduce_nc_fused_reader.cpp:196) uses IDs to determine device ownership and scans activation planes sequentially; IDs do not gather arbitrary planes of an expanded input. A true top-k gather reduction or combined down/reduce kernel needs new indexing and arithmetic-order validation. The current compute scans ascending expert planes; an arbitrary top-k order may change rounding.

Also, `zeros_like(typecast(logits,bfloat16))` can use `zeros_like(logits,dtype=bfloat16)` and omit the temporary cast. This removes an earlier32 KiB transient and potentially a preparation program; it is not a live32 KiB saving at down.

## Synchronization, capacity planning, and excluded explanations

Preparation executes all32 physical prefill lengths plus decode1, synchronizes after each call and during cleanup, restores state, and verifies state equality ([probe lines298–341](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_ple_decoder.py:298)). It completes before the public input/output tensors are resident. Its success is therefore real but is not a full-residency capacity test.

The public failure is synchronous in `SparseMatmulDeviceOperation::create_output_tensors`, before this down operation is launched and before the probe's post-prefill synchronization. [Output creation](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:651) precedes [device launch](/home/ttuser/qwen38-flash/tt-metal/ttnn/api/ttnn/device_operation.hpp:630). Earlier work can still be queued. This traceback neither establishes its completed numerical correctness nor indicates a reduction-kernel stall. This down invocation's new program cannot have allocated a binary before its output allocation failed; prior prepared programs remain resident.

The caller admits logical context length and selects Q32 chunks; it does not reserve a full forward workspace. Output shape is determined by operands, explicit config, and compact indices, not by a free-memory admission heuristic. No demonstrated planner/output-capacity contradiction calls for changing the allocator. Warmup uses small sources but has the same physical Q signatures; adding the two full public tensors later is the relevant residency transition.

Reducing Q from32 to31 or16 does not shrink a single expanded expert tensor: each expert still has a padded32-row tile. For4767, hidden input/output still physically pad to4768 rows; the last chunk is31 rows. The historical direct4767 pass covers that neighbor, not packed4767. Do not infer a packed4767 pass or reduce the public limit as a repair.

The [previous capacity report](../autofix_capacity/AUTODEBUG.md) concerns an earlier GR-down40 MiB product failure. Current code already bypasses same-dtype FP32 weight casts, releases accurate-sum products, and uses direct grouped uploads ([current projection](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:33)). Those historical headlines must not be carried forward as unresolved causes of this MoE down failure.

## Minimal experiments, in priority order

These are follow-ups for the serial hardware owner, not actions performed here. Preserve public4768/4767, all-tail preparation, the resident table, grouped64 exact QKV, route/numeric gates, repeated prefill and decode/reset/trace gates.

1. **Run current-source direct/unpack as the matched control.** Its old shared-buffer owner now dies earlier than in the historical passing source. Record source hashes and the same preparation/free-block measurements. This separates a source revision effect from the packing switch.
2. **Allocate activated20 before packed40 and use it as the packed-reader destination.** Compare with current packed-SiLU, changing no other policy. Capture addresses of packed/activated and free intervals immediately before/after both allocations, after releasing packed, and before down. Falsification: if the surviving activation does not separate the needed intervals or early placement still fails at identical capacity, demote this particular placement explanation.
3. **Isolate setup ownership:** retain only original shared gate/up/route through reduction metadata setup. Compare setup address maps and public4768. Keep expert release-before-upload intact. This is setup-only and does not add forward work.
4. **Apply the dead-local and RM-first score changes independently**, then index deduplication. The first has direct but small live savings; the second has a later transient saving; the third reduces persistent metadata. Measure preparation binaries/free ranges and the failure boundary rather than adding their payload savings and claiming guaranteed coalescence. Confirm exact score row order, all tails, and clean latency.
5. **If placement fixes are insufficient, reserve down output explicitly as a controlled alternative.** Sparse matmul accepts **`optional_output_tensor=...`**, not `output_tensor` ([binding](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1239)). Reserve `[1,512,32,2560]` BF16 DRAM before fragmentation and expose exact logical-Q views with explicit padded shape. The op validates shape and zero-fills the supplied output in place before sparse writes ([reuse](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:651)). Never alias activated. Reservation removes the fresh80 MiB request but raises the packed gate-stage live set to **80+40+20=140 MiB**, versus the current100 MiB local maximum when down succeeds. It can move the failure earlier; it is not a free capacity saving. Scope workspace ownership to serialized calls and verify trace/replay use.

For each run, capture post-construction, post-preparation, post-public-buffer allocation, pre/post-packed, post-activation/release, pre-down, and pre-reduction boundaries with chunk index, logical/padded shape, dtype, page size, buffer address, and allocator free ranges. The current free/largest snapshots alone cannot identify the blocking live allocation. A small matched control plus these measurements is more decisive than changing several fusions at once.

## Other Potential Issues / broader alternatives

- A **compact prefill union** could bound expert planes at `10Q ≤ 320`, reducing down's maximum allocation from80 to50 MiB at Q32. Existing indices mode supports compact output, but production currently only builds compact routes for Q1. This requires device union/deduplication and consistent per-token scores, not just toggling the guard. A guessed `nnz` is unsafe and alone does not shrink default output; compact mask mode needs matching descriptors/output and writer semantics.
- **Split down width**, e.g. two1280-column outputs, reduce each and concatenate small results. Each expanded allocation is40 MiB. This needs appropriate pre-stored weight halves and performance validation; slicing the huge resident down weight inside forward would introduce a different capacity problem.
- A **true down-plus-weighted-reduction kernel** could avoid the80 MiB materialization. Existing fused reduction is only postprocessing. Preserve the BF16 down-output boundary and the current weighted-accumulation order; its [compute kernel](/home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/deepseek_moe_fast_reduce_nc_fused_compute.cpp:54) uses hardware MAC accumulation, not a separately materialized BF16 weighted product. This is a broader implementation, not the minimal diagnosis.
- Moving activation to L1, forcibly relocating it after production, or changing precision could change the allocation limit, but adds layout/copy/numerical and performance questions absent from the minimal placement experiment.
- No allocator defect, packed activation numerical bug, stale previous down tensor, leaked full expert matrix, or current GR cast/product leak is supported by the exact observed ledger. The identity of the free-hole separators and the minimum intervention that passes every gate remain hardware questions.

## Draft review and static verification

Performed: provenance/frozen-source hash comparison; exact integer tensor/page/bank accounting; current versus frozen ownership and allocation-order comparison; Python, sparse output-spec/allocation, allocator, program binary ownership, and fused-reduction reader/validation source tracing. No target modules were imported or executed.

Claims were challenged against the observations: leftover packed/slice buffers were rejected by exact accounting; pre-public largest-hole loss was rejected because packed-SiLU starts with a larger hole; dead locals were not overstated as enough to close the deficit; weighted reduction was placed after the failing allocation; setup placement was separated from runtime placement and from duplicate weight ownership. Address-specific causality and performance/capacity recovery remain explicitly unverified.

An independent xhigh review passed after checking the ledger, frozen/current ownership sequence, program configurations, workspace aliasing and peak, RM-first score conversion, shared-index bounds, and weighted-MAC arithmetic. It prompted the more precise distinction between a source-derived ledger and an observed address inventory. All report links resolve, and the runtime SHA256 remains unchanged from the packed-SiLU capture.
