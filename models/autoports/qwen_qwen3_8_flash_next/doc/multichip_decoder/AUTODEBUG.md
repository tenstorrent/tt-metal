# AutoDebug: EP4 MeshMoe short-token reduction

## Findings

**Confirmed cause:** the original `MeshMoe` passes a ROW_MAJOR combine tensor with token extent `n` into a reduction that requires a multiple of 32 tokens. Physical tile padding of the earlier input does not satisfy this contract. At this boundary, every original logical `n=1..31` is rejected; `n=32` avoids this particular check.

**Minimal repair:** keep routing and expert execution at the original `n`; explicitly zero-pad only the combine contributions, reduction weights, and reduction route IDs to 32 token rows, run `post_combine_reduce`, then slice its result back to `n`. The parent added this form of repair while this investigation was running. Its current shapes and dtypes satisfy the inspected native contracts.

**Next failure audit:** padding contributions alone next fails the weights dimension check. Padding contributions and weights while leaving short IDs creates out-of-bounds device reads: the writer unconditionally fetches 32 index pages and the validator does not check their token extent. With all three padded correctly, this inspection found **no additional inevitable native validation/runtime failure** for the stated EP4 configuration. Conditional concerns and a separate native L1 accounting discrepancy appear below; they are not promoted to explanations of the logged failure.

No hardware was run, no implementation was edited, and AutoDebug was not relaunched. Three independent xhigh source audits covered dispatch/combine, fused SwiGLU, and post-combine reduction. Only this report was written by the investigation.

## Evidence and revision boundary

- Checkout HEAD: `fb462731c5e32e50a3d2efadb8df3067d35dbaed`.
- Focus caller: `models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py`.
- Native paths below are relative to `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/`.
- `doc/multichip_decoder/mesh3.log:60,84-86` records `Number of tokens 1 must be divisible by 32 for hardware tilization`, thrown by the post-reduce factory. Here and below, `doc/` is under the model's autoport directory.
- `mesh3_provenance.json` records the same HEAD and original caller SHA256 `d0b15f08155996d7de9d2ff17bd9c0ef89f3ddf24f7866adca583bc065936b47`. The caller is untracked and was edited concurrently by the parent. A later inspected caller SHA256 was `194c6f1052c3028c9a8b61f723f3141ad0f30fbfb5d12a60ceefeac4d2811d23`; it contains reduction-only padding around lines 666-703. Line references to the caller refer to that inspected padded form unless explicitly labeled original.

Direct observation versus interpretation:

1. The log proves a host factory rejection at token count 1. Its stack enters through `prefill_forward` and `_chunk`; it is not a completed decode measurement. `mesh3.json` has `passed=false` and no result rows. The same `MeshMoe` call is also used by one-token decode, so the defect covers decode as well.
2. `baseline1.json` records a successful baseline implementation. It does not prove this native MeshMoe path passes at 32 tokens.
3. Reaching the reduction factory does not prove dispatch, fused experts, or combine finished successfully. The probe synchronizes only after the model call returns (`tests/probe_multichip_decoder.py:179-184` in the inspected tree); the exception prevents that boundary. The contracts below were checked from source, not inferred from asynchronous completion.

## Exact failure and repair boundary

`post_combine_reduce/post_combine_reduce.cpp:16-36` forwards inputs without padding or layout conversion. For this ordinary gathered path, the misleadingly named `logical_combine_shape()` actually returns `combine_output.padded_shape()` (`device/post_combine_reduce_types.hpp:39-45`). A ROW_MAJOR combine tensor nevertheless retains token extent `n`; it has no tile-height expansion of that axis.

The factory multiplies dimensions preceding `expert_dim` to obtain the token count (`device/post_combine_reduce_program_factory.cpp:39-51`). With `[1,1,n,10,2560]` and `expert_dim=3`, that count is `n`. Lines 76-82 require it to be positive and divisible by 32. The first divergent preparation is therefore the model/native boundary: `combine(seq_len_per_chip=n)` correctly returns `n` rows, but the original caller forwards those directly to an operation requiring 32-row chunks.

The complete reduction input contract is:

| Object | Before reduction-only padding | Required reduction input | Padding contents |
|---|---|---|---|
| Combined expert contributions | BF16 ROW_MAJOR `[1,1,n,10,2560]` | BF16 ROW_MAJOR `[1,1,32,10,2560]` | All contribution elements in rows `n..31` are zero |
| Routing weights | BF16 ROW_MAJOR `[1,1,n,10]` | BF16 ROW_MAJOR, interleaved `[1,1,32,10]` | Zero weights in rows `n..31` |
| Selected global expert IDs | UINT16 ROW_MAJOR `[1,1,n,10]` | UINT16 ROW_MAJOR `[1,1,32,10]` | Valid in-range ID 0 is sufficient |
| Per-rank reduction table | INT32 ROW_MAJOR `[1,512]` | Same | Local 128 IDs map to 0; remaining IDs map to -1 |
| Reduction output | — | BF16 TILE `[1,1,32,2560]` | Slice token dimension to original `n` |

Rank-five weights `[1,1,32,10,1]` are also supported, but rank-four per-token weights are explicitly accepted and avoid one page per scalar. Native validation is in `post_combine_reduce/device/post_combine_reduce_device_operation.cpp:65-119`. The inspected parent repair uses rank-four weights consistently, including the unpadded `n=32` branch.

The parent's zero workspaces have the right types and extents: contributions `[320,2560]` BF16 ROW_MAJOR, scores `[32,10]` BF16 ROW_MAJOR, and IDs `[32,10]` UINT16 ROW_MAJOR (`MeshWorkspace`, caller lines 525-545). Concatenating flattened `[n*10,2560]` contributions with `(32-n)*10` zero rows preserves token/slot order. For scores and IDs, concatenate `[n,10]` with `[32-n,10]`, then restore the leading singleton dimensions. These inputs obey concat's equal-dtype/layout/rank constraints (`ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp:152-169`); the ROW_MAJOR factory copies whole sticks and does not require token alignment.

Do not merely relabel an `n`-row allocation as 32 rows. Do not pad `x` before the router: arbitrary padded rows would acquire expert selections and inflate counts/work. No dispatch metadata, count, offset, shared-expert, or mutable attention-state padding is required by this repair. Returning the slice before adding the shared expert preserves the public logical shape.

### Deterministic next failures from incomplete padding

- **Only contributions padded:** `post_combine_reduce_device_operation.cpp:79-86` compares weights' padded dimensions with the combine shape and rejects token extent `n` versus 32. This happens before the tilization divisibility check.
- **Contributions and weights padded, IDs left at `n`:** the validator checks index allocation/layout/dtype, but not index token count (`:38-44,110-119`). `deepseek_moe_post_combine_reduce_writer.cpp:118-135` unconditionally reads pages `token_start_idx+i`, `i=0..31`. For `n=1`, 31 index page reads exceed the provided one-token tensor's declared page extent. Stale/invalid IDs can also index outside the 512-entry table. This is a concrete missing validation guard; the exact device manifestation is unverified. Complete caller padding avoids it. A native hardening change would validate index token/slot geometry against the combine shape before launch.
- **IDs changed to INT32/UINT32:** native validation rejects them. Despite stale binding documentation describing INT32, this operation requires UINT16. Its compute helper reads UINT16 directly (`compute.cpp:27-45,98`); it does not expand the buffer in place.

There is no need to change dispatch's `experts_per_chip` from 512 to 128, widen `input_m_tiles` to workspace capacity, or change `cluster_axis` to repair the short-token error. Those changes alter separate contracts.

## Routing, counts, and selected local experts

The current pipeline deliberately dispatches a local copy of every selected global expert region on each rank, computes only its owned experts, then masks nonowned contributions during reduction. The two dispatch tables serve different purposes.

- Dispatch/combine wrappers recognize `mesh=1x4`, `cluster_axis=0`, `dispatch_group_size=1` as a singleton communication domain and canonicalize to zero links and Linear topology (`dispatch/dispatch.cpp:41-64`; `combine/combine.cpp:37-61`). No cross-chip expert routing is needed here because the mixed input is replicated.
- In this domain, native dispatch requires `num_routed_experts == experts_per_chip` (`dispatch/device/dispatch_program_factory.cpp:169-178`): both are 512 for the flat global namespace. The all-zero INT32 dispatch table maps every selected global ID to the current column/device (`reader_worker_dispatch.cpp:136-141,365-387`). Substituting 128 in this call violates the local dispatch requirement and does not describe the actual 512-entry count layout.
- `moe_fused_swiglu/moe_fused_swiglu.cpp:33-40` independently derives its local expert count from the weight-list length, which is 128. Local weight `j` on rank `r` is HF expert `128*r+j`, matching `self.ids[j]` (`MeshMoe` lines 561-576). The kernel looks up `counts[global_expert_idx_table[j]]` and the same global offset (`moe_fused_swiglu_reader.cpp:564-605`). Global count length 512 and local weight-list length 128 are therefore consistent.
- Count 0 produces `m_t=0`, `m_blocks=0`; count 1 through 32 produces `m_t=1`, `m_blocks=1` with `M_BLOCK=8` and `M_EFF_MIN=1`. Reader publishes this state to compute/writer mailboxes; both consume the mailbox even for skipped experts, and their M-block loops perform no matmul for zero counts (`reader.cpp:611-627`, `compute.cpp:521-552`, `writer.cpp:267-295`). Thus padding only the reduction preserves gate-selected local expert execution. The loop still visits 128 descriptors; it does not compute 128 experts.

Counts are formed from a binary scatter of the distinct top-10 IDs, not by testing whether rounded BF16 routing weights are nonzero (`MeshMoe:616-626`). This is the correct descriptor/data pairing: dispatch consumes those same IDs. Converting the ROW_MAJOR binary mask to TILE uses zero fill for its extra token rows (`ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.hpp:20-26`; `.cpp:204-229`), so physical padding does not add selections. Counts in `[0,32]` and the aligned offsets are exact integers in FP32; multiplication by `1/32`, ceiling, and multiplication by 32 do not introduce an ambiguity at these values. UINT32 conversion preserves them.

Let `c[e]` be the actual number of selected rows for expert `e`. The region size and start are

```
size[e]  = ceil(c[e] / 32) * 32
start[e] = sum(size[j] for j < e)
```

For all `n=1..32`, `c[e] <= n <= 32` and `sum(c)=10*n`. Each active expert therefore occupies exactly one 32-row region. If `A` experts are active, the total extent is `32*A <= 32*min(512,10*n) <= 10240`. Workspace capacity is 16192 rows.

| Case | Counts | Reserved extent | Fused per-expert maximum |
|---|---|---|---|
| `n=1` | Ten count-1 entries | 320 rows, starts 0,32,...,288 in expert order | One tile row |
| `n=32`, same ten experts selected each time | Ten count-32 entries | 320 rows | One tile row |
| `n=32`, all 320 selections distinct | 320 count-1 entries | 10240 rows | One tile row |

Consequently `input_m_tiles=ceil(n/32)=1` is sufficient; this bounds an individual expert's live input height, not the full flat dispatch buffer. The aligned global offsets select each expert's region. Reader applies the row-major start in token rows; writer divides the same start by 32 for its TILE output base (`reader.cpp:583-631`; `writer.cpp:286-290`).

## Dispatch metadata and combine ledger

| Stage | Data and ownership |
|---|---|
| Dispatch input | BF16 TILE `[1,n,2560]`; UINT16 ROW_MAJOR IDs `[1,n,10]`; UINT32 ROW_MAJOR offsets `[1,512]` |
| Dispatch output | BF16 ROW_MAJOR `[1,1,16192,2560]`; INT32 ROW_MAJOR metadata `[1,1,16192,3]` |
| Metadata for each live dispatched row | `[linearized source device, original token index, original top-k slot]` |
| Fused expert output | BF16 TILE `[16192,2560]`; only selected local expert regions are updated |
| Combine output | BF16 ROW_MAJOR `[1,1,n,10,2560]`, restoring each original token/slot |

Dispatch output specs fix the data/metadata dtypes and ROW_MAJOR layout (`dispatch_device_operation.cpp:169-189`). Fused SwiGLU explicitly accepts this BF16 ROW_MAJOR input, requires UINT32 ROW_MAJOR count/ID vectors, and derives BF16 output dtype from the supplied output tensor (`moe_fused_swiglu_device_operation.cpp:93-106,188-206,335-373`; public wrapper `moe_fused_swiglu.cpp:68-78`). The supplied compute config clears its three forbidden flags (`device_operation.cpp:300-309`).

For `n=1` and `n=32`, tiled dispatch has one 32-row input batch. Its route planner limits entries to `batch_count=n`; unreferenced padded input rows are not dispatched (`dispatch/.../reader_worker_dispatch.cpp:258-262,317-320,358-365`). Offsets are loaded into local scratch and incremented there (`:181-202,382-387`), leaving the caller's region-start tensor intact. Writer stores the three metadata words beside the matching payload page (`writer_worker_dispatch.cpp:430-437`).

Combine's local `counter_offset` is 0, including ranks 1-3 (`combine/device/combine_program_factory.cpp:824-831`). It scans the same global 512 counts. The TILE path reconstructs starts by the same aligned prefix (`reader_untilize.cpp:160-178,287-290`); it then uses only the real `expert_tokens` when consuming rows. Producer and consumer use full batch-sized CB pushes/pops with a separate `batch_count` for live rows (`writer_untilize.cpp:247-269`). Metadata restores `output_page = token_index*10 + topk_slot` (`:269-285`). Neither top-k sorting nor numeric weight ordering is required; IDs, slots, and weights remain paired.

Combine can copy stale nonowned expert output regions, since its counts cover all 512 IDs. The per-rank reduction table is essential: reduction checks only `table[expert_id] != -1`, not equality to the physical chip number (`post_combine_reduce/.../compute.cpp:97-120`; writer `:154-165`). Local entries 0 are correct on every rank. Changing the reduction table to all zero would count nonowned regions. Changing the dispatch table to the ownership mask without changing counts/layout would break the existing metadata/count agreement.

## Post-reduce kernel boundary cases

For the repaired shape: `num_tokens=32`, `K=10`, `H=2560`. The factory schedules one chunk on one core. It computes `emb_dim_cb_tiles=ceil(2560/1024)=3` and `emb_dim_out_tiles=2560/32=80`, within its eight-DST-tile limit. Both row batching and weight batching take their supported streaming fallbacks: `2*10*3*2048=122880 > 65536` and `2*10*2048=40960 > 32768` (`program_factory.cpp:57-82,115-138`).

The resulting CB ledger is coherent:

- Reader reserves/fills/pushes 3 pages per token/slot, 320 slots total; compute waits/pops 3 per slot, including skipped experts.
- Writer reserves/pushes one weight tile per slot; compute waits/pops one per slot, including skipped experts.
- Writer loads/pushes 32 index pages; compute waits once and pops 32 after the chunk.
- Compute reserves and produces 96 row-major scratch pages, tilizes 96 pages into the output CB; writer waits/pops all 96 and writes the 80 real tiles.

See reader `:75-99`, compute `:73-157`, writer `:117-148,172-235`. Padded rows contain zero contributions and weights. ID 0 is valid: rank 0 processes zero-valued slots as local; ranks 1-3 take the no-local fallback with a zero final weight. Either yields zero for explicitly zero contributions, without an out-of-range sentinel.

**Embedding 2560 is not a geometry defect.** Each scratch row has 3072 BF16 elements, with its first 2560 valid. `pack_tile<true>` selects scratch page `i*3+j`; it does not compact the row. `tilize<96>(1)` views the scratch as one 32-row matrix 3072 columns wide, emitting 96 horizontal tiles. Writing its first 80 tiles retains columns 0..2559 for **every** token, discarding padded columns rather than token rows. The relevant helper contract is `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:81,96-100`. Tests also exercise non-1024 widths such as 2880. The structured-test xfail for 2880 explicitly concerns its hardcoded test-data reshape, not native width support (`test_deepseek_moe_post_combine_reduce.py:56-64`).

## Other Potential Issues and limitations

### Native L1 estimate omits actual auxiliary table sizes

`moe_fused_swiglu/device/moe_fused_swiglu_geometry.cpp:441-446` estimates L1 using `cb_allocations(...,64,64,...)` for the index and count scratch allocations. The program factory computes the actual index-page and count/offset scratch sizes separately, but its final `l1_need` check still calls that nominal estimator (`moe_fused_swiglu_program_factory.cpp:418-439`). For this EP4 call:

- Index scratch: 128 UINT32 entries = 512 bytes, rather than 64.
- Count scratch including direct-write offsets: two 512-entry UINT32 pages = 4096 bytes, rather than 64.
- Actual CB footprint exceeds that estimator by `(512-64)+(4096-64)=4480` bytes per core.

These two scratch buffers are independent allocations, outside the alias groups (`geometry.cpp:336-337,401-409`), so this is a concrete accounting defect. `L1_CB_RESERVE=70656` is not an extra scratch allowance: it represents the kernel-config ring already below the default allocator base (`tt_metal/hal.cpp:49-53`; `hybrid_routed_expert_ffn/device/hybrid_program_factory.cpp:2988-2995`). Metal later allocates and validates the actual CB region (`tt_metal/impl/program/program.cpp:3561-3565`), rejecting L1 overflow or overlap (`:2638-2645,2695-2702`). Thus the undercount can defeat near-capacity blocking choices or move an error to the later allocator; it is not evidence of unchecked memory corruption.

It does **not** establish a next failure for this call: padding the reduction leaves fused-expert allocations unchanged, and the logged run got beyond their host setup without a recorded allocation error. The minimal native correction is to sum the same actual `cb_allocations(...,idx_page,counts_scratch_bytes,...)` used to build the descriptor for the final footprint check; making earlier budget-based blocking choices auxiliary-aware would close the corresponding planner gap. Do not attribute the token-divisibility exception to this defect.

### All-nonlocal fallback needs finite data

When no selected expert belongs to a rank, the reducer forces the final slot through with weight zero to initialize its accumulator (`compute.cpp:103-108`; writer `:177-187`). Multiplication by zero does not sanitize NaN/Inf. Current expert output starts zeroed, and combine defaults to `init_zeros=true`; finite live computations support the intended finite-data invariant. Nevertheless, an earlier nonfinite result in a reused nonowned region could propagate through this fallback. No supplied observation proves that precondition. Explicitly zero padded contribution rows, as in the current repair, avoid adding this risk for artificial rows.

### Refuted or unproven suspects

- UINT16 indices are not expanded in place; the current writer and compute keep them UINT16. The old validation comment and binding doc are stale.
- A concern about the per-token writer's two UINT16 L1 stores is unproven. They occur at aligned tile offsets 0 and 2 within one word/16-byte region; the dispatch-plan warning about subword stores does not demonstrate corruption at those addresses. A sibling explicitly supports halfword copies (`ttnn/cpp/ttnn/operations/data_movement/reshape_view/codegen/kernels/riscv_copy.h:16-34`). This is not a justified next-failure diagnosis.
- The source does not establish a route-axis mismatch: this caller intentionally uses local dispatch plus per-rank expert ownership and final cross-chip summation.
- No claim of end-to-end numerical correctness, device completion, or trace/cache safety follows from passing these host contracts.

## Static checks and focused follow-up

After the first draft, the investigation rechecked the headline claim against the actual shape helper and validators, instantiated the K10/H2560 CB and tilize geometry, and reviewed each suspected second failure against its triggering conditions. The non-1024 width and halfword-store suspicions were rejected as headline findings. The L1 undercount was independently confirmed but retained as a separate capacity issue because it cannot explain this exception or a change caused solely by reduction padding.

A host-only Python check exercised all `n=1..32` with six routing patterns each (192 cases): concentrated picks on each rank, spread picks, and mixed picks. It checked unique top-10 IDs, count totals, count bounds, 32-row offset alignment, capacity bounds, one-tile per-expert heights, and a unique EP owner for every selection. All checks passed; it imported neither TTNN nor a device runtime. This validates the arithmetic model, not the native kernels.

Nearby tests leave the exact production case uncovered. `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_dispatch_combine_local.py:20-24,113-131` covers local 1x1/1x4 dispatch/combine with H2560, but E64/K6/n128 and no fused experts/reduction. `tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_post_combine_reduce_row_index.py:34-37,70-98` compares per-page and per-token weights with K4/K8 and H2880/H7168. The direct fused-SwiGLU test constructs one local expert (`test_moe_fused_swiglu.py:185-205`). Their presence is contract evidence, not a claim that they were run in this investigation.

After implementation, the smallest useful hardware matrix is original logical `n=1,2,31,32`, then all other `n=1..32`; include changing routes on consecutive cached calls, picks crossing IDs 127/128,255/256,383/384,511, and tokens with no local expert on at least one rank. Include a public 33-token prefill split into 32+1 and ordinary decode. Check output shape/finite values and per-rank selected expert contributions against a reference, then synchronize/read outputs. Run this through the parent's hardware workflow; this report does not claim those checks passed.
