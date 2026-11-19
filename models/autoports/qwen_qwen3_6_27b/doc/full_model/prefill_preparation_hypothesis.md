# Prefill preparation hypothesis investigation

Scope: source-only AutoFix investigation of `doc/full_model/AUTODEBUG_sampling.md` section 4. No TTNN import, device access, tests, or implementation edits were performed. Paths below are relative to the checkout; `Q/` means `models/autoports/qwen_qwen3_6_27b/`.

## Finding

The diagnosis is supported. Preparing only decode/sample/reset does not bound later prefill signatures. Merely warming a list of prompt lengths is insufficient, even if that list includes the acceptance prompts. Absolute starts, whole-prompt embeddings/terminal shapes, variable slices and concatenations, and exact logical-row CCL persistence all introduce additional signatures.

A bounded design is available, but it requires a prefill scheduler and state-lifetime repair, not an extra warmup loop. **Use bounded physical prefill chunks, runtime SDPA positions, a stable fixed-shape fill-page-table view, and live FP32 GDN state spanning the entire public prefill call.** Prepare every actual physical variant and all per-layer persistent CCL resources before the first decode capture. Retain the accepted projections, precision and residual layout. The existing native flexible SDPA is the crucial available primitive; a new attention kernel is not required for aligned chunks.

No hardware claim is made here. The precise bucket choice and extra padding/masking work require the owner's reduced probe and timing comparison.

## Source-proven causes

- `Q/tt/model.py:130-144` embeds and rotates the entire requested length, calls every layer, and applies terminal work to the entire result or an exact last-token slice. Those operations depend on the whole logical length, even though each decoder later chunks internally.
- `Q/tt/flat_prefill.py:171-216` splits a logical length into 512-token pieces, but slices from the whole input, builds rotations with variable slice bounds, and concatenates a variable number of outputs. Full attention uses one-token work until `start_pos` is aligned to `_attention_chunk`, normally 512.
- `Q/tt/optimized_decoder.py:1383-1413` uses `state.page_table[:, start_pos // 32:]` and scalar `chunk_start_idx=start_pos`. Both the sliced table shape and scalar SDPA key vary with absolute position.
- `Q/tt/multichip_decoder.py:828-858` has the same shrinking-table issue on its ordered cache-fill path. `batched_fill_pair` is not selected for B1 or captured prefill.
- `Q/tt/multichip_decoder.py:446-513` retains exact logical-shape CCL outputs for row counts <=32. Prewarming row 1 and row 32 leaves row 2 through 31 missing. These resources are layer-owned, so compiling one representative layer does not allocate the corresponding buffers in other layers.
- `Q/tt/generator.py` currently creates row-state slices and prefill token/position tensors in `prefill_forward`; standalone `prepare()` does not prepare these paths. Slot slicing and `slice_write` also need bounded preparation for every configured fixed slot.

## Existing primitives and restrictions

| Need | Source-supported primitive | Important constraint |
| --- | --- | --- |
| Absolute SDPA position without a new program per position | `chunked_scaled_dot_product_attention(..., chunk_start_idx_tensor=stable_int32_scalar)` | `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp:267-310,520-585` dispatches the flexible path. Start must still be a multiple of both q and k chunk sizes. This does **not** eliminate the unaligned-resume path. Keep k chunk 512; changing it to 32 would change the accepted numerical policy. |
| KV fill at a different absolute page without shrinking shapes | Refresh a fixed-shape table containing the physical pages for the current chunk, then ordinary `paged_fill_cache` | Fill has no absolute-start tensor. Its table values are runtime data. A B1 chunk of <=512 needs 16 page entries; use owned valid pages in every padded entry. Keep the full fixed-capacity table separately for SDPA. |
| Paged fill dtype | Explicit cast to the selected cache dtype before fill | `paged_cache_nanobind.cpp:117-147`: raw tile copy. Decode update accepts BF16/FP32 and repacks. Do not collapse this dtype split. |
| Logical GDN convolution length with padded physical storage | Existing KDA convolution `actual_end` tensor | Native chronology reads device `actual_start`/`actual_end` (`qkv_causal_conv1d_silu/device/kernels/dataflow/qkv_conv_dataflow.hpp:118-130`); final carry is formed from real final tokens (`history_exchange_qkv_causal_conv1d_silu.cpp:95-118`). |
| Dynamic last-token selection | `ttnn.gather` with fixed-shape UINT32 index, then terminal/head | `data_movement/gather/gather_nanobind.cpp:17-78`: same-rank, same-layout indices; no broadcast. Gather the local 1280-wide residual before the LM head, not a full vocabulary. Probe the precise shape first. |
| Dynamic arbitrary slice | Not provided by ordinary `ttnn.slice` | `slice_nanobind.cpp:51-56` takes integer lists; use bounded local slices or tensor-index gather. `slice_write` also takes integer bounds. |

A fixed fill table can be uploaded by the public prefill driver between chunks. This is scheduling metadata already owned by that driver, not host computation of model activations, and does not require per-token decode page refresh. Allocate the table before capture and update its contents; never replace its backing allocation. For already full-capacity external tables, derive the chunk table from the caller's mapping, not an assumed contiguous physical allocation.

## Smallest conservative implementation route

1. Give `Model` a device-only **one bounded chunk** entrypoint. Its input embedding/rotary and terminal shapes must be bounded too; moving only the layer loop is insufficient. Preserve `[B,S,1280]` residual ownership across the stack and all accepted `_norm`, `_linear`, `_gdn`, `_finish`, projection/correction, and collective implementations.
2. Allocate persistent FP32 prefill recurrent state for each GDN layer/slot before capture. At the start of one public prefill/resume call, convert the caller's BF16 state into it. Run every internal chunk against that FP32 state; publish BF16 once at the end. Convolution carry remains BF16 and advances per real chunk. This directly preserves `flat_prefill.py`'s live-state/publication distinction.
3. Initially allow exact logical chunk sizes 1..512 rather than inventing padding semantics. This is a **complete bounded repertoire**, independent of 262144 context, not a list of known test lengths. It preserves existing arithmetic and projection branches most directly. Compile the actual helper signatures once; allocate layer-owned CCL entries for every reachable <=32-row signature in every layer before capture. The complete shape warmup may be too costly: measure setup time, then use the padded-bucket reduction below. Do not claim the exact-size scheme is cheap.
4. Remove absolute-start signatures: flexible SDPA scalar tensor; fixed 16-entry fill table; persistent current-position input for the existing unaligned attention decode helper instead of `moreh_full(absolute, ...)`; bounded local input/rotary slices only. Whole-prompt device concatenation must disappear. Return host diagnostic logits chunk by chunk at the explicit diagnostic boundary, and retain only a bounded device output for token-out mode.
5. For normal start 0, chunk sizes are 512 plus one remainder. For nonzero resume, first handle the prefix through the next 512 boundary, then aligned 512 chunks and a tail. **Do not implement the prefix by feeding the entire stack one token at a time.** That would replace fast GDN projection/recurrence prefill with decode-like work for as many as511 tokens. The full-attention layer may retain its inherited one-token unaligned attention behavior, while GDN processes the head segment as a bulk prefill. This changes the GDN internal chunk partition relative to the old layer-major wrapper, so compare boundary logits and resumed state against the original wrapper before adopting it. Keep FP32 state live throughout.
6. For that unaligned full-attention head, avoid variable growing concatenations. A bounded head output buffer plus bounded local writes is viable; warm all local scalar slice/write indices (0..511), or use runtime gather/scatter after proving exact shapes. None of those indices need to encode the 262K absolute offset. This is also why blindly calling existing `flat_prefill_forward` from a new outer loop is not sufficient.
7. The generator performs all preparation before captures, restores KV/recurrent/RNG inputs, and makes future unsupported **physical** signatures an implementation error. Logical valid lengths must always map to an existing physical signature. Keep trace IDs and backing state unchanged across requests.

The conservative repertoire proves finiteness but does not automatically meet setup/performance requirements. In particular, warming every head-length path by executing its full token loop would repeat excessive work: warm its one-token compute, bounded slice/write helpers, and final output view signatures separately. Do not warm 512 complete unaligned full-model runs.

## Reducing the repertoire without losing optimized projection families

A small bucket family such as one-token plus physical 32/128/256/512 rows is plausible; 128 must remain represented for the primary TTFT workload, and512 retains accepted long prefill projection geometry. Choosing only32 rows would discard the accepted512-row prefill optimization. Choosing only512 inflates primary128-token work.

Padding alone is **incorrect** for GDN. `Q/tt/multichip_decoder.py:1122-1286` shows why:

- Convolution history must end at the logical final token, not the padded final token. For the B1 native convolution path, bind a persistent runtime `actual_end=logical_count` instead of selecting `conv_ends[s]` from the physical length. Start is local zero; absolute RoPE/cache position is a separate input.
- Compute beta/decay normally, then set **both beta and g to exactly zero for invalid rows** before `chunk_gated_delta_rule`. Zero g gives multiplicative decay1, and zero beta gives no update. Padding token IDs alone leaves sigmoid(beta)>0 and negative decay, corrupting state.
- Also zero invalid convolution outputs (at least q/k/v, especially q/k before normalization) or otherwise prove finite neutral contributions. Keep valid rows and all accepted arithmetic unchanged. The existing short-tail code already pads beta/g with zeros; generalize that semantic rule to runtime masks.
- Keep state FP32 across all internal chunks and BF16 publication only at the public call boundary. Repeatedly calling the existing `prefill_forward` on BF16 state would insert new rounding boundaries.
- Batched optimized tiled/flat convolution embeds logical `S` in kernel geometry (`Q/tt/tiled_prefill_conv_candidate.py`, reader uses compile-time S for carry extraction). It does **not** acquire dynamic-end correctness merely by changing the B1 native call. Either extend the accepted batched convolution reader to runtime valid lengths/carry positions, or retain exact bounded sizes for that path until it is proven. Do not silently route accepted batched prefill through a slower convolution.

For aligned full attention, causal masking means trailing padded Q/K/V cannot influence real earlier rows. Padded cache writes must remain within owned capacity, and the next real continuation must overwrite padded entries before reading them. At the context edge select the smallest physical bucket that fits the rounded cache allocation. Keep the full table padded to the attention rounded read window, as already required. Inactive/future rows must not be published as valid.

Prefer dynamic last-hidden gather plus a one-row terminal path for token-out. For all-logit diagnostics, evaluate the physical chunk and slice on the host after the explicit readback; this avoids introducing logical-length terminal/slice programs. The public API concatenates/slices logical outputs, never exposes padding.

## Rejected shortcuts

- Warm only33/128/512 or the current reference length: leaves valid2-token, different-slot, tail and resume signatures uncovered.
- Warm only decoder layers while whole-model embedding/rope/head still use arbitrary lengths: does not bound the actual generator graph.
- Change k chunk512 to32 to admit more starts: changes the accepted SDPA numerical policy.
- Global token-by-token prefill for arbitrary lengths: discards optimized prefill projection families and makes max context impractical.
- Outer chunk loop using existing BF16 recurrent state: rounds after every chunk and violates the inherited FP32 live-state policy.
- Page-table shrinking slices, scalar absolute SDPA starts, or `moreh_full` absolute position construction in the runtime head: preserves absolute-position program proliferation.
- Broad tracker exclusions or decode trace retirement: hide the lifetime error rather than preparing the generator.

## Discriminating owner checks

Use one real layer of each kind first, with the real terminal and cache shapes. No hardware checks were run by this investigator.

1. Record program-cache count and every layer's CCL-buffer key set after complete preparation. Replay decode; prefill2,17,33,127,128,129,511,512,513,1025; replay after each. Counts and persistent key sets must not grow.
2. Resume at1,31,32,127,511,512,513 using real prefix state, changing row order/fixed slot. Test both same and changed physical page mappings. Compare against uninterrupted logits and each GDN state, not merely sampled text.
3. Use the runtime SDPA start tensor at0,512,1024 with identical physical Q/cache/table shapes; prove cache count remains fixed and output changes correctly. This directly discriminates absolute-offset key removal from accidental reuse at only one start.
4. Compare exact-tail and padded-bucket GDN at2,31,33,127,129,511 with nonzero initial state and convolution carry. Verify logical final state and history; poison padded token IDs to prove invalid rows cannot alter real outputs/state.
5. With allocation tracking and program-cache tracking enabled, interleave prepared eager prefill, reset and decode/sample traces, ending in decode replay. Include existing captures before changing requests. Do not restart counters or generator.
6. Benchmark128 and512 prefill after the bounded change. Added scheduling must preserve the accepted bulk projection paths; a successful tracker run does not establish preserved TTFT.

Conclusion: the bounded-preparation hypothesis is supported by existing native interfaces. The current wrapper cannot satisfy it unchanged. The minimal repair is a bounded model-level chunk scheduler plus runtime absolute position/page mapping and explicit FP32 prefill-state lifetime. A small padded repertoire needs the stated GDN mask/carry work; it is not safe to implement by padding IDs alone.


## Implementation handoff

Implemented `tt/bounded_prefill.py` and `Model.allocate_prefill(...)`. No accepted decoder file or generator file was edited by this investigator.

- Construct a B1 helper and, for configured batch >1, a second helper with `prefill_batch_size=configured_batch`. Both bind the same caller-owned cache. Full-batch inputs are supplied in fixed-slot order and share logical start/end. Mixed lengths/starts use the B1 helper on selected slots.
- Call both helpers' `prepare()` before any capture. They compile all1..512 physical lengths and eager unaligned-head helpers, using representative layer-policy/tensor-spec groups. Each non-representative layer receives its own equivalent persistent CCL buffers; no layer shares scratch backing storage with another. The owner's strict tracked full-stack replay must validate that native program signatures are shared as expected.
- Preparation changes cache state. Run the generator's prepared reset and restore token/position/RNG inputs before capture. Helpers import BF16 state when each request begins; they keep FP32 work state until all chunks finish.
- `helper.run(tokens_segment_host, slot=slot, start_pos=absolute_start, page_table=slot_order_page_host, return_all_logits=...)` yields `(absolute_start, logical_count, device_logits)`. Consume the iterator fully, including after its final yield, so BF16 state publication occurs. `False` emits only final last-token logits; `True` emits each bounded chunk for host diagnostic readback/concatenation.
- `_diagnostic_lengths=[...]` is available only on `prepare` for fast reduced probes. Runtime rejects a chunk not in that list. This is not acceptance preparation or a public context restriction.
- B>1 GDN directly calls the accepted `flat_prefill._flat_chunk`. B>1 full attention retains compact projections, `_finish_flat`, original geometry/dtypes, and bulk batched KV fill when host page ownership is disjoint; it warms the inherited ordered fallback too. B1 follows the accepted `_legacy_chunk` operation sequence. The only attention changes are fixed fill-table shape and the native runtime SDPA position scalar.
- Source inspection confirmed `ConcatDeviceOperation::compute_program_hash` includes its input-alias partition (`concat_device_operation.cpp:102-135`). Warmup now uses distinct cloned singleton inputs, matching actual unaligned-head outputs, rather than repeated references to one tensor.
- Per-slot publication stages tiled recurrent state through temporary ROW_MAJOR storage and copies the converted result into the original TILE tensor. It does not pass a persistent trace tensor to `experimental.slice_write`'s storage-rebinding TILE branch (`slice_write.cpp:139-148`). Convolution history is already ROW_MAJOR.

CPU checks run: `python_env/bin/python -m black --target-version py310 .../tt/bounded_prefill.py`; `python_env/bin/python -m py_compile .../tt/bounded_prefill.py .../tt/model.py`; AST-extracted `chunk_intervals` arithmetic over starts0,1,31,32,127,511,512,513,262013 and short/long/end262144 intervals. All passed. No device tests or performance measurements were run.

Recommended first device probe: reduced layers `[0,3]`, B1, capacity1024, diagnostic lengths1,2,17,33,128,512. Compare ordinary prefill33 and513 to the old wrapper at matching positions. Exercise resume31->33 using the corresponding prefix cache and verify both GDN state and final logits. Then the same configured B2 with uniform33 and two B1 slot33 calls; compare logits and K/V/state slot content. Record all persistent cache buffer addresses before/after slot publication. The next run should use full preparation and tracked generator decode before/after changed lengths and changed page mappings, verifying program-cache counts remain fixed. A final all-layer run is still required after the reduced probe succeeds.


### Ready-to-run reduced probe

`tests/probe_bounded_prefill.py` now implements the native-versus-public comparison with real layers0/3, capacity2048, lengths33/128/513, and resumed31/512 continuations through545. It compares matching native/public BF16 publication boundaries for resumed cases, reports logits PCC and top1/top5/top100 agreement, compares every GDN convolution/recurrent state, and checks persistent cache buffer addresses. No trace is captured. `--batch` uses the configured uniform path (default1). Partial JSON is written after each case and on exceptions.

Suggested owner commands (not executed by this investigator):

```bash
timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_bounded_prefill --output models/autoports/qwen_qwen3_6_27b/doc/full_model/bounded_prefill_b1.json
timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_bounded_prefill --batch 2 --output models/autoports/qwen_qwen3_6_27b/doc/full_model/bounded_prefill_b2.json
```

Run using the established bringup environment exports. CPU metric checks exercised identical tensors, nonfinite rejection, and shape mismatch rejection; all passed. Syntax and Black checks passed.
