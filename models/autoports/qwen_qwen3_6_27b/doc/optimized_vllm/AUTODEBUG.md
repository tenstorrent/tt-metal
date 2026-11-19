# AutoDebug: unaligned continuation serializes complete attention layers

## Verdict and evidence limits

**The performance defect is confirmed by source: a continuation not aligned to 512 runs complete full-attention layers, including their MLPs, token by token until the next 512 boundary.** Each such token also makes three host-to-device metadata copies per full-attention layer. The first continuation at position 481 has only **31** fallback tokens; subsequent scheduler offsets can make this much worse. There is no evidence here that the existing fallback corrupts state.

The smallest promising repair is to peel only to the next **32-token cache-tile boundary**, then reuse bounded vector prefill with its existing runtime start tensor and **unchanged 512-key attention blocking**. This requires a focused correctness gate: current kernels appear to support a 32-aligned start inside a 512-key block, but the Python binding still documents 512 alignment for tensor starts. Do not treat missing host validation as proof of support. A conservative alternative that stays inside the existing attention contract is described below.

Inspection base: `d67807951e5411a4c1a79144da81b7fbc7852a43`, matching `stage_start.json`. I read Python, C++, device kernels, and tests, and ran only AST/config inspection and independent integer arithmetic. **No model imports/execution, device queries, resets, serving requests, profiling, or source edits were performed.** This report is the only authored artifact. The baseline remains owned by the other session.

No new latency, throughput, accuracy, or hardware-completion measurements were obtained. Counts below are source-derived Python operation counts, not measured device-kernel counts or speedup estimates. The report's scheduler example assumes the stated 512 budget and 31 one-token decoders; it is not an independently observed scheduler trace.

## 1. Complete causal chain

Relevant model files are under `models/autoports/qwen_qwen3_6_27b/tt/`.

1. `generator_vllm.py:157–179` passes scheduler starts and ends to `QwenGenerator.prefill_forward` without rounding them.
2. `generator.py:815–827,857–880` preserves each exact logical interval and chooses the single-row engine unless rows qualify for uniform full-batch execution. Thus a 32-slot serving configuration can still use **physical prefill batch 1** for one continuing prompt.
3. `bounded_prefill.py:20–25,403` splits at the next multiple of 512. Its initial interval has `H = min(public_length, (-start) % 512)` tokens when `start % 512 != 0`; later intervals begin aligned.
4. `bounded_prefill.py:430–451` sends each full-attention layer in that initial interval to `_attention_head`. GDN layers still process the whole interval through `_layer_chunk`.
5. `_attention_head`, lines 245–256, loops over all `H` tokens. Each iteration writes `row_index`, `head_positions`, and `single_position` through `_write` (`from_torch` plus `copy_host_to_device_tensor`, lines 120–127), gathers one hidden row, gathers its RoPE, and calls `_attention_head_step`.
6. `_attention_head_step`, lines 238–243, performs input normalization, `_attention_decode`, and **`_finish(..., prefill=True)`**. `_finish` includes attention reduction/residual, post-attention norm, SwiGLU/gate/up work, down projection, and the second reduction/residual (`multichip_decoder.py:745–819`). This is complete layer work per token, not just an unavoidable sequential cache update.
7. Under compact attention, `_attention_decode` selects `_local_compact_attention_decode` (`multichip_decoder.py:1115–1117`). Its path includes QKVG projection, Q/K norm and RoPE, one fused K/V update, one paged decode SDPA, gate, and output projection (`975–1113`). The noncompact branch also has one update and decode SDPA per token (`1118–1167`).

The HF config has 64 layers: 48 GDN and **16 full-attention** layers. Therefore one public call with fallback length `H` incurs `16*H` full-attention/MLP invocations and `48*H` explicit head metadata copies, in addition to ordinary chunk uploads and the work of other layers. The row-index payload alone is logically `B*1*1280*4` bytes per token/layer, replicated onto the mesh; it is not merely a scalar position upload. Actual transfer sizes and elapsed cost remain unmeasured.

An independent arithmetic check of repeated 481-token scheduler segments gives:

| Logical interval | Current fallback tokens per FA layer | With a 32-token peel | Current head metadata copies, 16 FA layers |
|---|---:|---:|---:|
| 0 → 481 | 0 | 0 | 0 |
| 481 → 962 | 31 | 31 | 1,488 |
| 962 → 1443 | 62 | 30 | 2,976 |
| 1443 → 1924 | 93 | 29 | 4,464 |
| 7215 → 7696 | 465 | 17 | 22,320 |
| 7696 → 8177 | 481 | 16 | 23,088 |
| 8177 → 8658 | 15 | 15 | 720 |

This predicts offset-dependent serving stalls even when the scheduler token budget is constant. A start of 31 with enough remaining tokens produces 481 fallback tokens; a start of 1 can produce 511. Conversely, aligned initial prefill avoids this fallback. The first start481 case alone cannot demonstrate the proposed improvement: it already reaches the next tile and the next 512 boundary simultaneously.

No explicit synchronization or device readback appears in `_attention_head` or `run`; their operations submit device work. Timing Python return alone cannot establish device latency. Normal host logit reads, token sampling readback, or an explicit synchronization establish later completion boundaries. Preserve queue ordering between cache update and SDPA, between internal GDN chunks, and between publication and a following decode.

## 2. Lowered cache and SDPA contracts

Paths in this section are repository relative.

### Cache fill is a tile copy, not an arbitrary token-span write

`ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache.cpp:69–92` exposes no starting-token argument for `paged_fill_cache`. Optional valid-length and modulo controls do not turn it into an arbitrary-start span update. The fill factory derives work from the input's **padded** sequence extent (`device/fill_cache/paged_fill_cache_program_factory.cpp:136–153`). Its writer maps sequence tiles through the page table and writes full tiles (`device/kernels/dataflow/writer_fill_cache_interleaved.cpp:214–226,276–294`).

For the actual cache shape `[physical_pages, 1, 32, 256]`, shifting the page table changes the origin by multiples of **32**. At start481, using `page_table[:, 481//32:]` and filling from the first input row would write that row at480 and overwrite existing prefix data. Padding the entire logical sequence and blindly filling it does not preserve a partial first tile.

The fill writer explicitly drains and skips a `-1` destination (`writer_fill_cache_interleaved.cpp:279–282`). This permits fixed-width **fill-only** tables with unused trailing entries set to `-1`.

### Update supports an arbitrary single row, not a general contiguous sequence

`paged_update_cache` associates update indices with the input batch (`device/update_cache/paged_update_cache_device_operation.cpp:240–257`). Its writer computes both the destination tile and byte offset within that tile, patches a row into an untilized cache tile, then retilizes/writes it (`device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp:98–115,135–192`).

The production model calls **fused** K/V update. The tiled fused factory is selected for tiled inputs (`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:16–30`), and its writer implements the same row-level read/modify/write (`device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp:108–127,143–175`). This explains why the current serial head preserves earlier rows of a shared tile.

Do not reinterpret successive tokens of one user as independent update-batch entries writing the same tile: their read/modify/write workers can conflict, and page-table batch ownership is a different axis. That is not a safe one-call vector replacement.

### A tensor start is runtime data, but is not a token-remainder implementation

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:234–251` validates the tensor's dtype, shape, and storage. Checks requiring start divisibility by Q and K chunk sizes are inside the **legacy scalar** branch (`388–404`). The tensor path does not inspect its value on the host.

Nevertheless, all three kernels divide the runtime value by `q_chunk_size`:

- reader: `device/kernels/dataflow/reader_interleaved.cpp:350–374`;
- compute: `device/kernels/compute/sdpa.cpp:98–108`;
- writer: `device/kernels/dataflow/writer_interleaved.cpp:192–204`.

Production bounded prefill uses `q_chunk_size=32`, `k_chunk_size=512` (`bounded_prefill.py:190–202`; `optimized_decoder.py:1251–1258`). Passing481 directly would position the first query at480. Therefore changing only the fallback branch to vector prefill is incorrect.

For a **32-aligned** start, current source is more encouraging: reader absolute Q positioning and its causal K-loop bounds (`reader_interleaved.cpp:575–583,662–672`) match standard compute (`compute_common.hpp:1713–1753`) and streaming compute (`compute_streaming.hpp:2292–2296`). Diagonal masks use the query's tile position within the key block (`compute_common.hpp:1300–1347`); streaming compute also limits active K columns (`compute_streaming.hpp:2445–2454`). For starts32,480,992 with Q32/K512, the absolute query-tile indices are1,15,31, with first K-loop counts1,1,2 respectively. No remainder is lost and the three stages agree on those counts.

The actual model passes `layer.ckc`, with HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`, and `packer_l1_acc=True` (`multichip_decoder.py:184–190`). The SDPA factory therefore selects **standard compute**, not streaming (`sdpa_program_factory.cpp:80–82,1398`); causal lightweight masks remain enabled (`1405–1408`). Paged/chunked calls disable K/V chains (`1421–1428`, `kv_chain_mode_for:248–250`). Thus standard compute plus the paged interleaved reader/writer is the primary contract traced here; streaming arithmetic is only a secondary consistency check. The projection weight policy's LoFi settings do not change this attention compute config.

**Unresolved contract discrepancy:** `sdpa_nanobind.cpp:549–552` still explicitly requires both Q- and K-chunk alignment for **tensor** starts, citing issue35225. The current arithmetic supports a source hypothesis that this restriction is stale for this path; it does not prove Blackhole correctness. Existing chunked tests iterate prefill chunks that are K-aligned (`tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py:263–312,649–678`), so their cache/trace assertions do not adjudicate the proposed new boundary. A passing focused reduced op test is required before using this relaxation.

Keep the attention table separate from the fill table. SDPA page mapping has no `-1` skip handling (`device/kernels/dataflow/dataflow_common.hpp:53–72,281–289`). Also, flexible SDPA derives compile-time K extent as `Q.logical_seq + page_table_capacity` (`sdpa_program_factory.cpp:1228–1258`); its reader's runtime bound is based on start plus Q tiles (`reader_interleaved.cpp:408–417`), not an independent actual-capacity check. Preserve the real rounded-to32 Q extent. For example, start2016, length32, capacity2048 needs Q32, not an unnecessarily expanded Q512 that can access beyond the page table.

## 3. Concrete minimal repair candidate

**First test, then implement the 32-token peel in `bounded_prefill.py`; leave decoder implementation and precision configuration unchanged.** Shape-dependent kernel selection and rounding can still change, so this does not promise bitwise arithmetic equivalence. This is the smallest code intervention supported by the present source, conditional on resolving the documented SDPA restriction above.

Use separate notions of tile eligibility and the existing 512 maximum interval:

```text
while position < end:
    if position % 32:
        count = min(end - position, 32 - position % 32)
        full-attention path = existing exact token head
    else:
        count = min(end - position, 512 - position % 512)
        full-attention path = existing vector _layer_chunk
    run the exact count through every layer
    position += count
```

Required details, all part of the repair rather than optional cleanup:

- Change both interval selection and the predicates controlling fill-table setup / `_attention_head` (`bounded_prefill.py:403,430–451`). Changing only `chunk_intervals` would still select the slow path at32,64,… . Retain the 512 maximum shape and split vector bodies at512 boundaries.
- Refresh `self.start` with the **true** 32-aligned position, using the existing device tensor. Keep Q32/K512, grid, compute config, weight/activation/KV dtypes, and numerical policy. Do not switch to scalar absolute offsets: scalar offsets alter effective K geometry and program-cache keys (`sdpa_program_factory.cpp:1235–1238`; scalar cache-key regression at `test_sdpa_chunked.py:315ff`).
- Shift the fill table by `position//32`. It must remain `[B,16]`: right-pad a shortened slice with `-1`. The present `row_table[:, first_page:first_page+16]` (`bounded_prefill.py:434`) is only guaranteed to retain16 columns at512-aligned starts. At2016/capacity2048, it has one column. Padding is for fill only; the full attention table remains the scheduler's valid map.
- Check `ceil(count/32)` active write destinations with `disjoint_active_write_pages`, retain the existing bulk/ordered branch selection, and never permit padded fill to touch another request's page. Do not demand uniqueness of unused virtual pages. After a 32-aligned origin, padding the final tile writes only future rows, not the logical prefix; certify physical ownership of that complete tile. The helper checks only uniqueness among active writes: protection of prefix/inactive read pages still relies on the scheduler's valid ownership/COW contract. Ordered execution alone cannot make conflicting shared-page contents correct.
- Preserve exact host token/position lengths. Pad only Q/K/V tiles as the existing helper does, slice attention output back to `count`, and send **no padded tokens through GDN**. For481→962 the result remains31 exact head tokens then450 real body tokens, not512 logical tokens.
- Keep `_import_state` once per public `run`, FP32 recurrent work state throughout all internal chunks, and `_publish_state` once after iterator exhaustion (`bounded_prefill.py:105–115,129–159,402,456`). Never implement this by recursively calling the public generator per peel. Additional internal boundaries can select different GDN kernels or rounding, so they require state/accuracy tests even though the ownership contract is preserved.
- Adjust counters/telemetry to distinguish 32-aligned vector eligibility from512 boundary alignment. Do not mislabel a vector chunk at32 as512-aligned or continue counting it as a token head.

With this schedule, the serial length becomes `min(public_length, (-start)%32)`, at most31. This removes most repeated norm/QKV/MLP work and host copies at offsets such as962 or7696. It does not eliminate those copies for the residual head, nor establish an elapsed speedup.

### Conservative alternative if the tensor K-alignment gate fails

The smallest contract-preserving improvement is to move **only `_finish` outside `_attention_head`'s token loop**: keep per-token normalization, QKV/RoPE, exact fused update and decode SDPA, collect their attention-projection outputs, then perform one block residual/MLP finish against the original hidden block. The finish is token-independent, so its logical operation can be batched without changing attention causality. Use the existing flat/nonflat finish conventions and warm the concatenation of wider attention outputs. This reduces `H` MLP/finish calls to one but retains `H` attention/QKV calls and `3H` metadata copies per FA layer; numerical agreement still needs testing.

A larger second step could hoist input norm/QKV/RoPE and output gate/projection as well, leaving only token cache-update/SDPA calls in the loop. That requires factoring the compact implementation at `multichip_decoder.py:975–1113`, matching sharding, gate behavior, and `decode=False` precision routing. It is less local than the proposed peel.

Another mathematically sound option keeps K512 and left-pads **Q only, after RoPE**, to the preceding512 boundary. Call chunked SDPA using the existing runtime tensor set to that aligned boundary, then discard dummy-query outputs. K/V writes must still preserve the initial partial cache tile with exact updates, followed by tile fills at the real positions. Dummy Q does not justify dummy K/V writes. Variable padding/slicing can introduce offset-by-length program signatures; this option needs a fixed-shape runtime-index design or a complete finite warmup plan. Do not replace the runtime start with a scalar or silently assume existing warmup covers it.

Reducing K blocking to32 for the bridge avoids the documented alignment restriction but conflicts with the model's explicit long-context rounding rationale (`optimized_decoder.py:1251–1254`). It is a separate precision candidate, not the recommended minimal repair. A true arbitrary-start bulk fill/SDPA primitive would require boundary-tile merging plus token-remainder handling across reader, masks, compute, and writer, and is outside the smallest model-side fix.

## 4. Warmup, trace lifetime, and external-cache invariants

`QwenGenerator.prepare` warms single and optional uniform prefill **before** decode/sample/reset captures (`generator.py:614–680`). `BoundedPrefill.prepare` covers lengths1..512, representatives of each layer/state/policy signature, head gathers, distinct-output concat signatures, and per-layer persistent collective buffers (`bounded_prefill.py:258–367`). Preserve this generator-wide order. Representative sharing is for program signatures; `_prepare_layer_buffers` gives each layer its own storage.

For the candidate, exercise the new vector-at-non512-aligned-start branch during preparation, plus shortened fill tables and both ownership branches. Warm only legal start/length combinations with valid attention pages: do not blindly run every length1..512 at start32 in a capacity512 table. The unchanged start0 sweep already covers all physical lengths; a legal nonzero example such as32→512 adds the new runtime-offset branch without an unnecessary offset-by-length sweep. Existing length coverage handles unchanged shape-dependent operations, but does not prove new branch/signature coverage. All new persistent metadata, slice/gather/concat programs, and per-layer collective buffers must exist before first capture. Warm publication for every slot as today. Restore page tables and start/position metadata, reset mutated KV/GDN state, and restore RNG/request inputs before capture as the generator already does. Keep preparation idempotent.

Test with `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, not only the performance configuration that disables tracking (`doc/optimized_vllm/run_sweep.py:94`). The tracker can explicitly exclude program-cache allocations (`tt_metal/impl/allocator/trace_allocation_tracker.cpp:183–192`), so unchanged trace IDs alone are insufficient. Require stable program count and persistent bindings after complete preparation, no later captures/retirements, and a final successful decode replay. Transient allocations must be freed before replay; do not retain yielded logits across it (`generator.py:881–886`).

The scheduler-allocated cache must remain the identical object (`generator_vllm.py:96–121`, `generator.py:95,727ff`). Retain each K/V allocation and traced recurrent/table/input/output binding, remap state and page metadata using existing code, and write into persistent buffers rather than rebinding them. In particular, the row-major staging/copy sequence in `_publish_state` exists because tiled `slice_write` can reassign its output tensor (`bounded_prefill.py:150–157`); do not simplify it away. Never release or recapture generator traces to make an unprepared request work.

## 5. Discriminating follow-up tests — not run here

Run these only when the baseline owner releases the hardware. Record revision, exact shapes/dtypes/configs, page mappings, synchronized results and timings; do not run an unbounded full-model sweep as the first experiment.

### A. Resolve the SDPA boundary before changing the model

Use one device/local geometry `[B,6,Q,256]`, KV `[pages,1,32,256]`, **BF16 Q and `bfloat8_b` K/V cache**, Q32/K512 and production compute config. These are the selected runtime dtypes (`doc/datatype_sweep/selected_precision_config.json:54–60`); the adapter's BF16 allocation metadata must not be mistaken for the actual cache dtype. Compare against the same quantized K/V values consumed by the device, and optionally repeat with BF16 cache to isolate packing effects. Test tensor starts32,480,992 (non-K-aligned) against a dense host reference with absolute-position causal masks, alongside starts0,512 as controls. Use Q lengths1,31,32,33 and a longer bridge ending at a512 boundary, with only the existing round-to32 padding. At capacity2048 include2016→2048. Use shuffled physical pages and distinguishable prefix K/V; exercise first and subsequent causal Q tiles, not just the final output row. Require existing local attention accuracy gates and finite outputs.

Reuse the same start buffer, then replace it with another same-spec buffer, update Q contents and offsets repeatedly, and verify results plus stable program count. This checks runtime reads and descriptor cache rebinding rather than only an initial cache miss. Completion must be observed by result readback/synchronization. The raw481 start should be a host-side eligibility rejection test for the proposed helper, not an assumed-supported SDPA case.

### B. Reduced real model, including GDN after full attention

Use layers **`[0,3,4]`** (GDN→full attention→GDN), optionally `[0,3,4,7]`. Existing `probe_bounded_prefill.py:189` uses `[0,3]`: its state checks cannot see an attention error propagated into a later GDN state. Also, that probe explicitly makes no trace-lifetime claim (`3–6,200–206`).

| Case | Distinguishing failure it can expose |
|---|---|
| 31→545; 32→545; 33→546 | Old large head versus new tile peel; non-K-aligned vector start; partial first tile |
| 1→545; 511→545; 512→545 | Largest current head; one-token peel; unchanged aligned control |
| Separate public calls0→481→962→1443→1924 | Scheduler progression and BF16 publication at every public boundary |
| 2015→2048 and2016→2048, capacity2048 | One-token peel then short fixed fill table; Q/table capacity limit |
| Lengths1,31,32,33,127,128,129,511,512,513 | Exact logical tails and physical padding around tile/chunk boundaries |

Compare to the unchanged native decoder and/or old bounded implementation **with identical public call boundaries**, as `probe_bounded_prefill.py:123–143` already does for continuation. A monolithic call is a useful extra check but has different recurrent-publication rounding. Compare every logical logit, both GDN conv/recurrent states, and the next several decode outputs where capacity permits. The cases ending at2048 with capacity2048 stop after logits/state/KV checks: decode position2048 is invalid (`generator.py:1041–1042`). Repeat equivalent continuations at larger capacity for following-decode numerics; reset/start a new valid request for the final replay check in the boundary-capacity test. Reuse existing gates (`PCC >= .995`, top1/top5 agreement >=.98, top100 containment1.0, finite outputs), reporting numeric errors rather than weakening tolerances for a candidate.

Snapshot physical KV through the virtual-to-physical mapping before continuation. Require exact unchanged **decoded** prefix values at positions `<start` and exact untouched other-request state/pages; compare newly written logical K/V and allow only the identified final-tile padding footprint in future rows. Optionally compare raw cache bytes separately: BFP8 row-update repacking is not itself proof of prefix corruption, and equivalent encodings should not be confused with changed decoded values. Use unique, shuffled active pages and a small physical pool with aliased **unused** virtual tails. Exercise ordered fill with batching disabled or the single-row engine. An active shared page is a separate ownership-negative test: require exclusion/COW, or prove that all owners require identical contents. Do not expect ordered writes to preserve distinct K/V for conflicting owners: all rows are filled before batched SDPA (`bounded_prefill.py:184–190`), so the last writer wins. Record external cache identity and every device buffer address.

Cover B1 prefill into a nonzero slot of a capacity32 generator, a small uniform B3 engine, and ragged/reordered rows routed through the B1 engine. A B1-only generator does not exercise serving's inactive slots or remap/publication behavior.

### C. One prepared generator, live traces, new offsets

Extend `probe_full_trace_contract.py` and `probe_full_mixed_prefill.py`: prepare once (all physical lengths, both enabled engine widths), freeze trace IDs, capture/retirement counts, program-cache size, and persistent bindings; then alternate the cases above with changed token inputs, page-only reloads, slot remaps, and decode/sample replay. Include intermediate `sampling_mask=False` continuation, which must publish state without sampling or a full-logits host read (`generator.py:891–896`). End in decode **after** the last new shape, so a late allocation cannot hide at workload teardown. Include tracked-allocation/program-cache acceptance and its positive control separately from timing; the existing `--quick` probe is explicitly diagnostic, not acceptance.

### D. Performance discrimination

From identical initialized prefix/state snapshots, warm and measure old versus candidate at starts31,32,481,962,7696 plus aligned controls. First prove operation counters: at most31 token-head steps per FA layer for the peel; vector processing of the remainder; head copies follow the same reduction. Use an explicit synchronization around each timed eager block (or a demonstrated completed read), and report host submission separately if measured. Then repeat the same 32-user serving harness and scheduler budget to assess TTFT, ITL distribution, and throughput. Start481 alone is a poor performance discriminator. No speedup or ITL target is claimed by this inspection.

## 6. Claim review and other potential issues

After drafting, I rechecked the headline against interval construction, caller slicing, state publication, compact/noncompact dispatch, the actual fused-update writer, and SDPA reader/compute/writer arithmetic. An independent xhigh review checked the draft against the code. The final draft incorporates ownership, runtime-dtype, compute-variant, BFP8-prefix, and capacity-edge test clarifications.

| Candidate explanation | Evidence / precondition | Adjudication |
|---|---|---|
| Complete layer repeated across an unaligned head | Direct loop and `_finish` call; reached whenever a public start is non512-aligned | **Confirmed performance mechanism**; not a measured fraction of total serving time |
| Start481 implies481 serialized tokens | Contradicted by `512-481=31` | Rejected; later offsets explain the growing cost |
| Tensor start supports any absolute token | All kernels divide by Q chunk size | Rejected;481 loses its remainder |
| Tensor start at32 with K512 is safe because validation accepts it | Kernel arithmetic agrees, documentation still forbids it; no targeted hardware result | **Repair hypothesis / test gate**, not established support |
| Existing fallback corrupts KV or GDN | Exact row updates and once-per-public-call publication explain intended preservation | No source evidence of corruption from this mechanism |
| Padding fill table with `-1` makes padded SDPA reads safe | SDPA has no sentinel skip | Rejected; keep distinct tables and bounded Q extent |
| Smaller K blocking is a free scheduling optimization | Explicit512-key long-context rounding policy | Separate precision experiment; not a minimal accepted-policy fix |

Additional findings are repair hazards/test gaps, not root causes of the current fallback: shortened fill slices at future32-aligned origins; hypothetical overextended Q reads near capacity; downstream GDN omitted by `[0,3]` tests; new offset-dependent padding programs that could compile after capture. Also, active-page aliasing cannot generally be repaired by the existing ordered-fill branch; no evidence shows the reported scheduler supplies such invalid ownership, and prefix caching is disabled by this adapter (`generator_vllm.py:33–34`). None is promoted as a present runtime failure without its required preconditions being introduced or observed.

The confirmed defect can be addressed entirely on the model side while preserving the scheduler/plugin interface and external cache. The decisive unresolved fact is whether the current Blackhole Q32/K512 tensor-start implementation meets accuracy and lifetime gates at non-K-aligned tile starts. If it does, the 32-token peel is the most local substantial fix; if it does not, batch the token-independent finish while retaining the proven exact cache/attention path.
