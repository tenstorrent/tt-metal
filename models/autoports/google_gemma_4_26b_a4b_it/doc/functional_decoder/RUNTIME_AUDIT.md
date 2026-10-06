# Integrated runtime and continuation audit

Source-only re-audit, 2026-09-25. No hardware commands or implementation edits were performed. This supersedes the previous audit: prefill now executes locally in `tt/decode_attention.py`, and decode uses `tt/precise_attention.py`, rather than the previously inspected imported attention forward paths. Scope also includes `functional_decoder.py`, `precision_ops.py`, `routing_precision.py`, reachable imported helpers, and selected TTNN lowering.

## Runtime host boundaries

No reachable explicit host tensor computation, upload or readback was found in the integrated default forward paths under the caller-owned device-tensor contract.

- PyTorch and `from_torch` in `FunctionalDecoder.from_state_dict` and `PrecisePagedAttention.__init__` construct weights, positions and offsets during setup. `norm_weight` and router weight conversion are also setup-only. Imported expert sparsity is created in `Gemma4Experts.__init__`.
- Forward `rms_norm`, `rotary`, QKV/output projections and router selection use TTNN device operations. Fixed Python head/token/slot loops and tensor shape metadata do not extract device values.
- Router `zeros_like` receives BF16 TILE device input (`routing_precision.py:84-89`). C++ `creation.cpp:236-244` selects the on-device `fill` path for this case. It does not take that helper's host construction fallback.
- Local prefill contains no `ttnn.zeros`, `from_torch`, or imported short-tail padding helper. Its tail consists of cloned device K/V (`decode_attention.py:111-136`). The old report's dependency on `_clone_sliding_prefill_tail` and `use_persistent_tail` is obsolete. The wrapper still physically pads short final sliding chunks to 1024, ensuring the filler-Q slice can cover retained history.
- Decode resolves RoPE, page IDs and cache rows using device `embedding`/`gather` and tensor arithmetic. `gather.cpp` lowers to device layout/slice operations and `prim::gather`; `embedding.cpp` lowers to device layout conversion and an embedding primitive. No host cache readback occurs.
- Imported full-attention `chunked_prefill_sdpa` is still reachable, but receives scalar offsets here. Its `ttnn.full` branch requires a tensor offset plus a nonzero inner offset (`models/demos/gemma4/tt/attention/operations.py:370-380`) and is unreachable. TP=1 returns before imported CCL buffer creation (`tt/ccl.py:250-251`). Shared MLP and expert forwards remain device-only.

`tests/runtime_audit.py` rejects PyTorch operations and named Python TTNN host entry points. It does not intercept C++-internal host construction; targeted lowering checks supplement that guard rather than prove every library/configuration path.

## Decode QKV SFPU row-dot follow-up

The latest `QKVLinear` prepares a transposed FP32 weight during setup (`routing_precision.py:18-20`) and, for logical sequence length one, computes output rows in fixed groups of 256 using device slice, repeat, multiply, width reduction, transpose and concat (`:29-40`). Public batched decode invokes one slot at a time (`functional_decoder.py:204-219`), so every slot takes this path. Its input is the local FP32 RMSNorm output (`:97-98`), and the current caller supplies no memory-config override (`decode_attention.py:45-49`). Prefill retains the existing linear path.

No new host fallback or cache-address computation was found. Dispatch and loop bounds depend only on fixed tensor shapes. The real configuration has hidden width 2816 and fused output widths 8192 (sliding) and 10240 (full), including duplicated tied K/V for full attention. Both widths divide exactly into 256-row groups. Ascending group concatenation preserves fused Q/K/V order, with no partial final group. For logical input height one, `repeat.cpp:140-148,450-451,510-535` selects device TILE-to-ROW_MAJOR conversion, logical-row repeat and tilization. Its zero-repetition `ttnn::zeros` branch is unreachable for these strictly positive repetition factors. Thus tile padding does not multiply the logical row count.

The extra persistent FP32 QKV copy is **88 MiB sliding / 110 MiB full**, while the original BF16 weight remains live. Each repeated input and product has shape `[1,1,256,2816]`, **2.75 MiB apiece** in FP32; the output list retains reduced groups, not every full product. Layout conversion, slicing and reduction can add temporaries, so these figures are not peak-memory measurements. Setup can also transiently hold both FP32 conversion and transpose allocations. This adds a real capacity/performance cost, but source inspection found no invalid address or premature explicit deallocation.

The static device allocation/operation sequence is compatible with capture under the existing warm-exact-signature contract. Trace validity must be established using this new path, including its repeat/layout/reduction/concat operations; artifacts from the earlier matmul implementation cannot establish that. The latest saved traced results below exercise the integrated path. No independent device or performance verification was performed for this audit.

## Source findings and verified fix status

### Large physical-page row indices: fixed, focused device evidence inspected

The original `PrecisePagedAttention` converted physical page IDs to FP32, multiplied by `KV_heads * 32`, added within-page row offsets, then converted to uint32. Sliding attention has 8 KV heads: page 65536 begins at flattened row 16777216. Adding row offset 1 rounds back to 16777216 in FP32, so distinct cache rows alias. A nine-request full-context sliding pool can exceed this boundary; small batch-32 tests do not exercise it.

Verified current source: setup `row_offsets` is uint32 TILE (`precise_attention.py:26-29`), and `cache_row_indices` retains uint32 physical IDs, left-shifts by 8 for sliding or 6 for full attention, adds uint32 offsets, and returns ROW_MAJOR indices (`:32-39`). Production attention calls that helper (`:57-58`); there is no intervening FP32 row-address conversion. Float position/window arithmetic remains separate and below the FP32 exact-integer boundary.

The main agent executed `tests/cache_indices.py`; I inspected its source and existing `cache_indices.json`, without running hardware. The test calls the production helper under `device_only()` for physical IDs **0, 65535, 65536, 65537, 262143, 262144**, with **8 and 2 KV heads**, and compares every resulting row against integer arithmetic. The saved result reports exact equality for both and a clean runtime guard. This proves the tested index arithmetic, not allocation or attention over a whole large pool.

### Cache format rejection before mutation: fixed in source

The effective decode requires **BF16 caches with 32-token pages**. Originally the page check ran only after decode had enqueued K/V updates, and public prefill admitted other sizes. Verified current source: both public `prefill_forward` (`functional_decoder.py:126-130`) and `decode_forward` (`:201-203`) inspect every supplied cache's page size and dtype and reject invalid values before entering attention or writing cache rows. These metadata checks do not read tensor contents. The deeper 32-token check remains as a defensive check.

### Physical chunk upper bound: fixed in source

Verified current setup requires positive multiples of 128, **chunk_size <= 16384**, divisibility into HF context, and at least one window for sliding (`functional_decoder.py:39-46`). Local square SDPA therefore sees at most 16384 rows for full attention or 17408 including sliding history, below the imported 32768 limit (`attention/operations.py:26-54`). These checks constrain physical execution chunks, not logical request lengths; default 1024 is unchanged.

All three identified source issues are addressed in the inspected implementation. Device evidence above was produced by the main agent and inspected independently; this audit itself remains source-only.

## Cache and prefix contract

The active-only position documentation is accurate: `decode_forward` now requires nonnegative valid cache positions. **Inactive -1 positions are unsupported by the new attention.** Although the native cache writer skips -1, precise attention would produce an all-masked score row; subtracting its maximum then yields a uniform softmax rather than a skipped lane. Both RoPE indices and cache positions must name valid allocated rows. There should be no inherited claim that passing -1 is safe.

For `start_pos > 0`, the wrapper selects the request's page-table row and processes tokens with absolute device positions. The cache writer replaces one row within its page/tile, preserving earlier prefix rows (`experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp:88-139`). Remaining layer operations are token-local, so this is a valid causal decomposition. It does not consume the local prefill tail. A fresh request clears that tail; intermediate fresh-prefill chunks are at least one window, and only the final chunk can contain padding. Padded future rows cannot affect kept causal outputs, and local cache fill writes only tile-ceil(valid) (`decode_attention.py:104-108`).

The documented **128-token-rounded cache capacity** remains required by imported full-attention chunked prefill, which pads final Q to 128 and validates cache length against the padded end (`attention/operations.py:363-368`; `transformer/sdpa/device/sdpa_device_operation.cpp:366-374`). The previous report's native decode 64-token read-window explanation no longer applies: precise decode gathers explicit pages and applies its own mask. Capacity must include planned continuation/decode tokens, with valid physical mappings even for allocated future columns because full decode gathers the whole page-table row.

RoPE must cover physical prefill padding: tile rounding for one chunk, chunk rounding for multiple sliding chunks. Chunk-divides-context validation keeps supported padded end positions within the full HF table. Logical request lengths remain arbitrary within the HF context.

## Evidence and remaining checks

I inspected existing artifacts produced by the main agent's hardware workflow, without running them:

- `reuse_sliding_exact_qkv_integrated.json` and `reuse_full_exact_qkv_integrated.json` each record nine passing requests, trace reuse and a clean runtime guard.
- `batch32_sliding_exact_qkv_integrated.json` and `batch32_full_exact_qkv_integrated.json` each record 32 passing prefill and 32 passing decode checks, traced repeated equality, clean runtime guards, and disjoint randomly permuted page mappings.
- `headline_sliding_exact_qkv_integrated.json` and `headline_full_exact_qkv_integrated.json` record real-weight S4096 prefill PCC **0.998526 / 0.999441** and 128 traced decode steps with minimum PCC **0.999043 / 0.999774**, respectively; both record repeated equality and clean prefill/decode runtime guards.

These establish the recorded integrated-QKV cases, not final long-context memory capacity or performance.

Focused index arithmetic is now covered by `cache_indices.json`. Remaining runtime evidence should cover partial-page prefix continuation and prefix preservation, final long contexts, full-pool capacity/performance, and the device-only guard under the final implementation. Rejection ordering and chunk bounds were verified directly from source; this audit did not run separate rejection tests.

Resource caveat for pending long/performance runs: `embedding.cpp:29-32` converts TILE weights to ROW_MAJOR before lookup. Here those weights are the flattened whole K/V cache, so even sliding decode's 33-page selection currently triggers whole-pool device layout conversions per slot. This is not a host fallback, but small-cache accuracy results do not establish full-pool memory capacity or latency.

## Final full-context correction and follow-up evidence

Full attention now consumes the existing device page-table row directly: its
logical selection is the identity over every page. Sliding attention retains
the bounded device gather. This removes the native wide ROW_MAJOR gather that
corrupted full-attention tables above width 1920; it introduces no host work.
`AUTODEBUG_long_context.md`, `page_gather_boundary.json`, and the focused
long-attention probe record the failing primitive and corrected cache/attention
control. `long_full_262144_fixed.json` passes prefill and traced decode over the
full allocated context under the runtime guards. `long_sliding_262144_final.json`
and `long_sliding_262143_final.json` cover the other layer kind. Both
`continuation_*.json` now pass the partial-page and request-isolation checks.
The old remaining-check list above describes the earlier audit snapshot.

## Allocation warning classification

Some diagnostic/reuse logs contain the allocator's generic warning about
allocating buffers while a trace exists. `tt_metal/impl/allocator/allocator.cpp`
emits this for any post-capture allocation; it does not report a detected
address overlap. The tests retain stable decode inputs, outputs and caches from
capture, release temporary request tensors before replay, and preserve the
trace-owned buffers until trace release. Sliding prefill tail buffers can
persist between requests, but decode does not consume them and fresh prefill
replaces them. Nine requests with changed page maps and repeated trace outputs
pass for both layer kinds. This controls the tested lifetime pattern; callers
must still honor the documented buffer/trace ownership contract. It does not
establish safety for arbitrary allocation/deallocation patterns.
