# Final selected TP4 inter-layer contract

This contract is derived from installed `tt/multichip_decoder.py`, its `_STAGE5_DEFAULTS`, `tt/flat_prefill.py`, and page-table ownership code. It documents the stage-5 decoder boundary for later model assembly; it does not perform or claim full-model bring-up. Final measured evidence must identify the same installed source and effective policy.

## Boundary to preserve

| Mode | Logical tensor per rank | Meaning and storage |
|---|---|---|
| Prefill | `[B,S,1280]` | BF16, tiled, interleaved DRAM; rank r holds hidden channels `[1280*r:1280*(r+1)]` |
| Native decode | `[1,B,1280]` | Same BF16 hidden shard, compact live rows; one physical tile row for B<=32 |
| Legacy decode compatibility | `[B,1,1280]` | Accepted at the public entry, compacted internally, returned in the same legacy shape; use native compact shape for a layer stack |

The mesh is exactly `1x4` with Linear fabric topology. Tensor parallel degree and rank order are fixed across the stack. The residual is already reduced to its owned hidden channels; it is not an unreduced partial sum. Feed one layer's returned tensor directly to the next. Do not gather, reduce again, concatenate rank tensors, replicate hidden state, or normalize at the layer boundary. Do not insert an inter-layer sharding/layout conversion. Host `upload_residual`/`upload_decode_residual` are setup ingress, not layer transitions; host concatenation in correctness scripts is readback only.

`decode_forward` recognizes `[1,B,1280]` from state batch metadata and returns that shape unchanged. Legacy `[B,1,1280]` callers pay internal compatibility reshaping; chaining that interface is unnecessary. B1 shapes coincide. Each state remains owned by its layer and batch. Output has the same rank-local hidden interpretation for GDN→GDN, GDN→full attention, and full attention→GDN transitions.

## Placement and collective ownership

Selected `residual_layout='sharded'`, `flat_decode=True`, `flat_norm=True`, `flat_ccl=True`, and `ccl_decode_memory='DRAM'` preserve the boundary. Compact decode uses DRAM persistent CCL outputs/intermediates; layer-private `_ccl_buffers` are prepared by exact-signature warmup before tracing. Decode normalization may temporarily gather the complete hidden activation and full-attention normalization may temporarily use L1 width shards; those are intralayer projection-input layouts, not residual layouts.

All selected collectives are intralayer:

- Hidden norm either all-gathers the activation before normalization or gathers local statistics, normalizes owned channels, and gathers normalized activation for the column-parallel projection. GDN decode gathers first; full attention gathers first for decode or single-user prefill. Multi-user prefill uses the selected distributed path.
- Attention-output and MLP-down row-parallel projections reduce-scatter to 1280 owned hidden channels. Eligible prefill row counts use the selected fused matmul/reduce-scatter path; otherwise projection and reduce-scatter are separate.
- `_reduce` returns an already-width1280 result directly. The subsequent residual add remains rank-local. `_finish`/`_finish_flat` return the owned hidden shard.

No selected inter-layer collective remains. This is a source contract; final direct-stack boundary instrumentation provides the measured evidence. `distributed_decode_norm='off'` is selected for both kinds. DiT/minimal native norm experiments remain opt-in; they must not be assumed part of final owner capacity or numerical results. The failed minimal64-owner allocation is not evidence for reducing the selected path's context.

## Prefill, tails and cache contract

Public prefill retains logical `[B,S,1280]` and accepts valid non-aligned sequence lengths. The implementation owns tile padding, slicing, temporal chunking and cache masking. Internally, eligible multi-user chunks flatten to `[1,B*S,1280]` and restore public shape before returning. The selected temporal chunk is 512 tokens: B32S513 processes a 16384-row bulk chunk and a separate S1 tail, not one 16416-row projection. B1 and S1 tails retain the inherited ordered dispatch. Do not demand aligned public lengths to avoid internal work.

The list interface carries separate tensors, states, starts and rotary inputs for independent rows; scalar starts broadcast as specified by `flat_prefill_forward`. Preserve per-entry lengths and start positions rather than merging unrelated cache histories. Public context bounds remain the pinned HF limit, 262144 tokens; final-stage smoke evidence must not be relabelled a newly executed maximum-context test.

Full attention keeps one KV head per rank with BF8_B tiled cache tensors `[physical_pages,1,32,256]`. The page size is 32. Physical IDs may be shuffled and noncontiguous. Storage depends on maximum physical page ID plus one, not merely live page count. Tables with at least16 columns are padded to a multiple of16 with new nonaliasing page IDs. Page-table ownership/certification and ordered fill rules remain authoritative; metadata replacement is setup-only and requires trace recapture when storage changes. Decode positions are device int32 values per user, not a single host scalar. Preserve the untouched-cache-row checks and mutable-table compatibility behavior.

GDN persists BF16 local12-head recurrent state `[B,12,128,128]` and row-major BF16 convolution history `[B,3,2560]`. Selected cumulative-exp recurrence with column grain1 keeps live update/readout arithmetic separate from final BF16 persistence. State precision, update order, history ownership, and rank-local head mapping must remain unchanged during later assembly.

## Trace and later full-model preservation

Use complete ordered decoder calls on CQ0. Allocate all layer-private state, CCL semaphores, exact-shape persistent buffers, stable input/position/rope banks and output slots before first capture. Warm every intended shape/program, including prefill ordered-cache behavior and decode. Do not share mutable CCL scratch or semaphore owners across independently live layers; sharing immutable representative weights in a resource test is not a production ownership policy. Copy changed input values into stable storage between replays, without replacing addresses or adding host transfers inside capture.

A later full-model implementation should keep embeddings initially partitioned to the same hidden ownership and preserve the compact boundary throughout the stack; final normalization/head handling is a separate later-stage design. This document does not authorize or implement that work. Reuse final direct-stack evidence, 64-owner resource evidence, context smoke, cache ownership gates and final effective policies; do not rediscover the inter-layer format by adding gathers.

## Capacity evidence boundaries

`final28_capacity.py` waits for passing final artifacts and produces separate sections for: (1) source-derived 64-distinct-layer estimates including future embedding/head storage and reserve; (2) sampled allocator readings from 64 independent communication/state owners with only two representative weight sets shared; and (3) real single-layer context smoke with explicit physical reservation. Sampled readings are not allocator high-water marks. A positive source-estimated margin is not a successful all-distinct-weight full-model allocation. Historical stage-4 max-context runs remain identified as historical; the final32K smoke only establishes its actual tested length and continuation.

Capacity reconciliation is deliberately gated and has not been executed during preparation. After final validation completes, run `python models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/final28_capacity.py --performance <final-TP4-performance.json> [--performance <each-other-final-performance.json> ...]` to inspect the proposed report. Add `--write` to write `final28_capacity.json` and only the `optimized_multichip_decoder` context-contract section. The default manifest is `final_validation28_manifest.json`; repeated `--validation-manifest` supports separately completed validation groups. Missing summaries, receipts, artifacts, stale source/policies, failed gates, or a negative estimated max-context margin stop before updates. The script never launches devices or downgrades supported context automatically.
