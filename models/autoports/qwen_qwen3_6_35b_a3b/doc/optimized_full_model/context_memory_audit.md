# Optimized full-model context and memory audit

CPU-only inspection; no TTNN import, device access, reset or source change. Units below are bytes per device unless explicitly per bank. Allocator DRAM census fields are multiplied by eight banks, not by four mesh chips. The context contract remains 262144; this audit does not edit it.

## Matched context 1024 endpoint comparison

Both `before.json` and `final/performance.json` pass with batch 1/context 1024. All 40 individual layer-load censuses match exactly. These are allocator snapshots at named endpoints, not sampled runtime peaks.

| Endpoint | Before allocated | Final allocated | Delta | Final free | L1 allocated/bank before→final |
|---|---:|---:|---:|---:|---:|
| all_weights_and_rope | 18,843,353,088 | 18,978,734,080 | +135,380,992 | 14,999,981,056 | 571,456→571,456 |
| generator_persistent_state | 19,020,463,616 | 19,155,844,608 | +135,380,992 | 14,822,870,528 | 571,456→571,456 |
| performance_after_measurement | 19,298,695,168 | 19,433,862,656 | +135,167,488 | 14,544,852,480 | 581,696→581,696 |

## New head storage and retained legacy head

The measured production geometry is K=2048, local N=62080, eight DRAM banks, three readers and bank storage width=7,776. `tt/lm_head.py` rounds each bank width to a multiple of 32×readers. Thus physical width is 62,208, adding 128 padded columns per device. A BFP8 tile occupies 1088 bytes: useful head storage is 64×1940×1088=135,086,080; physical new bank-sharded storage is 8×64×243×1088=135,364,608. Padding costs 278,528 bytes. The new copy therefore costs more than the useful 135,086,080-byte estimate.

`tt/model.py` retains `head_weight` for the legacy many-row terminal path and constructs `dram_head.weight` as another resident buffer. Combined old-plus-new head payload is 270,450,688 bytes/device. The authoritative matched constructor census grows by 135,380,992, which is 16,384 above the calculated new tensor payload. This small residual is explicitly unassigned allocation/setup accounting; no complete unique-allocation inventory was collected here. Post-measurement growth is 135,167,488, or 213,504 less than constructor growth. Endpoint allocation/lifetime differences must not be mislabeled as a change to weight payload.

L1 endpoint allocation is unchanged, but new kernel-local circular buffers can alter transient pressure. The state slab declares 116,736 bytes of CB storage per participating worker; endpoint equality is not a runtime high-water measurement.

## Preserved maximum context and qualified projection

The unchanged `doc/context_contract.json` records stage 6 all 40 maximum-context acceptance: one nonaligned 262143-token prefill, then sampled-token replay at final valid position 262143 with next-position sentinel 262144. Its prepared allocation is 25,372,641,280, free space 8,606,073,856, and allocator capacity 33,978,715,136 bytes/device. Post-prefill and post-decode censuses matched the prepared census. This is historical source-bound acceptance, not a fresh stage 7 maximum-context run.

KV remains BF16 with one local 256-dimensional head per chip, page 32 and the same ten full-attention layers; recurrent backing remains FP32. No KV dtype/layout change warrants a new maximum sweep in this optimization stage. The contract remains 262144 without reduction. Batch 32 at maximum context was not measured and must not be inferred.

A conservative simple projection adds the matched-context constructor increase 135,380,992 to historical stage 6 prepared allocation: projected allocation 25,508,022,272, projected free 8,470,692,864 bytes/device (7.888947 GiB). This is headroom arithmetic under unchanged large-buffer geometry, not proof of new-source max allocation success, fragmentation tolerance or runtime peak. Trace region 200,000,000 bytes is already reflected in the allocator configuration; it is reported separately and must not be subtracted from measured free space again.

## Current nonaligned/context evidence

`final/prefill_boundaries.json` passes all 40 at configured context 4096 for lengths 64, 255, 256, 257, 1025, 2049. The four nonaligned lengths 255, 257, 1025, 2049 each run two queued model and sampling replays, with no token/position/page/activity uploads, readback, capture or release during the replay segment. Prepared and final censuses match. The repeated AIME-token fixture proves execution/state contracts, not long-form quality.

`final/prefill_context32769.json` now passes all 40 layers at configured context 65,536 with a nonaligned 32,769-token prompt (remainder one for the 32-token page/tile, 64-token linear chunk, 256-token attention chunk and 2,048-token MoE chunk). Two queued model replays and two queued sampling replays advance next position to 32,771. Token, position, page-table and activity refreshes, readbacks, captures and releases are all zero during that replay segment. Program-cache entries remain 9,590 and trace IDs remain model 0/sampling 1.

Prepared and final allocator snapshots match exactly: DRAM allocated 2,628,042,944 and free 1,619,296,448 bytes per bank, across eight banks; this is **21,024,343,552 allocated and 12,954,371,584 free bytes per device**. L1 reports 591,936 allocated and 868,352 free bytes per bank, across 110 banks. These are endpoint snapshots, not runtime peaks. The recorded first-token elapsed time is 691,479.282 ms with all physical variants warmed; the long prompt runs only once, so this is a smoke measurement rather than a repeated same-length warmed TTFT benchmark. Repeated AIME tokens do not establish long-form output quality.

The installed bounded `wait_for` exited 0 after 3,724 seconds on `FINAL_MEASUREMENTS_DONE`; the driver reports the long-context command exited 0. All 38 source hashes recorded by the smoke match the files inspected here. This supplies fresh stage 7 nonaligned long-context evidence at context 65,536. It does not replace the historical 262,144-context acceptance or convert the maximum-context headroom projection into a fresh maximum run.

Exact source/artifact SHA256 values, raw endpoint rows, arithmetic and completed wait state are in `context_memory_audit.json`. Runtime peak remains unmeasured.
