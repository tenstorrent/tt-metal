// Per-feature explanations shown in the artifact's Roadmap table (click a feature to expand).
// what: what the feature is and where it sits today; why: mechanism + numbers; todo: rough implementation steps;
// notes: interactions and caveats. Keys match study.js FEATURES.
'use strict';
module.exports = {
  pool: {
    what: 'Replace the static 1M-token slots with a few contiguous <b>lanes</b> per stage plus a shared <b>paged pool</b>. Lanes are the KV buffers the existing attention kernels run on. The pool is the rest of KV memory, split into pages keyed by the hash of their content (64-token blocks), shared by every conversation with the same prefix and evicted least-recently-used. Before a request runs, its cached prefix is copied from pool pages into a lane; afterwards its new tokens are copied back to the pool.',
    why: 'Today a conversation owns a whole 1M slot even when its context is 50k tokens, so 4 galaxies keep only 20 conversations warm. AgentX needs hundreds of live sessions to fill the pipeline, so almost every request misses: 0.3% hit rate at 256 lanes in the simulation. A pool stores only the tokens that exist: the same 21M tokens hold about 150 median (140k) contexts instead of 20. It shares identical prefixes (sub-agents start from the main agent\'s system prompt and tools) and evicts from the tail, so a conversation keeps its head. The copies are cheap: a median hit is about 65-107 MB per chip, about 0.5 ms of DRAM time, overlapped with compute.',
    todo: [
      'Page allocator and page table per stage in device DRAM; prefix index from block hash to page on the host.',
      'A page-list gather/scatter op that copies many pages into or out of a lane in one launch (per-page host copies would be dispatch-bound).',
      'A per-stage lane table: lanes are assigned per stage and only while that stage works on the request (today slot_id is global across stages).',
      'Scheduler: LRU eviction, pinning of in-flight prefixes, copy-in of the next request while the current chunk computes, copy-out after the last chunk.',
      'KV migration to decode reads from lane or pool pages instead of a fixed slot.',
    ],
    notes: 'Prerequisite for the host-DRAM tier. With fixed 1M lanes the dense layers still scan 1M tokens per chunk, so bounded dense gather (or variable-size lanes) is still needed. The lane count limits how many requests can share a batched chunk.',
  },
  host: {
    what: 'A second KV tier in the host servers\' RAM behind the device pool. Pages the device evicts are written to host RAM instead of dropped. When a request hits a host-resident prefix, those pages are copied back over PCIe into its lane while the request waits in the queue.',
    why: 'M3 KV is about 127 KB per token today (73 KB with the index-cache fixes). 4 galaxies hold about 21M tokens on device, but the live working set of AgentX near the throughput limit is several times larger, so the device pool alone still evicts conversations that come back minutes later. 1 TB of host RAM per galaxy adds about 32M tokens (56M with the index fixes). A host hit costs a PCIe copy: a 150k-token prefix is about 19 GB, about 75 ms over 4 × 64 GB/s, small next to TTFT and done while the request is queued. In the simulation host RAM size matters a lot (0.5 → 2 TB per galaxy: 34.9k → 52.8k goodput at 4 galaxies with today\'s kernels) and PCIe bandwidth barely does (16 → 256 GB/s changes it by at most 7%).',
    todo: [
      'Host page store in pinned memory with its own LRU.',
      'Asynchronous device→host write-back on eviction and host→device fetch before admission, using PCIe DMA from the host runtime.',
      'Scheduler: start the fetch when the request arrives; a fetching request must not block the pipeline.',
      'Measure the real galaxy host RAM size and achievable PCIe DMA rate; both are assumptions here (1 TB, 64 GB/s per galaxy).',
    ],
    notes: 'Requires the paged pool. AgentX caps host DRAM (3 TB per system, proportional to the accelerator allocation), so a compliant submission may get less than the default 4 TB at 4 galaxies.',
  },
  bounded: {
    what: 'The three dense layers use ring-joint SDPA. Its ring walks the whole KV slot capacity on every chunk (<span class="mono">cache_global = kv_cache.max_seq_len</span> in the dense attention prefill path), not just the tokens that are actually cached. The MSA layers already gather only the valid prefix. This feature bounds the dense ring to [0, kv_len).',
    why: 'With production 1M slots each chunk scans 1M tokens in every dense layer. Fitted from runs B/C, that costs about 100 ms per dense layer per chunk, regardless of chunk size or of how much is cached. The single-dense-layer stages then become the bottleneck: a run-C-like pipeline caps at about 18k processed tok/s. Bounded, the cost follows the real context (median about 140k), and those stages fall back to about a third of the MoE stage time.',
    todo: [
      'Pass kv_len (per chunk, as a device scalar so traces stay valid) into the dense ring-joint SDPA.',
      'Restrict the ring\'s K/V chunk loop and the gathered size to valid blocks, as the MSA path does with its gathered size.',
      'PCC at several cache depths; re-check the compile buckets.',
    ],
    notes: 'Redundant once lanes are request-sized (variable-size lanes) or paging exists. That is why its leave-one-out value is ×1.00 in the full stack. It also matters little if the scan kernel runs at link rate ("roofline kernels"). Low complexity and P0 while lanes are 1M: the best effort-to-gain ratio in the list.',
  },
  idxdedup: {
    what: 'The index-key cache used by the MSA indexer (one 128-wide key per token per layer, bf16) is stored on all 4 TP columns. That is 1024 of the 2112 bytes per token per layer: 48% of all KV memory. Store it once per SP row and gather it inside the indexer. The dense layers also allocate a zero-filled index cache, which can be dropped.',
    why: 'KV per token per layer drops from 2112 to 1344 bytes (−36%), so the same memory holds 57% more tokens. KV capacity is what limits goodput on this traffic, so the gain is ×1.14-1.29 in every scenario.',
    todo: [
      'Allocate index_k sharded over TP (by token blocks) or on one TP column, instead of replicated.',
      'In the indexer, all-gather index keys over TP before scoring, or score per TP shard and merge top-k. The extra traffic is small next to the K/V gather.',
      'Update the KV migration table and the decode-side layout.',
      'PCC.',
    ],
    notes: 'Stacks with index_k bf8. Its value grows with more lanes (more concurrent contexts).',
  },
  idxbf8: {
    what: 'Store the index-key cache in bf8 (1.0625 B per value) instead of bf16 (2 B).',
    why: 'KV per token per layer drops from 2112 to 1632 bytes (from 1344 to 1224 with de-replication): about 10-30% more capacity, for ×1.05-1.10 goodput.',
    todo: [
      'bf8 is already the runner\'s default; the measured runs opt into bf16 with M3_INDEX_CACHE_BF16=1.',
      'Validate that top-k block selection and end-to-end PCC hold at long contexts with bf8 index keys.',
    ],
    notes: 'Mostly an accuracy sign-off, not implementation work.',
  },
  async: {
    what: 'Today each stage receives a chunk, computes it, then sends the activation to the next stage (about 63 MB for a 5120-token chunk), and the send blocks the stage. Measured: about 18 ms per chunk at 5120 (7.5 ms at 2048) of blocking send, plus 14-16 ms of hop latency. With async handoff the send overlaps with computing the next chunk.',
    why: 'The blocking send adds 10-35% to the pipeline period. Its share grows with more stages (fewer layers per stage) and with faster kernels: ×1.13 at 4 galaxies with today\'s kernels, ×1.52 at 8 galaxies with roofline kernels. It also cuts TTFT, since 32 hops × 15 ms is about 0.5 s.',
    todo: [
      'Double-buffered D2D send/receive: post the send and start the next chunk immediately; the receiver pre-posts buffers.',
      'Make sure the transfer uses enough links (the activation is spread over the stage\'s chips).',
      'Re-measure with PREFILL_SYNC_PER_CHUNK=0: the per-chunk sync used for timing in the benchmark may be part of the measured blocking.',
    ],
    notes: 'Independent of the other features. Its value grows as the kernels get faster.',
  },
  batch: {
    what: 'Put several requests\' new tokens into one chunk, up to a token budget (8-16k works best). Projections, norms and the whole MoE run on the concatenated tokens; attention runs per request (sequential).',
    why: 'Most AgentX requests are small (median 1600 new tokens), so most chunks are small. An MoE layer has a large per-chunk cost that does not depend on tokens: expert weights (16 experts × 32 MB per chip per layer at [2,4]) and about 10 collectives per layer with ~40 µs latency each. More tokens per chunk amortise it. At chunk 2048 each expert sees only about 64 tokens (T×4/128), far below the 260-460 tokens per expert where the matmuls become compute-bound. Measured: ×1.13-1.35, larger with roofline kernels, at slightly higher TTFT. With sequential attention it does not matter which requests share a chunk.',
    todo: [
      'Chunk metadata per segment: lane/slot, start position, length.',
      'An attention loop over segments; each segment\'s KV is read and written in its own lane.',
      'MoE dispatch/combine buffers and activations sized for the budget instead of the chunk.',
      'A packing scheduler: fill the budget, split long requests across chunks, keep segments of one request in order.',
      'Needs the variable layout for token packing; with the fixed layout only whole chunks can be packed.',
    ],
    notes: 'Each request in a chunk needs a lane on the stage, so batching wants more (or variable-size) lanes. Fused multi-user attention adds nothing on top.',
  },
  var: {
    what: 'Today the chunk size is fixed per launch. KV is laid out block-cyclically over SP with period chunk/SP, RoPE is built with the same layout, the MoE dispatch buffers are sized by the chunk, and the MSA cache read requires cached_len % chunk == 0. Variable layout allows any segment length (a multiple of 32·SP tokens), and each layer\'s new K/V rows are routed to the SP device that owns them with a small all-to-all.',
    why: 'It removes padding: 15% of processed tokens at chunk 2048, and about half at 5120, on the AgentX mix. It also removes alignment loss on resume. It enables token-packed batching. The extra all-to-all is about 0.1 ms per layer.',
    todo: [
      'Absolute-position RoPE, independent of chunk size.',
      'An all-to-all KV write op, or a fixed fine-grained block-cyclic cache layout (e.g. 128-token blocks) written through a scatter.',
      'MSA cache read and ring attention without chunk-aligned starts.',
      'MoE buffers sized for the maximum budget; compile/trace buckets for variable token counts.',
    ],
    notes: 'Includes unaligned resume. Most of its value is realised together with batching.',
  },
  unaligned: {
    what: 'Resume a conversation at any 32-token boundary instead of rounding the cached prefix down to a chunk multiple. PR #57636, in review.',
    why: 'With chunk 2048 the rounding throws away 1024 cached tokens per request on average, which is a large fraction of a typical 1600-token turn. up to ×1.18 when added before the variable layout.',
    todo: ['Land PR #57636 and run the two-turn 60-layer PCC check.'],
    notes: 'The variable layout includes it, which is why it shows ×1.00 in the full stack.',
  },
  arena: {
    what: 'Instead of fixed 1M lanes, give each request a lane of its actual length from one contiguous arena per stage (2-8M tokens).',
    why: 'Frees the memory that fixed lanes reserve (4 × 1M tokens per stage) for the pool. Makes the dense scan request-sized, a substitute for bounded dense gather. Lets many small requests be in flight at once for batching. Measured ×0.99-1.03 once the other features are on (×1.7 at 4 galaxies and ×3.3 at 8 galaxies with today\'s kernels when it stands in for a missing bounded dense gather; the two are substitutes, and whichever is added first gets the credit), because they already cover these effects.',
    todo: [
      'Kernels take a base offset + length instead of a slot index.',
      'An allocator with fragmentation handling (compaction, or size classes).',
    ],
    notes: 'Low priority if bounded dense gather is done and lanes are few.',
  },
  msa: {
    what: 'Today every MSA layer all-gathers the whole cached K/V and index keys over SP for each request (ag_kv / ag_index_k). Instead, each SP rank scores its own index keys, a small top-k merge picks the 16 blocks per query, and only those blocks are fetched.',
    why: 'The gather grows with context: at [2,4] about 0.007 ms per 1k cached tokens per layer, so about 1 ms per layer at 140k, against an 18 ms layer today. The gain is ×1.01-1.03 in the full stack at SP=2; one early greedy step shows ×1.20, but that is a configuration still dominated by cache thrashing; it would matter more with larger SP or much faster kernels.',
    todo: [
      'A distributed top-k merge.',
      'A block-fetch op for the selected K/V blocks.',
      'Changes to the indexer_score_msa / sparse_sdpa_msa inputs.',
    ],
    notes: 'Not worth it at [2,4] stages.',
  },
  fused: {
    what: 'One attention kernel over all requests in a batched chunk, instead of one attention call per request.',
    why: 'It could save launches and fill cores better for short segments. The model includes both effects (per-op latency and a 110-core wave-quantisation penalty for short segments). It finds ≤2%, because batched segments average about 1.6k tokens, which already fill the cores.',
    todo: ['New multi-user MSA and ring-joint kernels.'],
    notes: 'Not recommended: sequential per-request attention is as good. That confirms the "batch the MoE, attention per request" plan.',
  },
  srpt: {
    what: 'Order the queue by new tokens (shortest first), with 30 s aging so long requests are not starved.',
    why: 'Short requests stop waiting behind 50k-token prefills, so p90 TTFT drops and more load fits under the SLO: ×1.01-1.07.',
    todo: ['Scheduler-only change.'],
    notes: 'Cheap; its gain grows near saturation.',
  },
};
