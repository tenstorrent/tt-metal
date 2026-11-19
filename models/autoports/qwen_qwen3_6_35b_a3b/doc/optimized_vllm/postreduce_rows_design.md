# Prefix-row post-combine reduction candidate

Status: source prepared for root-owned build and hardware validation; no performance or hardware correctness claim. Default off. Source-only hypothesis follows `remaining_grouped_dataflow.md`.

## API and scope

`ttnn.experimental.deepseek_prefill.post_combine_reduce(..., valid_token_rows=0)` preserves the full existing path. Values 1 through 32 request a prefix of a single physical 32-token chunk; 32 is an explicit identity control. The output shape remains unchanged, with omitted rows explicitly positive zero. The caller guarantees those rows are padding. Neither their combined inputs nor routing data is read. This is not a general sparse-row API, and it never skips an inactive row inside the valid prefix.

Cache-hit validation requires the dispatch-table path (indices and table), no row-indexed combine, an expert dimension immediately before embedding, at least one expert, exactly 32 physical tokens, and a positive embedding width divisible by 1024. The existing BF16/layout/metadata checks remain. The option is a `PostCombineReduceParams` operation attribute and is forwarded through public C++, primitive C++, nanobind, and every kernel's compile-time arguments, distinguishing cached programs for different prefix lengths. Default 0 leaves the original loops, output size, allocation sizes, and communication protocol intact.

## Arithmetic and data movement

The reader transfers only the prefix's expert rows. The writer loads only its indices and per-token weights and streams only its expert weight tiles. Compute uses the original expert iteration, dispatch-table lookups, first-local initialization, scalar multiply, BF16 packing and accumulation, including the original no-local-expert case for every valid row. The full 32-row tilize/output write still executes.

At Qwen width 2048 and top-k 8, combined-input reads decrease from 1 MiB per chunk to 64 KiB for B2 or 128 KiB for B4. Omitted scratch zeroing writes 120 KiB or 112 KiB of local L1 respectively. This is a source-derived byte count, not measured latency or a roofline denominator change. Upstream combine initialization remains unchanged.

## Circular-buffer ordering

c17 is still allocated for one full chunk; compute is its sole reserve/push producer. Before publishing the first c0 input, reader zeroes only c17's omitted-row tail and completes `write_zeros_l1_barrier()`. This is safe because:

1. Host validation permits exactly one chunk, hence c17's initial write pointer is its allocation base and cannot wrap before initialization. Reader also asserts the single-chunk/start-zero condition when zeroing.
2. Compute reserves c17 without advancing its write pointer, then waits on c0 before its first pack. c0 publication therefore orders the completed tail zeroing before any compute pack or c17 publication.
3. Compute packs only the disjoint prefix. No writer touches c17. There is no simultaneous write to a shared byte, and reader never alters c17's CB counters.
4. At least one valid row and one expert are required. Compute consequently cannot publish/tilize c17 without consuming an input whose publication followed zero completion.
5. Width divisibility by 1024 makes each scratch row exactly `emb_dim_cb_tiles * BF16_tile_bytes`, with no ambiguous per-row packing gap. The full output-tail zeros reach tilization.

Indices c3 reserve/push/wait/pop counts agree on the prefix length; full c17/c16 counts remain unchanged. Every replay starts new kernel CB state. No new semaphore, host read, blocking device readback, trace input or external cache ownership is introduced.

## Required discriminator and acceptance gates

- Eager and nonblocking traced exact BF16 output bits on all four chips, comparing default 0 against 1, 2, 4, 31 and 32. For full-output equality, omitted rows must have exact zero combined inputs and off-rank IDs, matching the caller's existing initialized carrier.
- Separately poison omitted combined inputs with stale finite/nonfinite/signed-zero values and omitted routing data; the specialized path must still write exact positive-zero output tails. Do not demand default-output equivalence for deliberately violated padding semantics.
- Valid rows cover no local expert, one local expert at first/middle/last slot, several local experts, asymmetric per-rank ownership, mixed BF16 values and signed zeros. Compare exact bits, not only PCC or token identity.
- Exercise legacy one-weight-per-page and packed per-token weights. Preserve all input buffers; retain old output tensors while launching other prefix lengths.
- Reuse traces with changed then unchanged inputs, scores, IDs and dispatch ownership; interleave prefix programs so wrong cache identity or stale tail contents are exposed. Include the full32 positive control before and after short-prefix runs.
- Validate that >32, multi-chunk input, weight-only skip, row-indexed combine, nonadjacent expert dimension, zero experts and unsupported embedding width fail before launch. Negative values must fail unsigned nanobind conversion.
- Run Watcher and trace-allocation tracking with no program-cache skip; collect clean trace release, mesh close and device-holder checks. No profiling.
- Only after microgates pass: reduced B2/B4/B32 full-state/stale-input probe, full40 exact logits and serial96 against unchanged control, then matched full-context native and real vLLM metrics. Keep or reject using serving evidence; do not infer a performance win from this source audit.
