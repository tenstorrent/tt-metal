# Reference review: Thatch-cloud Blackhole Qwen

Reviewed commit `95589fb2ebdd7aa400e1daa89dcb3a8d457756bc` from
<https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B>.
Local clone: `/Users/anatarajan/Documents/Tenstorrent.Blackhole-Qwen3.8-27B`.
This was source/report inspection only. Their two P150A cards use TP2,
T16 speculative verification, and a different precision/state path. Reported
committed throughput is not interchangeable with our TP4 one-token decode.

## Candidate ideas

1. [Shared Q/K](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/shared-qk-recurrence-results.md):
   repeated verifier 68.01854 -> 66.81181 ms; complete-request gain only 0.70%
   on the repeat. Our adapter repeats Q/K three times, then the four value
   partitions each normalize again. Sharing our existing exact FP32 normalization
   is a concrete experiment; preserve FP32 recurrent state and include prep time.
2. [Direct windows](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/gdn-direct-window.md):
   roughly 3.5-ms verifier saving by removing materialized causal windows.
   Full-request controls suffered large stalls; no repeatable request gain.
   Our packed convolution also has window concatenation/layout work to inspect.
3. [Block weight stream](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/mlp-block-stream.md):
   same packed BF4 bytes, 48 small reads -> one contiguous 27,648-B span;
   verifier 62.99 -> 60.27 ms in one ABBA pilot. Extra packed pool 3,227,516,928
   bytes/card; packing 423.25 ms; setup-inclusive latency regressed. We already
   use DRAM-sharded weights; check reader granularity and KV capacity first.
4. [Bounded phase clocks](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/gdn-recurrence-phase-diagnostic.md):
   small persistent per-processor records could avoid our unbounded Tracy export.
   Intervals contain waits and overlap; do not sum them into arithmetic time.

## Negative evidence and applicability limits

- [Bulk-read overlap](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/mlp-bulk-pipeline.md)
  repeated TG 117.02 -> 116.32 and verifier 60.17 -> 60.26 ms: rejected.
- [More down-projection workers](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/mlp-down-grid.md)
  changed 32 -> 80 productive workers but only about 0.301-ms verifier saving,
  with -0.40% aggregate committed throughput. Core count is not bandwidth proof.
- [Programmable prefetch](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/dram-prefetch-verdict-2026-09-19.md)
  was slower on the tested Qwen geometry; Llama ring geometry does not transfer
  directly. Earlier mismatched-worker speedup was retracted. No firmware changes
  are authorized or needed for our source review.
- [KV multicast](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/optimisation/ttnn-op/sdpa_decode_qwen/README.md)
  requires equal page-table rows. It cannot combine independent conversations'
  caches. Their query slicing also rejects single-KV-head geometry such as ours.
- [Launch-count verdict](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/blob/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/docs/k1-launch-count-verdict.md)
  explicitly retracts its headline: the image did not contain the candidate, so
  both arms executed the same implementation. Do not infer a hardware ceiling or
  retire launch reduction from that comparison. Verify actual compiled sources.

None of these reports establishes a long-context full-Galaxy improvement for
our implementation. Shared normalization and direct windows are pending local
candidates, not completed changes.
