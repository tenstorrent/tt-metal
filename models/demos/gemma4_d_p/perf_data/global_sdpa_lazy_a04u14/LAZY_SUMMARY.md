# Lazy rescale for Gemma4 global ring SDPA (bh-glx-110-a04u14, 2026-10-21)

Base: `mmanzoor/gemma4-gsdpa-ksplit8-q128-v-from-k` (`d1fd17139e7`). Layer benchmark
`test_prefill_layer_perf_chunk_n[...chunk{N}-global-sz{2048,4096,8192}-ctx_256k-8x4]` at a ~133k-token prefix
(chunk 64 / 32 / 16), global ring SDPA median over 32 devices. Raw log: `results.txt`.

## Measured

| Chunk | Reference | Rescale removed (ceiling, wrong numerics) | Ceiling | Row groups with unchanged max |
|---:|---:|---:|---:|---:|
| 2K | 2354.4 us | 2101.9 us | -10.7% | 87.0% (99.8% with factors >= 2^-4) |
| 4K | 3655.0 us | 3225.3 us | -11.8% | 86.4% (99.8% with factors >= 2^-4) |
| 8K | 8341.2 us | 7604.0 us | -8.8% | not measured (counting patch did not fit, below) |

- Counting overhead: reading the max / factor tiles from L1 on the UNPACK thread made the op 7.5% (2K) and 9% (4K)
  slower (`glazy_count.patch`). A per-step L1-scalar decision costs most of the gain; the decision has to come
  from the SFPU/FPU as one flag, or per row group closer to its use.
- 8K is at the kernel config buffer limit: the counting patch was 1440 B over (72096 > 70656 B); the
  v-from-k-overlap branch hit the same limit there.

## Implementation: `mmanzoor/gemma4-gsdpa-lazy-rescale` (`df515ed52d6`, WIP, DEADLOCKS)

On an eligible step (not first / last, no save CBs, no sliding window, not in-place latent V, not head-serial) the
UNPACK thread compares prev.max and cur.max bit-for-bit and mailboxes the decision to MATH and PACK. A skip step packs
P@V with L1 accumulation onto prev.out (single-entry accumulator CB, so its write pointer is back on its data),
corrects only the sums (`salad_correct_sum_only`), and swaps the out halves so the caller's prev/cur swap keeps the
accumulator. Counters LAZY-STEPS / LAZY-SKIPS (`lazy.py`).

Result: SDPA Gemma4 global accuracy suite passed 3 cases, then hung in the q128 x 5-band case at chunk 3 (likely
the first chunk with skips). Killed after ~55 min (rc 143). Not root-caused: suspect a CB count mismatch on the skip
path, or pack->unpack ordering of the in-place writes before the next step reads them. No tt-triage capture.
Perf / e2e / KV PCC not run (stopped by the user).

## Next

1. Reproduce the hanging case alone under `timeout` and capture with tt-triage.
2. Decision on the SFPU/FPU (one flag), or per row group, to remove the 7.5-9% overhead.
3. Code size for 8K, or enable only where it fits.
4. Threshold variant (keep a stale max unless it rises by > ln 2^4): ~100% skippable.
5. Rebase onto main first: main #61291 makes k_splits exact and rejects segmented accumulation with a K split.

## Files

`patches/glazy_count.patch` (count only), `patches/nosalad.patch` (ceiling), `glazy.py`, `lazy.py`, `runner.sh`,
`queue.txt`, `results.txt`, per-run logs and tracy reports.
