# Denoise SDPA q-chunk sweep — 2026-07-24

Roadmap item 1 from the #51080 analysis: raise `GEMMA4_PREFILL_SDPA_QCHUNK` from 32 so the
denoise SDPA's `q_num_chunks` drops 8 → 2, on the theory that the SDPA's per-q-chunk K/V
re-stream is the dominant `p_max`-proportional term.

**Verdict: bit-exact CONFIRMED, predicted speedup REFUTED at this configuration. Do not change
the default.**

## What the knob does

`_denoise_sdpa_program_config` (`tt/diffusion_attention.py:71-92`) reads the env var and passes
it through `_largest_tile_divisor(q_seq_len=256, ...)`:

| `GEMMA4_PREFILL_SDPA_QCHUNK` | `q_chunk_size` | `q_num_chunks` | work units (4 local Q heads) |
|---|---|---|---|
| 32 (shipped) | 32 | 8 | 32 |
| 64 | 64 | 4 | 16 |
| 128 | 128 | 2 | 8 |
| 256 | 256 | 1 | 4 — **half of the 8×1 grid idle** |

The knob is live on this path: `DG_SDPA_FULLCANVAS=1` (default) sets `chunk_size = q_seq_len`,
so `denoise_attention` takes the single-SDPA branch with this program config.

## Method

`demo/serving_smoke.py`, full 30 layers, `--gumbel-mode argmax` (deterministic),
`--disable-eos-stop`, `--max-seq-len 2048`, 4 blocks × 12 denoise steps, 2 **interleaved**
repetitions per config (32,64,128,32,64,128) so device drift is spread across configs rather
than absorbed by whichever ran first. Harness: `doc/optimize_perf/sweep_denoise_qchunk.sh`.

Steady state = `mean(per_block_latency_s[1:])`. Block 0 is discarded because it carries program
compilation — measuring it is what produced a bogus "+50% regression" on a first pass
(block0 ≈ 5.5 s vs steady ≈ 1.7 s).

## Results

All six runs produced identical committed tokens: `committed_sha256 = 7452a6b3f9b6af39…`,
1 distinct value across all configs and reps.

| QCHUNK | n | steady mean (s) | per-rep | vs 32 |
|---|---|---|---|---|
| 32 | 2 | 1.706 | [1.937, 1.476] | — |
| 64 | 2 | 1.711 | [1.919, 1.503] | +0.3% |
| 128 | 2 | 1.693 | [1.902, 1.484] | −0.8% |

No SDPA fallback fired in any run (`_FALLBACK_COUNTS` / `_warn_sdpa_fallback_once` silent), so
the L1 CB-clash path that would have explained a regression is ruled out.

## Reading

* **Bit-exactness holds.** q-chunking regroups the Q axis across cores and leaves the flash
  K-reduction order untouched (`k_chunk_size` unchanged), and the committed tokens confirm it
  over 6 runs. The knob is safe to use.
* **The predicted win does not appear.** The spread between configs (−0.8%…+0.3%) is far
  smaller than the spread *within* a config across reps (1.937 vs 1.476 s, ~30%). n=2 cannot
  resolve a sub-1% effect, and there is no effect worth resolving here. Note rep 1 was slower
  than rep 2 for **all three** configs — consistent drift, which is exactly what interleaving
  was there to expose.
* **This test structurally cannot show the predicted win.** On the eager path the prefix reader
  reads `prompt_len`, not `p_max`, so `k_seq_len` only grows 288 → 1056 across the four blocks.
  The SDPA re-stream term is proportional to the *fixed* span, which only exists on the
  up-front traced path. A large-span retest needs `serving_smoke --upfront --reveal-pmax <big>`.

## Action

Leave `GEMMA4_PREFILL_SDPA_QCHUNK` at 32. Re-test at a large fixed span on the traced path
before spending anything further on this lever; if a win shows up there it is a long-context
lever only, not a fix for the shipped 4096-class config.
