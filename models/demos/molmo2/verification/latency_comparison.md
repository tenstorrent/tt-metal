# Molmo2-8B latency / accuracy vs. earlier reference runs

Suite: the 105 multiple-choice video requests in `test.jsonl` (temperature 0, max_tokens 16), run through
`run_suite.py` against OpenAI-compatible servers. Date of the new runs: 2026-10-14.

## What the reference numbers are

| Reference | Source | Hardware / stack | Notes |
|---|---|---|---|
| HF baseline | `test_results.jsonl` (`runtime_sec`) | HF PyTorch. The hardware is not recorded anywhere in this folder. | 100 valid entries (5 baseline entries errored with empty responses). |
| TT T3K, Docker | `video_test_results_docker.md` / `_full.md` (2026-03-28) | T3K, vllm-tt-metal docker `0.11.0-3035237ebd`, video prefill untraced | 105/105 ok, mean 10.27 s. The old log says the end-to-end time includes ~5–8 s of GCS download. |
| TT T3K demo | `comprehensive_test_results_20260331.md` | T3K, demo, traces on | decode 32.7–35.5 tok/s, TTFT 373 ms (text) / 2.4 s (image) / 0.58 s (8-frame video). |

## New runs (tt-metal rebased on main, vllm-tt-plugin + vLLM 0.30.0, Galaxy 1x8 slices, batch 1 per server)

Per-request latency, seconds, wall time as seen by the client (each request fetches its video over https):

| | n | mean | median | p90 | max |
|---|---|---|---|---|---|
| HF baseline (`test_results.jsonl`) | 100 | 3.61 | 3.11 | 6.38 | 9.64 |
| TT T3K Docker 2026-03-28 | 105 | 10.27 | – | – | – |
| **New, 3 servers (slices 1-3)** | 105 | **3.62** | 2.65 | 7.27 | 10.93 |
| **New, 4 servers (slices 0-3)** | 105 | 3.76 | 3.11 | 6.88 | 11.19 |

Per test, new/HF-baseline ratio: median 1.03x (faster on 47 of 100 tests). Versus the 2026-03 T3K Docker run the mean
is 2.8x lower (10.27 s -> 3.62 s); that older figure also included the video download, as do the new ones.

Device-side (server logs, 3 servers, 213 video prefills / 1,616 decode steps):

| | new | earlier reference |
|---|---|---|
| prefill, median (S median 2.4k tokens, 30-frame-class videos) | 1.67 s (mean 2.05 s, p90 3.49 s) | not comparable (reference TTFTs are for 8-frame videos) |
| decode, median step | 52 ms = **19.2 tok/s** (mean 54 ms) | **~35 tok/s** (T3K demo, 2026-03-31) |

**Decode throughput is lower than the earlier reference (19 vs ~35 tok/s).** It was not reproduced on this stack with the
traced decode; an untraced decode measured ~14 tok/s. The server reads the full logits back to the host each step, and
`lm_head`/`wo` are replicated per chip; see the optimization notes in BRINGUP_LOG.

## Accuracy against the baseline answers (first A-D letter)

| Run | correct | notes |
|---|---|---|
| 3 servers, run 1 | 96/100 | included one request that never returned (slice 2, first video after start-up, vision pooling) |
| 3 servers, run 2 | **97/100** | misses: idx 22, 26, 84 |
| 4 servers, run 3 | **97/100** | same misses |

README's earlier claim was 98/100 (misses 22, 26, 55, 56, 76). 22 and 26 are shared; 84 is new; 55/56/76 now pass.

Raw per-request results of run 2: `results_galaxy_slices_main_2026-10-14.jsonl`.

## Caveats

* The new latencies were measured with 3-4 servers receiving requests concurrently (one in flight per server);
  a clean one-request-at-a-time pass was not possible because the Galaxy needed a power cycle afterwards.
* The new numbers are from Galaxy slices (8 Wormhole chips each), the old Docker numbers from a T3K; both are 8-chip
  Wormhole meshes, but not the same host.
* Slice 0 of this Galaxy hangs on long prefills (see BRINGUP_LOG, Session 11): requests up to ~5k tokens were fine,
  a 34k-token video stalled it.
