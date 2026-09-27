# perf_2026_09_25 — streaming CFM round (rebuilt)

Evidence for the streaming (chunk-causal, bucketed) CFM work in `tt/flow/decoder.py`. The 2026-09-23/24 originals
of the first three scripts were lost with an instance before they were committed; these are recreations, re-run
on the N150 on 2026-09-25, not copies of the old numbers.

| script | device | what it answers |
|---|---|---|
| `masked_sdpa_investigation.py` | yes | Does fused SDPA take the padding + chunk-causal mask correctly, and how fast is it compared with the explicit chain? |
| `cfm_streaming_real_checkpoint_check.py` | yes | Real `flow.pt`: bucketed vs. exact-length streaming solve (aligned and non-aligned valid lengths, trace reuse), and the per-bucket cost |
| `cfm_trace_cache_thrashing_simulation.py` | no | Under the real hop schedule, what trace-cache capacity would help, and by how much (measured costs) |
| `nonstreaming_timing_old_vs_new.py` | yes | Is the non-streaming path's timing unchanged against the pre-change decoder (`5c65be445c`)? |
| `../perf_2026_09_23/bucket_sizing_simulation.py` (PART 3) | no | The real growing hop schedule (`real_hop_schedule_lengths`) against `GeometryWeightCache` |

Run every script with `/home/user/tt-metal/python_env/bin/python`. python_env's `.pth` files put the repo, `ttnn/` and `tools/` on `sys.path`,
so no `PYTHONPATH` is needed (`/opt/venv` lacks `graphviz`, so `ttnn` does not import there). Device scripts that
load checkpoints need `HF_HOME=/home/user/models`.

## Results (2026-09-25, N150)

**Masked SDPA, op level** (`[2, 8, T, 64]` bf16, q_chunk 128 / k_chunk 256):

- Building the mask on device as the broadcast sum `chunk[1,1,T,T] + pad[B,1,1,T]` exactly matches the host sum.
- Accuracy vs. float64: PCC 0.9997 at every T and valid length.
- Speed: 2.2x / 4.9x / 7.1x faster than the explicit chain at T = 384 / 768 / 1536. A mask costs 2.1–3.5x over no mask.
- The op's default program config (no q/k chunk sizes) is slower than q128/k256 at T = 768 and 1536. At T = 384 it is slightly faster.
- Leak probe (large values in the padded rows of V): a chunk-only mask collapses to PCC 0.13 / 0.30 / 0.76 at the non-aligned lengths. The full mask stays at 0.9997.
- With random V, the same missing padding term still scores above 0.99, which is why the probe is needed.

**Real checkpoint, 10-step streaming solve** (first run):

- The TT bucketed solve is bit-identical to the TT exact-length solve (PCC 1.0, max|diff| 0) at all six (valid, bucket) pairs: 300/310→384, 650/660→768, 1500/1510→1536.
- Against the fp32 torch chunk-causal reference: 0.9962–0.9988.
- In that run the non-aligned cases did not reuse the aligned case's trace; the script was fixed afterwards (see its docstring). The card then became unrecoverable at teardown. The fixed version was re-run on 2026-09-27 (below).

| bucket (mel) | capture | recapture (warm) | streaming steady | ms/Euler step | non-streaming steady |
|---|---|---|---|---|---|
| 384 | 734.8 | 744.4 | 501.6 | 50.16 | 406.6 |
| 768 | 1069.6 | 1071.9 | 820.4 | 82.04 | 733.4 |
| 1536 | 2071.7 | 2063.9 | 1628.4 | 162.84 | 1377.7 |

All times are ms per 10-step traced solve. Bucket 1536 is over the 1 s per-chunk budget; that is re-measured, not extrapolated.

**Trace-cache thrashing** (5 back-to-back 30 s utterances, measured costs):

- Every CFM call in an utterance lands in a distinct bucket, so a single-slot cache never hits.
- Any capacity below the session's full bucket range gives zero hits.
- A full-range cache would cut total CFM time from 64.5 s to 52.5 s (−19%). Capture overhead is 23% of single-slot time, not the ~100% the lost 09-24 analysis stated.
- A lazy multi-slot cache is not allocation-safe, so `TtCausalConditionalCFM` refuses any capacity other than 1 (see its docstring). This number is a ceiling for a future pre-warmed design, not an available win.

**Real hop schedule** (upstream `CosyVoice2Model.tts`, re-read 2026-09-25):

- The hop goes 25(+prompt pad) → 50 → 100 → 100 …, plus one finalize call. Mid-stream valid lengths are always chunk-aligned; the finalize call's length is arbitrary.
- `tts()` never resets `token_hop_len`, so later utterances on the same model object start at a hop of 100. Both variants are simulated.
- Per-utterance reset: 9 flow calls per 30 s utterance, 80.0% `GeometryWeightCache` hit rate, breaking point 96.8 MB/bucket.

## Re-runs on 2026-09-27 (N150, new pod, KMD 2.3.0)

Both scripts were written on 2026-09-25 but not run before the card went down. Both ran on 2026-09-27 against the
streaming decoder as committed, with a clean exit and a clean teardown each time.

**`cfm_streaming_real_checkpoint_check.py`, fixed version, with the pass/fail gate** (run twice, identical accuracy):

- Every non-aligned case (310 / 660 / 1510) and its chunk-only control **reused** the aligned case's trace, so the
  per-solve padding refresh on reuse is verified with real weights.
- TT bucket vs TT exact-length: **bit-identical (max|diff| 0) at all six pairs.** TT vs fp32 torch: 0.9962 / 0.9964 /
  0.9978 / 0.9978 / 0.9988 / 0.9988. The non-streaming TT-vs-torch baseline at 300 / 650 / 1500 is 0.9993 / 0.9987 /
  0.9988.
- **Why the gate is on max|diff|, not PCC:** the chunk-only control (padding visible, garbage in the padded rows)
  differs from TT exact by max|diff| 2.42 / 3.09 / 2.06, yet scores PCC 0.9986 / 0.9992 / 0.9997. That would pass
  any 0.99 PCC gate. The script now fails unless bucket vs exact is within max|diff| 0.05, every chunk-only control
  fails that same check, and every non-aligned case and control reused the aligned trace. Verdict: PASS.
- Timing, ms per 10-step traced solve, within about 1–2% of the 2026-09-25 table (the simulation below keeps the
  09-25 costs):

  | bucket (mel) | capture | recapture (warm) | streaming steady | ms/Euler step | non-streaming steady |
  |---|---|---|---|---|---|
  | 384 | 743.9 | 710.7 | 500.2 | 50.02 | 403.3 |
  | 768 | 1061.8 | 1061.3 | 813.5 | 81.35 | 728.0 |
  | 1536 | 2079.9 | 2070.3 | 1645.1 | 164.51 | 1387.2 |

**`nonstreaming_timing_old_vs_new.py`**: old decoder (`5c65be445c`) vs the streaming decoder, non-streaming,
all-ones mask, mean ms over 8 solves:

| T | mode | old | new | new/old | outputs bit-identical |
|---|---|---|---|---|---|
| 384 | eager | 616.7 | 621.2 | 1.007 | True |
| 384 | traced | 409.6 | 409.7 | 1.000 | True |
| 768 | eager | 762.9 | 763.0 | 1.000 | True |
| 768 | traced | 728.9 | 727.8 | 0.999 | True |
| 1536 | eager | 1427.0 | 1430.4 | 1.002 | True |
| 1536 | traced | 1386.5 | 1387.1 | 1.000 | True |

Every old/new min–max range overlaps, and the outputs are bit-identical: the non-streaming path is unchanged in both
numbers and timing.
