# pplx-decider-v1-27b on one Blackhole p150a: benchmark report

Implementation: `models/demos/pplx_decider_v1_27b` (`TTDecider.predict`), measured at commit
`da43bdb9eff` (stage 3A + 3B, optimized decoder), precision C0 (BFP8_B weights, BF16 activations,
HiFi2; vision tower BF16). Model revision `b01a5cbaca5391f73bd55103d4f27e8982cd5e60`. All numbers
are measured on this host on 2026-10-11 unless marked otherwise. The "before" figures are the first
stage-11 run (2026-10-09, model code `b98508eac2c`, before stage 3), with the same subsets, templates,
commands and weight cache. Machine-readable results: `results.json`, `predictions.csv`,
`subsets_manifest.json`, `identity.json` in this folder.

## Accuracy (200 seeded examples per benchmark, batch 1)

| benchmark | subset (config / split) | correct / n | TT accuracy | 95% CI (Wilson) | model card | delta vs card | before stage 3 | card, Qwen3.8-27B base |
|---|---|---:|---:|---|---:|---:|---:|---:|
| WinoGrande | winogrande_xl / validation | 174 / 200 | 87.0% | 81.6–91.0% | 83.30% | +3.70 pp | 87.0% (174) | 73.10% |
| FinancialPhraseBank | sentences_50agree / train (only split) | 168 / 200 | 84.0% | 78.3–88.4% | 84.18% | −0.18 pp | 82.5% (165) | 75.68% |
| Belebele | eng_Latn / test | 193 / 200 | 96.5% | 93.0–98.3% | 94.00% | +2.50 pp | 96.5% (193) | 93.20% |

- Every model-card figure lies inside the TT 95% interval.
- No prompt was rejected or truncated; prompt lengths are 102–317 tokens (buckets 128 and 1024).
- The model card's figures were measured through the Perplexity API with a prompt protocol that is not
  in the snapshot. The templates here are one fixed template per benchmark, written before any TT result
  was seen. Treat the deltas as indicative, not as a pass/fail gate. Agreement with HF itself is
  established separately (stage 6: 25/25 text decisions; stage 12B: 8/8 image decisions; both re-run
green on the stage-3B code).

Change against the run before stage 3 (same 600 prompts):

| benchmark | predictions changed | correct before -> after | max \|prob diff\| |
|---|---:|---|---:|
| WinoGrande | 2 | 174 -> 174 (one lost, one gained) | 0.084 |
| FinancialPhraseBank | 3 | 165 -> 168 | 0.045 |
| Belebele | 0 | 193 -> 193 | 0.042 |

Over all 600 prompts the per-prompt max |prob diff| has median 0.0017 and p90 0.014. All 5 changed
predictions are near-ties: in one of the two runs the top two options have equal probability (the
app's first-maximum rule then picks), and in the other run they differ by at most 0.08. Examples:
WinoGrande 673 went from 0.528 / 0.472 to 0.5 / 0.5, and FPB 3534 went from 0.38 / 0.32 / 0.30 to
0.333 each. The equal values are probably BF16 readout logits that round to the same value
(inferred). The +1.5 pp on FPB is three such flips. It is noise from the numerics, not a gain in
accuracy. The stage-3B numerics change (RMSNorm without fp32 accumulation, SDPA exp approximation) is
the likely source of the probability shifts (inferred). Stage 3A changed only matmul program configs.

FinancialPhraseBank confusion (rows = label, columns = TT prediction):

| label \ pred | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 27 | 2 | 0 |
| neutral | 2 | 94 | 27 |
| positive | 0 | 1 | 47 |

Most FPB errors are neutral sentences classified as positive (27 of 32 errors).

## Latency per bucket (HEAD, default path)

Burst = each pass after at least 10 s idle; sustained = back to back. Request = tokenize + upload +
forward + 255-probability readback. Stage-6 harness `tests/perf/test_model_perf.py::test_model_perf`.

| bucket | prompt tokens | request burst ms (before -> after stage 3) | request sustained ms (before -> after) | decisions/s burst | decisions/s sustained |
|---:|---:|---|---|---:|---:|
| 128 | 100 | 148.5 -> **114.8** (−22.7%) | 147.6 -> **113.8** (−22.9%) | 8.71 | 8.79 |
| 1024 | 178 | 366.0 -> **315.1** (−13.9%) | 462.3 -> **414.1** (−10.4%) | 3.17 | 2.41 |
| 2048 | 2006 | 665.4 -> **563.6** (−15.3%) | 914.5 -> **821.0** (−10.2%) | 1.77 | 1.22 |
| 4096 | 3906 | 1480.8 -> **1227.3** (−17.1%) | 1912.9 -> **1679.0** (−12.2%) | 0.81 | 0.60 |
| 8192 | 7928 | 3535.8 -> **2882.3** (−18.5%) | 3999.4 -> **3510.1** (−12.2%) | 0.35 | 0.28 |

The burst figures match the stage-3B review's request burst (114.8 / 315.2 / 564.0 / 1227.7 /
2898.9 ms) to within 0.6%.

Image requests (one image, bucket 1024):

| golden row | patches | image tokens | request burst ms (before -> after) | request sustained ms (before -> after) | answer (expected) |
|---|---:|---:|---|---|---|
| v01_dominant_color | 256 | 64 | 385.6 -> **335.5** (−13.0%) | 474.1 -> **418.9** (−11.6%) | blue (blue) |
| v04_tallest_bar | 1024 | 256 | 405.2 -> **353.8** (−12.7%) | 504.7 -> **450.2** (−10.8%) | Thu (Thu) |

## Mixed-workload throughput (the 600 benchmark prompts, back to back, batch 1)

| pass | order | wall s (before -> after) | decisions/s (before -> after) | p50 ms | p90 ms | p99 ms |
|---|---|---|---|---:|---:|---:|
| throughput | seeded shuffle | 220.9 -> **193.6** | 2.716 -> **3.099** (+14.1%) | 449.3 -> 399.9 | 473.4 -> 425.1 | 478.9 -> 432.5 |
| accuracy | file order | 222.0 -> **196.0** | 2.703 -> **3.062** (+13.3%) | 475.4 -> 429.8 | 478.8 -> 436.7 | 481.5 -> 439.7 |

By bucket in the throughput pass, p50 is 190.6 -> 151.1 ms for bucket 128 and 458.2 -> 409.5 ms
for bucket 1024. Both passes chose the same option on 600/600 prompts, with zero probability
difference.

## Roofline (burst device forward, executed bucket length)

Text projection parameters P = 24.35 B (MLP 17.11 B, DeltaNet 5.56 B, attention 1.68 B). The
embedding gather and the vision tower are excluded. FLOPs = 2 × P × bucket tokens + attention and
DeltaNet terms. Peak = 359.4 TFLOP/s at HiFi2 (inferred: 130 Tensix × 1.35 GHz, per
`tech_reports/GEMM_FLOPS`). DRAM bandwidth is 512 GB/s (spec sheet, inferred).

| bucket | total GFLOP | device forward ms (before -> after) | TFLOP/s (before -> after) | % of peak | useful share of executed FLOPs |
|---:|---:|---|---|---:|---:|
| 128 | 6,281 | 146.8 -> 113.1 | 42.8 -> 55.5 | 15.4% | 78% |
| 1024 | 50,428 | 363.7 -> 313.0 | 138.6 -> 161.1 | 44.8% | 17% |
| 2048 | 101,269 | 656.9 -> 555.0 | 154.2 -> 182.5 | 50.8% | 98% |
| 4096 | 204,187 | 1,464.4 -> 1,212.7 | 139.4 -> 168.4 | 46.8% | 95% |
| 8192 | 414,971 | 3,460.8 -> 2,857.5 | 119.9 -> 145.2 | 40.4% | 97% |

- At bucket 128 the forward reads weights at 229 GB/s (45% of DRAM bandwidth; before: 176 GB/s,
  34%). It is not compute bound.
- Sustained (back to back), the device forward runs at 34–35% of peak for buckets 1024 and above
  (before: 29–31%).
- The mixed run executes 110.7 TFLOP/s (31% of peak; before: 97, 27%). Only 24.8 TFLOP/s of that is
  on real tokens, because short prompts (178–317 tokens) are padded to the 1024 bucket.

## Protocol

- Prompts use the app's own `decision_messages` and chat template (`enable_thinking=False`), via
  `demo/decider.py`. `state` = the sentence or passage, `question.instructions` = the task question,
  `criteria` = the answer options, question type `choice`. Templates live in `benchmark/datasets.py`.
- Subsets: seed 20260920, 200 examples per benchmark. The id lists and their sha256 are in
  `subsets_manifest.json`. The refresh reused the first run's subset files, which are byte-identical
  (same sha256). FinancialPhraseBank uses `sentences_50agree` (all 4,846 sentences, majority
  label), because the card does not state the configuration.
- Commands: `python -m models.demos.pplx_decider_v1_27b.benchmark.run_benchmark` (accuracy and
  throughput), the stage-6 bucket harness `tests/perf/test_model_perf.py`, and
  `python -m models.demos.pplx_decider_v1_27b.benchmark.report` (tables and roofline). Raw outputs:
  `$PPLX_DECIDER_ARTIFACTS/stage11_refresh/` (this run, including
  `compare_vs_stage11.json`), `$PPLX_DECIDER_ARTIFACTS/stage11/` (before stage 3).

One rendered example (WinoGrande, 105 tokens, bucket 128):

```
State:
Michael just bought brand new wheels for his truck unlike Leslie because _ wheels were new and perfect.

Question:
Which option correctly fills the blank (_) in the sentence?

Options:
A: Michael
B: Leslie

Return only the letter code of the best option.
```

## Limitations

- Batch 1 only. Batched requests with left padding are not implemented (DeltaNet would need the HF
  padding-mask zeroing).
- Sustained back-to-back requests at buckets 1024 and above run 1.22–1.46× slower than burst. The
  suspected cause is clock or power throttling (inferred; device clocks were not read).
- The card's prompt protocol is unknown, so accuracy deltas against the card are indicative.
- No HF reference run on the 600 benchmark prompts; HF agreement rests on the stage-6 and 12B goldens.
- Stage 3 (optimized decoder) is done: 3A tuned the prefill matmul program configs and 3B tuned the
  non-matmul ops and chunking. 3A resolved the S=128 regression from the stage-2 fusions. The 128
  bucket is now 22.7% faster per request than before stage 3. Stage 7 (further optimization) was
  skipped by person decision.
- Stage 3B runs RMSNorm without fp32 dest accumulation (HiFi4 kept); this was kept by decision. It
  costs up to −2.3e-4 decoder-layer PCC (suite minimum 0.999742 at L63 S4096, bar 0.995) and buys
  about 1% of layer time at 2048–8192. It probably accounts for the near-tie flips above (inferred).
- The dtype sweep (stage 8) kept BFP8 everywhere. A BFP4 MLP with LoFi was 17–21% faster and freed
  about 8 GiB of DRAM, but missed the logit-PCC gate on one 230-option prompt (0.98895 < 0.99).
- Multi-image prompts are supported by the code but untested.
