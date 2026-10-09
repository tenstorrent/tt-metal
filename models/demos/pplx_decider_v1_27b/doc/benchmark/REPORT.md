# pplx-decider-v1-27b on one Blackhole p150a: benchmark report

Implementation: `models/demos/pplx_decider_v1_27b` (`TTDecider.predict`), measured at commit
`b98508eac2c`, precision C0 (BFP8_B weights, BF16 activations, HiFi2; vision tower BF16).
Model revision `b01a5cbaca5391f73bd55103d4f27e8982cd5e60`. All numbers are measured on this host on
2026-10-09 unless marked otherwise. Machine-readable results: `results.json`, `predictions.csv`,
`subsets_manifest.json`, `identity.json` in this folder.

## Accuracy (200 seeded examples per benchmark, batch 1)

| benchmark | subset (config / split) | correct / n | TT accuracy | 95% CI (Wilson) | model card | delta vs card | card, Qwen3.8-27B base |
|---|---|---:|---:|---|---:|---:|---:|
| WinoGrande | winogrande_xl / validation | 174 / 200 | 87.0% | 81.6–91.0% | 83.30% | +3.70 pp | 73.10% |
| FinancialPhraseBank | sentences_50agree / train (only split) | 165 / 200 | 82.5% | 76.6–87.1% | 84.18% | −1.68 pp | 75.68% |
| Belebele | eng_Latn / test | 193 / 200 | 96.5% | 93.0–98.3% | 94.00% | +2.50 pp | 93.20% |

- Every model-card figure lies inside the TT 95% interval.
- No prompt was rejected or truncated; prompt lengths are 102–317 tokens (buckets 128 and 1024).
- The model card's figures were measured through the Perplexity API with a prompt protocol that is not
  in the snapshot. The templates here are one fixed template per benchmark, written before any TT result
  was seen. Treat the deltas as indicative, not as a pass/fail gate. Agreement with HF itself is
  established separately (stage 6: 25/25 text decisions; stage 12B: 8/8 image decisions).

FinancialPhraseBank confusion (rows = label, columns = TT prediction):

| label \ pred | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 26 | 2 | 1 |
| neutral | 2 | 92 | 29 |
| positive | 0 | 1 | 47 |

Most FPB errors are neutral sentences classified as positive (29 of 35 errors).

## Latency per bucket (current HEAD, default path)

Burst = each pass after at least 10 s idle; sustained = back to back. Request = tokenize + upload +
forward + 255-probability readback.

| bucket | prompt tokens | request burst ms | request sustained ms | decisions/s burst | decisions/s sustained |
|---:|---:|---:|---:|---:|---:|
| 128 | 100 | 148.5 | 147.6 | 6.73 | 6.78 |
| 1024 | 178 | 366.0 | 462.3 | 2.73 | 2.16 |
| 2048 | 2006 | 665.4 | 914.5 | 1.50 | 1.09 |
| 4096 | 3906 | 1480.8 | 1912.9 | 0.68 | 0.52 |
| 8192 | 7928 | 3535.8 | 3999.4 | 0.28 | 0.25 |

Image requests (one image, bucket 1024):

| golden row | patches | image tokens | request burst ms | request sustained ms | answer (expected) |
|---|---:|---:|---:|---:|---|
| v01_dominant_color | 256 | 64 | 385.6 | 474.1 | blue (blue) |
| v04_tallest_bar | 1024 | 256 | 405.2 | 504.7 | Thu (Thu) |

## Mixed-workload throughput (the 600 benchmark prompts, back to back, batch 1)

| pass | order | wall s | decisions/s | p50 ms | p90 ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| throughput | seeded shuffle | 220.9 | 2.716 | 449.3 | 473.4 | 478.9 |
| accuracy | file order | 222.0 | 2.703 | 475.4 | 478.8 | 481.5 |

The two passes chose the same option on 600/600 prompts with zero probability difference.

## Roofline (burst device forward, executed bucket length)

Text projection parameters P = 24.35 B (MLP 17.11 B, DeltaNet 5.56 B, attention 1.68 B); embedding
gather and the vision tower excluded. FLOPs = 2 × P × bucket tokens + attention and DeltaNet terms.
Peak = 359.4 TFLOP/s at HiFi2 (inferred: 130 Tensix × 1.35 GHz, per `tech_reports/GEMM_FLOPS`);
DRAM 512 GB/s (spec sheet, inferred).

| bucket | total GFLOP | device forward ms | TFLOP/s | % of peak | useful share of executed FLOPs |
|---:|---:|---:|---:|---:|---:|
| 128 | 6,281 | 146.8 | 42.8 | 11.9% | 78% |
| 1024 | 50,428 | 363.7 | 138.6 | 38.6% | 17% |
| 2048 | 101,269 | 656.9 | 154.2 | 42.9% | 98% |
| 4096 | 204,187 | 1,464.4 | 139.4 | 38.8% | 95% |
| 8192 | 414,971 | 3,460.8 | 119.9 | 33.4% | 97% |

- At bucket 128 the forward reads weights at 176 GB/s (34% of DRAM bandwidth); it is not compute bound.
- Sustained (back to back), the device forward drops to 29–31% of peak for buckets of 1024 and above.
- The mixed run executes 97 TFLOP/s (27% of peak) but only 21.7 TFLOP/s on real tokens, because short
  prompts (178–317 tokens) are padded to the 1024 bucket.

## Protocol

- Prompts use the app's own `decision_messages` and chat template (`enable_thinking=False`), via
  `demo/decider.py`. `state` = the sentence or passage, `question.instructions` = the task question,
  `criteria` = the answer options, question type `choice`. Templates live in `benchmark/datasets.py`.
- Subsets: seed 20260920, 200 examples per benchmark; id lists and their sha256 are in
  `subsets_manifest.json`. FinancialPhraseBank uses `sentences_50agree` (all 4,846 sentences, majority
  label), because the card does not state the configuration.
- Commands: `python -m models.demos.pplx_decider_v1_27b.benchmark.run_benchmark` (accuracy and
  throughput), the stage-6 bucket harness `tests/perf/test_model_perf.py`, and
  `python -m models.demos.pplx_decider_v1_27b.benchmark.report` (tables and roofline). Raw outputs:
  `/local/ttuser/gtobar/artifacts/pplx_decider/stage11/`.

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
- Sustained back-to-back requests run 1.25–1.36× slower than burst; the cause is suspected clock or
  power throttling (inferred, device clocks not read).
- The card's prompt protocol is unknown, so accuracy deltas against the card are indicative.
- No HF reference run on the 600 benchmark prompts; HF agreement rests on the stage-6 and 12B goldens.
- Optimization stages 3 and 7 were deferred: no program-config or sharding tuning was done, and the
  stage-2 fusions made the 128 bucket 12–14% slower per layer (about +17 ms per request).
- The dtype sweep (stage 8) kept BFP8 everywhere. A BFP4 MLP with LoFi was 17–21% faster and freed
  about 8 GiB of DRAM, but missed the logit-PCC gate on one 230-option prompt (0.98895 < 0.99).
- Multi-image prompts are supported by the code but untested.
