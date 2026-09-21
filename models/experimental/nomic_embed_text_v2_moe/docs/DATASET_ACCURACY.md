# Nomic Embed Text v2 MoE: accuracy on real datasets

Retrieval accuracy of the TTNN port against the vendored PyTorch reference on two public
datasets. Every other gate in this port uses random token ids or synthetic activations; ids
drawn uniformly from a 250k vocabulary are semantically meaningless, so they say nothing about
real text and nothing at all about retrieval quality, which is what the model is for.

Measured on a Blackhole p300c, grid 11x10, bfloat16 activations and weights with HiFi4 and
fp32 destination accumulation, weights read from the pinned checkpoint. One run, 2026-09-18.

## 1. What was measured

| Dataset | Split | Content |
|---|---|---|
| [`mteb/scifact`](https://huggingface.co/datasets/mteb/scifact) | test | English scientific claim verification, 5183 abstracts, 300 scored queries |
| [`mteb/XQuADRetrieval`](https://huggingface.co/datasets/mteb/XQuADRetrieval) | validation | 12 languages, 6 scripts, 240 passages and ~1185 queries each |

SciFact is one of the 15 BEIR subsets Nomic report in the paper, so its absolute number is
comparable to published work. XQuAD is the same SQuAD content translated into every
language, which is why it beats MIRACL for a port check: content is held fixed, so a
per-language difference is attributable to the language rather than to question difficulty.
MIRACL's per-language query sets are independent and confound the two.

Both sides were driven through identical tokenization and pooling, with the trained task
prefixes applied (`search_document:` for passages, `search_query:` for queries). Texts were
batched length-sorted within 32 rows and 4096 tokens. 22562 encodes per side, 45124 total.

nDCG@10 is scored against each dataset's own qrels, so the reference column is an absolute
retrieval number and the delta is the port's cost.

## 2. Results

| Split | docs | queries | nDCG@10 ref | nDCG@10 TT | delta | worst 1-cos | mean 1-cos |
|---|---|---|---|---|---|---|---|
| scifact | 5183 | 300 | 0.7288 | 0.7246 | -0.0042 | 9.57e-02 | 3.88e-03 |
| xquad-ar | 240 | 1186 | 0.9375 | 0.9371 | -0.0004 | 7.75e-02 | 1.81e-03 |
| xquad-de | 240 | 1181 | 0.9639 | 0.9625 | -0.0014 | 1.08e-01 | 1.17e-03 |
| xquad-el | 240 | 1184 | 0.9554 | 0.9540 | -0.0014 | 3.43e-02 | 8.03e-04 |
| xquad-en | 240 | 1185 | 0.9773 | 0.9777 | +0.0004 | 5.97e-02 | 9.54e-04 |
| xquad-es | 240 | 1184 | 0.9691 | 0.9678 | -0.0013 | 7.71e-02 | 1.75e-03 |
| xquad-hi | 240 | 1183 | 0.9516 | 0.9505 | -0.0011 | 1.50e-01 | 2.57e-03 |
| xquad-ro | 240 | 1184 | 0.9665 | 0.9640 | -0.0025 | 5.07e-02 | 1.08e-03 |
| xquad-ru | 240 | 1185 | 0.9555 | 0.9551 | -0.0003 | 8.23e-02 | 1.31e-03 |
| xquad-th | 240 | 1180 | 0.9532 | 0.9527 | -0.0004 | 8.59e-02 | 1.83e-03 |
| xquad-tr | 240 | 1184 | 0.9520 | 0.9504 | -0.0015 | 7.29e-02 | 8.77e-04 |
| xquad-vi | 240 | 1182 | 0.9565 | 0.9573 | +0.0008 | 6.99e-02 | 1.14e-03 |
| xquad-zh | 240 | 1181 | 0.9590 | 0.9581 | -0.0009 | 5.52e-02 | 1.37e-03 |

nDCG@10 delta: worst -0.0042, best +0.0008, mean -0.0011, positive on 2 of 13 splits.

Robustness: no hangs, no non-finite values, and every embedding norm within [0.9950, 1.0047]
across all 45124 encodes. No board reset was needed at any point.

Two languages score above the reference. A port cannot beat its oracle, so that is the useful
part of the signal: the deltas are noise around zero, not a one-directional loss.

## 3. The per-row cosine tolerance does not hold on real data

[`tests/pcc/test_ttnn_model.py`](../tests/pcc/test_ttnn_model.py) gates the port on
`COSINE_TOLERANCE = 0.01`, a per-row bound on `1 - cosine` between the port's pooled embedding
and the reference's, asserted on random token ids over a handful of shapes and seeds.

**Every one of the 13 splits exceeds it on its worst row.** The worst overall is 1.50e-01, 15x
the bound. The means do not: they run 8.0e-04 to 3.9e-03, inside it everywhere.

So it is a tail, not a shift, and the nDCG column shows it costs nothing: the split with the
worst tail (xquad-hi, 1.50e-01) loses 0.0011 nDCG, while the split with the largest nDCG loss
(scifact, -0.0042) has a milder tail than four others. Rows drift without reordering results.

That test's docstring puts `1 - cosine` at 9.1e-05 to 7.6e-03 over tens of rows; this is 22562,
and widening the sample moves the extreme. That is sampling, not regression. A fixed per-row
0.01 assert would fail immediately here and should not gate this data: gate on the mean or a
high quantile alongside the nDCG delta, and report the worst case rather than asserting it.
This run captured only mean and max, so a quantile has to be measured before setting one.

## 4. Short sequences are not the worst case here

The [README](../README.md) records short sequences as the weak case, on the argument that one
rerouted token is a larger share of the pooled mean: 1/74 at `2x37` against 1/1024 at `2x512`.
Aggregated by text length across all 13 splits, real data does not order that way:

| tokens | n | worst 1-cos |
|---|---|---|
| 0-32 | 14138 | 9.31e-03 |
| 33-64 | 413 | 1.50e-01 |
| 65-128 | 341 | 8.37e-02 |
| 129-256 | 2821 | 7.62e-02 |
| 257-512 | 4094 | 9.57e-02 |
| 513+, truncated to 512 | 755 | 4.66e-03 |

The shortest bucket holds under 0.01 across 14138 texts, the 33-64 bucket is the worst at
1.50e-01 across 413, and the fully packed 512-token sequences are the best behaved of all.

The two are not the same measurement: that figure is a per-shape pooled cosine over random-id
draws at fixed `B x S`, this is per-text over real inputs batched by length. Still, the recorded
ordering does not hold for real text, and the fully packed case being cleanest suggests the tail
tracks padding and batch composition rather than length.

## 5. Reproducing

From a standalone harness rather than the test suite, since none of these are asserted:

- load the configs above through `datasets`, with `{lang}-corpus`, `{lang}-queries` and
  `{lang}-qrels` for XQuAD (split `validation`) and `corpus`, `queries`, `default` for SciFact
  (split `test`)
- encode with `reference.embedding.encode` and `tt.model.encode`, same texts, same prefixes
- score nDCG@10 against the qrels on both sides, and compare the embeddings row by row

Rerun cost: the reference side is the constraint at roughly 11 texts/s on this 16-core host
with no GPU, about 35 minutes for its 22562 encodes; the device side is several minutes. The
reference embeddings are deterministic at a pinned revision, so caching them makes every rerun
after the first device-only.
