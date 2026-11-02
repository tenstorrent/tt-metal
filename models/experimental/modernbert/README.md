# ModernBERT

## Platforms
Wormhole (N300)

All results below were produced on an N300. N150 and Blackhole have not been run.

## Introduction
[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (Answer.AI / LightOn, Dec 2024) is a modern encoder-only transformer and the current drop-in replacement for BERT for retrieval, embeddings, reranking and classification ([paper](https://arxiv.org/abs/2412.13663)). tt-metal already supports the older BERT family (`bert`, `sentence_bert`, `bge_large_en`) but had no modern encoder.

This demo brings up `answerdotai/ModernBERT-base` (~149M params, Apache-2.0) for [bounty #50522](https://github.com/tenstorrent/tt-metal/issues/50522).

Model source: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), pinned at revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`.

## Model Details

| Parameter | Value |
| -- | -- |
| Parameters | 149,014,272 |
| Layers | 22 |
| Hidden size | 768 |
| Intermediate size | 1152 (`Wi` emits 2× for the GeGLU split) |
| Attention heads | 12 (head_dim 64) |
| Vocab | 50368 |
| Context length | 8192 (this demo is validated to 768) |
| Positional encoding | RoPE, dual theta — global 160000, local 10000 |
| Activation | GeGLU (gated GELU) |
| Attention | Full attention every 3rd layer; local sliding window otherwise |
| Bias | Bias-free everywhere except the final MLM decoder |

### Architecture notes

These are the details that differ from the older BERT demos, each verified against
the HuggingFace reference on device:

- **Layer 0 has no attention LayerNorm.** HF uses `nn.Identity()` there; the paper states the first attention LayerNorm is removed because the post-embedding LayerNorm already normalises the input. The checkpoint therefore holds 134 encoder tensors, not 135.
- **The local band is ±64 tokens (width 129).** `config.local_attention` is 128. HF's `attn.sliding_window` attribute holds 65, which is an internal half-representation and is *not* the band width.
- **GeGLU applies the activation to the first half.** `input, gate = Wi(x).chunk(2)`, then `Wo(gelu(input) * gate)` — inverted relative to the common SwiGLU convention.
- **RoPE thetas live in `config.rope_parameters[layer_type]["rope_theta"]`** in transformers 5.x; the `global_rope_theta` / `local_rope_theta` names do not exist there.
- **`Wqkv` is fused** and reshapes to `(B, S, 3, heads, head_dim)`, so Q/K/V split contiguously.
- **The MLM decoder is the only biased layer**, and its weight is tied to the token embeddings.

## Layout

```
models/experimental/modernbert/
├── README.md
├── common.py                       # config, weights, tokenizer, sample inputs
├── demo/
│   └── demo.py                     # masked-LM and embedding-similarity demos
├── reference/
│   └── modernbert.py               # from-scratch torch reference (not an HF wrapper)
├── tt/
│   ├── model_config.py             # precision policy
│   ├── weights.py                  # torch -> ttnn weight preparation
│   ├── modernbert_embeddings.py
│   ├── modernbert_rope.py
│   ├── modernbert_masks.py
│   ├── modernbert_attention.py     # global and local variants
│   ├── modernbert_mlp.py           # GeGLU
│   ├── modernbert_layer.py
│   ├── modernbert_model.py
│   └── modernbert_head.py          # MLM prediction head + decoder
├── runner/
│   ├── performant_runner.py        # trace / 2CQ / trace+2CQ dispatch
│   └── performant_runner_infra.py  # model, weights and reference for the runner
└── tests/
    ├── pcc_utils.py
    ├── test_model_config.py            # precision and core-grid policy, no device
    ├── test_reference_parity.py        # torch reference vs HF, plus negative controls
    ├── test_reference_coverage.py      # batching, padding, bf16
    ├── test_ttnn_embeddings.py
    ├── test_ttnn_rope.py
    ├── test_ttnn_mlp.py
    ├── test_ttnn_attention.py
    ├── test_ttnn_layer.py
    ├── test_ttnn_model.py
    ├── test_ttnn_mlm.py
    ├── test_modernbert_perf.py         # end-to-end timing
    ├── test_modernbert_device_perf.py  # device-kernel throughput assertion
    ├── test_modernbert_performant.py   # trace only, 2CQ only, and both
    └── test_modernbert_profile.py      # single forward pass, profiler target
```

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Model weights are downloaded automatically from HuggingFace on first run (public, Apache-2.0, ~600 MB)

## How to Run

Demos — masked-token prediction and embedding similarity, both printed alongside the HuggingFace reference:

```
pytest --disable-warnings models/experimental/modernbert/demo/demo.py
```

Full model correctness:

```
pytest --disable-warnings models/experimental/modernbert/tests/test_ttnn_model.py
```

Masked-LM head, logits and top-1 predictions:

```
pytest --disable-warnings models/experimental/modernbert/tests/test_ttnn_mlm.py
```

Everything under `tests/`. `demo.py` is separate: it does not match pytest's
`test_*.py` pattern, so the suite does not cover it and it must be run by its
explicit path.

```
pytest --disable-warnings models/experimental/modernbert/tests/
```

## Accuracy

Measured on N300, `transformers==5.10.2`. Matmul weights are bfloat8_b; activations,
the embedding table and the norm weights are bfloat16. PCC is against the fp32
HuggingFace reference unless stated otherwise.

| Component | seq 256 | seq 512 |
| -- | -- | -- |
| Embeddings | 0.99999552 | 0.99999555 |
| RoPE (both thetas) | 0.99999426 | 0.99999407 |
| GeGLU MLP | 0.99963329 | 0.99962656 |
| Attention — global | 0.99976246 | 0.99976627 |
| Attention — local | 0.99985688 | 0.99985822 |
| Encoder layer 0 / 1 | 0.99977086 / 0.99975562 | — |
| **Full model (`last_hidden_state`)** | **0.99328900** | **0.99220112** |
| MLM logits | 0.99364840 | — |

Full model under other conditions, all at seq 256 unless noted:

| Condition | PCC |
| -- | -- |
| batch 2 / batch 4 | 0.99328900 / 0.99443195 |
| padded input (200 real of 256), unpadded region | 0.99479703 |
| seq 768 | 0.99361037 |
| vs bf16 reference (isolates TTNN's own error) | 0.99441069 |

**The binding number is seq 512 at 0.99220112**, which clears the 0.99 gate by
2.20e-03 — the thinnest margin anywhere in the model, and the one to watch if
anything further is traded for speed.

Two deliberate trades account for the distance from a pure-bfloat16 build:
bfloat8_b weights cost 3.4e-03 of MLM PCC for 8.7% at b1s256, and the tanh GELU a
further 1.7e-03 for 9.0% at b8s256. The L1 attention chain cost nothing — it changes
placement, not arithmetic, so PCC is identical to eight decimals.

**Masked-token top-1 agreement is 8/8** on a 256-token passage and 4/4 on the demo
sentences, and has not moved through any of those trades. `test_ttnn_mlm`, not
hidden-state PCC, is the gate — see the bfloat8_b-activations row under Precision
policy for what hidden-state PCC misses.

The torch reference reproduces HuggingFace at PCC 1.0000000000, so it contributes
no error of its own. The fp32 figures above still include the bfloat16 penalty; the
bf16 row is the one that isolates TTNN.

### Negative controls

Correctness tests are paired with deliberately broken variants, so a passing suite
means something. Each control breaks one architectural detail; the tests assert the
result falls below threshold, and the values observed are:

| Broken detail | PCC |
| -- | -- |
| GeGLU gate order swapped | 0.2447 |
| Q/K/V split permuted | 0.5080 |
| Rotary theta shared across layer types | 0.8863 |
| Sliding band removed | 0.8234 |
| Band narrowed to ±32 | 0.8755 |
| Window set to 65 instead of 128 | 0.9520 |
| Embedding LayerNorm skipped | 0.9043 |
| LayerNorm applied at layer 0 | 0.9555 |

A padding control is separate from these: masking a region and comparing it against
the unmasked run scores 0.9778, confirming the mask is actually applied.

Local-attention tests run at seq ≥ 256. The band spans `[i-64, i+64]`, so removing
it changes nothing at seq ≤ 65, which the suite asserts directly. Detection also
weakens as sequences shorten: removing the band scored 0.932 at seq 128 against
0.823 at seq 256, which is why the tests sit well above the boundary.

## Performance

N300, no trace or multiple command queues. 20 timed iterations after 3 warmup,
ttnn 0.75.0 without the profiler. Each figure is the **median** of the three repeat
runs listed under "Reproducibility" below, not the best one.

| Batch | Seq | Stage 1 | current | change | sequences/s | tokens/s |
| -- | -- | -- | -- | -- | -- | -- |
| 1 | 256 | 13.81 ms | **10.45 ms** | −24.3% | 95.7 | 24,492 |
| 1 | 512 | 21.29 ms | **10.61 ms** | −50.2% | 94.3 | 48,269 |
| 1 | 768 | not measured | **15.34 ms** | — | 65.2 | 50,073 |
| 2 | 256 | not measured | **10.34 ms** | — | 193.4 | 49,508 |
| 4 | 256 | 44.53 ms | **15.56 ms** | −65.1% | 257.0 | 65,800 |
| 8 | 256 | not measured | **21.70 ms** | — | 368.7 | 94,391 |

Batch 8 is the throughput configuration and was added after Stage 1, so it has no
baseline to compare against. Throughput scales 95.7 → 193.4 → 257.0 → 368.7
sequences/s across batch 1, 2, 4 and 8.

The Stage 1 column was measured at this work's branch point. The base has since
moved and is slower at small batch for host-dispatch reasons unrelated to these
changes, so the small-batch deltas understate what the optimisation work bought.

**Small shapes are host-bound.** b1s256, b1s512 and b2s256 all land within 0.5% of
each other at ~10.5 ms despite b1s512 doing twice the attention work of b1s256 and
b2s256 twice the rows. That floor is host dispatch, not device time, and it is why
trace matters more at small batch than large — see *Trace and two command queues*.
Only b8s256 is device-bound, which is why it is the shape the device-perf test
asserts on.

Two changes account for most of the Stage 3 gain at b8s256, each measured as an
A/B against the configuration immediately before it:

| change | b8s256 |
| -- | --: |
| tanh GELU instead of the exact erf | −9.0% |
| attention chain in L1-interleaved | −21.7% |

See the GELU and `_L1_ATTENTION` notes in `tt/model_config.py`.

### How this compares to the existing encoders

The nearest published point in this repo is `sentence_bert`, a BERT-base backbone
at batch 8 with the same precision policy. Both figures below are single-device;
sentence_bert also publishes 772 sentences/s data-parallel across both N300 chips,
which this demo does not implement.

| | sentence_bert | this demo |
| -- | -- | -- |
| shape | b8 s384 | b8 s256 |
| sequences/s, trace+2CQ | 433 | 366.1 |
| device-kernel samples/s | 460 | 374.3 |

ModernBERT-base carries 110.3M matmul parameters per token against BERT-base's
84.9M, a factor of 1.30 — 22 layers of a narrower block versus 12 of a wider one.
Normalising for that, this demo reaches **about 73% of sentence_bert's throughput
per unit of matmul work, a gap of 1.37×.**


### Reproducibility

Run-to-run spread is shape-dependent, and knowing it is a precondition for reading
any small number here. Repeats of one unchanged configuration, separate processes:

| shape | runs (ms) | spread |
| -- | -- | --: |
| b4s256 | 15.50, 15.56, 15.65 | 1.0% |
| b8s256 | 21.66, 21.70, 21.89 | 1.1% |
| b1s256 | 10.39, 10.45, 10.57 | 1.7% |
| b1s768 | 15.31, 15.34, 15.60 | 1.9% |
| b2s256 | 10.21, 10.34, 10.46 | 2.4% |
| b1s512 | 10.46, 10.61, 10.76 | 2.9% |

**No sub-1% effect is readable at any shape on this card.** This is not cosmetic:
a 48-core SDPA grid measured 6.4% faster on the isolated op and 1.2% slower end to
end, and only b8s256's 0.05% spread made that call possible. The host-bound shapes
carry the widest spread, since they are timing dispatch rather than the device.

- **Isolated op benchmarks overstate in-model cost by 20-80%**, and can invert a
  ranking outright. They are used here as a screen; every shipped configuration was
  confirmed end to end.
- **One model per process.** A script that opens a device once and builds several
  models does not get the pytest fixture's per-test reset. The same configuration
  measured +8.5% as the fourth model built in one probe, and +29.7% after a trace
  capture, against the same configuration in a fresh process.

```
pytest --disable-warnings models/experimental/modernbert/tests/test_modernbert_perf.py -s
```

### Where the time goes

b8s256 on a profiler-enabled build: 21,339 us of device kernel, 374.9
samples/s. Profiling instruments every kernel, so this total runs slightly above
the 21.62 ms wall clock measured without it.

| op | n | us/op | total us | % |
| -- | --: | --: | --: | --: |
| Matmul | 110 | 80.2 | 8824.0 | 41.4 |
| SDPA | 22 | 153.0 | 3366.1 | 15.8 |
| RotaryEmbeddingHf | 44 | 51.5 | 2265.1 | 10.6 |
| reshards, both directions | 134 | 16.0 | 2140.2 | 10.0 |
| LayerNorm | 45 | 46.1 | 2073.5 | 9.7 |
| head create / concat | 44 | 33.6 | 1480.1 | 6.9 |
| residual adds + gate multiply | 66 | 17.3 | 1144.1 | 5.4 |
| embedding | 1 | 45.9 | 45.9 | 0.2 |

Three things in that table are not what the op names suggest:

**SDPA is two populations.** The 8 full-attention layers cost 116.7 us each and the
14 sliding-window layers 173.8 — the sliding ones are 49% *more* expensive despite
attending to half as many positions. The difference is the materialised mask,
re-read per head. That mask exists only to work around upstream #51223 item 2.

**Rotary costs more than its row.** 88 of the 134 reshards exist to give it a
sharded layout, which puts its true share near 17%. Unsharding it measured worse
(1.6% worse), so both options are poor.

**Matmul splits by where its input lives.** Both halves now read and write L1:

| in → out | n | us/op |
| -- | --: | --: |
| L1 block-sharded → L1 block-sharded (GeGLU) | 66 | 76.0 |
| L1 interleaved → L1 interleaved (attention) | 44 | 86.5 |

FPU utilisation by shape, from the `PM FPU UTIL (%)` column of the perf sheet:
Wqkv (768→2304) 70.4%, mlp Wo (1152→768) 56.5%, Wi act and gate (768→1152) 53.6%,
attn Wo (768→768) 50.1%.

The remaining gap is the shard, not the core count — the GeGLU block measures
398.4 us/layer block-sharded on 48 cores against 436.6 interleaved on 64, and
698.5 interleaved in DRAM. **The 48-core cap is not what that block is losing to.**

Zero-padding the intermediate to
1280 puts the GeGLU on all 64 cores: 1152 is 36 tiles and 36 is not divisible by 8,
while 1280 is 40 tiles, which shards 8 ways. The padding is exact — `gelu(0) * 0`
is 0 and the matching `Wo` columns are zero — so it changes core count and nothing
else. Measured at b8s256, alternating with the shipped config:

| grid | intermediate | ms | PCC |
| -- | --: | --: | --: |
| 6×8 (48 cores) | 1152 | 21.65 | 0.99428837 |
| 8×8 (64 cores) | 1280 | 21.65 | 0.99447997 |

A dead heat, so the shipped 6-wide config stays: the padding is 11% more weight
bytes per layer (2.65 → 2.95 MB), and that extra DRAM traffic cancels the compute
the extra 16 cores save. The block is bound by weight bandwidth, not core count.

To regenerate the breakdown:

```
python -m tracy -p -r -v -m pytest models/experimental/modernbert/tests/test_modernbert_profile.py
```

### Core count is set by tile-rows, not by tuning

Reading a low `CORE COUNT` as a tuning opportunity is the obvious mistake here, so
it is worth stating what the number actually tracks. Measured at seq 256:

| op | b1, 256 rows | b4, 1024 rows | b8, 2048 rows |
| -- | --: | --: | --: |
| LayerNorm | 8 | 32 | 64 |
| head creation | 8 | 32 | 64 |
| concat heads | 8 | 32 | 64 |
| embedding | 8 | 32 | 64 |

That is `min(rows / 32, 64)` at every point: one core per tile-row, capped by the
worker grid. These four ops are not core-starved by a missing config — ttnn cannot
give a row-parallel op more cores than the tensor has tile-rows, and only batch
supplies more of them. The full grid is reached at b8s256 and there is nothing left
to widen.

That argument covers the row-parallel ops in the table and nothing else, which is
worth saying explicitly because it was over-applied once already. Matmul is not
row-parallel in this sense, and a low core count there *was* a missing config: the
`Wqkv` projection sat on 36 of 64 cores at batch 4 and 8 for exactly that reason.
Treat a low `CORE COUNT` as a question, not as an answer in either direction.

It also explains a Stage 2 result that was recorded as unexplained: sharded
LayerNorm lost to interleaved at b1s256, 35.2 against 30.0 us. Spreading 8
tile-rows over more cores means splitting the width, which buys parallelism the
shape does not contain and pays a cross-core reduction for it.

No head-creation op exposes a core grid to Python in any case —
`nlp_create_qkv_heads` and `create_qkv_heads` take `input`, `input_kv`,
`num_heads`, `num_kv_heads`, `transpose_k_heads`, `memory_config` and
`output_tensors`, and nothing else. The experimental op was benchmarked against
the shipped one regardless: 47.8 against 47.8 us at b1s256, 62.4 against 62.7 at
b4s256.

To regenerate the op breakdown:

```
python -m tracy -p -r -v -m pytest models/experimental/modernbert/tests/test_modernbert_profile.py
```

### Trace and two command queues

`runner/performant_runner.py` captures the forward pass as a Metal Trace and
replays it while cq1 stages the next `input_ids`. Each mode has its own test so a
failure localises to one mechanism rather than to "the fast path". Every row
uploads a fresh input each iteration.

| b1s256 | ms | vs untraced |
| -- | -- | -- |
| untraced, single queue | 10.49 | — |
| 2CQ only | 10.50 | +0.1% |
| trace only | **7.34** | **−30.0%** |
| trace + 2CQ | 7.45 | −29.0% |

**The whole gain is trace; 2CQ costs a little.** cq1 exists to overlap input upload
with compute, and the only input is `input_ids` — a few KB against 7 ms of device
work. At b8s256 trace is worth −1.8% (22.31 → 21.90 ms), because that shape is
device-bound and has almost no host gap to recover.

Trace is what removes the ~10.5 ms host floor the small shapes sit on: the traced
figure is the device time, and it is the same 7.3 ms whether dispatch is fast or
slow. That is the whole reason this path exists on an encoder whose device work is
only a few milliseconds.

The traced path asserts that its output *changes* when a different input is written
through cq1. A trace bakes in buffer addresses, so an input written to the wrong
buffer does not fail loudly — the trace replays whatever that buffer held last time
and scores a perfect PCC against the previous iteration.

```
pytest --disable-warnings models/experimental/modernbert/tests/test_modernbert_performant.py -s
```


### Measured and rejected

Recorded so they are not attempted again from first principles. Each delta is
relative to the baseline that change was measured against, not to the shipped
configuration.

| change | result |
| -- | -- |
| bfloat8_b activations | 15% faster; MLM logit PCC 0.99535 → 0.93988, top-1 breaks |
| `MathFidelity.LoFi` | hangs the device, ETH heartbeat timeout, needs `tt-smi -r 0` |
| HiFi2 | −7.3%, but MLM top-1 drops to 6/8 (logit PCC 0.99369 → 0.96495) |
| sharded LayerNorm as its own op | 35.2 us against 30.0 interleaved at b1s256, before any reshard cost |
| height-sharded LayerNorm | rejected upstream: `layernorm_device_operation.cpp` has a TODO |
| sharding the attention block | `split_query_key_value_and_split_heads` cannot produce the sharded output shape this model needs |
| Wqkv reading the GeGLU shard directly | +5.9%; a sharded in0 costs it 16 cores |
| Wqkv on an 8×8 core grid | 85.1 us against 73.8 for ttnn's choice — a program config was the answer, not a grid |
| SDPA on an 8×6 grid | 6.4% faster isolated, +1.2% slower in the model |
| SDPA q_chunk below 256 | monotonically slower; the perfectly balanced setting is second worst |
| SDPA `exp_approx_mode` | identical in time and PCC |
| rotary math fidelity | HiFi4/3/2/LoFi all within noise; the op default is the most accurate |
| unsharding the rotary | 1.6% worse |
| GeGLU sharded below the work threshold | b1s256 +63%, b2s256 +50% |
| `rotary_embedding_llama` | silently wrong — interleaved Meta convention, not HF rotate_half |
| experimental head-creation op | 47.8 us against 47.8 |
| DRAM-sharded matmul weights | a decode-mode technique; weights are 3.4% of Wqkv's time |
| GeGLU on 64 cores via 1280 padding | runs correctly, 21.65 ms either way — bandwidth-bound, not core-bound |

## Precision policy

Full detail lives in `tt/model_config.py`; this is the summary.

| | choice | why |
| -- | -- | -- |
| matmul weights | bfloat8_b | −8.9% at b1s256 for 3.4e-03 of MLM PCC |
| embedding + norm weights | bfloat16 | bfloat8_b is a tiled block format — it cannot hold a row-major embedding table, and a 1-D norm weight would share its exponent with 31 rows of padding |
| activations | bfloat16 | bfloat8_b is 15% faster and breaks the MLM head |
| math fidelity | HiFi3 | LoFi hangs; HiFi2 is −7.3% but takes MLM top-1 to 6/8 (logit PCC 0.99369 → 0.96495) |
| fp32 accumulation | on | costs ~2e-5 to disable |
| packer L1 accumulation | on | worth 7.5% |
| GELU | tanh in the encoder, exact erf in the MLM head | the exact erf is 11% of the pass across 22 layers; in the head it is one op, where tanh costs 5.9e-05 of logit PCC and buys nothing |

ModernBERT develops channel-localised activation outliers from layer 16 onward:
`max|x|` jumps 205 → 34101 between layers 15 and 16 and stays there until
`final_norm`, while the median channel maximum stays near 34. At that magnitude the
accumulation format matters more than the storage format, which is why fp32
accumulation stays on everywhere.

**Fidelity had to be chosen on the model, not on a matmul.** On one outlier-scale
matmul HiFi2 loses 3.7e-06; on the whole model it loses 3.6e-03 — a thousandfold
difference, because the error compounds through 22 layers and the residual stream
carries it forward. Any precision decision measured on a single op will understate
its cost here.

**bfloat8_b activations are the cautionary case.** They are the largest speedup
available anywhere in this model — 15% at batch 4 and 8 — and `last_hidden_state`
PCC survives at 0.99483, improving at batch 1 because a shared exponent per block
tracks the layer-16 outliers better than a fixed bfloat16 mantissa. What breaks is
the head: MLM logit PCC falls 0.99535 → 0.93988 and masked top-1 stops matching.
**Hidden-state PCC is not a sufficient gate for this model**, and a 15% speedup
would have shipped on it had `test_ttnn_mlm` not existed.

## Known limitations

- Sequence length and batch size are fixed at model construction; rotary caches and
  masks are built per shape. Attention masks carry a materialised batch dimension;
  whether SDPA accepts a `(1, 1, S, S)` mask at batch > 1 is untested.
- Validated at seq 256 / 512 / 768 and batch 1–8. Longer contexts up to 8192 are untested.
- ModernBERT-large (28 layers, ~395M) is not implemented; only base.
- `MathFidelity.LoFi` hangs. The forward pass never returns and the device is left
  reporting `Timed out waiting for ETH heartbeat ... ETH core e8-6 (NOC0)`, needing
  `tt-smi -r 0`. Reproduced twice from a clean device, so the guide's default
  fidelity is unavailable here; see *Precision policy* for what was measured
  instead. Not yet reported upstream.
- Attention is not sharded. `split_query_key_value_and_split_heads` requires the
  tensor's batch dimension to equal the number of cores it shards across, and
  activations here are `(1, batch * seq, hidden)` with batch folded into rows.
- Rotary embedding runs as 44 separate calls, two per layer.
  `rotary_embedding_llama_fused_qk` would halve that, but it rejects the tensor:
  `only supports decode mode with seq_len=1`. ModernBERT is an encoder, so every
  pass is a full-sequence prefill.
- SDPA's `sliding_window_size` argument is not used, and must not be: combining it
  with a `compute_kernel_config` hangs the device on this build (each alone is fine).
  The window is passed to SDPA as an ordinary additive mask instead, which is the
  same tensor the layer already builds. See the attention module docstring.
- The seq-512 path is sensitive to how much device memory is already in use.
  Larger SDPA chunk geometries deadlock there outright (see `_SDPA_CHUNKS`), and
  even the shipped 128/128 can hang if several models are resident at once.
  Allocate one model at a time: a module-scoped `device` fixture holds every model
  a parametrised test builds unless each is freed, so release model, weights and
  inputs in a `finally` block before constructing the next shape.

## Bringup Stage Status

- [x] Stage 1 — Bring-Up
- [x] Stage 2 — Basic Optimizations
- [x] Stage 3 — Deeper Optimization

Stage 3 covers SDPA on both attention paths (one call site, `layer_type` selects
only the mask), the L1-resident inter-layer chain, core utilisation, and
trace / two-command-queue integration — `trace`, `2cq` and `trace_2cq` each have
their own test in `tests/test_modernbert_performant.py`.
