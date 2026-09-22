# Nomic Embed Text v2 MoE

TTNN bring-up of [`nomic-ai/nomic-embed-text-v2-moe`](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
for a single Blackhole chip.

Encoder-only multilingual text-embedding transformer with a Mixture-of-Experts FFN on every
other layer. 475M total parameters, ~305M active per token. Produces sentence embeddings; no
decoder, no KV cache, no generation.

## Embedding specification

B strings in, B unit-norm vectors out, in six steps across three stages. One call runs all of
them: `reference/embedding.py::encode` on the PyTorch side, `tt/model.py::encode` on the device.
The two take the same arguments and return the same `(B, dim)` torch tensor, so the port is a
drop-in for the reference.

| Step | Operation | Reference | TTNN |
|---|---|---|---|
| 1-2 | prefix, tokenize | `reference/preprocessing.py` | the same file, reused verbatim: host-only work with no device equivalent |
| 3 | encoder | `reference/inference.py` over `modeling_nomic_moe.py` | `tt/model.py`, `TtNomicBertModel` |
| 4-6 | pool, truncate, normalize | `reference/postprocessing.py` | `tt/pooling.py` |

The steps below describe the reference; each TTNN counterpart performs the same operation at the
same point, on device.

### Preprocessing, `preprocessing.py`

| Step | Operation | Output |
|---|---|---|
| 1 | `apply_prompt`: prepend a task prefix to each string | B prefixed strings |
| 2 | `tokenize`: tokenize and right-pad | `input_ids`, `attention_mask`, both `(B, S)` int64 |

Prefixes are trained-in, so the same string embeds differently as a query than as a document.
`prompt_prefix` takes one `NomicPromptPrefix` for the batch or one per string:

```python
apply_prompt(
    ["How do you say hello in Spanish?", "¡Hola!"],
    [NomicPromptPrefix.QUERY, NomicPromptPrefix.PASSAGE],
)
```

S is the batch's longest tokenized sequence, truncated at 512.

### Inference, `inference.py`

| Step | Operation | Output |
|---|---|---|
| 3 | `forward`: run the encoder | `last_hidden_state`, `(B, S, 768)` fp32 |

One 768-wide vector per token.

### Postprocessing, `postprocessing.py`

| Step | Operation | Output |
|---|---|---|
| 4 | `mean_pool`: average the token vectors, excluding padding | `pooled`, `(B, 768)` fp32 |
| 5 | `matryoshka_truncate`: keep the leading `dim` features, optional | `pooled`, `(B, dim)` fp32 |
| 6 | `l2_normalize`: scale to unit norm | `embeddings`, `(B, dim)` fp32 |

Step 4 collapses the sequence axis, masked so a string's embedding never depends on its
batch-mates. Step 5 is optional; `dim` is 768 without it. Unit norm makes a dot product of two
rows their cosine similarity:

```python
similarity = float(embeddings[0] @ embeddings[1])
```

## Layout

```
README.md                     this file: layout, setup, test commands
common.py                     pinned revisions, contracts, checkpoint resolution, test helpers
docs/                         architecture and operator mapping; see Documentation below
reference/
  modeling_nomic_moe.py       golden PyTorch reference
  configuration_nomic_moe.py  config projected from the pinned config.json snapshot
  loader.py                   checkpoint contract, generated from the config
  preprocessing.py            task prefixes, tokenization
  inference.py                runs the model
  postprocessing.py           mean pooling, Matryoshka truncation, L2 normalize
  embedding.py                drives the three stages end to end
  hf_reference.py             containment for the transformers native-class trap
  config.json                 pinned config snapshot for the no-network tests
tests/
  conftest.py                 session-scoped checkpoint, model and tokenizer fixtures
  pcc/                        correctness tests: one file per operator group, module and the model
  pcc/module_common.py        shared scaffolding for the module tests, not a test file itself
tt/
  model_config.py             dtypes, layout and compute kernel configs, bound to a device
  common.py                   weight reorientation, rotary tables, attention mask, reshapes
  embeddings.py               word lookup, token-type embedding folded into the table
  attention.py                fused QKV, rotary, bidirectional SDPA, output projection
  mlp.py                      dense FFN, even-numbered layers
  router.py                   fp32 softmax, top-k, dense routing weights
  experts.py                  all experts as two broadcast-batch matmuls, gate and reduce
  moe.py                      router plus experts, odd-numbered layers
  block.py                    one encoder block, post-norm with fused residual adds
  encoder.py                  the 12 blocks in sequence
  pooling.py                  mean pool, Matryoshka truncation, L2 normalize
  model.py                    the whole model, plus the text-to-embedding driver
demo/
  demo.py                     embeds queries and passages, prints the similarity matrix
```

`tt/model.py` is the host boundary: every module below it takes and returns device tensors, while
`TtNomicBertModel` takes torch token ids, because that is what the tokenizer produces and because
the rotary tables and the attention mask are host builds that depend on S. Its `encode()` mirrors
`reference/embedding.py` and reuses `reference/preprocessing.py` verbatim for the prefixes and
tokenization, which have no device equivalent.

Modules take the full state dict plus a prefix and move their weights to device once at
construction, following `models/tt_transformers`.

## Documentation

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the model: dimensions, pinned
revisions, checkpoint contract, operator inventory and embedding pipeline.

[`docs/OPERATOR_MAPPING.md`](docs/OPERATOR_MAPPING.md) maps that inventory onto TTNN: the
operator each aten call becomes, the PCC it reaches, the API and shape differences, and the
negative controls for the ways an operator can be wrong without failing.

[`docs/DATASET_ACCURACY.md`](docs/DATASET_ACCURACY.md) reports retrieval accuracy against the
reference on SciFact and XQuADRetrieval, 12 languages and 45124 encodes, where the test suite
uses random token ids. It also records where the per-row cosine bound asserted by the test
suite, and the short-sequence expectation stated below, fail to hold on real text.

All three are hand-written, and every number in them was measured.

## Setup

```bash
cd /path/to/tt-metal
source python_env/bin/activate
```

Weights resolve from the Hugging Face cache at a pinned revision. Pre-fetch (1.8 GB):

```bash
python -c "from models.experimental.nomic_embed_text_v2_moe.common import resolve_checkpoint; print(resolve_checkpoint())"
```

## Accuracy

Against the PyTorch reference, measured on a Blackhole p300c:

| Level | Gate | Measured |
|---|---|---|
| operators | PCC >= 0.999 | 0.99999 or better; SDPA is the tightest at 0.99976 |
| modules | PCC >= 0.99, to 0.999 by module | 0.9998 or better |
| 12-block encoder | PCC >= 0.99 | 0.99366 to 0.99834 |
| end to end, `last_hidden_state` | PCC >= 0.98 | 0.98162 to 0.99808 |
| end to end, embedding | cosine within 0.01, plus retrieval agreement | 1 - cosine of 9.1e-05 to 7.6e-03 |

The end-to-end gate is the pooled embedding cosine and the retrieval ranking, not PCC alone.
All-token PCC over a few hundred tokens moves with how many of them were routed differently, so
it is a sanity floor rather than an accuracy measure; the cosine is stable and the ranking is what
the embeddings are used for. Short sequences are the worst case, not long ones, because one
rerouted token is a larger share of the pooled mean: 1/74 at `2x37` against 1/1024 at `2x512`.

Routing differences are inherent rather than a defect. Roughly 1% to 2% of tokens sit within the
softmax's own 1.4e-3 error of a top-2 tie at every MoE layer, so a token can legitimately visit a
different pair of experts on device than in torch. The model card's published 0.9118 reproduces on
device.

## Tests

```bash
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v
```

Green on a single Blackhole chip inside a QuietBox, where every measurement above was taken. To
run one level, select by file or name:

```bash
pytest .../tests/pcc/test_ttnn_operators*.py -v     # operators
pytest .../tests/pcc/ -k "ttnn and not operators"   # modules and the model
pytest .../tests/pcc/test_ttnn_model.py -v          # end to end
```

### Demo

```bash
python models/experimental/nomic_embed_text_v2_moe/demo/demo.py --compare
```

Embeds a built-in set of queries and passages, prints the similarity matrix and the top passage
per query, and with `--compare` reports the per-row cosine against the PyTorch reference. Pass
`--query` and `--passage` (both repeatable) for your own text.

Tests skip rather than fail when what they need is absent: the checkpoint, the network, or a
Blackhole device.

The `test_ttnn_*.py` files need a Blackhole device and skip elsewhere;
`test_checkpoint_contract.py` and `test_reference_vs_hf_e2e.py` need only the weights. Do not set
`TT_VISIBLE_DEVICES`; on a p300c it fails with `Custom fabric mesh graph descriptor path must
be specified for CUSTOM cluster type`.

### Correctness traps

These failures do not raise exceptions, so the test suite includes measurements and negative controls for each:

1. **Do not use `AutoModel.from_pretrained` directly.**
   With `transformers >= 5`, it may resolve this model to the native `nomic_bert` implementation, which targets v1.5 and does not include the MoE layers. Always load through `reference/hf_reference.load_hf_model`.

2. **Do not renormalize the MoE top-2 routing weights.**
   `moe_normalize_expert_weights` is `false` in this checkpoint, so the two selected weights are used as they come out of the softmax and sum to less than 1. Dividing them by their top-2 sum, which Mixtral and Switch both do and which is the easy thing to copy by reflex, still scores around `0.99` PCC.

3. **Use max-absolute error for shared-bias validation.**
   PCC can hide the shared-bias bug because it mean-centers the resulting offset. The expert bias must be added once after the weighted expert sum, not inside the expert loop.

4. **Pass `is_causal=False` to SDPA explicitly.**
   `ttnn.transformer.scaled_dot_product_attention` defaults it to `True` where torch defaults to `False`. This is an encoder, so leaving the default applies a decoder mask: every token still gets finite output, computed from its prefix alone, at PCC 0.44.

5. **Pad the attention mask with dtype-min, not zero.**
   The mask is `(B, 1, S, S)` in `TILE_LAYOUT`, so S rounds up to a multiple of 32 and the pad columns take whatever the conversion fills them with. Zero is additively neutral, meaning "attend here", so SDPA counts those columns in the softmax denominator. At S=37 that took the output norm to 0.69x while PCC moved only 0.9998 to 0.9974, so gate it on the norm. `tt/common.py::additive_attention_mask` builds in `ROW_MAJOR` and converts with `pad_value=finfo.min`.


## References

- Model: <https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe> @ `1066b6599d09`
- Modelling code: <https://huggingface.co/nomic-ai/nomic-bert-2048> (fetched from `main` via `auto_map`)
- Paper: <https://arxiv.org/pdf/2502.07972>
- Matryoshka Representation Learning: <https://arxiv.org/pdf/2205.13147>
- Porting-to-ttnn skill: <https://github.com/sott0n/tt-agent-skills/tree/main/tt-metal/skills/porting-models-to-ttnn>
- Umbrella issue: <https://github.com/tenstorrent/tt-metal/issues/54916>
