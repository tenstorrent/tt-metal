# Nomic Embed Text v2 MoE

TTNN bring-up of [`nomic-ai/nomic-embed-text-v2-moe`](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
for a single Blackhole chip.

Encoder-only multilingual text-embedding transformer with a Mixture-of-Experts FFN on every
other layer. 475M total parameters, ~305M active per token. Produces sentence embeddings; no
decoder, no KV cache, no generation.

## Embedding specification

**In:** B strings, plus a `prompt_name` selecting which task prefix step 1 prepends.

**Out:** one unit-norm vector per string.

```
 in   ["Hello!", "¡Hola!"]                                      B strings

  1   apply_prompt          ["search_document: Hello!", ...]    B strings
  2   tokenize              input_ids                           (B, S)        int64
                            attention_mask                      (B, S)        int64
  3   forward               last_hidden_state                   (B, S, 768)   fp32
  4   mean_pool             pooled                              (B, 768)      fp32
  5   matryoshka_truncate   pooled                              (B, dim)      fp32   optional
  6   l2_normalize          embeddings                          (B, dim)      fp32

out   embeddings[0] @ embeddings[1]                             float        similarity
```

Steps 1 and 2 are `preprocessing.py`, step 3 `inference.py`, steps 4 to 6 `postprocessing.py`.
`embedding.encode` runs all six in one call.

1. The prefix is trained-in, and it is prepended to the text rather than injected as a special
   token, so it becomes ordinary tokens the encoder attends to. The same text under `"query"`
   and under `"passage"` gives different vectors.
2. S is the longest tokenized text in the batch, so it varies with the input. Shorter rows are
   right-padded and `attention_mask` records which positions are real.
3. One 768-wide vector per token, still per token.
4. Where the sequence axis disappears: B*S token vectors become B text vectors. Mask-weighted,
   so padding is excluded and a text's embedding never depends on its batch-mates.
5. Optional. The leading features carry the most information, so a narrower vector stays
   usable: the Matryoshka property.
6. Unit norm is what makes the dot product on the output line a cosine similarity.

## Layout

```
README.md                     this file: layout, setup, test commands
common.py                     pinned revisions, contracts, checkpoint resolution, test helpers
docs/ARCHITECTURE.md          what the model is; see Documentation below
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
  pcc/                        correctness tests
tt/                           TTNN implementation (Phase 1)
```

## Documentation

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the model: dimensions, pinned
revisions, checkpoint contract, operator inventory and embedding pipeline. Hand-written, and
every number in it was measured against the pinned checkpoint.

## Setup

```bash
cd /path/to/tt-metal
source python_env/bin/activate
```

Weights resolve from the Hugging Face cache at a pinned revision. Pre-fetch (1.8 GB):

```bash
python -c "from models.experimental.nomic_embed_text_v2_moe.common import resolve_checkpoint; print(resolve_checkpoint())"
```

Tests needing the checkpoint skip rather than fail when it is absent.

## Tests

```bash
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v
```

Every test needs the checkpoint and a warm HF cache or network; they skip rather than fail when
the checkpoint is absent.

| File | Covers | Needs weights |
|---|---|---|
| `test_checkpoint_contract.py` | 148 keys/shapes/dtypes generated from the config, absence assertions, strict load | yes |
| `test_reference_vs_hf_e2e.py` | end to end vs upstream: per-layer parity, tokenizer, prefixes, model-card similarity, Matryoshka, ragged batches | yes, plus network |

Phase 0 is CPU-only. Do not set `TT_VISIBLE_DEVICES`; on a p300c it fails with
`Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`.

### Quick test

```python
from models.experimental.nomic_embed_text_v2_moe.common import load_tokenizer
from models.experimental.nomic_embed_text_v2_moe.reference import inference, postprocessing, preprocessing
from models.experimental.nomic_embed_text_v2_moe.reference.loader import load_pretrained_reference_model

model, tokenizer = load_pretrained_reference_model(), load_tokenizer()
texts = ["Hello!", "¡Hola!"]

prefixed = preprocessing.apply_prompt(texts, "passage")                 # 2 strings, prefixed
encoded = preprocessing.tokenize(tokenizer, prefixed)                   # ids, mask (2, 10) i64
last_hidden_state = inference.forward(model, encoded["input_ids"], encoded["attention_mask"])  # (2,10,768)
pooled = postprocessing.mean_pool(last_hidden_state, encoded["attention_mask"])  # (2, 768), ~15
embeddings = postprocessing.l2_normalize(pooled)                        # (2, 768), norms 1.0

print(float(embeddings[0] @ embeddings[1]))   # 0.911788
```

`mean_pool` is where the sequence axis disappears. Row 0 is padded here, since `"Hello!"`
tokenizes shorter than `"¡Hola!"`, which is why pooling is mask-weighted: `<pad>` has a
non-zero embedding, so counting it would make row 0 depend on its batch-mate. Insert
`pooled = postprocessing.matryoshka_truncate(pooled, 256)` before the normalize for `(2, 256)`.

### Correctness traps

These failures do not raise exceptions, so the test suite includes measurements and negative controls for each:

1. **Do not use `AutoModel.from_pretrained` directly.**
   With `transformers >= 5`, it may resolve this model to the native `nomic_bert` implementation, which targets v1.5 and does not include the MoE layers. Always load through `reference/hf_reference.load_hf_model`.

2. **Do not renormalize the MoE top-2 routing weights.**
   `moe_normalize_expert_weights` is `false` in this checkpoint, so the two selected weights are used as they come out of the softmax and sum to less than 1. Dividing them by their top-2 sum, which Mixtral and Switch both do and which is the easy thing to copy by reflex, still scores around `0.99` PCC.

3. **Use max-absolute error for shared-bias validation.**
   PCC can hide the shared-bias bug because it mean-centers the resulting offset. The expert bias must be added once after the weighted expert sum, not inside the expert loop.


## References

- Model: <https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe> @ `1066b6599d09`
- Modelling code: <https://huggingface.co/nomic-ai/nomic-bert-2048> (fetched from `main` via `auto_map`)
- Paper: <https://arxiv.org/pdf/2502.07972>
- Matryoshka Representation Learning: <https://arxiv.org/pdf/2205.13147>
- Umbrella issue: <https://github.com/tenstorrent/tt-metal/issues/54916>
