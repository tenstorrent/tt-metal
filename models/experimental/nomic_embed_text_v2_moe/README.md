# Nomic Embed Text v2 MoE

TTNN bring-up of [`nomic-ai/nomic-embed-text-v2-moe`](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
for a single Blackhole chip.

Encoder-only multilingual text-embedding transformer with a Mixture-of-Experts FFN on every
other layer. 475M total parameters, ~305M active per token. Produces sentence embeddings; no
decoder, no KV cache, no generation.

## Status

| Phase | Issue | State |
|---|---|---|
| 0: architectural overview | [#54917](https://github.com/tenstorrent/tt-metal/issues/54917) | done, PR [#55503](https://github.com/tenstorrent/tt-metal/pull/55503) |
| 0: PyTorch reference | [#54919](https://github.com/tenstorrent/tt-metal/issues/54919) | done, same PR |
| 1: first working TTNN PoC | [#54918](https://github.com/tenstorrent/tt-metal/issues/54918) | not started |
| 2: device performance | | not started |

`tt/` is empty and `tests/perf/` does not exist yet; both arrive with Phase 1. A perf test in
this repo means a device performance test: it needs the device fixture, the
`models_device_performance_bare_metal` marker and `prep_device_perf_report`. A CPU-only
placeholder would be dead code that could pollute perf dashboards selecting by directory and
marker.

## Layout

```
README.md                     this file: status, layout, setup, test commands
common.py                     pinned revisions, contracts, checkpoint resolution, test helpers
docs/ARCHITECTURE.md          what the model is; see Documentation below
reference/
  modeling_nomic_moe.py       golden PyTorch reference
  configuration_nomic_moe.py  config projected from the pinned config.json snapshot
  loader.py                   checkpoint contract, generated from the config
  pipeline.py                 task prefixes, pooling, Matryoshka, L2 normalize
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
# everything, ~26 s with a warm cache
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v

# structural backbone: no network, no weights, no device, ~9 s
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -m "not needs_weights" -v
```

| File | Covers | Needs weights |
|---|---|---|
| `test_reference_modules.py` | rotary, QKV layout, post-norm structure, router, experts, GELU, pooling, each with a negative control | no |
| `test_checkpoint_contract.py` | 148 keys/shapes/dtypes generated from the config, absence assertions, strict load | yes |
| `test_reference_vs_hf.py` | end-to-end and per-layer parity with upstream | yes, plus network |
| `test_embedding_pipeline.py` | tokenizer, prefixes, model-card similarity, Matryoshka, ragged batches | yes, plus network |

Phase 0 is CPU-only. Do not set `TT_VISIBLE_DEVICES`; on a p300c it fails with
`Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`.

## Three things that fail silently

Each is measured, and each has a negative control in the test suite.

1. `AutoModel.from_pretrained` returns the wrong model without raising. transformers >= 5
   ships a native `nomic_bert` targeting v1.5, no MoE, registered for this `model_type`.
   Always load through `reference/hf_reference.load_hf_model`.
2. The MoE top-2 weights are not renormalized. Mixtral and Switch both divide by the top-k
   sum, so copying either by reflex produces a bug that lands right on a 0.99 PCC gate.
3. PCC cannot catch the shared-bias bug at all. The expert bias is added once after the
   weighted sum; folding it into the loop gives a near-constant offset that PCC mean-centres
   away. Gate that class of bug on max-abs.

## References

- Model: <https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe> @ `1066b6599d09`
- Modelling code: <https://huggingface.co/nomic-ai/nomic-bert-2048> (fetched from `main` via `auto_map`)
- Paper: <https://arxiv.org/pdf/2502.07972>
- Matryoshka Representation Learning: <https://arxiv.org/pdf/2205.13147>
- Umbrella issue: <https://github.com/tenstorrent/tt-metal/issues/54916>
