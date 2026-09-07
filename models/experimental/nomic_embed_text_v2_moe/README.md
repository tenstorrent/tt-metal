# Nomic Embed Text v2 MoE

TTNN bring-up of [`nomic-ai/nomic-embed-text-v2-moe`](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
for a single Blackhole chip.

Encoder-only multilingual text-embedding transformer with a Mixture-of-Experts FFN on every
other layer. 475M total parameters, ~305M active per token. Produces sentence embeddings; no
decoder, no KV cache, no generation.

## Status

| Phase | Issue | State |
|---|---|---|
| 0: architectural overview and PyTorch reference | [#54917](https://github.com/tenstorrent/tt-metal/issues/54917) | done, PR [#55503](https://github.com/tenstorrent/tt-metal/pull/55503) |
| 1: first working TTNN PoC | [#54918](https://github.com/tenstorrent/tt-metal/issues/54918) | not started |
| 2: device performance | | not started |

`tt/` is empty and `tests/perf/` does not exist yet; both arrive with Phase 1. A perf test in
this repo means a device performance test: it needs the device fixture, the
`models_device_performance_bare_metal` marker and `prep_device_perf_report`. A CPU-only
placeholder would be dead code that could pollute perf dashboards selecting by directory and
marker.

## Layout

```
reference/
  ARCHITECTURE.md            verified architecture and operator mapping
  IMPLEMENTATION_PLAN.md     approved multi-phase plan
  modeling_nomic_moe.py      golden PyTorch reference
  configuration_nomic_moe.py config that validates every baked-in assumption
  loader.py                  checkpoint contract, generated from the config
  pipeline.py                prefixes, pooling, Matryoshka, L2 normalize
  hf_reference.py            containment for the transformers native-class trap
  config.json                pinned config snapshot for the no-network tests
common.py                    pinned revisions, checkpoint resolution, metrics, helpers
tests/pcc/                   correctness tests
tt/                          TTNN implementation (Phase 1)
```

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
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_reference_modules.py -v
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

Detail in [`reference/ARCHITECTURE.md`](reference/ARCHITECTURE.md).

1. `AutoModel.from_pretrained` returns the wrong model without raising. transformers >= 5
   ships a native `nomic_bert` targeting v1.5, no MoE, registered for this `model_type`.
   Loading this checkpoint into it discards every expert tensor and randomly initialises
   `gate_proj`/`up_proj`. Always use `hf_reference.load_hf_model`.
2. The MoE top-2 weights are not renormalized; they sum to ~0.70 on real text. Mixtral and
   Switch both divide by the top-k sum. That bug scores PCC ~0.99, right at a typical gate.
3. PCC cannot catch the shared-bias bug. The expert bias is one `[768]` vector added once
   after the weighted sum; folding it into the loop gives a near-constant offset that PCC
   mean-centres away (0.9999998 on real weights). Gate that class on max-abs.

## References

- Model: <https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe> @ `1066b6599d09`
- Modelling code: <https://huggingface.co/nomic-ai/nomic-bert-2048> @ `7710840340a0`
- Paper: <https://arxiv.org/pdf/2502.07972>
- Matryoshka Representation Learning: <https://arxiv.org/pdf/2205.13147>
- Umbrella issue: <https://github.com/tenstorrent/tt-metal/issues/54916>
