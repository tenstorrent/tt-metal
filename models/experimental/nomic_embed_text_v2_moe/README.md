# Nomic Embed Text v2 MoE

TTNN bring-up of [`nomic-ai/nomic-embed-text-v2-moe`](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
for a single Blackhole chip.

An encoder-only multilingual text-embedding transformer with a Mixture-of-Experts FFN on every
other layer — 475M total parameters, ~305M active per token. It produces sentence embeddings;
there is no decoder, no KV cache and no generation.

## Status

| Phase | Issue | State |
|---|---|---|
| 0 — architectural overview + PyTorch reference | [#54917](https://github.com/tenstorrent/tt-metal/issues/54917) | **this PR** |
| 1 — first working TTNN PoC | [#54918](https://github.com/tenstorrent/tt-metal/issues/54918) | not started |
| 2 — device performance | — | not started |

`tt/` is empty and `tests/perf/` does not exist yet; both arrive with Phase 1. In this repo a
perf test means a *device* performance test — it needs the device fixture, the
`models_device_performance_bare_metal` marker and `prep_device_perf_report`. A CPU-only
placeholder would be dead code that could pollute perf dashboards selecting by directory and
marker.

## Layout

```
reference/
  ARCHITECTURE.md            verified architecture + operator mapping  <- start here
  IMPLEMENTATION_PLAN.md     the approved multi-phase plan
  modeling_nomic_moe.py      the golden PyTorch reference
  configuration_nomic_moe.py config that RAISES on every baked-in assumption
  loader.py                  checkpoint contract, generated from the config
  pipeline.py                prefixes, pooling, Matryoshka, L2 normalize
  hf_reference.py            containment for the transformers native-class trap
  config.json                pinned config snapshot (no-network tests)
common.py                    pinned revisions, checkpoint resolution, hook harness
tests/pcc/                   correctness tests
tt/                          TTNN implementation (Phase 1)
```

## Setup

```bash
cd /path/to/tt-metal
source python_env/bin/activate
```

The tests resolve weights from the Hugging Face cache at a pinned revision. To pre-fetch (1.8 GB):

```bash
python -c "from models.experimental.nomic_embed_text_v2_moe.common import resolve_checkpoint; print(resolve_checkpoint())"
```

Tests that need the checkpoint **skip** rather than fail when it is absent.

## Tests

```bash
# everything (~26 s with a warm cache)
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v

# the structural backbone: no network, no weights, no device (~9 s)
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_reference_modules.py -v

# parity against the upstream HF model, 13-point per-layer ladder
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/test_reference_vs_hf.py -v
```

| File | What it covers | Needs weights |
|---|---|---|
| `test_reference_modules.py` | rotary, QKV layout, post-norm structure, router, experts, GELU, pooling — each with a negative control | no |
| `test_checkpoint_contract.py` | 148 keys/shapes/dtypes generated from the config, absence assertions, `strict=True` load | yes |
| `test_reference_vs_hf.py` | end-to-end and 13-point per-layer parity with upstream | yes + network |
| `test_embedding_pipeline.py` | tokenizer, prefixes, model-card similarity, Matryoshka, ragged batches | yes + network |

Phase 0 is CPU-only. No `TT_VISIBLE_DEVICES` is needed or wanted — on a p300c it fails with
`Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`.

## Three things that will bite you

Full detail in [`reference/ARCHITECTURE.md`](reference/ARCHITECTURE.md); these are the ones that
fail *silently*.

**1. `AutoModel.from_pretrained` gives you the wrong model, without raising.**
`transformers` ≥ 5 ships a native `nomic_bert` targeting nomic-embed-text-**v1.5** — no MoE,
SwiGLU MLP — registered for this exact `model_type`. Loading this checkpoint into it discards
every expert tensor as UNEXPECTED and randomly initialises `gate_proj`/`up_proj`. You get a
working model that computes the wrong thing. Always go through `hf_reference.load_hf_model`,
which forces remote code and asserts the resolved class came from `transformers_modules`.

**2. The MoE top-2 weights are not renormalized.** They sum to ~0.70 on real text. Mixtral,
Switch and most reference code divide by the top-k sum; this model does not. The bug scores
PCC ~0.99 — right at a typical gate.

**3. PCC cannot catch the shared-bias bug.** The expert bias is one `[768]` vector added once
*after* the weighted sum. Folding it into the per-expert loop produces a near-constant offset,
and PCC mean-centres — it scores **0.9999998** against real weights. Gate that class of bug on
max-abs, which the tests do.

## References

- Model: <https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe> @ `1066b6599d09`
- Modelling code: <https://huggingface.co/nomic-ai/nomic-bert-2048> @ `7710840340a0`
- Paper: <https://arxiv.org/pdf/2502.07972>
- Matryoshka Representation Learning: <https://arxiv.org/pdf/2205.13147>
- Umbrella issue: <https://github.com/tenstorrent/tt-metal/issues/54916>
