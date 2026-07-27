# Upstream numerical validation

Compares our CPU reference blocks against the **upstream implementations**, block by block, at
PCC. This is the gate the `models/experimental/xtts_v2` references cleared against coqui
(`tests/_coqui_groundtruth.py`); it lives under `scripts/` rather than `tests/` because it needs
its own virtualenv and vendored upstream source, so it cannot run in CI.

**Result at the pinned commit: 27/27 checks pass.** See `reference/PROVENANCE.md`.

## What is compared against what

| Block | Upstream reference | Why that one |
|---|---|---|
| 1 AR backbone | `mistral_inference` (Mistral's own) | vLLM-Omni delegates the backbone to vLLM's `MistralForCausalLM`, which needs paged attention and a GPU. `mistral_inference` is authoritative for this architecture *and* reads the same `consolidated.safetensors` + `params.json`, so it settles the RoPE convention. |
| 2 Flow matching | `vllm_omni` `FlowMatchingAudioTransformer` | plain `nn.Module`; runs on CPU once vLLM imports are stubbed |
| 3 Codec decoder | `vllm_omni` `VoxtralTTSAudioTokenizer` | same; `flash_attn` is deliberately absent so upstream's own SDPA fallback provides the ALiBi + causal + sliding-window path |

Two substitutions are needed to run on a CPU box, both documented in the scripts:

- **vLLM / mistral_common / transformers imports are stubbed** (`upstream_loader.py`). The classes
  under test never touch them at runtime, so their math is unaffected. Real `torch` and real
  `einops`.
- **xformers `memory_efficient_attention` → `torch.nn.functional.scaled_dot_product_attention`**
  in `mistral_inference` (no CPU kernel for xformers). This is a *third* implementation,
  independent of both ours and xformers', so agreement validates our attention as well as our
  layer wiring — it is not circular.

## Setup

```bash
# 1) isolated CPU venv
python3 -m venv /tmp/cmp_venv
/tmp/cmp_venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/tmp/cmp_venv/bin/pip install einops numpy mistral-inference
#    (mistral-inference drags in a CUDA xformers; the import warning is harmless, we patch it out)

# 2) fetch the two upstream model files at the pinned commit
python models/experimental/voxtral_tts/scripts/upstream_compare/fetch_upstream.py

# 3) run (needs the checkpoint in reference/weights/)
export TT_METAL_HOME=$PWD
PYTHONPATH=$TT_METAL_HOME /tmp/cmp_venv/bin/python \
    models/experimental/voxtral_tts/scripts/upstream_compare/compare_blocks.py
PYTHONPATH=$TT_METAL_HOME /tmp/cmp_venv/bin/python \
    models/experimental/voxtral_tts/scripts/upstream_compare/compare_backbone.py
```

Both scripts exit non-zero if any check falls below its gate.

## Memory

`compare_backbone.py` holds our full 26-layer fp32 stack (~14 GB) plus one upstream layer, so it
wants ~20 GB of RAM. It swaps weights through a single upstream block rather than instantiating
26 of them, specifically to avoid needing ~28 GB.
