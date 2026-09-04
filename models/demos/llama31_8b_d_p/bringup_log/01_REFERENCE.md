# 01 — Reference implementation

Phase P1. What the torch oracle is, how it is invoked, its dtype policy, and how it was validated.
Date (UTC): 2026-09-04. Gate: `G-REF` (`06_GATES.md`).

## 1. The strategy — two oracles, one gate holding them together

`DEC-005`. Llama is first-class in `transformers` (`model_type: llama`, no `trust_remote_code`), so
**no `reference/` package is vendored** — the recipe's option 2 is explicitly not taken, and the
package has no `reference/` directory at all.

| Oracle | Used by | Why |
|---|---|---|
| **A — hand-written torch math**, in `tests/unit/test_reference_model.py` | every P5 module gate (`G-RMS`, `G-ROPE`, `G-MLP`, `G-ATTN`, `G-KV`) | no checkpoint, no HF model construction, ~1 s per test, runs on a bare card. Identical random weights drive both sides, the `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:12` pattern |
| **B — HF `LlamaDecoderLayer` / `LlamaForCausalLM`** | P6/P7 layer- and model-level gates (`G-LAYER`, `G-MODEL`, `G-GOLDEN`) | the only oracle that can be wrong about the architecture *in the same way HF is*; real weights; `models/tt_transformers/tt/model_config.py:4393` `reference_decoder` and friends wrap it |
| **`G-REF`** | — | proves A is a faithful transcription of B. Measured **bit-exact**, both with random and with real layer-0 weights |

Oracle B is reachable two ways: bare (`LlamaDecoderLayer(cfg, idx)` + `load_state_dict`, what this
package does) or through `models/tt_transformers`' accessors (`reference_transformer:4037`,
`reference_decoder:4393`, `reference_attention:4410`, `reference_mlp:4365`,
`reference_rms_norm:4167`, `reference_embedding:4379`, `reference_lm_head:4027`). The accessors
**raise without `HF_MODEL`** (`models/tt_transformers/tt/model_config.py:702`) and construct a whole
`ModelArgs`; the bare route needs only the bundled `config.json`, so `G-REF` uses the bare route and
P6 may use either.

## 2. How to invoke it

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct   # only for the real-weight tests

pytest models/demos/llama31_8b_d_p/tests/unit/test_reference_model.py -x -q
```

Host only — the file imports no `ttnn` and takes no `mesh_device` fixture (`DEC-010`). 13.5 s for
9 tests. Without `HF_MODEL` the two real-weight tests skip (`requires_hf_reference`) and the other
seven still run.

Public entry points, all in `tests/unit/test_reference_model.py` and importable by later phases:

| Function | Signature | Returns |
|---|---|---|
| `llama3_inv_freq` | `(head_dim=128, theta=500000.0, scaling=None)` | `[64]` fp32 `inv_freq`; `scaling=None` is the unscaled control |
| `build_cos_sin` | `(seq_len, inv_freq=None, start_pos=0)` | `(cos, sin)`, each `[S, 128]` fp32, **HF convention** (`cat` of halves) |
| `rms_norm` | `(x, weight, eps=1e-05)` | `[B, S, 4096]` |
| `attention` | `(x, w, cos, sin, mask)` | `[B, S, 4096]`; `mask=None` is the non-causal control, not a default |
| `mlp` | `(x, w)` | `[B, S, 4096]` |
| `decoder_layer` | `(x, w, cos, sin, mask)` | `[B, S, 4096]` |
| `random_layer_weights` / `real_layer_weights` | `(seed=0, scale=0.02)` / `(layer_idx=0)` | HF `[out, in]` weight dict |
| `hf_decoder_layer` | `(w, layer_idx=0)` | an fp32 eager-attention `LlamaDecoderLayer` loaded from `w` |

Supporting fixtures live in `tests/test_factory.py` (`llama_config_dims`, `requires_hf_reference`,
`load_hf_state_dict`, and the noise-floor helpers) and `conftest.py` (session `state_dict`,
`--skip-model-load`).

## 3. Dtype policy — fp32, and it is not a detail

`DEC-006`. **The reference computes in fp32 end to end.** Checkpoint tensors (`torch_dtype:
bfloat16`) are cast with `.float()` at load; the HF layer is built bare and `.float()`ed rather than
`from_pretrained`ed, which would load at bf16. Nothing is rounded back to bf16 anywhere in this
gate, because there is no device tensor in it.

Why this is load-bearing, not fussiness: a bf16-weight reference **shares the device's own
rounding** and inflates every PCC downstream — measured in the recipe as 0.9999867 (bf16-weight
reference) against 0.99995 (fp32-weight reference) for the *same device output* (`BRINGUP_RECIPE.md:402`).
The fp32 reference is strictly harder and it is the only one the noise-floor method is defined
against: the floor *is* "quantise what the device stores, compute the rest in fp32"
(`tests/test_factory.py::quantize_like_device`, from `models/demos/common/bringup/examples/noise_floor.py:33`).

At the comparison boundary in P5, the device output is compared against this fp32 reference, and the
floor is computed by quantising the reference's *inputs and weights* to the device dtype. Two
numbers per gate, never one.

## 4. `transformers` 5.12.1 — the five traps, and what this package does about each

Measured on this box, `transformers 5.12.1` / `huggingface_hub 1.16.1` / `torch 2.11.0+cpu`.

| # | Trap | Status here |
|---|---|---|
| 1 | **`rope_theta` is not an attribute** | **Reproduced and regression-tested.** `cfg.rope_theta` raises `AttributeError`; `getattr(cfg, "rope_theta", 10000.0)` returns **10000.0**, silently; `cfg.to_dict()` has no `rope_theta` (only `rope_parameters`); `cfg.rope_scaling` *does* contain `rope_theta: 500000.0`. This package reads theta and scaling in **exactly one place** — `llama_config_dims()` → `get_rope_theta` / `get_rope_scaling` (`models/tt_transformers/tt/common.py:165`, `:183`) on the **raw `config.json` dict** — and asserts non-`None` at import. `test_rope_theta_is_not_an_attribute_on_transformers_5` fails if the trap ever stops reproducing. See `07_RISKS.md` R-005 |
| 2 | **dict-vs-object `hf_config`** | Decided: **dict**, everywhere, and it is the bundled `config.json` dict. `llama_config_dims()` is the single constructor. The templates pass an object (`models/demos/minimax_m3/tt/dense_mlp.py:47` does `hf_config.hidden_size`), so P5 modules must not copy that accessor style without a normalising constructor — carried into `03_OUTLINE.md` as a P3 obligation (`tt/model_config.py` owns the one normalisation) |
| 3 | **`attention_mask=None` is silently non-causal** | **Reproduced as a negative control.** `cfg._attn_implementation = "eager"` **and** an explicit `[1,1,S,S]` additive mask. `test_reference_attention_is_causal` perturbs the last token: with the mask, rows `[:-1]` move by `max|Δ| = 0.0`; with `mask=None`, by `3.515e+00`. Both oracles behave identically, which is itself evidence the transcription is faithful |
| 4 | **`get_rot_transformation_mat(dhead=32)` ignores its argument** | Confirmed by reading: `models/tt_transformers/tt/common.py:562` takes `dhead=32` and `:564` immediately reassigns `dhead = 32`. Not used in P1; the rule for P5.5 is **call it with no arguments**. `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:213` already does |
| 5 | **HF wrappers branch on the forward signature** | Checked, and the branch resolves the *new* way here: `inspect.signature(LlamaDecoderLayer.forward)` and `LlamaAttention.forward` both contain `position_embeddings`, so `HfDecoderWrapper` / `HfAttentionWrapper` take the position-embeddings path and RoPE is applied exactly once. This package sidesteps the wrappers entirely in `G-REF` (it calls the layer directly with `position_embeddings=(cos, sin)`), so a wrapper regression cannot masquerade as a model bug here — but P6 must re-check if it uses the accessors |

## 5. Determinism

`test_handwritten_reference_is_deterministic` and `test_hf_reference_is_deterministic`: same seeded
input, two runs, `torch.equal` plus a SHA-256 of the output bytes.

| Oracle | Input | SHA-256, run 1 | SHA-256, run 2 |
|---|---|---|---|
| hand-written | seed 0, `randn[1,128,4096]`, random layer weights | `0a16330068bce3807911a80462fad1b38c84cfd3f08ba9098df38156f67cb96b` | identical |
| HF `LlamaDecoderLayer` | same input, same weights | `0a16330068bce3807911a80462fad1b38c84cfd3f08ba9098df38156f67cb96b` | identical |

The two oracles produce **the same hash**, which is the bit-exactness result stated as a hash rather
than as a PCC.

## 6. Validation summary (the `G-REF` numbers)

| Check | Result |
|---|---|
| hand-written vs HF, random weights, seq 128 | **PCC 1.0**, `max|Δ| = 0.0` |
| hand-written vs HF, real layer-0 weights, seq 64 | **PCC 1.0**, `max|Δ| = 0.0` |
| causality, both oracles | `max|Δ| = 0.0` masked / `3.515e+00` unmasked |
| llama3 `inv_freq` vs `models/tt_transformers/tt/common.py:489` `precompute_freqs` | `max|Δ| = 0.0` (exact) |
| llama3 scaling active (control) | scaled vs unscaled `inv_freq` `max|Δ| = 8.894e-04`; `cos` at position 8192 differs by `1.882e+00` |
| bundled `config.json` == checkpoint's | byte-identical (`filecmp`, `shallow=False`) |
| negative controls rejected | wrong theta **0.87437**, GQA `repeat` instead of `repeat_interleave` **0.55396**, no RoPE **0.86480** — all below the 0.9999 assertion |

**Read the bit-exactness honestly.** Two oracles agreeing bit-exactly proves the transcription is
faithful; it does **not** prove either is right about Llama's architecture — both could share a
misreading (`07_RISKS.md` R-006). What guards against that is the P0 card's per-row provenance and
`G-MODEL`'s top-1 check against HF with real weights.

## 7. What this reference deliberately does not include

- **No `reference/` package** and no vendored modeling code (recipe P1; `DEC-005`).
- **No decode path** — this bring-up is prefill only.
- **No KV cache in the reference** — the golden KV trace is generated in P7 by
  `scripts/generate_golden_kv_cache.py`, whose oracle is HF's own loop
  (template: `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py`).
- **No `bfloat16` anywhere.** If a future test needs a bf16 reference for an A/B, it must go through
  `quantize_like_device`, so the rounding is explicit and confined to the tensors the device stores.
