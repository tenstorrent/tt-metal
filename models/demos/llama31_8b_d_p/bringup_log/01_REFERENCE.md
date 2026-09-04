<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 01 — Reference implementation

**Phase:** P1 · **Date (UTC):** 2026-09-03 · **Gate:** `G-REF` — **PASS**

---

## 1. Which option, and why

The recipe ranks three strategies (`BRINGUP_RECIPE.md:292-313`). The chosen combination is the one
the recipe recommends — *hand-written torch math inside each unit test, driven by identical random
weights, plus an HF reference for the layer/model level* — with **one forced substitution**:

> The HF reference is reached through `transformers` **directly**
> (`LlamaConfig`, `LlamaDecoderLayer`, `LlamaForCausalLM`), **not** through
> `models/tt_transformers/tt/model_config.py`'s `ModelArgs.reference_*` accessors.

That substitution is not a preference. `ModelArgs.__init__` **raises** without a staged checkpoint —
`models/tt_transformers/tt/model_config.py:702`:

```python
raise ValueError("Please set HF_MODEL to a HuggingFace name ...")
```

(`HF_MODEL` read at `:683`, `self.CKPT_DIR = HF_MODEL` at `:687`.) Every accessor —
`reference_transformer` (`:4037`), `reference_decoder` (`:4393`), `reference_attention` (`:4410`),
`reference_mlp` (`:4365`), `reference_rms_norm` (`:4167`), `reference_embedding` (`:4379`),
`reference_lm_head` (`:4027`) — is a method on that object and funnels through
`reference_transformer`, which calls `model_cls.from_pretrained(self.CKPT_DIR, ...)` at `:4126-4144`.
All seven recipe line numbers are correct; the objects are simply unreachable here, because there is
no checkpoint on this machine (`07_RISKS.md` `R-003`).

Going straight to `transformers` is also *closer to the recipe's own stated rationale* for option 1
— "nothing to vendor, nothing to keep in sync" (`:295-296`). Logged as `DEC-004`.

**No `reference/model.py` is vendored** (`DEC-003`, `DEC-004`). Llama is first-class in
`transformers` with no `trust_remote_code`, which is exactly the condition under which the recipe
says a self-contained reference is *not* needed (`:301-304`).

### The two oracles, and their division of labour

| Oracle | Where it lives | Used by | Cost |
|---|---|---|---|
| **Hand-written torch math** | inside each `tests/unit/test_*_vs_ref.py` | every P5/P6 module gate | microseconds; no HF construction, no checkpoint, no network |
| **HF `transformers`** (`LlamaDecoderLayer`, `LlamaForCausalLM`) | imported | `G-REF` (to validate the hand-written one), `G-LAYER` / `G-MODEL` | seconds per build |

`G-REF` is what makes this split legitimate: it proves the cheap oracle *is* the expensive one, so
the P5/P6 gates lose nothing by using it.

---

## 2. How it is invoked

### Dimensions only — no HF, no network, no checkpoint, no device

```python
from models.demos.llama31_8b_d_p.tests.test_factory import llama_config_dims, rope_theta, rope_scaling

dims = llama_config_dims()      # raw configs/Llama-3.1-8B-Instruct/config.json + derived keys
dims["head_dim"]                # 128  (derived: hidden_size // num_attention_heads; key is absent)
dims["gqa_group_size"]          # 4    (derived: 32 // 8)
rope_theta(dims)                # 500000.0, via tt_transformers get_rope_theta -- transformers-5.x safe
rope_scaling(dims)              # asserts low_freq_factor==1.0 and high_freq_factor==4.0 (see R-006)
```

`llama_config_dims()` reads the **bundled** config so nothing in the P5/P6 inner loop touches the
network or `HF_MODEL`. `configs/Llama-3.1-8B-Instruct/config.json` is a **verbatim** copy of
`models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`
(sha256 `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`, asserted equal by
`test_bundled_config_matches_upstream`) — `DEC-005`.

**Never read θ as `LlamaConfig.rope_theta`.** Under the installed `transformers` 5.12.1 that
attribute exists and is `None`; the value moved into `rope_parameters`. `rope_theta()` routes through
`models/tt_transformers/tt/common.py:165` `get_rope_theta(config: dict, default=None)` and asserts
non-`None`. See `07_RISKS.md` `R-002` — this is a silent-wrong-RoPE trap, and two call sites in
`gpt_oss_d_p` are latently affected by it.

### One HF decoder layer, random weights (no checkpoint)

```python
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

cfg = LlamaConfig(**dims_subset, rope_theta=..., rope_scaling=dims["rope_scaling"], dtype=torch.float32)
cfg._attn_implementation = "eager"        # see section 3
layer  = LlamaDecoderLayer(cfg, layer_idx=0).to(torch.float32).eval()
cos, sin = LlamaRotaryEmbedding(cfg)(dummy_fp32_tensor, position_ids)
out = layer(hidden_states=x, attention_mask=causal, position_ids=pos, position_embeddings=(cos, sin))
```

Exact builders: `tests/unit/test_reference_model.py::_hf_config`, `::_build_layer`,
`::_rope_tables`, `::_run_hf`.

`LlamaDecoderLayer.forward` in `transformers` 5.12.1 returns a **bare tensor**, not a tuple, and
takes `position_embeddings=(cos, sin)` — the rotary tables are computed by the *model*, not the
layer, so a per-layer test must build them itself.

### The whole model with real state-dict key names, still no checkpoint

```python
from transformers import LlamaForCausalLM
model = LlamaForCausalLM(cfg)              # randomly initialised, REAL key names
sorted(model.state_dict())                 # model.embed_tokens.weight, model.layers.0..., lm_head.weight
```

This is what makes HF→Meta key-mapping work testable with no weights at all (P6.2 / `G-WEIGHTS`
structural half). The mapping itself is `models/tt_transformers/tt/load_checkpoints.py:800`
`map_hf_to_meta_keys` — see `02_SURVEY.md`.

### Real weights, when a checkpoint exists

`tests/test_factory.py` exposes `requires_hf_reference`, a `pytest.mark.skipif` on `HF_MODEL` not
being a directory, and `conftest.py` exposes a session-scoped `state_dict` fixture that returns `{}`
rather than raising. Real-weight tests therefore **skip** on this machine and **run** unchanged on a
machine with weights. `mesh_device` and `reset_seeds` are *not* redefined — they come from the
repo-root `conftest.py:554` and `conftest.py:34` (both verified) per `BRINGUP_RECIPE.md:329-330`.

---

## 3. dtype policy

**Compute the reference in fp32; cast only at the comparison boundary.**
(Recipe `:318-319`, following the `minimax_m3` golden-KV script header.)

| Stage | dtype |
|---|---|
| reference weights, activations, RoPE tables, all matmuls | **fp32** (`REF_DTYPE = torch.float32`) |
| reference softmax | fp32 accumulate — HF's `eager_attention_forward` does `softmax(..., dtype=torch.float32)` and casts back |
| the `G-REF` cross-comparison | fp32 vs fp32 |
| P5+ device comparison | the *device* tensor is brought to host and compared in fp32; the reference is never computed in bf16 |
| golden KV artefacts (P7) | generated in fp32, stored fp32 |

Two further pins that make bit-identical reruns possible at all:

1. **`cfg._attn_implementation = "eager"`.** This forces
   `transformers.models.llama.modeling_llama.eager_attention_forward` — explicit
   `matmul → *scale → +mask → softmax → matmul` — rather than a fused SDPA backend whose reduction
   order is unspecified and can vary with backend selection. Without this pin, assertion (a) of
   `G-REF` is not reliably achievable.
2. **The causal mask is built explicitly**, `[1, 1, S, S]` additive, `-inf` strictly above the
   diagonal (`_causal_mask`). `eager_attention_forward` applies whatever mask it is handed **and
   nothing more** — passing `attention_mask=None` yields *non-causal* attention, silently. The same
   explicit construction is what `BRINGUP_RECIPE.md:691-692` requires of the P5.5 attention
   reference.

Norm weights are initialised to `1 + 0.1·N(0,1)`, not left at HF's default of exactly ones. A norm
whose gain is all-ones makes the weight multiply a no-op and would hide a whole class of
weight-loading bug.

---

## 4. Determinism, measured

`G-REF` builds the oracle **twice from the same seed** — rebuilding, not re-running one object — so
what is proved is that the whole `seed → weights → forward` chain is reproducible.

| Case | dims | sha256 of the output hidden state | `torch.equal` |
|---|---|---|---|
| HF `LlamaDecoderLayer` | full (4096 / 32Q / 8KV / 128 / 14336, S=128) | `82cea4baa3e1e5210f88107b7044ee7be25f733331148cc4c1fd5ab84d28fb4b` (run 0 **and** run 1) | **True**, `max\|Δ\| = 0.0` |
| HF `LlamaDecoderLayer` | tiny (256 / 8Q / 2KV / 32 / 512, S=64) | `e19a867af264f74f01c6225b9489dc854af534cadde1deb0a9643a1ae904071c` (run 0 **and** run 1) | **True**, `max\|Δ\| = 0.0` |
| hand-written reference | full | `82cea4baa3e1e5210f88107b7044ee7be25f733331148cc4c1fd5ab84d28fb4b` | **True** |
| hand-written reference | tiny | `e19a867af264f74f01c6225b9489dc854af534cadde1deb0a9643a1ae904071c` | **True** |

---

## 5. Cross-reference: the two oracles agree **exactly**

| dims | PCC | `max\|Δ\|` | rel-L2 | threshold |
|---|---|---|---|---|
| full | **1.0** | **0.000e+00** | **0.000e+00** | ≥ 0.9999 |
| tiny | **1.0** | **0.000e+00** | **0.000e+00** | ≥ 0.9999 |

The agreement is **bit-exact**, not merely within tolerance — the hand-written hashes in §4 are the
same strings as the HF hashes. Read this honestly:

- **What it proves.** The hand-written reference is a *correct transcription* of the Llama decoder
  layer: same norm placement, same residual structure, same projection orientation
  (`x @ W.T` for HF's `[out, in]` storage), same RoPE convention (HF `rotate_half`, halves
  concatenated), same GQA expansion factor, same `scaling = head_dim**-0.5`, same SwiGLU
  (`down(silu(gate) * up)`), no biases, plain RMSNorm with no `+1` fold.
- **Why it is bit-exact rather than merely close.** Both paths reduce to the *same sequence* of
  `torch.matmul` / `softmax` / elementwise calls on the same fp32 tensors, and torch's CPU kernels
  are deterministic, so there is no reassociation difference to produce one. `repeat_interleave`
  (ours) and `expand`+`reshape` (HF's `repeat_kv`) differ in memory strategy but not in values.
- **What it therefore does *not* prove.** It is not an independent *numerical* check — a shared
  misreading of the architecture would be invisible. The independence that matters is against the
  **device** implementation, which is what P5/P6 supply. What `G-REF` buys is the licence to use the
  cheap oracle in those gates: any P5 PCC failure is then attributable to the TT code, not to two
  disagreeing references.

**Additional cross-check, not required by the gate but the highest-value de-risking in P1:**
`models/tt_transformers/tt/common.py:489` `precompute_freqs` — the helper P5.3 will reuse — agrees
with HF's llama3 RoPE **exactly** on the frequency tables:

| check | measured |
|---|---|
| `precompute_freqs` cos vs HF cos (first half), S=256, head_dim=128 | `max\|Δ\| = 0.000e+00` |
| `precompute_freqs` sin vs HF sin (first half) | `max\|Δ\| = 0.000e+00` |

That removes the largest single risk in P5.3 before a line of `tt/rope.py` exists. The test also
pins the **convention difference** Appendix B names as the classic RoPE bug:

| convention | table shape | expansion to `head_dim` |
|---|---|---|
| HF | `[S, head_dim]` | `cat(freqs, freqs)` — **halves concatenated**; verified `cos[:, 64:] == cos[:, :64]` bit-exactly |
| Meta / `tt_transformers` | `precompute_freqs` → `[S, head_dim/2]`, then `gather_cos_sin` (`common.py:525`) | `stack([c, c], -1).flatten(-2)` — **pairs interleaved** |

Same underlying `[S, head_dim/2]` frequency table; only the expansion differs. Mixing the two is
what produces "attention PCC ~0.5–0.9 with norms fine".

---

## 6. The llama3 RoPE scaling is active, measured

A RoPE test that passes with scaling silently disabled is worthless (`BRINGUP_RECIPE.md:650-652`),
so the fact is pinned in P1 rather than only at `G-ROPE`.

| quantity | value | source |
|---|---|---|
| `factor` | 8.0 | `config.json:27` |
| `low_freq_factor` / `high_freq_factor` | 1.0 / 4.0 | `config.json:28-29` |
| `original_max_position_embeddings` | 8192 | `config.json:30` |
| θ | 500000.0 | `config.json:32` (via `get_rope_theta`) |
| `low_freq_wavelen = orig / low_freq_factor` | **8192.0** | measured |
| `high_freq_wavelen = orig / high_freq_factor` | **2048.0** | measured |
| `inv_freq` slots that differ, scaled vs unscaled | **35 / 64** | measured |
| max relative deviation | **0.875000** | measured |
| analytic expectation `1 − 1/factor` | **0.875000** | exact match |

The test additionally asserts the *shape* of the schedule, not just that something changed:

- every frequency with `wavelength > low_freq_wavelen` equals `unscaled / factor` **exactly**
  (`rtol=1e-12`) — the low-frequency limb;
- every frequency with `wavelength < high_freq_wavelen` is **bit-identical** to unscaled
  (`rtol=0, atol=0`) — the high-frequency limb;
- the maximum relative deviation is exactly `1 − 1/factor`.

So a regression that disabled scaling, changed `factor`, or swapped the two limbs would all fail
distinctly, rather than all failing as "PCC dropped".

---

## 7. Gate `G-REF` — verdict

```
pytest models/demos/llama31_8b_d_p/tests/unit/test_reference_model.py -x -q
=> 9 passed, 1 warning in 13.84s
```

Raw log: `bringup_log/raw/G-REF_20260903T161226Z.log`. Host only; no device opened.

| Recipe condition (`:337-341`) | Result |
|---|---|
| (a) fixed-seed hidden state twice, **bit-identical** | ✅ 4/4 cases; sha256 pairs identical; `max\|Δ\| = 0.0` |
| (b) hand-written vs HF agree **PCC ≥ 0.9999** on one layer | ✅ PCC = **1.0** both dim sets (bit-exact) |
| (c) `01_REFERENCE.md` documents invocation + dtype policy | ✅ §2 and §3 |

**Verdict: PASS.** The oracle is deterministic, self-consistent, needs no checkpoint and no device,
and its dtype policy is fp32-until-the-comparison-boundary.

Tests, and what each is for:

| test | proves |
|---|---|
| `test_reference_is_deterministic[full,tiny]` | (a) for the HF oracle |
| `test_handwritten_reference_is_deterministic[full,tiny]` | (a) for the oracle the module gates use |
| `test_handwritten_matches_hf_decoder_layer[full,tiny]` | (b), plus the per-layer key set is exactly the 9 the card lists and contains no bias |
| `test_llama3_rope_scaling_is_active` | scaling is on, with the right factor and the right limb structure |
| `test_tt_transformers_precompute_freqs_matches_hf` | the repo helper P5.3 reuses matches HF; the Meta-vs-HF expansion difference is pinned |
| `test_bundled_config_matches_upstream` | the bundled config has not drifted; `head_dim` is still absent and still derives to 128 |

---

## 8. Files delivered in P1

| Path | Purpose |
|---|---|
| `configs/Llama-3.1-8B-Instruct/config.json` | bundled dims, verbatim copy (`DEC-005`) |
| `tests/test_factory.py` | `llama_config_dims`, `rope_theta`, `rope_scaling`, `requires_hf_reference`, `TestFactory.setup_test` |
| `conftest.py` | session `state_dict` fixture + `--skip-model-load`; does **not** redefine `mesh_device` / `reset_seeds` |
| `tests/unit/test_reference_model.py` | the `G-REF` gate |

---

## 9. What P3+ must carry forward from this phase

1. **Route every θ read through `get_rope_theta`** and assert non-`None`. Do not copy
   `getattr(cfg, "rope_theta", DEFAULT)` from `gpt_oss_d_p` (`R-002`).
2. **`precompute_freqs` is validated against HF** — reuse it in `tt/rope.py`, do not rewrite it.
3. **Assert `low_freq_factor == 1.0` / `high_freq_factor == 4.0`** before delegating to
   `apply_scaling` / `compute_llama3_parameters`, because those values are hard-coded in the callee
   and a differing config would be silently ignored (`R-006`). `test_factory.rope_scaling()` already
   does this — call it.
4. **The device must use the *Meta interleaved* expansion** with
   `ttnn.experimental.rotary_embedding_llama`, while the torch reference uses the *HF concatenated*
   one. Build both from **one** frequency set inside the test, as
   `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py::_build_cos_sin` (`:83`) does, so
   the test cannot silently compare two different RoPEs and call it a pass.
5. **Reuse `_torch_decoder_layer`, `_rms_norm`, `_apply_rope_hf`, `_causal_mask`** from
   `tests/unit/test_reference_model.py` in the P5/P6 module tests rather than re-transcribing the
   math — they are now gate-validated. If they are needed by three or more test files, promote them
   to `tests/reference_math.py` (that is a `DEC`, not a silent refactor).
6. `cfg._attn_implementation = "eager"` and an **explicit** causal mask are load-bearing. Omitting
   the mask gives non-causal attention with no error.
7. The moment a checkpoint is staged, `ModelArgs.reference_*` becomes usable and is worth switching
   to for `G-LAYER` / `G-MODEL` — it handles the Meta↔HF weight conversion for you
   (`reference_mlp` monkey-patches `load_state_dict` at
   `models/tt_transformers/tt/model_config.py:4368-4376`). `R-005`.
