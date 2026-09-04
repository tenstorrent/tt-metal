# Nomic Embed Text v2 MoE — TTNN Bring-Up Plan (single Blackhole chip)

## Context

`nomic-ai/nomic-embed-text-v2-moe` is a multilingual **encoder-only** text-embedding model
with a **Mixture-of-Experts** FFN (475M total / 305M active). Umbrella issue
[#54916](https://github.com/tenstorrent/tt-metal/issues/54916) tracks making it run on one
Blackhole chip as a TTNN port of the golden PyTorch reference, under `models/experimental/`.

| Issue | Title | Body |
|---|---|---|
| [#54917](https://github.com/tenstorrent/tt-metal/issues/54917) | **Phase 0 — Architectural Overview** | Get familiar with the model architecture; get familiar with the PyTorch implementation; be able to run the model on small inputs. |
| [#54918](https://github.com/tenstorrent/tt-metal/issues/54918) | **Phase 1 — First working PoC** | The `ttnn` variant becomes functional on a single BH chip. Performance out of scope; only correct output matters. |

The branch is `avelickov/54917-…-architectural-overview`, so **this PR is Phase 0**, per
`CLAUDE.md`'s "keep the PR limited to the linked sub-issue".

**Your decisions:** Phase 0 only in this PR · vendored reference under `reference/` as the
golden · weights downloaded and assumptions settled up front · router gated on agreement +
routed-subset PCC · no `tests/perf/` until Phase 1.

**Status: every load-bearing assumption has been executed and resolved.** Nothing below is
inferred from the model name, the paper, or a code read alone. The residue is in §D.

---

# What was verified, and how

## Model side — run against the real checkpoint

Downloaded `model.safetensors` (1.77 GB) at sha `1066b6599d099fbb93dfcb64f9c37a7c9e503e85`
and executed the HF remote model.

| Item | Result |
|---|---|
| Checkpoint contract | **148 tensors, 475,292,928 params, all F32.** MoE at layers **[1,3,5,7,9,11]**, dense at **[0,2,4,6,8,10]**. No router bias, no per-expert bias. No `position_embeddings` / `pooler` / `cls.` / `inv_freq` / `norm_factor` / `lm_head` / `ln_f`. |
| **Expert weight orientation** — the silent-failure item | `x @ w1[e].T` with `w1[e]=[3072,768]`, and `act @ w2[e]` **untransposed** with `w2[e]=[3072,768]` → **PCC 1.00000000** against HF. The alternatives score **−0.013** and **0.008** (garbage) or raise a shape error. |
| `<pad>` embedding row | `word_embeddings.weight[1]` absmax **1.5e-2 — NOT zero**. ⇒ **do not pass `padding_idx`** to `ttnn.embedding`; zeroing it would diverge from HF. |
| `strict=True` load vs the HF module tree | **0 missing, 0 unexpected.** Validates the name-isomorphism strategy (no weight remapper). |
| transformers 5.12.1 runs the remote model | **Yes.** Config *and* model resolve from `transformers_modules`, not native. Forward returns finite `[2,10,768]`. |
| Remote **code** repo sha | `7710840340a098cfb869c4f65e87cf2b1b70caca` — a **second** sha to pin, in `nomic-ai/nomic-bert-2048`. |
| **Model card's 0.9118** | Reproduced: **0.911788** (diff 1.2e-5). ⇒ the megablocks-vs-pure-torch worry is a non-issue; the card's number matches the pure-torch path. |
| top-2 weights on real data | rowsum mean **0.6655** ⇒ confirmed **not renormalized**. |
| Tokenizer | `XLMRobertaTokenizer`, `is_fast=True`, len **250002**, pad 1 / bos 0 / eos 2. `protobuf` not needed. The `nomic_bert → BertTokenizer` mapping trap is **inert** — `tokenizer_config.json` outranks it. |
| `Wqkv` layout | **three-major confirmed against the real model**: `qkv[..., 0:768]` ≡ `rearrange(…, three=3)`; head-major does not match. |
| Config | Every field re-confirmed: 768/12/12/3072, eps **1e-5**, `prenorm False`, `causal False`, `add_pooling_layer False`, rotary base 10000 / fraction 1.0 / **interleaved False**, experts 8 / top_k 2 / every_n 2, `moe_normalize_expert_weights False`, dropouts 0.0. |

**Two upstream quirks the reference must not inherit** (both executed):
- HF's `forward` **requires** `attention_mask` — omitting it raises `AttributeError`.
- `matryoshka_dim=256` returns `last_hidden_state` of shape `(2, 10, 768)` — it slices the
  **sequence** axis. An upstream bug. Use the model card's path instead.

**Per-layer output magnitudes** (for setting Phase-1 tolerances): layer 0 absmax 7.2 / std
0.37; layer 1 (MoE) absmax **21.2** / std 0.64; layer 5 absmax 20.9; layer 11 absmax 5.9.

## Device side — run on Blackhole

| Gate | Result |
|---|---|
| **Full MoE layer, real dims** (E=8, I=3072, H=768, T=512), batched dense-all-experts | **PCC 0.998942** ✓ |
| Broadcast-batch matmul `[1,1,T,768] × [1,8,768,3072]` | ✓ PCC 0.999979 |
| `fast_reduce_nc(dims=[1])` | ✓ 0.999992 |
| `reshape [1,1,T,H] ↔ [B,1,S,H]` | ✓ round-trip **exact** |
| `nlp_create_qkv_heads` on three-major | ✓ **0.999999** (head-major control: 0.082) |
| `rotary_embedding_hf`, NeoX, batch folded into heads | ✓ **0.999995** (interleaved control: 0.287) |
| SDPA `is_causal=False` + additive mask | ✓ 0.999776 |
| `layer_norm(residual_input_tensor=…)` / `embedding` | ✓ 0.999995 / 0.999999 |
| `ttnn.topk` on W=8 | ✓ **fp32 and bf16 both work**; indices all < 8; exact index match vs torch |

### Four device findings that change the design

1. **`ttnn.softmax` needs an explicit HiFi4 compute-kernel config.** Default: maxabs **1.91e-2**
   vs torch. With `compute_kernel_config=HiFi4`: **1.31e-3** — a 14× improvement.
   `numeric_stable=True` changes nothing. Not optional for the router.
2. **`ttnn.scatter` rejects `float32`** — in TILE *and* ROW_MAJOR (`TT_FATAL … !(input_dtype ==
   FLOAT32 && input_layout == TILE)`). bf16 works in both. ⇒ the router **cannot** stay fp32
   end-to-end; cast to bf16 before the scatter.
3. **`TT_VISIBLE_DEVICES=0` fails on this p300c** — `TT_FATAL: Custom fabric mesh graph
   descriptor path must be specified for CUSTOM cluster type`. Use the plain `device` fixture
   / `open_device(device_id=0)` with no `TT_VISIBLE_DEVICES`.
4. **The compute grid is 11×10 = 110 cores**, `dram_grid_size().x = 8` — not the (8,10) that
   `models/tt_transformers/tt/model_config.py:1815` implies. Always derive from
   `mesh_device.compute_with_storage_grid_size()`.

### GELU, quantified

| | vs exact erf (maxabs) |
|---|---|
| `ttnn.gelu` default (`Accurate`), fp32 | **9.5e-7** — essentially exact |
| `ttnn.gelu` default (`Accurate`), bf16 | 1.58e-2 — the bf16 noise floor |
| `fast_and_approximate_mode=True` (`FastLut`) | **2.34e-2**, *independent of dtype* |

The LUT error (2.3e-2) **exceeds** the bf16 noise floor (1.6e-2), so it is not swamped. The
repo's BERT idiom `fused_activation=(ttnn.UnaryOpType.GELU, True)` selects that LUT — do not
copy it.

### Why PCC is the wrong gate for the MoE — measured, not argued

Against the **real** weights, comparing each known bug to HF's own MoE-layer output:

| | PCC vs HF |
|---|---|
| Correct implementation | **1.00000000** |
| **BUG: bias inside the expert loop** | **0.99999980** |
| **BUG: renormalized top-2** | 0.99266372 |

The bias bug is *structurally invisible* to PCC: it produces a near-constant additive offset
`(Σw − 1)·b`, and PCC mean-centers before correlating. No threshold catches it. It must be
caught by max-abs / `allclose`. The renormalization bug at 0.993 would pass any gate set at
0.99. Both need dedicated negative-control tests.

---

# Assumption register — final state

## §D — Residual, and how each is handled

Everything else was executed. These four remain, all non-blocking:

| | Assumption | Handling |
|---|---|---|
| D1 | The **router's ~0.6% index disagreement** is inherent, not a bug | Measured **99.41%** set-agreement at real dims (T=512). Top-2 over 8 experts is a discrete decision; bf16 softmax+topk cannot match fp32 torch on near-ties. Gate per your decision: agreement ≥99% **and** every disagreeing token shown to have a near-tied margin. |
| D2 | End-to-end 12-layer TTNN PCC | Not yet measurable — no TTNN model exists. Per-layer inputs verified; the ≥0.98 target at step 10 is an estimate informed by the measured per-layer magnitudes and bge_m3's 0.94-at-bf8 precedent. |
| D3 | Blackhole DRAM headroom for ~951 MB resident + ~50 MB transient | No documented per-chip figure in-repo; `tt-smi` reports speed, not capacity. Confirm by allocation at Phase 1 step 1. |
| D4 | `fp32_dest_acc_en` corrupting BH expert matmuls | Taken from two in-repo findings (#49068), not independently reproduced. We default it **off** for matmuls, which is the safe side either way. |

## §E — Corrections to my earlier drafts

1. **"Phase 0 tests only need to avoid the `device` fixture."** Was wrong for an unbuilt tree —
   the root conftest imports `ttnn.device` and calls `ttnn.get_arch_name()` in
   `pytest_addoption`. **Now moot:** with the build in place, all test-infra imports work and
   `pytest --collect-only` succeeds (369 tests). No `--confcutdir`, no `comp_pcc` shim.
2. **"`ttnn.topk` requires bf16."** Wrong — fp32 works. What genuinely fails is `bfloat8_b`.
3. **"Matryoshka order is load-bearing."** Over-stated: it changes the norm (1.0 vs ≈0.57) but
   both are positive multiples of the same vector, so **cosine is identical**. Follow the card's
   order; pin the equivalence as a lemma.
4. **Concat-dense as the MoE choice.** Revised to **batched dense-all-experts** — same
   arithmetic position as the reference, ⅓ the peak memory, no static `expand` matrix, and now
   measured at PCC 0.9989 on hardware.
5. **The tokenizer trap.** Real for the config/model, **inert** for the tokenizer.
6. **`TT_VISIBLE_DEVICES=0` to pin one chip.** Fails on this p300c; see device finding 3.
7. **Grid (8,10).** It is **11×10**.

### The trap that remains fully live

`transformers 5.12.1` ships a native `transformers.models.nomic_bert` targeting
**nomic-embed-text-v1.5** — separate q/k/v/o with `bias=False`, SwiGLU, `layer_norm_eps 1e-12`,
rope theta 1000.0, **no MoE**. Its `@strict` config silently accepts the v2-moe `config.json`
and drops the GPT-2-style keys. A bare `AutoConfig`/`AutoModel` yields a plausible, wrong model
with no error. The fix, now verified to work: force remote-code resolution and **assert the
resolved class module starts with `transformers_modules`**.

---

# Phase 0 — this PR

## File layout

```
models/experimental/nomic_embed_text_v2_moe/          # dirs already exist, empty + untracked
├── README.md                  # setup, test commands, the trap warnings
├── __init__.py  .gitignore    # .gitignore blocks *.pt|*.pth|*.bin|*.npz|*.safetensors
├── common.py                  # BOTH pinned shas, checkpoint resolution, hook harness
├── reference/
│   ├── ARCHITECTURE.md        # the #54917 deliverable: verified facts + operator mapping
│   ├── config.json            # vendored pinned snapshot (~1.5 KB) for the no-network tests
│   ├── configuration_nomic_moe.py   # NomicMoEConfig + from_hf_config that RAISES on 17 assumptions
│   ├── modeling_nomic_moe.py  # ~380 lines, upstream-identical names → strict=True is the proof
│   ├── loader.py              # expected_checkpoint_keys(config) GENERATED, not stored
│   ├── pipeline.py            # prompts, tokenize, mean_pool, truncate, l2_normalize, encode
│   └── hf_reference.py        # trap containment; bypasses AutoModel.from_pretrained
├── tests/
│   ├── conftest.py
│   └── pcc/
│       ├── test_checkpoint_contract.py
│       ├── test_reference_modules.py      # no network, no weights — the backbone
│       ├── test_reference_vs_hf.py        # 13-point per-layer parity ladder
│       └── test_embedding_pipeline.py     # 0.9118, matryoshka, small-input smoke
└── tt/                        # stays empty this PR
```

**No `tests/perf/`** (your call): in this repo it means *device* performance and needs the
device fixture, the `models_device_performance_bare_metal` marker, and `prep_perf_report`. A
CPU-only version would be dead code that could pollute perf dashboards selecting by directory
+ marker. README states it arrives in Phase 1.

## The vendored reference

Three fidelity pillars, all now validated:

1. **Name isomorphism instead of a remapper** — mirror upstream's names (`Wqkv`, `emb_ln`,
   `mlp.experts.mlp.w1`) so `load_state_dict(strict=True)` *is* the structural proof.
   **Verified clean: 0 missing, 0 unexpected.**
2. **A validating config** — `from_hf_config` raises on all 17 baked-in assumptions
   (`prenorm`, `causal`, rotary fraction/interleaved/scale_base, `activation_function`,
   `moe_normalize_expert_weights`, `num_shared_experts`, `expert_choice_router`, dropouts,
   `type_vocab_size`, the three bias flags, `moe_top_k`, divisibility). Converts silent
   divergence into a loud error.
3. **Per-layer parity** — forward hooks at `emb_ln` and each `encoder.layers.{i}` give a
   13-point PCC ladder. End-to-end PCC alone can mask compensating errors.

~380 lines covering only the inference path. Excluded (~85% of upstream's 2556): the vision
tower, all task heads, the pooler, gated-MLP variants, DynamicNTK rotary, xPos, the megablocks
bridge, the custom `from_pretrained`, KV-cache, gradient checkpointing, the pre-norm branch,
and every `use_flash_attn`/`fused_*` path. `einops` is dropped so layout choices are explicit.

### Silent-divergence traps the reference must avoid

| Trap | Why it's silent |
|---|---|
| Renormalized top-k | Every Mixtral/Switch impl does it. Nomic must not — measured rowsum 0.6655. **PCC 0.993, passes a 0.99 gate.** |
| Softmax over top-k instead of all 8 | Implicitly renormalizes. |
| `w2` viewed `(E,H,F)` | `24576*768 == 8*768*3072`, so the view succeeds and the matmul typechecks. **Measured PCC −0.013 — pure garbage, no error raised.** |
| `w1` expert axis inner | Expert `e` owns rows `e*3072…`; the expert axis is outer. |
| Per-expert bias | One shared `[768]`, added once after the weighted sum. **PCC 0.9999998 — invisible.** |
| `repeat_interleave` for cos/sin | Gives the GPT-J layout; with NeoX `rotate_half` the 2×2 map isn't a rotation. Control measured at 0.287. |
| Pre-doubled rotary cache | Upstream caches `[S,32]` and doubles at apply time. |
| tanh GELU | Differs by ~5e-4 from erf; looks like a hardware precision problem later. |
| Applying the MoE `attention_mask` | It's an **inverted** pad mask (1 = pad) and upstream **ignores** it. Applying it would zero the real tokens. |
| Zeroing the `<pad>` embedding row | **Measured non-zero.** Do not pass `padding_idx`. |

## Phase 0 tests

| Test | Asserts | Network |
|---|---|---|
| `test_checkpoint_contract.py` | 148 keys == `expected_checkpoint_keys(config)` **generated from the config** (so the `i%2==1` predicate is itself under test); shapes/dtypes/param count; absence assertions; `strict=True` clean; vendored config == live config at the pinned sha; `<pad>` row non-zero | weights |
| `test_reference_modules.py` | **No network, no weights, <15 s.** Rotary closed form; `rotate_half` is NeoX **and explicitly not** interleaved; position-0 identity; per-plane norm preservation; relative-position identity; three-major `Wqkv` via an integer-bias probe; post-norm structure via zeroed branches; MoE placement; top-2 rowsum < 1 with a renormalized negative control; **bias-added-once via `allclose`, not PCC**; loop ≡ dense; exact-erf GELU with a tanh negative control | no |
| `test_reference_vs_hf.py` | Resolved class module starts with `transformers_modules` (the canary); function-level rotary parity; end-to-end and **13-point per-layer** ladder at PCC ≥ 0.9999999; records the `matryoshka_dim` sequence-slice bug and the mandatory-mask behaviour | yes |
| `test_embedding_pipeline.py` | Tokenizer identity + `AutoTokenizer` precedence canary; **0.9118 at abs=1e-4** (measured 0.911788); prefix actually changes the embedding; matryoshka at `d ∈ {768,512,256}` with the **cosine-invariance lemma**; the #54917 smoke set `(B,S) ∈ {(1,1),(1,4),(2,8),(3,17),(1,512)}` plus a ragged batch; padding invariance measured and logged | weights |

**Goldens stay far under the 500 KB cap:** nothing is a committed tensor. Committed data is
`config.json` (~1.5 KB), three integers, two shas, one float. Measured numbers go into
`ARCHITECTURE.md` as documentation, not assertions.

---

# Phase 1 — TTNN design (later PR, #54918)

Correctness only. `bfloat16` / `TILE_LAYOUT` / `DRAM_MEMORY_CONFIG`, no sharding. Canonical
activation layout is a flattened token axis `[1,1,B*S,768]` — every sub-block except attention
is token-wise, and flattening removes all batch-broadcast ambiguity from the MoE matmuls
(reshape verified exact). Do **not** copy `models/demos/blackhole/sentence_bert/ttnn/common.py`'s
program configs — that file is byte-identical to the Wormhole one and hardcodes a `(6,8)` grid.

## Operator mapping

| Reference | TTNN |
|---|---|
| `word_embeddings` | `ttnn.embedding(ids, weight, layout=TILE)` — **no `padding_idx`** (row 1 is non-zero) |
| `token_type_embeddings` | **fold to a constant**: `type_vocab_size==1`, so pre-add the `[1,768]` row into the table in fp32 at load |
| `nn.LayerNorm` + residual | `ttnn.layer_norm(x, epsilon=1e-5, weight=…, bias=…, residual_input_tensor=h)` — 2 torch ops → 1 |
| fused `Wqkv` | `ttnn.linear(h, W, bias=b)`, `W = cat([Wq.T,Wk.T,Wv.T],-1)` → `[768,2304]` |
| three-major split | `nlp_create_qkv_heads(qkv4, num_heads=12, num_kv_heads=12, transpose_k_heads=False)` — **not GQA**, **no permutation** |
| rotary | `rotary_embedding_hf(x, cos, sin, is_decode_mode=False)`; cos/sin built by **concat**-duplicating `[S,32]` → `[1,1,S,64]` bf16 TILE; batch folded into the head axis |
| SDPA | `scaled_dot_product_attention(q,k,v, attn_mask=…, is_causal=False, scale=1/8, …)` — `is_causal` **defaults True**, so passing False is mandatory |
| head concat | `ttnn.transformer.concatenate_heads` (returns rank-3) |
| GELU | `ttnn.gelu(x)` (default `Accurate`) as its **own op** — never the repo's `(GELU, True)` LUT |
| router | fp32 `ttnn.linear` → `ttnn.softmax(dim=-1, compute_kernel_config=HiFi4)` → **cast bf16** → `ttnn.topk(k=2)` → `ttnn.scatter(zeros_like, …)`. **No sum-normalization** — delete gemma4's `router.py:113-117` |
| experts | 2 broadcast-batch `ttnn.matmul` + `ttnn.gelu` + `ttnn.permute` + `ttnn.mul` + `fast_reduce_nc(dims=[1])` + `ttnn.add(bias)` |
| padding mask | `eq → to_layout(TILE) → reshape → where(pad,-100000.,0.) → expand → typecast`, built **once per forward**, shared across all 12 layers. Drive it from the tokenizer's `attention_mask`, not `input_ids != pad_token_id` |
| mean pool | `matmul(keep[B,1,1,S], hidden[B,1,S,D])` ÷ `clip(sum(keep,-1,keepdim=True), 1., S)` |
| L2 normalize | no single op: `mul → sum → rsqrt(+1e-12) → mul` |

## MoE — the verified formulation

```python
w1_tt = w1.view(8,3072,768).transpose(1,2).unsqueeze(0)   # [1,8,768,3072]  one transpose
w2_tt = w2.view(8,3072,768).unsqueeze(0)                  # [1,8,3072,768]  PURE VIEW
bias  = experts_bias.reshape(1,1,1,768)

logits = ttnn.linear(typecast(h, fp32), rw, compute_kernel_config=HIFI)   # fp32
p      = ttnn.softmax(logits, dim=-1, compute_kernel_config=HIFI)         # HiFi4 REQUIRED
v, i   = ttnn.topk(typecast(p, bf16), k=2, dim=-1)                        # scatter rejects fp32
dense  = ttnn.scatter(ttnn.zeros_like(p_bf16), dim=-1, index=i, src=v)    # NO renormalization

h1 = ttnn.matmul(h, w1_tt, compute_kernel_config=HIFI)   # [1,8,T,3072] broadcast-batch
a  = ttnn.gelu(h1)                                        # Accurate variant
o  = ttnn.matmul(a, w2_tt, compute_kernel_config=HIFI)   # [1,8,T,768]
o  = ttnn.mul(o, ttnn.permute(dense, (0,3,2,1)))         # [1,8,T,1] broadcast
y  = ttnn.experimental.fast_reduce_nc(o, dims=[1])        # [1,1,T,768]
return ttnn.add(y, bias)                                  # ONE shared bias, AFTER the sum
```

Measured **PCC 0.998942** at real dims on Blackhole. Peak transient ≈50 MB at B=1,S=512;
resident weights ≈951 MB.

## Bring-up order and gates

| # | Step | Gate |
|---|---|---|
| 1 | Embeddings | ≥0.999 + token-type-fold exactness |
| 2 | Rotary | ≥0.999 + analytic single-position probe + **interleaved oracle must fail** |
| 3 | Attention (no mask, then 25% padding) | ≥0.99, **kept positions only** |
| 4 | Dense MLP | ≥0.999 |
| 5 | **Router** | set-agreement **≥99%**; every disagreeing token shown to have a near-tied softmax margin; weights ≥0.999 on agreeing tokens; not-renormalized assertion |
| 6 | Experts (injected routing) | ≥0.998 **and max-abs** (not PCC) vs the bias-inside oracle |
| 7 | MoE layer | ≥0.998 on correctly-routed tokens + both negative controls |
| 8 | One dense + one MoE block | ≥0.99 |
| 9 | 12-layer encoder | ≥0.98 |
| 10 | Pool + truncate + normalize | ≥0.99 |
| 11 | End-to-end | ≥0.99 + cosine vs the reference within 0.005 + top-1 retrieval agreement |

**Steps 5–7 carry non-PCC assertions because PCC provably cannot catch their two most likely
bugs** (measured: 0.9999998 and 0.993). Step 5 is the hardest — routing is a discrete decision
and a flip changes a token's output entirely. Step 9 is the *least* worrying: post-norm
re-centers after every sub-block, so bf16 error cannot compound as it does in pre-norm.

## Blackhole specifics

`fp32_dest_acc_en=False` + HiFi4 for matmuls; `fp32_dest_acc_en=True` for norms/softmax;
`ttnn.Tile([32,32])` only (TinyTile broken, #31385); derive grid (**11×10**) and
`dram_grid_size().x` (**8**) from the device; avoid `topk_router_gpt` (unavailable on BH) and
`moe_compute` (fails PCC on BH); **no `TT_VISIBLE_DEVICES`** on this p300c.

---

# Phase 2 — device performance (later)

`@run_for_blackhole()` + `@pytest.mark.models_device_performance_bare_metal` driving
`run_device_perf` → `check_device_perf(assert_on_fail=True)` → `prep_device_perf_report`, with
`command` pointing at the Phase 1 e2e PCC test so the number always describes shipped code.
`device_params = {"l1_small_size": 16384, "trace_region_size": 0, "num_command_queues": 1}`.
Trace + 2CQ are not worth it in Phase 1 — ~200 DRAM-interleaved ops at M=512 are compute-bound,
the input is 2 KB, and an oversized trace reservation is itself an OOM risk.

---

# Verification

## Phase 0

```bash
cd /localdev/avelickov/tt-metal && source python_env/bin/activate
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc/ -v
pre-commit run --from-ref origin/main --to-ref HEAD
```

Weights are already cached at
`~/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v2-moe/snapshots/1066b65…`.

**Acceptance for #54917:** `ARCHITECTURE.md` documents the verified architecture and operator
mapping; the vendored reference matches HF on the 13-point ladder; the model runs on small
inputs and reproduces 0.911788.

## Phase 1

```bash
pytest models/experimental/nomic_embed_text_v2_moe/tests/pcc -v          # no TT_VISIBLE_DEVICES
TT_METAL_WATCHER=10 pytest .../tests/pcc/test_ttnn_model.py -v
```

Run Watcher and the profiler in separate runs — they contend for debug resources.

---

# Out of scope

No decoder, generation, KV cache, or multi-chip infrastructure; no optimization before
correctness; no `tt/` code in this PR; no `CLAUDE.md` edits without showing you the diff.

Once Phase 0 lands, `CLAUDE.md` is a good home for the durable facts established here — the
two pinned shas, the expert-orientation contract, the non-zero `<pad>` row, the HiFi4 softmax
requirement, and the `TT_VISIBLE_DEVICES`/grid notes. I'll propose that diff separately.
