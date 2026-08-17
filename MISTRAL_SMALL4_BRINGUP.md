# Mistral Small 4 119B — prefill bring-up

Working notes for the shared branch `mistral-small4-bringup`. Short on purpose: it records what is
on the branch and what still needs doing, not how to do it.

## Goal

Bring up `mistralai/Mistral-Small-4-119B-2603` prefill by **reusing `ttMLA` and `ttMoE` from
`deepseek_v3_d_p`**, not by writing a new model. Mistral is in-family — DeepSeek-style MLA attention
with identical weight naming, plus GPT-OSS-style MoE routing.

Order (from the prefill framework owner): chunked MLA → MoE → weight loading → transformer test = "something functional".
**Runner integration comes last** — no block test needs it.

## Environment

```bash
cd <checkout> && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
./python_env/bin/pytest ...
```

The working checkout for this branch is now **`/data/kmabee/tt-metal`**. Ready-made scripts (outside
git, so branch switches leave them alone) live in `/data/kmabee/mistral4_repro_logs/`: `00_build.sh`,
`10_reproduce.sh` (re-runs every measured row below, one log per row), `20_serve.sh` (the demo).

The caches are deliberately **outside** both checkouts, in `/home/kmabee/` — `mistral4_ttnn_cache`
(65 GB) and `mistral4_ref_cache`. They are keyed by `{variant}_{arch}_{N}dev/{sp}x{tp}`, not by
checkout path, so they survive moving the branch between checkouts. Export
`TT_MISTRAL4_PREFILL_TTNN_CACHE` and `TT_MISTRAL4_PREFILL_HOST_REF_CACHE` at them or the 36-layer row
costs 870 s instead of ~54 s.

Checkpoint (already downloaded, 113 GB) — point the env var at it, the adapter has no default path:

```bash
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
```

Block tests run on the **galaxy** at production shapes, not the QB2.

## Shapes

```
36 layers · hidden 4096 · vocab 131072 · 32 heads · 1M context · bf16/fp8
MLA:  q_lora 1024 · kv_lora 256 · qk_nope 64 · qk_rope 64 · v_head 128 · YaRN factor 128
MoE:  128 routed · top-4 · 1 shared · moe_inter 2048 · n_group 1 · first_k_dense_replace 0
```

Checkpoint is a `Mistral3ForConditionalGeneration` wrapper: the LM sits under `language_model.`
alongside a Pixtral `vision_tower` we ignore for text-only prefill.

## What is already on this branch

Everything lives under `models/demos/deepseek_v3_d_p/` unless noted.

| file | what |
|---|---|
| `reference/mistral_small4_config.py` | dims class + `mistral4_hf_config()` returning a real `Mistral4Config`; every field traced to `config.json` |
| `tt/runners/adapters/mistral_small4.py` | `MistralSmall4Adapter(MLAPrefillAdapter)`, incl. the HF reference classes |
| `tests/test_mla.py` | `test_mistral4_mla`, plus `mistral_small4` in `test_mla_chunked_prefill`'s variant list |
| `tests/test_prefill_block.py` | `test_mistral4_prefill_block` |
| `tests/test_prefill_transformer.py` | `test_mistral4_prefill_transformer` |
| `tt/mla/mla.py`, `reference/mla_reference.py` | `mla_disable_yarn_mscale` flag (see below) |
| `utils/test_utils.py` | per-tensor fp8 dequant |
| `utils/transformer_helpers.py` | stacked+fused expert split, zero router bias, reference binding by signature |
| `models/demos/common/prefill/adapter.py` | `"mistral_small4"` in `ADAPTER_PATHS` *(outside the folder)* |

**Mistral runs end to end on the galaxy with real weights.** embed -> MLA+MoE stack -> norm ->
lm_head -> sample, at production shapes on 32 chips. Measured so far, all at `(8, 4)`:

| what | result |
|---|---|
| MLA, plain + chunked, random weights | green |
| decoder block, PCC vs HF reference | **0.988** (bar 0.95) |
| transformer, 2 / 5 layers, random weights | passed |
| transformer, 2 layers, **real checkpoint** | passed |
| **transformer, all 36 layers, real checkpoint** | **passed** — 870 s cold (builds the cache), **54 s warm**, `tt_forward` ~3.0 s |

### Re-run in `/data/kmabee/tt-metal`, 2026-08-17 — what actually reproduces

Every row above was re-run after moving the branch to this checkout (`10_reproduce.sh`, one log per
row in `/data/kmabee/mistral4_repro_logs/`). The numerics reproduce **exactly**; two rows fail for
reasons that are not numerics.

| row | result | note |
|---|---|---|
| MLA plain, 8x4 line, PCC | **3 passed, 1 failed** | `seq25k-max_sl` hits the DeepSeek-tuned matmul config — see below |
| MLA chunked, 8x4 line, 13 rows | **13 failed → all 13 pass** | three stacked setup bugs, all fixed on this branch — see below. Includes `deep-50k+5k` and the `rot-*` scenarios previously listed as missing coverage |
| decoder block, 3 rows | **1 passed, 2 failed** | output PCC **0.987949** (doc: 0.988 ✓). Failure is the known KVPE **PE** part: 0.997590 vs the 0.999 bar. KV part 0.999898 passes |
| transformer 2 layers, real ckpt | **passed** | 62 s |
| transformer **36 layers, real ckpt** | **passed** | 84 s warm. First token `ID=2`, prob 0.9689 |
| full-model PCC, 36 layers | **xfailed** (as marked) | per-layer values **bit-identical** to those below; `TT==HF match: True` |

The 36-layer per-layer PCCs came back identical to the last digit — `layer_0` 0.999347, `layer_15`
0.974683, `layer_16` 0.965732, `layer_27` 0.941491, `layer_31` 0.905825, `layer_32` 0.173501, `norm`
0.959047, `lm_head` 0.962114. So the move changed nothing numerically, and the decay below is
reproducible rather than a one-off.

**`tt_forward` needs a qualifier: this run measured 31.96 s, not ~3.0 s.** Same forward, same shapes
— the difference is a cold program cache. A lone forward in a fresh process pays first-call program
compilation; the ~3 s figure is the warm number, and the demo independently measures **2.93 s** per
warm forward at 1k, which confirms it. Quote whichever you mean, but say which: for anything
latency-facing the warm number is the honest one, and for "how long will this test take" the cold one
is. Also from this run: `tt_transformer_creation` 30.6 s and `weights_creation` **0.09 ms** — a pure
TTNN-cache hit that never opens the checkpoint.

### ⚠ Full-model numerics: measured, and NOT yet correct

The rows above are **smoke** runs — they prove wiring, not numerics. The full-model PCC check now
exists (`pcc-json_prompts-pretrained`, no golden trace needed — the reference is computed
layer-by-layer) and it **fails**. Two separate things in that failure:

**Real:** per-layer PCC decays monotonically and compounds.

```
embed    1.000000     layer_11 0.992188     layer_20 0.946975     layer_28 0.904723
layer_0  0.999347     layer_13 0.983110     layer_23 0.942357     layer_31 0.905825
layer_5  0.998855     layer_15 0.974683 ←   layer_27 0.941491     norm     0.959047
layer_9  0.996916     layer_16 0.965732 ✗   (last PASS at 15)     lm_head  0.962114
```

Bar is 0.97, so it drops out from layer_16 on. Prime suspect: expert weights are stored
**`BFLOAT4_B`** and Mistral activates only 4 of 128 experts at `moe_intermediate 2048`, so 4-bit
expert error is diluted less here than for DeepSeek/Kimi. **Try a higher-precision expert dtype
first.**

**Likely an artifact, do not chase as numerics:** `layer_32..34` read 0.17–0.21 and `layer_35` 0.46,
which cannot be reconciled with `norm` 0.959, `lm_head` 0.962, and an **exact first-token match**
against the reference. `norm` is computed from `layer_35`'s output. Isolate the tail-layer comparison
before believing those four numbers. Likewise every `*_kvpe_*` entry reports `-1.0`, which is
`_compare_intermediate_pcc`'s "missing from TT intermediates" — a harness gap for this variant.

**The one unambiguously good number:** the full 36-layer model with real weights picks the **same
next token as the HF reference** — `ID=2 ['</s>']` at position 1023, `TT==HF match: True`. So the
output token is right even though the intermediate hidden states drift.

Two things to know about the adapter as written:

- `supports_pretrained = True`. The real checkpoint loads: per-tensor fp8 dequant, the stacked+fused
  expert split and the zero router bias are all handled, on both the random and the layer-by-layer
  pretrained paths.
- `default_gate_mode = "GPT_DEVICE"`. This started as an argument and is now **confirmed against the
  reference implementation**: `Mistral4MoE.route_tokens_to_experts` is
  `softmax(-1)` over all experts -> top-k -> gather -> renormalize -> x1.0, and with `n_group = 1`
  the group mask is all-ones so the grouping collapses out entirely. That is the same rule as the
  GPT-OSS gate. Still worth an independent per-expert token-count assertion.

## Next steps, in the order worth doing them

The detail per area is in "What still needs doing" below; this is the sequencing. The first three
are all about making the numbers trustworthy, because every later decision reads off them.

**1. Raise the expert weight dtype and re-run the full-model PCC.** One knob, decisive either way.
`routed_expert_weights_dtype=ttnn.bfloat4_b` is the default in three places —
`tt/moe/tt_moe.py:153`, `tt/moe/tt_routed_expert.py:252`, `utils/transformer_helpers.py:573`. Try
`bfloat8_b`. If the per-layer decay flattens, the 4-bit expert theory is confirmed and the price is
known (expert weights dominate the 65 GB TTNN cache, so budget roughly double that and one ~870 s
cold rebuild). If PCC is unchanged, experts are *exonerated* and the error is in MLA / norm /
routing — which is worth as much as a fix.

**2. Isolate the `layer_32..34 ≈ 0.17` reading before believing it.** It cannot coexist with `norm`
0.959, `lm_head` 0.962 and an exact first-token match, since `norm` is computed from `layer_35`.
Do this *before* step 1's re-run so that run's numbers are readable. Cheapest probe: print shapes
and a few element values for TT vs reference at `layer_31` and `layer_32` — a slot/offset mismatch
in the tail of the layer-by-layer reference shows up immediately.

**3. Close the `*_kvpe_* = -1.0` gap — it is missing coverage, not a bad score.**
`_compare_intermediate_pcc` (`tests/test_prefill_transformer.py:111-113`) returns `-1.0` for
"missing from TT intermediates": the reference emits those 72 labels (36 layers x kv/pe) and the TT
forward never puts KVPE into `tt_intermediates`. The plumbing already exists and is simply not
wired: `TtPrefillBlock.forward` takes `return_kv_cache` (`tt/tt_prefill_block.py:474`) and returns
`ttMLA.kv_cache_to_host(...)` (line 691), but `TtPrefillTransformer.forward` never passes it and
discards the second return value (`h, _ = ret`, `tt/tt_prefill_transformer.py:479`). So: pass
`return_kv_cache=return_intermediates` per layer, split the result at `kv_lora_rank` into
`layer_{i}_kvpe_kv` / `_pe`, and derive the compressed reference line the way the block test already
does (`_derive_mla_kvpe`, `tests/test_prefill_block.py:82` — note it must be computed in fp32; bf16
costs ~2e-3 PCC against a 0.999 bar). That turns 72 dead rows into the strongest per-layer attention
check available — and it is how you localise step 1 if the experts come back innocent.

**4. Assert per-expert token counts.** A routing bug that collapses onto one expert still produces
plausible text and still passes a loose PCC, so nothing above would catch it. Blocker: `TorchMoe`'s
gate hardcodes sigmoid/`noaux_tc`, so `test_ttnn_moe` cannot currently express Mistral's softmax
top-4 rule — it needs a small extension first.

**5. Capture a golden trace.** Removes the CPU-reference recompute from every PCC run, unblocks
`test_prefill_transformer_chunked`, and gives CI something to regress against.

Then the shape/coverage work (packed-FP8 alignment at `kv_lora_rank = 256`; chunked coverage at
`production-50k+5k`, `deep-*`, `with_determinism`, `metadata`; the embedding and LM-head probes),
and only last the runner + pipeline integration for PP=4 on 8x1 — nothing above needs it.

Separately: if generating text matters beyond the demo, **decode is a second bring-up, not a tweak**
— this folder has no decode path at all; the kernels live in `models/demos/deepseek_v3/`.

## What still needs doing

**Attention** — MLA is the most Mistral-divergent block.
- Broader chunked coverage: `with_determinism`, `metadata`, and the non-`line` topologies. The
  `cpu`-reference `rot-*`, `deep-*` (incl. `deep-50k+5k`), `maxedge` and `plain-5k` scenarios at
  8x4/line **now pass** (13 rows) after the config-path fixes below
- `seq25k-max_sl` plain MLA still fails on the DeepSeek-tuned matmul config (below)
- KV cache layout at `kv_lora_rank = 256`
- `reference_attention_cls` is now wired (transformers' `Mistral4Attention`), so `run_model`'s
  second reference check no longer silently no-ops — confirm it actually reports a PCC line.

**MoE** — the largest remaining unknown.
- Gate/routing at 128 experts, top-4, no correction bias
- Dispatch + combine, routed + shared experts
- **Assert per-expert token counts.** A routing bug that collapses onto one expert still produces
  plausible output and passes a loose PCC.

**Embeddings + LM head**
- `tests/pcc/test_parallel_embedding.py` and `tests/pcc/test_lm_head.py` at Mistral vocab/emb
- LM head in both column- and row-parallel modes
- These two are the per-stage probes we bisect with when full-model PCC is wrong

**Weights** — all of this now works; what is left is coverage, not construction
- Loading, dequant and the expert split are done and exercised end to end on the real checkpoint
- The TTNN cache builds at ~24 s/layer; the full 36-layer cache is **65 GB** for the whole 32-chip
  mesh (~1.8 GB/layer, ~57 MB/device/layer) at `$TT_MISTRAL4_PREFILL_TTNN_CACHE`
- Still to do: a golden trace to compare against, and the packed-FP8 KV format decision

**Integration**
- `test_prefill_block_chunked`, then the transformer test
- Adapter is done; runner + pipeline (`common/prefill/docs/ADDING_A_PREFILL_MODEL.md`) last

## Harness bugs found while re-running everything in `/data/kmabee/tt-metal`

None is a model or numerics problem; all are cases of Mistral inheriting another resident's
assumptions. Re-running the suite on a fresh checkout is what surfaced them — which is the argument
for doing that periodically rather than trusting a green run from a week ago.

**1. Mistral silently inherits DeepSeek-tuned MLA matmul program configs, and at one sequence length
that config is illegal.** `test_mistral4_mla[...seq25k-max_sl...]` dies in `_q_a_latent`
(`tt/mla/mla.py:978`) with:

```
TT_FATAL: MatmulMultiCoreReuseMultiCastProgramConfig: Kt (32) must be divisible by in0_block_w (14)
```

`_resolve_mm_cfg` (`tt/mla/mla.py:721`) keys the tuned table on `(weight_name, seq_len_local)` **only
— nothing about the variant's dimensions**. `seq25k` at sp=8 gives `seq_len_local = 25600/8 = 3200`,
which hits the `3200` bucket in `tt/mla/mla_config.py:82` (`in0_block_w=14, per_core_M=10,
per_core_N=5` — exactly what the error reports). `in0_block_w=14` divides DeepSeek's `Kt`
(hidden 7168 / tp 4 = 1792 → 56 tiles, 56 % 14 == 0) but not Mistral's (hidden 4096 / tp 4 = 1024 →
**32 tiles, 32 % 14 ≠ 0**).

The crash is the lucky case. The buckets are `640`, `3200`, `4096`, and `640 = 5120/8` is the seq5k
local length — so **the seq5k rows that pass are also running DeepSeek's tuning**, silently, just with
a divisibility-compatible value. Valid but not tuned for these shapes, with no diagnostic. Fix
properly by keying tuned configs on the variant (or the actual K/N), not only on seq length; a
`Kt % in0_block_w` guard that falls back to defaults is the cheap stopgap.

**2. The chunked MLA test's config path had diverged from the transformer's — three stacked bugs,
all fixed; 13 rows now pass.** None was numerics. They surfaced one at a time, each hiding the next,
which is the interesting part: **the last one would not have crashed at all.**

Two different config fixtures exist and only one was maintained:

| fixture | resolver | used by |
|---|---|---|
| `config_only` | `_resolve_config_only` → hand-built `mistral4_hf_config()` | transformer test, the demo |
| `hf_config` | `_resolve_hf_config` → `AutoConfig` + `_unwrap_multimodal_config` | chunked MLA test, via `pretrained_transformer_weights` |

The builder applies four post-construction fixups. The AutoConfig path applied **none** of them.

1. **`quantization_config` lost on unwrap.** `ValueError: Found float8 tensor
   'mlp.experts.down_proj' in a checkpoint whose config has no quantization_config`. It is a property
   of the *checkpoint*, so HF puts it on the outer `Mistral3Config`; unwrapping to the inner
   `Mistral4Config` discarded it. Mistral Small 4 is the first resident that is **both**
   multimodal-wrapped and quantized. `_unwrap_multimodal_config` now carries it across.
2. **`KeyError: 'mlp.gate.e_score_correction_bias'`**, then the stacked expert layout. The fixture
   had its own copy of the weight extraction and had not learned either Mistral shape. Now reuses
   `_extract_routed_experts_flat` and substitutes zeros for the absent router bias — the same
   treatment `transformer_helpers.py` already applied on the other path.
3. **`AttributeError: 'Mistral4Config' object has no attribute 'rope_theta'`** — the builder hoists
   it out of `rope_parameters`; AutoConfig leaves it there. **This is the one that matters:** the very
   next fixup in that list is `mla_disable_yarn_mscale`, and its absence does *not* raise. It silently
   re-applies DeepSeek's mscale² = 2.2058x to the attention logits — the wrong-softmax-temperature bug
   that cost real time to find the first time. So fixing only `rope_theta` would have turned a crash
   into quietly wrong numbers. Both fixups now live in a shared `normalize_mistral4_config()` that
   **both** paths call, and the invariant asserts moved in with them so an AutoConfig-loaded config is
   checked as strictly as a hand-built one. Verified: the two paths now agree on `rope_theta` and the
   mscale flag.

The general lesson for the next model: **two config paths is one too many.** Any normalization added
to the hand-built builder is invisible to the AutoConfig path, and the failure mode degrades from
`AttributeError` to silent numerical error as you fix your way down the list.

One useful side-effect of fix 1: the checkpoint's real `quantization_config` reads
`weight_block_size: None`, i.e. per-tensor fp8 — an independent cross-validation of the hand-built
config's assumption, which until now rested on reading `config.json` by eye.

**Verified no regression** after all three fixes: the 36-layer smoke row still passes with the same
first token (`ID=2`, prob 0.9689), and the decoder block still reports PCC 0.987949 / KVPE KV
0.999898 / KVPE PE 0.997590 — bit-identical to before.

## Mistral-specific facts worth not rediscovering

Each of these was checked against the code or the checkpoint.

- **fp8 is per-tensor.** `weight_block_size` is `null`; dense weights carry a rank-0 scalar
  `*_scale_inv`, stacked expert tensors carry `[128, 1, 1]`. The shared dequantizer asserts
  `tensor.ndim == inv_scale.ndim` and a matching `block_shape` rank, so it **raises** on both — a
  loud failure, not silent corruption. **Handled** by `is_per_tensor_fp8` /
  `_dequantize_per_tensor_fp8_state_dict` in `utils/test_utils.py`; verified against real checkpoint
  tensors.
- **Experts are stacked and fused.** `mlp.experts.gate_up_proj` is `[128, 4096, 4096]`, matching
  neither `experts.{i}.*` nor `experts_stacked.*`. transformers 5.12 ships `mistral4` natively, and
  `modeling_mistral4.py` declares it `[num_experts, 2*intermediate_dim, hidden_dim]` consumed with
  `.chunk(2, dim=-1)` — so gate is the first half of the output dim, up the second. Contiguous, not
  interleaved. **Handled** by `_extract_routed_experts*` in `utils/transformer_helpers.py`, on both
  the random and the pretrained paths.
- **Router is softmax affinity with no correction bias.** The grouped-topk kernel implements only
  `sigmoid` and `sqrtsoftplus`, so that path is not usable as-is. `n_group = 1` is *not* unusual —
  Kimi, GLM and V4-Flash are ungrouped too; only DeepSeek-V3 uses 8 groups. Note
  `TtMoEGatePrefill.check_cache_complete` requires an `e_score_correction_bias` cache entry that
  Mistral has no weight for — zeros are substituted, which is exact (zero is the identity everywhere
  the bias is read), not a placeholder.
- **rope: `rope_parameters` → `rope_scaling`, and `rope_theta` must be hoisted out of it.** The
  config builder does both. `original_max_position_embeddings` stays at the checkpoint's 8192 (the
  pre-extension length) — YaRN's frequency ramp is computed against it, so substituting `max_seq`
  changes the rope. GLM's builder does substitute, but only because GLM's `factor` is 1.0 and the
  value is inert there.
- **⚠ Mistral applies NO YaRN mscale — the softmax scale IS the bare `qk_head_dim**-0.5`.** This is
  the one real bug found so far, and it was silent. `Mistral4Attention.__init__` sets
  `self.scaling = qk_head_dim ** -0.5` unconditionally, and `Mistral4RotaryEmbedding.attention_scaling`
  is `1.0`, so no mscale is applied in the softmax scale *or* baked into cos/sin. DeepSeek folds
  `mscale**2` in whenever `rope_scaling["mscale_all_dim"]` is truthy — which Mistral's is (1.0) — so
  both `tt/mla/mla.py` and the CPU `MLAReference` were multiplying the attention logits by **2.2058**.
  No crash, no shape error, just a wrong softmax temperature. Handled by the
  `mla_disable_yarn_mscale` flag set in the config builder.
  **How it was caught, because the method generalises:** A/B the two CPU references against each
  other on identical weights (`MLAReference` vs `Mistral4Attention`) — 0.948 before, 0.99999 after.
  That is a one-minute CPU test. A green device PCC only means the device agrees with *the reference
  you picked*; when a model's own implementation differs from the family's, check the references
  against each other **first**.
- `config.json` exposes `llama_4_scaling_beta: 0.1`, the same constant `mla.py` hardcodes in the
  mscale formula. They agree today, and nothing reads the field.
- **`kv_lora_rank = 256`** is unprecedented here (family uses 512). It makes the packed-FP8 KV
  cache's rope offset 264 bytes, which is not 16-byte aligned and fails `validate_scaled()`. It does
  **not** affect the MLA tests, which pass the tiled format explicitly. It binds at serving, where 1M
  context likely wants the packed format. Smallest fix looks like 8 bytes of padding (264 → 272);
  worth taking to the MLA owners as a proposal.
- **Expert weights land on device as `BFLOAT4_B`** (4 bits), not 8. Cache entries are named
  `layer_N.routed_expert.local_K_{gate,up,down}_dtype_BFLOAT4_B_...`, and the built cache measures
  ~2.8 GB per layer across the whole 32-chip mesh (~87 MB/device/layer). So the `~3.6 GB/device`
  weight figure in the planning docs — which assumed 1 byte/param — is conservative by about 2x.
- **`first_k_dense_replace = 0`** — every layer is MoE. DeepSeek-V3 and GLM have 3 dense layers,
  Kimi 1. This is the first resident with none.
- **`embed_tokens` and `lm_head` are unquantized bf16** with no scale tensors — the only large
  matmul weights that are. Norms and the router gate are unquantized too; everything else is fp8.

## Debugging numerics here — four things that cost real time

These are generic to onboarding a model into this package, not specific to Mistral. Each is a
CPU-only check that runs in about a minute, and each one, skipped, produced a number that looked
exactly like a device bug and was not one.

**1. A green PCC only means the device agrees with the reference you picked.**
Two CPU references can both look reasonable and disagree with each other. Before spending galaxy
time, forward both on identical weights and compare them to each other. Onboarding a model whose
reference comes from current `transformers`, into a package whose other references are vendored
copies, is exactly when they diverge — the vendored DeepSeek reference folds a YaRN `mscale**2` into
the softmax scale, `Mistral4Attention` does not, and the gap was a 2.2058x multiplier on the
attention logits with no crash to reveal it (references at 0.948; 0.99999 once reconciled).

**2. Build rope tables in fp32, always.**
`position * inv_freq` in bf16 loses its low bits, so phase error grows with position. A bf16 rotary
holds to roughly 2k tokens and then falls off a cliff — measured at `qk_rope_head_dim = 64`: PCC
0.99997 at seq 512, 0.99980 at 2048, **0.958 at 5120**. Monotonic degradation with sequence length is
the signature. Two checks separate it from a real fault: compare the rope *tables* directly (they
should be bit-identical, max abs diff 0.0), and re-run in fp32 (the absorbed and unabsorbed MLA
formulations agree to 1.0000000 there).

**3. A disk-cached reference is only as good as its key's field list.**
`cpu_mla_reference` and the transformer reference cache both hash a chosen set of "math-affecting"
config fields. A field you added that moves the reference, but did not add to the hash, does **not**
fail loudly — it serves a stale ground truth, and the discrepancy appears to be in whatever you
changed. This cost a 0.829 that read as a device regression. If you add a config field that changes
reference output, add it to the hash in the same commit.

**4. Beware checks that cannot fail.**
Comparing rope tables at position 0 only shows all-ones on both sides, always. Compare full tables,
and prefer PCC over `allclose` on a single row. Same class: `load_state_dict(..., strict=False)`
silently leaves a missing weight at its random init — check `missing_keys` / `unexpected_keys` rather
than assuming the load worked.

**And two harness-level traps in this package specifically:**

* Reference decoder layers differ by `transformers` generation. Bind by signature — use
  `decoder_layer_kwargs` in `utils/transformer_helpers.py` rather than hand-writing the call. The
  cache kwarg is `past_key_value` on the vendored layers and `past_key_values` on `transformers >= 5`
  (wrong name lands silently in `**kwargs`, so no KV is captured), and `position_embeddings` is
  required on `>= 5` (omitting it raises `cannot unpack non-iterable NoneType` from inside attention).
* A comparison that reports `-1.0` did not measure anything — that is
  `_compare_intermediate_pcc`'s value for "missing from TT intermediates" or a raised exception.
  Treat it as no coverage, not as a bad score.

## Interactive demo — chatting with the model today

`demo/serve_mistral4_interactive.py` serves the 36-layer model with real weights behind an
OpenAI-shaped, SSE-streaming API, so `~/scripts/client_demo.sh` talks to it unmodified.

```bash
PREFILL_SERVE_SEQ_LEN=512 bash /data/kmabee/mistral4_repro_logs/20_serve.sh   # one shell
bash ~/scripts/client_demo.sh 32                                              # another
```

**It works, and the model answers correctly.** Verified end to end by `21_demo_smoke.sh` (starts the
server, waits for `model_ready`, sends one streaming and one non-streaming request, records timings,
shuts down):

```
"Name three French cities."      -> "Paris, Lyon, Marseille."
"What is the capital of France?" -> "Paris."          (streamed token-by-token, stopped on eos id 2)
```

Ready in ~60 s from the warm TTNN cache. **At window 512: ~1.05 s/token, TTFT 979 ms** (17-token
reply, `"tell me a joke"` → *"Why don't skeletons fight each other? They don't have the guts."*).

The very first run of a *new* serve configuration is much slower — 90.8 s for 6 tokens — because it
compiles programs. That cost does **not** recur: tt-metal caches JIT artifacts on disk, so later
processes hit the cache (`JIT cache stats: 134/135 hits`) and answer in under a second. Same reason a
lone `tt_forward` in a fresh process can read 32 s instead of ~3 s. When you time this, generate
enough tokens to leave warm-up behind and read the per-token log lines rather than the request mean —
averaging a 2-token reply once yielded a "2.80 s/token" figure that was nearly 3x the truth.

That the answers are correct is the useful part: it is qualitative evidence that the per-layer PCC
drift down to ~0.94 is benign at the output, since two factual questions come back right after
36 real layers and several sampled tokens.

**It does not decode.** There is no decode path in this folder — not one `*decode*` file. The server
generates by re-running the whole prefill for every token: `prefill(prompt[0:n])` gives token `n`,
then `prefill(prompt[0:n+1])` gives token `n+1`. Three consequences worth understanding:

- **Per-token cost is one full prefill of the padded *window*, not of the real tokens.** Measured at
  window 512 over a 17-token reply: **~1.05 s/token, TTFT 979 ms.**
- **Do not measure this with wall clock. Process-to-process variance is ~25% at a fixed
  configuration** — window 512 / `actual_isl` 64 / pad tail gave min 1401 ms in one process and
  1760 ms in another. That band is wider than every effect worth chasing, and four separate
  hypotheses died inside it after each looking real: cost scaling with the window, cost scaling with
  `actual_isl`, a ~288 ms first-touch `build_padding_config` penalty (it reproduced with the
  *opposite* sign), and a large win from suppressing the ~1539 DEBUG log lines emitted per forward.
  **None of those is established, in either direction.** Use
  `demo/profile_prefill.py::test_ops` under tracy, which reads `DEVICE KERNEL DURATION` from
  hardware counters and is immune to host noise; `tests/analyze_ops_perf.py` summarizes the CSV.
  Wall clock is only useful as `test_walltime` min compared against that device total, to get the
  host share.
- **The minimum window is 512, not 256.** Window 256 fails at the MoE, not at attention:
  `TT_FATAL: Token count (32) must be divisible by the 64-core grid used by masked_bincount`. At
  sp=8 a 256 window gives 32 tokens/chip, and `masked_bincount` needs a multiple of **64**. So the
  binding constraint is 64 tokens/chip → `64 * sp_factor` = 512, stricter than the 32-row
  tile-alignment bound. The server refuses non-multiples of `32 * sp` up front, but 256 gets through
  that check and dies on the device — worth tightening if anyone leans on it.
- **A 1-token "decode-as-chunked-prefill" shortcut does not exist,** for the same class of reason:
  the sequence is SP-sharded and tile-aligned per shard, so the smallest expressible chunk is
  hundreds of tokens, not 1. Real single-token decode needs the decode kernels in
  `models/demos/deepseek_v3/` (`moe_decoder_block_2d` and friends) — a separate bring-up.
- **Mistral's chat template injects a ~545-token default system prompt** when the caller sends none
  (tool-use instructions, knowledge-cutoff blurb, Le Chat branding). That alone would force a
  1024-token window for a one-line question, so the server substitutes a short system prompt (31
  tokens for the same question). `PREFILL_SERVE_SYSTEM_PROMPT=""` restores Mistral's real default.

Correctness of the re-prefill trick: only the first `actual_isl` window entries are real and the LM
head reads row `actual_isl - 1`. Attention is causal, so that row can never see the pad tail, and
stale KV beyond `actual_isl` from the previous step is unobservable. Right padding is therefore
load-bearing — with left padding the head reads `seq_len - 1`, the wrong row for a partial window.

The demo doubles as a **qualitative numerics probe**, which is why it is worth having while the
full-model PCC is still failing: coherent multi-turn text is evidence the per-layer drift below is
benign at the output, and word salad is evidence for the `BFLOAT4_B` expert theory. It is weaker
evidence than a PCC number but it exercises 36 real layers over many sampled tokens, which no
current test does.

Two details that were needed to make a real client work, both worth keeping if this is copied:
`/tt-liveness` must return `model_ready: true` (the client polls for exactly that key and otherwise
waits forever), and the `asyncio.Lock` must be held across the whole generation loop, not just the
construction of the streaming response — `batch_size=1` shares one KV slot, so overlapping requests
would corrupt each other. `fastapi`/`uvicorn` are not in `create_venv.sh`; install them into the
uv-managed venv with `VIRTUAL_ENV=$PWD/python_env ./python_env/bin/uv pip install fastapi uvicorn`
(there is no `pip` in that venv).

## Gotchas

- **A mesh mismatch is a SKIP, not a failure.** Several block tests are parametrised for meshes
  smaller than 32 and skip on a galaxy — `N skipped` with exit code 0 reads like success. Check the
  passed/skipped counts, not the exit status. `test_mistral4_mla`'s `(8,1)` id does this too: it
  needs an 8-chip carve via `TT_VISIBLE_DEVICES` and skips otherwise.
- **`--collect-only -q -k ...` before running.** `test_mla.py` collects thousands of cases; confirm
  you selected the number you meant. Note `-k` matches the **module path** too, so
  `-k deepseek_v3` selects *every* test under `models/demos/deepseek_v3_d_p/` — including Kimi's and
  Mistral's. Filter on the variant id (`dsv3`, `kimi`, `mistral4`) instead.
- **Point cache env vars somewhere writable in `$HOME`.** The shared `/mnt/models/...` tree is
  read-only; a cache *write* fails with `errno=13` from `serialization.cpp:74`. Reads from a complete
  cache are fine.
- **The cache path is keyed on `ttnn.get_num_devices()`**, not the mesh shape
  (`{name}_{arch}_{N}dev/{sp}x{tp}`). Carving to 8 visible devices changes `N` from 32 to 8, so
  nothing cached at 32 is found.
- **After any hang, `tt-smi -r` before the next run.** A wedged ethernet core makes the *next*
  person's run fail with a misleading timeout instead of their real error.

## Open

- `(8,1)` vs `(4,2)` for a PP=4 stage on one galaxy — compute vs CCL. Block correctness is
  split-independent, so this blocks nothing here.
- KV cache dtype/layout for the prefill↔decode ABI, given `kv_lora_rank = 256`.
- Vision scope — is text-only acceptable for v1?
- No Mistral golden trace exists. `test_prefill_transformer_chunked` cannot run without one, and
  every other resident's lives under `/mnt/models/deepseek-prefill-cache/golden/`. Capturing one is
  wall-clock work that nothing else shortens.

## Who to ask

Per-area owners for MLA, MoE and the prefill runner are listed in the internal bring-up channel;
ask there rather than in this file, so the routing stays current as people move around.
