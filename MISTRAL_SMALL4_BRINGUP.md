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

### ⚠ Full-model numerics: measured, still off — but three of the four scary numbers were the ruler

The rows above are **smoke** runs — they prove wiring, not numerics. The full-model PCC check
(`pcc-json_prompts-pretrained`, no golden trace needed — the reference is computed layer-by-layer)
**fails**, and a 2026-08-18 pass fixed the two instruments it was being read through. What is left
is smaller and better localised than it looked.

**Read the nPCC column, not the PCC column.** Raw PCC over a whole hidden-state tensor is, for this
model, close to a measurement of a few hundred outlier channels: massive activations develop by
layer 5 (absmax 429 vs rms 0.35, with the **top 0.01% of elements holding 95% of the total squared
energy**) and max/rms is still 120–165 through layers 30–35. `_compare_intermediate_pcc` now also
reports **nPCC**, the same PCC after per-token RMS normalisation — the transform RMSNorm itself
applies, and the reason `norm` was always a better-behaved number than the layer feeding it.

```
layer   PCC     nPCC        layer   PCC     nPCC        layer   PCC     nPCC
0       0.9993  0.9993      16      0.9657  0.9276      31      0.9058  0.9287
5       0.9989  0.9952      19      0.9516  0.9210      32      0.1735  0.9361
10      0.9953  0.9787      23      0.9424  0.9280      33      0.1946  0.9394
13      0.9831  0.9452      27      0.9415  0.9292      35      0.4581  0.9850
15      0.9747  0.9353      28      0.9047  0.9241      norm    0.9590  0.9581
```

Two conclusions change:

* **The `layer_32..34 ≈ 0.17` "tail collapse" does not exist.** Under nPCC those layers read
  0.936 / 0.939 / 0.942, in line with their neighbours, and `norm` at 0.958 sits right where it
  should. It was outlier domination, exactly as the contradiction with `norm` 0.959 and the exact
  first-token match implied. The distortion runs both ways — raw PCC also *flattered* the early
  layers (layer_10 reads 0.995 raw but 0.979 normalised).
* **The error does not compound without bound; it plateaus.** nPCC falls to ~0.92 by layer 16–19
  and then sits at 0.92–0.94 for the remaining seventeen layers. "Monotonic decay that compounds"
  was an artifact of the same ruler. A plateau reads as a bounded per-layer noise floor —
  quantisation-shaped — not an error that accumulates.

**The 72 `*_kvpe_*` rows now report real numbers, and they localise the error.** They used to read
`-1.0`; that was never "missing from TT intermediates" (the handoff's guess) but a shape mismatch,
because `Mistral4Attention` caches EXPANDED per-head keys `[1, 32, seq, 128]` while the device
caches the compressed MLA latent `[1, 1, seq, 320]`. The slice `ref[..., :kv_lora_rank]` on a
128-wide last dim just clamped, and comp_pcc died on 4194304 vs 262144. `test_prefill_block.py` had
already solved this for one layer; that helper is now shared as
`transformer_helpers.derive_mla_kvpe` and used per layer by the layer-by-layer reference.

```
layer      0      5     10     13     16     20     27     31     35
kvpe_kv  1.000  0.990  0.977  0.884  0.835  0.894  0.863  0.927  0.911
kvpe_pe  1.000  1.000  0.998  0.988  0.979  0.987  0.990  0.987  0.992
```

* **The RoPE half is flat at ~0.99 at every depth.** Rope-phase drift is ruled out as the
  compounding error — worth knowing, because the float32-rope trap in `decoder_layer_kwargs` makes
  that the natural first suspect.
* **The compressed KV latent is the worst-matching quantity measured anywhere in this model**, and
  it is consistently *worse than the hidden state that feeds it* (layer 16: hidden 0.966, latent
  0.835). So `kv_a_proj` + `kv_a_layernorm` — the 4096 -> 256 compression — **amplifies** relative
  error rather than passing it through. `KV_LORA_RANK = 256` is the dimension this config already
  flags as unprecedented in this family (DeepSeek-V3 / Kimi / GLM all use 512). Like the hidden
  states it decays early then plateaus (~0.83–0.93 from layer 13 on).
* Attention output survives it: the decoder-block PCC is 0.988 and the KVPE error does not visibly
  accelerate the hidden-state curve. So this is a lead, not yet a demonstrated cause.

**The one unambiguously good number:** the full 36-layer model with real weights picks the **same
next token as the HF reference** — `ID=2 ['</s>']` at position 1023, `TT==HF match: True`.

**Expert dtype `BFLOAT4_B` -> `BFLOAT8_B`: measured 2026-08-18. Real, consistent, and not the
whole story.** Every single row improves, none regresses, and the tail `norm` / `lm_head` cross the
0.97 bar for the first time — but the plateau only moves from ~0.928 to ~0.940, nowhere near 0.99,
so the row is still xfail.

```
                     bf4      bf8     delta
nPCC plateau (L20-35)  ~0.928   ~0.940   +0.0121 (mean)
norm                   0.9590   0.9701   +0.0110   <- now PASS
lm_head                0.9621   0.9764   +0.0143   <- now PASS
kvpe_kv (mean, 36 L)        —        —   +0.0151
first token            ID=2 match       ID=2 match
TTNN cache               65 GB   119 GB   1.83x, ~16 min cold build
```

So: **4-bit experts are a contributor, not the cause.** Budget ~+0.012 nPCC for 1.83x the weight
cache and decide whether that trade is worth it on its own merits; do not expect it to fix the
model. Reproduce with `TT_PREFILL_ROUTED_EXPERT_WEIGHTS_DTYPE=bfloat8_b` (one knob, rebinds every
default — see `utils/expert_dtypes.py`) plus a separate `TT_MISTRAL4_PREFILL_TTNN_CACHE` (Gotchas).

One useful cross-check fell out of it: `layer_0_kvpe_kv` is the **only** KVPE row that did not move
(+0.0000). Layer 0's KV latent is projected from the embedding, which no expert touches, while every
later layer's inherits the accumulated improvement. The KV amplification itself survives the change
(layer 16 latent 0.866 out of a ~0.94 hidden), so it is an error source **independent of expert
precision**.

Two things to know about the adapter as written:

- `supports_pretrained = True`. The real checkpoint loads: per-tensor fp8 dequant, the stacked+fused
  expert split and the zero router bias are all handled, on both the random and the layer-by-layer
  pretrained paths.
- `default_gate_mode = "GPT_DEVICE"`. This started as an argument and is now **confirmed against the
  reference implementation**: `Mistral4MoE.route_tokens_to_experts` is
  `softmax(-1)` over all experts -> top-k -> gather -> renormalize -> x1.0, and with `n_group = 1`
  the group mask is all-ones so the grouping collapses out entirely. That is the same rule as the
  GPT-OSS gate. Still worth an independent per-expert token-count assertion.

### Chunked prefill: running, and validated for the first time (2026-08-18)

`test_mistral4_prefill_transformer_chunked` (3 x CHUNK = 15,360 tokens, GPT_DEVICE gate,
`l1_small_size=768`) against the host-generated golden trace:

```
L1 + L10   2 passed    L10: min per-layer PCC 0.990904, KV cache min 0.978404
L36        1 failed    min per-layer PCC 0.280504 < 0.88  -- chunk 0, layers 32-34 only
```

The L36 failure is **the same layer-32..34 metric artifact**, reproduced independently at a
different ISL against a different reference — and it is **chunk-0-only** (`chunk 1 layer 33 =
0.974`, `layer 35 = 0.993` for the same layers). Chunk 0 is the slice holding **token 0**, whose
hidden state at layer 10 has L2 norm **757.9** against a per-position median of **10.66** — the
attention sink. One row sets the PCC of any slice containing it. See the comment at
`LAYER_PCC_THRESHOLD` before treating a chunk-0 tail-layer number as a device bug.

Everything else at L36 lands 0.96–0.99, so the chunked path itself is sound.

## Next steps, in the order worth doing them

The detail per area is in "What still needs doing" below; this is the sequencing. The first three
are all about making the numbers trustworthy, because every later decision reads off them.

**1. Raise the expert weight dtype and re-run the full-model PCC.** One knob, decisive either way.
Set `TT_PREFILL_ROUTED_EXPERT_WEIGHTS_DTYPE=bfloat8_b` (it rebinds every default at once via
`utils/expert_dtypes.py`) **and point `TT_MISTRAL4_PREFILL_TTNN_CACHE` at a fresh directory** — the
cache path encodes no dtype and the expert weights roughly double, ~65 GB -> ~120 GB, on a ~30 min
cold build. Read the **nPCC** column: the question is whether the 0.92–0.94 plateau moves. If it
does, the 4-bit expert theory is confirmed and the price is known. If it does not, experts are
*exonerated* and the KV-latent lead below inherits the investigation.

~~**2. Isolate the `layer_32..34 ≈ 0.17` reading before believing it.**~~ **Done, 2026-08-18: it was
the ruler.** Those layers read 0.936–0.942 under nPCC and the "tail collapse" is gone; see the
numerics section. The reusable lesson is that raw whole-tensor PCC is not a safe per-layer metric on
a model with massive activations, in *either* direction — it also flattered the early layers.

~~**3. Close the `*_kvpe_* = -1.0` gap.**~~ **Done, 2026-08-18, and it was not the diagnosed cause.**
The rows were never "missing from TT intermediates": `return_kv_cache` was already being passed and
`ref_kvpe_list` already populated. `Mistral4Attention` caches expanded per-head keys
`[1, 32, seq, 128]`, the device caches the compressed latent `[1, 1, seq, 320]`, and
`ref[..., :kv_lora_rank]` silently clamped a 128-wide last dim instead of failing. All 72 rows now
report; the RoPE half is flat at 0.99 and the KV latent is the worst number in the model. That
latent is now the best lead after expert dtype — see the numerics section.

**3b. Chase the KV latent (new).** `kv_a_proj` + `kv_a_layernorm` amplify relative error (layer 16:
hidden 0.966 in, latent 0.835 out) at `KV_LORA_RANK = 256`, half of what the rest of the family
uses. Worth checking the latent's own activation distribution and the dtype it is projected and
normalised in, the same way the reference had to be computed in fp32 to avoid a spurious 2e-3.

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

## Prefill throughput — what this model actually does

**Do not confuse the demo's generation rate with prefill throughput.** The demo generates ~8 tok/s
because it re-prefills a whole window per token; prefill throughput is `window / time per forward`,
and it is a different number by three orders of magnitude. Measured traced, with a **full** window of
real tokens (`actual_isl == window`), every row verified to sample the same token as eager:

| window | eager | traced | **prefill tok/s** | speedup |
|---|---|---|---|---|
| 512 | 1967 ms | 83.9 ms | 6,105 | 23.5x |
| 1024 | 2057 ms | 90.8 ms | **11,276** | 22.7x |
| 2048 | 1681 ms | 117.7 ms | **17,403** | 14.3x |
| 4096 | 1696 ms | 167.3 ms | **24,487** | 10.1x |
| 5120 (production 5k) | 1990 ms | 195.4 ms | **26,202** | 10.2x |
| 25600 (production 25k) | 1900 ms | 766.6 ms | **33,395** ← peak | 2.5x |
| 51200 | — | 4190.2 ms | 12,218 | — |
| 102400 | — | **fails** | — | — |

Reproduce with `40_prefill_throughput.sh` and `41_long_isl.sh`. Untraced, the same rows are
**260–2,415 tok/s**, so the trace work is worth ~10x on this metric, not a few percent.

### bf8 experts cost 2.5% of prefill throughput (measured 2026-08-18 12:31)

The accuracy win from `bfloat8_b` experts is not free. Same-session A/B at the 25k peak window, full
window of real tokens, traced min of 3 reps:

| window | bf4 tok/s (control) | bf8 tok/s | delta | bf4 token | bf8 token |
|---|---|---|---|---|---|
| 5,120 | 25,930 | 25,772 | −0.6% | 1010 | 2 |
| 25,600 | **33,550** | 32,713 | **−2.5%** | 2 | 1048 |

Both controls reproduce the overnight table within 1% (26,202 and 33,395), so **~33.5k tok/s is the
real peak, and bf8 does not beat it.** Expect that: bf8 nearly doubles expert weight bytes
(0.5625 → 1.0625 B/param) and `UnifiedRoutedExpertFfn` is 11.2% of device time.

**The penalty grows with window** — 0.6% at 5k, 2.5% at 25k. It is not a fixed tax: the larger the
window, the more the expert FFN is bandwidth-bound, so the doubled weight bytes hurt more exactly
where we want to run. Do not price bf8 off the 5k number.

Two things worth noting from these runs:

- **The two dtypes sample DIFFERENT tokens at 25k** (2 vs 1048) on identical input. Within each arm
  traced matches eager, so this is the expert dtype changing the model's output, not a trace bug. We
  cannot say which is right — there is no HF reference at 25k (the PCC reference is 1k, the golden
  trace 15,360). So bf4-vs-bf8 is not a cosmetic choice at long context, and "which token is correct"
  is an open question that needs a 25k reference.
- **`eager` is no longer ~1900 ms.** The older table shows ~1900 ms eager at *every* window, which is
  a host-bound signature, not device time; here eager (769 ms) ≈ traced (763 ms) because at a 25k
  window the device dominates and tracing buys ~1%. The trace still buys ~10-20x at small windows.
  Compare traced-to-traced across sessions, never eager.

**Single-shot has a ceiling, and throughput peaks well before it.** 51,200 still runs and still
matches eager, but 102,400 dies on **L1, not DRAM**:

```
Statically allocated circular buffers on core range [0-1 - 10-7] grow to
1721216 B which is beyond max L1 size of 1572864 B
```

1.72 MB of statically-allocated circular buffers against a 1.57 MB budget — on-chip buffer sizing,
which is kernel-config work, not a memory-capacity problem.

**More decisive than the ceiling is the shape of the curve.** Up to 25.6k the cost is *sub*linear
(5,120 -> 25,600 is 5x the tokens for 3.9x the time, because the fixed ~67 ms keeps amortizing).
From 25,600 -> 51,200 it inverts hard: 2x the tokens for **5.5x** the time, an exponent of ~2.45, as
the quadratic attention term takes over. So **peak single-shot prefill throughput is ~33k tok/s at a
~25k window**, and past that it falls off a cliff.

That is the real argument for chunked prefill at long context: even with the L1 limit fixed, a
single-shot 256k prefill would be catastrophically slow rather than merely large. **Chunking is what
keeps each chunk inside the efficient regime**, and this measurement gives whoever wires the runner a
starting point — chunk size somewhere in **8k–25k** is where this model wants to sit.

**Read the speedup column — it is the cost model in one place.** Host dispatch is ~1.1–1.9 s at
*every* window, because the op COUNT is fixed (2316 per forward) and only the work per op grows;
device time meanwhile goes 84 ms -> 767 ms. So trace buys 23x at 512, where host dominates, and only
2.5x at 25k, where the device finally does. Anyone tuning this should know which regime they are in
before optimising.

Fitting those four points gives **≈ 67 ms fixed + ~24.5 µs per window token**. That single line
explains most of what is confusing about this model's performance:

- Throughput *rises* with window because the 67 ms of fixed per-layer cost (CCL, MoE plumbing, small
  ops) amortizes over more tokens. It is the same fixed cost that makes the generation demo look slow.
- Per-*token* generation cost barely moves with window (512 -> 1024 is +19%) for the same reason —
  you are mostly paying the fixed part either way.
- Extrapolating is unsafe past ~4k: prefill attention is quadratic while this fit is linear, so the
  fit overstates long-context throughput. Measure, do not extrapolate.

**Comparison to the Gemma4 number — superseded 2026-08-18.** The old comparison here put our
single-shot 1024-window figure against Gemma4's 256k aggregate and concluded "we pass it", which was
apples to oranges and flattered us. We now have the matched measurement: see
"Gemma-4-31B-it side by side" in the long-context section. At 256k the two land within 5%.

**A load-bearing caveat about full occupancy:** window 512 costs 83.9 ms with all 512 tokens real
versus 80.1 ms with only 96 real — **+4.7% for 8x the routed tokens**. Benchmarks that leave the
window mostly padded therefore measure something close to the real thing here, but that is a property
of this fixed-cost-dominated regime, not a general licence. `profile_prefill.py` fills the window on
purpose.

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

**~8 tok/s (0.13 s/token), after the trace fix below.** Ready in ~96 s from the warm TTNN cache
(~60 s of that is the model, the rest is capturing the trace once). Before tracing it was ~1.05
s/token; the answers are identical either way.

### Why it was slow, and why it is not any more

Profiling said the forward was **~95% host-bound**, which was not what anyone assumed:

```
device kernel time     80.5 ms   (per device; mesh range 52-112, mean 73)
wall clock           1760    ms
=> host share        ~95%, spread over 2316 op dispatches at ~0.55 ms each
```

The 32 chips were idle ~95% of the wall clock while Python dispatched ops one at a time. Worse,
**1051 of those 2316 calls (45%) account for 5.9 ms of device time but ~578 ms of host time** —
`Slice` alone is 216 calls costing 0.2 ms on device and ~119 ms to launch, a **565x** ratio. A
prefill that "does too much arithmetic" was never the problem.

One bug blocked the fix. `get_rope_tensors()` recomputed cos/sin on host and uploaded three tensors
**every forward**, though its only argument (`self.seq_len`) is fixed for the life of the object. That
is wasted host work, and it is fatal to tracing — a host→device write inside a capture raises
`TT_FATAL: Writes are not supported during trace capture`. The chunked path already prebuilds its
tables once (`self.indexed_rope`); the single-shot path now does too. **Eagerly this bug is invisible,
just slow; it only announces itself when you try to trace.**

With that fixed, capturing the block stack gives **1715 ms → 80.1 ms per token, 21.4x, same sampled
token**. That 80.1 ms against 80.5 ms of measured device kernel time is the tell: dispatch overhead is
gone and what is left is silicon. In the server it lands at ~0.13 s/token — the extra ~50 ms is the
SSE/HTTP layer and the eagerly-run tail, which only became visible once the big cost went away.

Three things make the capture work, all worth knowing before touching it:

- `SubDeviceTraceController` splits the capture at the MoE's sub-device swaps (**73 segments**). A
  single naive capture cannot survive them.
- `actual_isl` is held **constant** so the op sequence and the MoE's memoized padding config stay
  invariant. Safe by causality: the LM head reads row `n-1` and nothing attends past it, so marking
  not-yet-generated positions "real" cannot change those logits. The server captures once at startup
  with `actual_isl = isl_total`, which makes the capture independent of prompt length.
- The tail (norm/LM-head/sample) ends in a blocking D2H and is excluded via `stop_after_blocks`, run
  eagerly instead. That is ~36 of 2316 ops, and it is also what lets each step pick the correct row
  `n-1` while the captured stack stays fixed.

**Release traces and MoE sub-device managers in a `finally`.** Leaving either registered segfaults
`close_mesh_device` in teardown, and one failed capture stranded the 32-chip galaxy for ~3 hours
before anyone noticed. Run these harnesses under `timeout`.

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
- **Window scaling, settled (measure it traced, not eagerly):** window 512 → **80.1 ms/token**,
  window 1024 → **95.2 ms/token**. Doubling the window costs **+19%**, not 2x, and both matched the
  eager token. At these sizes the cost is dominated by fixed per-layer work — CCL is 30% of device
  time, plus a long tail of small ops — not by sequence-length work.

  **This is the number that decides how much decode is worth.** Decode's naive appeal is that it does
  ~1/512th of the attention arithmetic, but we are not arithmetic-bound here: the fixed per-layer
  cost that decode still has to pay is most of the bill. Extrapolating the measured points (~65 ms
  fixed + ~0.03 ms per window token) puts a 4k window near ~190 ms/token and 8k near ~310 ms — and
  that *understates* it, since prefill attention is quadratic while this fit is linear. So decode's
  advantage is small for a short-context demo and grows with context: worth little at 512, roughly
  5x at 8k, and decisive beyond that.

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


## Long-context chunked prefill — 2026-08-18

**The 33k tok/s figure does not carry over to long-context chunked prefill.** Two reasons, and the
second one is a bug-shaped finding:

1. **33k is a SINGLE-SHOT number at a 25,600 window.** 256k cannot be done single-shot (L1 dies at
   ~100k), so it must be chunked, and chunked prefill at `CHUNK = 5120` is a different regime: even
   its *first* chunk is a 5,120-token forward, which single-shot measures at 26,202 tok/s — already
   below the 33k peak. The peak exists because a bigger single forward amortises better, and chunking
   at 5k gives that up. Chunked throughput can never reach a single-shot peak measured at 5x the
   chunk size.
2. **Chunked prefill is host-bound unless traced, and the untraced flatness is a trap.** Untraced,
   per-chunk time is **0.67 s flat** — identical at chunk 0 and chunk 19, and identical whether the
   allocated cache is 25,600 or 102,400. Both invariances are the tell: it is measuring host dispatch,
   not the device. `use_trace=True` (and `trace_region_size` in `device_params`, or the trace silently
   comes out of general DRAM) is mandatory for any chunked timing claim.

`test_mistral4_prefill_transformer_chunked_no_pcc` is the row for this: no golden trace needed at
`preload_isl=0` (synthetic token ids), so it reaches lengths Mistral's 15,360-token golden cannot
cover. `PREFILL_NOPCC_SEQ_CACHE` overrides the 100k cache cap (use a whole multiple of `CHUNK`;
261,120 = 51 chunks for "256k").

### Measured: full 256k chunked prefill takes ~25 s at ~10.5k tok/s

Traced, 36 layers, `CHUNK = 5120`, `preload_isl = 0`, medians over the second of two iterations
(`T11_traced_256k.log`; end-to-end iterations 24.74 s and 24.91 s, so this is reproducible):

| context | how | wall time | **aggregate tok/s** |
|---|---|---|---|
| 25,600 | single-shot (traced) | 0.76 s | **33,550** ← the old headline |
| 102,400 | chunked, 20 x 5120 | **5.57 s** | **18,381** |
| 261,120 ("256k") | chunked, 51 x 5120 | **24.88 s** | **10,494** |

Per-chunk cost rises linearly with position — 0.178 s at chunk 0 to 0.837 s at chunk 50 — so the
*instantaneous* rate decays from 28,700 tok/s to 6,117 tok/s across one prefill. Fit
`t(k) = 0.149 + 0.01354*k` s: **every additional 5,120 tokens already in the KV cache costs +13.5 ms
on every subsequent chunk.** That is causal attention, and it is why a single "tok/s" number cannot
describe long-context prefill.

**So: no, 33k does not hold at 256k — it is ~10.5k, about 3.2x lower, and the full 256k prefill
takes ~25 s.** The 100k fit predicted 23.7 s for 51 chunks against 24.88 s measured, i.e.
extrapolation from 100k understates by ~5%; measure, do not extrapolate.

What did NOT go wrong at 256k, worth recording: KV cache is 0.4 GiB/chip (never close to 31.9), and
L1 is fine because every chunk is a 5,120-token forward — chunking is exactly what buys past the
~100k single-shot L1 ceiling.

**Caveat: this is a TIMING run only.** Synthetic token ids, no PCC. Correctness at 256k is
unvalidated — the golden trace is 15,360 tokens, so the deepest *validated* chunked run remains
3 chunks. Do not quote 256k as working, only as running at this speed.

## Pursuing pipeline parallelism (PP=4) — a running log

*Started 2026-08-18. Goal: `PP=4 x (8,1)` — four pipeline stages of 9 layers, each stage SP/CP=8 x
**TP=1** on 8 chips. Motivation and attribution in "Where PP=4 x (8,1) actually stands" below:
TP=1 deletes 100% of this model's collectives, which measure 26.8% of device time.*

### ⚠ Read this before quoting anything below: what has and has NOT been measured

**No PP throughput number exists.** Nothing in this section changed the 25k / 100k / 256k figures.
Those three — **33,550 / 18,381 / 10,494 tok/s** — are all **single-rank `SP=8 x TP=4`, no pipeline
parallelism**, exactly as before. The PP work so far is *correctness and de-risking only*.

**The target geometry is `PP=4 x (8,1)`: 4 ranks x 8 chips, SP/CP=8, TP=1.** Not `(4,2)`
(SP=4, TP=2). `(4,2)` is the alternative Marko left open — and thought "might be better" — but the
CCL audit argues for `(8,1)` from where we are: we already run SP=8, so `(8,1)` keeps the
ring-attention length unchanged and drops TP 4->1 (all TP collectives vanish), whereas `(4,2)` would
cut SP 8->4 *and* keep 2-way TP collectives. See the reasoning in "Where PP=4 x (8,1) actually stands".

**And critically: no run so far has used the target geometry end to end.** What each milestone
actually ran:

| milestone | what ran | mesh per stage | chips | stages concurrent? | what it proves |
|---|---|---|---|---|---|
| M1 | one MLA block | `(32,1)` TP=1 | 32 | n/a | MLA needs no TP axis |
| M2c | 1 stage, 2 layers | `(8,1)` SP=8 TP=1 | 8 | n/a | the whole stack runs at the stage shape |
| M2d | 1 stage, 9 layers | `(8,1)` SP=8 TP=1 | 8 | n/a | a full-depth stage runs |
| M3 | 4 stages x 9 layers | **`(8,4)` SP=8 TP=4** | 32 | **NO — sequential** | slicing is numerically exact |
| — | 4 stages x 9 layers | `(8,1)` SP=8 TP=1 | 4 x 8 | yes | **not run — this is M5** |

So M2c/M2d ran the right *geometry* but only one stage; M3 ran the right *slicing* but at TP=4 on all
32 chips, one stage at a time. **The two halves have not been combined**, and until they are there is
no throughput to report.

**Why a PP throughput number needs M5 (or the proxy below).** PP's gain is *concurrency*: in steady
state the pipeline retires one chunk per **slowest-stage** time while 4 chunks are in flight. So

```
PP=4 steady-state throughput  ~=  1 / T(9 layers, 8 chips, TP=1)
single-rank throughput         =  1 / T(36 layers, 32 chips, TP=4)   <- what we measure today
```

M3 cannot produce that: running 4 stages *sequentially* on one mesh is strictly slower than the
single-rank model (same total work, four model builds, no overlap). It was never meant to be a perf
test.

### M6 — one traced stage: the first real PP performance number (2026-08-18)

Short of M5 there is one legitimate estimate: time **one traced stage** at the target geometry and
compare it to the traced whole model, because in steady state the pipeline retires one chunk per
slowest-stage time. Done — `demo/profile_prefill.py` now takes a `mesh-8x1` param and a
`PROFILE_NUM_LAYERS` knob, and a 9-layer pretrained cache at `mistral_small4_bh_8dev/8x1` builds in
~4 min (9/36 of the full 65 GB; it came to 18 GB).

**The per-chip compute is identical between the two configurations**, which is what makes the
comparison fair: single-rank is 36 layers x hidden/4 = 36,864 "layer-widths" per chip; one stage is
9 layers x hidden/1 = 36,864. Same work, same 32 chips in total. The only differences are TP=4's
collectives and its 4x narrower matmuls.

| window | full model (36L, 32 chips, TP=4) | one stage (9L, 8 chips, TP=1) | ratio | today tok/s | **PP=4 tok/s** |
|---|---|---|---|---|---|
| 5,120 | 195.4 ms | **121.1 ms** | **1.61x** | 26,203 | **42,279** |
| 25,600 | 763.0 ms | **423.8 ms** | **1.80x** | 33,552 | **60,406** |

So PP=4 x (8,1) is worth **1.6-1.8x** on single-shot prefill — *more* than the 1.37x the 26.8% CCL
share alone predicts, because TP=1 also makes every matmul 4x wider and therefore more efficient. Note
the sampled token from a 9-layer run is meaningless (2810, not 2): it is a third of a model with an LM
head bolted on. Only the *timing* is meaningful here.

**Long context is a different story, and the honest answer is a bracket.** PP removes TP collectives;
it does **not** touch the SP ring-attention term, which is what grows with KV depth and what dominates
at 256k. Applying the measured 1.61x (CHUNK=5120) to the chunked fit `t(k) = 0.149 + 0.01354k`:

| context | today | **optimistic** (whole per-chunk scales) | **conservative** (only the constant scales) |
|---|---|---|---|
| 102,400 | 5.55 s / 18,442 tok/s | 3.44 s / **29,757** | 4.42 s / **23,170** |
| 261,120 | 24.86 s / 10,503 tok/s | 15.41 s / **16,946** | 21.97 s / **11,884** |

**The deeper the context, the less PP buys** — at 256k the conservative case is only +13%, the
optimistic +61%. PP's win is largest at short-to-medium context, where the collectives are a large
fraction of a chunk; by 256k attention has swamped them. That is a materially different conclusion
from the earlier flat "~1.37x, so 13-14k at 256k" estimate, and it argues for PP as a *serving*
optimisation (many medium requests) more than a 256k-single-request one.

```bash
# one traced stage at the target geometry (25,600 window; use 5120 for the chunk-sized number)
pp/pp07_stage_tput.sh 25600 9
#   TT_VISIBLE_DEVICES=0..7  PROFILE_NUM_LAYERS=9  PROFILE_SEQ_LEN=25600 PROFILE_ISL=25600
#   pytest demo/profile_prefill.py::test_trace -s -q -k mesh-8x1
# prints "traced min 423.8 ms"; PP=4 steady state = 25600 / 0.4238
```

### M7 — the concurrent pipeline RAN, and the first e2e number is bad. Here is exactly why.

**Built and measured 2026-08-18.** `tests/test_prefill_pipeline_concurrent.py` carves four `(8,1)`
submeshes from the `(8,4)` parent, builds one 9-layer stage per submesh
(`first_layer_idx` 0/9/18/27), and pipelines independent requests with a one-iteration lag, enqueuing
all four stages before the single readback sync. It needed a 36-layer weight cache at
`mistral_small4_bh_32dev/8x1`, built **through a submesh** so the sharding matches what the stages
load (`pp/build_pp_cache.py`, 65 GB, ~25 min) — deliberately not a symlink of the 8-device cache, since
guessing at flatbuffer equivalence is how placeholders get loaded as weights.

It works, and the result is a clean negative:

```
window 5120, 12 iterations (first 4 discarded as pipeline fill)
  steady-state iteration: min 2047.3 ms, median 2158.8 ms
  ->  2,501 tok/s (min) / 2,372 tok/s (median)
  vs single-rank at the same window: 26,203 tok/s     ... 10x WORSE
```

**Diagnosis, and it is unambiguous.** One *eager* 36-layer forward at this window costs ~1990 ms (the
throughput table above). The 4-stage pipeline costs 2047-2190 ms for the same 36 layers.
**The pipeline costs the same as the monolithic model, i.e. zero overlap was obtained.**

The reason is the same trap that made untraced chunked prefill look flat: **eager dispatch is
host-bound, so issuing the ops IS the work.** ttnn enqueues are asynchronous *per device*, but one
Python thread still has to issue all 36 layers' worth of ops before any readback — splitting them
across four submeshes does not reduce that, it just relabels which device each op lands on. Four
stages that each take ~500 ms of *host issue* time serialize into ~2000 ms no matter how idle the
chips are.

**So concurrency requires tracing, not just submeshes.** A traced stage replays with a single host
call, which is what makes four of them genuinely overlap:

```
traced 9-layer stage (M6)                      121.1 ms
four traced stages, if fully overlapped   ->   ~42,300 tok/s   (matches the M6 projection)
this eager pipeline                            2,501 tok/s
```

**Nothing here invalidates M6.** M6's 1.61x/1.80x came from *traced* per-stage times; this run just
shows that a PP driver without trace capture cannot realise them.

**The remaining step is per-stage trace capture.** The right vehicle is `TtPrefillRuntime` rather than
raw `TtPrefillTransformer`: it already implements segmented trace capture (`capture_trace()`), the
metadata-tensor path for per-chunk scalars, and it is PP-aware (`first_layer_idx` / `is_first_rank` /
`is_last_rank`), and its `prefill_chunk` already returns a non-last rank's output activation for the
driver to forward. Per iteration that reduces to 4 replays + 3 host hand-offs (5 MB each), which is
where the overlap finally appears. Chunked long-context (100k / 256k) then follows from the same
driver, since `TtPrefillRuntime` owns the chunked KV bookkeeping.

```bash
pp/pp08a_build_cache.sh 5120 36     # one-time: 36-layer 8x1 cache via submesh (65 GB, ~25 min)
pp/pp08_concurrent.sh 5120 12       # the eager pipeline above (PP_WINDOW / PP_ITERS)
```

### M8 — TRACED PP=4, and the pipeline finally overlaps (2026-08-18)

Added per-stage segmented trace capture to the concurrent driver (`SubDeviceTraceController` per
submesh — a single trace cannot span the MoE sub-device swaps), plus a non-blocking
`replay(blocking=False)` on that controller so four stages can be in flight before the one sync.
Capture is clean: **19 segments, 6.5 MB per stage**, output `[1, 1, 640, 4096]` per device.

Every stage is built **headless** (`is_last_rank=False`). The norm/LM-head/sample tail ends in a
blocking host readback, which cannot live inside a trace; throughput does not need the token, and a
real PP prefill's last stage is headless anyway (the runtime has `kv_only_last_layer` for exactly this).
So all four stages capture identically.

**The progression at window 5120, `PP=4 x (8,1)`, tells the whole story:**

| driver | steady-state iteration | **tok/s** | vs single-rank (26,203) |
|---|---|---|---|
| eager, 4 submeshes | 2047 ms | 2,501 | **0.10x** — host-bound, zero overlap |
| traced + host hand-off | 1269 ms | 4,035 | 0.15x — hand-off dominates |
| **traced, hand-off isolated** | **148.3 ms** | **34,531** | **1.32x** |

Two lessons, both measured:

1. **Tracing is necessary but not sufficient.** Tracing alone took 2,501 -> 4,035 tok/s. The rest of the
   gap was the hand-off.
2. **The host hand-off is fatal, and I had underestimated it by ~8x.** The activation per *device* is
   `[1,1,640,4096]`, but the composed tensor a host round-trip moves is `[1,1,5120,4096]` bf16 =
   **42 MB per hop**, so three hops move ~250 MB per iteration in each direction. Measured cost:
   **1269 - 148 = 1121 ms**, i.e. 88% of the iteration. A device-to-device transport is not an
   optimisation here, it is the difference between 4k and 34k tok/s.

With the hand-off isolated, 148.3 ms against the ideal 4x121.1 ms/4 = 121.1 ms single-stage time is
**82% overlap efficiency** — the four submeshes really are running concurrently, with ~27 ms of
sync/dispatch overhead per iteration.

**So PP=4 x (8,1) is worth 1.32x at a 5120 window, measured end to end**, once the hand-off is done on
device. That is below the 1.61x the single-stage ratio suggested (M6), the difference being the ~27 ms
of imperfect overlap.

### M10 — PP=4 at 100k and 256k, and the master comparison table (2026-08-18)

The concurrent driver runs one independent single-shot request per stage slot, so it measures
throughput with an EMPTY KV — it cannot model the KV-depth growth that dominates long context.
Rather than build a chunked multi-stage driver, the numbers come from the identity that makes PP
work: **steady state retires one chunk per slowest-stage time**, so summing one stage's per-chunk
times over a context gives the PP total. `test_mistral4_prefill_transformer_chunked_no_pcc` now
takes `mesh-8x1` + `L9`, which is exactly one stage, chunked at depth. (Its `(sp,tp)==(8,4)`
assert had to be relaxed to the two PP stage shapes.)

**One PP=4 x (8,1) stage, traced, chunked at depth (9 layers, 8 chips):**

| context | chunks | per-chunk first -> last | sum |
|---|---|---|---|
| 102,400 | 20 | 114 -> 290 ms | **3.75 s** |
| 261,120 | 51 | 117 -> 569 ms | **17.27 s** |

**PP=4 x (8,1) long-context throughput**, bracketed by the overlap efficiency actually measured
on the concurrent driver (81% at a 5k window, 69% at 25.6k):

| context | single-rank | PP ideal | PP @81% | PP @69% | realistic gain |
|---|---|---|---|---|---|
| 102,400 | 18,381 | 27,307 | **22,118** | 18,842 | **1.20x** |
| 261,120 | 10,494 | 15,118 | **12,246** | 10,431 | **1.17x** |

**So PP=4 x (8,1) is worth ~1.2x at long context** — real, but the *smallest* gain of any context
measured, and at the pessimistic end of the overlap bracket it is a wash. The reason is the one
that has held throughout: PP removes TP collectives, and by 256k the SP ring-attention term it
cannot touch dominates.

### M11 — PP=4 at 100k / 256k, measured directly (2026-08-18)

The M10 figures were derived (one stage's per-chunk sum x overlap efficiency). Now measured with all
four stages actually running chunked: `test_mistral4_pp4_concurrent_longctx` keeps each stage's 9
layers' KV for the whole context and flows chunk c through stages 0..3 over four iterations, so each
stage's attention sees everything it wrote for earlier chunks — chunked prefill, pipelined. One trace
still serves every chunk: the per-chunk scalars (slot_id / actual_start / actual_end) live in
1-element uint32 DRAM tensors refreshed in place between replays.

| context | single-rank | **PP steady state** | gain | PP whole request | gain | (M10 derived) |
|---|---|---|---|---|---|---|
| 102,400 | 18,381 | **22,917** | **1.25x** | 19,343 | 1.05x | 22,118 |
| 261,120 | 10,494 | **12,332** | **1.18x** | 12,264 | 1.17x | 12,246 |

**The derived method held up** — 3.5% and 0.7% off the measured steady state. Worth knowing, because
the stage-sum trick needs only 8 chips and one stage's cache, so it is the cheap way to sweep a new
machine before committing to the full 4-submesh setup.

**But the two PP columns differ, and that distinction matters more than the gain.** "Steady state"
counts only full-pipeline iterations — what a server streaming requests sees. "Whole request" includes
pipeline fill and drain, 3 of 23 iterations at 100k and 3 of 54 at 256k. At 256k those are amortised
(1.17x vs 1.18x, near-identical) but **at 100k a single request keeps only 1.05x of the 1.25x** —
fill/drain eats 84% of the benefit. So PP=4 is a *serving* win, and for one-off long prefills it is
close to a wash at 100k and worth ~1.18x at 256k.

### Master comparison — every configuration measured on this box

All traced, 36 layers, real weights, 32 chips, BH Galaxy (**8 kW**). Single-shot for 5,120 / 25,600;
chunked at CHUNK=5120 for 102,400 / 261,120.

| config | 5,120 | 25,600 | 102,400 | 261,120 | notes |
|---|---|---|---|---|---|
| **single-rank SP=8 x TP=4** (today) | 26,203 | **33,552** | 18,381 | 10,494 | the baseline; all four measured |
| single-rank, bf8 experts | 25,772 | 32,713 | — | — | -0.6% / -2.5%; +0.012 nPCC |
| **PP=4 x (8,1)** | **34,270** | **41,384** | **22,917** | **12,332** | all four MEASURED; 1.31x / 1.23x / 1.25x / 1.18x |
| PP=4 x (8,1), whole request incl. fill/drain | — | — | 19,343 | 12,264 | 1.05x / 1.17x — one request pays pipeline fill |
| PP=4 x (4,2) | 24,416 | 17,483 | — | — | **worse than no PP** (0.93x / 0.52x) |
| PP=4 x (8,1), eager | 2,501 | — | — | — | host-bound, zero overlap |
| PP=4 x (8,1), host hand-off | 4,035 | — | — | — | hand-off costs 1121 ms/iter |
| Gemma-4-31B-it (TP=8 x CP=4) | — | — | — | 11,058 | external reference, same box |

Reading it: **PP=4 x (8,1) is the best configuration at every context measured**, by 1.31x at 5k
falling to ~1.17-1.20x at 256k. `(4,2)` is never worth it. Tracing is not optional anywhere. And
the hand-off must be device-to-device — a host round-trip moves 42 MB per hop and costs more than
the entire pipeline.

### Re-running all of this on a higher-power machine

This box is **8 kW**; a **12 kW** machine should raise every number here, and prefill is exactly the
regime that benefits (device-bound at large windows — at 25.6k eager and traced are within 1%, so
the device, not the host, is the limit). Everything needed to reproduce the whole table:

```bash
# 0. environment (same on any box; caches are keyed by {arch}_{Ndev}/{sp}x{tp} so they do NOT
#    transfer between machines with different device counts -- expect cold builds)
cd <checkout> && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export MISTRAL4_HF_MODEL=<path to Mistral-Small-4-119B-2603>
export TT_MISTRAL4_PREFILL_TTNN_CACHE=$HOME/mistral4_ttnn_cache          # 65 GB, single-rank 8x4
export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=$HOME/mistral4_ref_cache

# 1. single-rank baseline: 5,120 / 25,600 (single-shot) -- the 26,203 / 33,552 row
PROFILE_SEQ_LEN=25600 PROFILE_ISL=25600 TRACE_MAX_NEW=0 PROFILE_REPS=3 PROFILE_WARMUP=1   pytest models/demos/deepseek_v3_d_p/demo/profile_prefill.py::test_trace -s -q   # "traced min"

# 2. single-rank long context: 102,400 / 261,120 (chunked) -- the 18,381 / 10,494 row
PREFILL_NOPCC_SEQ_CACHE=261120 pytest   models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py   -k "chunked_no_pcc and mistral4 and mesh-8x4 and L36 and chunks51 and two_iters and traced" -q -s

# 3. PP=4 caches, one per stage shape (65 GB each, ~25 min, built through a submesh)
TT_MISTRAL4_PREFILL_TTNN_CACHE=$HOME/mistral4_ttnn_cache_pp   python <repro>/pp/build_pp_cache.py            # PP_SP=8 PP_TP=1 ; repeat with PP_SP=4 PP_TP=2

# 4. PP=4 concurrent throughput at 5,120 / 25,600 -- the 34,270 / 41,384 row
TT_MISTRAL4_PREFILL_TTNN_CACHE=$HOME/mistral4_ttnn_cache_pp PP_WINDOW=25600 PP_ITERS=12 PP_HANDOFF=none   pytest models/demos/deepseek_v3_d_p/tests/test_prefill_pipeline_concurrent.py -k 8x1 -q -s
#   PP_HANDOFF=host shows what a naive host hand-off costs; -k 4x2 runs the losing variant

# 5. PP=4 long context, MEASURED: all four stages chunked and concurrent -- the 22,917 / 12,332 row
TT_MISTRAL4_PREFILL_TTNN_CACHE=$HOME/mistral4_ttnn_cache_pp \
PP_CONTEXT=261120 PP_WINDOW=5120 PP_HANDOFF=none \
  pytest models/demos/deepseek_v3_d_p/tests/test_prefill_pipeline_concurrent.py -k "longctx and 8x1" -q -s
#   PP_CONTEXT=102400 for the 100k row. Prints both "total ... -> N tok/s" (whole request, incl.
#   pipeline fill/drain) and "steady-state median ... -> N tok/s" (server case). Wrapper:
#   pp/pp12_longctx.sh <context> <chunk> <handoff>

# 5b. the cheap sweep: ONE stage chunked at depth, summed -- 8 chips, one stage's cache, ~1 min.
#     Landed within 0.7-3.5% of step 5 on this box, so use it to triage a new machine first.
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TT_MISTRAL4_PREFILL_TTNN_CACHE=$HOME/mistral4_ttnn_cache_8x1 \
PREFILL_NOPCC_SEQ_CACHE=261120 pytest \
  models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py \
  -k "chunked_no_pcc and mistral4 and mesh-8x1 and L9 and chunks51 and two_iters and traced" -q -s
#   sum the per-chunk medians it prints; wrapper: pp/pp11_stage_at_depth.sh chunks51 261120
```

Ready-made wrappers for all of the above are in `/data/kmabee/mistral4_repro_logs/{night2,pp}/`
(`run_tput_ab.sh`, `run_longctx.sh`, `pp08a_build_cache.sh`, `pp09_traced_pp.sh`,
`pp11_stage_at_depth.sh`), plus `analyze_longctx.py` for the per-chunk breakdown.

**What to watch for on the new box, in priority order:**

1. **Is the gain still there?** PP's win comes from removing collectives. If more power mainly
   raises compute clocks, CCL becomes a *larger* fraction and PP should win by MORE. If it raises
   memory/link bandwidth instead, the gap narrows. Either way the diagnostic is the same: compare
   one traced stage against the whole model (step 4 vs step 1) — per-chip compute is identical
   between them by construction, so the ratio isolates collectives plus matmul width.
2. **Overlap efficiency.** 81% at 5k and 69% at 25.6k on this box. If contention is power-limited
   it should improve, which lifts the long-context numbers most.
3. **Re-check the L1 ceiling.** Single-shot dies at ~100k on L1 circular buffers here; that is a
   per-core memory limit, not a power one, so it should NOT move. If it does, something else
   differs about the part.
4. **Do not compare eager numbers across boxes.** They measure host dispatch, not the device.

### M9 — both PP=4 variants measured, and `(8,1)` wins decisively (2026-08-18)

Ran the traced concurrent pipeline for **both** candidates from the parallelisation table. Each is 8
chips, so four of them tile the 32-chip galaxy; each needed its own 36-layer weight cache at
`32dev/{sp}x{tp}` (65 GB apiece, ~25 min, built through a submesh).

| window | single-rank (SP=8 x TP=4) | **PP=4 x (8,1)** | ratio | **PP=4 x (4,2)** | ratio | 8x1 / 4x2 | overlap eff |
|---|---|---|---|---|---|---|---|
| 5,120 | 26,203 tok/s | **34,270** | **1.31x** | 24,416 | **0.93x** | 1.40x | 81% |
| 25,600 | 33,552 tok/s | **41,384** | **1.23x** | 17,483 | **0.52x** | 2.37x | 69% |

*(overlap eff = one-stage time / measured iteration; 100% would be perfect 4-way overlap.)*

**Marko's open question — "4x2 or 8x1 is still a good question, 4x2 might be better" — is answered, and
it is `(8,1)`.** `(4,2)` is not merely worse than `(8,1)` (by 1.40x at 5k, **2.37x** at 25.6k); it is
**worse than not using PP at all** (0.93x and 0.52x). The reason is exactly the mechanism the CCL audit
predicted: `(4,2)` keeps 2-way TP collectives *and* halves the sequence split (SP 8 -> 4), so each rank
carries twice the tokens through a shorter ring. `(8,1)` instead preserves the SP=8 we already run and
deletes every TP collective. Nothing about "compute vs CCL" favoured `(4,2)` once measured.

**PP=4 x (8,1) is worth 1.23-1.31x on prefill throughput, measured end to end.** That is real but below
the 1.61-1.80x the single-stage ratio suggested (M6), and the gap is overlap efficiency: 81% at 5k
falling to **69% at 25.6k**. The drop with window size is worth noting — four stages hammering the same
galaxy's DRAM concurrently contend more as activations grow, so PP's benefit erodes exactly where the
single-stage numbers looked best.

**All three headline contexts are now measured** — 25k directly from this driver
(**41,384 vs 33,552, +23%**), and 100k / 256k from one stage chunked at depth (see M10 below).

```bash
pp/pp08a_build_cache.sh 5120 36 8 1        # one-time cache per shape (65 GB, ~25 min); "... 4 2" for 4x2
pp/pp09_traced_pp.sh 25600 12 8x1 none     # 41,384 tok/s  <- the 25k PP number
pp/pp09_traced_pp.sh 25600 12 4x2 none     # 17,483 tok/s  <- the losing variant
pp/pp09_traced_pp.sh 5120  12 8x1 host     # what a naive host hand-off costs (4,035 tok/s)
```

```bash
pp/pp08a_build_cache.sh 5120 36 8 1        # one-time 8x1 cache via submesh (65 GB, ~25 min)
pp/pp09_traced_pp.sh 5120 12 8x1 none      # the 34,531 tok/s number (handoff isolated)
pp/pp09_traced_pp.sh 5120 12 8x1 host      # what a naive host hand-off costs
```

### Original M7 plan (kept for the reasoning; superseded by the result above)

**These numbers do not exist yet, and M6 cannot produce them** — it times one stage in isolation. A
true e2e number needs the four stages running **concurrently on disjoint chips**, because that is
where PP's throughput comes from.

**The good news: it probably does not need the multi-process M5 driver.** The `(8,4)` mesh is 8 rows x
4 columns, the SP axis IS axis 0, so **each column is exactly an `(8,1)` stage** — and
`mesh_device.create_submeshes(ttnn.MeshShape(8, 1))` exists (used by `tt_transformers`'
`create_submeshes` for data parallel). That gives 4 concurrent stages inside ONE process:

```
parent (8,4) --create_submeshes(MeshShape(8,1))--> [col0, col1, col2, col3]
   stage r on submesh r, 9 layers, first_layer_idx = 9r, its own 9-slot KV cache
```

Pipelining it needs no D2D socket either, because ttnn ops are enqueued asynchronously. Enqueue all
four stages for four *different* chunks before blocking on any readback, with a one-iteration lag:

```
iteration t:  stage0(tokens[t])        # each enqueues on its own submesh -> all 4 run concurrently
              stage1(h0 from t-1)
              stage2(h1 from t-2)
              stage3(h2 from t-3)      # last rank produces the token
              then one sync + read back h0,h1,h2 for the next iteration
steady state: one chunk retired per max(stage time) -> throughput = CHUNK / iteration_time
```

The handoff can go through the host to start with: 3 hops x `[1, 1, chunk/8, 4096]` bf16 = **5 MB per
hop**, ~3 ms total at PCIe speeds against a ~121 ms stage — small enough that a host-mediated prototype
gives a usable number, and a real D2D publish only improves it.

**Two prerequisites, one unverified:**

1. ~~**Does a submesh inherit working fabric for its column?**~~ **✅ YES — probed 2026-08-18.**
   `pp/probe_submesh.py` on an `(8,4)` parent:

   ```
   created 4 submeshes: [MeshShape([8,1]) x 4]
   all_gather(cluster_axis=0) on submesh 0: [1,1,32,128] -> [1,1,32,1024]   # 8x on the gather dim = SP=8
   all_gather on a SECOND submesh: OK                                        # independently usable
   PROBE_RESULT: SUBMESH_PP_VIABLE
   ```

   So the SP-axis collectives that ring attention needs **do** work inside a column, and two submeshes
   can be driven independently. **The one open technical risk for the concurrent approach is closed.**
2. **A weight cache at `32dev/8x1`.** The cache path is keyed on `get_num_devices()`, which is 32 in a
   submesh session, so the existing `8dev/8x1` cache (18 GB, built for the carve) does not resolve.
   Either build it (~4 min for 9 layers) or symlink — the per-8-device shards should be byte-identical,
   since a `(8,1)` submesh maps tensors exactly as an 8-device full mesh does.

If the probe passes, the driver is a contained piece of work and yields measured 25k / 100k / 256k PP
throughput directly comparable to today's 33,550 / 18,381 / 10,494. If it fails, PP throughput needs
the real multi-process M5 path (4 ranks, `TT_VISIBLE_DEVICES` per rank, D2D socket transport).

**Still projections, not PP throughput.** These are single-stage times, so they assume the pipeline is
perfectly full and the inter-stage handoff is free. The handoff is 3 hops of
`[1, 1, chunk/8, 4096]` bf16 = 5 MB each at CHUNK=5120, and fill/drain costs 3 stage-times per
request. Both are M5.

### The strategy: smallest probe first, because one question gates everything

`PP=4 x (8,1)` only pays off if **TP=1 works at all**. Every dense-path collective in this model is on
the TP axis, so TP=1 is the shape where they all vanish — but it is also a shape nothing has ever run.
If MLA or MoE cannot build without a TP axis, the whole proposal dies before any pipeline plumbing
matters. So the order is: prove TP=1 on one block, then the whole model, then slice it into stages,
and only then worry about ranks and transport.

A useful accident of the geometry makes this cheap: **`(32,1)` is TP=1 without needing an 8-chip
carve.** It is CI-listed for BLACKHOLE_GALAXY with FABRIC_1D, and at 32 devices
`experts_per_chip = 128/32 = 4` — the same as today's `(8,4)` — so the MoE side is unchanged and TP is
the only variable. It is not the PP stage shape (that is `(8,1)`, 16 experts/chip, and needs the
carve plus a cold weight cache), but it isolates the risky variable for minutes instead of an hour.

### What already exists, and the one thing that does not

Much more scaffolding is in place than the handoff implied:

| piece | state |
|---|---|
| `TtPrefillTransformer` rank slicing (`first_layer_idx`, `is_first_rank`, `is_last_rank`) | **done** — embedding only on rank 0, norm/LM-head/sample only on the last, non-last ranks return the hidden state |
| `TtPrefillRuntimeConfig` PP fields | **done** — same three knobs, defaults make a single-rank runtime own the whole model |
| `prefill_chunk` output on a non-last rank | **done** — returns the rank's output activation (`_trace_output` under trace) |
| `make_chunk_input` on a non-first rank | **placeholder** — `make_placeholder_activation()` returns zeros |
| KV chunk table across stages | **done** — `allgather_kv_stage_layout` merges per-rank layer ranges into one table (tt-blaze layer->mesh merge) |
| cache completeness per rank | **done** — `check_cache_complete(..., first_layer_idx, is_first_rank, is_last_rank)` |
| **inter-stage activation transport** | **MISSING** — the code says so: *"today via a placeholder; via a D2D-socket publish op once that lands"* |
| test rows that can express PP | **missing** — the chunked rows hard-`assert (sp, tp) == (8, 4)` |

So the gap is narrower than "build PP": it is **the transport plus a driver**. And transport is only
needed for *multi-rank*. Four stages inside one process can hand the tensor over directly, which is
what makes an early correctness milestone reachable.

### Milestones

**M1 — MLA at TP=1. ✅ PASSED (2026-08-18, 61 s).** `test_mistral4_mla` at `(32,1)`, FABRIC_1D,
seq5k, random weights: **output PCC 0.9970, KVPE kv 0.99984, pe 0.99984**. Compare the `(8,4)` row's
0.9969 — TP=1 is not worse. The single most important result so far: attention has no hidden TP
dependency.

```bash
pp/pp01_mla_tp1.sh 32x1     # -k "test_mistral4_mla and 32x1 and line and check_pcc and seq5k and max_sl"
```

**M2 — whole transformer at TP=1.** Added `mesh-32x1` and `mesh-8x1` rows to
`test_mistral4_prefill_transformer` (2 layers, random weights, so no pretrained cache is needed). This
is the step that found real work, in two distinct flavours.

**M2a — the first genuine TP=1 bug, fixed.** `ttnn.all_gather` TT_FATALs on a length-1 axis rather
than degenerating to a copy:

```
tt_prefill_block.py:555 -> tt_distributed_rms_norm.py:274
TT_FATAL: all_gather collective will only work for num_devices > 1, got 1
```

The distributed RMSNorm gathers per-device `sum(x^2)` across `cluster_axis=1`. At TP=1 there is nothing
to gather — each device already holds the full hidden dim, so its local statistic *is* the global one.
Fixed by short-circuiting the gather to a pass-through, deliberately keeping the
`rms_norm_pre_all_gather` / `rms_norm_post_all_gather` pair rather than switching to a plain
`ttnn.rms_norm`, so the TP=1 path cannot drift numerically from the TP>1 one. Note `tt_reduce.py`
already had the equivalent guard (`if mesh_device.shape[cluster_axis] > 1`) — the norm simply never
got one, because nothing had ever run TP=1. After the fix the norm emits `[1, 1, 32, 4096]`, i.e. the
undivided hidden dim, and the stack runs through MLA unchanged.

**M2b — `(32,1)` turns out to be the wrong probe, for a reason worth knowing.** The next failure was
not about TP at all:

```
tt_moe_routing_setup.py:220 -> masked_bincount
TT_FATAL: Token count (32) must be divisible by the 64-core grid used by masked_bincount
```

`(32,1)` over-shards the *sequence*: SP=32 leaves `isl/32` tokens per chip — 32 at isl 1k — and the
MoE routing setup needs the per-chip token count to be a multiple of its 64-core grid. The real stage
shape `(8,1)` gives `isl/8 = 128` and is fine. So the convenient no-carve probe is only good up to the
MoE, and validating the stage means using the stage geometry: 8 chips via `TT_VISIBLE_DEVICES=0..7`.

**This also constrains PP chunk sizes, which is a keeper:** at SP=8, `chunk/8` must be a multiple of
64, so **chunk must be a multiple of 512**. `CHUNK = 5120` gives 640/chip and is fine; an
arbitrary chunk size is not.

**M2c — the real stage geometry, `(8,1)` on an 8-chip carve. ✅ PASSED (194 s).** The whole
transformer — embedding, MLA, MoE with 16 experts/chip, distributed norms, LM head, sampling — runs at
**SP=8 x TP=1**, 2 layers, random weights. The carve itself works on a galaxy whose CI fabric table
lists only 32-chip meshes, which was not a given.

**This is the milestone that de-risks the proposal.** TP=1 was the one assumption that could have
killed `PP=4 x (8,1)` outright, and it cost exactly one fix (the RMSNorm guard) plus discovering that
the sequence-shard floor forces the stage geometry rather than a convenient 32-chip probe.

```bash
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pp/pp03_xf_stage_shape.sh 2_layers
# -k "test_mistral4_prefill_transformer and mesh-8x1 and 2_layers and 1k and smoke-random-random"
```

**M2d — a full-depth stage (9 layers = 36/4) at `(8,1)`. ✅ PASSED (347 s).** Exactly one stage's
worth of work, at the stage's geometry. **TP=1 is now de-risked end to end**: one block (M1), the whole
stack (M2c), and a full stage's depth (M2d).

```bash
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pp/pp03_xf_stage_shape.sh 9_layers
```

**M3 — 4 stages in one process.** New test `tests/test_prefill_pipeline_stages.py`: four
`TtPrefillTransformer` instances with `first_layer_idx` 0/9/18/27 and the boundary flags set, run in
sequence with each stage's output activation handed straight into the next, asserted to sample the
**same token** as a single-rank 36-layer run over the same weights.

Two deliberate choices in it. It runs at **mesh-8x4, not 8x1**: TP=1 is already covered by M2, and
(8,4) is where the warm pretrained cache lives, so both configurations read byte-identical weights and
the comparison is exact rather than approximate — this test is about *slicing*, not about TP. And each
stage gets its **own 9-slot KV cache**, because `forward` writes at the LOCAL `cache_layer_idx` while
blocks are built at the GLOBAL `layer_idx`; a real rank owns only its slice's cache, and one shared
cache would have the four stages overwrite each other. (That asymmetry is the single easiest thing to
get wrong in PP, and it is why this milestone exists.)

**✅ PASSED (190 s), and more strongly than asked:**

```
rank 0: layers 0-8   -> activation [1, 1, 128, 1024]      (128 = isl/sp, 1024 = hidden/tp)
rank 1: layers 9-17  -> activation [1, 1, 128, 1024]
rank 2: layers 18-26 -> activation [1, 1, 128, 1024]
rank 3: tail ran     -> token 2 (p=0.7147)
single-rank 36L      -> token 2 (p=0.7147)
```

Not just the same token — **the same probability to four decimals**, i.e. the sliced model is
numerically indistinguishable from the monolithic one, which is what you want from a hand-off that is
supposed to be lossless. `PP=4` layer slicing works.

One bug found on the way, and it was mine, not the framework's: the first attempt failed on rank 3
with `matmul ... width=1024 height=4096`. `TtPrefillTransformer` defaults
`lm_head_is_column_parallel=False`, which builds a row-parallel LM head expecting the full 4096 hidden,
while `test_prefill_transformer` passes `True`. Ranks 0-2 have no tail so they never noticed. Worth
recording because **a real PP driver must set it too** — the last rank is the only one that builds a
tail, so this default is invisible until the moment it matters.

```bash
pp/pp05_stages.sh   # pytest tests/test_prefill_pipeline_stages.py -q -s
```

**M3 — 4 stages in one process.** Build four runtimes with `first_layer_idx` 0/9/18/27 and the
boundary flags set, run them in sequence passing the hidden state directly, and check the final token
against the single-rank 36-layer result. No transport, no multi-rank — validates layer slicing and
boundary gating, which is the part most likely to be subtly wrong.

**M4 — the CCL number. ⚠ ATTEMPTED, blocked on tooling, not on the model.** Tried profiling the
`(8,1)` 9-layer stage with
`python -m tracy -r -p -v -m pytest tests/test_prefill_transformer.py -k "... mesh-8x1 ..."`. Tracy
runs and writes a host-side report but ends with **"No device logs found"** — the pytest inside it does
not appear to select/run the test (a trailing `-q` was even absorbed into the trace filename,
`-q_2026_08_18_15_29_51.tracy`, so argument passing is the suspect). Not chased further; it is harness
plumbing, and it also leaves a `serve_wasm` web-UI subprocess behind that has to be killed.

The known-good recipe is `demo/profile_prefill.py::test_ops` under tracy — but that file hardcodes
`(8, 4)` in its `_marks` and loads real weights, so pointing it at a TP=1 stage needs either a
`(8,1)` mesh param plus a cold `mistral_small4_bh_8dev/8x1` weight cache (~16 min), or a random-weight
mode. That is the concrete next step for M4.

**Worth noting what is already evidence, though:** M2a is itself a runtime demonstration that a TP
collective was on the critical path — the stage *could not run* until the RMSNorm's all-gather was
bypassed. And no other collective blocked TP=1 afterwards, which is consistent with the audit (the
rest either self-disable, like `tt_reduce`, or are not reached). So the direction is confirmed; what
M4 would add is the *magnitude* on a real stage, which the 26.8% figure already brackets.

**M5 — real PP.** Transport (D2D socket publish/wait) + a 4-rank driver + bubbles. This is the
genuinely new engineering; everything above de-risks it.

### Is it worth it? What the numbers say now

**Yes, on the evidence so far — with one honest limit.** PP does not reduce total work; it removes the
TP collectives and replaces them with three inter-stage handoffs. So the ceiling is set by what the
collectives cost, and that is measured: **26.8% of device time, 30.8% counting the distributed
LayerNorm pair that reverts to a plain norm at TP=1.** If handoffs and bubbles were free, that is
~1.37x on prefill throughput.

**Superseded by M6, which measured it rather than inferring it:** one stage is **1.61x** faster than
the whole model at a 5,120 window and **1.80x** at 25,600 — better than the 1.37x the CCL share alone
predicts, because TP=1 also widens every matmul 4x. But the win shrinks with context, because PP does
not touch the ring-attention term: at 256k the projection brackets **11,900-16,900 tok/s** against
today's 10,494 (i.e. +13% to +61%, not the flat 1.37x guessed here). Full numbers and commands in "M6 —
one traced stage" above.

**The limit worth stating plainly:** PP buys *throughput*, not single-request latency, and only when
the pipeline is full. It needs >=4 chunks in flight, so a short prompt gets nothing (or worse). Long
context is exactly where it pays, and 256k gives 51 chunks — which is why this is worth doing *for the
256k target specifically*, and not as a general default.

**Two things that could still eat the gain**, neither yet measured: the per-chunk D2D activation
handoff (3 hops of `[1, 1, chunk/8, 4096]` bf16 = 5 MB per hop at CHUNK=5120), and pipeline fill/drain
at the ends of a request. Both are M5 work.

### Running total

| milestone | state |
|---|---|
| M1 MLA at TP=1 | ✅ PCC 0.9970 / KVPE 0.9998 |
| M2a `all_gather` on a length-1 axis | ✅ fixed (RMSNorm stats gather -> pass-through) |
| M2b `(32,1)` is not a valid probe | ✅ understood (masked_bincount needs tokens/chip % 64 == 0) |
| M2c whole transformer at `(8,1)` | ✅ passes, 2 layers |
| M2d full stage depth (9 layers) at `(8,1)` | ✅ passes |
| M3 four stages in one process | ✅ passes — same token AND same probability as single-rank |
| M4 TP=1 CCL profile | ⚠ attempted — tracy captured no device logs (harness, not model) |
| M6 one traced stage at `(8,1)` | ✅ **1.61x @5k, 1.80x @25.6k** vs the whole model |
| M5 transport + 4-rank driver + bubbles | not started |
| M7 concurrent 4-submesh pipeline | ✅ built + measured — **2,501 tok/s, 10x worse**: eager dispatch is host-bound, zero overlap |
| M8 per-stage TRACE capture in the PP driver | ✅ traced replay overlaps (81% eff); host hand-off costs 1121 ms/iter |
| M9 both variants measured | ✅ **(8,1) = 1.23-1.31x single-rank; (4,2) = 0.52-0.93x, worse than no PP** |
| M10 PP=4 at 100k / 256k (derived) | ✅ ~22,118 / ~12,246 from one stage chunked at depth |
| M11 PP=4 at 100k / 256k (measured, 4 stages chunked) | ✅ **22,917 / 12,332 steady (1.25x / 1.18x)**; 19,343 / 12,264 incl. fill/drain |

**Code changed for PP so far** — deliberately small: one guard in `tt_distributed_rms_norm.py`, three
new test parametrizations (`mesh-32x1`, `mesh-8x1`, `9_layers`), and one new test file
(`tests/test_prefill_pipeline_stages.py`). Nothing in the model's structure had to move, which is the
strongest signal that the existing rank-slicing scaffolding was built correctly.

### Where PP=4 x (8,1) actually stands — nothing was decided against it

**Origin and attribution.** `PP=4 x (8,1)` (PP=4, SP/CP=8, **TP=1**) is **Marko's idea**
(`draft_question_parallelization.md`, table row "PP=4 x (8,1) — Marko idea"). He also corrected our
memory arithmetic on 2026-08-16: `experts_per_chip = num_routed_experts // num_devices`, so experts
spread over *every* device and TP shards only the small MLA weights — weights land ~3.6 GB/device in
any arrangement, and our user-capacity ranking had been inverted.

**But he did not settle on TP=1, and it is worth not misquoting him.** His follow-up:

> "no difference for expert weights. 4x2 or 8x1 is still a good question 4x2 might be better — its
> compute vs ccl analysis"

So Marko raised `(8,1)`, explicitly left `(8,1)` vs `(4,2)` open, and thought `4x2` *might* win. What
he asked for was the compute-vs-CCL analysis, not a conclusion.

**The CCL motivation is real and independently measured twice.** Sasha's profiled Gemma-4 layer put
CCL at **30.6%** of device time, larger than SDPA. Our own tracy run puts Mistral's CCL at **26.8%**
pure (30.8% counting the distributed-LayerNorm pair). Same story on both models.

**What this branch has now added to Marko's open question** (2026-08-18):

1. **Capacity: settled, and it fits better than today.** `PP=4/TP=1/CP=8` needs **4.91 GiB/chip** of
   weights against 31.9 available — *less* than today's `SP=8 x TP=4` at 5.41, because the TP=1
   dense-replication penalty (x4) is exactly cancelled by holding 9 layers instead of 36 (/4), and the
   expert term is identical either way (16 experts x 9 layers = 144 expert-layers/chip, same as
   4 x 36). This independently reproduces Marko's "no difference for expert weights" from the other
   direction. (His ~3.6 GB counts experts at fp8; our 4.91 GiB counts experts at bf4 plus dense at
   bf16 — different bases, same conclusion: memory does not bind.)
2. **CCL: settled, and it is 100% TP-axis.** An AST audit of all 17 CCL call sites under `tt/`: 14
   explicitly TP (10 `self.tp_axis`, 4 literal `1`); `tt_reduce` and `tt_distributed_rms_norm` both
   default `cluster_axis=1` and `tt_reduce` self-disables via `mesh_device.shape[cluster_axis] > 1`;
   and the single `sp_axis` site (`tt/mla/mla.py:1832`) is gated on `_has_indexer`, i.e. sparse-DSA
   only, so it never fires for dense Mistral. **TP=1 therefore deletes all of it.**
3. **Bubbles: effectively settled for long context.** PP=4 needs >=4 chunks in flight. The 256k run is
   **51 chunks**; 100k is 20. Bubbles only bite short prompts, which is exactly where PP is the wrong
   tool anyway. This was the third of the three unknowns and the 256k measurement closes it.
4. **`(8,1)` beats `(4,2)` *from where we already are*, which is a sharper argument than the original
   framing.** We run SP=8 today. `PP=4 x (8,1)` keeps SP=8 — **the ring-attention length on the SP
   axis is unchanged** — and only drops TP 4 -> 1, so the TP collectives vanish with no offsetting
   longer ring. `PP=4 x (4,2)` instead cuts SP 8 -> 4 *and* keeps 2-way TP collectives. Attention
   parallelism is 8-way either way (8 sequence ranks, or 4 sequence x 2 head ranks), so compute is
   roughly a wash and CCL is the only real difference — which is Marko's own framing, now answered.

**So: no, PP was not dropped, and nothing argued against it.** What was declined was one *proxy
measurement* — rebuilding the weight cache for an 8-chip `TT_VISIBLE_DEVICES` carve to profile
SP=8 x TP=1 at 36 layers under tracy (~40 min, ~70 GB). That configuration is not the PP config
either, so it is a proxy whose expected result the audit already derives; it is a confirmation run,
worth doing when convenient but not load-bearing.

**What PP actually still needs is engineering, not analysis:** the runner/pipeline integration —
4 ranks, `first_layer_idx` per rank, inter-stage D2D activation handoff, and the layer-completion /
migration plumbing. `TtPrefillRuntime` and `allgather_kv_stage_layout` are already PP-aware
(per-rank layer ranges, one merged KV chunk table across stages), so the scaffolding exists. The
chunked test rows, by contrast, hard-`assert (sp, tp) == (8, 4)`, so they cannot express PP as written.

### Gemma-4-31B-it side by side — the closest external check we have

Sasha's `svuckovic/gemma4-prefill` branch prefills **262,144 tokens in 23.7 s = 11,058 tok/s** on the
same 32-chip Blackhole Galaxy, traced, reproduced three consecutive runs (11058 / 11059 / 11050).
Notes in `gemma4_cp_prefill_branch_report.md` §6. That is the one directly comparable long-context
number in reach, and **both configurations are single-rank two-axis — neither uses pipeline
parallelism**, so the comparison is clean:

| | Gemma-4-31B-it | Mistral Small 4 119B |
|---|---|---|
| mesh (32 chips) | TP=8 x **CP=4** | **CP(SP)=8** x TP=4 |
| chunk | 4,096 (64 chunks) | 5,120 (51 chunks) |
| tokens | 262,144 | 261,120 |
| **time** | **23.7 s** | **24.88 s** |
| **aggregate tok/s** | **11,058** | **10,494** |
| per-chunk, shallow -> deep | 190-195 ms -> 489 ms | 178 ms -> 837 ms |
| µs/token, shallow -> deep | 46-48 -> 119.4 | **34.8** -> 163.5 |
| KV cache / device @256k | 13.5 GiB (their design table) | **0.4 GiB** (measured) |
| params (active/token) | 31B dense (31B) | 119B MoE (**~10B**) |
| layers | 60 | 36 |
| attention | interleaved sliding(1024) + global | full causal every layer, MLA |

**Do our numbers make sense? Yes — and the agreement is the point.** Two independent
implementations, different architectures, same hardware, both ~10-11k tok/s at 256k, ours 5% behind.
Nothing about our number is anomalous. Four things the table actually tells us:

1. **Our mesh is the mirror of theirs.** They chose TP=8 x CP=4; we run CP=8 x TP=4. We already have
   twice their context parallelism and half their tensor parallelism. Worth knowing before anyone
   proposes "add CP" as a lever — we are the config they could not afford (their design table has
   TP=4/CP=8 at 27 GiB/dev, OOM).
2. **We win shallow, they win deep.** 34.8 vs ~47 µs/token on the first chunk; 163.5 vs 119.4 on the
   last. That is the sliding-window signature: most of Gemma4's 60 layers are windowed at 1024, so
   their attention growth is capped, while all 36 of ours are full causal. Our quadratic coefficient
   is ~1.8x theirs (5.2e-10 vs 2.8e-10 s per token-of-chunk per token-of-KV).
3. **The crossover sits right at ~256k, and past it they pull ahead.** Extending both linear
   per-chunk fits to 512k: Gemma4 ~63 s against our ~85 s, so ~1.35x. Treat the direction as solid and
   the magnitude as soft — this is my arithmetic on their two published points (190-195 ms shallow,
   489 ms at depth 63), and that line reproduces only 21.8 s of their own reported 23.7 s at 256k, so
   their true curve is ~8% steeper than my fit and the real gap at 512k is probably nearer 1.25x. The
   architectural conclusion is what matters: full-attention MLA is competitive to 256k and loses
   ground beyond it, because their windowed layers cap what ours cannot.
4. **KV footprint is our decisive advantage, and it is MLA.** 0.4 GiB/device against their 13.5.
   Their chunk size and mesh were *forced* by KV capacity; ours are free choices. It is also why
   PP=4/TP=1/CP=8 is even discussable for us.

And the honest asymmetry: **we are doing ~3x less FFN work per token (~10B active vs 31B dense) and
still coming in 5% behind.** That gap is MoE overhead — Dispatch + Combine + PostCombineReduce is 17%
of our device time, more than the 11% the expert FFN it feeds costs. So there is headroom in our
number that a dense model does not have.

Two caveats, in both directions. Theirs: their report says the perf loop "has been exercised roughly
once", the last two perf commits are marked *"Not run — the device was off limits"*, and the archived
baseline CSV is not in the repo — though the 11,058 headline itself was reproduced three times.
**Ours: the 256k run is timing-only** (synthetic tokens, no PCC), whereas they have per-layer PCC
0.9993-0.9999 vs HF and a passing full graph at their config. On correctness-at-depth they are ahead
of us, and closing that needs a ~256k golden trace (~30 s/layer x 36 on the host).

### Reproducing the three headline numbers

Common environment for all three (from `/data/kmabee/mistral4_repro_logs/night2/`, which has these
as `run_tput_ab.sh` and `run_longctx.sh`):

```bash
cd /data/kmabee/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/home/kmabee/mistral4_ttnn_cache      # 65 GB, bf4 experts
export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=/home/kmabee/mistral4_ref_cache
```

**1. 25,600 single-shot — 0.763 s, 33,550 tok/s** (the peak; `PROFILE_ISL == PROFILE_SEQ_LEN` so every
token is real and routed, which is what makes it a prefill number rather than a generation number):

```bash
PROFILE_SEQ_LEN=25600 PROFILE_ISL=25600 TRACE_MAX_NEW=0 PROFILE_REPS=3 PROFILE_WARMUP=1 \
PROFILE_OUT=/tmp/tput_25600.json \
  ./python_env/bin/pytest models/demos/deepseek_v3_d_p/demo/profile_prefill.py::test_trace -s -q
# prints "traced min 763.0 ms" (and traced_min in PROFILE_OUT); tok/s = 25600 / 0.763.  ~2 min.
```

**2. 102,400 chunked (20 x 5120) — 5.57 s, 18,381 tok/s**:

```bash
PREFILL_NOPCC_SEQ_CACHE=102400 \
  ./python_env/bin/pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py \
  -k "test_mistral4_prefill_transformer_chunked_no_pcc and L36 and chunks20 and two_iters and traced" \
  -q -rs -s
# ~2.5 min. Prints the per-chunk median table + "iter 1 done (20 chunks) in 5.579 seconds".
```

**3. 261,120 chunked (51 x 5120) — 24.88 s, 10,494 tok/s — the 256k headline**:

```bash
PREFILL_NOPCC_SEQ_CACHE=261120 \
  ./python_env/bin/pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py \
  -k "test_mistral4_prefill_transformer_chunked_no_pcc and L36 and chunks51 and two_iters and traced" \
  -q -rs -s
# ~3.5 min wall (46 s model build + 51 chunks x 2 iterations).
# Prints "iter 1 done (51 chunks) in 24.912 seconds"; tok/s = 261120 / 24.88.
```

Summarise any of the chunked logs with
`./python_env/bin/python /data/kmabee/mistral4_repro_logs/night2/analyze_longctx.py <log>` — it prints
per-chunk time, cumulative time, instantaneous and aggregate tok/s, and the `t(k) = a + b*k` fit.

All three numbers are with the **default `bfloat4_b` experts**. Nothing above sets
`TT_PREFILL_ROUTED_EXPERT_WEIGHTS_DTYPE`, so nothing needs unsetting; `bfloat8_b` costs ~2.5% at the
25k window (and needs its own cache dir) — see the bf8 section above.

Four things that will bite you here:

- **`-s` is required** or you never see the table: it is logged at INFO, and pytest swallows captured
  output on a passing test.
- **`traced` is required.** Drop it and you get the `notrace` row, which reports ~0.67 s per chunk
  flat and is measuring host dispatch (see above). `traced` also needs `trace_region_size` in the
  row's `device_params` — it is there now; if you copy this row for another variant, carry it over.
- **`PREFILL_NOPCC_SEQ_CACHE` must be >= n_chunks x 5120**, and should be a whole multiple of `CHUNK`
  (the block-cyclic slab math assumes complete slabs). 51 x 5120 = 261,120, hence that value rather
  than 262,144.
- **Several `chunks*` ids are prefixes of each other**, and `-k` is a substring match: `chunks5` also
  selects `chunks51`, `chunks2` also selects `chunks20`, `chunks1` also selects `chunks10`. The extra
  test then fails its `total_len <= SEQ_CACHE_NOPCC` assert unless the cache happens to fit it (this
  is a real thing that happened: `-k chunks5` with a 25,600 cache reported "1 failed, 1 passed", the
  failure being `261120 <= 25600` from the accidentally-selected `chunks51`). The three commands above
  use `chunks20` / `chunks51`, which are unambiguous; if you want `chunks1`, `chunks2` or `chunks5`,
  pin the full test id instead.

**The lever is chunk size, and it is ABI-locked.** Chunk 0 of a 5,120 chunked run reaches
~28.7k tok/s while a single 25,600 forward reaches 33.5k — a bigger chunk amortises better and would
lift the whole curve, which is what the earlier throughput sweep argued for (8k–25k). But
`PREFILL_CHUNK_OUTPUT_TOKENS = 5120` is baked into the disaggregation address table (see the KV cache
section), so raising `CHUNK` is an ABI change, not a tuning knob.

### `logical_n` is a placeholder, but it does NOT cost full-cache attention

Worth writing down because it looks alarming and is not. `tt/mla/mla.py:1004`, on the metadata
(trace-safe) path the chunked runtime uses:

```python
if metadata is not None:
    ring_logical_n = kvpe_cache.storage.shape[2] * self.sp_factor   # global cache capacity
else:
    ring_logical_n = kv_actual_isl + chunk_size_global              # the tight bound
```

The comment calls it "a placeholder = global cache capacity", which reads like every chunk paying
attention over the whole allocated cache. **It does not.** Two measurements say so:

* traced per-chunk time grows **linearly with position** (0.170 s at chunk 0 to 0.399 s at chunk 19),
  so the work tracks the actual accumulated KV, not the cache size;
* untraced, per-chunk time was identical at a 25,600 and a 102,400 cache — a 4x cache change with no
  time change.

So `logical_n` is a sizing/extent hint and the kernel bounds real work by `kv_actual_isl_tensor`.
Nothing to fix for performance; the misleading part is only the name and the comment.

## The KV cache: what we store and why

**What Mistral uses today.** `MlaKvCacheFormat.BFP8_TILE` — `ttnn.bfloat8_b`, `ttnn.TILE_LAYOUT`,
logical row `kv_lora_rank 256 + qk_rope_head_dim 64 = 320`. Selected in exactly one place,
`allocate_mla_kvpe_cache` (`utils/kv_cache_utils.py`), which is the default allocator for
`TtPrefillRuntime` *and* what `MLAPrefillAdapter.allocate_kv_cache` calls — so all dense MLA
residents share one definition. Not variant-specific: Mistral inherits it.

```
per-device shape   [num_users * num_layers, 1, max_seq_len / sp, 320]     # user-major slots
dim 1 == 1         MLA's whole point: ONE latent row serves all 32 heads
seq axis           SP-sharded (/8), and BLOCK-CYCLIC ordered under chunked prefill
across TP          replicated (TensorTopology is 1D Replicate over all 32 chips)
DRAM               ND-sharded, shard_shape [1, 1, 32, 320], ROUND_ROBIN_1D over DRAM banks
bytes/token/layer  340   (10 tiles x 1088 B per 32-token chunk; 1088 = 1024 data + 64 exponent)
```

**Why bfloat8_b.** 340 B/token/layer against bf16's 640 — **1.88x less** KV DRAM and bandwidth — and
it is the format the dense SDPA/matmul path consumes natively, so no conversion on the read. It is
also what the decode side already allocates (`deepseek_v3/tt/mla/mla1d.py` `_convert_cache`:
`bfloat8_b` + `TILE_LAYOUT`), which is the point: `init_kvpe_cache`'s default and
`allocate_dflash_kv_cache`'s dtype comment both say "align w/ decode KV cache" explicitly.

**Why TILE.** Dense SDPA reads tiles. ROW_MAJOR is needed only by the sparse/DSA gather path, which
is why the other two formats are row-major. `head_dim` must be a whole number of tiles, and
`320 / 32 = 10` ✓ (`_dram_chunk_size_bytes` raises otherwise — an unaligned head_dim would silently
undersize the address table and corrupt the migration).

**The two formats we do NOT use, and why.**

| format | dtype / layout | B/token/layer @320 | why not |
|---|---|---|---|
| `BFP8_TILE` | bfloat8_b / TILE | **340** | — this is ours |
| `BF16_RM` | bfloat16 / ROW_MAJOR | 640 | 1.88x the bytes; row-major only needed for sparse SDPA |
| `SCALED_FP8` | fp8_e4m3 / ROW_MAJOR | 392 | **rejected at 256** — and bigger than bfp8 here anyway |

`SCALED_FP8` packs fp8 latent + fp32 per-128-block scales + **bf16** rope into one mixed row:
`rope_offset_bytes = 256 + (256/128)*4 = 264`, and `264 % 16 = 8`, so `validate_scaled()` raises —
it needs 16-byte field alignment. DeepSeek's `kv_lora_rank = 512` gives `512 + 16 = 528`, `% 16 = 0`,
so it is fine there. This is `kv_lora_rank = 256` being unprecedented in the family again. Note it
would not even be a win: 392 B > 340 B, because the rope half stays bf16. Scaled-FP8 exists to shrink
the *sparse* path's row-major cache (656 B vs BF16_RM's 1152 B for DeepSeek), not the dense one. So
Mistral loses nothing by being excluded — but the 264-byte misalignment is a live trap for anyone who
opts into it, and dense `allocate_mla_kvpe_cache` never selects it.

**Whole-model KV** (36 layers, 1 user, bfp8; `night2/T8_kv_cache_math.txt`):

```
context     5,120   25,600   131,072   262,144   1,048,576
total     0.06 GiB 0.29 GiB  1.49 GiB  2.99 GiB   11.95 GiB
per chip  0.01     0.04      0.19      0.37        1.49 GiB   (SP=8)
```

Against 31.9 GiB/chip this is not a constraint at any context we care about.

### For disaggregation, dtype is the easy part

Migration is **not** a tensor hand-off — it is raw-DRAM RDMA driven by a `KvChunkAddressTable`
(`tt/runners/kv_chunk_table.py` → `ttnn.experimental.disaggregation.export_to_protobuf_file`). The
worker copies bytes at addresses the table names, so *everything* about the physical layout is ABI,
not just dtype. Where prefill and decode stand:

- **dtype ✓ / layout ✓ / row width ✓** — both sides are `bfloat8_b` + `TILE` at
  `kv_lora_rank + qk_rope_head_dim`. This is already consistent, and it was chosen to be.
- **row width ✓, by construction** — decode computes
  `kvpe_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim`, the same formula prefill uses.
  Both land on 320 for Mistral from the same config fields, so the width cannot drift.
- **slot organisation ✗** — prefill is `[num_users*num_layers, 1, seq/sp, 320]`, user-major
  contiguous, ND-sharded round-robin over DRAM banks, replicated across TP. Decode defaults to
  **paged**: `(max_num_blocks, 1, block_size, kvpe_dim)` with `ShardTensorToMesh(dim 0)`.
  Contiguous-slot vs paged-block, and TP-replicated-plus-SP-sharded vs dim-0-sharded, is the real
  seam. `mla1d.py` does expose the hook for it — `kv_cache_override: KvCacheConfig` replaces the
  paged shape with a caller-supplied 4D shape — so making decode's cache match prefill's slots is a
  deployment-time decision someone has to make explicitly, not something that happens by default.
- **block-cyclic order ✗ (unverified for Mistral)** — under chunked prefill device *c* holds blocks
  *c, c+sp, c+2sp, …* (`block_cyclic_reorder`), so physical order is not global token order. Every
  host-side check un-rotates first (`reverse_reorder_tensor_chunks`). Whoever consumes the migrated
  bytes must know the same permutation.
- **`PREFILL_CHUNK_OUTPUT_TOKENS = 5 * 1024` is baked into the table** — the address math asserts
  `seq_len % 5120 == 0` and strides by `5120 / sp = 640` tokens. So **the disagg ABI currently
  assumes 5k chunks**, which collides with the 8k–25k chunk sizes the throughput curve argues for.
  Changing chunk size is an ABI change, not a tuning knob. Worth knowing before picking a chunk size.

**The concrete gap:** `tests/test_kv_cache_table.py` has per-variant table tests for kimi, glm_5_1
and glm_5_2 — the GLM ones verify the physical→global mapping via `blockcyclic_positions` — and
**there is no Mistral row**. That test is the ABI conformance check, and it is where the 320-wide row
and the 256 latent would be exercised. Adding `test_mistral4_kv_cache_table` is the next step here;
it needs no golden trace and no decode stack.

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
- **Changing the expert dtype needs a different cache DIRECTORY, not just the env var.** The TTNN
  cache path is `{name}_{arch}_{N}dev/{sp}x{tp}` — it encodes nothing about dtype. `as_tensor` does
  stamp dtype into each *filename* (`..._dtype_BFLOAT8_B_layout_TILE.tensorbin`), so the two dtypes
  can coexist in one directory, but the expert weights roughly double in size going bf4 -> bf8
  (~65 GB -> ~120 GB). Use `TT_PREFILL_ROUTED_EXPERT_WEIGHTS_DTYPE` with a separate
  `TT_MISTRAL4_PREFILL_TTNN_CACHE`, and expect a full cold build.
  `TtRoutedExpert.check_cache_complete` is dtype-aware as of 2026-08-18 — before that its glob was
  `local_{i}_{proj}*.tensorbin`, which matched a bf4 file for a bf8 request, reported the cache
  complete, skipped the weight load, and then dumped the **empty placeholder** as the expert
  weights. Random experts still emit plausible text and a PCC number, so nothing announces it. The
  same trap was already fixed for `TtIndexer`; check any new `check_cache_complete` you write
  against `test_routed_expert_check_cache_complete_is_dtype_aware`.
- **After any hang, `tt-smi -r` before the next run.** A wedged ethernet core makes the *next*
  person's run fail with a misleading timeout instead of their real error.

## Open

- `(8,1)` vs `(4,2)` for a PP=4 stage on one galaxy — compute vs CCL. Block correctness is
  split-independent, so this blocks nothing here. Two of the three unknowns are now closed
  (2026-08-18):

  **Capacity is a non-issue.** `PP=4 / TP=1 / CP=8` needs **4.91 GiB/chip** of weights against
  31.9 available — *less* than today's `SP=8 x TP=4` at 5.41, because the TP=1 dense-replication
  penalty (x4) is exactly cancelled by holding 9 layers instead of 36 (/4), and the expert term is
  identical either way (16 experts x 9 layers = 144 expert-layers/chip, same as 4 x 36). bf8 experts
  fit too (6.60), and a 1M-token KV cache is 2.81 GiB/chip. DRAM was never the constraint — L1 is.

  **All of the CCL is on the TP axis**, so TP=1 deletes all of it. An AST audit of the 17 CCL call
  sites under `tt/`: 14 explicitly TP (10 `self.tp_axis`, 4 literal `1`); `tt_reduce` and
  `tt_distributed_rms_norm` both default `cluster_axis=1` and `tt_reduce` self-disables via
  `mesh_device.shape[cluster_axis] > 1`; and the single `sp_axis` site (`tt/mla/mla.py:1832`) is
  gated on `_has_indexer`, i.e. sparse-DSA only, so it never fires for dense Mistral. Measured, that
  is 26.8% of device time (30.8% counting the distributed-LayerNorm halves, which revert to a plain
  LayerNorm at TP=1). Per-chip matmul FLOPs are unchanged (4x the work per layer, a quarter of the
  layers) and `RingJointSDPA` (9.1%) survives because CP stays 8.

  **What is left is pipeline bubbles**, now the only open PP question: 4 stages want >=4 chunks in
  flight. 8k–25k chunks over 256k gives 10–32 chunks (fine); a short prompt gives 1–3, where PP is
  strictly worse. That couples the PP decision to the chunk-size decision.
- KV cache dtype/layout for the prefill↔decode ABI, given `kv_lora_rank = 256`. **Written down
  2026-08-18 — dtype and layout already agree; the slot organisation and the chunk-size constant are
  what do not.** See "The KV cache: what we store and why" below.
- Vision scope — is text-only acceptable for v1?
- ~~No Mistral golden trace exists.~~ **Resolved 2026-08-18.** Every other resident's golden is a
  recorded vLLM trace under `/mnt/models/deepseek-prefill-cache/golden/`; Mistral's is *generated*
  on the host instead, by `tt/runners/generate_prompt_trace.py`, which runs the layer-by-layer CPU
  reference and writes the same layout. It was already there for the runner's KV validation — it
  needed the KVPE fix (it built its golden from `ref_kvpe_list`, so for Mistral it would have
  written a 128-wide row with an empty `pe` half), plus `hidden_states/` + `n_layers` for the
  transformer test, and the adapter's config loader rather than `AutoConfig`.

  ```
  python -m models.demos.deepseek_v3_d_p.tt.runners.generate_prompt_trace --model mistral_small4 \
      --prompt-file models/demos/deepseek_v3_d_p/demo/test_prompt_25k.json \
      --isl 15360 --num-layers 36 --out <dir>          # ~30 s/layer, no device, 5 GB
  ```

  A 36-layer / 15,360-token (3 x CHUNK) golden is at
  `/data/kmabee/mistral4_golden_traces/mistral4_15360_36L`. Point `PREFILL_TRACE_DIR` at it;
  `prefill_trace_default` stays empty rather than hard-coding a personal path on a shared branch.
  `test_mistral4_prefill_transformer_chunked` (L1 / L10 / L36) reads it.

## Who to ask

Per-area owners for MLA, MoE and the prefill runner are listed in the internal bring-up channel;
ask there rather than in this file, so the routing stays current as people move around.
