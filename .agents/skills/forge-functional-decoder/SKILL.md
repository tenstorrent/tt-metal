---
name: forge-functional-decoder
description: "Bring up a functionally correct TTNN decoder layer for the model-specific autoport directory by TRANSLATING a tt-forge / tt-torch codegen-emitted TTNN model into the functional-decoder output shape. Use this instead of $functional-decoder when a forge emit already exists and you must adapt it (not author from scratch). Scope is emit-driven — translate whichever forward paths the emit actually ships (a prefill graph generalized across seq lens, a single-token decode graph, or both), test each, and provide an honest deferring stub only for a path the emit does not contain."
---

# Forge → Functional Decoder Bringup

## Mission Context

If this skill is used as part of `$model-bringup`, follow that skill's mission, workspace, and
reporting contract. This stage is a drop-in replacement for the `functional-decoder` stage: it
produces the same artifacts under `models/autoports/<model>/`, but instead of hand-authoring a TTNN
decoder from the HF architecture, you translate an existing tt-forge / tt-torch codegen emit into
that shape. The emit is a starting point; downstream stages (optimized, multichip, full-model,
datatype-sweep, vLLM, release) build on what you leave here, exactly as they would after a normal
functional-decoder stage.

**Scope is emit-driven: translate whatever the emit actually contains — do not assume which forward
paths are present.** An emit may ship a prefill graph, a single-token decode graph, or both; different
emits differ. Classify each path from the workload the emit was generated for, not from a fixed list
of op names. Read the harness (`model_pt.py` / `main.py`) and the graph's input/output signature and
ask:

- Does it process a full sequence (emitted seq length > 1, attention over the sequence itself, no
  incremental single-token cache append)? That is a prefill path.
- Does it process a single token step (emitted seq length == 1, a persistent KV cache passed in as an
  input plus a cache position/index, and the graph appends to that cache in place)? That is a decode
  path.

Signals to read: the emitted input shapes (`input_ids` / hidden seq dim, tokens-per-step), whether a
`past_key_values` / cache is a persistent input, and whether `cache_position` / an update index is
threaded through. The specific attention/cache ops the graph uses are useful corroboration, but derive
the classification from the workload, not from matching op names. Translate each path the emit ships,
independently; the presence or absence of one path implies nothing about the other.

For each forward path the emit contains:

- Translate it into a correct, tested method (`prefill_forward` / `decode_forward`). It is semantic
  ground truth — capturing it is the whole point of a forge-seeded bringup, so do not leave it for the
  optimize stage to rediscover blind. Keep the runtime on correctness-first defaults; the emit's device-
  and shape-specialized parts are handled as described under "The Forge Emit" below (layout kept out of
  the runtime, precision normalized to bf16 at load), not smuggled into the forward.

For each forward path the emit does not contain:

- Implement it as an explicit, documented `NotImplementedError` stub whose message states the emit did
  not ship that graph and the path is delivered by a later stage. Do not fabricate it, and do not claim
  it is "pending" — nothing more is coming from forge for this model.

Record which paths were present (and the workload-signature evidence for the classification) in the
README and context contract.

## Your Part

Implement:

```text
models/autoports/<model>/tt/functional_decoder.py
```

Derive `<model>` from the HF model id: lowercase it and replace every non-alphanumeric character with
an underscore (e.g. `Org/Model-Name` → `org_model_name`). Downstream tooling resolves the autoport dir
from the HF model id, so add no extra qualifiers.

Expose a model-appropriate `FunctionalDecoder` that subclasses
`models.common.lightweightmodule.LightweightModule` with a real weight-loading boundary:

```python
class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...   # real+tested if the emit ships a prefill graph; else honest stub
    def decode_forward(self, ...): ...    # real+tested if the emit ships a decode graph; else honest stub
```

Also produce the full artifact set (same as the functional-decoder stage):

```text
models/autoports/<model>/tt/__init__.py
models/autoports/<model>/tests/__init__.py
models/autoports/<model>/tests/test_functional_decoder.py     # PCC vs HF for each translated path, incl. >=1 real-weight test
models/autoports/<model>/doc/functional_decoder/README.md
models/autoports/<model>/doc/functional_decoder/work_log.md
models/autoports/<model>/doc/context_contract.json
```

## The Forge Emit (your input)

The emit lives under `FORGE_DIR` (passed to you):

```text
FORGE_DIR/model/graph_0/
  model_ttnn.py   # THE reference: ModelTTNN -> per-layer decoder class -> attention / MLP classes
  params.py       # loads HF weights live and maps FX-mangled names -> HF state-dict keys (_state_dict_key)
  consteval.py    # all load-time constant folding (weight fusion/transposes/casts, RoPE tables, mask & index constants, reshapes, ...)
  model_pt.py     # torch reference (the HF model) + tokenizer input, and the emitted workload batch
  utils.py        # calculate_pcc, load_tensor helpers
  main.py         # end-to-end harness (open 1x1 mesh, run, PCC vs model_pt)
```

> **Emit variants.** The tree above is the tt-forge / tt-torch shape. If your input is instead an
> **EmitPy emit** (a flat `main.py` + `consteval.py` package from tt-mlir's TTNN→EmitPy — e.g. the
> `qb2-ttnn-to-emitpy/` packages), it is a full-model, often tensor-parallel graph that is **not**
> modularized into per-layer classes: use **`$forge-functional-decoder-from-ir`**, which documents
> segmenting one layer from the flat graph. And whatever the source, if the emit is **tensor-parallel**
> (weights sharded across a mesh + `all_reduce` / `all_gather` / `reduce_scatter` / `mesh_shard`),
> collapse it to single-device dense math in the runtime — drop the collectives and matmul the **full**
> HF weight (the state_dict is already un-sharded; use the shard/collective structure only to learn each
> projection's parallel axis and fusion order) — and record the distribution facts as
> `doc/functional_decoder/multichip_provenance.json` (mesh axes/sizes; per sharded tensor and collective
> its axis, `cluster_axis`, `reduce_type`, and emitted dtype) for the multichip stage to consume.

Read `model_ttnn.py` first — it is typically modularized into clean per-layer classes (a post-codegen
refactor) and is your semantic ground truth. Start from it and take its math and structure as-is — op
sequence, fusion order, eps, scale, residual wiring — do not re-derive them. The one thing you keep out
of the runtime forward is the emit's static-shape layout specialization (this is the reason it is a
functional stage, not a copy of the emit):

- Layout: keep the emit's compiler-chosen shard specs, matmul program configs, and per-core grids out of the
  runtime forward — they are static-shape glue tied to its target device grid, so a runtime wired to them
  won't take an arbitrary prefill `seq_len` or move to another grid, and layout tuning is the optimize
  stage's job (it derives its own layout from `$shard-advise`). Let each op fall back to its simplest
  working layout — plain `ttnn.bfloat16` / `ttnn.TILE_LAYOUT` / `ttnn.DRAM_MEMORY_CONFIG` for most ops,
  which is also what lets a prefill path accept an arbitrary `seq_len` — except where an op needs
  otherwise: some decode ops (KV-cache update, decode SDPA, decode head-concat) require an L1-sharded
  layout to run at all, so give those the minimal workload-derived layout (e.g. batch/heads), still
  without adopting the emit's full compiler-chosen grid.

Run the ops on framework-default compute-kernel config (`math_fidelity`, `fp32_dest_acc_en`,
`packer_l1_acc`), adopting the emit's setting for a numerically sensitive op (norms, etc.) only if PCC is
at or below the bar — a correctness fix, not an optimization. The emit's load-time weight and constant
preprocessing (`consteval.py`) is reproduced at the load boundary, not the runtime — covered under
Per-layer semantics below.

**Preserve the emitted workload batch.** Read it from `model_pt.BATCH_SIZE` and keep it — it is a
workload dimension downstream stages (optimize, full-model, release) serve, not layout glue to drop.
Parameterize the layer by `batch` (default = emitted) and, for prefill, by `seq_len`, so both accept
arbitrary values.

### Per-layer semantics (read them from the emit — do not assume)

Extract the exact structure from the emit's per-layer decoder / attention / MLP classes and its
`consteval.py`. Do not hardcode shapes, orders, or flags from memory or from another model — read them,
and confirm shape constants against `AutoConfig`.

`consteval.py` is the emit's load-time host preprocessing, so reproduce it in `from_state_dict` (plus
setup-boundary builders for the RoPE/mask tables) — none of it belongs in the runtime forward, which
stays torch-free. Read it in full and keep every step at load:

- Fusion: reproduce exactly — whether projections are fused and the exact slice/concat order, which can
  differ per layer (specifics under Attention below); a wrong order permutes Q/K/V.
- Transposes: pre-transpose the weight at load and keep the matmul `transpose_b=False`; do not defer it
  into a runtime `transpose_b=True`, which would push const-eval work into the forward.
- Dtype casts: recast to bf16 for the correctness baseline, recording the emit's per-weight bf8/bf4
  policy in the README / context contract for datatype-sweep.
- Shape-frozen constants (masks, index/position tables, cache length): rebuild as
  shape/position-parameterized builders at the setup boundary, not the baked tensor.

The block is a chain of residual updates around normed sublayers; do not assume a canonical shape — read
it from the emit's per-layer class. What generalizes: each residual add uses the tensor from before that
update's norms (not a normed one), and in a standalone single-layer decoder the input hidden state is the
first residual. Everything else varies, so take it from the emit: how many norms and where — a sublayer
may be wrapped by both a pre- and a post-norm inside the residual (not just a single pre-norm); whether
the feedforward is one MLP or several parallel branches summed before the residual (e.g. a dense MLP plus
a MoE branch); per-head Q/K/V norms (and a norm may be scale-free); and a possible final per-layer scalar.

Attention (GQA, optionally with per-head Q/K-norm, plus RoPE) — read every concrete value from the emit:

- Whether Q/K/V come from a fused QKV matmul or separate projections, and if fused, the exact slice
  offsets and fusion/concat order the emit's `consteval.py` uses. This order is emit-specific and can
  even differ per layer — reproduce exactly what the emit does, do not assume a canonical order.
- The batch axis in the reshapes: carry the emitted `batch`, do not hard-code it.
- Per-head `rms_norm` on Q and/or K over `head_dim` only if the emit / state-dict actually has
  `q_norm` / `k_norm`.
- RoPE via `ttnn.experimental.rotary_embedding` on Q and K.
- SDPA scale `1 / sqrt(head_dim)`. For prefill you may pass `is_causal=True` and omit an explicit
  causal mask (this generalizes across `seq_len` cleanly) — verify PCC either way.
- `concatenate_heads` → `o_proj` matmul.

Config: read hidden size, layer count, Q/KV head counts, `head_dim`, intermediate size, RMSNorm eps,
`rope_theta`, `max_position_embeddings`, and vocab from `AutoConfig`, not from memory. Batch comes from
the emit (`model_pt.BATCH_SIZE`), not from config — read and preserve it.

MLP — read the emit's form (commonly SwiGLU: `down_proj( silu(gate_proj(x)) * up_proj(x) )`).

RoPE: build cos/sin at the setup / test-harness boundary (not in the runtime forward), parameterized by
position — `arange(seq_len)` for prefill, the current cache position for decode. Source them however is
cleanest and matches PCC: the model's HF rotary module, or the emit's own `inv_freq`-based construction
in `consteval`. `ttnn.experimental.rotary_embedding` consumes them.

### Weight-key mapping

`model_ttnn.py` references weights from a flat dict keyed with FX-mangled names, e.g.
`L__self___model_layers_{i}_self_attn_q_proj.weight`, `..._self_attn_{k,v,o}_proj.weight`,
`..._self_attn_{q,k}_norm_weight`, `..._mlp_{gate,up,down}_proj.weight`, `..._input_layernorm_weight`,
`..._post_attention_layernorm_weight`. `params.py::_state_dict_key` maps these to canonical HF keys
(`model.layers.{i}.self_attn.q_proj.weight`, etc.) by stripping the `L__self___` prefix, converting a
trailing `_weight`→`.weight`, and rewriting `_`→`.` on the structural path segments.

Your `from_state_dict` should accept a canonical HF state_dict keyed `model.layers.{layer_idx}.*` (and
tolerate layer-local `*` and `model.language_model.layers.{i}.*` forms, as the example autoports do).
Inside `from_state_dict`, do all torch→ttnn conversion, transposes, QKV fusion, and RoPE-table / mask
construction. Keep the runtime forwards free of `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host
fallback.

## How To Approach It

1. Load `AutoConfig` for `HF_MODEL`; confirm the shapes/config from the real config, not memory.
2. Read `FORGE_DIR/model/graph_0/model_ttnn.py` + `consteval.py` + `params.py` to lock the exact math
   (eps, the emit's QKV slice/fusion order, per-head QK-norm presence, RoPE, SDPA scale, matmul
   transpose flags / weight-prep factoring, MLP form) and the emitted batch (`model_pt.BATCH_SIZE`).
   Read `consteval.py` in full and account for every constant-folding step it does — the set is
   emit-specific and may go beyond the usual weight fusion/transposes/casts (e.g. per-layer fusion
   variants, slice/reorder steps, `inv_freq` reshapes, mask/index constants); do not assume a fixed list.
3. Classify which forward graphs the emit ships from its workload signature (per Mission Context): read
   the harness (`model_pt.py` / `main.py`) and graph I/O — emitted seq length, persistent KV-cache
   input, `cache_position` / update index — to decide prefill (full sequence, no incremental append) vs
   decode (single token, appends to a passed-in cache). Then read the math of each path present (for
   decode: KV-cache append at the cache position, single-token attention over the cache, RoPE-at-position,
   the emitted decode batch).
4. Write `FunctionalDecoder` for a single decoder layer on a 1x1 mesh, bf16 / TILE / DRAM defaults,
   parameterized by `batch` (default = emitted value) and, for prefill, `seq_len`. `from_state_dict`
   builds all ttnn weights + RoPE tables + mask; each translated forward runs the block and returns the
   layer output at the emitted batch (match the example autoports' shape convention; document what you
   pick).
5. Build an HF single-layer reference (the model's decoder layer class from `transformers`, or slice
   `layer_idx` out of the full model) at the same batch and compare PCC for each translated path.
6. For any path the emit does not ship, `raise NotImplementedError` with a message stating the emit did
   not contain that graph and it is delivered by a later stage.

## Evidence To Leave (done criteria — "works for now" bar)

Done means all of these are true and recorded:

- `tt/functional_decoder.py` with `FunctionalDecoder(LightweightModule)`, `from_state_dict`, and a
  `prefill_forward` / `decode_forward` that is translated-and-tested for each path the emit ships and a
  documented stub for each it does not. `forward(mode=...)` dispatcher optional.
- `tests/test_functional_decoder.py`: for each translated path, at least one real-weight single-layer
  test at the emitted batch that passes PCC ≥ 0.99 vs the HF reference layer (a single bf16 layer should
  comfortably clear this; aim for ≥ 0.995 if achievable). For a prefill path, include a synthetic-weight
  test at real shapes and a couple of non-trivial `seq_len` values (a small smoke length and one larger
  length), all at the emitted batch. For a decode path, include a real decode PCC test vs a single HF
  decode step at the emitted batch. A path the emit does not ship may `pytest.xfail` / `skip` with the
  not-emitted reason. Include a static runtime-fallback audit (grep the runtime method source for
  `torch` / `from_torch` / `to_torch`).
- `doc/context_contract.json`: records `hf_advertised_context` (= HF `max_position_embeddings`) and
  `current_supported_context` (the largest prefill seq you validated, or the decode cache length if the
  emit is decode-only). If this forge-seeded starting point supports less than advertised, set
  `limiting_reason` and evidence honestly (state plainly in a `notes` field whether it is a functional
  starting point vs a DRAM limit; downstream stages extend context). Also record which forward paths
  were translated vs stubbed and why (a `decode_status` / `paths` style field), reflecting the
  classification evidence. Keep the JSON parseable by `.agents/scripts/check_context_contract.py`.
- `doc/functional_decoder/README.md` + `work_log.md`: the runtime contract (translated forward
  signatures + shapes), a validation summary table (PCC per test, real-weight PCC), the forge-emit
  provenance (which files you translated, and the path-classification result), and a Limitations section
  (static-shape origin, any context below advertised, and which forward paths were stubbed and deferred).
- A short note that the runtime forwards have no torch/host fallback.

Watcher / tracy / long-context / paged-KV evidence from the full `functional-decoder` skill is not
required for this pass — but do not claim it either. If a quick warmed timing is easy, include it;
otherwise record it as not-yet-measured.

## Reuse Pointers

- `models/common/lightweightmodule.py` — base class.
- `models/common/modules/attention/attention_1d.py`, `.../rope/rope_1d.py`,
  `.../rmsnorm/rmsnorm_1d.py` — clean config-first patterns if you want cleaner building blocks than
  the raw emit.
- The `functional-decoder` skill (`.agents/skills/functional-decoder/SKILL.md`) — the full contract
  this stage is a scoped subset of; consult it for TTNN correctness defaults and the artifact format.
- Existing example autoports (for shape/format of `functional_decoder.py`, tests, README, contract):
  any completed `models/autoports/*/` in the agentic-research project.

## Code Quality Defaults

- Prefer explicit model contracts over permissive fallback logic.
- Do not silently infer or patch invalid config values; fail directly.
- Keep the emit's shape-specialized layout out of the runtime entirely; run its math on simple `bf16` /
  `TILE` / `DRAM` defaults.
- Preserve the emitted workload batch (`model_pt.BATCH_SIZE`); drop layout glue, not batch.
- Be honest in the docs and contract about which paths are translated, stubbed, and supported this pass.
