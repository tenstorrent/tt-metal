---
name: forge-functional-decoder
description: Bring up a functionally correct TTNN decoder layer for the model-specific autoport directory by TRANSLATING a tt-forge / tt-torch codegen-emitted TTNN model into the functional-decoder output shape. Use this instead of $functional-decoder when a forge emit already exists and you must adapt it (not author from scratch). Scope for this pass is PREFILL only; decode is a documented stub until an emitted decode version is provided.
---

# Forge → Functional Decoder Bringup

## Mission Context

If this skill is used as part of `$model-bringup`, follow that skill's mission, workspace, and
reporting contract. This stage is a **drop-in replacement for the `functional-decoder` stage**: it
produces the same artifacts under `models/autoports/<model>/`, but instead of hand-authoring a TTNN
decoder from the HF architecture, you **translate an existing tt-forge / tt-torch codegen emit** into
that shape. The emit is a *starting point*; downstream stages (optimized, multichip, full-model,
datatype-sweep, vLLM, release) build on what you leave here, exactly as they would after a normal
functional-decoder stage.

**Scope for this pass is PREFILL only.** The forge emit provided is a static-shape prefill graph with
no decode / no incremental-KV path. Deliver a correct, tested `prefill_forward`. Implement
`decode_forward` as an explicit, documented `NotImplementedError` stub that points at the pending
emitted-decode version — do **not** fabricate a decode/paged path. Record decode as a known limitation
in the README and the context contract. A future emitted-decode version will be dropped in and tested
in a later pass.

## Your Part

Implement:

```text
models/autoports/<model>/tt/functional_decoder.py
```

Derive `<model>` from the HF model id: lowercase it and replace every non-alphanumeric character with
an underscore (e.g. `Qwen/Qwen3-4B` → `qwen_qwen3_4b`). Downstream tooling resolves the autoport dir
from the HF model id, so add no extra qualifiers.

Expose a model-appropriate `FunctionalDecoder` that subclasses
`models.common.lightweightmodule.LightweightModule` with a real weight-loading boundary:

```python
class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...   # real, tested
    def decode_forward(self, ...): ...    # documented NotImplementedError stub for now
```

Also produce the full artifact set (same as the functional-decoder stage):

```text
models/autoports/<model>/tt/__init__.py
models/autoports/<model>/tests/__init__.py
models/autoports/<model>/tests/test_functional_decoder.py     # prefill PCC vs HF, incl. >=1 real-weight test
models/autoports/<model>/doc/functional_decoder/README.md
models/autoports/<model>/doc/functional_decoder/work_log.md
models/autoports/<model>/doc/functional_decoder/forge_sharding_recommendations.json  # emitted per-op layout, for the optimize stage to seed from
models/autoports/<model>/doc/context_contract.json
```

## The Forge Emit (your input)

The emit lives under `FORGE_DIR` (passed to you), for Qwen3-4B:

```text
FORGE_DIR/model/graph_0/
  model_ttnn.py   # THE reference: ModelTTNN -> Qwen3DecoderLayer -> Qwen3Attention / Qwen3MLP (36 layers)
  params.py       # loads HF weights live and maps FX-mangled names -> HF state-dict keys (_state_dict_key)
  consteval.py    # QKV fusion, RoPE cos/sin tables, causal mask (all baked for seq_len=16)
  model_pt.py     # torch reference (HF Qwen3-4B) + tokenizer input
  utils.py        # calculate_pcc, load_tensor helpers
  main.py         # end-to-end harness (open 1x1 mesh, run, PCC vs model_pt)
```

Read `model_ttnn.py` first — it is already modularized into clean per-layer classes (a post-codegen
refactor). It is your semantic ground truth. **Do not** copy its hand-tuned seq-specialized shard
specs, `MatmulMultiCoreReuseMultiCast1DProgramConfig`s, or per-core grids **into the runtime
forward** — those are static-shape layout glue that will not generalize. Reimplement the *math* on
simple `ttnn.bfloat16` / `ttnn.TILE_LAYOUT` / `ttnn.DRAM_MEMORY_CONFIG` defaults (the
functional-decoder correctness-first convention), so the layer accepts an arbitrary `seq_len`. One
refinement: the emit's compute-kernel precision (`math_fidelity`, `fp32_dest_acc_en`, `packer_l1_acc`) is
seq-independent, so keep the runtime on defaults but adopt the emit's setting for a numerically sensitive
op (norms, etc.) when prefill PCC is at or below the bar — a correctness fix, not an optimization.

**But do not throw the layout knowledge away.** The emit's layouts, shard widths, program configs,
fusions, and precision are the best seed the downstream `optimize` stage has; discarding them makes it
rediscover sharding blind and re-lose wins the emit already had. Record them verbatim into
`doc/functional_decoder/forge_sharding_recommendations.json` (schema below) — provenance, not runtime
code, so it does not affect arbitrary-`seq_len` correctness.

**Preserve the emitted workload batch size.** The emit is generated for a specific batch (read it
from `model_pt.BATCH_SIZE`, e.g. **32** for Llama-3.1-8B). That batch is part of the workload the
downstream stages (optimize, full-model, release) are expected to serve, so **do not silently
retarget to batch-1**. Batch is a runtime/workload dimension, *not* static layout glue — keep it. The
thing you drop is the seq=16 *layout* specialization (shard specs, program configs, per-core grids),
not the batch. Parameterize the layer by `batch` (defaulting to the emitted `BATCH_SIZE`) and by
`seq_len`, so it accepts arbitrary values of both. If you have a documented reason to also support
batch-1 (single-user latency), add it as an *additional* supported value — do not make it the sole
target and do not lose the emitted batch.

### Forge sharding recommendations (`forge_sharding_recommendations.json`)

Capture, per emitted op, what *your* emit did, so `optimize` can seed from it instead of a blank DRAM
slate. Record as much as the emit specifies — completeness is the goal; `optimize` decides how much to
use. The skeleton below is the field set with placeholder `<...>` values, not real data: fill every
field from your own emit rather than copying the example.

```json
{
  "provenance": {
    "emit": "<path to the model_ttnn.py you translated>",
    "emitted_batch": "<emitted BATCH_SIZE>",
    "emitted_seq_len": "<emitted seq_len>",
    "emitted_grid": "<the emit's program grid, copied raw>",
    "note": "Grids/shard-core counts are verbatim from the emit's authoring device and MUST be legalized to the optimize target's real compute grid; only the intent (layout class, shard-width fraction, weights layout, fused activation, compute-kernel precision) ports as-is."
  },
  "ops": {
    "<op_role>": {
      "memory_layout": "<L1_WIDTH_SHARDED | L1_HEIGHT_SHARDED | L1_INTERLEAVED | DRAM>",
      "shard_shape": "<[rows, cols] from the emit, or null>",
      "shard_orientation": "<ROW_MAJOR | COL_MAJOR, or null>",
      "core_count": "<int from the emit, or null>",
      "output_dtype": "<the emit's output dtype for this op, e.g. BFLOAT16>",
      "program_config": "<the ENTIRE emitted program config, all fields, serialized verbatim; keep the class name; null if the emit passes none>",
      "weights_layout": "<interleaved | dram_width_sharded (matmuls only)>",
      "fused_activation": "<activation fused into this op, or null>",
      "compute_kernel": "<the ENTIRE emitted compute-kernel config, all fields verbatim, or null>"
    }
  }
}
```

One entry per op the emit lays out, using this model's actual op set; use `null` for defaults, and record
ops the emit deliberately ran in DRAM as `DRAM` rather than omitting them. Do not invent ops or values.
For `program_config` and `compute_kernel`, serialize the complete emitted object — every field of
whatever class it is (matmul, DRAM-sharded, SDPA, compute-kernel, etc.) — via the helpers in
`models/common/tensor_utils.py`, not a hand-picked subset. Copy grids and shard specs verbatim too;
legalizing them to the target device is `optimize`'s job, not yours.

### Per-layer semantics (extracted from `Qwen3DecoderLayer` / `Qwen3Attention` / `Qwen3MLP`)

Decoder block (`Qwen3DecoderLayer.forward(hidden_states, residual, cos, sin)`):

1. `x = rms_norm(hidden_states, eps=1e-6, weight=input_layernorm_weight)`
2. `attn_out, v, k = attention(x, cos, sin)`
3. `attn_residual = attn_out + residual`
4. `y = rms_norm(attn_residual, eps=1e-6, weight=post_attention_layernorm_weight)`
5. `mlp_out = mlp(y)`
6. `out = mlp_out + attn_residual`

Note the residual convention: the residual passed *in* is added to the attention output (not the
pre-norm hidden). In a standalone single-layer decoder, the input hidden state IS the residual, so a
clean equivalent is: `h = x; a = attn(norm1(h)); h = h + a; m = mlp(norm2(h)); h = h + m`.

Attention (`Qwen3Attention.forward`), GQA with per-head QK-norm and RoPE:

- One **fused QKV** matmul → output width 6144, sliced in **`[V, Q, K]`** order:
  `V = [:, 0:1024]`, `Q = [:, 1024:5120]` (4096 wide), `K = [:, 5120:6144]`.
  (The fused weight is built by consteval as `concat([V^T, Q^T, K^T], dim=1)`.)
- In the reshapes below the leading `1` is the **batch axis**; carry the emitted `batch`
  (e.g. `[batch, seq, 8, 128]`), do not hard-code it to 1.
- `V`: reshape `[1, seq, 8, 128]` → permute `[0,2,1,3]` → `[1, 8, seq, 128]`. No norm, no RoPE.
- `K`: reshape `[1, seq, 8, 128]` → `rms_norm(eps=1e-6, weight=k_norm_weight)` over head_dim →
  permute `[0,2,1,3]` → `rotary_embedding(cos, sin)`.
- `Q`: reshape `[1, seq, 32, 128]` → `rms_norm(eps=1e-6, weight=q_norm_weight)` over head_dim →
  permute `[0,2,1,3]` → `rotary_embedding(cos, sin)`.
- `attn = scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask, is_causal=False,
  scale=1/sqrt(128) ≈ 0.0883883)`. (You may instead pass `is_causal=True` and omit the mask, which
  generalizes across seq_len cleanly — verify PCC either way.)
- `concatenate_heads` → reshape `[seq, 4096]` → `o_proj` matmul (the emit uses `transpose_b=True`
  because `o_proj.weight` is the raw HF `[hidden, num_heads*head_dim]` layout).

Shapes/config (Qwen3-4B; confirm from `AutoConfig`): hidden 2560, 36 layers, 32 Q heads, 8 KV heads
(GQA), head_dim 128, intermediate ≈ 9728, RMSNorm eps 1e-6, `rope_theta` per config,
`max_position_embeddings` = 40960, vocab 151936. **Batch** comes from the emit
(`model_pt.BATCH_SIZE`), not from config — read and preserve it.

MLP (`Qwen3MLP.forward`) — SwiGLU: `down_proj( silu(gate_proj(x)) * up_proj(x) )`.

RoPE: cos/sin are built in `consteval._build_rotary_cos_sin_cpu` from `rotary_emb.inv_freq`:
`angles = inv_freq[:,None] * positions[None,:]`, then `concat([angles, angles], dim=-1)`, then
`cos`/`sin`. For arbitrary `seq_len`, rebuild these tables for `positions = arange(seq_len)` (do it at
setup / test-harness boundary, not inside the runtime forward). `ttnn.experimental.rotary_embedding`
consumes them.

### Weight-key mapping

`model_ttnn.py` references weights from a flat dict keyed with FX-mangled names, e.g.
`L__self___model_layers_{i}_self_attn_q_proj.weight`,
`..._self_attn_{k,v,o}_proj.weight`, `..._self_attn_{q,k}_norm_weight`,
`..._mlp_{gate,up,down}_proj.weight`, `..._input_layernorm_weight`,
`..._post_attention_layernorm_weight`. `params.py::_state_dict_key` maps these to canonical HF keys
(`model.layers.{i}.self_attn.q_proj.weight`, etc.) by stripping the `L__self___` prefix, converting a
trailing `_weight`→`.weight`, and rewriting `_`→`.` on the structural path segments.

Your `from_state_dict` should accept a **canonical HF state_dict** keyed
`model.layers.{layer_idx}.*` (and tolerate layer-local `*` and `model.language_model.layers.{i}.*`
forms, as the example autoports do). Inside `from_state_dict`, do all torch→ttnn conversion,
transposes, QKV fusion (if you keep the fused matmul; a plain separate q/k/v is also fine for
correctness), and RoPE-table / mask construction. Keep the runtime `prefill_forward` free of `torch`,
`ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

## How To Approach It

1. Load `AutoConfig` for `HF_MODEL`; confirm the shapes/config above from the real config, not memory.
2. Read `FORGE_DIR/model/graph_0/model_ttnn.py` + `consteval.py` + `params.py` to lock the exact math
   (eps, slice order `[V,Q,K]`, per-head QK-norm, RoPE, SDPA scale, o_proj transpose, SwiGLU) **and
   the emitted batch** (`model_pt.BATCH_SIZE`).
3. Write `FunctionalDecoder` for a single decoder layer on a 1x1 mesh, bf16 / TILE / DRAM defaults,
   parameterized by `seq_len` **and `batch` (default = emitted `BATCH_SIZE`)**.
   `from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, batch=<emitted>)`
   builds all ttnn weights + RoPE tables + mask; `prefill_forward(hidden_states, *, position_cos,
   position_sin, ...)` runs the block and returns the layer output `[1, batch, seq, hidden]` at the
   **emitted batch** (match the example autoports' prefill shape convention; document whatever you pick).
4. Build an HF single-layer reference (`Qwen3DecoderLayer` from `transformers`, or slice layer
   `layer_idx` out of the full model) **at the same batch** and compare prefill PCC.
5. `decode_forward` raises `NotImplementedError("decode path pending emitted-decode forge version")`.

## Evidence To Leave (done criteria — "works for now" bar)

Done means all of these are true and recorded:

- `tt/functional_decoder.py` with `FunctionalDecoder(LightweightModule)`, `from_state_dict`,
  `prefill_forward`, and a documented `decode_forward` stub. `forward(mode=...)` dispatcher optional.
- `tests/test_functional_decoder.py`: at least one **real-weight** single-layer prefill test **at the
  emitted batch** that passes **PCC ≥ 0.99** vs the HF reference layer (the emit itself captures ~0.977
  full-model bf16, so a single bf16 layer should comfortably clear 0.99; aim for ≥ 0.995 if
  achievable). Include a synthetic-weight prefill test at real shapes, and a couple of non-trivial
  `seq_len` values (a small smoke length and one larger length), all at the emitted batch. A
  `decode_forward` test may `pytest.xfail`/`skip` with the pending-decode reason. Include a static
  runtime-fallback audit (grep the runtime method source for `torch`/`from_torch`/`to_torch`).
- `doc/context_contract.json`: records `hf_advertised_context` (= HF `max_position_embeddings`) and
  `current_supported_context` (the largest prefill seq you validated). Because this is a
  forge-seeded prefill-only starting point, `current_supported_context` will be below advertised;
  set `limiting_reason` and evidence honestly (this is a functional starting point, not a DRAM limit —
  state that plainly in a `notes` field; downstream stages extend context). Also record a
  `decode_status: "pending_emitted_decode_version"` field. Keep the JSON parseable by
  `.agents/scripts/check_context_contract.py`.
- `doc/functional_decoder/README.md` + `work_log.md`: the runtime contract (prefill/decode
  signatures + shapes), a validation summary table (PCC per test, real-weight PCC), the forge-emit
  provenance (which files you translated), and an explicit **Limitations** section (prefill-only,
  static-shape origin, decode pending, no paged KV yet, context below advertised).
- A short note that the runtime `prefill_forward` has no torch/host fallback.
- `doc/functional_decoder/forge_sharding_recommendations.json`: the emit's per-op layout captured
  verbatim per the schema above, every emitted op present (including DRAM ones), referenced from the
  README as the seed input for `optimize`.

Watcher / tracy / long-context / paged-KV evidence from the full `functional-decoder` skill is
**not required** for this pass (the emit does not support those yet) — but do not claim them either.
If a quick warmed-prefill timing is easy, include it; otherwise record it as not-yet-measured.

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
- Keep the seq-specialized forge layout glue out of the *runtime*, but capture it in
  `forge_sharding_recommendations.json` for the optimize stage — drop it from the code, not from the record.
- **Preserve the emitted workload batch** (`model_pt.BATCH_SIZE`); drop layout glue, not batch.
- Be honest in the docs and contract about what is and isn't supported this pass.
