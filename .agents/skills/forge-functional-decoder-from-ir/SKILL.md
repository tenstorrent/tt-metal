---
name: forge-functional-decoder-from-ir
description: "Bring up a functionally correct TTNN decoder layer for the model-specific autoport directory by TRANSLATING a compiler-emitted TTNN IR graph (a .mlir dump from tt-xla / tt-mlir) into the functional-decoder output shape. Use this instead of $functional-decoder or $forge-functional-decoder when your starting point is a TTNN IR dump (not a tt-forge python emit). Scope is graph-driven — translate whichever forward paths the IR actually ships (a prefill graph generalized across seq lens, a single-token decode graph, or both), test each, and provide an honest deferring stub only for a path the IR does not contain."
---

# TTNN IR → Functional Decoder Bringup

## Mission Context

If this skill is used as part of `$model-bringup`, follow that skill's mission, workspace, and
reporting contract. This stage is a drop-in replacement for the `functional-decoder` stage and a
sibling of `$forge-functional-decoder`: it produces the same artifacts under
`models/autoports/<model>/`, but instead of translating a tt-forge python emit it translates a
**compiler-emitted TTNN IR graph** — a `.mlir` dump (e.g. a tt-xla nightly `ttnn-mlir-*` artifact).
Downstream stages (optimized, multichip, full-model, datatype-sweep, vLLM, release) build on what you
leave here, exactly as they would after a normal functional-decoder stage.

Read `$forge-functional-decoder`'s SKILL.md once: **every rule there about what a functional decoder
is** — single decoder layer on a 1x1 mesh, bf16 / TILE / DRAM correctness-first defaults, layout
specialization kept out of the runtime, torch-free runtime forwards, preserve the emitted workload
batch, honest stubs for absent paths, and the full artifact/evidence contract — **applies here
unchanged.** This document only replaces the *input* (a TTNN IR graph instead of a python emit) and
adds the two things that are genuinely different about reading an IR graph: it is **flat** (not
modularized per layer) and it may be **tensor-parallel** (multi-device collectives + sharded weights).
Do not re-derive the shared contract; take it from the forge skill.

**Scope is graph-driven: translate whatever the IR actually contains — do not assume which forward
paths are present.** A model's IR dump usually ships several graphs; classify each from its workload,
not from a fixed list of op names:

- Fills the whole KV cache over a full sequence (`ttnn.fill_cache`, emitted seq length > 1, attention
  over the sequence itself, no incremental single-token append)? That is a **prefill** path.
- Processes a single token step (emitted seq length == 1, a persistent KV cache passed in as an input
  plus a cache/page position, and the graph appends in place — `ttnn.paged_update_cache` /
  `ttnn.update_cache` and a decode SDPA such as `scaled_dot_product_attention_decode`)? That is a
  **decode** path.

`scripts/classify_graphs.sh <ir-dir>` reports these signals for every `.mlir` in a directory (fill
vs paged-update vs decode-SDPA counts, whether the graph also returns full-vocab logits, and whether
it is a `runtime` dump). Prefer the **compiler, non-runtime** graph for each path (the `runtime`
variant is the same math wrapped in trace/execute plumbing). Whether a graph additionally returns raw
logits does not change the decoder-layer math — pick either; note which you used. Translate each path
the IR ships, independently; the presence or absence of one path implies nothing about the other.

For each forward path present: translate it into a correct, tested method (`prefill_forward` /
`decode_forward`) — it is semantic ground truth, so do not defer it to the optimize stage. For each
path absent: an explicit, documented `NotImplementedError` stub stating the IR did not contain that
graph and the path is delivered by a later stage. Record which paths were present and the
workload-signature evidence for the classification in the README and context contract.

## Your Part

Implement `models/autoports/<model>/tt/functional_decoder.py` and the full artifact set — **identical
to the `$forge-functional-decoder` "Your Part" and artifact list** (`tt/__init__.py`,
`tests/__init__.py`, `tests/test_functional_decoder.py`, `doc/functional_decoder/README.md`,
`doc/functional_decoder/work_log.md`, `doc/context_contract.json`). Same class contract:

```python
class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...
    def prefill_forward(self, ...): ...   # real+tested if the IR ships a prefill graph; else honest stub
    def decode_forward(self, ...): ...    # real+tested if the IR ships a decode graph; else honest stub
```

Derive `<model>` from the HF model id exactly as the forge skill says (lowercase, non-alphanumerics
→ `_`). Match the shape conventions of the example autoports (`hidden_states = [1, batch, seq,
hidden]` in and out; document what you pick).

## The TTNN IR (your input)

Your input is a directory `IR_DIR` of one or more `.mlir` graphs for a single model (plus the HF model
id `HF_MODEL`). The graphs are **full-model** TTNN dialect dumps (all decoder layers inlined), device-
and shape-specialized for the workload they were captured on — often **tensor-parallel across a
multi-chip mesh**.

**Step 0 — get a readable emit.** MLIR is readable, but the fastest path to the op sequence, eps,
activations, and weight names is to lower each graph to a flat ttnn python emit — the IR analogue of
the forge `model_ttnn.py`:

```bash
scripts/ir_to_emit.sh <IR_DIR>/<prefill-graph>.mlir /tmp/<model>/prefill
scripts/ir_to_emit.sh <IR_DIR>/<decode-graph>.mlir  /tmp/<model>/decode
```

This runs tt-mlir's `--ttnn-common-to-emitpy-pipeline` + `ttmlir-translate --mlir-to-python` (with the
known schema-skew normalizations baked in; if the converter reports a new `failed to satisfy
constraint: NN-bit {signed,signless} integer attribute` error, add a scoped sed for that op in
`ir_to_emit.sh` — it is an attribute-signedness skew, not a semantic change; note the direction is
build-vs-producer dependent, so only add a sed when the *current* build actually rejects the attribute).
Read **both** the emit (`.py`, for op semantics + weight names) and the raw `.mlir` (for exact tensor
shapes and layouts). The emit's weights are synthetic `ttnn.ones` — it is a structure/math reference
only; real weights are loaded from HF in `from_state_dict`.

**Pre-generated emit shortcut.** If a runnable EmitPy emit for this model already exists, read it directly
and **skip `ir_to_emit.sh`** (e.g. the `qb2-ttnn-to-emitpy/<model>/` packages: a `g0_prefill/` and a
`g1_decode/`, each self-contained with `main.py` = flat emitted forward + synthetic `ttnn.ones` weights,
`consteval.py` = weight-prep, `utils.py` = PCC/`load_tensor` helpers, `ttir_cpu.py` = cpu-hoisted-op impls).
`g0` = prefill (`fill_cache`), `g1` = decode (`paged_update_cache` + decode SDPA). It is the same
full-model, often tensor-parallel graph the converter would produce, so everything below applies unchanged.

The emit differs from a forge `model_ttnn.py` in two ways you must handle. Everything else about the
translation — dropping layout specialization, correctness-first defaults, per-layer semantics read
from the source, load-time weight prep — is **the same as the forge skill; follow it.**

### Difference 1 — the graph is FLAT: segment the repeating decoder layer

The emit is one long function (a `trace_*` / `_main`-style body), not per-layer classes. You must find
the single repeating decoder block and translate exactly one layer:

- The block repeats `num_hidden_layers` times (confirm the count against `AutoConfig`). Delimit layers
  by the norm sites — a decoder layer has a fixed, small number of `ttnn.rms_norm` / `ttnn.layer_norm`
  calls (e.g. an `input_layernorm` and a `post_attention_layernorm`); the count between two identical
  markers is one layer. The trailing final-norm + lm_head (and any `ttnn.argmax` / sampling) are **not**
  part of the layer — exclude them.
- Confirm the boundary with the **weight names**: `create_weights_for__main()` in the emit maps each
  `arg_N` to a weight key carrying its layer index (see Weight-key mapping). All args whose key is
  `...layers.{i}...` belong to layer `i`; translate one representative `i` (a middle layer, not 0 or
  the last, to avoid first/last-layer special-casing).
- Cross-check the op multiset per layer against the HF architecture (norms, Q/K/V/O projections, MLP
  branches). Read the layer's math from the emit — do not assume a canonical block shape; consult the
  forge skill's "Per-layer semantics" for the full list of things that vary (norm placement, fused vs
  separate QKV, per-head QK-norm, MLP form, RoPE, SDPA scale).

### Difference 2 — the graph may be TENSOR-PARALLEL: collapse to single-device dense math

A multi-chip capture shards weights across devices and inserts collectives (`ttnn.all_reduce`,
`ttnn.all_gather`, `ttnn.reduce_scatter`, `ttnn.mesh_shard`, `ttnn.collective_permute`). Your
functional decoder runs a **single dense layer on a 1x1 mesh**, so translate to the mathematically
equivalent un-sharded computation — this is the IR analogue of the forge skill's "keep the emit's
static-shape layout specialization out of the runtime":

- **Collectives collapse.** A tensor-parallel linear is `[local matmul on a weight shard] -> all_reduce
  (sum) / all_gather (concat)`; on one device the equivalent is the single dense matmul against the
  **full** weight. Drop `all_reduce` / `all_gather` / `reduce_scatter` / `mesh_shard` from the runtime
  and use the full-width weight. Verify by PCC against the HF reference, which is dense — a correct
  collapse matches; a wrong one (e.g. treating an all_gather concat as a sum) will not.
- **Recombine sharded weights at load.** The per-device weight args are slices of the HF weight along
  the TP-sharded axis (columns for column-parallel Q/K/V/gate/up, rows for row-parallel O/down). In
  `from_state_dict` you load the **canonical full HF weight** (not the shards) — the state_dict already
  has the un-sharded tensor — so you generally do not reconstruct shards at all; you just skip the
  collective and matmul the full weight. Use the shard/collective structure only to *understand* which
  axis each projection is parallel over and to get the fusion/slice order right.
- Determine the mesh/TP degree from the raw `.mlir` (mesh shape attribute, `all_reduce` cluster axis,
  shard specs) and record it as provenance, but keep it out of the   runtime forward. Capture it
  structured and complete, not just prose — write `doc/functional_decoder/multichip_provenance.json`
  recording the mesh (axis names + sizes) and **every** sharded tensor and collective the layer's
  graph contains — the tensors named here are examples, not a closed set: per tensor (weights, biases,
  norm/QK-norm/RoPE tables, KV cache, and any MoE router/expert tensors) its parallel kind (tensor
  column/row, sequence, expert, data, or replicated — possibly across more than one mesh axis on a
  2D/ND mesh), its shard axis(es) + mesh dim(s); the boundary activation/residual shardings; and every
  collective (`all_reduce`/`all_gather`/`reduce_scatter`/`all_to_all`/`mesh_shard`/`collective_permute`/…)
  with its cluster/mesh axis. The `$multichip` stage reads this as a known-correct sharding prior. `head_dim`, head counts, and intermediate size come from `AutoConfig` at full
  (un-sharded) width — the per-device shapes in the graph are `full / tp_degree`; do not bake the
  per-device shape into the layer.

## Weight-key mapping

`create_weights_for__main()` in the emit maps each `arg_N` (a `ttnn.ones` placeholder) to a weight key,
but the key is FX/parametrization-mangled, e.g.
`"model.model.layers.0.self_attn.v_proj.parametrizations.weight.original"`. Normalize to a canonical HF
key: strip a leading duplicated `model.` prefix, and rewrite a
`.parametrizations.<name>.original` tail back to `.<name>` (so the example becomes
`model.layers.0.self_attn.v_proj.weight`). Your `from_state_dict` accepts a canonical HF state_dict
keyed `model.layers.{layer_idx}.*` and tolerates the layer-local `*` and
`model.language_model.layers.{i}.*` forms, exactly as the example autoports do. Do all torch→ttnn
conversion, transposes, QKV fusion, and RoPE-table / mask construction inside `from_state_dict`; keep
the runtime forwards free of `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

## How To Approach It

1. Load `AutoConfig` for `HF_MODEL`; confirm hidden size, layer count, Q/KV head counts, `head_dim`,
   intermediate size, RMSNorm eps, `rope_theta`, `max_position_embeddings`, vocab — from the real
   config, not memory.
2. `scripts/classify_graphs.sh IR_DIR` → decide which graphs are prefill / decode; pick the compiler
   (non-runtime) graph for each present path. Note the emitted batch and seq from the graph I/O
   (raw `.mlir` `func.func @main` signature) and preserve the batch downstream.
3. `scripts/ir_to_emit.sh` each chosen graph → flat python emit. Read the emit for op sequence, eps,
   activations (e.g. fused `SILU` on a gate matmul ⇒ SwiGLU), fusion/slice order, SDPA scale, matmul
   transpose flags; read the raw `.mlir` for exact shapes. Segment one representative decoder layer
   (Difference 1). Identify the TP structure and plan the collapse (Difference 2).
4. Write `FunctionalDecoder` for a single decoder layer on a 1x1 mesh, bf16 / TILE / DRAM defaults,
   parameterized by `batch` (default = emitted) and, for prefill, `seq_len`. `from_state_dict` builds
   all ttnn weights (full un-sharded HF weights, pre-transposed, QKV-fused in the emit's order) + RoPE
   tables + mask. Each translated forward runs the block on simple layouts and returns the layer output
   at the emitted batch. Decode ops that require it (KV-cache update, decode SDPA, head concat) get the
   minimal workload-derived L1 layout only — never the emit's full compiler-chosen grid.
5. Build an HF single-layer reference (the model's decoder layer class from `transformers`, or slice
   `layer_idx` out of the full model) at the same batch and compare PCC for each translated path.
6. For any path the IR does not ship, `raise NotImplementedError` with the not-emitted reason.

## Evidence To Leave (done criteria)

**Identical to `$forge-functional-decoder`'s "Evidence To Leave"** — do not restate a weaker bar here.
In summary: `tt/functional_decoder.py` with the class contract and translated-and-tested forwards for
each path present + honest stubs for absent ones; `tests/test_functional_decoder.py` with, per
translated path, ≥1 real-weight single-layer test at the emitted batch passing PCC ≥ 0.99 (aim
≥ 0.995), a synthetic-weight test at real shapes, prefill smoke + one larger `seq_len`, a decode test
vs one HF decode step, and a static grep audit that the runtime forwards contain no
`torch`/`from_torch`/`to_torch`; `doc/context_contract.json` parseable by
`.agents/scripts/check_context_contract.py` with `hf_advertised_context`, `current_supported_context`,
and per-path status; `README.md` + `work_log.md` with the runtime contract, a PCC table, the
**IR-graph provenance** (which `.mlir` files you translated, their classification signals, the TP
degree/mesh you collapsed, captured structured in `doc/functional_decoder/multichip_provenance.json`
for the `$multichip` stage), and a Limitations section. Watcher / tracy / long-context / paged-KV
evidence is not required this pass but do not claim it.

## Reuse Pointers

- `$forge-functional-decoder` SKILL.md — the shared contract this skill inherits; read it first.
- `scripts/ir_to_emit.sh`, `scripts/classify_graphs.sh` — the IR→emit converter and graph classifier
  shipped with this skill.
- `models/common/lightweightmodule.py`; `models/common/modules/attention/attention_1d.py`,
  `.../rope/rope_1d.py`, `.../rmsnorm/rmsnorm_1d.py` — clean config-first building blocks.
- `$functional-decoder` — the full contract this stage is a scoped subset of (TTNN correctness
  defaults, artifact format).
- Existing example autoports (shape/format of `functional_decoder.py`, tests, README, contract): any
  completed `models/autoports/*/` in the agentic-research project.

## Code Quality Defaults

- Prefer explicit model contracts over permissive fallback logic; fail directly on invalid config.
- Keep the graph's shape-specialized layout, program configs, per-core grids, and collectives out of
  the runtime entirely; run its math on simple `bf16` / `TILE` / `DRAM` defaults on a 1x1 mesh.
- Translate the tensor-parallel graph to equivalent single-device dense math (full weights, no
  collectives); confirm every collapse by PCC against the dense HF reference.
- Preserve the emitted workload batch; drop layout/TP glue, not batch.
- Be honest in the docs and contract about which paths are translated, stubbed, and supported, and
  about the IR provenance and TP collapse.
