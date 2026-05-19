# Module-class references

This directory holds **module-class reference fingerprints** that the
`tt_hw_planner perf` pipeline uses to compare your model's submodules
against the equivalent submodules of a well-optimized peer model.

## What lives here

- `schema.yaml` — the schema documentation. Never loaded by the DB; copy
  its structure when authoring a new reference.
- `_*.yaml` — drafts. Ignored by the loader.
- `<name>.yaml` — active references. One file per
  `(reference_model, box, mesh, dtype)` combination.

The `reference_db.py` module enumerates every `*.yaml` in this
directory (excluding `_*` and `schema.yaml`) at startup.

## Authoring a reference

1. Run `tt_hw_planner perf collect` against a well-optimized peer model
   on the target box. The optimizer blocks you applied are recorded in
   `perf-data/<run_id>/applied_blocks.json` (if any).

2. Run `tt_hw_planner perf join <run_id>` to produce `joined.json`.

3. Copy `schema.yaml` to a new file named after the run, e.g.
   `llama_3.1_8b_qb2_bfp8.yaml`.

4. Fill in the `provenance` block from `run_meta.json`:
   ```
   model_id           -> meta["model_id"]
   arch_family        -> short tag (llama / qwen / mistral / ...)
   box                -> meta["box"]
   mesh_shape         -> meta["mesh_shape"]
   dtype              -> meta["dtype"]
   source_run_id      -> meta["run_id"]
   ```

5. For each distinct submodule **role** in the model (one entry per
   *role*, not per layer — use `*` in `role_path` to match all
   layers), fill in:
   - `role_path`: dot-separated attribute path with `*` for the layer
     index. Inspect a few rows from `joined.json` to find the actual
     paths recorded.
   - `module_class`: the Python class name of the `nn.Module`.
   - `metrics`: pick a representative layer (e.g. layer 5 of 32) and
     read the median runtime / utilization from the joined rows.
   - `config_summary`: copy the `arguments` dict of one of the rows
     attributed to this module — that's the kernel config that produced
     the metrics.
   - `optimizer_blocks_applied`: list every block from
     `applied_blocks.json` that touched this submodule. The
     suggestion engine will recommend these to user-runs whose
     equivalent module is slower.

6. Validate: run
   ```
   python -m scripts.tt_hw_planner perf suggest <user_run_id>
   ```
   on a run of a peer model. If your new reference is picked up
   correctly you should see per-module suggestions in the output.

## What the matcher does

When the user clicks a node in the cytoscape view, the matcher
(`reference_db.find_module_reference`) is invoked with the user's
`(attribute_path, module_class, arch_family, mesh_shape, dtype, box)`.
Scoring:

- `role_path` match: +100
- `module_class` match: +30
- exact mesh: +20
- exact dtype: +10
- exact box: +5
- exact-family (vs aliased): +50

The matcher walks family aliases in the order defined by
`_FAMILY_ALIASES` in `reference_db.py`. The first strong match
(score >= 100) wins.

## What a "good" reference looks like

- Covers every distinct submodule **role** in the model (typically 6-12
  entries for a transformer: rms_norm, q_proj, k_proj, v_proj, sdpa,
  o_proj, gate_proj, up_proj, down_proj, mlp_act, residual_add,
  lm_head).
- Each entry is anchored in a **specific, reproducible run**
  (`source_run_id` in provenance).
- `optimizer_blocks_applied` lists the **exact** block name + parameter
  string the runner emits when applying a block (e.g.
  `matmul_tuner@in1_block_w=4`, not `matmul_tuner`).

## When a reference is "outdated"

A reference's metrics encode a specific tt-metal commit's perf. When
tt-metal lands a major op-level change (new matmul kernel, new
dispatcher, etc.), re-collect from the same peer model and overwrite
the YAML. Use `provenance.notes` to log what changed.
