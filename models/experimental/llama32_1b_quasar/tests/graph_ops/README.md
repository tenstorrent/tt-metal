# Per-op tests generated from a graph capture (`tests/graph_ops/`)

Isolated per-op tests **generated mechanically from a ttnn graph capture** of the
model run, so each op is exercised with the exact shapes, dtypes, layouts, memory
configs and program configs the model really used.

This is the capture-driven counterpart to [`tests/ops/`](../ops/README.md), which
was authored from a *static* read of the model source. Both suites answer the same
question — "does this one op work on the emulator?" — from opposite directions; see
[Which suite to use](#which-suite-to-use).

## Quick start

```bash
# whole suite
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/ -v

# one op, one case
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_rms_norm.py -k 00_32x2048 -v

# the subset that fits the 2-node Quasar emulator
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/ -m emulator

# shapes/dtypes/finiteness only — no PCC against a torch golden
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/
```

## Regenerating

```bash
# 1. capture a model run (writes generated/ttnn/reports/<report_name>/graph_capture.python_io.json)
#    MESH_DEVICE is required — the demo skips at import without it; this capture used N150.
MESH_DEVICE=N150 \
TTNN_CONFIG_OVERRIDES='{"enable_graph_report": true, "report_name": "llama32_1b_demo_aug20_0223"}' \
    pytest models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/demo.py -k "token-accuracy"

# 2. generate the suite
python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
    --capture generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json \
    --out    models/experimental/llama32_1b_quasar/tests/graph_ops

# 3. validate the generated data offline (no ttnn, no device)
python models/experimental/llama32_1b_quasar/tests/graph_ops/validate_cases.py
```

The `.python_io.json` sidecar is what carries the python-level call arguments; it is
written by `ttnn.graph.end_graph_capture_to_file` (`ttnn/ttnn/graph.py:285,444`) when
`enable_graph_report` is on. The main `graph_capture.json` alone is not enough — it
records the C++ op graph, not the python call arguments.

## Files

| File | Role |
| --- | --- |
| `test_<op>.py` | **generated.** A `CASES` list (pure data) + a two-line test body. One file per op. |
| `graph_case.py` | Runtime. Turns a case back into ttnn objects, runs it, checks the result. **All interpretation lives here.** |
| `generate_from_graph_capture.py` | The generator. Model-agnostic: `--capture`, `--out`. |
| `validate_cases.py` | Offline consistency check of generated data vs `graph_case.py`'s tables. |
| `conftest.py` | Tags emulator-appropriate cases with the `emulator` marker. |

Do not hand-edit `test_<op>.py` — fix `graph_case.py` (or the generator) instead. The
generated files are data only, so a fidelity improvement applies to every op at once
and survives regeneration.

## What a case looks like

```python
{
    "id": "00_32x2048_bf16_ws-l1",           # greppable: pytest -k 00_32x2048
    "op": "ttnn.rms_norm",
    "count": 327,                            # times this exact call occurred in the run
    "args": [
        {"k": "t", "shape": [1, 1, 32, 2048], "dtype": "BFLOAT16", "layout": "TILE",
         "mem": {"layout": "WIDTH_SHARDED", "buffer": "L1",
                 "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"}}},
    ],
    "kwargs": {
        "epsilon": {"k": "lit", "v": 1e-05},
        "weight": {"k": "t", "shape": [1, 1, 64, 32], ...},
        "program_config": {"kind": "LayerNormShardedMultiCoreProgramConfig",
                           "fields": {"compute_with_storage_grid_size": [8, 4], "subblock_w": 2,
                                      "block_h": 1, "block_w": 2, "inplace": 0, ...}, "k": "cfg"},
        ...
    },
    "outs": [
        {"shape": [1, 1, 32, 2048], "dtype": "BFLOAT16", ...},      # one spec per returned tensor
    ],
}
```

`count` is the useful part when triaging: it tells you which shape actually dominates
a run, so you can fix the case worth 327 calls before the one worth 2.

Each output spec is recovered by tracking `tensor_id` across the whole capture — an
op's output shows up later as some other op's input, which is what pins down its
shape and dtype. `outs` holds one entry per tensor the op returned, in order, so a
multi-output op is fully checked (`nlp_create_qkv_heads` asserts K and V, not just
Q). An output that is never consumed again carries `None` — its length still pins
the op's output count, and the tensor itself is checked for finiteness only.

## Fidelity: what is exact, what is not

Exact, straight from the capture:

- input shapes, dtypes, layouts;
- memory configs including shard grids, shard shapes and orientation (so a
  DRAM-sharded weight really is DRAM-sharded across 12 banks, and a width-sharded
  activation really spans its 32 captured cores);
- program configs field-for-field (`in0_block_w`, `per_core_M/N`, LayerNorm's
  `block_h/block_w/subblock_w` and `legacy_reduction`/`legacy_rsqrt`/`use_welford`,
  SDPA's chunk sizes and `max_cores_per_head_batch`, `MinimalMatmulConfig` blocks);
- scalar kwargs (`epsilon`, `scale`, `num_heads`, `is_decode_mode`, activations …);
- the shape, dtype, layout and memory config of every returned tensor to assert
  against — a relayout op (`to_memory_config`, `interleaved_to_sharded`) that hands
  its input back untouched fails on the placement check. The shard *detail* (grid,
  per-core shape, orientation) is compared only when the captured grid exists on the
  device under test; on a smaller one the op derives its own grid, so only the memory
  layout and buffer type are asserted there.

Not recoverable from a capture, and how each is handled:

| Gap | Handling |
| --- | --- |
| `compute_kernel_config` — recorded only as an object address | dropped; the op's default is used. Affects math fidelity / fp32 accumulation, i.e. PCC, not shapes or program structure. |
| tensor **values** | random `bfloat16` noise, except where a value carries meaning: index tensors (next row) and the tensor a paged-cache op writes (`CACHE_SENTINEL`, so the write can be verified). |
| index tensors (page tables, `cur_pos`, `update_idxs`, token ids) — random values would fault the device | filled semantically by `graph_case.INDEX_VALUES` (page ids `arange`, small valid positions, ids `< vocab`). |
| golden outputs | a torch reference is used wherever it is unambiguous (`graph_case.GOLDEN`: matmul, elementwise, relayout/identity ops, transpose, concat, slice, embedding, rms_norm, reshape/unsqueeze views, and the `nlp_*` head splits/concats — the last four mirror the references `tests/ops/` verified on hardware, so a permutation or a mis-split of the fused QKV fails here too). **66 of 74 cases (8,868 captured calls) have one.** Rope and SDPA do not: rope's Meta-format cos/sin plus the tile-wise transformation matrix make a reference easy to get subtly wrong (`tests/ops/test_rotary_embedding_llama.py:33` reached the same conclusion), and SDPA would mean reimplementing chunked flash attention. Those 6 cases (1,020 calls) get shape/dtype/placement/finiteness only. |
| the paged caches' values (2 cases, 680 calls) — the capture never sees their output again | `graph_case.POSTCONDITION` instead of a golden: the tensor they write is filled with `CACHE_SENTINEL` (1024.0, which standard-normal noise cannot produce), and since the page table and positions are generated here (`INDEX_VALUES`), the **destination** is known too. The case fails unless exactly the cells `page_table[user, pos // block_size]` / slot `pos % block_size` selects hold the sentinel, and none outside do — so writing nothing, writing too little, or writing to the wrong page all fail. The paged layout this assumes is the op's own (`tests/ttnn/nightly/.../test_paged_update_cache.py:449,641`), documented at `graph_case._paged_write_region`. |
| an output the capture never observed again (`"outs"` entry `None`: `untilize`, two `sharded_to_interleaved` cases, the paged caches) | what the **call itself** pins down is used instead of nothing (`graph_case._derived_spec`): a `memory_config` argument is where the output must land, `untilize`/`tilize` fix the output layout, and a relayout/move op preserves its input's shape. So a `sharded_to_interleaved` that hands back its width-sharded input fails even though identity PCC cannot see it. Failure messages say `implied by the call` rather than `captured`. |
| memory configs of tensors passed **inside a python list** (e.g. `ttnn.concat`) | that repr carries shape/dtype/layout only, so list elements are uploaded DRAM-interleaved. Flagged in the generated file's docstring. |
| optional core-grid restrictions (`sub_core_grids`, `allowed_worker_cores`) | `std::nullopt` in every capture so far; a non-null value **skips** the case rather than running a different core set. |
| mesh placement | replicated onto the `(1, 1)` mesh the capture ran on. |
| an input whose logical shape does not fill its shard (rope's cos/sin: 1 row in a 32-row shard; decode's 8 KV heads in a 32-row shard) | built DRAM-interleaved and moved with `to_memory_config`. Passing that memory config straight to `from_torch` pads the **logical** shape up to the shard, which changes what the op computes — rope then returns 32 rows where the capture recorded 8. Costs one extra relayout op per such input. |
| an output with more elements than all its inputs combined (batch-1 decode result in a tensor padded to 32 tile rows) | the op cannot write all of it, and the untouched padding is stale L1. Finiteness is asserted over the **leading** elements the inputs account for — row-major puts the logical data first — so a NaN in the computed region still fails while stale padding is ignored. |

Each generated file's docstring lists the gaps that apply to *that* op, so you never
have to guess whether a failure is a real bug or a reconstruction artifact.

## Which suite to use

| | `tests/ops/` (static analysis) | `tests/graph_ops/` (this suite) |
| --- | --- | --- |
| Derived from | reading the model source | a recorded run |
| Configs | idealized — interleaved, no program_config, so the op runs anywhere | verbatim — sharded grids, DRAM-sharded weights, real program configs |
| Correctness signal | strong: torch golden per op, tuned PCC | golden where unambiguous (PCC > 0.999, lowered only for block-float dtypes and reduction-heavy ops — see `graph_case._PCC_BY_DTYPE` / `_PCC_BY_OP`); otherwise shape/dtype/placement/finite |
| Coverage | the call sites a human found | every call the model actually made, with call counts |
| Catches | wrong math in an op | shape/shard/program-config-specific hangs, faults, allocation failures |
| Cost to add a model | hours of reading per model | minutes: one capture + one generator run |
| Fails when | the op is genuinely broken | the op is broken **or** the exact captured config is unsupported |

They are complementary, and the practical split is:

- **A new model, or an op that hangs/faults in the demo but passes in isolation** →
  start here. The static suite's idealized interleaved stand-in is exactly what makes
  those bugs disappear; this suite reproduces the config that breaks.
- **An op that produces wrong numbers** → `tests/ops/`, where the torch reference is
  authoritative and the shapes are small.

Suggested workflow going forward: generate this suite first (it is nearly free and
tells you what the model really calls), then hand-write a `tests/ops/` case only for
the ops where you need a trustworthy golden.

## Coverage of the current capture

Source: `generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json`
(16,667 recorded python-level calls).

- **27 op files, 74 distinct cases, covering 10,568 captured calls.**
- **Value coverage: 66 of 74 cases (8,868 of 10,568 calls) are checked against a torch
  golden**, 2 more (680 calls) against a paged-cache postcondition, and 6 (1,020 calls
  — rope and SDPA) on shape/dtype/placement/finiteness only.
- 5 ops are deliberately not generated (`graph_case`/generator `SKIP_OPS`):
  `ttnn.deallocate` (nothing to assert), `ttnn.from_torch` / `ttnn.as_tensor`
  (exercised by every case's input upload), `ttnn.load_tensor` (needs a
  `model_cache/` file the demo produces), `ttnn.to_torch` (used by every assertion).
- **30 of 74 cases are marked `emulator`**, spread over 17 of the 27 ops. The
  remaining cases are prefill-shaped (1024-row activations) or width-sharded across
  32 captured cores, which the emulator's grid cannot allocate — `graph_case`
  skips those at run time with the grid it finds, so they are not failures. The row
  cap is measured on the *activation*, which is arg 1 for the paged-cache ops
  (`conftest._PRIMARY_ARG`): their arg 0 is a legitimately tall KV cache, not a
  prefill-sized input.

## Extending

- **A golden reference for another op** — add it to `graph_case.GOLDEN`. Return
  `None` from the reference when a case is outside what it models; the runner falls
  back to shape/dtype/finiteness rather than failing on a bad reference.
- **A new index-tensor heuristic** — add `(op, arg key) -> fn` to
  `graph_case.INDEX_VALUES`.
- **A program config this capture is the first to use** — add its ctor fields to
  `graph_case._PROGRAM_CONFIG_FIELDS` (verified against the nanobind `.def(nb::init…)`
  for that class). `validate_cases.py` fails loudly on a config field with no mapping,
  so a stale table cannot silently drop a field.
- **A new model** — point the generator at that model's capture and `--out` at its
  own `tests/graph_ops/`. The generated files import their runtime from the `--out`
  directory itself (derive it, or set `--runtime-module`), so copy `graph_case.py`
  there: only its import of `..ops.op_utils` (for `assert_pcc` / `from_tt` / the mesh
  fixture parametrization) is model-specific.
