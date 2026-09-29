# Per-op tests generated from a Qwen3-VL graph capture (`tests/qwen3_vl_ops/`)

Isolated per-op tests **generated mechanically from a ttnn graph capture** of a
Qwen3-VL-4B-Instruct demo run, so each op is exercised with the exact shapes, dtypes,
layouts, memory configs and program configs the model really used.

This is the Qwen3-VL sibling of
[`models/experimental/llama32_1b_quasar/tests/graph_ops/`](../../../../llama32_1b_quasar/tests/graph_ops/README.md),
and it shares that suite's generator and runtime — read its README for the full
discussion of how a case is reconstructed and what a capture cannot tell you. This
file covers what is specific to Qwen3-VL.

## Quick start

```bash
# whole suite
pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/ -v

# one op, one case
pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/test_rms_norm.py -k 00_32x2560 -v

# the subset that fits the 2-node Quasar emulator
pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/ -m emulator

# shapes/dtypes/finiteness only — no PCC against a torch golden
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/
```

## Coverage of the current capture

Source: `generated/ttnn/reports/qwen3_vl_demo_aug27_1509/`, a full
`-k "batch-1"` demo run of `Qwen/Qwen3-VL-4B-Instruct` on N150 — vision tower, text
prefill, and 200 generated tokens x 2 repeat batches of decode. 749,552 recorded
python-level calls over 42 distinct ops.

- **38 op files, 134 distinct cases, covering 511,259 captured calls.**
- **Value coverage: 121 of 134 cases (438,269 of 511,259 calls) are checked against a
  torch golden**, 2 more (28,944 calls) against the paged-cache postcondition, and 11
  (44,046 calls) on shape/dtype/placement/finiteness only — rope, both SDPAs, `argmax`,
  `pad` and `scatter` (see `graph_case.GOLDEN`'s header comment for why each).
- **37 of 134 cases are marked `emulator`**, spread over 19 of the 38 ops.
- 4 ops are deliberately not generated (`SKIP_OPS`): `ttnn.deallocate` (234,969 calls,
  nothing to assert), `ttnn.from_torch` / `ttnn.as_tensor` (exercised by every case's
  input upload) and `ttnn.to_torch` (used by every assertion).

The call counts are dominated by decode — 200 tokens x 2 batches x 36 layers — so
`count` ranks the decode-shaped case of an op above its prefill-shaped one. The vision
tower still contributes its own cases: `layer_norm`, the 24-block attention shapes, and
the patch-merger `linear`s all appear with their real (much taller) activations.

Both halves of the model are represented: `ttnn.layer_norm` / `ttnn.transformer.
scaled_dot_product_attention` / `ttnn.experimental.nlp_create_qkv_heads` come from the
vision tower and text prefill, while `ttnn.experimental.paged_update_cache` /
`paged_scaled_dot_product_attention_decode` / `nlp_create_qkv_heads_decode` are the
decode path. `ttnn.scatter` and `ttnn.mesh_partition` are Qwen3-VL-specific: the
vision-token merge and the vision tower's output partition.

`PROGRAM_FACTORIES.md` lists which C++ program factory each device op actually
selected in this run, with its program-cache hit rate — useful for turning a failing
case into a named factory to go read. It reads a factory identity that the checked-in
graph tracer does not record (the capture it was built from came from a branch that
does), so `program_factories.py` finds nothing in a capture taken from this tree.

## Regenerating

```bash
# 1. capture a model run. Device trace has to be off (demo.py's `enable_trace`): ttnn
#    logging syncs the device and reads tensors per op, both hard-fatal in a trace region.
MESH_DEVICE=N150 HF_MODEL=Qwen/Qwen3-VL-4B-Instruct \
TTNN_CONFIG_OVERRIDES='{"enable_logging":true,"enable_fast_runtime_mode":false,"enable_graph_report":true,"enable_detailed_buffer_report":false,"report_name":"qwen3_vl_demo"}' \
    pytest models/experimental/ops/quasar/qwen3_vl/demo/demo.py -k "batch-1"

# 2. drop the per-record captured_graph the generator does not read (5.0 GB -> 1.1 GB)
python models/experimental/ops/quasar/tests/qwen3_vl_ops/slim_python_io.py \
    generated/ttnn/reports/<report>/graph_capture.python_io.json \
    generated/ttnn/reports/<report>/graph_capture.python_io.slim.json

# 3. generate the suite (the generator is shared with the llama suite)
python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
    --capture generated/ttnn/reports/<report>/graph_capture.python_io.slim.json \
    --out    models/experimental/ops/quasar/tests/qwen3_vl_ops

# 4. validate without a device: the generated data (no ttnn), then the goldens (imports
#    ttnn, opens nothing)
python models/experimental/ops/quasar/tests/qwen3_vl_ops/validate_cases.py
python models/experimental/ops/quasar/tests/qwen3_vl_ops/validate_goldens.py

# 5. optional: which program factory each op selected (reads the C++ graph)
python models/experimental/ops/quasar/tests/qwen3_vl_ops/program_factories.py \
    generated/ttnn/reports/<report>/graph_capture.json > \
    models/experimental/ops/quasar/tests/qwen3_vl_ops/PROGRAM_FACTORIES.md
```

Step 2 is the one thing this model needs that the llama suite does not. A run this
long records ~750k calls and every record carries its own `captured_graph` (the C++
node list for that one op), which is ~97% of the bytes and which the generator never
reads. The llama captures are ~100 MB and go straight into step 3.

**Keep the capture if you want to regenerate.** `ttnn.CONFIG.delete_reports_on_start`
wipes `generated/ttnn/reports/` when the next run starts, so the slim file is gone
after the next demo invocation unless you move it somewhere else.

Do not hand-edit `test_<op>.py` — fix `graph_case.py` (or the generator) instead. The
generated files are data only, so a fidelity improvement applies to every op at once
and survives regeneration.

## Files

| File | Role |
| --- | --- |
| `test_<op>.py` | **generated.** A `CASES` list (pure data) + a two-line test body. One file per op. |
| `graph_case.py` | Runtime. Turns a case back into ttnn objects, runs it, checks the result. **All interpretation lives here.** Copied from the llama suite, plus the goldens below. |
| `op_utils.py` | The four model-specific hooks `graph_case` needs (mesh parametrization, `from_tt`, `torch_rand`, `assert_pcc`), re-exported from the llama op suite as `tests/yolo_ops` does. |
| `conftest.py` | Tags emulator-appropriate cases with the `emulator` marker. |
| `validate_cases.py` | Offline consistency check of generated data vs `graph_case.py`'s tables. |
| `validate_goldens.py` | Checks that every `GOLDEN` reference actually runs against its cases — no device, though it imports ttnn. It cannot check the answer, only that the reference does not crash or silently return None. |
| `slim_python_io.py` | Step 2 above: strips `captured_graph` from a capture, streaming. |
| `program_factories.py` | Step 5 above: per-op program-factory summary from the C++ graph. |
| `PROGRAM_FACTORIES.md` | Its output for the current capture. |

The generator itself is not copied — it is model-agnostic and lives in the llama
suite; `--out` decides which directory's `graph_case.py` the generated files import.

## What is different from the llama suite

**Tensor-list arguments.** The committed `concat` cases carry a memory config per list
element, which the llama suite's cases do not (its README records that list elements
reach the capture with shape/dtype/layout only and are uploaded DRAM-interleaved).
Those placements came from the capture these files were generated from. The ttnn
serializer and generator changes behind it are **not part of this PR**, so step 3 above
does not reproduce them today — regenerating drops the `concat` cases rather than
rebuilding them. Known gap between the committed data and the checked-in tooling.

**Goldens added for the ops this model introduced** (`graph_case.GOLDEN`):
`layer_norm` (the vision norm, weight/bias tile-padded like rms_norm's gamma),
`split`, `plus_one`, `expand`, `zeros_like`, `to_layout` (identity),
`unsqueeze` (view), and `mesh_partition` — the last one only while the captured output
keeps the input's shape, which is what `ttnn::mesh_partition` returns on the
single-device mesh the capture ran on (`mesh_partition.cpp:17`).

Deliberately left without a golden: `argmax` (random bfloat16 logits tie at the
maximum often enough that torch's index and the op's index are both correct and
different), `scatter` and `pad` (index / pad-value semantics would have to be
re-derived, and a subtly wrong reference is worse than the structural checks).

## Fidelity

The llama README's fidelity table applies verbatim — random tensor values,
`compute_kernel_config` dropped, index tensors filled semantically, replicated onto
the `(1, 1)` mesh the capture ran on. Each
generated file's docstring lists the gaps that apply to *that* op, so a failure can be
read as a real bug or a reconstruction artifact without guessing.

One capture-specific caveat: this run had **device trace capture disabled**, since
`enable_logging` and a ttnn trace region cannot coexist. The decode ops are therefore
recorded as the eager calls that a traced run would replay, which is the same op with
the same config — but a bug that only appears when the op runs from a captured trace
is out of this suite's reach by construction.
