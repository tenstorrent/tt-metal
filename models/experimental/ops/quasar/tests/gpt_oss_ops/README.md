# Per-op tests generated from a GPT-OSS graph capture (`tests/gpt_oss_ops/`)

Isolated per-op tests **generated mechanically from a ttnn graph capture** of a gpt-oss-20b demo
run, so each op is exercised with the exact shapes, dtypes, layouts, memory configs and program
configs the model really used.

Sibling of [`tests/qwen3_vl_ops/`](../qwen3_vl_ops/README.md) and of the original
[`models/experimental/llama32_1b_quasar/tests/graph_ops/`](../../../../llama32_1b_quasar/tests/graph_ops/README.md),
sharing their generator and runtime — read the llama README for the full discussion of how a case
is reconstructed and what a capture cannot tell you. This file covers what is specific to GPT-OSS.

## Quick start

```bash
# whole suite
pytest models/experimental/ops/quasar/tests/gpt_oss_ops/ -v

# one op, one case
pytest models/experimental/ops/quasar/tests/gpt_oss_ops/test_sparse_matmul.py -k 00_1x2880 -v

# the subset that fits the 2-node Quasar emulator
pytest models/experimental/ops/quasar/tests/gpt_oss_ops/ -m emulator

# shapes/dtypes/finiteness only — no PCC against a torch golden
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/ops/quasar/tests/gpt_oss_ops/
```

## Coverage of the current capture

Source: a `-k "prefill_128"` run of gpt-oss-20b on a single Blackhole device (1x1 mesh), with
device trace off and the decode length capped — see [Regenerating](#regenerating). 15,141 recorded
python-level calls over 42 ops.

- **38 op files, 119 distinct cases, covering 14,198 captured calls.**
- **Value coverage: 101 of 119 cases (11,798 of 14,198 calls) are checked against a torch
  golden**, 2 more (480 calls) against the paged-cache postcondition, and 16 (1,920 calls) on
  shape/dtype/placement/finiteness only: both SDPAs, rope, `topk`, `scatter` and `sparse_matmul`
  (`graph_case.GOLDEN`'s header comment says why each).
- **88 of 119 cases are marked `emulator`**, over 32 of the 38 ops — a much larger fraction than
  the qwen3_vl suite, because this capture is batch-1 decode-dominated rather than
  prefill-dominated.
- 4 ops are deliberately not generated (`SKIP_OPS`): `ttnn.deallocate`, `ttnn.from_torch` /
  `ttnn.as_tensor` (exercised by every case's input upload) and `ttnn.to_torch` (used by every
  assertion).

The MoE machinery is what distinguishes this suite from its siblings. `ttnn.sparse_matmul` (720
calls, 4 cases) is the expert matmul; `topk` + `softmax` + `sigmoid` are the router; `clamp` (480
calls) is the SwiGLU limit; `ttnn.experimental.fast_reduce_nc` combines the weighted expert
outputs (`gpt_oss/tt/experts/operations.py:66`); and `scatter` / `permute` / `repeat` move tokens
between the token-major and expert-major layouts.

## Regenerating

**Prerequisite:** the capture step needs the `enable_torch_tracer` split from
`martemov/llama-graph-tracing-bbradel-fork`, which is not on main yet. Without it,
`enable_logging` + `enable_graph_report` turns on the legacy python torch tracer
(`ttnn/ttnn/decorators.py:997-999`), which wraps every incoming `torch.Tensor` in a
`TracedTorchTensor` that nanobind cannot dtype-convert — `ttnn.from_torch` aborts before the model
is built. The committed cases came from a capture taken with that branch merged locally; nothing
in this directory depends on it, only the act of re-capturing does.

```bash
# 1. capture a run. The demo turns device trace off and caps generated tokens by itself when
#    ttnn logging is on (text_demo.py) -- logging syncs the device and reads tensors per op, both
#    hard-fatal inside a trace region, and 200 eager decode steps only inflate the record count.
MESH_DEVICE=<sku> HF_MODEL=<path to gpt-oss-20b> \
TTNN_CONFIG_OVERRIDES='{"enable_logging":true,"enable_fast_runtime_mode":false,"enable_graph_report":true,"enable_detailed_buffer_report":false,"report_name":"gpt_oss_demo"}' \
    pytest models/experimental/ops/quasar/gpt_oss/demo/text_demo.py -k "prefill_128 and not 128k"

# 2. drop the per-record captured_graph the generator does not read (124 MB -> 19 MB)
python models/experimental/ops/quasar/tests/gpt_oss_ops/slim_python_io.py \
    generated/ttnn/reports/<report>/graph_capture.python_io.json \
    generated/ttnn/reports/<report>/graph_capture.python_io.slim.json

# 3. generate the suite (the generator is shared with the llama and qwen3_vl suites)
python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
    --capture generated/ttnn/reports/<report>/graph_capture.python_io.slim.json \
    --out    models/experimental/ops/quasar/tests/gpt_oss_ops

# 4. validate without a device: the generated data (no ttnn), then the goldens (imports ttnn,
#    opens nothing)
python models/experimental/ops/quasar/tests/gpt_oss_ops/validate_cases.py
python models/experimental/ops/quasar/tests/gpt_oss_ops/validate_goldens.py
```

**Keep the capture if you want to regenerate.** `ttnn.CONFIG.delete_reports_on_start` wipes
`generated/ttnn/reports/` when the next run starts.

Do not hand-edit `test_<op>.py` — fix `graph_case.py` (or the generator) instead. The generated
files are data only, so a fidelity improvement applies to every op at once and survives
regeneration.

## Files

| File | Role |
| --- | --- |
| `test_<op>.py` | **generated.** A `CASES` list (pure data) + a two-line test body. One file per op. |
| `graph_case.py` | Runtime. Turns a case back into ttnn objects, runs it, checks the result. **All interpretation lives here.** |
| `op_utils.py` | The four model-specific hooks `graph_case` needs, re-exported from the llama op suite. |
| `conftest.py` | Tags emulator-appropriate cases with the `emulator` marker. |
| `validate_cases.py` | Offline consistency check of generated data vs `graph_case.py`'s tables. |
| `validate_goldens.py` | Checks that every `GOLDEN` reference runs against its cases — no device, though it imports ttnn. |
| `slim_python_io.py` | Step 2 above: strips `captured_graph` from a capture, streaming. |

## What this capture needed that the siblings did not

**`ttnn.Tile` arguments.** `sparse_matmul` passes an explicit `output_tile`, which reaches the
capture as `Tile with shape: [32, 32]`. The generator had no branch for it, so all 720 calls of
the model's central op were dropped as unreconstructible. Added to `parse_argument` (a `tile`
spec) and to `graph_case._build_value` (`ttnn.Tile(shape)`).

**Four more `MatmulMultiCoreReuseMultiCast1DProgramConfig` fields.** This is the first capture to
use that config, and `graph_case._PROGRAM_CONFIG_FIELDS` predated `gather_in0`,
`num_global_cb_receivers`, `untilize_out` and `stream_in1` — all real ctor kwargs
(`matmul_nanobind.cpp:401-406`). `validate_cases.py` flagged every one, which is what it is for.
`hop_cores` is handled like `allowed_worker_cores` but compares against *empty* rather than null:
its ctor default is an empty CoreRangeSet, so the repr prints `{}` rather than `std::nullopt`.

**Goldens for the MoE ops** (`graph_case.GOLDEN`): `matmul`, `sigmoid`, `softmax`, `clamp`,
`permute`, `repeat`, `sum`, and `fast_reduce_nc` (a keepdim sum over the expert dim — each
expert's output is already scaled by its routing weight, so combining them is addition).

Deliberately without a golden: `topk` (random bfloat16 values tie often enough that torch's
indices and the op's are both correct and different — the same reason `argmax` has none) and
`sparse_matmul` (the result depends on the sparsity tensor's routing semantics, which the capture
records as data rather than meaning).

## Fidelity

The llama README's fidelity table applies verbatim — random tensor values,
`compute_kernel_config` dropped, index tensors filled semantically, replicated onto the `(1, 1)`
mesh the capture ran on. Each generated file's docstring lists the gaps that apply to that op.

Two caveats specific to this capture:

- **device trace was off**, so the decode ops are recorded as the eager calls a traced run would
  replay. A bug that only appears when the op runs from a captured trace is out of reach here.
- **the decode length was capped**, so `count` reflects a handful of steps rather than a full
  generation. The *set* of captured calls is unaffected (every decode step issues the same ops),
  but `count` ranks cases, so it should be read as relative weight within a step, not as the
  model's real call volume.
