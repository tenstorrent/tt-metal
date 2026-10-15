# Per-op tests generated from a Panoptic-DeepLab graph capture (`tests/panoptic_deeplab_ops/`)

Isolated per-op tests **generated mechanically from a ttnn graph capture** of a
Panoptic-DeepLab run (the Quasar fork in
[`models/experimental/ops/quasar/panoptic_deeplab/`](../../panoptic_deeplab/README.md)),
so each op is exercised with the exact shapes, dtypes, layouts, memory configs and
conv / program configs the model really used.

This is the CNN sibling of [`tests/qwen3_vl_ops/`](../qwen3_vl_ops/README.md). Both share
the generator in
[`models/experimental/llama32_1b_quasar/tests/graph_ops/`](../../../../llama32_1b_quasar/tests/graph_ops/README.md).
Read that suite's README for how a case is reconstructed and what a capture cannot tell
you. This file covers what is specific to Panoptic-DeepLab.

## Quick start

The model was captured on a 20-core Blackhole grid (5x4, `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3"`),
so run the suite on that grid. Conv, pool and upsample choose their own shard grid from
the device. On a larger grid their outputs are sharded differently from the capture, and
the placement check reports that difference.

```bash
# whole suite (one file per pytest invocation is fine too)
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/ops/quasar/tests/panoptic_deeplab_ops/ -v

# one op, one case
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" \
    pytest models/experimental/ops/quasar/tests/panoptic_deeplab_ops/test_conv2d.py -k 21_512x1024x8 -v

# the subset that fits the 2-node Quasar emulator
pytest models/experimental/ops/quasar/tests/panoptic_deeplab_ops/ -m emulator

# shapes/dtypes/finiteness only — no PCC against a torch golden
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/ops/quasar/tests/panoptic_deeplab_ops/
```

## Running on Quasar (craq-sim / emulator)

Two switches, both off on Wormhole/Blackhole by default:

- `TTNN_GRAPH_OPS_BF8_AS_BF16=1`: Quasar has no block-float formats, so build every captured
  bfloat8_b / bfloat4_b tensor (and a `DataType::BFLOAT8_B` inside a `Conv2dConfig`) as bfloat16.
  Same switch as the qwen3_vl / gpt_oss suites. The PCC floors still use the captured dtype.
- `TTNN_GRAPH_OPS_QUASAR_FORKS=auto|1|0` (default `auto`, i.e. on when the device is Quasar):
  call `ttnn.experimental.quasar.<op>` for the ops whose base implementation has no Quasar kernels
  (`op_utils._QUASAR_FORKED_OPS`: conv2d, add, multiply, move, max_pool2d, avg_pool2d). Every other op
  runs the base implementation, so failures still show which base ops are missing on Quasar.

```bash
TT_METAL_SIMULATOR=/path/to/craq-sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1 \
TTNN_GRAPH_OPS_BF8_AS_BF16=1 pytest models/experimental/ops/quasar/tests/panoptic_deeplab_ops/test_add.py
```

What fails on Quasar, and why, is tracked in
[`panoptic_deeplab/QUASAR_GAPS.md`](../../panoptic_deeplab/QUASAR_GAPS.md). Some cases hang on
craq-sim; pytest's `--timeout` cannot interrupt a hang inside the simulator, so run one file per
process under an external timeout.

## Coverage of the current capture

Source: `generated/ttnn/reports/panoptic_deeplab_20_cores_oct14_1821/`, one inference of
the full `PANOPTIC_DEEPLAB` model (ResNet-52 backbone, ASPP, semantic and instance-embedding
heads) at 512x1024. The run used the Detectron2 Cityscapes weights on ttsim Blackhole with a
5x4 grid. It passed its own PCC checks (semantic 0.989, center 0.966, offset 0.992).
1,470 python-level calls were recorded over 22 distinct ops.

- **19 op files, 161 distinct cases, covering 394 captured calls.** That is every call
  outside `SKIP_OPS`; no call was unreconstructible.
- **Value coverage: 160 of 161 cases (392 of 394 calls) are checked against a torch
  golden.** The exception is one `unsqueeze_to_4D` case (2 calls) whose output the capture
  never observed again; it gets shape/dtype/placement/finiteness only. Ops that only move,
  relayout or view data are held to **bit-exact** output (see below), the rest to PCC.
- **All 161 pass** on ttsim Blackhole with the 5x4 grid. One file per pytest invocation took
  about 8 min for `conv2d` and under 15 s for each other file.
- **23 of 161 cases are marked `emulator`**, spread over 11 of the 19 ops: one 1x1 `conv2d`
  on the ASPP pooling branch's single pixel, and the stride-16/32 activations
  (e.g. 32x64x256) of `add`, `concat`, `relu`, `reshape`, `to_layout`, `to_memory_config`,
  `typecast`, `unsqueeze_to_4D` and `upsample`, plus `to_device`. The cap is the VAD-variant
  suite's 48x48x256 elements on the primary activation, not the LLM suites' 128 rows, which
  is smaller than a 12x12 feature map. Everything at stride 4 or 2 (up to 256x512x256) is
  far over it, as is anything sharded over more than 8 cores. The classifier treats the
  `input_tensor` keyword and a `concat` list's first operand as the primary activation
  (`conftest._primary_input`).

- 3 ops are deliberately not generated (`SKIP_OPS`): `ttnn.deallocate` (nothing to assert),
  `ttnn.from_torch` (exercised by every case's input upload) and `ttnn.to_torch` (used by
  every assertion).

A CNN capture looks very different from the LLM ones. There is no decode loop, so almost
every call happens once. `count` is 1 or 2 for most cases, and the suite is about breadth:
**all 51 distinct `conv2d` configurations** are kept. Run the generator with
`--max-cases 64` (see below); the default cap of 32 would keep an arbitrary 32 of them.

## Regenerating

```bash
# 1. capture one inference. Needs the Detectron2 weights in
#    models/experimental/ops/quasar/panoptic_deeplab/weights/model_final_bd324a.pkl
#    (see the model README). The test's own 600 s timeout is too short on a simulator,
#    hence -p no:timeout.
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" \
TTNN_CONFIG_OVERRIDES='{"enable_logging":true,"enable_fast_runtime_mode":false,"enable_graph_report":true,"enable_detailed_buffer_report":false,"report_name":"panoptic_deeplab_20_cores"}' \
    pytest -p no:timeout models/experimental/ops/quasar/panoptic_deeplab/tests/pcc/test_tt_model.py \
        -k panoptic_deeplab_20_cores

# 2. generate the suite (the generator is shared with the llama and qwen suites).
#    --max-cases 64 keeps every distinct conv2d (51); the default 32 would drop 19 layers.
python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
    --capture generated/ttnn/reports/<report>/graph_capture.python_io.json \
    --out    models/experimental/ops/quasar/tests/panoptic_deeplab_ops \
    --max-cases 64

# 3. validate without a device: the generated data (no ttnn), then the goldens (imports
#    ttnn, opens nothing)
python models/experimental/ops/quasar/tests/panoptic_deeplab_ops/validate_cases.py
python models/experimental/ops/quasar/tests/panoptic_deeplab_ops/validate_goldens.py
```

The capture is small (12 MB, 1,470 records), so the qwen suite's `slim_python_io.py` step
is not needed.

**Keep the capture if you want to regenerate.** `ttnn.CONFIG.delete_reports_on_start`
wipes `generated/ttnn/reports/` when the next run starts.

Do not hand-edit `test_<op>.py`. Fix `graph_case.py` (or the generator) instead.

## Files

| File | Role |
| --- | --- |
| `test_<op>.py` | **generated.** A `CASES` list (pure data) + a two-line test body. One file per op. |
| `graph_case.py` | Runtime. Turns a case back into ttnn objects, runs it, checks the result. **All interpretation lives here.** Copied from the qwen suite, plus the CNN support below. |
| `op_utils.py` | The model-specific hooks `graph_case` needs. Re-exported from the llama op suite, except `with_default_mesh`, which opens the mesh with PDL's `l1_small_size`. |
| `conftest.py` | Tags emulator-appropriate cases with the `emulator` marker. |
| `validate_cases.py` | Offline consistency check of generated data vs `graph_case.py`'s tables. |
| `validate_goldens.py` | Checks that every `GOLDEN` reference runs against its cases (no device, imports ttnn). |

## What is different from the LLM suites

All of these are generic. A future CNN capture gets them for free.

**Host weights.** The capture is the model's *first* inference, where `ttnn.conv2d`
receives raw `[O, I, kH, kW]` weights and `[1, 1, 1, O]` bias **on host** and prepares
them itself. A host tensor still prints a MemoryConfig, so the generator now records
`storage_type` as `"host": True` on the spec, and `graph_case.build_tensor` leaves such
inputs on host. If it uploaded them instead, conv2d would treat them as already prepared.
The steady-state path, a second inference reusing the prepared device weights, is not in
this capture.

**`compute_config` dropped, then restored from the model.** Conv's spelling of
`compute_kernel_config` is captured only as an object address, so it is in the generator's
`DROP_KWARGS` (without that, every conv case was unreconstructible). The model builds every
conv's compute config from the same defaults (`panoptic_deeplab/tt/model_configs.py`: LoFi,
fp32 dest accumulation, no packer L1 accumulation), so `op_utils.default_kwargs` passes
exactly that to every conv2d case. Not restored: the `packer_l1_acc=True` the model sets on
its deep 1x1 bottleneck and ASPP convs, since a case does not say which layer it was.

**Goldens from the values the device holds; bit-exact data movement.** Each input is read
back after upload (`graph_case._held_values`), so a bfloat8_b input's quantization is in the
reference. Ops that only move, relayout or view data (`graph_case.LOSSLESS`: reshape,
permute, slice, concat, `to_layout`, `to_memory_config`, move, reshard, ...) must then
reproduce it bit for bit, as in the yolo and VAD-variant suites (`assert_lossless`), instead
of passing a PCC floor that is 0.97 for bfloat8_b. Two exceptions keep PCC (against the
read-back inputs): a cast to a narrower dtype, which rounds by design; and a bfloat8_b
`permute` or `concat`, which repack their output through the packer instead of copying tiles
(`graph_case._REPACKS_BLOCK_FLOAT`). On ttsim Blackhole a width concat of bfloat8_b changes
~9% of elements by one mantissa step even for tile-aligned operands, and even when an operand
is concatenated with itself, while a `dim=2` concat and every other data-movement op here is
bit-exact. Permute regroups values under new shared exponents and rounds them differently
from the host quantizer. 6 cases fall under this (all 5 `concat` cases and `permute` 00); 3
of them (concat 02/03, permute 00) fail an exact check.

**Nested return value.** `conv2d(..., return_output_dim=True, return_weights_and_bias=True)`
returns `(out, [h, w], [weight, bias])`. The capture records its tensors depth first:
`out, weight[, bias]`. `graph_case._output_tensors` flattens the result the same way. The
prepared weight and bias are never consumed again, so their specs are `None`. They are
checked for finiteness only; dtype/layout derived from the call's kwargs apply to output 0
alone (`_derived_spec`).

**`Conv2dConfig` / `Op2DSliceConfig`.** These print every field as its C++ value (`DataType::BFLOAT8_B`,
`0`/`1`, `std::nullopt`), and their python ctors take exactly those fields. So they are
rebuilt field by field from that spelling (`graph_case.build_conv2d_config` /
`build_slice_config`) rather than through `_PROGRAM_CONFIG_FIELDS`. Any field the
runtime cannot convert skips the case instead of falling back to a ctor default.
`Conv2dConfig.activation` (a `UnaryWithParam`) and `ttnn.add(..., activations=[UnaryWithParam(RELU)])`
are parsed into a new `uwp` argument kind.

**`MatmulMultiCoreReuseMultiCast1DProgramConfig`** now also maps `gather_in0`,
`num_global_cb_receivers`, `untilize_out` and `stream_in1`, which the current repr prints. The
ctor defaults `num_global_cb_receivers` to 1, while the model's configs carry 0, so
dropping the field would have changed the config. `hop_cores` must be empty (its default),
or the case skips.

**L1_SMALL.** Halo and sliding-window config tensors live in the L1_SMALL region, so
`op_utils.with_default_mesh` opens the mesh with `l1_small_size=PDL_L1_SMALL_SIZE`, as
the model test does.

**Goldens added** (`graph_case.GOLDEN`): `conv2d` (torch `conv2d` on the NHWC activation
unflattened with the captured geometry, with the bias and the `Conv2dConfig.activation`
applied), `max_pool2d`, `avg_pool2d`, `upsample` (nearest), `permute`, `relu`, `matmul`,
`move` and `reshard` (identity), and the post-op `activations` of `add`. The stem conv's
input is `[1, 512, 1024, 8]` for `in_channels=3`. The model zero-pads RGB to 8 channels,
so the case does the same (`_conv_input`) and the golden is unambiguous.

**Case ids.** Conv, pool and upsample are called entirely by keyword, so the generator now
takes the id's shape/dtype/placement from the first tensor keyword argument
(`input_tensor`) when there is no positional one.

## Fidelity

The llama README's fidelity table applies: random tensor values (weights included),
`compute_kernel_config` dropped (conv2d's restored from the model source, see above),
replicated onto the `(1, 1)` mesh the capture ran on. Each generated file's docstring lists the gaps that apply to *that* op.

The capture came from **ttsim**, the functional simulator, not silicon. Op selection,
shapes and configs come from the host-side ttnn code and are the same either way. Only
the timing-dependent behaviour of the run itself is not represented.
