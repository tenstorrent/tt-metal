# Per-op tests (`tests/ops/`)

Isolated tests for **every individual ttnn op** the model executes on the
token-accuracy path
(`tests/demos/llama32_1b/demo.py -k "token-accuracy"`).

## Why

The end-to-end demo is extremely slow on the Quasar emulator. These tests let you
exercise a single op — with the exact shapes / dtypes / memory configs the model
uses — in isolation, e.g.:

```bash
MESH_DEVICE=N150 pytest models/experimental/llama32_1b_quasar/tests/ops/test_linear.py -v
```

One file per op. Each file's docstring cites the model call site(s) (`file:line`)
that the parametrizations are derived from.

## Conventions

- Shared dims/helpers live in `op_utils.py` (imported as `U`). Do not re-hardcode
  Llama-3.2-1B dims — use `U.DIM`, `U.N_HEADS`, `U.HEAD_DIM`, etc.
- `test_rms_norm.py` is the **canonical template**; new op files mirror it.
- Tests use the `ttnn_mesh_device` fixture (from `tests/conftest.py`), parametrized
  `indirect`; default `[(1, 1)]` for the emulator. Multi-device ops (all_gather,
  reduce_scatter, …) parametrize `[(1, 2)]` and skip on a single device.
- Correctness: `U.assert_pcc(torch_ref, tt_out)` where a torch reference exists;
  otherwise `U.assert_shape_dtype(tt_out, shape=…, dtype=…)` (shape/dtype/finite).

## Status

Authored from static analysis of the model source (no device was available at
authoring time). Comments marked `# TODO: verify on device` flag op signatures /
sharded configs that need a first confirming run on hardware or the emulator.
