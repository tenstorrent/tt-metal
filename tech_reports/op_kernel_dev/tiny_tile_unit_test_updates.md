# Updating Unit Tests for Tiny-Tile Support

This guide describes how to modify an existing TTNN unit test so it also exercises
**tiny tiles** once the op itself has tiny-tile support (see
[`tiny_tile_support_for_ops.md`](./tiny_tile_support_for_ops.md) for the op/kernel side).

The reference implementation is
`tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py`.
Every other test listed at the bottom of this doc should follow the same pattern.

---

## Guiding principles

1. **Do not add new tests.** No new test functions, no new `parametrize` rows, no
   new dtype/shape combinations. Tiny tiles are folded into the cases that already
   run.
2. **Tiny tiles only for `bfloat16` and `float32`.** Blocked formats
   (`bfloat8_b`, `bfloat4_b`) and any non-float dtype must keep the standard
   `32×32` tile. A case that mixes dtypes uses a tiny tile only when **every**
   dtype involved supports it.
3. **Tiny tiles only for `TILE_LAYOUT`.** `ROW_MAJOR_LAYOUT` has no tile concept,
   so it always resolves to the standard tile.
4. **Keep the diff minimal.** The intended change per test is one added line that
   computes a tile, plus `tile=<tile>` threaded into the existing `from_torch`
   call. Do not restructure tests, rename variables, or touch assertions.
5. **Share the common code.** The dtype gate and tile selector are moved into a
   shared utility instead of being copy-pasted into each test file.

---

## What "tiny tile" means for tests

- Standard tile: `ttnn.Tile((32, 32))`.
- Tiny tile: **height smaller than 32, width still 32** — e.g. `ttnn.Tile((8, 32))`
  or `ttnn.Tile((16, 32))`. Non-32 widths are out of scope.
- A tensor is created with a specific tile by passing `tile=` to `ttnn.from_torch`.

The choice of tile is data driven: given the dtype(s) and layout a case already
uses, we deterministically pick tiny vs. standard. This is why no new parametrize
rows are needed — the existing rows now transparently cover the tiny-tile path
wherever their dtype/layout allows it.

---

## Step 1 — Add the shared utility functions

These two helpers live in `tests/ttnn/utils_for_testing.py` (importable from both
`tests/ttnn/...` and `tests/tt_eager/...`). Import them from there; do not
redefine them locally in each test file.

```python
def dtype_supports_tiny_tile(dtype):
    return dtype == ttnn.bfloat16 or dtype == ttnn.float32


def select_tile(*dtypes, layout=ttnn.TILE_LAYOUT):
    if layout == ttnn.TILE_LAYOUT and all(dtype_supports_tiny_tile(dtype) for dtype in dtypes):
        return ttnn.Tile((8, 32))
    return ttnn.Tile((32, 32))
```

Behavior:

- `select_tile(dtype)` — single-dtype case.
- `select_tile(in_dtype, out_dtype)` — dtype-conversion case; tiny only if both qualify.
- `select_tile(dtype, layout=layout)` — parametrized-layout case; standard for row-major.
- Returns a tiny `8×32` tile only when the layout is `TILE_LAYOUT` **and** all
  supplied dtypes are `bfloat16`/`float32`; otherwise the standard `32×32` tile.

In each test file, import them:

```python
from tests.ttnn.utils_for_testing import select_tile
```

(`dtype_supports_tiny_tile` usually does not need to be imported directly; import
it only if a test needs to gate on it explicitly.)

---

## Step 2 — Apply the mechanical edit in each test

For every tensor the op tiles, pick the tile from the case's own dtype/layout and
thread it into `from_torch` (or `ttnn.Tensor(..., tile=tile)`).

**Before:**

```python
torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
```

**After:**

```python
tile = select_tile(dtype, layout=layout)
torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout, tile=tile)
```

If the test builds tensors with `ttnn.Tensor(...)` instead of `from_torch`, pass
the same `tile` there (and prefer setting `layout=` at construction rather than
a follow-up `.to(layout)`):

```python
tile = select_tile(tt_dtype, layout=input_layout)
tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype, layout=input_layout, tile=tile)
```

Pick the `select_tile` arguments to match what the case actually uses:

| Case in the test                         | Call                                       |
| ---------------------------------------- | ------------------------------------------ |
| Single dtype, fixed `TILE_LAYOUT`        | `select_tile(dtype)`                       |
| Single dtype, parametrized `layout`      | `select_tile(dtype, layout=layout)`        |
| Two dtypes (e.g. input + output/convert) | `select_tile(in_dtype, out_dtype)`         |
| Two dtypes, parametrized `layout`        | `select_tile(in_dtype, out_dtype, layout=layout)` |
| Fixed `bfloat16` tensor                  | `select_tile(ttnn.bfloat16)`               |

---

## Special cases

### Tests that use `allocate_tensor_on_device` + `copy_host_to_device_tensor`

`allocate_tensor_on_device` does not take a tile cleanly. Collapse the
allocate/copy pair into a single `from_torch` that carries `tile`, `device`, and
`memory_config` (this is exactly what the reference `test_..._hash` test did):

**Before:**

```python
input_tensor = ttnn.from_torch(input_tensor_torch, first_dtype, layout=ttnn.TILE_LAYOUT)
input_tensor_device = ttnn.allocate_tensor_on_device(
    input_tensor.shape, input_tensor.dtype, input_tensor.layout, device, input_memory_config
)
ttnn.copy_host_to_device_tensor(input_tensor, input_tensor_device)
```

**After:**

```python
tile = select_tile(first_dtype, second_dtype)
input_tensor_device = ttnn.from_torch(
    input_tensor_torch,
    first_dtype,
    layout=ttnn.TILE_LAYOUT,
    tile=tile,
    device=device,
    memory_config=input_memory_config,
)
```

### Dtype-conversion paths

When the op converts dtype (e.g. input `bfloat16` → output `bfloat8_b`), pass
**all** dtypes that touch a tiled buffer to `select_tile`, so a blocked output
dtype correctly forces the standard tile even if the input is `bfloat16`.

### Row-major cases

Leave them alone functionally — `select_tile(..., layout=ttnn.ROW_MAJOR_LAYOUT)`
returns the standard tile, and `from_torch` still accepts `tile=`. If a test
never creates a tiled tensor at all, no change is needed.

### `from_torch` calls for auxiliary tensors that the op does not tile

Only thread `tile` into tensors that are actually tiled by the op under test. Do
not add `tile=` to `uint32` index tensors, row-major helper tensors, or golden
inputs that never reach the tiled buffer.

---

## Step 3 — Verify

- Run the modified test file; the existing `32×32` cases must stay green.
- Confirm the tiny-tile path actually executes for the `bfloat16`/`float32` +
  `TILE_LAYOUT` rows (e.g. temporarily assert the tile height is `8` for one such
  case, then remove the probe).
- Confirm blocked-dtype and row-major rows still run on the standard tile.

---

## Ops and their primary unit tests

Apply Steps 1–2 to the primary test for each op; the "also" / nightly files use
the identical pattern.

| Op | Primary unit test |
| --- | --- |
| `interleaved_to_sharded` | `tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py` (reference) |
| `sharded_to_interleaved` | `tests/ttnn/unit_tests/operations/data_movement/test_sharded_to_interleaved_oob.py`; also `test_core.py`, `tests/ttnn/unit_tests/base_functionality/test_to_memory_config.py` |
| `reshard` | `tests/tt_eager/python_api_testing/unit_testing/misc/test_reshard.py`; also `test_nd_reshard.py`, `test_core.py` |
| `reshape_view` | `tests/ttnn/unit_tests/base_functionality/test_reshape.py`; nightly `test_tm_reshape.py`, `test_universal_input_tm_reshape.py` |
| `copy` | `tests/ttnn/unit_tests/base_functionality/test_copy.py`; nightly `test_copy_ops.py` |
| `unary` | `tests/ttnn/unit_tests/operations/eltwise/test_unary.py`; also `test_unary_sharding.py`, `test_unary_ops_ttnn.py`, `test_unary_fp32.py`, other `test_unary_*.py` |
| `binary_ng` | `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py`; also `test_binaryng_ND.py`, `test_binaryng_fp32.py`, `test_binary_ng_program_cache.py`, `test_elt_binary.py` |
| `concat` | `tests/ttnn/unit_tests/operations/data_movement/test_concat.py`; also `test_concat_iterative.py`; nightly `test_concat.py`, `test_concat_block_sharded.py`, `test_concat_memory_configs_and_layouts.py` |
| `slice` | `tests/ttnn/unit_tests/operations/data_movement/test_slice.py`; also `test_slice_write.py`; nightly `test_universal_input_tm_slice.py` |
| `typecast` | `tests/ttnn/unit_tests/operations/eltwise/test_eltwise_typecast.py`; also `test_typecast_sharded.py`, `test_typecast_int.py` |

---

## Checklist per test file

- [ ] Import `select_tile` from `tests.ttnn.utils_for_testing`.
- [ ] For each tiled `from_torch` / `ttnn.Tensor(...)`, add `tile = select_tile(<dtypes>, layout=<layout>)` and pass `tile=tile`.
- [ ] Collapse any `allocate_tensor_on_device` + `copy_host_to_device_tensor` into a single `from_torch(..., tile=...)`.
- [ ] Include every dtype that reaches a tiled buffer in the `select_tile` call (conversion paths).
- [ ] No new tests, parametrize rows, or dtype/shape combinations added.
- [ ] Existing `32×32` and row-major coverage unchanged and green.
