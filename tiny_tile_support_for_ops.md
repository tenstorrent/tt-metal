# Adding Tiny-Tile Support to TTNN Ops

This guide generalizes the changes made to enable non-standard (tiny) tiles in the sharding data-movement ops (`interleaved_to_sharded`, `sharded_to_interleaved`, and `reshard`). Use it as a checklist when porting the same support to other ops.

**Reference commits (I2S / S2I / reshard):**

- Relax validation and replace hardcoded `TILE_HEIGHT` / `TILE_WIDTH` sizing
- Preserve page config in output specs; plumb `Tile` into CB descriptors
- Propagate tile-aware unit sizing through program factories (and shared helpers)
- Hash tile shape into the program cache; reject unsupported dtype × tiny-tile combos

---



## What “tiny tile” means here

- Standard tile: `32×32` (`tt::constants::TILE_HEIGHT` × `tt::constants::TILE_WIDTH`).
- Tiny tile (as enabled for these ops): **height may be smaller than 32** (e.g. `16×32`); **width remains 32**.
- Ops that only move data (no compute face packing assumptions beyond page size) can often support this with host-side sizing + CB tile metadata. Ops with compute kernels may need additional kernel / face-geometry work beyond this guide.

---



## Failure modes if you skip a step


| Mistake                                                           | Typical symptom                                                                                                |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `PageConfig(layout)` instead of `tensor_spec().page_config()`     | Output L1 bank / page size assumes 32×32; undersized vs CB sized from real tile → corruption or alloc failures |
| `tt::tile_size(format)` instead of `tile.get_tile_size(format)`   | Wrong page / unit size for tiny tiles                                                                          |
| Hardcoded `TILE_HEIGHT` / `TILE_WIDTH` / `TILE_HW` in unit counts | Wrong number of tiles per shard / row / height                                                                 |
| CB format descriptor without `.tile = TileDescriptor(...)`        | JIT `get_tile_size(cb)` / unpack strides fall back to 32×32 → L1 stride corruption                             |
| Program hash omits tile H/W                                       | Cache hit reuses a 32×32 program for a tiny-tile tensor                                                        |
| Validation still requires 32×32 height                            | Op rejects valid tiny-tile inputs before the factory runs                                                      |


---



## Checklist (apply in this order)



### 1. Validation: allow tiny height, keep width (and dtype) constraints

In `validate_inputs` (or equivalent):

1. **Stop requiring** `tile.get_height() == TILE_HEIGHT`.
2. **Keep requiring** `tile.get_width() == TILE_WIDTH` unless the op truly supports non-32 widths.
3. If the op (or its output dtype) uses **blocked formats** (`BFLOAT8_B`, `BFLOAT4_B`), reject tiny heights when those dtypes are involved. Prefer checking the dtype that actually lands in the tiled buffer the kernels touch (for I2S that was **output** dtype after conversion; for S2I, **input** dtype).
4. If a preallocated output exists, require `out_tile == in_tile`.

Example pattern:

```cpp
if (input_tensor.layout() == Layout::TILE) {
    auto tile = input_tensor.tensor_spec().tile();
    if (tile.get_width() != tt::constants::TILE_WIDTH) {
        return {false, fmt::format("op requires tile width {}, got {}",
                                   tt::constants::TILE_WIDTH, tile.get_width())};
    }
    if (tile.get_height() < tt::constants::TILE_HEIGHT &&
        (/* relevant dtype */ == DataType::BFLOAT8_B ||
         /* relevant dtype */ == DataType::BFLOAT4_B)) {
        return {false, "Tiny tile heights are not supported for blocked data types "
                       "like BFLOAT8_B or BFLOAT4_B"};
    }
    // optional: out_tile == tile for preallocated output
}
```

Update error strings so they no longer claim “requires standard 32×32” when only width is constrained.

### 2. Output specs: preserve the input page config

`PageConfig(layout)` defaults to a **32×32** tile. For tiny tiles that undersizes the output tensor relative to CBs sized from the real tile.

**Do this:**

```cpp
return TensorSpec(
    input_tensor.logical_shape(),
    TensorLayout(
        output_dtype,
        input_tensor.tensor_spec().page_config(),  // not PageConfig(layout)
        output_mem_config));
```

Or, when using `TensorLayout::fromPaddedShape`, pass `input_tensor.tensor_spec().page_config()` the same way.

This is required for both sharded and interleaved outputs that should inherit the input tile shape.

### 3. Program factory: derive all tile geometry from `tensor_spec().tile()`

Replace every use of global tile constants with the tensor’s tile:


| Old (32×32-only)                  | New (tile-aware)                        |
| --------------------------------- | --------------------------------------- |
| `tt::tile_size(data_format)`      | `tile.get_tile_size(data_format)`       |
| `shard_h / TILE_HEIGHT`           | `shard_h / tile.get_height()`           |
| `shard_w / TILE_WIDTH`            | `shard_w / tile.get_width()`            |
| `numel / TILE_HW`                 | `numel / tile.get_tile_hw()`            |
| `padded_w / TILE_WIDTH`           | `padded_w / tile.get_width()`           |
| `volume / padded_w / TILE_HEIGHT` | `volume / padded_w / tile.get_height()` |


Also:

- Assert shard shape (or shard numel) is divisible by the **actual** tile dimensions.
- Keep separate `input_tile` / `output_tile` when dtype conversion can change format but tile shape must still match.
- Audit **shared helpers** used by the op (e.g. slice starting index from tile grid) the same way—any `TILE_HEIGHT` / `TILE_WIDTH` there will break multi-slice or partial paths.



### 4. Circular buffers: attach `TileDescriptor` on TILE-layout CBs

For every CB that holds tiled data, set `CBFormatDescriptor::tile` from the tensor tile. Omitting it makes `CircularBufferConfig` / JIT assume 32×32 for `get_tile_size(cb)` and unpack strides.

```cpp
void push_cb(..., std::optional<Tile> tile = std::nullopt) {
    CBFormatDescriptor format_desc;
    format_desc.buffer_index = ...;
    format_desc.data_format = data_format;
    format_desc.page_size = page_size;
    if (tile.has_value()) {
        format_desc.tile = TileDescriptor(tile.value());
    }
    cb.format_descriptors.push_back(std::move(format_desc));
    // ...
}
```

Pass the tile into **all** TILE-layout CBs for the op (input, output, and scratch CBs that carry tiled pages). ROW_MAJOR paths can leave `tile` as `nullopt`.

Keep the changes to the existing code as minimal as possible. For example if format_desc is not explicitly instantiated in th original code, then keep it like that.

### 5. Program hash: include tile height and width

If the op overrides `compute_program_hash`, add tile dimensions so tiny-tile and 32×32 programs do not collide:

```cpp
const auto& tile = input_tensor.tensor_spec().tile();
return hash_operation<ThisOp>(
    /* existing fields */,
    tile.get_height(),
    tile.get_width());
```

If the op relies on the default hash, confirm tile shape is already covered by hashed tensor specs; if not, override and include it explicitly.

### 6. Kernels

For pure data-movement kernels that already use `get_tile_size(cb_id)` (or stick sizes from runtime args), **host-side CB tile metadata + correct page sizes are usually enough**—no kernel rewrite was required for I2S/S2I readers/writers once CBs carried the real tile.

Still verify:

- No hardcoded `32` / `TILE_HEIGHT` / `TILE_WIDTH` in the kernel for address math.
- Compile-time args that encode tile geometry are either unused or updated from the host tile.
- Compute kernels (pack/unpack, face layout, untilize/tilize) may need extra work; do not assume this checklist alone is sufficient for those ops.



### 7. Tests

- Keep existing 32×32 coverage green.
- Add / replay cases with tiny tiles (e.g. `16×32`) for the layouts and memory configs the op supports.
- Cover dtype-conversion paths separately from same-dtype paths.
- Confirm blocked dtypes with tiny height fail validation with a clear message (if that restriction applies).

---



## Suggested search terms when auditing an op

```text
TILE_HEIGHT
TILE_WIDTH
TILE_HW
tt::tile_size
PageConfig(
get_height() !=
requires standard 32x32
compute_program_hash
CBFormatDescriptor
push_*_cb
```

Touch every hit in the op’s device operation, program factory(ies), and any shared helpers it calls.

---



## Minimal change surface (typical data-movement op)

1. `*_device_operation.cpp` / `*_op.cpp` — validate, `compute_output_specs`, `compute_program_hash`
2. `*_program_factory*.cpp` — unit sizing, CB tile plumbing
3. Shared helpers used for tile-grid indexing
4. Unit / replay tests for tiny tile + regression for 32×32

Kernels often stay unchanged for copy-style ops once (1)–(3) are correct.

---



## Out of scope / follow-ups

- Non-32 **tile widths** (not enabled by the I2S/S2I work).
- Tiny tiles with blocked formats in paths that still reject them.
- Compute ops that depend on face geometry, untilize/tilize, or matmul tile assumptions—those need a separate kernel-level plan after this host/CB checklist.
