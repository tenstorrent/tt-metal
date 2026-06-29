# Pre-existing bug: `untilize_multi_core_block` corrupts partial-tile-row × width-cliff corner

**Status:** pre-existing on `origin/main` (NOT a Metal 2.0 migration regression). Introduced by the
ProgramDescriptor migration **#43840** ("Migrate data_movement family (26 ops)"). `untilize` has **not**
been ported to Metal 2.0 — it is pure descriptor-path code, byte-identical between this branch and main.

## Symptom
`tests/ttnn/unit_tests/operations/data_movement/test_tilize_with_val_padding.py::test_run_tilize_large_row_input`
fails for `(48, 5210112)` and `(180, 5210116)` with `Max ATOL Delta ~5.6`. Passes for `(32, 15916)` and
`(16, 5210112)`. The test is a TILE↔ROW_MAJOR round-trip; the failing leg is the **device untilize**, not tilize.

## Isolation (device, single op)
- Forward tilize → `to_torch`: **atol 0.0** (bit-exact), including huge widths.
- Untilize a host-built TILE tensor (`from_torch(layout=TILE)` then `to_layout(ROW_MAJOR)`) — **no migrated
  factory in the path** — reproduces the failure. So tilize and the Metal 2.0 framework are not involved.

## Error signature (the smoking gun)
For `(48, 5210112)` (48 rows → padded to 64 = two tile-rows; 162816 tile-cols):
- **Wrong rows: exactly 32–47** — the *second*, partially-padded tile-row. Rows 0–31 are perfect.
- **Wrong cols: exactly the last 768** (cols 5209344–5210112 = 24 tiles = the **width-cliff** block).
- `(64, 5210112)` — identical huge width but *full* tile-rows (64/32=2, no padding) — **passes**.

=> The bug lives in `untilize_multi_core_block`'s handling of the **(width-cliff block) × (non-first /
partial tile-row)** corner. Not width-dependent per se; it's the partial last tile-row combined with the
column cliff. A program-cache / runtime-args defect cannot produce this surgical pattern (it would corrupt
addresses wholesale), which is further proof the Metal 2.0 adapter is not the cause.

## Repro
```python
import torch, ttnn
dev = ttnn.open_device(device_id=0)
x = torch.randn((48, 5210112), dtype=torch.bfloat16)
tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
rm   = ttnn.to_layout(tile, layout=ttnn.ROW_MAJOR_LAYOUT)   # device untilize
back = ttnn.to_torch(rm)[:48, :5210112]
print((back - x).abs().max())   # ~5.6 ; should be 0.0
```

## Notes for the fix (separate effort — out of Metal 2.0 scope)
- Factory: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp`
  (+ its dataflow kernels). Routed from `untilize_device_operation.cpp` `select_program_factory` for
  `num_tiles_per_row > 32` block-split cases.
- Compare the per-core block-size / start-id arithmetic for the **last column block** when the core also
  owns the **partial last tile-row** against the legacy pre-#43840 implementation.
- Test is not referenced in any CI yaml, which is why this regression went unnoticed on main.
