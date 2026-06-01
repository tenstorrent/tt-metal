# Fused Partial RoPE Device Op — Design

## Summary

Implement a new Blackhole ttnn C++ device op, `ttnn.experimental.fused_partial_rope`,
that fuses the entire partial-RoPE calculation currently done in Python by
`_apply_rope` in `models/experimental/deepseek_v4_flash/tt/attention.py`
(lines 151–165). The op takes a full `[.., D]` tensor, applies interleaved RoPE
to only the trailing `rope_dim` channels (via the existing `rotate_half`
matrix-matmul formulation), passes the leading "nope" channels through
untouched, and returns a full `[.., D]` tensor — all in a single device op.

## Motivation

`_apply_rope` currently issues a chain of primitive ttnn ops per call
(`slice` × 2, `matmul`, `multiply` × 2, `add`, `concat`), materializing several
intermediate tensors. This runs multiple times per decode step (q, kv, output,
compressor). Fusing into one device op removes the intermediates and host-side
op dispatch overhead on the traced-decode hot path.

## Reference semantics (must match exactly)

From `_apply_rope(x, cos, sin, rot, rope_dim)`:

- `D = x.shape[-1]`, `Rd = rope_dim`.
- If `D == Rd`: `nope = None`, `rope = x`.
- Else: `nope = x[..., :D-Rd]`, `rope = x[..., D-Rd:]`.
- `rotated = rope * cos + (rope @ rot) * sin`, computed at **HiFi4**.
- Output: `rotated` if no nope, else `concat([nope, rotated], dim=-1)`.

`cos` / `sin` are `[1, 1, L, Rd]` tables (broadcast over batch/heads, already
`repeat_interleave(2)`-expanded on host). `rot` is the `[Rd, Rd]` interleaved
`rotate_half` matrix.

## Concrete decode dimensions (deepseek_v4_flash)

- `head_dim (D) = 512` → 16 tiles wide.
- `qk_rope_head_dim (Rd) = 64` → trailing 2 tiles get RoPE.
- nope = 448 → leading 14 tiles pass through.
- `H (num_attention_heads) = 64`.
- `rot` = `[64, 64]` → 2×2 tiles.

Call-site shapes (all `B == 1`, `S == 1` decode):
- q: `[1, 1, H=64, 512]`
- kv: `[1, 1, 1, 512]`
- compressor: `[1, 1, n_win, 512]`

## Op identity & placement

- Name: `fused_partial_rope`, exposed as `ttnn.experimental.fused_partial_rope`.
- Directory: `ttnn/cpp/ttnn/operations/experimental/transformer/fused_partial_rope/`
- Structure mirrors the existing sibling op
  `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/`:
  - `fused_partial_rope.{hpp,cpp}` — op wrapper
  - `fused_partial_rope_nanobind.{hpp,cpp}` — python bindings
  - `device/fused_partial_rope_device_operation.{hpp,cpp}` — op + program factory
  - `device/kernels/dataflow/` — reader + writer
  - `device/kernels/compute/` — compute kernel
  - `sources.cmake` (+ CMake wiring)

Python signature:

```python
ttnn.experimental.fused_partial_rope(x, cos, sin, rot, rope_dim, memory_config=None) -> Tensor
```

## Data layout & sharding (v1 target)

- **Single-core, height-sharded L1** input and output (like the existing
  `_height_sharded_l1_config` helper in `attention.py`). One shard holds the
  full `[rows, 512]` block; shard width = 512 (16 tiles), height = padded rows.
- `cos` / `sin`: `[1, 1, rows, 64]`, height-sharded to the same grid (caller
  slices tables to `rows`).
- `rot`: `[64, 64]`, resident in L1 (tiny constant).
- Output: a new height-sharded tensor with the same config as input.
- TILE layout throughout.

Trade-off: single-core caps throughput but matches the fixed traced-decode
shape (H=64 → 2 tile-rows). Multi-core distribution is deferred to v2.

## Kernel plan

One program, three kernels on a single core:

- **Reader**: input/`cos`/`sin` shards are already resident (sharded-in); set up
  CBs over them and read `rot` into a CB.
- **Compute**:
  1. Copy the leading 14 nope tiles per row directly to the output shard.
  2. For the trailing 2 rope tiles per row: `mm = matmul_tiles(rope, rot)`
     (HiFi4), then `rotated = mul_tiles(rope, cos) + mul_tiles(mm, sin)`.
  3. Write `rotated` into the trailing 2 tiles of the output shard.
- **Writer**: commit output CB → shard (mostly a no-op with sharded output).

Compile-time args: `D`, `Rd`, `rows`, tile counts (nope tiles, rope tiles).
Compute config: `MathFidelity::HiFi4`.

Edge case: when `D == Rd` (no nope), skip the pass-through copy and rotate the
whole input.

## Testing & validation

- Unit test `models/experimental/deepseek_v4_flash/tests/test_fused_partial_rope.py`
  (or the ttnn unit-test tree): random `x` `[1,1,H,512]`, host `cos/sin/rot`,
  run the op vs a pure-torch reference of the reference semantics; assert
  PCC ≥ 0.999.
- Cover shapes: `H=64`, `rows=1` (kv), a compressor `n_win` row count, and the
  `D == Rd` no-nope edge case.
- Integration: gate the `_apply_rope` call sites behind a flag, swap in the op,
  run the decode demo to confirm parity.

## Non-goals (v1)

- Multi-core sharding / halo distribution.
- Wormhole support (Blackhole only).
- Native (non-matmul) rotate_half formulation.
- Prefill-optimized (large-seq) layouts.
