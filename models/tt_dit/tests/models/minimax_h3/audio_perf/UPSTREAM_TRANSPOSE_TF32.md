# fp32 `ttnn.transpose` on ROW_MAJOR silently truncates to TF32 (and takes `ttnn.concat` with it)

> **Note:** some `audio_perf/` scripts cited below were removed once their conclusions were
> captured here and in `ITEM1_RESULT.md` / `ITEM2_RESULT.md`. Recover any of them with
> `git log -- models/tt_dit/tests/models/minimax_h3/audio_perf`. See `README.md` for what survives.

Draft for an upstream issue. Found while chasing an accuracy divergence in the MiniMax-H3 audio
decoder; the audio angle is incidental, the defect is general.

## Summary

`ttnn.transpose(x, -2, -1)` on a **float32 ROW_MAJOR** tensor returns `x & 0xFFFFE000` — the low 13
mantissa bits zeroed. That is TF32 (1 sign + 8 exponent + 10 mantissa), applied by **truncation, not
round-to-nearest**. It happens at every shape tested, not just awkward ones.

`ttnn.concat(..., dim=-1)` inherits this for float32 inputs whose row length in bytes is not a
multiple of the buffer alignment (64 B on Blackhole), because
`build_non_aligned_last_dim_concat` (`ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp:186`)
routes those through a `ttnn.transpose(-2,-1)` round trip.

Both ops are pure data movement. Neither documents a precision loss, and neither warns.

## Severity

Silent. No exception, no log line, no dtype change — the output tensor still reports `float32`. Any
fp32 model doing a row-major transpose loses 13 bits of mantissa with no indication.

It is easy to miss end-to-end even when testing against a CPU reference: a ~1e-03 perturbation sits
well under the tolerance any whole-model gate has to allow. In our case the audio decode passes at
42.9 dB PSNR against a 28 dB threshold, so the loss was invisible there and only showed up when a
single op was compared against CPU on its own.

## Reproduction

`models/tt_dit/tests/models/minimax_h3/audio_perf/transpose_tf32.py`. Minimal form:

```python
import torch, ttnn
device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
x = torch.randn(1, 64, 64)
xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
got = ttnn.to_torch(ttnn.transpose(xd, -2, -1)).float()
ref = x.transpose(-2, -1).contiguous()

assert torch.equal(got, ref)                       # fails, maxdiff 3.893e-03
mask = (ref.view(torch.int32) & torch.tensor(-8192, dtype=torch.int32)).view(torch.float32)
assert torch.equal(got, mask)                      # passes -- exactly fp32 & 0xFFFFE000
```

Measured (Blackhole, 32 chips, fp32):

| op | layout | shape / width | result |
|---|---|---|---|
| `transpose(-2,-1)` | ROW_MAJOR | (1,1024,8), (1,1024,16), (1,64,64), (1,32,32) | **all `== ref & 0xFFFFE000`** |
| `transpose(-2,-1)` | TILE | same four | exact |
| `transpose(-2,-1)` | ROW_MAJOR, bf16 | (1,1024,8), (1,1024,16) | exact |
| `concat(dim=-1)` | ROW_MAJOR | C=8 (32 B row), C=24 (96 B) | **corrupt**, 3.9e-03 / 1.9e-03 |
| `concat(dim=-1)` | ROW_MAJOR | C=16 (64 B), C=32 (128 B) | exact |
| `concat(dim=-1)` | ROW_MAJOR, bf16 | C=8, C=24 | exact |

The bf16 rows are the control: bf16 has 7 mantissa bits, so a TF32 mask cannot change it, and indeed
nothing changes. The concat pass/fail split lands exactly on `row_bytes % 64 == 0`, which is the
predicate at `concat.cpp:195`.

Attribution checks, all clean, ruling out the alternatives:

- `from_torch` → `to_torch` round trips are exact at C=8/16/32 in both layouts, so upload/download is
  not the source.
- The affected values are not relocated source data (0 / 1024 wrong values occur anywhere in the
  correct row), so it is numeric conversion rather than a copy-offset bug.
- The output is *not* bf16-rounded — it sits further from `bf16(ref)` than from `ref`.

## Suspected mechanism

Row-major transpose goes through the matrix engine, whose SrcA register holds fp32 operands at TF32
precision. The tiled path avoids it, hence TILE being exact. Not confirmed against the kernel — the
bit pattern is the evidence.

Tangentially, `reader_concat_stick_layout_interleaved_start_id.cpp:49` carries a bare
`// FIX RM CONCAT WIDTH` comment directly above the `WIDTH_CONCAT` loop, so the width-concat path was
already regarded as unfinished. That loop is not itself the culprit here (its CB is `Float32` and it
is pure NOC copy), but it is adjacent.

## Suggested resolution

In rough order of preference:

1. Make row-major fp32 transpose exact — route it through a data-movement path rather than the matrix
   engine, or through the tiled path that is already correct.
2. Failing that, have `concat`'s non-aligned fallback avoid transpose for fp32 (padding to an aligned
   width, or `ttnn.pad`-style stick writes, both work).
3. At minimum, make the loss loud: document it, and warn when an fp32 tensor takes either path.

## Workarounds in the meantime

- `ttnn.pad` instead of concat-against-zeros. Exact at every width, and **4–30× faster** here
  (0.08 ms vs 2.38 ms at C=8, on (2, 20701, C)).
- `ttnn.repeat` instead of concatenating N copies. Exact at every width tested.
- Keep fp32 transposes in TILE layout.
- Pad the channel count so the row is a multiple of 64 B before any last-dim concat.
