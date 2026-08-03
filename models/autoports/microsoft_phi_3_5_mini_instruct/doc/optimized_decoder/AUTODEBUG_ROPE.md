# AutoDebug: Phi-3.5 head-96 RoPE

## Starting evidence

The optimized decoder currently implements exact HF rotate-half with ordinary
TTNN operations:

```text
slice first 48 + slice second 48 + neg + concat + 2 multiply + add
```

Decode first moves Q and K from the height-sharded output of
`nlp_create_qkv_heads_decode` to DRAM interleaved, then moves both results back.
The b1 profile confirms that the non-tile-aligned 48-wide slices/concat expand
into `Untilize`, `UntilizeWithUnpadding`, and `TilizeWithValPadding` helpers.
The stage documentation currently rejects fused RoPE only because 96 is not
divisible by 64. That rejection is incomplete: a correct adapted 128-wide HF
representation exists, and a lower-movement adjacent-pair representation exists
for `rotary_embedding_llama`.

This investigation was source-only. It did not reserve or use TT hardware and
did not edit implementation or test files.

## Exact TTNN contracts

### `ttnn.experimental.rotary_embedding_hf`

- All inputs are device tensors in TILE layout.
- Padded head width must be 32 or divisible by 64. The validation is in
  `rotary_embedding_hf_device_operation.cpp:29-53`.
- Prefill is documented/implemented as input `[1, heads, seq, D]` with cos/sin
  `[1, 1, seq, D]`; decode is input `[1, batch, heads, D]` with cos/sin
  `[1, batch, 1, D]`.
- Decode input must be HEIGHT_SHARDED, and cos/sin must be sharded
  (`rotary_embedding_hf_device_operation.cpp:55-68`).
- The multi-tile kernel hard-codes `half_Wt = Wt / 2` and exchanges whole
  width tiles (`rotary_embedding_hf_sharded.cpp:23-27,55-79`). It cannot put
  the midpoint inside a tile.

Therefore a logical/padded width of 96 (`Wt=3`) is rejected. Naively appending
32 zeros to make width 128 changes the rotate-half midpoint from 48 to 64 and
is numerically wrong.

### `ttnn.experimental.rotary_embedding_llama`

- All input, cos, sin, and transformation tensors must be BF16 TILE tensors.
- Head width may be up to 256; decode only requires it to be a multiple of 32.
  Thus head width 96 is legal
  (`rotary_embedding_llama_device_operation.cpp:65-83`).
- Decode requires HEIGHT_SHARDED input/cos/sin/transformation tensors, input
  shape `[1, batch, heads, D]`, and at most one batch row per available core
  (`rotary_embedding_llama_device_operation.cpp:85-128`).
- Prefill requires interleaved input; head-broadcast cos/sin are supported
  (`rotary_embedding_llama_device_operation.cpp:129-180`).
- The standard 32x32 transformation matrix performs adjacent-pair rotation:
  row-vector `[a,b] -> [-b,a]`
  (`models/common/tensor_utils.py:19-40`).

Existing integration examples are
`models/tt_transformers/tt/attention.py:_mllama_rope_decode`,
`_mllama_rope_prefill`, and
`models/tt_transformers/tt/rope.py:RotarySetup`.

### Required decode cos/sin shape

`nlp_create_qkv_heads_decode` returns Q/K with logical shape
`[1, batch, heads, D]`
(`nlp_create_qkv_heads_decode_device_operation.cpp:114-156`).
The current optimized decoder reshapes embedded cos/sin to
`[1, 1, batch, 96]`. That broadcasts per-position rows over the batch axis as
if they were head rows when batch=heads=32. Both fused APIs require
`[1, batch, 1, D]`; the repo reference integration explicitly performs the
transpose (`models/tt_transformers/tt/rope.py:703-714`).

The existing b32 trace test does use different positions, but its zero-prefix
reference is insensitive to this mistake: applying the same wrong orthogonal
rotation to the current Q and K preserves their dot product. A sequential
nonzero-cache test with different old/new positions is needed.

## Hypothesis experiments

### H1: simple right padding to 128 preserves Phi rotate-half

Prediction: refuted, because the fused midpoint becomes 64.

Source-only experiment:

```bash
python - <<'PY'
import torch
def rh(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
torch.manual_seed(96)
x, c, s = (torch.randn(2, 3, 96) for _ in range(3))
ref = x*c + rh(x)*s
xp, cp, sp = (torch.nn.functional.pad(v, (0, 32)) for v in (x, c, s))
got = (xp*cp + rh(xp)*sp)[..., :96]
print((got-ref).abs().max().item(), torch.equal(got, ref))
PY
```

Result: `9.490667343139648 False`.

Verdict: **refuted**.

### H2: two 48-wide halves packed into two 64-wide halves preserve semantics

Define:

```text
pack96(x)   = [x[0:48], zero[16], x[48:96], zero[16]]
unpack96(y) = [y[0:48], y[64:112]]
```

The 128-wide fused midpoint now lies between the two padded logical halves.

Source-only experiment used random `[2,3,96]` input/cos/sin and compared
`unpack96(fused_hf_math(pack96(...)))` with exact HF rotate-half.

Result: `adapted_max_abs 0.0`, `adapted_equal True`.

Verdict: **verified mathematically**. Device legality/performance remains to be
measured.

The smallest TTNN helper is:

```python
def pack96(value, memory_config):
    shape = list(value.shape)
    first = ttnn.slice(value, [0, 0, 0, 0], [*shape[:-1], 48])
    second = ttnn.slice(value, [0, 0, 0, 48], [*shape[:-1], 96])
    first = ttnn.pad(first, [(0, 0), (0, 0), (0, 0), (0, 16)], 0.0)
    second = ttnn.pad(second, [(0, 0), (0, 0), (0, 0), (0, 16)], 0.0)
    return ttnn.concat((first, second), dim=-1, memory_config=memory_config)

def unpack96(value, memory_config):
    shape = list(value.shape)
    first = ttnn.slice(value, [0, 0, 0, 0], [*shape[:-1], 48])
    second = ttnn.slice(value, [0, 0, 0, 64], [*shape[:-1], 112])
    return ttnn.concat((first, second), dim=-1, memory_config=memory_config)
```

For TILE tensors, padding each logical 48-wide slice by 16 promotes its already
64-wide padded storage to logical width 64; `ttnn.pad` fills implicit padding
then views it (`pad.cpp:269-326`). The final 48+48 unpack concat is still
non-aligned and deliberately triggers untilize/unpadding/retilize
(`concat.cpp:93-139`). The adapted HF path therefore cannot eliminate every
layout helper.

Cos/sin should be packed once when the short/long tables are created, not on
each invocation. Decode must reshape/transpose embedded rows to
`[1,batch,1,128]`, then place input and cos/sin in matching HEIGHT_SHARDED
configs with shard widths 128. Prefill batch 32 should reshape packed
`[batch,heads,seq,128]` to `[1,batch*heads,seq,128]` for the documented prefill
contract and restore the shape afterward.

### H3: a load-time adjacent-pair basis permits fused head-96 RoPE with no
runtime pack/unpack

Phi tables are generated as `cat(freqs, freqs)`. Let

```python
pair_index = torch.stack(
    (torch.arange(48), torch.arange(48, 96)), dim=-1
).flatten()
```

Applying `pair_index` to Q/K output coordinates and cos/sin table coordinates
turns each canonical half pair `[a_i,b_i]` into adjacent `[a_i,b_i]`.
The llama transformation matrix then produces `[-b_i,a_i]` directly. Q and K
can remain in this coordinate basis through cache and attention because the
same permutation is applied to both; V and the attended output remain in their
canonical coordinates.

Source-only experiment compared the adjacent-pair result, inverse-permuted only
for checking, against HF rotate-half at D=96.

Result:

```text
llama_basis_output_equal True
unrotated_dot_max_abs 9.5367431640625e-07
rotated_dot_max_abs 1.9073486328125e-06
```

The tiny dot differences are reduction-order effects after permutation, not a
semantic mismatch.

Verdict: **verified mathematically and supported by source contracts**. This is
the preferred device experiment because it can remove the runtime 48-wide
slice/concat topology instead of adding another pack/unpack topology.

## Proposed smallest implementation experiment

Add a temporary `rope_mode` policy with `manual`, `llama_interleaved`, and
`hf_padded128`; keep only a proven winner.

For `llama_interleaved`:

1. Before transposing/loading the QKV weight, reshape canonical
   `[3*hidden,hidden]` to `[3,heads,96,hidden]`, apply `pair_index` to the
   head coordinate of Q and K only, leave V unchanged, then restore shape.
2. Apply the same `pair_index` to all short/long cos/sin tables at load time.
3. Create one BF16 TILE interleaved 32x32 adjacent-pair transformation tensor
   for prefill. Create the decode form by repeating it across the fixed decoder
   batch and HEIGHT-sharding it like `RotarySetup`.
4. Prefill: call `rotary_embedding_llama(..., is_decode_mode=False)` directly
   on interleaved Q and K. No head-width padding is needed.
5. Decode: embedding produces rows which must become `[1,batch,1,96]` via the
   same transpose used by `HfRotarySetup`; shard cos/sin on the Q/K batch grid
   with shard `(32,96)`, then call
   `rotary_embedding_llama(..., is_decode_mode=True)` directly on the existing
   height-sharded Q and K. Do not cross Q/K to DRAM.
6. Keep K in the adjacent-pair basis in the paged cache. Prefill and decode
   must use the same basis. No inverse permutation is needed in runtime.

If that API path fails after adapting tensor/grid contracts, run the
`hf_padded128` experiment described in H2. A first TTNN error is not a rejection:
log logical/padded shapes, dtypes, layouts, shard grids/shapes, and try the
matching interleaved or HEIGHT_SHARDED form required above.

## Verification required from the main agent

1. Add a focused RoPE-boundary test at b1 and b32 with non-contiguous,
   per-user positions. Test-only inverse permutation may compare internal
   interleaved Q/K to the HF oracle.
2. Add a two-step or prefill-then-decode test with nonzero cached K/V and
   different previous/current positions. This detects the current
   `[1,1,batch,D]` broadcast blind spot.
3. Run the full optimized correctness suite, including long RoPE, non-aligned
   prefill, trace replay, and real weights.
4. Measure warmed prefill and traced decode at b1 and b32 for `manual`,
   `llama_interleaved`, and (if needed) `hf_padded128`. Batch-1 decode remains
   the primary gate and b32 may not regress.
5. Collect a focused Tracy/`tt-perf-report` window. For the preferred path the
   old Q/K `Slice`, `Neg`, `Concat`, multiply/add, DRAM-crossing, and
   concat-generated untilize/tilize cluster should be gone and replaced by two
   fused rotary rows.
6. Run watcher correctness separately after selecting the path.

## Final status

The earlier “width 96 is unsupported, so retain the composite” conclusion is
not sufficient.

- Direct 96-wide `rotary_embedding_hf`: **source-level blocker** (midpoint must
  be on a 32-wide tile boundary).
- Naive right-pad to 128: **refuted**.
- Structured 48→64 half packing around the 128-wide HF op: **mathematically
  verified**, but necessarily retains non-aligned unpack/layout work and needs
  device A/B evidence.
- Load-time adjacent-pair Q/K/cos/sin basis plus legal 96-wide
  `rotary_embedding_llama`: **mathematically verified and preferred for device
  A/B**, because it preserves context capacity and can remove the avoidable
  runtime movement.

No implementation choice is proven until the focused hardware correctness,
performance, profiler, trace, and watcher gates above pass.
