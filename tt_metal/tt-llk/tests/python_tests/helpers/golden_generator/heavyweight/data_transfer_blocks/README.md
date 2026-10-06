<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `data_transfer_blocks` — what the hardware does to a buffer

Each block answers one question: **given a buffer sitting in L1, what does the
consumer see after the hardware has moved it?** Blocks take and return what the
hardware takes and returns, which means **L1 takes bytes**:

```
l1_bytes  ──l1_to_srcA──▶  src values  ──[math]──▶  Dest  ──dest_to_l1──▶  l1_bytes
```

Writing a test never calls into this package. It is reached through
[`operations/`](../operations/README.md), which turns these blocks into
chainable steps. You come here to change *what a transfer costs*, or to find out
what one already costs.

## Contents

| File | What is in it |
|---|---|
| `data_transfer_blocks.py` | `DataTransferBlocks` — the shared base, one method per boundary, plus the format model. It cannot be constructed without an architecture's format set. |
| `l1_codec.py` | `pack_to_l1` / `unpack_from_l1` — the bytes boundary |
| `pack_effects.py` | ReLU, edge masking, and what stochastic rounding costs us |
| `quasar_data_transfer/` | MX formats, no block float |
| `wormhole_data_transfer/` `blackhole_data_transfer/` | Block float, no MX |

---

# The one idea to take away

**There is no quantization step anywhere in this package.**

Storing a tensor as `MxFp8R` *is* the quantization. By the time the L1 bytes
exist the loss has already happened, and reading them back just reports it.

A golden that quantizes and *then* packs has run two rounding rules over the
same data and will disagree with silicon at every subnormal and every lattice
midpoint. If you are migrating a test that pre-quantizes its stimuli, delete
that step — the stimuli and the golden go through the same packer, so they land
on the same lattice by construction.

What is left is the loss each boundary takes — unpack into a src register, the
FPU's write to Dest, Dest fed back into a src register, and the packer's ReLU,
edge mask and requantization. Exactly one of those varies by architecture: **the
unpacker lands every L1 format in one of a small number of src-register storage
families, and the family — not the L1 format — is what the FPU reads.**

---

# `data_transfer_blocks.py`

## The blocks

| Method | Models |
|---|---|
| `l1_to_srcA(l1_bytes, l1_format, src_format=None, **geometry)` | The T0 unpack. Implemented in the base by delegating to `_l1_to_src`; what varies by architecture is the format metadata, and an architecture overrides this only if its unpack genuinely differs. |
| `l1_to_srcB(...)` | Delegates to `l1_to_srcA` — SrcA and SrcB share the datum layout. |
| `l1_to_srcS(l1_bytes, l1_format, src_format=None, *, dest_acc=False, **geometry)` | Not a delegate: SrcS has its own format rules (`srcs_format`). Float32 keeps full fp32 here, where SrcA truncates it to Tf32, and with `dest_acc` every float but Float16/Float16_b widens to Float32. `use_srcs=True` is defaulted into the geometry: SrcS uses a per-slice L1 layout, so the buffer must have been *packed* that way. |
| `l1_to_dest(l1_bytes, l1_format, dest_format, **geometry)` | Straight into Dest, bypassing the src registers — so the value keeps **more** mantissa than the same buffer read through `l1_to_srcA`. How a feedback loop starts. |
| `src_to_dest(values, dest_format, current=None)` | Where a math result lands. |
| `dest_to_srcA(dest_values, dest_format, src_format)` | The feedback path. Lossy; see the table below. |
| `dest_to_srcB(...)` | Same conversion. |
| `dest_to_l1(dest_values, l1_format, dest_format, *, relu_type, relu_threshold, edge_mask, stoch_rnd, **geometry)` | The T2 pack, mirroring `l1_to_srcA`. |
| `pack_to_l1` / `unpack_from_l1` | L1 access, format-checked against the architecture. |
| `supports(l1_format)` · `supported_dest_formats` | What this architecture can read, and hold in Dest. |

`src_format` on the unpack blocks overrides the storage format the unpacker
lands the buffer in; `None` lets the architecture choose. The override is a real
knob, but not an unrestricted one: which src formats an L1 format can land in is
a property of the unpacker, held per architecture in `UNPACK_TO_SRC_FORMATS` and
checked on **both** the explicit and the defaulted path. `Int32` has no legal
SrcA/SrcB target at all — it reaches Dest or SrcS only — and an integer input
does not become a float in a src register. An architecture that leaves the table
empty is unmodelled at this level and falls back to the weaker "is this a src
storage format" check; that is not a claim that every pair is legal.

## Two src storage families, and a name that lies

A src register datum is **19 bits: 1 sign + 8 exponent + 10 mantissa**, whatever
format is stored in it (`SRC_MANT_BITS = 10`).

```python
SRC_STORAGE_FORMATS = {Float16, Float16_b, Tf32}
```

Only two families exist in the datapath. **`Float16_b` is an alias for `Tf32`** —
the hardware converts it, and the distinction survives only in the row/column
mask path, which is not on the datum path.

> **Trap.** Nothing truncates a src mantissa to bf16's 7 bits. A
> `Float32 → Float16_b` src keeps **10** mantissa bits, not 7. The src format
> sets the exponent *range*, not the mantissa width. Precision in a src register
> is `min(input mantissa bits, 10)`.

### `_to_src_storage` — the two losses the unpacker takes

| src family | Mantissa | Exponent range |
|---|---|---|
| `Float16` | Truncate to 10 bits, **then** cast. Casting straight to fp16 would round, and the unpacker does not round. | 5-bit. Above it saturates to infinity; below the smallest normal flushes to zero, so fp16 subnormals never reach the register. |
| `Float16_b` · `Tf32` | Truncate to 10 bits. | fp32's range. Clips nothing. |
| `Float32` | Not a legal SrcA/SrcB target at all — a 19-bit datum cannot hold it, so `l1_to_srcA` refuses it. A `Float32` **input** lands as `Tf32`, `Float16` or `Float16_b`, all of which keep 10 bits. Full fp32 survives only through `l1_to_dest` and `l1_to_srcS`. | |
| integer | Passes through unchanged. | |

Truncation, never rounding: `_truncate_src_mantissa` calls
`truncate_mantissa(values, SRC_MANT_BITS)`, which masks off the low
`23 - 10 = 13` bits of the float32 mantissa.

## Dest storage

```python
DEST_STORAGE_FORMATS = {Float32, Int32, Float16, Float16_b, Int16, Int8, UInt8}
DEST_32_BIT_FORMATS  = {Float32, Int32}
```

Dest is **32-bit when accumulation is enabled and 16-bit otherwise** — that is
the whole of what `DestAccumulation` controls. `Tf32` has no separate Dest
encoding; it lives in a `Float32` container.

`resolve_dest_format(dest_format, l1_input_format, dest_acc)` is what every
`run()` calls: it derives the Dest format from the input when none is given, and
otherwise checks the one supplied against `dest_acc` using
`DEST_32_BIT_FORMATS`. The two are not independent settings, so `Float32` with
`dest_acc=False` is rejected rather than run — unchecked it produces a
believable answer at the wrong precision, which looks like a maths bug further
down the chain.

### `src_to_dest(values, dest_format, current=None)`

The maths belongs to the operation; what belongs here is **where it lands**. A
Dest slot holds only what `dest_format` can represent, and the value is rounded
on the way **in**, not on the way out to L1.

`current` is Dest's existing contents, which a multi-pass op passes to
accumulate rather than replace — the FPU has an accumulate-enable bit, not a
second instruction. Because the rounding happens on this write, **it happens on
every pass**. Accumulating at full precision and rounding once at the end would
model an accumulator the hardware does not have.

### `_to_dest_storage` — the FPU writes no denormal

Narrows to the Dest dtype, then flushes anything below that format's smallest
normal. The rule it models: when the FPU's result is valid and the operation is
floating point, a zero exponent implies a zero mantissa.

> **Why this one matters disproportionately.** `torch.float16` *does* carry
> subnormals, so a naive cast keeps a value the hardware zeroed. The MX packer
> then rounds that survivor **up** onto the output lattice — so the device
> reports `0` and the golden reports the output format's smallest representable
> value. The signature in a log is unmistakable: a handful of datums out of
> thousands, PCC exactly `1.000000000`, and every diff is *golden = smallest
> lattice value, device = 0*.

Only `Float16` Dest meets the threshold in practice; the wider formats bottom
out near 2⁻¹²⁶.

### `dest_format_for(l1_input_format, dest_acc)`

Derives the Dest format from the **input** side, because that is what the
hardware does.

| Input | `dest_acc=False` | `dest_acc=True` |
|---|---|---|
| integer | the input format | `Int32` |
| float | 16-bit member of the src family — `Float16` if the src is `Float16`, else `Float16_b` | `Float32` |
| `Float32` · `Tf32` | **Raises.** The input alone does not decide the family: the device picks `Float16` or `Float16_b` from the *output* format (`infer_unpack_out`), and the two clip differently. Pass the `dest_format` the harness configured, or enable `dest_acc`. | `Float32` |

> **Dest follows the unpacker's src output, never the pack format.** An MX input
> unpacks to the 8-bit-exponent family, so Dest is bf16 *whatever you pack to*.
> This cannot be inferred from `out_format`, which is why `OpConfig.dest_format`
> is a required field.

### `_dest_to_src_storage` — the feedback path

Driven by the **Dest** format, not the src format. The src format is consulted
*only* to decide whether a wide Dest needs its exponent rebiasing. Every case is
bit-slicing, so nothing rounds: bits below the target width are dropped, not
folded in.

| Dest format | What reaches the src register |
|---|---|
| `Float16_b` | 7 explicit mantissa bits; the low 3 arrive as **zeros**. A bf16 Dest does not come back with a src register's full 10. |
| `Int16` | **Carried unchanged.** It shares the bf16 path, but the format label there is a transport trick rather than a conversion, so 257 comes back 257 — not rounded to 256 the way the row above would suggest. |
| `Float16` | Passes as fp16 — 10 mantissa bits, 5-bit exponent. |
| `Int32` | **Saturates to INT8**, clamped to ±127. A wide integer Dest cannot survive the trip, and the clamp is silent. |
| `Float32` · `Tf32` | 10 mantissa bits and the 8-bit exponent — unless the src register is `Float16`, in which case the exponent is rebiased and values below the fp16 normal range flush to zero. |

## Where an architecture plugs in

| Hook | Notes |
|---|---|
| `SUPPORTED_L1_FORMATS` | Every L1 access is checked against it, so asking Quasar for `Bfp8_b` raises rather than quietly quantizing to something the hardware cannot store. |
| `_src_format(l1_format)` | The L1 → src-register mapping. Override where an architecture diverges; `src_format()` does the support check first, because the base mapping is built from exponent-family predicates that would otherwise return a plausible answer for a format the hardware cannot read. |
| `UNPACK_TO_SRC_FORMATS` | Legal `L1 format -> src format` pairs, checked on both the explicit and the defaulted path. Empty means unmodelled at this level, not "everything is legal". |
| `EDGE_MASK_MASKED_WHEN_SET` | Edge-mask polarity: `True` where a set register bit masks the datum (Quasar), `False` where it keeps it (Wormhole/Blackhole). |
| `PACK_TO_L1_FORMATS` | Legal `Dest format -> L1 format` pairs for the packer, the mirror of `UNPACK_TO_SRC_FORMATS`. Empty means unmodelled. |
| `HAS_SRCS` | Whether the architecture has a SrcS register. `True` on Quasar only — Wormhole and Blackhole have no `UNP_S` unpacker and no `_is_srcs_32bit_mode_`, so `l1_to_srcS` and `srcs_format` refuse there rather than answering for a register that does not exist. |
| `PROMOTES_OUTLIER_TO_32_BIT_DEST` | Whether `TestConfig` forces `dest_acc` on for an exponent-B input with a `Float16` output. `True` everywhere except Quasar, which it skips by name, so on Wormhole/Blackhole a `Float16_b` Dest packed to `Float16` is refused — the device would have run a 32-bit Dest. |
| `l1_to_srcA` | Concrete in the base. Override only for an architecture whose unpack differs beyond its format metadata. |

### The format divide

| | L1 formats | Dest formats | Has | Lacks |
|---|---|---|---|---|
| Quasar | 15 | `Float16` `Float16_b` `Float32` `Int8` `Int16` `Int32` `UInt8` | MX — `MxFp8R` `MxFp8P` `MxFp4` `MxInt8` `MxInt4` `MxInt2` | **No block float.** No `MxFp4_2x_A/B` — src-register formats, never L1 formats. |
| Blackhole | 14 | `Float16` `Float16_b` `Float32` `Int8` `Int32` `UInt8` | Block float — `Bfp8` `Bfp8_b` `Bfp4_b` `Bfp2_b` | **No MX.** |
| Wormhole | 13 | `Float16` `Float16_b` `Float32` `Int8` `Int32` `UInt8` | Block float — `Bfp8` `Bfp8_b` `Bfp4_b` `Bfp2_b` | **No MX.** No `Fp8_e4m3` — Wormhole's only fp8 is Lf8 (e5m2). |

Quasar's L1 → src mapping in full:

| L1 format | → src | |
|---|---|---|
| `Float16` | `Float16` | 5-bit exponent family |
| `Fp8_e4m3` | `Float16` | L1-only encoding; the unpacker converts it |
| `Float16_b` | `Float16_b` | = Tf32 |
| `Float32` · `Tf32` | `Tf32` | 8-bit exponent, 10-bit mantissa |
| all six MX formats | `Float16_b` | **whatever you pack to** |
| `Int8` `Int16` `Int32` `UInt8` | unchanged | |

---

# `l1_codec.py`

A thin dispatch over the real codecs in `helpers.pack` and `helpers.unpack` —
one entry point each way. Deliberately the **same** codecs the test harness
writes device L1 with, which is what makes the bytes the golden reads identical
to the bytes the device read.

| Member | Notes |
|---|---|
| `PACKERS` | Format → packer, 19 entries, mirroring `StimuliConfig.get_packer`. `MODELLED_L1_FORMATS` is its key set — the formats this golden can actually move bytes for, which is **narrower** than an architecture's `SUPPORTED_L1_FORMATS`. `Tf32` (all) and `Bfp8` (WH/BH) are real L1 formats with no codec here, so `supports()` returns False for them and the error says it is a model gap, not a hardware limit. |
| `datums_per_tile(num_faces=4, face_r_dim=16)` | `num_faces × face_r_dim × 16`. The geometry primitive everything counts in. |
| `pack_to_l1(tensor, l1_format, *, tile_count=None, num_faces=4, face_r_dim=16, use_srcs=False, dest_acc=False)` | Where precision is lost for the block-scaled formats. |
| `unpack_from_l1(packed, l1_format, *, tile_count=None, tile_stride_bytes=None, num_faces=4, face_r_dim=16, use_srcs=False, dest_acc=False, twos_complement=False)` | Reads bytes back as values. |
| `_call_accepted(fn, tensor, **kwargs)` | Calls a packer passing only the keywords it declares. |

### `_call_accepted`, and the typo that does not raise

The packers take different subsets of the tile geometry — `pack_fp16` takes
none, `pack_bfp8_b` takes faces, the MX packers also take the SrcS layout flag
and extra rounding controls. Filtering by signature keeps one call site.

The filtering applies to the **codec functions**, not to the entry points.
`pack_to_l1` and `unpack_from_l1` both name every parameter keyword-only with no
`**kwargs`, so a misspelled geometry key raises `TypeError` on either path.
What `_call_accepted` drops is a *correctly spelled* key that a given packer
does not declare — `num_faces` handed to `pack_fp16`, say.

### Multi-tile layout

The underlying codecs handle exactly **one tile**, so `pack_to_l1` splits a
multi-tile tensor and packs tile by tile, matching how it is read back.
`pack_to_l1` also casts bfloat16 up to float32 first — losslessly, since numpy
has no bfloat16 and most packers go straight to `.numpy()`.

> **`tile_stride_bytes` defaults to `tile_bytes_for`**, what `pack_to_l1`
> actually writes at this geometry — which is not the dense datum count: the BFP
> packers hold a minimum of 16 exponents (48 bytes rather than 34 for a 1×32
> `Bfp8_b` tile), and `use_srcs` writes 16-byte-aligned slices (1152 bytes rather
> than 1056, or 1280 under `dest_acc`). Left to its own devices,
> `unpack_res_tiles` assumes a full 32×32 tile stride for backward
> compatibility — correct only when the geometry really is 32×32. Pass the
> device's stride explicitly when reading a buffer laid out some other way.

> **Two source strides, only one of which matches.** `Golden.run` lays tiles out
> back to back, `datums_per_tile` apart. `write_matrix_w_tile_dimensions`
> (`use_dense_tile_dimensions=True`) strides the source the same way, but the
> *default* `StimuliConfig.write_matrix` always strides by `MAX_TILE_ELEMENTS`
> whatever the tile size, writing only `num_faces × face_r_dim × 16` of each
> stride. They coincide only at 1024 datums per tile, or on a single tile where
> the stride never applies — below that, over several tiles, the two read
> different source elements from tile 1 on. `operations/golden.py`'s
> `check_source_layout` raises for those rather than silently computing on data
> the device never saw.

---

# `pack_effects.py`

The packer does more than convert format. Three knobs change the values that
reach L1, and the hardware applies them in this order:

```
Dest → ReLU → round → edge mask → format convert → L1
```

Rounding sits between ReLU and the mask in hardware, but the mask *replaces* a
datum outright, so masking before the format conversion gives the same bytes.
Two of the three are exactly modellable.

## `apply_relu(values, relu_type, threshold, dest_format)`

| `PackerReluType` | Behaviour |
|---|---|
| `NoRelu` | Pass through. |
| `ZeroRelu` | `relu(values)`. |
| `MinThresholdRelu` | Below the threshold is flushed; above it passes through **untouched**. |
| `MaxThresholdRelu` | Clamp into `[0, limit]`. |

The threshold is first narrowed by `_encode_threshold` to the 16 bits the
packer's configuration register actually holds — fp16 for the exponent-A family,
bf16 otherwise. **Comparing against the un-narrowed value is a real and easy
bug.**

## `PackEdgeMask(masks, select, mode)`

Masking at tile edges. Up to `EDGE_MASK_COUNT = 4` masks of
`EDGE_MASK_WIDTH = 16` bits, one bit per datum of a 16-datum row.

| Field | Notes |
|---|---|
| `masks` | The raw register values. **What a set bit means depends on the architecture**: on Quasar it masks the datum (the packer inverts the register before the gasket), on Wormhole/Blackhole it keeps it. The blocks resolve this through `EDGE_MASK_MASKED_WHEN_SET`, so `dest_to_l1` applies the right one. |
| `select` | Which mask each **row** uses: one index for every row, or one per row. The hardware selector is 2 bits per row, so a row cannot mix masks. `from_face_select_words(masks, words)` builds it from Quasar's four `EDGE_MASK_SELECT_FACE*` words. |
| `mode` | `EdgeMaskMode.ZERO`, or `NEG_SATURATE` — the latter exists so a masked datum *loses* a following max-reduce rather than winning it at zero. |

`keep(count, masked_when_set=...)` gives the boolean survival mask;
`apply(values, masked_when_set=...)` substitutes zero or `-inf`. Construction
validates mask count, bit width, selector range and mode.

## `is_deterministic(stoch_rnd)`

Stochastic rounding draws from a pseudo-random sequence seeded on device and
**cannot be reproduced here**. With it enabled the golden returns the
round-to-nearest result, which hardware matches only in expectation — each datum
may land one ULP of the output format either side. Check this and compare with
PCC rather than exactly.

Returns `True` for `No` only. `Fpu` randomises the FPU's write to Dest rather
than the packer's write to L1, so the packer behaves normally under it — but the
value reaching the packer is already off, which the golden can no more follow.
It only actually diverges when that Dest write has to round at all (a 16-bit
Dest, or a long enough accumulation), so a test that knows its Dest write is
exact can still compare exactly under `Fpu`; this predicate just will not make
that call for it. `STOCH_RND_EFFECTS` maps each mode to the stage it randomises,
which is what `dest_to_l1` puts in its warning.

---

# Changing a block

1. **Find the boundary, not the symptom.** A disagreement that is one lattice
   step wide at an MX output is usually several ULPs wide in Dest. Re-run with a
   `Float16` output first — a Float16 Dest packed to a Float16 output is a
   straight copy, so the device's result *is* its Dest, and you can compare in
   ULPs instead of lattice steps.
2. **Change the block, not the operation.** If a loss belongs to a transfer,
   every operation that crosses that boundary should inherit it for free.
3. **Validate bit-exactly, at LoFi.** A tolerance that passes tells you the
   tolerance is wide enough. Bit-exactness on every datum tells you the model is
   right — and then a later disagreement is a finding rather than noise.
4. **Check the other architectures still construct.** The base class is shared
   and the format sets differ; a change to the shared machinery can break
   Wormhole or Blackhole without touching a Quasar test.

## What this package does not model

- **Stochastic rounding** — see above.
- **A bit-for-bit model of the unpacker's conversion hardware.** The L1 → SrcA
  path is *composed* out of `pack_to_l1` plus the storage-precision rules,
  which agrees with silicon on everything measured here. Where a conversion
  rule is subtle it is written down as a rule, not reproduced as logic.
- **Partial-face multi-tile layout** — see the sharp edge above.
- **Integer SrcS under `dest_acc`.** `srcs_format` carries an integer input
  through unchanged, while `infer_unpack_out` — the rule the harness programs —
  returns `Float32` for it once `dest_acc` is on. That `Float32` is the
  fall-through after an fp16 carve-out and only the carve-out is justified in
  its comment, so this is an open question, not a settled rule. It is not
  cosmetic: `_is_srcs_32bit_mode_` keys on the same format, so it also selects
  the slice layout the buffer is read at. Nothing exercises integer SrcS against
  silicon on either side; if the harness's version turns out to be right, this
  method and the layout both have to change.
- **`Tf32` and `Bfp8` L1 bytes.** Neither has a codec here, on any
  architecture, so they raise an explicit "no L1 codec" rather than a bare
  `KeyError`. A gap in the model, not a statement about the hardware.
