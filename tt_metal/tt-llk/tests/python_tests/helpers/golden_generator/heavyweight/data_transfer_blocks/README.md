<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `data_transfer_blocks`: what the hardware does to a buffer

Each block answers one question. **Given a buffer sitting in L1, what does the
consumer see after the hardware has moved it?** Blocks take and return what the
hardware takes and returns, which means **L1 takes bytes**:

```
l1_bytes  ──l1_to_srcA──▶  src values  ──[math]──▶  Dest  ──dest_to_l1──▶  l1_bytes
```

Writing a test never calls into this package. You reach it through
[`operations/`](../operations/README.md), which turns these blocks into
chainable steps. Come here to change what a transfer costs, or to find out what
one already costs.

## Contents

`DataTransferBlocks` is a **façade**. The transfers themselves live on one unit
per hardware stage in `units/`, and an architecture (`arch.py`) is a set of
those units injected into the façade. Nothing subclasses the façade and nothing
overrides a transfer. See [Where an architecture plugs in](#where-an-architecture-plugs-in).

| File | What is in it |
|---|---|
| `data_transfer_blocks.py` | `DataTransferBlocks`, the façade. One method per boundary, each a single delegation to a unit. |
| `arch.py` | `ArchGolden` and the three architectures, each assembling its units. |
| `l1_support.py` | `L1Support`: an architecture's L1 format set, the support checks, and the L1 read/write path. Shared by the unpacker and the packer, so it belongs to neither. |
| `storage.py` | The precision rules as pure functions: what a src datum, a Dest slot and the feedback path each hold. Testable without an architecture. |
| `l1_codec.py` | `pack_to_l1` and `unpack_from_l1`, the bytes boundary. |
| `units/unpack.py` | `Unpack`: L1 into SrcA, SrcB, SrcS or Dest, both legality tables, and the Dest-format derivation. |
| `units/fpu.py` | `Fpu`: the arithmetic, the multiplier's mantissa split, the write to Dest, and Dest back to a src register. |
| `units/fidelity.py` | The multiplier's fidelity decomposition: phase tables, the mantissa split, the pre-carry denormal flush. Applied by `Fpu` and touched by no other unit. |
| `units/pack.py` | `Pack`: Dest into L1, the pack legality table and the edge-mask polarity. |
| `units/pack_effects.py` | ReLU, edge masking, and what stochastic rounding costs us. Applied by `Pack` and touched by no other unit. |

There is no `units/sfpu.py`. Nothing here models the SFPU, so the class would be
empty and every architecture would have to supply a no-op. It arrives with the
first real SFPU block.

**Each module has a chapter below.** The code points at those chapters rather
than repeating them. A docstring tells you what you need at the call site, and
the chapter tells you why.

[`data_transfer_blocks.py`](#data_transfer_blockspy) ·
[`storage.py`](#storagepy-register-storage) ·
[`units/fpu.py`](#unitsfpupy) ·
[`units/fidelity.py`](#unitsfidelitypy-in-full) ·
[`units/unpack.py`](#unitsunpackpy) ·
[`units/pack.py`](#unitspackpy) ·
[`units/pack_effects.py`](#unitspack_effectspy) ·
[`l1_codec.py`](#l1_codecpy) ·
[`arch.py`](#where-an-architecture-plugs-in)

---

# The one idea to take away

**There is no quantization step anywhere in this package.**

Storing a tensor as `MxFp8R` *is* the quantization. By the time the L1 bytes
exist the loss has already happened, and reading them back just reports it.

A golden that quantizes and *then* packs has run two rounding rules over the
same data. It will disagree with silicon at every subnormal and every lattice
midpoint. If you are migrating a test that pre-quantizes its stimuli, delete
that step. The stimuli and the golden go through the same packer, so they land
on the same lattice by construction.

What is left is the loss each boundary takes: the unpack into a src register,
the FPU's write to Dest, Dest fed back into a src register, and the packer's
ReLU, edge mask and requantization. Two of those vary by architecture.

The first and larger one is this. **The unpacker lands every L1 format in one of
a small number of src-register storage families, and it is the family, not the
L1 format, that the FPU reads.** The second is how wide the FPU's multiplier is
per fidelity phase, which only Quasar models here.

---

# `data_transfer_blocks.py`

## The blocks

| Method | Models |
|---|---|
| `l1_to_srcA(l1_bytes, l1_format, src_format=None, **geometry)` | The T0 unpack. Delegates to the unpacker. What varies by architecture is the format metadata, not this method. |
| `l1_to_srcB(...)` | Delegates to the same place as `l1_to_srcA`, because SrcA and SrcB share the datum layout. |
| `l1_to_srcS(l1_bytes, l1_format, src_format=None, *, dest_acc=False, **geometry)` | Not a delegate, because SrcS has its own format rules (`srcs_format`). Float32 keeps full fp32 here, where SrcA truncates it to Tf32, and with `dest_acc` every float except Float16 and Float16_b widens to Float32. `use_srcs=True` is defaulted into the geometry, because SrcS uses a per-slice L1 layout and the buffer must have been *packed* that way. |
| `l1_to_dest(l1_bytes, l1_format, dest_format, **geometry)` | Straight into Dest, bypassing the src registers, so the value keeps **more** mantissa than the same buffer read through `l1_to_srcA`. This is how a feedback loop starts. |
| `src_to_dest(values, dest_format, current=None)` | Where a math result lands. |
| `dest_to_srcA(dest_values, dest_format, src_format)` | The feedback path. Lossy, and the table below says by how much. |
| `dest_to_srcB(...)` | The same conversion. |
| `dest_to_l1(dest_values, l1_format, dest_format, *, relu_type, relu_threshold, edge_mask, stoch_rnd, **geometry)` | The T2 pack, mirroring `l1_to_srcA`. |
| `pack_to_l1` and `unpack_from_l1` | L1 access, format-checked against the architecture. |
| `supports(l1_format)` and `supported_dest_formats` | What this architecture can read, and what it can hold in Dest. |
| `fpu_*` (seven methods) | Not transfers. These are what the FPU **computes**, and how its multiplier behaves. They are forwarded one at a time so an operation never holds the unit. See [The FPU computes](#the-fpu-computes). |

`src_format` on the unpack blocks overrides the storage format the unpacker
lands the buffer in, and `None` lets the architecture choose. The override is a
real knob but not an unrestricted one. Which src formats an L1 format can land
in is a property of the unpacker, held per architecture in
`Unpack.to_src_formats` and checked on **both** the explicit and the defaulted
path.

`Int32` has no legal SrcA or SrcB target at all, since it reaches Dest and SrcS
only, and an integer input never becomes a float in a src register. An
architecture that leaves the table empty is unmodelled at this level and falls
back to the weaker "is this a src storage format" check. That fallback is not a
claim that every pair is legal.

---

# `storage.py`: register storage

The precision rules as pure functions over tensors, with no idea which unit
applies them. They sit apart from the units because several units need the same
rule. `flush_subnormals` is taken by both the unpacker and the FPU, and the Dest
narrowing is applied on the unpack-to-Dest path, on the FPU's write, and again
by the packer as a safety net. Keeping them at module level also makes each one
testable against a tensor without constructing an architecture.

Two rules are worth stating once:

- **Every narrowing on the register paths truncates.** It does not round.
  `truncate_mantissa` is the single primitive behind all of them.
- **`saturate_to_integer` clamps before it narrows**, because a plain cast
  wraps. A Dest result of 200 narrowed to `Int8` lands as `-56`, and a later
  clamp in the packer would then see `-56` where it should have seen `127`. It
  clamps in `float64`, because `int32`'s max is not representable in `float32`.
  Clamping there would round the bound *up* to 2³¹ and leave the cast
  undefined, which is the bug being fixed. NaN survives a clamp, so it is
  mapped first.

## Two src storage families, and a name that lies

A src register datum is **19 bits: 1 sign, 8 exponent, 10 mantissa**, whatever
format is stored in it (`SRC_MANT_BITS = 10`).

```python
SRC_STORAGE_FORMATS = {Float16, Float16_b, Tf32}
```

Only two families exist in the datapath. **`Float16_b` is an alias for `Tf32`.**
The hardware converts it, and the distinction survives only in the row and
column mask path, which is not on the datum path.

> **Trap.** Nothing truncates a src mantissa to bf16's 7 bits. A
> `Float32 → Float16_b` src keeps **10** mantissa bits, not 7. The src format
> sets the exponent *range*, not the mantissa width. Precision in a src register
> is `min(input mantissa bits, 10)`.

### `storage.to_src_storage`: the two losses the unpacker takes

| src family | Mantissa | Exponent range |
|---|---|---|
| `Float16` | Truncate to 10 bits, **then** cast. Casting straight to fp16 would round, and the unpacker does not round. | 5-bit. Above it, values saturate to infinity. Below the smallest normal they flush to zero, so fp16 subnormals never reach the register. |
| `Float16_b` and `Tf32` | Truncate to 10 bits. | fp32's range. Clips nothing. |
| `Float32` | Not a legal SrcA or SrcB target at all, because a 19-bit datum cannot hold it, so `l1_to_srcA` refuses it. A `Float32` **input** lands as `Tf32`, `Float16` or `Float16_b`, all of which keep 10 bits. Full fp32 survives only through `l1_to_dest` and `l1_to_srcS`. | |
| integer | Passes through unchanged. | |

Truncation, never rounding. `storage.truncate_src_mantissa` calls
`truncate_mantissa(values, SRC_MANT_BITS)`, which masks off the low
`23 - 10 = 13` bits of the float32 mantissa.

## Dest storage

```python
DEST_STORAGE_FORMATS = {Float32, Int32, Float16, Float16_b, Int16, Int8, UInt8}
DEST_32_BIT_FORMATS  = {Float32, Int32}
```

Dest is **32-bit when accumulation is enabled and 16-bit otherwise**. That is
the whole of what `DestAccumulation` controls. `Tf32` has no separate Dest
encoding and lives in a `Float32` container.

`resolve_dest_format(dest_format, l1_input_format, dest_acc)` is what every
`run()` calls. It derives the Dest format from the input when none is given, and
otherwise checks the one supplied against `dest_acc` using
`DEST_32_BIT_FORMATS`. The two are not independent settings, so `Float32` with
`dest_acc=False` is rejected rather than run. Unchecked, it produces a
believable answer at the wrong precision, which looks like a maths bug further
down the chain.

### `src_to_dest(values, dest_format, current=None)`

Which instruction ran belongs to the operation. What belongs here is **where its
result lands**. A Dest slot holds only what `dest_format` can represent, and the
value is rounded on the way **in**, not on the way out to L1.

`current` is Dest's existing contents, which a multi-pass op passes in to
accumulate rather than replace. The FPU has an accumulate-enable bit, not a
second instruction. Because the rounding happens on this write, **it happens on
every pass**. Accumulating at full precision and rounding once at the end would
model an accumulator the hardware does not have.

### `storage.to_dest_storage`: the FPU writes no denormal

Narrows to the Dest dtype, then flushes anything below that format's smallest
normal. The rule it models is that when the FPU's result is valid and the
operation is floating point, a zero exponent implies a zero mantissa.

> **Why this one matters disproportionately.** `torch.float16` *does* carry
> subnormals, so a naive cast keeps a value the hardware zeroed. The MX packer
> then rounds that survivor **up** onto the output lattice, so the device
> reports `0` and the golden reports the output format's smallest representable
> value. The signature in a log is unmistakable: a handful of datums out of
> thousands, PCC exactly `1.000000000`, and every diff reading *golden =
> smallest lattice value, device = 0*.

Only a `Float16` Dest meets the threshold in practice. The wider formats bottom
out near 2⁻¹²⁶.

### `dest_format_for(l1_input_format, dest_acc)`

Derives the Dest format from the **input** side, because that is what the
hardware does.

| Input | `dest_acc=False` | `dest_acc=True` |
|---|---|---|
| integer | the input format | `Int32` |
| float | the 16-bit member of the src family, so `Float16` if the src is `Float16` and `Float16_b` otherwise | `Float32` |
| `Float32` and `Tf32` | **Raises.** The input alone does not decide the family. The device picks `Float16` or `Float16_b` from the *output* format (`infer_unpack_out`), and the two clip differently. Pass the `dest_format` the harness configured, or enable `dest_acc`. | `Float32` |

> **Dest follows the unpacker's src output, never the pack format.** An MX input
> unpacks to the 8-bit-exponent family, so Dest is bf16 *whatever you pack to*.
> This cannot be inferred from `out_format`, which is why `OpConfig.dest_format`
> is a required field.

### `storage.dest_to_src_storage`: the feedback path

Driven by the **Dest** format, not the src format. The src format is consulted
*only* to decide whether a wide Dest needs its exponent rebiasing. Every case is
bit-slicing, so nothing rounds. Bits below the target width are dropped, not
folded in.

| Dest format | What reaches the src register |
|---|---|
| `Float16_b` | 7 explicit mantissa bits, and the low 3 arrive as **zeros**. A bf16 Dest does not come back with a src register's full 10. |
| `Int16` | **Carried unchanged.** It shares the bf16 path, but the format label there is a transport trick rather than a conversion, so 257 comes back as 257 and is not rounded to 256 the way the row above would suggest. |
| `Float16` | Passes as fp16, so 10 mantissa bits and a 5-bit exponent. |
| `Int32` | **Saturates to INT8**, clamped to ±127. A wide integer Dest cannot survive the trip, and the clamp is silent. |
| `Float32` and `Tf32` | 10 mantissa bits and the 8-bit exponent, unless the src register is `Float16`. In that case the exponent is rebiased and values below the fp16 normal range flush to zero. |

---

# `units/fpu.py`

## The FPU computes

Every other unit here only moves and converts. The FPU also *computes*, so the
arithmetic is its, and so is everything about how this device performs it. An
operation says **which** instruction runs and in what order. The FPU says what
that instruction does.

Its methods are forwarded one at a time rather than handing the unit over, so
nothing outside this package holds an `Fpu` and `units/` stays private. Each
keeps the `fpu_` prefix. Unlike a transfer, these are not named for a hardware
path, and a bare `phases()` or `operand_halves()` would not say which unit
decides the answer.

| Call | What it is for |
|---|---|
| `blocks.fpu_elementwise(operation, srcA, srcB)` | One element-wise op (`Elwadd`, `Elwsub` or `Elwmul`) computed in a type exact for the operands it was handed. |
| `blocks.fpu_sum_of_products(srcB_rows, srcA_cols)` | One matrix multiply, accumulated in `float64`, which is wider than its operands just as the hardware's network is. Which rows and columns go in is the operation's business. |
| `blocks.fpu_models_fidelity(in_format)` | Whether a multiply on these datums decomposes into phases here. It takes the **format**, because an operation asks while it is building its pipeline, before any register holds a value. |
| `blocks.fpu_phases(math_fidelity)` | The phase indices to run. `LoFi` is one phase and `HiFi4` is four. |
| `blocks.fpu_operand_halves(srcA, srcB, phase)` | The mantissa window each operand contributes on one phase, with non-finite lanes already resolved. For a caller that has to arrange the halves before multiplying, such as a matmul slicing one K face out of them, rather than one that just wants the phase's product. |
| `blocks.fpu_partial_product(srcA, srcB, phase, dest_format=None)` | One whole fidelity phase (window, multiply, flush) so the multiply itself never leaves the unit. `dest_format` applies the pre-carry denormal flush, for a product that *is* a lane result. A matmul passes nothing and does not use this, since it needs the halves unmultiplied. |
| `blocks.fpu_warn_split_unmodelled(op_name)` | Says so when a multiply falls back to the exact product. |

### Integer arithmetic is integer

The unpacker lands an integer input as an integer tensor, so an integer op here
really does integer arithmetic. Two rules follow, and both have already caught a
wrong answer.

* **Use `int64`, not the operand's own width.** torch integers *wrap*: `int8`
  `100 + 100` is `-56`, and the saturation in `storage.to_dest_storage` would
  then clamp the wrong number. `int64` is exact for every product reachable
  through SrcA and SrcB, since `Int32` never lands in one.
* **No fidelity split on an integer.** The decomposition slices a mantissa, and
  an integer datum has none to slice. Applied anyway it truncates the operand's
  low bits. An `Int16` `3 * 10922` came back as `32640` instead of `32766`.
  `models_fidelity` is what keeps that from happening.

Dest accumulation picks its own width the same way. Two integers add in
`int64`. A float result going into an integer Dest uses `float64`, because
`float32` holds integers exactly only to 2²⁴ and an `Int32` Dest reaches 2³¹. A
float Dest uses `float32`.

### One invariant worth repeating

**Every phase reads the original srcA and srcB**, not the previous phase's
sliced operands. Feeding a phase its predecessor's output zeroes every phase
after the first, which turns fidelity into a silent no-op. The result still
looks plausible, just more accurate than the device.

## `units/fidelity.py` in full

The multiplier's decomposition, applied by `Fpu` and shared by the element-wise
multiply and the matmul because the two decompose identically.

| Member | Notes |
|---|---|
| `split_mantissa(values, keep_bits)` | `(high, low)` where `low = value - high`. |
| `operand_halves(srcA, srcB, split, phase)` | The slice of each operand a phase multiplies. |
| `min_normal_exponent(dest_format)` | The lowest exponent Dest holds as a normal, or `None` for an integer. |
| `ieee_exponent(values)` | `frexp`'s exponent minus one, which is the field an IEEE float stores. |
| `flush_pre_carry_denormals(product, srcA, srcB, min_exponent)` | The flush rule. Element-wise only, because a matmul's products never become lane results. |
| `fidelity_phases(math_fidelity)` | A `range` of phase numbers. `FIDELITY_PHASES` maps a fidelity to a count, and `PHASE_OPERAND_HALVES` names each phase's two halves, **biggest term first**, so stopping early drops what was contributing least. |
| `resolve_non_finite(a_half, b_half, srcA, srcB, phase)` | Repairs operand positions holding an infinity or a NaN. See below. |
| `non_finite_lanes(srcA, srcB)` | Where either operand is non-finite, so the split and flush rules can skip them. |
| `warn_unmodelled_split(op_name, arch_name, stacklevel=2)` | The warning text for an architecture whose split this package does not model. |

#### `split_mantissa`: why subtract rather than mask

The low half is `value - high`, not a masked mantissa read back as a float. The
implicit leading 1 belongs to the high half, so masking in place does not give
the remainder.

**This is provably equivalent to the lightweight golden's mask-and-reassemble.**
Its Quasar masks are:

```python
(0b11111111000, 0b11111111000)    # phase 0: top 8 of an 11-bit {implicit, mantissa}
(0b00000000111, 0b11111111000)    # phase 1: mant[2:0], implicit position cleared
```

`reassemble_float_after_fidelity` runs the masked field through
`calculate_fractional_part` on an 11-bit string, so the low half's implicit bit
is *not* restored. `0b00000000111` becomes `0.0000000111 × 2^e`, which is
`mant[2:0] × 2^(e-10)`, exactly this remainder. Verified bit-identical across
10,000 values at four widths.

Two corollaries are worth keeping, because both have been chased as leads and
are dead ends.

- The Quasar masks are **symmetric**, 7 explicit bits from each operand,
  matching `QuasarArch.fpu.mantissa_split == (7, 7)`. The asymmetric 4/6 split
  in that module's comment belongs to its `else` branch, which is
  **Wormhole and Blackhole**, where SrcA also drops its least significant bit.
- Mask-and-reassemble versus subtract is **not** a source of divergence between
  the two goldens.

#### `flush_pre_carry_denormals`: the pre-carry rule

The FP lane forms the exponent by adding the two *stored src* exponents and
rebiasing into Dest's range, then flushes the whole term, mantissa included,
when that lands below the lowest normal. The decision is taken **before** the
mantissa product's carry into the next binade, so the rule is one binade coarser
than testing the finished magnitude. When the mantissas multiply to 2.0 or more,
a product anywhere in `[tiny, 2·tiny)` is a legitimate Dest normal that hardware
still returns as zero.

```
fp16 src, 16-bit Dest (bias 15):
  0.025390625 (2^-6 × 1.625)  ×  0.0024414062 (2^-9 × 1.25)
  exponents sum to -15  → lane flushes
  product is 6.199e-05, above fp16 tiny 6.104e-05 → a magnitude rule keeps it
```

This only bites for a five-bit-exponent Dest, meaning an fp16 src into a 16-bit
Dest. Every other combination accumulates with an eight-bit exponent and never
reaches the threshold. It is only *visible* through an MX output, where one
zeroed element shifts the block scale enough to fail the block-aware compare.

> Use the **datum's** exponent, not a split half's. One exponent field serves
> every phase, because a phase selects a mantissa window and not an exponent, so
> the sum is identical on all four. Using a half's exponent flushes almost
> everything, and scores worse than no flush at all.

#### Infinities and NaNs

A non-finite datum has no mantissa to split, and the three obvious fixes are all
wrong.

| Attempt | What goes wrong |
|---|---|
| Split it like anything else | `split_mantissa` does `value - high`, and `Inf - Inf` is `NaN`. |
| Zero just the low half | A later phase multiplies the surviving infinity by a zero half, and `Inf × 0` is `NaN` by IEEE. |
| Let `frexp` report it | `frexp` gives exponent 0 for a non-finite value, which reads as a magnitude near 2⁻¹, so the flush rule treats an infinity as a tiny number and zeroes it. |

What `resolve_non_finite` does instead is carry the **raw value** at those
positions on phase 0, where it is the whole of that datum's contribution, and
zero them on every later phase. Finite positions keep their truncated halves, so
LoFi stays truncated. `Inf + 0` stays `Inf`, and a NaN propagates.

The same approach works for a matmul. The phase-0 term carries the infinity into
every output position that sums over it, and later phases add exact zeros there.

`non_finite_lanes` is the matching guard inside `flush_pre_carry_denormals`.
Without it, `Inf × b` flushes to zero for any small `|b|`.

## Where an architecture plugs in

An architecture is **assembled, not subclassed**. `ArchGolden` holds one unit
per hardware stage, and `DataTransferBlocks` is a façade that delegates to them.
Nothing overrides a method. There is exactly one implementation of each
transfer, and architectures differ in the data their units are constructed with.
See [arch.py](arch.py).

| Where | What it carries |
|---|---|
| `L1Support(name, formats)` | The architecture's L1 format set, the support checks, and the L1 read/write path. Shared by the unpacker and the packer, so it belongs to neither. `name` appears in refusals, so they say `Quasar cannot read Bfp8_b` rather than naming the shared class. |
| `Unpack(l1, to_src_formats, to_dest_formats, has_srcs)` | The L1-to-src mapping, both legality tables, and whether this architecture has a SrcS register at all. Also the Dest-format derivation, because Dest takes the format the *unpacker* landed in the src register, which makes that question the unpacker's. |
| `Fpu(l1, mantissa_split)` | The arithmetic, the write to Dest and the Dest-to-src feedback. `mantissa_split` is the explicit mantissa bits the multiplier takes from each operand per fidelity phase: `(7, 7)` on Quasar, and `None` on Wormhole and Blackhole, where it is unmodelled and says so. |
| `Pack(l1, to_l1_formats, edge_mask_masked_when_set, promotes_outlier_to_32_bit_dest)` | The Dest-to-L1 legality table, the edge-mask polarity, and the harness's outlier promotion. |

An empty table means **unmodelled at that level of detail**, and a weaker check
applies instead. It is not a claim that every pair is legal. Wormhole and
Blackhole leave the unpacker's tables empty today.

There is no `Sfpu` unit. Nothing in this golden models the SFPU, so the class
would be empty and every architecture would have to supply a no-op. It is added
with the first real SFPU block.

### Adding an architecture

You need one `L1Support` and one class in [arch.py](arch.py):

```python
_NEWARCH_L1 = L1Support(name="NewArch", formats=frozenset({...}))


class NewArch(ArchGolden):
    l1 = _NEWARCH_L1
    unpack = Unpack(_NEWARCH_L1, to_src_formats={...}, has_srcs=False)
    fpu = Fpu(_NEWARCH_L1)         # add mantissa_split=(a, b) to model fidelity
    pack = Pack(_NEWARCH_L1)
```

and a factory beside the others in
[data_transfer_blocks.py](data_transfer_blocks.py):

```python
NewArchDataTransferBlocks = partial(DataTransferBlocks, NewArch())
```

Use `partial`, not a plain function. A plain function is a descriptor, so
assigning one as a **class attribute** binds it as a method and passes `self` as
the architecture. A dataclass field, which is how the operations layer holds
these, stores on the instance and is safe either way. `partial` is safe in both
places and accepts the `__name__` and `__doc__` the factories carry.

### When a unit's behaviour genuinely differs

Only then does a unit get a second implementation, and only that unit. Write a
new class implementing the same surface, and name it after what differs rather
than numbering it:

```python
class UnpackWithoutTf32(Unpack):   # or a standalone class on the same surface
    ...

class NewArch(ArchGolden):
    unpack = UnpackWithoutTf32(_NEWARCH_L1, ...)
    fpu = Fpu(_NEWARCH_L1)         # unchanged units are reused as-is
    pack = Pack(_NEWARCH_L1)
```

Do **not** reach for this to express a different format table. That is what the
constructor arguments are for. The tripwire is this: if more than two or three
units end up with per-architecture variants, the data-versus-behaviour split has
stopped paying and inheritance is the better tool again.

### The format divide

| | L1 formats | Dest formats | Has | Lacks |
|---|---|---|---|---|
| Quasar | 15 | `Float16` `Float16_b` `Float32` `Int8` `Int16` `Int32` `UInt8` | MX: `MxFp8R` `MxFp8P` `MxFp4` `MxInt8` `MxInt4` `MxInt2` | **No block float.** No `MxFp4_2x_A/B`, which are src-register formats and never L1 formats. |
| Blackhole | 14 | `Float16` `Float16_b` `Float32` `Int8` `Int32` `UInt8` | Block float: `Bfp8` `Bfp8_b` `Bfp4_b` `Bfp2_b` | **No MX.** |
| Wormhole | 13 | `Float16` `Float16_b` `Float32` `Int8` `Int32` `UInt8` | Block float: `Bfp8` `Bfp8_b` `Bfp4_b` `Bfp2_b` | **No MX.** No `Fp8_e4m3`, because Wormhole's only fp8 is Lf8 (e5m2). |

Quasar's L1-to-src mapping in full:

| L1 format | lands in src as | |
|---|---|---|
| `Float16` | `Float16` | the 5-bit exponent family |
| `Fp8_e4m3` | `Float16` | an L1-only encoding, which the unpacker converts |
| `Float16_b` | `Float16_b` | the same thing as Tf32 |
| `Float32` and `Tf32` | `Tf32` | 8-bit exponent, 10-bit mantissa |
| all six MX formats | `Float16_b` | **whatever you pack to** |
| `Int8` `Int16` `Int32` `UInt8` | unchanged | |

---

# `units/unpack.py`

L1 into SrcA, SrcB, SrcS or Dest. It owns the two legality tables that vary by
architecture, the L1-to-src-format mapping, and the Dest format derivation. That
last one belongs here because **Dest takes the format the unpacker landed in the
src register**, never the pack format.

## The format mapping

`src_format(l1_format)` resolves every L1 format to one of three src families:

| Input | Lands as | Why |
|---|---|---|
| `Float32`, `Tf32` | `Tf32` | 8-bit exponent, 10-bit mantissa |
| any MX format | `Float16_b` | the unpacker converts MX into the 8-bit-exponent family **whatever the pack format**, so math and Dest see bf16 |
| `Fp8_e4m3` | `Float16` | special-cased, see below |
| exponent-A | `Float16` | |
| exponent-B | `Float16_b` | |
| anything else | itself | correct for integers only |

`Fp8_e4m3` has to be named explicitly because it reports **neither** exponent
family. `is_exponent_A()` and `is_exponent_B()` are both `False` for it. Without
the case it would reach the final pass-through and stay `Fp8_e4m3` in a src
register, which no architecture does.

That pass-through is the one to watch. A *float* arriving there is a format
whose exponent family `helpers.format_config` does not know, and it needs a case
adding rather than a silent identity mapping.

## SrcS and the integer divergence

SrcS does not narrow `Float32` to `Tf32` the way SrcA and SrcB must, and with
accumulation on it widens instead.

- MX lands in `Float16_b`, as it does in SrcA.
- `Float32` stays `Float32`, with or without `dest_acc`.
- With `dest_acc`, `Float16` and `Float16_b` stay themselves, because the
  unpacker cannot convert fp16 to a 32-bit SrcS datum. Every other float widens
  to `Float32`.
- Otherwise a format lands as it would in SrcA.

> **Integers pass through unchanged, and that is a deliberate divergence** from
> `infer_unpack_out` in `helpers.data_format_inference`, which the float cases
> above do follow. Its SrcS branch returns `Float32` for everything except
> `Float16` and `Float16_b` once `dest_acc` is on, so an `Int8` input is
> programmed as a `Float32` SrcS. That is a conversion the unpacker does not
> perform, and one that would make a golden report `1.0` where the device holds
> the integer `1`. The blanket widening reads as a rule written for the float
> path, and integer SrcS with accumulation is not swept, so nothing has forced
> the question. **Do not "fix" this by aligning with it.**

The **slice layout** follows the SrcS element width, not `dest_acc`. It is
32-bit only for `Float32`, `Int32` and `Tf32`. MX lands as `Float16_b`, so an MX
buffer holds 144-byte slices whatever `dest_acc` says, and reading it at 80
would take the wrong stride through data the device wrote.

## The 32-bit guard

`is_valid_src_format` refuses `Float32`, `Int32` and `UInt32` outright, before
consulting any table. No 32-bit datum reaches SrcA or SrcB on any architecture
modelled here, and without the guard the unmodelled-architecture fallback's
`src_format == l1_format` arm would hand back full int32 values for a register
that cannot hold them. `Int32` reaches Dest and SrcS only.

## The two families a wide float could land in

**A `Float32` or `Tf32` input with a 16-bit Dest raises** rather than guessing.
Both families are legal targets and they behave differently:

| Target | Mantissa | Exponent | Clips |
|---|---|---|---|
| `Float16` | 10 bits | 5-bit | saturates above 65504, flushes below 2⁻¹⁴ |
| `Float16_b` or `Tf32` | bf16's 7 in Dest | full fp32 range | nothing |

Which one a kernel gets is a harness decision. `infer_unpack_out` picks the
family from the **output** format, to limit exponent mixing, so guessing here
would model a different kernel than the one under test. Pass the `src_format`
and `dest_format` the harness configured, or enable `dest_acc`, where both
families give a `Float32` Dest and the question disappears.

The same refusal appears twice, in `l1_to_src` when defaulting the src format
and in `dest_format_for` when deriving the Dest format, because it is the same
ambiguity reached from two directions.

## `l1_to_dest` truncates

There is no rounding stage in the unpack datapath, unlike the FPU's write to
Dest. Narrowing a `Float16_b` or `Tf32` Dest with a plain cast would round, so
the mantissa is truncated explicitly first.

## `resolve_dest_format` checks the pairing

An explicit `dest_format` is validated against `dest_acc` rather than taken on
trust, because the two are not independent. **A Dest slot is 32-bit exactly when
accumulation is on.** `Float32` with `dest_acc=False` describes a machine state
that cannot exist, and left unchecked it runs and produces a plausible answer at
the wrong precision, full fp32 where the device had 16 bits or the reverse. That
is indistinguishable from a maths bug downstream.

Deriving the format is not a reason to skip the check. `dest_format_for` applies
the same pairing to integers, and refuses an input that cannot drive an `Int32`
Dest at all. `INT32_DEST_COMPATIBLE` is `Int8`, `UInt8` and `Int32`. `Int16` is
absent because the FPU takes it for MOV only.

---

# `l1_codec.py`

A thin dispatch over the real codecs in `helpers.pack` and `helpers.unpack`,
with one entry point each way. These are deliberately the **same** codecs the
test harness writes device L1 with, which is what makes the bytes the golden
reads identical to the bytes the device read.

| Member | Notes |
|---|---|
| `PACKERS` | Format to packer, 19 entries, mirroring `StimuliConfig.get_packer`. `MODELLED_L1_FORMATS` is its key set, the formats this golden can actually move bytes for, which is **narrower** than an architecture's `SUPPORTED_L1_FORMATS`. `Tf32` on every architecture and `Bfp8` on Wormhole and Blackhole are real L1 formats with no codec here, so `supports()` returns False for them and the error says it is a model gap rather than a hardware limit. |
| `datums_per_tile(num_faces=4, face_r_dim=16)` | `num_faces × face_r_dim × 16`. The geometry primitive everything counts in. |
| `pack_to_l1(tensor, l1_format, *, tile_count=None, num_faces=4, face_r_dim=16, use_srcs=False, dest_acc=False)` | Where precision is lost for the block-scaled formats. |
| `unpack_from_l1(packed, l1_format, *, tile_count=None, tile_stride_bytes=None, num_faces=4, face_r_dim=16, use_srcs=False, dest_acc=False, twos_complement=False)` | Reads bytes back as values. |
| `_call_accepted(fn, tensor, **kwargs)` | Calls a packer passing only the keywords it declares. |

### `MODELLED_L1_FORMATS`: what this module can actually move

```python
MODELLED_L1_FORMATS = frozenset(PACKERS)
```

Looping a dict yields its **keys**, so `frozenset(PACKERS)` is the set of every
format that has a packer. It is identical to `frozenset(PACKERS.keys())`, just
shorter. Deriving it rather than typing it out means there is one list and not
two that can drift. Add a packer and it is modelled automatically.

Deriving it from the packers alone is only sound because **the two directions
cover the same formats**. There is no format with a packer and no unpacker, or
the reverse. That holds today, and all 19 round-trip, but nothing enforces it,
since the unpack side dispatches inside `unpack_res_tiles` rather than from a
matching table.

It answers a different question from an architecture's L1 format set, and
`L1Support.supports` demands **both**:

| Question | Answered by |
|---|---|
| Can the *hardware* hold this format? | `L1Support.formats` |
| Can *this code* read and write it? | `MODELLED_L1_FORMATS` |

A format can be completely real on the device and still have no codec here, as
`Tf32` is everywhere and `Bfp8` is on Wormhole and Blackhole. Without the second
gate those sail past the check and fail deep inside `unpack_res_tiles` with a
bare `KeyError`, which is what `_no_codec_message` exists to replace.

### `_call_accepted`, and the typo that does not raise

The packers take different subsets of the tile geometry. `pack_fp16` takes none,
`pack_bfp8_b` takes faces, and the MX packers also take the SrcS layout flag and
extra rounding controls. Filtering by signature keeps one call site.

The filtering applies to the **codec functions**, not to the entry points.
`pack_to_l1` and `unpack_from_l1` both name every parameter keyword-only with no
`**kwargs`, so a misspelled geometry key raises `TypeError` on either path. What
`_call_accepted` drops is a *correctly spelled* key that a given packer does not
declare, such as `num_faces` handed to `pack_fp16`.

### Multi-tile layout

The underlying codecs handle exactly **one tile**, so `pack_to_l1` splits a
multi-tile tensor and packs tile by tile, matching how it is read back.
`pack_to_l1` also casts bfloat16 up to float32 first. That is lossless, and it
is needed because numpy has no bfloat16 and most packers go straight to
`.numpy()`.

> **`tile_stride_bytes` defaults to `tile_bytes_for`**, what `pack_to_l1`
> actually writes at this geometry, which is not the dense datum count. The BFP
> packers hold a minimum of 16 exponents, so a 1×32 `Bfp8_b` tile takes 48 bytes
> rather than 34, and `use_srcs` writes 16-byte-aligned slices, taking 1152
> bytes rather than 1056, or 1280 under `dest_acc`. Left to its own devices,
> `unpack_res_tiles` assumes a full 32×32 tile stride for backward
> compatibility, which is correct only when the geometry really is 32×32. Pass
> the device's stride explicitly when reading a buffer laid out some other way.

> **Two source strides, only one of which matches.** `Golden.run` lays tiles out
> back to back, `datums_per_tile` apart. `write_matrix_w_tile_dimensions`
> (`use_dense_tile_dimensions=True`) strides the source the same way, but the
> *default* `StimuliConfig.write_matrix` always strides by `MAX_TILE_ELEMENTS`
> whatever the tile size, writing only `num_faces × face_r_dim × 16` of each
> stride. They coincide only at 1024 datums per tile, or on a single tile where
> the stride never applies. Below that, over several tiles, the two read
> different source elements from tile 1 onwards. `operations/golden.py`'s
> `check_source_layout` raises for those rather than silently computing on data
> the device never saw.

---

# `units/pack.py`

Dest into L1, the T2 half that mirrors the unpacker. It owns the Dest-to-L1
legality table, the edge-mask polarity, and the order the packer applies its
effects.

`dest_format` **cannot be inferred from `l1_format`.** Dest follows the
*unpacker's* output, meaning the input format, never the pack format. Use
`Unpack.dest_format_for` to derive it from the input side.

## What the packer will and will not convert

Two rules and one exception:

- **Narrowing out is the ordinary case.** A `Float32` Dest reaches every float
  and MX format.
- **No upcast to a 32-bit output.** No 16-bit Dest produces `Float32`, and
  nothing crosses between integer and float.
- **The integer rows are tighter than either rule implies**, and are worth
  reading rather than inferring.

Quasar's table in full:

| Dest | Reaches |
|---|---|
| `Float32` | `Float32`, `Tf32`, `Float16`, `Float16_b`, `Fp8_e4m3`, all six MX |
| `Float16` | `Float16`, `Float16_b`, `Fp8_e4m3`, all six MX |
| `Float16_b` | `Float16`, `Float16_b`, `Fp8_e4m3`, all six MX |
| `Int32` | `Int32`, `Int8`, `UInt8`, but **not `Int16`** |
| `Int16` | `Int16` only |
| `Int8` | `Int8` only |
| `UInt8` | `UInt8` only |

Without `check_pack_pair`, a `Float16` Dest packed to `Float32` returns bytes
that decode to believable numbers for a conversion the hardware refuses to
perform.

## The outlier promotion

`check_outlier_promotion` refuses a `Float16_b` Dest packed to `Float16`,
because on every architecture but Quasar that state never runs. `TestConfig`
spots an exponent-B input with a `Float16` output, turns `dest_acc` on, and the
device uses a 32-bit Dest instead. A `Float16_b` Dest is the tell, since that is
what an exponent-B input resolves to without accumulation.

Model it as written and the golden runs bf16 against a device running fp32. That
is a real precision gap, and it arrives as an arithmetic disagreement rather
than a wiring mistake. See `promotes_outlier_to_32_bit_dest` in
[The blocks](#the-blocks).

## `UnmodelledHardwareWarning`

A `RuntimeWarning` subclass rather than a `UserWarning`, deliberately.
`pytest.ini` sets `ignore::UserWarning` for the whole suite, so a `UserWarning`
would reach nobody running the tests, and they are the only audience for a
warning that says "this answer is approximate". Worse, `pytest.warns` bypasses
the filter, so the unit test asserting the warning would keep passing while the
suite stayed silent.

---

# `units/pack_effects.py`

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
| `MinThresholdRelu` | Below the threshold is flushed, and above it passes through **untouched**. |
| `MaxThresholdRelu` | Clamp into `[0, limit]`. |

The threshold is first narrowed by `_encode_threshold` to the 16 bits the
packer's configuration register actually holds, which is fp16 for the
exponent-A family and bf16 otherwise. **Comparing against the un-narrowed value
is a real and easy bug.**

### Why the threshold is encoded twice

The field is 16 bits wide whatever the pack format, and the two branches narrow
*differently*:

| Dest format | Register reads the field as | So the threshold is |
|---|---|---|
| `Float16`, `Bfp8` | fp16 | **rounded** |
| everything else | bf16, the top half of the fp32 | **truncated** |

Narrowing with a plain bf16 cast would round where the register truncates, which
shifts the threshold a full bf16 step for any value with mantissa bits below the
cut. `0.3` becomes `0.30078` instead of `0.29883`, and every datum between the
two is then clamped differently.

> This duplicates the encoding in `PackGolden` **on purpose**. Heavyweight owns
> its own model of the packer so the old golden can be retired without stranding
> it. The two were bit-identical across every format and threshold swept when
> this was written, and **nothing enforces that they stay so**. There is no test
> comparing them, deliberately, since one would reintroduce the dependency.
> Treat the agreement as a fact about that moment, not a guarantee.

## `PackEdgeMask(masks, select, mode)`

Masking at tile edges. Up to `EDGE_MASK_COUNT = 4` masks of
`EDGE_MASK_WIDTH = 16` bits, one bit per datum of a 16-datum row.

| Field | Notes |
|---|---|
| `masks` | The raw register values. **What a set bit means depends on the architecture.** On Quasar it masks the datum, because the packer inverts the register before the gasket. On Wormhole and Blackhole it keeps it. `Pack` resolves this through its `edge_mask_masked_when_set`, so `dest_to_l1` applies the right one. |
| `select` | Which mask each **row** uses, either one index for every row or one per row. The hardware selector is 2 bits per row, so a row cannot mix masks. `from_face_select_words(masks, words)` builds it from Quasar's four `EDGE_MASK_SELECT_FACE*` words. |
| `mode` | `EdgeMaskMode.ZERO`, or `NEG_SATURATE`. The latter exists so a masked datum *loses* a following max-reduce rather than winning it at zero. |

`keep(count, masked_when_set=...)` gives the boolean survival mask, and
`apply(values, masked_when_set=...)` substitutes zero or `-inf`. Construction
validates mask count, bit width, selector range and mode.

## `is_deterministic(stoch_rnd)`

Stochastic rounding draws from a pseudo-random sequence seeded on device and
**cannot be reproduced here**. With it enabled the golden returns the
round-to-nearest result, which hardware matches only in expectation, so each
datum may land one ULP of the output format either side. Check this and compare
with PCC rather than exactly.

It returns `True` for `No` only. `Fpu` randomises the FPU's write to Dest rather
than the packer's write to L1, so the packer behaves normally under it, but the
value reaching the packer is already off and the golden can no more follow that.
It only actually diverges when that Dest write has to round at all, meaning a
16-bit Dest or a long enough accumulation, so a test that knows its Dest write
is exact can still compare exactly under `Fpu`. This predicate simply will not
make that call for it. `STOCH_RND_EFFECTS` maps each mode to the stage it
randomises, which is what `dest_to_l1` puts in its warning.

---

# Changing a block

1. **Find the boundary, not the symptom.** A disagreement that is one lattice
   step wide at an MX output is usually several ULPs wide in Dest. Re-run with a
   `Float16` output first. A Float16 Dest packed to a Float16 output is a
   straight copy, so the device's result *is* its Dest, and you can compare in
   ULPs instead of lattice steps.
2. **Change the block, not the operation.** If a loss belongs to a transfer,
   every operation that crosses that boundary should inherit it for free.
3. **Validate bit-exactly, at LoFi.** A tolerance that passes tells you the
   tolerance is wide enough. Bit-exactness on every datum tells you the model is
   right, and then a later disagreement is a finding rather than noise.
4. **Check the other architectures still construct.** The units are shared and
   only the data they are built with differs, so a change to a unit can break
   Wormhole or Blackhole without touching a Quasar test.

## What this package does not model

- **Stochastic rounding.** See above.
- **Wormhole's and Blackhole's mantissa split.** Their `Fpu` carries no
  `mantissa_split`, so a multiply there is the exact product and
  `math_fidelity` is ignored. That holds at *every* fidelity, HiFi4 included,
  since their per-phase masks never reach SrcA's least significant bit. It warns
  rather than passing quietly.
- **A bit-for-bit model of the unpacker's conversion hardware.** The L1-to-SrcA
  path is *composed* out of `pack_to_l1` plus the storage-precision rules, which
  agrees with silicon on everything measured here. Where a conversion rule is
  subtle it is written down as a rule, not reproduced as logic.
- **Partial-face multi-tile layout.** See the sharp edge above.
- **Integer SrcS under `dest_acc`.** `srcs_format` carries an integer input
  through unchanged, while `infer_unpack_out`, the rule the harness programs,
  returns `Float32` for it once `dest_acc` is on. That `Float32` is the
  fall-through after an fp16 carve-out, and only the carve-out is justified in
  its comment, so this is an open question rather than a settled rule. It is not
  cosmetic. `_is_srcs_32bit_mode_` keys on the same format, so it also selects
  the slice layout the buffer is read at. Nothing exercises integer SrcS against
  silicon on either side, and if the harness's version turns out to be right,
  this method and the layout both have to change.
- **`Tf32` and `Bfp8` L1 bytes.** Neither has a codec here, on any
  architecture, so they raise an explicit "no L1 codec" rather than a bare
  `KeyError`. That is a gap in the model, not a statement about the hardware.
