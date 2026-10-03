<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `operations` — goldens as pipelines of data-transfer blocks

An operation here does not compute its result. It declares the sequence of
transfers the hardware performs — L1 to src register, src to Dest, Dest to L1 —
and the result is whatever falls out of the end. Precision is lost at each
boundary because that is where silicon loses it.

Writing a test never touches the data-transfer blocks. Pick the golden for your
architecture and hand it tensors:

```python
from helpers.golden_generator.heavyweight.operations.quasar_operations import (
    QuasarDataCopyGolden,
)

result = QuasarDataCopyGolden().run(stimuli, in_format, out_format)
```

## Contents

| File | What is in it |
|---|---|
| `chain.py` | `Chain`, `Step`, `Registers`, `StageRecord` — the execution engine. No hardware. |
| `golden.py` | `Golden`, `OpConfig` — turns blocks into chainable steps, and runs them. |
| `datacopy.py` | `DataCopyGolden` |
| `matmul.py` | `MatmulGolden` |
| `fidelity.py` | Phase tables, `split_mantissa`, the pre-carry flush — shared by eltwise and matmul |
| `eltwise.py` | `EltwiseBinaryGolden` |
| `reuse_dest.py` | `EltwiseBinaryReuseDestGolden` |
| `quasar_operations/` `wormhole_operations/` `blackhole_operations/` | Per-architecture bindings |

Layering, each knowing strictly less about the hardware than the one below:

```
Chain · Step · Registers        no hardware at all
        ↓
Golden · OpConfig               blocks → steps          ──self.blocks──▶  DataTransferBlocks
        ↓                                                                        ↓
DataCopyGolden · MatmulGolden · Eltwise…                          Quasar… · Wormhole… · Blackhole…
        ↓
QuasarEltwiseBinaryGolden · …   binds blocks_class + arch constants
```

---

# `chain.py`

## `Registers`

The named slots a chain passes between steps. Holds whatever a step puts there —
L1 byte buffers, src tensors, Dest, counters. **No slot is special.**

Steps address slots **by name**, not by passing values along. That is deliberate:
a math op reads two src registers plus a Dest slot, and a feedback op reads the
slot it is about to write, neither of which a value-passing pipeline expresses.

Names used in practice: `in0`, `in1`, `in0_t1`, `srcA`, `srcB`, `srcS`, `dest`,
`out`, and `seed` for reuse-dest.

| Member | Notes |
|---|---|
| `Registers(**initial)` | Construct with slots pre-populated. |
| `regs[name]` | Raises `KeyError` naming the available slots if nothing wrote it. |
| `regs[name] = value` | Any type. |
| `name in regs` | |
| `regs.get(name, default=None)` | **Does not raise.** Returns the default. |
| `regs.names()` | Sorted slot names. |

> `get` versus `[]` matters: a step using `get` on an unwritten slot proceeds with
> `None` instead of failing. See the `accumulate` note under `src_to_dest`.

## `Step`

A dataclass. Four fields, one of which does the work.

| Field | Type | Role |
|---|---|---|
| `name` | `str` | Display only — the trace and `Chain.__repr__`. Factories build it as `f"l1_to_srcA({source})"` so a trace says *which* buffer. |
| `run` | `Callable[[Registers], None]` | The work. Takes the whole namespace, returns nothing — a step **writes into** the registers rather than returning a result. |
| `reads` | `Sequence[str]` = `()` | Declaration. Used by `dry_run` **only**; no effect at run time. |
| `writes` | `Sequence[str]` = `()` | Declaration. Used by `dry_run` and to decide what the trace prints. |

`reads` and `writes` are **not enforced**. `run` can touch any slot. A step that
touches an undeclared slot still works, but its trace omits the change and
`dry_run` stops being sound.

Steps are built by closure, so configuration lives inside `run` rather than in
extra fields:

```python
def l1_to_srcA(self, cfg, source="in0", into="srcA", index=0, src_format=None):
    l1_format = cfg.in_formats[index]          # resolved at build time

    def run(regs):                              # called later, by the chain
        regs[into] = self.blocks.l1_to_srcA(
            regs[source], l1_format, src_format, **cfg.geometry
        )

    return Step(f"l1_to_srcA({source})", run, reads=(source,), writes=(into,))
```

## `Chain`

An ordered list of steps run over shared registers. Build by construction, with
`then` / `repeat`, or by concatenating with `+`.

| Member | Notes |
|---|---|
| `Chain(steps=())` | |
| `.then(*steps)` | Appends, returns `self`, so calls read left to right. |
| `.repeat(times, steps)` | A loop, unrolled. **Re-runs the same step objects** — a body needing distinct inputs per iteration must be built with a comprehension giving each iteration its own steps. |
| `chain_a + chain_b` | Concatenation. |
| `len(chain)`, `iter(chain)` | |
| `repr(chain)` | `Chain(l1_to_srcA(in0) -> datacopy -> dest_to_l1(out))` — what a failure prints. |

### `Chain.run(registers=None, *, result=None, trace=None)`

The whole engine:

```python
regs = registers if registers is not None else Registers()
for index, step in enumerate(self.steps):
    step.run(regs)
    if trace is not None:
        trace.append(StageRecord(index, step.name, tuple(step.writes), …))
return regs[result] if result is not None else regs
```

| Argument | Notes |
|---|---|
| `registers` | Pre-populated namespace. A fresh one if omitted. |
| `result` | Return just this slot instead of the whole namespace. Ops pass `"out"`. |
| `trace` | A list **you** own; the chain appends one `StageRecord` per step, in place. `None` (default) skips the machinery entirely. |

The check is `if trace is not None`, not `if trace` — an empty list is falsy, and
an empty list is the only sensible thing to pass.

### `Chain.dry_run(available=())`

Static check: walks the chain tracking a **set of slot names** and reports every
step that reads one nothing has written yet. Calls no `run`, needs no data,
touches no torch. Returns a list of problem strings; empty means sound.

`available` must be seeded with the slots the caller pre-loads, or the inputs
look like problems:

```python
chain.dry_run()                                     # 3 false positives
chain.dry_run(["seed", "in0", "in1", "in1_t1"])     # []
```

Limits: it checks **ordering only** — nothing about formats, tile indices,
operand order or arithmetic — and it is only as good as the `reads`/`writes`
declarations.

## `StageRecord` and `summarise`

One record per *executed* step, appended to `trace`. A `Step` is persistent and
reusable; a record is created fresh on each execution, so one step can produce
many records.

| Field | Notes |
|---|---|
| `index` | Position **within one chain run** — it restarts at 0 for each block of a multi-block op. |
| `name` | Copied from the step. |
| `writes` | Copied from the step, as a tuple. |
| `summary` | `summarise()` of each written slot, **as a string computed at that moment**. |

That `summary` is text, not a tensor reference, is the point: `dest` is
overwritten repeatedly, so a reference would show the final value in every row.

`summarise(value)` formats by type — `(1024,) torch.float16 [0, 0.9922]` for a
tensor (shape, dtype, finite range), `1056 B` for a list or bytes, `repr`
otherwise. `__str__` pads to fixed columns so records line up:

```
 0. l1_to_dest(seed)         -> dest       (1024,) torch.float16 [8.637e-05, 0.9995]
 1. dest_to_srcA(dest)       -> srcA       (1024,) torch.float32 [8.637e-05, 0.9995]
 2. l1_to_srcB(in1)          -> srcB       (1024,) torch.float16 [0.002256, 1]
 3. reuse_dest:elwmul        -> dest       (1024,) torch.float16 [0, 0.9883]
 ...
 7. dest_to_l1(out)          -> out        1056 B
```

Read it as a story — above, Dest's minimum collapses to 0 at step 3, so you know
which transfer drove values under the flush threshold rather than only that the
final answer differed.

---

# `golden.py`

## `OpConfig`

How one invocation is configured. **Not** a register file: these are the knobs
the chain is *built* from, fixed before it runs. Data flows through `Registers`.

| Field | Default | Notes |
|---|---|---|
| `in_formats` | required | L1 format per operand. |
| `out_format` | required | L1 format to pack to. |
| `dest_format` | required | **Cannot be inferred from `out_format`.** Dest follows the *unpacker's* output, i.e. the input format. Derive it with `blocks.dest_format_for(in_format, dest_acc)`. |
| `geometry` | `{}` | `num_faces`, `face_r_dim`; spread as `**cfg.geometry` into every codec call. |
| `tiles_per_output` | `1` | Input tiles folded into one output tile. *How* they fold is the operation's business. |
| `relu_type` | `NoRelu` | Packer ReLU. |
| `relu_threshold` | `0.0` | Narrowed to the packer register's 16 bits before comparison. |
| `edge_mask` | `None` | `PackEdgeMask`. |
| `stoch_rnd` | `No` | Accepted, but only `No` is reproducible — anything else returns round-to-nearest and warns. `Fpu` included: it randomises the Dest write, not the pack. |

## `Golden`

| Member | Notes |
|---|---|
| `blocks_class` | Class attribute, set by the architecture subclass. Construction takes no arguments. |
| `op_name` | Used as the math step's trace name. |
| `Golden(blocks=None)` | Falls back to `blocks_class`; raises a pointed `TypeError` if neither is set, so using an arch-independent class by accident fails loudly. |
| `build_chain(cfg)` | **Abstract.** Every op declares its own pipeline. |
| `source(operand, tile=0)` | Slot name for one operand's buffer in one tile: `in0`, `in1`, then `in0_t1`, `in0_t2`. Tile 0 keeps the plain name so a non-folding op's trace stays uncluttered. |

### Step factories

Each returns a `Step`; none does work. Common arguments: `cfg`, `source` (slot
read), `into` (slot written), `index` (which entry of `cfg.in_formats` applies).

| Factory | Default `source` → `into` | Notes |
|---|---|---|
| `l1_to_srcA(cfg, source="in0", into="srcA", index=0, src_format=None)` | `in0` → `srcA` | `src_format` overrides the storage format the unpacker lands it in; `None` lets the architecture choose. |
| `l1_to_srcB(cfg, source="in1", into="srcB", index=1, src_format=None)` | `in1` → `srcB` | |
| `l1_to_srcS(cfg, source="in0", into="srcS", index=0, src_format=None)` | `in0` → `srcS` | Buffer must have been packed with `use_srcs=True`. |
| `l1_to_dest(cfg, source="in0", into="dest", index=0)` | `in0` → `dest` | Seeds Dest from L1, bypassing the src registers, so it keeps more mantissa. |
| `dest_to_srcA(cfg, source="dest", into="srcA", src_format=None)` | `dest` → `srcA` | Re-quantizes to a 19-bit datum. Conversion is driven by the **Dest** format. |
| `dest_to_srcB(cfg, source="dest", into="srcB", src_format=None)` | `dest` → `srcB` | |
| `dest_to_l1(cfg, into="out", source="dest")` | `dest` → `out` | Applies Dest precision, ReLU, edge mask, then really packs. |

### `src_to_dest(cfg, fn, *, reads=("srcA",), into="dest", accumulate=False)`

The only factory taking a function. `fn(regs) -> Tensor` supplies the arithmetic
— the operation's business — while the block decides how the result lands in a
Dest slot.

| Argument | Notes |
|---|---|
| `fn` | Reads whatever slots it needs out of `regs`. |
| `reads` | Declared for the trace and `dry_run`; must match what `fn` touches. |
| `into` | Dest slot. |
| `accumulate` | Add to Dest's existing contents instead of replacing, and **round to Dest precision on every pass** — the FPU has an accumulate-enable bit, not a wide accumulator. The step's trace name gains `+=`. |

> **Known gap.** With `accumulate=True` the step reads Dest via `regs.get(into)`
> but `reads` does not include it, so `dry_run` cannot see that dependency — and
> because `get` returns `None` rather than raising, an accumulating step placed
> before anything wrote Dest **silently degrades to replace**. Safe in the shipped
> ops (eltwise phase 0 uses `accumulate=False`; reuse-dest seeds with
> `l1_to_dest`), but worth knowing when adding an op.

### Running

#### `run(stimuli, in_formats, out_format, *, dest_acc=False, dest_format=None, num_faces=4, face_r_dim=16, num_tiles_per_output=1, trace=None, **pack_effects)`

Takes and returns what a test has: **tensors**. The blocks only handle L1 bytes,
so this is the adapter on either side — stimuli are packed into L1 buffers before
the chain runs, and the output buffer is read back after.

| Argument | Notes |
|---|---|
| `stimuli` | One tensor or a sequence, one per operand. Already tilized. |
| `in_formats` | One format applied to all operands, or one per operand. |
| `out_format` | Pack format. |
| `dest_acc` | Picks the 32- or 16-bit member of the Dest family. |
| `dest_format` | Explicit override; otherwise derived via `dest_format_for`. |
| `num_faces`, `face_r_dim` | Tile geometry. |
| `num_tiles_per_output` | `> 1` routes to `_run_blocked`. |
| `trace` | List to collect `StageRecord`s. |
| `**pack_effects` | Forwarded into `OpConfig` — `relu_type`, `relu_threshold`, `edge_mask`, `stoch_rnd`. |

> Do **not** pre-quantize stimuli to the output lattice. The stimuli and the
> golden go through the same packer, so they land on the same lattice by
> construction; pre-quantizing with a different rounding rule and then packing
> rounds twice.

#### `_run_blocked(stimuli, in_formats, cfg, trace)`

Taken automatically when `num_tiles_per_output > 1`. One chain run per block, so
each block folds its tiles into a single Dest and packs once — the output holds
one tile per *block*. Raises if the tile count is not a multiple of the depth.

#### `run_l1(l1_buffers, cfg, *, trace=None)`

Runs on real L1 buffers and returns an L1 buffer, skipping both conversions —
for chaining operations the way a kernel does.

All three set `self.last_chain`, which failures print to show the pipeline that
actually ran.

---

# `fidelity.py`

The phase machinery, shared because eltwise and matmul decompose identically.

| Member | Notes |
|---|---|
| `split_mantissa(values, keep_bits)` | `(high, low)` where `low = value - high`. |
| `operand_halves(srcA, srcB, split, phase)` | The slice of each operand a phase multiplies. |
| `min_normal_exponent(dest_format)` | Lowest exponent Dest holds as a normal; `None` for integer. |
| `ieee_exponent(values)` | `frexp`'s exponent minus one — the field an IEEE float stores. |
| `flush_pre_carry_denormals(product, srcA, srcB, min_exponent)` | The FPU's flush rule. Eltwise only; see the matmul section for why. |

### `split_mantissa` — why subtract rather than mask

The low half is `value - high`, not a masked mantissa re-read as a float: the
implicit leading 1 belongs to the high half, so masking in place does not give
the remainder.

**This is provably equivalent to the lightweight golden's mask-and-reassemble.**
Its Quasar masks are

```python
(0b11111111000, 0b11111111000)    # phase 0 — top 8 of an 11-bit {implicit, mantissa}
(0b00000000111, 0b11111111000)    # phase 1 — mant[2:0], implicit position cleared
```

and `reassemble_float_after_fidelity` runs the masked field through
`calculate_fractional_part` on an 11-bit string, so the low half's implicit bit
is *not* restored: `0b00000000111` → `0.0000000111 × 2^e` = `mant[2:0] × 2^(e-10)`,
exactly this remainder. Verified bit-identical across 10,000 values at four
widths.

Two corollaries worth keeping, because both have been chased as leads and are
dead:

- The Quasar masks are **symmetric** — 7 explicit bits from each operand,
  matching `MANTISSA_SPLIT = (7, 7)`. The asymmetric 4/6 split in that module's
  comment is its `else` branch, i.e. **Wormhole/Blackhole**, where SrcA also
  drops its least significant bit.
- Mask-and-reassemble versus subtract is **not** a source of divergence between
  the two goldens.

### `flush_pre_carry_denormals` — the pre-carry rule

The FP lane forms the exponent by adding the two *stored src* exponents and
rebiasing into Dest's range, then flushes the whole term — mantissa included —
when that lands below the lowest normal. The decision is taken **before** the
mantissa product's carry into the next binade, so the rule is one binade coarser
than testing the finished magnitude: when the mantissas multiply to ≥ 2.0 a
product anywhere in `[tiny, 2·tiny)` is a legitimate Dest normal that hardware
still returns as zero.

```
fp16 src, 16-bit Dest (bias 15):
  0.025390625 (2^-6 × 1.625)  ×  0.0024414062 (2^-9 × 1.25)
  exponents sum to -15  → lane flushes
  product is 6.199e-05, above fp16 tiny 6.104e-05 → a magnitude rule keeps it
```

Only bites for a five-bit-exponent Dest (fp16 src into 16-bit Dest); every other
combination accumulates with an eight-bit exponent and never reaches the
threshold. Only *visible* through an MX output, where one zeroed element shifts
the block scale enough to fail the block-aware compare.

> Use the **datum's** exponent, not a split half's. One exponent field serves
> every phase — a phase selects a mantissa window, not an exponent — so the sum
> is identical on all four. Using a half's exponent flushes almost everything,
> and scores worse than no flush at all.

---

# The operations

## `DataCopyGolden` — `datacopy.py`

```python
Chain([
    self.l1_to_srcA(cfg, source="in0"),
    self.src_to_dest(cfg, lambda regs: regs["srcA"], reads=("srcA",)),
    self.dest_to_l1(cfg, into="out"),
])
```

The dataflow with nothing in the middle. Worth more than it looks: because
`dest_to_l1` really packs, a format-converting copy is **requantized onto the
output lattice**, which a golden that only casts dtypes is not.

## `MatmulGolden` — `matmul.py`

```python
MatmulGolden(math_fidelity=MathFidelity.HiFi4, blocks=None)
```

A matmul decomposes into fidelity phases exactly as an element-wise multiply
does — the multiplier is narrower than a src datum, so each pass multiplies a
different slice of the operands' mantissas and the partial products accumulate
in Dest. The only difference is that the partial product is a *matrix* product.

| Member | Notes |
|---|---|
| `MANTISSA_SPLIT` | `(srcA_bits, srcB_bits)`. Quasar `(7, 7)`. `None` → the product is computed exactly and fidelity is ignored. |
| `OPERAND_REGISTERS` | Which src register each stimulus lands in, in argument order. `("srcB", "srcA")` on every architecture. |
| `models_fidelity` | True when `MANTISSA_SPLIT` is set. |
| `_product(srcA, srcB)` | The one place operand order lives, so a subclass inverting it overrides neither `apply` nor `partial_product`. |
| `TILE_DIM` | 32. |

### Accumulation: exact within a pass, rounded once at the Dest write

Two things the RTL rules out, both of which would be natural to assume:

- **No per-product denormal flush.** `SOP_IN_MAN_PREC = (MAN_PREC_A+1) + (MAN_PREC_B+1)`
  — products enter the sum-of-products network at full exact significand width
  and never become FP lane results, so `RES_A2`'s no-denormal rule does not
  reach them. This is the one place matmul and eltwise genuinely differ: there,
  each product *is* a lane result and the flush applies.
- **No per-MAC rounding.** `SOP_PREC = SOP_IN_MAN_PREC + SOP_GUARD_BITS` and
  `MAN_PREC_ACC = 23` — a wide fixed-point adder tree with guard bits, which
  rounds once. `_product` therefore sums in float64, so the model does not
  introduce a float32 accumulation error the hardware does not have.

Only the accumulated sum reaches Dest, where `src_to_dest` applies Dest
precision and the no-denormal rule as usual.

### Operand order

`run([arg0, arg1])` mirrors `_llk_unpack_matmul_init_(arg0, arg1)` and produces
`arg0 @ arg1`. What the caller passes never changes; the model's job is to get
right which register each operand rides, because that is what the mantissa split
is indexed by.

**Every architecture sends the first operand to SrcB and the second to SrcA**, so
this lives in the base class and no architecture overrides it. Wormhole and
Blackhole are not the mirror image of Quasar — `llk_math_matmul.h` says "D = in0
* in1, where in0 is loaded to SrcB and in1 to SrcA" on both, and
`llk_unpack_AB_matmul.h` agrees with "in0/inA - loaded to SrcB".

```python
OPERAND_REGISTERS = ("srcB", "srcA")    # init's arg0 → SrcB, arg1 → SrcA
def _product(self, srcA, srcB): return srcB @ srcA     # Dest = SrcB @ SrcA
```

Both halves are load-bearing and must agree. Either one alone transposes the
result into something wrong everywhere but still plausible-looking:

> This was a live bug until it was caught by cross-checking against the
> lightweight golden: the two disagreed on 1018/1024 datums purely because
> `stimuli[0]` was being routed to SrcA.

Flipping *both* is a no-op on the numbers, which is why the base class carried
the wrong routing harmlessly for a while: with `("srcA", "srcB")` and
`srcA @ srcB` the product is still `arg0 @ arg1`. It only becomes a real bug once
an architecture defines an asymmetric `MANTISSA_SPLIT`, because the split is
indexed by *register* — so the SrcA penalty would be charged to the wrong
operand. Wormhole/Blackhole will hit exactly that when their split is filled in,
SrcA there losing its least significant bit.

### Cross-validation against the lightweight golden

`helpers.golden_generators.MatmulGolden` (note the trailing **s** — a different
module) also models matmul fidelity and is exercised by four passing suites, so
it is a useful offline reference. Over 3 seeds × 4 fidelities, 32×32, fp16:

| fidelity | agreement |
|---|---|
| LoFi | **bit-exact, 0/1024, every seed** |
| HiFi2 · HiFi3 | 0–3 datums, 1 ULP |
| HiFi4 | ~345/1024 datums, 1 ULP |

bf16 operands make fidelity a **no-op** (0/1024 change LoFi → HiFi4), as a
7-bit mantissa must.

**The HiFi4 gap is a discriminating prediction, not noise.** This model gives
HiFi3 == HiFi4, because phase 4's AL×BL falls below Dest resolution; the
lightweight golden's HiFi4 differs from its HiFi3. Hardware gave HiFi3 == HiFi4
*byte-identical* for eltwise reuse-dest — same phase machinery, same argument —
which favours this one, but it has not been checked for matmul. A matmul run at
HiFi3 and HiFi4 settles it.

## `EltwiseBinaryGolden` — `eltwise.py`

```python
EltwiseBinaryGolden(operation=MathOperation.Elwadd,
                    math_fidelity=MathFidelity.HiFi4,
                    blocks=None)
```

Add and subtract are exact — the FPU splits only **multiplies** into fidelity
phases, so a multiply runs one accumulate step per phase and the HiFi4 chain has
four of them.

| Member | Notes |
|---|---|
| `MANTISSA_SPLIT` | `(srcA_bits, srcB_bits)` the multiplier takes per phase. Quasar `(7, 7)` — symmetric. `None` means the split is not modelled and the multiply is exact. |
| `models_fidelity` | True only for `Elwmul` with `MANTISSA_SPLIT` set. |

A consequence worth stating: a **bf16 operand carries only 7 explicit bits, so its
low half is always zero** and every phase after AH_BH contributes nothing.
Fidelity is a no-op for bf16 and only changes the result for fp16, Tf32 and
Float32 sources.

### Fidelity details

`split_mantissa`, the phase tables and the pre-carry flush all live in
[`fidelity.py`](#fidelitypy) — see that section. `EltwiseBinaryGolden` keeps
`split_mantissa` as a class member because the split is documented per
operation, but it delegates to the shared implementation.

`partial_product(regs, *, phase, dest_format=None)` takes the phase's operand
halves, multiplies them element-wise, and applies the pre-carry flush when
`dest_format` is given. Every phase reads the **original** `srcA`/`srcB` —
feeding a phase the previous phase's sliced operands zeroes every phase after
the first and turns fidelity into a silent no-op.

Unlike matmul, each product here **is** an FP lane result written to Dest, so
the flush applies.

## `EltwiseBinaryReuseDestGolden` — `reuse_dest.py`

```python
EltwiseBinaryReuseDestGolden(operation=MathOperation.Elwadd,
                             math_fidelity=MathFidelity.LoFi,
                             reuse_dest_type=EltwiseBinaryReuseDestType.DEST_TO_SRCA,
                             blocks=None)
```

Folds several input tiles into one output tile through Dest. Unlike an
accumulating op **nothing sums**: each pass *replaces* Dest with `srcA op srcB`,
and the feedback happens through the **operand** — one src register is loaded
from Dest rather than from L1.

```
l1_to_dest(seed)
repeat: dest_to_srcA, l1_to_srcB, math        # or the srcB mirror
dest_to_l1
```

The round trip is lossy on purpose: Dest is wider than a src register, so feeding
it back re-quantizes, and the chain models that where the hardware does it rather
than carrying full precision through the loop.

> **`DEST_TO_SRCB` swaps which stimulus is read.** The Dest feedback occupies
> SrcB, so the L1 operand must arrive through **SrcA** — the second stimulus
> tensor is never read and operand A multiplies against itself. Silent and total
> if you get it wrong.

### `run(stimuli, in_formats, out_format, *, inner_dim=1, output_tiles_in_block=1, dest_acc=False, dest_format=None, num_faces=4, face_r_dim=16, trace=None, dest_out=None, **pack_effects)`

Custom, because the kernel walks a block at a time.

| Argument | Notes |
|---|---|
| `inner_dim` | Input tiles collapsing into each output tile. |
| `output_tiles_in_block` | The inputs a given output tile consumes are **strided by this**, not contiguous: `block * inner_dim * otib + tile * otib + in_block`. |
| `dest_out` | Pass a list to collect each output tile's Dest contents **before the pack**. Separates "the math left a different value in Dest" from "the packer treated the same value differently" — a distinction the packed output alone cannot make. |

`SEED = "seed"` is the slot the seed tile is placed in before the chain runs.

---

# Architecture bindings

A subclass exists to bind `blocks_class` and state the constants that differ.
13–26 lines each.

```python
class QuasarEltwiseBinaryGolden(EltwiseBinaryGolden):
    blocks_class = QuasarDataTransferBlocks
    MANTISSA_SPLIT = (7, 7)
```

| Architecture | Has | Lacks |
|---|---|---|
| Quasar | MX — `MxFp8R`, `MxFp8P`, `MxFp4`, `MxInt8`, `MxInt4`, `MxInt2` | **No block float.** No `MxFp4_2x_A/B` — those are src-register formats, never L1 formats. |
| Wormhole · Blackhole | Block float — `Bfp8`, `Bfp8_b`, `Bfp4_b`, `Bfp2_b` | **No MX.** |

Available today: `QuasarDataCopyGolden`, `QuasarMatmulGolden`,
`QuasarEltwiseBinaryGolden`, `QuasarEltwiseBinaryReuseDestGolden`, plus
datacopy / eltwise / matmul for Wormhole and Blackhole.

---

# Adding an operation

1. Write the pipeline in `build_chain`, composing the step factories. Check it
   with `chain.dry_run([...pre-loaded slots...])`.
2. If the maths is not one of the existing `apply` implementations, pass your own
   `fn` to `src_to_dest`, and make `reads` match what it touches.
3. Subclass per architecture, setting `blocks_class` and any arch constants.
4. Validate against silicon at **full Dest width first** — run with a `Float16`
   output so the packed result *is* Dest, and compare in ULPs. Only then move to
   an MX output, where a lattice step is hundreds of ULPs wide and will hide a
   real modelling error.

## What this package does not model

- **Stochastic rounding** — device-seeded; golden returns round-to-nearest.
  Compare with PCC.
- **The HiFi phase residual** — raising fidelity moves the device 88–93 % as far
  as it moves the golden, on both reuse-dest feedback paths, with the *maximum*
  movement matching exactly (which rules out a wrong phase weight, a wrong split
  and a wrong accumulation rounding mode — all three would move the extremes).
  Currently sub-tolerance. Rejected explanations, on the record:
  truncating Dest accumulation instead of RNE (drops max movement to 66 where
  the device measures 71); a fitted `2^-15` magnitude flush (fit the numbers
  well, no mechanism); per-operand mantissa asymmetry (WH/BH only, Quasar is
  symmetric); and mask-versus-subtract in the split (provably equivalent).
- **Whether the matmul AL×BL phase is observable** — this model says HiFi3 ==
  HiFi4; the lightweight golden says otherwise. Unresolved for matmul, though
  hardware agrees with this model for eltwise.
- **Multi-tile layout for sub-1024-datum tiles** — `Golden.run` lays tiles out
  back to back, which matches `write_matrix_w_tile_dimensions`
  (`use_dense_tile_dimensions=True`). The *default* `StimuliConfig.write_matrix`
  strides the source by `MAX_TILE_ELEMENTS` whatever the tile size, so the two
  only coincide at 1024 datums per tile, or on a single tile. `check_source_layout`
  raises for the rest rather than quietly computing on data the device never saw.
