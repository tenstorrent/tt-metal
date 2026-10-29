# The heavyweight golden — a user manual

Start here. This is the orientation and how-to; the two subpackage READMEs are
the reference:

| | |
|---|---|
| [`operations/README.md`](operations/README.md) | every class and argument in the chain engine and the ops |
| [`data_transfer_blocks/README.md`](data_transfer_blocks/README.md) | every block, every conversion rule, and what is deliberately not modelled |

## Contents

- [Where things live](#where-things-live)
- [What this is, and why](#what-this-is-and-why)
- [Using it: write a test](#using-it-write-a-test)
- [How the layers connect](#how-the-layers-connect)
- [Adding things](#adding-things)
- [When a comparison fails](#when-a-comparison-fails)
- [What is not modelled](#what-is-not-modelled)

---

## Where things live

```
heavyweight/
├── operations/                 the pipelines, and the engine that runs them
│   ├── chain.py                Chain · Step · Registers — no hardware at all
│   ├── golden.py               Golden · OpConfig — blocks into chainable steps
│   ├── arch.py                 ArchProfile + the per-architecture factories
│   └── pipelines/              one module per operation
│       ├── datacopy.py · eltwise.py · matmul.py · reuse_dest.py
│
├── data_transfer_blocks/       what the hardware does to a buffer
│   ├── data_transfer_blocks.py the façade — one delegation per boundary
│   ├── arch.py                 ArchGolden + the three architectures
│   ├── l1_support.py           L1Support — format set, checks, L1 read/write
│   ├── storage.py              the precision rules, as pure functions
│   ├── l1_codec.py             bytes ↔ values
│   └── units/                  one module per hardware stage
│       ├── unpack.py · fpu.py · pack.py
│       ├── fidelity.py         the multiplier's phases, split and flush
│       └── pack_effects.py     ReLU, edge masking, stochastic rounding
│
└── mismatch.py                 failure reporting (lands with the test wiring)
```

**Two files named `arch.py`, on purpose.** They answer different questions:
`operations/arch.py` says which blocks an *operation* runs with;
`data_transfer_blocks/arch.py` says which units, format tables and multiplier a
*device* has. The operations one is what a test touches.

---

## What this is, and why

A golden that **does not compute its operation**. It declares the sequence of
transfers the hardware performs — L1 to src register, src to Dest, Dest to L1 —
and the answer is whatever falls out of the end.

The reason is narrow and worth stating plainly. Almost none of the goldens we
argue with are wrong about arithmetic. They are wrong about **where precision
is lost**, and precision is lost at boundaries:

- the unpacker landing an L1 datum in a 19-bit src register
- the FPU writing into a 16-bit Dest slot
- the packer requantizing onto an MX lattice whose scale is shared by 32 elements

A golden that computes in float32 and casts at the end has collapsed all of
those into one. When it disagrees with silicon it can only tell you *that*,
never *where*. Modelling the transfers puts each loss where the hardware takes
it, so a disagreement is localisable instead of a tolerance to widen.

It also means a loss fixed in one block is fixed for **every** operation that
crosses that boundary, rather than per-golden.

### Does it work

On Quasar reuse-dest ELWMUL — four chained multiplies through Dest, Float16 in
and out, 4096 datums, both feedback directions — it is **bit-identical to
silicon at LoFi, 0/4096 datums differing on both paths**. One result validating
input quantization, mantissa truncation, the multiply, Dest storage and
flush-to-zero, the Dest→Src feedback, tile indexing and the pack simultaneously.

Bit-exactness is the bar worth holding: a golden that agrees within a tolerance
only tells you the tolerance is wide enough.

---

## Using it: write a test

Pick the golden for your architecture and hand it tensors. There is no setup.

```python
from helpers.golden_generator.heavyweight.operations import QuasarEltwiseBinaryGolden

golden = QuasarEltwiseBinaryGolden(MathOperation.Elwmul, MathFidelity.HiFi4)
expected = golden.run(
    [src_A, src_B],            # the same tensors the harness packed into L1
    formats.input_format,
    formats.output_format,
    dest_acc=(dest_acc == DestAccumulation.Yes),
    num_faces=num_faces,
    face_r_dim=tile_shape.face_r_dim,
)
```

What comes back is a tensor of **output-format values**, ready to compare
against `res_from_L1`.

### The four goldens available today

| Class | Constructor takes |
|---|---|
| `QuasarDataCopyGolden` | — |
| `QuasarEltwiseBinaryGolden` | `operation`, `math_fidelity` |
| `QuasarMatmulGolden` | `math_fidelity` |
| `QuasarEltwiseBinaryReuseDestGolden` | `operation`, `math_fidelity`, `reuse_type` |

Wormhole and Blackhole have datacopy / eltwise / matmul equivalents —
`WormholeMatmulGolden`, `BlackholeEltwiseBinaryGolden` and so on. All of them
come from the same place: `heavyweight.operations`.

They are **factories, not classes**. Each is the generic op bound to an
`ArchProfile` in [`operations/arch.py`](operations/arch.py), which names the
architecture's blocks once rather than once per operation. Everything else
about the hardware the operation asks the blocks for.

### Arguments you will actually use

`run(stimuli, in_formats, out_format, *, dest_acc, dest_format, num_faces,
face_r_dim, num_tiles_per_output, dense_layout, trace, **pack_effects)`

| | |
|---|---|
| `dest_acc` | Takes the `DestAccumulation` enum or a bool. **Pass the enum.** It has no `__bool__`, so `DestAccumulation.No` is truthy — handed to a bool parameter it reads as "accumulation on" and silently selects a 32-bit Dest. |
| `dest_format` | Usually leave it out; it is derived from the *input* format, never the output. Pass it when the harness configured one explicitly. |
| `num_tiles_per_output` | More than one input tile folded into a single Dest before packing. |
| `dense_layout` | Say `True` if the stimuli came from `use_dense_tile_dimensions=True`. The golden cannot tell from a tensor which of the harness's two writers produced it, and guessing wrong computes on data the device never saw. |
| `trace` | Pass a list; it fills with one `StageRecord` per step. See [when a comparison fails](#when-a-comparison-fails). |
| `**pack_effects` | `relu_type`, `relu_threshold`, `edge_mask`, `stoch_rnd`. |

Reuse-dest takes `inner_dim` and `output_tiles_in_block` instead of
`num_tiles_per_output`, plus `dest_out` to collect the pre-pack Dest.

### One wiring detail

The heavyweight goldens are built **directly**, not through
`get_golden_generator` — `register_golden` instantiates with no arguments and
these take their operation and fidelity at construction. That means they miss
the swaps `TestConfig.setup_mode` installs, so guard the call:

```python
if not is_perf and should_compute_direct_golden("the heavyweight eltwise golden"):
    expected = golden.run(...)
```

Without it a compile-only CI job computes a full golden for every variant and
throws it away.

---

## How the layers connect

Four layers, each with one job. This is the whole architecture.

```
your test
   │  golden.run(...)
   ▼
Golden / build_chain          decides WHEN: wraps each block call in a Step
   │  self.blocks.l1_to_srcA
   ▼
DataTransferBlocks            routing only: self.arch.unpack.l1_to_src(...)
   │
   ▼
Unpack / Fpu / Pack           decides WHETHER: is this pair legal here
   │
   ▼
L1Support · storage           decides WHAT: bytes↔values, and what a register holds
```

### An operation is a list of transfers

`build_chain` produces `Step`s — a name, a closure, and declared reads/writes:

```
0. l1_to_srcA(in0)       reads=['in0']           writes=['srcA']
1. l1_to_srcB(in1)       reads=['in1']           writes=['srcB']
2. eltwise:elwmul        reads=['srcA','srcB']   writes=['dest']
3. dest_to_l1(out)       reads=['dest']          writes=['out']
```

Steps 0, 1 and 3 call blocks; step 2 is the operation's own arithmetic. Nothing
runs at build time — `chain.run()` fires the closures later. That separation is
what lets a chain be **checked** rather than only executed:

| Check | Catches |
|---|---|
| `Chain.dry_run(available)` | A step reading a register nothing wrote — a typo or an out-of-order chain, found without any arithmetic. |
| `Chain.unread(available)` | The quieter mistake: a value *staged* that no step reads, so the run completes and answers from less input than it was given. |

### What one Step actually does

Instrumented, a single `l1_to_srcA` descends like this:

```
-> DataTransferBlocks.l1_to_srcA     (facade — one delegation, nothing else)
  -> Unpack.l1_to_src                (unit)
    -> Unpack.is_valid_src_format    (is this L1->src pair legal on this arch?)
    -> L1Support.read                (bytes -> values)
    -> storage.to_src_storage        (values -> what a 19-bit datum holds)
```

The façade is genuinely thin — every method is a single delegation to
`self.arch.<unit>`. That is why the operations layer can do

```python
unpack = getattr(self.blocks, f"l1_to_{register}")
```

and never need to know a `Unpack` exists.

### Architectures are assembled, not subclassed

`ArchGolden` holds one unit per hardware stage. **Nothing overrides a method** —
there is exactly one implementation of each transfer, and architectures differ
in the data their units are constructed with. In Python an override is invisible
at the call site, so this removes the question "which implementation am I
reading?" entirely.

```python
class QuasarArch(ArchGolden):
    l1 = _QUASAR_L1
    unpack = Unpack(_QUASAR_L1, to_src_formats={...}, has_srcs=True)
    fpu = Fpu(_QUASAR_L1)
    pack = Pack(_QUASAR_L1, edge_mask_masked_when_set=True, ...)
```

---

## Adding things

### A new operation

Subclass `Golden` and declare the chain. The arithmetic is yours; the transfers
are not.

```python
class MyOpGolden(Golden):
    op_name = "myop"

    def build_chain(self, cfg: OpConfig) -> Chain:
        return (
            Chain([self.l1_to_srcA(cfg, source=self.source(0), index=0)])
            .then(self.src_to_dest(cfg, self.apply, reads=("srcA",)))
            .then(self.dest_to_l1(cfg, into="out"))
        )

    def apply(self, regs: Registers) -> torch.Tensor:
        return regs["srcA"].abs()      # your maths, on register-precision values
```

Then one factory per architecture, beside the others in
[`operations/arch.py`](operations/arch.py):

```python
QuasarMyOpGolden = _factory(MyOpGolden, QUASAR)
WormholeMyOpGolden = _factory(MyOpGolden, WORMHOLE)
```

No subclass: the architecture is an argument, so there is nothing for one to
add. If your op multiplies, ask `self.blocks.fpu_*` for the phases and the operand
windows — the FPU carries them, so you do not restate them.

Three things to get right:

1. **Declare `reads` honestly.** `dry_run` and `unread` both work off it, and a
   step that touches an undeclared slot still runs — it just stops being
   checkable and drops out of the trace.
2. **Accumulate with `accumulate=True`, not by summing yourself.** Rounding
   happens on every Dest write; accumulating at full precision and rounding
   once models an accumulator the hardware does not have.
3. **Read operands from the registers, not the stimuli.** `regs["srcA"]` is
   already at src-register precision. Reaching back to the input tensor skips
   the unpacker's loss, which is the entire point of the model.

### A new architecture

One `L1Support` and one `ArchGolden` in
[`data_transfer_blocks/arch.py`](data_transfer_blocks/arch.py), plus a factory.
Full recipe in that subpackage's README under *Where an architecture plugs in*.

### A unit whose behaviour genuinely differs

Only when the hardware really changed — not to express a different format
table, which is what the constructor arguments are for. Write a second
implementation of that one unit and name it after what differs, not `V2`. Every
unchanged unit is reused as-is.

**Tripwire:** if more than two or three units end up with per-architecture
variants, the data-vs-behaviour split has stopped paying and inheritance is the
better tool again.

### A new L1 format

The codec lives in `helpers/pack.py` and `helpers/unpack.py`, outside this
package — `l1_codec.py` only dispatches to it. Add the format to the relevant
architecture's `L1Support.formats` and to the unpacker's tables. `supports()`
answers two questions, and they are different: the architecture can hold the
format **and** this golden has a codec for it.

### Before you extend anything

Run the host tests. They are fast and need no device:

```
CHIP_ARCH=quasar pytest test_heavyweight_data_transfer_blocks.py \
    test_heavyweight_mismatch.py --compile-producer -q
```

They cover the blocks and the failure report. **The operations layer has no
host tests** — the chains, the fidelity phases, matmul's operand order and the
Dest-format derivation are exercised only by the three Quasar device tests, so
changing them needs hardware to verify.

---

## When a comparison fails

`passed_test` prints the failing tiles with the bad datums highlighted. It
cannot answer two things, and `mismatch.describe_mismatch` adds both:

- **how badly** — for an MX output the tolerance is a number of lattice steps,
  relative to each element's own magnitude. Ranked by absolute error, the top of
  a table is large datums that comfortably pass while the one that failed sits
  further down.
- **which stage** — it prints the golden's pre-pack Dest beside the packed
  result. A device zero against a golden Dest above the Dest format's smallest
  normal is not explained by flush-to-zero alone.

Use the two together:

```python
assert passed_test(golden_tensor, res_tensor, formats.output_format), describe_mismatch(
    golden_tensor, res_tensor,
    context=f"{formats.input_format}->{formats.output_format}",
    output_format=formats.output_format,
    chain=golden.last_chain,
    dest=golden_dest,
    dest_format=golden.last_dest_format,
)
```

`trace=[]` on the `run` call fills with one record per step — shape, dtype and
finite range at each boundary, which localises *which transfer* changed the
value.

To settle a flush-to-zero question, re-run with an output **in the Dest
format**: only then is the device's packed result its Dest. A Float16 output
does not do this unless Dest is Float16 too.

---

## What is not modelled

Stated plainly, because the package is larger than the part validated against
silicon.

- **Validated against silicon:** Quasar eltwise binary, eltwise reuse-dest and
  matmul, across the formats and fidelities those tests sweep.
- **Written but unexercised:** the Wormhole and Blackhole bindings and their
  blocks, the datacopy goldens, `Golden.run_l1`, `l1_to_srcS`, and several
  `Chain` helpers. Carefully reasoned and reviewed, but no datum has been
  compared against hardware through them.

Known gaps, each raising an explicit error rather than guessing:

| Gap | Behaviour |
|---|---|
| No L1 codec for `Tf32` (all) or `Bfp8` (WH/BH) | `supports()` is False; the error says it is a model gap, not a hardware limit |
| No mantissa split on WH/BH | A multiply there is computed exactly and **warns** — wrong at every fidelity, HiFi4 included |
| No SFPU unit | Nothing models the SFPU yet |
| Stochastic rounding | Device-seeded; the golden returns round-to-nearest and warns. Compare with PCC |
| Multi-tile sub-1024-datum layout | Refused unless `dense_layout=True` — the golden cannot tell which stimuli writer was used |

Those warnings are a `RuntimeWarning` subclass on purpose: `pytest.ini` sets
`ignore::UserWarning` suite-wide, so a plain `UserWarning` would reach nobody.
