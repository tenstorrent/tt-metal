# Reducing RISC-V instruction issue with the MOP expander and replay buffer

This describes a method, and an open tool that implements it, for minimising the
number of RISC-V instructions a baby core spends issuing a fixed sequence of
Tensix instructions, using the two levers the hardware gives you for that,
without changing the Tensix stream that comes out the other end. The question it
tackles was raised in
[tt-llk#1638](https://github.com/tenstorrent/tt-llk/issues/1638).

The tool, **tt-mop**, is open source under Apache 2.0 at
[github.com/Zaneham/tt-mop](https://github.com/Zaneham/tt-mop). I wrote it as the
optimiser behind my open CUDA compiler's Tensix backend, and it turned out worth
writing up. The goal here is that someone on the LLK team can read this, run the
tool on an op, understand what its numbers mean (and where they lie), and extend
it to a new op or architecture without me!

## The two levers

A baby core issues Tensix instructions one at a time, so a stream with long
repeated runs wastes the core pushing one instruction per cycle. Both levers to
avoid that are already used by hand all over the LLK:

- **The MOP expander.** One `MOP` issue expands a template into a whole run.
  The double-loop template (`ckernel_template`, in
  `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_template.h`) covers up
  to 32 outer by 64 inner instructions from a single issue. The exact
  emit-vs-swallow behaviour of the NOP slots is pinned down in the ISA docs'
  [MOPExpander.md](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MOPExpander.md).
- **The replay buffer.** `lltt::record` captures a short sequence, a later replay
  reissues it for free. On Wormhole and Blackhole the buffer holds up to 32
  instructions and is shared between threads, so the Math thread cannot grab
  slots the SFPU needs.

## Verifier-first, and why not an LLM

The issue floats an AI agent generating suggestions to filter, and I went
deliberately the other way: a deterministic search with a verifier in front of
it. For this problem a plausible-but-wrong arrangement is worse than none. An
off-by-one replay offset or a wrong loop count looks completely reasonable and
only bites on silicon. So every arrangement the tool proposes gets expanded back
into the exact Tensix stream and checked word for word against the original
before it's allowed to count. Nothing incorrect comes out, so there's no
"filter the good ones" step left to do.

On top of that the expander is fuzzed against a port of
[ttsim](https://github.com/tenstorrent/ttsim), so the thing doing the checking is
itself checked against Tenstorrent's own simulator. ttsim is ground truth: if the
two ever disagree, ttsim wins and it's a bug in mine. Net effect, you can trust
the tool without a chip in hand.

## How it is built

The pieces, in dependency order:

- **Plan IR** (`include/ttmop.h`, `tm_planop_t`). A plan is a sequence of ops:
  direct emit, program a MOP template (double-loop or unpack), run a MOP, load
  the replay buffer, run the replay buffer. Both MOP template shapes are
  modelled, including the last-instruction substitution and the `loop_op1`
  alternation that doubles the inner count.
- **Expander** (`src/ttmop_expand.c`, `tm_expand`). Plays a plan forward into the
  exact Tensix word stream it produces. MOP bodies, replays and direct emits all
  route through one `tm_issue` path, so a MOP whose body is a REPLAY word expands
  exactly as the frontend would compose it (the matmul's actual shape). The
  frontend's NOP-eat is modelled too, so the stream is the NOP-free backend
  trace.
- **Verifier** (`tm_verify`). Expands a plan and compares it to the target word
  for word, reporting the first divergence. Every candidate plan and the
  all-direct fallback both pass through this.
- **The oracle** (`tests/ttsim_ref.c`). A faithful port of ttsim's `mop_expander`
  / `replay_expander` / `tensix_push_inst_fifo`. The fuzzer (`tests/tfuzz.c`) runs
  thousands of random valid plans through both the expander and the oracle every
  test run and asserts they agree.
- **Cost model** (`src/ttmop_opt.c`, `tm_cost`). Counts RISC-V issues per plan op.
  The constants are in `ttmop.h` (`TM_COST_*`). **Read the limitations section
  before trusting any number this produces.**
- **Optimiser** (`tm_optimise`). Weighs replay-block and MOP-run candidates in one
  greedy covering, picks whichever is cheaper per segment, and gates every plan
  through the verifier.

Build and test:

```
make            # the tool
make test       # the suite, including the oracle's fuzz pass
```

Clean under `-Wall -Wextra -Werror -pedantic` and a pile of other flags.

## Using it on an op

The CLI works on a flat little-endian file of 32-bit Tensix words (the kind ttas
emits):

```
ttmop --opt f.bin [slots]      # optimise, report the saving
ttmop --suggest f.bin [slots]  # print an apply-this-by-offset arrangement
ttmop --dump f.bin             # print the words
ttmop --demo                   # expand a sample plan
```

`slots` is the replay-buffer budget for this thread (default 32; pass 16 if Math
and SFPU are sharing the buffer).

The procedure for a real op:

1. **Get the op's Tensix stream.** Either lift the `TTI_*` sequence and recorded
   block by hand and encode it with the `TT_OP_*` macros from `ckernel_ops.h`, or
   capture the issued stream out of ttsim. The stream you want is the NOP-free
   backend trace, the instructions the engine actually receives.
2. **Write it as a flat little-endian `.bin`.**
3. **Run `ttmop --opt`** to see the arrangement and the modelled saving, then
   **`ttmop --suggest`** for the by-offset version a human can apply.
4. **Read the number against the limitations below before acting on it.**

### Worked example: the Quasar matmul

`tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_math_matmul.h` records a 15-MVMUL
block and programs `ckernel_template(1, FIDELITY_PHASES, REPLAY(0,15),
matmul_op)`, a double-loop that alternates a replay of the block with a single
MVMUL. At HiFi4 that expands to 64 MVMULs. Fed that raw stream, tt-mop reports:

```
RISCV issues : 64 -> 23   (saved 41, 64.1%)
verified     : yes
```

It loads the 15-block once and replays it three times, the same record-once-
replay trick the C++ hand-codes. It's also, spoiler, **wrong about that being a
win**, which is exactly why it's the first limitation below.

## What the tool actually found: the ops are already well tuned

Run over the existing LLK, the most useful thing the tool does is confirm you
already did it right. The hand-written arrangements are at or near optimal:

- **`transpose_dest`** records one buffer and composes the unpack template out of
  overlapping replay sub-ranges. That is a lovely use of the buffer and the tool
  does not improve on it.
- **matmul** does record-once-loop, which the tool arrives at independently.
- **reduce** wraps its pool loop.

Two questions came up on the issue that this answers directly.

**Can the tool model the overlapping replay-buffer usage `transpose_dest`
relies on?** Yes. The plan IR expresses a MOP whose body replays a sub-range of a
recorded buffer, and the expander composes overlapping replay sub-ranges the way
the frontend does, so an arrangement of that shape round-trips through the
verifier. What it does not yet do is *prefer* it on cost grounds, for the
amortisation reason below.

**Where is the real headroom?** Not in per-op tuning, that is already thorough.
The surface an automated pass can see that a human tuning one op cannot is:

- **Whole fused kernels rather than single ops.** Per-op tuning cannot see
  repetition that spans op boundaries; a pass over the fused stream can.
- **Newer ops that have not had the `transpose_dest` treatment yet.**
- **Quasar ops.** Many are ported from Blackhole and Wormhole and use the same
  `ckernel_template` and `lltt::record` mechanism, so the expander already models
  them. What changes per arch is the opcode set and the buffer size. New Quasar
  instructions (`MVMULDI`, the direct-indexing matmul variant) need to exist in
  the stream producer before the tool can chew a stream containing them.

## Extending it

- **New instructions.** tt-mop works on opcodes, so a new arch's instructions just
  need to be encodable in whatever produces the stream.
- **New architectures.** Quasar uses the same mechanism as Wormhole and Blackhole,
  so the expander already models the behaviour. What changes per arch is the
  opcode set and the replay-buffer size (`TM_REPLAY_SLOTS`).
- **Keeping the oracle honest.** The oracle is only as good as the ttsim it was
  ported from. As the spec and ttsim are scrubbed, re-port the changed logic into
  `tests/ttsim_ref.c` so the fuzz pass keeps catching real divergence.

## Limitations

Read these before you trust a number. The verifier guarantees the *stream* is
correct. It does not guarantee the *cost* is meaningful, and on real ops that gap
matters.

### 1. The cost model is one-shot; the LLK ops are program-once-run-many

This is the big one, and the Quasar matmul above is the worked example.

tt-mop costs a plan as if the stream is produced **once**. But an LLK op does not
produce its stream once. It **programs the MOP at init and runs it per call.**
Costed in tt-mop's own model:

- **Hand-written:** record the 15-block (1 + 15 = 16) + program the MOP (sync and
  nine config stores, ~10) + run (1) = **~27 to set up, then ~1 per call.** One
  `ckernel_template::run()` issue expands to all 64 Tensix instructions, every
  call.
- **tt-mop's "23":** the cost to build the stream from cold. As a per-call number
  that is 23, or ~7 if the buffer is pre-loaded once.

So for a matmul called N times, the hand-written arrangement is `27 + N` and
tt-mop's plan is `16 + 7N` at best. For any real N the hand-written MOP wins
decisively, because it amortises the setup and collapses the whole stream into a
single run instruction. **tt-mop "found a 64% saving" that would make the op
slower the moment you call it twice.**

The verifier did its job, the 23-issue stream really is correct. The cost model
is what misled, because it has no concept of amortisation, which is the entire
reason the MOP hardware exists. This is also why every hand-tuned op uses the MOP
even where a one-shot replay looks cheaper.

**The fix:** an amortisation-aware cost model. Separate the setup cost (paid once)
from the per-call run cost, take the op's expected invocation count as an input,
and minimise `setup + N * per_call` rather than the one-shot total. Until that
exists, tt-mop systematically under-values the MOP for repeated ops and should
not be used to talk anyone out of a MOP arrangement they already have. It's
sound for *finding* a correct arrangement; it's not yet sound for *comparing*
arrangements by cost.

### 2. The cost constants are uncalibrated

The `TM_COST_*` values in `ttmop.h` are best-guess, not measured against real
RISC-V emission. The numbers are model-relative. Calibrating them against actual
issue counts is a prerequisite for the comparisons in (1) to mean anything in
cycles.

### 3. Greedy covering is not provably optimal

The optimiser takes the highest-saving candidate that fits and does not collide
with one already chosen, then the next. That is optimal when the chosen blocks
are disjoint (the common case) and the verifier keeps it correct regardless, but
optimal covering under collision is unsolved here. A run that a chosen replay
block cuts across is dropped whole rather than re-fitted to the unclaimed prefix.

### 4. Targets must be the NOP-free backend trace

The expander models the frontend eating every NOP, so a target carrying literal
NOPs will not round-trip and the optimiser will (correctly) report it unverified.
Feed it the executed backend stream, not the issued instruction list with
padding.

### 5. Long single-instruction runs leave a tail

A single instruction repeated more than 2048 times exceeds what one MOP
double-loop can cover (32 outer by 64 inner), so the remainder is emitted direct.
Tiling a long run across several MOP loops is not yet implemented.

### 6. ISA coverage is per-architecture

The tool reasons about opcodes; it does not itself know an arch's full
instruction set. New instructions (Quasar's `MVMULDI`, for instance) need to
exist in the stream producer before tt-mop can optimise a stream that contains
them.

## References

- tt-mop, the tool: [github.com/Zaneham/tt-mop](https://github.com/Zaneham/tt-mop) (Apache 2.0)
- ttsim, the oracle: [github.com/tenstorrent/ttsim](https://github.com/tenstorrent/ttsim)
- ttas, the assembler that emits the input streams: [github.com/Zaneham/ttas](https://github.com/Zaneham/ttas)
- MOP expander ISA reference: [tt-isa-documentation MOPExpander.md](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MOPExpander.md)
- Where the question came up: [tt-llk#1638](https://github.com/tenstorrent/tt-llk/issues/1638)
