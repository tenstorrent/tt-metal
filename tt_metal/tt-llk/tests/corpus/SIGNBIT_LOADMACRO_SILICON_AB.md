# Typed Signbit SFPLOADMACRO silicon A/B

This test-only A/B compares the existing Blackhole production Signbit LLK with
a fresh semantic C++ body.  The fresh source expresses a typed destination
load, logical sign-bit shift, integer-to-float conversion, typed store, and Dst
advance.  It contains no raw instruction words, fixed LREG selection, macro
descriptor, replay slot, or hand-written schedule.

The default-off compiler pass recognizes only the exact closed typed region.
For a canonical single-block loop with one structural preheader and no other
Tensix/config/call/asm owner, it materializes the immutable descriptor once in
the preheader and emits one delayed SFPLOADMACRO launch plus three drain slots
per iteration.  Unsupported, multiply-owned, QSR, odd-address, out-of-range,
live-out, and LREG-pressure shapes retain the original body.

## Gates

- SFPI-GCC: `476abbd2e` (rebased local review SHA; commits are pushed by the
  landing workflow only after all gates below).
- CRAQ: `51c3e76edb8efa80b60c7332273213102d7f887a`.
- TT-Metal selector base: `981b74ee` before this evidence commit.
- Compiler focused DejaGNU: 96/96; compiler C/C++ selftests pass.
- WH/BH compile: pass. QSR deliberately retains the explicit typed sequence.
- Linked WH/BH helper: exactly four SFPCONFIG writes before the loop, one macro
  launch in the loop, and three explicit drain NOPs. BH uses auto-increment
  address mode 6; WH uses mode 2.
- Paired CRAQ correctness: production/fresh pass on both WH and BH (4/4).
- Paired BH silicon correctness: production/fresh pass (2/2), using the existing
  Float32 `passed_test` contract: element tolerance rtol=0.05, atol=0.05 and
  PCC > 0.99 on signed finite nonzero stimuli.

## Serialized Blackhole result

All values are the raw `TILE_LOOP` row's `mean(MATH_ISOLATE)` from independent
profiler processes. They are scoped device-cycle measurements, not wall time or
whole-kernel throughput.

| selector | sample 1 | sample 2 | sample 3 | mean | result |
|---|---:|---:|---:|---:|---:|
| production hand-tuned LLK | 23246 | 23246 | 23246 | 23246 | baseline |
| fresh typed + compiler macro | 21508 | 21508 | 21508 | 21508 | -1738 (-7.4766%) |

With `tile_cnt=8` and `loop_factor=16`, the scoped improvement is 13.578125
cycles per tile. The MATH_ISOLATE text size also falls from 2916 to 2764 bytes
(-152 bytes).

Evidence is archived at
`/localdev/nkapre/signbit-competitive-silicon/archive`. The aggregate SHA-256
over the six raw/post CSV files is
`760f83b21f391229b633d9c324b10a61c388523875f638392270814e0f0057c6`.
The aggregate hashes over the five BH and five WH math ELFs are respectively
`29343dc741a46eb3bb17cfe6cd84f8ef86edca1eb256c6670714e712d65c7ba7`
and `e8cebec07fe52f3d68991722d91034c3b8ffa60677adba3eb4c12fe35122235b`.
