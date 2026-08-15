# TopK typed-SFPI conversion blocker

## Decision

**BLOCKED before selector — do not claim a generated TopK A/B yet.**  The
current typed SFPI interface cannot represent the full architectural state
changed by TopK's two central instructions.  Copying the production header and
leaving these operations as opaque TTI would produce a nominal selector, but it
would not expose the compare/sort core to the compiler and therefore would not
test the intended compiler flow.

Production LLKs are untouched.  The compile-only reproducer is
`topk_typed_index_tracking_probe.cpp`; run it with
`run_topk_typed_probe.sh [output-directory]`.

## Reproduced gap

`_init_topk()` sets bit 2 of `SFPU_CONTROL_REG`, enabling index tracking.  In
that mode, every physical `SFPSWAP` on value registers L0--L3 also conditionally
moves the corresponding index registers L4--L7.  The GCC pattern
`rvtt_sfpswap_int` has two inputs and two outputs only.  Neither its RTL nor
`sfpi::min_max` names the companion index pair.

Likewise, physical `SFPTRANSP` transforms the L0--L3 and L4--L7 groups.  The
typed `__builtin_rvtt_sfptransp` accepts four inputs, returns four outputs, and
`rvtt_sfptransp_int` constrains only L0--L3.  It has no dataflow edge for the
simultaneously transformed L4--L7 group.

The probe compiles for both `tt-bh-tensix` and `tt-wh-tensix`.  Both assemblies
show:

```text
# READ L0
# READ L2
SFPSWAP L2, L0, 1
# WRITE L0
# WRITE L2
# READ L4
# WRITE L4
# READ L6
# WRITE L6
```

The L4/L6 operations after `SFPSWAP` are independent zero-length fixed-register
round trips; the swap itself has no L4/L6 output.  The transpose probe similarly
emits one `SFPTRANSP` for L0--L3, followed by independent L4--L7 reads/writes.
Thus GCC is free to reason from a false unchanged-index state even though
silicon changed it.  Volatility prevents deletion of the instruction; it does
not repair the missing defs or establish the required value/index relationship.

Probe evidence generated during this audit is in
`/localdev/nkapre/topk-probe`.  Assembly SHA-256 values were:

- BH: `6cbd032eb5cadbc629acab17a0c2532f68fbb214cf0e43c7b22d90cc29200564`
- WH: `ef2de2bb9cd7ead1c72bce7f7ef2832a39fb993a36d14bad24da825133d3db1a`

## Required general compiler fix

1. Add an indexed compare/swap builtin whose RTL describes four logical
   results: both value results and both companion-index results, while emitting
   one physical `SFPSWAP`.  Its constraints must enforce the architectural
   `index_reg = value_reg + 4` relationship; ordinary independent allocation is
   not legal.
2. Add an eight-value transpose form whose RTL describes both LREG groups while
   emitting one `SFPTRANSP`.  Keep the current four-value helper for callers
   that prove L4--L7 dead, but it is insufficient for TopK.
3. Give both forms the existing Tensix issue/delay attributes so scheduling can
   operate on them, and teach replay formation to retain the multi-result
   operations as single encodable instructions.
4. Add post-RA verification: indexed swap operands must be L0--L3 and every
   live companion must occupy exactly L4--L7; TopK transpose must expose all
   eight hard-register defs.  Fail compilation rather than silently accepting
   a partial model.
5. Only then build the test-only whole-TopK selector, preserving the current
   RWC/DST loads, stores, `SFPCONFIG` direction changes and replay boundaries.

This is a general SFPU paired-state fix.  It also applies to any future kernel
that uses index-tracking `SFPSWAP` or keeps L4--L7 live across `SFPTRANSP`; it
must not be implemented as a TopK-name special case.

## Why no CRAQ or silicon numbers

The functional gate must follow a semantically complete compiler model.  A
physical test could happen to pass today because the allocator selected the
expected fixed LREGs, while the RTL still permits later scheduling/allocation
changes to reuse or misinterpret a hidden companion value.  Running performance
on that accidental encoding would bless an unsound interface, so this audit
stops at the reproducible WH/BH compiler discriminator.
