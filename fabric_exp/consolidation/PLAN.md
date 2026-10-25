# Router consolidation (from 2026-10-24, after RESULTS §81)

Scope: ERISC-only mode (no DRISC offload), Blackhole P150x8, router kernel v2 at flags 13129088 as the starting point.

## Goals, in order
1. **Stable codegen.** A router source whose performance does not swing when someone makes a small, unrelated edit,
   or when the toolchain or optimization level changes. New contributors must not get tripped by it.
2. **Mainline-worthy.** Incorporates all or most of the study's optimizations, with no experiment flags or
   benchmark-only code, and performs in generic scenarios (1D, 2D, VC1, other slot counts), not just the 2D VC0
   benchmark configs.
3. Only then: cut further cycles of non-useful work.

## Rules
- The source must not assume compiler behavior (register allocation, scalarization of aggregates, loop peeling,
  layout). No register pinning, no layout steering. `variants/README.md`'s "Strict E0 Codegen Rules" are observations
  about GCC 15.1 at -O3 on s77–s80, not rules to write code by.
- Production already builds the router two ways: `-O3`, and `-Os` whenever VC1 is active
  (`FabricEriscDatamoverBuilder::get_kernel_opt_level`). A design must hold up at both.
- Holistic: designs are whole routers (both ERISCs, main loop, outer loop, connection handling), judged as whole
  builds by suite means across configs, opt levels and perturbations. Disassembly explains results; it does not
  drive edits. Ideas that lost as local patches may win inside a better overall shape (the runtime forwarding loop:
  −20% forwarding in §35/§36, nearly even in §81), so a past loss is not a refutation unless it lost on substance.
- Restructuring the router is allowed when it gets to a performant and stable shape.
- Standing study rules still hold: no RX work on E0; one worker channel per router; keep the per-pass worker
  connection check; a design that hangs even once is out; never score telemetry-on runs; benchmark-only bits never
  go in validation yamls.

## Phases
- **0. Make stability measurable.** (a) `ROUTER_MAP.md`: everything each ERISC does per pass and per packet, with
  costs, essential vs overhead. (b) `LEDGER.md`: every past experiment, classified (in the build / lost on
  substance / lost in context / open). (c) Offline builds from logged JIT commands, a perturbation suite (-O3/-O2/-Os,
  toolchain knobs, benign source edits) and a codegen proxy calibrated against variants with saved ELFs and hardware
  numbers. (d) Today's fragility on hardware: s79mc at -O3/-O2/-Os (job A, `p0a_runs.sh`) plus benign edits.
- **1. Strategies (agents, offline only).** Whole-router designs from the map and the ledger; each returns a
  semantic review plus stability and proxy scores. A faithful consolidation of s79m is the reference.
- **2. Hardware.** Batched broker jobs on the shortlist: suites + validation, each finalist also at -O2/-Os and
  with a benign edit. VC1 testing here.
- **3. Further cycle cuts** on the stable base.

Agents never run device jobs and never edit the live tree.
