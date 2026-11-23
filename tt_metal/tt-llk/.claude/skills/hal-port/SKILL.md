---
name: hal-port
description: Port a Blackhole LLK header (raw TT_/TTI_ instructions, cfg[] MMIO, regfile) to the HAL in tests/hal_llk, verify it instruction-for-instruction against production, test it on silicon through the chip queue, and optionally optimise it with measured, kept-only-if-faster changes. Use for any "port X to HAL", "HAL-ify", "optimise the HAL port of X" or "measure HAL vs TT" task on Blackhole.
---

# Porting Blackhole LLK to the HAL

`tests/hal_llk/` is a complete mirror of `tt_llk_blackhole/llk_lib/` (core headers at the top, `experimental/`
below). Building with `TT_LLK_IMPL_INCLUDE_DIRS=hal_llk` puts it first on the include path, so every test compiles
against it; without it the production headers are used. A header that has not been ported yet is an unchanged copy
of production, so the tree always builds. Porting a header means rewriting that file in place.

Read before you start:
- `references/porting-rules.md` — what a port may and may not change (same instructions, same access kinds,
  only two kinds of improvement), the TT/TTI → HAL mapping table, and how to verify.
- `references/hal-edge-cases.md` — every place where a correct-looking HAL call cost cycles, code size or
  correctness when eltwise binary was ported, and how to avoid it. Check your port against each item.

## Workflow (one work package = a set of headers, usually one thread's family)

1. **Scope.** List the headers of your work package and every production helper they call
   (`tt_llk_blackhole/common/inc/c{unpack,math,pack}_common.h`, `ckernel_template.h`, ...). Find the tests that
   exercise them: `grep -l '#include "<header>"' tests/sources/*.cpp`, then the `tests/python_tests/test_*.py` and
   `perf_*.py` that name those sources. Untested headers: say so in your report.
2. **Port** each header in place in `tests/hal_llk/` following the rules. Production helpers that emit instructions
   are ported into the same header under `namespace hal_port` with their production names; public `_llk_*`
   functions keep their exact signatures. Headers already ported by others (e.g. `llk_unpack_common.h`) may only be
   extended, never changed in behaviour; coordinate through the plan owner if you must.
3. **Check codegen** against production for at least one representative variant per thread:
   `bash .claude/skills/hal-port/scripts/compare_codegen.sh <test_x.py> <unpack|math|pack> [--perf]`
   (builds both implementations into private trees, pairs variants by `build.h`, prints `.text` sizes, Tensix
   mnemonic counts and a register-normalised diff of `run_kernel`). Every difference must be an allowed improvement
   or a listed HAL-GAP.
4. **Test on silicon through the chip queue** — never call pytest or run_test.sh directly while other agents share
   the 4 chips:
   ```
   Q=.claude/scripts/chip_queue.py
   python3 $Q run --worktree $PWD --test test_x.py --impl hal --owner <your-wp> --maxfail 20
   ```
   (`$PWD` = your worktree's `tt_metal/tt-llk`). `run` prints one result line (verdict, counts, log path) and exits
   0 only on PASS. Submit several jobs with `submit` and collect them with `wait <id>` to keep all chips busy.
   Every test that includes one of your headers must pass with `--impl hal`, with the same pass/skip counts as
   `--impl tt`. While other threads are still being ported elsewhere, validate your thread alone with a thread view:
   `bash .claude/skills/hal-port/scripts/thread_view.sh <unpack|math|pack> /tmp/bh-port/val/<your-wp>` and pass
   that directory as `--impl` (your thread's headers from tests/hal_llk, the other threads' from production). Known flaky/hanging-on-production tests are listed in the plan; check against `--impl tt` before
   blaming the port.
5. **Measure** (only when asked to optimise, or for the final report): run each affected `perf_*.py` with
   `--perf --impl tt` and `--perf --impl hal --tag <unique>`, then compare the two reports with
   `scripts/perf_compare.py <tt-run-dir> <hal-run-dir>` (geomean %, per run type and zone, faster/slower counts).
6. **Report** in this order: files changed; per function its production source and any deviation; HAL-GAPs;
   allowed improvements applied with their instruction-count effect; compare_codegen header lines; functional
   verdict lines (hal and tt) per test; for optimisation work the perf table. Never commit; the plan owner merges.

## Optimising a ported header

Only after the port passes functionally. One change at a time, each with a saved patch, a full functional run and a
perf run against the previous step. Keep a change only if every test passes and no config of any affected perf test
gets more than 2% slower end to end (L1_TO_L1). Typical sources of gain found so far (details in the edge-case
reference): per-tile RISC instruction count in RISC-issue-bound loops, MMIO polls that a STALLWAIT or Auto TTSync can
replace, replay/MOP structure that removes per-face RISC work, and stalls that wait for more than the hardware needs.
Anything that changes a cross-thread contract (data-valid hand-off, semaphores, context counts) or sticky per-thread
state (Auto TTSync, config contexts) must be scoped to the op that relies on it and undone in its uninit.
