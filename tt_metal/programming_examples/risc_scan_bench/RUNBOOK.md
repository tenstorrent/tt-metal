# risc_scan_bench RUNBOOK — Gate-2 scalar-RISC scan/emit rate constant "X"

Single-core (0,0) microbenchmark measuring the BRISC/NCRISC scan and emit rate
over N = 65536 bf16 values resident in L1. This is the one unmeasured constant
that prices every RISC-side materialization candidate of the top-k selector
campaign (RADIX_BUCKET_GPU.md Gate 2; storm/research/materialization.md §2/§7;
storm/research/risc-radix.md §3).

## Build

The repo build dir at `build/` is already configured with
`BUILD_PROGRAMMING_EXAMPLES:BOOL=ON`, so an incremental build of just this
target suffices (CMake re-runs automatically on the CMakeLists change):

```bash
cd /home/nachiket/tt-metal
ninja -C build metal_example_risc_scan_bench
```

Full-path alternative (rebuilds everything, only if the incremental route
fails):

```bash
cd /home/nachiket/tt-metal
./build_metal.sh --build-programming-examples
```

Binary lands at `build/programming_examples/metal_example_risc_scan_bench`.

## Run (device-using — always under the device lock)

```bash
cd /home/nachiket/tt-metal
export TT_METAL_HOME=/home/nachiket/tt-metal
flock /tmp/tt-device.lock ./build/programming_examples/metal_example_risc_scan_bench
```

Run it 3 times and quote the median of each constant; on-core wall-clock
variance is small but the first invocation pays JIT compiles (the timed loops
themselves are warmed up in-kernel, so even run 1's numbers are valid).

Exit code 0 = every (variant, pattern) cell verified bit-exactly against the
CPU reference. Any FAIL prints a `******** FAIL ... ********` line and the
process exits 1 — do not use the numbers from a failing run.

## What it prints

A table of `variant | pattern | cycles | cyc/elem | cyc/origelem | PASS/FAIL`
for 7 variants x 3 patterns (uniform-random u16 bits, clustered/same-bin
adversary = one high byte, all-equal), then the harvested decision constants:

| constant | variant | meaning |
|---|---|---|
| `X_load_floor` | v7 uniform | pure lbu-stride-2 load+sum floor, cyc/elem |
| `X_hist` | v1 uniform | 256-bin high-byte histogram in local data RAM, cyc/elem |
| `X_hist_clust` | v3 clustered | 4 interleaved sub-histograms (+timed merge) on the same-bin adversary, cyc/elem |
| `X_dense_emit` | v4 uniform | threshold-compare dense emit (candidate (a) inner loop), cyc/elem |
| `X_consumer` | v5 uniform | skip-zero-words compressed-stream consumer (candidate (e) inner loop), **cyc/WORD** |
| `X_dual_agg` | v6 uniform | dual-RISC v1 split, aggregate cyc/elem (max of the two RISCs / N) |

v2 (histogram in L1) prices the non-coalesced L1 store rule vs v1 directly.
All-equal cells for v4/v5 deliberately overflow the 4096-entry emit cap: the
overflow flag must fire and the count must stay exact (loud, never silent).

## Decision rule (from the research reports)

- **X_consumer <= ~5–6 cyc/word** → Gate-2 candidates (a)/(e) stay ALIVE:
  the (e) fused-key consumer prior of 0.12–0.31 cyc/original-element holds and
  Gate 2 proceeds on the (e)+(b) composition (materialization.md §7).
- **X_consumer >= 10 cyc/word** → the RISC arm is KILLED as a primary path:
  demote dual-RISC to Gate-4 correctness oracle / Gate-2 emit engine only
  (risc-radix.md §3 decision rule). One 16:1 cascade stage (cost ≈ 1/16 of a
  pass) is the documented escape hatch before declaring FAIL.
- Selector-arm side rule: **X_hist (v1 uniform) <= 6 AND X_hist_clust (v3
  clustered) <= 8** keeps the multicore width-sharded selector arm alive →
  proceed to the tree-reduction bench; **>= 10** demotes it.
- Sanity anchors: X_load_floor bounds everything from below (if X_hist ≈
  X_load_floor the histogram stores are free; if X_hist >> X_load_floor the
  store path is the problem). v2−v1 isolates the L1-store tax. v6 vs v1 shows
  dual-RISC scaling (ideal: X_dual_agg ≈ X_hist/2).

## Caveats / known risks

- Cycles are tensix wall-clock ticks (`RISCV_DEBUG_REG_WALL_CLOCK_L`), same
  clock domain as the baby RISCs — ratios are exact, absolute µs need AICLK.
- Kernel is JIT-compiled (likely `-Os`); if a number looks absurd, inspect the
  generated ELF/disassembly under `built/.../kernels/scan_bench/...` before
  believing it — compiler codegen is the top estimate-breaker called out in
  risc-radix.md §4.5.
- If the JIT link ever fails with a local-memory overflow on the kernel, drop
  `NSUB_HIST` in `kernels/scan_bench.cpp` from 4 to 2 (v3 uses 4 KiB of the
  8 KiB BRISC/NCRISC local data RAM for its sub-histograms).
- v3's timed region includes the 256x4 sub-histogram merge — that is the true
  cost of the mitigation, so v3 can only beat v1 on same-bin adversaries.
- Input is scanned flat (ROW_MAJOR-equivalent). The TILE face-run traversal tax
  (~0.25 instr/elem, risc-radix.md §4.1) is NOT measured here; add it on paper.
