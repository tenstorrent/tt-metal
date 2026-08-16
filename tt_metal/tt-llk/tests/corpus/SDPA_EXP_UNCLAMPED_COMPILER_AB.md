# SDPA Exp upper-unclamped compiler A/B

## Scope and attribution

This lane measures the generic SFPI counted-loop replay-hoist pass on the clean
semantic SDPA Exp upper-unclamped implementation. Production is unchanged. The
primary OFF/ON comparison uses identical C++ source, input, golden, compiler
binary, and final link; the sole compiler-input difference is
`-mtt-tensix-optimize-replay-hoist`. There is no operation-name compiler hook.

The exact paired correctness node is:

`test_sfpu_sdpa_exp_unclamped.py::test_sfpu_sdpa_exp_unclamped[formats:Float16_b->Float16_b-input_range:(-20.0, 0.0)-scale_en:True-scale:16256-num_tiles:1]`

The profiler fixture narrows the timestamp region to the one-tile SFPU call and
runs in `MATH_ISOLATE`. Each reported sample is a separate pytest process with
a unique `RUNNER_TEMP`, raw/post CSV, log, ELF, and JUnit result.

## Correctness and executable gate

Both OFF and ON passed the exact node under CRAQ simulator commit
`8386fe45bc8e332ad84db55f7105b71e43785e7e` and physical Blackhole. The final
linked executable images reproduce the scout checkpoint exactly:

| compiler arm | final ELF SHA256 | `.text` SHA256 | `.text` bytes | loop form |
| --- | --- | --- | ---: | --- |
| OFF | `4af1cc40a37bf77b14d754d56a5676f789580fcd98abc01d7896b5ce2bec7ed2` | `c9ac3da4baff4acb004716ed907591dc93ecc653b6e14e22653d9a9cba317815` | 1028 | eight executions of an explicit 24-instruction body |
| ON | `0376200b3bda28a401deb9e8cab35d76668835acf527f9af4b4642fca1c872ef` | `c770a99af8f3d8f3667763d110770bdffd2b7008a869504d9af8cede85c3ffb0` | 1036 | one 24-slot capture plus seven replay launches |

The compiler is SFPI-GCC
`e17a4f8fdd733cf523d5d8d4c37c15be41b4433d`; the actual driver binary SHA256
is `a6fe054dea8b08e1131a0e233679e1b149ea56c30f10490bb98a7bbbd405f041`.

## Blackhole silicon 2x2

The production handwritten control final linked `.text` is byte-identical with
the pass OFF and ON: SHA256
`45accd4b547f9c8451967636ee17bce0989636eb149a0afbafe1fee251961b07`,
1392 bytes. Its SFPU body is therefore profiled once and reused for both control
cells.

| implementation | pass OFF, body cycles r1/r2/r3 | pass ON, body cycles r1/r2/r3 |
| --- | ---: | ---: |
| clean semantic upper-unclamped Exp | 1048 / 1048 / 1048 | 945 / 945 / 945 |
| production handwritten Exp | 632 / 632 / 632 | 632 / 632 / 632 (same executable; reused) |

The same-source compiler effect is **-103 cycles, -9.828244%, or 1.108995x**:
a reproducible compiler-only win. Competitiveness is separate: optimized
semantic C++ is **+313 cycles, +49.525316% slower** than handwritten production.
The overall broad replacement classification is therefore **loss**, not win or
tie; replay hoisting closes part of the gap but does not reach the handwritten
kernel.

## Evidence

The complete archive is
`/localdev/nkapre/sdpa-exp-unclamped-evidence`. It contains final-source paired
CRAQ JUnit results, final linked ELFs and disassemblies, production controls,
two physical-device captures, compiler/simulator provenance, and SHA256
manifests. `silicon-final` is the primary device set with per-process raw/post
CSVs. The aggregate `SHA256SUMS` file hashes to
`a7e6320fc3f5afc038d24ea610a28368206d75500454876d38a6851a4ee20bdd`;
the independently verified `silicon-final/SHA256SUMS` hashes to
`73db341a90d53f92970526586af86bff830b87d4789703162001d664e2ceb162`.
