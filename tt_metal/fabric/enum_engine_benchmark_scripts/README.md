# Enumeration-engine benchmark harnesses

Standalone harnesses used for `../ENUM_ENGINE_BENCHMARK.md` (cold gimsatul vs warm incremental CaDiCaL on the dumped
2×4-128 base CNF). They operate on a plain DIMACS file with **full-model blocking** (¬ the whole assignment) so they
isolate raw SAT-engine speed, independent of the tt-metal pipeline.

| file | approach | build / run |
|---|---|---|
| `warm_enum.cpp` | single warm incremental CaDiCaL | `clang++ -O3 -std=c++17 -I<cadical/src> warm_enum.cpp libcadical.a -lpthread -o warm_enum` → `warm_enum <cnf> <N> [seed]` |
| `mt_warm_enum.cpp` | 16-way clause-sharing warm CaDiCaL (learned + blocking pools; optional phase hint) | same link → `mt_warm_enum <cnf> <N> <workers> <hint:0|1> [seedbase]` |
| `cold_enum.sh` | repeated cold gimsatul (DIMACS round-trip + append blocking clause) | `cold_enum.sh <cnf> <N> <threads>` (edit `$GIM` path) |
| `hybrid_enum.sh` | gimsatul finds #1 cold, then `warm_enum` for the rest | `hybrid_enum.sh <cnf> <N> <threads> [cadical-seed]` |

Dump the CNF: run `generate_rank_bindings` on the 2×4-128 pipeline MGD onto SC36 with `TT_TOPO_SAT_NO_MINHOST=1` and
`TT_TOPO_SAT_DUMP_DIMACS=<out.cnf>` (the `write_dimacs` hook lives on `ridvan/exp-clause-sharing-portfolio`).
gimsatul: build v1.1.3 from https://github.com/arminbiere/gimsatul (`./configure && make`).

Note: absolute paths in the shell scripts point at the experiment scratchpad — adjust `SC=`, `$GIM`, and the harness
paths for your environment.
