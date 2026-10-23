# perf-optimization-audit — validation & output

## Prove it in the disassembly (assembly diff — the primary evidence)
Most findings are claims about the **emitted instruction stream** ("this removes N SFPSTOREs", "the FMA shadow is empty", "the compiler already lowered `0.0f` to `vConst0`"). Do **not** assert these from the C++ — compile before/after and diff the actual assembly. This is also how you discharge the provenance lens: if the disassembly of the *unchanged* code already shows the win, the finding is a NON-ISSUE (the compiler did it).

Get the real toolchain (the compiler pinned to this build, plus `objdump`) via the repo's setup script — do **not** hand-roll a compiler:
```bash
cd tt_metal/tt-llk/tests
./setup_testing_env.sh          # downloads the pinned sfpi tarball (compiler + objdump + gdb) into tests/sfpi/
cat sfpi/sfpi.version           # the installed version — record it in the report
```
The binaries land under `tt_metal/tt-llk/tests/sfpi/compiler/bin/` (`riscv-tt-elf-g++`, `riscv-tt-elf-objdump`, …). Workflow: compile the kernel (matching the build's arch `-mcpu`/flags and `APPROXIMATION_MODE`/`ITERATIONS` template args) to an object, `riscv-tt-elf-objdump -d` it, then diff the before vs after disassembly for the SFPU op stream — count SFPLOAD/SFPSTORE/SFPMAD/SFPNOP, confirm const-reg lowering, confirm the shadow got filled. The instruction delta (per element × ITERATIONS) **is** the win; report it, and note the exact `sfpi.version` used (it must match the pin from the freshness contract — flag any mismatch).

## Measure — don't assert (cycle-level validation)
For claims about *cycles* (bubbles filled, stalls removed) rather than instruction count, point at the perf-counter harness rather than eyeballing it: `tests/sources/*_perf.cpp` + `START_PERF_MEASURE`, the **`INSTRN_THREAD`** bank (instruction issue + stall-reason counters), the **`FPU`** bank (FPU/SFPU active cycles), and the Python-derived **stall% / backpressure% / utilisation%** metrics (`tests/python_tests/helpers/metrics.py`). See `docs/performance_counters/performance_counters.md`. A "win" that doesn't shrink the instruction count *or* move a counter isn't one. For runtime validation of correctness (numerical equivalence) after a rewrite, use the `run-test` skill (never run pytest directly).

## Output
Head the report with the compiler version (+ the `tests/sfpi/sfpi.version` actually used for disassembly) and ISA-doc revision consulted (and that latency/throughput were derived fresh). For each finding: `file:line`, kernel/function, provenance (sfpi vs raw), the check (section letter + #), current vs proposed sequence, the **semantic-equivalence argument** (+ edge cases checked), the **assembly-diff evidence** (before/after instruction counts from `objdump`, or an explicit note that it's un-disassembled and therefore a SUGGESTION), estimated magnitude, per-arch verdict, and the one-line fix — with the recommended perf-counter metric to confirm any cycle-level claim. End with totals per verdict per arch and any explicit coverage bound. Keep correctness untouched: if a change has any ordering/hazard implication, defer that half to the matching correctness audit and say so.
