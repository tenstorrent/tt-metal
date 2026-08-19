# CCL generated ops — first hardware-graded run (4-chip Blackhole QuietBox)

Dashboard rows **942 / 943 / 944** (`runtime_backend=hw`, `arch=blackhole`) on
http://bgdepyc01:8090. Every prior row for these ops was graded on the functional
craq-sim; these are the first graded on silicon.

Hardware: 4x Blackhole p150a, fw 19.5.0.0, FABRIC_1D, mesh `(1,4)` via
`CCL_HW_MESH_SHAPE` (the committed pins want 8 devices; this box has 4).
Runner: `run_hw_golden.sh` — plain pytest over `test_golden.py` + `test_translated.py`.
`eval_test_runner.sh`'s multidevice path always sets up craq-sim, so it cannot be
used here; `golden_results.txt` is produced with that script's exact convention
(`TOTAL = len(results)`) so the rows stay comparable to existing ones.

| op | dashboard | passed | failed | xfail | skipped | scorer | grade | wall |
|----|----------:|-------:|-------:|------:|--------:|-------:|-------|-----:|
| point_to_point | 942 | 383 | **24** | 0 | 36 | 383/443 | A (93.2) | 234s |
| all_gather     | 943 |  36 |   0 | 295 | 64 |  36/395 | D (52.0) | 150s |
| all_reduce     | 944 |  11 |   0 |   1 |  0 |   11/12 | A (93.3) |  15s |

## Read these two numbers carefully

**all_gather's "D" is a scoring-convention artifact, not a quality signal.** It had
**zero failures**. 295 of its cells are out-of-SUPPORTED cells that xfail *by design*
(the op's `validate()` correctly raises `UnsupportedAxisValue`) — that is the op
behaving correctly. The canonical `TOTAL = len(results)` counts those against the
pass rate. The prior sim runs used bounded `-k` filters and so ran far fewer xfail
cells; that makes the raw percentages non-comparable in all_gather's favour. Judge
it on `failed=0`.

**`execution_time` here is grading wall-clock, not a generation pipeline.** The
scorer's 25%-weighted `execution_time` term measures *pipeline* duration
(planner/implementer/verifier). These runs re-grade already-committed ops, so no
pipeline ran and the duration is just how long the golden suites took on silicon
(15-234s). It is NOT comparable to the sim rows' 4-7h full-regeneration durations.
The honest cross-row comparison is `golden_passed/golden_total` and
`runtime_backend`, not the grade.

## The one real finding: 24 point_to_point failures, 100% predicted by DRAM alignment

All 24 failures are `layout=ROW_MAJOR` + `alignment=non_tile_aligned`. Computing
`row_bytes = last_dim * sizeof(dtype)` over all 160 ROW_MAJOR cells:

    row_bytes % 64 != 0  ->  24 cells, 24 failed, 0 passed
    row_bytes % 64 == 0  -> 136 cells,  0 failed, 136 passed
    mismatches: 0

`row_bytes % 64 != 0` predicts failure with **100% accuracy**. 64 is the Blackhole
DRAM alignment (Wormhole's is 32). Example: `1x1x32x48` fails for BFLOAT16/UINT16
(96 B, unaligned) but passes for FLOAT32/INT32/UINT32 (192 B, aligned) — same shape,
outcome flips purely on the byte count. Linear and Ring fail equally, so it is not
routing.

This is the **same signature** the shipped C++ `point_to_point` shows on this box
(see `CCL_HW_VALIDATION_BH4.md` on branch `wransom/ccl_hw_4chip_review`): DRAM-only,
`page_size % 64 != 0`, exactly the odd-indexed pages corrupted, and the identical
shape in L1 passes.

**Two independent implementations — the generated Python op and the shipped C++ op —
fail on precisely the same predicate.** That argues the defect is not in either op's
own logic but in a shared layer below both (interleaved DRAM page addressing on
Blackhole when the page is smaller than the 64 B alignment). It is invisible on
Wormhole, where a 32 B row-major page is already aligned, and invisible on craq-sim,
which does not model real DRAM page padding — which is why neither the sim runs nor
the Wormhole/T3K CI ever caught it.
