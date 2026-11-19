# V6 dynamic corrected-FMA device probe preparation

The new [probe](../../../tests/v6_fma_sfpu_corrected_probe.py) implements the full short-expansion midpoint correction using a real runtime FP32 input tensor. It is a separate source from the already-measured primitive probe and leaves that source and the earlier CPU proof unchanged. Current frozen preparation artifacts are `v6_fma_sfpu_corrected_source_only_v2.json`, `.compute.cpp.snapshot`, and `.fixtures.pt`. **This preparation did not run hardware or compile the device source.**

CPU fixture generation and a scalar transcription of the exact generated graph pass bitwise on 8,192 deterministic bounded random lanes, 616 constructed/actual midpoint and cancellation lanes, and 306 signed-zero lanes. There are another 102 separately tagged subnormal-control lanes; their CPU transcription passes but device subnormal parity is explicitly outside the bounded claim. Golden values come from scalar `libm.fmaf`, not an approximation or a changed decoder reference.

The graph uses general TwoSum and fixed expansion slots, retaining zero components instead of performing data-dependent compaction. It compares the exact residual with the signed half-spacing, uses output significand parity at a tie, and explicitly constructs negative zero when both the product zero and accumulator zero have negative sign. All arithmetic intermediates go through FP32 DST scratch boundaries, as in the previous measured primitive. Inputs remain in DST tiles 0, 1, and 2; scratch uses slots 96 through 185 plus the current lane offset (at most 216); corrected output occupies tile 7 (slots 224 through 255). Thus scratch cannot overwrite operands or output as the loop visits all 32 vector groups.

The single inline `SOURCE_CODE` compute kernel uses the existing sharded-CB generic-op pattern from `tests/ttnn/unit_tests/kernel_lib/reduce/test_sfpu_reduce_integer_column_avg.py`. Its input CB is configured with `UnpackToDestFp32`; `copy_tile` loads all three operands directly into destination tiles. The output contains the corrected tile and all three untouched input tiles. The report checks all 1,024 lanes in every tile and bit-compares each input round trip, including signed zero. C++ compilation, vector register allocation, DST iteration, and pack/unpack behavior remain hardware validation obligations.

The default invocation emits CPU evidence and source only. The parent controls serial hardware use:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fma_sfpu_corrected_probe --run-device --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_fma_sfpu_corrected_device.json
```

The output path must be fresh. A run records component pass/fail, exact input-roundtrip differences, per-category bit/numerical differences and examples, source/fixture hashes, output tensors, exceptions, and device closure. The subnormal category does not affect the bounded gate; signed-zero and actual midpoint categories do. This component does not qualify a general FP32 FMA, custom GEMM, whole chunk graph, decoder, trace replay, or latency.

## First compilation failure and local source repair

The parent's first device invocation (`v6_fma_sfpu_corrected_device.json/.log`) closed cleanly after the installed SFPI 7.95/GCC 15.1 compiler failed in `rvtt_cc::process_block_stmts`, line145. No kernel dispatch occurred. The exact tested Python source is preserved as `v6_fma_sfpu_corrected_probe.first_device.py.snapshot` (SHA `17e8beca8f1f8c5b80a687f5192d4371e24ff028c9df268469a14055c8664859`).

A standalone translation unit reproduced the identical compiler assertion without hardware. Isolated sign, neighbor, and first selection blocks compiled successfully; the second selection block failed. The failing expression combines six predicates and a nested disjunction to select negative zero. Replacing only this expression with flat predicated integer flags lets the **entire standalone generated arithmetic graph compile successfully**. The arithmetic operations, graph construction, fixtures, scalar oracle, input/output handling, and gates are unchanged. An AST comparison confirms that only `Graph.cpp` changed; a 64-case Boolean truth table confirms predicate equivalence. The emitted standalone assembly preserves separate multiply/add and DST load/store instructions and contains no SFPMAD contraction. Full kernel linkage and silicon behavior still require the parent's next run.

Repair evidence, commands, hashes, AST delta, predicate checks, and assembly counts are in `v6_fma_sfpu_corrected_cc_repair.json`; original and revised standalone C++/compiler logs and successful assembly sit alongside it. Current source-only fixture/compute artifacts are `v6_fma_sfpu_corrected_source_only_v3.*`. Current Python SHA is `185217ed00091eddcdf15db9494071901edaa472d3935b986b4b512834f6832f`. No compiler, shared/native source, or hardware was modified by the repair investigation.
