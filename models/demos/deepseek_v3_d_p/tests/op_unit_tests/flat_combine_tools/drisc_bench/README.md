# DRISC flat-expert weight delivery benchmark (source copy)

Built and run in a tt-metal tree with tests enabled (`cmake -DTT_METAL_BUILD_TESTS=ON .` in the build dir, then
`ninja unit_tests_api`): copy `test_drisc_flat_bench.cpp` to `tests/tt_metal/tt_metal/api/` (add it to that dir's
`sources.cmake`), the two kernels to `tests/tt_metal/tt_metal/test_kernels/misc/`.

    python dump_flat_plan_json.py 6144 2048 8 64 5120 plan.json   # on a device; then convert to the text format
    TT_METAL_SLOW_DISPATCH_MODE=1 DRISC_FLAT_PLAN=flat_plan_glm53.txt ./unit_tests_api --gtest_filter='DriscFlatBench*'

Needs Blackhole firmware >= 19.12.0.0 and slow dispatch (programs with DRAM-core kernels throw under fast dispatch).
DRISC kernels on NOC1 hung (DRISC_FLAT_NOC=1): leave it 0. Results: FLAT_COMBINE_OVERLAP_WORKLOG.md 8f.
