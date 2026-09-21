# PR 1 foundation and streaming-primitive validation

Date: 2026-09-21. This validates the compatibility resolver and the first shared
streaming helper extraction. D/C/B/E device implementations and the public recipe
API remain pending. The frozen research evidence is unchanged.

## FP32 recurrent-state extraction

The next staged change adds `streaming/fp32_state.hpp` and
`streaming/fp32_state_sfpu.hpp`. These isolate the C/D state rescale, L1 add,
reciprocal, final normalization, and single-tile pack configuration. They are
not yet called by the production attention entrypoint. No recipe dispatch,
data-movement kernel, block size, or input-buffer depth is changed.

The arithmetic comes from the frozen snapshot's
`experiments/sdpa-l2/compute-sprint-v3/fp32/early_guard.hpp` and its
`hybrid-mixed-v1/candidate/.../ckernel_sfpu_sdpa.h` dependency. Important retained
contracts are full FP32 unpack-to-destination, multiplication rounded before
the L1 addition, batched-unpack zero-flag clearing, an identity path with no
correction-buffer access, and the two-iteration reciprocal. The extraction
replaces the research runtime identity branch with compile-time specializations
and makes pack-cache ownership explicit. The surrounding fused C/D loop and
its specialized exponential/subtraction schedules remain to be integrated.

`test_sdpa_fp32_state.py` uses the existing generic-op descriptor interface and
stock unary reader/writer. Its 11 cases pass on Blackhole P100, both normally
and with Watcher/device assertions. Each invocation runs 12 records (forcing
CB wraparound) and two actual trace replays. Rescale tests additionally retain
the first buffers while launching changed data at fresh addresses and verify
that the program-cache count does not grow.

The final rebuilt-tree check passed all 22 device tests (11 state plus 11
legacy compatibility) under Watcher, followed by all 13 policy/resolver host
tests. Incremental build/install and clang-format/Black checks also passed.

| Primitive check | Observed error against independent host reference |
| --- | --- |
| Rescale then add, 1/2/3 tiles, and first-column denominator | Exact FP32 match |
| Identity add, 2/4 tiles, and first-column denominator | Exact FP32 match |
| Normalization to FP32, widths 1/4 | Relative L2 5.53–5.60e-8; max absolute error 5.96e-8 |
| Normalization to BF16, widths 1/4 | Relative L2 0.1633–0.1646% |

Inputs include cancellation, tiny updates, large finite common modes, and
identity/nonidentity corrections. Normalization uses positive finite sums;
zero/nonfinite denominator behavior is preserved in code but is not qualified
by these tests. FP32 output is a diagnostic of the primitive, not a new public
recipe/output contract. These are component tests against host arithmetic,
not a differential run of the complete frozen attention kernel, and do not
establish end-to-end accuracy or performance parity.

Reproduce with the environment below and:

```bash
bash scripts/run_safe_pytest.sh --dev \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_fp32_state.py
```

Raw logs and XML are `sdpa-pr1-fp32-state{,-watcher}.{log,xml}` in the same
external artifact locations listed below. Device JIT caches are separate
`sdpa-pr1-fp32-state-jit` and `sdpa-pr1-fp32-state-watcher` directories under
`/localdev/cglagovich`.

## Environment and recovery

- Main base: `dfaf6dc802f0a1321bbb2578ba7c4a0fb9b71ab8`.
- Foundation commit: `27f81aaf7fa`; tests and helper extraction follow it.
- IRD reservation: `227464`, `yyzo-bh-04`, one Blackhole P100.
- Container: `yyzo-bh-04-special-cglagovich-for-reservation-227464`.
- Remote checkout: `/localdev/cglagovich/tt-metal-sdpa-pr1`.
- Clang 20, system SFPI 7.80.0, Python 3.10.19, Torch 2.11.0+cpu.
- Exact pinned submodules were fetched, not copied from the old research build.

The previous reservation expired. The fresh container could fetch GitHub
dependencies normally. The network home filesystem was full, so both ccache
and the separate firmware/device JIT cache were redirected to `/localdev`.
The initial build failures were cache writes, not C++ diagnostics. No existing
user files were deleted. Missing Python runtime dependencies were installed
from the existing `pyproject.toml`; no project dependency was added.

## Results

| Check | Result |
| --- | --- |
| Full build, Python bindings, TTNN test targets | Passed |
| Registered policy/resolver GoogleTests against new libraries | 13 passed |
| Legacy compatibility suite before helper extraction | 11 passed |
| Same suite after extraction, independent JIT cache | 11 passed |
| Before/after device output SHA256 comparison | All 10 identical |
| Same compatibility suite with Watcher/device assertions | 11 passed |
| Existing `test_sdpa_prefill.py` | 8 passed, 2 existing skips |

The compatibility suite covers:

- Omitted config versus explicit HiFi2, and explicit LoFi with omitted versus
  explicit exponential approximation.
- All eight combinations of FP32 destination, math approximation, and exp
  approximation booleans at HiFi2.
- A cache hit on a second set of live input addresses, two actual trace
  replays per combination, and unchanged inputs.
- Python's empty config constructor contract without launching invalid fidelity.

The existing prefill suite adds GQA, a dense bias with a non-tile-aligned
sequence, causal/noncausal sliding windows, packed BF8 inputs, and attention
sinks. Its two skips are already present for a profiling/OOM-sensitive geometry;
no new skip was introduced.

The paired baseline is the **unrefactored kernel in the PR1 foundation**, not
a separate build of pristine main. The foundation's legacy resolver is separately
tested against the existing field/default contract. No new-recipe numerical
or throughput claim follows from these tests.

## Compatibility details found during bring-up

1. C++ `ComputeKernelConfig{}` defaults to LoFi. Python
   `WormholeComputeKernelConfig()` explicitly supplies `MathFidelity.Invalid`.
   Python does not export the canonical C++ name `ComputeKernelConfig`.
   Tests now distinguish these contracts; production defaults were not changed.
2. On this base, legacy non-streaming FP32 at Q256/K512 needed 1,606,656 bytes
   through the end of its static CB region, exceeding 1,572,864-byte L1.
   Compatibility tests use Q128/K512 for FP32 and Q256/K512 for BF16. This is
   not a matched-geometry performance comparison or a frozen-recipe change.
3. The numerical smoke uses original BF16 normal Q/K/V, shape
   `[1, 2, 1024, 128]`, noncausal attention and an independent FP64 reference.
   Across its eight legacy configurations, relative L2 was 2.4698–2.7333% and
   PCC was 0.999671–0.999745. These are legacy results, unchanged by extraction;
   the broad smoke thresholds are not the new recipes' acceptance criteria.

## Reproduction

From the reserved container's production checkout:

```bash
export CCACHE_DIR=/localdev/cglagovich/sdpa-pr1-ccache
export CCACHE_MAXSIZE=20G
export TT_METAL_CACHE=/localdev/cglagovich/sdpa-pr1-jit-cache
export CMAKE_BUILD_PARALLEL_LEVEL=12
./build_metal.sh --enable-ccache --build-dir build_pr1 --build-ttnn-tests --use-system-sfpi

export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD:$PWD/ttnn:$PWD/tools"
export LD_LIBRARY_PATH="$PWD/build_pr1/lib"
export OMP_NUM_THREADS=8
flock -x /tmp/tt-device.lock build_pr1/test/ttnn/unit_tests_ttnn \
  '--gtest_filter=SDPAPrecisionPolicy.*:SDPANumerics.*'
bash scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_numerics_compatibility.py
bash scripts/run_safe_pytest.sh --dev \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_numerics_compatibility.py
bash scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py
```

The paired runs used `sdpa-pr1-jit-before` and `sdpa-pr1-jit-after` under
`/localdev/cglagovich`; Watcher used `sdpa-pr1-jit-watcher`. No old research
library or kernel-source override was used. All device runs used the repository's
cooperative locking runner.

Raw XML reports and logs are retained outside the production tree at
`/Users/cglagovich/dev/sdpa-pr1-validation-20260921/`, with remote originals
under `/localdev/cglagovich/sdpa-pr1-*.{xml,log}`. XML properties contain the
per-case relative L2, PCC, and output hashes. These artifacts do not modify
the immutable research snapshot.
