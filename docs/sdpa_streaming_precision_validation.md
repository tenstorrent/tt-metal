# PR 1 foundation and shared-buffer validation

Date: 2026-09-21. This validates the compatibility resolver and the first shared
streaming helper extraction. D/C/B/E device implementations and the public recipe
API remain pending. The frozen research evidence is unchanged.

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
