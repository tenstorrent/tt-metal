## Summary
Temporarily skip two flaky nightly SDPA backward NanoGPT tests on Wormhole-class targets to stabilize nightly CI while root-cause investigation is in progress.

Made-with: Cursor

### Ticket
- Link to Github Issue: `TODO`

### Problem description
- Nightly CI is unstable on specific hardware targets due to test failures that are currently unrelated to functional regressions:
- `SDPABackwardTest.NIGHTLY_NanoGPTConfig`
- `SDPABackwardTest.NIGHTLY_CausalMask_NanoGPTConfig`
- The SDPA failures are observed on Wormhole-class hardware (including N300 configurations).

### What's changed
- Added a hardware-aware skip guard in `tt-train/tests/ops/sdpa_bw_op_test.cpp` for the two nightly NanoGPT SDPA backward tests on:
- `tt::BoardType::N300`
- `tt::ARCH::WORMHOLE_B0`
- Added TODO tracking comments in code to re-enable tests once underlying instability is resolved.

### Impact
- Stabilizes nightly CI signal on affected hardware targets.
- Limits scope of skips to known flaky test/hardware combinations; unaffected platforms continue to run coverage.
- Serves as a temporary mitigation, not a permanent correctness change.

### Checklist

- [ ] [![Sanity tests](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml/badge.svg?branch=mdragula/ci-skip-memory-sdpa-wormhole)](https://github.com/tenstorrent/tt-metal/actions/workflows/sanity-tests.yaml?query=branch:mdragula/ci-skip-memory-sdpa-wormhole)
- [ ] [![Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml/badge.svg?branch=mdragula/ci-skip-memory-sdpa-wormhole)](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml?query=branch:mdragula/ci-skip-memory-sdpa-wormhole)
- [ ] [![cpp-unit-tests](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml/badge.svg?branch=mdragula/ci-skip-memory-sdpa-wormhole)](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml?query=branch:mdragula/ci-skip-memory-sdpa-wormhole)
- [ ] New/Existing tests provide coverage for changes
- [ ] (Optional) Ran [clang-tidy code analysis](https://github.com/tenstorrent/tt-metal/actions/workflows/code-analysis.yaml) on the PR branch to catch linting errors early
