<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# JIT-build compile-throughput CI benchmark

Part of [#46305](https://github.com/tenstorrent/tt-metal/issues/46305) (extend the
runtime microbenchmark suite) — this covers the **"jit building"** item, alongside
the op-to-op latency benchmark ([#49771](https://github.com/tenstorrent/tt-metal/pull/49771)).

## What it measures

Host-side **local JIT compile throughput**: how fast metal compiles kernels on the
host (fork the RISC-V toolchain, produce ELFs). Kernels are compiled, **not**
dispatched — the device is only used for build config (arch/grid).

It adapts the existing `DISABLED_TensixCompileStress` gtest in
[`test_compile_stress.cpp`](./test_compile_stress.cpp), which creates N compute
kernels with unique `{id, seed}` compile-time args so each one **bypasses the JIT
cache**, spreads them across grid-sized programs, and compiles all programs in
parallel, reporting `total_elapsed_ms`.

The gated metric is **`compile_ms_min`** (wall-clock to cold-compile N kernels;
lower is better). `kernels_per_sec_max` is recorded alongside it.

> **Real device, not mock.** The test's original mock path (`TT_METAL_COMPILE_STRESS_MOCK=1`,
> the default) is for device-less compile hosts / the remote-server harness; on a
> device-equipped host its real→mock transition throws because
> `MeshDispatchFixture::SetUpTestSuite` already opened the device. The runtime-perf
> SKUs have hardware, so the driver sets `TT_METAL_COMPILE_STRESS_MOCK=0` to compile
> against the real attached device. Compilation is host-side either way.
>
> This is the **local** compile path (`TT_METAL_JIT_SERVER_ENABLE=0`). The remote
> compile-server stress mode (`run_compile_stress_harness.py`, multi-client against
> `TT_METAL_JIT_SERVER_ENDPOINTS`) is a separate, non-gated use-case and is not run
> by CI here.

## How CI runs it

The `jit_build` suite in [`tests/perf/suites.yaml`](../../../perf/suites.yaml) runs it
through the generic runtime perf framework ([`tests/perf`](../../../perf/README.md)) as
`runtime_perf_jit_build` on `wh_n300_civ2` and `bh_p150_perf`. Each of the 3
repetitions launches the gtest in a fresh process with:

- a per-rep seed → every rep is a genuine **cold** compile,
- an isolated `TT_METAL_CACHE` in a scratch dir → no disk-cache carryover between reps,
- a fresh process → fresh in-memory `JitBuildCache`,
- `TT_METAL_JIT_SERVER_ENABLE=0` and `CCACHE_DISABLE=1` → local, uncached compile path only,
- `TT_METAL_COMPILE_STRESS_MOCK=0` → real attached device, arch pinned to `$ARCH_NAME`.

The gtest writes `compile_ms` for case `compile/num_kernels:300` to `$TT_PERF_OUTPUT`;
the framework takes the **fastest** rep (min wall-clock rejects upward CPU-contention
noise on shared runners) and compares it to
[`goldens/jit_build.json`](./goldens/jit_build.json). The job fails if compile time is
more than 15% slower or 15% faster than the golden; a faster run means the golden
should be updated with `python -m tests.perf update --from-run <run-id>`.

Run it locally against a build:

```bash
python -m pytest --noconftest -p tests.perf.plugin "tests/perf/test_suites.py::test_perf[jit_build]" \
    --perf-environment=bh_p150_perf
```

The goldens started from the worst cold-compile min observed in CI per SKU
(50700 ms on Wormhole, 44800 ms on Blackhole).
