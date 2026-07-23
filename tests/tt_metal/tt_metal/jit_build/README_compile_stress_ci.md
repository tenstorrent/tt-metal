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

`compile_stress_ci.py` is the driver + regression gate, wired into the
**(Runtime) Performance Tests** pipeline as `runtime_perf_jit_build`
(`tests/pipeline_reorg/runtime_perf_tests.yaml`), on `wh_n300_civ2` and
`bh_p150_perf`. CI config: `--num-kernels 300 --repetitions 3` (≈2–4 min/SKU;
throughput is CPU-bound at roughly a few hundred kernels/min).

For each of `--repetitions` runs it launches the gtest in a fresh process with:

- a unique per-rep seed (`BASE_SEED + rep`) → every rep is a genuine **cold** compile,
- an isolated `TT_METAL_CACHE` dir → no disk-cache carryover between reps,
- a fresh process → fresh in-memory `JitBuildCache`,
- `TT_METAL_JIT_SERVER_ENABLE=0` → local compile path only,
- `TT_METAL_COMPILE_STRESS_MOCK=0` → real attached device, arch pinned to `$ARCH_NAME`.

It then takes the **fastest** rep (min wall-clock rejects upward CPU-contention
noise on shared runners) and compares it to a per-arch golden.

Run it locally against a build:

```bash
./tests/tt_metal/tt_metal/jit_build/compile_stress_ci.py \
    --arch blackhole --num-kernels 300 --repetitions 3
```

## Goldens and gating

- [`compile_stress_golden.json`](./compile_stress_golden.json) — Wormhole (`wh_n300_civ2`)
- [`compile_stress_blackhole_golden.json`](./compile_stress_blackhole_golden.json) — Blackhole (`bh_p150_perf`)

A golden metric set to `null` stays in **record mode**: the value is printed but the
job is **not** gated on it. A non-null metric is **gated** — the job fails if the
measured value regresses beyond `tolerance_pct` (default 15%).

Both goldens are **armed on `compile_ms_min`** (15% tolerance), set to the worst
`compile_ms min` observed across two CI runs per SKU:

| SKU | golden `compile_ms_min` | fails above (~min × 1.15) |
| --- | --- | --- |
| `wh_n300_civ2` (wormhole_b0) | 50700 ms | ~58.3 s |
| `bh_p150_perf` (blackhole)   | 19500 ms | ~22.4 s |

`kernels_per_sec_max` stays `null` on both — the time gate is sufficient and
throughput is just its inverse.

### Re-tuning the gate

1. Read `compile_ms  min` off the passing CI runs for the SKU.
2. Update `metrics.compile_ms_min` in the matching golden (use the worst stable
   min so normal variance doesn't flake the gate).
3. Adjust `tolerance_pct` if 15% is too tight/loose for observed CI variance.
