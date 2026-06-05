# Hardware-free, parallel precompile collect (mock mode) — reduction A/B

## Result (full `test_reduction_ops.py` nightly, 4958 tests, N300)

| Path | Wall | Speedup vs cold | Notes |
|---|---|---|---|
| Cold inline (real) | 916s | — | inline JIT compile |
| Two-pass, real-device collect | 824s (533 collect + 291 warm) | **1.11×** | collect = 292s serial body-run + 241s compile, single device |
| **Two-pass, MOCK-xdist collect** | **607s (317 collect + 290 warm)** | **1.51×** | collect is **hardware-free + 8-way parallel** |

Correctness: warm run = **2478 passed / 2150 skipped / 330 xfailed**, identical to cold; **99.2% cache hit** (12 of ~1500 kernels recompiled inline). The device is **not used at all** during the 317s collect.

## Why the collect was the bottleneck
The serial collect (run every test body to discover programs, on the one device) doesn't parallelize: btop shows ~1 core busy during it, all 8 only during the final compile burst. The win is to make the collect **hardware-free** so it can run `pytest-xdist -n <cores>` (and, ultimately, on machines with no device).

## The hard part: making the mock build_key match the real one
The collect only pays off if its kernels are looked up under the **same `build_key`** by the real run. `compute_build_key` (build_env_manager.cpp) hashes: `dispatch_core_type`, `dispatch_core_axis`, `num_hw_cqs`, `harvesting_mask` (only if coord-virt off), and `get_compile_hash_string()`. Each mock-vs-real divergence, and its fix:

1. **fast-dispatch CQ-config segfault** — mock crashes in `Device::configure_command_queue_programs` → `DataMovementKernel::configure` (configuring CQ kernels on cores that don't exist). **Fix:** `TT_METAL_SLOW_DISPATCH_MODE=1` skips CQ setup. Free, because the build_key is **dispatch-mode-invariant on real HW** (real fast == real slow == `8475153408042195026`).
2. **harvesting** — generic mock descriptor (`wormhole_N300.yaml`) has the wrong harvesting (grid 8,9 vs real 8,8). **Fix:** capture the **real** cluster descriptor from UMD (`tt_umd.TopologyDiscovery.create_cluster_descriptor().serialize_to_file(...)`) and point mock at it via `TT_METAL_MOCK_CLUSTER_DESC_PATH`. Works for any board (not relying on coord-virt to exclude harvesting).
3. **multi-erisc** — `get_compile_hash_string()` hashes `enable_2_erisc_mode`; mock force-disables it (`rtoptions.cpp`, non-Silicon target). **Fix:** the `TT_METAL_KEEP_2_ERISC_MODE` escape hatch (rtoptions.cpp patch) keeps it on for build purposes.
4. **dispatch_core_type** — force ETH (it's a settable `DispatchCoreConfig`); mock's `get_default_dispatch_core_type()` already returns ETH on this board.

With all four, mock's build_key == real's `8475153408042195026`, verified by the firmware lookup + the 99.2% warm hit.

## The other catch: only explicit CompileProgram compiles under mock
Under mock the op **launch** path does not JIT-compile op kernels (only firmware). The up-front collector's **explicit `detail::CompileProgram`** does (host-side g++, hardware-independent). So the mock collect MUST go through `up_front_collect_plugin` (which calls CompileProgram at session end), not a bare suite run.

## Reproduce
```bash
# 1. mock-xdist collect (hardware-free, warms the cache)
./mock_precompile_collect.sh /tmp/reduction_mock 8 \
    tests/ttnn/nightly/unit_tests/operations/reduction/test_reduction_ops.py
# 2. real warm run over it
TT_METAL_CACHE=/tmp/reduction_mock scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/nightly/unit_tests/operations/reduction/test_reduction_ops.py
```
Requires the `rtoptions.cpp` `TT_METAL_KEEP_2_ERISC_MODE` patch compiled into `libtt_metal.so`
(`ninja -C build_Release`, then sync `build_Release/tt_metal/libtt_metal.so` → `build_Release/lib/`).

## Ceiling
The 317s collect ≈ ~76s parallel body-run + ~241s compile (8-core-bound). The compile is the floor on this 8-core box; more cores or the remote farm would push the collect (and the win) much further. Body-run can be trimmed more with the generic fake-tensor path (lever 2).
