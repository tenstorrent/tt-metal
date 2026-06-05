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
3. **multi-erisc** — `get_compile_hash_string()` hashes `enable_2_erisc_mode`; mock force-disables it (`rtoptions.cpp:350`, non-Silicon target). The hashed value is **resolved per-arch/firmware**, so it can't be hardcoded or re-derived under mock:
   - It flows: rtoptions default `true` → maybe disabled → `verify_fw_capabilities()` (metal_env.cpp:154) overwrites it from the firmware-capability resolver (`firmware_capability.cpp`).
   - The resolver is **arch-conditional**: `WORMHOLE_B0: break` (pass-through, default `true` flows through); `BLACKHOLE` downgrades to **false** if eth-fw < `1.7.0`. Plus `TT_METAL_DISABLE_MULTI_AERISC=1` (`rtoptions.cpp:710`) forces false on any arch. Mock can't re-derive the Blackhole downgrade (no eth-fw query).
   - **Fix (device-derived, general):** CAPTURE the *resolved* value off a real device — `ttnn.cluster.get_enable_2_erisc_mode()` (new binding → `tt_metal::GetEnable2EriscMode` → rtoptions, read after `verify_fw_capabilities`) — and REPLAY it under mock via **`TT_METAL_FORCE_2_ERISC_MODE=0|1`** (rtoptions.cpp escape hatch). This is correct on any arch, including a downgraded Blackhole, exactly like harvesting/dispatch. Verified: real N300 getter returns `1`; mock `FORCE=1` → key `8475153408042195026` (== real), `FORCE=0` → `9000709522989198091` (single-erisc). (The earlier `TT_METAL_KEEP_2_ERISC_MODE` keep-on hatch is retained for back-compat but is Wormhole-only — it can only produce the `true` case.)
4. **dispatch_core_type** — **device-derived, not forced.** `get_default_dispatch_core_type()` (device.py) returns ETH iff `get_cluster_type()` ∈ {N300, T3K, N300_2x2}; under mock the cluster type comes from the captured real descriptor → ETH on this N300. So the runner forces nothing — dispatch falls out of the descriptor (same as harvesting). (My early isolation probe used `FORCE_DISPATCH=ETH` to vary one knob; the real collect does not.) On a board whose default is WORKER, the same descriptor-driven default would correctly pick WORKER — general.

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
The runner captures the real cluster descriptor + resolved `enable_2_erisc_mode` once (brief real-HW
opens, cached in `/tmp`), then replays them under mock. Requires the `rtoptions.cpp`
`TT_METAL_FORCE_2_ERISC_MODE` hatch + the `ttnn.cluster.get_enable_2_erisc_mode()` binding compiled
in — build with `./build_metal.sh` (it installs the `.so`s to their import locations).

## Ceiling
The 317s collect ≈ ~76s parallel body-run + ~241s compile (8-core-bound). The compile is the floor on this 8-core box; more cores or the remote farm would push the collect (and the win) much further. Body-run can be trimmed more with the generic fake-tensor path (lever 2).
