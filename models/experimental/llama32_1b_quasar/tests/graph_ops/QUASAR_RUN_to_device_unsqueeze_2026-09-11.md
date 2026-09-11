# Quasar run report — `ttnn.to_device` + `ttnn.unsqueeze_to_4D` (craq-sim + ZEBU RTL), 2026-09-11

**Branch:** `vsureshTT/quasar_uplift_round_2` (working tree; no source edits made in this session). **Host libs:** built from this tree — `ttnn/ttnn/_ttnn.so` 18:08 (craq-sim runs) / 18:56 (emulator jobs 1–2, another agent's install landed between the two phases; no op in this report has host-side changes, so the difference is immaterial here).
**Tests:** `tests/graph_ops/test_to_device.py` (4 captured cases), `tests/prototype_ops/test_to_device.py` (10 cases), `tests/graph_ops/test_unsqueeze_to_4D.py` (4 captured cases), `tests/prototype_ops/test_unsqueeze_to_4D.py` (2 cases).
**Review artifact — uncommitted, delete before merge.** Temp test copies (`test_tmp_unsqueeze_to_4D_qsr.py`, one per directory) were deleted after the run; their content is reproduced in §4.

## 0. Summary

| Op | craq-sim | ZEBU RTL (1x3, 1 worker) | Device work | Blocker |
|---|---|---|---|---|
| `ttnn.to_device` (graph_ops, 4 cases) | **PASS 4/4** as checked in | **PASS** (case `01_1_i32_int-dram`) | **none** — plain host→device DRAM write, no program, no kernel JIT-built | — |
| `ttnn.experimental.quasar.to_device` (prototype_ops, 10 cases) | **PASS 10/10** as checked in | **PASS** (`uint32-rot_idxs-batch1`) | **none** — same `Tensor::to_device` call as core (§3) | — |
| `ttnn.unsqueeze_to_4D` (graph_ops, 4 cases) | as checked in: **1 PASS / 3 FAIL (harness B1)** · temp copy: **PASS 4/4**, `out.buffer_address() == in.buffer_address()` on all 4 | temp copy case `00_32x64_bf16_int-dram`: **PASS** (2nd launch; 1st = HANG, ZEBU unit held by another user, §2) | **none** — `ttnn::reshape` → `ttnn::experimental::view`, buffer aliased | B1 (harness tilize, not the op) |
| `ttnn.unsqueeze_to_4D` (prototype_ops, 2 cases) | as checked in: **FAIL 2/2 (B1)** · temp copy: **PASS 2/2**, addresses equal | temp copy `rope_3d`: **PASS** | none | B1 |

Neither op is RED. The only failures are the known harness-level `from_torch(layout=TILE, device=)` legacy tilize (B1, §5), which fires before the op under test is reached.

## 1. craq-sim (Quasar functional simulator)

Environment: `source /localdev/vsuresh/qsr-sim/env.sh light_ops` → `TT_METAL_SIMULATOR=/localdev/vsuresh/qsr-sim/libttsim.so`, `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_FORCE_JIT_COMPILE=1`, fresh private kernel cache `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-light_ops` (created empty at the start of this session). Sim grid 8x4 workers, 2 DRAM banks (no case here is sharded, so no grid skip occurred). Every run went through `qsr_test timeout 1800 …` (shared build lock). Logs: `scratchpad/qsr_light_ops/{A..F}_*.log`.

**Kernel-cache evidence of "no device kernel":** with `TT_METAL_FORCE_JIT_COMPILE=1` any launched program JIT-builds its kernels into `$TT_METAL_CACHE/<hash>/<kernel>`. After all six runs the cache contains **only** `firmware/{dm0,dispatch_dm0,trisc0..3}` — no kernel directory was ever created. So no case in any of the four files (including the 20 PASSing to_device cases and the 6 PASSing unsqueeze cases) enqueued a program.

### 1a. `graph_ops/test_to_device.py` — as checked in (run A, 4 passed in 6.39 s)

| Case | Input (host) | Result | On device |
|---|---|---|---|
| `00_1x128_i32_int-dram` | `[1,128]` INT32 ROW_MAJOR host → DRAM interleaved | **PASS** | host→device write only (no kernel) |
| `01_1_i32_int-dram` | `[1]` INT32 ROW_MAJOR | **PASS** | write only |
| `02_1x32_u32_int-dram` | `[1,1,1,32]` UINT32 ROW_MAJOR | **PASS** | write only |
| `03_1x32_u32_int-dram` | `[1,32]` UINT32 ROW_MAJOR | **PASS** | write only |

Harness checks that passed: output shape / dtype / layout / memory config vs the capture, finiteness, and the `_ref_identity` golden (read-back equals the uploaded data). No implicit tilize / typecast / pad was triggered: the input is ROW_MAJOR and INT32/UINT32 is a native Quasar format for a plain buffer write (no `ValidateProgramSpec` is involved since no `ProgramSpec` is built).

### 1b. `prototype_ops/test_to_device.py` — as checked in (run B, 10 passed in 8.50 s)

Entry point exercised: **`ttnn.experimental.quasar.to_device`** (the test calls it as written). It is a thin wrapper — `ttnn/cpp/ttnn/operations/experimental/quasar/to_device/to_device.cpp` does `return tensor.to_device(mesh_device, memory_config.value_or(DRAM_MEMORY_CONFIG), queue_id);`, i.e. exactly the same `Tensor::to_device` call that core `ttnn.to_device` (`ttnn-nanobind/operations/core.cpp:137` → `operations::core::to_device`) makes. There is no Quasar-specific device code behind it.

| Case | Input (host) | Result |
|---|---|---|
| `uint32-rot_idxs-batch1` | `[1,32]` UINT32 ROW_MAJOR | **PASS** |
| `uint32-rot_idxs-batch32` | `[1,32]` UINT32 ROW_MAJOR | **PASS** |
| `uint32-activation-seq128` | `[1,1,128,2048]` UINT32 ROW_MAJOR | **PASS** |
| `uint32-activation-seq512` | `[1,1,512,2048]` UINT32 ROW_MAJOR | **PASS** |
| `uint32-activation-seq1024` | `[1,1,1024,2048]` UINT32 ROW_MAJOR | **PASS** |
| `bf16-rot_idxs-batch1` | `[1,32]` bf16 **TILE** (tilized on host — `from_torch` without `device=`) | **PASS** |
| `bf16-rot_idxs-batch32` | `[1,32]` bf16 TILE | **PASS** |
| `bf16-activation-seq128` | `[1,1,128,2048]` bf16 TILE | **PASS** |
| `bf16-activation-seq512` | `[1,1,512,2048]` bf16 TILE | **PASS** |
| `bf16-activation-seq1024` | `[1,1,1024,2048]` bf16 TILE (4 MiB) | **PASS** |

All 10 are pure writes (cache evidence above). The bf16 TILE rows do **not** hit B1 because the test builds the tensor on host (no `device=`), so the tilize happens in `Tensor::from_span` on the host.

### 1c. `graph_ops/test_unsqueeze_to_4D.py`

**As checked in (run C, 3 failed / 1 passed in 6.64 s):**

| Case | Input | Result | Error |
|---|---|---|---|
| `00_32x64_bf16_int-dram` | `[1,32,64]` bf16 TILE DRAM | **FAIL (harness B1)** | `TT_FATAL @ tt_metal/impl/kernels/kernel.hpp:450: … DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.` — backtrace: `convert_python_tensor_to_tt_tensor → ttnn::to_layout → ttnn::tilize → prim::tilize (TilizeDeviceOperation) → Program(ProgramDescriptor) → CreateDataMovementKernel`. Raised inside `graph_case.build_tensor`'s `ttnn.from_torch(..., layout=TILE, device=mesh)`, **before `ttnn.unsqueeze_to_4D` is called**. |
| `01_32x2048_bf16_int-dram` | `[1,32,2048]` bf16 TILE | **FAIL (B1)** | same |
| `02_32_u32_int-dram` | `[32]` UINT32 ROW_MAJOR | **PASS** | — (ROW_MAJOR input, no tilize; op returned the `[1,1,1,32]` view) |
| `03_1024x2048_bf16_int-dram` | `[1,1024,2048]` bf16 TILE | **FAIL (B1)** | same |

**Temp copy `graph_ops/test_tmp_unsqueeze_to_4D_qsr.py` (run D, 4 passed in 6.21 s)** — identical CASES, `G.build_tensor` monkeypatched to host-tilize + `ttnn.to_device` (§4), op wrapped to assert `out.buffer_address() == in.buffer_address()`:

| Case | in → out shape | in_addr | out_addr | Result |
|---|---|---|---|---|
| `00_32x64_bf16_int-dram` | `(1,32,64)` → `(1,1,32,64)` | 24088704 | 24088704 | **PASS** (view) |
| `01_32x2048_bf16_int-dram` | `(1,32,2048)` → `(1,1,32,2048)` | 24088704 | 24088704 | **PASS** (view) |
| `02_32_u32_int-dram` | `(32,)` → `(1,1,1,32)` | 24088704 | 24088704 | **PASS** (view) |
| `03_1024x2048_bf16_int-dram` | `(1,1024,2048)` → `(1,1,1024,2048)` | 24088704 | 24088704 | **PASS** (view) |

Harness golden `_ref_view` (PCC ≥ 0.999 vs the leading elements) plus shape/dtype/layout/mem-config checks all passed.

### 1d. `prototype_ops/test_unsqueeze_to_4D.py`

**As checked in (run E, 2 failed in 7.02 s):** both `rope_3d` (`(1,32,64)`) and `rank2` (`(32,2048)`) **FAIL (B1)** with the same `kernel.hpp:450 DataMovementKernel is not supported on Quasar` TT_FATAL, raised in `U.to_tt` → `ttnn.from_torch(..., layout=TILE, device=mesh)` before the op.

**Temp copy `prototype_ops/test_tmp_unsqueeze_to_4D_qsr.py` (run F, 2 passed in 5.36 s)** — `U.to_tt` replaced by host tilize + `ttnn.to_device`, plus the buffer-address assert:

| Case | in → out | in_addr / out_addr | Result |
|---|---|---|---|
| `rope_3d` | `(1,32,64)` → `(1,1,32,64)` | 24088704 / 24088704 | **PASS** (view, PCC ≥ 0.999) |
| `rank2` | `(32,2048)` → `(1,1,32,2048)` | 24088704 / 24088704 | **PASS** (view) |

## 2. ZEBU RTL emulator (`emu-quasar-1x3`, 1 worker core)

Environment: `source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3` (`TT_METAL_SIMULATOR=/localdev/vsuresh/tt-umd-simulators/build/emu-quasar-1x3`, `NNG_SOCKET_ADDR=tcp://aus-wh-03:$P_USER_DBD_PORT`, `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-emu-1x3`, slow dispatch). One case per test file, one device session per job, each via `qsr_emu timeout 2700 …` (exclusive emu lock). Launcher logs moved to `scratchpad/qsr_light_ops/emu*_launcher_*.log`; pytest logs `EMU*_*.log`.

| Job | Test / case | Result | Wall | Notes |
|---|---|---|---|---|
| 1 | `graph_ops/test_to_device.py -k 01_1_i32` (`[1]` INT32 ROW_MAJOR → DRAM) | **PASS** | 66.7 s incl. boot | pure write; no kernel |
| 2 | `prototype_ops/test_to_device.py -k uint32-rot_idxs-batch1` (`[1,32]` UINT32 via `ttnn.experimental.quasar.to_device`) | **PASS** | 51.4 s incl. boot | pure write; no kernel |
| 3 (attempt 1, 19:09) | `graph_ops/test_tmp_unsqueeze_to_4D_qsr.py -k 00_32x64` (temp copy, `(1,32,64)` bf16 TILE, host-tilized) | **HANG → 2700 s timeout** (infra, not the op) | 45 min | ZEBU launch refused: launcher log `zServer ERROR WRP0625E: Hostname = soc-zebu-03 is used by gchoudhary … U8.M2.S0..U8.M3.S3 is used by gchoudhary (SessionId:568a6036) … Relaunch your command to try again later`, then `zpost.split ERROR Test log file test-start.log does not exist`; the tt-metal client stayed at `Waiting for ack msg from remote...` (sim child `quasar-1x3_run_` defunct) until `timeout 2700` killed it. Not killed by hand (protocol). `ssh soc-l-04 ps … zrun|vovsh` showed **no** dangling process of mine, so the stale-process cleanup step was not applicable — the unit was simply occupied. Launcher log: `emu3_attempt1_launcher_FAILED_unit_in_use.log`. |
| 3 (retry, 19:59) | same | **PASS** | 40.7 s test (+ ~4 min lock wait/boot) | `[qsr] unsqueeze_to_4D (1, 32, 64) -> (1, 1, 32, 64): in_addr=24088704 out_addr=24088704 same=True`; PCC golden passed; zero device work |
| 4 (20:00) | `prototype_ops/test_tmp_unsqueeze_to_4D_qsr.py -k rope_3d` (temp copy, `(1,32,64)` bf16 TILE) | **PASS** | 38.8 s test (+ ~6 min lock wait/boot) | `[qsr] unsqueeze_to_4D (1, 32, 64) -> (1, 1, 32, 64): in_addr=24088704 out_addr=24088704 same=True`; PCC ≥ 0.999; zero device work |

RTL total: 4 test files × 1 case = **4 PASS** in 5 ZEBU launches (one refused by unit contention). Jobs 1–2 ran on `ttnn/ttnn/_ttnn.so` 18:56, jobs 3-retry / 4 on `_ttnn.so` 19:55 (other agents' host installs between jobs; irrelevant to these kernel-free ops — no host-side change touches `Tensor::to_device` or `reshape_view`'s view path).

## 3. What each op actually did on device

- **`ttnn.to_device` / `ttnn.experimental.quasar.to_device`:** both resolve to `Tensor::to_device(mesh_device, memory_config, queue_id)` — a DRAM buffer allocation plus a host→device write over the dispatch/slow-dispatch path. No `ProgramSpec`/`Program`, no kernel; no implicit tilize / typecast / pad in any of the 14 captured+prototype shapes (INT32 / UINT32 ROW_MAJOR; bf16 TILE already tilized on host). Confirmed on both platforms by PASS + the empty kernel cache (§1). The Quasar prototype entry point adds nothing over core (it exists only because a plain `mod.def("to_device")` collided with core's registration at import — comment in `to_device_nanobind.cpp:37`).
- **`ttnn.unsqueeze_to_4D`:** `ttnn/cpp/ttnn/operations/core/core.cpp:20-30` → `ttnn::reshape(tensor, logical.to_rank(4), padded.to_rank(4))` → `reshape_view/reshape.cpp` view predicates → `ttnn::experimental::view` → `tt::tt_metal::view_device` aliasing the same `MeshBuffer` under a new `TensorSpec`. Zero device work; proven by `out.buffer_address() == in.buffer_address()` (24088704 on every case, both files) and by the empty kernel cache. This matches the reshape_view GREEN report (`ttnn/cpp/ttnn/operations/data_movement/reshape_view/QUASAR_UPLIFT_REPORT.md` §1). Neither `ReshapeViewRMProgramFactory` nor the tiled factory was reached.

## 4. Harness workarounds used (temp copies only; repo tests untouched; copies deleted after the run)

Both unsqueeze test files build their bf16 TILE inputs with `ttnn.from_torch(..., layout=TILE_LAYOUT, device=mesh, mesh_mapper=…)`, which on Quasar runs the legacy on-device `ttnn::tilize` (B1). The temp copies, placed next to the originals and named `test_tmp_unsqueeze_to_4D_qsr.py`, changed only the input materialization and added a view assert:

```python
# graph_ops copy: monkeypatch G.build_tensor (same idiom as the earlier test_tmp_reshape/untilize_qsr.py copies)
def _build_tensor_host_tilize(spec, mesh_device, case, op_name, key):
    data = G._torch_data(spec, case, op_name, key)
    memory_config = G.build_memory_config(spec.get("mem"), mesh_device) or ttnn.DRAM_MEMORY_CONFIG
    host = ttnn.from_torch(data, dtype=G.DTYPE[spec["dtype"]], layout=G.LAYOUT[spec["layout"]],
                           mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device))
    return ttnn.to_device(host, mesh_device, memory_config=memory_config), data
G.build_tensor = _build_tensor_host_tilize

def _unsqueeze_checked(t):            # _OP; run_case(..., op_name="ttnn.unsqueeze_to_4D") keeps the golden
    out = ttnn.unsqueeze_to_4D(t)
    assert t.buffer_address() == out.buffer_address()
    return out
```
```python
# prototype_ops copy: U.to_tt -> host from_torch(mesh_mapper=replicate) + ttnn.to_device(..., DRAM_MEMORY_CONFIG);
# same buffer_address assert; CASES / reference / PCC unchanged.
```

`to_device` needed no workaround in either file. No temp copy shrank any shape for the emulator (case 00 / `rope_3d` are one tile row and the op does no device work).

## 5. Blockers (symptom → owner)

- **B1 — harness helper, NOT either op under test (pre-existing, same as the reshape/untilize reports):** `ttnn.from_torch(…, layout=TILE_LAYOUT, device=…, mesh_mapper=…)` for bf16 dispatches the legacy `ttnn::tilize` (`TilizeDeviceOperation`, `Program(ProgramDescriptor)` → `CreateDataMovementKernel`) → `TT_FATAL @ tt_metal/impl/kernels/kernel.hpp:450: DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.` Fails as-checked-in cases `00/01/03` of `graph_ops/test_unsqueeze_to_4D.py` and both cases of `prototype_ops/test_unsqueeze_to_4D.py` before the op runs. **Owner:** `tilize` op uplift (legacy `TilizeDeviceOperation` is the op launching the Gen1 kernel) / `llama32_1b_quasar` harness owners (`graph_case.build_tensor`, `op_utils.to_tt`) — the host-tilize + `ttnn.to_device` idiom is the working alternative. Not an op edit; nothing was ported.
- **No op-level blocker** for `ttnn.to_device`, `ttnn.experimental.quasar.to_device` or `ttnn.unsqueeze_to_4D`: every case that reaches the op passes on craq-sim, and the RTL cases that ran pass.
- **Infra (not a code issue):** ZEBU unit contention — job 3's first launch was refused because the 1x3 slots were held by another user's session (§2); the client hung to its 2700 s timeout. Recovery was a single relaunch once the unit freed (PASS); no stale `zrun`/`vovsh` of mine was involved, so no cleanup was performed.

## 6. Reproduce

```bash
# craq-sim
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh light_ops
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_to_device.py -v -s          # 4 PASS
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_to_device.py -v -s      # 10 PASS
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_unsqueeze_to_4D.py -v -s     # 02 PASS, 00/01/03 FAIL (B1)
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_unsqueeze_to_4D.py -v -s # 2 FAIL (B1)
# temp copies (recreate from §4 next to the originals as test_tmp_unsqueeze_to_4D_qsr.py) -> 4 PASS / 2 PASS
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_unsqueeze_to_4D_qsr.py -v -s
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_unsqueeze_to_4D_qsr.py -v -s
# "no kernel" evidence: only firmware/ under the fresh cache
find $TT_METAL_CACHE -mindepth 2 -maxdepth 2 -not -name firmware   # -> empty

# ZEBU RTL (one case per file, one job at a time)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_to_device.py -k 01_1_i32 -v -s
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_to_device.py -k uint32-rot_idxs-batch1 -v -s
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_unsqueeze_to_4D_qsr.py -k 00_32x64 -v -s
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/prototype_ops/test_tmp_unsqueeze_to_4D_qsr.py -k rope_3d -v -s
```

Logs: `/tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_light_ops/` (`A..F_*.log` craq-sim, `EMU1..4_*.log` + `emu*_launcher_*.log` RTL).
