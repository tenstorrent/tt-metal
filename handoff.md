# MP Fabric / ttsimd Daemon — Agent Handoff

Last updated: 2026-05-28 (session 2 — sem-zero clamp fix)

## Latest session breakthroughs (2026-05-28)

### MAJOR WIN: mesh_socket TIMEOUT → FAIL in 22s (sem-zero clamp)

Three sequential fixes landed today and reduced mesh_socket from TIMEOUT(1800s) to FAIL(22s) with full fabric init succeeded:

| # | Failure | Fix | Commit |
|---|---------|-----|--------|
| 1 | `ImportError: write_to_device_buffer(..., CoreRangeSet*)` | Restored 7-arg overload | tt-metal `da3a463eff5` |
| 2 | Fabric router sync timeout (cross-rank chip mismatch) | Honor `TTSIMD_PHYSICAL_CHIP_IDS` in `ttsimd::translate_chip_id_locked` | craq-sim `6c819313` |
| 3 | `tile_mmio_rd8: addr=0xffb123a3` daemon abort | Permissive WH RISCV_DEBUG byte MMIO | craq-sim `acaaca6a` |
| **4** | **mesh_socket TIMEOUT at "Creating sockets" (dispatch_go_signal=0 forever)** | **Clamp `t_tile_dispatch_kernel` sem-zero so it cannot overwrite BRISC kernel text** | **craq-sim `6f54fa6f`** |

Fix #4 root-cause walkthrough (runtime evidence in `.cursor/debug-ae7d0a.log`):

- Instrumented `tt_metal/llrt/llrt.cpp:write_binary_to_address` (H_BIN_WRITE) confirmed host writes 31076 B BRISC binary to chip(0,0) noc(18,25) tile_id=56 L1 0x81e0, first_word=0xfe010113.
- Instrumented `craq-sim/src/tile.cpp:tile_wr_bytes` (H_TILE_WR) confirmed the write reaches the SAME `g_t_tiles[56].sram` at addr 0x81e0 size 31076.
- Instrumented `craq-sim/src/libttsim.cpp:t_tile_dispatch_kernel` (H_DISPATCH_TENSIX) reads launch_msg fields after the write: `sem_offset=0x30, sem_base=0x8190, sem_bytes=256, kernel_entry=0x81e0`.
- `[sem_base, sem_base+sem_bytes) = [0x8190, 0x8290)` OVERLAPS `kernel_entry=0x81e0` → the 256 B fixed sem zero wipes the first 128 B of the kernel binary. Dispatch FW boots into all-zero L1, never raises GO, host CQ stalls forever.
- Fix clamps `sem_bytes = min(sem_bytes, kernel_text_base - sem_base)` so the zero never extends into the kernel-text window.

### New failure point (next blocker)

`2.mp/mesh_socket_shrunk` now reaches **"Fabric initialized on 4 devices"** on both ranks (8/8 chips), then aborts inside `libttsim_clock_all_devices` during `FabricFirmwareInitializer::wait_for_fabric_router_sync`:

```
ttsimd.log: [162528] ERROR: UndefinedBehavior: rv32_step: invalid pc=0x49960
mpi: libttsim_rpc: failed to read response/callback prefix
stack: TTSimCommunicator::advance_clock → libttsim_clock_all_devices → rv32_step
```

The dispatch BRISC FW is now actually executing (great — sem-zero clamp worked) and runs far enough to jump to L1 `0x49960`, where there's no valid instruction. Likely causes:
1. FD dispatch kernel returns from a function and the stack/return-addr was scribbled (no kernel-config layout protection in sim).
2. A jump table or vtable lookup mis-resolves to an unmapped L1 page.
3. Out-of-bounds NOC-write from BRISC clobbers BRISC's own instruction stream.

Suggested next steps:
- Add a deeper rv32_step error message that prints surrounding L1 bytes + the last 8 PC values for diagnosis.
- Compare BRISC's intended jump targets against `kernel_config_base + binary_size` to confirm no kernel-config writes are landing in the BRISC text region.
- Check whether a second launch (rd_ptr=1) in the launch ring is firing with a yet-uninitialized kernel_config slot.

### Latest results

- Job 12830: `mesh_socket_shrunk` → `FAIL rc=134 dur=22s` (vs prior `TIMEOUT 1800s`). Results dir: `mp-run-shrunk-20260528T022207Z/`.

## Goal

Get **multiprocess (MP) fabric tests** passing under **shared `ttsimd` daemon mode** on Galaxy Slurm compute (`bh-glx-*`). These tests use `tt-run` + MPI ranks + craq-sim simulator (not real T3K hardware).

Success means the **shrunk MP sweep** (`scripts/run-nkapre-mp-ttsim-sweep-shrunk.sh`) passes all 10 suites in daemon mode, starting with `2.mp/mesh_socket_shrunk`.

---

## Where to work (important)

| Repo | Path | Branch | PR |
|------|------|--------|-----|
| **tt-metal (canonical)** | `/data/rsong/tt-metal3` | `nkapre/multichip-mp-ttsim-mock-cluster-rank-binding-20260527` | [#45379](https://github.com/tenstorrent/tt-metal/pull/45379) |
| **craq-sim** | `/data/rsong/craq-sim` | `nkapre/multichip-mp-daemon` | [#51](https://github.com/tenstorrent/craq-sim/pull/51) (draft) |

**Do NOT use `/data/rsong/tt-metal-fork` as the primary tt-metal worktree.** Use it only for reference docs (`nkapre-fork-test-commands.md`, this file).

Both branches are **already pushed** and up to date with their remotes as of 2026-05-27.

Stale remote: `origin/nkapre/multichip-mp-ttsim-host-fixes` on tt-metal is an old single-commit tip; the canonical branch is the long PR branch above.

---

## Current failure (start here)

**Suite:** `2.mp/mesh_socket_shrunk`
**Latest run:** `/data/rsong/tt-metal3/craq-parity-results/mp-run-shrunk-20260527T234044Z-tt-metal3-bundled/`
**Result:** `FAIL rc=134 duration=37s`

### Failure chain (runtime evidence)

1. **`ttsimd` aborts** during fabric init:
   ```
   [21043] ERROR: AssertionFailure: send_eth_data: Invalid file descriptor
   ```
   Source: `craq-sim/src/eth_io.cpp:27` — `write_fd == -1`.

2. **MPI ranks lose daemon connection:**
   ```
   libttsim_rpc: failed to read response/callback prefix
   Signal: Aborted (6)
   ```
   Stack: `FabricFirmwareInitializer::wait_for_fabric_router_sync` → `MeshDeviceImpl::create`.

3. Test gets far enough to compile fabric ERISC routers and dispatch kernels, then dies at **fabric router sync polling** — not a slow timeout, but a **daemon crash** triggered by eth I/O.

### What is NOT the problem anymore

Previous layers of bugs were fixed on craq-sim `nkapre/multichip-mp-daemon`:

| Old failure | Fix |
|-------------|-----|
| `rv32_step: invalid pc=0x49960` | ERISC dispatch guard in `libttsim.cpp` — skip if kernel binary missing/OOB |
| `tensix_decode_and_execute_inner: opcode=0x0` | Tensix dispatch guard — skip if L1 has no valid RV32 instruction |
| `rv32_mem_rd: unaligned addr=0x3` | Soft-reset / BRISC release guards + tolerant unaligned reads in `riscv_impl.h` |
| `eth_switch_enqueue: no destination registered for MAC 0xab` | Lazy cross-mesh peer auto-pairing in `eth_switch.cpp` (`354a7977`) |
| Per-rank chip registry collisions | `ttsimd` per-connection chip_id translation (`09266298`) |

Mock cluster misconfiguration on tt-metal was also fixed: **`--mock-cluster-rank-binding`** (not `TT_METAL_MOCK_CLUSTER_DESC_PATH`, which `tt-run` blocks).

---

## Next debugging target: `send_eth_data: Invalid file descriptor`

### Hypotheses to test (with instrumentation)

1. **H_ETH_FD_UNINIT** — Eth tile `write_fd` never set via `configure_eth_io()` before first `send_eth_data()` call in daemon mode (cross-rank peer path uses fd before socketpair/pipe setup).

2. **H_ETH_FD_STALE** — fd was valid in one MPI rank's registration but the daemon-side eth link table entry for the destination tile has `write_fd=-1` after lazy peer pairing.

3. **H_ETH_CROSS_RANK** — Cross-rank eth traffic goes through `eth_switch` auto-pairing but the paired endpoint's I/O fds were not propagated to the sending tile's `EthSendConfig`.

4. **H_CHIP_ID_MISMATCH** — With `TT_METAL_NO_CHIP_ID_REMAP=1` + `TTSIMD_PHYSICAL_CHIP_IDS=1`, UMD registers eth links on chip IDs that don't match what the eth tile send path resolves at runtime.

5. **H_DAEMON_RPC_ORDER** — Multiple MPI ranks register/configure eth links concurrently; one rank's registration is overwritten or partially applied before send.

### Where to instrument

| File | What to log |
|------|-------------|
| `craq-sim/src/eth_io.cpp` | `send_eth_data(write_fd, ...)` — log fd, errno context |
| `craq-sim/src/tile.cpp` | `e_tile_send_partial_transaction` — log `config.write_fd`, dest MAC, tile_id |
| `craq-sim/src/eth_switch.cpp` | `eth_switch_enqueue*`, lazy peer pairing — log resolved dest, fd state |
| `craq-sim/src/libttsim.cpp` | eth link registration from UMD — log chip/channel/MAC/fd |

Debug log path (NDJSON): `/data/rsong/tt-metal-fork/.cursor/debug-ae7d0a.log`
Session ID: `ae7d0a`

Existing agent logs in craq-sim (keep during debug): `#region agent log` blocks in `eth_switch.cpp`, `libttsim.cpp`, `tile.cpp`, `riscv_impl.h`.

---

## How to run tests

### Rules (Galaxy Slurm)

- **Never run sweeps on the login node.** Submit Slurm jobs only.
- Compute guard: `scripts/lib/require-bh-glx-compute.sh`
- Default node: `bh-glx-b06u08`, partition `bh_sc5_B2B9_D12`

### Submit shrunk MP sweep

From **login node**, in `/data/rsong/tt-metal3`:

```bash
./scripts/submit-nkapre-mp-ttsim-shrunk-slurm.sh
```

Single suite only:

```bash
SUITE_FILTER=mesh_socket ./scripts/submit-nkapre-mp-ttsim-shrunk-slurm.sh
```

### Manual repro on compute (after `salloc` or ssh to `bh-glx-*`)

```bash
cd /data/rsong/tt-metal3
source python_env/bin/activate   # MUST use bundled venv (create_venv.sh --bundle-python)
export CRAQ_SIM=/data/rsong/craq-sim
export TTSIM_USE_DAEMON=1
SUITE_FILTER=mesh_socket ./scripts/run-nkapre-mp-ttsim-sweep-shrunk.sh
```

### Rebuild after code changes

**craq-sim** (WH target for these tests):

```bash
cd /data/rsong/craq-sim
python3 make.py wh release    # produces src/_out/release_wh/libttsim.so
cd daemon && make             # produces daemon/_out/ttsimd
```

**tt-metal3**:

```bash
cd /data/rsong/tt-metal3/build_Debug
ninja -j32 tt_metal tt-umd test_mesh_socket_main multi_host_fabric_tests test_tt_fabric \
  distributed_multiprocess_tests unit_tests_dual_rank_2x2 unit_tests_dual_rank_2x4 unit_tests_ttnn
```

Stale binaries cause ABI mismatches (`bad_function_call`, mysterious SIGABRT). Always rebuild test binaries after `libtt_metal.so` changes.

---

## Launch configuration (must be correct)

Every MP fabric test in the shrunk sweep uses:

```
--mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml
```

**Both 2x2 and 2x4 suites use the T3K 2x4 big-mesh mapping file.** Do not use `t3k_cluster_desc.yaml` or the env var `TT_METAL_MOCK_CLUSTER_DESC_PATH`.

Daemon mode env (set by sweep script):

```
TT_METAL_NO_CHIP_ID_REMAP=1
TTSIMD_PHYSICAL_CHIP_IDS=1
TTSIM_USE_DAEMON=auto   # launches ttsimd before mpirun
TT_TTSIMD_SOCKET=/tmp/ttsimd-<pid>.sock
```

Simulator artifacts copied per run:

```
CRAQ_SIM/src/_out/release_wh/libttsim.so  →  $RESULTS_DIR/sim_wh_multichip/
tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml  →  soc_descriptor.yaml
```

---

## Shrunk sweep suites (10 total)

| Suite | Binary | Notes |
|-------|--------|-------|
| `2.mp/2x2_fabric_ubench_shrunk` | `test_tt_fabric` | Single unicast, 64 bytes |
| `2.mp/multi_host_fabric_shrunk` | `multi_host_fabric_tests` | `MultiHopUnicast` only |
| `2.mp/mesh_socket_shrunk` | `test_mesh_socket_main` | **Current blocker** |
| `2.mp/BigMeshDualRankTest2x4` | `distributed_multiprocess_tests` | |
| `2.mp/BigMeshDualRankMeshShapeSweep` | same | |
| `2.mp/ttnn_dual_rank_2x2_shrunk` | `unit_tests_dual_rank_2x2` | |
| `2.mp/ttnn_dual_rank_2x4_shrunk` | `unit_tests_dual_rank_2x4` | |
| `2.mp/ttnn_launch_op` | `unit_tests_ttnn` | |
| `2.mp/py_data_parallel` | pytest | |
| `2.mp/py_submesh` | pytest | Often TIMEOUT (900s), not crash |

Historical peak (pre-daemon-crash-fix, job 11839 on tt-metal2): **6 PASS / 3 FAIL / 1 TIMEOUT**. Fabric suites failed on MAC 0xab before eth_switch fixes. Non-fabric suites mostly pass.

---

## Key commits already landed

### craq-sim `nkapre/multichip-mp-daemon`

- `354a7977` — eth_switch lazy cross-mesh peer auto-pairing
- `09266298` — ttsimd per-connection chip_id translation
- `21e6570b` — park cores when kernel binary missing; tolerate unaligned reads

### tt-metal3 (base: `ridvan/nkapre-multichip-metal-v2`)

- `625eeccc6af` — host-side MP daemon fixes (SystemMesh chip filtering, NO_REMAP, DRAM-backed CQ, fabric router sizing, shrunk sweep scripts)
- `3a345824c09` — `--mock-cluster-rank-binding` fix + Slurm wrappers

---

## Uncommitted local work (not on remote)

In `/data/rsong/tt-metal2` only (WIP, do not treat as canonical):

- `fabric_firmware_initializer.cpp` — `H_FAB_ROUTER_SYNC` poll logging + `advance_device_execution` in sync loop
- `command_queue_common.cpp` — `get_cq_dispatch_go_signal()` helper
- `llrt.cpp` — `sim_arm_launch_watcher()` after GO signal in sim mode
- Various dispatch/device tweaks

These were being used to debug a **prior** failure mode (CQ stall: `dispatch_go_signal=0`). That may still be relevant **after** the eth fd crash is fixed. Consider cherry-picking into tt-metal3 once validated.

---

## Architecture reminder

```
MPI rank 0 (TT_VISIBLE_DEVICES=0,1)          MPI rank 1 (TT_VISIBLE_DEVICES=2,3)
  └─ tt-umd → libttsim_rpc.so ──┐              └─ tt-umd → libttsim_rpc.so ──┐
                                 │  UNIX socket                                │
                                 └──────────────► ttsimd (single process) ◄─────┘
                                                    ├─ chip registry (global)
                                                    ├─ eth_switch (global)
                                                    └─ libttsim.so (loaded once)
```

Each rank sees disjoint physical chip IDs under one daemon. Eth cross-rank traffic must go through the **shared** `eth_switch`, not private per-rank state.

---

## Known gotchas

1. **`python_env` must be bundled** — symlinked venv breaks on compute nodes (`python3: not found`). Recreate with `create_venv.sh --bundle-python` in tt-metal3 if needed.

2. **Slurm job script output path** — `scripts/slurm-nkapre-mp-ttsim-shrunk-job.sh` still writes Slurm logs to `/data/rsong/tt-metal2/craq-parity-results/` (copy-paste artifact). Harmless but confusing.

3. **In-process vs daemon** — In-process sim (`TTSIM_USE_DAEMON=0`) does not exercise true MP fabric. Many tests skip or pass for wrong reasons. Always test with daemon mode for MP fabric.

4. **Do not add `ALLOW_LOGIN_RUN` bypasses** — workspace rule enforces Slurm-only for sweeps.

5. **Debug instrumentation** — Do not remove `#region agent log` blocks until post-fix verification logs confirm success.

---

## Suggested next-agent workflow

1. Add instrumentation to `send_eth_data` / eth tile send path / eth_switch peer pairing (hypotheses H_ETH_* above).
2. Clear debug log: delete `/data/rsong/tt-metal-fork/.cursor/debug-ae7d0a.log`.
3. Rebuild craq-sim WH + ttsimd.
4. Run `SUITE_FILTER=mesh_socket` shrunk sweep on compute.
5. Analyze logs — confirm/reject each hypothesis with cited log lines.
6. Fix with evidence; keep instrumentation for verification run.
7. Push fixes to `craq-sim:nkapre/multichip-mp-daemon` and/or `tt-metal3` PR branch.
8. Once mesh_socket passes, run full 10-suite shrunk sweep.
9. If CQ stall reappears (`dispatch_go_signal=0`), pick up tt-metal2 WIP (`sim_arm_launch_watcher`, go_signal polling).

---

## Reference docs

- Test commands (full, not shrunk): `/data/rsong/tt-metal-fork/nkapre-fork-test-commands.md`
- Agent transcript: `/home/rsong/.cursor/projects/data-rsong-tt-metal-fork/agent-transcripts/ae7d0a21-b1f5-4de8-b388-59e2602cae57/ae7d0a21-b1f5-4de8-b388-59e2602cae57.jsonl`
- Latest mesh_socket log: `.../mp-run-shrunk-20260527T234044Z-tt-metal3-bundled/2.mp_mesh_socket_shrunk.log`
- Latest ttsimd log: `.../mp-run-shrunk-20260527T234044Z-tt-metal3-bundled/ttsimd.log`
