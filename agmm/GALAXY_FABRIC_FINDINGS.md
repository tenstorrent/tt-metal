# AGMM sweep on bh-glx-120-b10u02 — fabric bring-up findings (2026-07-14)

Investigation log for why the AGMM block-size sweep (`agmm/run_sweeps.py`,
shape `s2_fabric_n512`) could not run its `num_links=2` all-gather ring on this
Blackhole galaxy. **Diagnosis is complete; the remaining blocker is hardware.**

## TL;DR

The sweep harness itself is fully working. The op cannot run at `num_links=2`
on this host because of a chain of three independent facts:

1. **2 ethernet links are hardware-faulty** (data corruption under traffic,
   survives retrain) — `tray2/asic1/ch3` and `tray3/asic1/ch1`.
2. This is a **subtorus-wired galaxy**: a single-galaxy ring is a sub-torus that
   only exists on the physical wrap cabling; a plain-mesh / default fabric setup
   leaves the ring axis with only **1 usable channel per hop**.
3. The swept op **`all_gather_minimal_matmul_async` is 1D-fabric-only** — its
   kernel is hard-coded to `LowLatencyPacketHeaderT<1>` and will not compile
   under any `FABRIC_2D_TORUS_*`. So the natural fix for #2 (a 2D torus
   descriptor) is incompatible with this op.

Net: with a 1D fabric (the only kind this op supports) and the current faulty
links, only `num_links=1` is satisfiable. `num_links=2` needs the hardware fixed.

## Environment

- Host: `bh-glx-120-b10u02` (BH_GALAXY_REV_C, single host, 32 chips)
- Branch: `cglagovich/agmm_analysis` @ `5e9e8b2ab11`, Release build in `build_Release/`
- Python env: created via `./create_venv.sh` → `python_env/` (uv, py3.10)

## What is confirmed working (harness validated up to the op)

- Venv + full import stack (`ttnn`, `tracy`, sweep worker).
- `serve_wasm.py` lazy-`websockets` fix is present.
- **Run invocation:** must have the venv on `PATH` so the profiler subprocess's
  bare `python3 -m tracy` resolves to the venv (not `/usr/bin/python3`):
  ```bash
  export PATH="$PWD/python_env/bin:$PATH"
  TT_METAL_HOME=$PWD PYTHONPATH=$PWD python3 agmm/run_sweeps.py --mode full --ids s2_fabric_n512
  ```
- The harness runs end-to-end up to the op: profiler subprocess launches,
  spec→worker handoff works, device inits, the block grid + L1 filter compute
  (185 combos for `s2_fabric_n512`). Only the on-device op execution fails.

## The blocker chain, with evidence

### 1. Two faulty ethernet links (hardware)

`run_cluster_validation --send-traffic` (ran twice, with a `topology
--retrain_6u` in between — fault persists identically):

| Host | Tray | ASIC | Ch | Port | Type | Failure | Mismatch Words | Retrains | CRC/FEC |
|---|---|---|---|---|---|---|---|---|---|
| bh-glx-120-b10u02 | 2 | 1 | 3 | 2 | QSFP_DD | Data Mismatch | 90889 | 0 | 0 |
| bh-glx-120-b10u02 | 3 | 1 | 1 | 3 | QSFP_DD | Data Mismatch | 90889 | 0 | 0 |

Signature = PHY trains clean (0 retrains, 0 CRC, 0 FEC codewords) but data is
corrupted under traffic and the workload times out. Classic marginal
cable/connector/SerDes fault. **Not** missing cables — `umd/topology` finds 2
links on all 64 chip-pairs (`links-per-pair {2:64}`). **Not** software-retrainable.

`fabric_builder.cpp:146`: usable `num_links = min(channels per direction)`, so a
degraded hop caps the whole ring.

### 2. Subtorus wiring → 1D ring sees 1 channel/hop

`FABRIC_1D_RING` (the op's fabric) on a ubb galaxy resolves to
`FabricType::TORUS_XY` (`fabric_host_utils.cpp:42`). Bring-up succeeds and the
mesh opens, but the op aborts at warmup:
```
TT_FATAL: Requested link index 1 is out of bounds. 1 ethernet channels available
to forward b/w src (M0, D0) and dst (M0, D1)
```
i.e. only 1 active channel in the forwarding direction — cannot satisfy
`num_links=2`. Note D0 pins to tray1/asic1, which is NOT one of the 2 faulty
links above, so the ring is short of a 2nd channel on more than just those hops.

### 3. The op is 1D-fabric-only (blocks the torus fix)

Setting `FABRIC_2D_TORUS_X` / `FABRIC_2D_TORUS_Y` with the matching mesh-graph
descriptor (verified applied: `Using custom mesh graph descriptor: ...`,
`cluster_type ClusterType.BLACKHOLE_GALAXY`, and a correctly-formed torus —
logical degree histogram `{3:8, 4:24}` for an 8×4 4-axis ring) makes the
**kernel fail to compile**:
```
matmul_dataflow_common.hpp:999: cannot convert 'HybridMeshPacketHeader*'
(HybridMeshPacketHeaderT<35>) to 'LowLatencyPacketHeader*' (LowLatencyPacketHeaderT<1>)
... no matching function for fabric_unicast_noc_unicast_write_with_state<...>
Fatal Python error: Segmentation fault
```
So `all_gather_minimal_matmul_async` cannot use 2D torus fabric at all. deepseek_v3
runs rings on this galaxy class, but via its own 2D-capable CCL ops — not this one.

## Things tried and ruled out

- Galaxy reset (`tt-smi -glx_reset` needs sudo; user ran `tt-smi -r`) — cleared a
  wedged MMIO state but not the link faults.
- `topology --retrain_6u` — no effect on the data-mismatch links.
- Full 4×8 torus (`FABRIC_2D_TORUS_X` + `single_bh_galaxy_torus_x`) — descriptor
  applied and topology correct, but same "1 channel available" (op still 1D; and
  the ring is link-degraded). Also incompatible per #3.
- 4×4 subtorus (`FABRIC_2D_TORUS_Y` + `single_bh_galaxy_subtorus_y4` +
  `TT_VISIBLE_DEVICES` for ASICs 3,4,7,8 of trays 1-4, avoiding the faulty ASIC-1
  links). Pinned descriptor: placement solver rejected the fixed pinnings
  (`could not fit ... relax pinnings`). Non-pinned (RELAXED): **mesh mapped
  successfully onto the 16 healthy chips** — but the op then hit the #3 kernel
  compile error. This confirms the mesh/topology side works; the op is the wall.

For reference, the 16-chip carve mapping derived from `umd/topology`
(`tt-smi -glx_list_tray_to_device` + `asic_locations`):
`TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31` (ASICs 3,4,7,8 of
trays 1-4).

## Recommended next steps (for the galaxy owner)

1. **Fix the 2 faulty links** — reseat/replace the QSFP_DD cables at
   **tray 2 port 2** and **tray 3 port 3**, then confirm clean with:
   ```bash
   build_Release/tools/scaleout/run_cluster_validation --send-traffic
   ```
   Target: `Num Unhealthy Links: 0`.
2. Re-run the sweep unchanged (config is `FABRIC_1D_RING`, `num_links=2`):
   ```bash
   export PATH="$PWD/python_env/bin:$PATH"
   TT_METAL_HOME=$PWD PYTHONPATH=$PWD python3 agmm/run_sweeps.py --mode full --ids s2_fabric_n512
   ```
   If a healthy 1D ring still reports only 1 channel/hop, the subtorus 1D-ring
   wiring genuinely provides 1 channel/direction and `num_links=2` is not
   achievable for this op on this board — see #3 options below.
3. If 2-link numbers are needed but the 1D op can't get 2 channels here:
   either (a) run on a galaxy whose 1D ring provides 2 channels/hop, or (b) if a
   2D-fabric variant of the all-gather-matmul op exists/lands, use the 4×4
   subtorus recipe (proven to map here): `FABRIC_2D_TORUS_Y` +
   `single_bh_galaxy_subtorus_y4_graph_descriptor.textproto` +
   `TT_VISIBLE_DEVICES=<the 16 above>`.

## To just validate the harness end-to-end now (no hardware fix)

Temporarily set `num_links: 1` in `DEVICE_CONFIGS["bh_4x8"]` and run — a 1D ring
needs only link 0 (which is available). This exercises the full pipeline
(search → measure → roofline → history); the roofline fabric term will reflect 1
link rather than the target 2.

## State on exit

- Working tree clean: `bh_4x8` reverted to its original `FABRIC_1D_RING`,
  `num_links=2`, `(4,8)`, `cluster_axis=0` config. No code changes committed.
- Device healthy (topology discovery clean); orphaned sweep/tracy/profiler
  processes from a mid-run segfault were killed.
- `agmm/sweep_history.csv` contains failed-status rows from these attempts
  (`SWEEP_FAILED` / `NO_COMBOS`); no `OK` rows were produced.
