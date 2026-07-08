# Ring/Torus topology for the 16-chip pi0.5 prefill — investigation & status

**Status:** the perf win is real but **not realizable** on the current tt-metal stack.
Blocked by a fabric bug: **tt-metal issue [#22524](https://github.com/tenstorrent/tt-metal/issues/22524)**.

---

## TL;DR

- **Hypothesis (confirmed):** the TP=8 prefill's all-reduce CCL runs faster with a *ring/torus*
  wraparound than with the linear (adjacent-hop) topology we ship.
  - **Prefill stage: 12.96 ms (Linear, current) → 10.66 ms (Torus), −2.30 ms (−18%), 3-cam.**
    This nearly closes the gap to the standalone 8-chip build's 9.62 ms prefill.
- **Blocker (confirmed):** the 8-stage streamed **denoise deadlocks** (hangs) under the torus
  fabric — both traced and eager. So the prefill win cannot be realized end-to-end.
- **Root cause:** tt-metal **#22524** — the routing tables don't encode the torus *wraparound*
  (dateline) links, **and** the torus fabric's VC / bubble-flow-control deadlocks under
  *concurrent* socket transfers. This is a fabric-team fix, not a pipeline change.
- **Current shipped config is unchanged and correct:** `FABRIC_2D` + `Topology.Linear`,
  e2e ≈ 29.4 ms (2CQ, 3-cam).

---

## Background: the 16-chip topology

The 16-chip pipeline opens a `(2, 8)` mesh (rows 0–1 of the BH Galaxy):

- **Row 0 — prefill:** SigLIP DP + TP=8 VLM. The TP=8 all-reduce (`reduce_scatter` +
  `all_gather`) is a **ring along the 8-column X axis** — it benefits from the
  `col7 ↔ col0` wraparound.
- **Row 1 — denoise:** 8-stage streamed matmul_decode. Inter-stage **hop sockets**
  `chip(1,c) → chip(1,c+1)` + the **velocity-wrap** `chip(1,7) → chip(1,0)` — also a
  **ring along the X axis** (needs the same `col7 → col0` wraparound edge).
- **KV handoff:** per-chip **cross-row sockets** `chip(0,c) → chip(1,c)` — along the
  **2-row Y axis**.

To get a real `col7 ↔ col0` wraparound you need a **torus** fabric (`FABRIC_2D_TORUS_*`).
Plain `FABRIC_2D` has no wraparound (the ring/CCL falls back to Linear = adjacent hops).

---

## What was tried (chronological)

| # | Attempt | Result | Why |
|---|---------|--------|-----|
| 1 | `Topology.Ring` under `FABRIC_2D` | **Fatal** | `fabric_context.cpp`: `TT_FATAL(!(2D) || !(RING), "2D routing mode cannot be combined with LINE or RING")`. Ring needs 1D dateline deadlock-avoidance; incompatible with 2D routing. |
| 2 | `Topology.Ring` under `FABRIC_1D` on the **carved** `(1,8)` prefill row | **"Could not find any forwarding direction"** (`fabric.cpp:149` TT_FATAL) | A carved sub-slice has no *physical* `col7↔col0` wraparound edge; the 1D neighbor-lookup workaround finds no direction. |
| 3 | `FABRIC_1D_RING` on a standalone `(1,8)` sub-slice | **eth-handshake timeout** | Ring needs the 8 chips to *be* the SystemMesh torus (a physical ring). A sub-slice's wraparound partner isn't cabled as a ring edge. |
| 4 | **`FABRIC_2D_TORUS_XY` + `Topology.Torus`** (the no-host-bounce ring) | **Prefill routes + is FASTER (10.66 ms); denoise DEADLOCKS** | Torus = 2D routing (KV sockets keep working) *plus* wraparound datelines. Prefill CCL benefits. But the denoise ring/KV deadlock — see root cause. |
| 5 | Isolated `all_reduce` micro-probe (small 32×2048 tensor) | Ring ≈ Linear (0.32 ms both) — **misleading** | Latency-bound at that size. The *real* prefill CCL (36 all-reduces on 1024-tok block-sharded data) is bandwidth-bound and *does* benefit — hence #4's 10.66 ms. |
| 6 | Minimal socket-ring reproducers (see below) | single hop **PASS**; sequential ring **PASS**; chained ring **PASS**; ring+compute **PASS**; **8 concurrent cross-row transfers DEADLOCK** | The deadlock is specifically **concurrent** transfers on the torus, not the ring/compute per se. |
| 7 | Full pipeline under torus (traced) | **Hangs in denoise trace capture** | The socket ops route individually but the pipeline deadlocks. |
| 8 | Full pipeline under torus (eager denoise) | **Hangs in the eager denoise / KV handoff** | Same deadlock, not a trace-capture artifact. |
| 9 | **C++ patch** — extend the #22524 1D neighbor-lookup workaround into the `is_2d_fabric` branch for torus wraparound hops (original + refined to disambiguate the 2-row Y axis) | **Insufficient — still deadlocks** | *Decisive:* with the patch, the cross-row KV routes over the **same adjacent direction as plain `FABRIC_2D`** (where it works), yet still deadlocks under torus. So the deadlock is the torus **VC / bubble-flow-control under concurrent transfers**, NOT the routing direction. A routing-direction workaround cannot fix a VC/buffer deadlock. |

---

## Root cause — tt-metal #22524

Two coupled fabric problems, both under `FabricConfig::FABRIC_2D_TORUS_{X,Y,XY}`:

1. **Routing tables omit the wraparound (dateline) links.** `ControlPlane::get_forwarding_direction`
   (`tt_metal/fabric/control_plane.cpp`) just reads the intra-mesh routing table, which
   `routing_table_generator.cpp` never populates with torus wraparound edges. So a wraparound
   hop is routed the **long way** (opposite direction) instead of over the dateline.
   - There **is** a workaround, but **only for 1D fabric** — `tt_metal/fabric/fabric.cpp`,
     `append_fabric_connection_rt_args`, the `else` (`!is_2d_fabric`) branch loops
     `get_chip_neighbors` (physical topology) to find the wraparound direction. The
     comment literally says: `// TODO: Workaround for #22524 routing tables not having
     wraparound links`. The **2D-torus path is deliberately NOT covered** (uses the broken
     routing table).
   - Upstream commit that added the 1D workaround:
     `06a3fe01eb307597cff58a90c7dffbbc3b0bf7b2` ("#22524: Workaround for 1d fabric routing…").

2. **The torus VC / bubble-flow-control deadlocks under concurrent transfers.** Even when the
   direction is correct, N transfers hitting a shared torus dateline concurrently exhaust its
   finite per-link buffer pool → circular wait → deadlock (see `fabric_context.cpp`
   `need_deadlock_avoidance_support` / `bubble_flow_control_enabled_`). A **single** or
   **sequential** transfer never fills the pool, which is exactly why our isolated hops and
   sequential rings pass while the concurrent KV handoff / denoise pipeline deadlocks.

**Why the C++ patch (attempt #9) can't fix it:** the patch only corrects problem (1) — the
routing *direction*. Problem (2) is a VC/buffer flow-control issue, independent of direction.
Proof: the cross-row KV hop, routed via the *same adjacent direction as plain `FABRIC_2D`*
(where it works), still deadlocks under `torus_xy`. The only difference is the torus fabric
*mode* (its VC/flow-control), so that's the cause.

### Upstream status (checked 2026-07-07, `origin/main` tip `15236612f9e`)

- #22524 is **still open**. The 1D-only workaround + `// TODO: … #22524` comment are still in
  `fabric.cpp`. `get_forwarding_direction` still reads the wraparound-less routing table.
  `routing_table_generator.cpp` still has **no** torus wraparound population.
- Recent torus activity on main is **subtorus descriptor/validation** work
  (`#45629`, `#46401`, `#45724`) — **not** the routing-table wraparound / VC fix.
- **Pulling latest tt-metal does NOT fix this.**

---

## The real fix (fabric team, #22524)

1. Populate the torus **wraparound edges** in the intra-mesh routing table so
   `get_forwarding_direction` returns the correct dateline direction for 2D torus
   (removes the `// TODO #22524` workaround entirely).
2. **Dedicate virtual channels / buffers on the torus dateline** so *concurrent* transfers
   don't exhaust a shared pool (the actual deadlock we hit).

Neither is a pi0.5 / one-function change; both are in `tt_metal/fabric/` (routing-table
generator + EDM/router VC allocation).

---

## Measurements (3-cam, N=5, this branch)

| Metric | Linear (`FABRIC_2D`, shipped) | Torus (`FABRIC_2D_TORUS_XY`) |
|--------|------------------------------:|-----------------------------:|
| Prefill stage | **12.96 ms** | **10.66 ms** (−2.30) |
| SigLIP | 5.05 ms | 5.04 ms |
| Denoise | 11.4 ms | **deadlocks** |
| e2e (2CQ) | **29.4 ms** | not achievable |
| Isolated all_reduce, 1024-tok (FABRIC_1D standalone) | ring 0.336 / linear 0.330 ms | — |

8-chip reference (standalone `FABRIC_1D_RING`, for context): prefill **9.62 ms**.

---

## Reproducers

Minimal, model-free demonstrations that **concurrent** torus transfers deadlock while
sequential/single ones pass. Copied to `_bench_runs/torus_ring_repro/`. Run under a chosen
fabric via `FAB=torus_xy|2d`:

```bash
cd <repo>; export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
tt-smi -glx_reset && sleep 90            # torus opens are wedge-prone; settle
FAB=torus_xy python_env/bin/python _bench_runs/torus_ring_repro/ring_chained_repro.py   # PASS (sequential ring)
FAB=torus_xy python_env/bin/python _bench_runs/torus_ring_repro/ring_xrow_repro.py      # DEADLOCK (8 concurrent cross-row)
```

- `ring_socket_repro.py` — 8 forward hops + wrap, independent tensors → **PASS**.
- `ring_chained_repro.py` — chained data dependency, realistic 1024 payload → **PASS**.
- `ring_compute_repro.py` — ring + a matmul per stage → **PASS**.
- `ring_xrow_repro.py` — 8 concurrent cross-row (Y) transfers + ring → **DEADLOCK**.

These are the tightest repro to attach to **#22524**.

---

## Key code locations

- `tt_metal/fabric/fabric.cpp` — `append_fabric_connection_rt_args`: the `is_2d_fabric`
  branch (`get_forwarding_direction`, routing tables) vs the 1D `else` branch (`#22524`
  neighbor-lookup workaround). `TT_FATAL(forwarding_direction.has_value(), "Could not find
  any forwarding direction …")`.
- `tt_metal/fabric/control_plane.cpp` — `get_forwarding_direction` (reads intra-mesh table).
- `tt_metal/fabric/routing_table_generator.cpp` — intra-mesh table generation (no wraparound).
- `tt_metal/fabric/fabric_context.cpp` — `TT_FATAL` forbidding 2D+Ring;
  `need_deadlock_avoidance_support`; `bubble_flow_control_enabled_`.
- `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` — `FabricConfig`
  (`FABRIC_2D_TORUS_X/Y/XY`), `FabricType` (`TORUS_X/Y/XY`).
- pi0.5 side: `models/experimental/pi0_5/tt/tt_bh_glx/mesh_setup.py` (`open_decode_16_mesh`,
  fabric select), `stage_prefill_tp4.py` (`_ccl_base` topology), `pipeline_16_decode.py`
  (KV sockets), `tt/tt_pipeline/denoise_pipeline.py` + `_transport.py` (hop/wrap sockets).

---

## How to re-enable torus for testing (once #22524 is fixed)

The env-gated wiring was reverted to keep the tree clean. To re-test:

1. `open_decode_16_mesh`: select the fabric via an env (`PI0_16_FABRIC=torus_xy` →
   `FabricConfig.FABRIC_2D_TORUS_XY`) instead of hard-coded `FABRIC_2D`.
2. `stage_prefill_tp4._ccl_base`: add a `torus` option (`PI0_CCL_TOPOLOGY=torus` →
   `Topology.Torus`).
3. Run the prefill-stage probe / full breakdown under `PI0_16_FABRIC=torus_xy
   PI0_CCL_TOPOLOGY=torus`.

(See git history of this branch for the exact ~30-line diff.)
