# Progress journal

## 2026-07-15 — Session 0: planning

- Created job folder `claude_job/tt_stack_walkthrough/` with `prompt.md` (task brief) and `PLAN.md`
  (full plan: verified layer map with file paths, CCL signatures, workload verification design,
  execution steps, cautions, definition of done).
- Decisions: new job folder (separate from `dual_t3k_ops`); TTFM = document + live-demo the in-repo
  `run_fabric_manager` split lifecycle; Big-Mesh only.
- Stack layers already mapped from source (see PLAN.md): tt-topology (external), tt-fabric control
  plane, TTFM (`run_fabric_manager` + `FabricManagerMode`), tt-run/mpirun-ulfm, ttnn (open_mesh_device
  + CCL). Nothing run on hardware yet for this job.

## 2026-07-15 — Session 1: built + proved the stack

Done (all four steps complete):
1. **Preflight** — `test_system_health` → `[ PASSED ] 3 tests.` on both hosts.
2. **Proof workload** — wrote `scripts/stack_workload.py` (add, matmul, all_gather, all_reduce,
   reduce_scatter; per-local-shard torch goldens; reused `local_coords_and_tensors`/`pcc` from
   `../dual_t3k_ops/scripts/bigmesh_ops.py`). Ran under tt-run across both hosts (background + poll).
   - First run crashed rank 1 with `SIGBUS Non-existent physical address` in `fetch_queue_write`
     partway through (host CQ/pinned-buffer accumulation). Wedged the remote fabric → recovered with
     `tt-smi -r` both hosts. **Fix:** `ttnn.deallocate()` intermediate tensors + barrier between ops
     (new gotcha #14 in findings.md).
   - Re-run: **ALL PASS** for all 5 ops, exit 0, clean teardown. Captured `scripts/PASS_output.txt`.
3. **Fabric-manager live demo** — `scripts/fm_demo/`. Phase 1 `--initialize-fabric` on one T3K (2×4)
   **succeeded** (fabric up on 8 chips, node IDs printed, `fabric_status.txt` written, fabric left up).
   Phase 2 ENABLED attach and Phase 3 `--terminate-fabric` **both failed** on the T3K eth-heartbeat vs
   running-routers conflict (`Stuck at 0xabcd…` in remote-chip discovery) — a real, documented finding
   (new gotcha #15): `run_fabric_manager` hard-sets `TT_MESH_HOST_RANK=0` (single-host tool) and its
   ENABLED/TERMINATE halves are Galaxy-shaped, not T3K-shaped. Recovered with `tt-smi -r` both hosts.
4. **Docs** — wrote `STACK.md` (5-layer walkthrough + citations + runtime trace + hand-off diagram +
   embedded PASS output + FM transcript, cross-links GUIDE.md/topology.html), `findings.md`, this
   journal. System left healthy (both hosts `[ PASSED ] 3 tests.`).

### Optional — done
- Rendered a team-presentable HTML version `STACK.html` (self-contained, theme-aware, terminal-styled
  proof block + layer cards + FM-phase badges) and published it as an Artifact:
  https://claude.ai/code/artifact/3d6f1379-298d-455f-961e-4d92456e77cf (private until shared).

## 2026-07-15 — Session 2: fix the failed fabric-manager phase (user request)

The separate-process CLI ENABLED/TERMINATE failed on the T3K eth-heartbeat wall. Root-caused in source:
the wall only bites a FRESH process (new UMD Cluster ⇒ remote-over-eth re-discovery, which the running
EDM routers block). Fix: drive the SAME FabricManagerMode split lifecycle in ONE process
(`scripts/fm_lifecycle.py`) — the Cluster is built once, before fabric is up, and reused.
- **ENABLED attach now SUCCEEDS on the T3K** — captured `"Fabric initialized through Fabric Manager"`
  (fabric_firmware_initializer.cpp:320), the line the CLI could never reach. Evidence:
  `scripts/fm_demo/04_lifecycle.log` + `04_TRANSCRIPT_inprocess.md`.
- **Remaining boundary (honest):** a workload dispatched under the in-process ENABLED reattach hangs on
  its first device write on a T3K (remote-chip dispatch tunnels over the ethernet the ENABLED-mode
  control-plane reconfigure disturbs). Diagnosed live (futex_wait + 99% spin, 0 kernels built = hung op,
  not a compile). So a fully-green ENABLED workload is not achievable on this T3K/commit; the fully-green
  fabric proof stays the DEFAULT-mode 16-chip workload (`scripts/PASS_output.txt`).
- Killed the hung demo (SIGTERM) → fabric left up → `tt-smi -r` both hosts → health GREEN on both.
- Updated STACK.md §3, findings.md (gotcha #15), fm_demo/TRANSCRIPT.md, and re-published the Artifact.
- System left healthy; no leftover processes.
