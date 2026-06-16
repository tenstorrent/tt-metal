# nsight-counters

## Phase 3a fully wired (CSV column); 2a==2b OOM coupling found
**2026-06-15 17:59** · `tt-metal@3e0ebf5fed9-dirty`

Delta on top of the MVP entry below. 10 commits now on the branch.

- **Phase 3a complete**: `NOC BYTES FROM COUNTERS` column now emitted by
  `process_ops_logs.append_device_data` as a tt-npe-absent fallback
  (`noc_bytes_from_trace_dir`). Additive, guarded by the process_ops_logs golden
  regression test (5 passed). Validated end-to-end re-processing the real bh
  noc-trace capture: eltwise = exactly 402,653,184 bytes. BW % applied
  downstream via `noc_bandwidth.noc_bw_util_pct` with the part peak.
- **Verified Phase 2b reframing**: the cpp fast path (`load_device_perf_report`)
  already streams (csv.DictReader, ~ops×devices, bounded). The 140 GB OOM is the
  *legacy* `pd.read_csv` of the full per-core device log, and counters *force*
  the legacy path (CLI disables cpp post-process when perf_counter_groups set).
  => **Phase 2a (counters on the cpp path) IS the Phase 2b fix.** Chunking the
  legacy path is a strictly worse band-aid.
- **Phase 2a tractability**: `ProgramsPerfResults` is per-PROGRAM by construction
  (AnalysisDimension::PROGRAM). Per-core/per-RISC counters need a new dimension +
  result struct + nanobind serialization + CSV writer changes + rebuild + device
  cross-validation vs the Python path. Large; not a safe unsupervised slice.
- **Phase 4**: largest open Q is whether the serial GPU-zone path
  (`TracyTTDevice.hpp` PushStart/PushEndMarker) can reach Tracy `ZoneText` or
  needs a new QueueType. Not yet read.


## MVP landed: counter-trust gate + sizing + scope + NoC bytes + workflow
**2026-06-15 11:00** · `tt-metal@faa44d88545-dirty`

Branch `worktree-smarton+profiler+nsight-counters-2026-06-15` (off origin/main +
the 2 profiler-infra-fixes prereqs cherry-picked). Implementing
`/home/smarton/tracy-nsight-counters-plan.md` (company-wide profiler infra).
All work is TDD'd; host-only tests: 24 passed. Device: bh 4x8 (chip reports 130
compute cores, not 140 — peak table assumes 140, ~7% basis gap absorbed by
tolerances).

**Delivered + validated (commits 127aa7a..faa44d8):**
- **Phase 0** — `tests/ttnn/tracy/test_counter_utilization_sanity.py` +
  `counter_sanity_workload.py`. Gate: grid-normalized FPU counter must equal the
  achieved fraction of the analytical peak. Device-verified: FPU 21.9% vs
  achieved 21.5% of HiFi2; eltwise 77.6% DRAM BW, FPU idle. Also added `sfpu`
  CLI group alias (rides FPU bit).
- **Phase 1a** — `tools/tracy/perf_counter_sizing.py` markers/zone (BH: fpu3
  pack5 unpack22 l1*16 instrn59), L1 250-slot headroom, PROGRAM_SUPPORT_COUNT
  recommender. Cross-checked vs real capture (FPU=3 markers/zone). Finding: all
  9 BH groups = 169 markers < 250, so per-zone L1 never overflows on BH; the
  DRAM program-support count is the real limit.
- **Phase 2.5** — `tools/tracy/perf_counter_scope.py` deterministic per-op-grid
  compute-core sampling, env `TT_METAL_PROFILER_COMPUTE_CORE_SAMPLE=K`, wired
  into `process_ops_logs.py` (default off). Device-verified: sampled matmul FPU
  Util Median 25.07% == full-grid 25.04%. Preserves per-core util dist, NOT
  grid-summed metrics (documented).
- **Phase 3a** — `tools/tracy/noc_bandwidth.py` per-op NoC bytes/BW% from the
  profiler's own noc_trace*.json, independent of tt-npe (absent on this host).
  Device-verified Phase 3 gate: 8192² bf16 eltwise = 402.7 MB vs analytical
  402.65 MB. NOTE: `--collect-noc-traces` ran clean on a SINGLE chip → the
  Phase 3b hang is fabric/multi-chip init, not local NoC.
- **Phase 5** — `tools/tracy/capture_counters.py` (pins buffer, applies scope,
  archives CSV+.tracy+scope to ~/traces/<ts>), `tt_metal/tools/profiler/PERF_COUNTERS.md`,
  `/home/smarton/profiling-recipe.md`. Device-verified archive + sampled run.
- **Phase 1c** — L1_0/L1_1 multi-pass merge already exists
  (`process_model_log.py` run_multi_pass/merge_pass_csv); remaining 1c work is
  folding a NoC-events pass into the same zone-key merge.

**Not done (heavy / high-risk — deliberately not faked):**
- **Phase 2a** per-RISC C++ fast post-process (extend `profiler_analysis.{hpp,cpp}`
  + nanobind `GetLatestProgramsPerfData` to per-core/per-RISC). Large C++; needs
  rebuild + device.
- **Phase 2b** streaming/bounded-memory Python post-process. Big refactor of the
  pandas full-load in `process_ops_logs.py`; needs a mesh-scale (140 GB) capture
  to validate the OOM fix. Phase 2.5 sampling already cuts the compute-counter
  row explosion.
- **Phase 3a CSV wiring** — `noc_bandwidth` math is validated but not yet emitted
  as a CSV column. Insertion point: `analyzeNoCTraces` returns None without
  tt-npe (`process_ops_logs.py:1792`); add a fallback that calls
  `noc_bytes_from_trace_dir` and writes a "NoC BW UTIL (FROM COUNTERS) (%)"
  column joined by run_host_id using DEVICE FW DURATION. Held back pending the
  profiler golden-regression suite (`tests/tt_metal/tools/profiler/test_device_logs.py`).
- **Phase 3b** noc-trace fabric/ethernet-init hang — needs a multi-chip bisect;
  NOT attempted (would risk wedging the shared 32-chip box for other jobs).
- **Phase 4** Tracy GUI zone tooltips — vendored-fork protocol change
  (`tt_metal/third_party/tracy/...`; note: real path is under tt_metal/, not the
  plan's `third_party/tracy`). Largest item; needs rebuild + GUI validation.

**Resume:** `source python_env/bin/activate; PYTHONPATH=tools pytest
tests/ttnn/tracy/test_counter_*.py tests/ttnn/tracy/test_noc_bandwidth.py -q`.
Device runs via tt-device-mcp (queue shared, same owner). See [[validate-trace-at-real-scale]].
