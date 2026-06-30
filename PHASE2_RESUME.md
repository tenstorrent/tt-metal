# CCL op-gen — Phase 2 resume (machine-switch-proof)

**Written:** 2026-06-24 EOD · **Updated:** 2026-06-25 (session 10, overnight) · **Status:** Phase 1 DONE+green;
**Phase 2 headline DONE** — the pipeline GENERATED `all_gather` (the harder ring op) and it verified GREEN
on the WH sim (22/22 acceptance + 8/8 precision). See memory "session 10" entry + OVERNIGHT_REPORT.md.
Committed into git at the root of `origin/wransom/ccl_pipeline_phase1` (survives even if weka home isn't mounted).

## The work is on REMOTE git (source of truth — machine-independent)
- **tt-metal:** `origin/wransom/ccl_pipeline_phase1` @ `2f20bdc3643` (branched off `origin/wransom/ccl_help` @ `b7d848979e1`).
  Contains: **W4** (B1 fix — `MeshProgramDescriptor.semaphores` slot + adapter parking + nanobind + single-device cache test, `90b5925f7f5`),
  **W2** (`scripts/run_multidevice_sim_pytest.py` + `scripts/multidevice_sim_topologies.yaml` — runner covers BH p2p + WH all_gather, both green),
  the **pipeline-generated p2p op** + the **green (2,4) confirmation** (`tests/ttnn/unit_tests/operations/point_to_point/test_p2p_confirm_topology.py`),
  and the **pipeline-generated `all_gather` op** (`ttnn/ttnn/operations/all_gather/`, verifier-green on the WH sim, `2f20bdc3643`).
- **pipeline (tt_ops_code_gen):** `origin/wransom/ccl_pipeline_prompts` @ `0e30fb7` (off `wransom/refresh-matmul-helper-references`).
  Contains: **W1** (CCL fabric-dataflow + op-internal-sem prompts + verifier multi-device routing + topology-match guidance) + **dashboard wiring** (`run_eval.py` CCL grading branch via the multichip runner).
- **CCL helper:** `origin/wransom/ccl_help` @ `b7d848979e1` (the typestate `FabricStreamSender` + bound `ccl_packet_dims`/`ccl_dm_route`).
- Design/report docs (weka home, also referenced from memory): `OVERNIGHT_REPORT.md` (session-10 results), `B1_MESH_SEMAPHORE_DESIGN.md`,
  `CCL_GOLDEN_TESTS_DESIGN.md`, `DASHBOARD_WIRING_PLAN.md`, `all_gather_phase2.txt` + `point_to_point_phase1.txt` (op prompts), `SETUP.md`.
- Full state + Phase-2 sizing is in auto-memory `project_ccl_op_gen.md` (the "session 10" entry).

## What's MACHINE-LOCAL (on `/localdev`, needs re-setup on a new box)
craq-sim clone, the staged sims (`sim-bh`, `sim-wh`), the built worktree (`tt-metal-ccl-pipeline` + `build_Release/` + `python_env`).
`sim-wh` was already STALE (rebuild needed for any WH work).

## Fresh-machine re-setup recipe (blackhole; only if /localdev isn't already set up)

### 1. craq-sim + sims (~20-30 min; software, no device, no sudo)
```bash
# Clone + the multichip branch (ebae5463 includes Nachiket's WH all-MMIO fix; or pull latest multichip)
git clone git@github.com:tenstorrent/craq-sim.git /localdev/wransom/craq-sim
git -C /localdev/wransom/craq-sim checkout multichip   # (was at ebae5463)
# Build BH (and WH if doing all_gather). EPYC/Zen2/3 has NO AVX-512 -> -march=x86-64-v3 (v4 SIGILLs).
env -C /localdev/wransom/craq-sim ./make.py --env TTSIM_MARCH=-march=x86-64-v3 --env TTSIM_LTO=0 src/_out/release_bh/libttsim.so
env -C /localdev/wransom/craq-sim ./make.py --env TTSIM_MARCH=-march=x86-64-v3 --env TTSIM_LTO=0 src/_out/release_wh/libttsim.so   # WH (Phase-2 all_gather)
# Stage sim-bh: libttsim.so + soc_descriptor.yaml(=data/bh/blackhole_140_arch.yaml) + tensix_isa.json + tensix_regs.json
mkdir -p /localdev/wransom/sim-bh && cp /localdev/wransom/craq-sim/src/_out/release_bh/libttsim.so /localdev/wransom/sim-bh/
cp /localdev/wransom/craq-sim/data/bh/blackhole_140_arch.yaml /localdev/wransom/sim-bh/soc_descriptor.yaml
cp /localdev/wransom/craq-sim/data/bh/tensix_isa.json /localdev/wransom/craq-sim/data/bh/tensix_regs.json /localdev/wransom/sim-bh/
# Stage sim-wh: soc_descriptor.yaml = tt-metal's wormhole_b0_80_arch.yaml; tensix jsons from craq-sim data/wh
# (see SETUP.md / the memory for the exact WH staging; the WH all-MMIO descriptors are craq-sim data/wh/wormhole_t3k_all_mmio.yaml + ..._mesh_graph_descriptor.textproto)
```
The runner's `scripts/multidevice_sim_topologies.yaml` resolves these via env (`TT_SIM_BH_SO`/`TT_SIM_WH_SO`/`CRAQSIM_DATA`)
with defaults at the paths above — so once staged, `run_multidevice_sim_pytest.py --list` shows `paths = OK`.

### 2. Worktree + build (~30-45 min build)
```bash
# From a tt-metal checkout (clone if none): add a worktree on the phase1 branch
git fetch origin wransom/ccl_pipeline_phase1
git worktree add /localdev/wransom/tt-metal-ccl-pipeline origin/wransom/ccl_pipeline_phase1   # detached, or -b a fresh throwaway
cd /localdev/wransom/tt-metal-ccl-pipeline
git submodule update --init --recursive          # heavy (umd etc.)
./build_metal.sh -c                              # W4 is C++ -> needs a build; ccache helps
./create_venv.sh                                 # python_env
# (if a nested llama submodule shows dirty after recursive init: cd into it, git reset --hard + git clean -fd)
```

### 3. Pipeline wiring (for run_op.py regeneration)
```bash
WT=/localdev/wransom/tt-metal-ccl-pipeline
mkdir -p "$WT/tt_metal/third_party/tt_ops_code_gen"
git clone git@github.com:tenstorrent/tt_ops_code_gen.git "$WT/tt_metal/third_party/tt_ops_code_gen"
git -C "$WT/tt_metal/third_party/tt_ops_code_gen" checkout wransom/ccl_pipeline_prompts
ln -sfn tt_metal/third_party/tt_ops_code_gen "$WT/.claude"
ln -sfn tt_metal/third_party/tt_ops_code_gen/eval "$WT/eval"
ln -sfn tt_metal/third_party/tt_ops_code_gen/QUICK_START.md "$WT/QUICK_START.md"
ln -sfn tt_metal/third_party/tt_ops_code_gen/pipeline-improvements.md "$WT/pipeline-improvements.md"
# Hide the wiring from git so auto-commits never grab it (worktree gitdir is under the main repo's .git/worktrees/<name>/info/exclude):
GITDIR=$(cat "$WT/.git" | sed 's/gitdir: //'); printf '%s\n' '/.claude' '/eval' '/QUICK_START.md' '/pipeline-improvements.md' '/tt_metal/third_party/tt_ops_code_gen/' >> "$GITDIR/info/exclude"
```

## Gotchas (carry these forward)
- **Baseline gate:** fires on the SESSION cwd matching `*/tt-metal` or `$TT_METAL_HOME`. The worktree path
  `.../tt-metal-ccl-pipeline` does NOT match, so the nested pipeline agents (cwd = worktree) are GATE-FREE.
  For my own direct `pytest tests/...` calls, prepend `# CONFIRMED:` once the baseline is established, or
  `clear_pending.sh`. The multi-device RUNNER invocation (`run_multidevice_sim_pytest.py`) never trips the gate.
- **sim build:** `-march=x86-64-v3` on EPYC/Zen (no AVX-512).
- **Detach long jobs:** `run_op.py`/builds via `setsid bash -c "...; echo EXIT=$? > <sentinel>" &` + poll the sentinel,
  so a disconnect doesn't kill them (harness-tracked background jobs DIE on disconnect).
- **Test↔topology coupling (the Phase-1 finding):** a generated CCL op's acceptance test MUST open a `mesh_device`
  shape matching the runner topology's `mesh_shape` (see `--list`) + `fabric_config`, else fabric init HANGS with
  `Fabric Router Sync: Timeout` — a test/topology mismatch, NOT a sim or op defect.
- run_op.py auto-commits + auto-pushes to the WORKTREE branch (the throwaway phase1 branch — safe). Generate ONLY on a throwaway.

## Phase-2 menu (pick on resume; with a full session, likely #1 or #2 + #3)
1. **WH `all_gather` via the runner** — 2nd op + 2nd arch (rebuild sim-wh, align the WH topology descriptor + the test's mesh shape, verify green). Headline advance.
2. **Dashboard wiring** (`run_eval`↔multichip) — wire run_eval.py to drive the multichip runner so a real multichip run ingests to the dashboard (today run_eval is single-chip only).
3. **Close the p2p verifier gap** — re-run the verifier with a topology-matched test so the pipeline's OWN verifier reports green (quick; the op is already proven green).
4. **Full matrix + RING** (`FABRIC_1D_RING`, full dtype/shape) — needs a RING-capable descriptor.
5. **CCL golden suite** — build the `CCL_GOLDEN_TESTS_DESIGN.md` design (largest; mesh_device fixture + shard→collective→gather oracle).
