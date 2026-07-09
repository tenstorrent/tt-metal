# LTX 3s loop — ltx-rt base, MEDIUM/720p (project knowledge pack)

**North star:** drive the LTX-2.3 distilled AV e2e (704×1280 "medium"/720p, 145f, +audio,
BH galaxy 4×8) to **E2E ≤ 3s**, without shipping a different-scene generation.

## Base + harness
- Worktree `/home/smarton/tt-metal/.claude/worktrees/ltxperf-tip`, branch
  `smarton/optimizer/ltx-3s-rt-2026-07-07` (= origin/ltx-rt superset, built rc=0). ltx-rt already
  has quant presets, prewarm, the distilled test — do NOT re-port; use them.
- **Config = `LTX_QUALITY=medium`** = `all_bf8_lofi` quant + S1 6-step (FAST_S1_SIGMAS) + **S2 1-step**
  (`0.909375,0.0`). Resolution `HEIGHT=704 WIDTH=1280`, `NUM_FRAMES=145`, `LTX_TRACED=1`.
- **Run discipline (ON-HOST + broker, NOT ssh/devrun):** device jobs go through the tt-device-mcp
  broker (owner `[claude]smarton`, workspace this worktree). Canonical runner
  `models/tt_dit/tests/models/ltx/run_ltx.sh` (prewarm-off-device + broker) with
  `ENV_YAML=tmp/ltxrt_env.yaml MESH=bh_4x8sp1tp0_ring`, OR a direct `tt_device_job_run_bg` with the
  env in `tmp/ltxrt_env.yaml`. **GUARD every run with `tt-workflows/scripts/broker_watch.sh <job>`**
  (early wedge-abort — a wedge once burned 2700s). NEVER SIGKILL a device-holding proc; route kills
  through the broker.
- Metric `ms_view` = gen#1 (traced steady-state) E2E compute total. gen#0 (cold, trace capture) is
  never the metric. Report the full stage table (S1/upsample/S2/VAE/audio/Total).
- **PATIENCE (lesson, cost a wasted run):** gen#0 trace capture is SILENT for 10–20 min — the broker log
  stops growing while the device captures. Do NOT kill a run on "no log growth"; only kill on a real
  wedge marker (broker_watch handles those) or a total-runtime timeout > ~30 min with no gen#1. The
  `Index 3/6 is larger than runtime args size` storm during warmup/capture is a BENIGN red herring (runs
  complete through it) — never treat it as a hang. Cold first-run on a fresh cache ≈ 15–20 min; warm ≈ 10–12.

## Quality gate (metric_name = qgate)
Decoded-frame **mean-PCC vs a same-config high-fidelity reference** via `~/compare_videos.py`
(the worktree `python_env` has cv2/skimage). Build the reference ONCE: `LTX_QUALITY=medium` at HiFi
(medium sigmas but SDPA/linears bf16/HiFi2 — or the shipped `high` tier at 720p) → `tmp/ref_medium720.mp4`.
A perf win that drops qgate < 0.85 is a REJECT.

## THE BOTTLENECK (source+measured this session — read before picking ideas)
The DiT denoise is **SDPA-COMPUTE-bound, NOT communication-bound.** Evidence: (1) the SP K/V
all-gather fabric critical path is `max(fwd,bwd)=4` hops, ~1ms of the denoise, hidden under quadratic
SDPA compute; (2) bf8 collectives were perf-FLAT (rules out bandwidth); (3) SP4×TP8 (fewer SP hops)
was **+1.5s WORSE** (halving SP doubles per-device SDPA compute); (4) the P0 `LTX_KV_WINDOW` −0.30s
was a COMPUTE cut (fewer attended-KV), not fabric. **So optimize COMPUTE, not CCL.** At 720p the
sequence is ~0.43× the 1080p N, so SDPA compute is ~0.19× — 3s is reachable here.

## Idea backlog (rank by removing SDPA/matmul compute or host/dispatch bubbles; profile first)
1. **SDPA LoFi fidelity** — `LTX_QUANT=all_bf8_lofi_sdpa_lofi_fp32acc` (ring SDPA HiFi2→LoFi, fp32
   acc). Halves the SDPA matmul phases if S2 is attention-compute-bound. Config-only. qgate.
2. **Tracy/nsight profile** (tt-profile) of gen#1 S2 — split SDPA (QK^T/softmax/AV) vs FFN vs the
   TP-fused matmuls vs host bubbles. Pick the true dominant op; don't grind a non-bottleneck.
3. **Per-block L4 temporal windowing as a COMPUTE cut** — apply the ring-iters/window clamp only to
   depth-map-tolerant blocks (`opt/l4/depth_map*`), cutting attended-KV compute where quality holds.
4. **SDPA chunk retune** — `ring_sdpa_chunk_by_n` / `sdpa_chunk_by_shape` in attention_ltx.py for the
   720p shapes (add `(True,8,4,<720p N>)` entries); q/k chunk sizing for the medium sequence.
5. **Host/dispatch bubbles** — D2H→host→H2D between stages, per-step Euler on host; on-device solver
   (WAN pattern, `gather_output=False`) to keep the latent on-device across steps.
6. **L1 sharding (next tier, user-directed)** — keep activations/KV in L1 across ops to cut DRAM
   round-trips; research other tt_dit models (WAN, mochi) + glean + web; plan; implement; optimize.
7. VAE decode + audio — measure at 720p; may be a meaningful E2E fraction once denoise shrinks.

## Milestones
- M0: baseline `LTX_QUALITY=medium` 720p = ? s @ qgate (measuring now, job 233113-75).
- M1: SDPA-LoFi + chunk retune → first compute win.
- M2: profile-guided top-op attack (SDPA or FFN).
- M3: L1 sharding.
- M4: honest report of the residual gap to 3s.
