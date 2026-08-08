# Kokoro-82M — Optimization Report (single P150 / 1x1 mesh)

Date: 2026-07-07 · Board: P300 (Blackhole), `TT_VISIBLE_DEVICES=0` (single chip) · engine `cc` · metric `device_ms`

## TL;DR
The `optimize` phase now runs cleanly end-to-end and produced the **first real on-device
Tracy profile** of the pipeline. The **auto-tuner applied 0 kernel changes** because it is
blocked by two hard gates on this machine (details below). The baseline profile itself is a
genuine, reproducible deliverable and identifies exactly where the time goes.

## 2026-07-08 · Session 3 — QB2 (sjc2-t3020): full loop ran on HW, discovery bug fixed, NO net win
Moved the whole flow to the **QB2** box (`sjc2-t3020`, 4× P150 Blackhole, **249 GB RAM**) to get past
the 61 GB desktop OOM. Fresh HTTPS clone of the fork into `/home/ttuser/work/tt-metal`, branch
`tvardhineni/models_bringup` @ `21fb1d6`. Single-chip: `TT_VISIBLE_DEVICES=0` +
`p150_mesh_graph_descriptor.textproto`. (NOTE: the optimizer's model-dir revert-to-HEAD will wipe an
*uncommitted* edit to this report mid-run — write session notes only after the run, or commit them.)

**Environment / unblocks**
- **RAM OOM gone** (249 GB); the run that died on "Importing ops logs" at 61 GB now profiles cleanly.
  Chunked-parser fix **not** needed.
- **`create_venv.sh` now yields CPython 3.12.12** (was 3.10). Final `uv pip check` fails only on an
  unrelated `fiftyone`↔`sse-starlette` pin (claude-agent-sdk bump) — cosmetic; re-ran with
  `--skip-compat-check --force`.
- **tt-lang blocker RESOLVED:** on 3.12, `ensure_tt_lang` reports **`tt-lang available (1.0.1)`**
  (cp312 wheels), so the kernel rung is finally live.
- **Sanity passed** on device 0: e2e waveform PCC **0.9198**, log-spectrogram **0.9933** (1 passed
  ~67 s); `demo_tts` → `kokoro_tt.wav` (WAVE 16-bit mono 24 kHz).

**Discovery bug (Session-2's open TODO) — ROOT-CAUSED + FIXED**
- The `discovery failed (before_loop produced no manifest)` error was **not** a manifest-handoff bug.
  `before_loop` finishes all work, writes a **valid** `manifest.json`, sets `state=BEFORE_LOOP_DONE`,
  and `main()` returns 0. The process then **segfaults in `_ttnn.so` during interpreter teardown**
  (kernel log: `python[…]: segfault at 0 … in _ttnn.so`, null-deref at finalization) → child exits
  with a signal → `cc_optimize/run.py::discover()` saw `rc != 0` and **discarded the good manifest**.
- **Fix (uncommitted, in `cc_optimize/run.py`):** `discover()` now accepts a **fresh** manifest whose
  sibling `state.json` is `BEFORE_LOOP_DONE` even when `rc != 0` (teardown-only crash); added
  `_run_state()` + manifest-mtime freshness check. Only a genuinely incomplete discovery fails now.
  (A plain `ttnn.open/close` exits 0, so the crash is specific to the discovery process teardown.)

**Loop ran for the first time on Kokoro — outcome: NO net device win (stopped manually mid round 1, ~48 min)**
- `device_ms` **89.25 → ~89.26** (unchanged) when stopped. Baseline buckets on this branch are much
  lower than Session 1/2 (datamove 41.2 / eltwise 29.9 / matmul 17.5 / host 16.2; layout-churn now
  **596× = 20.3 ms**, was 3139× = 109 ms). Roofline floor ≈ 42 ms. Full-model e2e (BEFORE) = 2345 ms.
- **Biggest gap `TilizeWithValPadding` (~14.7 ms) + `UntilizeWithUnpadding` (~4.1 ms) cleared as
  `structural`/irreducible** — the per-op knob+kernel ladder cannot touch them. This is the same
  layout-churn; it needs a source-level layout fix, not a lever.
- **tt-lang rung actually exercised (the payoff of 3.12):** the agent authored a real fused LSTM-cell
  tt-lang kernel (`tt/lstm_cell_ttl.py`, `@ttl.operation grid=auto`, compute + 2 datamovement kernels)
  collapsing the LSTM's 5-op elementwise sequence, validated standalone on-device (max abs err
  ~0.0016), wired it in, and **passed PCC = 0.9910**. But **`measure_candidate` = 89.26 ms →
  `beat_baseline: false`** → reverted. Reason: that eltwise op is *dispatch/count-bound* (thousands
  of tiny ops), so fusing one cell doesn't reduce net device time.
- Session totals: 4 commits / 3 reverts (all PCC-gated), 2 knobs distilled; all reverted at stop.

**Conclusion**
- The SELECT→APPLY→gate→measure loop now runs cleanly on real HW (incl. the first working tt-lang
  kernel + PCC pass), but the **auto-tuner cannot meaningfully speed up Kokoro**: cost is dominated by
  **structural layout churn** (tilize/untilize, ruled irreducible) and **many small dispatch-bound
  eltwise/host ops**. The real wins remain the **source-level, tt-lang-independent** targets already
  documented below (kill layout churn; collapse the remaining host ops). This run's value is the
  *proof on hardware* that the per-op ladder is exhausted here.
- **Workspace left clean:** model source reverted to HEAD `21fb1d6ab` (agent's `lstm_cell_ttl.py`
  removed). Uncommitted only: `cc_optimize/run.py` `discover()` fix, the one-time
  `tt_metal/impl/profiler/profiler.cpp` heal (+ `.perfauto_bak`), and `kokoro_tt.wav`.
- Run log `/tmp/optimize.log`; artifacts under `models/experimental/perf_automation/runs/2026-07-08T00-26-11/`.

## 2026-07-08 · Session 3b — masked bidirectional LSTM (TRACE ENABLER) — PROTOTYPED + VALIDATED
The long-standing trace blocker was that a fixed-capacity padded bidirectional LSTM lets the reverse
pass start in the padded tail and pollute every real frame (PCC 0.9933 -> ~0.28). Searched the whole
repo: **no other model has trace-compatible bidirectional/masked recurrence** (transfuser GRU is
host-only; mamba `prefix_scan` is forward-only; gemma4 has trace-safe single-step buffer reset;
informer has an on-device seq mask for attention). So we prototyped it in `_stubs/l_s_t_m.py`.

**Approach (opt-in via env; default path byte-identical):** run the SAME `C` unrolled cells every call
(trace-shaped) and gate each `h`/`c` update with a per-timestep validity mask `m_t = (t < T_valid)`,
so padded frames are a state no-op. A reverse pass then keeps the initial zero state until the true
last frame `T_valid-1`, reproducing the dynamic pass. Env knobs (all default OFF):
`KOKORO_LSTM_TRACE_CAP`/`KOKORO_LSTM_TRACE_PAD` (force fixed capacity), `KOKORO_LSTM_TRACE_BIASPAD`
(fill padded gates with the linear bias = emulates padding `x` pre-projection), `KOKORO_LSTM_TRACE_NOMASK`
(debug: disable mask).

**Validated on the authoritative `test_e2e_tts` gate (4 runs):**

| # | scenario | e2e PCC | log-spec PCC | result |
|---|---|---:|---:|---|
| 1 | default (dynamic, no pad) | 0.9198 | 0.9933 | PASS — refactor safe |
| 2 | masked, zero-pad +32 | 0.9198 | 0.9933 | PASS |
| 3 | bias-pad +32, **NO mask** (control) | **0.0839** | 0.9256 | **FAIL** — reproduces the collapse |
| 4 | bias-pad +32, **WITH mask** | 0.9198 | 0.9933 | **PASS** (bit-identical to dynamic) |

Run #3 is the documented failure (polluted backward pass: `pred_dur` mismatch, waveform PCC 0.08); run
#4 proves the mask is the mechanism that fixes it. Two findings: (a) zero-padding the *projected* gate
input is itself benign (a zero-input cell from zero state is a fixed point), so run #2 holds even
unmasked; (b) the mask is REQUIRED once any nonzero contribution reaches padded gates (the realistic
pre-projection / bias-leak case, run #3->#4). The mask covers both, so it's the robust choice.

**Trace + 2CQ ladder (correctness of the recurrence is done):**
- **[DONE] Step 1 — device-resident mask.** Replaced the host scalar `m = 1.0 if t < T_valid` with a
  pure-ttnn `ttnn.lt(iota, T_valid)` device mask (`KOKORO_LSTM_TRACE_DEVMASK=1`), so the per-step gate
  has NO host control flow → the op sequence is identical for any length (trace-shaped). Validated on
  `test_e2e_tts`: DEVMASK+zero-pad+32 and DEVMASK+bias-pad+32 both hold e2e 0.9198 / log-spec 0.9933
  PASS (bit-identical to host-scalar mask). Trace version keeps `iota` resident and writes `T_valid`
  into a resident buffer OUTSIDE the trace, leaving the `ttnn.lt` as the only per-capture op.
- **[DONE] Step 1b — shared masked-scan primitive.** Discovered the bi-LSTM cell-unroll was
  **triplicated** (`l_s_t_m` on the frame axis; inline copies inside `duration_encoder` and
  `text_encoder` on the token axis). Extracted ONE primitive `_stubs/_lstm_scan.py::run_bilstm`
  (dynamic + masked paths, host/device mask, env knobs) and refactored all three stubs onto it.
  Validated: `test_l_s_t_m` PASS; `duration_encoder`/`text_encoder` per-stub PCC harness SKIPs
  (synthetic-input incompatible — the test itself says "validate via the top-level demo"), so the
  authoritative gate is e2e. Default e2e byte-identical (0.9198 / 0.9933 PASS); and with ALL THREE
  LSTMs forced onto the masked device path (`PAD=32 DEVMASK=1`) e2e still 0.9198 / 0.9933 PASS — the
  masked primitive is correct at both the token and frame axes simultaneously.
- **[TODO] Step 2 — bucketed capacities** `C` (power-of-2, token axis ≤512 / frame axis ≤2048), one
  trace per bucket, mirroring `tt_transformers/tt/common.py`.
- **[DONE] Step 3 — `pipeline.py::prosody_trace_setup/step` wired + captured.** setup pads the token
  axis to a bucketed `C = next_pow2(T)` in `[32, max_position_embeddings]` (folds in Step 2) and builds
  the resident mask; step runs `prosody_fwd` (duration_encoder + LSTM + duration_proj) pure-ttnn with
  the trace ctx pushed. `trace_capture_selftest` (device `trace_region_size=1<<28`) now reports
  **`prosody: captured host-free, execute_trace PCC=1.00000 OK`** (was single-CQ fallback), alongside
  the pre-existing `encode`. Default e2e unchanged. Two trace-API rules baked into the primitive: (1) NO
  tensor creation inside a trace (host writes → fatal) — zero-state is a resident cache, mask built in
  setup; (2) `open_device` needs a large `trace_region_size`.
- **[TODO] Step 4 — capacity-pin the duration→frame alignment** (the other data-dependent axis,
  `sum(pred_dur)`): extend the resident `_align_col` buffer (pipeline.py:390) to fixed frame capacity
  + validity mask instead of slicing to dynamic `T`.
- **[TODO] Step 5 — 2nd command queue:** CQ1 stages next-utterance inputs (`*_write_inputs` hooks
  already stubbed) while CQ0 runs `execute_trace`; replicate for decode/vocode.
- **[TODO] Step 6 — verify + measure:** `trace_capture_selftest` + emit-e2e trace+2CQ gate; confirm
  prosody captures host-free at PCC ≥ 0.99 and measure the e2e win (host/dispatch-bound cost is exactly
  what trace+2CQ targets — the structural lever the per-op auto-tuner could not pull).
All changes confined to `_stubs/l_s_t_m.py` so far (opt-in); default gate unchanged (run #1).

## 2026-07-07 · Session 2 — auto-optimizer run from the persistent repo (PAUSED, resume tomorrow)
Re-ran the `optimize` tool, this time correctly from the **persistent main repo**
(`/home/ttdeploy/teja/tt-metal-tv`), not the ephemeral `/tmp` worktree. Key outcomes:

- **Auth: solved with zero tool changes.** The engine runs on the user's `claude` CLI **login**
  (`~/.claude/.credentials.json`) — `no ANTHROPIC_API_KEY in env — using claude login credentials`.
  No API key and no code change needed. (A LiteLLM `.env.agent` was briefly trialed then fully
  reverted at the user's request — `config.py` is back at HEAD, `.env.agent` deleted.)
- **Ran in-place from `teja/tt-metal-tv`** (emitted demo → in-place). `startup_reset` was a **no-op**
  (`clean (fbe239993)`): it only restores *uncommitted* model-dir changes to HEAD (scoped
  `git checkout HEAD -- <model dir>`, never `reset --hard`, never touches other files). Everything
  was already committed, so nothing was disturbed.
- **Profiler self-heal (one-time rebuild).** The stock tt-metal device profiler crashes on mesh
  captures via "orphan markers" (unpaired Tracy zone start/stop). The tool patched
  `tt_metal/impl/profiler/profiler.cpp` and **recompiled `libtt_metal`** (~2–3 min; C++ fix must be
  compiled in) so baseline capture doesn't crash. **This edited a TRACKED source file** — left in
  place, uncommitted, with original saved at `profiler.cpp.perfauto_bak` (revert:
  `git checkout -- tt_metal/impl/profiler/profiler.cpp`). One-time; future runs skip it.
- **Reproducible baseline captured: device 621.748 ms** — matches Session-1's 621.85 ms and the
  identical bucket split (eltwise 237.8 / datamove 221.9 incl. 109.3 ms layout-churn / matmul 152.8
  / reduction 9.1 / host_overhead 139.8). Roofline floor 278.97 ms → Σgap 342.78 ms. (NB: this is
  the *capped* perf test — `TT_PERF_LAYERS=2` + a tiny phoneme string — so host_overhead here is the
  truncated-forward figure, not the full-utterance number.)
- **Then it FAILED before the lever loop:** `discovery failed (before_loop produced no manifest)` →
  `run failed`. So **0 optimization attempts happened** — this is a tooling/manifest bug, not a
  hardware or auth problem. **TODO tomorrow: debug why `before_loop` emitted no discovery manifest**
  (the baseline + discovery evidence were produced; the manifest handoff is what broke), then let the
  SELECT→APPLY→PCC-gate→re-measure loop actually run against the layout-churn + grid/fidelity buckets.
- **Resume command (persistent repo, claude login):**
  ```bash
  cd /home/ttdeploy/teja/tt-metal-tv
  export PATH="$PATH:/home/ttdeploy/.tenstorrent-venv/bin"
  export TT_MESH_GRAPH_DESC_PATH=/home/ttdeploy/teja/tt-metal-tv/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
  ./python_env/bin/python -m scripts.tt_hw_planner optimize hexgrad/Kokoro-82M \
    --pcc-test models/demos/kokoro_82m/tests/e2e/test_e2e_tts.py::test_e2e_tts \
    --engine cc --in-place --max-rounds 2 --metric device_ms --devices 0
  # run log: /home/ttdeploy/kokoro_optimize_run.log ; run artifacts under models/experimental/perf_automation/runs/
  ```

## Real baseline (Tracy, on device)
- **Total device time: 621.85 ms** (median of 1; wall incl. compile/setup = 565.6 s).
- **Full-model end-to-end: 2307 ms** (all 52 layers, prefill + 1 decode; `eager_wall`).
- Roofline achievable floor ≈ **278.6 ms** → ~**343 ms** of theoretical headroom.

| bucket | ms | % | notes |
|---|---:|---:|---|
| eltwise | 237.8 | 38.2% | hifi4, partial grid, dram_interleaved |
| datamove | 221.9 | 35.7% | **layout-churn 3139× = 109.3 ms** (TILE↔RM conversions) |
| matmul | 152.9 | 24.6% | hifi4, partial grid |
| reduction | 9.1 | 1.5% | |
| other | 0.08 | 0.0% | |
| host_overhead | 138.7 | 22.3% | host fallback in decode (`source=op_gap`) |

Biggest concrete, tt-lang-independent targets:
1. **Layout churn — 109 ms** in datamove from 3139 TILE↔ROW_MAJOR conversions.
2. **Host overhead — 139 ms** decode-region host round-trips (`from_device`/`from_torch`/`to_torch`).
3. eltwise/matmul on **partial grids at hifi4** — grid + fidelity/dtype levers.

## Why the auto-tuner made 0 changes (2 hard blockers)
1. **`tt-lang` lever unavailable on this env.** The lever ladder is
   `grid → dtype → tt-lang → cpp → host`. The top op's next rung was `tt-lang`, and the engine
   **halts** when a material op needs a tt-lang kernel that isn't installed. `tt-lang` ships
   **cp312-only wheels** (all versions 1.0.1–1.1.5), but `python_env` is **CPython 3.10** →
   not installable without a 3.12 rebuild of ttnn. So this rung is off the table here.
2. **Not fully on-device → trace + 2CQ blocked.** Scorecard: `fully on device: NO` due to host
   round-trips (`ttnn.from_device`, `ttnn.from_torch`, `ttnn.to_torch`). Trace/2CQ (and thus
   TTFT / T/s numbers) can't be enabled until the 14 variable-length bookkeeping host ops are
   moved to a fixed-capacity on-device path.

## Environment fixes applied to get here (so re-runs profile cleanly)
The optimize phase failed 3× on environment issues before this clean run; all now resolved:
1. `tt-smi` not on PATH → added `/home/ttdeploy/.tenstorrent-venv/bin`.
2. **Tracy tools not found** → the `/tmp/...` worktree's `build/tools/profiler/bin/` was empty;
   symlinked `tracy-capture`, `tracy-capture-daemon`, `tracy-csvexport` from the real repo build.
3. **`CUSTOM cluster type` fatal** → P300 board with 1 visible chip isn't a recognized cluster;
   set `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`
   (clean 1×1 Blackhole topology). Verified: perf test passes in ~62 s on one chip.

## What remains (real optimization work, none env-blocked)
These do **not** need tt-lang and are the actual path to a faster model:
- **Kill layout churn (~109 ms):** keep tensors in a consistent layout across the datamove-heavy
  stages; audit the 3139 conversions and remove redundant TILE↔RM round-trips.
- **Eliminate the 14 host ops → fixed-capacity on-device alignment**, which also **unlocks
  trace + 2CQ** for prosody/decode/vocode (encode already traces host-free at PCC 1.0).
- **Grid + fidelity tuning** on the eltwise/matmul buckets (partial grid, hifi4).

## Reproduce
```bash
cd /tmp/tt_hw_planner_hexgrad_Kokoro-82M_1783376840   # isolation worktree
export PATH="$PATH:/home/ttdeploy/.tenstorrent-venv/bin"
export TT_MESH_GRAPH_DESC_PATH=/home/ttdeploy/teja/tt-metal-tv/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
PY=/home/ttdeploy/teja/tt-metal-tv/python_env/bin/python
# baseline perf test (single chip, ~62s):
$PY -m pytest models/demos/kokoro_82m/tests/e2e/test_main_perf.py::test_main_perf -svv
# full optimize sweep:
$PY -m scripts.tt_hw_planner optimize hexgrad/Kokoro-82M \
  --pcc-test models/demos/kokoro_82m/tests/e2e/test_e2e_tts.py::test_e2e_tts \
  --engine cc --in-place --max-rounds 2 --metric device_ms --devices 0
```

## Host-op elimination + trace/2CQ attempt (manual, on device)
Attacked the 14 host ops directly. Findings (all verified on the P150 chip):

- **Alignment is now fully on-device.** The duration→frame expansion (previously a
  host `torch.round`/`clamp`/`sum` + a Python T-loop of `torch.zeros`/`scatter_` +
  upload — 4 op *types* and a per-token loop) is replaced by an on-device
  `cumsum` + a resident column-index buffer compared with `ge`/`lt`. Verified against
  the authoritative gate `test_e2e_tts.py`: **bit-identical** result — `pred_dur`
  exact, waveform PCC 0.9198, **log-spectrogram PCC 0.9933**, 20/20 modules. Host op
  *types* dropped 14 → 10; the per-token host loop is gone. Saved as
  `pipeline.ONDEVICE_ALIGN.py`.
  - **Measured payoff (re-profiled on device, same Tracy path):** device compute
    unchanged at **621.7 ms** (was 621.85 — noise; same neural math), but
    **host_overhead dropped 138.7 ms → 80.6 ms (−58 ms, −42%)**. Bucket breakdown
    otherwise identical (eltwise 237.8 / datamove 221.9 incl. 109 ms layout-churn /
    matmul 152.9 / reduction 9.1). Full-model e2e 2313 ms (unchanged). This is the
    honest win: moving the alignment bookkeeping off the host removed ~58 ms of host
    op-gap without touching correctness.
- **trace + 2CQ is architecturally blocked — confirmed, not a tooling gap.** The
  prosody predictor's LSTMs (shared + all 3 duration-encoder LSTMs) are **bidirectional**.
  A fixed-capacity padded trace makes the backward pass start in the zero-padded tail
  and pollute the hidden state of *every* real frame → global corruption. Measured
  directly: naive fixed-capacity padding drops log-spectrogram PCC **0.9933 → 0.28**.
  This is exactly why the tool kept prosody/decode/vocode as single-CQ. Unlocking
  trace here requires **masked bidirectional recurrence** (reset the backward state at
  the true sequence boundary) in the LSTM stub + bucketed trace lengths — a separate,
  larger project, not a config flag.
- **Remaining 10 host ops** are NOT alignment: they are the embedding **one-hot gather**
  (`albert_embeddings.py`/`text_encoder.py`: `torch.zeros`+`scatter_`+`round`) and the
  **NSF source masks** (`_build_source.forward`: `torch.ones`/`repeat`+index-assign).
  Both can be moved on-device (one-hot via `eq` against a resident class-index buffer;
  masks via resident buffers), which would flip the verdict to `fully on device: True`
  — but it does **not** unlock trace/2CQ (still bi-LSTM-bound), so it is a
  correctness-flag/marginal-host-time win only.

## Artifacts (in `models/demos/kokoro_82m/optimize_artifacts/`)
- `baseline_profile.json` — real per-bucket device profile (device_ms=621.85)
- `iter_baseline_report.csv` — per-op device report (tt-metal format)
- `summary.md` — tool's own optimize summary (note: its "no baseline profile found" line is a
  reporting quirk; the Tracy baseline of 621.85 ms is in `baseline_profile.json` and the run log)
- `manifest.json` — discovery manifest
