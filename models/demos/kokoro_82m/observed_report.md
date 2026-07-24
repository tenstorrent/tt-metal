# Kokoro-82M bring-up — observed report (tt_hw_planner)

Living log of what the `tt_hw_planner` tool actually did during the
`hexgrad/Kokoro-82M` bring-up, plus tool-behavior observations and
enhancement opportunities. We own this branch and are enhancing the tool,
so this doubles as a tool-QA / feature-gap record.

- Session: `python -m scripts.tt_hw_planner auto-up hexgrad/Kokoro-82M --box QB2 --mesh 1,1`
- Locked defaults: `--auto --auto-agent=claude --auto-model-tiered --auto-max-iters=24 --auto-max-attempts-per-component=5 --isolation=worktree`
- Hardware: QB2 = 4× Blackhole p150c (128 GB). Run pinned to **mesh [1,1] = single P150**; other 3 chips idle. Verified real device: `/dev/tenstorrent/0–3`, `ttnn.open_device()` opens/closes cleanly.
- Isolation worktree: `/tmp/tt_hw_planner_hexgrad_Kokoro-82M_1783376840` (detached HEAD off `tvardhineni/models_bringup`).

---

## Timeline — what the tool did

1. **Pre-flight / plan.** Categorized Kokoro as TTS (StyleTTS2 + ISTFTNet). Memory plan: per-chip 1.1 GB, headroom 27.5 GB. Selected mesh [1,1].
2. **HF load FAILED (expected).** `AutoModel*` all raise `Unrecognized model ... should have a model_type key in config.json` (Kokoro ships no `model_type`). Tool tried 7 auto-classes (CausalLM, SpeechSeq2Seq, TextToWaveform, TextToSpectrogram, ImageTextToText, ImageClassification, AutoModel) — all failed.
3. **Reference loader (model-local) recovered it.** `models/demos/kokoro_82m/tests/pcc/_reference_loader.py` (added by us) stubs `misaki` and constructs `kokoro.KModel` (a plain `nn.Module`) on CPU. Component discovery then walks the real module tree.
4. **Scaffold.** 25 components discovered (1 REUSE + 24 NEW); 52 TTNN stubs generated under `_stubs/`; per-component PCC test per component under `tests/pcc/`.
5. **Capture (real IO).** With our learned capture driver (`scripts/tt_hw_planner/learned_drivers/kmodel.py`, drives `KModel.forward_with_tokens`), **21/23 components captured REAL inputs**. 2 uncaptured (STFT path) fall back to synthetic. Fired 414 submodules across all 5 top-level modules (bert, bert_encoder, predictor, text_encoder, decoder).
6. **Graduation gate (preflight).** `instance_norm1d` GRADUATED to native TTNN immediately — this is the **REUSE** component inherited from the sibling `coqui/XTTS-v2` template (genuinely native ttnn ops, but not Kokoro-specific).
7. **Bring-up loop (cc engine).** Tiered models: light=haiku, heavy=sonnet, super_heavy=opus. Round 1 target = `ada_i_n1d` (see issue below).

### Live status (updated)
- **Bring-up CONVERGED: 21/25 on device (84%)** — 20 NEW native (PCC-verified) + 1 REUSE. Loop finished in 1 round (~15 min) after the in-place restart.
- **On CPU (4, PENDING "retry next run"):** `decoder`, `generator`, `sine_gen`, `source_module_hn_n_s_f` — the ISTFTNet vocoder core (iSTFT/HiFi generator + harmonic NSF source). Hardest pieces; next target for hand-fix.
- Loop restarted **in-place** (`up --isolation none`, `--auto-agent-timeout 480`, `--auto-max-attempts-per-component 4`) so a single component can no longer thrash unbounded (the first run spent ~47 min on one op). The norm fix (#1) cascaded to the 3 other norm components + unblocked the ALBERT stack → 1 graduated jumped to 20.

---

## Interventions (manual, verified on-device)

### #1 — `ada_i_n1d` stuck at PCC 0.9878, fixed to 0.99998
- **Symptom:** the cc loop ran ~47 min / 247 tool calls on `ada_i_n1d` and plateaued at a *repeating* `PCC=0.9878` (then regressed to 0.5/0.85/0.93). A repeating ceiling = structural bug, not tunable noise; the LLM kept retrying instead of changing the reduction.
- **Root cause:** InstanceNorm over the time axis `T` where `x=[1,512,25]`. In TILE_LAYOUT, `T=25` pads to 32. The stub did `mean=ttnn.mean(x,dim=2)` then `xc=x-mean`, which writes `-mean` into the 7 padding slots; `ttnn.mean(xc*xc)` then folds that padding into the variance → systematic error → hard PCC ceiling ~0.988.
- **Fix (reusable norm recipe):** compute `Var = E[x²] − E[x]²` from **sums divided by the true length**, so zero-padding never participates in a reduction and there's no `x−mean` padding pollution:
  ```python
  n = int(x.shape[-1])
  inv_n = 1.0 / float(n)
  mean    = ttnn.multiply(ttnn.sum(x, dim=2, keepdim=True), inv_n)
  mean_x2 = ttnn.multiply(ttnn.sum(ttnn.multiply(x, x), dim=2, keepdim=True), inv_n)
  var     = ttnn.subtract(mean_x2, ttnn.multiply(mean, mean))
  norm_x  = ttnn.multiply(ttnn.subtract(x, mean), ttnn.rsqrt(ttnn.add(var, eps)))
  ```
- **Result:** on-device PCC **0.9878 → 0.99998**, PASSED, verified in **~7 s** (vs 47 min of loop thrashing). Backup: `models/demos/kokoro_82m/ada_i_n1d.FIXED.py`.
- **Generalizes to:** `ada_i_n_res_block1`, `adain_res_blk1d`, `ada_layer_norm`, and any norm/reduction over a non-tile-aligned axis.
- **Tool takeaways:** (a) a *repeating* sub-target PCC should trigger a structural-change prompt (or hand-off), not more same-shape retries; (b) the emitter should default norm/reduction stubs to the sum/N `E[x²]−E[x]²` recipe; (c) per-attempt agent timeout is essential — one attempt ran ~47 min unbounded.

---

## Tool mechanisms observed (working as intended)

- **Real device PCC gate.** Each test builds torch reference, runs the ttnn port on the P150, reads the tensor back, asserts `comp_pcc(...) >= 0.99`. Skips are counted as **fail** (loop refuses to converge on a skip).
- **Anti-fakery: AST-native gate.** Even if a torch-passthrough stub trivially passes PCC (`torch ≡ torch`), `_stub_body_is_native` / `target_is_ast_native` refuses to graduate when the forward calls `self.torch_module(...)` / `self._torch_module.forward()` / a `no_grad` fallback. Confirmed in `bringup_loop.py` (`_calls_self_torch_module`, `_calls_fallback`) and `bringup_mcp.py:133`.
- **Snapshot gate.** Graduation requires a `.py.last_good_native` snapshot AND the current body still native (`_stub_has_graduated_from_autofill`) — a stub can't be silently reverted after passing.
- **Compute-split reporting.** Runtime CPU fallbacks are surfaced as `NEW-native (partial runtime CPU fallback)` in the split — partial cheating is visible, not hidden.

---

## Issues & enhancement opportunities

1. **Non-transformers models need a model-local reference loader.** `AutoModel` can't construct Kokoro. We hand-wrote `_reference_loader.py`.
   - *Enhancement:* when `AutoModel*` fails with "Unrecognized model / no model_type", tool should auto-scaffold a reference-loader stub (pip package hint + `sys.modules` stubbing pattern) instead of aborting discovery.

2. **Scaffold clobbered the model-local `_reference_loader.py`.** Sibling-templating copied XTTS-v2's loader over ours (→ `ModuleNotFoundError: No module named 'TTS'` at capture).
   - *Fixed by us:* `scaffold_demo_folder.py` now excludes `_reference_loader.py` (`_EXCLUDE_FILE_NAMES`); `bringup_plan.py` points discovery `demo_dir` at the NEW model's dir.

3. **Generic capture drivers can't drive `KModel`.** `forward(phonemes: str, ref_s)` needs the misaki G2P stack; introspected/vision/text drivers failed → only 6/23 captured real IO.
   - *Fixed by us:* learned driver calling `forward_with_tokens(input_ids, ref_s)` → 21/23 real IO.
   - *Enhancement:* generic driver layer could probe for alternate tensor-native entrypoints (`forward_with_tokens`, `*_with_tokens`, `generate`) before giving up / before falling to synthetic.

4. **2 components still synthetic (STFT path).** `decoder.generator.stft` not fired under `disable_complex=True`; framework's direct-submodule probe passes a 4D `[1,1,64,788]` into a conv1d → `RuntimeError` (expects 2D/3D).
   - *Impact:* those 2 get PCC on random data (weaker guarantee), not real captured IO.
   - *Enhancement:* driver could construct STFT-shaped inputs, or capture could reshape 4D→3D before invoking conv-based submodules.

5. **AdaIN / small-axis instance-norm is slow to converge (current bottleneck).** `ada_i_n1d` input `x=[1,512,25]` (`[B,C,T]`, T=25), `s=[1,128]`. Trivial math, but three TTNN sharp edges:
   - reduce over **non-tile-aligned T=25** (padded to 32 → padding zeros contaminate mean/var unless masked);
   - **per-channel broadcast** of `gamma/beta` `[1,512,1]` over T in `[B,C,T]` layout;
   - **PCC 0.99 precision** — variance over only 25 samples divided by `sqrt(var+eps)` amplifies bf16 error → needs float32 + masked reduction.
   - Each iteration is a full on-device pytest (kernel compile + run) → high wall-clock per attempt.
   - *Generalizes to:* `ada_i_n_res_block1`, `adain_res_blk1d`, `ada_layer_norm`.
   - *Enhancement:* ship a reusable "instance/layer norm over short non-tile-aligned axis" recipe/primitive (transpose so reduce axis is tile-aligned, or masked reduction) as a hint the emitter can reach for, instead of rediscovering it per component per model.

---

## FINAL — end-to-end bring-up SUCCEEDED (verified on P150)
- **e2e test PASSES** (`tests/e2e/test_e2e_tts.py`), independently re-run on device: **log-spectrogram PCC = 0.9933** (gate ≥ 0.95), Gate 2 **20/20 modules invoked** (`missing: []`), `pred_dur` **exact match** to HF.
- **Real audio generated:** `demo/demo_tts.py` → `kokoro_tt.wav` (2.10 s @ 24 kHz, 50,400 samples) for *"kokoro is open source"*. Copied to `models/demos/kokoro_82m/kokoro_tt.wav`.
- **All neural compute on-device:** the 4 previously-CPU vocoder components (`decoder`, `generator`, `source_module_hn_n_s_f`, `sine_gen`) were **re-expressed as an explicit native TTNN chain** in `tt/pipeline.py` (built from graduated primitives — custom_s_t_f_t, ada_i_n_res_block1, upsample, etc.), so they are NOT torch fallback in the e2e path. `host_op_selftest`: zero neural aten ops (14 residual host ops = integer bookkeeping for variable-length TTS: duration rounding, alignment scatter, one-hot).
- Per-stage PCC 0.998–1.0; decode+vocode math **exact (1.0000)** given identical F0/N/asr.
- **Why raw-waveform PCC is low (0.075) and not a bug:** NSF vocoder F0-phase is chaotically sensitive — even HF perturbing its OWN F0 by 1e-6 drops waveform PCC to 0.95. Phase-invariant log-spectrogram PCC (0.9933) is the correct fidelity metric; Gate 3 asserts it.
- emit-e2e's driver agent was killed by the `[cc-watchdog]` at 3600s on a LATER step (trace/2CQ selftest, which legitimately single-CQ-falls-back for data-dependent decode length) — AFTER the gates already passed and the report was written. The e2e deliverable is complete.

## Remaining
- **Full `optimize` phase NOT done** (perf). Trace/2CQ is partial: `encode` stage captured host-free (execute_trace PCC 1.0); `prosody`/`decode`/`vocode` single-CQ fallback (data-dependent `sum(pred_dur)` length). Fusion / memory-config / sharding / perf-vs-target all remain.
- Optional: reduce residual 14 host bookkeeping ops; push variable-length stages onto traced fixed-capacity path.

## Realistic timeline read (retrospective)
- Bring-up + e2e **DONE today**: 20 NEW native + 1 REUSE, full-native e2e pipeline, PCC 0.9933, real audio.
- The norm fix (#1) + in-place restart with guardrails was the unlock (1 → 20 graduated in one round). Optimize is the remaining phase.

---

# OPTIMIZE PHASE — what we hit and how we unblocked it

## Environment blockers (the `optimize` command failed 3× before it ran clean)
Each was a real environment/hardware-config issue, not a model bug. Fixed in order:

1. **`tt-smi` not on PATH** → `BEFORE-LOOP FAILED: FileNotFoundError: 'tt-smi'`.
   - *Fix:* added `/home/ttdeploy/.tenstorrent-venv/bin` to `PATH`.
2. **Tracy profiler tools not found** → `TracyRunError: Tracy tools were not found`.
   - *Cause:* the `optimize` run executes inside a `/tmp/...` git **worktree** whose
     `build/tools/profiler/bin/` was empty; the Tracy binaries only exist in the real
     repo build. The profiler resolves them under `$TT_METAL_HOME/build/...` (default
     `cwd`), so it looked in the worktree and found nothing → wrongly assumed a wheel install.
   - *Fix:* symlinked `tracy-capture`, `tracy-capture-daemon`, `tracy-csvexport` from the
     real repo build into the worktree.
3. **`CUSTOM cluster type` fatal** → `TT_FATAL: Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`.
   - *Cause:* the box is a **P300** (2-chip board). The cluster classifier only recognizes
     P300 with 2 or 4 chips; pinning to a single chip (`--devices 0` → `TT_VISIBLE_DEVICES=0`)
     leaves 1 visible chip on a P300 → falls into the `CUSTOM` branch → fabric demands a
     mesh-graph descriptor. (This is why the earlier bring-up/e2e runs, which saw all chips
     → recognized `P300_X2`, never hit it.)
   - *Fix:* `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`
     (a clean 1×1 Blackhole topology). Verified: perf test passes in ~62 s on one chip.
   - *Enhancement:* the tool/profiler-heal should (a) set `TT_METAL_HOME` to the real repo
     build when run from a worktree, and (b) auto-select the single-chip mesh-graph
     descriptor when a known board is under-populated, instead of a hard `TT_FATAL`.

## Real device baseline (first true on-device profile) — 621.85 ms
- eltwise 237.8 ms (38.2%) · datamove 221.9 ms (35.7%, incl. **layout-churn 3139× = 109 ms**)
  · matmul 152.9 ms (24.6%) · reduction 9.1 ms · **host_overhead 138.7 ms (22.3%)**.
- Roofline achievable floor ≈ 278.6 ms (~343 ms headroom). Full-model e2e = 2307 ms.

## Auto-tuner made 0 changes — two hard gates
1. **`tt-lang` unavailable on this env.** Lever ladder is `grid → dtype → tt-lang → cpp → host`;
   the top op reached the `tt-lang` rung and the engine **halts** without it. Every `tt-lang`
   wheel is **cp312-only**, but `python_env` is **CPython 3.10** → not installable.
   - *Enhancement:* when tt-lang is known-unavailable, the engine should **soft-skip** the
     tt-lang rung (fall through to cpp/host levers) instead of hard-halting the whole run.
2. **Not fully on-device → trace + 2CQ blocked** by residual host round-trips.

## Manual host-op elimination + trace attempt (real results)
- **Alignment moved fully on-device.** The duration→frame expansion (host `round`/`clamp`/`sum`
  + a Python per-token loop of `zeros`/`scatter_` + upload) is now an on-device `cumsum` +
  a resident column-index buffer compared with `ge`/`lt`. Verified **bit-identical** against
  the trusted gate `test_e2e_tts.py`: `pred_dur` exact, **log-spectrogram PCC 0.9933**,
  20/20 modules. Host op *types* 14 → 10; per-token host loop gone. (`pipeline.ONDEVICE_ALIGN.py`.)
  - *Gotcha we hit:* a first ad-hoc verification harness read PCC 0.27 — that was a **harness
    bug** (it built the pipeline from a separate, non-determinized model), NOT a pipeline
    regression. The real gate never dropped. Lesson: validate against the committed gate, not
    a bespoke harness.
- **trace + 2CQ is architecturally blocked (verified, not a tooling gap).** The prosody LSTMs
  (shared + all 3 duration-encoder) are **bidirectional**; a fixed-capacity padded trace makes
  the backward pass start in the zero-padded tail and corrupt **every** real frame. Measured:
  naive padding drops log-spectrogram PCC **0.9933 → 0.28**. The tool's single-CQ fallback was
  correct. Unlocking it needs **masked bidirectional recurrence** + bucketed trace lengths.
- **Remaining 10 host ops are not alignment:** embedding **one-hot gather**
  (`albert_embeddings`/`text_encoder`: `zeros`+`scatter_`+`round`) and **NSF source masks**
  (`_build_source`: `ones`/`repeat`+index-assign). Movable on-device, but does not unlock
  trace/2CQ (still bi-LSTM-bound).
