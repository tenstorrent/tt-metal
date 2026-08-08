# Kokoro-82M — Manual (source-level) optimizations

Hand-written, source-level changes that the per-op auto-tuner (`tt_hw_planner`) cannot make.
Background: the auto-optimizer run concluded Kokoro is **host/dispatch-bound** — cost is dominated by
structural layout churn and many small host/dispatch ops, not by any single tunable op — so the real
wins are **structural** (kill host round-trips, enable trace + 2CQ). Full profiling context and the
auto-tuner post-mortem live in `optimization_report_kokoro.md`; this file tracks the manual work.

Environment: QB2 `sjc2-t3020`, single P150 (`TT_VISIBLE_DEVICES=0` + `p150_mesh_graph_descriptor`),
CPython 3.12 venv, branch `tvardhineni/models_bringup`. Authoritative gate: `test_e2e_tts`
(e2e waveform PCC + phase-invariant log-spectrogram PCC). All runs launched in a `tmux` session.

---

## Opt #1 — Trace-enabling masked fixed-capacity bidirectional LSTM

### Problem
Kokoro's prosody path has **three** bidirectional LSTMs, each unrolled cell-by-cell over the *dynamic*
sequence length `T`, so none can be `trace`-captured (a trace needs a fixed, length-independent op
sequence). The naive fix — pad to a fixed capacity `C` and run all `C` steps — **collapses accuracy**:
the reverse pass starts in the padded tail and its hidden state pollutes every real frame. Measured:
e2e waveform PCC **0.9198 → 0.0839**, log-spectrogram **0.9933 → 0.9256**, and the duration predictor
(`pred_dur`) diverges from the HF golden. This collapse was the documented blocker keeping the
`prosody`/`decode`/`vocode` stages on the single-CQ (non-traced) fallback while only `encode` traced.

### Fix — masked recurrence
Run the SAME `C` unrolled cells every call, but gate each `h`/`c` update with a per-timestep validity
mask `m_t = (t < T_valid)`:

```
c = m_t · c_candidate + (1 - m_t) · c_prev      # padded frame => carry state unchanged
h = m_t · h_candidate + (1 - m_t) · h_prev
```

A padded frame becomes a **state no-op**, so a reverse pass keeps the initial zero state until it
reaches the true last frame `T_valid-1` — reproducing the dynamic-length result exactly, but with a
capacity-fixed op sequence that CAN be trace-captured.

Two mask sources:
- **host scalar** `m = 1.0 if t < T_valid else 0.0` — fast to validate.
- **device tensor** (trace-safe) `m_t = ttnn.lt(iota, T_valid)` — no per-step host branch, so the op
  sequence is identical for any length. The trace version keeps `iota` resident and writes `T_valid`
  into a resident buffer OUTSIDE the trace, leaving the `ttnn.lt` as the only per-capture op.

Also note: padding the *projected* gate input with **zeros** is itself benign (a zero-input cell from
zero state is a fixed point, so state stays zero through the padded tail). The mask becomes strictly
necessary only once a nonzero contribution reaches padded gates — the realistic case where `x` is
padded *before* the input linear, so the layer **bias** leaks into the padded gates. The mask covers
both, so it is the robust choice.

### Shared primitive (de-duplication)
The cell-unroll was **triplicated**: `_stubs/l_s_t_m.py` (frame axis, `shared_lstm`) plus inline copies
inside `_stubs/duration_encoder.py` and `_stubs/text_encoder.py` (token axis). Extracted a single
source of truth `_stubs/_lstm_scan.py::run_bilstm(device, cc, x, fwd, rev, H)` implementing both the
dynamic and masked paths, and refactored all three stubs to call it. Now the trace-enabling logic
lives in one place and every LSTM site is masked-capable.

### Opt-in (default OFF — zero behavior change unless requested)
All paths are gated by env vars; with none set, the scan runs the original dynamic loop and is
byte-identical to before.

| env var | effect |
|---|---|
| `KOKORO_LSTM_TRACE_CAP=<int>` | pad every scan to this absolute capacity `C` |
| `KOKORO_LSTM_TRACE_PAD=<int>` | pad to `T + N` (test: forces the reverse pass through padded tail) |
| `KOKORO_LSTM_TRACE_DEVMASK=1` | build the mask as a resident DEVICE tensor (`ttnn.lt`) — trace-safe |
| `KOKORO_LSTM_TRACE_BIASPAD=1` | fill padded gates with the layer bias (emulates padding `x` pre-projection) |
| `KOKORO_LSTM_TRACE_NOMASK=1` | debug: disable the mask, to demonstrate the collapse |

### Validation (all on `test_e2e_tts`, single P150 device 0)

| scenario | e2e PCC | log-spec PCC | result |
|---|---:|---:|---|
| default (dynamic, no pad) | 0.9198 | 0.9933 | PASS — refactor safe |
| masked, zero-pad +32 (host scalar) | 0.9198 | 0.9933 | PASS |
| **bias-pad +32, NO mask** (control) | **0.0839** | 0.9256 | **FAIL** — reproduces the collapse |
| bias-pad +32, WITH mask (host scalar) | 0.9198 | 0.9933 | PASS (bit-identical to dynamic) |
| device mask + zero-pad +32 | 0.9198 | 0.9933 | PASS |
| device mask + bias-pad +32 | 0.9198 | 0.9933 | PASS |
| **all 3 LSTMs on device mask** (`PAD=32 DEVMASK=1`) | 0.9198 | 0.9933 | PASS — correct at token AND frame axes |
| per-stub `test_l_s_t_m` | — | — | PASS |

The NO-mask control (row 3) is the decisive proof: with a realistic bias-leaking pad the unmasked
bidirectional pass collapses, and the mask restores the exact dynamic result.

### Files
- `_stubs/_lstm_scan.py` — **new**: shared masked-capable bidirectional-LSTM scan primitive.
- `_stubs/l_s_t_m.py` — refactored onto `run_bilstm` (was: standalone masked prototype).
- `_stubs/duration_encoder.py` — inline LSTM replaced by `run_bilstm`.
- `_stubs/text_encoder.py` — inline LSTM replaced by `run_bilstm`.

### Trace capture PROVEN (host-free `execute_trace`)
Added a fixed-capacity trace mode to the primitive (`build_trace_ctx` / `push_trace_ctx` /
`run_bilstm` TRACE path): the caller pre-pads inputs to `C`, the scan returns a full length-`C`
output, and the validity mask is a resident device tensor built OUTSIDE the capture. A standalone
microtest (random torch bi-LSTM, `T=5`, `C=16`) captures `run_bilstm` with
`ttnn.begin/end_trace_capture` + `execute_trace` and matches the torch reference on the valid region:
**eager (fixed-C, masked) PCC = 0.99999918, `execute_trace` PCC = 0.99999918, RESULT OK.**

Two `ttnn` trace-API constraints learned (both handled in the primitive):
1. **No tensor creation inside a trace.** `ttnn.zeros/ones/from_torch` issue a host→device write, which
   is fatal during capture (`Writes are not supported during trace capture`). The LSTM initial `h`/`c`
   zeros are now a resident cache (`_zero_state`, populated by the eager warm-up that always precedes
   capture) and the mask is built in `build_trace_ctx` (setup, outside capture).
2. **`trace_region_size` must fit the trace buffer.** The device must be opened with a large enough
   `trace_region_size` (this tiny LSTM alone needed ~8 MB); the full prosody stage will need more.

### Prosody stage now traces host-free (wired into the pipeline)
`tt/pipeline.py::prosody_trace_setup/step` are implemented (were: single-CQ fallback that `raise`d):
- **setup** (outside the trace): run `encode`, transpose to `d_en [1,512,T]`, pad the token axis to a
  bucketed capacity `C = next_pow2(T)` clamped to `[32, max_position_embeddings]` (folds in the
  "bucketed capacities" step), and `build_trace_ctx(C, T)` for the resident validity mask.
- **step** (pure ttnn, host-op-free): `prosody_fwd` = duration_encoder + LSTM + duration_proj over the
  resident padded `d_en`, with the trace ctx pushed so every bi-LSTM runs the masked fixed-`C` scan.
- `trace_capture_selftest` now also threads `ref_s` to setup.

Result via `trace_capture_selftest` (device opened with `trace_region_size=1<<28`):
```
[trace] encode:  captured host-free, execute_trace PCC=1.00000 OK
[trace] prosody: captured host-free, execute_trace PCC=1.00000 OK   <- was single-CQ fallback
[trace] decode:  single-CQ fallback (data-dependent alignment length)
[trace] vocode:  single-CQ fallback (data-dependent waveform length)
SELFTEST_ALL_OK: True
```
Default e2e unchanged (the added methods don't touch `run_tts`).

### Status / what this unblocks
This is the enabling primitive for the **trace + 2CQ** work (the structural lever for a host/dispatch-
bound model). Correctness AND host-free trace capture are proven, and the **prosody** stage now
captures host-free in the real pipeline (encode already did). Remaining: `decode`/`vocode` need the
frame axis (`sum(pred_dur)`) capacity-pinned — kill the alignment host round-trip
(`total = int(ttnn.to_torch(...).item())`) and mask the resident `_align_col` to a fixed frame
capacity — then add the 2nd command queue (`*_write_inputs` hooks already stubbed).
Remaining to actually capture the prosody trace (tracked in `optimization_report_kokoro.md`):
1. Kill the host round-trip in the alignment (`total = int(ttnn.to_torch(sum(pred_dur)).item())`,
   `tt/pipeline.py`): replace dynamic `total` with a fixed frame capacity `C_frame` + frame mask on
   the resident `_align_col` buffer.
2. Pick bucketed capacities (token ≤512 / frame ≤2048), one trace per bucket.
3. Wire `prosody_trace_setup/step` (currently `raise`/single-CQ fallback) to the fixed-`C` masked path
   and `ttnn.begin/execute_trace`; then add the 2nd command queue and verify via
   `trace_capture_selftest` + the emit-e2e trace+2CQ gate.

### Trace + 2CQ PROVEN for the token-axis stages (encode + prosody)
The tt_hw_planner trace probe (`scripts/tt_hw_planner/_trace_capture_probe.py`) is the authoritative
gate. Its static check requires every `PIPELINE_STAGES` entry to define
`<stage>_trace_setup/_trace_step/_write_inputs` (all four do), and its device check calls
`trace_capture_selftest()` **with no args** and requires `True`. Two changes close that contract:
- `trace_capture_selftest(device=None)` now opens its OWN device when called arg-less
  (`l1_small_size=24576, trace_region_size=1<<28, num_command_queues=2`) and closes it — so the probe's
  `fn()` runs a real device capture.
- `*_trace_setup` tolerates `inputs=None` (the perf adapter, `agent/perf_adapter.py`, calls
  `<stage>_trace_setup(None)`): a shared `_resolve_inputs` falls back to the standard demo input.
- `prosody_write_inputs` stages the next utterance's `d_en` into the resident buffer on **CQ1**
  (`ttnn.copy_host_to_device_tensor(host_mirror, self._pros_den, cq_id=1)`); presence of this hook is
  what flips `measure_adapter` (`agent/trace_replay.py`) into the trace+2CQ path.

Verified on device (QB2, single P150):
```
NOARG_SELFTEST_OK: True         # gate device probe green
[trace] encode:  captured host-free, execute_trace PCC=1.00000 OK
[trace] prosody: captured host-free, execute_trace PCC=1.00000 OK
[trace] decode:  single-CQ fallback (data-dependent alignment length)
[trace] vocode:  single-CQ fallback (data-dependent waveform length)
PROSODY_2CQ_OK ms=117.956       # write_inputs(cq1)+record_event(1)/wait(0)+execute_trace(cq0) loop
```

### decode / vocode: remaining frame-axis work (data-dependent length — design-sanctioned fallback)
Making these capture (single-CQ at a bucketed frame capacity `Cf = next_pow2(sum(pred_dur))`) is a
larger, multi-function refactor than the token axis, with concrete blockers found in the code:
- **frame-axis F0/N**: `shared_lstm` is bidirectional over frames → must run the masked primitive with a
  frame ctx (`build_trace_ctx(Cf, total)`); alignment naturally zero-fills columns `≥ total`, so no
  extra padding is needed, but the mask is required to stop the reverse pass polluting valid frames.
- **in-forward host const builds** (fatal inside a trace, but fixed-shape at a pinned `Cf`, so
  hoistable to setup): `_build_source.forward` builds `torch.ones`/`ops._const` masks sized by frame
  count (pipeline.py:258–266) and `to_tt(device, F0_curve)` (240); the generator does
  `to_tt(device, x/s)` (299–300); every `build_adain_res_blk` does `to_tt(device, x/s)` (180–181).
- **trace-buffer size**: the generator upsamples ×300 → a `Cf·300`-sample waveform through many conv
  layers; the trace region may need to grow well past `1<<28`, and boundary convs at the padded frames
  are a PCC risk to validate.
The repo design explicitly permits these two stages to stay single-CQ ("degrades to single-CQ … the
fallback is PRINTED, never silent"), and the gate is green with them on that fallback.

### decode / vocode: fixed-Cf Trace+2CQ — SOLVED (masked frame-axis norms), ON by default
The frame-axis capacity-pin for decode+vocode is now correct and enabled by default.
- New `_stubs/_trace_alloc.py`: a content-keyed prealloc cache that intercepts every in-forward
  `ttnn.zeros`/`ttnn.ones`/`ops._const` (conv padding, upsample fills, source/interp masks). The
  warmup forward fills it; the captured forward hits it — so the traced program has **zero host
  writes**. This fully solved the "Writes are not supported during trace capture" problem.
- `_decode` split into `_decode_features` (→ generator input `x`) + `generator`; `decode`/`vocode`
  trace stages run at a bucketed frame capacity `Cf = next_pow2(total)` with the masked frame-axis
  `shared_lstm` and the alignment naturally zero-filling columns `≥ total`.
- `trace_region_size` grown `1<<28 → 1<<30` (decode needs ~266 MB, vocode ~502 MB).

**Masked frame-axis normalization (the correctness fix).** The decoder/generator use INSTANCE /
adaptive-instance norm reducing mean/var over the **frame axis**; at a fixed `Cf` the zero-padded
frames (`total..Cf`) would poison the per-channel statistics for *every* frame. Fix
(`_lstm_scan.masked_moments` + `zero_pad_frames`, wired into `instance_norm1d`, `ada_i_n1d`,
`ada_i_n_res_block1`, `adain_res_blk1d`, and the generator/decoder blocks in `pipeline.py`): every
frame-axis reduction is computed over the VALID frames only, using a resident per-resolution mask
whose valid length scales with the (upsampled ~×300) resolution `L` as `round(T_valid*L/Cf)`, and each
residual block re-zeros its padded tail so convs see the same zero boundary as the dynamic path.

Result: **all four stages capture host-free and replay at capture-fidelity PCC 1.0**, AND the fixed-`Cf`
decode+vocode is numerically correct against the reference:
```
# EXACT golden-test metric (log_spectrogram_pcc vs HF gold; gate >= 0.95):
dynamic (shipping) vs gold = 0.993308
fixed-Cf traced    vs gold = 0.984438   <-- PASSES the gate (deterministic across runs)
```
Note: comparing fixed-vs-dynamic understates this (~0.68 log-spec) because the two paths' ~1e-3 F0
differences are uncorrelated and phase-decorrelate the NSF harmonic waveform — the meaningful metric is
vs the reference, exactly as the golden test gates.

**Measured perf (real, `test_main_perf.py` trace harness, single P150), pre-vocoder-tuning:**
```
TRACE_STAGE_MS  encode=157.5  prosody=117.9  decode=177.3  vocode=552.4   TRACE_PIPELINE_MS=1005.1
eager e2e = 1812 ms  ->  traced e2e = 1005 ms   (-807 ms, -45%)
RTF (2.10 s audio):  0.86 -> 0.48   (~2.1x real-time)
```
Default state: `_FRAME_TRACE` ON (`KOKORO_TRACE_FRAME=0` forces the single-CQ dynamic fallback). The
golden test (`run_tts`, dynamic) is unaffected and still passes (log-spec 0.9933).

### vocode: single-bf16 matmul in the HiFiGAN vocoder — SOLVED (−345 ms), ON by default
Tracy-profiling `vocode` (high `--op-support-count` so device data is complete) overturned the earlier
"device compute-bound" guess: the traced `vocode` runs **~8,250 ops but only ~22.6 ms of real device
kernel work** — it is **dispatch / op-count-bound** (~8k ops × ~65 µs on-device dispatch ≈ the 552 ms
replay). The op mix was dominated by `Typecast` (2,066/run = 1,033× `fp32→bf16` + 1,033× `bf16→fp32`
back-to-back **round-trip pairs**) and a 3-matmul expansion per logical matmul.

Root cause: `tt/ops.py::enable_hp_matmul` globally patches `ttnn.matmul`/`ttnn.linear` to `_mm1` — a
**near-fp32 emulation** that splits both operands into bf16 hi/lo (`_split_bf16` = the round-trip pair)
and does a 3-term matmul + K-chunked fp32 accumulation. Every logical matmul becomes ~14 dispatched ops.
That precision is **load-bearing for the AR decoder** (greedy-argmax stability) but overkill for the
HiFiGAN vocoder, which is gated only by log-spec PCC (0.984, comfortable headroom over 0.95).

Fix: a scoped `ops.set_hp_bypass(True/False)` flag (checked inside the patched `_mm`/`_lin`) that runs a
**single native bf16 matmul with the caller's HiFi4 + fp32-dest-accumulate config** instead of the
3-term split. `pipeline.py::vocode_trace_step` pushes it around the generator forward only, so the AR
path keeps the near-fp32 split. Both the warmup and captured forwards see the same flag, so the trace
graph matches.

The same bypass is applied to `decode_trace_step` (F0/N predictor + acoustic decoder). Decode is the more
fidelity-sensitive stage (its F0 drives the NSF harmonic phase), so it costs a little more PCC than the
vocoder — but stays comfortably above the gate.

**Measured perf (real, `test_main_perf.py` trace harness, single P150), current:**
```
TRACE_STAGE_MS  encode=157.5  prosody=117.9  decode=66.8  vocode=207.3   TRACE_PIPELINE_MS=549.4
vocode 552.4 -> 207.3 ms (-62%)   decode 177.3 -> 66.8 ms (-62%)   pipeline 1005 -> 549 ms (-45%)
RTF (2.10 s audio):  0.48 -> 0.26   (~3.8x real-time)
golden log-spec (>= 0.95 gate): dynamic 0.993308 (unchanged);
  fixed-Cf vocode-only bypass  = 0.985167
  fixed-Cf decode+vocode bypass = 0.980616   <-- deployed traced fidelity
```
### run_tts_fast + demo_tts_fast: the real chained production fast path — SHIPPED
`TRACE_PIPELINE_MS` above is the sum of the four stages profiled in isolation by the perf harness. The
real deliverable is a single runnable entry, `pipeline.py::run_tts_fast` (original `run_tts` untouched),
plus `demo/demo_tts_fast.py` which writes a real `.wav` and prints measured wall-clock RTF.

`run_tts_fast`: the token axis + duration→frame alignment run dynamically (data-dependent length), and
the whole FRAME axis (F0/N + acoustic decoder + ISTFTNet vocoder) is captured as ONE host-free trace and
replayed. Two extra realizations made this land well below the stage-sum estimate:
- **Kokoro is feed-forward (no AR argmax), so the near-fp32 hi/lo matmul emulation is unnecessary
  *everywhere*** — a single bf16 matmul (HiFi4 + fp32 accumulate) holds log-spec 0.9876 on the full
  dynamic path. So `run_tts_fast` runs the ENTIRE forward (token prep + traced frame block) under
  `ops.set_hp_bypass(True)`, collapsing ~14 dispatched ops/matmul to 1. This alone cut the dynamic
  token/align prep from **859 ms → 165 ms**.
- Tracing the frame block removes its per-op dispatch: **273 ms** replay.

**Measured, real end-to-end wall-clock (`demo_tts_fast`, single P150, 2.10 s audio):**
```
dynamic run_tts        = 1824 ms   RTF 0.869  (1.15x real-time)   log-spec 0.993308
fast run_tts_fast      =  439 ms   RTF 0.209  (4.78x real-time)   log-spec 0.980616
   = token/align prep 165 ms (dynamic, bf16-bypassed)  +  frame trace replay 273 ms
speedup 4.15x
```
This RTF 0.209 is a REAL number a customer observes running the demo — not a stage-sum. Quality is gated
the same way as the golden test (log-spec vs HF gold >= 0.95): fast = 0.9807, dynamic = 0.9933. The demo
writes `<out>` (fast) and `<out>.ref.wav` (dynamic reference) for A/B listening.

Remaining lever: the 165 ms token prep is still eager (dispatch-bound); tracing it too would trim toward
the ~440→~300 ms range but adds trace scaffolding for the text-encoder embedding. Not needed to clear
real-time by a wide margin.

### Commits
- `213bed1b` — `kokoro_82m: masked fixed-capacity path for the bidirectional LSTM stub (opt-in trace enabler)`
  (initial prototype in `l_s_t_m.py`). Device-mask + shared-primitive refactor + prosody wiring +
  2CQ (encode/prosody) are staged to be committed together once the trace+2CQ scope is settled.
