# FIBO SmolLM3 encoder trace (v2) — design

**Date:** 2026-07-22
**Branch:** `fibo-pipeline`
**Status:** Implemented

> **Outcome (2026-07-22): the prior replay-noise bug does NOT reproduce on the simplified encoder; the
> trace works and is fast.** Measured on the 4×8 Galaxy:
> - **Bit-exact & stable:** traced == untraced (PCC 1.0002 vs untraced, 0.99989 vs HF on the JSON
>   prompt); flat across **16 isolated replays** and **3 full-pipeline generations** (gen 2/3 identical
>   to gen 1 at PCC 0.9999999) — the exact "after the first run" condition, now clean.
> - **Perf: 3.58× on the real JSON encode** (untraced 1021.8 ms → traced replay 285.6 ms; one-time
>   capture 2281.1 ms).
> - The empty-prompt PCC dip (~0.979 vs HF) is the pre-existing short-sample effect, **identical
>   traced and untraced** — not a trace bug.
> - Therefore the planned **Phase 1/2 (root-cause + fix the noise) collapsed** — the simplification
>   (SP-only, single-output `_forward`, removed FSDP/mask paths) had already eliminated the cause.
>   Delivered: the per-bucket `Tracer` in the wrapper, driven by a **per-call `traced` flag** on
>   `encode_prompt` — threaded `pipeline.__call__ → _encode → encode_prompt(traced=...)`, i.e. the SAME
>   `traced` flag the DiT denoise uses (no separate build-time knob). `traced=True` requires a device
>   `trace_region_size`, exactly like the denoise trace, so profile/no-trace-region paths (which call
>   with `traced=False`) are unaffected. Plus a corrected replay-stability gate (asserts traced-replay
>   == captured baseline + json-vs-HF ≥ 0.99).

> Second attempt at tracing the encoder forward. The first attempt
> (`docs/superpowers/specs/2026-07-15-fibo-encoder-trace-design.md`, commits `9219018b5ba` …,
> reverted in `382b9c02825`) got the trace **bit-exact on capture** but **"produced noise after the
> first run"** on replay, and was reverted. This design targets that replay-stability bug directly:
> the deliverable is a forward trace that stays correct across many sequential encodes, not just the
> first.

## Goal

Eliminate the SmolLM3 encoder forward's host op-dispatch overhead by capturing it as a ttnn trace and
replaying it — while keeping replays bit-exact across an arbitrary number of sequential encodes
(multiple pipeline generations, pos + neg each).

## Motivation

The whole-mesh SP × TP encoder forward is host-dispatch-bound: ~1000 ops (36 layers × ~28 ops) with
only ~14 ms of summed device compute per forward but multi-second op-to-op gaps (the device idles
while the host enqueues each op). A ttnn trace records the device program once and replays it with no
per-op host dispatch, directly targeting the gaps. The fixed 1024-token bucket already makes the
forward a static shape (pos and neg both pad to 1024), which is the precondition for tracing.

The readback fix (`_read_seq_sharded` via `ttnn.get_device_tensors`, the ~20× win) is **already
shipped** and is out of scope here — it is correctness-neutral and works untraced.

## Non-goals

- The host readback / device→host→device round-trip into the DiT (a separate, larger lever).
- Any change to the SP math, weights, or the encoder's public output contract.
- Multi-bucket support: only the shipped 1024 bucket is traced (pos and neg share it).

## Why the prior attempt failed (working hypothesis)

The encoder forward's per-layer CCL all-gathers (SP K/V gather on the sp axis, TP gathers on the tp
axis) go through `CCLManager`, which selects ping-pong **buffers** and **global semaphores** via
**Python-side indices** (`ag_ping_pong_idx`, `_ping_pong_buffer_indices`, etc.) that flip on every
call. A ttnn trace bakes whichever buffer/semaphore the Python index pointed at *during capture*;
`ttnn.execute_trace` replays those fixed addresses and **does not advance the Python indices**. If the
manager's ping-pong phase (or semaphore reset state) at the start of a replay differs from what
capture baked — because an untraced call (e.g. the `__init__` allocation-run encode) advanced the
indices, or because capture's `prep_run` + capture double-execution left an unexpected phase — replays
read/write the wrong ping-pong slot and drift to noise after the first run.

The denoise trace uses the *same* `all_gather_persistent_buffer` and is stable across many replays, so
CCL-in-trace is not fundamentally broken — the denoise pipeline is simply architected so nothing
untraced touches its CCLManager after capture and the phase is deterministic. The fix is to give the
encoder trace the same guarantees.

## Design

### 1. Phase-split `encode_prompt`

- `_prep_inputs(prompt) -> (tt_ids, tt_cos, tt_sin, seq_len)` — host tokenize + `pick_bucket` + pad +
  `create_rope_tensors` + `from_torch` (SP-sharded on the sp axis). No CCL ops.
- `_forward(tt_ids, tt_cos, tt_sin) -> ttnn.Tensor` — the ~1000-op device forward, returning the
  **single stacked** hidden-states tensor (`ttnn.concat(all_hidden_states, dim=0)`), exactly what
  `_read_seq_sharded` already consumes. This is the unit wrapped by the `Tracer`. A single output
  (not the prior design's 38-tensor tuple) is simpler and removes multi-output buffer aliasing as a
  possible contributor to the old replay bug.
- readback — unchanged `_read_seq_sharded` + host slice to `seq_len` + host split into the hidden
  list + host-derived `prompt_embeds`. No CCL ops.

The `prompt_embeds = cat(hidden[-1], hidden[-2])` contract is unchanged (derived on host after
readback).

### 2. Single-bucket Tracer

One `Tracer(self._forward, device=self._submesh, prep_run=True, clone_prep_inputs=False)` on the
wrapper (created only when `use_trace and not use_torch`). Both pos and neg pad to bucket 1024 and
share it: the first encode captures, all later encodes replay. Inputs are handed as already-SP-sharded
**device** tensors (the `Tracer` copies device→device into its persistent input buffers, preserving
the sharding; host tensors would lose it).

### 3. The replay-stability fix

The exact fix is confirmed by the Phase-1 investigation; the design commits to the *property* and
lists the candidate mechanisms in priority order:

**Property:** every traced `_forward` (capture and every replay) must see the encoder CCLManager in an
identical ping-pong buffer/semaphore phase, and nothing untraced may touch that manager after capture.

- **(a) Capture-owned, phase-reset encoder CCLManager (preferred).** Ensure the encoder's
  `_encoder_ccl_manager` is dedicated to the traced forward and is in a known phase when capture runs
  — e.g. reset its ping-pong indices (`ag_ping_pong_idx`, `_ping_pong_buffer_indices`, semaphore
  phase) immediately before capture, and ensure the `__init__` allocation-run encode does not leave it
  in a conflicting phase (capture after the warmup, or use a fresh manager for the trace). Mirrors the
  denoise trace's isolation.
- **(b) Non-ping-pong all-gather on the traced path.** If (a) cannot be made deterministic, force the
  encoder's traced all-gathers to a single fixed buffer + fixed semaphore set (no Python alternation),
  trading some overlap for trace-determinism.

Phase 1 measures the ping-pong/semaphore phase at capture vs each replay and diffs it against the
denoise trace's phasing to pick (a) vs (b).

### 4. Guard flag

`use_trace: bool` on `SmolLM3TextEncoderWrapper.__init__`:
- Pipeline + `test_fibo_encode_perf` → `use_trace=True` (their device params set
  `trace_region_size`).
- `test_fibo_encode_device_profile` (no trace region, wants real per-op timings) and `use_torch=True`
  → `use_trace=False`; the untraced path stays byte-for-byte current behavior.

Default value is decided once the correctness gate passes (lean toward `True` to match the prior
design, but only after the gate is green).

## Correctness gate (primary deliverable)

A new test reproduces and guards the reverted failure: build the wrapper with `use_trace=True`, run
**N sequential encodes** alternating pos/neg across ≥2 simulated generations (e.g. the JSON prompt and
`""`, repeated), and assert **every** readback — not just the first — stays PCC ≥ 0.99 vs the HF
reference. The existing untraced `test_smollm3_encoder_sp` continues to guard the SP math.

## Plan shape (investigation-first)

- **Phase 1 — reproduce & root-cause.** Re-apply the forward Tracer behind `use_trace`, run the
  N-encode gate, confirm the noise reproduces on ≥2nd run, then instrument the encoder CCLManager
  ping-pong buffer/semaphore phase at capture vs each replay and diff against the denoise trace.
  Output: the confirmed cause.
- **Phase 2 — fix.** Apply (a) or (b) per Phase 1, re-run the N-encode gate to PCC 0.99.
- **Phase 3 — perf + integrate.** Confirm `test_fibo_encode_perf` drops (forward dispatch gone), wire
  the flag into the pipeline, verify no denoise-trace or pipeline latent-PCC regression.

## Verification

1. **N-encode replay-stability gate** (PCC 0.99 on every encode) — primary; directly reproduces the
   reverted failure.
2. `test_fibo_encode_perf` (4×8, `-s`) — encode forward wall-clock drops toward the readback floor.
3. `test_smollm3_encoder_sp` — unchanged (SP math).
4. Full pipeline latent-PCC — unchanged (no denoise-trace regression from the encoder trace / shared
   trace region).

## Risks

- **CCL-in-trace phase is subtle.** The fix depends on the Phase-1 finding; (b) is the fallback if (a)
  is not deterministic. Both are bounded to the encoder's traced all-gathers.
- **Trace region sharing.** The encoder trace and the denoise trace both live in the device
  `trace_region_size`; Phase 3 verifies they coexist (they capture on separate CCLManagers, as the
  pipeline already isolates encoder / transformer / VAE managers).
- **Warmup interaction.** The `__init__` allocation-run encode runs the forward untraced and advances
  the encoder manager's ping-pong phase; the fix must account for capture happening after it.
- **Modest headline win.** The forward is only ~14 ms of compute; the gain is the dispatch gaps. If
  Phase 1 shows the untraced encode is already dominated by prep/readback (both untraceable here), the
  perf upside is small — the gate still requires correctness, and the flag lets us ship it off if the
  win doesn't justify the complexity.
