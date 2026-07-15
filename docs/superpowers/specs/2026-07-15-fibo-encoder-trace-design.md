# FIBO encoder trace (capture/replay) design

**Date:** 2026-07-15
**Branch:** `fibo-pipeline`
**Status:** Implemented

> **Outcome:** Trace is bit-exact (traced == untraced, PCC 100.0000% on the JSON prompt). Encode
> dropped ~12.5 s → **10.8 s (~13%)** — smaller than hoped: the device forward is only ~14 ms, so the
> encode is **host-readback-bound** (the 37-tensor `to_torch` of hidden states gathered over the SP
> axis, ~260 MB device→host per prompt), which the trace does not cover. Also discovered: the encoder's
> vs-HF accuracy is 99.97% for realistic (833-token) prompts and only dips to 98.5% for pathologically
> short (4-token) ones — a pre-existing small-sample/padding effect, not a bug, and unrelated to tracing.
> **Next lever: the hidden-state readback** (concat device-side before one `to_torch`, or keep them
> on-device for the DiT to skip the device→host→device round-trip).

## Goal

Eliminate the host op-dispatch overhead of the SmolLM3 encoder forward by capturing it as a ttnn
trace and replaying it, validated first through the encoder-only perf test
(`test_fibo_encode_perf`).

## Motivation (profile)

The whole-mesh SP=8 × TP=4 encoder measures ~12.5 s per encode (pos+neg) yet the device-op profile
(`encode_report_4x8/`) shows only ~14 ms of summed device compute per forward, with **multi-second
"Op-to-Op Gap" values** between ops. The encode is **host-dispatch-bound**: the device idles while
the host enqueues each of the ~1000 ops (36 layers × ~28 ops). A ttnn trace records the whole device
program once and replays it from device with no per-op host dispatch, which directly targets this.

The fixed **1024-token bucket** (already shipped) makes the forward a **static shape**, which is the
precondition for tracing — positive and negative prompts both pad to 1024, so a single trace serves
both, every run.

## Scope

- **In:** trace the encoder **device forward** in `SmolLM3TextEncoderWrapper`; validate via
  `test_fibo_encode_perf`.
- **Out (deferred):** the 37-tensor `to_torch` host readback and the device→host→device round-trip
  of hidden states into the DiT; any pipeline-wide change beyond what the wrapper gives for free.

## Current state

`SmolLM3TextEncoderWrapper.encode_prompt` (device path) does, per prompt:
1. tokenize → `pick_bucket` → pad `input_ids` to bucket → `create_rope_tensors(1, bucket)` (host).
2. `from_torch` `input_ids`/`cos`/`sin` to device, sharded on the SP axis when `sp_factor > 1`.
3. `self._encoder.encode(tt_ids, attention_mask=None, pos_embeds=(cos, sin))` → `(prompt_embeds,
   all_hidden_states)` on device.
4. `to_torch` (gathered over the SP axis) + host slice `[:, :seq_len, :]`.

Step 3 is the ~1000-op device forward that suffers the host-dispatch gaps.

The `Tracer` utility (`models/tt_dit/utils/tracing.py`) already implements capture-once/replay for a
device-only function: it holds persistent input buffers, validates input shape/dtype/layout, copies
new inputs into the buffers, and replays. The denoise path uses it with on-device sequence-sharded
tensors and dict/tuple inputs, so it fits the encoder.

## Design

### 1. Phase split in `encode_prompt`

Refactor the device path of `encode_prompt` into:
- `_prep_inputs(prompt) -> (tt_ids, tt_cos, tt_sin, seq_len, bucket)` — host prep + `from_torch`
  (steps 1–2, unchanged logic).
- `_forward(tt_ids, tt_cos, tt_sin) -> tuple[ttnn.Tensor, ...]` — returns
  `(prompt_embeds, *all_hidden_states)` (a flat tuple, since the `Tracer` output must be tensors or
  containers of tensors). This is the unit wrapped by the `Tracer`.
- readback (step 4, unchanged) — `to_torch` each output, slice to `seq_len`, and re-split the flat
  tuple back into `(prompt_embeds, list_of_hidden_states)`.

### 2. Tracer, keyed by bucket

- `self._tracers: dict[int, Tracer]` on the wrapper (only when `use_trace and not use_torch`).
- On each device encode: `bucket = pick_bucket(...)`; get-or-create `Tracer(self._forward,
  device=self._device, prep_run=True, clone_prep_inputs=False)` for that bucket; call it with the
  freshly-prepped on-device tensors. First call for a bucket captures + executes; later calls copy
  the new inputs into the trace buffers and replay.
- Pos and neg (both bucket 1024) share the same `Tracer`; capture happens on the first encode
  (warmup), all subsequent encodes replay.
- Inputs are **already sharded on-device** tensors → the `Tracer` copies device→device into its
  persistent buffers, preserving the sharding (same as the denoise tracer). Do **not** hand the
  `Tracer` host tensors (its host→device move would not reproduce the SP sharding).

### 3. Multi-tensor output

The traced `_forward` returns 38 tensors (`prompt_embeds` + 37 hidden states) at full bucket length.
Confirm the `Tracer` supports a tuple-of-tensors output (denoise returns a single tensor). If it does
not, extend the `Tracer` (or wrap the outputs in a supported container). Host slicing to the true
`seq_len` stays outside the trace (per-prompt, host-side).

### 4. Guard flag

Add `use_trace: bool = True` to `SmolLM3TextEncoderWrapper.__init__`:
- Pipeline and `test_fibo_encode_perf` → `use_trace=True` (their `_DEVICE_PARAMS` sets
  `trace_region_size=200_000_000`; the encode-only test captures no denoise trace, so the region is
  free for the encoder trace).
- `test_fibo_encode_device_profile` (uses `_PROFILE_DEVICE_PARAMS`, **no** trace region, wants real
  per-op timings) and `use_torch=True` → `use_trace=False`; the untraced path is byte-for-byte the
  current behavior.

## Correctness

- Trace replay executes the identical captured ops, so numerics equal the untraced path. Add a test:
  build the wrapper with `use_trace=True`, encode a prompt (capture) and again (replay), and assert
  both match the untraced/HF reference via `assert_quality(pcc>=0.99)`.
- The existing `test_smollm3_encoder_sp` (untraced) continues to guard the SP math.

## Verification

- `test_fibo_encode_perf` (4×8, `-s`): encode wall-clock should drop sharply from ~12.5 s toward the
  device-compute floor + host readback (warmup captures, 3 measured runs replay). Headline metric.
- No regression in `test_smollm3_encoder_sp` and the new traced-correctness test.

## Risks

- **Tracer tuple output** (38 tensors): the main unknown; verify support, extend if needed.
- **Trace region coexistence**: in the full pipeline the encoder trace + 2 denoise traces must fit
  `trace_region_size`. Out of scope here (encode-only test captures no denoise trace); flag for the
  pipeline-wide follow-up.
- **CCL ping-pong state**: the encoder uses its own `CCLManager`; once its trace is captured, only
  traced encodes run on it, so the baked ping-pong state is never desynced by untraced ops (same
  discipline as the denoise/VAE managers).

## Follow-ups (out of scope)

- Reduce/eliminate the 37-tensor readback and the device→host→device round-trip into the DiT.
- Encoder trace inside the full generation pipeline (coexistence with denoise traces).
- Multiple buckets (2048, …) — the keyed-`dict` structure already supports it; add buckets when a
  prompt needs them.
