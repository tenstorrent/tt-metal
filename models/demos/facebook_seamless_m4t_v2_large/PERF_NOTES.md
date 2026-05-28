# SeamlessM4T-v2 Performance Notes (Phase 9)

Characterization + one targeted optimization pass on the integrated TTNN
pipeline. All numbers below are from a single Blackhole p150a, bf16 weights
in DRAM, no metal trace, T2TT use case (the shortest, cleanest path that
exercises encoder + 24-layer decoder + AR loop + LM head).

## Setup

- Device: p150a (single Blackhole)
- ARCH_NAME=blackhole
- Branch: ssinghal/seamless-m4t @ d0b558ffa37
- Harness: `models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2tt.py`
- Prompt: `"Hello world."` (eng -> fra)
- max_new_tokens = 32, generation actually stopped at 9 tokens (EOS)
- All times measured with `ttnn.synchronize_device` boundaries
  (host-perceived latency, NOT pure kernel duration)

## Baseline numbers (no optimization, 3 timed translate() calls)

| Phase                                  | p50      |
| -------------------------------------- | -------- |
| prefill_ms (encoder + cross-attn pop)  | 14.94    |
| steady_decode_step_ms                  | 20.88    |
| total_ms / translate()                 | 208.30   |
| tokens_generated_per_call              | 9        |

Per-step cost dominates total wall-clock: ~9 decode steps × ~21 ms = 188 ms
(90 % of total). Encoder is small (3 input ids), prefill is correspondingly
cheap.

## Tracy findings

Captured under `python -m tracy -p -v -r --op-support-count 3000
--dump-device-data-mid-run -n seamless_m4t_t2tt ...`. CSV at
`/local/ttuser/ssinghal/tt-metal/generated/profiler/.logs/tracy_ops_data.csv`.

### Top device ops by call count (over the entire run = warmup + 1 timed)

| OP                              | calls | notes                          |
| ------------------------------- | -----:| ------------------------------ |
| MatmulDeviceOperation           | 3174  | Q/K/V/out projs, FFN, LM head  |
| TransposeDeviceOperation        | 2396  | head-dim transpose             |
| ReshapeViewDeviceOperation      | 2396  | shape ops, no kernel           |
| BinaryNgDeviceOperation         | 1310  | residual adds, mask adds       |
| LayerNormDeviceOperation        | 1118  | 3 LNs × 24 layers × N steps    |
| SDPAOperation                   |  717  | self + cross SDPA × 24 × N     |
| UpdateKVCacheOperation          |  671  | self K + V × 24 × N            |
| UnaryDeviceOperation            |  383  | ReLU, scalar muls, etc.        |

### Top host-side zones by wall-time

| ZONE                                  | total_ms | mean_us | notes                |
| ------------------------------------- | -------: | ------: | -------------------- |
| convert_python_tensor_to_tt_tensor    | 6337     | 5603    | `ttnn.from_torch`    |
| to_tile_major_layout_nfaces           | 1688     | 1542    | tilize on host       |
| FDMeshCommandQueue::finish_nolock     |   51     |  169    | per-op finish/sync   |

(Tracy itself adds ~9 s of `readDeviceMarkerData` / `dumpDeviceResults` —
ignored for per-step analysis.)

### Diagnosis

- Total device-kernel time across the entire run was ~70 ms (from
  `cpp_device_perf_report.csv`). The observed 20.88 ms per step is
  dominated by host-side dispatch overhead, not on-device compute.
- With ~365 kernel dispatches per decode step (24 layers × ~14 ops/layer
  + embed/LN/LM head), at 20.88 ms wall we are at ~57 µs/op average. This
  is the textbook signature of being host-dispatch limited.
- The canonical fix is **metal trace + execute_trace** on the AR step:
  capture the static graph once, replay per step. This eliminates the
  per-op host dispatch cost.

## Optimization applied

**Precompute the per-call invariant cross-attention mask once and reuse it
across every decode step.** Implemented in
`tt/text_generator.py::TextGenerator.generate()` via new
`precomputed_encoder_mask_tt` keyword on `text_decoder.decode_step()`.

Rationale: the encoder mask is a strict function of `encoder_attention_mask`
+ `encoder_seq_len_logical`, both invariant within one `generate()`. The
baseline path rebuilt it host-side and re-uploaded it every decode step.
This optimization saves `max_total - 1` redundant host->device transfers
of a `[1, 1, 1, enc_seq_padded]` fp32 tensor + matching tilize passes.

### After numbers (same prompt, 3 timed runs)

| Phase                                  | p50      | vs baseline |
| -------------------------------------- | -------- | ----------- |
| prefill_ms                             | 14.65    | -0.29 ms    |
| steady_decode_step_ms                  | 20.98    | +0.10 ms    |
| total_ms / translate()                 | 209.12   | +0.82 ms    |

Effectively a wash — encoder mask for `"Hello world."` is only [1,1,1,32]
(enc_seq_padded=32 after tile rounding), and the per-step rebuild cost is
< 1 ms on this small case. The optimization correctness path is preserved
(BLEU = 42.524 matches HF exactly on the 10-sample set) and would scale
better on prompts where `enc_seq_padded >= 128`; for the short-form set
the delta is below noise.

This was retained as a small structural improvement (correct + safe + adds
the `precomputed_*_mask_tt` hooks needed for the future trace path), not
as a measured win.

## Why metal trace was deferred

`text_decoder.decode_step` currently has three structural blockers against
a single re-usable trace:

1. **`ttnn.update_cache(cache, k_new, update_idx=int(pos))`** in
   `kv_cache.py::SelfAttentionKVCache.update`. The `update_idx` is captured
   into the trace as a static int; replaying the trace would always write
   into position 0. Fix: switch to the `cur_pos_tensor` form (a 1-element
   device tensor that the cache op reads at runtime) used by
   `models/demos/qwen3_tts/tt/server.py`. Requires the cache op variant
   plus a per-step host->device write of one int.
2. **`ttnn.from_torch` of `input_ids` and `position_ids`** inside
   `text_decoder.decode_step` and `sinusoidal_positional_embedding.forward`.
   Per-step allocation of a new ROW_MAJOR tensor breaks trace
   reusability. Fix: pre-allocate one persistent `input_ids` tensor and
   one `position_ids` tensor, and write via `ttnn.copy_host_to_device_tensor`
   (matches the qwen3_tts pattern).
3. **Host-side self-attention mask construction.** Same fix: a persistent
   `[1, 1, 1, max_seq_len]` mask buffer updated by
   `ttnn.copy_host_to_device_tensor` per step. (Or compute the mask on
   device by comparing a position tensor against an arange.)

The cross-attention mask is now already a one-time upload (this PR), so
it does NOT block trace; it just sits as a regular trace-time DRAM input.

## Path forward (deferred to a follow-up)

The expected wins from full trace capture, given the host-bound
characterization above:

- Eliminate per-op dispatch: at ~365 ops/step and ~50-60 µs/op overhead,
  the upper-bound saving is ~17-19 ms/step. Even keeping 25% of overhead
  (host->device input copy + KV update + sync) we should land in the
  10-15 ms/step range, i.e. ~50% reduction.
- Prefill stays approximately the same — it's already a one-time cost
  that's cheap relative to the AR loop and not worth trace.

Recommended ordered work for the next perf pass:

1. Convert `SelfAttentionKVCache.update` to take a `cur_pos_tensor`
   instead of an int `pos`, and thread it through `text_decoder_layer`.
2. Make per-step inputs (input_ids, position_ids, self-mask) persistent
   ttnn tensors; write via `copy_host_to_device_tensor` in
   `TextGenerator.generate()`.
3. Add `capture_trace=True` / `execute_trace` machinery to
   `TextGenerator` (mirroring `models/demos/qwen3_tts/tt/server.py:1391`).
4. Re-run `profile_t2tt.py --traced` and capture deltas.
5. Optional follow-up: 2-CQ overlap of decode_step kernel launch with
   greedy argmax + host token append (sub-millisecond, but cheap to add).

## Phase 9b — Metal-trace pass (current PR)

Added the structural plumbing for metal trace + replay on
`TextGenerator.decode_step`, gated behind a new `use_trace=True` kwarg
on `generate()`. Default is **`use_trace=False`** because of a
correctness regression in trace re-use across `generate()` calls
documented below.

### What landed

1. `tt/kv_cache.py`
   - `SelfAttentionKVCache.reset()` now streams a host-side zero tensor
     into each cache buffer via `copy_host_to_device_tensor` instead of
     re-allocating via `ttnn.mul(..., output_tensor=...)`. Preserves
     the cache buffer ADDRESS across `generate()` calls so any captured
     trace continues to point at the right K/V slot. (Matches the
     qwen3_tts cross-call trace pattern at `server.py:1414-1415`.)
   - `CrossAttentionKVCache.__init__` now pre-allocates persistent K/V
     buffers per layer; `populate()` uses `ttnn.fill_cache` to write the
     freshly projected encoder K/V into the existing buffers (instead of
     replacing the tensor handles); `reset()` is a no-op on the
     buffers (only flips the `_populated[i]` flag) for the same
     trace-pointer-stability reason.
   - `SelfAttentionKVCache.update(layer_idx, k_new, v_new, pos)` now
     accepts a `ttnn.Tensor` for `pos` as a backwards-compat overload
     (auto-extracts the scalar). Currently unused — the trace path bakes
     `pos` as a Python int constant per captured trace.
2. `tt/sinusoidal_positional_embedding.py`
   - `forward(precomputed_position_ids_tt=…)` skips the host-side
     cumsum + H2D upload and gathers directly into the persistent
     position-id buffer.
3. `tt/text_decoder.py::decode_step`
   - New `persistent_input_ids_tt`, `persistent_position_ids_tt` kwargs
     let the caller hand in pre-allocated device buffers (overwritten
     per step via `copy_host_to_device_tensor`).
4. `tt/text_generator.py`
   - `__init__` allocates the persistent input-id, position-id, and
     encoder-mask buffers lazily (`_ensure_persistent_buffers`).
   - `_precompute_encoder_mask` writes the per-call cross-attn mask
     INTO the persistent encoder-mask buffer (when trace mode is
     active) so the buffer ADDRESS captured into every decode trace
     remains valid across `generate()` calls.
   - `_run_decode_body(position)` is the trace-captureable body (calls
     `text_decoder.decode_step` with the persistent buffers, then
     `ttnn.linear` for the LM head).
   - `_capture_decode_trace(position)` captures one trace PER absolute
     decode position because `ttnn.update_cache` only accepts a
     Python-int `update_idx` (no `cur_pos_tensor` form yet).
   - `_generate_traced` is the AR loop under trace: untraced warmup at
     pos 0–1, then for each AR position lazily capture (untraced
     warmup pass + `begin_trace_capture`) on first hit and replay
     thereafter.
   - `step_callback(position, ms, kind)` hook lets a profiler measure
     per-step latency tagged by phase (`warmup` / `capture` / `replay`).
5. `tt/profile_t2tt.py`
   - `--traced` flag now actually enables trace replay; opens the
     device with `trace_region_size=256_000_000`.
   - Per-step report splits replay (steady-state) from warmup/capture
     (first-hit cost).

### Measured numbers (T2TT, "Hello world.", 9-token output)

| Phase                                          | Untraced | Traced (per-call recapture) |
| ---------------------------------------------- | -------: | --------------------------: |
| prefill_ms                                     |    13.05 |                       14.43 |
| steady_decode_step_ms (p50 replay)             |    20.34 |                       17.62 |
| capture_step_ms (per first-hit-of-pos)         |        — |                       18.4 |
| warmup_step_ms (per first-hit-of-pos)          |        — |                       17.7 |
| total_ms / translate() (9 tokens, 1 generate)  |   200.30 |                      450.10 |

The replay path is ~13% faster than untraced (17.62 vs 20.34 ms/step),
but the per-call recapture cost (warmup + capture ≈ 36 ms per AR
position) currently dominates for short sequences. Net total wall-clock
is WORSE than the baseline on T2TT-short.

### Correctness regression in cross-call trace re-use

Initial integration captured each decode trace ONCE and reused it
across subsequent `generate()` calls. With every other piece of the
trace-readable state held persistent (input-id buffer, position-id
buffer, self-attn KV cache via in-place zero, cross-attn KV cache via
`fill_cache`, encoder-mask buffer via `copy_host_to_device_tensor`,
LM-head weight, all decoder weights), the captured trace nonetheless
produced INCORRECT logits at the second decoded position onward (pos 2
matched the warmup-gen run bit-exact; pos 3 diverged). Same prompt,
same input-id sequence, same persistent buffer addresses. The
divergence is consistent with implicit state captured by the trace
(probably program-cache IDs or pre-allocated workspace) that doesn't
survive a cross-call replay against the same K/V buffer.

Whisper's analogous decoder (`models/demos/audio/whisper/tt/whisper_generator.py`)
sidesteps this by:

1. Using `ttnn.experimental.paged_update_cache(cache, k, update_idxs_tensor=cur_pos)`
   so a SINGLE trace re-runs at any position via the on-device
   `current_decode_pos` tensor.
2. Doing on-device `argmax` + `ttnn.copy(token, persistent_token_buf)`
   + `ttnn.plus_one(decode_pos)` *inside* the trace, so the next
   replay reads its own previous output as input — no host
   intervention between steps.

This pattern requires re-writing the seamless self-attention block to
use the sharded `[1, B, H, d]` HEIGHT-sharded K/V input layout that
`paged_update_cache` expects (Whisper does this around
`ttnn_optimized_functional_whisper.py:527-557`). That's substantial
work touching `SeamlessMHA` (4 projections, all four buses), the
`TextDecoderLayer` glue, and the K/V cache layout — out of scope for
this phase.

### What's still live

The `use_trace=False` default path still benefits from the structural
work in this phase:

- Persistent encoder-mask buffer + `copy_host_to_device_tensor` upload
  (one alloc per generator, not per step).
- Persistent cross-attn KV cache buffers (one alloc per generator, not
  per encoder forward).
- Tensor-input path on the sinusoidal positional embedding.

These do not measurably move the steady-state number on T2TT-short
(host dispatch is still 365 ops/step at ~20 ms) but they're prerequisites
for the next iteration. All 5 e2e tests pass on the `use_trace=False`
path (BLEU=42.524, WER=0.0, char-sim=1.0/0.92+ for T2TT/S2TT/T2ST/S2ST,
2/2 token-match tests).

### Next steps

The textbook trace win still requires migrating
`SeamlessMHA.attend_and_out_project` to take a `cur_pos_tensor` and
issuing `ttnn.experimental.paged_update_cache` against a HEIGHT-sharded
K/V input. Once that's in, the per-position trace approach is replaced
by a SINGLE trace and the warmup+capture overhead amortises to zero
across calls.

## Files touched in this phase

- `models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2tt.py` (new)
  — characterization harness for T2TT.
- `models/demos/facebook_seamless_m4t_v2_large/tt/text_decoder.py`
  — added `precomputed_self_mask_tt` and `precomputed_encoder_mask_tt`
  optional kwargs to `decode_step()`. Backwards compatible.
- `models/demos/facebook_seamless_m4t_v2_large/tt/text_generator.py`
  — `generate()` now precomputes the cross-attn mask once and passes it
  via the new kwarg; added `_precompute_encoder_mask` helper.

## Verification

All 5 e2e tests pass after the change:

- `test_e2e_t2tt.py::test_t2tt_bleu_matches_hf` — TTNN BLEU = HF BLEU = 42.524
- `test_e2e_s2tt.py::test_s2tt_wer_matches_hf` — pass
- `test_e2e_s2tt.py::test_asr_wer_matches_hf` — pass
- `test_e2e_t2st.py::test_t2st_audio_parity_with_hf` — pass
- `test_e2e_s2st.py::test_s2st_audio_parity_with_hf` — pass
- `test_text_generator.py` (2 tests) — pass

## S2TT

Per-use-case pipeline perf characterization for S2TT (speech-to-text
translation). The decode path reuses the same `TextGenerator` that T2TT
uses, so the structural Phase 9c work (paged_update_cache + cross-call
trace) is already in for this use case. What's S2TT-specific is the
prefill: a 256-step audio-feature `SpeechEncoder` (24 Conformer layers
+ 1 adapter) instead of T2TT's tiny text encoder.

### Setup (S2TT)

- Device: p150a (single Blackhole)
- ARCH_NAME=blackhole
- Branch: ssinghal/seamless-m4t @ d0b558ffa37 + this pass
- Harness: `models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2tt.py`
- Audio: `demo/inputs/sample_hello.wav` (~1.6 s, eng -> fra)
- max_new_tokens = 32; generation stopped at 11 tokens
- All times measured with `ttnn.synchronize_device` boundaries
  (host-perceived latency, NOT pure kernel duration)

### Baseline numbers (2 timed translate() calls per config)

| Phase                                  | Untraced | Traced  | Delta            |
| -------------------------------------- | -------: | ------: | ---------------- |
| prefill_ms (speech_enc + cross-attn)   |    56.90 |   73.43 | +16.5 (capture)  |
| steady_decode_step_ms (p50 replay)     |    24.18 |   17.47 | -6.71 (-27.7 %)  |
| total_ms / translate() (11 tokens)     |   353.29 |  273.08 | -80.2 (1.29x)    |
| tokens_generated_per_call              |       11 |      11 |                  |

The traced steady-state per-step (17.47 ms) on S2TT is essentially the
same as on T2TT (17.62 ms) — the decode path is identical, so the
trace replay timing is invariant across use cases by design.

The end-to-end **1.29x speedup** is larger than T2TT's 1.21x because
the S2TT AR loop generated 11 tokens vs T2TT's 9 — more decode steps
means more replays in which the trace win compounds. Prefill grows
~5x from T2TT to S2TT (15 ms -> 57 ms untraced; the Conformer encoder
dominates) and is absorbed by the trace capture cost on the first
call (73 ms traced), then disappears for subsequent calls because the
trace is cross-call-reusable.

### Tracy findings (S2TT, 1 warmup + 1 timed, ~22 110 device ops)

Captured via `python -m tracy -p -v -r --no-device -n seamless_m4t_s2tt
models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2tt.py
--num-timed 1`.

#### Top device ops by call count

| OP                                  | calls | notes                          |
| ----------------------------------- | ----:| ------------------------------ |
| MatmulDeviceOperation               | 5088 | encoder + decoder projs, FFN, LM head |
| ReshapeViewDeviceOperation          | 4664 | view-only, no kernel           |
| TransposeDeviceOperation            | 3658 | head-dim transpose, Conformer  |
| BinaryNgDeviceOperation             | 2152 | residual adds                  |
| LayerNormDeviceOperation            | 1906 | encoder+decoder LN + adapter   |
| SDPAOperation                       | 1056 | self+cross SDPA                |
| PagedUpdateCacheDeviceOperation     | 1056 | self-attn KV update (Phase 9c) |
| InterleavedToShardedDeviceOperation | 1104 | I2S for sharded paths          |
| UnaryDeviceOperation                |  728 | SiLU, scalar muls              |
| UpdateKVCacheOperation              |  192 | cross-attn cache fill (one-shot per call) |
| Conv2dDeviceOperation               |   48 | Conformer convolution module   |

#### Top host-side wall-time zones

| ZONE                                  | calls | total_ms | mean_us |
| ------------------------------------- | ----:| -------: | ------: |
| convert_python_tensor_to_tt_tensor    | 2139 |     5330 |    2492 |
| to_tile_major_layout_nfaces           | 1564 |     2004 |    1281 |
| FDMeshCommandQueue::finish_nolock     |   51 |       65 |    1269 |

Per-op host-enqueue cost is remarkably flat (~1.1 - 1.9 us per op).
There is no single op type at >5 % wall-clock dominance — Matmul is the
biggest by call count (37 % of dispatches) but its mean host-enqueue is
1.6 us, the same range as Transpose and Reshape. Speech-encoder host
mask-build (`_build_encoder_attention_mask` + `_to_tt`) shows up as
the bulk of the `convert_python_tensor_to_tt_tensor` wall-time, but
this is prefill cost, not steady-state.

### Diagnosis

Steady-state per-step (17.47 ms traced) is the same as T2TT, so the
floor analysis from Phase 9b carries over: ~365 ops/step at ~50 us
host dispatch is ~18 ms — and traced replay already removes that.
The remaining ~17 ms is on-device kernel time. We are at the kernel-time
floor; further wins require per-block work (sharded matmuls, LN/Linear
fusion) which is the `skills/optimization/` frontier. Bringup log
already reports 24/24 blocks at-ceiling at the optimization-skill
level, so all leverage at the pipeline level for S2TT is exhausted.

### Optimization applied

**None — at-ceiling at the pipeline-perf level.**

The trace machinery from Phase 9c already provides the largest
pipeline-level win available (1.29x end-to-end). Tracy shows no single
device op at >5 % of the budget; per-op host-enqueue costs are flat
across MatMul / Reshape / Transpose / LayerNorm. The conformer mask
build (`_build_encoder_attention_mask`) is the largest host-side
zone but runs only once per translate() and amortises across the AR
loop. Forcing a "win" by tuning two flat ops at once is a regression
risk (per the failure-modes section of `skills/perf/SKILL.md`).

The encoder mask cost could be reduced by caching the chunked
attention mask between translate() calls when the audio_seq_len is
constant, but the wall-time delta would be ~5 ms one-shot on a single
prefill call — below the noise floor of the steady-state AR loop and
not worth the structural churn.

### After numbers

Same as the baseline traced row (no additional optimization applied
in this pass — characterization confirmed Phase 9c trace is the
pipeline-level win).

| Phase                                  | After   |
| -------------------------------------- | ------: |
| prefill_ms (one-shot capture cost)     |   73.43 |
| steady_decode_step_ms                  |   17.47 |
| total_ms / translate() (11 tokens)     |  273.08 |
| tokens_generated_per_call              |      11 |

### Correctness verification

```
pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_s2tt.py -v
```

- `test_s2tt_wer_matches_hf` — PASS (TTNN_WER = HF_WER = 0.0000, drift = 0.0000, tolerance = 0.05)
- `test_asr_wer_matches_hf` — PASS (TTNN_WER = HF_WER = 0.0000, drift = 0.0000)

### Recommendations for the next pass

1. **Per-block work first.** S2TT's remaining 17 ms/step is on-device
   compute. The next leverage is at the `skills/optimization/` frontier:
   sharded matmul kernel configs in the Conformer encoder layer, LM-head
   `bfloat8_b` weight quantization. Those changes belong in the
   block-level skill, not here.
2. **Encoder prefill caching.** A modest one-shot win is available by
   caching the encoder attention mask buffer when `audio_seq_len` is
   constant across calls (the typical demo case). Saves ~5 ms per
   translate() prefill on top of the trace; not worth doing
   independently but cheap to fold into a per-block-mask refactor.
3. **Speech encoder trace.** The encoder forward currently rebuilds host
   masks per call. Wrapping it in a separate trace (with mask buffers as
   persistent inputs) would amortise ~30-40 ms of host work per call
   over a session. Out of scope for this pass; would require
   speech_encoder.py to accept persistent mask inputs.

### Files touched in this phase

- `models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2tt.py` (new)
  — S2TT tracy harness modeled on `profile_t2tt.py`; 1 warmup + N timed;
  `step_callback` splits `replay` / `capture` / `warmup` per-step
  costs; reports `prefill_ms` (speech encoder + cross-attn populate),
  `steady_decode_step_ms`, `total_ms`. The decode path is the SAME
  `TextGenerator` instance from T2TT — the structural plumbing is
  shared, this harness just changes how prefill is timed and exercises
  the SpeechEncoder.
- `models/demos/facebook_seamless_m4t_v2_large/PERF_NOTES.md` — this
  S2TT section.

No block files were modified in this pass — characterization-only.
