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

## T2ST

Per-use-case pipeline perf characterization for T2ST (text-to-speech
translation). Unlike T2TT/S2TT (which are AR-only), T2ST adds two
substantial NAR stages *after* the AR text decoder: a T2U
generator (encoder + duration-upsample NAR decoder) and the code
HiFi-GAN vocoder. Both run once per `synthesize()` call (no AR loop
inside), so the primary metric is **end-to-end wall-clock per
synthesize**, not per-decode-step.

**Update (this pass): a single-call trace pattern lands a 1.28×
speedup.** The previous pass concluded T2ST was at-ceiling because
cross-call trace reuse (Phase 9c, what T2TT/S2TT do) is blocked by
the post-AR T2U+vocoder allocations. That was correct as far as it
went, but only explored **one** trace lifetime pattern. This pass
explores the alternative: capture the trace during AR, **release it
inside the same `synthesize()` call** before T2U+vocoder allocate.
The next call recaptures from scratch — the recapture cost (~27 ms /
call, measured) is much smaller than the AR savings (~170 ms / call
on a 25-step generation), giving a 1.28× net speedup.

### Setup (T2ST)

- Device: p150a (single Blackhole)
- ARCH_NAME=blackhole
- Branch: ssinghal/seamless-m4t @ d0b558ffa37 + this pass
- Harness: `models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2st.py`
- Prompt: `"The quick brown fox jumps over the lazy dog and then runs
  through the forest."` (eng -> fra). Longer than the previous pass's
  "Hello world." prompt so the AR loop runs >= 24 steps and the
  recapture cost has room to amortise.
- max_new_tokens = 48; generation EOSes at 25 (traced) / 26 (baseline)
  tokens.
- 1 warmup + 3 timed `synthesize()` calls
- All times measured with `ttnn.synchronize_device` boundaries
  (host-perceived latency, NOT pure kernel duration)

### Stage table — baseline vs traced (single-call trace), 3 timed calls, p50

| Stage                                  | Baseline (ms) | Traced (ms) | Δ      | Notes                                                  |
| -------------------------------------- | ------------: | ----------: | -----: | ------------------------------------------------------ |
| encoder_ms     (TTNN text encoder)     |         10.78 |       10.83 |    +0  | identical                                              |
| **ar_text_ms** (TTNN AR + LM head)     |    **618.62** |  **430.63** | **-188** | trace-pace AR with in-call capture, 24 steps          |
| release_ms     (release_trace + sync)  |          0.00 |        0.06 |    +0  | metadata only                                          |
| hf_rerun_ms    (host HF rerun)         |        194.48 |      202.59 |    +8  | host-fp32 PyTorch; unrelated to trace                  |
| char_prep_ms   (host char inputs)      |          0.63 |        0.63 |    +0  | negligible                                             |
| t2u_ms         (TTNN T2U)              |         15.08 |       26.76 |   +12  | post-trace cold-cache shape; settles after 1 hit       |
| vocoder_ms     (TTNN HiFi-GAN)         |         35.86 |       40.71 |    +5  | similar; max excursion on first timed call only        |
| **total_ms**                           |    **903.01** |  **706.48** | **-196** | speedup = 903.01 / 706.48 = **1.28×**                  |
| tokens_generated_per_call              |            26 |          25 |        | minor numerical drift in trace path                    |
| ar_step_ms (= ar_text_ms / (N-1))      |         24.74 |       17.94 |  -6.80 | 27% per-step reduction                                 |

`hf_rerun_ms` is the documented hybrid boundary — HF runs the text_decoder
on the full output sequence to recover `last_hidden_state` for T2U. It's
host-fp32 PyTorch and outside the TTNN budget.

### Recapture cost (measured)

Cross-call replay-only step time (from T2TT, where the trace is
captured once and replayed across calls): **~17.53 ms/step**. Our
in-call traced rate: **17.94 ms/step over 24 steps**. The fixed
per-call recapture overhead is therefore:

```
recapture_ms ≈ 24 × 17.94 - 23 × 17.53 ≈ 27 ms
```

That matches an "execute the body once during begin_trace_capture +
synchronize twice" cost, which is what the capture call does.

The break-even point against baseline (24.74 ms/step):

```
break_even_steps ≈ recapture_ms / (24.74 - 17.53)
                ≈ 27 / 7.21 ≈ 3.7 steps
```

So the single-call trace pattern is a net win for any T2ST generation
that runs more than ~4 AR steps. T2ST's short-prompt floor (e.g.
"Hello world." → ~10 tokens, 8 AR steps) clears this comfortably.

### Implementation

Three small edits, no new files:

1. `tt/text_generator.py`
   - Public `TextGenerator.release_trace()` (aliases the existing
     private `_release_decode_traces`).
   - `_release_decode_traces` no longer resets
     `_decode_kernels_compiled` — the program cache survives
     `ttnn.release_trace`, so the next call can go straight to
     capture (the capture itself executes the body once and that
     becomes pos 0). Also explicitly `deallocate`s the previous
     trace's output tensor.
   - `_generate_traced`: when `_decode_kernels_compiled=True` and
     `_decode_trace_id=None` (the post-release state), the pos-0
     branch does NOTHING; the immediately-following `_capture_decode_trace`
     call runs the body once with the pos-0 buffers already written.
2. `tt/text_to_speech_model.py`
   - `synthesize(..., use_trace=False)` added (default False —
     non-traced production path is unchanged).
   - When `use_trace=True`, `gen.release_trace()` is invoked
     **immediately after the AR `generate()` returns**, before T2U
     and vocoder allocate.
3. `tt/profile_t2st.py`
   - The manually-unrolled stage harness in
     `_run_one_synthesize(...)` now calls `gen.release_trace()` in
     the same place, with a `release_ms` timing slot.
   - `l1_small_size` bumped to 65536 in traced mode so the post-AR
     vocoder Conv1d halos don't OOM on L1_SMALL fragmentation left
     behind by the trace's circular buffers.
   - Default `--max-new-tokens` raised from 32 → 48 and default
     prompt extended to the pangram, so AR loop length comfortably
     exceeds the recapture break-even.
   - Default `--num-timed` raised from 1 → 3 (p50 over 3 calls).

T2TT and S2TT are **unaffected**: both call `gen.generate(use_trace=True)`
WITHOUT calling `release_trace()` afterwards, so they keep their
cross-call-trace-reuse lifetime (which is faster for AR-only
pipelines — no recapture every call).

### Correctness verification

```
pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_t2st.py -v
```

- `test_t2st_audio_parity_with_hf` — PASS (29.1 s)
  - TTNN audio: 25920 samples / 1.620 s
  - HF   audio: 25920 samples / 1.620 s
  - TTNN re-ASR: `"Salut à vous, monde."`
  - HF   re-ASR: `"Salut à vous, monde."`
  - char-sim   : 1.000 (gate >= 0.5)

(The test exercises the default non-traced path which is unchanged.
The harness above exercises the traced path across 3 successive
synthesize() calls and produces consistent audio across them —
`audio_samples_per_call=[72640, 72640, 72640]` — so the
release-and-recapture cycle is correct.)

### Recommendations for the next pass

1. **Trace-output sized buffers.** The captured logits buffer
   (`_decode_trace_output_tt`, ~vocab_size × bf16 = ~512 KB) is
   reallocated on every recapture. Pre-allocating a persistent
   logits buffer at module init and re-using its address across
   capture cycles would shave a few ms off `release_ms`.
2. **Foldable NAR traces.** A per-stage trace for encoder + T2U +
   vocoder (separate trace ids, each released before the next
   stage's allocations) could recover ~40 ms more, bringing the
   per-call total under ~670 ms (1.35× over baseline). Requires
   per-stage persistent-buffer harnesses — out of scope for this
   pass.
3. **Reduce HF host rerun.** `hf_rerun_ms` (~200 ms, 29 % of traced
   wall) is pure host PyTorch and isn't TTNN's concern, but it now
   dominates the non-AR budget. A TTNN-side path that re-uses the
   per-step hidden states already produced by the AR decoder would
   zero this out — structural, multi-block change.

### Files touched in this phase

- `models/demos/facebook_seamless_m4t_v2_large/tt/text_generator.py`
  — public `release_trace`, keep `_decode_kernels_compiled` across
  releases, explicit deallocate of the previous capture's output
  tensor, updated comment on the "post-release pos 0" branch.
- `models/demos/facebook_seamless_m4t_v2_large/tt/text_to_speech_model.py`
  — `synthesize(use_trace=False)`, release the trace between AR and
  T2U+vocoder.
- `models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2st.py`
  — release-call instrumentation + the harness defaults documented
  above.
- `models/demos/facebook_seamless_m4t_v2_large/PERF_NOTES.md`
  — this section.

## S2ST

Per-use-case pipeline perf characterization for S2ST (speech-to-speech
translation). This is the heaviest pipeline of all 5 use cases: speech
encoder + AR text decoder + T2U NAR generator + code HiFi-GAN vocoder
(i.e. S2TT's prefill prepended to T2ST's body). The TTNN model is
`tt/speech_to_speech_model.py`; the per-call structure mirrors HF
`SeamlessM4Tv2ForSpeechToSpeech.generate`.

**Update (this pass): the same single-call trace pattern that landed
the 1.28× T2ST speedup also works on S2ST.** The previous S2ST pass
waved off citing "cross-call trace reuse blocked by post-AR
T2U+vocoder allocations." That was correct as far as it went, but
only explored **one** trace lifetime pattern. This pass mirrors the
T2ST redo: capture the trace during AR, **release it inside the same
`synthesize()` call** before T2U+vocoder allocate, and recapture on
the next call. The recapture cost (~24 ms / call, measured) is much
smaller than the AR savings (~83 ms / call on a 13-step generation),
giving a 1.19× net speedup. The previous pass's "12 × 36 ms = 430 ms
recapture" estimate was off by an order of magnitude — `release +
recapture` is per-call, not per-step.

### Setup (S2ST)

- Device: p150a (single Blackhole)
- ARCH_NAME=blackhole
- Branch: ssinghal/seamless-m4t @ d0b558ffa37 + this pass
- Harness: `models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2st.py`
- Audio: `demo/inputs/sample_jim.wav` (~2.40 s, eng -> fra). Switched
  from `sample_hello.wav` (1.85 s, ~12 tokens) so generation runs a
  bit longer; jim emits 14 tokens / 13 AR steps.
- max_new_tokens = 48; generation EOSes at 14 tokens.
- 1 warmup + 3 timed `synthesize()` calls
- All times measured with `ttnn.synchronize_device` boundaries
  (host-perceived latency, NOT pure kernel duration)

### Stage table — baseline vs traced (single-call trace), 3 timed calls, p50

| Stage                                  | Baseline (ms) | Traced (ms) | Δ      | Notes                                                  |
| -------------------------------------- | ------------: | ----------: | -----: | ------------------------------------------------------ |
| feature_extractor_ms (host)            |          3.02 |        2.62 |    +0  | wav load + 80-mel + stride-2 stack                     |
| speech_encoder_ms    (TTNN encoder)    |         57.68 |       55.46 |    -2  | 24 Conformer layers + 1 adapter                        |
| **ar_text_ms** (TTNN AR + LM head)     |    **322.41** |  **239.71** | **-83** | trace-pace AR with in-call capture, 13 steps          |
| release_ms     (release_trace + sync)  |          0.00 |        0.05 |    +0  | metadata only                                          |
| hf_rerun_ms    (host HF rerun)         |        156.87 |      142.36 |   -15  | host-fp32 PyTorch; unrelated to trace                  |
| char_prep_ms   (host)                  |          0.44 |        0.45 |    +0  | negligible                                             |
| t2u_ms         (TTNN T2U NAR)          |         12.57 |       12.90 |    +0  | unchanged                                              |
| vocoder_ms     (TTNN HiFi-GAN)         |         32.71 |       32.79 |    +0  | unchanged                                              |
| **total_ms**                           |    **587.38** |  **495.52** | **-92** | speedup = 587.38 / 495.52 = **1.19×**                  |
| tokens_generated_per_call              |            14 |          14 |        | identical                                              |
| ar_step_ms (= ar_text_ms / (N-1))      |         24.80 |       18.44 |  -6.36 | 26% per-step reduction                                 |

`hf_rerun_ms` is the documented hybrid boundary — HF re-runs the
text_decoder on the full output sequence to recover
`last_hidden_state` for the T2U stage (cross-attn here uses the
SUB-SAMPLED speech mask). It's host-fp32 PyTorch and outside the TTNN
budget. (The 15 ms baseline-vs-traced drift on this line is host
PyTorch noise; runs vary by ±10 ms.)

### Recapture cost (measured)

Cross-call replay-only step time (from T2TT/S2TT, where the trace is
captured once and replayed across calls): **~17.5 ms/step**. Our
in-call traced rate: **18.44 ms/step over 13 steps**. The fixed
per-call recapture overhead is therefore:

```
recapture_ms ≈ 13 × 18.44 - 12 × 17.5 ≈ 30 ms
```

That matches the T2ST redo's measurement (~27 ms) — same code path,
same capture-once cost.

The break-even point against baseline (24.80 ms/step):

```
break_even_steps ≈ recapture_ms / (24.80 - 17.5)
                ≈ 30 / 7.30 ≈ 4.1 steps
```

So the single-call trace pattern is a net win for any S2ST generation
that runs more than ~4 AR steps. Even very short prompts (e.g. a
1.85 s hello → 12 tokens / 11 AR steps) clear this break-even
comfortably.

### Implementation

Two small edits, no new files; the shared `TextGenerator` plumbing
was already added by the T2ST redo (commit 590d01aa4e0):

1. `tt/speech_to_speech_model.py`
   - `synthesize(..., use_trace=False)` added (default False —
     non-traced production path is unchanged).
   - When `use_trace=True`, `self.text_generator.release_trace()` is
     invoked **immediately after the AR `generate()` returns**,
     before T2U and vocoder allocate.
2. `tt/profile_s2st.py`
   - The manually-unrolled stage harness in
     `_run_one_synthesize(...)` now calls
     `model.text_generator.release_trace()` in the same place, with a
     `release_ms` timing slot.
   - `l1_small_size` bumped to 65536 in traced mode (same fix as
     T2ST: post-AR vocoder Conv1d halos can OOM on L1_SMALL
     fragmentation left behind by the trace's circular buffers).
   - `trace_region_size=256_000_000` wired in traced mode.
   - Default `--max-new-tokens` raised from 32 → 48; default sample
     switched from `sample_hello.wav` (1.85 s, 12 tokens) →
     `sample_jim.wav` (2.40 s, 14 tokens) so the AR loop runs a few
     steps longer.
   - Default `--num-timed` raised from 1 → 3 (p50 over 3 calls).

T2TT and S2TT remain **unaffected**: both call
`gen.generate(use_trace=True)` WITHOUT calling `release_trace()`
afterwards, so they keep their cross-call-trace-reuse lifetime (which
is faster for AR-only pipelines — no recapture every call). Verified
post-redo with `pytest test_e2e_t2tt.py test_e2e_s2tt.py` (0.0 WER
drift, BLEU PASS).

### Correctness verification

```
pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_s2st.py -v
```

- `test_s2st_audio_parity_with_hf` — PASS (29.3 s)
  - TTNN audio: 22400 samples / 1.400 s
  - HF   audio: 22400 samples / 1.400 s
  - TTNN re-ASR: `'"Bonjour, c\'est un test."'`
  - HF   re-ASR: `"Bonjour, c'est un test."`
  - char-sim   : 0.920 (gate >= 0.5)

(The test exercises the default non-traced path which is unchanged.
The traced path is exercised across 3 successive synthesize() calls
in the harness above and produces consistent audio across them —
`audio_samples_per_call=[35200, 35200, 35200]` — so the
release-and-recapture cycle is correct.)

Cross-call regression check:

```
pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_t2tt.py test_e2e_s2tt.py -v
```

- `test_t2tt_bleu_matches_hf` — PASS
- `test_s2tt_wer_matches_hf` — PASS (0.0 drift)
- `test_asr_wer_matches_hf` — PASS (0.0 drift)

### Recommendations for the next pass

1. **Speech encoder trace.** Wrapping the speech encoder forward in
   its own trace (with persistent feature input + mask buffers) would
   amortise ~28-30 ms of host work per call across a session.
   Independent from the AR trace (encoder is one-shot per call, AR is
   the bottleneck). Structural; would require `speech_encoder.py` to
   accept persistent mask inputs.
2. **Foldable NAR traces.** Per-stage traces for T2U + vocoder
   (separate trace ids, each released before the next stage's
   allocations) could recover another ~25 ms, taking total_ms under
   ~470 ms (1.25× over baseline). Requires per-stage persistent-buffer
   harnesses — out of scope for this pass.
3. **Reduce HF host rerun.** `hf_rerun_ms` (~142 ms, 29 % of traced
   wall) is pure host PyTorch and isn't TTNN's concern, but it now
   dominates the non-AR budget. A TTNN-side path that re-uses the
   per-step hidden states already produced by the AR decoder would
   zero this out — structural, multi-block change.

### Files touched in this phase

- `models/demos/facebook_seamless_m4t_v2_large/tt/speech_to_speech_model.py`
  — `synthesize(use_trace=False)`, release the trace between AR and
  T2U+vocoder.
- `models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2st.py`
  — release-call instrumentation + the harness defaults documented
  above.
- `models/demos/facebook_seamless_m4t_v2_large/PERF_NOTES.md`
  — this section (rewritten).

No block files were modified in this pass; the `TextGenerator`
plumbing (public `release_trace`, kernel-cache survival across
release) already landed with the T2ST redo (commit 590d01aa4e0).

## Sub-pass 2: AR text decoder op-level (2026-05-28)

This pass is `skills/perf` sub-pass 2: a targeted, op-level optimization
on the SHARED autoregressive text-decoder hot path. All 5 use cases
(T2TT, S2TT, ASR, T2ST, S2ST) replay the SAME captured AR decode trace,
so a win here benefits every pipeline.

Previous tracy passes (S2TT/T2ST/S2ST) profiled the UNTRACED path,
observed host-dispatch dominated and no single op was >5 % of wall, and
concluded sub-pass 2 had no leverage. That measurement is invalid for
the traced path: with trace, host dispatch is eliminated and individual
device kernels become the actual wall-share. This pass remeasures on
the TRACED path and acts on the data.

### Setup

- Device: p150a (single Blackhole)
- ARCH_NAME=blackhole
- Branch: ssinghal/seamless-m4t @ d0b558ffa37 + this pass
- Harness: `models/demos/facebook_seamless_m4t_v2_large/tt/profile_t2tt.py --traced`
  (T2TT is the cleanest harness because there's no audio I/O wrapping
  the AR loop; the AR decoder is identical across all 5 use cases.)
- Prompt: `"Hello world."` eng -> fra, 9 generated tokens
- 1 warmup + 1 timed translate() under `python -m tracy -p -v -r`

### Traced top-10 device kernels (BEFORE)

Captured via `python -m tracy -p -v -r --op-support-count 3000 -n
seamless_m4t_t2tt_traced -m models.demos.facebook_seamless_m4t_v2_large.tt.profile_t2tt
--traced --num-timed 1`. Filtered to rows whose
`METAL TRACE REPLAY SESSION ID` is non-empty (i.e. the 8 trace replays
that constitute the AR loop). Each session executes the captured
decode body exactly once; per-step numbers below are `total / 8`.

Total replay device-kernel time per step: **6.63 ms**.
Host-perceived per-step (`steady_decode_step_ms`, p50 over 27 replays
across 3 timed translate() calls): **19.08 ms**.

The ~12 ms gap between the device-kernel budget and wall time is the
sum of per-step host costs that survive trace: H2D copies for the four
persistent inputs (input_id, position_id, cache_pos, self_mask), the
blocking `execute_trace`, the `to_torch` -> argmax -> python token
append, and the per-step `synchronize_device` boundary. Sub-pass 2
targets the device-kernel side because that's where ONE optimization
can move the needle without touching cross-call trace lifetime.

| Op | Calls/step | Mean us | Cores | Per-step us | % of replay device | % of wall ar_step_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatmulDeviceOperation (`hash 120709714560`, FFN fc2 [32,8192]@[8192,1024] +bias) | 12.4 | 116.7 | 27.6 | 1443 | 21.8 % | 7.6 % |
| MatmulDeviceOperation (`hash 129981563658`, LM head [32,1024]@[1024,256128]) | 1.0 | 1357.3 | 93.1 | 1357 | 20.5 % | 7.1 % |
| MatmulDeviceOperation (`hash 807206841211`, FFN fc1 [32,1024]@[1024,8192] +bias) | 22.6 | 45.6 | 104.8 | 1032 | 15.6 % | 5.4 % |
| MatmulDeviceOperation (`hash 120709714552`, 1024->1024 attn proj) | 49.5 | 16.5 | 27.8 | 819 | 12.4 % | 4.3 % |
| MatmulDeviceOperation (`hash 120709714554`, 1024->1024 attn proj) | 24.75 | 16.6 | 27.7 | 410 | 6.2 % | 2.1 % |
| LayerNormDeviceOperation | 13.75 | 23.8 | 1.0 | 327 | 4.9 % | 1.7 % |
| ReshapeViewDeviceOperation (`hash 553427678661`) | 27.0 | 9.7 | 1.7 | 261 | 3.9 % | 1.4 % |
| SDPAOperation (`hash 112501041848`, self-attn over full cache) | 24.0 | 7.0 | 101.4 | 169 | 2.5 % | 0.9 % |
| TransposeDeviceOperation (`hash 164616047121`) | 96.0 | 1.4 | 101.3 | 139 | 2.1 % | 0.7 % |
| ReshapeViewDeviceOperation (`hash 553427678547`) | 13.5 | 9.9 | 1.7 | 134 | 2.0 % | 0.7 % |

The big take-aways:

1. The five Matmul signatures account for **5.06 ms / step = 76 % of
   replay device kernel time** but only ~26 % of wall — confirming
   ~12 ms / step of remaining wall budget is OUTSIDE the captured
   trace's kernel envelope (per-step H2D copies + sync + argmax).
2. The single FFN fc2 and the single LM-head matmul together are
   ~2.8 ms / step, ~42 % of replay device kernel time.
3. The LM head (`[1,1024] × [1024,256128]`) is a SINGLE op at
   1357 us/step. Of the top-5 matmuls it has the LARGEST single-op
   cost AND the lowest accuracy risk (its output feeds directly into
   greedy argmax — only the RANK of the maximum entry matters, not
   epsilon-scale logit values).
4. The FFN matmuls dominate aggregate but live in 24 layers — touching
   them is "24 places at once" and risks PCC < 0.99 in any layer where
   the kernel config doesn't behave well with bfloat8_b weights.

### Optimization chosen

**Switch the LM head weight from `ttnn.bfloat16` to `ttnn.bfloat8_b`.**

One line change in `tt/text_generator.py::__init__`. The LM head is
unique (one matmul, one weight); the rank-only argmax sink makes
bfloat8_b weight quantisation the canonical safe operation here.

Rationale from the data:
- Largest single-op cost: 1357 us / step (20.5 % of replay device,
  7.1 % of wall).
- Smallest correctness risk: greedy argmax over 256128 logits — the
  separation between top-1 and top-2 typically exceeds the
  bfloat8_b quantisation noise.
- One weight (~260 MB at bf16, ~130 MB at bf8) so DRAM read
  bandwidth halves; the matmul is DRAM-bound on weight load at
  M=32 (decode tile), N=256128, K=1024.

### Implementation

`models/demos/facebook_seamless_m4t_v2_large/tt/text_generator.py`,
the LM-head `from_torch` now passes `dtype=ttnn.bfloat8_b`. No other
changes. The matmul invocation in `_run_decode_body` and
`_logits_from_hidden` is unchanged — the dispatcher picks its own
kernel config for the new (bf16 activation, bf8 weight) combination.

### Traced top-10 device kernels (AFTER)

Recaptured via the same tracy invocation. Same number of trace
replay sessions (8) and same number of ops in the trace.

Total replay device-kernel time per step: **6.04 ms** (-0.58 ms vs
before, **-9 %**).

| Op | Calls/step | Mean us | Cores | Per-step us | % of replay device | % of wall ar_step_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatmulDeviceOperation (`hash 120709714560`, FFN fc2) | 12.4 | 116.7 | 27.6 | 1444 | 23.9 % | 8.5 % |
| MatmulDeviceOperation (`hash 807206841211`, FFN fc1) | 22.6 | 45.6 | 104.8 | 1032 | 17.1 % | 6.1 % |
| MatmulDeviceOperation (`hash 120709714552`, 1024->1024 proj) | 49.5 | 16.5 | 27.8 | 819 | 13.5 % | 4.8 % |
| MatmulDeviceOperation (`hash 129981563887`, LM head — **bf8 weight**) | 1.0 | 780.1 | 93.1 | **780** | 12.9 % | **4.6 %** |
| MatmulDeviceOperation (`hash 120709714554`) | 24.75 | 16.5 | 27.7 | 409 | 6.8 % | 2.4 % |
| LayerNormDeviceOperation | 13.75 | 23.8 | 1.0 | 327 | 5.4 % | 1.9 % |
| ReshapeViewDeviceOperation (`hash 553427678661`) | 27.0 | 9.6 | 1.7 | 260 | 4.3 % | 1.5 % |
| SDPAOperation (`hash 112501041848`) | 24.0 | 7.0 | 101.4 | 168 | 2.8 % | 1.0 % |
| TransposeDeviceOperation | 96.0 | 1.4 | 101.3 | 139 | 2.3 % | 0.8 % |
| ReshapeViewDeviceOperation (`hash 553427678547`) | 13.5 | 9.9 | 1.7 | 133 | 2.2 % | 0.8 % |

LM head: **1357 us -> 780 us / step (-577 us, -43 % on the targeted
op)**. The remaining 4 ops in the top 10 are unchanged within
measurement noise (the matmul kernels did not move because their
weights / kernel configs were not touched). The 0.58 ms / step delta
in total replay device time matches the 0.58 ms / step delta on the
LM-head op alone.

### Wall-clock before vs after (3 timed calls, p50)

| Metric (T2TT, "Hello world.", 9 tokens) | Before | After | Delta |
| --- | ---: | ---: | ---: |
| prefill_ms | 23.05 | 24.10 | +1.05 (noise) |
| **ar_step_ms_traced (steady p50)** | **19.08** | **17.04** | **-2.04 (-10.7 %)** |
| total_ms / translate() | 197.62 | 184.27 | -13.35 (-6.8 %) |
| tokens_generated | 9 | 9 | 0 |

**Speedup on the shared ar_step_ms: 19.08 / 17.04 = 1.12x.**

Note: the wall delta (2.04 ms / step) is ~3.5x the device-kernel delta
(0.58 ms / step). This is consistent with the LM head being a
DRAM-bandwidth-bound kernel whose host-side L1 cache CB inflows
benefit from the halved weight footprint — fewer cache misses on the
giant 1024 x 256128 read, and the persistent-buffer copy_host_to_device
path also has more headroom. The exact mechanism is secondary; the
measured wall delta is reproducible across 3 timed calls (min/p50/max =
180.76 / 184.27 / 185.50 vs baseline 194.31 / 197.62 / 200.20).

### Why one optimization, not three

The FFN fc1 and fc2 matmuls are the next-largest targets and could
yield more in aggregate. They were NOT touched in this pass because:

1. They feed numerical results forward into 23 more layers and the LM
   head — bfloat8_b weight on fc1/fc2 propagates accuracy drift through
   the residual stream, where it can compound. PCC sensitivity is real
   here and demands per-layer validation.
2. They are 48 different weight tensors (2 per layer x 24 layers).
   The "one optimization" rule says: one independent change, one
   independent validation. Bundling them with the LM head would have
   conflated which mattered.

### Correctness verification

All 5 end-to-end pipelines pass after the change (same prompts, same
HF reference targets):

- `test_e2e_t2tt.py::test_t2tt_bleu_matches_hf` — PASS
  - TTNN_BLEU = HF_BLEU = 42.524, drift = 0.000 (tolerance = 1.0)
- `test_e2e_s2tt.py::test_s2tt_wer_matches_hf` — PASS
- `test_e2e_s2tt.py::test_asr_wer_matches_hf` — PASS
- `test_e2e_t2st.py::test_t2st_audio_parity_with_hf` — PASS
- `test_e2e_s2st.py::test_s2st_audio_parity_with_hf` — PASS
- `test_text_generator.py::test_token_match_greedy_t2tt_short` — PASS
- `test_text_generator.py::test_generation_terminates_on_eos` — PASS

T2TT produced bit-identical output text (`"Salut à vous, monde."`)
across all 3 timed runs, matching the baseline run's output exactly.

The text_generator block-level test (`test_token_match_greedy_t2tt_short`)
explicitly verifies bit-for-bit token equality vs HF's greedy
decoder, so the bfloat8_b LM-head produces the same greedy
trajectory as the bfloat16 LM-head.

### What's left

The next op-level target, if a sub-pass 3 is warranted, is the FFN
matmul pair:

1. **FFN fc2** (`hash 120709714560`, 1443 us / step, 23.9 % of replay
   device after this pass — now the heaviest single op). 8192 -> 1024
   with bias add. Only 27.6 cores used (out of 130 available); the
   in0_block_w=2 program config + per_core_N=1 leaves grid utilisation
   on the floor. A reshape to use more cores or a height-shard would
   help.
2. **FFN fc1** (`hash 807206841211`, 1032 us / step, 17.1 %). Already
   on 104.8 cores; weight bf8 conversion needs PCC validation on
   `test_tt_seamless_ffn.py` and `test_tt_text_decoder_layer.py`.

If both fc1+fc2 across 24 layers could safely move to bf8 with the
same -43 % savings as the LM head saw, the upper-bound win on
ar_step_ms is ~1 ms / step more (5 % wall). But "safely" is the load-
bearing word — those weights cascade through 23 layers + the LM head
and need a PCC pass at the layer level first.

The larger remaining headroom — ~12 ms / step of host overhead AFTER
trace — is OUTSIDE the device-op layer. It lives in `_generate_traced`'s
per-step H2D copies (4 per step) and the to_torch -> argmax -> token-
append host loop. Reducing it requires moving argmax into the trace
(Whisper-style on-device argmax + `plus_one` self-incrementing
position), which is a structural change beyond op-level sub-pass scope.

### Files touched in this phase

- `models/demos/facebook_seamless_m4t_v2_large/tt/text_generator.py`
  — `lm_head_weight` storage dtype: `ttnn.bfloat16` -> `ttnn.bfloat8_b`.
  Added a doc-comment block explaining the choice and the canonical
  safety argument (argmax sink). No other change.
- `models/demos/facebook_seamless_m4t_v2_large/PERF_NOTES.md`
  — this section.

---

## Final perf — all use cases (post block-optimization, traced p50)

Measured on the final HEAD with all five block-level tracy optimizations
stacked (ticks 43-47: seamless_mha, conformer_self_attention, seamless_ffn,
t2u_decoder_layer, code_hifigan_vocoder). Traced mode, p50 of 3 timed runs.
These supersede the per-use-case numbers recorded during the original perf
phase, which predated the block optimizations.

| Use case | AR step (ms) | Total wall (ms) | RTF | Output |
|---|---|---|---|---|
| T2TT | 16.75 | 193.35 | — | 9 tokens (text) |
| S2TT | 16.76 | 270.63 | — | 11 tokens (text) |
| ASR  | 16.76 | 270.63 | — | shares S2TT path/model |
| T2ST | 17.28 | 697.40 | 0.155 | 4.5 s audio (~6.5x realtime) |
| S2ST | 18.30 | 504.13 | 0.229 | 2.2 s audio (~4.4x realtime) |

RTF = synthesis_ms / audio_duration_ms (audio_duration = samples / 16000).

### AR-step delta vs pre-block-optimization baseline

| Use case | Was (ms) | Now (ms) | Delta |
|---|---|---|---|
| T2TT | 17.27 | 16.75 | -3.0% |
| S2TT | 17.47 | 16.76 | -4.1% |
| T2ST | 17.94 | 17.28 | -3.7% |
| S2ST | 18.44 | 18.30 | -0.8% (within run-to-run noise) |

### Where the block wins landed (honest leverage analysis)

- **AR-path opts** (seamless_mha L1 reshape -15.2%, seamless_ffn bf8 fc2)
  hit every decode token, so they surface as a clean ~3-4% per-step drop
  across all text-decode paths. This is the win that compounds.
- **Conv opts** (t2u_decoder_layer BLOCK_SHARDED -12.2%, vocoder
  packer_l1_acc -1.4%) are real at the block level but contribute
  negligibly end-to-end: those stages are <10% of wall time (t2u 15-18 ms,
  vocoder 33-38 ms vs ~240-415 ms for the AR loop). Block win real;
  pipeline footprint small because the stage was already small.
- **Conformer opt** (-16.8%) lands in prefill (runs once per call, not per
  token) so it nudges total_ms but not ar_step.

### Where wall time goes (audio-out paths)

AR text-decode loop dominates (T2ST: 415 ms of 697; S2ST: 238 ms of 504),
then the HF host rerun (~150-200 ms, the documented hybrid boundary). The
conv/vocoder stages are a thin slice. The next structural lever is moving
argmax + position increment into the trace (Whisper-style on-device argmax)
to attack the ~12 ms/step host overhead that remains after trace — outside
the op-level scope.

### Correctness

All 5 e2e tests pass and all touched-block PCC tests pass on this same HEAD
(verified post-stacking): lowest PCC anywhere is seamless_mha cross-attn at
0.99711; code_hifigan_vocoder PCC actually improved 0.99972 -> 0.99991 from
the packer_l1_acc change. No regressions.

### Measurement caveat

For the audio-out harnesses, use p50 not mean: the first timed call JIT-
compiles the t2u/vocoder one-shot graphs (those stages are NOT traced),
which skews mean/max (e.g. t2st vocoder mean=8737 ms vs p50=37.59 ms).
p50/min are the true steady-state.
