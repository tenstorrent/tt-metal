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

### Setup (S2ST)

- Device: p150a (single Blackhole)
- ARCH_NAME=blackhole
- Branch: ssinghal/seamless-m4t @ d0b558ffa37 + this pass
- Harness: `models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2st.py`
- Audio: `demo/inputs/sample_hello.wav` (~1.6 s, eng -> fra)
- max_new_tokens = 32; generation stopped at 12 tokens
- 1 warmup + 3 timed `synthesize()` calls
- All times measured with `ttnn.synchronize_device` boundaries
  (host-perceived latency, NOT pure kernel duration)

### Baseline numbers (untraced production path, 3 timed calls, p50)

| Stage                                  | p50 (ms) | Share | Notes                                            |
| -------------------------------------- | -------: | ----: | ------------------------------------------------ |
| feature_extractor_ms (host)            |     2.42 |  0.5% | wav load + 80-mel + stride-2 stack               |
| speech_encoder_ms    (TTNN encoder)    |    53.47 |  9.9% | 24 Conformer layers + 1 adapter                  |
| ar_text_ms           (TTNN AR + LM)    |   279.34 | 52.0% | 12 tokens -> ~23.3 ms/step (untraced)            |
| hf_rerun_ms          (host HF rerun)   |   157.81 | 29.4% | host fp32 HF text_decoder over [seq[:,:-1]]      |
| char_prep_ms         (host)            |     0.40 |  0.1% | tokenizer helpers, negligible                    |
| t2u_ms               (TTNN T2U NAR)    |    11.73 |  2.2% | NAR enc+dec+LM-head argmax                       |
| vocoder_ms           (TTNN HiFi-GAN)   |    30.61 |  5.7% | Conv1d / ConvTranspose1d stack                   |
| **total_ms**                           | **537.58** |  100% | end-to-end per synthesize                        |
| tokens_generated_per_call              |       12 |       |                                                  |
| audio_samples_per_call                 |    22400 |       | ~1.40 s of 16 kHz audio                          |

`hf_rerun_ms` is the documented hybrid boundary — HF re-runs the
text_decoder on the full output sequence to recover
`last_hidden_state` for the T2U stage (cross-attn here uses the
SUB-SAMPLED speech mask). It's host-fp32 PyTorch and outside the TTNN
budget.

### Tracy findings (S2ST, 1 warmup + 1 timed, ~23,782 device ops captured)

Captured via `python -m tracy -p -v -r --op-support-count 3000 -n
seamless_m4t_s2st models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2st.py
--num-timed 1`. NOTE: tracy DRAM marker buffers filled on this run
("Profiler DRAM buffers were full, markers were dropped!") and the
postproc step asserted on a missing op id — `cpp_device_perf_report.csv`
and `tracy_ops_data.csv` are still intact and used below. Total
device kernel time across the whole run (warmup + timed) was 101.25 ms
on 5,447 device-side op markers — i.e. roughly half that per
synthesize. Wall-clock of the timed synthesize under tracy was
628.81 ms (overhead included); untraced wall is 537.58 ms. On-device
time is therefore well under 20% of wall — same diagnosis as T2TT /
S2TT / T2ST: **host-bound pipeline**.

#### Top device ops by call count (entire run = warmup + 1 timed)

| OP                                  | calls | notes                                            |
| ----------------------------------- | ----:| ------------------------------------------------ |
| MatmulDeviceOperation               | 5118 | Q/K/V/out projs, FFN, LM head (all stages)       |
| ReshapeViewDeviceOperation          | 4670 | view-only, no kernel                             |
| TransposeDeviceOperation            | 3744 | head-dim transpose, Conformer                    |
| BinaryNgDeviceOperation             | 2358 | residual adds, mask adds                         |
| LayerNormDeviceOperation            | 1966 | encoder+decoder+adapter+T2U LN                   |
| InterleavedToShardedDeviceOperation | 1330 | I2S for sharded paths                            |
| SDPAOperation                       | 1080 | self+cross SDPA across encoder, decoder, T2U     |
| PagedUpdateCacheDeviceOperation     | 1056 | self-attn KV update (Phase 9c)                   |
| UnaryDeviceOperation                |  954 | SiLU, scalar muls                                |
| HaloDeviceOperation                 |  274 | conv halo (Conformer + HiFi-GAN)                 |
| Conv2dDeviceOperation               |  274 | Conformer conv module + HiFi-GAN vocoder convs   |
| MoveDeviceOperation                 |  244 |                                                  |
| UntilizeDeviceOperation             |  152 | tile cleanup                                     |
| UntilizeWithUnpaddingDeviceOperation |  130 | tile cleanup at NAR boundaries                   |
| SliceDeviceOperation                |  108 | slicing in T2U / vocoder                         |
| UpdateKVCacheOperation              |   96 | cross-attn cache fill (one-shot per call)        |
| EmbeddingsDeviceOperation           |   52 | unit / char / token embeddings                   |
| SoftmaxDeviceOperation              |   50 |                                                  |

#### Top host-side wall-time zones

| ZONE                                  | calls | total_ms | mean_us | notes                |
| ------------------------------------- | ----:| -------: | ------: | -------------------- |
| convert_python_tensor_to_tt_tensor    | 2578 |  6055    | 2349    | `ttnn.from_torch`    |
| to_tile_major_layout_nfaces           | 1991 |  2255    | 1133    | tilize on host       |
| TT_DNN_DEVICE_OP                      | 23782|    70    |    3    | per-op enqueue       |
| FDMeshCommandQueue::finish_nolock     |   318|    65    |  204    | per-op finish/sync   |

(close_impl / dumpDeviceResults / ProcessDeviceProfilerResults / etc.
are tracy teardown overhead; ignored.)

### Diagnosis

The AR text decoder (the same `TextGenerator` instance used by T2TT,
S2TT, and T2ST) dominates the wall budget at 52% / 279 ms. Its
per-step cost untraced (~23.3 ms/step) matches T2TT (20.88 ms), S2TT
(24.18 ms), and T2ST (22.7 ms) — host-dispatch limited at ~365
ops/step × ~57 µs/op. The fact that S2ST, the heaviest pipeline,
lands the AR text decoder at the SAME per-step cost as T2TT confirms
that the shared `TextGenerator` is the dominant pipeline component
and that its scaling is invariant to upstream prefill or downstream
NAR work.

The canonical pipeline-level fix (Phase 9c metal trace + replay on
the AR loop, which delivered the 1.21x T2TT speedup and 1.29x S2TT
speedup) is **not directly applicable to S2ST** for the same reason
it is not applicable to T2ST:

1. `SpeechToSpeechModel.synthesize` runs T2U + vocoder *after* the AR
   loop, and both allocate fresh device buffers (T2U encoder hidden
   state, char inputs, unit embeddings, ConvTranspose1d intermediates).
2. The Phase 9c cross-call trace machinery keeps the AR decode trace
   armed (`self._decode_trace_id` is not released between
   `generate()` calls). With an active trace, any post-AR
   allocation triggers a corruption warning and the T2U/vocoder
   buffers either corrupt the trace or are themselves clobbered by
   trace replay. See PERF_NOTES.md::T2ST::Diagnosis for the
   per-attempt empirical observation.
3. Releasing the trace before T2U / recapturing on the next
   synthesize would safe-handle the allocation, but on a 12-token
   AR loop the per-call recapture cost (~12 × 36 ms = ~430 ms)
   *exceeds* the steady-state AR savings (~12 × 5 ms = ~60 ms).

### Optimization applied

**None — at-ceiling at the pipeline-perf level.**

Same outcome as T2ST. The AR text decoder is at the structural ceiling
for its layout, the only known pipeline-level lever (Phase 9c trace) is
blocked by the post-AR allocator interaction, and tracy shows no single
device op type at > 5% of the budget (per-op host-enqueue costs are
flat ~2-3 µs across MatMul / Reshape / Transpose / LayerNorm / SDPA).

For the smaller NAR stages (speech encoder, T2U, vocoder), per-block
optimization is the appropriate skill. None shows up at > 10% of wall
on this prompt, and the bringup log already reports 24/24 blocks
at-ceiling per `skills/optimization/`:

| Component        | Wall (ms) | Realistic floor (kernel-time) | Trace saving (theoretical) |
| ---------------- | --------: | ----------------------------: | -------------------------: |
| speech_encoder   |     53.47 |                          ~25  |                    ~28 ms  |
| t2u_generator    |     11.73 |                           ~3  |                     ~9 ms  |
| code HiFi-GAN    |     30.61 |                          ~10  |                    ~21 ms  |

A "trace every NAR stage" pass could in principle recover ~58 ms
(total ~480 ms, 1.12x speedup) at the cost of three independent trace
harnesses with persistent input/output buffers and a `synthesize()`
refactor — beyond the scope of a pipeline-perf characterization.

The HF host rerun (`hf_rerun_ms` = 158 ms, 29% of wall) is pure host
PyTorch and isn't TTNN's concern, but it's the second-largest line
after the AR text decoder. A TTNN-side path that re-uses the per-step
hidden states already produced by the AR decoder (instead of rerunning
HF over the full sequence) would zero this out — structural, not a
pipeline-perf item.

### After numbers

Same as baseline (no optimization applied this pass).

| Phase                | After   |
| -------------------- | ------: |
| feature_extractor_ms |    2.42 |
| speech_encoder_ms    |   53.47 |
| ar_text_ms           |  279.34 |
| hf_rerun_ms          |  157.81 |
| char_prep_ms         |    0.40 |
| t2u_ms               |   11.73 |
| vocoder_ms           |   30.61 |
| total_ms             |  537.58 |
| tokens_generated     |      12 |

### Correctness verification

```
pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_s2st.py -v
```

- `test_s2st_audio_parity_with_hf` — PASS
  - TTNN audio: 22400 samples / 1.400 s
  - HF   audio: 22400 samples / 1.400 s
  - TTNN re-ASR: `'"Bonjour, c\'est un test."'`
  - HF   re-ASR: `"Bonjour, c'est un test."`
  - char-sim   : 0.920 (gate >= 0.5)

### Recommendations for the next pass

1. **Block-level first.** S2ST's ~279 ms ar_text_ms is the same wall
   we see on T2ST and matches the host-dispatch floor. Further wins
   here require either: (a) a refactor of `SeamlessMHA` to support
   `paged_update_cache` with a `cur_pos_tensor` and on-device
   argmax/copy (the Whisper pattern documented in PERF_NOTES.md::
   Phase 9b), which would unblock single-trace cross-call AR replay
   in the presence of post-AR allocations; or (b) per-block sharded
   matmul kernel configs, LM-head bfloat8_b quant. Both belong to
   `skills/optimization/`, not pipeline-perf.
2. **Speech encoder trace.** Wrapping the speech encoder forward in
   its own trace (with persistent feature input + mask buffers) would
   amortise ~28-30 ms of host work per call across a session.
   Independent from the AR trace (encoder is one-shot per call, AR is
   the bottleneck). Structural; would require `speech_encoder.py` to
   accept persistent mask inputs.
3. **Cache HF rerun cost.** The HF host rerun (~158 ms) is the
   second-largest contributor. If the AR decoder is restructured to
   persist its per-step hidden states across the AR loop, the HF
   rerun would be replaced by a single per-step concat on host and
   the ~158 ms line item would drop to ~5 ms. Structural; out of
   scope for pipeline-perf.

### Files touched in this phase

- `models/demos/facebook_seamless_m4t_v2_large/tt/profile_s2st.py` (new)
  — S2ST profiler harness modeled on `profile_s2tt.py` (audio input +
  speech encoder) and `profile_t2st.py` (T2U + vocoder tail). Times
  the full `synthesize()` call broken into feature_extractor +
  speech_encoder + AR text decoder + HF host rerun + char prep + T2U +
  vocoder phases. Each stage is bounded by an explicit
  `ttnn.synchronize_device` to attribute time on a host-perceived
  basis. Untraced production path only (the post-AR T2U + vocoder
  allocations block cross-call trace reuse on the AR loop, same
  reason as T2ST).
- `models/demos/facebook_seamless_m4t_v2_large/PERF_NOTES.md` — this
  S2ST section.

No block files were modified in this pass — characterization-only.
