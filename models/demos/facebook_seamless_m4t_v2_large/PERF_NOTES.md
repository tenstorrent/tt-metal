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
