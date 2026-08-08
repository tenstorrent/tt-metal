# Higgs Audio v2 on TTNN

Higgs Audio v2 (`bosonai/higgs-audio-v2-generation-3B-base`) text-to-audio model running on
Tenstorrent Wormhole. The LLM backbone (prefill + autoregressive audio-token decode) runs on
the TTNN `HiggsAudioTTModel`. The **entire token→waveform codec is ported to TTNN** (`tt/codec.py`:
RVQ dequant via `ttnn.embedding` + per-codebook `project_out`, the `fc2` projection, and the DAC
decoder conv stack are all on-device) and is exercised end-to-end on real audio by
`test_perf_e2e_n300.py`; enable it with `HIGGS_TTNN_CODEC=1` (by default the codec runs on host). The
HF `HiggsAudioV2Processor` glue (chat templating, delay-pattern revert, token/file I/O — not codec
compute) stays on host.

### Model assets

The model directory is resolved with **no hard-coded path**:

1. `--model-dir` (demo) or `$HIGGS_MODEL_DIR` (all entry points) — a **pre-staged local directory**
   (offline, IRD containers, or a lab box's `/data` cache);
2. otherwise the model ID `bosonai/higgs-audio-v2-generation-3B-base` is **auto-fetched via
   `huggingface_hub.snapshot_download`** into the standard HF cache (`$HF_HOME` / `$HF_HUB_CACHE`;
   `HF_HUB_OFFLINE=1` resolves cache-only).

So in an IRD container, just set `HF_HOME=/localdev/...` (writable) or point `HIGGS_MODEL_DIR` /
`--model-dir` at a pre-staged copy — no source patch needed. Required layout of that directory:
`config.json`, the transformers-native `HiggsAudioV2ForConditionalGeneration` weights, and
`tokenizer/` (the `HiggsAudioV2TokenizerModel` codec).

- Backbone: Llama-3.2-3B + **DualFFN** (per-token-type FFN), 28 layers, dim 3072, 24/8 heads.
- Audio head: 8 codebooks × codebook\_size 1026, delay-pattern decode (24 kHz, 25 frames/s).
- Three generation modes: text-to-speech, voice cloning, multi-speaker dialog.

## Setup

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # optional: force cache-only (no network)
# Point at a pre-staged assets dir (skips the HF-hub fetch); omit to auto-fetch into $HF_HOME:
export HIGGS_MODEL_DIR=/path/to/higgs               # e.g. a writable cache in your container
```

## Demo — three modes

```bash
D=models/demos/audio/higgs_audio_v2/demo/demo.py

# 1) Text-to-speech
python $D --mode tts --text "Tenstorrent hardware now speaks." --out-dir ./out

# 2) Voice cloning (condition on a reference utterance)
python $D --mode voiceclone --ref-audio reference.wav --text "Spoken in the cloned voice." --out-dir ./out

# 3) Multi-speaker dialog (voice-conditioned per speaker, as Higgs intends)
python $D --mode multispeaker --ref-audio speakerA.wav --ref-audio-b speakerB.wav --out-dir ./out
```

Outputs are 24 kHz mono `.wav` (`higgs_<mode>.wav`). Multi-speaker conditions each speaker on a
reference clip (`--ref-audio` = A, `--ref-audio-b` = B) — text-only speaker tags are not the
Higgs format and produce repetitive output.

### Precision presets and decoding

**Sampling is the recommended decode method** (the default). Greedy is not used, for two reasons:

1. **Greedy does not self-terminate.** EOS is almost never the single most-likely token, so pure
   argmax never selects it — greedy *always* runs to `max_new_tokens` and must be length-capped +
   truncated. Sampling draws EOS from the tail
   of the distribution, so it stops naturally for typical seeds (the demo seed 1234 terminates at
   207 frames / \~7.6 s). An occasional "runaway" seed still fails to emit EOS and falls back to the
   same cap + truncation — but greedy needs that fallback for *every* input, sampling only rarely.
2. **Pure greedy collapses at bf8.** The audio-token argmax is razor-margin; a bf8-rounded marginal
   pick flips and the decode locks onto a single repeating frame → silence (measured: pure greedy
   `ras_win_len=0` @ bf8 → stuck row, rms 0.0, \~11 rows). Repetition-Aware Sampling (RAS, on by
   default) resamples the repeated codebooks and *does* keep greedy alive at bf8 (greedy+RAS @ bf8 →
   201 rows, non-silent) — but it still doesn't fix the EOS problem in (1), so we don't use it.

Two precision presets via `--precision` (default `performance`), a linked pair with sampling:

- **`performance`** (bf8/bf4 weights, the default, fast). Sampling's occasional 2nd-place pick makes
  the razor-margin argmax flip harmless → clean audio, and it self-terminates for typical seeds.
- **`accuracy`** (bf16 + HiFi4, higher precision, slower). The argmax is faithful; greedy would also
  produce clean audio here, but is still avoided because it does not self-terminate (see above).

Decoding runs the **fully on-device sampler by default**: forward + temperature + top-k + gumbel-max
draw + delay-pattern stuffing + feedback all run on TTNN; the host reads back only the 8 chosen
tokens per step (no per-step logit transfer, no host sampling). Pass `--host-sample` for the hybrid
path (host reads logits and samples in Python each step).

Set `HIGGS_TTNN_CODEC=1` to run the fully on-device TTNN codec for the token→waveform step (RVQ
dequant + `fc2` + DAC decoder conv stack); by default the codec runs on host.

### Decode execution modes: device vs mixed (blocking vs non-blocking)

The autoregressive decode runs in one of two modes. They trade *where sampling happens* against *how
the host and device synchronize each step* — the audio quality is identical either way.

| Mode | Flag | Sampling | Readback / step | Dispatch | Clean wall-clock |
| ---- | ---- | -------- | --------------- | -------- | ---------------- |
| **Device** (default) | `ondevice_sample=True` | on device (top-k + temperature + gumbel-max + delay) | 8 tokens | **non-blocking** | ~55 tok/s / RTF 0.46 |
| **Mixed** | `--host-sample` (`ondevice_sample=False`) | on host (Python) | full logits `[8×1026]` | non-blocking | ~52 tok/s / RTF 0.478 |

**Device mode dispatches non-blocking by default.** The 8-token readback (`to_torch(cur_tokens)`) is
queued after the trace on the **same command queue**, so FIFO ordering guarantees it reads the fully
written `cur_tokens` — no race, no per-step `synchronize` barrier. This is ~4% faster than blocking and
produces bit-identical tokens (verified across seeds, including early-EOS). Set `HIGGS_ONDEV_BLOCKING=1`
to force the blocking path. Device-only compute is ~61 tok/s / RTF 0.41; the per-step gumbel-noise
upload + token readback bring the wall-clock to ~55 tok/s / RTF 0.46.

The one thing that does **not** work is *overlapping* the readback with the next step (snapshotting
`cur_tokens` into a side buffer to run concurrently) — that races the trace's in-place write and
corrupts frames; a 2nd command queue orders it but costs more than it saves. The default doesn't
overlap — it only drops the redundant barrier — so it stays correct.

**Mixed mode** reads back a **write-once trace output** (`logits`) rather than the self-mutating
feedback buffer — the loop is closed on the host — so it has no hazard either and is used by the perf
gate as the stable reference (~52 tok/s / RTF 0.478).

**Why device mode still returns to the host every step:** a captured trace has no data-dependent
control flow, so it cannot stop itself at the natural EOS — the host must read the 8 tokens to detect
end-of-stream and break the loop. (It also streams fresh gumbel noise each step, since `ttnn.rand` is
baked at trace-capture time, and advances the KV position.) Only the EOS stop is fundamental; the noise
and position could move to on-device counters, and decoding K steps between host barriers (chunked
decode) is the natural next optimization. Neither mode changes the single-stream floor: decode is
DRAM-bandwidth-bound (~12 ms/step to read the 2.17 GB of weights), so one stream is bounded around
RTF 0.4–0.5 end-to-end regardless of mode.

### How generation works

1. `processor.apply_chat_template(...)` → `input_ids` ending at `<|audio_out_bos|>` (plus reference
   `audio_input_ids` for voice clone / multi-speaker).
2. TTNN `prefill_text` fills the KV cache. With audio in the prompt, the reference-audio embeddings
   are spliced into the `<|AUDIO_OUT|>` placeholders and routed through the **audio DualFFN branch**
   via a per-token mask blend (matches HF `HiggsAudioV2Model.forward`).
3. Free-running decode from the all-BOS frame at pos=S: by default the **fully on-device sampler**
   produces 8 per-codebook tokens/step (temperature + top-k + gumbel-max draw + delay-pattern
   stuffing + feedback all on device, only the tokens are read back); with `--host-sample` the host
   samples from read-back logits. Continues until the all-EOS row or `--max-new-tokens`.
4. The `[1, T, 8]` delay-patterned stream → `processor.batch_decode` (delay revert + codec) →
   silence trim → save.

## Functional results (RAS, trimmed)

| Mode          | Duration | Terminates    | Notes                                            |
| ------------- | -------- | ------------- | ------------------------------------------------ |
| TTS           | 7.6 s    | natural EOS   | seed 1234, 207 frames, clean                     |
| Voice clone   | 12.1 s   | natural EOS   | DualFFN-conditioned on the reference             |
| Multi-speaker | 4.8 s    | EOS @128 rows | distinct A/B voices, self-sim 0.27 (HF ref 0.20) |

Durations are per-seed: sampling self-terminates for typical seeds (the default seed 1234 does), but an
occasional runaway seed emits no EOS and is capped at `--max-new-tokens` (default 750) + trailing-EOS
truncated — the same fallback greedy needs for every input.

***

# Performance report

| Field            | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| Device           | **N300** — one Wormhole chip (batching amortizes on-chip)          |
| Batch            | 1 (single-stream RTF) or 2–8 (on-chip batched throughput)          |
| Decode precision | `performance`: FF1/FF3 bf4, FF2/WQKV/WO/KV bf8, audio-LM-head bf16 |
| Codec frame rate | 25 Hz → RTF = 25 / decode\_tok\_per\_s                             |
| Measurement      | block-traced on-device decode, 64 steps after 4 warmup             |

## Headline — single-stream, real generated audio (N300, one chip)

The honest per-utterance numbers, measured through the actual traced generator producing real audio:

| Metric                                                                      | Value          |
| --------------------------------------------------------------------------- | -------------- |
| Decode throughput (traced generator, real sampling)                         | **52.7 tok/s** |
| Decode RTF (single stream)                                                  | **0.475**      |
| End-to-end single-utterance RTF (prefill + decode + TTNN codec, real audio) | **0.490**      |

`tests/test_perf_e2e_n300.py`. **Stage-1 (RTF < 0.5) is met single-stream.** The device decode-step
ceiling (traced, on-device argmax) is 63.5 tok/s / RTF 0.394 / 15.76 ms/step.

## Multi-stream: on-chip batching *serving throughput* (NOT single-stream latency)

Decode is DRAM-bandwidth-bound — one \~2.17 GB weight read per step dominates. Batching B **different**
utterances into a single decode step amortizes that one read across all B streams, so aggregate
throughput climbs on a **single chip** (unlike data-parallel, which just runs independent model copies
on separate chips). Each stream is prefilled into its own KV-cache row and decoded in lockstep;
sampling, RAS, the delay-pattern state machine and EOS run per-stream on host, so streams terminate
independently (**ragged EOS**). Measured with B genuinely different prompts (`tests/test_perf_batch_n300.py`).

The batch ceiling on this box is DRAM capacity — a full 3B model already fills
\~10.4 GB, so `max_batch_size=32` (a full decode tile) does not fit alongside real-length KV. The tile
rule that the decode hidden's batch pad to 32 is met by padding the *fed* hidden while keeping
current_pos/KV at the real batch, so intermediate batches 2–31 work. Tensor-parallel (splitting *one*
stream across two chips) was measured *slower* (35.5 tok/s, CCL-bound at batch-1).

## Profiling (tracy `-r`, per-op device kernel time, per chip)

Per-step device-kernel time = **15.9 ms**: Matmul **76.6%** (12.21 ms), LayerNorm 4.8%, SDPA
(flash-attn decode) 4.7%, residual add 4.2%, rest \~9%. The matmul time matches `weight_bytes
(~2.17 GB) / effective DRAM BW (~178 GB/s)` → **decode is DRAM-bandwidth-bound**, not compute-bound
(M=1 per step), via the tracy per-op device profiler.

## Optimizations applied

- **Tracing**: full decode chained on-device in one `execute_trace`.
- **DualFFN**: decode runs a single FFN branch (no wasted compute); prefill blends both branches by token type.
- **Flash attention**: `scaled_dot_product_attention` decode kernel.
- **Sharded/fused flow**: DRAM-sharded matmuls, distributed fused RMSNorm, L1 width-sharded activations, paged KV cache (bf8).
- **Precision**: per-tensor bf8/bf4 weights at the accuracy cliff (FF1/FF3 bf4).
- **On-chip batching**: B different utterances share one weight read per decode step (aggregate 1.92×/3.40×/5.35× at B=2/4/8).
- **Codec ported to TTNN**: RVQ dequant + `fc2` + DAC decoder conv stack all on-device (`tt/codec.py`), exercised on real audio by `test_perf_e2e_n300.py`.

## Tuning limits (measured negative results)

- **bf4 beyond FF1/FF3**: `bf4_all` hits 77.4 tok/s / RTF 0.323 but accuracy collapses to 0.62
  (`mlp_bf4` → 0.65). The audio argmax is razor-margin; bf4 is tapped out.
- **Wider matmul core grids** (32→48/64): *slower* (55.5 tok/s) — 12 DRAM banks already saturated; DRAM-channel bound.
- **Tensor-parallel across 2 chips**: *slower* (35.5 tok/s) — per-layer CCL costs more than the bandwidth at batch-1.
- **Single-stream RTF<0.2** (needs 125 tok/s on one chip): infeasible — even zeroing all non-matmul ops
  leaves the \~12.2 ms DRAM-bound matmul floor (RTF 0.31). Only batching raises *aggregate* throughput
  (5.35× at B=8), never single-stream latency.

## Reproduction

```bash
# single-stream real-audio end-to-end RTF (traced generate + TTNN codec): decode-RTF 0.475, e2e 0.490
pytest -s models/demos/audio/higgs_audio_v2/tests/test_perf_e2e_n300.py
# on-chip batched serving throughput (B different prompts, ragged EOS): B=4 -> 3.40x aggregate tok/s
HIGGS_BATCH=4 pytest -s models/demos/audio/higgs_audio_v2/tests/test_perf_batch_n300.py
# accuracy gate: 0.967
pytest -s models/demos/audio/higgs_audio_v2/tests/test_accuracy_native.py
```
