# Higgs Audio v2 on TTNN

Higgs Audio v2 (`bosonai/higgs-audio-v2-generation-3B-base`) text-to-audio model running on
Tenstorrent Wormhole. The LLM backbone (prefill + autoregressive audio-token decode) runs on
the TTNN `HiggsAudioTTModel`; the DAC audio codec (token↔waveform) is ported to TTNN
(`tt/codec.py`) and the HF `HiggsAudioV2Processor` is reused for chat templating and I/O.

- Backbone: Llama-3.2-3B + **DualFFN** (per-token-type FFN), 28 layers, dim 3072, 24/8 heads.
- Audio head: 8 codebooks × codebook_size 1026, delay-pattern decode (24 kHz, 25 frames/s).
- Three generation modes: text-to-speech, voice cloning, multi-speaker dialog.

## Setup

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # offline weights at /data/hf_cache/higgs
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

### Sampling and on-device codec

Decoding uses Higgs's repetition-aware sampling (RAS): `--temperature 1.0 --top-k 50 --top-p 0.95`
(`--temperature 0` = greedy, which degenerates). Set `HIGGS_TTNN_CODEC=1` to run the ported TTNN
DacDecoder for token→waveform on-device (PCC 0.9997 vs HF); by default the codec runs on host.

### How generation works

1. `processor.apply_chat_template(...)` → `input_ids` ending at `<|audio_out_bos|>` (plus reference
   `audio_input_ids` for voice clone / multi-speaker).
2. TTNN `prefill_text` fills the KV cache. With audio in the prompt, the reference-audio embeddings
   are spliced into the `<|AUDIO_OUT|>` placeholders and routed through the **audio DualFFN branch**
   via a per-token mask blend (matches HF `HiggsAudioV2Model.forward`).
3. Free-running decode from the all-BOS frame at pos=S: per step, sample 8 per-codebook tokens
   (RAS), apply the delay-pattern state machine, feed back into `decode_step_audio` until the
   all-EOS row or `--max-new-tokens`.
4. The `[1, T, 8]` delay-patterned stream → `processor.batch_decode` (delay revert + codec) →
   silence trim → save.

## Functional results (RAS, trimmed)

| Mode | Duration | Terminates | Notes |
|---|---|---|---|
| TTS | 8.6 s | natural EOS | clean |
| Voice clone | 12.1 s | natural EOS | DualFFN-conditioned on the reference |
| Multi-speaker | 4.8 s | EOS @128 rows | distinct A/B voices, self-sim 0.27 (HF ref 0.20) |


---

# Performance report


| Field | Value |
|---|---|
| Device | **N300** (2 Wormhole chips, data-parallel) |
| Batch | 1 per stream (one stream per chip) |
| Decode precision | `performance`: FF1/FF3 bf4, FF2/WQKV/WO/KV bf8, audio-LM-head bf16 |
| Codec frame rate | 25 Hz → RTF = 25 / decode_tok_per_s |
| Measurement | block-traced on-device decode, 64 steps after 4 warmup |

## Headline — N300 (decode)

| Metric | Value |
|---|---|
| Aggregate throughput (2 streams) | **126.9 tok/s** |
| Amortized RTF | **0.197** (RTF<0.2 stretch goal met) |
| Per stream | 63.4 tok/s / RTF 0.394, 15.76 ms/step (2.00× linear scaling, zero CCL) |

`tests/test_perf_dp_n300.py`. Tensor-parallel was measured *slower* (35.5 tok/s, CCL-bound at
batch-1) — data-parallel (one independent stream per chip) is the correct multi-chip lever.

## End-to-end (LLM decode + TTNN codec), directly measured on N300

One process, traced decode + the ported TTNN codec timed together on the two chips
(`tests/test_perf_e2e_n300.py`): decode 15.74 ms/frame + codec 0.52 ms/frame →
**aggregate end-to-end RTF = 0.2113**. The codec adds ~0.5 ms/frame (codec-only RTF 0.012) — not a
bottleneck; the LLM decode dominates.

## Profiling (tracy `-r`, per-op device kernel time, per chip)

Per-step device-kernel time = **15.9 ms**: Matmul **76.6%** (12.21 ms), LayerNorm 4.8%, SDPA
(flash-attn decode) 4.7%, residual add 4.2%, rest ~9%. The matmul time matches `weight_bytes
(~2.17 GB) / effective DRAM BW (~178 GB/s)` → **decode is DRAM-bandwidth-bound**, not compute-bound
(M=1 per step), via the tracy per-op device profiler.

## Optimizations applied

- **Tracing**: full decode chained on-device in one `execute_trace`.
- **DualFFN**: decode runs a single FFN branch (no wasted compute); prefill blends both branches by token type.
- **Flash attention**: `scaled_dot_product_attention` decode kernel.
- **Sharded/fused flow**: DRAM-sharded matmuls, distributed fused RMSNorm, L1 width-sharded activations, paged KV cache (bf8).
- **Precision**: per-tensor bf8/bf4 weights at the accuracy cliff (FF1/FF3 bf4).
- **Multi-chip data-parallel**: independent stream per chip, zero CCL, linear scaling.
- **Codec on TTNN**: DAC decoder conv stack ported (`tt/codec.py`), PCC 0.998 vs HF.

## Tuning limits (measured negative results)

- **bf4 beyond FF1/FF3**: `bf4_all` hits 77.4 tok/s / RTF 0.323 but accuracy collapses to 0.62
  (`mlp_bf4` → 0.65). The audio argmax is razor-margin; bf4 is tapped out.
- **Wider matmul core grids** (32→48/64): *slower* (55.5 tok/s) — 12 DRAM banks already saturated; DRAM-channel bound.
- **Tensor-parallel across 2 chips**: *slower* (35.5 tok/s) — per-layer CCL costs more than the bandwidth at batch-1.
- **Single-stream RTF<0.2** (needs 125 tok/s on one chip): infeasible — even zeroing all non-matmul ops
  leaves the ~12.2 ms DRAM-bound matmul floor (RTF 0.31). Reachable only via batching/data-parallel.

## Reproduction

```bash
# N300 data-parallel decode: 126.9 tok/s aggregate, amortized RTF 0.197
HIGGS_PRECISION=performance pytest -s models/demos/audio/higgs_audio_v2/tests/test_perf_dp_n300.py
# end-to-end RTF (decode + codec), N300: 0.2113
pytest -s models/demos/audio/higgs_audio_v2/tests/test_perf_e2e_n300.py
# accuracy gate: 0.967
pytest -s models/demos/audio/higgs_audio_v2/tests/test_accuracy_native.py
# codec decode PCC vs HF: 0.9997
pytest -s models/demos/audio/higgs_audio_v2/tests/test_codec_e2e.py
```
