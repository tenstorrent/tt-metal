# SpeechT5 Voice Conversion (VC) on Tenstorrent

This directory contains an initial SpeechT5 voice-conversion bring-up path using TTNN.

## Stage-1 Status (As Of Now)

Current status: **Partially complete**.

The implementation path exists end-to-end, but Stage-1 bounty acceptance is **not yet fully satisfied**
until cloud validation/performance/accuracy checks are completed.

### Stage-1 Checklist

- [x] Speech pre-net input processing path is integrated (currently via HF reference path in the demo).
- [x] Shared encoder is executed on TTNN.
- [x] Shared decoder with speaker conditioning is executed on TTNN.
- [x] Speech post-net is executed on TTNN.
- [x] HiFi-GAN vocoder path is integrated (CPU reference path).
- [ ] Confirmed run on N150/N300 with no errors (pending cloud run).
- [ ] Throughput >= 30 tokens/s (pending cloud perf measurement).
- [ ] RTF < 0.5 for typical 3-5s utterances (pending cloud perf measurement).
- [ ] Speaker similarity > 70% cosine (pending cloud validation run).
- [ ] Content preservation WER < 3.0 (pending cloud validation run).
- [ ] Token-level accuracy > 95% vs PyTorch reference (pending cloud validation run).
- [ ] Final audio quality verification report (pending listening + objective checks).
- [x] Setup and run instructions are documented.

## Current Stage-1 Bring-up Scope

Pipeline in `demo_ttnn.py`:

1. Speech encoder prenet (`feature_encoder + feature_projection + positional conv/sinusoidal`) on CPU (HuggingFace reference)
2. Shared SpeechT5 encoder on TTNN (`TTNNSpeechT5Encoder.forward_from_hidden_states`)
3. Shared SpeechT5 decoder on TTNN (with speaker conditioning)
4. Speech decoder postnet on TTNN
5. HiFi-GAN vocoder on CPU

This gets the VC flow running end-to-end and provides performance counters (`token/s`, `TTFT`, `RTF`) for iteration.

## Requirements

From `tt-metal` root:

```bash
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
```

You also need:

- `transformers`
- `datasets`
- `soundfile`
- access to HuggingFace model weights (`microsoft/speecht5_vc`, `microsoft/speecht5_hifigan`)

## Run

Input audio must be mono 16 kHz WAV.

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_vc/demo_ttnn.py \
  --input_wav /path/to/source_16khz_mono.wav \
  --output converted.wav \
  --l1_small_size 24576 \
  --num_command_queues 2 \
  --max_steps 800 \
  --perf_report speecht5_vc_perf.json
```

Optional target speaker embedding:

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_vc/demo_ttnn.py \
  --input_wav /path/to/source_16khz_mono.wav \
  --speaker_embedding /path/to/target_xvector.npy \
  --output converted.wav
```

If `--speaker_embedding` is omitted, a default CMU-Arctic x-vector is used.

## Cloud Validation (Stage-1 Closeout)

Use this script to generate a Stage-1 validation report JSON with pass/fail checks.

Script:

- `models/experimental/speecht5_vc/validate_stage1.py`

Example:

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_vc/validate_stage1.py \
  --input_wavs /path/a.wav /path/b.wav \
  --output_dir ./vc_stage1_outputs \
  --l1_small_size 24576 \
  --num_command_queues 2 \
  --retry_on_l1_clash \
  --report_json ./vc_stage1_report.json
```

Optional transcript-aware WER mode (if you already have reference text):

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_vc/validate_stage1.py \
  --input_wavs /path/a.wav /path/b.wav \
  --reference_texts "first reference sentence" "second reference sentence" \
  --report_json ./vc_stage1_report.json
```

Report includes:

- throughput (`token_per_sec`)
- real-time factor (`rtf`)
- token-level parity proxy (`token_accuracy_percent`, frame-cosine based)
- speaker similarity (`speaker_cosine`)
- content preservation (`wer_percent`)
- threshold checks + overall pass flag

Useful stability flags on cloud:

- `--l1_small_size 24576` (default): lower L1 reservation to reduce circular-buffer clashes.
- `--retry_on_l1_clash`: automatically retries once with smaller L1 size if a clash is detected.
- `--disable_program_cache`: useful for debugging cache-related runtime issues.

## Output

- Converted waveform at 16 kHz
- Console summary with `steps`, `token/s`, `TTFT`, `RTF`
- Optional JSON perf report via `--perf_report`

## Next Steps

Planned optimization work:

1. Run cloud validation on N150/N300 and attach logs.
2. Use `validate_stage1.py` report outputs as Stage-1 closeout evidence.
3. Generate and attach perf report with throughput + RTF.
4. Port speech encoder prenet fully to TTNN (if required by final acceptance interpretation).
5. Stage-2 optimization: sharding/trace/cache and fused-op tuning.
