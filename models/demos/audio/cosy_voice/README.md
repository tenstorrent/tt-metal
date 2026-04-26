# CosyVoice

## Platforms

- Wormhole single-device

## Introduction

This demo brings up the public CosyVoice 300M model family on TT-Metal with a minimal public surface.
The demo is scoped to the CosyVoice 1.0 model split:

- `CosyVoice-300M-SFT` for SFT mode
- `CosyVoice-300M` for zero-shot and cross-lingual modes
- `CosyVoice-300M-Instruct` for instruct mode

The implementation is organized around the real CosyVoice pipeline:

- semantic token generation LLM
- flow-based acoustic decoder
- HiFT vocoder

The pinned upstream reference repo is used for frontend processing and reference behavior:

- https://github.com/FunAudioLLM/CosyVoice

## Prerequisites

- Cloned `tt-metal`
- Installed repo-local TT runtime
- Cloned the official CosyVoice repo
- Downloaded CosyVoice checkpoints

The public runners expect a local CosyVoice checkout and local model directories. A typical layout is:

```text
/path/to/CosyVoice
/path/to/pretrained_models/CosyVoice-300M
/path/to/pretrained_models/CosyVoice-300M-SFT
/path/to/pretrained_models/CosyVoice-300M-Instruct
```

## Runtime Verification

Use the repo-local runtime before running the demo or validation:

```bash
python_env/bin/python - <<'PY'
import sys
import ttnn
print(sys.executable)
print(ttnn.__file__)
PY
```

## How To Run

### Demo

SFT:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/demo.py \
  --mode sft \
  --text '你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？' \
  --speaker-id '中文女' \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output /tmp/cosy_voice_sft.wav
```

Zero-shot:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/demo.py \
  --mode zero_shot \
  --text '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。' \
  --prompt-text '希望你以后能够做的比我还好呦。' \
  --prompt-audio /path/to/CosyVoice/asset/zero_shot_prompt.wav \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output /tmp/cosy_voice_zero_shot.wav
```

Cross-lingual:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/demo.py \
  --mode cross_lingual \
  --text '<|en|>And then later on, fully acquiring that company.' \
  --prompt-audio /path/to/CosyVoice/asset/cross_lingual_prompt.wav \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output /tmp/cosy_voice_cross_lingual.wav
```

Instruct:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/demo.py \
  --mode instruct \
  --text '在面对挑战时，他展现了非凡的勇气与智慧。' \
  --speaker-id '中文男' \
  --instruction "Theo 'Crimson', is a fiery, passionate rebel leader.<|endofprompt|>" \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output /tmp/cosy_voice_instruct.wav
```

### Validation

Accuracy manifest:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/validate_tt.py \
  --suite accuracy \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output-json /tmp/cosy_voice_accuracy.json
```

The accuracy suite measures semantic-token parity against deterministic greedy
reference targets under teacher forcing. This avoids treating the reference
model's sampled top-k outputs as a deterministic correctness oracle.

Quality manifest:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/validate_tt.py \
  --suite quality \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output-json /tmp/cosy_voice_quality.json
```

The quality suite measures:

- output waveform validity
- Whisper-based transcription WER against the case transcript
- CampPlus speaker-embedding similarity against the target speaker or prompt audio

Performance manifest:

```bash
python_env/bin/python models/demos/audio/cosy_voice/demo/validate_tt.py \
  --suite performance \
  --reference-repo /path/to/CosyVoice \
  --model-root /path/to/pretrained_models \
  --output-json /tmp/cosy_voice_performance.json
```

## Known Limitations

- This branch is single-device only.
- The public tree is intentionally small and excludes multi-device and streaming runtime variants.
- The frontend stays on the pinned upstream reference stack.
- The current public backend is `tt_semantic_tt_flow_frontend_length_regulator_torch_decoder_reference_vocoder`: the semantic LLM path is on TT, the flow acoustic frontend up through the learned length regulator is on TT, and the diffusion decoder loop is now implemented in-tree in torch parity form.
- The HiFT vocoder still runs on the pinned reference path.
- The public quality suite is real, but it evaluates the current mixed backend rather than a full end-to-end TT pipeline.
- Public metrics should only be published from benchmark-backed runs of the current backend split.
