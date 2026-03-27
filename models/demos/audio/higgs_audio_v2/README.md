# Higgs Audio v2

This public Higgs Audio v2 bring-up is intentionally small:

- single-device Wormhole
- `batch_size=1`
- greedy audio generation only
- public modes: TTS, voice clone, and multi-speaker dialog
- public validation through two end-to-end hardware tests

Measured performance for the current minimal suite is recorded in [PERF.md](/vol/stor-vol-1/loc/tt-metal/models/demos/audio/higgs_audio_v2/PERF.md).

## Environment

Run every command from the repo root and set the runtime environment first:

```bash
export TT_METAL_HOME=/vol/stor-vol-1/loc/tt-metal
export HIGGS_AUDIO_REPO=/abs/path/to/higgs-audio
export PYTHONPATH=$HIGGS_AUDIO_REPO:$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}
```

Use the repo-local Python environment and in-repo TTNN build:

```bash
python_env/bin/python - <<'PY'
import os
import sys
import ttnn
print(sys.executable)
print(os.environ["TT_METAL_HOME"])
print(ttnn.__file__)
PY
```

The public runtime expects the upstream Higgs repo to be first on `PYTHONPATH`. It also needs access to:

- `bosonai/higgs-audio-v2-generation-3B-base`
- `bosonai/higgs-audio-v2-tokenizer`

## Assets

Fetch the pinned reference clips once:

```bash
python_env/bin/python models/demos/audio/higgs_audio_v2/demo/fetch_reference_audio_assets.py \
  --reference-audio-manifest models/demos/audio/higgs_audio_v2/demo/reference_audio_manifest.json
```

## Demo

TTS:

```bash
python_env/bin/python models/demos/audio/higgs_audio_v2/demo/demo.py \
  --model-path bosonai/higgs-audio-v2-generation-3B-base \
  --audio-tokenizer-path bosonai/higgs-audio-v2-tokenizer \
  --transcript "Please welcome everyone to the review and keep the tone calm and clear." \
  --out-path generated/higgs_audio_v2/demo_tts.wav
```

Voice clone:

```bash
python_env/bin/python models/demos/audio/higgs_audio_v2/demo/demo.py \
  --model-path bosonai/higgs-audio-v2-generation-3B-base \
  --audio-tokenizer-path bosonai/higgs-audio-v2-tokenizer \
  --transcript "This public demo keeps a real voice reference and runs on TT hardware." \
  --ref-audio /root/.cache/tt-metal/higgs_audio_v2_reference_audio/boson-ai-higgs-audio-8b1539a02d5764317724a904a344a0f1be8a736e/voice_prompts/en_woman.wav \
  --ref-transcript "The device would work during the day as well, if you took steps to either block direct sunlight or point it away from the sun." \
  --out-path generated/higgs_audio_v2/demo_clone.wav
```

Dialog:

```bash
python_env/bin/python models/demos/audio/higgs_audio_v2/demo/demo.py \
  --model-path bosonai/higgs-audio-v2-generation-3B-base \
  --audio-tokenizer-path bosonai/higgs-audio-v2-tokenizer \
  --messages-json models/demos/audio/higgs_audio_v2/demo/fixtures/multi_speaker_prompt.json \
  --reference-audio-manifest models/demos/audio/higgs_audio_v2/demo/reference_audio_manifest.json \
  --out-path generated/higgs_audio_v2/demo_dialog.wav
```

## Validation

Accuracy validation is the real TTNN-vs-PyTorch parity gate:

```bash
python_env/bin/python -m pytest models/demos/audio/higgs_audio_v2/tests/test_accuracy.py -q
```

Performance validation is the traced TTNN throughput and RTF gate:

```bash
python_env/bin/python -m pytest models/demos/audio/higgs_audio_v2/tests/test_performance.py -q
```

Run both together with:

```bash
python_env/bin/python -m pytest models/demos/audio/higgs_audio_v2/tests -q
```

The public thresholds enforced by the tests are:

- token accuracy against the PyTorch reference `>= 0.95`
- autoregressive tokens/s `>= 60`
- RTF `< 0.5`

## Notes

- The public demo keeps the CLI minimal on purpose. There is no sampling path and no multi-device path.
- The demo decodes up to `256` audio steps by default and will stop earlier if the model emits audio EOS.
- The validation cases use short real reference-audio windows for clone and dialog prompts so the public accuracy path stays device-runnable.
- If the board gets wedged after a failed run, recover with `tt-smi -r 0`.
