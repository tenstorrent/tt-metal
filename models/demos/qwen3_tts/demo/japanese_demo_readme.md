# Japanese TTS (Qwen3-TTS)

From the tt-metal repo root, after `source python_env/bin/activate`.

Pass `--language japanese`. Weights: full HF snapshot (including `speech_tokenizer/config.json`), not safetensors-only.

Sample text: `sample_ja.txt`. Speaker ref: `jim_reference.wav`.

## N150 (1.7B)

```bash
MESH_DEVICE=N150 \
python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
  --text "$(cat models/demos/qwen3_tts/demo/sample_ja.txt)" \
  --language japanese \
  --ref-audio models/demos/qwen3_tts/demo/jim_reference.wav \
  --output /tmp/qwen3-tts-1.7B.wav
```

0.6B: add `--hf-id Qwen/Qwen3-TTS-12Hz-0.6B-Base --output /tmp/qwen3-tts-0.6B.wav`.

## N300 TP=2 (1.7B and 0.6B)

Needs a 2-chip N300 (or equivalent mesh). Weights are sharded with
``ShardTensorToMesh`` (``tt/mesh_utils.py``). Prints RTF and writes both wavs.

```bash
MESH_DEVICE=N300 \
python models/demos/qwen3_tts/demo/demo_jp_tts_tp2.py \
  --output-dir /tmp
```

Outputs:

- `/tmp/qwen3-tts-1.7B-tp2.wav`
- `/tmp/qwen3-tts-0.6B-tp2.wav`
