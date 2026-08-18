# Japanese TTS (Qwen3-TTS, N150)

From the tt-metal repo root, after `source python_env/bin/activate`.

Do not use English-only defaults. Pass `--language japanese`.

Weights (once):

```bash
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

Need the full download (including `speech_tokenizer/config.json`), not a `*.safetensors`-only copy.

Sample text: `sample_ja.txt` (`こんにちは。今日はいい天気ですね。`). Speaker ref: `jim_reference.wav`.

Assuming WH-Galaxy, one N150, device 0. `TT_VISIBLE_DEVICES` remaps that chip to logical 0, so `--device-id` stays `0`.

```bash
MESH_DEVICE=N150 TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto \
PYTHONPATH=$PWD \
python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
  --text "$(cat models/demos/qwen3_tts/demo/sample_ja.txt)" \
  --language japanese \
  --ref-audio models/demos/qwen3_tts/demo/jim_reference.wav \
  --output /tmp/jp_tts.wav \
  --device-id 0
```
