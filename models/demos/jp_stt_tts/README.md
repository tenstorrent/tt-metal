# Japanese STT , TTS (tt-metal, N150)

From the tt-metal repo root, after `source python_env/bin/activate`.


Checked-in inputs we ran:

- `sample_ja.txt` — TTS prompt (`こんにちは。今日はいい天気ですね。`)
- `sample_ja.wav` — TTS output / STT input (2s, 24 kHz)
- `refs/jim_reference.wav` — TTS speaker ref

## Weights

```bash
./models/demos/jp_stt_tts/download_weights.sh
```

## TTS — Qwen3-TTS, Japanese

# Assuming we are running on WH-Galaxy and using only device 0

```bash
MESH_DEVICE=N150 TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto \
PYTHONPATH=$PWD \
python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
  --text "$(cat models/demos/jp_stt_tts/sample_ja.txt)" \
  --language japanese \
  --ref-audio models/demos/jp_stt_tts/refs/jim_reference.wav \
  --output /tmp/jp_tts.wav \
  --device-id 0
```

Female ref: `--ref-audio models/demos/jp_stt_tts/refs/female_reference_24k.wav --ref-text "$(cat models/demos/jp_stt_tts/refs/female_reference.txt)"`

## STT — Whisper large-v3, Japanese

Note : Do not use distil-whisper for Japanese.

# Assuming we are running on WH-Galaxy and using only device 2


```bash
MESH_DEVICE=N150 TT_VISIBLE_DEVICES=2 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto \
PYTHONPATH=$PWD \
python models/demos/jp_stt_tts/run_jp_stt.py --wav models/demos/jp_stt_tts/sample_ja.wav --warmup
```
