# Japanese STT (Whisper large-v3, N150)

From the tt-metal repo root, after `source python_env/bin/activate`.

Do **not** use `distil-whisper/distil-large-v3` for Japanese. Use `openai/whisper-large-v3` and `language=ja`.

Weights (once):

```bash
hf download openai/whisper-large-v3
```

Sample wav we ran: `sample_ja.wav` (2s, 24 kHz). The script resamples to 16 kHz. Transcription is printed as `TEXT: ...`.

Assuming WH-Galaxy, one N150, device 2. `TT_VISIBLE_DEVICES` remaps that chip to logical 0.

```bash
MESH_DEVICE=N150 TT_VISIBLE_DEVICES=2 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto \
PYTHONPATH=$PWD \
python models/demos/audio/whisper/demo/run_jp_stt.py \
  --wav models/demos/audio/whisper/demo/sample_ja.wav \
  --warmup
```
