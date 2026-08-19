# Japanese STT (Whisper large-v3-turbo, N150)

From the tt-metal repo root, after `source python_env/bin/activate`.

Turbo is multilingual (4 decoder layers). Pass `language=ja`. Do **not** use `distil-whisper/distil-large-v3` for Japanese (English-only). Full `openai/whisper-large-v3` is the vanilla JP demo in `japanese_demo_readme.md`.

Weights (once):

```bash
hf download openai/whisper-large-v3-turbo
```

Sample wav: `sample_ja.wav` (same clip as large-v3). The script resamples to 16 kHz. Transcription is printed as `TEXT: ...`.

Assuming WH-Galaxy, one N150, device 2. `TT_VISIBLE_DEVICES` remaps that chip to logical 0.

```bash
MESH_DEVICE=N150 TT_VISIBLE_DEVICES=2 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto \
PYTHONPATH=$PWD \
python models/demos/audio/whisper/demo/run_jp_stt_turbo.py \
  --wav models/demos/audio/whisper/demo/sample_ja.wav \
  --warmup
```
