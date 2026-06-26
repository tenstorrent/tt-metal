# Qwen3-TTS OpenAI-compatible API server (Tenstorrent P300)

Wraps the `models/demos/qwen3_tts` Qwen3-TTS-12Hz-1.7B pipeline (branch `yito/qwen3-tts`)
in a FastAPI server exposing the OpenAI **`POST /v1/audio/speech`** contract, packaged as a
container that boots straight into the server on a Blackhole **P300**.

Mirrors the Qwen3-ASR server (`qwen3asr-server-build/`): dev-image base, ttnn wired to the
mounted tt-metal source, single device behind one lock, first-request warmup.

## Files
- `Dockerfile` — dev image + ffmpeg + server deps; p150 mesh descriptor (correct for the
  1-visible-chip P300), port 8003.
- `entrypoint.sh` — wire ttnn → `uvicorn ...qwen3_tts_server:app`.
- `build_ttmetal_tts.sh` — compile the `yito/qwen3-tts` worktree (produces `_ttnn.so`).
- `build_baked.sh` — optional self-contained image (no `/work` mount needed).
- Server module: `tt-metal-qwen3-tts/models/demos/qwen3_tts/server/qwen3_tts_server.py`.

## Build
```bash
# 1. tt-metal source (already done): git worktree at branch yito/qwen3-tts
#    /home/ttuser/ttwork/tt-metal-qwen3-tts
# 2. Compile tt-metal (long; produces _ttnn.so)
./build_ttmetal_tts.sh
# 3. Build the server image (mount-based)
docker build -t qwen3-tts-server:latest .
# 4. (optional) self-contained image
./build_baked.sh
```

## Run (P300 single chip)
```bash
docker run -d --device /dev/tenstorrent/0 \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /home/ttuser/ttwork/tt-metal-qwen3-tts:/work \
  -v /home/ttuser/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ttuser/ttwork/qwen3_tts_voices:/models/qwen3_tts_voices \
  -p 8003:8003 --cap-add ALL qwen3-tts-server:latest
```
If init fails with "Timed out while waiting for active ethernet core ...", reset the board
(`tt-smi -r`) and retry — a prior crashed run leaves cores in a bad state.

## Call (OpenAI Audio Speech)
```bash
curl localhost:8003/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"こんにちは、世界","voice":"default","response_format":"mp3"}' \
  --output out.mp3
```
Body fields: `input` (text, required), `model`, `voice`, `response_format`
(`mp3`|`opus`|`aac`|`flac`|`wav`|`pcm`), `speed` (0.25–4.0), and extension `language`
(default `japanese`). Also: `GET /health`, `GET /v1/audio/voices`.

## Voices (cloning)
Save speaker embeddings once from reference audio, then mount the directory; each
`<name>.safetensors` becomes a selectable `voice`:
```bash
# inside a dev container with the device mounted:
python models/demos/qwen3_tts/demo/demo_tts.py --backend tt \
  --ref_audio ref.wav --save_speaker /models/qwen3_tts_voices/alloy.safetensors \
  --text "テスト"
```
`voice: "default"` (or any unknown name) uses the model's built-in voice.
