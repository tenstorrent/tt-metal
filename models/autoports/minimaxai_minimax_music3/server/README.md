# MiniMax-Music3 HTTP server (tt-model `tt-dit-server` kind)

`server/app.py` is a FastAPI app (ASGI attribute `app`) that puts the whole MiniMax-Music3 pipeline (`tt/pipeline.py`,
optimized dtype policy) behind the OpenAI-style audio endpoint the model card documents for SGLang-Omni.

## Launch

```bash
source ~/mm3-bringup/common.sh
cd $MM3_WT && HF_MODEL=$MM3_WEIGHTS MM3_MESH_SHAPE=1x1 $MM3_PY -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app
```

This is exactly how tt-model-manager's `TtDitServerLauncher` starts a `tt-dit-server` app inside its container
(`python -m uvicorn --host 0.0.0.0 --port <port> --lifespan on <module:attr>`). The server is ready when uvicorn prints
`Application startup complete`: by then the weights are resolved, the chip is open, the pipeline is loaded and every trace
is warm (a 2 s warm-up song has been generated).

Environment:

| variable | meaning | default |
|---|---|---|
| `HF_MODEL` | HF repo id (`MiniMaxAI/MiniMax-Music3`, downloaded with `snapshot_download` minus the raw `.pth` / `qwen_7B` files) or a local snapshot directory | `$MM3_WEIGHTS` |
| `MM3_MESH_SHAPE` | mesh shape the launcher resolved (`RxC`); v1 is single-chip and always opens a 1x1 mesh on device 0, the value is logged. The tt-model manifest must set `runtime.mesh_shape_env: MM3_MESH_SHAPE`, otherwise the launcher exports `FLUX2_MESH_SHAPE` and the app falls back to `1x1` | `1x1` |
| `MESH_DEVICE` | board name from the launcher (`P150` / `P300` / `P300x2`), logged and reported by `/health` | unset |
| `MM3_REQUEST_TIMEOUT_S` | per-request timeout; the request gets a 504 but the device finishes the song before the next one starts | `1800` |
| `MM3_DTYPE_POLICY` | `tt/pipeline.py` preset (`optimized` / `functional`) | `optimized` |
| `MM3_WARMUP_SECONDS`, `MM3_WARMUP_STEPS` | the warm-up song | `2.0`, `4` |
| `MM3_TRACE_REGION_SIZE` | device trace region in bytes | `200000000` |
| `TT_DIT_CACHE_DIR` | DiT weight cache (`/weight-cache` in the container) | `generated/tt_dit_cache` |
| `MM3_HF_REVISION` | pin the HF revision when `HF_MODEL` is a repo id | `main` |

## Endpoints

`POST /v1/audio/speech` (JSON) -> `audio/wav` (44.1 kHz, stereo, 16-bit PCM):

| field | meaning |
|---|---|
| `model` | `MiniMaxAI/MiniMax-Music3` (optional; anything else -> 404) |
| `input` | lyrics; structure tags like `[Verse]` on their own lines |
| `instructions` | music description (caption) |
| `response_format` | `wav` (the only format; others -> 400) |
| `seed` | optional; the seed used is returned in `X-MM3-Seed` |
| `max_new_tokens` | max audio frames at 25 fps (default 1500 = 60 s, max 9000); the song may end earlier at the end token |
| `audio_duration` | seconds, alias of `max_new_tokens / 25` |
| `num_inference_steps` | Euler steps per 8 s window (default 30, 1..200) |
| `stream` | must be `false` (`true` -> 501) |

Response headers: `X-MM3-Frames`, `X-MM3-Seed`, `X-MM3-Stopped-By` (`max_frames` / `end_token` / `context`),
`X-MM3-Prompt-Tokens`, `X-MM3-Generation-Seconds`, `X-MM3-Sampling-Rate`. Limits: prompt <= 5000 tokens, frames <= 9000
(400 otherwise). The backbone context is 10240 positions, so a song holds at most `10240 - prompt_tokens` frames; a
longer request is cut there and reports `X-MM3-Stopped-By: context` (the reference has the same cap). Every error,
body-validation errors included, uses the OpenAI shape `{"error": {"message", "type"}}` (400 / 404 / 422 / 501 / 504).

`GET /health` -> `{"status": "ok", "model", "device": {mesh_shape, device_ids, arch, ...}, "warm": true, "busy", ...}`.
`GET /v1/models` -> OpenAI list shape.

The model card's request, verbatim:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe",
    "instructions": "A warm acoustic pop song with intimate female vocals, fingerpicked guitar, soft piano, and a gradual emotional build into a wide final chorus.",
    "response_format": "wav",
    "seed": 7,
    "max_new_tokens": 750,
    "stream": false
  }' \
  --output minimax_music3.wav
```

## Concurrency and shutdown

One generation at a time: requests queue on an `asyncio.Lock`; the device work runs on a single dedicated worker
thread so `/health` keeps answering while the chip is busy (`"busy": true`). SIGTERM runs the lifespan shutdown:
the vocoder worker process is stopped, the pipeline released and the mesh closed.

## Tests

`tests/test_server.py` starts the server in a subprocess (under the hardware lock), waits for readiness, and checks
`/health`, `/v1/models`, a 10 s generation (seed 7), bad requests, a second request and the model-card curl. Evidence
goes to `doc/server/results.json`, wavs and the server log to `generated/`. Run: `~/mm3-bringup/checks/08.sh`.
