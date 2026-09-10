# Stage 08: HTTP server for tt-model's `tt-dit-server` kind

MiniMax-Music3 (lyrics + caption -> stereo 44.1 kHz song) served on one Blackhole chip through the OpenAI-style audio
endpoint the model card documents for SGLang-Omni. Attempt 1; headless, so every decision below was taken without asking.

## What was built

| file | content |
|---|---|
| `server/app.py` | FastAPI app (`app`): lifespan (weights -> device -> `MiniMaxMusic3Pipeline.load(optimized)` -> 2 s warm-up song), `POST /v1/audio/speech`, `GET /health`, `GET /v1/models`, one-request-at-a-time lock, request timeout |
| `server/__init__.py`, `server/README.md` | package marker; launch command, environment, request contract |
| `tests/test_server.py` | subprocess launch of the uvicorn command, readiness wait, the six gate tests, SIGTERM clean-exit check |
| `doc/server/results.json` | evidence written by the tests (startup time, `/health`, wav statistics, status codes, curl) |

Launch (identical to what `TtDitServerLauncher` runs in the container, see `~/tt-model-manager/src/tt_kernel/launchers.py`):

```bash
source ~/mm3-bringup/common.sh
cd $MM3_WT && HF_MODEL=$MM3_WEIGHTS MM3_MESH_SHAPE=1x1 $MM3_PY -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app
```

Gate: `~/mm3-bringup/checks/08.sh` (= `pytest tests/test_server.py -m "not slow" -x` under the hardware lock).

## Design

- **Weights.** `HF_MODEL` may be a local snapshot directory (used as is; `$MM3_WEIGHTS/language_model`, which
  `common.sh` exports as `HF_MODEL` for tt_transformers, resolves to its parent) or an HF repo id, which goes through
  `huggingface_hub.snapshot_download(ignore_patterns=[flowmatching_vae.pth, dav.pth, qwen_7B/*, assets/*, figures/*, scripts/*])`.
  Decision: `ignore_patterns` instead of the `allow_patterns` the stage prompt mentions - the intent is "everything
  except the raw checkpoints", and an exclusion list cannot silently drop a subfolder the diffusers layout needs.
  `MM3_HF_REVISION` pins the revision. Verified on the host (`doc/server/repo_id_resolution.txt`):

  ```bash
  cd $MM3_WT && HF_HUB_OFFLINE=1 MM3_HF_REVISION=$MM3_HF_REVISION $MM3_PY -c "from models.autoports.minimaxai_minimax_music3.server.app import resolve_weights; print(resolve_weights('MiniMaxAI/MiniMax-Music3'))"
  # -> /home/jashan/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-Music3/snapshots/fbdf52fbaaca799592917417eb05f1899f1255ec
  ```
- **Device.** Always `ttnn.open_mesh_device(MeshShape(1, 1))` on device 0 with the 200 MB trace region from stage 07;
  `MM3_MESH_SHAPE` / `MESH_DEVICE` are logged and reported by `/health` (v1 is single-chip; a shape other than `1x1`
  logs a warning). No fabric (this host handshakes the unused chips otherwise, see `tests/conftest.py`).
- **One device thread.** Load, warm-up, every generation and the shutdown run on a single `ThreadPoolExecutor(1)`
  thread, so all ttnn calls come from one thread and the asyncio loop only ever waits on futures: `/health` answers while
  a song is generating (`"busy": true`). Requests serialize on an `asyncio.Lock`; a request that exceeds
  `MM3_REQUEST_TIMEOUT_S` (default 1800 s) gets a 504, but the underlying task is shielded so the lock is held until
  the device is actually idle (a generation cannot be interrupted mid-trace).
- **Warm-up.** `MiniMaxMusic3Pipeline.load` already captures the AR traces and the 200-frame-window DiT trace; the
  lifespan then generates a 2 s / 4-step song (smallest window shape, vocoder-worker round trip, wav encoder) before it
  yields, so `Application startup complete` means "fully warm".
- **Contract.** `input` = lyrics, `instructions` = caption, `response_format` must be `wav` (422 otherwise, a body-validation error),
  `max_new_tokens` = frames at 25 fps (default 1500, max 9000), `audio_duration` seconds as an alias (both given and
  disagreeing -> 400), `num_inference_steps` 1..200 (default 30), `stream: true` -> 501, `model` other than
  `MiniMaxAI/MiniMax-Music3` -> 404, prompt over 5000 tokens -> 400 (tokenized on the host before the lock is taken).
  Validation errors (pydantic) are rewritten into the same OpenAI error shape (422). The validation tokenizer is a
  second `PromptEncoder` instance run off the event loop, separate from the pipeline's, so the device thread and the
  request handler never share the tokenizer's BPE cache. A failed start closes the mesh before re-raising (uvicorn then
  exits non-zero); a generation that outlives `MM3_REQUEST_TIMEOUT_S` is logged when it finishes and still counted.
  Output: 16-bit PCM stereo WAV at 44.1 kHz (the model card says 32 kHz for SGLang-Omni; the released vocoder config
  and the stage-06 evidence say 44.1 kHz, and the stage prompt asks for 44.1 kHz), plus `X-MM3-*` headers with the
  frames, seed, stop reason, prompt tokens and generation time. Errors use the OpenAI `{"error": {...}}` shape.
- **Shutdown.** uvicorn turns SIGTERM into the lifespan shutdown: vocoder worker process stopped, pipeline released,
  mesh closed (`device closed` in the log); the tests accept exit code 0 or -15 (uvicorn re-raises the captured SIGTERM; -15 is what was observed).

## Evidence

Hardware: Blackhole p300c, board id `000004613193411b` (`tt-smi -s`), device 0 of the 1x1 mesh, host `qbge-devex-02`.
Code: commit `1f3a581da33` (the final `results.json` run was recorded at `fd28ebad279`, whose `server/` and `tests/` are identical) on `jashan/minimax-music3` (`models/autoports/minimaxai_minimax_music3`).

Gate run (`~/mm3-bringup/checks/08.sh`, 2026-09-10 01:09-01:11, idle host, warm weight caches; `doc/server/results.json`,
pytest output `generated/gate08.log`, server stdout `generated/server_test.log`): **7 passed in 148 s, `GATE_OK`**. Two
earlier full runs (00:58 and 01:02, before the review fixes) gave the same passes with timings within 0.2 s.

| test | result | measured |
|---|---|---|
| startup (`Application startup complete` + `/health` 200) | pass | 22 s from `Popen` to ready: 14 s pipeline load (warm weight caches) + 4.5 s 2 s / 4-step warm-up song |
| `/health` | pass | `status ok`, `warm true`, `busy false`, `device {mesh_shape 1x1, device_ids [3], arch blackhole, mesh_device p300c}` |
| `/v1/models` | pass | list shape, id `MiniMaxAI/MiniMax-Music3` |
| `POST /v1/audio/speech` 10 s, seed 7, 30 steps, `/health` polled every 0.5 s meanwhile | pass | 250 frames, 9.996 s of 44.1 kHz stereo PCM_16 wav (1.76 MB), RMS 0.095, peak 0.87, 104 prompt tokens, stopped_by `max_frames`, 26.25 s generation / 26.7 s client wall -> `generated/server_speech_seed7_10s.wav`; `/health` answered all 52 polls (`busy: true` seen), max 82 ms / mean 14 ms |
| bad requests | pass | empty lyrics 422, malformed JSON 422, `mp3` 422, `num_inference_steps 0` 422, 9001 frames 400, 0.01 s 400, 6000-word lyrics 400 ("5000"), `stream: true` 501, wrong model 404, all in the OpenAI error shape; `/health` ok afterwards |
| second request (`audio_duration: 2.0`, seed 11, 6 steps) | pass | 50 frames, 1.997 s, RMS 0.013, 4.45 s generation; `requests_served` 2 |
| two concurrent 2 s requests (seeds 21 / 22) | pass | both 200; generations 4.39 s + 4.38 s, client walls 4.4 s and 8.78 s (the second waited behind the lock), pair wall 8.78 s >= sum |
| model card curl (750 frames, seed 7) | pass | 30.02 s of audio (5.3 MB), RMS 0.109, peak 1.00, 83.2 s wall -> `generated/minimax_music3_model_card_curl.wav` |
| SIGTERM | pass | lifespan shutdown logged `device closed`; exit code -15 (uvicorn re-raises the captured SIGTERM after a clean shutdown, `Server.capture_signals`) |

The 10 s song reproduces the stage-07 optimized timing (26.3 s total in `doc/optimize/README.md`), so the HTTP layer,
the wav encoder and the worker-thread hand-off add nothing measurable (26.25 s generation vs 26.7 s client wall including
the 1.8 MB transfer). The independent stage review (host-only) found the 10 s server wav bit-identical after 16-bit
quantization to the stage-06/07 `generated/free_running_seed7_10s.wav` of the same prompt / seed / steps / policy. Device id
`3` is UMD's logical id of the one chip `TT_METAL_VISIBLE_DEVICES=0` exposes; the mesh is 1x1 and `/health` also
reports `visible_devices`.

Findings fixed during the stage:

- The first test run hung on startup: the test process called `ttnn.get_num_devices()` to decide whether hardware is
  present, which builds UMD's cluster and takes the `CHIP_IN_USE_0_PCIe` lock, so the server subprocess waited on it
  forever. The test now probes `/dev/tenstorrent/<TT_METAL_VISIBLE_DEVICES>` only and never builds the cluster
  (`conftest.py` still imports ttnn; the import alone does not take the lock).
- uvicorn 0.52 exits with -15 after SIGTERM by design (it re-raises the signal once the lifespan shutdown finished);
  the test accepts 0 or -SIGTERM and additionally requires the `device closed` log line.

Stage review (fresh subagent, read-only, `stage-review` skill): **clean-pass**; its hardening items were applied before
the gate run (OpenAI-shaped 422 errors, mesh closed on a failed start, done-callback for a timed-out generation, separate
validation tokenizer off the event loop, real board name as `MESH_DEVICE` in the test, concurrent-request test, README
wording on the context cap / `mesh_shape_env` / errors).

```
$ ~/mm3-bringup/checks/08.sh
...
======================== 7 passed in 147.60s (0:02:27) =========================
GATE_OK tests/test_server.py
```

The gate was run again on the committed code (`1f3a581da33`, after the pre-commit black / isort reformat) at 01:13-01:15:
`7 passed in 148.36s`, `GATE_OK`; `doc/server/results.json` holds this final run (10 s song 26.46 s, curl 83.6 s, `/health` fine during generation).

## Open risks

- Not tested here: the 504 timeout path, SIGTERM while a song is generating, `MM3_MESH_SHAPE` other than `1x1`
  (only the warning + 1x1 fallback code path exists), the repo-id download from an empty cache (only the resolution to
  the cached snapshot was run), and the container launch itself (stage 09; the manifest must set
  `runtime.mesh_shape_env: MM3_MESH_SHAPE`).
- One song at a time: a second request waits behind the lock for the whole first generation (up to minutes for a
  60 s song); the container launcher's HTTP health check keeps working meanwhile.
- The 504 path leaves the device busy until the song finishes; the request that timed out is discarded.
- The SGLang-Omni contract in the model card says the response is 32 kHz; ours is 44.1 kHz (the checkpoint's vocoder
  sampling rate). Clients that hard-code 32 kHz would resample wrongly.
- Only the local snapshot path was exercised end to end; the repo-id path was verified to resolve the cached snapshot
  offline but not to download from scratch (the weights are already present on this host).
