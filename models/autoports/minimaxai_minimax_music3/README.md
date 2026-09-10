# MiniMax-Music3 on one Blackhole chip (TTNN autoport)

`MiniMaxAI/MiniMax-Music3` turns lyrics plus a caption into a stereo song. This directory runs the whole
model on a single Tenstorrent Blackhole chip with TTNN and serves it through tt-model-manager's
`tt-dit-server` container kind.

The reference is the merged diffusers implementation (`MiniMaxMusic3Blocks`, Apache-2.0). Its parts and where they
live here:

| part | reference | device code | notes |
|---|---|---|---|
| language model (Qwen3, 36 layers, hidden 4096, batch 2 for CFG) | `Qwen3ForCausalLM` | `tt/llm.py` on `models/tt_transformers` | traced decode, paged prefill, tile-aligned logits window read back |
| RVQ depth decoder (4 layers, 7 audio heads, 2..9 steps) | `minimax_music3_rvq_depth_decoder.py` | `tt/depth_decoder.py` | two traces (seed + step), nothing allocated per frame |
| autoregressive generator (CFG 1.5, top-k 50, end token) | `encoders.py` | `tt/ar_generator.py`, `tt/prompt.py` | reference sampling arithmetic on the host from device logits |
| flow-matching DiT (36 layers, dim 2048, partial RoPE) + Euler scheduler | `transformer_minimax_music3.py` | `tt/flow_transformer.py`, `tt/denoiser.py`, `tt/scheduler.py`, `tt/condition_encoder.py` | traced 30-step loop per 200-frame window, CFG 1.7 |
| vocoder (44.1 kHz stereo) | `minimax_music3_vocoder.py` | `tt/vocoder.py`, `tt/vocoder_worker.py` | fp32 on the host in a worker process, overlapped with the next window's denoising (a TTNN vocoder exists but is slower, see limitations) |
| pipeline (chunking, crop / stitch) | `MiniMaxMusic3Blocks` | `tt/pipeline.py` | `MiniMaxMusic3Pipeline.load(mesh_device)` then `generate(...)` |
| HTTP server | model card (`/v1/audio/speech`) | `server/app.py` | FastAPI, one song at a time, OpenAI error shape |

Vendored torch reference pieces (with their Apache-2.0 headers and diffusers source paths) live in `reference/`.
Per-stage work logs with every measured number are under `doc/<stage>/README.md`
(`llm`, `depth_decoder`, `ar_generator`, `flow_dit`, `pipeline`, `optimize`, `server`, `container`, `review`).

## Hardware

Developed and measured on one chip of a Blackhole P300 (board `p300c`, `tt-smi -s` board id `000004613193411b`) with
`TT_METAL_VISIBLE_DEVICES` pinning a single chip, opened as a 1x1 mesh on device 0. The tt-model manifest declares
`p150`, `p300` and `p300x2` profiles; only the `p300x2` profile was exercised on hardware, and every profile is
single-chip (see limitations).

## How to run the tests

All tests are pytest under `tests/`. Hardware tests open device 0 as a 1x1 mesh; they skip when no device is
present. Long sweeps are marked `slow`.

```bash
cd <tt-metal root>
export HF_MODEL=<path to the MiniMax-Music3 snapshot>/language_model   # tt_transformers reads the Qwen3 backbone here
export MM3_WEIGHTS=<path to the MiniMax-Music3 snapshot>               # the rest of the checkpoint
python -m pytest models/autoports/minimaxai_minimax_music3/tests -m "not slow" -x -q
```

Stage gates in order of cost (each is a single file): `test_llm.py`, `test_depth_decoder.py`, `test_flow_transformer.py`,
`test_ar_generator.py`, `test_pipeline.py`, `test_optimized.py`, `test_server.py`. Every comparison against the
diffusers reference uses `models.common.utility_functions.comp_pcc`; the bars are in the test files and in the
stage logs. `tests/conftest.py` documents the environment variables (weights, golden reference directory, dtype policy).

The golden outputs (one 10 s clip, seed 7, 30 Euler steps, fp32 CPU diffusers) are not in the repository; the tests
that need them skip when the directory `MM3_REF` points at (default `~/mm3-bringup/reference`) is missing. `scripts/dump_ref_trajectory.py` and
`scripts/dump_scheduler_triples.py` regenerate the per-step reference data from the diffusers venv.

## How to serve

Host (same command tt-model-manager runs inside the container):

```bash
HF_MODEL=<snapshot dir or MiniMaxAI/MiniMax-Music3> MM3_MESH_SHAPE=1x1 \
python -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app
```

Container through tt-model-manager (manifest `tt-model.yaml`, package published as `jashansinghTT/MiniMax-Music3-tt`):

```bash
tt-model pull jashansinghTT/MiniMax-Music3-tt
tt-model serve jashansinghTT/MiniMax-Music3-tt --profile p300x2 --port 8000
curl http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model":"MiniMaxAI/MiniMax-Music3","input":"[Verse]\n...","instructions":"A warm acoustic pop song ...","max_new_tokens":750,"seed":7}' \
  --output song.wav
```

The server is ready when uvicorn prints `Application startup complete` (weights resolved, traces warm). Request and
response fields, headers, error codes and environment variables are in `server/README.md`.

## Performance (measured, one chip, warm caches, idle host)

Free-running 10 s song, seed 7, 30 steps, 250 frames, two DiT windows (`doc/optimize/perf.json`):

| metric | functional (stage 06) | optimized (default) |
|---|---|---|
| AR frames/s (realtime = 25) | 13.3 | 21.7 |
| per frame: LLM step / depth loop / host | 38.0 / 33.1 / 3.9 ms | 21.8 / 23.9 / 0.4 ms |
| DiT per 200-frame window, 30 steps (689 / 516 latents) | 3.51 / 2.77 s | 2.55 / 2.45 s |
| host vocoder per window (fp32) | 6.77 / 5.11 s, serial | 6.83 / 5.22 s, overlapped with denoising |
| total, 10 s clip | 37.1 s | 26.3 s |
| resident device DRAM after load | 21.7 GB | 15.1 GB |

Served through the container (`doc/container/README.md`): the model-card request for 750 frames (30 s of audio)
took 85.9 s generation time against 83.6 s for the same request on the host, with byte-identical wavs.

Accuracy against the fp32 golden run, optimized policy (teacher-forced codes and golden noises so the paths match):
frame-hidden PCC 0.99903 (per-frame minimum 0.979), latent PCC 0.99903 / 0.99868 for the two windows, stitched wav
PCC 0.99855 with a 1.27 dB log-mel RMS difference. The per-stage bars (>= 0.98 PCC, <= 2.0 dB) and the functional
policy's numbers are in `doc/optimize/README.md`.

## Known limitations

- **Single chip only.** The `p300` and `p300x2` profiles open one chip (`MeshShape(1, 1)` on device 0) even though
  the board has two. The natural follow-up is to run the AR loop on one chip and the DiT + vocoder of already
  emitted windows on the other (window pipelining); it needs every DiT-phase temporary allocated as a persistent
  buffer before the AR traces are captured (trace lifetime rule) and a second device in the pipeline. Not done.
- **Below realtime for the AR loop.** 21.7 frames/s against 25 needed; the LLM step runs at the DRAM roofline of
  its bytes, the depth loop spends 4.2 ms of 23.9 in a 2-core head split. Levers are listed in
  `doc/optimize/README.md` (interleaved 1D-mcast decode matmuls, a faster head split for two tile rows).
- **Host vocoder.** The fp32 vocoder runs on the host in a worker process. The TTNN port (`vocoder="device"`) is
  kept in the tree but runs 2.3x slower (fp32 conv3d has no good blocking for these shapes) and its trace needs device-side padding.
- **No streaming, one request at a time.** `stream: true` returns 501. A second request waits behind the device
  lock for the whole first song; the health endpoint keeps answering. A request that hits `MM3_REQUEST_TIMEOUT_S`
  gets a 504 but the device finishes the song first.
- **Batch is fixed at 2** (conditional + unconditional CFG rows). There is no multi-request batching.
- **First request of a new window length** (a last chunk shorter than 200 frames that was not warmed) compiles the
  DiT programs and captures a trace once per process (about 5 s instead of 2.4 s); the 200-frame window is warmed at load.
- **Precision.** The optimized policy (bfp8 depth-decoder weights, LoFi on the 12-core DRAM-sharded decode matmuls,
  fp16 accumulation in the DiT) costs 0.5 dB of log-mel against the functional policy; the per-frame frame-hidden PCC
  minimum is 0.979. `MM3_DTYPE_POLICY=functional` restores the slower, closer path.
- **Sampling rate.** The checkpoint's vocoder produces 44.1 kHz; the model card's SGLang-Omni text says 32 kHz.
- `p150` / `p300` profiles, the weight download into an empty cache inside the container, and the 504 path were not
  exercised on hardware.
