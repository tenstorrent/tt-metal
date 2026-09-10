# Stage 09: tt-model v5.1 container package, local serve, publish, consumer pull

MiniMax-Music3 (lyrics + caption -> stereo 44.1 kHz song) on one Blackhole chip, packaged as a tt-model-manager
**container (v5.1)** package (`kind: tt-dit-server`), served locally through `tt-model serve`, published privately to
`jashansinghTT/MiniMax-Music3-tt` and pulled + served again from a fresh consumer cache. Attempt 1; headless, so every
decision below was taken without asking.

## What was built

| file | content |
|---|---|
| `tt-model.yaml` | the authored manifest: schema 5.1, `repo jashansinghTT/MiniMax-Music3-tt`, `name minimax-music3`, pinned weights, `kind tt-dit-server`, `arch blackhole`, `source.code` allowlist, runtime packages, three serve profiles, `verify:` assertions, model-card quickstart |
| `doc/container/README.md` | this work log |
| `generated/container/` (gitignored) | the served wavs, response headers, `/health` dumps and the consumer container log |

No code under `tt/`, `server/` or tt-model-manager changed. The stage-08 server already read `MM3_MESH_SHAPE`, `MESH_DEVICE`,
`HF_MODEL`, `TT_DIT_CACHE_DIR` and `TT_METAL_CACHE` the way the `TtDitServerLauncher` exports them, and `tt-model`
(`~/tt-model-manager`, branch `feat/source-extra-code`, commit `240112f` plus an uncommitted `shlex.quote` in the
`--print` path that is not mine) needed no change, so no `fix/*` branch or PR was opened (`~/mm3-bringup/state/09.prs.txt`
says so).

## The manifest, and why each field is what it is

- **`weights`**: `MiniMaxAI/MiniMax-Music3` at `fbdf52fbaaca799592917417eb05f1899f1255ec` (the revision every earlier
  stage validated against) with `ignore_patterns` for the raw MiniMax checkpoints the diffusers layout does not use.
- **`kind: tt-dit-server`**: the model is served by its own FastAPI app, no vLLM. The launcher runs
  `python -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app`,
  which is byte for byte the stage-08 launch command.
- **`source.code`** (8 entries): `models/common`, `models/tt_transformers`, `models/tt_dit`, and the autoport dir by
  subpath: `__init__.py`, `tt/`, `server/`, `reference/`, `tt-model.yaml`. Computed from the runtime import closure on
  the host: importing `server.app`, `tt.pipeline` and `models.tt_transformers.tt.model` loads 59 `models.*` modules, all
  inside these paths (checked with `sys.modules` against the allowlist). The autoport dir is 7.7 GB because of
  `generated/` (wav/pt dumps) plus 15 MB of `doc/`; neither ships, nor do `tests/` and `scripts/`. No shipped code
  reads data files relative to the model dir (grep for `Path(__file__)`, `.json`, `.yaml`: only the weights snapshot's
  `vocoder/config.json` and `model.safetensors.index.json` are read, and `server/app.py` uses `__file__` only for the
  default `TT_DIT_CACHE_DIR`, which the container overrides). Staged `code/` is 9.5 MB / 532 files.
- **`runtime.packages`**: the list from the stage prompt (`fastapi uvicorn pydantic>=2 transformers>=5.10 safetensors
  soundfile huggingface_hub numpy<2`) plus **`pillow`, `pytest`, `tqdm`**, which the import trace showed are imported at
  module level by `models/tt_transformers/tt/common.py` (PIL), `models/common/utility_functions.py` (pytest) and
  `models/tt_transformers/tt/model.py` (tqdm). Everything else in the closure (loguru, pandas, seaborn, networkx,
  graphviz, pyyaml, click, ml_dtypes) is a declared dependency of `ttnn` and arrives with its editable install; the
  launcher prepends tt-metal's torch pin (`torch==2.11.0`, from `tt_metal/python_env/requirements-dev.txt`). No
  diffusers at runtime (the vendored `reference/*.py` pieces cover it). `librosa` is imported lazily by
  `tt/audio_metrics.py` only for offline evaluation and is deliberately not shipped.
- **`runtime.mesh_shape_env: MM3_MESH_SHAPE`**: the variable `server/app.py` reads (it would otherwise get
  `FLUX2_MESH_SHAPE` and fall back to `1x1` silently).
- **`serve.env.MM3_HF_REVISION`**: the launcher exports only `HF_MODEL=<repo id>`; the app resolves it with
  `snapshot_download(revision=MM3_HF_REVISION)`, so this pin makes the container load exactly the validated weights even
  when the consumer's HF cache also holds a newer `main`. Decision: accept the duplicated sha (commented in the YAML as
  "keep in step with weights.revision") rather than change the app to parse the wire manifest.
- **Profiles** `p150` (P150, default), `p300` (P300), `p300x2` (P300x2): all single-chip in v1. The launcher derives
  `MM3_MESH_SHAPE` (`1x1` / `1x2` / `1x4`) and `MESH_DEVICE` from `mesh_device`; the app logs them, warns when the shape
  is not `1x1`, and opens a 1x1 mesh on device 0. Documented in `card.quickstart`. None of the published profiles pins
  `TT_METAL_VISIBLE_DEVICES` (gate assertion).
- **`verify`**: imports of the app module (asserting `app`), the pipeline module (asserting `MiniMaxMusic3Pipeline`),
  `models.tt_transformers.tt.model` (asserting `Transformer`), and the host-side libraries with a `numpy < 2` check. All
  ran inside the finished image (build log step `[runtime 21/22] RUN bash /ctx/verify.sh`, 14.7 s).
- **`ubuntu "22.04"`, `python "3.12"`**: the base image is `ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest`
  (digest `c2161d26c599...`, created 2026-09-09); the venv inside is Python 3.12.14, matching the host's 3.12.13.

Validation without building (skill step 8) and the argv preview matched the stage-08 launch flag for flag:

```
$ ~/.tenstorrent-venv/bin/python -c "from tt_kernel.container_manifest import load_container_manifest as L; m=L('$MM3_MODEL_DIR/tt-model.yaml', check_sources=True); print(m.name, m.kind, m.profile_names())"
minimax-music3 tt-dit-server ['p150', 'p300', 'p300x2']
p300x2 -> python -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app
   env {'HF_MODEL': 'MiniMaxAI/MiniMax-Music3', 'MESH_DEVICE': 'P300x2', 'MM3_MESH_SHAPE': '1x4', 'MM3_HF_REVISION': 'fbdf52fb...'}
```

## Build

Prerequisite found by `tt-model package`'s own check: the worktree had **uninitialised git submodules** (a worktree does
not populate them), and the in-image `build_metal.sh` needs `tt_metal/third_party/umd` and `tracy`. Fixed with
`git -C $MM3_WT submodule update --init --recursive` (network clone; `~/tt-metal` untouched). Manifest committed as
`501969ba911` before building so the image's provenance sha includes it.

```bash
source ~/mm3-bringup/common.sh; cd $MM3_WT
~/.tenstorrent-venv/bin/tt-model -v --no-color package --container $MM3_MODEL_DIR/tt-model.yaml --out $MM3_BUILDS   # log: $MM3_LOGS/09.build.log
```

| phase | measured |
|---|---|
| whole `package` (resolve, stage, docker build, OCI export) | 01:35:48 -> 01:44:56, **9 min 8 s** |
| `build_metal.sh` inside the builder stage | 487.5 s. Not a cold build: the `ccache` / CPM cache mounts on this host were warm from an earlier tt-model build of another model (the 1-3 h estimate did not apply) |
| engine install (`torch==2.11.0` + packages, `uv`) | 5.3 s (uv cache) |
| `verify.sh` inside the runtime image | 14.7 s, all assertions passed |
| image `tt-model/minimax-music3:ac48eefd11a5` | 866 MB content, 3.94 GB unpacked; OCI layout `image/` 827 MB in 28 blobs |
| staged package | `$MM3_BUILDS/minimax-music3/{tt_kernel_manifest.json, README.md, requirements.lock, code/, image/}` |

`package` froze the image's venv into `requirements.lock` and recorded `runtime.lock: requirements.lock` on the wire
manifest (resolved there: `transformers==5.17.0`, `huggingface_hub==1.30.0`, `numpy==1.26.4`, `torch==2.11.0+cpu`,
`fastapi==0.141.1`, `uvicorn==0.52.4`, `soundfile==0.14.0`, `safetensors==0.8.0`, `pillow==12.3.0`). The host ran
`transformers 5.15.0` / `numpy 2.3.5`; the served output below is bit-identical to the host's stage-08 numbers, so the
difference is inert for this model. The wire manifest's `built` block: tt-metal `501969ba9114...` (branch
`jashan/minimax-music3`, `dirty: true` because the stage-08 gate re-run had rewritten `doc/*/results.json`; nothing
under `doc/` ships), `code_sha256 6eb92a34...`, `image_digest sha256:ac48eefd11a5...`.

## Local serve through tt-model (author path)

Decisions for this host (4 chips, the stage's device pin is `TT_METAL_VISIBLE_DEVICES=0`):

- The published manifest must not pin a device, so the local run used a **copy** of the built
  `tt_kernel_manifest.json` (in the session scratch dir) with `TT_METAL_VISIBLE_DEVICES=0` added to the `p300x2`
  profile's `env`. `tt-model serve` has no `--env` flag; everything after the target goes to uvicorn. The image was
  already loaded, so the copy needed no `image/` beside it.
- Host port 8000 is held by the tt-studio backend container, so `--port 8010` (tt-model moves both the docker publish
  mapping and uvicorn's `--port`).
- Every device-facing command ran under `with_hw_lock`.

```bash
~/.tenstorrent-venv/bin/tt-model serve <scratch>/tt_kernel_manifest.json --profile p300x2 --port 8010 --print
docker run --name tt-model-minimax-music3-p300x2 --user 1001:1001 ... --device /dev/tenstorrent --ipc host \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G --volume /home/jashan/.cache/huggingface:/hf --env HF_HOME=/hf \
  --volume /home/jashan/.cache/tt-model/minimax-music3/cache:/cache --env TT_METAL_CACHE=/cache \
  --volume /home/jashan/.cache/tt-model/minimax-music3/weights:/weight-cache --env TT_DIT_CACHE_DIR=/weight-cache \
  --publish 8010:8010 --env HF_TOKEN --env HF_MODEL=MiniMaxAI/MiniMax-Music3 --env MESH_DEVICE=P300x2 \
  --env MM3_HF_REVISION=fbdf52fbaaca799592917417eb05f1899f1255ec --env MM3_MESH_SHAPE=1x4 --env TT_METAL_VISIBLE_DEVICES=0 \
  tt-model/minimax-music3:ac48eefd11a5 python -m uvicorn --host 0.0.0.0 --port 8010 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app

with_hw_lock timeout 3000 ~/.tenstorrent-venv/bin/tt-model --no-color serve <scratch>/tt_kernel_manifest.json --profile p300x2 --port 8010 --follow   # log: $MM3_LOGS/09.serve.log
```

Boot transcript (`$MM3_LOGS/09.serve.log`, container started 01:48:34):

| event | measured |
|---|---|
| weights resolved | `snapshot_download` at the pinned revision resolved offline to `/hf/hub/models--MiniMaxAI--MiniMax-Music3/snapshots/fbdf52fb...` (the bind-mounted host cache), 0.5 s |
| device | `MESH_DEVICE=P300x2 MM3_MESH_SHAPE=1x4 -> 1x1 mesh, device 0`; `/health` later reports `device_ids [3]` (UMD logical id of the chip `TT_METAL_VISIBLE_DEVICES=0` exposes, as in stage 08), `arch blackhole`, `visible_devices "0"` |
| cold caches | the mounted `/cache` and `/weight-cache` were empty: tt_transformers converted the 36 Qwen3 layers to bfp8 (`MusicLLM: device model built in 82s`), tt_dit converted the DiT weights (`Cache does not exist. Loading PyTorch state dict.`) |
| ready | `server warm: 50-frame warm-up song in 9.2 s; load 147 s`, `Application startup complete`, `ready at http://127.0.0.1:8010` at 01:51:17, **2 min 43 s** after `docker run` |
| host cache footprint afterwards | `~/.cache/tt-model/minimax-music3/cache` 9.9 GB (tt_transformers bfp8 tensors + JIT kernels), `weights/` 2.5 GB (DiT cache) |

Requests (all under the same container):

```bash
curl http://127.0.0.1:8010/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Verse]\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe",
  "instructions": "A warm acoustic pop song with intimate female vocals, fingerpicked guitar, soft piano, and a gradual emotional build into a wide final chorus.",
  "response_format": "wav", "seed": 7, "max_new_tokens": 250, "stream": false }' --output served_seed7_10s.wav
```

| request | result |
|---|---|
| `GET /health`, `GET /v1/models` | `status ok, warm true, busy false, dtype_policy optimized`; model id `MiniMaxAI/MiniMax-Music3` (`generated/container/health_before.json`) |
| model-card curl, 250 frames (10 s), seed 7 | HTTP 200 `audio/wav`, 1,763,372 bytes; headers `x-mm3-frames 250`, `x-mm3-seed 7`, `x-mm3-stopped-by max_frames`, `x-mm3-prompt-tokens 58`, `x-mm3-generation-seconds 29.30`; curl wall 29.31 s. Validated: 44,100 Hz, 2 channels, PCM_16, 9.996 s, RMS 0.0943, peak 1.000 -> `generated/container/served_seed7_10s.wav` (= `state/09.served_wav`) |
| model-card curl verbatim, 750 frames (30 s), seed 7 | HTTP 200, 30.02 s of audio, RMS 0.1087, peak 1.000, generation 85.89 s, wall 85.91 s -> `generated/container/served_model_card_curl_750.wav`; stage 08 measured 30.02 s / RMS 0.109 / 83.2 s for the same request on the host |
| `tt-model stop <scratch manifest> --profile p300x2` | `clean shutdown 2.2s`, container removed (`$MM3_LOGS/09.stop.log`) |

The 10 s song took 29.3 s in the container versus 26.3 s on the host in stages 07/08 (about 11 % slower). Not
investigated in this stage: the container runs with `MM3_TORCH_THREADS` defaulting to `cpu_count - 4` like the host,
but the host was also running a Llama-3.1-8B tt-studio container on chips 0/1 and the tt-studio stack during this run,
so the difference may be host contention rather than the container. Recorded as an open item.

## Publish (private, own namespace)

```bash
~/.tenstorrent-venv/bin/tt-model -v --no-color push $MM3_BUILDS/minimax-music3 --private   # log: $MM3_LOGS/09.push.log
Creating repo jashansinghTT/MiniMax-Music3-tt (private)
  • image/ 866.2 MB in 28 content-addressed blobs
✓ uploading to jashansinghTT/MiniMax-Music3-tt  19.5s
  ✓ pushed jashansinghTT/MiniMax-Music3-tt
```

Verified on the Hub with `HfApi.model_info(files_metadata=True)`: private, revision `3a6be827e854...`, 567 files,
874.4 MB (`tt_kernel_manifest.json`, `README.md` with the quickstart, `requirements.lock`, `code/` 532 files,
`image/` with 28 blobs). The 19.5 s upload is xet deduplication: the runtime layers are shared with the
`qwen3-coder-30b-a3b` container this host pushed earlier. `~/mm3-bringup/state/09.published.txt` =
`https://huggingface.co/jashansinghTT/MiniMax-Music3-tt`.

## Consumer path: fresh cache, pull, serve, curl

To make the pull real, the local image was deleted first (`docker image rm tt-model/minimax-music3:ac48eefd11a5`), and
tt-model ran with `HOME=~/mm3-bringup/consumer-home` so its local db, pulled-manifest dir and both per-model caches
(`~/.cache/tt-model/minimax-music3/{cache,weights}`) started empty. `HF_HOME` stayed the real host cache (the weights are
a pointer and were already there; the run therefore does not prove the 27 GB weight download).

```bash
HOME=$MM3_ROOT/consumer-home tt-model -v --no-color pull jashansinghTT/MiniMax-Music3-tt          # $MM3_LOGS/09.pull.log
  docker load tt-model/minimax-music3:ac48eefd11a5  28.2s  (from the Hub OCI layout)
  ✓ pulled jashansinghTT/MiniMax-Music3-tt
# TT_METAL_VISIBLE_DEVICES=0 added to the p300x2 profile of the PULLED manifest copy (this host only), then:
HOME=$MM3_ROOT/consumer-home with_hw_lock timeout 3000 tt-model --no-color serve jashansinghTT/MiniMax-Music3-tt --profile p300x2 --port 8010 --follow   # $MM3_LOGS/09.consumer_serve.log
```

| event | measured |
|---|---|
| boot with empty consumer caches | container start 01:57:49, `server warm: 50-frame warm-up song in 9.2 s; load 145 s`, `ready at http://127.0.0.1:8010` at 02:00:29 (**2 min 40 s**) |
| model-card curl, 250 frames, seed 7 | HTTP 200, `x-mm3-generation-seconds 29.47`, curl wall 29.48 s; 44.1 kHz stereo, 9.996 s, RMS 0.0943, peak 1.000 -> `generated/container/consumer_seed7_10s.wav` |
| consumer wav vs author-path wav | **bit-identical** (`np.array_equal`, max abs diff 0.0) |
| `docker logs` captured before stop | `generated/container/consumer_container.log` (643 lines); `/health` -> `generated/container/consumer_health.json` |
| `HOME=... tt-model stop jashansinghTT/MiniMax-Music3-tt` | `clean shutdown 1.9s`, no `tt-model-minimax-music3-*` container left (`$MM3_LOGS/09.consumer_stop.log`) |

## Gate

```
$ ~/mm3-bringup/checks/09.sh
manifest OK: minimax-music3 tt-dit-server ['p150', 'p300', 'p300x2']
served wav OK: 10.0s rms=0.0943
GATE09_OK published: https://huggingface.co/jashansinghTT/MiniMax-Music3-tt
```

Hardware: Blackhole p300c, board id `000004613193411b` (`tt-smi -s`: `BOARD_ID_HIGH 0x461`, `BOARD_ID_LOW 0x3193411b`),
host `qbge-devex-02`, chip `TT_METAL_VISIBLE_DEVICES=0`. Code: manifest commit `501969ba911` (built into the image);
this work log in the follow-up commit.

## Decisions taken because nobody could be asked

- `runtime.packages` extended with `pillow`, `pytest`, `tqdm` (module-level imports in shipped tt-metal code); no
  `librosa`; specs left as ranges per the stage prompt and frozen by `package` into `requirements.lock`.
- `MM3_HF_REVISION` duplicated into `serve.env` (see above) instead of a code change.
- Autoport dir listed by subpath; `tests/`, `scripts/`, `doc/`, `generated/` do not ship.
- Local runs pinned the device through a copied manifest, not through the published one; the consumer run reused the
  host HF cache but a fresh tt-model cache.
- `--port 8010` for both local runs because 8000 was taken on this host; the published manifest keeps `port: 8000`.
- The Llama-3.1-8B tt-studio container holding `/dev/tenstorrent/0` and `/dev/tenstorrent/1` was left running (never
  `docker stop` anything): the pinned chip was free and both serves opened it without contention.
- The stage-08 gate re-run's `doc/*/results.json` modifications (made by the pipeline's own check after commit
  `77a63529861`) were left as they were; they are evidence of that re-run, not of this stage.

## Open risks

- Not exercised: the `p150` and `p300` profiles on their own hardware (this box is a p300x2), the weight download into an
  empty HF cache from inside the container (the 27 GB snapshot was already on the host), and serving without the device
  pin on a multi-chip host where other chips are busy (the app opens `MeshShape(1, 1)` on device 0 regardless of the
  requested shape; UMD still enumerates every visible chip).
- The model card `README.md` generated by `package` renders the `max_num_seqs` / `max_model_len` columns empty for this
  kind; cosmetic.
- `tt-model stop` removes the container, so its logs vanish with it; capture `docker logs` first (done for the consumer
  run, missed for the author run, whose boot is in `$MM3_LOGS/09.serve.log`).
- The container's 10 s generation was 29.3 s vs 26.3 s on the host (see above); cause not isolated.
- `requirements.lock` resolved `transformers 5.17.0` and `huggingface_hub 1.30.0`, newer than the host's 5.15.0 / 1.16.1;
  outputs matched bit for bit, but a future rebuild without the lock could drift further. Re-authoring with
  `runtime.lock: requirements.lock` (committing the lock next to the YAML) would freeze it.
