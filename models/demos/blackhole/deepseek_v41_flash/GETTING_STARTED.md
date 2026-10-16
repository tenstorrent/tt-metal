# DeepSeek-V4.1-Flash on Blackhole Galaxy — tt-inference-server getting started

Text generation (chat and completions, optionally with image input) through an OpenAI-compatible vLLM server. Three serving configurations, all with speculative decoding
(5 draft tokens per round) and on-device sampling: 16 concurrent users with up to 131,072 tokens of context (the default), 16 users with up to 262,144 tokens, and 4 users with up
to 1,048,576 tokens (see "Long context"). What changed in this release is in `RELEASE_NOTES.md`.

## Docker image

| | |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-799a223-ffd1a57` |
| Size | 21.8 GB on disk (5.4 GB compressed), built 2026-10-16 |
| Access | `docker run` pulls the image automatically on a new host. Pulling needs access to the `tenstorrent` packages on ghcr.io (check the digest with `docker image inspect <image> --format '{{.RepoDigests}}'`) |
| tt-metal | `799a223987c18909ced4bab280e4bccf673a73b0` (branch `ssinghal/dsv4p1`) |
| tt-inference-server | `ffd1a57ceece25c02b668bbcc4682dba52dbd9da` (branch `ssinghal/dsv4p1`) |
| vllm-tt-plugin | `8cb999ebf8579e0733e378d852bb2139f7cac064` (branch `ssinghal/dsv4p1`) |
| How it was built | an overlay on the previous image (`0.24.0-3dce07e-233b250`): the compiled tt-metal and the Python environment are reused (no file outside `models/` differs in tt-metal between the two commits); `models/`, the tt-inference-server tree, the plugin package, `model_spec.json` and `BUILD_INFO` come from the shas above. `BUILD_INFO` in the image records them. |
| vLLM | 0.31.0 (torch 2.11.0+cpu, transformers 5.12.1) |
| Model | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (476 GB of safetensors, 48 shards) |
| Hardware | 1x Blackhole Galaxy (32 chips, 4x8 mesh) |

The image has no weights and no caches in it: the checkpoint, the converted-weight cache and the
compiled-kernel cache are mounted from the host (step 3).

What this build is configured for (the runtime spec in step 3 sets all of it; the table shows the default 16-user spec, the other two are in "Long context"):

| | |
|---|---|
| Concurrent requests | 16 (`max_num_seqs` 16; more requests wait in the vLLM queue) |
| Context | 131,072 tokens (prompt + generated) |
| Speculative decoding | fixed 5 draft tokens per round (DSpark drafter from the model's own weights), lossless |
| Sampling | on the device: temperature, top-k, top-p, per-request seed |
| KV cache | fp8 pool, no prefix caching |
| Chat template | non-thinking (the reply starts after `</think>`) |
| SwiGLU clamp | routed experts compute `silu(min(gate, 10)) * clamp(up, -10, 10)` (`swiglu_limit` 10 in the model config) in decode, prefill, the speculative verify block and the MTP drafter (`DSV41_SWIGLU_CLAMP_SITES`, default `decode,prefill,spec,mtp`; `DSV41_DEC_SWIGLU_CLAMP=0` turns it off everywhere) |

## Performance

Method (all tables): GSM8K 8-shot prompts (about 700 tokens on average) with 256 output tokens per request, `vllm bench serve` run by tt-inference-server's
`benchmarks` workflow from the same image as the server, a constant number of requests in flight. "Greedy" is temperature 0; "sampled" is temperature 0.6, top-k 20,
top-p 0.95. TPOT and TTFT are the report means.

### 16 users, 262,144-token context, image input on (this release)

Image `0.24.0-799a223-ffd1a57`, vLLM 0.31.0, `assets/runtime_spec_b16_k5_262144_vision.json`, one persistent server (the sweeps ran after the functional tests, no restart):

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.05 s | 26.4 ms | 37.8 | 1.09 s | 29.3 ms | 34.1 |
| 2 | 1.26 s | 34.8 ms | 28.8 | 1.30 s | 35.2 ms | 28.4 |
| 4 | 1.44 s | 36.1 ms | 27.7 | 1.40 s | 39.0 ms | 25.6 |
| 8 | 1.74 s | 53.4 ms | 18.7 | 1.86 s | 52.2 ms | 19.2 |
| 16 | 2.56 s | 71.0 ms | 14.1 | 2.63 s | 72.5 ms | 13.8 |

Total output throughput (greedy): 32.8 / 50.0 / 94.4 / 120.7 / 189.4 tok/s at 1 / 2 / 4 / 8 / 16 requests.

Compared with the table below (the previous image on the default 131,072-token spec, measured on other hosts and an earlier vLLM stack): TTFT is lower at every concurrency, TPOT is lower at 1, 4 and 16 requests and about 1 ms higher
(inside the run-to-run noise of about 3-4 ms) at 2 and 8. The longer context and the loaded vision tower do not cost speed: TPOT follows the actual position, and the
tower only runs for requests that carry images. The same server compared with the image before the two fixes of this release (RoPE per-row, scheduler limits), same host and spec:
TPOT within noise (29.95 / 33.0 / 41.4 / 49.8 / 73.9 ms greedy against 26.4 / 34.8 / 36.1 / 53.4 / 71.0 ms).

### 16 users, 131,072-token context, default spec (this release, no image input)

Image `0.24.0-799a223-ffd1a57`, `assets/runtime_spec_b16_k5_131072.json`, same method, measured on the same host a few hours after the 262,144-token table above:

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.03 s | 28.5 ms | 35.1 | 1.06 s | 28.0 ms | 35.8 |
| 2 | 1.33 s | 32.2 ms | 31.1 | 1.25 s | 31.8 ms | 31.4 |
| 4 | 1.38 s | 38.0 ms | 26.3 | 1.39 s | 37.5 ms | 26.6 |
| 8 | 1.74 s | 49.1 ms | 20.3 | 1.80 s | 49.6 ms | 20.2 |
| 16 | 2.40 s | 70.4 ms | 14.2 | 2.70 s | 68.0 ms | 14.7 |

Total output throughput (greedy): 30.8 / 53.2 / 85.5 / 125.4 / 196.6 tok/s at 1 / 2 / 4 / 8 / 16 requests.

Against the previous image's table below: TTFT is lower at every concurrency (1.03-2.40 s against 1.24-2.75 s), TPOT is lower at 2 to 16 requests (70.4 against 80.1 ms at 16) and 0.8 ms higher at one request, and the total throughput at 16 requests is 196.6 against 168.3 tok/s.
The previous image's numbers were measured on other hosts, so part of the difference can be the hosts.

### 16 users, 131,072-token context (previous image `0.24.0-3dce07e-233b250`, vLLM 0.31.0)

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.24 s | 27.7 ms | 36.1 | 1.20 s | 29.4 ms | 34.0 |
| 2 | 1.51 s | 33.7 ms | 29.7 | 1.41 s | 35.4 ms | 28.2 |
| 4 | 1.54 s | 43.7 ms | 22.9 | 1.60 s | 42.5 ms | 23.5 |
| 8 | 1.96 s | 52.5 ms | 19.0 | 1.99 s | 56.1 ms | 17.8 |
| 16 | 2.75 s | 80.1 ms | 12.5 | 2.96 s | 78.4 ms | 12.8 |

Total output throughput (greedy): 30.9 / 50.0 / 72.4 / 121.4 / 168.3 tok/s at 1 / 2 / 4 / 8 / 16 requests.
These are within noise of the image before it (vLLM 0.26): TPOT parity, no regression from vLLM 0.31.

- **TPOT is a serving metric, not decode speed.** The benchmark keeps the number of requests in flight
  constant, so every finishing request is replaced by a new one, and that new request's prefill (about
  1.2-1.5 s) pauses everyone else's decode. At 16 users most of the 75.8 ms is those pauses.
- **Decode speed with no arrivals.** One wave of 16 simultaneous requests (same prompts, 256 tokens): the
  server runs a spec round in about 107 ms and gets about 3.7 tokens per round, which is **about 30 ms per
  token (33 tok/s/user)** per user; the per-user mean measured by the client is 31.5-32.3 tok/s/user
  (the request that arrives first prefills alone and ends up slower, ~20 tok/s/user). The demo
  (`models/demos/blackhole/deepseek_v41_flash`) on the same 16 prompts gives 34.3 tok/s/user.
- **Prompts matter for speculative decoding.** Short zero-shot GSM8K questions accept more drafts (4.3
  tokens per round) and decode at about 42 tok/s/user in the demo; the 8-shot prompts above give 3.7-3.9
  tokens per round.
- **Speculative-decode statistics** during the benchmark (server log): 3.7-3.95 tokens per round (greedy),
  3.6-3.8 (sampled); a round takes 99 ms at one user and 108 ms at 16; the host gap between rounds is
  4.5-7 ms.
- **Accuracy on this image.** GSM8K 8-shot (`dsv41_gsm8k_cot`, all 1319 questions, `evals` workflow run from this image
  against the 16-user, 131,072-token server of this guide): **95.45 %** flexible-extract (PASS). Earlier images scored
  95.75 / 95.83 / 96.29 %; the model card lists 93.0 for the base model.
- A single-user request with a ~93,000-token prompt returned a correct two-sentence summary in 105 s on an
  earlier commit of this server (`e41a8985e97`, before the dense position bucket; host checkout, not this
  image). The long-prompt rate was not re-measured on this image.

## Startup time

Time from `docker run` until `GET /health` returns 200:

| | Time |
|---|---|
| Weight cache and kernel cache both populated for this configuration (a later start on the same host) | ~13 min (13.4 min, measured twice on one host) |
| Weight cache populated, this configuration not yet run on the host (kernel cache still to fill) | ~30-33 min (29.5 and 32.6 min measured on one host: about 13 min building the 40 layers, then about 16-19 min compiling kernels and capturing the prefill and spec traces) |
| This image, both caches populated (16-user speculative server, 40,000-token context) | 14.0 min (`docker run` to `/health`) |
| This image with the runtime spec of this guide (131,072-token context), weight cache populated, kernel cache from earlier runs on the host | 20.1 min (`docker run` 18:47:55, `/health` 19:07:59); 13.4 min on a repeat start of the same server (19:25:35 to 19:39:00) |
| This image (vLLM 0.31) with the runtime spec of this guide, weight and kernel caches populated | 16.6-22 min over five starts on three hosts (first start on a host that had run another image: 20.2 min) |
| Image `0.24.0-799a223-ffd1a57` with `assets/runtime_spec_b16_k5_131072.json` (16 users, 131,072 tokens), weight and kernel caches populated | 23.7 min (`docker run` 11:36:54, `/health` 12:00:35) |
| Same image, `assets/runtime_spec_b16_k5_262144_vision.json` (16 users, 262,144 tokens, image input) | 24.7 min (09:03:44 to 09:28:26) |
| Same image, `assets/runtime_spec_b4_k5_1048576.json` (4 users, 1,048,576 tokens) | 30.9 min (10:37:28 to 11:08:20): about 20 min for the model (40 layers and the Engram tables), 7 min for the speculative runner, 2 min for the warm-up and trace captures |
| First start with nothing cached | not re-measured on this image; earlier first starts of the same weights took 60-90 min |

Two caches make later starts fast; both are mounted in the `docker run` below:

- **Weight cache** (`/cache/weights`, about 610 GB): the checkpoint converted to the on-device layout. The
  first start writes it.
- **Kernel cache** (`/cache/kernels`, about 43 GB): compiled device kernels. Without this mount they are
  thrown away with the container and recompiled on every start.

Keep both on a disk local to the host (we used a local NVMe), not on a network file system.

## Launch the server

### 1. Prerequisites (once per host)

- Tenstorrent driver and firmware installed, all 32 chips visible (`ls /dev/tenstorrent`).
- 1 GiB hugepages mounted at `/dev/hugepages-1G`.
- No other process using the chips (one server per galaxy).
- Disk: 476 GB for the checkpoint, about 610 GB for the converted-weight cache and about 43 GB for the kernel
  cache (the weight and kernel caches should be on a local disk).
- Docker, and access to the image registry (see the table above).

### 2. Pull the image (optional)

`docker run` pulls the image by itself the first time. To pull ahead of time:

```bash
export IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-3dce07e-233b250
docker pull $IMAGE
```

### 3. Prepare the checkpoint, caches and runtime spec

```bash
export CKPT=/path/to/deepseek-v41-flash        # the model checkpoint (read-only for the container)
export CACHE=/path/to/dsv41_cache              # weights/ and kernels/ below; reused across starts
export STATE=/path/to/dsv41_state              # small scratch dirs the server writes to
export SPEC=$PWD/assets/runtime_spec_b16_k5_131072.json   # run from models/demos/blackhole/deepseek_v41_flash/ in tt-metal (see below)
export IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-3dce07e-233b250   # or the image you built locally

mkdir -p $CKPT $CACHE/weights $CACHE/unified $CACHE/kernels $STATE/cache_root $STATE/logs $STATE/traces

# the checkpoint (476 GB); a Hugging Face token is needed if the repo asks for one
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir $CKPT

# vLLM looks the model up by name in the Hugging Face cache even when offline: give it the three small
# files (config + tokenizer, from the checkpoint you just downloaded)
R=$STATE/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
mkdir -p $R/refs $R/snapshots/local && echo -n local > $R/refs/main
cp $CKPT/config.json $CKPT/tokenizer.json $CKPT/tokenizer_config.json $R/snapshots/local/
```

`$CACHE` and `$STATE` must be writable by the user the container runs as (the `docker run` below uses your
own user id); `$CKPT` only needs to be readable.

The runtime spec is `assets/runtime_spec_b16_k5_131072.json`, next to this guide (`models/demos/blackhole/deepseek_v41_flash/` in tt-metal). It
sets the server arguments and environment shown in the first table, with in-container paths (`/ckpt`,
`/cache/...`), so it does not need editing when you mount your directories at those paths.

| Flag | What it is for |
|---|---|
| `--network host` | The server listens on port 8000 of the host. |
| `--user $(id -u):$(id -g)` + `-v /etc/passwd:/etc/passwd:ro` | Run as your user so the cache directories stay yours; the image has no entry for your uid and Python crashes without one, so the mounted `passwd` provides it. |
| `-v $CKPT:/ckpt:ro` + `MODEL_WEIGHTS_DIR=/ckpt` | The model checkpoint. |
| `-v $CACHE:/cache` | Weight cache and kernel cache (see Startup time). |
| `-v $STATE/hf:/hf` + `HF_HOME=/hf` + `HF_HUB_OFFLINE=1` | The three-file Hugging Face cache entry; offline mode means no network at runtime. |
| `-v $STATE/cache_root:.../cache_root`, `-v $STATE/logs:.../logs` | Scratch directories the server writes at start (those paths in the image are not writable by your user). |
| `-v $STATE/traces:.../build/profiler/build_wasm/traces` | The profiler needs a writable directory here; without it the server aborts at start with `Critical nanobind error: refusing to add duplicate key "ROW_MAJOR"...`. |
| `-v $SPEC:.../rtspec.json:ro` + `RUNTIME_MODEL_SPEC_JSON_PATH` | The runtime spec above. |
| `--ulimit memlock=-1`, `--ipc host`, hugepages mount, `--device /dev/tenstorrent` | Device access and pinned memory. |

### 4. Start the server

```bash
docker run -d --rm --name dsv41 \
  --network host --ipc host --ulimit memlock=-1 \
  --user $(id -u):$(id -g) \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $CKPT:/ckpt:ro \
  -v $CACHE:/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v $STATE/hf:/hf \
  -v $STATE/cache_root:/home/container_app_user/cache_root \
  -v $STATE/logs:/home/container_app_user/logs \
  -v $STATE/traces:/home/container_app_user/tt-metal/build/profiler/build_wasm/traces \
  -v $SPEC:/home/container_app_user/model_specs/rtspec.json:ro \
  -e RUNTIME_MODEL_SPEC_JSON_PATH=/home/container_app_user/model_specs/rtspec.json \
  -e MODEL_WEIGHTS_DIR=/ckpt \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) \
  $IMAGE \
  --model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy --no-auth
```

### 5. Wait until it is ready

```bash
until curl -sf -o /dev/null http://localhost:8000/health; do sleep 10; done; echo ready
```

Progress is in `docker logs -f dsv41`. While loading you will see `built layer N (...s)` lines for the 40
layers; after that the server captures its decode and prefill traces before `/health` turns 200.

### 6. Stop the server

```bash
docker stop dsv41
```

If the server ever crashes or is killed, reset the board before starting again: `tt-smi -glx_reset`.

## Build the image locally from the repos and run the server with it

Use this to run commits that are not in a published image: build an image from local checkouts of
`tt-metal`, `tt-inference-server` and `vllm-tt-plugin`, then start the server from that image with the same
`docker run` as step 4. The build script is `scripts/build_local_dsv41_image.sh` in tt-inference-server; it
uses `vllm-tt-metal/vllm.tt-metal.local.Dockerfile`, which copies git exports of the three repos instead of
cloning them. It needs network access (PyPI, GitHub, ghcr.io, download.pytorch.org) and Docker with buildx;
it does not need the chips.

### A. Check out the three repos

```bash
mkdir -p ~/dsv41 && cd ~/dsv41
git clone https://github.com/tenstorrent/tt-metal.git
git clone https://github.com/tenstorrent/tt-inference-server.git
git clone https://github.com/tenstorrent/vllm-tt-plugin.git

git -C tt-metal checkout ssinghal/dsv4p1 && git -C tt-metal submodule update --init --recursive
git -C tt-inference-server checkout ssinghal/dsv4p1
git -C vllm-tt-plugin checkout ssinghal/dsv4p1
```

The branches must exist on the remote (or use your own branch or commit). The submodules of tt-metal must be
initialised: the script exports each one at the commit recorded in the superproject. **Only committed code
goes into the image**: each repo is exported at its `HEAD` when the script starts, and uncommitted changes are
ignored. The three commits are recorded in `/home/container_app_user/BUILD_INFO` in the image.

### B. Build the image

```bash
cd ~/dsv41/tt-inference-server
TT_METAL_DIR=~/dsv41/tt-metal \
TTIS_DIR=~/dsv41/tt-inference-server \
PLUGIN_DIR=~/dsv41/vllm-tt-plugin \
CTX_ROOT=~/dsv41/docker-ctx \
  scripts/build_local_dsv41_image.sh
```

The script exports the sources into `$CTX_ROOT/<tt-metal-11>-<tt-inference-server-8>-<plugin-7>`, builds the
toolchain base image `tt-metalium-ci-build:local-<tt-metal-11>` from tt-metal's own `dockerfile/docker-bake.hcl`
(4.7 GB), then builds the server image and prints `BUILT dsv41-flash-local:<tt-metal-11>-<tt-inference-server-8>-<plugin-7>`
(for this guide's commits: `dsv41-flash-local:3dce07e3792-233b250e-6b9ef2a`).

| Variable / flag | Use |
|---|---|
| `--skip-base` | do not build the toolchain base; reuse the image named by `BASE_TAG` (default `tt-metalium-ci-build:local-<tt-metal-11>`) |
| `BASE_TAG` | an existing toolchain base to reuse, for example `tt-metalium-ci-build:local-170e8b7e819` |
| `METAL_SHA`, `TTIS_SHA`, `PLUGIN_SHA` | export these commits instead of each repo's `HEAD` |
| `REUSE_CTX=1` | reuse an already exported context of the same tag (resume after a failed build) |
| `DOCKER_HOST` | build with another Docker daemon, for example a private one whose data root is on a large disk |
| `BUILD_NETWORK=host` | needed when the daemon runs with `--bridge=none` |
| `BUILD_JOBS` | parallel jobs of the tt-metal C++ build (default 16) |
| `IMAGE_REPO` | image name (default `dsv41-flash-local`) |

Notes on the build:

- **First build.** It compiles tt-metal in the builder stage, which is the long part; we did not time a
  complete first build from nothing. The C++ layers are keyed on the tt-metal tree **without** `models/`, so
  a change that only touches `models/`, the plugin or tt-inference-server rebuilds only the cheap layers (a
  rebuild of the final stage took about 15 minutes here, most of it exporting the 21 GB image).
- **Disk.** The finished image is 21.4 GB and the toolchain base 4.7 GB; the build cache adds more. Put the
  Docker data root on a large disk.
- **A board reset right after a build.** If the Docker data root is on a local disk that `tt-smi -glx_reset`
  detaches, run `sync` before the reset and restart the Docker daemon after it. We once reset immediately after
  a build and the image came back with empty files (the container then exited silently with code 0). The check
  below catches it.

Check the image:

```bash
IMAGE=dsv41-flash-local:3dce07e3792-233b250e-6b9ef2a     # the tag the script printed
docker run --rm --entrypoint cat $IMAGE /home/container_app_user/BUILD_INFO
docker run --rm --entrypoint wc $IMAGE -c /home/container_app_user/app/src/run_vllm_api_server.py   # 53132 bytes, not 0
```

Optional: give it the registry-style name used above, so the rest of this guide applies unchanged:

```bash
docker tag $IMAGE ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-<tt-metal-7>-<tt-inference-server-7>
```

### C. Run the server with the local image

Do steps 1 and 3 of "Launch the server" (prerequisites; checkout, caches and runtime spec) once, then use the
local image in step 4:

```bash
export IMAGE=dsv41-flash-local:3dce07e3792-233b250e-6b9ef2a     # the image you built
# step 4: the same docker run as above, which uses $IMAGE
```

Steps 5 (wait for `/health`) and 6 (stop) are unchanged, and so is the request section below. To change what the
server runs, commit the change in the repo, rebuild with the script (only the cheap layers rebuild for `models/`,
plugin and tt-inference-server changes), and start a new container from the new tag.

## Send requests

The server speaks the OpenAI API on port 8000. It was started with `--no-auth`, so no API key is needed.

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "A train travels 120 km in 1.5 hours. What is its average speed?"}],
  "max_tokens": 256,
  "temperature": 0.6, "top_p": 0.95
}'
```

Streaming (`"stream": true`) works the same way. On 2026-10-15 the steps of this guide were run again with the image tag above on two
hosts (`assets/runtime_spec_b16_k5_131072.json`, vLLM 0.31.0): the server was ready after 16.6-22 min, chat and completion requests were
answered correctly (arithmetic, a 20,000-token needle-in-a-haystack), GSM8K scored 95.45 %, and the 1 / 2 / 4 / 8 / 16-request benchmarks above ran.

### Request fields

Checked against a live server of this image by `ctr2/serving_api_test.py`-style checks (23 of 23 passed at 16 users / 262,144 tokens; see "Tested on this image").

| Field | Notes |
|---|---|
| `model` | `deepseek-ai/DeepSeek-V4.1-Flash`; an unknown name gets HTTP 404 (`NotFoundError`) |
| `max_tokens` | prompt + `max_tokens` must stay within the server's `max_model_len`; a longer prompt, or a `max_tokens` above `max_model_len`, gets HTTP 400 with the limit in the message |
| `temperature`, `top_p`, `top_k` | sampled on the device; `temperature` 0 (or `top_k` 1) is greedy (`top_k` 1 gave the same text as greedy). The model card recommends temperature 1.0 and top-p 0.95. Top-k up to 32 takes the fastest path. A negative `temperature` gets HTTP 400. |
| `seed` | a seeded request gives the same output whatever slot or batch position it runs in (same seed twice: identical text; different seeds: different text) |
| `stop`, `stream`, `stream_options.include_usage` | work as usual (`finish_reason` is `stop` / `length`, the stream ends with `[DONE]`, usage is sent when asked) |
| `logprobs`, structured output | **not supported** with speculative decoding: the request is rejected with HTTP 400 and an OpenAI-style error body that says to drop the control or to run the server without speculative decoding |
| `min_p`, `logit_bias` | **rejected** with HTTP 400 (`not yet supported with speculative decoding`) |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | accepted; per the server's own message these requests decode without speculation (slower) |
| `n` | `n` 2 returns two choices |
| `/v1/completions`, `/v1/models`, `/health` | work; `/health` stays 200 under overload and after a client disconnects mid-stream |
| Images | up to 16 per prompt with the vision runtime specs (see "Image input"); more get HTTP 400 |

More requests than `max_num_seqs` can be sent at once: the extra ones wait in vLLM's queue (tested with 4x the slot count, two waves, every answer correct).

## Image input (optional)

The server can also take images in chat messages. This needs a vision runtime spec instead of the text one (set `SPEC` to it in step 3; everything else in the
launch is unchanged): `assets/runtime_spec_b16_k5_131072_vision.json` (16 users, 131,072 tokens), `assets/runtime_spec_b16_k5_262144_vision.json` (16 users, 262,144 tokens)
or the 4-user long-context spec (see "Long context"). A vision spec is the text spec plus `limit-mm-per-prompt {"image": 16}`, `mm-processor-cache-gb 0`,
`chat-template-content-format openai`, `DSV41_VISION=1` and the modalities `text` + `image`. The vision tower costs about 22 MB of device memory per DRAM bank
(140 MiB of weights and about 40 MiB of input buffers per chip) and a slightly longer start; text-only requests work as before, with the same speed.

```bash
IMG=$(base64 -w0 photo.png)
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}},
    {"type": "text", "text": "What does the image say?"}]}],
  "max_tokens": 128, "temperature": 0
}'
```

Limits:

| | |
|---|---|
| Images per prompt | **16** (`--limit-mm-per-prompt`; the model card sets no limit of its own). A request with more is refused with an OpenAI-style HTTP 400 error and the server keeps serving: `{"error": {"message": "At most 16 image(s) may be provided in one prompt. (parameter=image)", "type": "BadRequestError", "param": "image", "code": 400}}` |
| Size of one image | at most 1,024 tokens in the model's input (`max_image_tokens`): a 31 x 31 token grid, one token per 42 x 42 pixels, about 1302 x 1302 pixels (8,649 ViT patches of 14 x 14); larger images are resized, keeping the aspect ratio. The smallest image is about 544 x 544 pixels (184 tokens). |
| Tokens of 16 images | 16 small images (512 x 384) are about 3,150 prompt tokens, a full-size image about 1,000 |

Checked on this image (16 users, 262,144 tokens, speculative decoding, vision on): text only; one image (a red square labelled "CAT 42" read correctly); two images; 4 image + 4 text requests at
once; streaming; 16 numbered images in one prompt (all 16 labels returned in order); one 2400 x 1800 image resized to 1,019 tokens and answered correctly; 17 images refused with the 400 above and the server
still serving; and a 16-concurrent stress run (number exact in 48 of 48 images, both colours right in 48 of 48).

## Long context

Two more serving configurations ship with this image. All three share one budget: about 1M tokens of context per mesh row (4 rows), users x context per row.
Use the spec as `SPEC` in step 3; the launch is otherwise unchanged. Startup takes 25-31 minutes with both caches populated (table in "Startup time").

| Configuration | Runtime spec | Users | Context | Speculative decoding | Image input |
|---|---|---|---|---|---|
| Default | `assets/runtime_spec_b16_k5_131072.json` (text), `assets/runtime_spec_b16_k5_131072_vision.json` | 16 | 131,072 | k = 5 | with the vision spec |
| Long context, 16 users | `assets/runtime_spec_b16_k5_262144_vision.json` | 16 | 262,144 | k = 5 | yes (up to 16 images per prompt) |
| Long context, 4 users | `assets/runtime_spec_b4_k5_1048576.json` | 4 | 1,048,576 | k = 5 | **no** (see below) |

What differs from the default spec: `max_model_len` and `max_num_seqs`, `DSV41_VLLM_CHUNK` 512 (prefill chunk), a smaller `trace_region_size` (0.8e9 at 262,144 tokens, 0.65e9 at 1,048,576: the tables of a long context leave no room for the 1.6e9 default),
and `DSV41_SPEC_MAX_CTX=262144` in the 16-user long-context spec (the measured context cap of speculative decoding at 16 users is 131,584 unless you set it; `DSV41_SPEC_MAX_CTX` is the final word, the vLLM-side guard follows it).
Above 131,584 tokens the speculative verify step uses the matmul indexer backend (forced at build).

What makes a long context fit (branch `ssinghal/dsv4p1`): compact RoPE position tables (768 B per position per chip, column-sharded from 262,144 positions), the prefill
indexer's key FIFO, scores and visibility mask sharded over the 8 mesh columns (from 131,072 index entries; `DSV41_PIDX_SHARD`), dense prefill masks only for the layers that
run the dense path, the kv-table FIFO shifted in place in row pieces (`DSV41_PF_FIFO_CHUNK`), one extra eager step before each trace capture (prefill, decode, spec), and the index-key hand-off
and the speculative score computed in pieces (`DSV41_KEY_EXPORT_PIECE`, `DSV41_SPEC_SCORE_PIECE`). The prefill kv table is still replicated over the mesh columns.

### 16 users, 262,144 tokens, with image input

Tested on this image with `assets/runtime_spec_b16_k5_262144_vision.json` on one persistent server: sanity, a 20,020-token prompt (4817 found in 23.5 s), a 100,020-token needle at the end and at the start of the prompt (4817 both times, prefill 115.7 s and 117.1 s at chunk 512),
the image checks of "Image input", 64 concurrent requests on the 16 slots (two waves, all correct, health 200), 23 of 23 API checks. Speed: the table in "Performance".

### 4 users, 1,048,576 tokens (text)

Tested on this image with `assets/runtime_spec_b4_k5_1048576.json` (text only, speculative decoding k = 5) on one persistent server: sanity, a 20,020-token prompt (4817 in 32.3 s), a 100,020-token needle at the end and at the start of the prompt (4817 both times,
prefill 157.7 s and 159.2 s at chunk 512; a prompt that long is a long first token), 16 concurrent requests on the 4 slots (two waves, all correct), 23 of 23 API checks.

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.49 s | 29.8 ms | 33.5 | 1.51 s | 30.9 ms | 32.4 |
| 2 | 1.74 s | 35.2 ms | 28.4 | 1.78 s | 39.6 ms | 25.3 |
| 4 | 1.86 s | 45.5 ms | 22.0 | 1.99 s | 44.8 ms | 22.3 |

Total output throughput (greedy): 28.1 / 47.3 / 74.5 tok/s at 1 / 2 / 4 requests.

The 1M server is slower than the 262,144-token one at the same concurrency (TPOT 45.5 against 36.1 ms at 4 requests, TTFT about 0.4 s higher): every decode step and every prefill chunk scores the index entries of a 1M context.
Speculative decoding runs at this context (3.31 accepted drafts per round, 1.81x plain decode in the 4-user demo).

**Image input does not fit at 4 users and about 1M tokens.** With the vision tower loaded (about 22 MB per DRAM bank), the server runs out of DRAM during the warm-up of the prefill trace: a 160 MiB buffer needs one contiguous block of 21 MB per bank and the largest block left is 17-19 MB (79-108 MB per bank are free in total, fragmented around the persistent tables).
This happened at 1,048,576 tokens with chunk 512 and chunk 256, and at 1,000,448 tokens. Image input is available with 16 users up to 262,144 tokens (above) and with 4 users up to 524,288 tokens (next section).

### 4 users, 524,288 tokens, with image input

`assets/runtime_spec_b4_k5_524288_vision.json` (4 users, 524,288 tokens, speculative decoding k = 5, image input, trace region 0.65e9). Tested on this image on one persistent server: ready 26 minutes after the start (warm-up 123 s), sanity and a 20,020-token prompt (4817), a 100,020-token needle at the end and at the start of the prompt (4817 both times, prefill 132.4 s and 134.0 s),
all the image checks of "Image input" (16 images in one prompt, the 17-image HTTP 400, 48 of 48 images in the stress run), 16 concurrent requests on the 4 slots (two waves, all correct), 23 of 23 API checks, and the sweeps:

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.26 s | 31.8 ms | 31.4 | 1.27 s | 31.3 ms | 31.9 |
| 2 | 1.61 s | 33.9 ms | 29.5 | 1.79 s | 33.3 ms | 30.0 |
| 4 | 1.79 s | 40.4 ms | 24.7 | 1.67 s | 43.5 ms | 23.0 |

Total output throughput (greedy): 27.3 / 48.2 / 81.2 tok/s at 1 / 2 / 4 requests.

Other (users x context) points measured on the same code (plain decode, chunk 512, trace region 800,000,000, 100,020-token needle at the end, 391 / Paris sanity; earlier tree, not re-run on this image):

| Users | Context | Tokens per mesh row | Ready | Needle (4817) prefill | |
|---|---|---|---|---|---|
| 4 | 1,048,576 | 1,048,576 | 13-24 min | 163 s | works |
| 8 | 524,288 | 1,048,576 | 22 min | 141 s | works |
| 16 | 262,144 | 1,048,576 | 27 min | 121 s | works |
| 8 | 1,048,576 | 2,097,152 | - | - | out of DRAM during warm-up (32 MB free per bank at a 128 MiB allocation) |

Users times context per mesh row is the budget that fits (the paged KV pool and the per-user prefill tables scale with it). A 2-users-per-row decode at a 262,144-slot index needs the per-user ranking of the decode indexer (`SELECT_CONCAT_BYTES`): without it the score rows of two users do not fit a core's L1.

Not measured: a prompt close to 1,000,000 tokens (estimated ~12 min of prefill from the 100k run).

## Run the GSM8K evaluation

The accuracy numbers in this guide come from tt-inference-server's `evals` workflow, run from the same image
against the running server. `assets/run_gsm8k_eval.sh` does that: it starts a second container of the image
(the client) that runs `run.py --workflow evals` against `http://127.0.0.1:8000`.

| | |
|---|---|
| Task | `dsv41_gsm8k_cot`: lm-eval's `gsm8k_cot` (the 8 chain-of-thought exemplars of Wei et al.), copied to `reference_config/evals/lm_eval_tasks/dsv41_gsm8k_cot.yaml` in tt-inference-server with the namespaced dataset id `openai/gsm8k`. The task and its `EvalTask` entry in `reference_config/evals/eval_config.py` are both in the image. |
| Questions | all 1319 of the GSM8K test split, 8-shot, through the chat API with the model's chat template |
| Sampling | temperature 1.0, top-p 0.95, `do_sample`, up to 2048 generated tokens, stop at `Q:` (the start of the next exemplar), streaming; `max_length` 40000 |
| Concurrency | `DSV41_EVAL_CONCURRENCY`, 16 here (the 16-user server); the task's own default is 32, for the 32-user build |
| Score | `exact_match,flexible-extract` in percent. The model card lists 93.0 for the base model (a different protocol); the workflow's check passes within 5 % of it. `strict-match` is also reported, but it is lower because the answers end "The answer is $64." and its regex wants a digit right after "is". |

Start the server as in "Launch the server" (steps 1-5), then:

```bash
cd models/demos/blackhole/deepseek_v41_flash        # in the tt-metal checkout
IMAGE=$IMAGE CKPT=$CKPT SPEC=$PWD/assets/runtime_spec_b16_k5_131072.json STATE=$STATE ./assets/run_gsm8k_eval.sh
```

- `CKPT`, `SPEC`, `STATE` and `IMAGE` are the variables of step 3. The script also takes `URL` (default
  `http://127.0.0.1:8000`), `CONCURRENCY` (default 16), `HF_CACHE` and `DOCKER`.
- **Checkpoint mount.** `eval_config.py` names the tokenizer by the checkpoint's site path
  (`/mnt/tt-data/ssinghal/deepseek-v41-flash`), so the script mounts your checkpoint at that path inside the
  client container.
- **Network.** The client downloads the dataset (`openai/gsm8k`) from the Hugging Face hub and, on its first run,
  installs the workflow's Python environment (`uv`, PyPI). Both go into `$STATE/eval/` and are reused later.
- **Entry script.** `assets/eval_client_entry.sh` is mounted as the client's entry point: `run.py` calls `git rev-parse HEAD` on
  its own tree, and the tree in the image is an empty `git init` owned by another user, so the script trusts it
  and commits it before starting `run.py`.
- **Output.** The report is `$STATE/eval/cache_root/workflow_logs/reports_output/evals/report_*.md` (the score table at the
  bottom); the run logs are in `$STATE/eval/cache_root/workflow_logs/run_logs/`.
- **Time.** About 20 minutes for the 1319 questions at concurrency 16 (20.4 min measured in a container run), plus the
  first-run setup (about 15 minutes measured: managed Python, the workflow environment and the dataset; later runs skip it).
- **Quick check.** Extra arguments go to `run.py`: `./assets/run_gsm8k_eval.sh --limit-samples-mode smoke-test` runs 8 questions.

Results with this protocol on this image (`0.24.0-799a223-ffd1a57`, speculative decoding k = 5, flexible-extract in percent, all 1319 questions, the standard error is about 0.6):

| Server | Concurrency | Flexible-extract | Strict-match |
|---|---|---|---|
| 16 users, 131,072 tokens (`assets/runtime_spec_b16_k5_131072.json`) | 16 | **95.22** | 73.9 |
| 16 users, 131,072 tokens, image input on | 16 | **95.38** | 72.5 |
| 4 users, 1,048,576 tokens (`assets/runtime_spec_b4_k5_1048576.json`) | 4 | **94.92** | 65.1 |
| 16 users, 262,144 tokens, image input on (`assets/runtime_spec_b16_k5_262144_vision.json`) | 16 | **93.56** and **93.48** (two runs on the same server) | 70.1 and 70.5 |

All pass the workflow's check (within 5 % of the model card's 93.0). The previous image scored 95.45 % (95.75 / 95.83 / 96.29 on earlier images); sampling at temperature 1.0 makes a run differ from the next by about half a percent.
**The 16-user 262,144-token server scores about 2 points lower than the other configurations** (two runs, 93.5 each, about 3 standard errors below 95.4), so treat 262,144 tokens at 16 users as a configuration that has not reached the accuracy of the default.
Checks made on this image to find the cause (16 users, text unless stated, same method): vision on or off makes no difference (131,072 tokens: 95.38 % with, 95.22 % without; 262,144 tokens: 93.5 % with, 93.18 % without and with the column-sharded RoPE tables switched off);
the new code is not the cause at the default context (95.22 % at 131,072 tokens on this image); 163,840 tokens scored 94.54 %. The cause is **not identified yet**. The 262,144-token server passes every functional test (a code hidden in a 100,020-token prompt is found at both ends, 23 of 23 API checks, overload, image checks) and its speed is the same as the default's;
where GSM8K-level accuracy matters, serve the default 131,072-token spec (or the 4-user 1,048,576-token spec, 94.92 %), and treat the 16-user 262,144-token configuration as one that scores about 2 points lower until this is resolved.


## Notes

- **Concurrency.** 16 requests decode at once; the rest wait. A new request's prefill pauses the running
  requests for about 1.2-1.5 s (700-token prompt), so mixed traffic shows lower per-user speed than the
  decode numbers above.
- **Context.** The server is built for 131,072 tokens at 16 users. Decode speed does not depend on that
  size for short prompts (the work follows the actual position), but TTFT of a long prompt grows with the
  prompt length.
- **No thinking mode.** The served chat template is the non-thinking format.
- **One server per galaxy.** Stop a running server before starting another (or running a demo).
- **The runtime spec disables the build-slot lock** (`DSV41_BUILD_SLOTS=0`). By default the server takes a
  slot in `/mnt/tt-data/ssinghal/buildslots`, a directory that only exists on our shared hosts, to limit how
  many hosts convert weights at once; on any other host the server fails at start with
  `PermissionError: ... '/mnt/tt-data'` unless it is disabled.
- **Running the workflows (`evals`, `benchmarks`) from the image.** The image ships the client side of
  tt-inference-server (`/home/container_app_user/tt-inference-server`, run with
  `/home/container_app_user/tt-metal/python_env/bin/python run.py ...`). `assets/run_gsm8k_eval.sh` wraps the
  `evals` workflow for GSM8K (section "Run the GSM8K evaluation"); there is no such wrapper for `benchmarks` in the
  repos yet.
- **Hugging Face cache must be a real directory.** vLLM resolves the model name through `$HF_HOME`. If `~/.cache/huggingface/hub` on your
  host is a symlink (for example to a shared file system) and you mount that path into the container, the link dangles inside it and the server
  fails to find the model. Mount the real directory created in step 3 (`$STATE/hf`) as `/hf`, as the `docker run` above does.
