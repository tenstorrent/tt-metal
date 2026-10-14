# DeepSeek-V4.1-Flash on Blackhole Galaxy — tt-inference-server getting started

Text generation (chat and completions) through an OpenAI-compatible vLLM server: 16 concurrent users, up to
131,072 tokens of context (prompt + output), speculative decoding on (5 draft tokens per round), on-device
sampling.

## Docker image

| | |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-b983060-233b250` |
| Size | 21.4 GB on disk (5.3 GB compressed), built 2026-10-14 |
| Access | `docker run` pulls the image automatically on a new host. Pulling needs access to the `tenstorrent` packages on ghcr.io (check the digest with `docker image inspect <image> --format '{{.RepoDigests}}'`) |
| tt-metal | `b983060a726…` in the tag (branch `ssinghal/dsv4p1`): this guide and its runtime spec on top of `234b28cdb725e52d08becb3f1e19c60884fe66ca`, the code the image was built from (it adds the routed-expert SwiGLU clamp at every MoE site, see below; `/home/container_app_user/BUILD_INFO` in the image records `234b28cdb725…`) |
| tt-inference-server | `233b250e85148bc25853786aff54c0949a994b76` (branch `ssinghal/dsv4p1`) |
| vllm-tt-plugin | `ad9eed9ad9f7295fda00b8ec788dc30b34587647` (branch `ssinghal/dsv4p1`; the scheduler no longer splits a prompt across prefill steps when the token budget holds a whole prompt) |
| Model | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (476 GB of safetensors, 48 shards) |
| Hardware | 1x Blackhole Galaxy (32 chips, 4x8 mesh) |

The image has no weights and no caches in it: the checkpoint, the converted-weight cache and the
compiled-kernel cache are mounted from the host (step 3).

What this build is configured for (the runtime spec in step 3 sets all of it):

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

**The speed numbers below were measured on the previous image (`0.24.0-901fa21-5b9d4ce`), before the SwiGLU clamp was added; they were not re-measured on this image.** The accuracy line at the end of this section was re-measured on this image.

All numbers: one Blackhole Galaxy, this image, 16-user server above, GSM8K 8-shot prompts (about 700
tokens each), 256 output tokens per request, `vllm bench serve` run by tt-inference-server's `benchmarks`
workflow from the same image. "Greedy" is temperature 0; "sampled" is temperature 0.6, top-k 20, top-p 0.95.

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|
| 1 | 1.24 s | 26.9 ms | 37.2 | 31.8 ms | 31.4 |
| 2 | 1.52 s | 31.7 ms | 31.5 | 33.3 ms | 30.0 |
| 4 | 1.53 s | 40.4 ms | 24.8 | 38.6 ms | 25.9 |
| 8 | 1.86 s | 53.1 ms | 18.8 | 53.0 ms | 18.9 |
| 16 | 2.62 s | 75.8 ms | 13.2 | 72.7 ms | 13.8 |

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
- **Accuracy.** GSM8K 8-shot (`dsv41_gsm8k_cot`, all 1319 questions, temperature 1.0, top-p 0.95, run through
  tt-inference-server's `evals` workflow from this image): **95.75 ± 0.56 %** flexible-extract (strict-match
  75.59 ± 1.18 %: the answers end "The answer is $64." and the strict regex wants a digit right after
  "is"). The model card lists 93.0 for the base model.
- **Accuracy on this image (with the SwiGLU clamp at all four sites).** GSM8K 8-shot (`dsv41_gsm8k_cot`, all 1319
  questions, `evals` workflow run from this image against a 16-user server with speculative decoding and a
  40,000-token context): **96.29 %** flexible-extract. The same configuration run as a host server (not in the
  container) also scored 96.29 %. Earlier runs of the previous image scored 95.75 / 95.83 %; the 131,072-token
  server configuration in this guide was not re-run for accuracy on this image.
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
| This image with the runtime spec of this guide (131,072-token context), weight cache populated, kernel cache from earlier runs on the host | 20.1 min (`docker run` 18:47:55, `/health` 19:07:59) |
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
export IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-b983060-233b250
docker pull $IMAGE
```

### 3. Prepare the checkpoint, caches and runtime spec

```bash
export CKPT=/path/to/deepseek-v41-flash        # the model checkpoint (read-only for the container)
export CACHE=/path/to/dsv41_cache              # weights/ and kernels/ below; reused across starts
export STATE=/path/to/dsv41_state              # small scratch dirs the server writes to
export SPEC=$PWD/assets/runtime_spec_b16_k5_131072.json   # run from models/demos/blackhole/deepseek_v41_flash/ in tt-metal (see below)
export IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-b983060-233b250   # or the image you built locally

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
(for this guide's commits: `dsv41-flash-local:234b28cdb72-233b250e-ad9eed9`).

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
IMAGE=dsv41-flash-local:234b28cdb72-233b250e-ad9eed9     # the tag the script printed
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
export IMAGE=dsv41-flash-local:234b28cdb72-233b250e-ad9eed9     # the image you built
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

Streaming (`"stream": true`) works the same way. On 2026-10-14 the steps of this guide were run again with the image tag above and
`assets/runtime_spec_b16_k5_131072.json`: the server was ready after 20.1 min, a chat request was answered correctly,
`/v1/models` reported `max_model_len` 131072, and `docker stop` ended the container cleanly. The commands in this guide were run end to end on one
Galaxy (a server started with the command in step 4, then the request above, a streaming request, the model
list, a seeded request twice, and the field checks in the table).

### Request fields

| Field | Notes |
|---|---|
| `model` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `max_tokens` | prompt + `max_tokens` must stay within 131,072 tokens |
| `temperature`, `top_p`, `top_k` | sampled on the device; `temperature` 0 (or `top_k` 1) is greedy. The model card recommends temperature 1.0 and top-p 0.95. Top-k up to 32 takes the fastest path. |
| `seed` | a seeded request gives the same output whatever slot or batch position it runs in |
| `stop`, `stream` | work as usual |
| `logprobs` | **not supported**: the request fails with an error that says to drop the control or to run the server without speculative decoding |
| `min_p`, `logit_bias` | **rejected** with HTTP 400 (`not yet supported with speculative decoding`) |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | accepted; per the server's own message these requests decode without speculation (slower) |
| `n` | `n` 2 returned two choices in a test |

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
  `/home/container_app_user/tt-metal/python_env/bin/python run.py ...`). Three things are needed that are
  not yet in the image: the directory must be a git repository with a commit (`run.py` calls
  `git rev-parse HEAD`), uv's managed Python must be installed on a mounted path
  (`UV_PYTHON_INSTALL_DIR`), and the workflow output directories must be mounted writable. Wrapper
  scripts for this are not part of the repos yet.
