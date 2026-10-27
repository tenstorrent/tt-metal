# DeepSeek-V4.1-Flash on Blackhole Galaxy: detailed guide (all configurations, request fields, image input, evaluation, performance tables; the short guide is GETTING_STARTED.md)

An OpenAI-compatible vLLM server for DeepSeek-V4.1-Flash on one Blackhole Galaxy (32 chips, 4x8 mesh).
It serves chat and completions, with optional image input.
All configurations use speculative decoding (5 draft tokens per round) and on-device sampling.
What changed in this release is in `RELEASE_NOTES.md`.

## Docker image

| | |
|---|---|
| Image | `dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6` (local build, not pushed to ghcr.io) |
| Image id / offline copy | `de6952f6f477`; `/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_afb78699c8b.tar` (5.38 GB); size on disk TODO; built 2026-10-27 |
| tt-metal | `afb78699c8b15104e8e9239d2bd1545a882091d9` (branch `ssinghal/dsv4p1-rebase-1017-v3`) |
| tt-inference-server | `5b6f9268ac7a36a19766425c6cc721503efe7646` (branch `ssinghal/dsv4p1-rebase-1017`) |
| vllm-tt-plugin | `dd4fbc6230610c4d12a436eaa29d45f7d2bc6c29` (branch `ssinghal/dsv4p1-replay-v3`) |
| vLLM | TODO: stack versions of this image (previous release: 0.31.0, torch 2.11.0+cpu, transformers 5.12.1); read them from the image (`BUILD_INFO`, pip list) |
| Model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (476 GB checkpoint) |
| Hardware | 1x Blackhole Galaxy |

`BUILD_INFO` in the image records the three commits.
The image is built from the three commits above without the documentation commits of branch `ssinghal/dsv4p1-rebase-1017-v3-docs` (documentation only; the `assets/` runtime specs in the image and in the repo are identical).
The compiled tt-metal and the Python environment are reused from the previous build; `models/`, the tt-inference-server tree and the plugin package come from the commits above.

The image has no weights and no caches.
The checkpoint and the two caches are mounted from the host (step 3).

## Configurations

Each configuration has a runtime spec in `assets/`.
Set `SPEC` to it in step 3.
Everything else in the launch stays the same.

| Configuration | Runtime spec | Users | Context | Image input |
|---|---|---|---|---|
| **Default, 16 users** | `runtime_spec_b16_k5_262144_vision.json` | 16 | 262,144 | yes |
| Smaller context, 16 users | `runtime_spec_b16_k5_131072.json` | 16 | 131,072 | no |
| Smaller context with images, 16 users | `runtime_spec_b16_k5_131072_vision.json` | 16 | 131,072 | yes |
| Long context, 4 users | `runtime_spec_b4_k5_1048576.json` | 4 | 1,048,576 | no |
| Long context with images, 4 users, 1M | `runtime_spec_b4_k5_1048576_vision.json` | 4 | 1,048,576 | yes |
| 16 users, 262,144, prefix caching + chunked prefill (opt-in) | `runtime_spec_b16_k5_262144_vision_apc.json` | 16 | 262,144 | yes |
| 4 users, 1M, prefix caching + chunked prefill (opt-in) | `runtime_spec_b4_k5_1048576_vision_apc.json` | 4 | 1,048,576 | yes |
| 4 users, 1M, prefix caching + chunked prefill, no images (opt-in) | `runtime_spec_b4_k5_1048576_apc.json` | 4 | 1,048,576 | no |
| 4 users, 524,288, prefix caching + chunked prefill (opt-in) | `runtime_spec_b4_k5_524288_vision_apc.json` | 4 | 524,288 | yes |

The 262,144-token and 1,048,576-token specs run with prefix caching and paged prefill on (`DSV41_APC=1`, `DSV41_PF_PAGED=read`) and the bounded prefill replay (`DSV41_PF_REPLAY`, default 2,560). The chunked-prefill token budget (`max_num_batched_tokens`) is 16,384 with `long-prefill-token-threshold` 4,096 at 4 users and 1,048,576 tokens, 67,584 at 16 users and 262,144 tokens. The `_apc` copies differ only at 16 users (budget 4,096). The 131,072-token specs keep prefix caching and paged prefill off (`DSV41_APC=0`, `DSV41_PF_PAGED=0`).

The image's own catalog (tt-inference-server `5b6f9268`) has four entries, all with prefix caching on, paged prefill on and image input: `deepseek-v41-flash` and `deepseek-v41-flash-vlm` (16 users, 262,144 tokens, budget 67,584) and `deepseek-v41-flash-1m` and `deepseek-v41-flash-vlm-1m` (4 users, 1,048,576 tokens, budget 16,384). A launch without a runtime spec (`--model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy`) serves the 4-user 1,048,576-token entry (`deepseek-v41-flash-1m` is the default implementation); `--impl deepseek-v41-flash` selects the 16-user entry. The catalog entries do not set `long-prefill-token-threshold` 4,096 in `vllm_args` the way the runtime specs do (the verification of this image booted the catalog default; whether the threshold is applied was not checked in its server log).

Shared settings:
- Speculative decoding: 5 draft tokens per round, lossless.
- Sampling on the device: temperature, top-k, top-p, per-request seed.
- KV cache: bf16 pool in the paged-prefill specs (prefix caching needs it), fp8 pool in the 131,072-token specs. Prefix caching is on in the 262,144 / 524,288 / 786,432 / 1,048,576-token specs; the `_apc` copies differ in the token budget at 16 users and 524,288 tokens (4,096).
- Chat template: non-thinking.
- More requests than users wait in the vLLM queue.

Context budget: about 1M tokens per mesh row (users x context per row).
Long contexts need a smaller trace region (0.8e9 at 262,144, 0.65e9 at 1M tokens).
`DSV41_SPEC_MAX_CTX` sets the context cap of speculative decoding.
It is 131,584 by default and 262,144 in the 16-user long-context spec.

## Quick start

### 1. Prerequisites (once per host)

- Tenstorrent driver and firmware installed, 32 chips visible (`ls /dev/tenstorrent`).
- 1 GiB hugepages mounted at `/dev/hugepages-1G`.
- No other process using the chips.
- Disk: 476 GB checkpoint, 610 GB weight cache, 43 GB kernel cache.
- Keep both caches on a local disk.
- Docker, and access to ghcr.io.

### 2. Load the image

The image is a local build: load the offline copy (`/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_afb78699c8b.tar`). It is not on ghcr.io yet.

```bash
export IMAGE=dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6   # local image: load the offline copy, there is nothing to pull
docker load -i /mnt/tt-data/ssinghal/dsv4-docker-ctx/img_afb78699c8b.tar
```

### 3. Prepare checkpoint, caches and spec

```bash
export CKPT=/path/to/deepseek-v41-flash-1017        # the checkpoint (read-only for the container)
export CACHE=/path/to/dsv41_cache_1017              # weights/ and kernels/, reused across starts
export STATE=/path/to/dsv41_state              # scratch dirs the server writes to
export SPEC=$PWD/assets/runtime_spec_b16_k5_262144_vision.json   # run from models/demos/blackhole/deepseek_v41_flash/ in tt-metal
export IMAGE=dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6

mkdir -p $CKPT $CACHE/weights $CACHE/unified $CACHE/kernels $STATE/cache_root $STATE/logs $STATE/traces

# the checkpoint (476 GB); a Hugging Face token may be needed
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir $CKPT

# vLLM looks the model up in the Hugging Face cache, even offline.
# Give it the three small files from the checkpoint.
R=$STATE/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
mkdir -p $R/refs $R/snapshots/local && echo -n local > $R/refs/main
cp $CKPT/config.json $CKPT/tokenizer.json $CKPT/tokenizer_config.json $R/snapshots/local/
```

`$CACHE` and `$STATE` must be writable by your user.
`$CKPT` only needs to be readable.
The specs use in-container paths (`/ckpt`, `/cache/...`).
Mount your directories there and the specs need no editing.

| Flag | Purpose |
|---|---|
| `--network host` | The server listens on port 8000. |
| `--user $(id -u):$(id -g)` + `-v /etc/passwd:/etc/passwd:ro` | Run as your user. Python needs a passwd entry for it. |
| `-v $CKPT:/ckpt:ro` + `MODEL_WEIGHTS_DIR=/ckpt` | The checkpoint. |
| `-v $CACHE:/cache` | Weight and kernel caches. |
| `-v $STATE/hf:/hf` + `HF_HOME=/hf` + `HF_HUB_OFFLINE=1` | The three-file Hugging Face entry. No network at runtime. |
| `-v $STATE/cache_root`, `-v $STATE/logs` | Scratch dirs the server writes at start. |
| `-v $STATE/traces:.../build/profiler/build_wasm/traces` | Needed by the profiler. Without it the server aborts at start (`refusing to add duplicate key "ROW_MAJOR"`). |
| `-v $SPEC:.../rtspec.json:ro` + `RUNTIME_MODEL_SPEC_JSON_PATH` | The runtime spec. |
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

Follow progress with `docker logs -f dsv41`.
You see `built layer N` lines for the 40 layers.
Then the server captures its traces.
After that `/health` returns 200.

### 6. Stop the server

```bash
docker stop dsv41
```

After a crash or a kill, reset the board before the next start: `tt-smi -glx_reset`.

## Startup time

Time from `docker run` to `/health` 200, with both caches populated:

| Configuration | Time |
|---|---|
| 16 users, 131,072 tokens | 23.7 min |
| 16 users, 262,144 tokens, images | 24.7 min |
| 4 users, 524,288 tokens, images | 26.4 min |
| 4 users, 1,048,576 tokens | 30.9 min on the previous release; this image: 20 to 30 min (1,217 / 1,526 / 1,773 / 1,787 s on four boots) |

The 1M start splits into about 20 min for the model (layers and Engram tables), 7 min for the speculative runner and 2 min for warm-up.
A first start with nothing cached takes much longer (60-90 min on earlier images, not re-measured).

Two caches make later starts fast:
- **Weight cache** (`/cache/weights`, about 610 GB): the converted checkpoint. The first start writes it.
- **Kernel cache** (`/cache/kernels`, about 43 GB): compiled kernels. Without it they are recompiled on every start.

## Send requests

The server speaks the OpenAI API on port 8000.
It runs with `--no-auth`.

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "A train travels 120 km in 1.5 hours. What is its average speed?"}],
  "max_tokens": 256,
  "temperature": 0.6, "top_p": 0.95
}'
```

Streaming (`"stream": true`) works the same way.

### Request fields

Checked on the previous image (`0.24.0-799a223-ffd1a57`) with the API tests in `tools/serving_tests/` (23 of 23 pass on each configuration). Not re-run on this image.

| Field | Behaviour |
|---|---|
| `model` | `deepseek-ai/DeepSeek-V4.1-Flash`. An unknown name gets HTTP 404. |
| `max_tokens` | Prompt plus `max_tokens` must fit `max_model_len`. Otherwise HTTP 400 with the limit. |
| `temperature`, `top_p`, `top_k` | Sampled on the device. `temperature` 0 or `top_k` 1 is greedy. Top-k up to 32 is fastest. A negative `temperature` gets HTTP 400. |
| `seed` | Same seed, same text, whatever slot or batch position. |
| `stop`, `stream`, `stream_options.include_usage` | Work as usual. |
| `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, `min_p` | Not supported with speculation. HTTP 400 with an OpenAI-style error. |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Accepted. Those requests decode without speculation (slower). |
| `n` | `n` 2 returns two choices. |
| `/v1/completions`, `/v1/models`, `/health` | Work. `/health` stays 200 under overload and after a client disconnect. |
| Images | Up to 16 per prompt with a vision spec. More gets HTTP 400. |

More requests than users can be sent at once.
The extra ones wait in the queue.
Tested at 4x the user count, two waves: every answer was correct.

## Image input

Use a vision spec (see Configurations).
It adds `limit-mm-per-prompt {"image": 16}`, `mm-processor-cache-gb 0`, `chat-template-content-format openai` and `DSV41_VISION=1`.
The vision tower takes about 22 MB of DRAM per bank.
Text requests run at the same speed as without it.

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

| Limit | Value |
|---|---|
| Images per prompt | **16**. The model card sets no limit. This is the server setting. |
| Error for more | HTTP 400, `{"error": {"message": "At most 16 image(s) may be provided in one prompt. (parameter=image)", "type": "BadRequestError", "param": "image", "code": 400}}`. The server keeps serving. |
| One image | At most 1,024 tokens: a 31 x 31 grid, 42 x 42 pixels per token, about 1302 x 1302 pixels. Larger images are resized. |
| Token cost | 16 small images (512 x 384): about 3,150 tokens. One full-size image: about 1,000. |

Verified configurations:
- 16 users up to 262,144 tokens.
- 4 users up to 524,288 tokens.

- 4 users up to 1,048,576 tokens (`runtime_spec_b4_k5_1048576_vision.json`).
  The Engram reduce-scatter of the prefill warm-up is split in 4 when vision is on, so the blocks it needs fit.

Checks on the previous image (vision on, speculation on); this image passed the image, 16-image and 16-concurrent stress tests on the 16-user 262,144 and the 4-user 1,048,576 vision configurations:
- Text, one image, two images, 4 image + 4 text at once, streaming.
- 16 numbered images in one prompt: all 16 labels returned.
- One 2400 x 1800 image: resized to 1,019 tokens, answered correctly.
- 17 images: the 400 above, server still serving.
- 16-concurrent stress: number exact in 48 of 48 images, colours right in 48 of 48.

## Long context

What makes a long context fit:
- Compact RoPE tables: 768 B per position per chip. Sharded over the mesh columns from 262,144 positions.
- Prefill indexer keys, scores and masks sharded over the 8 mesh columns (`DSV41_PIDX_SHARD`).
- Dense prefill masks only for layers that use them.
- The kv-table FIFO shifted in place (`DSV41_PF_FIFO_CHUNK`).
- One eager step before each trace capture.
- Index-key hand-off and speculative scoring in pieces (`DSV41_KEY_EXPORT_PIECE`, `DSV41_SPEC_SCORE_PIECE`).
- Above 131,584 tokens speculation uses the matmul indexer backend.

The prefill kv table is still replicated over the mesh columns.

Tested on one persistent server per configuration:

| Test | 16 users, 262,144 | 4 users, 1,048,576 | 4 users, 524,288 |
|---|---|---|---|
| 20,020-token prompt, code 4817 | 23.5 s | 32.3 s | 27.1 s |
| 100,020-token needle (end / start), code 4817 | 115.7 / 117.1 s | 157.7 / 159.2 s | 132.4 / 134.0 s |
| Overload (4x users, two waves) | 64/64 correct | 16/16 correct | 16/16 correct |
| API checks | 23/23 | 23/23 | 23/23 |

A prompt that long means a long first token.
Estimate for a 1,000,000-token prompt at 4 users: about 12 minutes (not measured).

Other points on an earlier tree (plain decode, not re-run on this image):

| Users | Context | Ready | 100k needle prefill | |
|---|---|---|---|---|
| 8 | 524,288 | 22 min | 141 s | works |
| 8 | 1,048,576 | - | - | out of DRAM at warm-up |

## Prefix caching and chunked prefill

On by default in the 262,144-token, 524,288-token, 786,432-token and 1,048,576-token specs (`enable-prefix-caching`, `DSV41_APC=1`, paged prefill `DSV41_PF_PAGED=read`); `--no-enable-prefix-caching` (or `DSV41_APC=0`) turns it off.
The token budget per step is 16,384 (`long-prefill-token-threshold` 4,096) at 4 users and 1,048,576 tokens and 67,584 at 16 users; the `_apc` copies at 16 users and 524,288 tokens use 4,096.
The 131,072-token specs keep prefix caching off.
The measured numbers below are from earlier images (APC was opt-in then); see `RELEASE_NOTES.md` and `RELEASE_DETAILS.md` for this image.
With the bounded prefill replay (`DSV41_PF_REPLAY`, default 2,560) a prefix-cache hit on an 8k prompt is slower than without it (3.84 s against 2.91 s), see Known issues in `RELEASE_NOTES.md`.

What it does:
- Requests that share a prefix (a system prompt, few-shot examples, a long document) skip the shared part.
- The model keeps snapshots of the prefill state in host memory (6 GiB, a snapshot at each power of two of the prompt length).
- A long prompt is split into 4,096-token steps, so decoding of the other users continues between steps.

Measured with speculation on (one server per row):

| Configuration | Burst TTFT p50 | Worst decode stall during a 32k prompt | GSM8K |
|---|---|---|---|
| 16 users, 262,144 tokens, default spec | 5.5 s | 32.7 s | 94.09 % (workflow) |
| 16 users, 262,144 tokens, default spec, this image | 2.2 s | 9.2 s | 95.60 % (workflow) |
| 4 users, 1,048,576 tokens, no APC, budget 4,096 | 15.5 s | 12.6 s | 96.5 / 97.7 % (own client, 256 questions) |
| 4 users, 1,048,576 tokens, default spec | 12.3 s | 12.7 s | 97.7 / 96.9 % (own client, 256 questions) |

The burst and stall numbers of the default spec and of the 4-user rows come from tests of the same code on a branch (images off); the 16-user row and the GSM8K scores marked workflow are from this image.

Long shared prefix (cold, then a hit): 100,000 tokens 107-124 s then 39-46 s; 300,000 tokens at 4 users 504 s then 69 s.
Limits:
- Prompts with images are not cached.
- A long prompt takes 1-3 s longer to its first token with the small budget.
- Setting `DSV41_PIDX_SHARD=0` at 262,144 tokens breaks cache hits (answers wrong from 16k); leave it unset.

## Performance

### Benchmark tables of this image: ISL x OSL up to 1,048,576 tokens, speculation on and plain decode

**These tables were measured on the v2.1 image `7ca53f64776-5b6f9268-15bcbd6` (1017 checkpoint, before the replay, drafter, attention-front, lean-CCL and fused-indexer work of this release), not on `afb78699c8b-5b6f9268-dd4fbc6`.** The workflow, the catalog and the checkpoint are the same; decode and prompt processing of this image are faster (see `RELEASE_NOTES.md`).

Setup: B=4 / 1,048,576-token catalog default (`deepseek-v41-flash-1m`, host .32 for spec-on, host .47 for plain decode), `vllm bench serve` through the `benchmarks` workflow of tt-inference-server (branch `ssinghal/dsv4p1-bench-1m`), 58 sweep points, 0 failed requests. Plain decode = the same catalog entry without `speculative_config` (DRAM free 596.4 MiB per bank against 415.1 with speculation, boot 1,177 s against 1,913 s). The 1M rows use ISL 1,048,320 / 1,047,296 / 1,044,224 for OSL 128 / 1,024 / 4,096 because ISL + OSL + 256 must fit in 1,048,576. Concurrency 1 and 4.

Each cell: TTFT / TPOT / tok/s per user / total tok/s, for plain decode | speculation on | speedup of tok/s per user. Source: `dsv4-logs/bench_1m_nospec/table_out.md` (generator `table.py`, raw results in `bench_1m_nospec/cache_root/` and `bench_1m/cache_root`, `cache_root_b`, `cache_root_c`; paths in the two `PATHS.md`).


### OSL 128  (cell: TTFT / TPOT / tok/s/user / total tok/s ; plain | spec-on | spec speedup in tok/s/user)
| ISL | conc | plain | spec-on | speedup |
|---|---|---|---|---|
| 128 | 1 | 0.4s / 34.4ms / 29.1 / 54 | 0.4s / 23.2ms / 43.0 / 76 | x1.48 |
| 128 | 4 | 0.8s / 37.3ms / 26.8 / 185 | 0.5s / 35.2ms / 28.4 / 202 | x1.06 |
| 1024 | 1 | 1.1s / 37.6ms / 26.6 / 196 | 1.1s / 26.6ms / 37.6 / 256 | x1.41 |
| 1024 | 4 | 1.7s / 39.3ms / 25.4 / 686 | 1.4s / 43.0ms / 23.3 / 644 | x0.91 |
| 2048 | 1 | 1.8s / 37.8ms / 26.5 / 332 | 1.8s / 27.8ms / 36.0 / 407 | x1.36 |
| 2048 | 4 | 2.4s / 39.5ms / 25.3 / 1166 | 2.3s / 53.2ms / 18.8 / 962 | x0.74 |
| 4096 | 1 | 3.1s / 37.8ms / 26.4 / 533 | 3.2s / 27.1ms / 36.9 / 638 | x1.40 |
| 4096 | 4 | 3.9s / 40.6ms / 24.7 / 1875 | 3.5s / 83.5ms / 12.0 / 1176 | x0.49 |
| 8192 | 1 | 5.8s / 38.6ms / 25.9 / 775 | 5.9s / 26.4ms / 37.9 / 904 | x1.46 |
| 8192 | 4 | 6.7s / 41.4ms / 24.2 / 2783 | 6.2s / 100.0ms / 10.0 / 1751 | x0.41 |
| 16384 | 1 | 11.2s / 38.7ms / 25.8 / 1025 | 11.3s / 26.6ms / 37.6 / 1128 | x1.46 |
| 16384 | 4 | 12.0s / 41.5ms / 24.1 / 3827 | 13.2s / 145.7ms / 6.9 / 2060 | x0.28 |
| 32768 | 1 | 22.0s / 40.0ms / 25.0 / 1215 | 22.0s / 28.6ms / 35.0 / 1283 | x1.40 |
| 32768 | 4 | 24.8s / 49.2ms / 20.3 / 4233 | 25.7s / 163.7ms / 6.1 / 2798 | x0.30 |
| 65536 | 1 | 43.9s / 42.6ms / 23.5 / 1333 | 44.2s / 28.5ms / 35.1 / 1372 | x1.49 |
| 65536 | 4 | 46.3s / 52.2ms / 19.1 / 4963 | 46.7s / 405.9ms / 2.5 / 2668 | x0.13 |
| 131072 | 1 | 88.7s / 48.2ms / 20.7 / 1384 | 89.0s / 51.1ms / 19.6 / 1375 | x0.94 |
| 131072 | 4 | 97.1s / 74.7ms / 13.4 / 4922 | 97.0s / 175.6ms / 5.7 / 4395 | x0.43 |
| 262016 | 1 | 179.4s / 52.3ms / 19.1 / 1409 | 180.3s / 34.8ms / 28.7 / 1419 | x1.50 |
| 262016 | 4 | 184.6s / 75.4ms / 13.3 / 5401 | 183.5s / 193.8ms / 5.2 / 5030 | x0.39 |
| 524160 | 1 | 362.7s / 68.0ms / 14.7 / 1412 | 362.1s / 43.6ms / 23.0 / 1426 | x1.56 |
| 524160 | 4 | 364.2s / 91.0ms / 11.0 / 5581 | 364.2s / 71.7ms / 14.0 / 5608 | x1.27 |
| 1048320 | 1 | 741.2s / 93.8ms / 10.7 / 1392 | 732.6s / 75.6ms / 13.2 / 1413 | x1.24 |
| 1048320 | 4 | 732.7s / 156.4ms / 6.4 / 5572 | 733.7s / 134.2ms / 7.4 / 5584 | x1.17 |

### OSL 1024  (cell: TTFT / TPOT / tok/s/user / total tok/s ; plain | spec-on | spec speedup in tok/s/user)
| ISL | conc | plain | spec-on | speedup |
|---|---|---|---|---|
| 128 | 1 | 0.4s / 36.0ms / 27.8 / 31 | 0.4s / 17.5ms / 57.2 / 63 | x2.06 |
| 128 | 4 | 0.8s / 38.2ms / 26.2 / 115 | 0.5s / 18.9ms / 52.8 / 216 | x2.02 |
| 1024 | 1 | 1.1s / 37.3ms / 26.8 / 52 | 1.1s / 23.1ms / 43.2 / 83 | x1.61 |
| 1024 | 4 | 1.6s / 39.6ms / 25.3 / 195 | 1.4s / 25.1ms / 39.8 / 244 | x1.58 |
| 8192 | 1 | 5.8s / 38.3ms / 26.1 / 205 | 5.9s / 18.5ms / 54.1 / 372 | x2.07 |
| 8192 | 4 | 6.8s / 40.5ms / 24.7 / 764 | 6.3s / 28.5ms / 35.1 / 911 | x1.42 |
| 10000 | 1 | 6.8s / 38.2ms / 26.2 / 240 | 6.8s / 18.4ms / 54.3 / 430 | x2.08 |
| 10000 | 4 | 7.5s / 40.5ms / 24.7 / 902 | 7.2s / 30.3ms / 33.0 / 1022 | x1.34 |
| 32768 | 1 | 22.1s / 40.0ms / 25.0 / 536 | 22.1s / 35.4ms / 28.2 / 580 | x1.13 |
| 32768 | 4 | 24.2s / 43.1ms / 23.2 / 1979 | 23.3s / 57.6ms / 17.3 / 1570 | x0.75 |
| 131072 | 1 | 89.0s / 48.0ms / 20.8 / 957 | 89.0s / 26.4ms / 37.9 / 1139 | x1.82 |
| 131072 | 4 | 96.3s / 53.0ms / 18.9 / 3509 | 94.5s / 133.8ms / 7.5 / 2245 | x0.40 |
| 262144 | 1 | 179.9s / 58.9ms / 17.0 / 1096 | 179.7s / 26.8ms / 37.3 / 1271 | x2.20 |
| 262144 | 4 | 185.8s / 64.0ms / 15.6 / 4188 | 182.2s / 288.8ms / 3.5 / 2182 | x0.22 |
| 524288 | 1 | 363.9s / 75.1ms / 13.3 / 1192 | 364.7s / 32.5ms / 30.7 / 1320 | x2.31 |
| 524288 | 4 | 367.2s / 80.9ms / 12.4 / 4668 | 379.5s / 227.4ms / 4.4 / 3106 | x0.36 |
| 1047296 | 1 | 739.9s / 90.5ms / 11.0 / 1259 | 741.1s / 44.1ms / 22.7 / 1333 | x2.05 |
| 1047296 | 4 | 729.5s / 97.0ms / 10.3 / 5059 | 839.8s / 590.9ms / 1.7 / 2895 | x0.16 |

### OSL 4096  (cell: TTFT / TPOT / tok/s/user / total tok/s ; plain | spec-on | spec speedup in tok/s/user)
| ISL | conc | plain | spec-on | speedup |
|---|---|---|---|---|
| 128 | 1 | 0.4s / 37.1ms / 27.0 / 28 | 0.4s / 12.5ms / 80.1 / 82 | x2.97 |
| 128 | 4 | 0.7s / 39.1ms / 25.6 / 105 | 0.6s / 14.5ms / 68.8 / 270 | x2.69 |
| 1024 | 1 | 1.1s / 37.3ms / 26.8 / 33 | 1.1s / 11.8ms / 84.8 / 104 | x3.17 |
| 1024 | 4 | 1.6s / 39.4ms / 25.4 / 126 | 1.4s / 16.3ms / 61.2 / 261 | x2.41 |
| 8192 | 1 | 5.9s / 38.3ms / 26.1 / 75 | 5.9s / 15.5ms / 64.7 / 178 | x2.48 |
| 8192 | 4 | 6.3s / 40.5ms / 24.7 / 286 | 6.1s / 20.7ms / 48.3 / 471 | x1.95 |
| 32768 | 1 | 22.0s / 39.8ms / 25.1 / 199 | 22.0s / 14.0ms / 71.6 / 465 | x2.85 |
| 32768 | 4 | 24.2s / 42.3ms / 23.7 / 747 | 23.2s / 23.7ms / 42.1 / 1174 | x1.78 |
| 131072 | 1 | 88.5s / 48.1ms / 20.8 / 474 | 89.5s / 76.6ms / 13.0 / 335 | x0.63 |
| 131072 | 4 | 96.2s / 50.5ms / 19.8 / 1784 | 92.3s / 57.9ms / 17.3 / 1557 | x0.87 |
| 262144 | 1 | 179.9s / 58.8ms / 17.0 / 633 | 180.7s / 20.5ms / 48.7 / 1006 | x2.87 |
| 262144 | 4 | 185.9s / 61.4ms / 16.3 / 2435 | 182.3s / 90.3ms / 11.1 / 1869 | x0.68 |
| 524288 | 1 | 363.4s / 75.0ms / 13.3 / 788 | 366.5s / 34.2ms / 29.3 / 1043 | x2.19 |
| 524288 | 4 | 367.0s / 77.6ms / 12.9 / 3085 | 370.7s / 141.9ms / 7.0 / 2182 | x0.55 |
| 1044224 | 1 | 737.8s / 90.1ms / 11.1 / 947 | 738.7s / 34.0ms / 29.4 / 1194 | x2.65 |
| 1044224 | 4 | 726.8s / 92.8ms / 10.8 / 3788 | 730.1s / 290.8ms / 3.4 / 2034 | x0.32 |

The tables below were measured on earlier images; the heading of each table names its image. They are kept as the long-form record and are not the numbers of this image. The prefix-caching and chunked-prefill numbers are in their own section.

Method for all tables:
- GSM8K 8-shot prompts, about 700 tokens, 256 output tokens.
- `vllm bench serve` from tt-inference-server's `benchmarks` workflow, run from the same image.
- A constant number of requests in flight.
- Greedy is temperature 0. Sampled is temperature 0.6, top-k 20, top-p 0.95.
- TTFT and TPOT are report means.

### 16 users, 262,144 tokens (default spec, image `0.24.0-c9cfb00-ffd1a57`)

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.08 s | 29.0 ms | 34.5 | 1.11 s | 30.9 ms | 32.3 |
| 2 | 1.28 s | 33.8 ms | 29.6 | 1.31 s | 32.4 ms | 30.9 |
| 4 | 1.49 s | 36.6 ms | 27.3 | 1.41 s | 40.5 ms | 24.7 |
| 8 | 1.85 s | 51.5 ms | 19.4 | 1.89 s | 52.5 ms | 19.0 |
| 16 | 2.52 s | 74.1 ms | 13.5 | 2.59 s | 74.9 ms | 13.4 |

Total output throughput (greedy): 30.2 / 51.1 / 89.1 / 131.6 / 183.6 tok/s at 1 / 2 / 4 / 8 / 16 requests.

### 16 users, 131,072 tokens (not a default any more, this release)

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.03 s | 28.5 ms | 35.1 | 1.06 s | 28.0 ms | 35.8 |
| 2 | 1.33 s | 32.2 ms | 31.1 | 1.25 s | 31.8 ms | 31.4 |
| 4 | 1.38 s | 38.0 ms | 26.3 | 1.39 s | 37.5 ms | 26.6 |
| 8 | 1.74 s | 49.1 ms | 20.3 | 1.80 s | 49.6 ms | 20.2 |
| 16 | 2.40 s | 70.4 ms | 14.2 | 2.70 s | 68.0 ms | 14.7 |

Total greedy throughput: 30.8 / 53.2 / 85.5 / 125.4 / 196.6 tok/s at 1 / 2 / 4 / 8 / 16 requests.

### 16 users, 262,144 tokens, images on (this release)

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.05 s | 26.4 ms | 37.8 | 1.09 s | 29.3 ms | 34.1 |
| 2 | 1.26 s | 34.8 ms | 28.8 | 1.30 s | 35.2 ms | 28.4 |
| 4 | 1.44 s | 36.1 ms | 27.7 | 1.40 s | 39.0 ms | 25.6 |
| 8 | 1.74 s | 53.4 ms | 18.7 | 1.86 s | 52.2 ms | 19.2 |
| 16 | 2.56 s | 71.0 ms | 14.1 | 2.63 s | 72.5 ms | 13.8 |

Total greedy throughput: 32.8 / 50.0 / 94.4 / 120.7 / 189.4 tok/s.

### 4 users, 1,048,576 tokens, images on (default spec, image `0.24.0-c9cfb00-ffd1a57`)

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.51 s | 29.1 ms | 34.4 | 1.54 s | 29.5 ms | 33.9 |
| 2 | 1.77 s | 34.3 ms | 29.1 | 1.80 s | 35.3 ms | 28.3 |
| 4 | 1.88 s | 44.2 ms | 22.6 | 2.00 s | 46.3 ms | 21.6 |

Total output throughput (greedy): 28.7 / 48.2 / 76.2 tok/s at 1 / 2 / 4 requests.

### 4 users, 1,048,576 tokens

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.49 s | 29.8 ms | 33.5 | 1.51 s | 30.9 ms | 32.4 |
| 2 | 1.74 s | 35.2 ms | 28.4 | 1.78 s | 39.6 ms | 25.3 |
| 4 | 1.86 s | 45.5 ms | 22.0 | 1.99 s | 44.8 ms | 22.3 |

Total greedy throughput: 28.1 / 47.3 / 74.5 tok/s.

### 4 users, 786,432 tokens, images on (default spec, image `0.24.0-c9cfb00-ffd1a57`)

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.37 s | 29.0 ms | 34.5 | 1.39 s | 28.8 ms | 34.8 |
| 2 | 1.58 s | 33.5 ms | 29.8 | 1.63 s | 35.1 ms | 28.5 |
| 4 | 1.70 s | 42.9 ms | 23.3 | 1.75 s | 43.4 ms | 23.0 |

Total output throughput (greedy): 29.2 / 50.0 / 79.3 tok/s at 1 / 2 / 4 requests.

### 4 users, 524,288 tokens, images on

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.26 s | 31.8 ms | 31.4 | 1.27 s | 31.3 ms | 31.9 |
| 2 | 1.61 s | 33.9 ms | 29.5 | 1.79 s | 33.3 ms | 30.0 |
| 4 | 1.79 s | 40.4 ms | 24.7 | 1.67 s | 43.5 ms | 23.0 |

Total greedy throughput: 27.3 / 48.2 / 81.2 tok/s.

### Previous image (`0.24.0-3dce07e-233b250`, 16 users, 131,072 tokens)

| Concurrent requests | Greedy TTFT | Greedy TPOT | Greedy tok/s/user | Sampled TTFT | Sampled TPOT | Sampled tok/s/user |
|---|---|---|---|---|---|---|
| 1 | 1.24 s | 27.7 ms | 36.1 | 1.20 s | 29.4 ms | 34.0 |
| 2 | 1.51 s | 33.7 ms | 29.7 | 1.41 s | 35.4 ms | 28.2 |
| 4 | 1.54 s | 43.7 ms | 22.9 | 1.60 s | 42.5 ms | 23.5 |
| 8 | 1.96 s | 52.5 ms | 19.0 | 1.99 s | 56.1 ms | 17.8 |
| 16 | 2.75 s | 80.1 ms | 12.5 | 2.96 s | 78.4 ms | 12.8 |

Total greedy throughput: 30.9 / 50.0 / 72.4 / 121.4 / 168.3 tok/s.

### How the numbers compare

- This release on the default spec is the same or faster than the previous image at every concurrency.
- TTFT is lower throughout (1.03-2.40 s against 1.24-2.75 s).
- TPOT is 0.8 ms higher at one request and up to 9.7 ms lower at 16.
- Total throughput at 16 requests is 196.6 against 168.3 tok/s.
- The previous image was measured on other hosts, so part of the gap may be the hosts.
- The 262,144-token server matches the default within the run-to-run noise (3-4 ms).
- The two fixes of this release do not change speculative decode speed.
- The 1M and 524,288-token servers are slower per user (TPOT +1 to +8 ms, TTFT +0.2-0.5 s).
- Every decode step and prefill chunk scores the index entries of the whole context.

### Reading TPOT

- **TPOT is a serving metric, not decode speed.** Finished requests are replaced by new ones. Each new prefill (1.2-1.5 s) pauses everyone's decode.
- **Decode with no arrivals:** a speculative round takes about 107 ms and gives about 3.7 tokens. That is about 30 ms per token per user.
- **Prompts matter.** Short zero-shot questions accept more drafts (4.3 tokens per round). The 8-shot prompts give 3.7-3.9.

## Accuracy: GSM8K 8-shot

All 1319 test questions, flexible-extract in percent, standard error about 0.6.

| Server | Concurrency | Flexible-extract | Strict-match |
|---|---|---|---|
| 16 users, 131,072 tokens | 16 | **95.22** | 73.9 |
| 16 users, 131,072 tokens, images on | 16 | **95.38** | 72.5 |
| 4 users, 1,048,576 tokens | 4 | **94.92** | 65.1 |
| 16 users, 262,144 tokens, images on | 16 | **93.56** and **93.48** (two runs) | 70.1 and 70.5 |
| Previous image, 16 users, 131,072 tokens | 16 | 95.45 | |
| **This image**, 16 users, 262,144 tokens, images on | 16 | **94.09** | 68.4 |
| **This image**, same, default spec | 16 | **95.60** | 66.2 |

All pass the workflow gate (within 5 % of the model card's 93.0).
Sampling at temperature 1.0 moves a score by about half a percent between runs.

**16 users at 262,144 tokens score about 2 points lower.**
That is about 3 standard errors, and it reproduced on a second run.
Checks so far:
- Image input on or off makes no difference (95.38 / 95.22 at 131,072; 93.5 / 93.18 at 262,144).
- Switching off the sharded RoPE tables makes no difference (93.18).
- The new code is fine at 131,072 tokens (95.22).
- 163,840 tokens scored 94.54.

The cause is not identified yet.
The 262,144-token server passes every functional test.
Where the last point of accuracy matters, use the 131,072-token spec or the 4-user 1M spec.

### Run the evaluation

`assets/run_gsm8k_eval.sh` starts a second container of the image.
It runs tt-inference-server's `evals` workflow against `http://127.0.0.1:8000`.

| | |
|---|---|
| Task | `dsv41_gsm8k_cot`: lm-eval `gsm8k_cot` (8 exemplars). |
| Sampling | temperature 1.0, top-p 0.95, up to 2048 tokens, stop at `Q:`, streaming. |
| Concurrency | `DSV41_EVAL_CONCURRENCY`: 16 for 16 users, 4 for 4 users. |
| Score | `exact_match,flexible-extract`. `strict-match` is lower by design: answers end "The answer is $64." |

Start the server (steps 1-5), then:

```bash
cd models/demos/blackhole/deepseek_v41_flash        # in the tt-metal checkout
IMAGE=$IMAGE CKPT=$CKPT SPEC=$PWD/assets/runtime_spec_b16_k5_262144_vision.json STATE=$STATE ./assets/run_gsm8k_eval.sh
```

- The script also takes `URL` (default `http://127.0.0.1:8000`), `CONCURRENCY` (default 16), `HF_CACHE` and `DOCKER`.
- It mounts your checkpoint at the site path that `eval_config.py` expects.
- The client downloads the dataset and, on first run, installs its Python environment (about 15 min). Both go into `$STATE/eval/`.
- The report is `$STATE/eval/cache_root/workflow_logs/reports_output/evals/report_*.md`.
- Time: about 25 min at 16 users, about 80 min at 4 users.
- Quick check: `./assets/run_gsm8k_eval.sh --limit-samples-mode smoke-test` runs 8 questions.

## Serving tests

`tools/serving_tests/` has the scripts used on this image.
Run them against a live server.
See the README there.

| Script | Checks |
|---|---|
| `sanity.py`, `long.py`, `needle_probe.py` | Short requests, a 20k-token prompt, 100k-token needles. |
| `serving_api_test.py <max_model_len>` | 23 OpenAI-API checks. |
| `overload_test.py <slots>` | 4x users at once, two waves. |
| `client_img.py`, `client_many.py`, `client_stress.py` | Image checks, 16 images, the 17-image error, image stress. |

## Build the image from the repos

Use this to run commits that are not in a published image. The commits of the image `afb78699c8b-5b6f9268-dd4fbc6` (below) exist only in the local repositories of the build host at the time of writing (branches `ssinghal/dsv4p1-rebase-1017-v3`, `ssinghal/dsv4p1-rebase-1017`, `ssinghal/dsv4p1-replay-v3`; nothing is pushed): TODO replace the checkout lines by pushed branch names once they are pushed.

```bash
mkdir -p ~/dsv41 && cd ~/dsv41
git clone https://github.com/tenstorrent/tt-metal.git
git clone https://github.com/tenstorrent/tt-inference-server.git
git clone https://github.com/tenstorrent/vllm-tt-plugin.git
git -C tt-metal checkout afb78699c8b && git -C tt-metal submodule update --init --recursive
git -C tt-inference-server checkout 5b6f9268
git -C vllm-tt-plugin checkout dd4fbc6

cd ~/dsv41/tt-inference-server
TT_METAL_DIR=~/dsv41/tt-metal TTIS_DIR=~/dsv41/tt-inference-server \
PLUGIN_DIR=~/dsv41/vllm-tt-plugin CTX_ROOT=~/dsv41/docker-ctx \
  scripts/build_local_dsv41_image.sh
```

The script prints `BUILT dsv41-flash-local:<tt-metal-11>-<tt-inference-server-8>-<plugin-7>`.
It needs network access and Docker with buildx.
It does not need the chips.

- Only committed code goes into the image. Each repo is exported at `HEAD`.
- Variables: `--skip-base` and `BASE_TAG` reuse a toolchain base. `METAL_SHA`, `TTIS_SHA`, `PLUGIN_SHA` pin commits. `REUSE_CTX=1` resumes. `DOCKER_HOST` and `BUILD_NETWORK=host` select a daemon and its network. `BUILD_JOBS` and `IMAGE_REPO` tune the build.
- The C++ layers depend on the tt-metal tree without `models/`. A change to `models/`, the plugin or tt-inference-server rebuilds only the cheap layers.
- The image is 21.8 GB and the toolchain base 4.7 GB. Put the Docker data root on a large disk.
- Run `sync` before a board reset if the Docker data root is on a local disk that `tt-smi -glx_reset` detaches. Restart the Docker daemon after the reset.

Check the image, then run it with step 4 (`export IMAGE=dsv41-flash-local:<tag>`):

```bash
docker run --rm --entrypoint cat $IMAGE /home/container_app_user/BUILD_INFO
docker run --rm --entrypoint wc $IMAGE -c /home/container_app_user/app/src/run_vllm_api_server.py   # not 0
```

## Notes

- **Concurrency.** Users decode at once. A new request's prefill pauses the others for about 1.2-1.5 s.
- **Context.** Decode speed follows the actual position, not the configured context. TTFT grows with prompt length.
- **No thinking mode.** The chat template is the non-thinking format.
- **One server per galaxy.** Stop a running server before starting another.
- **Build-slot lock.** The specs set `DSV41_BUILD_SLOTS=0`. Without it, a host without `/mnt/tt-data/ssinghal/buildslots` fails at start with `PermissionError`.
- **Workflows from the image.** The image has the client side of tt-inference-server in `/home/container_app_user/tt-inference-server`. Run it with `/home/container_app_user/tt-metal/python_env/bin/python run.py ...`.
- **Hugging Face cache must be a real directory.** A symlinked `~/.cache/huggingface/hub` dangles inside the container. Mount `$STATE/hf` as `/hf`, as in step 4.
