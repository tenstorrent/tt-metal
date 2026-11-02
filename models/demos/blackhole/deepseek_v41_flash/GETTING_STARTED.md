# DeepSeek-V4.1-Flash on Blackhole Galaxy: getting started (B=8)

OpenAI-compatible vLLM server on one Blackhole Galaxy (32 chips, 4x8 mesh). Release notes: `RELEASE_NOTES.md`.

## Docker image

Target registry name (**not pushed yet**) and the local image it was tagged from:

```
ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-785f0df-2406dcb
dsv41-flash-local:rc4c-785f0df9b28-2406dcb1-a83b4c6      # same content as the ghcr name (image id ae6ac9b615f5); plugin a83b4c6 with the tool-calling fix
```

- 21.7 GB on disk. No weights or caches inside.
- No tar has been written yet for the a83b4c6 image (`docker save` it before distributing); the earlier tar `/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_rc4c_785f0df9b28.tar` is the image without the tool-calling fix.
- The image contains the model catalog (`/home/container_app_user/model_specs/model_spec.json`, baked at build time). **No runtime spec file is needed**: the server picks the catalog entry from `--model`, `--tt-device` and `--impl`. (The `assets/runtime_spec_*.json` files are not used by rc4c.)

## Configurations

| | Users | Context | Select |
|---|---|---|---|
| **B=8** (default entry) | 8 | 1,048,576 | `--model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy` (no `--impl`) |

B=8: speculative decoding, image input (up to 250 images per prompt), tool calling / reasoning parser / non-thinking default (`tokenizer-mode deepseek_v41`), structured outputs, prefix caching on a 512-token grid (device block-paged at B=8), paged prefill. The catalog also has vision entries (`deepseek-v41-flash-vlm-b8-1m`) that differ only in the model type used by the evaluation workflows. B=4 entries exist in the catalog but were not validated for rc4c. All settings of an entry are in the catalog (`workflows/model_specs/dev/llm.yaml` and `vlm.yaml` of tt-inference-server `2406dcb15`); read them with:

```bash
docker run --rm --entrypoint cat dsv41-flash-local:rc4c-785f0df9b28-2406dcb1-a83b4c6 /home/container_app_user/model_specs/model_spec.json > model_spec.json
```

## Start

Once per host (driver, 32 chips in `/dev/tenstorrent`, 1 GiB hugepages at `/dev/hugepages-1G`, Docker; the caches on a local disk; the rc4c validation started a server only when `MemAvailable` was at least 300 GB, a limit set by the validation script, not a measured requirement). Cache disk sizes were not measured for rc4c (earlier images: checkpoint 476 GB, weight cache about 610 GB, kernel cache about 43 GB).

The weight cache is keyed by directory only, so each checkpoint needs its own empty `$CACHE` (the server stamps `$CACHE/weights/ckpt_id.json` with the checkpoint id on first use and refuses to start if the stamp names a different one).

```bash
export CKPT=/path/to/deepseek-v41-flash-1017   # checkpoint, read-only
export CACHE=/path/to/dsv41_cache_1017         # weights/ unified/ kernels/, reused across starts
export STATE=/path/to/dsv41_state              # scratch dirs the server writes
export IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-785f0df-2406dcb   # once pushed; before that the local name dsv41-flash-local:rc4c-785f0df9b28-2406dcb1-a83b4c6

mkdir -p $CKPT $CACHE/weights $CACHE/unified $CACHE/kernels $STATE/cache_root $STATE/logs $STATE/traces
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir $CKPT        # a token may be needed

# vLLM looks the model up in the Hugging Face cache, even offline: give it three small files
R=$STATE/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
mkdir -p $R/refs $R/snapshots/local && echo -n local > $R/refs/main
cp $CKPT/config.json $CKPT/tokenizer.json $CKPT/tokenizer_config.json $R/snapshots/local/
```

Start the server (B=8, the default entry):

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
  -e MODEL_WEIGHTS_DIR=/ckpt \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) \
  $IMAGE \
  --model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy --no-auth
```

This is the command the validation used (`dsv4-logs/rc4c_val/hold_acc.sh`: no runtime spec, no tree or plugin mounts, no environment overrides). B=16 is not part of this release.

Note: `docker -e DSV41_...` does not override the settings baked into the catalog entry; to change a setting, mount a modified `model_spec.json` over `/home/container_app_user/model_specs/model_spec.json`.

Wait for ready, then test and stop:

```bash
until curl -sf -o /dev/null http://localhost:8000/health; do sleep 10; done; echo ready

curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 64, "temperature": 0.6, "top_p": 0.95}'

docker stop dsv41          # after a crash or kill, reset the board first: tt-smi -glx_reset
```

Progress: `docker logs -f dsv41` (`built layer N` for the 40 layers, then trace capture, then `/health` returns 200).

## Request features

- `cache_salt` in the request body isolates the prefix cache between tenants (a request with another salt never hits).
- `usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens came from the prefix cache.
- Structured outputs: `response_format` (`json_schema`, `json_object`), `tool_choice` required / named with `strict: true`, guided `regex` / `choice` / `grammar`. Streaming requests with `logprobs`, `min_tokens` or `bad_words` are refused with HTTP 400 before the first event.
- Images: `image_url` content parts, up to 250 per prompt.
- Thinking is off by default (`default-chat-template-kwargs` in the catalog); a request can override it (`chat_template_kwargs`; not exercised in the rc4c validation files I read).

## Start-up time

From `docker run` to `/health` 200, with the weight and kernel caches populated (rc4c validation, `dsv4-logs/rc4c_val/srv/*/info`):

| Configuration | Time |
|---|---|
| B=8 (1,048,576 tokens) | 1,274-1,635 s (21-27 min) |

The first start with empty caches takes much longer (not measured for rc4c); it writes the weight cache and the kernel cache.
