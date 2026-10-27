# DeepSeek-V4.1-Flash on Blackhole Galaxy (B=4): getting started

OpenAI-compatible vLLM server on one Blackhole Galaxy (32 chips, 4x8 mesh). Release notes: `RELEASE_NOTES.md`.

## Docker image

```
ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-a06adeb-91da919
```

- Image id `68748a55fba8`, 21.6 GB on disk. Built from the commits listed in `RELEASE_NOTES.md`. No weights or caches inside.
- `docker pull` it on each host.

## Configuration

| | Users | Context | Runtime spec (`assets/`) |
|---|---|---|---|
| **B=4** | 4 | 1,048,576 | `runtime_spec_b4_k5_1048576_vision.json` |

Speculative decoding up to k = 5 (the server runs rounds of 3 or 5 drafts, whichever is faster), image input (up to 250 images per prompt: `limit-mm-per-prompt` in the spec), prefix caching on, paged prefill on (bf16 KV pool, which APC needs), bounded prefill replay on (`DSV41_PF_REPLAY=2560`, `0` turns it off), token budget `max_num_batched_tokens` 16,384 with `long-prefill-token-threshold` 4,096. To run without prefix caching pass `--no-enable-prefix-caching` (or `DSV41_APC=0`); `DSV41_PF_PAGED=0` selects the previous prefill. The `assets/` folder has more B=4 specs (524,288 and 786,432 tokens, `*_apc.json` variants).

## Start

Once per host (the host needs the driver, 32 chips in `/dev/tenstorrent`, 1 GiB hugepages at `/dev/hugepages-1G`, Docker; disk: 476 GB checkpoint, 610 GB weight cache, 43 GB kernel cache, caches on a local disk):

The weight cache is keyed by directory only, so each checkpoint needs its own empty `$CACHE` (the server stamps `$CACHE/weights/ckpt_id.json` with the checkpoint id on first use and refuses to start if the stamp names a different checkpoint). The default checkpoint is `DeepSeek-V4.1-Flash-1017`; never point it at a cache built from the earlier `DeepSeek-V4.1-Flash` checkpoint.

```bash
export CKPT=/path/to/deepseek-v41-flash-1017   # checkpoint, read-only
export CACHE=/path/to/dsv41_cache_1017         # weights/ unified/ kernels/, reused across starts
export STATE=/path/to/dsv41_state         # scratch dirs the server writes
export IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-a06adeb-91da919

mkdir -p $CKPT $CACHE/weights $CACHE/unified $CACHE/kernels $STATE/cache_root $STATE/logs $STATE/traces
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir $CKPT        # 476 GB; a token may be needed

# vLLM looks the model up in the Hugging Face cache, even offline: give it three small files
R=$STATE/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
mkdir -p $R/refs $R/snapshots/local && echo -n local > $R/refs/main
cp $CKPT/config.json $CKPT/tokenizer.json $CKPT/tokenizer_config.json $R/snapshots/local/
```

Get the runtime spec (a JSON file). It is inside the image (no checkout needed):

```bash
C=$(docker create $IMAGE)
docker cp $C:/home/container_app_user/tt-metal/models/demos/blackhole/deepseek_v41_flash/assets ./assets
docker rm $C
```

or from the tt-metal repo, branch `ssinghal/dsv4p1-rebase-1017-v3d` (the image is built from its commit `a06adeba8f1`): `models/demos/blackhole/deepseek_v41_flash/assets/`. The specs use in-container paths (`/ckpt`, `/cache/...`), so they need no editing.

```bash
export SPEC=$PWD/assets/runtime_spec_b4_k5_1048576_vision.json
```

Start the server:

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

## Start-up time

From `docker run` to `/health` 200, with the weight and kernel caches populated: 20 to 30 minutes (1,217 to 1,787 s on four boots).

The first start with empty caches takes much longer (60-90 minutes on earlier images, not re-measured); it writes the weight cache (about 610 GB) and the kernel cache (about 43 GB).
