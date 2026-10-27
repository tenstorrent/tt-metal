# DeepSeek-V4.1-Flash on Blackhole Galaxy: getting started

OpenAI-compatible vLLM server on one Blackhole Galaxy (32 chips, 4x8 mesh). Release notes: `RELEASE_NOTES.md`.

## Docker image

```
dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6
```

- Built locally from the commits listed in `RELEASE_NOTES.md`; not pushed to ghcr.io (a ghcr tag is TODO until the image is pushed). No weights or caches inside.
- Image id `de6952f6f477`. Offline copy (5.38 GB tar): `/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_afb78699c8b.tar` (`docker load -i <tar>`). Size on disk: TODO (read `docker images` on a host that has it loaded).
- Verified on hosts .34, .45 and .33 (load the tar on any other host first).

## Configurations

| | Users | Context | Runtime spec (`assets/`) |
|---|---|---|---|
| **B=16** | 16 | 262,144 | `runtime_spec_b16_k5_262144_vision.json` |
| **B=4** | 4 | 1,048,576 | `runtime_spec_b4_k5_1048576_vision.json` |

Both: speculative decoding up to k = 5 (the server runs rounds of 3 or 5 drafts, whichever is faster), image input, prefix caching on, paged prefill on (bf16 KV pool, which APC needs), bounded prefill replay on (`DSV41_PF_REPLAY=2560`, `0` turns it off), token budget `max_num_batched_tokens` 16,384 with `long-prefill-token-threshold` 4,096 (B=4). To run without prefix caching pass `--no-enable-prefix-caching` (or `DSV41_APC=0`); `DSV41_PF_PAGED=0` selects the previous prefill. The `assets/` folder has more specs (no images, 131,072 tokens with APC and paged prefill off, `*_apc.json` variants: same settings as the default specs at B=4; budget 4,096 at B=16).

## Start

Once per host (the host needs the driver, 32 chips in `/dev/tenstorrent`, 1 GiB hugepages at `/dev/hugepages-1G`, Docker; disk: 476 GB checkpoint, 610 GB weight cache, 43 GB kernel cache, caches on a local disk):

The weight cache is keyed by directory only, so each checkpoint needs its own empty `$CACHE` (the server stamps `$CACHE/weights/ckpt_id.json` with the checkpoint id on first use and refuses to start if the stamp names a different checkpoint). The default checkpoint is `DeepSeek-V4.1-Flash-1017`; never point it at a cache built from the earlier `DeepSeek-V4.1-Flash` checkpoint.

```bash
export CKPT=/path/to/deepseek-v41-flash-1017   # checkpoint, read-only
export CACHE=/path/to/dsv41_cache_1017         # weights/ unified/ kernels/, reused across starts
export STATE=/path/to/dsv41_state         # scratch dirs the server writes
export IMAGE=dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6

mkdir -p $CKPT $CACHE/weights $CACHE/unified $CACHE/kernels $STATE/cache_root $STATE/logs $STATE/traces
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir $CKPT        # 476 GB; a token may be needed

# vLLM looks the model up in the Hugging Face cache, even offline: give it three small files
R=$STATE/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
mkdir -p $R/refs $R/snapshots/local && echo -n local > $R/refs/main
cp $CKPT/config.json $CKPT/tokenizer.json $CKPT/tokenizer_config.json $R/snapshots/local/
```

Get the runtime specs (JSON files). They are inside the image (no checkout needed):

```bash
C=$(docker create $IMAGE)
docker cp $C:/home/container_app_user/tt-metal/models/demos/blackhole/deepseek_v41_flash/assets ./assets
docker rm $C
```

or from the tt-metal repo, branch `ssinghal/dsv4p1-rebase-1017-v3` (the image is built from its commit `afb78699c8b`): `models/demos/blackhole/deepseek_v41_flash/assets/`. The specs use in-container paths (`/ckpt`, `/cache/...`), so they need no editing.

Pick the configuration:

```bash
export SPEC=$PWD/assets/runtime_spec_b16_k5_262144_vision.json     # B=16
# export SPEC=$PWD/assets/runtime_spec_b4_k5_1048576_vision.json   # B=4
```

Start the server (the same command for both):

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

From `docker run` to `/health` 200, with the weight and kernel caches populated:

| Configuration | Time |
|---|---|
| B=16 (262,144 tokens) | not measured on this image (about 25 minutes on the previous release) |
| B=4 (1,048,576 tokens) | 20 to 30 minutes: 1,217 / 1,526 / 1,773 / 1,787 s on four boots with the weight and kernel caches populated (the previous release: about 31 minutes) |

<!-- sources: B=4 boots: image_tests/afb78699c8b-5b6f9268-dd4fbc6/results.md lane A / B / C / G boot rows; B=16 25 min and B=4 31 min = previous GETTING_STARTED.md; image_tests/74727b7f7ee-5b6f9268-15bcbd6/results.md lane boots 1839/1370/1791/1298 s; image_tests/7ca53f64776-5b6f9268-15bcbd6/results.md lane boots 1810/1230/1798 s -->

The first start with empty caches takes much longer (60-90 minutes on earlier images, not re-measured); it writes the weight cache (about 610 GB) and the kernel cache (about 43 GB).
