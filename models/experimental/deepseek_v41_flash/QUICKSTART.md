# DeepSeek-V4.1-Flash server: quick start

The bare minimum to get `demo/server.py` serving on a fresh machine. Everything else (flags, images, files,
sessions) is in the docstring at the top of `demo/server.py`.

**Needs:** a 32-chip Blackhole Galaxy, ~250 GB of free RAM (the Engram tables are staged in `/dev/shm`), and
~1.5 TB of disk (the checkpoint is ~480 GB, the converted weight cache ~700 GB).

## 1. Build tt-metal and the Python env

From the repo root:

```bash
sudo ./install_dependencies.sh
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
```

## 2. Build the Rust host runtime

The server runs its per-step host work on it. It isn't checked in, and without it the server quietly falls
back to slower Python threads and starts no web monitor.

```bash
# Only if `cargo` is missing:
curl https://sh.rustup.rs -sSf | sh -s -- -y && source ~/.cargo/env

models/experimental/deepseek_v41_flash/host_rs/build.sh
```

## 3. Get the checkpoint

The server looks in `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017` by default:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017
```

To keep it somewhere else, pass `--model-dir /path/to/checkpoint` to the server (and to the script in step 5).

## 4. (Optional) Pick a weight cache directory

The first run converts every weight into ttnn tiles and saves them here; later runs load them from the cache.
Without it the cache goes in `../cache` (relative to the repo root, where you run the commands from). If you
set it, set it in the shell you run both the fill script and the server from:

```bash
export DEEPSEEK_V4_CACHE_DIR=/path/to/cache
```

## 5. (Recommended) Pre-fill the expert cache

The routed experts are nearly all of the weights, and the server converts them one at a time on its first
start, which makes that start very slow. `tools/fill_expert_cache.py` writes the same cache files with many worker processes
instead. It runs on the host only (no device), skips experts already cached, and can be interrupted and rerun:

```bash
# The 40 decoder layers' routed experts (bfloat4_b, TP4).
python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --all-layers

# The DSpark drafter's experts, which it loads as bfloat8_b.
python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --dspark --dtype bfloat8_b
```

`--workers N` sets the process count (default: the CPU count, at most 32).

## 6. Run the server

```bash
python models/experimental/deepseek_v41_flash/demo/server.py --host 0.0.0.0 --port 8000
```

By default it runs DSpark speculative decode (text only; one reply decoding at a time, `--max-running N` lets
N take verify rounds in turn) with the KV-only prefill spread over every stage. It also starts the web monitor at `http://<host>:8090`, which keeps running after the server
exits. `--no-dspark` serves many users (40 by default) with images instead.

Startup is done when it logs `serving 'deepseek-v4.1-flash' on http://...`. Then:

```bash
curl -s --no-buffer -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```
