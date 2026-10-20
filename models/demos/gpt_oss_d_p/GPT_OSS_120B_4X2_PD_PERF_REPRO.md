# GPT-OSS 120B 4x2 P/D performance reproduction

This reproduces the two-Galaxy GPT-OSS 120B prefill/decode setup used to
measure prefill throughput and overlapped KV migration over RDMA.

## Expected result

After one warm-up request, a 65,536-token prompt with 32K prefill chunks
should produce approximately:

- TTFT: 3.14 s
- prefill service throughput: 21.8K tokens/s
- decode: about 330 tokens/s/user
- KV transfers: 72 successful, zero failures
- summed transfer time: 1.8-1.9 s
- migration span: 2.96 s
- migration tail exposed after prefill: 5-9 ms

The sum of transfer durations is not added to TTFT because transfers overlap
prefill. They still contend with prefill resources: compute-only throughput is
about 25.1K tokens/s, versus about 21.8K tokens/s with migration enabled.

## Revisions

Use the following branches:

```text
tt-metal: mg/gptoss-4x2-pd-perf-repro
tt-blaze: mg/gptoss-4x2-pd-perf-repro
tt-gen:   divanovic/gptoss120b-pd-1k-native
```

Pin these exact revisions:

```text
tt-metal: use the head printed by `git rev-parse origin/mg/gptoss-4x2-pd-perf-repro`
tt-blaze: d6d0b09f22f60a161d66fe4db90b9abed81ef82d
tt-gen:   f5a46649a2d1e9a4ccae8b1e2504c63925b16473
```

The tested frontend uses `ai-dynamo==1.5.0`. The tt-gen adapter, Python
bindings, native KV manager, and DMK transport must be built from the
matching internal tt-gen source. These instructions assume the user has
access to tt-gen.

Clone and build tt-gen:

```bash
git clone git@github.com:tenstorrent/tt-d-gen.git "$ROOT/tt-gen"
git -C "$ROOT/tt-gen" checkout f5a46649a2d1e9a4ccae8b1e2504c63925b16473
git -C "$ROOT/tt-gen" submodule update --init --recursive

cd "$ROOT/tt-gen"
./adapters/dynamo/setup_router_env.sh --with-test-deps
VIRTUAL_ENV=$PWD/adapters/dynamo/.venv ./build_dgen.sh --dynamo
./kv_manager/install-build-dependencies.sh
./build_dgen.sh --kv-manager --mooncake --blaze \
  --build-dir "$PWD/build-kv-manager"
```

The expected artifacts are:

```text
tt-gen/adapters/dynamo/.venv/bin/python
tt-gen/build-dynamo/bindings/python
tt-gen/build-kv-manager/kv_manager
tt-gen/build-kv-manager/device_io/dmk.elf
```

## Hardware and model prerequisites

- Two Blackhole Galaxies connected through the cluster RDMA network.
- Exactly 32 visible Tenstorrent PCIe devices on each host.
- GPT-OSS 120B checkpoint and tokenizer.
- Shared filesystem visible from both hosts.
- Open MPI 5.0.7 ULFM.
- Built tt-metal, tt-blaze, tt-gen Python adapter, native KV manager, and
  DMK device-I/O ELF.

Verify both machines before loading:

```bash
lspci -d 1e52: | wc -l
# Must print 32 on both hosts.
```

Define paths and addresses:

```bash
export ROOT=/shared/gptoss-4x2-pd-repro
export TT_METAL_HOME=$ROOT/tt-metal
export TT_BLAZE_HOME=$ROOT/tt-blaze
export TT_DGEN_HOME=$ROOT/tt-gen
export TT_GEN_NATIVE_BUILD=$TT_DGEN_HOME/build-kv-manager/kv_manager
export RUN=$ROOT/run
export MODEL=/path/to/gpt-oss-120b

export PREFILL_HOST=<prefill-hostname>
export PREFILL_IP=<prefill-rdma-ip>
export DECODE_HOST=<decode-hostname>
export DECODE_IP=<decode-rdma-ip>
```

Use a new `RUN` directory for every decoder-owner launch. The retained
decoder intentionally refuses stale owner or stop markers.

The checked-in reproduction assets are under:

```bash
export REPRO=$TT_METAL_HOME/models/demos/gpt_oss_d_p/repro
```

Generate concrete configs from the templates:

```bash
mkdir -p "$RUN"
envsubst < "$REPRO/manager-prefill.env.json.in" > "$RUN/manager-prefill.env.json"
envsubst < "$REPRO/manager-decode.env.json.in"  > "$RUN/manager-decode.env.json"
envsubst < "$REPRO/worker-prefill.json.in"     > "$RUN/worker-prefill.json"
envsubst < "$REPRO/worker-decode.json.in"      > "$RUN/worker-decode.json"

python3 -m json.tool "$RUN/manager-prefill.env.json" >/dev/null
python3 -m json.tool "$RUN/manager-decode.env.json"  >/dev/null
python3 -m json.tool "$RUN/worker-prefill.json"      >/dev/null
python3 -m json.tool "$RUN/worker-decode.json"       >/dev/null
```

## Required configuration

Prefill:

```text
PREFILL_MODEL=gpt_oss_d_p
PREFILL_NUM_LAYERS=36
PREFILL_NUM_USERS=2
PREFILL_SP=4
PREFILL_TP=8
PREFILL_CHUNK_SIZE=32768
PREFILL_MAX_SEQ_LEN=131072
PREFILL_ASYNC_LAYER_COMPLETION=1
PREFILL_LAYER_ACK_D2H=0
PREFILL_ENABLE_MIGRATION=1
PREFILL_MOCK_MIGRATION=0
PREFILL_MIGRATION_EXPORT_TO_FILE=1
PREFILL_USE_TRACE=0
PREFILL_TOPOLOGY=linear
PREFILL_FABRIC_MODE=1d
```

KV manager on both hosts:

```text
KV_MANAGER_TRANSFER_ENGINE_PROTOCOL=rdma
KV_MANAGER_TRANSPORT_KIND=zmq
KV_MANAGER_DEVICE_IO=dmk
KV_MANAGER_STAGING_BUFFER_BYTES=16777216
KV_MANAGER_SLAB_COUNT=4
KV_MANAGER_DEVICE_IDS=0,1,...,31
```

Use endpoint ID 1 for prefill and endpoint ID 0 for decode. Configure each
worker's peer to the other host's RDMA IP. Use all 36 layers and set
`chunk_size=32768`, `max_seq_len=131072`, and `max_slots=2`.

## Launch order

### 1. Start etcd and the OpenAI frontend on the prefill host

```bash
export ETCD_BIN=$TT_DGEN_HOME/.dynamo_local/etcd/etcd

"$ETCD_BIN" \
  --name gptoss-pd \
  --data-dir "$RUN/etcd-data" \
  --listen-client-urls "http://$PREFILL_IP:2379" \
  --advertise-client-urls "http://$PREFILL_IP:2379" \
  --listen-peer-urls "http://$PREFILL_IP:2380" \
  --initial-advertise-peer-urls "http://$PREFILL_IP:2380" \
  --initial-cluster "gptoss-pd=http://$PREFILL_IP:2380"

ETCD_ENDPOINTS="http://$PREFILL_IP:2379" \
$TT_DGEN_HOME/adapters/dynamo/.venv/bin/python -m dynamo.frontend \
  --http-port 8000 \
  --discovery-backend etcd \
  --event-plane zmq \
  --router-mode kv \
  --router-kv-events
```

### 2. Start the retained decode owner on the decode host

```bash
export HF_MODEL=$MODEL
export GPTOSS_DECODE_REPO=$TT_BLAZE_HOME
export GPTOSS_KV_DEPTH=131072
export GPTOSS_SAMPLE=1
export GPTOSS_CHAT_SOCKET_PREFIX=gptoss_4x2_pd_repro
export TT_MPI_IFACE=<cluster-interface>
export TT_METAL_CACHE_BASE=$HOME/.cache/tt-metal-cache/gptoss-4x2-pd
export BLAZE_TENSORBIN_CACHE=$ROOT/cache/decode-tensorbin
mkdir -p "$BLAZE_TENSORBIN_CACHE" "$TT_METAL_CACHE_BASE"

bash "$TT_BLAZE_HOME/scripts/gptoss_reload_4x2/launch_chat_showcase.sh" \
  "$RUN/decode" 36
```

Wait for:

```text
$RUN/decode/external.ready.json
```

Loading a cold 36-layer decoder takes roughly 35-45 minutes.

### 3. Start the prefill owner on the prefill host

Activate the tt-metal Python environment, export the prefill variables listed
above, then run:

```bash
export PREFILL_MODEL=gpt_oss_d_p
export PREFILL_HF_MODEL=$MODEL
export HF_MODEL=$MODEL
export PREFILL_TTNN_CACHE=$ROOT/cache/prefill-weights
# Use 0 for the first cache-populating launch and 1 on later launches.
export GPT_OSS_WEIGHTS_FROM_CACHE=0
export PREFILL_NUM_LAYERS=36
export PREFILL_NUM_USERS=2
export PREFILL_SP=4
export PREFILL_TP=8
export PREFILL_CHUNK_SIZE=32768
export PREFILL_MAX_SEQ_LEN=131072
export PREFILL_ASYNC_LAYER_COMPLETION=1
export PREFILL_LAYER_ACK_D2H=0
export PREFILL_H2D_SERVICE_ID=gptoss_4x2_pd_repro
export PREFILL_LAYER_COMPLETION_RING=/gptoss_4x2_pd_repro_completion
export PREFILL_ENABLE_MIGRATION=1
export PREFILL_MOCK_MIGRATION=0
export PREFILL_MIGRATION_EXPORT_TO_FILE=1
export PREFILL_MIGRATION_TABLE_PATH=$RUN/prefill/kv_table.pb
export PREFILL_MIGRATION_DEVICE_MAP_PATH=$RUN/prefill/device_map.txt
export PREFILL_USE_TRACE=0
export PREFILL_TOPOLOGY=linear
export PREFILL_FABRIC_MODE=1d
export PREFILL_GATE_FALLBACK_MODE=DEVICE_FP32
export PREFILL_MANIFEST=$TT_METAL_HOME/models/demos/gpt_oss_d_p/tt/runners/manifests/gpt_oss_120b_chat_36l.json

cd "$TT_METAL_HOME"
python -u -m models.demos.common.prefill.runners.prefill_runner
```

Wait for:

```text
setup complete, entering request loop
```

The prefill owner must export:

```text
$RUN/prefill/kv_table.pb
$RUN/prefill/device_map.txt
```

### 4. Start both KV managers

Start the native `kv_manager` once on each host. Load the generated JSON into
the environment before exec:

```bash
# Prefill host:
set -a
eval "$(jq -r 'to_entries[] | "export \(.key)=\(.value|@sh)"' "$RUN/manager-prefill.env.json")"
set +a
exec "$TT_GEN_NATIVE_BUILD/kv_manager"

# Decode host:
set -a
eval "$(jq -r 'to_entries[] | "export \(.key)=\(.value|@sh)"' "$RUN/manager-decode.env.json")"
set +a
exec "$TT_GEN_NATIVE_BUILD/kv_manager"
```

Do not start either manager until both model-owner table and device-map files
exist.

### 5. Start tt-gen workers

Run one worker on each host. Use the tt-gen adapter environment and bindings:

```bash
export PYTHONPATH=$TT_DGEN_HOME/adapters/dynamo:$TT_DGEN_HOME/build-dynamo/bindings/python
export ETCD_ENDPOINTS=http://$PREFILL_IP:2379

$TT_DGEN_HOME/adapters/dynamo/.venv/bin/python -m tt_dynamo.main \
  --config "$RUN/worker-prefill.json" \
  --model-path "$MODEL" \
  --served-model-name gpt-oss-120b-4x2-chat \
  --component gptoss_prefill \
  --reasoning-parser gpt_oss \
  --tool-call-parser harmony

$TT_DGEN_HOME/adapters/dynamo/.venv/bin/python -m tt_dynamo.main \
  --config "$RUN/worker-decode.json" \
  --model-path "$MODEL" \
  --served-model-name gpt-oss-120b-4x2-chat \
  --component gptoss_decode \
  --reasoning-parser gpt_oss \
  --tool-call-parser harmony
```

Both worker configs must use the KVM IDs and endpoint IDs described above.
Set `min_disagg_tokens=64` so the benchmark routes through P/D.

## Measurement

Generate a unique prompt whose chat-templated input is exactly 65,536 tokens.
Request 32 output tokens with streaming and usage enabled:

```json
{
  "model": "gpt-oss-120b-4x2-chat",
  "messages": [{"role": "user", "content": "<65536-token unique prompt>"}],
  "max_tokens": 32,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Measure TTFT from immediately before the HTTP POST until the first non-empty
streaming delta. Run once to warm caches and report the second request.

```bash
export FRONTEND_URL=http://$PREFILL_IP:8000
export TOKENIZER_PATH=$MODEL
export SERVED_MODEL_NAME=gpt-oss-120b-4x2-chat

python "$REPRO/async_bench.py" 65536  # warm-up
python "$REPRO/async_bench.py" 65536  # measured
python "$REPRO/analyze_migrations.py" "$RUN/manager-prefill.log" 72
```

Prefill service throughput is:

```text
templated input tokens / prefill worker service duration
```

Do not calculate it from TTFT; TTFT also includes routing and frontend
overhead. For the expected run, TTFT is about 3.14 s while the prefill service
rate is about 21.8K tokens/s.

## Validity checks

Reject a run if any log contains:

```text
successful=false
queue has no chunks
stays stale
condemn
reclaim
```

For a 65,536-token prompt with 32K chunks, require:

```text
72 successful transfers
2 chunks
36 layer transfers per chunk
0 failures
0 stale shards
```

`sum_ms` adds individual transfer durations. `span_ms` measures the
layer-by-layer migration window, including compute gaps. Neither is the
exposed migration penalty; the exposed penalty is only the migration tail
after prefill completes.

## Changing chunk size

Change both the prefill owner's `PREFILL_CHUNK_SIZE` and the prefill worker's
`chunk_size`, then restart the prefill owner, both managers, and both workers.
The decoder owner can remain loaded.

Measured 64K-prompt comparison:

```text
8K chunks:  about 20.3K tokens/s
16K chunks: about 21.6K tokens/s
32K chunks: about 21.8K tokens/s
64K chunks: about 20.0K tokens/s
```

The 32K setting is the best validated point. Grouping multiple layer
completion notifications is invalid with the current ordered migration
protocol and can deadlock.
