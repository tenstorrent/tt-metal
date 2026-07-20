# Prefill KV-migration — test command reference

How to validate a model's prefill KV-cache migration on a **single Blackhole galaxy**, no decode. The flow
is model-agnostic: pick a model with `PREFILL_MODEL` and fill in its dimensions/paths below. Three gates,
each stricter and more expensive than the last.

| Gate | What it exercises | Needs |
|------|-------------------|-------|
| **0 — KV PCC** | prefill writes correct KV (precondition for everything) | tt-metal tree only |
| **1 — mock migration** | the KV-chunk **address table** is correct, read device-lessly | tt-metal tree only |
| **2 — loopback migration** | the real DRAM→transport→DRAM copy + migrated-KV accuracy | + tt-llm-engine binaries |

---

## 0. Shared setup (every terminal)

```bash
cd <tt-metal>
export TT_METAL_HOME="$PWD" PYTHONPATH="$PWD"

export PREFILL_MODEL=<model>              # registry key (see common/prefill/adapter.py)
export PREFILL_SP=8 PREFILL_TP=4          # mesh (SP rows × TP cols)
export PREFILL_NUM_LAYERS=<L>
export PREFILL_CHUNK_SIZE=<C>             # per-prefill_chunk tokens; block-cyclic period
export PREFILL_MAX_SEQ_LEN=<S>            # per-user cache tokens; multiple of CHUNK_SIZE
export PREFILL_H2D_SERVICE_ID=<svc>
export PREFILL_HF_MODEL=<config/weights dir>   # (+ TT_CACHE_PATH if the tilized weight cache is elsewhere)
export PREFILL_TRACE_DIR=<golden dir>          # metadata.json + kv_cache/layer_*.safetensors
export NCHUNKS=<ceil(prompt_len / CHUNK_SIZE)>
```

Constraints: `MAX_SEQ_LEN % CHUNK_SIZE == 0` and `CHUNK_SIZE % (SP*32) == 0` (each SP shard stays
32-token-block aligned). `RUN="python -m models.demos.common.prefill.runners.prefill_runner"`.

---

## Gate 0 — KV PCC (precondition; no migration)

Confirms prefill writes correct KV vs the golden trace before migration means anything. Request mode:
the runner serves; the producer pushes the golden-trace chunks and PCC-checks the KV read back from
device. Two terminals (both with the shared setup above):

```bash
# terminal A — runner (serves; creates + exports the H2D service):
$RUN

# terminal B — producer (pushes $NCHUNKS chunks from the golden trace, PCC-checks KV):
PREFILL_PRODUCER_CHUNKS=$NCHUNKS PREFILL_PRODUCER_CHECK_PCC=1 \
    python -m models.demos.common.prefill.runners.prefill_producer
```

**Expect:** `[kv-pcc] min PCC across <L> layers … (overall …)` ≥ `PREFILL_STANDALONE_CHUNKED_PCC`.

---

## Gate 1 — Mock migration + producer read-back (table addresses; no endpoint)

The runner serializes the KV-chunk table + device map (`PREFILL_MOCK_MIGRATION=1`, no validation on the
runner side); the producer reads each chunk **device-lessly** via `read_dram_umd` — the same UMD path the
migration worker uses — and PCCs vs golden. This isolates "is `build_kv_chunk_table` correct?" with no
migration endpoint, worker, or MPI. Requires the producer to implement a read-back for this model's cache
layout (see `prefill_producer.py`).

```bash
# Terminal 1 — runner (serialize table + device map only):
env PREFILL_MOCK_MIGRATION=1 \
    PREFILL_MIGRATION_TABLE_PATH=/tmp/kv_chunk_table.pb \
    PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/kv_device_map.json \
    $RUN

# Terminal 2 — producer (push $NCHUNKS chunks, read the table back, PCC):
env PREFILL_MIGRATION_TABLE_PATH=/tmp/kv_chunk_table.pb \
    PREFILL_MIGRATION_DEVICE_MAP_PATH=/tmp/kv_device_map.json \
    PREFILL_PRODUCER_CHUNKS=$NCHUNKS PREFILL_PRODUCER_CHECK_PCC=1 \
    python -m models.demos.common.prefill.runners.prefill_producer
```

**Expect:** producer `[producer] KV cache PCC PASSED`.

---

## Gate 2 — Loopback migration (endpoint + runner + scheduler driver)

Exercises the real DRAM→DCN→DRAM copy. Three processes, one host. Needs the tt-llm-engine binaries built
(`migration_endpoint`, `migration_worker`, `prefill_scheduler_driver` — see the tt-llm-engine build
scripts). Point the migration layer at the same tt-metal tree the runner uses.

```bash
export ENGINE=<tt-llm-engine>
export MIG="$ENGINE/disaggregation/migration/build_RelWithDebInfo"
```

**Launch order: A → B (wait for `WORKER_READY`) → C.** Between runs, clear stale state:
```bash
pkill -f migration_endpoint ; pkill prte
rm -f /dev/shm/tt_h2d_* /dev/shm/tt_d2h_* /dev/shm/tt_prefill_layer_acks_* /tmp/migration_done.sentinel*
```

```bash
# ---- Terminal A — migration endpoint (loopback target) ----
"$MIG/bin/migration_endpoint" --endpoint-id 0 \
  --cmd-queue /prefill_mig_cmd_1 --table-queue /prefill_mig_tbl_1 --response-queue /prefill_mig_rsp_1 \
  --worker-bin "$MIG/bin/migration_worker"
```

### 2a — burst (src & dst vs golden)

One prompt on slot 0, migrated to slot 1; both slots PCC'd vs golden.

```bash
# ---- Terminal B — runner (wait for WORKER_READY before starting C) ----
# PREFILL_STANDALONE_CHUNKED_NCHUNKS MUST equal the chunks the driver pushes (burst = NCHUNKS): the driver
# never sends a shutdown push, so the runner uses this count to know prefill is done, then polls the DONE
# sentinel and validates. Omit it and the runner blocks and prints no PCC.
env PREFILL_NUM_USERS=2 \
    PREFILL_ENABLE_MIGRATION=1 PREFILL_VALIDATE_MIGRATION=1 \
    PREFILL_STANDALONE_CHUNKED_NCHUNKS=$NCHUNKS \
    PREFILL_MIGRATION_CLIENT_DIR="$MIG/python" \
    PREFILL_MIGRATION_TABLE_PATH=/tmp/kv_chunk_table.pb \
    MIGRATION_DONE_FILE=/tmp/migration_done.sentinel \
    $RUN

# ---- Terminal C — scheduler driver ----
env PREFILL_OUTPUT_TIMEOUT_S=600 \
  "$ENGINE/build-full/prefill_scheduler_driver" \
    --service-id $PREFILL_H2D_SERVICE_ID --chunk-size $PREFILL_CHUNK_SIZE --sp-factor $PREFILL_SP \
    --num-completions-acks-per-layer $PREFILL_NUM_LAYERS \
    --max-users 2 --token-json "$PREFILL_TRACE_DIR/metadata.json" \
    --migrate --dest-endpoint-id 0 --num-migrations 1
```

**Expect (terminal B):** `[kv-migrate-validate] BEFORE src_slot=0 …` + `AFTER dst_slot=1 …`, then
`ALL 1 migrated pair(s) PASSED`.

### 2b — pairwise (dst == src, golden-free, length-agnostic)

`N` concurrent src→dst migrations; each dst asserted bit-equal to its src. Set `PREFILL_NUM_USERS=2N`,
`--max-users 2N`, `--num-migrations N`, and `PREFILL_STANDALONE_CHUNKED_NCHUNKS = N · NCHUNKS`. Example `N=2`
(slots 0,1 → 2,3):

```bash
# ---- Terminal B ----
env PREFILL_NUM_USERS=4 \
    PREFILL_ENABLE_MIGRATION=1 PREFILL_VALIDATE_MIGRATION=1 PREFILL_MIGRATE_PAIRWISE=1 \
    PREFILL_STANDALONE_CHUNKED_NCHUNKS=$((2 * NCHUNKS)) \
    PREFILL_MIGRATION_CLIENT_DIR="$MIG/python" \
    PREFILL_MIGRATION_TABLE_PATH=/tmp/kv_chunk_table.pb \
    MIGRATION_DONE_FILE=/tmp/migration_done.sentinel \
    $RUN

# ---- Terminal C ----
env PREFILL_OUTPUT_TIMEOUT_S=600 \
  "$ENGINE/build-full/prefill_scheduler_driver" \
    --service-id $PREFILL_H2D_SERVICE_ID --chunk-size $PREFILL_CHUNK_SIZE --sp-factor $PREFILL_SP \
    --num-completions-acks-per-layer $PREFILL_NUM_LAYERS \
    --max-users 4 --token-json "$PREFILL_TRACE_DIR/metadata.json" \
    --migrate --dest-endpoint-id 0 --num-migrations 2
```

**Expect (terminal B):** `[kv-migrate-validate] AFTER pairwise src=… dst=… min_pcc=…` per pair, then
`ALL 2 pair(s) dst==src PASSED` (≥ 0.99).

---

## Notes / gotchas

- **PCC is logged by the runner (terminal B), not the driver/endpoint.** The driver only reports the
  transport (`SUBMIT`, `prefill_complete`, migration pairs). Accuracy lives in the runner's post-loop
  validation (`[kv-migrate-validate]` lines).
- **`PREFILL_STANDALONE_CHUNKED_NCHUNKS` must match what the driver pushes** (burst = `NCHUNKS`, pairwise =
  `N · NCHUNKS`). Too low → the runner exits mid-prefill; too high → it blocks waiting for chunks that never
  come.
- **`PREFILL_MIGRATION_CLIENT_DIR`** must point at the dir holding `_migration_client*.so`
  (`$MIG/python`).
- **Config-id order is the src↔dst contract** for multi-config caches: loopback is self-consistent, but a
  real prefill→decode run needs the decode endpoint to publish configs in the same order.
- **pairwise vs burst:** pairwise (dst==src) is golden-free and length-agnostic — the cheapest fidelity
  check. burst anchors both endpoints to the golden but reads the cache twice (src then dst).
