# Prefill KV-migration — configuration and gates

The prefill side on its own terms: the files that configure it, and three gates that can be run against it
directly with **no decode side present**. The tt-llm-engine launch harness generates and invokes this
underneath; this document is the layer beneath that. It is also the path to take when integrating a new
model, because the gates isolate failures that the harness necessarily reports as one.

Assumes the model is already integrated — adapter registered, golden trace staged, weight cache populated
(`ADDING_A_PREFILL_MODEL.md`, alongside this file).

| Gate | What it exercises | Needs |
|------|-------------------|-------|
| **0 — standalone PCC** | Prefill writes correct KV (precondition for everything) | tt-metal tree only |
| **1 — mock migration** | The KV-chunk address table is correct, read device-lessly | tt-metal tree only |
| **2 — loopback migration** | The real DRAM → transport → DRAM copy, and migrated-KV accuracy | + tt-llm-engine binaries |

Gate 2 covers the same ground as the harness's own prefill-loopback stage. The difference is only that the
harness drives it end to end instead of three terminals.

---

## The three config files

| File | Location | Owns |
|------|----------|------|
| model manifest (`.json`) | your model package | `PREFILL_MODEL` + model-wide knobs |
| rank binding (`.yaml`) | `runners/topology_configuration/` | mesh topology + runner env |
| producer manifest (`.yaml`) | `runners/producer_manifests/` | workload + migration scenario |

Rank bindings and producer manifests are shared and model-agnostic; the only model-specific file is the one
in your own package. Swapping models means swapping the model manifest, not editing the other two.

Precedence differs by process. **Runner** (under `tt-run`): `global_env` > model manifest > code default. A
shell-exported `PREFILL_*` is *not* an override, because `tt-run` forwards only `TT_`, `ARCH_`, `WH_`,
`TTNN_`, `DEEPSEEK_` and `MESH_` prefixed variables (`ENV_PASSTHROUGH_PREFIXES` in
`ttnn/ttnn/distributed/ttrun.py`). **Producer** (an ordinary process): CLI flag > exported env > manifest
`env:` block > manifest typed block.

That forwarding rule is the reason the harness generates `<run_dir>/prefill_topology.yaml` rather than
exporting variables: a `PREFILL_*` set in the shell never reaches the runner, so `pd_producer.prefill_env`
must be merged into the rank binding's `global_env` instead. It is also why Gates 0 and 1 below run the
runner as a **bare python process** — exported `PREFILL_*` works there because `tt-run` is not in the path.

---

## Shared setup

```bash
cd <tt-metal>
export TT_METAL_HOME="$PWD" PYTHONPATH="$PWD"
source python_env/bin/activate
export HOST=$(hostname)
export RUN=./models/demos/common/prefill/runners/run_pipeline_prefill.sh
export BINDING=models/demos/common/prefill/runners/topology_configuration/<binding>.yaml
export MANIFEST=models/demos/common/prefill/runners/producer_manifests/<manifest>.yaml
export MODEL_MANIFEST=models/demos/<your_model>/tt/runners/manifests/<your_model>.json
```

Two shape constraints apply: `MAX_SEQ_LEN % CHUNK_SIZE == 0` and `CHUNK_SIZE % (SP*32) == 0` (each SP shard
stays 32-token-block aligned).

`PREFILL_ENABLE_MIGRATION=1` is **single-rank only** — the runner rejects it for `num_ranks > 1` (pipelined
migration is not implemented). Gate 2 therefore uses a 1-rank binding.

---

## Model manifest

```json
// models/demos/my_model/tt/runners/manifests/my_model.json
{ "env": { "PREFILL_MODEL": "my_model", "PREFILL_GATE_FALLBACK_MODE": "DEVICE_FP32" } }
```

The `env:` map is applied verbatim by `setdefault`, so a rank binding's `global_env` still wins. A manifest
may also carry a `users[]` + `migration{}` block for the pairwise-validation path; a plain model-config
manifest omits it.

## Rank binding

Copy `topology_configuration/pipeline_prefill_request_1rank.yaml` and add what migration needs:

```yaml
rank_bindings:
  - {rank: 0, mesh_id: 0, mesh_host_rank: 0, env_overrides: {}}
mesh_graph_desc_path: tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_x_graph_descriptor.textproto
global_env:
  PREFILL_MANIFEST: "models/demos/my_model/tt/runners/manifests/my_model.json"
  PREFILL_NUM_LAYERS: "<L>"        # the runner defaults to 61 — pin the real depth
  PREFILL_MAX_SEQ_LEN: "<S>"
  PREFILL_CHUNK_SIZE: "<C>"        # defaults to 5*1024
  PREFILL_FABRIC_MODE: "2d"
  PREFILL_H2D_SERVICE_ID: "my_prefill"
  PREFILL_NUM_USERS: "4"           # must cover every dst slot: src 0,1 -> dst 2,3

  PREFILL_ENABLE_MIGRATION: "1"
  PREFILL_MIGRATION_WAIT_READY_MS: "120000"
  PREFILL_MIGRATION_TABLE_PATH: "/tmp/prefill_kv_chunk_table.pb"
  PREFILL_MIGRATION_DEVICE_MAP_PATH: "/tmp/prefill_kv_device_map.json"
  PREFILL_MIGRATION_CMD_QUEUE: "/mig_ep1_cmd"      # endpoint default is /mig_ep<id>_{cmd,table,resp}
  PREFILL_MIGRATION_TABLE_QUEUE: "/mig_ep1_table"
  PREFILL_MIGRATION_RESP_QUEUE: "/mig_ep1_resp"
  PREFILL_MIGRATION_CLIENT_DIR: "<tt-llm-engine>/disaggregation/migration/build_RelWithDebInfo/python"

  PREFILL_VALIDATE_MIGRATION: "1"
  PREFILL_MIGRATE_PAIRWISE: "1"                    # dst == src; omit for burst (vs golden)
  PREFILL_STANDALONE_CHUNKED_NCHUNKS: "22"         # = num_users x chunks the producer pushes
  MIGRATION_DONE_FILE: "/tmp/migration_done.sentinel"
  PREFILL_MIGRATE_WAIT_S: "1200"
```

---

## Two producer entry points

`prefill_producer` and `migration_driver` are separate modules, and which one you run matters:

| module | what it is |
|--------|------------|
| `prefill_producer` | The plain runner test. H2D push, ack drain, golden PCC. Knows nothing about migration. |
| `migration_driver` | The migration entry point. Drives the H2D half **with `prefill_producer`'s own helpers**, then owns the `MigrationLayerClient` attach, the cross-endpoint pairing, the src→dst mapping, the `migrate()` calls, and both sidecar files. |

The dependency runs one way: `prefill_producer` imports nothing from `migration_driver`, so a runner-only run
can never pull migration in. Migration has to share the producer's process — it needs that run's
resident-slot state and must migrate while the runner still holds the KV in device DRAM — which is why one
module covers both halves rather than chaining two.

So: **any run that migrates uses `migration_driver`**; only the no-migration gates (Gate 0 and Gate 1 below)
use `prefill_producer`. Invoking `migration_driver` *is* the opt-in, so `migration: {issue: true}` is
redundant there and an explicit `false` is warned about rather than honoured.

The harness follows the same rule. Its step is still **named** `prefill_producer` (that name appears in its
step chains and gate tables, and in `<run_dir>/prefill_producer.log`), but the module it launches is
`migration_driver` — see `producer_module` in `disaggregation/launch_harness/config.py`. Every `pd_producer`
mode that launches a producer at all (`pd`, `prefill_loopback`) migrates, so this is unconditional.

One env var moved with the split: the src-KV dump is now `--dump-src-kv`, whose default reads
**`PREFILL_MIGRATION_DUMP_SRC_KV`**, not the older `PREFILL_PRODUCER_DUMP_SRC_KV`. Under the old name the
flag silently resolves to `None`, no `src_slot<N>.pt` is written, and the decode side's
`--migration-validate-src-kv-pt` then fails to load — a prefill-side cause that surfaces as a decode-side
failure. `PREFILL_PRODUCER_CHECK_PCC` is unchanged.

---

## Producer manifest

Both modules read the same manifest; `migration_driver` additionally applies its typed `migration:` block.
Start from `producer_manifests/prefill_producer_manifest.example.yaml`, which lists every field with its env
var and default. A migration scenario needs little:

```yaml
transport:
  sp: 8
  tp: 4
  h2d_service_id: my_prefill      # [MATCH RUNNER]
  connect_timeout_s: 60

workload:
  num_users: 2                    # src slots 0,1
  chunks: "11"
  max_requests: 2                 # one resident request per slot
  interleave: round_robin
  check_pcc: false                # the RUNNER validates migrated KV on-device

migration:
  issue: true                     # attach the client and migrate after prefill
  dest_endpoint_id: 1             # == our own id => loopback
  dst_slot_offset: 2              # dst = src + 2  (0->2, 1->3)
  timeout_ms: 3600000
  done_file: /tmp/migration_done.sentinel                # [MATCH RUNNER]
  table_path: /tmp/prefill_kv_chunk_table.pb             # [MATCH RUNNER]
  cmd_queue: /mig_ep1_cmd                                # [MATCH ENDPOINT]
  table_queue: /mig_ep1_table
  resp_queue: /mig_ep1_resp
```

`client_dir` is deliberately left out of the shared manifests: it is per-checkout, and a placeholder path
there fails later as a confusing `No module named '_migration_client'`. Export it instead — it must point at
the directory holding `_migration_client*.so`:

```bash
export PREFILL_MIGRATION_CLIENT_DIR=<tt-llm-engine>/disaggregation/migration/build_RelWithDebInfo/python
```

The shared producer manifests carry **no** `model:` **block** — that is what keeps them model-agnostic.
Export the four model and transport variables so they match the runner, or add a `model:` block for a
self-contained file:

```bash
export PREFILL_MODEL=my_model PREFILL_NUM_LAYERS=<L> \
       PREFILL_MAX_SEQ_LEN=<S> PREFILL_CHUNK_SIZE=<C>
```

Left unset, the producer falls back to the adapter defaults (61 layers), which will not match a deeper
runner. The ack count is `num_layers × chunks`, so the drain hangs.

For an arbitrary mapping instead of a uniform offset, replace `dst_slot_offset` with `pairs` (CLI
equivalent: `--migrations "0:3,1:2"`). Source and destination sets must be disjoint in loopback, and
duplicate destinations are rejected.

```yaml
  pairs:
    - {src: 0, dst: 3}
    - {src: 1, dst: 2}
```

### Per-slot prompts

By default every slot replays one shared prompt (`PREFILL_TRACE_DIR`, or the adapter default) and the
schedule invents each request's depth. `PREFILL_PRODUCER_SLOT_TRACES="dirA,dirB,..."` (manifest:
`workload.slot_prompts`) instead assigns trace *i* to slot *i*, cycling by `slot % len` when there are fewer
entries than users. Because a trace dir carries **both** the tokens (`metadata.json`) and the golden KV
(`kv_cache/layer_*.safetensors`), per-slot traces give per-slot prompts *and* per-slot goldens: each slot's
read-back is PCC'd against its own golden.

Two consequences follow automatically. Depth stops being synthetic — a slot pushes exactly
`ceil(real_len / CHUNK_SIZE)` chunks with `actual_isl = real_len`, so the random chunk draw and
`mid_end_prob` are bypassed. And a prompt longer than the per-user cache is clamped to
`MAX_SEQ_LEN / CHUNK_SIZE` chunks with a warning, so distinct full-length traces still fit the KV budget.

This is the discriminating multi-prompt check: with one shared prompt every slot's KV is identical, so
crossed slot→prompt wiring or a slot-index bug in the address table still passes. Note that
`slot_lengths` is keyed by slot, so with `max_requests > num_users` a recycled slot replays the *same*
prompt; set `max_requests == num_users` for one resident request per slot.

Coverage: `tests/test_producer_slot_prompts.py` (device-free, host logic) and
`tests/test_producer_runner_e2e.py::test_producer_runner_multiprompt_pcc` (on-device, opt-in via
`PREFILL_CI_MULTIPROMPT_TRACES`).

---

## Gate 0 — standalone KV PCC

A precondition, with no migration involved. Single rank, so run the runner directly rather than under
`tt-run`, and export the model config (see the forwarding note above).

```bash
env PREFILL_MANIFEST=$MODEL_MANIFEST \
    PREFILL_NUM_LAYERS=<L> PREFILL_MAX_SEQ_LEN=<S> PREFILL_CHUNK_SIZE=<C> \
    PREFILL_STANDALONE=1 PREFILL_STANDALONE_PCC=1 \
    PREFILL_STANDALONE_NCHUNKS=$NCHUNKS PREFILL_STANDALONE_CHUNKED_NCHUNKS=$NCHUNKS \
    python -m models.demos.common.prefill.runners.prefill_runner
```

Expect `[kv-pcc]` per-layer lines, with the minimum at or above `PREFILL_STANDALONE_CHUNKED_PCC` (runner
default `0.88`). The exact wording is your model's, since the validator lives in your package. `PREFILL_MODEL`
comes from the model manifest here; exporting it directly also works.

## Gate 1 — mock migration and producer read-back

The runner serialises the KV-chunk table and device map and nothing else; the producer reads each chunk
device-lessly via `read_dram_umd` — the same UMD path the migration worker uses — and PCCs against golden.
This isolates one question: is `build_kv_chunk_table` correct? No endpoint, no worker and no MPI are
involved. Requires the producer to implement a read-back for this model's cache layout (see
`prefill_producer.py`).

In the binding, replace the migration block with:

```yaml
  PREFILL_MOCK_MIGRATION: "1"
  PREFILL_MIGRATION_TABLE_PATH: "/tmp/prefill_kv_chunk_table.pb"
  PREFILL_MIGRATION_DEVICE_MAP_PATH: "/tmp/prefill_kv_device_map.json"
  PREFILL_NUM_USERS: "2"
```

In the producer manifest, set `check_pcc: true` and `migration: {issue: false}`.

```bash
# Terminal 1 — runner (stays open across producer runs)
$RUN $BINDING $HOST:1

# Terminal 2 — producer
python -m models.demos.common.prefill.runners.prefill_producer --manifest $MANIFEST
```

`prefill_producer` here, deliberately: this gate migrates nothing, so it is one of the two runs that does
**not** use `migration_driver`.

Expect `[producer] KV cache PCC PASSED` (threshold `PREFILL_STANDALONE_CHUNKED_PCC`, producer default
`0.93` — note this differs from the runner's `0.88` default for the same variable).

This gate is not a prerequisite for the producer's golden PCC on the real-migration path, because the runner
serialises the device map there too — `serialize_device_map` is called from **both** the
`PREFILL_ENABLE_MIGRATION=1` and the `PREFILL_MOCK_MIGRATION=1` branch (`prefill_runner.py`, the two calls
under each branch). It has to be: `publish_table_and_wait_ready` hands the device map to the worker over the
migration client and leaves nothing on disk, while the producer's device-less read-back resolves chips from
the JSON sidecar. If that call ever regresses out of the real branch, the symptom is quiet — the producer
waits 60 s, logs `device map ... not found; skipping KV read`, and both the golden PCC **and** the src-KV
dump are silently lost. Gate 1 remains the cheapest way to separate a table problem from a transport problem.

## Gate 2 — loopback migration

The real DRAM → transport → DRAM copy. Loopback means `dest_endpoint_id` equals the endpoint's own id:
source and destination slots share one table, routed through the endpoint's internal A→B worker pair. Three
processes on one host — the shared-memory queues and UMD access are host-local, so all three must co-locate.
Needs the tt-llm-engine binaries built (`migration_endpoint`, `migration_worker`), pointed at the same
tt-metal tree the runner uses.

Launch order is A → B (wait for `WORKER_READY`) → C. Between runs, clear stale state:

```bash
pkill -f migration_endpoint ; pkill prte
rm -f /dev/shm/tt_h2d_* /dev/shm/tt_d2h_* /dev/shm/tt_prefill_layer_acks_* /tmp/migration_done.sentinel*
```

```bash
# ---- Terminal A — migration endpoint ----
cd <tt-llm-engine>/disaggregation/migration
./launch_migration_endpoints.sh --name_server_host $HOST \
    --prefill_hosts $HOST --prefill_endpoint_id 1

ls /dev/shm/mig_ep1_*      # all three queues must exist BEFORE starting B

# ---- Terminal B — runner (wait for WORKER_READY before starting C) ----
cd $TT_METAL_HOME && $RUN $BINDING $HOST:1

# ---- Terminal C — prefill + migrate ----
cd $TT_METAL_HOME
python -m models.demos.common.prefill.runners.migration_driver --manifest $MANIFEST
```

`migration_driver`, not `prefill_producer` — this gate migrates. See "Two producer entry points" above.

The driver prefills, drains the acks, migrates each pair, then writes the DONE sentinel; the runner is
polling for it and validates on-device. **PCC is logged by the runner (terminal B), not the driver** — the
driver only reports the transport (the `MIGRATE slot … complete` lines). Accuracy lives in the runner's
post-loop `[kv-migrate-validate]` output.

**2a — pairwise** (`PREFILL_MIGRATE_PAIRWISE=1`). Each destination is asserted bit-equal to its source,
golden-free and length-agnostic. Also catches cross-talk between concurrent migrations. Expect, in terminal
B, `[kv-migrate-validate] AFTER pairwise src=0 dst=2 min_pcc=…` per pair, then `ALL <N> pair(s) dst==src
PASSED` at or above `PREFILL_MIGRATE_PAIRWISE_PCC` (default `0.99`).

**2b — burst** (omit `PREFILL_MIGRATE_PAIRWISE`). Source and destination are both PCC'd against the same
golden. Expect `BEFORE src_slot=0 …` and `AFTER dst_slot=2 …`, then `ALL <N> migrated pair(s) PASSED`.

The two are either/or per run; run twice for both signals. Pairwise is the cheaper fidelity check — burst
anchors both slots to the golden but reads the cache twice.

For a multi-config cache (a sparse model publishes its index cache alongside the KV cache in one merged
table), **config-id order is the src↔dst contract**. Loopback is self-consistent by construction, but a real
prefill→decode run needs the decode endpoint to publish its configs in the same order.

---

## Runtime hooks each gate requires

Validation is driven by the optional runtime hooks in `ADDING_A_PREFILL_MODEL.md` §2. Implement only what the
gates you intend to run require.

| Gate | Hook | Signature requirement |
|------|------|-----------------------|
| 0 | `kv_cache_pcc_check` | accepts `trace_dir`, `first_layer_idx` |
| 1 | `build_kv_chunk_table` | serialises the block-cyclic layout; issues no comms |
| 2a pairwise | `read_slot_kv` | returns bare host tensors, `[num_layers, heads(or 1), seq_cache, head_dim]` |
| 2b burst | `kv_cache_pcc_check` | **also** accepts `real_len=` |
| 2a + golden anchor | `kv_cache_pcc_check` | **also** accepts `pt_path_override=` |

`runners/validation.py` calls these by keyword, so a runtime missing `real_len` or `pt_path_override` raises
`TypeError` before any PCC runs. It presents as a crash rather than a validation failure.

## Values that must agree across the three processes

| Value | Runner `global_env` | Producer manifest | Endpoint |
|-------|---------------------|-------------------|----------|
| H2D socket | `PREFILL_H2D_SERVICE_ID` | `transport.h2d_service_id` | — |
| layer depth | `PREFILL_NUM_LAYERS` | same var, exported | — |
| chunk size | `PREFILL_CHUNK_SIZE` | same var, exported | — |
| mesh | `PREFILL_SP` / `PREFILL_TP` | `transport.sp` / `.tp` | — |
| KV table | `PREFILL_MIGRATION_TABLE_PATH` | `migration.table_path` | — |
| sentinel | `MIGRATION_DONE_FILE` | `migration.done_file` | — |
| queues | `PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE` | `migration.{cmd,table,resp}_queue` | `--prefill-{cmd,table,resp}-queue` |
| client `.so` | `PREFILL_MIGRATION_CLIENT_DIR` | exported (not in the manifest) | — |
| chunk budget | `PREFILL_STANDALONE_CHUNKED_NCHUNKS` | `num_users` × `chunks` | — |
| slot count | `PREFILL_NUM_USERS` | ≥ max dst slot + 1 | — |

`PREFILL_STANDALONE_CHUNKED_NCHUNKS` is the one worth double-checking: the driver never sends a shutdown
push, so the runner uses that count to know prefill is done before it polls the DONE sentinel. Too low and
the runner exits mid-prefill; too high and it blocks on chunks that never come, printing no PCC either way.

Deriving every row of this table from a single place is the main thing the harness buys. It also rejects
attempts to set any of them by hand.
