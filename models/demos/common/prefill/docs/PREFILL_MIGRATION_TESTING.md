# Prefill KV-migration — configuration and gates

The prefill side on its own terms: the files that configure it, and two gates that can be run against it
directly with **no decode side present**. The tt-llm-engine launch harness generates and invokes this
underneath; this document is the layer beneath that. It is also the path to take when integrating a new
model, because the gates isolate failures that the harness necessarily reports as one.

Assumes the model is already integrated — adapter registered, golden trace staged, weight cache populated
(`ADDING_A_PREFILL_MODEL.md`, alongside this file). Why migration exists, what the source table is, and
how the runner publishes it are [`KV_MIGRATION_SPEC.md`](KV_MIGRATION_SPEC.md) (prefill first principles;
the decode counterpart is tt-blaze `docs/kv_migration_first_principles.md`). File-export into tt-d-gen
`kv_manager` is that same spec, §10. This file is how to *run* the gates.

| Gate | What it exercises | Needs |
|------|-------------------|-------|
| **1 — mock migration** | Prefill writes correct KV (precondition for everything) and the KV-chunk address table is correct, read device-lessly | tt-metal tree only |
| **2 — loopback migration** | The real DRAM → transport → DRAM copy, and the destination slots read back by the driver (`--verify-migration`, default `dst-bytes`) | + tt-llm-engine binaries |

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

# multi-host only: ONE rank-ordered host list, rank 0 first, reused by all three terminals
export HOSTS=<H0>,<H1>            # endpoint:  --prefill_hosts
export HOSTSP=<H0>:1,<H1>:1       # runner + driver: the --host list, passed to both launchers
```

Two shape constraints apply: `MAX_SEQ_LEN % CHUNK_SIZE == 0` and `CHUNK_SIZE % (SP*32) == 0` (each SP shard
stays 32-token-block aligned).

**`PREFILL_MOCK_MIGRATION=1` is single-rank only** — the runner rejects it for `num_ranks > 1`, because each
rank would publish a table covering just its own layer slice and a merged mock table is not implemented. So
**Gate 1** needs a 1-rank binding. `PREFILL_ENABLE_MIGRATION=1` (Gate 2) has no such restriction: the real
path merges the per-rank stage layouts through the worker
(`deliver_device_map_and_gather_stage_layout`), so a pipelined runner publishes one table spanning every
rank's layers. Gate 2 runs on 1, 2 or 4 ranks — see *Covering every rank* below for what that costs on the
read-back side.

---

## Model manifest

```json
// models/demos/my_model/tt/runners/manifests/my_model.json
{ "env": { "PREFILL_MODEL": "my_model", "PREFILL_GATE_FALLBACK_MODE": "DEVICE_FP32" } }
```

The `env:` map is applied verbatim by `setdefault`, so a rank binding's `global_env` still wins. It is the
only block the runner reads.

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
```

Nothing about validation appears here: the runner publishes the KV-chunk table and the device map and
that is the whole of its involvement. Both read-backs — the source `check_pcc` and the destination
`--verify-migration` — run out in the driver process against those two published artefacts.

The runner has no on-device check of its own to turn on, so there is no both-sides case to avoid: every
verdict in either gate comes from the driver process. A binding that sets a runner-side validation or
chunk-bound variable is carrying dead config — nothing reads it.

---

## Two producer entry points

`prefill_producer` and `migration_driver` are separate modules, and which one you run matters:

| module | what it is |
|--------|------------|
| `prefill_producer` | The plain runner test. H2D push, ack drain, golden PCC. Knows nothing about migration. |
| `migration_driver` | The migration entry point. Drives the H2D half **with `prefill_producer`'s own helpers**, then owns the `MigrationLayerClient` attach, the cross-endpoint pairing, the src→dst mapping, the `migrate()` calls, both sidecar files, and — for loopback — the destination read-back that proves the copy landed. On a multi-host runner, launch it with `run_migration_driver.sh` so those read-backs cover every rank (*Covering every rank*, below). |

The dependency runs one way: `prefill_producer` imports nothing from `migration_driver`, so a runner-only run
can never pull migration in. Migration has to share the producer's process — it needs that run's
resident-slot state and must migrate while the runner still holds the KV in device DRAM — which is why one
module covers both halves rather than chaining two.

So: **any run that migrates uses `migration_driver`**; only the no-migration gate (Gate 1 below)
uses `prefill_producer`. Invoking `migration_driver` *is* the opt-in, so `migration: {issue: true}` is
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
  check_pcc: true                 # SOURCE-slot golden PCC; the destination has its own gate below

migration:
  issue: true                     # attach the client and migrate after prefill
  dest_endpoint_id: 1             # == our own id => loopback
  dst_slot_offset: 2              # dst = src + 2  (0->2, 1->3)
  timeout_ms: 3600000
  done_file: /tmp/migration_done.sentinel                # driver-side only; no runner counterpart
  table_path: /tmp/prefill_kv_chunk_table.pb             # [MATCH RUNNER] shared storage if multi-host
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

### Multi-turn conversations

`PREFILL_PRODUCER_MULTI_TURN_PROB` (manifest: `workload.multi_turn_prob`, default `0.0`) is the probability
that a recycled slot **continues** its conversation instead of starting a fresh one. A continued turn
resumes writing at the previous turn's length **aligned down to 32** and replays the ≤31 dropped tokens as
part of its first chunk; aligning up instead would leave a permanent unwritten hole mid-sequence, and a
sub-tile write offset is rejected outright (`update_padded_kv_cache` asserts `kv_actual_global % 32 == 0`,
and the kernel's staircase disagrees with the host mirror off-tile). The replay is PCC-idempotent because
the rope table is keyed on absolute position, though it is not bit-idempotent.

Every length the producer reports stays **absolute** — measured from cache position 0, never relative to
the turn — because `actual_end` is a cache position. When a conversation no longer has room for another
full chunk the slot restarts from 0 rather than overrunning the per-user cache.

At the default `0.0` nothing draws from the rng and the schedule is byte-for-byte what it was before the
knob existed, so existing legs are unaffected.

Coverage: the multi-turn scheduling described here has no dedicated automated test yet — a device-free
host-logic test for turn continuation (align-down/replay and absolute-length bookkeeping) is a TODO. The
closest existing coverage is the on-device producer/runner PCC gate
`tests/test_producer_runner_e2e.py::test_producer_runner_pcc`, which drives the producer end to end and
fails if any resident slot's KV PCC is below threshold, though it is not multi-turn-specific.

---

## Verifying the migrated destination

`migration_driver` reads the destination slots back **itself**, in the driver process — one per host on a
multi-host runner, see *Covering every rank* — over the same device-less `read_dram_umd` path the source-side
`check_pcc` uses. The runner does no work for it: it does not hold the check, does not poll the sentinel, and
its own on-device checks are not part of this path. It does have to stay **alive** — the reads go to device
DRAM — which is the default, since the driver sends no shutdown push unless `PREFILL_SEND_SHUTDOWN=1` (and
even then, only after every rank's read-back).

Order within a run: `check_pcc` and `--dump-src-kv` read the *sources* before the migrate; this reads the
*destinations* after `wait_complete` lands, which is also after the DONE sentinel is published. So the
sentinel means "copied", not "verified".

```
--verify-migration {off | dst-bytes | dst-golden | both}     # env: PREFILL_VERIFY_MIGRATION
--verify-migration-layers 0,30,60                            # env: PREFILL_VERIFY_MIGRATION_LAYERS
```

| mode | asks | needs a golden? | model-specific? |
|------|------|-----------------|-----------------|
| `dst-bytes` *(default)* | is dst **byte-identical** to src, chunk by chunk? | no | no — nothing is decoded |
| `dst-golden` | does dst PCC to the src's golden trace? | yes | yes — decodes via the per-layout branch |
| `both` | runs each in turn, reporting transport and model correctness separately | yes | yes |
| `off` | nothing; the run proves transport only | — | — |

`dst-bytes` is the default because migration is a byte copy: a correct destination is bit-identical, so
there is no threshold to tune and no undefined-correlation hole over the all-zero pad tail or a dense
layer's `index_k` — exactly the regions a PCC-based fidelity check has to paper over with a threshold and
an all-zero short-circuit. `dst-golden` is not strictly stronger, it is differently strong: it proves the copy carries
**model-correct** data rather than merely the same bytes the source held, but it decodes through the
per-model layout branch, needs the golden on disk, and its PCC is undefined over the pad tail. `both` is
the honest answer when you want transport and model correctness reported separately.

For loopback migrations, both feed the **exit code**, alongside the source `check_pcc`: a driver run that
migrates and exits 0 has verified the destination unless you passed `off`. Cross-endpoint verification is
skipped as described below.

Three limits are worth internalising before reading a PASS:

- **Loopback only.** Cross-endpoint P→D is skipped with a warning — the destination lives in the decode
  galaxy's table, a separate address space, and looking `dst` up in *our* table would confidently read the
  wrong slot. Verify P→D on the decode side against a `--dump-src-kv` reference instead.
- **Cross-talk is invisible with one prompt.** If every slot replays the same trace, all sources are
  byte-identical and a copy landing in the wrong destination is indistinguishable from a correct one. Use
  `PREFILL_PRODUCER_SLOT_TRACES` (see *Per-slot prompts*) if you want that property covered. This is a
  property of identical sources, so it applies to `dst-golden` equally.
- **A layer subset makes a PASS a sample.** `--verify-migration-layers` (and, implicitly,
  `PREFILL_MIGRATION_LAYERS`) restricts the read; the driver logs a warning saying so. With a subset only
  config 0 is checked, because a sparse model's compacted index config indexes rows by full-indexer rank
  rather than global layer id.

Cost is the reason the subset flag exists: `dst-bytes` reads *both* slots, so it roughly doubles a
`check_pcc` pass — a full-depth Kimi pair is ~215k UMD reads. Chunks whose chips are not in this host's
device map are skipped rather than failed, and the skip count is printed so a `PASSED` line cannot be
mistaken for whole-model coverage — that skip is the whole subject of the next section. A check that
compared nothing is reported as a **failure**.

These two flags have no typed field in the manifest's `migration:` block. Set them on the CLI, export
them, or put them in the manifest's raw `env:` passthrough:

```yaml
env:
  PREFILL_VERIFY_MIGRATION: "both"
  PREFILL_VERIFY_MIGRATION_LAYERS: "0,30,60"
```

---

## Covering every rank — the multi-host driver

`read_dram_umd` is **host-local**. It reaches the chips in the machine the process is running on and nothing
else, and that applies to *both* driver read-backs — the source `check_pcc` and the destination
`--verify-migration`. On a pipelined runner spanning N hosts, one driver process therefore verifies 1/N of
the layers: every other rank's layers resolve to no local `unique_id` and are skipped. The run still says
`PASSED`, and it is telling the truth about a fraction of the model.

The fix is one process per host, and placing them is the **launcher's** job — `run_migration_driver.sh`,
the sibling of `run_pipeline_prefill.sh`. The module itself spawns nothing; it only splits by rank once MPI
has placed it, which keeps launch concerns (host list, MPI transport, env forwarding) in shell on both
sides of the run.

```bash
# multi-host: <manifest> <host_list> [tcp_iface] [extra driver args...]
./models/demos/common/prefill/runners/run_migration_driver.sh $MANIFEST "$HOSTSP"

# one galaxy: no host list, no MPI — or invoke the module directly, same thing
./models/demos/common/prefill/runners/run_migration_driver.sh $MANIFEST
```

The host list is in **rank order, rank 0 first**, and is the same list you gave `run_pipeline_prefill.sh` —
which is why it stays a command-line argument rather than a manifest field, exactly as on the runner side.
Pass one host (or none) and the script runs the module directly: one process, no MPI, this host's layers.

Roles, once launched:

| rank | does |
|------|------|
| 0 (the host you launched from) | the whole run — H2D feed, ack drain, `MigrationLayerClient`, `migrate()`, both sidecars — **plus** its own host's read-backs |
| every other | a device-less **validator**: no H2D connect, no client, no migrate. It reads its own host's KV back and votes |

Rank 0 must be the runner's rank-0 host: it alone attaches the H2D service and the `/mig_ep*` queues, which
exist only there. The script prints the rank-0 host next to `hostname` so a wrong order is visible before
anything hangs.

Three collectives over the distributed context (host-side MPI, no mesh device) sequence it, so every rank
must see the same env — the script forwards every exported `PREFILL_*`/`MIGRATION_*` variable and each rank
applies the same manifest itself:

| barrier | when | releases |
|---------|------|----------|
| **GO#1** | rank 0 has drained every LayerAck (so all layers are written) | the source `check_pcc` on every rank |
| **GO#2** | every `migrate()`'s `wait_complete` has returned | the destination check on every rank |
| **DONE** | all read-backs finished | the verdict fold — and it holds rank 0's shutdown sentinel until no rank is still reading |

The migrated `(src, dst, real_len)` triples are broadcast at GO#2 rather than re-derived per rank, so the
validators check what was actually migrated. Every rank still logs its own `N chunk(s) skipped — their chips
are not in this host's device map`; that is the *other* ranks' half. The real verdict is the fold:

```
[migration_driver] rank=0: ok=True
[migration_driver] rank=1: ok=True
```

Any rank failing fails the run's exit code. Two configuration requirements, and they pull in opposite
directions:

- `PREFILL_MIGRATION_TABLE_PATH` on **shared** storage — rank 0 writes the merged table, every validator
  reads it from its own host. The driver exits with a clear error if it points at `/tmp`, `/dev/shm`,
  `/run` or `/var/tmp`.
- `PREFILL_MIGRATION_DEVICE_MAP_PATH` **host-local** — each runner rank serializes its own host's
  `fabric_node → unique_id` map under that name, and that is precisely what filters each driver rank to its
  own layers. On shared storage the ranks race `<path>.tmp` → `os.replace`.

**The MPI interface must match the runner's.** These hosts are multi-homed and `docker0` carries the *same*
address on every one, so unpinned OpenMPI advertises addresses on one NIC and connects on another. The
script passes the same transport arguments `ttrun` gives the runner (`--mca btl self,tcp --mca
btl_tcp_if_include <iface>`), defaulting to `ens5f0np0` — the same default `run_pipeline_prefill.sh` uses.
Override it with the script's 3rd argument if you launched the runner with a different NIC (its own 3rd
argument). Get this wrong and `MPI_Init` never completes: **every rank logs `applied manifest`
and then goes silent**, usually alongside OpenMPI's *"accepted a TCP connection … cannot find a
corresponding process entry for that peer"*. The driver announces `joining the distributed context (N
rank(s) expected)` first so the hang is visible rather than mysterious.

`--dump-src-kv` stays rank-0-only (every rank would write the same `src_slot<N>.pt` with a different layer
subset and clobber the rest), so its dump covers one host's layers and warns when it does. Extra driver
flags go after the script's three positional arguments and reach every rank verbatim.

---

## Gate 1 — mock migration and producer read-back

This is also the KV-correctness precondition: prefill must write correct KV before migration means
anything. The runner serialises the KV-chunk table and device map and nothing else; the producer reads
each chunk device-lessly via `read_dram_umd` — the same UMD path the migration worker uses — and PCCs
against golden. This isolates one question: is `build_kv_chunk_table` correct? No endpoint, no worker and
no MPI are involved. Requires the producer to implement a read-back for this model's cache layout (see
`prefill_producer.py`). `PREFILL_MAX_SEQ_LEN` must be at least `chunks * CHUNK_SIZE` or the runner
asserts when a chunk overruns the cache — it no longer derives from a chunk count.

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

`prefill_producer` here, deliberately: this gate migrates nothing, so it is the one run that does
**not** use `migration_driver`.

Expect `[producer] KV cache PCC PASSED` (threshold `PREFILL_STANDALONE_CHUNKED_PCC`, producer default
`0.93`).

This gate is not a prerequisite for the producer's golden PCC on the real-migration path, because the runner
serialises the device map there too: one `serialize_device_map` call sits above the mock/real split inside the
`_migration_enabled` block, so **every rank on either path** publishes its own host-local sidecar. It has to
be: `deliver_device_map_and_gather_stage_layout` hands the map to the co-located *worker* over the migration
client and leaves nothing on disk, while every device-less read-back resolves chips from the JSON.

That call **had** regressed to living under `if _mock_migration:` only, and the symptom is exactly as quiet as
you would fear — a real-migration run published no sidecar, each reader polled 60 s, logged `device map ... not
found; skipping KV read`, and every PCC plus the `--dump-src-kv` reference vanished with nothing raising. If
you see that line, check this call before suspecting the table. Beware the confusing variant too: a *stale*
map left in `/tmp` by an earlier mock run makes the same misconfiguration look like it works. Gate 1 remains
the cheapest way to separate a table problem from a transport problem.

## Gate 2 — loopback migration

The real DRAM → transport → DRAM copy. Loopback means `dest_endpoint_id` equals the endpoint's own id:
source and destination slots share one table, routed through the endpoint's internal A→B worker pair. Needs
the tt-llm-engine binaries built (`migration_endpoint`, `migration_worker`), pointed at the same tt-metal
tree the runner uses.

Three terminals, and the shared-memory queues plus the H2D socket are host-local, so **A, B and C all start
on the same host** — the runner's rank-0 host. On a **pipelined runner** that stays true and each side simply
fans out from there, always in rank order with rank 0 first: the endpoint gets `--prefill_hosts <H0>,…,<Hn>`
(one `migration_endpoint` per host, each reading its own DRAM), the runner gets `<H0>:1,…,<Hn>:1`, and the
driver is launched with `run_migration_driver.sh <manifest> <same list>`, which places one process per host
(*Covering every rank*, above). Host order must be identical in all three.

Launch order is A → B (wait for `WORKER_READY`) → C. Between runs, clear stale state:

```bash
pkill -f migration_endpoint ; pkill prte
rm -f /dev/shm/tt_h2d_* /dev/shm/tt_d2h_* /dev/shm/tt_prefill_layer_acks_* /tmp/migration_done.sentinel*
```

```bash
# ---- Terminal A — migration endpoint ----
cd <tt-llm-engine>/disaggregation/migration
./launch_migration_endpoints.sh --name_server_host $HOST \
    --prefill_hosts $HOST --prefill_endpoint_id 1     # multi-host: --prefill_hosts $HOSTS

ls /dev/shm/mig_ep1_*      # all three queues must exist BEFORE starting B

# ---- Terminal B — runner (wait for WORKER_READY before starting C) ----
cd $TT_METAL_HOME && $RUN $BINDING $HOST:1            # multi-host: $RUN $BINDING $HOSTSP
                                                      # (with an N-rank $BINDING)

# ---- Terminal C — prefill + migrate ----
cd $TT_METAL_HOME
./models/demos/common/prefill/runners/run_migration_driver.sh $MANIFEST      # one host
./models/demos/common/prefill/runners/run_migration_driver.sh $MANIFEST "$HOSTSP"   # multi-host
```

`migration_driver`, not `prefill_producer` — this gate migrates. See "Two producer entry points" above.

The driver prefills, drains the acks, migrates each pair, writes the DONE sentinel, and then reads the
destination slots back itself. **Everything lands in terminal C** — transport as the `MIGRATE slot …
complete` lines, accuracy as the `verify …` lines below. Terminal B is a pure serving loop and logs no PCC
at all. Multi-host, terminal C's lines are `[1,N]`-tagged per rank and the run's actual verdict is the
`rank=N: ok=…` fold at the end.

**2a — byte compare** (the default, `--verify-migration dst-bytes`). Each destination is asserted
byte-identical to its source, chunk by chunk, golden-free and model-agnostic. Expect one
`verify bytes: slot 0 -> 2 config 0: …` planning line per pair and config, then:

```
[migration_driver] verify bytes PASSED: 2 pair(s), 1342 chunk(s) byte-identical dst == src
```

**2b — golden anchor** (`--verify-migration dst-golden`). Each destination is PCC'd against the source
slot's golden trace, at `PREFILL_STANDALONE_CHUNKED_PCC` (same threshold as `check_pcc`). Expect an
`AFTER dst_slot=2 (src=0) min_pcc=…` line per pair — the destination half of the number `check_pcc: true`
already reports for the sources — then `verify golden PASSED: <N> migrated dst slot(s) >= <thr>`.

`--verify-migration both` runs them in one pass, which is the useful default when a model is new: the byte
compare isolates the transport, and the golden PCC tells you whether what was transported was right in the
first place. See *Verifying the migrated destination* above for the cost and the three coverage caveats
(loopback only, cross-talk needs per-slot prompts, a layer subset is a sample).

The runner's request loop is unbounded and the driver's chunks are not a shutdown, so set
`PREFILL_SEND_SHUTDOWN=1` on the driver to close the stream when it is done — otherwise terminal B sits in
`recv` until you SIGTERM it. The driver sends that sentinel **last**: after the destination read-back and,
multi-host, after the DONE barrier, so it never tears the mesh down under a UMD read in flight on any rank.
Leaving it unset is the useful choice while iterating — terminal B stays up and you can rerun terminal C
against it.

For a multi-config cache (a sparse model publishes its index cache alongside the KV cache in one merged
table), **config-id order is the src↔dst contract**. Loopback is self-consistent by construction, but a real
prefill→decode run needs the decode endpoint to publish its configs in the same order.

---

## Runtime hooks each gate requires

Both gates run off the optional runtime hooks in `ADDING_A_PREFILL_MODEL.md` §2. Implement only what the
gates you intend to run require.

| Gate | Hook | Signature requirement |
|------|------|-----------------------|
| 1 | `build_kv_chunk_table` | serialises the block-cyclic layout; issues no comms |
| 2 | `kv_migration_base_address` | this rank's KV base DRAM address, for the cross-stage table merge |
| 2 `dst-bytes` | **none** | nothing is decoded — the byte compare is model-agnostic |
| 2 `dst-golden` | none beyond Gate 1 | reuses the producer's own read-back, not a runtime hook |

The destination check adds no per-model surface: `dst-bytes` compares raw chunks and never decodes, and
`dst-golden` rides on `prefill_producer._read_slot_kv_and_check_pcc` — the same layout branch Gate 1
already needs — rather than on a runtime hook. A new model whose cache is neither MLA nor the M3 triple
gets a branch there, and both gates work.

## Values that must agree across the three processes

| Value | Runner `global_env` | Producer manifest | Endpoint |
|-------|---------------------|-------------------|----------|
| H2D socket | `PREFILL_H2D_SERVICE_ID` | `transport.h2d_service_id` | — |
| layer depth | `PREFILL_NUM_LAYERS` | same var, exported | — |
| chunk size | `PREFILL_CHUNK_SIZE` | same var, exported | — |
| mesh | `PREFILL_SP` / `PREFILL_TP` | `transport.sp` / `.tp` | — |
| KV table | `PREFILL_MIGRATION_TABLE_PATH` | `migration.table_path` | — |
| device map | `PREFILL_MIGRATION_DEVICE_MAP_PATH` | `migration.device_map_path` | — |
| sentinel | — (driver-side only) | `migration.done_file` | — |
| queues | `PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE` | `migration.{cmd,table,resp}_queue` | `--prefill-{cmd,table,resp}-queue` |
| client `.so` | `PREFILL_MIGRATION_CLIENT_DIR` | exported (not in the manifest) | — |
| host list | the `--host` list given to `run_pipeline_prefill.sh` | `run_migration_driver.sh`'s 2nd arg, same order | `--prefill_hosts`, same order |
| MPI NIC | `run_pipeline_prefill.sh`'s 3rd arg (`ens5f0np0`) | `run_migration_driver.sh`'s 3rd arg, same value | — |
| slot count | `PREFILL_NUM_USERS` | ≥ max dst slot + 1 | — |

The two path rows pull in opposite directions once more than one host is involved: the **table** must be on
shared storage (rank 0 writes it, every driver rank reads it) and the **device map** must not be (each rank
publishes its own host's chips under that name, which is what scopes each driver rank to its own layers).

The sentinel has no runner-side counterpart: the driver writes it for an external P→D consumer, and
nothing in the runner reads it. It also means "copied", not "verified" — the destination read-back runs
after it is published.

`PREFILL_STANDALONE_CHUNKED_NCHUNKS` is absent because nothing reads it: the request loop no longer takes a
bound and always runs until the stream closes. That is what the driver needs — a bounded runner would head
for teardown while the driver was still migrating and reading back — but it means the driver must close the
stream itself with `PREFILL_SEND_SHUTDOWN=1`, or terminal B sits in `recv` after the gate has passed.

Deriving every row of this table from a single place is the main thing the harness buys. It also rejects
attempts to set any of them by hand.

---

## The endpoint's processes (Gate 2 only)

| Process | Role |
|---|---|
| `migration_endpoint` | Owns the outward queues `/mig_ep<id>_{cmd,table,resp}` the runner talks to; relays commands inward and `WORKER_READY` outward. Copies no KV itself. |
| `migration_worker` × 2 | The processes that touch DRAM: sender ("A") and loopback receiver ("B"), each with queues `/ep_<id>_{a,b}_{cmd,table,resp}`. Started by an MPI launcher — `prte` holds a pool of *slots* (one per process), `prun` requests them. |

A worker reports `WORKER_READY` only once it holds the KV chunk table, the device map, and its A↔B link. The
runner supplies the first two, then waits — it cannot distinguish a slow worker from an absent one.

---

## Troubleshooting: the runner times out in `wait_ready`

```
RuntimeError: MigrationLayerClient::wait_ready: timeout after 120000ms
```

Almost always: the two workers were never started, so nothing can answer. Confirm in the endpoint log
(`/tmp/launch_mig_ep_<id>_*.log`; it holds binary bytes, so `grep` needs `-a`):

```bash
grep -a JOB_FAILED_TO_MAP $(ls -t /tmp/launch_mig_ep_1_*.log | head -1)
```

The workers are launched as one request for **two slots on the host**, and `prte` started inside a batch
allocation takes its slot count from the allocation instead of from the `--host <node>:2` it was given. On a
partition that advertises a whole accelerator node as one CPU (`scontrol show node <n>` → `CPUTot=1`) the
request for two is refused. A larger allocation cannot fix this — the node has one CPU to give. Detach the
launcher from the scheduler instead, before launching the endpoint:

```bash
# shellcheck disable=SC2046
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p')
export PRTE_MCA_ras="^slurm"     # take the node pool from --host, not from the allocation
```
