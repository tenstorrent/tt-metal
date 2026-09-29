# MiniMax-M3 KV migration runbook — prefill loopback, decode loopback, prefill → decode, KV Manager

Copy-paste commands for the four M3 migration tests on the Blackhole galaxies. Paths are this
checkout's (`/data/philei/...`). The mechanism and config layering are explained elsewhere; this file
only tells you what to type.

| § | Test | Galaxies | Verdict comes from | Reference |
|---|------|----------|--------------------|-----------|
| 1 | prefill → prefill loopback (Gate 2 / P2) | 1 | `migration_driver` (terminal C) | `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` |
| 2 | decode → decode loopback | 1 | `[kv-slice]` lines from every decoder rank | `tt-blaze/docs/DECODE_MIGRATION.md` |
| 3 | prefill → decode, two galaxies (harness) | 2 | `pd_migration_complete` step | `tt-llm-engine/disaggregation/launch_harness/README.md` |
| 4 | prefill → decode through the KV Manager (tt-d-gen, `kv_manager: "kvm"`) | 2 | request checks + the KVM log, then blaze's `[kv-golden]` | `tt-d-gen/kv_manager/scripts/fleet/kvm-fleet-launcher.md` |

Do §1 and §2 before §3, and §3 before §4. The harness reports a failure on either side as a single
failure. §4 adds a serving stack and a second transport on top of §3.

Every migrating run needs a **bf8 prefill index cache** (`M3_INDEX_CACHE_BF16=0`). Decode stores index_k
in bfp8, and the workers copy raw chunk bytes, so building a migration table rejects a bf16 index_k.
The checked-in migration bindings already set it.

---

## 0. Trees, builds, environment

Three trees. Each builds against its **own** tt-metal; they meet only at the KV chunk-table protobuf
and the `/dev/shm` queues, so their tt-metal versions need not match. Never override `TT_METAL_DIR`
for the engine build.

```bash
export PREFILL=/data/philei/tt-metal
export BLAZE=/data/philei/tt-metal/tt-blaze
export ENGINE=/data/philei/tt-metal/tt-llm-engine
```

### 0.1 Build (compute node only — never the login node)

**a) prefill tt-metal** — Python bindings + venv, the runner is Python:

```bash
cd $PREFILL
./build_metal.sh --build-type Release --enable-ccache
./create_venv.sh
```

**b) engine migration layer** — `migration_endpoint`, `migration_worker`, `_migration_client.so`.
The bundled submodule is built first; the migration layer probes that build for `libtt_metal.so`
and **silently falls back to SimulatedDram** if it is missing, so check the RUNPATH afterwards.

```bash
cd $ENGINE
git submodule update --init --recursive tt-metal
cd $ENGINE/tt-metal && ./build_metal.sh -c --without-python-bindings
cd $ENGINE
rm -rf disaggregation/migration/build_RelWithDebInfo        # stale CPM cache from another metal tree
env -u TT_METAL_DIR -u TT_METAL_BUILD_DIR \
  ./build_migration_layer.sh --build-type RelWithDebInfo --jobs $(nproc)

ls disaggregation/migration/build_RelWithDebInfo/bin/{migration_endpoint,migration_worker} \
   disaggregation/migration/build_RelWithDebInfo/python/_migration_client*.so
readelf -d disaggregation/migration/build_RelWithDebInfo/bin/migration_worker | grep RUNPATH
# must resolve into $ENGINE/tt-metal/build/...
```

**c) tt-blaze decode** — `--bundle-python` because the galaxy nodes have no `~/.local` python for
a symlinked venv:

```bash
cd $BLAZE
git submodule update --init tt-metal
git submodule status tt-metal            # no leading '+' => checkout matches the pin
./build_blaze.sh --development --enable-ccache
cd $BLAZE/tt-metal && ./create_venv.sh --bundle-python
```

Device-free tests, login node is fine, run after every rebuild:

```bash
cd $PREFILL && source python_env/bin/activate
pytest models/demos/minimax_m3/tests/test_kv_chunk_table_merge.py \
       models/demos/common/prefill/tests/test_prefill_producer_kv_decode.py -q

cd $BLAZE && source env.sh
pytest tests/blaze/migration/test_minimax_m3_kv_migration_spec.py tests/blaze/infra/test_multi_config_kv_table.py -q
```

### 0.2 Allocation

Galaxy nodes are `bh-glx-120-b*` in partition `bh_sc5_B2B9_D12` (`sinfo -h -o "%P %N %T" | grep b0`).
The short names `bh-glx-b08u02` etc. used in the yaml files resolve **on the compute nodes only**,
not from `slurm-login`.

```bash
# one galaxy (§1, §2)
salloc -p bh_sc5_B2B9_D12 -N 1 --exclusive -t 06:00:00
# two galaxies (§3): prefill = b08u02, decode = b09u02
salloc -p bh_sc5_B2B9_D12 -N 2 --nodelist=bh-glx-120-b08u02,bh-glx-120-b09u02 --exclusive -t 06:00:00
# extra terminals onto a node of a running allocation
srun --jobid <JOBID> -w bh-glx-120-b08u02 --overlap --pty bash
```

One-time, for anything that spans two nodes (§1 two-galaxy variant, §3): passwordless node→node ssh
from the host you drive from. `/data/philei/.ssh/galaxy0` has a passphrase, so the harness's batch-mode
ssh cannot use it; the passwordless pair at `/data/philei/.mig_ssh/id_ed25519` is for this. Home dirs
are node-local, so every node in the run needs the public half in its `authorized_keys`, and the
driving host additionally needs a `Host bh-glx-*` block. Redo after a reimage.

```bash
# on EVERY node of the run (srun into it; no ssh needed for this step)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
grep -qf /data/philei/.mig_ssh/id_ed25519.pub ~/.ssh/authorized_keys 2>/dev/null || cat /data/philei/.mig_ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
# on the DRIVING host only (prefill host for §3)
grep -q '^Host bh-glx-\*' ~/.ssh/config 2>/dev/null || printf 'Host bh-glx-*\n  IdentityFile /data/philei/.mig_ssh/id_ed25519\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n' >> ~/.ssh/config
chmod 600 ~/.ssh/config
ssh -o BatchMode=yes <other-node> hostname      # and to itself: the harness ssh's to the prefill host too
```

The galaxies need no fabric link between them for §1.3 or §3: KV goes device DRAM → host staging
buffer → MPI over the host NIC (`--tcp-transport`, or RoCE via `setup_roce_mpi.sh`) → remote host →
device DRAM. Any two reachable galaxies work.

### 0.3 Per-terminal preamble

Prefill-side terminals (§1, §3):

```bash
cd /data/philei/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
ulimit -Su $(ulimit -Hu)          # FIRST: RLIMIT_NPROC is per-user node-wide; a live runner exhausts it
export HOST=$(hostname -s)
export RUN=./models/demos/common/prefill/runners/run_pipeline_prefill.sh
export M3=models/demos/minimax_m3/tt/runners/manifests
export ENGINE=/data/philei/tt-metal/tt-llm-engine
export BLAZE=/data/philei/tt-metal/tt-blaze
export MIG=$ENGINE/disaggregation/migration/build_RelWithDebInfo
export PREFILL_MIGRATION_CLIENT_DIR=$MIG/python    # the loopback bindings expand this
mkdir -p /data/philei/tmp /data/philei/tmp/m3_pd /data/philei/disagg_runs
```

Decode-side terminals (§2):

```bash
export BLAZE=/data/philei/tt-metal/tt-blaze ENGINE=/data/philei/tt-metal/tt-llm-engine
cd $BLAZE && source env.sh        # venv + PYTHONPATH + TT_METAL_HOME=$BLAZE/tt-metal
ulimit -Su $(ulimit -Hu)
export DECODE_HOST=$(hostname -s)
mkdir -p /data/philei/disagg_runs
```

### 0.4 Slurm detach — endpoint terminals and any multi-rank runner terminal

The migration endpoint spawns two workers and asks `prte` for 2 slots on the node; Slurm advertises
a galaxy as `CPUTot=1`, so inside an allocation the spawn fails with `PMIX_ERR_JOB_FAILED_TO_MAP` /
"All nodes which are allocated for this job are already filled" and the model side later times out
in `wait_ready`. The same bites `tt-run` with more than one rank per host. Run this in the endpoint
terminal, in any runner terminal using `$HOST:2`/`$HOST:4`, and in the §3 harness shell:

```bash
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p'); export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
env | grep -c SLURM_        # must print 0
```

Not in terminals you use for `srun --overlap` attaches — those need their Slurm env.

### 0.5 Migration endpoint (shared by §1 and §2)

Same launcher and binary on both sides; only the role flags differ. `--tcp-transport` is required:
`disaggregation/migration/.roce_env` exists, so the launcher defaults to RoCEv2 and hard-fails without
a usable RoCE device. Loopback is a same-galaxy DRAM→DRAM copy, TCP is the right data plane.
`TT_METAL_HOME` must be the **engine's** submodule in this terminal: the workers were built against it
and resolve SOC descriptors from it (the launcher defaults to it only when the variable is unset).

```bash
# after 0.3 + 0.4 in this terminal
export TT_METAL_HOME=$ENGINE/tt-metal
cd $ENGINE/disaggregation/migration

# prefill side, endpoint id 1   (§1)
./launch_migration_endpoints.sh --name_server_host $HOST --prefill_hosts $HOST --prefill_endpoint_id 1 --tcp-transport
# decode side, endpoint id 0    (§2)
./launch_migration_endpoints.sh --name_server_host $DECODE_HOST --decode_hosts $DECODE_HOST --decode_endpoint_id 0 --tcp-transport
```

The launcher holds the terminal and streams the endpoint log to it; export `MIGRATION_LOG_TEE=1` to also
keep a copy under `/tmp/mig_logs_<uid>/` (worker stdout/stderr always land there). Ready when the three outward queues exist **and two**
`migration_worker` processes run (the launcher warns but does not fail if the spawn failed):

```bash
ls /dev/shm/mig_ep<ID>_{cmd,table,resp}
for p in /proc/[0-9]*; do f=$(tr '\0' '\n' <$p/cmdline 2>/dev/null | head -1); [ "${f##*/}" = migration_worker ] && echo $p; done | wc -l   # want 2
grep -a -E "FAILED_TO_MAP|already filled" $(ls -t /tmp/mig_logs_$(id -u)/launch_mig_ep_<ID>_*.log | head -1)  # want nothing
```

For endpoint 1, `/data/philei/tt-metal/m3_gate2_endpoint.sh` wraps the launch with a 2-rank
placement preflight and `/data/philei/tt-metal/m3_gate2_check.sh` runs the readiness checks above.

**One endpoint launch per runner launch.** A worker exits on a second `SET_TABLE`, and the runner
publishes a table at startup, so a second runner against a live endpoint kills both workers.

### 0.6 Cleanup between runs (Ctrl-C the endpoint terminal first)

```bash
pkill -f migration_endpoint; pkill prun; pkill prted; pkill prte
rm -f /dev/shm/mig_ep[01]_* /dev/shm/ep_[01]_[ab]_* /dev/shm/ttmig.* \
      /dev/shm/tt_h2d_* /dev/shm/tt_d2h_* /dev/shm/tt_prefill_layer_acks_* \
      /tmp/m3_kv_chunk_table.pb /tmp/m3_kv_device_map*.json /tmp/m3_migration_done.sentinel* \
      /data/philei/tmp/m3_kv_chunk_table_pp.pb \
      /data/philei/tmp/m3_pd/migration_done.sentinel* /data/philei/tmp/m3_pd/prefill_migration_handoff.json
```

Two of these are silent footguns: the driver waits only for the table file to *exist*, and the
decode driver polls only for the sentinel's *existence*, so a leftover of either makes the next run
proceed on stale data.

---

## 1. Prefill → prefill loopback (one galaxy)

Prefill slots 0,1 (2 chunks × 5120 tokens each, golden `longbook_10240`), migrate 0→2 and 1→3
through endpoint 1's internal A→B worker pair, then the driver reads the destinations back over UMD.
Checked-in config: `$M3/m3_binding_loopback_migration_1rank.yaml` (runner) and
`$M3/m3_producer_loopback_migration.yaml` (driver). Three terminals, all on the galaxy node.

```bash
# ---- Terminal A — endpoint id 1 (0.3, 0.4, then 0.5 prefill line). Confirm READY:
/data/philei/tt-metal/m3_gate2_check.sh

# ---- Terminal B — runner (0.3). WORKER_READY comes after minutes of silence: the 9-config,
#      691200-entry table is built one Python call at a time.
$RUN $M3/m3_binding_loopback_migration_1rank.yaml $HOST:1
#   wait for:  [migration] WORKER_READY: table=/tmp/m3_kv_chunk_table.pb
#              [migration] LayerAck channel ready at /tt_prefill_layer_acks_m3_prefill

# ---- Terminal C — prefill + migrate + verify (0.3). Every verdict lands HERE.
PREFILL_SEND_SHUTDOWN=1 python -m models.demos.common.prefill.runners.migration_driver \
  --manifest $M3/m3_producer_loopback_migration.yaml
```

`migration_driver`, not `prefill_producer` — the producer knows nothing about migration.
`PREFILL_SEND_SHUTDOWN=1` closes the runner's request loop after the last read-back; leave it off
while iterating and terminal B stays up for another terminal-C run.

**Expect in C:**

```
[producer] KV cache PCC PASSED                               <- source slots vs golden (check_pcc)
MIGRATE slot 0 -> 2 ... complete / MIGRATE slot 1 -> 3 ... complete
[migration_driver] verify bytes PASSED: 2 pair(s), N chunk(s) byte-identical dst == src
```

`verify bytes` is the default `--verify-migration dst-bytes`, across all 9 configs. The table names
them `"00"`..`"08"`, matching blaze: `00`–`03` = K heads 0–3, `04`–`07` = V heads 0–3, `08` = index_k.
index_k rows exist only on MSA layers, because blaze publishes it only there. The driver therefore logs
`N chunk(s) not compared — their table rows are unpublished on both sides`; at 10240 tokens N is 1920
(3 dense layers × 320 chunks × 2 pairs). For the golden-anchored destination check as well, append
`--verify-migration both` (expect an extra `verify golden PASSED`). Exit code 0 means every check
passed. `[spsc-trace] ... wait_complete` backtraces in C are queue instrumentation, not errors.

### 1.1 Cheaper pre-check — mock migration, no endpoint (Gate 1)

Isolates the address table from the transport. Two terminals, no MPI, nothing migrates:

```bash
$RUN $M3/m3_binding_mock_migration_1rank.yaml $HOST:1                           # terminal 1
python -m models.demos.common.prefill.runners.prefill_producer \
  --manifest $M3/m3_producer_mock_migration.yaml                                # terminal 2
# expect: [producer] layer acks 240/240 ... [producer] KV cache PCC PASSED
```

### 1.2 Variant — 2-stage intragalaxy pipeline loopback (P2)

Same three terminals; the runner spans two Z-linked 4×4 sub-meshes (rank 0 = layers 0–29, rank 1 =
30–59) and rank 0 publishes one merged 60-layer table. Differences from §1:

```bash
# Terminal B needs 0.4 (Slurm detach) because it is 2 ranks on one host, and the sub-mesh weight cache:
export PP_CACHE=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
TT_CACHE_PATH=$PP_CACHE $RUN $M3/m3_binding_loopback_migration_intragalaxy_2rank.yaml $HOST:2

# Terminal C
PREFILL_SEND_SHUTDOWN=1 python -m models.demos.common.prefill.runners.migration_driver \
  --manifest $M3/m3_producer_loopback_migration_2rank.yaml
```

Expect both ranks to log their device maps (`/tmp/m3_kv_device_map_r{0,1}.json`), rank 0 to
publish `/data/philei/tmp/m3_kv_chunk_table_pp.pb` (roughly twice the Gate 2 build time), and the
same three verdict lines in C, with source PCC over `60/60 local layers`.

### 1.3 Variant — two galaxies, 2 stages (validated 2026-08-25, b08u02 + b09u02)

Endpoint id 1 whose workers span both hosts; runner rank 0 on A. All three terminals on A. Needs the
node→node ssh from 0.2 and the Slurm detach in **both** the endpoint and the runner terminal.

```bash
export A=bh-glx-b08u02 B=bh-glx-b09u02
# Terminal A (0.3, 0.4, TT_METAL_HOME=$ENGINE/tt-metal)
$ENGINE/disaggregation/migration/launch_migration_endpoints.sh --name_server_host $A --prefill_hosts $A,$B --prefill_endpoint_id 1 --tcp-transport
# Terminal B (0.3, 0.4)
$RUN $M3/m3_binding_loopback_migration_2galaxy_2rank.yaml $A:1,$B:1
# Terminal C (0.3) — one driver process per host so host B's layers are read back too
./models/demos/common/prefill/runners/run_migration_driver.sh $M3/m3_producer_loopback_migration_2galaxy.yaml "$A:1,$B:1"
```

The per-host driver prints its own skip count for the other host's chunks; the verdict is the
`rank=N: ok=True` fold at the end. The table path in that binding is on `/data/philei/tmp`, which
both hosts see (the driver rejects `/tmp` for multi-host).

---

## 2. Decode → decode loopback (one galaxy)

`blaze.decode_migration_driver` prefills one prompt into slot 0, migrates every KV chunk of every
layer to slot 2 through endpoint 0, byte-compares the two slots per head per 32-token chunk, then
keeps decoding from slot 2. Topology: the 4-stage ring embed → dense(layer 0) → sparse(layer 3) →
lm-head, 4 ranks × 8 chips, rank binding `blitz_decode_single_galaxy_4stage_rank_bindings.yaml`.
Manifest: `$BLAZE/tests/testfiles/m3_migration_loopback.yaml` (synthetic weights — the byte compare is
weight-independent; the decoded text is gibberish by design).

Config smoke test first, login node is fine:

```bash
cd $BLAZE && source env.sh
python3 -m blaze.decode_migration_driver --manifest $BLAZE/tests/testfiles/m3_migration_loopback.yaml --dry-run
```

On the galaxy node:

```bash
# ---- Terminal A — endpoint id 0 (0.3 decode preamble, 0.4, then 0.5 decode line). Confirm:
ls /dev/shm/mig_ep0_{cmd,table,resp}

# ---- Terminal B — driver (0.3 decode preamble). PRINT=1 in front prints the ttrun command and exits.
set -o pipefail
$BLAZE/tests/testfiles/run_decode_migration.sh m3_migration_loopback.yaml $DECODE_HOST:4 \
  2>&1 | tee /data/philei/disagg_runs/m3_decode_loopback.log
```

The launcher exports `TT_METAL_HOME`/`PYTHONPATH`, raises `ulimit -Su`, sets `OPENBLAS_NUM_THREADS=8`
and passes `--oversubscribe` itself (4 ranks on one host), so terminal B needs no Slurm detach.

**Expect** — one line per KV-owning rank and config; the two decoder stages report, embed and
lm-head `SKIP`:

```
[kv-slice] mesh1 k_h0 slot0->slot2: PASS over 64 positions x 1 head(s)      <- layer 0 (dense): k_h0..3, v_h0..3; index_k SKIP
[kv-slice] mesh2/layer3 index_k slot0->slot2: PASS over ...                  <- layer 3 (sparse): all 9 configs
[decode-driver] stage N validation: ...
```

Pass = every decoder-stage line is `PASS`. `FAIL (N mismatched chunk(s))` logs
`(head, chunk, pos, dim, max_abs)` — `pos` at a bank-slice boundary points at `bph`/`st_pb`, at a
128-token boundary at the block-cyclic `index_k` mapping, at a slot boundary at the slot stride.
**`SKIP` everywhere means nothing was validated.**

```bash
grep -E "\[kv-slice\]|\[decode-driver\] stage" /data/philei/disagg_runs/m3_decode_loopback.log
```

If every rank dies in `build_pipeline` with `resolve_graph_layout(): incompatible function arguments`, the
`_ttnn.so` was built from an older tt-metal than blaze's Python expects (blaze passes `nodes=` since
`5c6720ae8`): `git submodule status tt-metal` shows a leading `+`. Sync and rebuild per §0.1 c).

If every rank dies in `open_mesh_device` with `Timed out while waiting for active ethernet core ... Try
resetting the board`, a previous run left the fabric routers up (typically ranks that crashed after opening
the mesh). Tear the endpoint down, `tt-smi -glx_reset`, relaunch the endpoint, rerun.

If the driver dies in `wait_ready`, the cause is worker-side and not in the driver output:

```bash
tail -80 $(ls -t /tmp/mig_logs_$(id -u)/launch_mig_ep_0_*.log | head -1)
for f in /tmp/mig_logs_$(id -u)/launch_mig_wkr_0_*/*/*/std{out,err}; do [ -s "$f" ] && { echo "== $f"; tail -40 "$f"; }; done
```

M3 constraints baked into the manifest: `n-slots` must exceed the largest slot in `migrate-pairs`
**plus one** scratch slot for the teardown dummy token (`0:2` → `n-slots: 3`; the driver now rejects
less at startup); the prompt is 41 tokens so the migrated range `[0, 64)` crosses a bank boundary;
the cache depth is not a knob (`kv_cache_depth(DEFAULT_MAX_SEQ_LEN)` = 66560, a multiple of 1024 as the
spec requires). `model:` is the registry key `MiniMaxAI/MiniMax-M3` (`docs/DECODE_MIGRATION.md`'s
`minimax_m3` is not a registry key). `double-torus` is accepted and ignored by M3; the fabric comes
from the 4-mesh MGD the rank binding names.

---

## 3. Prefill → decode across two galaxies (harness)

Prefill galaxy `bh-glx-b08u02` (1 rank, SP=8 × TP=4, 60 layers, 10240 tokens, endpoint 1) pushes KV
into a tt-blaze 4-stage decode ring on `bh-glx-b09u02` (layers {0, 3}, endpoint 0). One command from
the prefill host: `tt-llm-engine`'s launch harness, scenario `pd_migration`, config
`$ENGINE/disaggregation/launch_harness/disagg_harness_m3_pd.yaml`.

### 3.1 Prerequisites and known limits

- All three trees carry the P→D changes: tt-metal #55911 (M3 runtime accepts the runner's
  `metadata_msg`), tt-blaze #4014 (positional table config ids, K/V `load_golden`, dummy-slot gate),
  tt-llm-engine #404 (`pd` mode sizes the prefill table like the decode table, completion-step fixes,
  the M3 config). Without the blaze change every decode stage skips or the golden check fails.
- The tt-blaze submodule is at its pin and rebuilt (`git submodule status tt-metal` shows no `+`).
- The harness does not detach from Slurm; the shell launching it must (§0.4), or the endpoint workers
  fail to map and the prefill runner times out in `wait_ready` while the launcher still reports
  `ENDPOINT_ALIVE`.
- `TT_METAL_HOME` in the harness shell reaches only the endpoint step; set it to the engine submodule
  (§0.5). The prefill steps get `prefill.tree` from the yaml.
- Use Slurm hostnames and run the harness on the prefill node; the short `bh-glx-b*` aliases do not
  resolve everywhere.
- Checks:
  - CHECK 1 = prefill source vs golden (producer, over UMD).
  - CHECK 2 = `[kv-src]` source vs destination. Not available for M3: there is no source dump.
  - CHECK 3 = `[kv-golden]` destination vs golden, K/V and, on the sparse layer, index_k.
    `longbook_10240` carries index_k for the MSA layers.
- The migration layer's own tt-metal must read the compressed (`STRIDED_ROWS`) tables that tt-metal
  ≥ #57098 exports. At the engine's old pin (e49ac055d6) those rows import as empty: the sender logs
  `no destination rows for config N ...; skipped` and migrates nothing. Two fixes:
  - use tt-llm-engine#419 (tt-metal cc834df1b8);
  - or set `KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES=2080374784` in the harness `prefill.env` and
    `decode.flags.env`, which makes the exporters mirror the per-chunk entries.

  With blaze ≥ #4502 every K/V config is compressible, so without either fix nothing migrates.
- The decode driver's wait for the sentinel has no timeout.
- Last passing run: 2026-09-29, b08u08 → b09u02, tt-blaze main + tt-llm-engine#419, no dual-write.
  CHECK 3: K/V ≥ 0.999, index_k 0.99976.

### 3.2 Run

Prerequisites: §0.1 builds on all three trees, §0.2 two-node allocation and node→node ssh (the
harness ssh's to the decode host for preclean, the decode endpoint, and the decode driver), and the
§1/§2 loopbacks passing on their respective galaxies.

```bash
# on bh-glx-120-b08u02, inside the allocation
# 0.3 prefill preamble, then:
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p'); export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
export TT_METAL_HOME=$ENGINE/tt-metal        # reaches the endpoint/worker step only; prefill steps get prefill.tree
ssh -o BatchMode=yes bh-glx-b09u02 hostname  # must print the decode host
cd $ENGINE
# Copy disaggregation/launch_harness/disagg_harness_m3_pd.yaml, fill in its [EDIT] fields, point CFG at the copy.
export CFG=/data/philei/disagg_runs/m3_pd_tools/disagg_harness_m3_pd.local.yaml

# Read the plan: every command, log path and gate, no hardware touched. Also runs the preflight
# (paths, shared dir, golden trace, launcher scripts).
python3 -m disaggregation.launch_harness pd_migration \
  --config "$CFG" --dry-run

python3 -m disaggregation.launch_harness pd_migration \
  --config "$CFG"
# --keep-up leaves runner/endpoints up after the verdict for a manual look; --run-id <name> names the run dir
```

Steps in order: `preclean_pd_migration` (sweeps processes, `/dev/shm` and the two handshake files on
both hosts) → `migration_endpoints` (ep1 on prefill, ep0 on decode, one PRRTE DVM) → `prefill_runner`
→ `decode_driver` (registers destination slots, blocks on the DONE sentinel) → `prefill_producer`
(the step is *named* that; it launches `migration_driver`: prefill, migrate, write handoff + DONE) →
`pd_migration_complete`.

Logs: `/data/philei/disagg_runs/<timestamp>/<step>.log`, plus the generated
`prefill_topology.yaml` and `decode_manifest.yaml` there. If `prefill_runner` times out in
`wait_ready`, look at `migration_endpoints.log` for `FAILED_TO_MAP` before anything else.

### 3.3 Reading the result

`pd_migration_complete.log`, a pass:

```
[pd-migration] CHECK 1 (prefill src == golden), producer over UMD:   ... kv_cache_pcc_complete ...
[pd-migration] CHECK 2 (prefill src == decode dst), transport fidelity:   (none)        <- expected for M3
[pd-migration] CHECK 3 (decode dst == golden), KV correctness:
  [kv-golden] mesh1/layer0 k_h0 slot0 layer0 head0 [0,10239): PASS pcc=0.999...
  [kv-golden] mesh2/layer3 index_k slot0 layer3 head0 [0,10239): PASS pcc=0.999759
[pd-migration] stage verdicts (4/4 stages reported): ...
[pd-migration] PASS: all 4 decode stages validated
```

Two decoder stages report; embed and lm-head `SKIP`. Every stage `SKIP` is a **failure** the harness
gates on — it means `decode.migration_layers` disagrees with what the ring owns. Read the truth off
the decode log:

```bash
grep "gathered layer_id->mesh_id" /data/philei/disagg_runs/*/decode_driver.log
# [migration] gathered layer_id->mesh_id=[(0, 1), (3, 2)]   -> migration_layers: "0,3"
```

### 3.4 Config facts worth knowing before editing the yaml

- `decode.migration_layers: "0,3"` is mandatory. The decode table has 4 rows; migrating `[0, 60)`
  aborts in the sender on the layer-4 lookup. The harness passes it as `PREFILL_MIGRATION_LAYERS`.
- `prefill.num_users: 2` is deliberate: the harness derives decode `n-slots = num_users + 1`, and
  the teardown dummy token needs a slot no validated pair names. `num_users: 1` → `n-slots: 2`, which
  the decode driver now rejects at startup.
- `prefill.max_seq_len` is injected as the decode side's `migration-validate-positions`. Do not set
  that flag by hand; a mismatch false-FAILs a correct migration.
- `engine.shared` (`/data/philei/tmp/m3_pd`) must be visible from both hosts; only the DONE sentinel
  and the handoff JSON cross it, KV never touches disk.
- `engine.transport: tcp`. RoCE (`roce`) needs `setup_roce_mpi.sh` first and buys bandwidth, not
  coverage.
- `decode.flags.weights: synthetic`: decode weights never enter the verdict, the migration overwrites
  the KV and CHECK 3 compares against the prefill golden.
- The prefill binding the yaml points at is the §1 one; the harness overrides its queue, endpoint,
  `PREFILL_NUM_USERS`, `PREFILL_MIGRATION_TABLE_PATH` and `PREFILL_MIGRATION_CLIENT_DIR` entries and
  writes the merged file to `<run_dir>/prefill_topology.yaml`.

---

## 4. Prefill → decode through the KV Manager (tt-d-gen, `kv_manager: "kvm"`)

The same two model runners as §3, but served:
- a Dynamo frontend and the tt-d-gen `tt_dynamo` prefill and decode workers;
- KV moves through a **KV Manager (KVM) fleet**, one `kv-manager` docker container per galaxy, instead
  of the migration-layer endpoints.

The workers route every prompt of at least `min_disagg_tokens` (1024) to prefill and hand the KV to
decode; shorter prompts stay on decode. Validated 2026-09-29: prefill bh-glx-120-b08u08 → decode
bh-glx-120-b09u02. The same driver without `--kvm` runs the legacy migration-layer transport.

### 4.1 What differs from §3

- **No endpoints and no `/dev/shm` command queues.**
  - Each runner writes its KV chunk table and device map to files at startup.
  - Each KVM loads both tables and pairs the prefill and decode rows **by config name**. The prefill
    table must therefore use blaze's names, `"00"`..`"08"` (see §1).
  - The KVM rejects a (layer, config) row that only one side publishes:
    `layer=0 config=8 is covered by only one side (source=1 destination=0)`. That is why the prefill
    table publishes index_k on the MSA layers only, like blaze.
- **Where the tables go.** Every KVM in the fleet reads the tables, so they must be on shared storage.
  The device maps hold host-local ASIC ids, so they stay in `/tmp`.

  | | prefill runner (binding `global_env`) | blaze decode | fleet conf |
  |---|---|---|---|
  | enable | `PREFILL_ENABLE_MIGRATION=1`, `PREFILL_MIGRATION_EXPORT_TO_FILE=1` | env `TT_MIGRATION_EXPORT_TO_FILE=1` | |
  | table | `PREFILL_MIGRATION_TABLE_PATH` | `--migration-table-path` | `PREFILL_TABLE` / `DECODE_TABLE` |
  | device map | `PREFILL_MIGRATION_DEVICE_MAP_PATH` | `--migration-device-map-path` | `PREFILL_DEVICE_MAP` / `DECODE_DEVICE_MAP` |

  The container writes into the table directory, so it must be world-writable (`chmod 777`).
- **Workers.** Both worker configs set `kv_manager: "kvm"`, and `kv_endpoint` points at the
  *prefill-side* KVM's ZMQ endpoint (`tcp://<prefill-ip>:9093`); only the prefill leader arms ZMQ.
  The `kv_cmd/table/resp_queue` keys configure the legacy client and are rejected under `kvm`.

### 4.2 Trees and one-time setup

| Tree | Where (validated run) | Notes |
|---|---|---|
| tt-d-gen | `/data/philei/tmp/dgen-m3`, branch `philei/minimax-m3-disagg` | See below. |
| tt-blaze | `/data/philei/tmp/blaze-pre4502` (148bf3af6a + #4650) | Linked at `$DGEN/third_party/tt-blaze`. Blaze main is not yet tested under `kvm`. |
| prefill tt-metal | this branch | Built per §0.1 a). |

The tt-d-gen branch carries:
- the worker configs `models/minimax-m3/dynamo.disagg.kvm.{prefill,decode}.json`;
- the runner binding `engine/tools/manifests/minimax_m3/runner_1rank_kvm.yaml`.

Everything below lives in `/data/philei/scripts/m3_pd/dgen`, outside the repos. `env.sh` there sets
the tree paths. It also keeps uv's interpreters and caches on `/data`, because home is node-local on
the galaxies.

```bash
S=/data/philei/scripts/m3_pd/dgen
source $S/env.sh

# tt-d-gen: libzmq from source + the Dynamo adapter venv, then the blaze-enabled engine module (compute node)
bash $S/build_deps.sh
bash $S/build_blaze.sh      # a later `build_dgen.sh --bindings` replaces the device module with a CPU build

# KVM image, on a galaxy host with docker
cd $DGEN && ./kv_manager/scripts/build_kv_manager_image.sh --image kv-manager:m3-local
```

- **A local image** is enough. `launch_kvm_fleet.sh up` ships it to the other fleet hosts through its
  `kvm-registry` container, pulled over `ssh -R localhost:5001`.
- **A GHCR image** (`ghcr.io/tenstorrent/tt-d-gen/kv-manager:kvm-<sha>`) needs `IMAGE_PULL_USER` and
  `IMAGE_PULL_TOKEN` with `read:packages`.
- **Per host:** `docker info` must work as your user on both hosts, and the node→node ssh from §0.2
  must be in place. The driver uses `/data/philei/.mig_ssh/id_ed25519`.

### 4.3 Run

`run_real_pd.sh` runs from the **login node, inside tmux**. It takes two RUNNING Slurm jobs of yours
holding the two galaxies. It only reads them (`squeue`) and reaches the hosts over ssh, not `srun`:
a process started in an `srun` step dies with the step.

```bash
export KVM_IMAGE=kv-manager:m3-local PREFILL_HOST=bh-glx-120-b08u08 DECODE_HOST=bh-glx-120-b09u02
S=/data/philei/scripts/m3_pd/dgen

# Configs, cross-config invariants, ports and docker on both hosts; starts nothing.
$S/run_real_pd.sh <prefill-job> <decode-job> --kvm --golden --dry-run

$S/run_real_pd.sh <prefill-job> <decode-job> --kvm --golden
```

Steps in order:
1. `glx_reset` + 90 s on both hosts (`--no-reset` skips it).
2. blaze 4-stage ring, launch-only (~9 min).
3. Prefill runner (~3 min).
4. KVM fleet: `plan`, `up`, `health`. It needs both runners' tables first.
5. etcd + frontend on the decode host.
6. Decode worker, then prefill worker.
7. The three requests.
8. **Hold until Enter**, then teardown in reverse.

`--golden` sends `longbook_5120`'s token ids as the third request. At teardown blaze then PCCs the
migrated KV against that golden (up to 30 min).

A device-free rehearsal of the script runs on any one node: `--mock` uses mock pipelines and a
`mock_kvm_server` per side, with `PREFILL_HOST=DECODE_HOST=<cpu node>` and `NIC`/`DECODE_NIC` set to
two different interfaces.

Ports it checks and holds:

| Host | Ports |
|---|---|
| decode | 8000 frontend, 12379/12380 etcd, 20020 worker health |
| prefill | 20021 worker health; 9093 KVM ZMQ, 18650 KVM control, 18081 KVM health |
| both | 19071 prefill↔decode rendezvous |

### 4.4 Reading the result

A pass, in the script's output:

```
[PASS] long_fact_prompt_migrates: HTTP 200 ..., prompt_tokens=3072 migrated_tokens=3008 prefill_admitted=1 decode_prefilled=64
[PASS] short_prompt_stays_local: HTTP 200 ..., prompt_tokens=182 migrated_tokens=0 prefill_admitted=0     <- < min_disagg_tokens
[PASS] golden_token_ids_migrate: HTTP 200 ..., prompt_tokens=5120 migrated_tokens=4928 prefill_admitted=1
[INFO] migration_id=...: all children completed, overall successful=true                                  <- kvm-prefill-0 log
REQUESTS PASS
```

The decode worker prefills the 64-token block holding the last prompt token itself, so migration stops
short of the prompt end. The golden check covers `[0, 5056)`.

After Enter (with `--golden`), in `<run>/blaze.log`:

```bash
grep -aE '\[kv-golden\]' <run>/blaze.log
# [kv-golden] mesh1/layer0 k_h1 slot1 layer0 head0 [0,5056): PASS pcc=0.999916      <- slot 1 = the golden request
# [kv-golden] mesh1/layer0 k_h1 slot0 layer0 head0 [0,5056): FAIL pcc=0.001353      <- expected, see below
```

Only slot 1 received the golden tokens; slot 0 holds the fact prompt. `--migration-validate-golden-pt`
validates both slots against `longbook_5120`, so slot 0's `FAIL` at PCC ≈ 0 is expected. **Judge the
run by slot 1.**

KVM-side detail: `docker logs kvm-prefill-0` on the prefill host. `grep -ac 'successful=false'` must
print 0.

### 4.5 Teardown and gotchas

- If the script dies without its trap (kill -9, lost tmux), the remote processes keep running. From
  the login node:

  ```bash
  K="-i /data/philei/.mig_ssh/id_ed25519"
  ssh $K <prefill-host> bash $S/pd_remote.sh <run> kvm_fleet <run>/fleet.conf down
  for h in <prefill-host> <decode-host>; do ssh $K $h FORCE_CLEAN=1 bash $S/pd_remote.sh <run> preclean; done
  ```

  `preclean` refuses while any of your runner, worker or endpoint processes live, unless
  `FORCE_CLEAN=1`.
- Teardown stops the decode ring mid-run, which wedges its Ethernet cores. Keep the reset on for the
  next bring-up.
- The validated run set `KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES=2080374784` on both workers and runners,
  for the legacy transport's older importer (§3.1). It has not been re-tested under `kvm` without it.
- `rendezvous :19071 connections ... 2` means both workers resolved their peer. Fewer means one side
  fell back to local prefill, and `migrated_tokens` stays 0.
