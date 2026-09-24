# Prefill KV-migration loopback on one Galaxy (sc1) — how it works

Gate 2 of `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`, run on
`bh-glx-110-c10u20` with Kimi K2.7 + DFlash.

Gate 1 is a mock: the table is synthetic and nothing crosses a wire. **Gate 2 is the first
configuration where real KV leaves DRAM, crosses the transport, and lands back in DRAM**, and where
the destination is read back and scored. Everything below exists to make that one sentence true and
checkable.

---

## 1. Three processes, and why the order is forced

| # | process | owns | tree |
|---|---|---|---|
| 1 | `migration_endpoint` | the shm queues, a self-spawned prte DVM, exactly 2 workers | `tt-llm-engine/tt-metal` |
| 2 | `prefill_runner` (under `tt-run`) | the 32 chips, the KV cache, the H2D server, the chunk table | `/data/nmilicevic/tt-metal` |
| 3 | `migration_driver` | the prompt feed, the migrate commands, all verification | `/data/nmilicevic/tt-metal` |

**The order is not a convention, it is shm semantics.** The endpoint creates its queues with
`O_CREAT`; the runner and driver attach `O_RDWR` only. Start a client first and it fails on a queue
that does not exist yet.

The endpoint launches a hardcoded pair, not a scalable set:

- **worker A** — sender, endpoint id `1`, `--ep-app-color 0`, queues `/ep_1_a_*`
- **worker B** — receiver, endpoint id `0x7FFF0000` = 2147418112 (`kWorkerLoopbackReceiverId`),
  `--ep-app-color 1`, queues `/ep_1_b_*`

`run_loopback()` calls `launch_worker_pair()` unconditionally. `--num-subordinates` scales a
different axis and will not give you more workers. There is no `--loopback` flag — the docs are
wrong; loopback is unconditional at this engine commit.

### One endpoint per runner — this is the constraint that bites

**A worker accepts exactly ONE `SET_TABLE` for its lifetime.** `ControlThread::handle_set_table`
throws on the second, and nothing catches it:

```
[ep 1] SET_TABLE -> both workers
[ctrl 2147418112] second SET_TABLE — fatal
[ctrl 1]          second SET_TABLE — fatal
terminate called without an active exception
[migration_worker pid=...] FATAL signal=6
```

Both workers die. The endpoint process stays alive, `prun` goes zombie, and the **next driver hangs
forever in `wait_complete` polling a dead pair** — with no error of its own. The only symptom on the
driver side is a migration that never completes. Always read
`prefill_migration_logs/<mode>/endpoint.log` when a migration stalls.

So:

| action | needs a fresh endpoint? |
|---|---|
| re-run the driver against a live runner | **no** — free, seconds to minutes |
| restart the runner so it reaches WORKER_READY again | **yes** — kill and relaunch the endpoint first |
| runner killed before it published a table | no — no SET_TABLE was sent |

That last row is how this constraint hides: a runner killed mid-weight-load never publishes, so the
endpoint looks reusable until the first restart that actually reaches publish.

**Recovery, in order** (all three must go; a half-restart leaves stale shm cursors):

```
kill -TERM <driver> <runner tree> <endpoint> <prte>     # enumerate with ps -eo pid,ppid,etime,args
rm -f /dev/shm/mig_ep1_* /dev/shm/ep_1_* /dev/shm/ttmig.*
./scratchpad/run_endpoint.sh <mode>                      # wait for "channel up (Phase 1)"
./scratchpad/run_runner.sh <mode>                        # wait for WORKER_READY
./scratchpad/run_driver.sh <mode>
```

Those `/dev/shm` entries are the migration rings, **not** the UMD lock. Remove them only once every
owning process is confirmed dead. Never delete a UMD lock file — a lock whose holder PID does not
exist means a live container holds the chips.

### Two trees on purpose

The engine pins its own tt-metal (`e49ac055d61`). Your working tree is `/data/nmilicevic/tt-metal`.
They are deliberately decoupled: the endpoint and workers build against the pin, the runner and
driver run your branch. Launching a worker without `TT_METAL_HOME=/data/nmilicevic/tt-llm-engine/tt-metal`
kills both workers with an unhelpful `YAML::BadFile`.

---

## 2. What loopback actually means

`dest_endpoint_id == src_endpoint_id`, so src and dst share one table and one set of chips. The
destination slot is just an offset:

```
dst_slot = src_slot + dst_slot_offset      # offset defaults to the producer's num_users
```

With `num_users: 4` that is `0->4, 1->5, 2->6, 3->7`, which is why **`PREFILL_NUM_USERS` on the
runner must be 2x the driver's `num_users`** — the runner has to allocate 8 slots for a 4-pair run.
That doubling is a loopback artifact and nothing else.

Because all slots get the same prompt unless `PREFILL_PRODUCER_SLOT_TRACES` is set, and only one
Kimi K2.7 golden exists, verification catches a wrong layer or a wrong offset but **cannot catch
data landing in the wrong destination slot**. Known blind spot.

---

## 3. The layer axis — the part that is easy to get wrong

Without DFlash the table has one config: main KV, 61 rows, row index == global layer id. Simple.

**With DFlash the table has 17 configs:**

| configs | what | rows | head_dim | chunk_size_bytes |
|---|---|---|---|---|
| `0` | main KV (verifier) | 61 | 576 | 19584 |
| `dflash_k_h00..h07` | drafter K, 8 KV heads | **67** | 128 | 4352 |
| `dflash_v_h00..h07` | drafter V, 8 KV heads | **67** | 128 | 4352 |

A drafter config's `num_layers` is the **global** axis: verifier depth + 6 draft layers. But the
drafter only populates the **tail** — rows 61..66 at full depth, rows 4..9 in a 4-layer smoke.
Rows below that have `noc_addr == 0`. The drafter taps model layers `(1, 12, 24, 35, 47, 58)` and
`row first+i` maps to `draft layer i`.

So the caches are *ragged*: config 0 is a 61-row prefix of a 67-row axis, and the drafter configs
are a 6-row suffix of it.

### The engine was built for this; the driver was not

`dcn_sender_backend.cpp` handles ragged perfectly:

- `:247` — `if (layer >= config(config_id).num_layers) continue;` — config 0 is skipped for layers 61-66.
- `:413` — if every destination row for a (config, layer) is unpopulated, skip rather than write NOC
  address 0.

The driver capped itself at the **verifier** depth in three places, all keyed to
`producer.NUM_LAYERS`:

1. `_issue` migrated `[0, 61)` — drafter rows never crossed the wire.
2. `_cache_plan` mapped `range(num_layers)` = rows 0..60 for *every* config — for a drafter config
   that is exactly the 61 empty rows and none of the 6 that hold data.
3. `_drain_layer_acks` waited for `61 * pushes` while the runtime emits `67 * pushes`, so it
   returned **before the drafter had written**.

Fixed by taking the axis from the table instead: `_global_layer_axis(table)` = `max(cfg.num_layers)`
over all configs. 67 at full depth, 10 in a 4-layer smoke, and **61 = unchanged** when DFlash is off.

### Why the acks exist at all

PR **#56877** (`95a3f01a17a`, 2026-09-22) added `TtPrefillRuntime.layer_ack_layers()`, which widens
the global ack count by the drafter's layer count, and wired `on_layer_complete` + `layer_ack_base`
into `drafter.forward`. Only the KV-tail rank writes those layers, so only its *local* count grows;
every rank must agree on the *global* count because the master router's reorder buffer keys on
`chunk * global + layer` and demands a dense sequence.

> **Do not trust the comment at `tt_prefill_runtime.py:1015-1023`.** It says real migration must not
> copy drafter KV yet. That is from `6b35ca8d35e` (#52413) and #56877 left it stale. It cost me a
> wrong answer. Report it.

---

## 4. The four checks and what each one actually proves

| check | knob | proves | does not prove |
|---|---|---|---|
| source KV PCC | `check_pcc: true` | the model computed correct KV | nothing about migration |
| drafter KV PCC | same, needs `variant: kimi_k2_7` | the drafter computed correct context K/V | nothing about migration |
| `dst-bytes` | `--verify-migration=dst-bytes` | dst is byte-identical to src, **golden-free** | that src was right |
| `dst-golden` | `--verify-migration=dst-golden` | dst matches the golden trace | — |

`--verify-migration=both` runs the last two. Use it.

Two silent-pass traps, both now fixed:

- **Unpopulated rows read as verified.** With no `noc_addr == 0` guard, the byte check read DRAM
  address 0 on both sides, got identical bytes, and counted them. One smoke reported 478720
  "byte-identical chunks" of which only 28160 were real. It also defeated that function's own
  `if not checked: FAILURE` guard.
- **`variant:` silently disables the drafter check.** `prefill_producer.py:48` does
  `sd("PREFILL_MODEL", model.get("variant"))`. The committed manifests say `variant: deepseek_v3_d_p`
  — a **valid but wrong** adapter with `supports_dflash=False` and an empty `dflash_golden_default`.
  The producer logs "the drafter half is NOT checked" and passes. Must be `variant: kimi_k2_7`.
  Still latent in the two non-dflash manifests (harmless there, still a bug).

A truncated run cannot score the drafter: 4 layers reach only target layer 1, so drafter-K decays
monotonically with target depth (0.839 at target 1 down to 0.043 at target 47). That decay is the
truncation signature and also proves the row mapping is right. Set `PREFILL_DFLASH_PCC: "0.0"` in a
smoke manifest to record without gating; the gate only means something at 61 layers.

### The drafter gate is 0.85, not the 0.88 module default

`prefill_producer.py` defaults `PREFILL_DFLASH_PCC` to 0.88. **That number is wrong for K2.7 at full
depth** — `runners/ci/run_multirank_dflash_pcc.sh:48` gates at 0.85, and says so directly: 0.88 "is a
K2.6 bring-up value that does not hold for K2.7 at full depth; override with PREFILL_DFLASH_PCC rather
than editing the module default."

The invariant: **the drafter gate tracks the verifier's gate and never sits above it.** Drafter V is
unnormalized and carries the verifier's own error, so a tighter drafter gate fires when the *verifier*
merely sits near its floor rather than when the drafter regresses. K is normalized by RMSNorm+RoPE and
clears both by a wide margin — so in practice this is a V gate.

Leaving it unset in a 61-layer manifest is a config bug and costs a full run. Measured at 61 layers:
verifier 0.885879, drafter-K 0.9545, drafter-V **0.871930** — rc=1 against 0.88, passing against 0.85.
The verifier sat only 0.036 above its own floor, exactly the condition the CI comment describes.

**A failing source-PCC gate is not a migration failure.** The source check runs before the destination
is verified, so rc=1 can sit alongside a byte-perfect migration. Read *which* check raised before
blaming the transport.

---

## 5. Running it

Two stages, because the weight load is the entire cost.

The Kimi K2.7 ttnn cache is 546 GB of pre-converted `.tensorbin` on NFS, ~9.1 GB/layer. Host RAM is
566 GB, so the working set does not fit in page cache and prewarming cannot help at full depth.
**The only real lever is `PREFILL_NUM_LAYERS`,** and load time is roughly linear in it.

`PREFILL_SEND_SHUTDOWN: "0"` keeps the runner serving after the driver exits, so **driver iterations
after the first load cost minutes, not another load.** Only a runner-side crash costs a reload.

### Stage A — smoke, ~4 min load

```
./scratchpad/run_endpoint.sh dflash_smoke4  # wait for "channel up (Phase 1)"
./scratchpad/run_runner.sh dflash_smoke4    # wait for WORKER_READY
./scratchpad/run_driver.sh dflash_smoke4    # LAYERS empty => verify every row
```

### Stage B — full depth

```
./scratchpad/run_endpoint.sh dflash         # FRESH endpoint, see section 1
./scratchpad/run_runner.sh dflash           # wait for WORKER_READY, 1.2-3.7 h (see "How long the load takes")
./scratchpad/run_driver.sh dflash           # LAYERS empty => verify every row, ~10 min
```

Or the whole thing as one command: `./scratchpad/run_full_chain.sh` tears the previous stack down,
rotates the logs, and runs endpoint -> runner -> driver in order, failing fast with a `FAIL` line if
any stage does not reach its marker.

Re-running the driver against that same live runner is free. Restarting the runner is not — go back
to the endpoint.

Logs land in `/data/nmilicevic/tt-metal/prefill_migration_logs/<mode>/{runner,driver}.log` with a
`.rc` beside each.

### How long the load takes

**Measure it; do not quote a number from last time.** The weight load is NFS-bandwidth-bound — 546 GB
of `.tensorbin` over `10.32.13.1:/models`, strictly per-layer at ~9.1 GB/layer. The DellEMC SDNAS
server is shared, so the rate moves with whoever else is reading it:

| observed | per layer | 61 layers |
|---|---|---|
| 123 MB/s | ~74 s | ~1.2 h |
| 48 MB/s | ~220 s | ~3.7 h |

Live rate, from the rank-0 python process:

```
p=$(ps -eo pid,args | awk '/prefill_runner/ && !/awk/ {print $1}' | tail -1); a=$(awk '/^read_bytes/{print $2}' /proc/$p/io); sleep 5; b=$(awk '/^read_bytes/{print $2}' /proc/$p/io); echo $(( (b-a)/5/1048576 )) MB/s
```

**The first layers lie.** They are page-cache hits left by the previous run: a run that died at layer
6 leaves 0..6 cached, so the next run does them at ~7 s each and only goes cold at layer 7. Averaging
from layer 0 understates the ETA badly. **Take the rate from the slow tail only** — per-layer deltas
above 60 s:

```
grep -aoE 'Building layer [0-9]+/[0-9]+' prefill_migration_logs/dflash/runner.log | tail -3
```

This is not the dense-vs-MoE split — the boundary moves with wherever the last run stopped.

The chain's runner wait is capped at 3600 x 5 s = 5 h, which covers the slow case.

### Sizing the byte check

`chunk_n_tokens` is 32, so a 56320-token slot is 1760 chunks per (pair, config, row).

**Checking every row at full depth is affordable — do not narrow it by default.** The naive count
treats each drafter config as carrying all 67 rows, which would be 1.9M chunks. It does not: a
drafter config populates only its 6-row tail, and the unpopulated-row guard skips the rest before
issuing any read. So the drafter's share of the work is **flat in model depth** — 6 rows whether
they sit at 4-9 or 61-66 — and only config 0 scales:

| | chunks compared at 61 layers |
|---|---|
| config 0, 61 rows x 1760 x 4 pairs | 429,440 |
| drafter, 6 rows x 1760 x 16 configs x 4 pairs | 675,840 |
| **total** | **1,105,280** |

Measured at smoke depth: config 0 runs ~7.6 s per 4 rows per pair, each drafter config ~2.5 s per
pair. Extrapolated that is ~156 s per pair at full depth, ~10 min for all four. Cheap enough that
`--verify-migration-layers` is for narrowing a *re-run* while chasing a specific failure, not for
the first pass.

Whenever you do narrow it, take verifier rows plus at least one drafter row (`0,30,60,61,66`) — a
subset PASS is a **sample**, not a proof, and the driver says so.

---

## 6. Gotchas that cost real time

- **`run_migration_driver.sh` args are positional** (`MANIFEST HOST_LIST TCP_IFACE`). Passthrough
  flags need **two empty positionals** first or the first flag is eaten as HOST_LIST.
- **`PREFILL_VERIFY_MIGRATION` in the manifest can never change the argparse default** — manifest env
  is applied after `parse_args()`. Pass `--verify-migration` on the CLI or you silently get
  `dst-bytes`.
- **`tt-run` forwards only `TT_`/`ARCH_`/`WH_`/`TTNN_`/`DEEPSEEK_`/`MESH_` prefixes.** A shell-exported
  `PREFILL_*` never reaches the runner; it must live in the rank-binding YAML's `global_env`. The
  model manifest JSON applies with `os.environ.setdefault`, so **`global_env` wins over the manifest**.
- **`run_pipeline_prefill.sh` defaults to the wrong hosts** (`bh-glx-d03u02:1,bh-glx-d03u08:1`). Pass
  `bh-glx-110-c10u20:1 ens5f0np0`.
- **The driver needs the venv explicitly**: `source python_env/bin/activate`, plus `TT_METAL_HOME`,
  `PYTHONPATH`, `LD_LIBRARY_PATH=.../build/lib`. Bare `/usr/bin/python3` imports `ttnn` as a namespace
  package (`__file__ is None`) and dies later inside `read_dram_umd`.
- **The producer's `model.num_layers` must equal the runner's `PREFILL_NUM_LAYERS`.** The PCC loop
  walks `range(NUM_LAYERS)` and a lookup on an unpublished layer resolves to a default address.
- **Pre-flight worth the entire load**: confirm `build/lib/_ttnn.so` is newer than the model source. A
  stale build fails at `runtime.compile()` *after* the full weight load and *before* any migration
  code runs.
- **`pgrep -f` matches itself.** Enumerate with `ps -eo pid,ppid,etime,args` and kill explicit PIDs.
- **A UMD `CHIP_IN_USE` lock whose holder PID does not exist means a live container holds the chips.**
  Check `docker ps`. Never delete the lock file.

---

## 7. Upstream bugs found

**tt-llm-engine** (`e79c4a07`, tt-metal pin `e49ac055d61`) — the migration layer does not build out
of the box:

1. `disaggregation/migration/CMakeLists.txt` — stale UMD library paths. `libdevice.so` and
   `libtt-umd.so` both moved in UMD `6d8cc51f`, which predates the pin. `find_library` misses are
   swallowed by `if(${LIB_VAR})`, so it degrades to an undefined-`tt::umd::*` **link** error rather
   than a configure error.
2. Same file — yaml-cpp never linked. `migration_worker` deliberately does not link `libtt_metal.so`
   (ULFM vs stock-OpenMPI ABI conflict) and compiles `metal_soc_descriptor.cpp` directly, which drags
   in yaml-cpp. Only the include dir was globbed.
3. `launch_migration_endpoints.sh:442` passes `--resp-queue`; the binary requires `--response-queue`.
4. `docs/launch.md` documents a `--loopback` flag that does not exist.
5. Launching a worker without `TT_METAL_HOME` set kills both workers with a bare `YAML::BadFile`.
6. A second `SET_TABLE` aborts the worker via `terminate called without an active exception` instead
   of returning an error to the endpoint. The endpoint survives, so the next client hangs in
   `wait_complete` with no diagnostic. Either make the table re-settable or fail the endpoint loudly.

Patched locally in `/data/nmilicevic/tt-llm-engine`, kept out of the main tree on purpose. The patch
uses path fallbacks so older pins still resolve — upstreamable as-is.

**tt-metal**: #56877 left the contradicting comment at `tt_prefill_runtime.py:1015-1023` in place.

---

## 8. Driver changes made here

`models/demos/common/prefill/runners/migration_driver.py`, uncommitted:

- new `_global_layer_axis(table)` — the migrate and ack axis comes from the table, not from
  `producer.NUM_LAYERS`.
- `main()` sets `driver.num_layers` from it after the table read; `_drain_layer_acks` uses it.
- `_cache_plan` maps `range(n_rows)` instead of `range(num_layers)`.
- `_verify_dst_vs_src_bytes` skips rows with `noc_addr == 0` on either side and warns with a count.

Net effect on the 4-layer dflash smoke: populated rows reachable by the byte check went from **4 to
100**. No behavior change when DFlash is off — verified against a non-dflash table: same axis, same
mapping.
