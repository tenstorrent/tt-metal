# Migration engine setup and process operations (single Galaxy)

How to get `tt-llm-engine`'s migration layer built and the three-process loopback running, plus how
to take it down without wedging the box. Concepts, layer axis and PCC gates live in
[PREFILL_MIGRATION_SC1_RUNBOOK.md](PREFILL_MIGRATION_SC1_RUNBOOK.md); this file is the mechanics.

Verified on `bh-glx-110-c10u20` (sc1) against tt-llm-engine `e79c4a07`, tt-metal submodule pin
`e49ac055d61`.

## 1. Two trees, on purpose

| tree | used by | why separate |
|---|---|---|
| your tt-metal checkout | runner, driver | the model, the manifests, `python_env` |
| `tt-llm-engine/tt-metal` (submodule, pinned) | endpoint, workers | the worker links UMD from *this* pin |

Do not point the engine at your own tt-metal build. The worker deliberately does **not** link
`libtt_metal.so` (ULFM vs stock-OpenMPI ABI conflict) and instead compiles
`tt_metal/llrt/metal_soc_descriptor.cpp` directly against the pinned tree's headers. Mixing trees
gives you link errors or, worse, a silent ABI mismatch.

```
git clone <tt-llm-engine> /data/<user>/tt-llm-engine
cd /data/<user>/tt-llm-engine
git submodule update --init --recursive     # pulls tt-metal at the pin; takes a while
```

## 2. The migration layer does not build out of the box

At this pin, `disaggregation/migration/CMakeLists.txt` needs two fixes before it links.

**a. Stale UMD library paths.** UMD commit `6d8cc51f` (2026-06-03, *predates the pin*) folded
`libdevice.so` into `libtt-umd.so` and moved it from `build/lib` to
`build/tt_metal/third_party/umd/lib`. `find_library` misses are swallowed by `if(${LIB_VAR})`, so
you get an undefined-`tt::umd::*` **link** error rather than a configure error. Fix by giving
`_W_UMD_TT_UMD_LIB` both paths (new first, old as fallback so older pins still resolve). The
`BUILD_DEVICE_TESTS` block has a duplicate loop with unprefixed variable names — patch it too.

**b. yaml-cpp is never linked.** Only the *include* dir was globbed, but
`metal_soc_descriptor.cpp` parses the per-arch SOC YAML and needs the archive. Add a `_W_YAML_LIB`
entry pointing at `${TT_METAL_BUILD_DIR}/_deps/yaml-cpp-build`, and put it **last in the loop** — a
static archive has to trail the objects that reference it.

Build:

```
cd /data/<user>/tt-llm-engine && ./build_migration_layer.sh --build-type RelWithDebInfo --jobs 32
```

Script gotchas:
- `--targets` accepts only the `--targets=value` form. `--build-type` and `--jobs` take either form.
- It builds tt-metal with `|| true`, so a **metal build failure is swallowed** and you only find out
  at the migration-layer link step. Read the log, don't trust the exit code.
- There is no passthrough for extra `-D` args, which is why patching the CMakeLists is the right
  route rather than seeding cache vars.

Artifacts land in `disaggregation/migration/build_RelWithDebInfo/bin/{migration_endpoint,migration_worker}`
and the Python client in `.../build_RelWithDebInfo/python`.

## 3. Launch order is forced: endpoint -> runner -> driver

Forced by shm semantics, not preference. The endpoint **creates** the queues (`O_CREAT`); the runner
publishes the KV chunk table into them; the driver attaches to both and issues the migrations. Start
them out of order and the later process attaches to queues that do not exist yet.

**Endpoint** — owns its prte DVM (self-spawned, no ssh) and exactly two workers:
A = sender id `1`, B = receiver id `0x7FFF0000`. The pair is hardcoded in `run_loopback()`;
`--num-subordinates` scales a different axis and will not give you more.

```
cd /data/<user>/tt-llm-engine/disaggregation/migration; export TT_METAL_HOME=/data/<user>/tt-llm-engine/tt-metal; ./build_RelWithDebInfo/bin/migration_endpoint --endpoint-id 1 --cmd-queue /mig_ep1_cmd --table-queue /mig_ep1_table --response-queue /mig_ep1_resp --worker-bin ./build_RelWithDebInfo/bin/migration_worker
```

`TT_METAL_HOME` is not optional: without it the SOC descriptor lookup fails with a bare
`YAML::BadFile` and no indication of what it was looking for.

**Runner** — owns the 32 chips, the KV cache and the H2D server. Builds the table at rank 0,
publishes it, then waits up to 3600 x 5 s for `WORKER_READY`.

```
cd /data/<user>/tt-metal; source python_env/bin/activate; ./models/demos/common/prefill/runners/run_pipeline_prefill.sh models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_migration_1rank_kimi27_dflash.yaml <host>:1 <iface>
```

**Driver** — a plain single process, no `tt-run`, so shell exports reach it directly.

```
cd /data/<user>/tt-metal; source python_env/bin/activate; export TT_METAL_HOME=/data/<user>/tt-metal; export PYTHONPATH=/data/<user>/tt-metal; export LD_LIBRARY_PATH=/data/<user>/tt-metal/build/lib; ./models/demos/common/prefill/runners/run_migration_driver.sh models/demos/common/prefill/runners/producer_manifests/producer_manifest_migration_loopback_kimi27_dflash.yaml "" "" --verify-migration=both
```

`run_migration_driver.sh` args are **positional** (`MANIFEST HOST_LIST TCP_IFACE`), so passthrough
flags need the two empty positionals first or your first flag is eaten as `HOST_LIST`.

The driver needs the venv explicitly. A bare `/usr/bin/python3` imports `ttnn` as a namespace package
(`__file__ is None`) and dies much later, inside `read_dram_umd`.

## 4. One endpoint per runner — the constraint that bites

**A worker accepts exactly one `SET_TABLE` for its lifetime.**
`ControlThread::handle_set_table` guards on `table_initialized`, logs
`[ctrl <id>] second SET_TABLE — fatal` and calls `std::terminate()` — a deliberate fail-fast, not an
escaped exception, so both workers die with `terminate called without an active exception` /
`FATAL signal=6`. The endpoint process stays alive and `prun` goes zombie, so **the next driver hangs
forever in `wait_complete` with no error of its own**: the fatal never reaches the client. Whenever a
migration stalls with a silent driver, `grep 'second SET_TABLE' endpoint.log` first.

A runner killed *before* it publishes sends no `SET_TABLE`, which is why the endpoint looks reusable
right up until the first restart that actually reaches `WORKER_READY`.

Consequences worth planning around:
- Restarting the **runner** means restarting the **endpoint** too.
- Re-running the **driver** against a live runner is free — seconds to a few minutes. Keep
  `PREFILL_SEND_SHUTDOWN: "0"` in the manifest so the runner stays serving, and a config or gate
  mistake costs one driver pass instead of another multi-hour weight load.

## 5. Shutdown and recovery

Order matters: driver, then runner tree, then endpoint, then prte, then the shm rings.

```
pkill -f migration_driver; pkill -f run_pipeline_prefill; pkill -f prefill_runner; pkill -f migration_endpoint; pkill -f migration_worker; pkill -f prte
rm -f /dev/shm/mig_ep1_* /dev/shm/ep_1_* /dev/shm/ttmig.*
```

Those are the 9 migration rings (`mig_ep1_{cmd,table,resp}`, `ep_1_{a,b}_{cmd,table,resp}`) and their
`ttmig.*.lock` files.

**Never delete `/dev/shm/TT_UMD_LOCK.*`.** If a UMD `CHIP_IN_USE` lock names a PID that does not
exist, that does **not** mean the lock is stale — it means a **live container** holds the chips.
Check `docker ps`. Deleting the lock lets two processes drive the same chips.

For a wedged device use `tt-smi -glx_reset_auto`. Do **not** use `tt-smi -r` on this Galaxy — it is
insufficient below CPLD v1.16 and leaves the device enumerable by tt-smi but unusable by ttnn
("Query mappings" / stale sysmem). Never reflash firmware; that is syseng-only.

## 6. Health checks

```
ps -o pid=,etime=,comm= -p <endpoint> -p <workerA> -p <workerB> -p <runner>
grep -c FATAL prefill_migration_logs/<mode>/endpoint.log      # expect 0
ls -1 /dev/shm | grep -cE 'mig_ep1|ep_1_'                     # expect 9
awk '/^read_bytes/{print $2}' /proc/<rank0-python-pid>/io     # weight-load progress, sample 5 s apart
```

## 7. Upstream bugs hit during bring-up

1. `CMakeLists.txt` stale UMD paths (§2a) — link error, not a configure error.
2. `CMakeLists.txt` yaml-cpp not linked (§2b).
3. `launch_migration_endpoints.sh:442` passes `--resp-queue`; `endpoint_main.cpp:61` accepts only
   `--response-queue`.
4. `docs/launch.md` documents a `--loopback` flag that does not exist. Loopback is unconditional at
   this pin — `EndpointOrchestrator::run()` always calls `run_loopback()`. Loopback vs cross-endpoint
   is a *protocol usage* distinction (`dest_endpoint_id == own id` routes to worker B), not a flag.
5. Missing `TT_METAL_HOME` surfaces as a bare `YAML::BadFile` with no path in the message.
6. A second `SET_TABLE` aborts both workers instead of being rejected, and the failure is only
   visible in `endpoint.log` (§4).
