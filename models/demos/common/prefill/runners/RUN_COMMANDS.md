# 4-rank prefill runner launch commands

## Prerequisites

Per host, once:

- A clone at the **same commit** on all four hosts. Ranks exchange a chunk plan
  derived from the checked-out code; a mismatch surfaces later as a D2D chunk
  plan error, not as a version warning.
- `bash build_metal.sh` and `bash create_venv.sh` in that clone.
- The model's TTNN weight cache reachable at the adapter's default path, or
  `PREFILL_TTNN_CACHE` pointed elsewhere:
  - kimi27: `/mnt/weka/model-cache/scratch/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill`
  - glm53: `/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/GLM-5.3-Cache-prefill`
- Passwordless ssh between the hosts, both directions.

`HOSTS` order is the **pipeline order**, which must follow the physical galaxy
ring — rank N talks to rank N+1 over an inter-galaxy cable. Hostfile order is
not it. If you do not know the ring for your quad, get the connectivity dump
from a multi-host `run_cluster_validation --print-connectivity`; a single-host
run cannot see the exit cables.

## 0. Set up the shell

```bash
export TT_METAL_HOME=/path/to/tt-metal
export HOSTS=host0,host1,host2,host3
cd $TT_METAL_HOME && source python_env/bin/activate
```

## 1. Recover the cluster — before every launch

```bash
LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib \
  bash $TT_METAL_HOME/tools/scaleout/exabox/recover.sh --hosts $HOSTS --max-attempts 3
```

Not optional. Fabric ethernet cores are not always freed on teardown, and a
launch onto un-retrained links fails during bringup with an error that reads
like a model bug. Roughly one bringup in eight still fails after a successful
recovery; run it again.

Give `recover.sh` only `build/lib`. A ULFM MPI library on `LD_LIBRARY_PATH`
makes the system `mpirun` fail with `undefined symbol: pmix_value_load`, and
recovery reports "MPI interface cannot be used" instead of the real cause.

Note the `MPI interface: <name>` line it prints — that interface name is what
`--tcp-interface` below wants.

## 2. Purge stale shared memory — before every launch

```bash
for h in ${HOSTS//,/ }; do
  ssh -o BatchMode=yes $h 'rm -f /dev/shm/*prefill*; pkill -f prefill_runner; true'
done
```

Before, not after. The runner unlinks its own segments at startup, but only the
rank-scoped names it can prove it owns, and the unlink is existence-based rather
than staleness-based. A segment left by a `kill -9`'d run makes the next run
unlink a ring another rank is already using.

## 3. Launch the runner

```bash
python3 $TT_METAL_HOME/ttnn/ttnn/distributed/ttrun.py \
  --mesh-graph-descriptor $TT_METAL_HOME/models/demos/common/prefill/runners/topology_configuration/ci/sc4_mgd.textproto \
  --hosts $HOSTS \
  --tcp-interface ens5f0np0 \
  --force-rediscovery \
  --mpi-args "-x PATH --tag-output" \
  bash -lc "cd $TT_METAL_HOME; \
    export PYTHONPATH=$TT_METAL_HOME; \
    export PREFILL_MANIFEST=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi27.json; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner"
```

The mesh-graph descriptor is per-SKU, not per-model; `sc4_mgd.textproto` is shared
across models. For glm53, swap the manifest path to `glm53.json`, or use the
static binding in 3b.

The runner allocates, captures its trace, warms up, logs `setup complete,
entering request loop`, and then blocks there until requests arrive. That idle
is by design: no device op completes for as long as it waits.

Replace `ens5f0np0` with the interface `recover.sh` reported. `ttrun` writes its
phase-1 discovery cache to `./generated/ttrun` under the launch directory, so
run it somewhere writable.

`PREFILL_MANIFEST` must be exported inside the `bash -lc` string. Passed through
`--mpi-args -x` it reaches only the local rank, and the other ranks fall back to
defaults — which shows up much later as a D2D chunk plan mismatch.

## 3b. GLM-5.3 from a static rank binding

Deployments that pin hosts instead of running discovery launch from
`topology_configuration/glm53_4rank.yaml`. Rank N runs mesh N of `ci/sc4_mgd.textproto`,
the same descriptor CI uses.

Pick the variant by its manifest name:

| `GLM53_VARIANT` | KV cache | MTP | Users |
|---|---|---|---|
| `glm53` | scaled FP8 | none | 54 |
| `glm53_mtp_kv_share` | scaled FP8 | 1 level | 53 |
| `glm53_bf16` | BF16 | none | 34 |
| `glm53_mtp_kv_share_bf16` | BF16 | 1 level | 31 |

```bash
cd $TT_METAL_HOME && source python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
export GLM53_VARIANT=glm53
python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface ens5f0np0 \
  --rank-binding models/demos/common/prefill/runners/topology_configuration/glm53_4rank.yaml \
  --mpi-args "--host host0:1,host1:1,host2:1,host3:1 --map-by slot --bind-to none --tag-output -x PATH" \
  -- python3 -m models.demos.common.prefill.runners.prefill_runner
```

Run it on `host0`. The `--host` order is the pipeline order from the prerequisites:
rank 0 first, each next host one cable further along the ring.

The manifest plus the runner defaults are the configuration. The runner applies
the manifest with `setdefault`, so any `PREFILL_*` variable already set in the
environment silently wins over both. Leave them unset. In particular,
`PREFILL_FABRIC_MODE` and `PREFILL_OVERLAP_SHARED_EXPERT` default to
`2d_torus_xy` and `1`, which is what CI runs; the `2d` fabric, and the torus
with overlap off, are not.

Set only what is specific to the deployment, in the binding's `global_env`.
`ttrun` does not forward `PREFILL_*` variables from the launching shell, so an
export there reaches rank 0 only.

- `PREFILL_NUM_USERS`, when the decode side caps slots below the variant's count.
- `PREFILL_H2D_SERVICE_ID`, matching the producer.
- Migration: `PREFILL_ENABLE_MIGRATION=1`, `PREFILL_MIGRATION_EXPORT_TO_FILE=1`,
  `PREFILL_MIGRATION_TABLE_PATH` on storage all hosts share, and
  `PREFILL_MIGRATION_DEVICE_MAP_PATH` on host-local storage.
- `PREFILL_HF_MODEL` / `PREFILL_TTNN_CACHE`, when the weights are not at the
  adapter defaults.

## Teardown

A clean shutdown needs no recovery. A killed run does — go back to step 1.

```bash
for h in ${HOSTS//,/ }; do
  ssh -o BatchMode=yes $h 'pkill -f prefill_runner; rm -f /dev/shm/*prefill*; true'
done
```
