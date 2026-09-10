# 4-rank prefill runner launch commands

Two ways to run: the CI script (runner + producer + PCC gate, what you want if
you are validating a change) or the runner alone (a server that idles until
some other producer connects).

## Prerequisites

Per host, once:

- A clone at the **same commit** on all four hosts. Ranks exchange a chunk plan
  derived from the checked-out code; a mismatch surfaces later as a D2D chunk
  plan error, not as a version warning.
- `bash build_metal.sh` and `bash create_venv.sh` in that clone.
- The model's TTNN weight cache reachable at the adapter's default path, or
  `PREFILL_TTNN_CACHE` pointed elsewhere:
  - kimi27: `/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill`
  - glm52: `/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache`
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
  ssh -o BatchMode=yes $h 'rm -f /dev/shm/*prefill*; pkill -f prefill_runner; pkill -f prefill_producer; true'
done
```

Before, not after. The runner unlinks its own segments at startup, but only the
rank-scoped names it can prove it owns, and the unlink is existence-based rather
than staleness-based. A segment left by a `kill -9`'d run makes the next run
unlink a ring another rank is already using.

## 3a. Runner + producer + PCC gate

```bash
export TTRUN_DIR=$HOME/ttop
export PREFILL_SUMMARIES=$HOME/prefill_summaries
mkdir -p $TTRUN_DIR $PREFILL_SUMMARIES
printf '%s\n' ${HOSTS//,/ } > $TTRUN_DIR/hostfile

bash $TT_METAL_HOME/models/demos/common/prefill/runners/ci/run_multirank_pcc.sh kimi27 sc4
bash $TT_METAL_HOME/models/demos/common/prefill/runners/ci/run_multirank_pcc.sh glm52 sc4
```

Pass `sc1` instead of `sc4` for the single-galaxy leg. The script launches the
runner under `ttrun`, waits for it to publish the KV chunk table, derives the
producer's host order from `ttrun`'s own discovery, runs the producer, and exits
non-zero unless every rank clears PCC 0.85.

`PREFILL_SUMMARIES` should contain the literal string `prefill_summaries`: the
script derives its per-model scratch directory by substituting into that path,
so without it both models write to the same directory.

Do not have `PREFILL_NUM_USERS` or `PREFILL_TP_SHARD_KV` exported in this shell.
The script picks them up as deliberate overrides and silently runs a shape other
than the manifest's.

## 3b. Runner only

No producer, no PCC gate. The runner allocates, captures its trace, warms up,
logs `setup complete, entering request loop`, and then blocks forever.

```bash
python3 $TT_METAL_HOME/ttnn/ttnn/distributed/ttrun.py \
  --mesh-graph-descriptor $TT_METAL_HOME/models/demos/common/prefill/runners/topology_configuration/ci/kimi27_sc4_mgd.textproto \
  --hosts $HOSTS \
  --tcp-interface ens5f0np0 \
  --force-rediscovery \
  --mpi-args "-x PATH --tag-output" \
  bash -lc "cd $TT_METAL_HOME; \
    export PYTHONPATH=$TT_METAL_HOME; \
    export PREFILL_MANIFEST=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi27.json; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner"
```

For glm52, swap both paths to `glm52_sc4_mgd.textproto` and `glm52.json`.

Replace `ens5f0np0` with the interface `recover.sh` reported. `ttrun` writes its
phase-1 discovery cache to `./generated/ttrun` under the launch directory, so
run it somewhere writable.

`PREFILL_MANIFEST` must be exported inside the `bash -lc` string. Passed through
`--mpi-args -x` it reaches only the local rank, and the other ranks fall back to
defaults — which shows up much later as a D2D chunk plan mismatch.

## Teardown

A clean shutdown needs no recovery. A killed run does — go back to step 1.

```bash
for h in ${HOSTS//,/ }; do
  ssh -o BatchMode=yes $h 'pkill -f prefill_runner; pkill -f prefill_producer; rm -f /dev/shm/*prefill*; true'
done
```
