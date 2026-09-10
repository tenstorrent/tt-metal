# 4-rank prefill runner launch commands

Set `HOSTS` to your quad in pipeline order, activate the venv, then run one block.

```bash
export TT_METAL_HOME=/path/to/tt-metal
export HOSTS=host0,host1,host2,host3
cd $TT_METAL_HOME && source python_env/bin/activate
```

## kimi27

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

## glm52

```bash
python3 $TT_METAL_HOME/ttnn/ttnn/distributed/ttrun.py \
  --mesh-graph-descriptor $TT_METAL_HOME/models/demos/common/prefill/runners/topology_configuration/ci/glm52_sc4_mgd.textproto \
  --hosts $HOSTS \
  --tcp-interface ens5f0np0 \
  --force-rediscovery \
  --mpi-args "-x PATH --tag-output" \
  bash -lc "cd $TT_METAL_HOME; \
    export PYTHONPATH=$TT_METAL_HOME; \
    export PREFILL_MANIFEST=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tt/runners/manifests/glm52.json; \
    exec python3 -m models.demos.common.prefill.runners.prefill_runner"
```

The runner allocates, captures its trace, warms up, then blocks in the request
loop waiting for a producer. `ci/run_multirank_pcc.sh <model> sc4` runs the same
launch with a producer attached and gates on per-rank PCC.

`PREFILL_MANIFEST` must be exported inside the `bash -lc` string. Passed through
`-x` it reaches only the local rank, and the other ranks fall back to defaults.

`PREFILL_FABRIC_MODE` defaults to `2d`. Set it only to run a torus mode, and
note that `2d_torus_xy` overflows the erisc kernel-config buffer on a blackhole
galaxy across two or more meshes.

Purge `/dev/shm/*prefill*` on every host after killing a run. The runner unlinks
its own segments on startup, but only the rank-scoped names it can prove it owns.
