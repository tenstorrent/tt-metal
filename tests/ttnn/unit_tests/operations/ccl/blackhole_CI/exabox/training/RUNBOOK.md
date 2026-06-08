# Runbook — DistributedContext send/recv exabox tests on QUAD_BH

End-to-end instructions for manually exercising the byte-level send/recv tests
in this folder. See `../multi_host/RUNBOOK.md` for the general exabox
prerequisites and troubleshooting catalogue these instructions build on.

---

## 1. What's in this folder

Three byte-level tests exercising `DistributedContext.send/recv` (host MPI
point-to-point), each mirroring a tt-train workload pattern.

| Test | What it exercises |
|---|---|
| `test_send_recv_training.py::test_round_robin_send_recv_32x4` | Adjacent-rank ring exchange in both directions |
| `test_send_recv_training.py::test_pipeline_activation_handoff_32x4` | Stage-boundary fwd activation + bwd gradient |
| `test_send_recv_training.py::test_remote_optimizer_grad_exchange_32x4` | All workers (1,2,3) → aggregator (0); aggregator → all workers |

Byte-level tests run in seconds; total wall time is dominated by `tt-run`
launch (~30 s per invocation).

---

## 2. Prerequisites

### 2.1 Build

`_ttml*.so` must exist with the byte-level send/recv Python bindings:

```bash
cd $TT_METAL_HOME
./build_metal.sh --build-tt-train
```

The bindings live in `tt-train/sources/ttml/nanobind/nb_core.cpp`
(see `DistributedContext::send`/`recv`). The conftest auto-symlinks
`build_Release/ttml/_ttml*.so` into the source package on first import.

### 2.2 Cluster

4-host BH Galaxy (`bh-glx-*`), each host a full Galaxy of 32 devices.
Default host list:

```
bh-glx-b06u02, bh-glx-b06u08, bh-glx-b07u02, bh-glx-b07u08
```

Override via `HOSTS=...` env in the launcher script, or substitute on the
command line.

### 2.3 SSH

Passwordless ssh from the launching host to every cluster host, including
to itself. Test once before launching:

```bash
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "hostname"
done
```

### 2.4 Chips

Run a reset before the first launch and after any hung run:

```bash
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  ssh -o BatchMode=yes "$h" "$TT_METAL_HOME/python_env/bin/tt-smi -glx_reset_auto" &
done
wait
```

Each reset takes ~60 s. Reset is complete when each host's stdout shows
`Re-initialized 32 boards`.

---

## 3. The single rule

**One `tt-run` invocation at a time. Wait for completion before the next.**

The cluster's chips are shared. Two `tt-run` launches will collide and
deadlock or corrupt each other.

---

## 4. Quick path — run everything

```bash
cd $TT_METAL_HOME
bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/scripts/run_training_tests.sh
```

Launches `pytest` once under the single-mesh binding and runs all three tests.

---

## 5. Manual per-test invocation

Binding: `tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml`
(stock — all 4 ranks share `mesh_id: 0`).

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08" \
  bash -c "source python_env/bin/activate && \
           MESH_DEVICE=QUAD_BH pytest --timeout=300 -v \
           tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_send_recv_training.py::test_round_robin_send_recv_32x4"
```

Substitute the test name to run the others:

- `::test_round_robin_send_recv_32x4`
- `::test_pipeline_activation_handoff_32x4`
- `::test_remote_optimizer_grad_exchange_32x4`

You can also run all three in one invocation by dropping the `::test_*`
suffix:

```bash
tt-run ... pytest --timeout=300 -v \
  tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_send_recv_training.py
```

**Expected output** (per test): `1 passed in <≈3-4s>`. Wall-time per test
~30 s (dominated by tt-run launch).

---

## 6. Troubleshooting

### 6.1 `pytest` hangs at 99% CPU after the test reports `passed`

MPI teardown sometimes can't tear down cleanly. Kill it manually and proceed:

```bash
# locally
ps -ef | grep -E 'tt-run|prterun|pytest|prted' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# on every host
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  ssh "$h" "ps -ef | grep -E 'pytest|prted' | grep -v grep | awk '{print \$2}' | xargs -r kill -9"
done
```

Confirm clean before the next launch:

```bash
ps -ef | grep -E 'tt-run|prterun|pytest|prted' | grep -v grep | wc -l
# → 0
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  echo -n "$h: "; ssh "$h" "ps -ef | grep -E 'pytest|prted' | grep -v grep | wc -l"
done
```

### 6.2 Test hangs with no log progress for 10+ minutes

Likely a partial MPI bring-up hang. Kill (§6.1) and reset chips (§2.4). If
it reproduces twice in a row, capture the per-rank log and escalate.

---

## 7. Sanity-check on a single process

The conftest skips cleanly if invoked without MPI:

```bash
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/ -v --no-header
```

Expected: 3 tests collected, 3 skipped with reason
`send/recv tests require multi-process launch (world_size=1); use tt-run.`

This confirms the conftest, `_ttml` bootstrap, and pytest collection all
work without touching the cluster. Useful to verify after a fresh build.

---

## 8. Logs and artifacts

When you launch a new run, redirect into a sibling file so previous runs
remain available:

```bash
mkdir -p /tmp/training_send_recv_runs
tt-run … > /tmp/training_send_recv_runs/run_$(date +%Y%m%d_%H%M%S).log 2>&1
```
