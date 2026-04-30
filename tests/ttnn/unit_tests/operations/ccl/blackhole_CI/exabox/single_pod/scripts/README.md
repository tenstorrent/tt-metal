# single-pod test runner scripts

Helpers to run the 16-rank single-pod tests on the 4-host BH cluster.

## Files

| Script | Purpose |
|---|---|
| `reset_chips.sh` | `tt-smi -glx_reset_auto` on all 4 hosts in parallel (~60s). |
| `recover_hung_run.sh` | `pkill -9` local + remote pytest/prte after a hang. |
| `run_chain_test.sh <test>` | Run a CCL chain test from `test_fake_moe_traffic.py` (fast dispatch). |
| `run_pipeline_test.sh <test> [test_file.py]` | Run a 16-stage pipeline test (slow dispatch). |
| `ssh_ulimit_wrapper.sh` | PRRTE ssh agent that injects `ulimit -n 65536;` into remote commands. Wired in by `_run_common.sh`. |
| `_hosts.sh` / `_run_common.sh` | Internal helpers (host list, shared launch logic). |

## Cheat sheet

```bash
cd tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/single_pod/scripts

# Once per session, and after any hang:
./reset_chips.sh

# CCL chain (fast dispatch)
./run_chain_test.sh test_fake_moe_chain_4x2_single_pod

# Pipeline framework (slow dispatch)
./run_pipeline_test.sh test_single_pod_pipeline_fake_moe

# Real DeepSeek Blitz pipeline (currently broken at MoE setup):
./run_pipeline_test.sh test_single_pod_pipeline_setup_and_decode test_single_pod_pipeline.py

# After a hang:
./recover_hung_run.sh
./reset_chips.sh
```

## Pre-flight assumptions

1. **Pipeline config bundle** has been generated and `/tmp/single_pod_current_dir.txt`
   points at it:
   ```bash
   cd $TT_METAL_HOME
   python models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py
   ```
2. **`/opt/openmpi-v5.0.7-ulfm/bin`** is reachable (the runners prepend it to `PATH`).
3. **SSH** to all 4 hosts works without a password prompt.

Override the host list per-shell if needed:

```bash
SINGLE_POD_HOSTS="hostA hostB hostC hostD" ./run_chain_test.sh ...
```

## Why two runners

The blitz_decode pipeline framework requires `TT_METAL_SLOW_DISPATCH_MODE=1`,
but the CCL chain tests use sub-device managers that are unsupported under
slow dispatch. So pipeline tests run with slow dispatch and CCL chain tests
run with fast dispatch — hence two thin entry points around the shared
`_run_common.sh`.

## Logs

Each invocation writes `/tmp/single_pod_<timestamp>_<test>.log`. The runner
prints the last 40 lines on exit. To follow live in another shell:

```bash
LATEST=$(ls -t /tmp/single_pod_*_test_*.log | head -1)
tail -f "$LATEST" | sed 's/\x1b\[[0-9;]*m//g'
```
