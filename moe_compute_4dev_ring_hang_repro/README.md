# moe_compute fused-combine deadlock on WH galaxy (4,8), cluster_axis=0

Minimal, self-contained repro of a hang in `ttnn.experimental.moe_compute` (Full mode,
the fused `selective_reduce_combine`) on a Wormhole 6U galaxy.

## Exact build to reproduce on

- **tt-metal commit: `837fbd9b55432dd64bf88782fd6c7115bac927e7`**
  ("Squash: Dynamic core placement for moe_compute + all follow-ups")
- This commit is the **head of PR #46863** ("Generalize core selection/placement for
  `moe_compute` with ROW dispatch support"), branch
  `gchoudhary/generalize-core-selection-placement-incoroprating-moe_gpt-test`.
- Submodules at this commit: umd `v0.9.6-42-g44455fa1`, tracy `v0.13.3-tt.0`.

```bash
git fetch origin 837fbd9b55432dd64bf88782fd6c7115bac927e7
git checkout 837fbd9b55432dd64bf88782fd6c7115bac927e7
git submodule update --init --recursive
./build_metal.sh            # or: ninja -C build _ttnn.so _ttnncpp.so
```

## Config (what the smoke sets up itself)

WH 6U galaxy, mesh **(4,8)**, **`cluster_axis=0`** => a **4-device dispatch ring** x 8
replicated cols, **COL** dispatch axis, **FABRIC_1D_RING**. GLM-4.7 MoE dims: 160 experts,
experts_per_device=5, hidden=5120, moe_intermediate=1536, K=8, bf4 experts, 64 tokens.
The script opens its own mesh + fabric + dispatch config and uses random weights (ttnn only,
no model load, ~2 min).

## Run

```bash
# env: ttnn on PYTHONPATH; USE_TORCH_XLA=0 ACCELERATE_USE_XLA=false
# RESET THE GALAXY BETWEEN HANGING RUNS:  tt-smi -glx_reset_auto

timeout 360 python3 moe_compute_smoke.py            # cluster_axis=0 (4-dev ring) -> HANG (exit 124)

python3 ccl_health.py                               # plain CCLs, no moe_compute -> PASS (control)
```

## Signal

A `ttnn.synchronize_device` placed after the fused `moe_compute` **never returns**
(process `timeout` -> exit 124). The smoke prints `moe_compute ok: returned 6 combine ...`
(the op enqueues) and then never reaches the epilogue / `SMOKE PASSED`. Under the watcher
(`TT_METAL_WATCHER=10`) a core parks in a ring-barrier NOC-semaphore wait
(`all_gather_async/.../minimal_default_writer.cpp`).

## Verified results on `837fbd9`

| run | dispatch ring | result |
|---|---|---|
| `moe_compute_smoke.py` (default, `cluster_axis=0`) | 4 devices | **HANG** (exit 124) |
| `SMOKE_CLUSTER_AXIS=1 moe_compute_smoke.py` | 8 devices | **HANG** (exit 124) |
| `ccl_health.py` (plain CCLs) | - | PASS |

So on this commit the fused combine deadlocks on **both** the 4-device and 8-device dispatch
rings (it is not ring-size specific here). For background and the standalone-op isolation
(standalone `selective_reduce_combine` and `moe_compute(compute_only)` pass on the same
device), see tt-metal issue #47523.

## Reset note

`moe_compute` hangs leave the galaxy wedged; reset with `tt-smi -glx_reset_auto` (NOT
`tt-smi -r`) before the next device run.
