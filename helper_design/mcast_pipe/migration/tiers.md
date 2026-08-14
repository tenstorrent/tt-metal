# `mcast_pipe` prepared rollout tiers — not started

This is a future work order, not authorization to migrate. Recompute it if the
tree changes before the user resumes the rollout.

| Tier | Atomic unit | Scope | Why next |
|---|---|---|---|
| 0 | current v10 fleet → v11 | 17 kernels, 14 bindings | API-version write-back is stale; three kernels also need recheck |
| 1 | `matmul-in0-mcast-interleaved` | 2 kernels, 5 bindings | Source integrated; mapped interleaved and sparse validation remains |
| 2 | `matmul-in0-mcast-block-sharded` | 1 hybrid kernel, 4 bindings | Source integrated; rotating-sender geometry and exact route require atomic validation |
| 3 | `conv2d-activation-block-sharded` | 1 hybrid kernel, 1 binding | Source integrated with streaming `send_from_cb`; Conv and performance validation remain |

Tier 0 must run first because migrated entries are recorded against API v10.
Within every tier, host and kernel bindings form one atomic unit and device tests
run sequentially. The authoritative tests and dispatch conditions are in
`test_map.json`; the authoritative status is in `ledger.json`.
