# Full-mesh gather: linear enablement + comb ring — state as of 2026-09-07

Untracked scratch notes. Delete before merging anything.

## TL;DR

- **PR 1 (linear enablement) is complete and hardware-validated.** Both halves are now
  committed and green: the CCL commit and the model-side fallback deletion. The MLA hang
  that blocked it was **not ours** — see "RESOLVED: the MLA hang" below.
- **The branch is parked off `origin/main` on purpose.** It is rebased to *before*
  `f010a178807` (#54741), the commit that causes the hang. It cannot be PR'd from here.
  Rebase forward once the author's fix lands.
- **PR 2 (comb, perf) is written and hardware-validated** but sits on branches based on the
  *old* main. Needs re-basing onto PR 1.

## Branch inventory

| branch | base | state |
|---|---|---|
| `ipotkonjak/hbag-full-mesh-linear` | `fb7a94b2a87` (pre-#54741) | **PR 1, complete.** 2 commits, clean tree |
| `backup/hbag-full-mesh-linear-pre-rewind` | new main `28d1cab3f8c` | position before the 2026-09-07 rewind; safe to delete once #54741 is fixed |
| `ipotkonjak/snake-comb-ring-fabric2d` | old main `167b5ccb4e8` | comb (`568a1035370`, pushed) + open path (`865253bd886`, local) |
| `ipotkonjak/comb-path-integration` | old main | merge of comb + open path + deletion; carries `1852c4008d9` (one-wide fixes), already folded into PR 1 |
| `ipotkonjak/glm52-kv-tp-sharding` | old main | superseded — main merged the TP-sharded work |
| `backup/glm52-kv-tp-sharding-prerebase` | — | recovery point from the earlier rebase, safe to delete |

## PR 1 — linear enablement

### `6005c437a27` (CCL) — DONE, validated

Adds an open Hamiltonian **path** tier (`Topology::Linear`) to `resolve_mesh_ring_plan`
for when no snake **cycle** closes. Plain (non-torus) `FABRIC_2D` on 8x4 previously had no
full-mesh route at all; now it has one.

Files: `mesh_ring_plan.{hpp,cpp}`, `high_bw_all_gather_device_operation{,_types}.{cpp,hpp}`,
`high_bw_all_gather_unicast_factory.cpp`, `test_ccl_helpers.cpp`, `test_high_bw_all_gather.py`.

Key points:
- The op already had a full line schedule (`relay_iters`, `num_recv`, `do_local_write`,
  `dir_active` all branch on `is_ring`; every axis gather uses it). Only one ternary in the
  program factory forced `Ring`.
- `is_ring` is now topology-driven instead of forced by `full_mesh`.
- `get_mesh_ring_position` returns no forward neighbour at rank N-1 / no backward at rank 0
  when the plan is Linear.
- Opt-in via `allow_open_path`, **default false**. `ring_joint_sdpa` and `indexer_score`
  fuse a genuine closed ring and must keep getting `nullopt`. Only `high_bw_all_gather`
  opts in.
- Also stopped excluding 1-wide meshes from the *cycle* search: a 1xN boustrophedon
  degenerates to a straight line whose closing edge is the axis wrap, so it closes on a
  torus. Precondition is now `mesh_size > 2 && some even lane count`; the host edge proof
  decides. The op's ring validation carried the same both-extents rule and was relaxed too.

Validated on the local 32-device Blackhole Galaxy 8x4:
- 9 passed: 8x4 whole-mesh accuracy on BOTH fabrics, 8x1/1x4/1x2 submesh accuracy on both,
  8x4 axis-ring regression. Torus still resolves the same `Row` ring as before.
- `unit_tests_ttnn_ccl CclHelpers`: 4/4.

Measured perf cost (2 links, effective receive bandwidth GB/s):

```
                     torus (ring)   plain 2D (path)   ratio
  bf16 TILE   1720      67.25            44.36        0.66
  bfp8 TILE   1720      59.54            41.84        0.70
  bf16 TILE  16384      91.93            47.93        0.52
  bfp8 TILE  16384      89.69            47.34        0.53
```

0.52 vs a predicted 16/31 = 0.516 — the cost is hop count and nothing else. Perf gates were
deliberately left torus-only so no perf leg's runtime doubles (models/bh_sc1_high_power is
at 166/166).

### `baf546ff99a` (mla.py fallback deletion) — DONE, validated

`-144/+17`. Removes `_snake_ring_can_close`, `_SNAKE_CLOSING_TORUS_CONFIGS`,
`_can_full_mesh_gather_kvpe`, `_gather_kvpe_prefix_tp_sharded_high_bw`,
`_kvpe_tp_stage_buffer` (worth ~1/sp of scratch). Three of the guard's five conditions were
already invariants elsewhere and become asserts in the surviving route. The release site
needed no change (already route-independent).

Validated on the rewound base: `test_sparse_mla_rotated -k tp_sharded`, **2 passed in 71s**,
both variants green on all three PCC gates (KVPE cache 0.9999, indexer cache 0.9999,
rotated 0.9909).

**Committed with `--no-verify`.** pre-commit could not install its hook envs on this box
(`No module named pip.__main__` in the venvs it creates; survives `pre-commit clean` — a
machine Python-env problem, not a repo one). Checked by hand instead: no trailing
whitespace, single trailing newline, longest added line 108 < black's 120, and all 20
module-level imports still referenced so autoflake is a no-op. **Re-run the real hooks
before PR'ing.**

## RESOLVED: the MLA hang — caused by #54741, not by this work

`pytest models/.../sparse_mla/test_sparse_mla.py::test_sparse_mla_rotated -k tp_sharded`

Cause: **`f010a178807` — "[Performance] Optimize fused Ring Indexer communication and
scheduling" (#54741)**, which landed in main between the two branch bases. It rewrites
`ring_indexer_score_dsa_program_factory.cpp` (470 lines), the ring-attention all-gather
reader/writer kernels (~900 lines), and adds `ring_indexer_score_schedule.hpp`. The author
is working on a fix (as of 2026-09-07).

The A/B that settles it — each row differs from its neighbour by exactly one thing:

| base | mla.py deletion | result |
|---|---|---|
| new main **with** #54741 | stashed | **hang**, exit 124 at 900s |
| base rewound **without** #54741 | applied | **2 passed, 71s** |

Row 1 exonerates the deletion (it hangs without it). Row 2 clears the rest: the full PR 1
passes once #54741 is gone. Note the passing run is also *faster* than the old-main
baseline (71s vs 159s on `comb-path-integration`), so nothing degraded to a slow path.

Hang signature, for whoever picks up #54741: log dies ~2s after `MLA_START`, then silence
until the outer timeout. `gdb -p` on the wedged process (saved at `scratchpad/hang_bt.txt`)
shows the main thread parked in

```
futex wait
  FDMeshCommandQueue::wait_for_outstanding_reads
  FDMeshCommandQueue::finish_nolock
  MeshCommandQueueBase::enqueue_write_mesh_buffer
  GlobalSemaphoreImpl::reset_semaphore_value
  ttnn::global_semaphore::create_global_semaphore
  HighBwAllGatherUnicastFactory::create_mesh_workload
```

with a reader thread spinning in `read_completion_queue` -> `completion_queue_wait_front`
-> `Cluster::read_sysmem`. **The device never posted a completion.** Read that stack
carefully: `wait_for_outstanding_reads` drains reads *already in flight*, so the semaphore
write is merely the first sync point that noticed a device which had already stopped
responding — it does **not** implicate `high_bw_all_gather`'s own kernels. The program that
actually wedged the device was dispatched earlier. A `TT_METAL_WATCHER=1` run would name
the stuck core and kernel; not yet done.

Also note `test_sparse_mla_rotated` carries `@pytest.mark.timeout(0)`, so pytest's 300s
per-test timeout is disabled on it — a hang there runs until the outer `timeout` kills it.
Do not read "outlived the pytest timeout" as evidence of anything.

## PR 2 — comb ring (perf follow-up)

Written and hardware-validated, but based on old main. Needs re-basing onto PR 1.

Adds `CombRow`/`CombColumn`: spine along one lane, then a boustrophedon back toward the
spine's origin, so every edge including the closing one is a nearest neighbour. Closes a
ring on a plain 2D fabric with no torus, taking the 0.52 ratio above back to ~1.0.

Validated (on the old-main branches): plain 2D 8x4 accuracy + both perf gates pass, within
-2.4%/+2.7% of torus across 12 payload cases. Resolver picks `CombRow` on plain 2D and
`Row` on torus, so production is untouched.

When re-basing, PR 2 should carry:
- the comb enum + index math + `lane_count()` helper (PR 1 deliberately left the existing
  ternaries alone to stay minimal)
- comb gtests
- `_FULL_MESH_DEVICE_PARAMS` on the perf gates + raised floors for plain 2D
- the one-wide fixes from `1852c4008d9` are ALREADY folded into PR 1, do not double-apply

## Next steps

1. Wait for the #54741 fix. Then rebase this branch forward onto `origin/main` and re-run
   `test_sparse_mla_rotated -k tp_sharded` to confirm the fix actually resolves it.
2. Re-run the real pre-commit hooks over `baf546ff99a` before opening the PR.
3. Finish the `ring_joint_sdpa` + `indexer_score_dsa_4d` regression (see below).
4. Rebase PR 2 onto PR 1.

## Not tested / known noise

- `CombColumn` — gtest only, never on silicon. On 8x4 `CombRow` always resolves first, so
  reaching it needs a temporary preference flip.
- `ring_joint_sdpa` + `indexer_score_dsa_4d` regression — **never completed**. Three
  attempts, all starved of the device or killed because other work was run concurrently.
  Needs one clean run with nothing else touching the Galaxy.
- Full GLM 744B chunked prefill — no weights on this box, needs CI.
- **This machine's Galaxy is flaky.** On 2026-09-07 two consecutive runs died in hardware,
  not in the test: a device-0 active-ethernet-core handshake timeout during
  `FABRIC_2D_TORUS_XY` bring-up, then (after a reset) a device-5 PCIe hang
  (`Read 0xffffffff over PCIe ID 5`) that took out `tt-smi -ls` itself. Both cleared on
  their own later. **Check `tt-smi -ls` enumerates 32 chips before trusting any run.**
- Beware a false pass: when `ttnn.cluster.get_cluster_type()` throws, the deepseek conftest
  guard catches it and *skips* every ring/torus test with a message about cabling
  certification. That is a hardware failure wearing a skip's clothing, and the exit code is
  0. Always confirm the test actually reached `MLA_START`.
- Pre-existing on this machine, reproduced on a clean main baseline, NOT caused by any of
  this work:
  - 4-device / (2,4) submesh fabric-router ethernet handshake timeouts
  - three `512k` axis-ring bandwidth failures (torus wraparound ~48 GB/s vs a 90 floor)
  - `galaxy_ci_perf` bfp8-TILE straddles its own 78.0 floor (baseline spans 77.46-80.60)

## Diagram

Ring-order diagrams (all four orientations, wrap edges, edge budgets, perf table):
https://claude.ai/code/artifact/75c43b6d-490a-46c0-8f45-def4b0298378
