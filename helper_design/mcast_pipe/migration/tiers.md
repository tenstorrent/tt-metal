# Tiers — mcast_pipe rollout re-entry @ v8 — 2026-06-27 (run mode: run-all)

Re-entry after the `llk_helper_library` rebase + `reconcile-dm-helper` (2026-06-27). Helper unit test
green (50/50 @ v8). **0 stale** (all 19 migrated @ v8 — no JIT breakage), so there is no Tier 0
remigration. Worklist = 1 `needs_recheck` (verify-only) + 3 `pending` (net-new), all on a single
WORMHOLE_B0 device.

## Tier V — verify-only (needs_recheck, no rewrite)
Set by reconcile after the rebase moved the file. Run its mapped test; clear the flag on green.

| kernel | validation_set | expectation |
|---|---|---|
| reader_mcast_receiver_unary_sharded_ln.cpp | tests/ttnn/unit_tests/operations/normalization/test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage | PASS (rebase edit was a 3-line comment; helper logic untouched) -> clear `needs_recheck` |

## Net-new pending — ALL DEFER (coverage-gap), NOT migrated
Phase 1 device-verification verdict: none of the 3 pending kernels is reachable/verifiable on this
single WORMHOLE_B0 device. They are SERVICE kernels (not op program factories), and their only tests
need hardware we don't have. Migrating them blind would be the "migrated but unverified" anti-pattern,
so they move `pending -> deferred` with flag `coverage-gap`.

| kernel | service / factory | candidate test | why unverifiable here |
|---|---|---|---|
| persistent_d2h_sender.cpp | ttnn/core/services/d2h_socket_service.cpp (D2HSocket) | test_send_recv_async_hd.py::test_send_async_d2h_basic | test is **blackhole-only** — SKIPPED on wormhole_b0 |
| persistent_d2d_receiver.cpp | ttnn/core/tensor/d2d_stream_service.cpp (D2DStream) | test_d2d_stream_service_multiprocess.py | requires **Galaxy** (2-process multi-mesh, 4x4 each) |
| persistent_d2d_sender.cpp | ttnn/core/tensor/d2d_stream_service.cpp (D2DStream) | test_d2d_stream_service_multiprocess.py | requires **Galaxy** (2-process multi-mesh) |

These mirror their already-deferred census twin `persistent_h2d_receiver` (coverage-gap). Candidate
tests are recorded in test_map.json so a re-entry on a Blackhole / Galaxy machine can migrate+verify them.

## Migration tiers (Tier 1+): EMPTY this run
No clean/refactor pending kernel is device-verifiable here -> nothing to migrate. No production op is
rewritten this run. No API-version change.
