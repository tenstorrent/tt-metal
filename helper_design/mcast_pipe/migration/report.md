# mcast_pipe rollout — final report (re-entry @ v8, 2026-06-27)

## Run header
- **Helper version (CURRENT):** `MCAST_PIPE_API_VERSION 8`. Unit test `test_mcast_pipe.py` green
  **50/50** at intake. Helper **NOT modified** this run (read-only) — **no API bump**.
- **Entry mode:** re-entry after a `llk_helper_library` **rebase** + `reconcile-dm-helper` (2026-06-27).
  **Migration mode:** `run-all`. **Machine:** single-chip **Wormhole b0** (num_devices=1).
- **Trigger:** the reconcile flagged 1 migrated kernel `needs_recheck` (rebase touched it) and added
  3 `pending` candidates. This run resolves both.
- **Headline:** **0 stale** (the rebase broke no JIT — all 19 migrated still compile @ v8); **1
  needs_recheck re-verified GREEN**; **3 net-new pending all DEFERRED on coverage** (no single-chip
  verification possible here). **0 migrations, 0 failures, 0 quarantined, no production code touched.**

## Rollout state @ v8 (from ledger.json) — 92 entries
| status | count |
|---|---|
| **migrated (current @ v8)** | **19** |
| pending | **0** |
| quarantined | **0** |
| deferred | **73** (50 prior + 18 quasar metal-2.0 + indexer_score CHAIN + lab_multicast + 3 persistent coverage-gap) |

**0 pending and 0 stale → the migratable-on-this-chip fleet is fully current at v8.** Census grew
69→92 in the reconcile; every one of the 23 new entries is either deferred (gap/in-flux/example) — none
were device-migratable on this machine this run.

## Tier V — verify-only (the needs_recheck kernel) ✅
| kernel | result | action |
|---|---|---|
| reader_mcast_receiver_unary_sharded_ln.cpp | `test_layer_norm_sharded_single_stage` **64 passed / 0 fail** (--dev, watcher clean) | `needs_recheck` flag **cleared**; `last_verified=2026-06-27` |

The rebase edit on this kernel was a 3-line explanatory comment (helper logic untouched), so the
verify-only passed as predicted. **Bonus finding:** its recorded test path was stale — the rebase moved
the suite `operations/normalization/test_layer_norm_sharded.py` → `operations/fused/...`. The
`validation_set` (ledger) and `validation` (test_map) for this kernel were corrected to the new path.
Only this one entry referenced the moved path.

## Net-new pending — all 3 DEFERRED (coverage-gap), none migrated
Phase 1 device-verification verdict: these are mesh/multi-device **service** kernels (not op program
factories) whose only tests cannot run on this single Wormhole-b0 box. Migrating them blind would be the
"migrated but unverified" anti-pattern, so they moved `pending → deferred` + flag `coverage-gap`.

| kernel | service | candidate test | blocked because |
|---|---|---|---|
| persistent_d2h_sender.cpp | d2h_socket_service (D2HSocket) | test_send_recv_async_hd.py::test_send_async_d2h_basic | test is **blackhole-only** — all 5 cases SKIPPED on wormhole_b0 |
| persistent_d2d_receiver.cpp | d2d_stream_service (D2DStream) | test_d2d_stream_service_multiprocess.py | requires **Galaxy** (2-process multi-mesh, 4x4 each) |
| persistent_d2d_sender.cpp | d2d_stream_service (D2DStream) | test_d2d_stream_service_multiprocess.py | requires **Galaxy** (2-process multi-mesh) |

Candidate tests are recorded in `test_map.json` (v8/51 entries) so a re-entry on a **Blackhole** (for
d2h) or **Galaxy** (for d2d) machine can map+device-verify+migrate them. They mirror their already-
deferred census twin `persistent_h2d_receiver` (same coverage-gap).

## Migration tiers (1+): EMPTY
No clean/refactor pending kernel was device-verifiable on this machine → nothing migrated. The 18 quasar
metal-2.0 kernels, `reader_indexer_score` (CHAIN design-gap), and the `lab_multicast` example remain
deferred per the reconcile.

## Commit hygiene
No code changes this run (verify-only + ledger/test_map status updates). All artifact edits are local;
nothing pushed/rebased/reset; helper header untouched.

## Hand-off
- **Nothing device-actionable remains on this Wormhole-b0 box** — 0 pending, 0 stale, fleet current @ v8.
- **Future re-entry on other hardware:** a **Blackhole** box unblocks `persistent_d2h_sender`; a
  **Galaxy** unblocks the two `persistent_d2d_*` kernels (run `apply-dm-helper` there).
- **Future re-entry after quasar stabilizes:** when the experimental metal-2.0 port lands and `#47797`
  closes, re-run `reconcile-dm-helper` then `apply-dm-helper` to migrate the 18 quasar call sites.
- **Future tune-dm-helper:** `reader_indexer_score` needs the CHAIN/`relay_multicast` path the helper
  can't express yet (GAP=CHAIN) — a candidate for a future helper round.
