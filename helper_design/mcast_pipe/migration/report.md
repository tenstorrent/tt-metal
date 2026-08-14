# `mcast_pipe` current rollout report — paused 2026-08-14

This is the live report. The detailed completed-v10 report is preserved at
`../archive/reports/migration_report_through_2026-08-06.md`.

## Result

The source tree and rollout inventory are reconciled at branch head
`9686814ea22`. All 91 census paths exist and match the ledger exactly. No kernel
was added, removed, renamed, or clobbered since the last write-back.

The helper is API v11 while the 17 migrated kernels and 14 migrated host
bindings remain recorded at v10. Four source-integrated kernels and ten required
bindings are pending; 70 kernels remain deliberately deferred. Three migrated
kernels carry `needs_recheck` because their source changed after the last
write-back.

## Intake validation performed during documentation cleanup

- `./build_metal.sh`: passed.
- Exact helper smoke case under `run_safe_pytest.sh --dev`: passed.
- `McastHostFixture.*`: 32/32 passed.
- Complete `test_mcast_pipe.py --dev`: 80/80 passed.

This proves that the materialized host and device helpers are healthy at intake.
It is not per-operation rollout validation and did not change any migration
status or API stamp.

## Work deliberately left for later

- Verify the v10 fleet against API v11 and clear the three `needs_recheck` flags.
- Validate Matmul in0 interleaved: two kernels and five bindings.
- Validate Matmul in0 block-sharded: one hybrid kernel and four bindings.
- Validate block-sharded Conv2D activation: one hybrid kernel and one binding.
- Write back ledger status only after each atomic unit's mapped correctness,
  exact-JIT, and performance requirements pass.

No migration run is active, no production source was changed by this cleanup,
and no apply mode has been selected.
