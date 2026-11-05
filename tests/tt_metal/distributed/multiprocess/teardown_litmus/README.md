# Teardown litmus (CPU only, mock dual galaxy)

One fault, one assertion per row, on tt-metal's own distributed stack: the distributed context, the control
plane's init inside `open_mesh_device`, the collectives, close and exit, plus the user's signals at tt-run.
Two ranks of 8x4 on the Wormhole 6U dual-host mock descriptors, launched by tt-run, no device.

The rows come from the teardown plan for tt-metal #56902 (tt-blaze `tools/teardown_probe`, lane L4, where
every row was first measured with its log). A row that passes today is a gate. A row that fails today runs
in report mode: `xfail(strict=True)` with the plan's error id in the reason, so the unexpected pass after the
fix fails the job and the marker is dropped in the same PR.

| row | fault | assertion | today |
|---|---|---|---|
| `test_clean` | none | exit 0, nothing left | gate |
| `test_dead_peer_barrier_ends` | rank 1 dies before a barrier | the survivor's barrier raises MPI_ERR_PROC_FAILED, the job ends, nothing left | gate |
| `test_dead_peer_recv_raises` | rank 1 dies before rank 0's recv | recv raises, the job ends | gate |
| `test_fabric_config_mismatch_ends_nonzero` | FABRIC_2D on one rank only | TT_FATAL on one rank, exit non-zero, nothing left | gate |
| `test_prte_timeout_ends_deadlock` | a send nobody receives, PRRTE `--timeout 60` | exit 241 after both ranks were past open, nothing left | gate |
| `test_exit_nofinalize_ends_nonzero` | rank 1 `_exit`s with the mesh open | exit non-zero, nothing left | gate |
| `test_invalid_rank_raises` | recv from a rank that does not exist | raises at once | gate |
| `test_use_before_init_raises` | a barrier before init | raises "not initialized" | gate |
| `test_raise_after_open_ends` | rank 1 raises with the mesh open | the job ends non-zero | report, P1 |
| `test_bad_shape_on_one_rank_ends` | rank 1 opens 9x9 | the job ends non-zero | report, P21 |
| `test_conflicting_descriptor_ends` | rank 1's descriptor is a T3K | the job ends non-zero | report, P21 |
| `test_barrier_skipped_ends` | rank 1 skips a barrier | the job ends | report, P16 |
| `test_send_without_recv_ends` | a send nobody receives | the job ends | report, P16 |
| `test_sigkill_before_init_ends` | rank 1 killed before MPI_Init | the job ends | report, P13 |
| `test_dead_peer_in_open_exit_nonzero` | rank 1 killed inside open | exit non-zero | report, P17 P4 |
| `test_ctrl_c_in_deadlock_leaves_nothing` | SIGINT at tt-run in a deadlock after open | tt-run ends, nothing left 3 s later | gate |
| `test_ctrl_c_in_init_leaves_nothing` | SIGINT at tt-run while a rank waits in the context init for a late peer | nothing left 3 s later | report, P6 |
| `test_sigterm_at_ttrun_ends` | SIGTERM at tt-run in a deadlock | the job ends, nothing left | report, P7 |
| `test_solve_rank_killed_ends` | a phase-1 rank killed | tt-run ends non-zero, nothing left | report, P18 |

Every row kills whatever it left (its own process group, then any launcher or probe process of this user
that appeared during the row) before the next one starts, so a hang in one row cannot fail the next.

Run from the repository root with a built tree:

```
pytest tests/tt_metal/distributed/multiprocess/teardown_litmus -v
```

In CI this is group `teardown-litmus` of `tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh`, entry
`fabric-cpu-teardown-litmus` of `tests/pipeline_reorg/fabric_cpu_only_unit_tests.yaml` (the
`fabric-cpu-only-unit-tests` job of `tm-fabric-tests-impl.yaml`). Deadlines: `TT_LITMUS_DEADLINE` (a row that
must end, 90 s) and `TT_LITMUS_DEADLINE_HANG` (a row that hangs today, 60 s).
