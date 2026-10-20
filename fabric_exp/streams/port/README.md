# Streams port into the experiment tree

The experiment tree currently carries `streams_p3f.patch` (applied 2026-10-16 after reversing `streams_p3d.patch`; the faster benchmark loops also run the existing single-pattern and round-robin benchmark sends).

- `streams_p2.patch`: test_infra changes from branch `aagarwal/fabric-test-streams` (worktree
  /home/aagarwal/tt-metal-streams) through P2, including the new `test_yamls/test_fabric_stream_sanity.yaml`.
  It leaves out the branch's measurement-only telemetry switch in test_tt_fabric.cpp (this tree already has it).
- `streams_p1.patch`: the earlier P1 port, kept for reference.
- Undo: `git apply -R streams_p3f.patch` from the repo root, if nothing else changed those files since.
  Otherwise restore from the backups: `exp_tree_test_infra_backup.tgz` (files touched by P1) and
  `exp_tree_p2_extra_backup.tgz` (the two more files P2 touches), both taken before any patch, then delete the
  new `test_fabric_stream_sanity.yaml`.
