# #155: device check of the t152 vol2col_rm sizing fix (blx03)

State 2026-10-06 05:10 UTC: three specs queued on the blx03 runner, FIFO, behind the legacy #140
eval-pack driver (pid 8381, still running; the runner starts once it ends).

| ID | what | CONFIG |
|---|---|---|
| t155-build | `~/fasth3/t155/build155.sh`: apply fix.diff (be371d08a9f C++) to ~/fasth3/t48 as a commit, incremental build_metal.sh --release. No device use. Log /var/tmp/fasth3/t155/build.log, marker `BUILD155_DONE rc=` | t155-build-t48 |
| t155-j1 | `job155.sh j1 64,128,5,4,4` | t152-exact_s2_res-64,128,5,4,4 |
| t155-j2 | `job155.sh j2 64,128,7,4,4 j1` (refuses, exit 21, unless j1 printed T155_PASS) | t152-exact_s2_res-64,128,7,4,4 |

- t48 tree before: bf7db12a14 (t149 guard on c4409b1fa2), clean. Factory blob equals aa4ade43a28's, so fix.diff
  (`git diff aa4ade43a28 be371d08a9f -- ttnn/`) applies (checked with git apply --check).
- Python: overlay /var/tmp/fasth3/t155/src = git archive 73347dfaec8 (models, conftest.py, pytest.ini).
  73347dfaec8 adds SWEEP_CHECK_ALL=1: without it a blocking slower than the table one gets no output check
  and one >1.5x slower is counted "slow" (failed) without the full timing.
- Job script gates: build marker rc=0 and fix in the factory (exit 20); for j2, j1 passed (21); no tray-2
  incident in server.log in the last 30 min (22; requeue with a new ID). Then TT_CONV3D_ALLOW_UNALIGNED_VOL2COL=1
  SWEEP_CHECK_ALL=1 SWEEP_MAX_SECONDS=300 pytest ...::test_bruteforce_sweep_ltx25_544p_145f_halo -k exact_s2_res.
  Prints T155_PASS pcc=... or T155_FAIL, and T155_TIMES (table_us and all_results us/op) from the json.
- Logs: /var/tmp/fasth3/t155/{j1,j2}.log, results_{j1,j2}/; runner markers /var/tmp/fasth3/runner/done/t155-*.done.

## Next
On wake: read the three done markers and the logs. Health: grep server.log HEALTH-GATE[post-job] after each job
(eth-heartbeat FROZEN = fail). No marker for j2: `ssh g14blx03 bash ~/fasth3/runner/runner-start.sh`, wait again.
