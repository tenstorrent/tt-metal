# t110 — gate fold + norm-AdaLN block A/B (2x4, Linear sp1/tp0)

Files: test_gate_adaln_ab.py (harness), run110.sh (broker job), driver.sh (detached driver on blx03).
Staged on g14blx03:/var/tmp/fasth3/t110/. Tree on blx03: ~/fasth3/t48 @83c11ee2b34 (built).

Launch: tt-project/harness/templates/blx03-launch.sh t110 /var/tmp/fasth3/t110/driver.sh
Done marker: "T110_DRIVER_DONE ab <rc>" in /var/tmp/fasth3/t110/driver.log (rc 9 = drop/reboot in OUR job -> stop all device work).

Caveat: gate fold only engages on Ring; harness forces it on Linear and runs the folded projection as
one minimal_matmul + slices. Gate ms here is a Linear lower bound; PCC is the real check (t51 Ring: -3.7% S1, -4.9% S2).

Next on resume:
  ssh g14blx03 'cat /var/tmp/fasth3/t110/driver.log; grep -E "T110_(AB|BLOCK|FOLD|FAIL|EXIT)" /var/tmp/fasth3/t110/run110.log'
Then build the table (S1/S2 ms/block per arm, PCC v/a, job id), write result.json, delete /var/tmp/fasth3/t110 on blx03.

## Result (blx03 broker job 456, 2026-10-03 07:07-07:09 UTC, exit 0, post-job health gate OK)
Traced block replay, median of 3 laps x 10 replays, 2x4 submesh of the (4,8) mesh, Linear sp1/tp0, t48 @83c11ee2b3.

| shape | arm   | ms/block | delta vs base   | PCC video | PCC audio |
|-------|-------|----------|-----------------|-----------|-----------|
| S1    | base  | 14.006   |                 |           |           |
| S1    | gate  | 14.015   | +0.009 (+0.07%) | 0.9999993 | 0.9999946 |
| S1    | adaln | 13.738   | -0.267 (-1.91%) | 0.9999632 | 0.9999820 |
| S1    | both  | 13.895*  | -0.111 (-0.79%) | 0.9999633 | 0.9999821 |
| S2    | base  | 60.002   |                 |           |           |
| S2    | gate  | 59.972   | -0.029 (-0.05%) | 1.0000000 | 0.9999999 |
| S2    | adaln | 59.317   | -0.685 (-1.14%) | 0.9999834 | 0.9999688 |
| S2    | both  | 59.009   | -0.993 (-1.66%) | 0.9999834 | 0.9999688 |
*S1 both laps 13.692/13.895/14.040: noisy; min 13.692 (-2.2%).
Base matches #85 job 161 (S1 14.003, S2 60.560). Raw lines: job456_results.log.
