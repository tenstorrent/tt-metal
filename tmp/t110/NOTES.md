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
