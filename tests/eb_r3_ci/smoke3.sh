#!/usr/bin/env bash
# Round 3 eltwise binary, third pass: the ci3 tree builds; its toggles reach the factory (rule log) and change programs.
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-} EB_R3_LOG_RULE=1
T=tests/eb_r3_ci/test_eb_bng.py
timeout 900 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $T -k "sharded_bcast or post_activation_sharded" 2>&1 | grep -E "EB_R3_RULE|passed|failed" | sort | uniq -c | head -20
EB_R3_PER_FACE=1 EB_R3_FIDELITY=HiFi3 EB_R3_MAIN_REINIT=1 EB_R3_PROBE_INIT=1 timeout 900 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $T -k "sharded_bcast" 2>&1 | grep -E "passed|failed" | tail -2
