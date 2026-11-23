"""Synthetic stream-json transcript for metrics.py v2: one case per fix."""
import json, subprocess, sys, tempfile
from pathlib import Path
MET = sys.argv[1]
D = Path(sys.argv[2]); L = D / "logs"; L.mkdir(parents=True, exist_ok=True)
(L / "baseline_test_x.txt").write_text("....\n===== 10 passed, 2 skipped, 30 deselected in 5.00s =====\n")
(L / "run_meta.txt").write_text("mode=B\ntests=test_x:add\n")
ev = [{"type": "system", "subtype": "init", "model": "m", "timestamp": "2026-01-01T00:00:00Z"}]
n = [0]
def call(name, inp, out, minute, is_error=False, denied=False):
    n[0] += 1; i = f"t{n[0]}"
    ts = f"2026-01-01T00:{minute:02d}:00Z"
    ev.append({"type": "assistant", "timestamp": ts, "message": {"id": f"m{n[0]}", "usage": {"input_tokens": 10},
               "content": [{"type": "tool_use", "id": i, "name": name, "input": inp}]}})
    if denied:
        ev.append({"type": "system", "subtype": "permission_denied", "tool_use_id": i})
    ev.append({"type": "user", "timestamp": ts, "message": {"content": [{"type": "tool_result", "tool_use_id": i,
               "is_error": is_error, "content": out}]}})
P = "python_tests/test_x_hal.py"
# 1 denied by event, unusual text (v1: hang because of 'timeout' in the text)
call("Bash", {"command": f"timeout 60 pytest {P} -k add"}, "This Bash command contains multiple operations; timeout", 1, True, True)
# 2 one variant, pass, 'timeout' word in output (v1: hang)
call("Bash", {"command": f"pytest '{P}::t[a]'"}, "pytest-timeout: 600\n1 passed in 1.0s", 2)
# 3 wrong selection with many passes (v1: full pass)
call("Bash", {"command": f"pytest {P} -k 'not add'"}, "===== 50 passed, 10 deselected in 9.0s =====", 3)
# 4 producer + consumer in one command; consumer passes the full selection (v1: compile_only, missed)
call("Edit", {"file_path": "/w/tests/hal_llk/a.h", "old_string": "x", "new_string": "y"}, "ok", 4)
call("Bash", {"command": f"pytest --compile-producer -n 8 {P} -k add -q; pytest --compile-consumer {P} -k add -q"},
     "10 passed, 2 skipped in 3.0s\n10 passed, 2 skipped, 30 deselected in 4.0s", 5)
# 5 @file written by Write -> attributed to the HAL test, 1 failed
call("Write", {"file_path": "/w/tests/hal_llk/ids.txt", "content": "python_tests/test_x_hal.py::t[a]\n"}, "ok", 6)
call("Bash", {"command": "pytest @hal_llk/ids.txt -q"}, "1 failed in 1.0s", 7)
# 6 code change via Bash, then --lf run that reruns everything (count equals baseline) -> final-code full pass
call("Bash", {"command": "sed -i 's/a/b/' hal_llk/a.h"}, "", 8)
call("Bash", {"command": f"pytest {P} -k add --lf -q"}, "10 passed, 2 skipped, 30 deselected in 4.0s", 9)
# 7 nothing ran -> not an iteration
call("Bash", {"command": f"pytest {P} -k zzz"}, "no tests ran in 0.1s\nerror: not found", 10)
ev.append({"type": "result", "subtype": "success", "duration_ms": 600000, "total_cost_usd": 1.0, "num_turns": 9})
(L / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
out = subprocess.run([sys.executable, "-I", MET, str(D), "--json", str(D / "m.json")], capture_output=True, text=True)
assert out.returncode == 0, out.stderr
m = json.loads((D / "m.json").read_text()); p = m["process"]
kinds = [(r["n"], r["seg"], r["kind"], r["selection"]) for r in m["pytest_timeline"]]
print(kinds)
exp = [(1, 1, "denied"), (2, 1, "pass"), (3, 1, "pass"), (4, 1, "compile_only"), (4, 2, "pass"), (5, 1, "fail"), (6, 1, "pass"), (7, 1, "empty")]
assert [(a, b, c) for a, b, c, _ in kinds] == exp, kinds
assert kinds[2][3] == "subset:-k" and kinds[4][3] == "full" and kinds[6][3] == "full", kinds
assert p["iterations"] == 5, p["iterations"]            # #2, #3, #4/2, #5 (@file), #6
f = p["first_full_pass"]["test_x_hal"]; z = p["final_code_first_full_pass"]["test_x_hal"]
assert (f["iteration"], f["pytest_command"]) == (3, 4), f
assert (z["iteration"], z["pytest_command"]) == (5, 6), z
assert p["code_changes"] == 2 and p["success_at_k"]["test_x_hal"]["5"] is True
assert m["counts"]["denied"] == 1 and m["pytest_runs_total"] == 7 and m["pytest_invocations_total"] == 8
# no baseline -> no full pass (v1 accepted the 1-variant pass)
(L / "baseline_test_x.txt").unlink()
subprocess.run([sys.executable, "-I", MET, str(D), "--json", str(D / "m2.json")], check=True, capture_output=True)
assert json.loads((D / "m2.json").read_text())["process"]["first_full_pass"] == {}
# --baseline from the command line works again
subprocess.run([sys.executable, "-I", MET, str(D), "--baseline", "test_x=10", "--json", str(D / "m3.json")], check=True, capture_output=True)
assert json.loads((D / "m3.json").read_text())["process"]["first_full_pass"]["test_x_hal"]["pytest_command"] == 4
print("SYNTHETIC TEST OK")
