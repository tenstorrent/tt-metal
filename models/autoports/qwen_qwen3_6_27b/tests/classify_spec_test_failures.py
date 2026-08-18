import glob
import json
import os
import re
import sys

base = os.path.expanduser("~/tt-inference-server/workflow_logs/reports_output/spec_tests")
files = sorted(glob.glob(os.path.join(base, "**", "report_data*.json"), recursive=True),
               key=os.path.getmtime)
if not files:
    print("no report_data json found")
    sys.exit(1)
path = files[-1]
print("report:", os.path.basename(path))

d = json.load(open(path))
blob = json.dumps(d)

# locate the conformance block
detailed = None
summary = None
stack = [d]
while stack:
    n = stack.pop()
    if isinstance(n, dict):
        if "detailed_test_results" in n:
            detailed = n["detailed_test_results"]
            summary = n.get("parameter_conformance_summary")
        stack.extend(n.values())
    elif isinstance(n, list):
        stack.extend(n)

if summary:
    print("\n=== per-case summary")
    for s in summary:
        print("  %-32s %-10s %s" % (s.get("test_case"), s.get("status", "").strip(), s.get("summary")))

if not detailed:
    print("no detailed_test_results")
    sys.exit(0)

print("\n=== failure classification (%d parametrizations)" % len(detailed))
buckets = {}
for r in detailed:
    st = (r.get("status") or "").strip()
    msg = r.get("message") or ""
    if "PASS" in st:
        kind = "PASS"
    elif "ReadTimeout" in msg or "Read timed out" in msg:
        kind = "TIMEOUT (read timeout=30)"
    elif "ConnectionRefused" in msg or "NewConnectionError" in msg:
        kind = "CONNECTION REFUSED"
    elif "assert" in msg.lower():
        kind = "GENUINE ASSERTION FAILURE"
    elif msg.strip():
        first = msg.strip().splitlines()[-1][:90]
        kind = "OTHER: " + first
    else:
        kind = "OTHER: (no message)"
    buckets.setdefault(kind, []).append(r.get("parametrization") or r.get("test_case"))

for kind, names in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    print("  %-38s %2d" % (kind[:38], len(names)))
    for nm in names[:6]:
        print("        %s" % nm)
    if len(names) > 6:
        print("        ... +%d more" % (len(names) - 6))

n_pass = len(buckets.get("PASS", []))
n_to = len(buckets.get("TIMEOUT (read timeout=30)", []))
n_assert = len(buckets.get("GENUINE ASSERTION FAILURE", []))
print("\n  passed=%d  timeouts=%d  genuine-assertions=%d  other=%d"
      % (n_pass, n_to, n_assert, len(detailed) - n_pass - n_to - n_assert))
print("  read-timeout value seen in messages:",
      sorted(set(re.findall(r"read timeout=(\d+)", blob))))
