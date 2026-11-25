import json, resource, subprocess, sys, time

label, times, log, cmd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
start = time.time()
with open(log, "w") as f:
    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
wall = time.time() - start
r = resource.getrusage(resource.RUSAGE_CHILDREN)
row = dict(label=label, wall_s=round(wall, 1), cpu_s=round(r.ru_utime + r.ru_stime, 1), rc=rc)
print(json.dumps(row))
open(times, "a").write(json.dumps(row) + "\n")
