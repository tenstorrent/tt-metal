import hashlib
import json
import pathlib
import subprocess
import sys
import time

O = pathlib.Path(__file__).resolve().parent
R = pathlib.Path.cwd()
a = json.loads((O / "admission.json").read_text())
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p, h in a["source_pins"].items():
    assert sha(R / p) == h, p
assert not (O / "launch.json").exists()
sys.path.insert(0, str(R / "models/autoports/google_gemma_4_31b_it/tools"))
from benchmark_server import device_owners

owners = device_owners()
assert not owners, owners
hf = []
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        c = (p / "cmdline").read_bytes()
    except OSError:
        continue
    if b"precision_repro_hf.py" in c:
        hf.append(p.name)
assert not hf, hf
launch = {
    "start_unix": time.time(),
    "admission_sha256": sha(O / "admission.json"),
    "device_owners": owners,
    "hf_processes": hf,
    "command": a["command"],
    "forecast": a["forecast_seconds"],
}
(O / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
with (O / "run.log").open("w") as f:
    r = subprocess.run(a["command"], stdout=f, stderr=subprocess.STDOUT)
completion = {
    **launch,
    "end_unix": time.time(),
    "elapsed_seconds": time.time() - launch["start_unix"],
    "exit_code": r.returncode,
}
(O / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
print(json.dumps(completion), flush=True)
raise SystemExit(r.returncode)
