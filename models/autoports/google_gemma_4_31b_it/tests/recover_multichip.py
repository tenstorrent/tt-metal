"""Owner-checked, bounded recovery for this stage's serialized device queue."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("label")
a = p.parse_args()
out = Path("models/autoports/google_gemma_4_31b_it/doc/multichip_decoder")
owners = []
for fds in Path("/proc").glob("[0-9]*/fd"):
    try:
        for fd in fds.iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("/dev/tenstorrent"):
                owners.append([fds.parent.name, target])
    except OSError:
        pass
assert not owners, owners
commands = [
    ([str(Path(sys.executable).with_name("tt-smi")), "-ls", "--local"], 60),
    ([str(Path(sys.executable).with_name("tt-smi")), "-r"], 180),
    ([str(Path(sys.executable).with_name("tt-smi")), "-ls", "--local"], 60),
    (
        [
            sys.executable,
            "-c",
            "import ttnn; ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D); m=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0); ttnn.close_mesh_device(m); print('MESH_SMOKE_OK')",
        ],
        90,
    ),
]
records = []
for i, (cmd, limit) in enumerate(commands):
    log = out / f"{a.label}_{i}.log"
    with log.open("w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=limit)
    records.append(dict(command=cmd, exit=result.returncode, log=str(log)))
    (out / f"{a.label}.json").write_text(json.dumps(dict(owners=owners, commands=records), indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
