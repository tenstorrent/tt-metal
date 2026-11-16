import datetime
import json
import subprocess
from pathlib import Path

s = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup")
rows = json.loads((s / "swe_runtime_preseed.json").read_text())
out = []


def run(args):
    return subprocess.check_output(args, text=True).strip()


for r in rows:
    cmd = (
        'git -C /testbed log -2 --format="%H %P %T %s"; git -C /testbed diff --name-status '
        + r["expected_base_commit"]
        + ' HEAD; git -C /testbed status --porcelain; for p in /solution /tests; do if test -e "$p"; then echo PRESENT:$p; else echo ABSENT:$p; fi; done'
    )
    a = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/bash", r["base"], "-lc", cmd])
    b = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/bash", r["tag"], "-lc", cmd])
    assert a == b
    bi = json.loads(run(["docker", "image", "inspect", r["base"]]))[0]
    pi = json.loads(run(["docker", "image", "inspect", r["tag"]]))[0]
    assert pi["RootFS"]["Layers"][:-1] == bi["RootFS"]["Layers"]
    out.append(
        dict(
            task=r["task"],
            command=cmd,
            original_and_prepared_output=a,
            outputs_equal=True,
            base_layers=len(bi["RootFS"]["Layers"]),
            prepared_layers=len(pi["RootFS"]["Layers"]),
            base_layers_preserved=True,
            additional_layer=pi["RootFS"]["Layers"][-1],
        )
    )
    print(r["task"], "original/prepared repository probes equal", flush=True)
(s / "swe_image_repository_provenance.json").write_text(
    json.dumps(
        dict(
            checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope="All probes used docker run --network none; no eval or model calls.",
            records=out,
        ),
        indent=2,
    )
    + "\n"
)
