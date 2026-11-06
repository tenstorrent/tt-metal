"""Host-only GNU tar/periodic idle export reproduction; synthetic whitespace padding."""
import argparse
import importlib.util
import json
import subprocess
import tempfile
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
spec = importlib.util.spec_from_file_location("phases", a.source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix="phase-archive-race-") as d:
    root = Path(d)
    rec = m.PhaseRecorder(root / "phases.jsonl", {}, auto_export_interval=0.05, retain_decode_tokens=True)
    rec.command("drain")
    # Blank JSONL whitespace creates a slow-to-copy fixture without artificial model events.
    with rec.export_lock:
        rec.export_stream.write(b" " * 8 * 1024 * 1024 + b"\n")
    first = rec.path.stat()
    time.sleep(0.25)
    second = rec.path.stat()
    cmd = ["tar", "--checkpoint=100", "--checkpoint-action=exec=sleep 0.03", "-cf", "/dev/null", "-C", d, "."]
    run = subprocess.run(cmd, capture_output=True, text=True)
    result = dict(
        idle_seconds=0.25,
        size_before=first.st_size,
        size_after=second.st_size,
        mtime_changed=first.st_mtime_ns != second.st_mtime_ns,
        tar_returncode=run.returncode,
        tar_stderr=run.stderr,
        fixture="8MiB blank whitespace; periodic snapshot append; GNUtar slowed at checkpoints; no model/device imports",
        source=a.source,
    )
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
