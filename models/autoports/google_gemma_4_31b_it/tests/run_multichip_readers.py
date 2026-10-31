"""Serialized real-operand reader microprofiles; invoke only with an idle mesh."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

base = Path("models/autoports/google_gemma_4_31b_it")
module = "models.autoports.google_gemma_4_31b_it.tests."
for layer, kind in ((0, "sliding_attention"), (5, "full_attention")):
    out = base / "doc/optimized_multichip_decoder/readers" / kind
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1", TT_METAL_TRACE_ALLOC_TRACKING="0")
    env.pop("TT_METAL_WATCHER", None)
    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-v",
        "--no-web-server",
        "--op-support-count",
        "16384",
        "-o",
        str(out / "profile"),
        "-m",
        module + "profile_multichip_readers",
        "--layer",
        str(layer),
        "--result",
        str(out / "host.json"),
    ]
    files = [
        base / "tt/multichip_decoder.py",
        base / "tt/optimized_decoder.py",
        base / "tests/profile_multichip_readers.py",
    ]
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                command=cmd,
                source_hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                watcher=False,
                profiler=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print("RUN", kind, flush=True)
    with (out / "profile.log").open("w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600, check=True)
    csvs = list((out / "profile/reports").glob("*/ops_perf_results_*.csv"))
    assert len(csvs) == 1, csvs
    subprocess.run(
        [
            sys.executable,
            "-m",
            module + "summarize_multichip_readers",
            str(csvs[0]),
            str(out / "host.json"),
            str(out / "performance.json"),
        ],
        check=True,
    )
    print("PASS", kind, flush=True)
