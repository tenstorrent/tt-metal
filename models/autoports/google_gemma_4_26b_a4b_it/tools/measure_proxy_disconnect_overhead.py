# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""CPU-only paired transport microbenchmark; no model/device requests."""

import argparse
import hashlib
import json
import socket
import statistics
import subprocess
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tti-root", type=Path, required=True)
    parser.add_argument("--baseline", default="4fcfdcfb")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    relative = "llm_module/agentic/abortable_request.py"
    old_source = subprocess.check_output(["git", "show", args.baseline + ":" + relative], cwd=args.tti_root)
    new_source = (args.tti_root / relative).read_bytes()
    modules = {}
    for name, source in (("baseline", old_source), ("candidate", new_source)):
        module = types.ModuleType(name)
        exec(compile(source, name, "exec"), module.__dict__)
        modules[name] = module

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    elapsed = {name: [] for name in modules}
    try:
        for index in range(args.repetitions + 2):
            # Alternate order, with two unrecorded warmup calls per candidate.
            names = list(modules) if index % 2 == 0 else list(reversed(modules))
            for name in names:
                downstream, peer = socket.socketpair()
                try:
                    start = time.perf_counter()
                    result = modules[name].post_until_disconnect(
                        "http://127.0.0.1:{}/v1/chat/completions".format(server.server_port),
                        b"{}",
                        {"Content-Type": "application/json"},
                        3,
                        downstream,
                    )
                    duration = time.perf_counter() - start
                    assert result == (b"{}", 200)
                    if index >= 2:
                        elapsed[name].append(duration)
                finally:
                    downstream.close()
                    peer.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    report = {
        "scope": "host direct-transport overhead, dummy localhost endpoint; not model or suite speed",
        "baseline_ref": args.baseline,
        "baseline_source_sha256": hashlib.sha256(old_source).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(new_source).hexdigest(),
        "repetitions_per_candidate": args.repetitions,
        "warmup_calls_per_candidate": 2,
        "measurements": {
            name: {
                "median_s": statistics.median(values),
                "mean_s": statistics.mean(values),
                "min_s": min(values),
                "max_s": max(values),
            }
            for name, values in elapsed.items()
        },
        "raw_seconds": elapsed,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_seconds"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
