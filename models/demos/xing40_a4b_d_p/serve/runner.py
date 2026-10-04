# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The serve stack's runner process: tt-metal's prefill_runner.main, unchanged, with two hooks (serve/README.md).

    python -m models.demos.xing40_a4b_d_p.serve.runner <run_dir>

  - build_runtime: sets the runtime's hidden_sink (tt/runners/adapter.py). After every chunk the engine sends (its 40
    layer acks already enqueued), the last layer's output goes through the final norm on the device; the row of the
    chunk's last real token (actual_end - 1, found through the server's row order, tt/layout.py:server_order) comes to
    the host and through the LM head (fp32, on the CPU, as the runner smoke test's stand-in decode does). The logits
    land in XING_SERVE_LOGITS_DIR as s<slot>_e<end>_<seq>.bin: one JSON header line (seq, slot, start, end, t_ns,
    argmax) then vocab float32 values; written to a temp name and renamed, so a reader never sees half a file. The
    front end (serve/server.py) takes the file of its request's (slot, PREFILL_DONE position) and deletes it; the
    files of chunks nobody reads (all but a prompt's last) are swept after 120 s.
  - run_request_loop: starts the engine daemon (XING_SERVE_DAEMON_CMD, an sh command built by serve/server.py:
    engine_daemon.py under tt-d-gen's Python, then the shutdown sentinel; its log <run_dir>/engine.log, its exit
    code <run_dir>/engine.rc) once the H2D service exists, then runs the runner's own loop until the sentinel.
Environment: the PREFILL_* settings serve/server.py sets (those of the runner tests: FABRIC_2D, D2H layer acks, mock
migration on the migration-enabled path).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

SWEEP_S = 120.0


def log(msg: str) -> None:
    print(f"[serve runner] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def logits_sink(rt, out_dir: Path):
    """hidden_sink for XingPrefillRuntime: last-token logits of every chunk -> out_dir (see module docstring)."""
    import torch

    import ttnn
    from models.demos.xing40_a4b_d_p.reference.weights import WeightLoader
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_host, server_order
    from models.demos.xing40_a4b_d_p.tt.runners.adapter import resolve_model_path

    t0 = time.time()
    lm_head = WeightLoader(resolve_model_path()).get("lm_head.weight").float()
    log(f"LM head {tuple(lm_head.shape)} fp32 on the host in {time.time() - t0:.0f} s")
    chunk, sp = rt.config.chunk_size, rt.config.mesh_shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    st = {"seq": 0, "swept": time.time()}

    def sink(h, slot: int, start: int, end: int) -> None:
        t = time.time()
        hidden = rt.model.final_norm(h)
        host = col_split_to_host(rt.mesh_device, hidden)  # [chunk, hidden] in the server's row order
        ttnn.deallocate(hidden)
        order = server_order(start, chunk, sp)
        row = int((order == end - 1 - start).nonzero().flatten()[0])
        logits = torch.nn.functional.linear(host[row].float(), lm_head).contiguous()
        st["seq"] += 1
        head = {
            "seq": st["seq"],
            "slot": slot,
            "start": start,
            "end": end,
            "t_ns": time.time_ns(),
            "argmax": int(logits.argmax()),
            "finite": bool(torch.isfinite(logits).all()),
        }
        name = f"s{slot}_e{end}_{st['seq']}.bin"
        tmp = out_dir / f".{name}.tmp"
        with open(tmp, "wb") as f:
            f.write((json.dumps(head) + "\n").encode())
            f.write(logits.numpy().tobytes())
        os.replace(tmp, out_dir / name)
        log(
            f"chunk slot {slot} [{start}, {end}): next {head['argmax']}"
            f"{'' if head['finite'] else ' NON-FINITE LOGITS'} (logits {time.time() - t:.2f} s)"
        )
        if t - st["swept"] > SWEEP_S / 4:
            st["swept"] = t
            for p in out_dir.glob("s*_e*_*.bin"):
                try:
                    if t - p.stat().st_mtime > SWEEP_S:
                        p.unlink()
                except FileNotFoundError:
                    pass

    return sink


def main(run_dir: str) -> int:
    out = Path(run_dir)
    from models.demos.common.prefill.runners import prefill_runner as PR

    daemon_cmd = json.loads(os.environ["XING_SERVE_DAEMON_CMD"])
    logits_dir = Path(os.environ["XING_SERVE_LOGITS_DIR"])
    build = PR.ADAPTER.build_runtime

    def serving_build(**kw):
        rt = build(**kw)
        rt.hidden_sink = logits_sink(rt, logits_dir)
        return rt

    PR.ADAPTER.build_runtime = serving_build
    loop = PR.run_request_loop

    def serving_loop(runtime, kv_caches, *a, **k):
        elog = (out / "engine.log").open("w")
        # In this process group, so the supervisor's group kill takes the daemon too.
        daemon = subprocess.Popen(daemon_cmd, env=dict(os.environ), stdout=elog, stderr=subprocess.STDOUT)
        log(f"engine daemon started (pid {daemon.pid}, log {out / 'engine.log'}); RUNNER_READY")
        try:
            loop(runtime, kv_caches, *a, **k)
            log("shutdown sentinel received; request loop done")
            daemon.wait(timeout=120)
        finally:
            if daemon.poll() is None:
                subprocess.run(["pkill", "-TERM", "-P", str(daemon.pid)])
                daemon.terminate()
                daemon.wait(timeout=30)
            elog.close()

    PR.run_request_loop = serving_loop
    try:
        PR.main()
    except BaseException as err:
        log(f"runner: {type(err).__name__}: {err}\n{traceback.format_exc()}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
