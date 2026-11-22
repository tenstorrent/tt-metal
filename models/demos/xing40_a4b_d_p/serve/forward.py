# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Byte-level TCP forwarder: expose the serve stack's port on a port the container publishes (serve/README.md).

    python models/demos/xing40_a4b_d_p/serve/forward.py [--listen 5555] [--to 8000]

On the bh-lb-17 reservation container, host port 54210 (P_USER_DBD_PORT) is published to container port 5555, so
with this running the chat is at http://bh-lb-17:54210/. Bytes are copied both ways untouched (SSE streams too).
Writes its pid to generated/xing_serve/forward.pid; stop with `kill $(cat generated/xing_serve/forward.pid)`.
"""

import argparse
import asyncio
import os
import signal
import time
from pathlib import Path


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def main(listen: int, to: int, pid_file: Path):
    async def handle(cr, cw):
        peer = cw.get_extra_info("peername")
        try:
            ur, uw = await asyncio.open_connection("127.0.0.1", to)
        except OSError as e:
            print(f"[forward] {time.strftime('%H:%M:%S')} {peer}: upstream :{to} down ({e})", flush=True)
            cw.close()
            return
        print(f"[forward] {time.strftime('%H:%M:%S')} {peer[0]}", flush=True)
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))

    srv = await asyncio.start_server(handle, "0.0.0.0", listen)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    print(f"[forward] 0.0.0.0:{listen} -> 127.0.0.1:{to} (pid {os.getpid()})", flush=True)
    stop = asyncio.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(s, stop.set)
    async with srv:
        await stop.wait()
    pid_file.unlink(missing_ok=True)
    print("[forward] stopped", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=5555)
    ap.add_argument("--to", type=int, default=int(os.environ.get("XING_SERVE_PORT", "8000")))
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    asyncio.run(
        main(a.listen, a.to, Path(os.environ.get("XING_SERVE_DIR", root / "generated" / "xing_serve")) / "forward.pid")
    )
