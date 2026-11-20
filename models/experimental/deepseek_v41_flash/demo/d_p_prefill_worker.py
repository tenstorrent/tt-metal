# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated prefill worker: metal_2's whole-mesh DeepSeek-V4.1-Flash prefill (layers 0-20, ``tt/prefill/d_p_bridge.py``)
on this host's Galaxy, serving decode servers on other hosts over TCP (``tt/prefill/d_p_wire.py``).

A decode server started with ``--prefill-backend d_p --d-p-worker HOST:PORT`` sends each prompt's ids; the worker
prefills them from position 0 and sends back the window rings, compressed KV and index keys of layers 0-20, which the
server commits into the session's decode caches (``PREFILL_D_P_INTEGRATION.md``). One request runs at a time.

Run (from the repo root, ttnn venv)::

    BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml BRINGUP_HF=... \\
      python models/experimental/deepseek_v41_flash/demo/d_p_prefill_worker.py --port 9041 --max-seq 131072
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import torch
from loguru import logger

from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import recv_msg, send_msg, send_states


def _serve_connection(conn: socket.socket, d_p) -> None:
    while True:
        try:
            header, blobs = recv_msg(conn)
        except ConnectionError:
            return
        op = header.get("op")
        try:
            if op == "info":
                send_msg(
                    conn,
                    {
                        "max_seq": d_p.max_seq,
                        "chunk": d_p.chunk,
                        "layers": d_p.layers,
                        "mesh": list(d_p.mesh_device.shape),
                    },
                )
            elif op == "prefill":
                ids = torch.frombuffer(blobs[0], dtype=torch.int32).tolist()
                t0 = time.perf_counter()
                n = d_p.prefill(ids, on_chunk=lambda _s, end, _sec: send_msg(conn, {"progress": end}))
                seconds = time.perf_counter() - t0
                host = d_p.host_states(n)
                logger.info(f"prefill: {n} tokens in {seconds:.2f} s = {n / seconds:.0f} tok/s")
                t0 = time.perf_counter()
                send_states(conn, host, seconds)
                logger.info(f"states of {n} tokens sent in {time.perf_counter() - t0:.2f} s")
            else:
                send_msg(conn, {"error": f"unknown op {op!r}"})
        except (ConnectionError, BrokenPipeError):
            return
        except Exception as e:  # noqa: BLE001 - reported to the client, the worker keeps serving
            logger.exception(f"{op} failed")
            send_msg(conn, {"error": f"{type(e).__name__}: {e}"})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9041)
    p.add_argument("--max-seq", type=int, default=131072, help="longest prompt served (rounded up to whole chunks)")
    p.add_argument(
        "--warm-len", type=int, default=None, help="dummy prompt run once before serving (default: --max-seq)"
    )
    args = p.parse_args(argv)
    if not os.environ.get("BRINGUP_SPEC"):
        print("set BRINGUP_SPEC (metal_2's galaxy_8x4-full spec) and BRINGUP_HF", file=sys.stderr)
        return 1

    from models.demos.common.bringup.testing.harness import close_mesh, open_mesh
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import DPPrefill

    mesh, fab = open_mesh()
    try:
        d_p = DPPrefill(mesh, max_seq=args.max_seq)
        warm = min(args.warm_len if args.warm_len is not None else d_p.max_seq, d_p.max_seq)
        if warm > 0:
            # Random ids: a constant prompt routes every token to the same experts and leaves most MoE programs cold.
            gen = torch.Generator().manual_seed(0)
            t0 = time.perf_counter()
            d_p.prefill(torch.randint(0, d_p.cfg.vocab_size, (warm,), generator=gen).tolist())
            logger.info(f"warmed {warm} tokens in {time.perf_counter() - t0:.0f} s")
        with socket.create_server((args.host, args.port), reuse_port=True) as srv:
            logger.info(f"d_p prefill worker on {socket.gethostname()}:{args.port}, prompts up to {d_p.max_seq} tokens")
            while True:
                conn, peer = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info(f"decode server {peer[0]}:{peer[1]} connected")
                with conn:
                    _serve_connection(conn, d_p)
                logger.info(f"decode server {peer[0]}:{peer[1]} disconnected")
    except KeyboardInterrupt:
        return 0
    finally:
        close_mesh(mesh, fab)


if __name__ == "__main__":
    sys.exit(main())
