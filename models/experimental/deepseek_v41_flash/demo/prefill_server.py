# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated prefill server: metal_2's whole-mesh DeepSeek-V4.1-Flash prefill (layers 0-20, ``tt/prefill/d_p_bridge.py``)
on this host's Galaxy, serving decode servers over TCP (protocol: ``tt/prefill/d_p_wire.py``).

A decode server sends a prompt's ids; the server prefills them from position 0 and sends back the KV cache entries
decode needs: the window rings of layers 0-20 and the compressed KV and index keys of the KV source layers, which the
decode server commits into the session's caches (``PREFILL_D_P_INTEGRATION.md``) and then replays the prompt's last
``window`` tokens itself. Any number of decode servers may stay connected; their prefills run one at a time.

Run (from the repo root, ttnn venv)::

    BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml BRINGUP_HF=... \\
      python models/experimental/deepseek_v41_flash/demo/prefill_server.py --port 9041

and try it with ``demo/prefill_client.py --server HOST:9041``.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

import torch
from loguru import logger

from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import VERSION, recv_msg, send_msg, send_states

# Decode commits whole 128-token units (its ALIGNMENT); the prompt's remainder is replayed by decode.
ALIGN = 128


class PrefillService:
    """The device side: one :class:`DPPrefill`, one prefill at a time (``lock``)."""

    def __init__(self, d_p):
        self.d_p = d_p
        self.lock = threading.Lock()
        cfg = d_p.cfg
        self.vocab = cfg.vocab_size
        self.info = {
            "version": VERSION,
            "max_seq": d_p.max_seq,
            "chunk": d_p.chunk,
            "align": ALIGN,
            "layers": d_p.layers,
            "kv_sources": {str(i): int(cfg.ratios[i]) for i in d_p.layers if cfg.is_kv_source(i)},
            "window": cfg.window,
            "vocab": cfg.vocab_size,
            "mesh": list(d_p.mesh_device.shape),
        }

    def check(self, ids: torch.Tensor) -> None:
        n = ids.numel()
        if not 0 < n <= self.d_p.max_seq or n % ALIGN:
            raise ValueError(f"prompt of {n} tokens: send a positive multiple of {ALIGN} up to {self.d_p.max_seq}")
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self.vocab:
            raise ValueError(f"token ids span [{lo}, {hi}], the vocabulary is [0, {self.vocab})")

    def prefill(self, conn: socket.socket, ids: torch.Tensor, peer: str) -> None:
        self.check(ids)
        gone = []

        def progress(_start, end, _seconds):
            # A client that hung up must not stop the prefill midway: the device work is already enqueued.
            if not gone:
                try:
                    send_msg(conn, {"progress": end})
                except OSError as e:
                    gone.append(e)

        t_wait = time.perf_counter()
        with self.lock:
            queued = time.perf_counter() - t_wait
            t0 = time.perf_counter()
            n = self.d_p.prefill(ids.tolist(), on_chunk=progress)
            seconds = time.perf_counter() - t0
            parts: dict = {}
            host = self.d_p.host_states(n, parts)
            readback = time.perf_counter() - t0 - seconds
        if gone:
            raise ConnectionError(f"{peer} hung up during the prefill: {gone[0]}")
        t0 = time.perf_counter()
        send_states(conn, host, seconds, queued)
        logger.info(
            f"{peer}: {n} tokens prefilled in {seconds:.2f} s ({n / seconds:.0f} tok/s), waited {queued:.2f} s, "
            f"read back in {readback:.2f} s ("
            + ", ".join(f"{k} {v:.2f} s" for k, v in parts.items())
            + f"), sent in {time.perf_counter() - t0:.2f} s"
        )


def _serve_connection(conn: socket.socket, peer: str, service: PrefillService) -> None:
    with conn:
        while True:
            try:
                header, blobs = recv_msg(conn)
            except (ConnectionError, OSError):
                break
            op = header.get("op")
            try:
                if op == "ping":
                    send_msg(conn, {"pong": True})
                elif op == "info":
                    send_msg(conn, service.info)
                elif op == "prefill":
                    if len(blobs) != 1 or len(blobs[0]) % 4:
                        raise ValueError("prefill takes one blob of int32 token ids")
                    service.prefill(conn, torch.frombuffer(blobs[0], dtype=torch.int32), peer)
                else:
                    raise ValueError(f"unknown op {op!r}")
            except (ConnectionError, BrokenPipeError):
                break
            except Exception as e:  # noqa: BLE001 - reported to the client; the server keeps serving
                if isinstance(e, ValueError):
                    logger.warning(f"{peer}: {op} rejected: {e}")
                else:
                    logger.exception(f"{peer}: {op} failed")
                try:
                    send_msg(conn, {"error": f"{type(e).__name__}: {e}"})
                except OSError:
                    break
    logger.info(f"{peer} disconnected")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9041)
    p.add_argument(
        "--max-seq",
        type=int,
        default=1044480,
        help="longest prompt served, rounded up to whole chunks (default: 204 chunks of 5120, metal_2's deployed 1M)",
    )
    p.add_argument(
        "--warm-len", type=int, default=None, help="random prompt run once before serving (default: --max-seq)"
    )
    args = p.parse_args(argv)
    if not os.environ.get("BRINGUP_SPEC"):
        print("set BRINGUP_SPEC (metal_2's galaxy_8x4-full spec) and BRINGUP_HF", file=sys.stderr)
        return 1
    # metal_2's settings give a process that has imported pytest (its model build does) ENGRAM_TABLE=mmap: random reads
    # of the ~190 GB checkpoint tables, which leave a 1M-token prompt host-bound for many minutes. Serve from RAM.
    os.environ.setdefault("DS41_ENGRAM_TABLE", "resident")

    from models.demos.common.bringup.testing.harness import close_mesh, open_mesh
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import DPPrefill

    mesh, fab = open_mesh()
    try:
        d_p = DPPrefill(mesh, max_seq=args.max_seq)
        service = PrefillService(d_p)
        warm = min(args.warm_len if args.warm_len is not None else d_p.max_seq, d_p.max_seq)
        if warm > 0:
            # Random ids: a constant prompt routes every token to the same experts and leaves most MoE programs cold.
            gen = torch.Generator().manual_seed(0)
            t0 = time.perf_counter()
            d_p.prefill(torch.randint(0, service.vocab, (warm,), generator=gen).tolist())
            logger.info(f"warmed {warm} tokens in {time.perf_counter() - t0:.0f} s")
        with socket.create_server((args.host, args.port), reuse_port=True) as srv:
            logger.info(
                f"prefill server on {socket.gethostname()}:{args.port}: layers {d_p.layers[0]}-{d_p.layers[-1]}, "
                f"prompts of up to {d_p.max_seq} tokens in multiples of {ALIGN}"
            )
            while True:
                conn, (host, port) = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                peer = f"{host}:{port}"
                logger.info(f"{peer} connected")
                threading.Thread(target=_serve_connection, args=(conn, peer, service), daemon=True).start()
    except KeyboardInterrupt:
        return 0
    finally:
        close_mesh(mesh, fab)


if __name__ == "__main__":
    sys.exit(main())
