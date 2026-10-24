# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The live disaggregated-prefill server for the tt-blaze GLM-5.3-Flash 48-ring (tt-blaze G53F-0085).

Holds the resident 45-layer mesh model (tt/mesh_model.py, one user slot) and serves the ring over TCP on
``GLM53F_SEED_PORT`` (default 7653). The protocol is tt-blaze ``blaze/models/glm_5_3_flash_4_GLX/seed_transport.py``'s:
frames of ``u32 header length | JSON header | u64 body length | body`` (body = safetensors bytes).

* ``{"op": "register", "layer": L}`` -- a ring rank owning decoder layer L; the server keeps the connection and pushes
  ``{"op": "seed", "n": S', "slot": s, "request": r}`` + that layer's state (the tt-blaze ``prefill_seed`` contract,
  the same tensors ``test_prefill_vs_blaze_golden._save_seed`` writes) and reads back ``{"op": "ack", ...}``.
* ``{"op": "driver"}`` then ``{"op": "prefill", "ids": [...], "slot": s}`` -- the ring's stage 0. The server prefills
  ``ids[:S']`` with S' = 4 * floor((len(ids) - 1) / 4), exports every layer's state, pushes it, waits for every ack and
  answers ``{"op": "done", "S_prime": S', "prefill_s", "export_s", "push_ack_s", "bytes", "next_token", ...}``.
* ``{"op": "stop"}`` -- stop every rank, answer the driver, exit.

    GLM53_MESH_SHAPE=8,4 GLM53F_SEED_PORT=7653 pytest models/demos/glm53_flash/tests/test_prefill_seed_server.py -s
"""

from __future__ import annotations

import json
import os
import select
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors.torch import save

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_prefill_vs_blaze_golden import _dsa_state_host, _kda_state_host
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
PORT = int(os.environ.get("GLM53F_SEED_PORT", "7653"))
MAX_SEQ = int(os.environ.get("GLM53F_PREFILL_MAXSEQ", "8192"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/mnt/tt-data/sdawle/glm53f/prefill/mesh_cache")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
POOL = 4


def _send(sock, header: dict, body: bytes = b"") -> None:
    h = json.dumps(header).encode()
    sock.sendall(struct.pack("<I", len(h)) + h + struct.pack("<Q", len(body)) + body)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 22, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def _recv(sock) -> tuple[dict, bytes]:
    (hl,) = struct.unpack("<I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, hl))
    (bl,) = struct.unpack("<Q", _recv_exact(sock, 8))
    return header, _recv_exact(sock, bl) if bl else b""


@pytest.mark.timeout(86400)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prefill_seed_server(mesh_device):
    layers = list(range(45))
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    t0 = time.time()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=1,
        max_seq_len=MAX_SEQ,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    logger.info(f"[seed-server] model built in {time.time() - t0:.0f} s; warm-up prefill")
    st, lg, _ = model.prefill(torch.tensor([785, 6722, 315, 9621], dtype=torch.int64), user=0)  # compile the programs
    ttnn.deallocate(st)
    ttnn.deallocate(lg)
    decs = {d.layer_idx: d for d in model.layers}

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(64)
    logger.info(f"[seed-server] READY on port {PORT} (max_seq_len {MAX_SEQ})")
    ranks: dict[int, socket.socket] = {}
    driver = None
    pending = [srv]
    request = 0
    pool = ThreadPoolExecutor(max_workers=48)
    while True:
        readable, _, _ = select.select(pending + ([driver] if driver else []), [], [])
        for s in readable:
            if s is srv:
                c, addr = srv.accept()
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hdr, _ = _recv(c)
                if hdr["op"] == "register":
                    ranks[int(hdr["layer"])] = c
                    logger.info(
                        f"[seed-server] layer {hdr['layer']} registered from {hdr.get('host', addr)} ({len(ranks)}/45)"
                    )
                elif hdr["op"] == "driver":
                    driver = c
                    logger.info(f"[seed-server] driver connected from {addr}")
                continue
            hdr, _ = _recv(s)  # the driver
            if hdr["op"] == "stop":
                for c in ranks.values():
                    _send(c, {"op": "stop"})
                _send(s, {"op": "stopped"})
                logger.info("[seed-server] stop")
                return
            if hdr["op"] == "verify":  # the per-request check: this mesh's own next token after the whole prompt
                st, lg, owner = model.prefill(torch.tensor([int(t) for t in hdr["ids"]], dtype=torch.int64), user=0)
                nt = model.first_token(lg, owner)
                ttnn.deallocate(lg)
                ttnn.deallocate(st)
                _send(s, {"op": "verified", "next_token": nt})
                continue
            assert hdr["op"] == "prefill", hdr
            ids = [int(t) for t in hdr["ids"]]
            slot = int(hdr.get("slot", 0))
            sp = POOL * ((len(ids) - 1) // POOL)
            request += 1
            rep = {"op": "done", "S_prime": sp, "request": request, "prompt_len": len(ids)}
            if sp == 0:
                _send(s, rep)
                continue
            while len(ranks) < len(layers):  # every decoder rank must be registered before the first push
                c, addr = srv.accept()
                h2, _ = _recv(c)
                ranks[int(h2["layer"])] = c
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            st, lg, owner = model.prefill(torch.tensor(ids[:sp], dtype=torch.int64), user=0)
            nxt = model.first_token(lg, owner)
            ttnn.synchronize_device(mesh_device)
            t2 = time.perf_counter()
            ttnn.deallocate(lg)
            ttnn.deallocate(st)
            # device -> host export on this thread (ttnn reads stay single-threaded), then the pushes in parallel
            bodies = {}
            for L in layers:
                dec = decs[L]
                state = _dsa_state_host(model, dec, sp) if dec.cache is not None else _kda_state_host(model, dec)
                bodies[L] = save({k: v.contiguous() for k, v in state.items()})
            t_exp = time.perf_counter()

            def push(L):
                tx = time.perf_counter()
                _send(ranks[L], {"op": "seed", "n": sp, "slot": slot, "request": request}, bodies[L])
                ack, _ = _recv(ranks[L])
                return L, len(bodies[L]), time.perf_counter() - tx, ack

            outs = list(pool.map(push, layers))
            t3 = time.perf_counter()
            nbytes = sum(o[1] for o in outs)
            rep.update(
                prefill_s=round(t2 - t1, 4),
                export_s=round(t_exp - t2, 4),
                push_ack_s=round(t3 - t_exp, 4),
                push_ack_s_max=round(max(o[2] for o in outs), 4),
                write_s_max=round(max(o[3].get("write_s", 0) for o in outs), 4),
                bytes=nbytes,
                next_token=nxt,
                server_s=round(t3 - t1, 4),
            )
            logger.info(f"[seed-server] request {request}: {rep}")
            _send(s, rep)
