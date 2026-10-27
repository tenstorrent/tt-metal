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
* ``{"op": "logits", "ids": [...], "cands": [...], "k": 8}`` -- diagnosis: this mesh's last-position logits after a
  whole prompt (top-k and the named candidates), e.g. to grade a ring divergence against an independent prefill.
* ``{"op": "needle", "n": N, "depth": d, "env": {...}}`` -- diagnosis (G53F-0170): arg's needle prompt of ~N tokens
  (``test_mesh_long_context.build_needle_ids``), its first token / top-5 and whether it is the needle key's first
  token; ``env`` sets runtime knobs first. Prompts longer than ``GLM53F_PREFILL_CHUNK`` use the chunked prefill.
* ``{"op": "stop"}`` -- stop every rank, answer the driver, then wait for the next ring (the server outlives ring runs;
  ``GLM53F_SEED_SERVER_ONESHOT=1`` exits instead). A ring that dies mid-run (a closed rank or driver socket) is dropped
  the same way, and a connection that closes before its first header is ignored.

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
from types import SimpleNamespace

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
#: prompts longer than this go through ``prefill_chunked`` (16 K chunks: the 1 M test's geometry, G53F-0089)
CHUNK = int(os.environ.get("GLM53F_PREFILL_CHUNK", "16384"))
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


#: tt-blaze G53F-0189: "1" = a seed body goes out as safetensors' byte layout built by hand -- the header,
#: then every tensor's own storage as a memoryview -- instead of ``safetensors.torch.save`` (1.27 s per 1 GiB DSA layer on
#: the export thread, 11 layers at 1M) plus ``_send``'s concatenation (another copy). Same bytes on the wire (the ring's
#: ``seed_transport.tensors_from_bytes`` loads them unchanged); "0" (default until the fake-ring measurement) = save +
#: concatenation.
ZERO_COPY = os.environ.get("GLM53F_SEED_ZEROCOPY", "0") == "1"
_ST_DTYPES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def st_parts(tensors: dict) -> list:
    """``safetensors.torch.save(tensors)``'s format as a list of buffers, without copying any tensor's data: the
    8-byte header length + the JSON header (space-padded to 8 bytes), then each tensor's contiguous storage in order
    of decreasing element size, then name (``data_offsets`` consecutive from 0, as the format's validation
    requires; every tensor starts aligned to its element size, as ``safetensors.torch.save`` lays them out, so the
    ring can map them in place with ``torch.frombuffer``). Their concatenation is a safetensors file;
    ``safetensors.torch.load`` reads it back to equal tensors (tests/test_seed_zerocopy.py)."""
    hdr, bufs, off = {}, [], 0
    for k in sorted(tensors, key=lambda k: (-tensors[k].element_size(), k)):
        t = tensors[k].contiguous()
        n = t.numel() * t.element_size()
        hdr[k] = {"dtype": _ST_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + n]}
        if n:
            bufs.append(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        off += n
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return [struct.pack("<Q", len(h)) + h, *bufs]


def parts_len(parts) -> int:
    return (
        parts.nbytes
        if isinstance(parts, memoryview)
        else len(parts)
        if isinstance(parts, bytes)
        else sum(memoryview(p).nbytes for p in parts)
    )


def _send_parts(sock, header: dict, parts: list) -> None:
    """``_send`` of a body given as buffers (``st_parts``): one small prefix, then every buffer as is."""
    h = json.dumps(header).encode()
    sock.sendall(struct.pack("<I", len(h)) + h + struct.pack("<Q", parts_len(parts)))
    for p in parts:
        sock.sendall(p)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 22, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


# tt-blaze G53F-0227: GLM53F_SEED_MTP=1 -- the server also builds the MTP layer's (checkpoint layer 45) prompt state for the
# ring's MTP stage: every chunk's final hidden + the next tokens fill the block's own latent cache / index, exported and
# pushed as "layer 45" (the ring's MTP stage registers that layer). Set by the server at start (``_MTP``).
SEED_MTP = os.environ.get("GLM53F_SEED_MTP", "0") == "1"
_MTP = {"block": None}


def _mtp_hook(model, nxt: list[int]):
    """``prefill_chunked``'s ``on_streams``: chunk [s0, e) -> the MTP rows [s0, e) (row p: H_p = the head's normed final
    hidden, t_{p+1} = ``nxt[p]``) into the MTP block's cache, state only (``TtMTPBlock.prefill_state_rows``)."""
    mtp, rows = _MTP["block"], model.mesh.rows

    def hook(streams, s0: int, e: int):
        T_pad = int(streams.shape[-2]) * rows
        h = model.head.collapse(streams)
        hn = model.head.norm(h)  # [1,1,T_loc,H] bf16 = the LM-head input (the MTP's H)
        if h is not streams:
            ttnn.deallocate(h)
        host = model.embedding.tokens_to_host(torch.tensor(nxt[s0:e], dtype=torch.int64), pad_to=T_pad)
        if rows > 1:
            ids_dev = model.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            ids_dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device)
        mtp.prefill_state_rows(ids_dev, hn, user=0, seq_len=e - s0, start=s0)
        ttnn.deallocate(ids_dev)
        ttnn.deallocate(hn)

    return hook


def _prefill(model, ids: list[int], mtp_next: list[int] | None = None):
    """``(streams, logits_last, owner_row)`` of a whole prompt for slot 0: one eager prefill up to ``CHUNK`` tokens,
    16 K chunks above (``GLM53MeshModel.prefill_chunked``). ``mtp_next`` (with the MTP block built, G53F-0227): the
    next token of every prompt row (``ids[1:] + [t]``), and the MTP state is built chunk by chunk alongside."""
    t = torch.tensor([int(v) for v in ids], dtype=torch.int64)
    chunk = int(os.environ.get("GLM53F_PREFILL_CHUNK", CHUNK))  # per call: the diagnosis ops' env can A/B it
    hook = None
    if _MTP["block"] is not None and mtp_next is not None:
        assert len(mtp_next) == len(ids), (len(mtp_next), len(ids))
        hook = _mtp_hook(model, [int(v) for v in mtp_next])
    if len(ids) > chunk or hook is not None:  # one chunk of prefill_chunked == prefill (start 0, no padding knob)
        return model.prefill_chunked(t, user=0, chunk=chunk, on_streams=hook)
    return model.prefill(t, user=0)


def _warm_tails(model, mesh_device):
    """tt-blaze G53F-0225 (``GLM53F_SEED_WARMUP_TAILS`` = "multi", "single" or "multi+single"; unset = off): create the
    programs of every ragged last-chunk shape before the first request. A chunk's programs are keyed by its padded
    length (32 rows per mesh row granularity, so CHUNK / (32 * rows) shapes per family); a request whose last chunk has
    a shape this process never ran pays its compile (14.35 s vs 0.672 s warm for a 4240-token tail, G53F-0221). "multi":
    one tail-sized ``prefill`` at start CHUNK per shape (a multi-chunk prompt's last chunk; the cache's stale rows below
    the start only feed the throw-away result); "single": the same lengths at start 0 (a prompt of at most one chunk).
    Values are not checked: the results are discarded, only the program cache is filled."""
    fams = [f for f in os.environ.get("GLM53F_SEED_WARMUP_TAILS", "").split("+") if f]
    if not fams:
        return
    unit = 32 * model.mesh.rows
    t_all, n = time.perf_counter(), 0
    for fam in fams:
        assert fam in ("multi", "single"), fam
        start = CHUNK if fam == "multi" else 0
        for k in range(1, CHUNK // unit + (0 if fam == "multi" else 1)):  # a full chunk at CHUNK is the warm-up's
            L = k * unit - 1  # T_loc = 32 k; one padding row on the last mesh row
            if start + L > MAX_SEQ:
                break
            ids = torch.tensor(([785, 6722, 315, 9621] * (L // 4 + 1))[:L], dtype=torch.int64)
            tw = time.perf_counter()
            st, lg, _ = model.prefill(ids, user=0, start=start)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(st)
            if lg is not None:
                ttnn.deallocate(lg)
            n += 1
            logger.info(f"[seed-server] tail warm-up {fam} {L} tokens at {start}: {time.perf_counter() - tw:.3f} s")
    logger.info(f"[seed-server] tail warm-up: {n} shapes in {time.perf_counter() - t_all:.1f} s")


class _SelectionTrace:
    """G53F-0170 diagnosis: every DSA layer's sparse index rows for the last ``rows`` prompt positions -- how many of
    the needle's token positions [lo, hi) each row selects (``TtDSA.on_indices``; mesh row r's slab read on its
    column 0, as ``test_mesh_slab_prefill.IndexRecorder``). Blocks entirely below the traced rows are not read back."""

    def __init__(self, model, S: int, lo: int, hi: int, rows: int = 4):
        self.mesh, self.S, self.lo, self.hi, self.first = model.mesh, S, lo, hi, S - rows
        self.hits: dict[int, dict[int, list[int]]] = {}
        self.errors: list[str] = []
        self.attns = [d.layer.attn for d in model.layers if d.cache is not None]
        for d in model.layers:
            if d.cache is not None:
                d.layer.attn.on_indices = self._hook(d.layer_idx)

    def _hook(self, layer: int):
        def rec(query_start, indices):
            try:  # a diagnosis hook must never take the prefill down with it
                self._record(layer, query_start, indices)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"layer {layer}: {e!r}")

        return rec

    def _record(self, layer: int, query_start, indices):
        starts = query_start if isinstance(query_start, tuple) else (query_start,)
        n = int(indices.shape[-2])
        if max(starts) + n <= self.first:
            return
        devs = self.mesh.device_tensors(indices)
        for r, s0 in enumerate(starts):
            if s0 + n <= self.first:
                continue
            t = ttnn.to_torch(devs[r * self.mesh.cols]).reshape(-1, indices.shape[-1]).to(torch.int64) & 0xFFFFFFFF
            for i in range(t.shape[0]):
                pos = s0 + i
                if self.first <= pos < self.S:
                    row = t[i][t[i] != 0xFFFFFFFF]
                    hit = int(((row >= self.lo) & (row < self.hi)).sum())
                    self.hits.setdefault(layer, {})[pos] = [hit, int(row.numel())]

    def close(self) -> dict:
        for a in self.attns:
            a.on_indices = None
        return {
            "lo": self.lo,
            "hi": self.hi,
            "rows_from": self.first,
            "errors": self.errors[:4],
            "layers": {str(L): {str(p): v for p, v in sorted(h.items())} for L, h in sorted(self.hits.items())},
        }


def _find_sub(ids: list[int], sub: list[int]) -> int:
    """First index of the token run ``sub`` in ``ids`` (-1 when absent)."""
    n = len(sub)
    for i in range(len(ids) - n + 1):
        if ids[i] == sub[0] and ids[i : i + n] == sub:
            return i
    return -1


def _dram_view(mesh_device) -> dict:
    """Per-bank DRAM allocator state in MB (G53F-0171: a request OOMed with 30 MB of contiguous DRAM left per bank
    after ~10 long requests -- leak vs fragmentation is read from this in every reply)."""
    try:
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        mb = lambda b: round(b / 1e6, 1)
        return {
            "dram_alloc_mb": mb(v.total_bytes_allocated_per_bank),
            "dram_free_mb": mb(v.total_bytes_free_per_bank),
            "dram_largest_free_mb": mb(v.largest_contiguous_bytes_free_per_bank),
        }
    except Exception as e:  # noqa: BLE001 -- diagnosis only
        return {"dram_view_error": repr(e)}


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
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(default_snapshot_dir()))
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
    push_layers = list(layers)
    mtp_layer = None  # G53F-0227: the MTP layer this server can seed (GLM53F_SEED_MTP=1)
    if SEED_MTP:
        from models.demos.glm53_flash.tt.mtp import MTP_LAYER, TtMTPBlock

        tm = time.time()
        _MTP["block"] = TtMTPBlock.from_checkpoint(
            mesh_device,
            cfg,
            embedding=model.embedding,
            head=model.head,
            weight_cache_root=CACHE_ROOT,
            mesh=model.mesh,
            max_batch=1,
            max_seq_len=MAX_SEQ,
            cache_only=True,
        )
        push_layers.append(MTP_LAYER)
        mtp_layer = MTP_LAYER
        logger.info(f"[seed-server] MTP layer {MTP_LAYER} cache builder in {time.time() - tm:.0f} s (G53F-0227)")
    # compile the programs at the lengths the ring will ask for (GLM53F_SEED_WARMUP, default 4 and 4092 tokens: the
    # first request at a new length otherwise pays its compile, 4.66 s vs 1.82 s at 4092, G53F-0085)
    # "N" (two passes) or "N:REPS": a long warm-up (e.g. 1040000:1) creates every DSA pools-bucket width's programs and
    # that length's tail before the first request (tt-blaze G53F-0205: a first 1M pass pays ~6 s for the bucket widths)
    for spec_ in [v for v in os.environ.get("GLM53F_SEED_WARMUP", "4,4092").split(",") if v.strip()]:
        n_, _, reps_ = spec_.partition(":")
        n, reps = int(n_), int(reps_ or 2)
        for rep in range(reps):  # the second pass is the warm cost (what a request at this length pays)
            ids = torch.tensor(([785, 6722, 315, 9621] * (n // 4 + 1))[: min(n, MAX_SEQ)], dtype=torch.int64)
            ttnn.synchronize_device(mesh_device)
            tw = time.perf_counter()
            st, lg, _ = _prefill(model, ids.tolist(), mtp_next=ids.tolist()[1:] + ids.tolist()[-1:])
            ttnn.synchronize_device(mesh_device)
            tp = time.perf_counter()
            ttnn.deallocate(st)
            ttnn.deallocate(lg)
            nbytes = 0
            for dec in model.layers:  # the export path too
                st_ = _dsa_state_host(model, dec, len(ids)) if dec.cache is not None else _kda_state_host(model, dec)
                nbytes += sum(v.numel() * v.element_size() for v in st_.values())
            te = time.perf_counter()
            logger.info(
                f"[seed-server] warm-up {len(ids)} tokens pass {rep}: prefill {tp - tw:.3f} s, export {te - tp:.3f} s "
                f"({nbytes / 2**20:.1f} MiB)"
            )
    _warm_tails(model, mesh_device)
    decs = {d.layer_idx: d for d in model.layers}
    if _MTP["block"] is not None:  # the MTP layer exports like a DSA layer: its cache + pooled-key index
        decs[push_layers[-1]] = SimpleNamespace(cache=_MTP["block"].cache, layer_idx=push_layers[-1])

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(64)
    logger.info(f"[seed-server] READY on port {PORT} (max_seq_len {MAX_SEQ}) {_dram_view(mesh_device)}")
    ranks: dict[int, socket.socket] = {}
    driver = None
    pending = [srv]
    request = 0
    pool = ThreadPoolExecutor(max_workers=48)
    oneshot = os.environ.get("GLM53F_SEED_SERVER_ONESHOT", "0") == "1"

    def _accept_one():
        c, addr = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            hdr, _ = _recv(c)
        except (ConnectionError, OSError, ValueError) as e:  # a probe or a dead peer: not a ring rank
            logger.warning(f"[seed-server] connection from {addr} closed before its header ({e!r})")
            c.close()
            return None, None, addr
        return c, hdr, addr

    def _drop_ring(why: str):
        nonlocal driver
        logger.warning(f"[seed-server] dropping the ring ({why}): {len(ranks)} ranks, driver {driver is not None}")
        for c in list(ranks.values()) + ([driver] if driver else []):
            try:
                c.close()
            except OSError:
                pass
        ranks.clear()
        driver = None

    while True:
        readable, _, _ = select.select(pending + ([driver] if driver else []), [], [])
        for s in readable:
            if s is srv:
                c, hdr, addr = _accept_one()
                if c is None:
                    continue
                if hdr["op"] == "register":
                    ranks[int(hdr["layer"])] = c
                    if int(hdr["layer"]) not in push_layers:  # e.g. an MTP ring's layer 45 on a server without MTP
                        logger.warning(
                            f"[seed-server] layer {hdr['layer']} registered but this server does not seed it"
                        )
                    logger.info(
                        f"[seed-server] layer {hdr['layer']} registered from {hdr.get('host', addr)} "
                        f"({sum(L in ranks for L in push_layers)}/{len(push_layers)})"
                    )
                elif hdr["op"] == "driver":
                    driver = c
                    logger.info(f"[seed-server] driver connected from {addr}")
                continue
            try:
                hdr, _ = _recv(s)  # the driver
            except (ConnectionError, OSError, ValueError) as e:
                _drop_ring(f"driver socket: {e!r}")
                continue
            if hdr["op"] == "stop":
                for c in ranks.values():
                    try:
                        _send(c, {"op": "stop"})
                    except OSError:
                        pass
                _send(s, {"op": "stopped"})
                logger.info("[seed-server] stop")
                if oneshot:
                    return
                _drop_ring("stop")
                logger.info(f"[seed-server] READY for the next ring on port {PORT}")
                continue
            if hdr["op"] == "needle":  # G53F-0170 diagnosis: a needle prompt (arg's builder) -> the first token + top-5
                from models.demos.glm53_flash.tests import test_mesh_long_context as LC

                for k_, v_ in (hdr.get("env") or {}).items():  # runtime knobs read per call (e.g. GLM53_TOPK_SEGMENTS)
                    if v_ is None:
                        os.environ.pop(k_, None)
                    else:
                        os.environ[k_] = str(v_)
                ids, _text = LC.build_needle_ids(tok, int(hdr["n"]), float(hdr["depth"]))
                trace = None
                if hdr.get("trace"):  # the selection of the last rows vs the needle's tokens (G53F-0170)
                    key = tok.encode(LC.NEEDLE_KEY, add_special_tokens=False)
                    at = _find_sub(ids, key)
                    if at < 0:  # mid-sentence: the key tokenizes with its leading space
                        key = tok.encode(" " + LC.NEEDLE_KEY, add_special_tokens=False)
                        at = _find_sub(ids, key)
                    trace = _SelectionTrace(model, len(ids), at, at + len(key), int(hdr["trace"]))
                ttnn.synchronize_device(mesh_device)
                tn = time.perf_counter()
                st, lg, owner = _prefill(model, ids)
                ttnn.synchronize_device(mesh_device)
                pre_s = time.perf_counter() - tn
                shard = model.mesh.device_tensors(lg)[owner * model.mesh.cols]
                vec = ttnn.to_torch(shard).float().reshape(-1)[: cfg.vocab_size]
                ttnn.deallocate(lg)
                ttnn.deallocate(st)
                top = torch.topk(vec, 5)
                want = tok.encode(LC.NEEDLE_KEY, add_special_tokens=False)[0]
                rep = {
                    "op": "needle",
                    "S": len(ids),
                    "depth": float(hdr["depth"]),
                    "prefill_s": round(pre_s, 3),
                    "top_ids": top.indices.tolist(),
                    "top_vals": top.values.tolist(),
                    "top_text": [tok.decode([int(i)]) for i in top.indices.tolist()],
                    "key_token": int(want),
                    "key_logit": float(vec[want]),
                    "found": int(top.indices[0]) == int(want),
                    "env": hdr.get("env") or {},
                    "trace": trace.close() if trace is not None else None,
                    **_dram_view(mesh_device),
                }
                logger.info(f"[seed-server] needle: {rep}")
                _send(s, rep)
                continue
            if hdr["op"] == "logits":  # diagnosis: this mesh's last-position logits after ``ids``
                tr = hdr.get("trace")
                trace = (
                    _SelectionTrace(model, len(hdr["ids"]), int(tr["lo"]), int(tr["hi"]), int(tr.get("rows", 4)))
                    if tr
                    else None
                )
                st, lg, owner = _prefill(model, hdr["ids"])
                shard = model.mesh.device_tensors(lg)[owner * model.mesh.cols]  # every chip holds the full vocab row
                vec = ttnn.to_torch(shard).float().reshape(-1)[: cfg.vocab_size]
                ttnn.deallocate(lg)
                ttnn.deallocate(st)
                top = torch.topk(vec, int(hdr.get("k", 8)))
                _send(
                    s,
                    {
                        "op": "logits",
                        "top_ids": top.indices.tolist(),
                        "top_vals": top.values.tolist(),
                        "cands": {str(int(t)): float(vec[int(t)]) for t in hdr.get("cands", [])},
                        "trace": trace.close() if trace is not None else None,
                        **_dram_view(mesh_device),
                    },
                )
                continue
            if hdr["op"] == "verify":  # the per-request check: this mesh's own next token after the whole prompt
                st, lg, owner = _prefill(model, hdr["ids"])
                nt = model.first_token(lg, owner)
                ttnn.deallocate(lg)
                ttnn.deallocate(st)
                _send(s, {"op": "verified", "next_token": nt})
                continue
            assert hdr["op"] == "prefill", hdr
            ids = [int(t) for t in hdr["ids"]]
            slot = int(hdr.get("slot", 0))
            sp = POOL * ((len(ids) - 1) // POOL)  # < len(ids): the MTP rows [0, sp) need t_1 .. t_sp, all prompt tokens
            request += 1
            rep = {"op": "done", "S_prime": sp, "request": request, "prompt_len": len(ids)}
            if sp == 0:
                _send(s, rep)
                continue
            # every trunk rank must be registered before the first push. The MTP layer is seeded only when the ring
            # registered it (an MTP ring builds stage 47 before its driver connects; a plain ring has no layer 45), so
            # one MTP server serves plain and MTP rings, and a plain ring skips the MTP build (G53F-0227)
            trunk = [L for L in push_layers if L != mtp_layer]
            while not all(L in ranks for L in trunk):
                c, h2, addr = _accept_one()
                if c is not None and h2["op"] == "register":
                    ranks[int(h2["layer"])] = c
            want_mtp = mtp_layer is not None and mtp_layer in ranks
            req_layers = trunk + ([mtp_layer] if want_mtp else [])
            rep["mtp"] = int(want_mtp)
            # tt-blaze G53F-0198: a test driver may A/B prefill knobs read per call (GLM53_INDEXER_POOL_BUCKET,
            # GLM53_MHC_PREFILL_HMIX, GLM53_PREFILL_ABLOCK, ...) with "opts": {"env": {name: value}}, set for this
            # request only; the ring's driver sends none
            env_opts = {str(k): str(v) for k, v in ((hdr.get("opts") or {}).get("env") or {}).items()}
            env_old = {k: os.environ.get(k) for k in env_opts}
            os.environ.update(env_opts)
            if env_opts:
                rep["env"] = env_opts
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            try:
                st, lg, owner = _prefill(model, ids[:sp], mtp_next=ids[1 : sp + 1] if want_mtp else None)
            finally:
                for k, v in env_old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            nxt = model.first_token(lg, owner)
            ttnn.synchronize_device(mesh_device)
            t2 = time.perf_counter()
            ttnn.deallocate(lg)
            ttnn.deallocate(st)
            # device -> host export on this thread (ttnn reads stay single-threaded), then the pushes in parallel
            bodies = {}

            def push(L):
                tx = time.perf_counter()
                seed_hdr = {"op": "seed", "n": sp, "slot": slot, "request": request}
                if isinstance(bodies[L], list):
                    _send_parts(ranks[L], seed_hdr, bodies[L])
                else:
                    _send(ranks[L], seed_hdr, bodies[L])
                ack, _ = _recv(ranks[L])
                return L, parts_len(bodies[L]), time.perf_counter() - tx, ack

            # GLM53F_SEED_OVERLAP=1 (default): each layer's push starts as soon as its export is done (the pushes are
            # socket I/O on the pool's threads; the ttnn reads stay on this thread), so push_ack_s is only the tail
            # after the last export (G53F-0171: 8.3 s export + 8.8 s push at 1M back to back); 0 = export all first
            overlap = os.environ.get("GLM53F_SEED_OVERLAP", "1") == "1"
            # G53F-0189: a test driver (prefill/fake_ring.py) may A/B the export per request with "opts"
            # {"zerocopy": 0|1, "export_partition": 0|1}; the ring's driver sends none (the env defaults apply)
            opts = hdr.get("opts") or {}
            zc = bool(int(opts.get("zerocopy", ZERO_COPY)))
            part_env = os.environ.get("GLM53F_EXPORT_PARTITION")
            if "export_partition" in opts:
                os.environ["GLM53F_EXPORT_PARTITION"] = str(int(opts["export_partition"]))
            rep.update(zerocopy=int(zc), export_partition=int(os.environ.get("GLM53F_EXPORT_PARTITION", "0") == "1"))
            futs = []
            # tt-blaze G53F-0219: the 11 DSA layers first (GLM53F_SEED_DSA_FIRST=1, default): their 1 GiB pushes + ring
            # writes are the tail after the last export, and the 34 small KDA exports (~0.9 s, flat in S) then run
            # under that tail instead of before it. Same bytes per layer; only the order of the pushes changes.
            order = req_layers
            if os.environ.get("GLM53F_SEED_DSA_FIRST", "1") == "1":
                order = [L for L in req_layers if decs[L].cache is not None] + [
                    L for L in req_layers if decs[L].cache is None
                ]
            try:
                for L in order:
                    dec = decs[L]
                    state = _dsa_state_host(model, dec, sp) if dec.cache is not None else _kda_state_host(model, dec)
                    bodies[L] = st_parts(state) if zc else save({k: v.contiguous() for k, v in state.items()})
                    if overlap:
                        futs.append(pool.submit(push, L))
            finally:
                if part_env is None:
                    os.environ.pop("GLM53F_EXPORT_PARTITION", None)
                else:
                    os.environ["GLM53F_EXPORT_PARTITION"] = part_env
            t_exp = time.perf_counter()

            try:
                outs = [f.result() for f in futs] if overlap else list(pool.map(push, order))
            except (ConnectionError, OSError, ValueError) as e:  # a rank died mid-push: this ring is gone
                try:
                    _send(s, {"op": "error", "request": request, "error": repr(e)})
                except OSError:
                    pass
                _drop_ring(f"push failed: {e!r}")
                continue
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
                **_dram_view(mesh_device),
            )
            logger.info(f"[seed-server] request {request}: {rep}")
            _send(s, rep)
