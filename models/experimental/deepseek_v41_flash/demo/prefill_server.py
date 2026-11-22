# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated prefill server: metal_2's whole-mesh DeepSeek-V4.1-Flash prefill (layers 0-20, ``tt/prefill/d_p_bridge.py``)
on this host's Galaxy, serving decode servers over TCP (protocol: ``tt/prefill/d_p_wire.py``).

A decode server sends a prompt's ids; the server prefills them and sends back the KV cache entries decode needs: the
window rings of layers 0-20 and the compressed KV and index keys of the KV source layers, which the decode server
commits into the session's caches (``PREFILL_D_P_INTEGRATION.md``) and then replays the prompt's last ``window``
tokens itself. Any number of decode servers may stay connected; their prefills run one at a time.

Prefix cache (``PREFILL_D_P_PREFIX_CACHE.md``): every prompt's state stays in one of ``--cache-slots`` device cache
slots until a prompt that continues none of them needs the least recently used slot. A prompt that continues a kept
one (the next turn of a conversation, the same prompt again) resumes from where the kept prompt ended instead of
position 0, and the reply says how many tokens came from the cache (``cached``). Independently, a decode server
that already holds a prompt's first ``have`` tokens' entries gets only the later ones.

Run (from the repo root, ttnn venv)::

    BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml BRINGUP_HF=... \\
      python models/experimental/deepseek_v41_flash/demo/prefill_server.py --port 9041

and try it with ``demo/prefill_client.py --server HOST:9041`` (``--prefix-check`` for the cache).
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

from models.experimental.deepseek_v41_flash.tt.prefill.d_p_push import FabricPush
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import VERSION, recv_msg, send_msg, send_states

# Decode commits whole 128-token units (its ALIGNMENT); the prompt's remainder is replayed by decode.
ALIGN = 128


class PrefixCache:
    """Host bookkeeping of the device cache slots: the ids each slot's state is of, and when it was last used."""

    def __init__(self, slots: int):
        self.ids: list = [None] * slots  # int32 tensor of the prompt slot k holds, None: nothing kept
        self.used = [0] * slots
        self.clock = 0

    def lookup(self, d_p, ids: torch.Tensor) -> tuple[int, int]:
        """(slot, start): the slot to prefill ``ids`` into and the position to start from. A hit is the kept prompt
        sharing the longest prefix with ``ids`` that ``d_p`` can continue exactly (``DPPrefill.resume_start``); a
        miss (start 0) takes an empty slot, else the least recently used one."""
        hits = []  # (start, last use, slot): the longest continuation, then the most recently used slot
        for k, held in enumerate(self.ids):
            if held is None:
                continue
            n = min(held.numel(), ids.numel())
            diff = (held[:n] != ids[:n]).nonzero()
            common = int(diff[0]) if diff.numel() else n
            if common >= ALIGN:
                r = d_p.resume_start(k, common, ids.numel())
                if r > 0:
                    hits.append((r, self.used[k], k))
        if hits:
            r, _, k = max(hits)
            return k, r
        return self.victim(), 0

    def victim(self) -> int:
        empty = [k for k, held in enumerate(self.ids) if held is None]
        return empty[0] if empty else min(range(len(self.ids)), key=self.used.__getitem__)

    def keep(self, slot: int, ids: torch.Tensor | None) -> None:
        """Slot ``slot`` now holds the state of ``ids`` (None: nothing usable)."""
        self.ids[slot] = None if ids is None else ids.clone()
        self.touch(slot)

    def touch(self, slot: int) -> None:
        self.clock += 1
        self.used[slot] = self.clock


class StageTotals:
    """Seconds per stage of every request served so far (the ``stats`` op), and how many requests."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.tokens = 0
        self.seconds: dict[str, float] = {}

    def add(self, tokens: int, stages: dict[str, float]) -> None:
        with self.lock:
            self.requests += 1
            self.tokens += tokens
            for stage, s in stages.items():
                self.seconds[stage] = self.seconds.get(stage, 0.0) + s

    def snapshot(self) -> dict:
        with self.lock:
            return {"requests": self.requests, "tokens": self.tokens, "seconds": dict(self.seconds)}


class PrefillService:
    """The device side: one :class:`DPPrefill` and its prefix cache, one prefill at a time (``lock``).

    Every request is profiled by stage (seconds, logged one line a request and summed in :attr:`totals`):
    ``queued`` waiting for the device, ``lookup`` the prefix cache, ``prefill.input`` / ``prefill.enqueue`` building
    and issuing the chunks' programs on the host, ``prefill.drain`` waiting for the device to finish them (all of it
    after the last chunk unless ``sync_chunks``, which waits after every chunk so the split is exact), then
    ``readback.*`` (TCP: reading the states to the host) or ``push.*`` (the fabric push, a launch after every chunk
    and one after the last, ``FabricPush``: ``begin`` mapping the targets, then summed over the launches ``plan``
    building the runs, ``enqueue`` dispatching the programs, ``stream`` writing the runs into the H2D sockets (the
    per-chunk launches happen during the prefill, outside ``prefill.*``), ``window`` reading and staging the rings,
    ``device`` waiting for the last launch), and ``reply`` sending the answer. ``total`` is from taking the request to
    the reply sent."""

    def __init__(self, d_p, sync_chunks: bool = False):
        self.d_p = d_p
        self.sync_chunks = sync_chunks
        self.pusher: FabricPush | None = None  # made by the first prefill_push (it holds an H2D socket a chip)
        self.totals = StageTotals()
        self.lock = threading.Lock()
        self.cache = PrefixCache(len(d_p.slots))
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
            "cache_slots": len(d_p.slots),
        }

    def check(self, ids: torch.Tensor, have: int) -> None:
        n = ids.numel()
        if not 0 < n <= self.d_p.max_seq or n % ALIGN:
            raise ValueError(f"prompt of {n} tokens: send a positive multiple of {ALIGN} up to {self.d_p.max_seq}")
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self.vocab:
            raise ValueError(f"token ids span [{lo}, {hi}], the vocabulary is [0, {self.vocab})")
        if not 0 <= have <= n or have % ALIGN:
            raise ValueError(f"have {have}: a multiple of {ALIGN} up to the prompt's {n} tokens")

    def prefill(
        self, conn: socket.socket, ids: torch.Tensor, have: int, use_cache: bool, peer: str, targets: dict | None = None
    ) -> None:
        """Prefill ``ids`` and reply with the entries from ``have`` on; with ``targets`` (``prefill_push``) write
        them into the client's device memory over TT-Fabric instead and reply with what was written."""
        self.check(ids, have)
        if targets is not None and (int(targets["seq_len"]) != ids.numel() or int(targets["start"]) != have):
            raise ValueError(
                f"push targets for [{targets['start']}, {targets['seq_len']}): the request is "
                f"[{have}, {ids.numel()})"
            )
        gone = []

        def progress(_start, end, _seconds):
            # A client that hung up must not stop the prefill midway: the device work is already enqueued.
            if not gone:
                try:
                    send_msg(conn, {"progress": end})
                except OSError as e:
                    gone.append(e)

        n = ids.numel()
        t_wait = time.perf_counter()
        stages: dict[str, float] = {}
        with self.lock:
            t0 = time.perf_counter()
            stages["queued"] = t0 - t_wait
            slot, start = self.cache.lookup(self.d_p, ids) if use_cache else (self.cache.victim(), 0)
            t1 = time.perf_counter()
            stages["lookup"] = t1 - t0
            on_chunk = progress
            if targets is not None:
                if self.pusher is None:
                    self.pusher = FabricPush(self.d_p)
                self.pusher.begin(targets, slot=slot)

                def on_chunk(s, end, seconds):
                    self.pusher.chunk(end)
                    progress(s, end, seconds)

            prefill_times: dict = {}
            if start < n:
                self.cache.keep(slot, None)  # (until the slot holds all of ids)
                self.d_p.prefill(
                    ids.tolist(),
                    on_chunk=on_chunk,
                    slot=slot,
                    start=start,
                    times=prefill_times,
                    sync_chunks=self.sync_chunks,
                )
                self.cache.keep(slot, ids)
            else:  # the slot holds the whole prompt already (it may hold more: what it keeps stays)
                self.cache.touch(slot)
            chunks = prefill_times.pop("chunks", 0)
            stages.update({f"prefill.{k}": v for k, v in prefill_times.items()})
            t2 = time.perf_counter()
            seconds = t2 - t1
            parts: dict = {}
            if targets is None:
                host = self.d_p.host_states(n, parts, slot=slot, base=have)
            else:
                pushed = self.pusher.finish(parts)
            readback = time.perf_counter() - t2
            moved_stage = "readback" if targets is None else "push"
            stages.update({f"{moved_stage}.{k}": v for k, v in parts.items()})
        if gone:
            raise ConnectionError(f"{peer} hung up during the prefill: {gone[0]}")
        t3 = time.perf_counter()
        if targets is None:
            send_states(conn, host, seconds, stages["queued"], cached=start)
            moved = f"entries from {have} read back in {readback:.2f} s"
        else:
            reply = {"seq_len": n, "base": have, "cached": start, "seconds": seconds, "queued": stages["queued"]}
            send_msg(conn, {**reply, "bytes": pushed["bytes"], "pages": pushed["pages"], "push_seconds": readback})
            moved = (
                f"entries from {have} pushed ({pushed['bytes'] / 2**20:.1f} MiB, {pushed['runs']} runs) in "
                f"{readback:.2f} s"
            )
        t4 = time.perf_counter()
        stages["reply"] = t4 - t3
        stages["total"] = t4 - t_wait
        self.totals.add(n - start, stages)
        rate = f"{(n - start) / seconds:.0f} tok/s" if seconds > 0 and start < n else "nothing to compute"
        logger.info(
            f"{peer}: {n} tokens, {'hit' if start else 'miss'} in slot {slot} ({start} cached), {n - start} prefilled "
            f"in {seconds:.2f} s ({chunks} chunks, {rate}), waited {stages['queued']:.2f} s, {moved}"
        )
        logger.info(
            f"{peer}: profile (ms{', chunks synced' if self.sync_chunks else ''}): "
            + ", ".join(f"{k} {1e3 * v:.1f}" for k, v in stages.items())
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
                elif op == "stats":
                    send_msg(conn, service.totals.snapshot())
                elif op in ("prefill", "prefill_push"):
                    if len(blobs) != 1 or len(blobs[0]) % 4:
                        raise ValueError(f"{op} takes one blob of int32 token ids")
                    if op == "prefill_push" and not isinstance(header.get("targets"), dict):
                        raise ValueError("prefill_push takes the session's targets")
                    service.prefill(
                        conn,
                        torch.frombuffer(blobs[0], dtype=torch.int32),
                        int(header.get("have", 0)),
                        bool(header.get("cache", True)),
                        peer,
                        header.get("targets") if op == "prefill_push" else None,
                    )
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
    p.add_argument(
        "--cache-slots",
        type=int,
        default=256,
        help="most prompts kept on the device for the prefix cache, each a slot of --max-seq tokens (default 256)",
    )
    p.add_argument(
        "--cache-reserve-gb",
        type=float,
        default=4.0,
        help="DRAM a chip left free after the cache slots, for the chunks' working buffers (default 4 GiB); "
        "fewer slots than --cache-slots are made when they would not leave it",
    )
    p.add_argument(
        "--profile-chunks",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEEPSEEK_V41_PREFILL_PROFILE_CHUNKS") == "1",
        help="wait for the device after every chunk, so the profile splits host enqueue from device time exactly "
        "(slower: the host no longer runs ahead of the device); env DEEPSEEK_V41_PREFILL_PROFILE_CHUNKS=1 (tt-run "
        "forwards it to the ranks)",
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
        d_p = DPPrefill(
            mesh,
            max_seq=args.max_seq,
            slots=args.cache_slots,
            dram_reserve=int(args.cache_reserve_gb * 2**30),
        )
        service = PrefillService(d_p, sync_chunks=args.profile_chunks)
        warm = min(args.warm_len if args.warm_len is not None else d_p.max_seq, d_p.max_seq)
        if warm > 0:
            # Random ids: a constant prompt routes every token to the same experts and leaves most MoE programs cold.
            gen = torch.Generator().manual_seed(0)
            t0 = time.perf_counter()
            d_p.prefill(torch.randint(0, service.vocab, (warm,), generator=gen).tolist())
            logger.info(f"warmed {warm} tokens in {time.perf_counter() - t0:.0f} s")
            # A prompt continued off the chunk grid, as a cache hit runs it: a partial first chunk at an odd start.
            first, total = d_p.chunk + ALIGN, 2 * (d_p.chunk + ALIGN)
            if total <= d_p.max_seq:
                ids = torch.randint(0, service.vocab, (total,), generator=gen).tolist()
                t0 = time.perf_counter()
                d_p.prefill(ids[:first])
                if d_p.resume_start(0, first, total) == first:
                    d_p.prefill(ids, start=first)
                    logger.info(
                        f"warmed a continued prompt ({first} + {total - first}) in {time.perf_counter() - t0:.0f} s"
                    )
                else:
                    logger.warning(f"a slot cannot continue its own {first}-token prompt: every request will miss")
        with socket.create_server((args.host, args.port), reuse_port=True) as srv:
            logger.info(
                f"prefill server on {socket.gethostname()}:{args.port}: layers {d_p.layers[0]}-{d_p.layers[-1]}, "
                f"prompts of up to {d_p.max_seq} tokens in multiples of {ALIGN}, {len(d_p.slots)} prefix cache slots"
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
