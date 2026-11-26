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

Prefill ahead (``prefill_warm``): a decode server whose decode slots are all taken sends the prompts of the requests
waiting for one, to be prefilled into the prefix cache while the device would otherwise idle. Their real
``prefill_push`` / ``prefill`` later finds them kept and only moves the entries. A warm-up runs after every other
request waiting and yields to one between chunks (:class:`DeviceQueue`), keeping what it computed.

Images (``--image-rows``, :class:`ImageBook`): a prompt may hold image-span tokens. The decode server names each
image by key with its rows' slots and sends its vision rows the first time (``d_p_wire``). The server writes them
into its embedding table past the vocabulary and rewrites the prompt's image ids into its own rows. In metal_2's
model, image tokens rank experts with ``gate.bias_vl`` (the MoE runs again for a chunk that holds any) and take no
part in an Engram n-gram (``TtDS41Model.enable_images``).

Run (from the repo root, ttnn venv)::

    python models/experimental/deepseek_v41_flash/demo/prefill_server.py --port 9041

The checkpoint is the default Hugging Face cache
(``~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017``).

and try it with ``demo/prefill_client.py --server HOST:9041`` (``--prefix-check`` for the cache).
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import socket
import sys
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import torch
from loguru import logger

from models.experimental.deepseek_v4_flash.demo.overhead import OVERHEAD
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_push import FabricPush
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import VERSION, recv_msg, send_msg, send_states

# Decode commits whole 128-token units (its ALIGNMENT); the prompt's remainder is replayed by decode.
ALIGN = 128
# The full Galaxy this server runs. bringup/spec.yaml is the partial 2x2 bring-up.
_GALAXY_SPEC = (
    Path(__file__).resolve().parents[4]
    / "models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml"
)


def _checkpoint() -> str:
    """The default Hugging Face snapshot. Raises FileNotFoundError when it has not been downloaded."""
    from models.experimental.deepseek_v4_flash.tt.weight_loader import resolve_snapshot_dir
    from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR

    return str(resolve_snapshot_dir(DEFAULT_MODEL_DIR))


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


class NeedImages(Exception):
    """A request names images whose rows this server does not hold and did not send them: reply ``{"need": keys}``."""

    def __init__(self, keys: list[str]):
        super().__init__(f"send the rows of images {keys}")
        self.keys = keys


class ImageBook:
    """The vision rows of the images this server's prompts hold, by image (the decode server's key, the sha1 of its
    bytes), in the prefill model's embedding table: rows ``vocab + FIRST_IMAGE_ROW + r`` for ``r`` in ``[0, rows)``.

    A decode server names each image of a prompt with its own rows' slots (``{"key", "slots"}``); :meth:`resolve`
    rewrites those ids into this table's, so the prefix cache compares a prompt by its images, not by where either
    server keeps them. An image is written once, the first time a request sends its rows; when the rows run out the
    least recently used images not in the request go, and with them every prefix-cache slot whose prompt holds one."""

    def __init__(self, d_p, cache: PrefixCache):
        from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import FIRST_IMAGE_ROW

        self.d_p, self.cache = d_p, cache
        self.vocab = int(d_p.vocab)
        self.first = self.vocab + FIRST_IMAGE_ROW
        self.capacity = int(d_p.image_rows) - FIRST_IMAGE_ROW
        self.free = deque(range(self.capacity))
        self.entries: OrderedDict[str, torch.Tensor] = OrderedDict()  # key -> its rows, least recently used first
        self.hidden = int(d_p.cfg.dim)

    def resolve(self, ids: torch.Tensor, images: list, blobs: list) -> torch.Tensor:
        """``ids`` (a decode server's) with each image token's id this table's; the images first written if their
        rows came with the request (``blobs``: the rows of the images whose entry says ``"rows": true``, in order).
        Raises :class:`NeedImages` when one is neither held nor sent. Device writes: call with the prefill lock."""
        keys = [str(image["key"]) for image in images]
        sent = [key for key, image in zip(keys, images) if image.get("rows")]
        if len(sent) != len(blobs):
            raise ValueError(f"{len(sent)} images say their rows are sent, the request has {len(blobs)} row blobs")
        rows_of = dict(zip(sent, blobs))
        need = [key for key in keys if key not in self.entries and key not in rows_of]
        if need:
            raise NeedImages(need)
        decode, local = [], []
        for key, image in zip(keys, images):
            slots = torch.as_tensor(image["slots"], dtype=torch.int64).reshape(-1)
            rows = self.entries.get(key)
            if rows is None:
                data = torch.frombuffer(rows_of[key], dtype=torch.bfloat16)
                if data.numel() != slots.numel() * self.hidden:
                    raise ValueError(
                        f"image {key[:12]}: {data.numel()} values for {slots.numel()} rows of {self.hidden}"
                    )
                rows = self._add(key, data.view(slots.numel(), self.hidden), set(keys))
            self.entries.move_to_end(key)
            if rows.numel() != slots.numel():
                raise ValueError(f"image {key[:12]}: {slots.numel()} slots, this server holds {rows.numel()} rows")
            decode.append(slots)
            local.append(rows)
        ids = ids.long()
        image = ids >= self.first
        if not bool(image.any()):
            return ids.to(torch.int32)
        decode, local = torch.cat(decode), torch.cat(local)
        lut = torch.full((int(ids[image].max()) - self.first + 1,), -1, dtype=torch.int64)
        inside = decode < lut.numel()
        lut[decode[inside]] = local[inside]
        mapped = lut[ids[image] - self.first]
        if bool((mapped < 0).any()):
            raise ValueError("image token ids that no image of the request names")
        out = ids.clone()
        out[image] = self.first + mapped
        return out.to(torch.int32)

    def _add(self, key: str, rows: torch.Tensor, keep: set) -> torch.Tensor:
        n = rows.shape[0]
        if n > self.capacity:
            raise ValueError(f"image {key[:12]}: {n} rows, this server holds {self.capacity} (--image-rows)")
        while len(self.free) < n:
            victim = next((k for k in self.entries if k not in keep), None)
            if victim is None:
                raise ValueError(f"the request's images need more than this server's {self.capacity} image rows")
            self._evict(victim)
        slots = torch.tensor([self.free.popleft() for _ in range(n)], dtype=torch.int64)
        try:
            self.d_p.write_image_rows(rows, (self.first + slots).tolist())
        except Exception:
            self.free.extend(slots.tolist())
            raise
        self.entries[key] = slots
        logger.info(f"image {key[:12]}: {n} rows written ({len(self.free)} of {self.capacity} free)")
        return slots

    def _evict(self, key: str) -> None:
        ids = self.first + self.entries.pop(key)
        for slot, held in enumerate(self.cache.ids):
            if held is not None and bool(torch.isin(held.long(), ids).any()):
                self.cache.keep(slot, None)  # (its prompt held the image: no later prompt may continue it)
        self.free.extend((ids - self.first).tolist())
        logger.info(f"image {key[:12]}: evicted ({len(self.free)} of {self.capacity} image rows free)")


class DeviceQueue:
    """The device, one request at a time (:meth:`hold`). Waiting requests are served in arrival order, except that a
    prefill a decode server needs now goes before every warm-up (``prefill_warm``: a prompt prefilled ahead into the
    prefix cache for a request still waiting for a decode slot), and a warm-up holding the device yields between
    chunks while one waits (:meth:`urgent`)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._held = False
        self._waiting: list[tuple[int, int]] = []  # (1 for a warm-up else 0, arrival)
        self._arrivals = itertools.count()

    @contextlib.contextmanager
    def hold(self, warm: bool = False):
        me = (int(bool(warm)), next(self._arrivals))
        with self._cv:
            self._waiting.append(me)
            try:
                while self._held or min(self._waiting) != me:
                    self._cv.wait()
            finally:
                self._waiting.remove(me)
                self._cv.notify_all()  # (the next in line, should this one leave without the device)
            self._held = True
        try:
            yield
        finally:
            with self._cv:
                self._held = False
                self._cv.notify_all()

    def urgent(self) -> bool:
        """A request other than a warm-up is waiting for the device."""
        with self._cv:
            return any(not warm for warm, _ in self._waiting)


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
    """The device side: one :class:`DPPrefill` and its prefix cache, one prefill at a time (``device``,
    :class:`DeviceQueue`: warm-ups after every other prefill, and yielding to one between chunks).

    A warm-up (``warm``, the ``prefill_warm`` op) only prefills a prompt into the prefix cache, for a decode server
    whose request for it is still waiting for a decode slot: its ``prefill`` / ``prefill_push`` later finds the prompt
    kept and computes nothing, or the rest from where a warm-up that yielded stopped. Warm-ups are counted apart
    (``warm_totals``, the ``stats`` op's ``warm``).

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
        self.warm_totals = StageTotals()
        self.device = DeviceQueue()
        self.cache = PrefixCache(len(d_p.slots))
        self.images = ImageBook(d_p, self.cache) if getattr(d_p, "image_rows", 0) else None
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
            # image-span tokens (ImageBook): how many images' rows the server holds at once; 0, or missing, none
            "image_rows": self.images.capacity if self.images is not None else 0,
            # takes prefill_warm (a prompt prefilled ahead into the prefix cache); missing: it does not
            "warm": True,
        }

    def check(self, ids: torch.Tensor, have: int, limit: int | None) -> None:
        """``limit``: the ids lie below it -- the vocabulary, or past it the image rows once ImageBook.resolve has
        rewritten a decode server's image ids (None: not checked above, those ids before that)."""
        n = ids.numel()
        if not 0 < n <= self.d_p.max_seq or n % ALIGN:
            raise ValueError(f"prompt of {n} tokens: send a positive multiple of {ALIGN} up to {self.d_p.max_seq}")
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or (limit is not None and hi >= limit):
            raise ValueError(
                f"token ids span [{lo}, {hi}], the vocabulary is [0, {self.vocab})"
                + ("" if limit == self.vocab else f", the image rows up to {limit}")
            )
        if not 0 <= have <= n or have % ALIGN:
            raise ValueError(f"have {have}: a multiple of {ALIGN} up to the prompt's {n} tokens")

    def prefill(
        self,
        conn: socket.socket,
        ids: torch.Tensor,
        have: int,
        use_cache: bool,
        peer: str,
        targets: dict | None = None,
        images: list | None = None,
        rows: list | None = None,
        warm: bool = False,
    ) -> None:
        """Prefill ``ids`` and reply with the entries from ``have`` on; with ``targets`` (``prefill_push``) write
        them into the client's device memory over TT-Fabric instead and reply with what was written. ``images``: the
        prompt's images (``[{"key", "slots", "rows"}]``, the decode server's), ``rows`` the row blobs sent with them;
        :class:`NeedImages` (nothing run) when one is neither held nor sent.

        ``warm`` (``prefill_warm``): only prefill ``ids`` into the prefix cache, after every other request waiting and
        stopping after a chunk once one comes in; reply ``{"seq_len", "done", "cached", "seconds", "queued"}``, where
        ``done`` is how far it got (the slot keeps ``ids[:done]``, so sending it again continues from there)."""
        if images and self.images is None:
            raise ValueError("this prefill server takes no images (--image-rows 0)")
        if warm and (images or targets is not None):
            raise ValueError("a warm-up takes a text prompt only, and writes nothing to the client")
        self.check(ids, have, None if images else self.vocab)
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
                    with OVERHEAD.time("net.send_progress"):
                        send_msg(conn, {"progress": end})
                except OSError as e:
                    gone.append(e)

        n = ids.numel()
        t_wait = time.perf_counter()
        stages: dict[str, float] = {}
        with self.device.hold(warm):
            t0 = time.perf_counter()
            stages["queued"] = t0 - t_wait
            if images:
                ids = self.images.resolve(ids, images, rows or [])
                self.check(ids, have, self.vocab + self.d_p.image_rows)
                stages["images"] = time.perf_counter() - t0
                t0 = time.perf_counter()
            slot, start = self.cache.lookup(self.d_p, ids) if use_cache or warm else (self.cache.victim(), 0)
            t1 = time.perf_counter()
            stages["lookup"] = t1 - t0
            on_chunk = None if warm else progress
            if targets is not None:
                if self.pusher is None:
                    self.pusher = FabricPush(self.d_p)
                self.pusher.begin(targets, slot=slot)

                def on_chunk(s, end, seconds):
                    with OVERHEAD.time("push.chunk"):
                        self.pusher.chunk(end)
                    progress(s, end, seconds)

            prefill_times: dict = {}
            done = n
            if start < n:
                self.cache.keep(slot, None)  # (until the slot holds all of ids)
                done = self.d_p.prefill(
                    ids.tolist(),
                    on_chunk=on_chunk,
                    slot=slot,
                    start=start,
                    times=prefill_times,
                    sync_chunks=self.sync_chunks,
                    stop=self.device.urgent if warm else None,
                )
                self.cache.keep(slot, ids if done >= n else ids[:done])
            else:  # the slot holds the whole prompt already (it may hold more: what it keeps stays)
                self.cache.touch(slot)
            chunks = prefill_times.pop("chunks", 0)
            stages.update({f"prefill.{k}": v for k, v in prefill_times.items()})
            t2 = time.perf_counter()
            seconds = t2 - t1
            parts: dict = {}
            if warm:
                pass
            elif targets is None:
                host = self.d_p.host_states(n, parts, slot=slot, base=have)
            else:
                pushed = self.pusher.finish(parts)
            readback = time.perf_counter() - t2
            moved_stage = "readback" if targets is None else "push"
            stages.update({f"{moved_stage}.{k}": v for k, v in parts.items()})
        if gone:
            raise ConnectionError(f"{peer} hung up during the prefill: {gone[0]}")
        t3 = time.perf_counter()
        if warm:
            reply = {"seq_len": n, "done": done, "cached": start, "seconds": seconds, "queued": stages["queued"]}
            send_msg(conn, reply)
            moved = "kept" if done >= n else f"yielded at {done} to a waiting prefill"
        elif targets is None:
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
        computed = max(done - start, 0)
        if warm:
            self.warm_totals.add(computed, stages)
            # Device time no request was waiting on yet (a decode server's request still waiting for a slot).
            OVERHEAD.add("compute.prefill_ahead", seconds, own_thread=False)
            OVERHEAD.add("wait.prefill_lock_ahead", stages["queued"], own_thread=False)
        else:
            self.totals.add(computed, stages)
            for stage, s in stages.items():
                if stage in _OVERHEAD_STAGES:
                    OVERHEAD.add(_OVERHEAD_STAGES[stage], s, own_thread=False)
                elif stage.startswith(("prefill.", "readback.", "push.")):
                    OVERHEAD.add(_OVERHEAD_STAGES.get(stage.split(".")[0] + ".*", "") + stage, s, own_thread=False)
        rate = f"{computed / seconds:.0f} tok/s" if seconds > 0 and computed else "nothing to compute"
        logger.info(
            f"{peer}: {'warm-up of ' if warm else ''}{n} tokens, {'hit' if start else 'miss'} in slot {slot} "
            f"({start} cached), {computed} prefilled in {seconds:.2f} s ({chunks} chunks, {rate}), waited "
            f"{stages['queued']:.2f} s, {moved}"
        )
        logger.info(
            f"{peer}: profile (ms{', chunks synced' if self.sync_chunks else ''}): "
            + ", ".join(f"{k} {1e3 * v:.1f}" for k, v in stages.items())
        )


# A request's profile stages (PrefillService.prefill) as overhead-table stages (overhead.py): which are the device
# computing, a wait, or host overhead. "enqueue" can block on a full command queue, so it may hold device time too.
_OVERHEAD_STAGES = {
    "queued": "wait.prefill_lock",
    "images": "prefill.images",
    "lookup": "prefill.cache_lookup",
    "reply": "net.send_reply",
    "prefill.drain": "compute.prefill_drain",
    "prefill.*": "",  # prefill.input (ids, Engram rows, H2D), prefill.enqueue (host dispatch)
    "readback.*": "d2h.",  # the KV entries read back to the host (without push)
    "push.*": "",  # the KV entries written into decode's DRAM over TT-Fabric
}


def _serve_connection(conn: socket.socket, peer: str, service: PrefillService) -> None:
    with conn:
        while True:
            times: dict = {}
            try:
                header, blobs = recv_msg(conn, times)
            except (ConnectionError, OSError):
                break
            OVERHEAD.add("idle.connection", times["wait"])
            OVERHEAD.add("net.recv_request", times["transfer"])
            op = header.get("op")
            try:
                if op == "ping":
                    send_msg(conn, {"pong": True})
                elif op == "info":
                    send_msg(conn, service.info)
                elif op == "stats":
                    send_msg(conn, {**service.totals.snapshot(), "warm": service.warm_totals.snapshot()})
                elif op in ("prefill", "prefill_push", "prefill_warm"):
                    images = header.get("images") or []
                    if not blobs or len(blobs[0]) % 4 or (len(blobs) > 1 and not images):
                        raise ValueError(f"{op} takes one blob of int32 token ids (then its images' rows)")
                    if op == "prefill_push" and not isinstance(header.get("targets"), dict):
                        raise ValueError("prefill_push takes the session's targets")
                    service.prefill(
                        conn,
                        torch.frombuffer(blobs[0], dtype=torch.int32),
                        int(header.get("have", 0)),
                        bool(header.get("cache", True)),
                        peer,
                        header.get("targets") if op == "prefill_push" else None,
                        images=images,
                        rows=blobs[1:],
                        warm=op == "prefill_warm",
                    )
                else:
                    raise ValueError(f"unknown op {op!r}")
            except (ConnectionError, BrokenPipeError):
                break
            except NeedImages as e:  # (nothing ran: the client sends the request again with those images' rows)
                logger.info(f"{peer}: {op}: asking for the rows of {len(e.keys)} image(s)")
                try:
                    send_msg(conn, {"need": e.keys})
                except OSError:
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
    p.add_argument(
        "--image-rows",
        type=int,
        default=16384,
        help="embedding rows for image-span tokens (3 span delimiters, then the images' vision rows; an image takes "
        "184-1024): 0 takes no images; 16384 costs 40 MiB a chip (default 16384)",
    )
    p.add_argument(
        "--load-threads",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V41_PREFILL_LOAD_THREADS", "10")),
        help="worker threads that read the next layers' weight files into the page cache and compute the dense "
        "weights' bfp8 statistics while the main thread builds each layer in order (metal_2's LOAD_THREADS; "
        "0: the serial build; default 10; env DEEPSEEK_V41_PREFILL_LOAD_THREADS, which tt-run forwards to the ranks)",
    )
    args = p.parse_args(argv)
    os.environ["DS41_LOAD_THREADS"] = str(max(0, args.load_threads))
    try:
        checkpoint = _checkpoint()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    # metal_2's settings give a process that has imported pytest (its model build does) ENGRAM_TABLE=mmap: random reads
    # of the ~190 GB checkpoint tables, which leave a 1M-token prompt host-bound for many minutes. Serve from RAM.
    os.environ.setdefault("DS41_ENGRAM_TABLE", "resident")
    # ... and the same pytest import turns ENGRAM_SHM_DIR off (its test value): the packed bfp8 rows (2 x 104 GB) were
    # rebuilt from the checkpoint at every start (~5 min on one thread, layer 1's load waiting ~80 s for it). Keep them
    # in tmpfs so a restart maps them in seconds (the first start builds them there; without room it builds in
    # anonymous memory as before, and logs it).
    os.environ.setdefault("DS41_ENGRAM_SHM_DIR", "/dev/shm/deepseek_v41_d_p_engram")

    from models.demos.common.bringup.core.spec import Spec
    from models.demos.common.bringup.testing.harness import close_mesh, open_mesh
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import DPPrefill

    spec = Spec.load(_GALAXY_SPEC)
    mesh, fab = open_mesh(spec)
    try:
        d_p = DPPrefill(
            mesh,
            max_seq=args.max_seq,
            slots=args.cache_slots,
            dram_reserve=int(args.cache_reserve_gb * 2**30),
            checkpoint=checkpoint,
            spec=spec,
            image_rows=args.image_rows,
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
