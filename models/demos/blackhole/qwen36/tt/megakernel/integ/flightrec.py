# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hang flight recorder for the engine-ON verify path (diagnostic only; OFF unless QWEN36_R16_MK_FLIGHTREC=N > 0).

Every verify replay's host-side inputs go into a ring of N slots in a file-backed mmap, /tmp/mkfr_<pid>.bin. If the
device hangs, the worker's Python thread is blocked inside a ttnn call and can no longer write anything. The ring is
already on disk, so an outside watchdog can copy it. Each slot holds the page-table rows, the row positions (= the
staged kvpos when the host stages it), the inactive rows, mi_prev, the bucket and staging flags, and each engine
program's host-staged RQBAL table (replay-block word 7).

File layout: header 4096 B = magic b"MKFR0001", then at 64 <QdQd> (stage_seq, stage_t, done_seq, done_t), then at 128
a JSON meta (pid, TT_VISIBLE_DEVICES, slots, slot size, device buffer addresses) zero-padded. Slots follow: <QQ> (seq,
nbytes) + a pickle of a dict of numpy arrays / ints. ``read(path)`` decodes a file.

QWEN36_R16_MK_L1FLOOR_EVERY (default 500 while the recorder is on): every that many stages, the lowest live L1
allocation from the device allocator is checked against QWEN36_R16_MK_L1FLOOR_MIN (default 1465216 B, the engine's
static CB region end plus margin). The result is logged as "[mk-l1floor]" and stored in the meta. If
QWEN36_R16_MK_L1FLOOR_ASSERT=1, the check raises when a buffer is below the floor.
"""

import json
import mmap
import os
import pickle
import struct
import time

MAGIC = b"MKFR0001"
HDR = 4096
SLOT = 256 * 1024
_REC = None


def enabled():
    return int(os.environ.get("QWEN36_R16_MK_FLIGHTREC", "0") or 0) > 0


def get():
    global _REC
    if _REC is None and enabled():
        _REC = FlightRec(int(os.environ["QWEN36_R16_MK_FLIGHTREC"]))
    return _REC


def _np(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().copy()
    except Exception:  # noqa: BLE001
        pass
    return x


class FlightRec:
    def __init__(self, n):
        self.n = max(4, n)
        self.path = f"/tmp/mkfr_{os.getpid()}.bin"
        size = HDR + self.n * SLOT
        self._f = open(self.path, "w+b")
        self._f.truncate(size)
        self.mm = mmap.mmap(self._f.fileno(), size)
        self.mm[0:8] = MAGIC
        self.seq = 0
        self.meta = {"pid": os.getpid(), "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"), "slots": self.n,
                     "slot_bytes": SLOT, "created": time.time(), "addrs": {}, "l1floor": None}
        self.l1_every = int(os.environ.get("QWEN36_R16_MK_L1FLOOR_EVERY", "500") or 0)
        self.l1_min = int(os.environ.get("QWEN36_R16_MK_L1FLOOR_MIN", "1465216"))
        self.l1_assert = os.environ.get("QWEN36_R16_MK_L1FLOOR_ASSERT", "0") == "1"
        self._write_meta()

    def _write_meta(self):
        b = json.dumps(self.meta, default=str).encode()[: HDR - 129]
        self.mm[128:HDR] = b + b"\0" * (HDR - 128 - len(b))

    def set_addrs(self, **named):
        """Device addresses of the persistent per-replay buffers (for a triage-side DRAM read after a hang)."""
        for k, t in named.items():
            if t is None:
                continue
            try:
                self.meta["addrs"][k] = int(t.buffer_address())
            except Exception as e:  # noqa: BLE001
                self.meta["addrs"][k] = f"? {type(e).__name__}"
        self._write_meta()

    def stage(self, rec):
        """One verify replay's host inputs, written before the replay is enqueued."""
        self.seq += 1
        now = time.time()
        rec = {k: _np(v) for k, v in rec.items()}
        rec["seq"], rec["t"] = self.seq, now
        b = pickle.dumps(rec, protocol=4)
        if len(b) > SLOT - 16:  # oversized page table: keep everything else
            rec["pt"], rec["pt_dropped"] = None, len(b)
            b = pickle.dumps(rec, protocol=4)
        off = HDR + (self.seq % self.n) * SLOT
        self.mm[off:off + 16] = struct.pack("<QQ", self.seq, len(b))
        self.mm[off + 16:off + 16 + len(b)] = b
        self.mm[64:80] = struct.pack("<Qd", self.seq, now)

    def done(self):
        """The host has seen the device finish the latest staged replay (its readback or event returned)."""
        self.mm[80:96] = struct.pack("<Qd", self.seq, time.time())

    def l1_floor(self, mesh):
        if not self.l1_every or self.seq % self.l1_every:
            return
        import ttnn
        from loguru import logger

        t0 = time.perf_counter()
        low = None
        try:
            bufs = [b for b in ttnn._ttnn.reports.get_buffers(mesh) if b.buffer_type == ttnn.BufferType.L1]
            low = min(bufs, key=lambda b: b.address, default=None)
        except Exception as e:  # noqa: BLE001
            logger.info(f"[mk-l1floor] get_buffers failed: {e}")
            return
        ms = (time.perf_counter() - t0) * 1e3
        a = None if low is None else int(low.address)
        ok = a is None or a >= self.l1_min
        self.meta["l1floor"] = {"seq": self.seq, "lowest": a, "min": self.l1_min, "ok": ok, "n_l1": len(bufs)}
        self._write_meta()
        desc = "none" if low is None else f"{a} (+{low.max_size_per_bank} {str(low.buffer_layout).split('.')[-1]})"
        logger.info(f"[mk-l1floor] stage {self.seq}: lowest L1 allocation {desc} of {len(bufs)} L1 buffers; floor "
                    f"{self.l1_min}: {'OK' if ok else 'BELOW'} ({ms:.1f} ms)")
        if not ok and self.l1_assert:
            raise AssertionError(f"[mk-l1floor] L1 allocation at {a} below the engine floor {self.l1_min} after capture")


def read(path):
    """Decode a recorder file: (meta, header counters, records sorted by seq)."""
    with open(path, "rb") as f:
        d = f.read()
    assert d[:8] == MAGIC, f"{path}: not a flight-recorder file"
    st_seq, st_t, dn_seq, dn_t = struct.unpack("<QdQd", d[64:96])
    meta = json.loads(d[128:HDR].rstrip(b"\0") or b"{}")
    recs = []
    for i in range(meta.get("slots", 0)):
        off = HDR + i * meta["slot_bytes"]
        seq, nb = struct.unpack("<QQ", d[off:off + 16])
        if seq and nb:
            try:
                recs.append(pickle.loads(d[off + 16:off + 16 + nb]))
            except Exception as e:  # noqa: BLE001 -- a slot torn by the hang mid-write
                recs.append({"seq": seq, "torn": str(e)})
    recs.sort(key=lambda r: r["seq"])
    return meta, {"stage_seq": st_seq, "stage_t": st_t, "done_seq": dn_seq, "done_t": dn_t}, recs


if __name__ == "__main__":
    import sys

    import numpy as np

    np.set_printoptions(linewidth=200, threshold=100000)
    for p in sys.argv[1:]:
        meta, hdr, recs = read(p)
        print(f"== {p}\nmeta {json.dumps(meta)}\nheader {hdr} (outstanding {hdr['stage_seq'] - hdr['done_seq']})")
        for r in recs[-int(os.environ.get("FR_LAST", "8")):]:
            print(f"-- seq {r['seq']} t {r.get('t')}")
            for k in sorted(r):
                if k not in ("seq", "t"):
                    v = r[k]
                    print(f"   {k}: {v if not hasattr(v, 'shape') else f'{v.dtype}{list(v.shape)}'}")
                    if k == "pt" and v is not None:
                        print("   pt rows (last 6 nonzero cols each):",
                              [list(row[np.nonzero(row)[0][-6:]]) if np.any(row) else [] for row in v])
