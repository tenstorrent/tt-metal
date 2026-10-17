# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Host side of the sampling application (tools/l2cpu/sampling/fw): boot through the generic L2cpuCtl, descriptors,
per-user parameters, requests, tokens, timing records and counters. Offsets: layout.py (l2cpu_sampling.h).

    fw, region = boot(device)                      # ttnn device open on a fresh chip reset; keep `region` alive
    fw.set_logits_desc(local_desc(S.L2S_DTYPE_BF16, 151936))
    fw.set_ctrl(1, 151936); fw.set_params([(0.7, 50, 0.9, 1234)])
    fw.write(S.L2S_OFF_LOGITS, row.tobytes()); fw.issue(); fw.wait_done(); fw.next_tokens(1)
"""
from __future__ import annotations

import os
import struct
import time

from .. import layout as L
from ..ctl import L2cpuCtl
from ..hw import REGION_ALIGN
from . import layout as S

L2CPU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_IMAGE = os.environ.get("L2S_FW_IMAGE") or os.path.join(L2CPU_DIR, "fw", "build", "bh-irq-sampling", "fw.bin")
CLINT_MTIME_PA = 0x0200_BFF8
UNCACHED_DELTA = 0x4000_0000_0000  # cached Memory Port PA - uncached System Port PA of the same local DRAM byte
REGION_ALLOC_BYTES = S.L2S_REGION_SIZE + REGION_ALIGN  # per bank: the base is rounded up inside the page

ERR = {
    v: k
    for k, v in {
        **{k: v for k, v in L.CONSTANTS.items() if k.startswith("L2CPU_ERR_")},
        **{k: v for k, v in S.CONSTANTS.items() if k.startswith("L2S_ERR_")},
    }.items()
}
BAD = {v: k[len("L2S_BAD_") :] for k, v in S.CONSTANTS.items() if k.startswith("L2S_BAD_")}


class SamplingError(RuntimeError):
    pass


def pack_desc(
    dtype,
    layout=S.L2S_LAYOUT_ROW_MAJOR,
    rows=32,
    cols=0,
    page_size=0,
    page_stride=0,
    banks=(),
    first_bank=0,
    location=S.L2S_LOC_NOC,
    row_stride=0,
    local_off=0,
):
    """Tensor descriptor bytes. banks: list of (x, y, noc, addr) for location NOC."""
    head = struct.pack(
        "<16I",
        dtype,
        layout,
        rows,
        cols,
        page_size,
        page_stride,
        len(banks),
        first_bank,
        location,
        row_stride,
        local_off,
        0,
        0,
        0,
        0,
        0,
    )
    body = b"".join(struct.pack("<BBBBIQ", x, y, noc, 0, 0, addr) for (x, y, noc, addr) in banks)
    d = head + body
    return d + bytes(S.L2S_DESC_SIZE - len(d))


def local_desc(dtype, cols, rows=32, uncached=False, row_stride=None):
    """Rows in the coherent (default) or uncached logits zone of the region."""
    es = 2 if dtype == S.L2S_DTYPE_BF16 else 4
    rs = row_stride or (cols * es + 63) // 64 * 64
    return pack_desc(
        dtype,
        rows=rows,
        cols=cols,
        location=S.L2S_LOC_REGION_UC if uncached else S.L2S_LOC_REGION,
        row_stride=rs,
        local_off=S.L2S_OFF_LOGITS_UC if uncached else S.L2S_OFF_LOGITS,
    )


class SamplingFw:
    def __init__(self, ctl: L2cpuCtl):
        self.ctl, self.hw, self.base = ctl, ctl.hw, ctl.base
        self.req = None

    # ---- region access ----
    def r32(self, off):
        return self.hw.pa_read32(self.base + off)

    def w32(self, off, v):
        self.hw.pa_write32(self.base + off, v & 0xFFFFFFFF)

    def read(self, off, n):
        return self.hw.pa_read(self.base + off, n)

    def write(self, off, data):
        self.hw.pa_write(self.base + off, data)

    def write_uncached(self, off, data):
        """System Port alias: the uncached logits zone only (never mix aliases on one line)."""
        self.hw.pa_write(self.base + off - UNCACHED_DELTA, data)

    def mtime(self):
        return self.hw.pa_read64(CLINT_MTIME_PA)

    # ---- identity ----
    def sync(self):
        """After start / restart: check the image and pick up the request sequence (WARM keeps it, COLD resets)."""
        app, lv = self.r32(L.L2CPU_OFF_APP_ID), self.r32(L.L2CPU_OFF_APP_LAYOUT_VERSION)
        if app != S.L2S_APP_ID or lv != S.L2S_LAYOUT_VERSION:
            raise SamplingError(
                f"image app {app} layout {lv}; host mirror app {S.L2S_APP_ID} layout {S.L2S_LAYOUT_VERSION}"
            )
        self.req = self.r32(S.L2S_OFF_REQ_SEQ)
        return self

    def build_flags(self):
        return self.r32(L.L2CPU_OFF_BUILD_FLAGS) >> L.L2CPU_BUILD_APP_SHIFT

    def mtime_hz(self):
        return self.r32(L.L2CPU_OFF_MTIME_HZ)

    # ---- configuration ----
    def set_logits_desc(self, desc: bytes):
        self.write(S.L2S_OFF_LOGITS_DESC, desc)

    def set_tokens_desc(self, desc: bytes):
        self.write(S.L2S_OFF_TOKENS_DESC, desc)

    def set_params(self, params):
        """params: (temperature, top_k, top_p, seed) for users 0.."""
        data = b"".join(
            struct.pack("<fIfIQ40x", float(t), int(k), float(p), 0, int(s) & (2**64 - 1)) for (t, k, p, s) in params
        )
        self.write(S.L2S_OFF_PARAMS, data)

    def set_ctrl(self, batch, vocab, vocab_padded=None, flags=0, step_seq_base=None):
        self.w32(S.L2S_OFF_BATCH, batch)
        self.w32(S.L2S_OFF_VOCAB, vocab)
        self.w32(S.L2S_OFF_VOCAB_PADDED, vocab_padded or vocab)
        self.w32(S.L2S_OFF_FLAGS, flags)
        if step_seq_base is not None:
            self.w32(S.L2S_OFF_STEP_SEQ_BASE, step_seq_base)

    def set_step(self, step):
        """The next request samples with this step (step = req_seq - step_seq_base)."""
        self.w32(S.L2S_OFF_STEP_SEQ_BASE, (self.req + 1 - step) & 0xFFFFFFFF)

    def set_stream(self, mode, timeout_us=0):
        """mode: L2S_STREAM_ROWS; timeout_us: 0 = firmware default (L2S_STREAM_TIMEOUT_US)."""
        self.write(S.L2S_OFF_STREAM_MODE, struct.pack("<III", mode, timeout_us, 0))

    def set_work_timeout_us(self, us):
        self.w32(S.L2S_OFF_WORK_TIMEOUT_US, us)

    # ---- requests ----
    def issue(self, doorbell=True):
        self.req = (self.req + 1) & 0xFFFFFFFF
        self.w32(S.L2S_OFF_REQ_SEQ, self.req)
        if doorbell:
            self.hw.doorbell(self.req | 0x80000000)  # never push 0
        return self.req

    def wait_done(self, timeout=5.0):
        t0 = time.time()
        while self.r32(S.L2S_OFF_DONE_SEQ) != self.req:
            if time.time() - t0 > timeout:
                raise SamplingError(
                    f"request {self.req}: done_seq {self.r32(S.L2S_OFF_DONE_SEQ)}; error "
                    f"{self.error()}\n{self.ctl.log_text()[0][-3000:]}"
                )

    def next_tokens(self, batch):
        d = self.read(S.L2S_OFF_NEXT_TOKENS, 64 * batch)
        return [struct.unpack_from("<I", d, 64 * u)[0] for u in range(batch)]

    def ring_last(self, batch):
        wr = self.r32(S.L2S_OFF_RING_WR)
        d = self.read(S.L2S_OFF_RING + ((wr - 1) % S.L2S_RING_SLOTS) * S.L2S_RING_SLOT_SIZE, S.L2S_RING_SLOT_SIZE)
        seq, b, step = struct.unpack_from("<III", d, 0)
        return dict(req_seq=seq, batch=b, step=step, tok=list(struct.unpack_from(f"<{batch}I", d, S.L2S_RS_TOK)))

    def timing_last(self):
        n = self.r32(S.L2S_OFF_TIMING_COUNT)
        d = self.read(S.L2S_OFF_TIMING + ((n - 1) % S.L2S_TIMING_SLOTS) * S.L2S_TIMING_SIZE, 56)
        f = struct.unpack("<IIQQIIIII3I", d)
        return dict(
            req_seq=f[0],
            batch=f[1],
            mtime_wake=f[2],
            mtime_publish=f[3],
            cyc_read=f[4],
            cyc_sample=f[5],
            cyc_write=f[6],
            cyc_wait=f[7],
            cyc_total=f[8],
            cyc_worker=list(f[9:12]),
        )

    # ---- status ----
    def counters(self, h):
        w, s, m, wi, req, users, nan, caps = struct.unpack("<8Q", self.read(L.L2CPU_OFF_COUNTERS + 64 * h, 64))
        return dict(wakeups=w, spurious=s, mailbox=m, work_items=wi, requests=req, users=users, nan=nan, kmax_caps=caps)

    def error(self):
        e = self.ctl.error()
        if e:
            e["name"] = ERR.get(e["code"], e["code"])
            if e["code"] == S.L2S_ERR_BAD_DESC:
                e["reason"] = BAD.get(e["arg"], e["arg"])
        return e


def allocate_region(device):
    """An interleaved ttnn DRAM buffer whose bank-5 slice holds the 64 MiB region. Allocate BEFORE any trace
    capture and keep it alive for the whole session."""
    import ttnn

    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 8, REGION_ALLOC_BYTES // 4]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def boot(device, image_path=DEFAULT_IMAGE, region=None, mhz=1750, log=print):
    """Fresh chip reset + ttnn device open: allocate the region, start the sampling image with L2cpuCtl.start
    (resident page, RNMI handlers, release), wait READY. Returns (SamplingFw, region tensor, start info)."""
    from ..bringup import region_base_pa
    from ..hw import L2cpuHw, TtnnClusterBackend

    region = region if region is not None else allocate_region(device)
    base = region_base_pa(region.buffer_address())
    ctl = L2cpuCtl(L2cpuHw(TtnnClusterBackend(0), guard=True, log=None), base, log=log, mhz=mhz)
    t0 = time.time()
    info = ctl.start(open(image_path, "rb").read())
    info["boot_s"] = time.time() - t0
    return SamplingFw(ctl).sync(), region, info
