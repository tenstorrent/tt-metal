# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.3: helpers of the packed device tests (test_packed_oracle / _reference / _walls): the PACKED switch per call,
a lane's rows of a device boundary, digests."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager

import torch

import ttnn
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

CHUNK = 8192
LANES = 4
PAD_ID = 0xFFFFFFFF


@contextmanager
def switch(key: str, value: str):
    """A call-time switch (bringup/hooks.py batch_call_settings) = value for the calls inside: the frozen snapshot's
    entry when the model froze its settings (DS41-314), and the environment."""
    name = settings.env_name(key)
    snap = settings.snap
    old_env = os.environ.get(name)
    old_snap = None if snap is None else snap.get(key)
    os.environ[name] = value
    if snap is not None:
        snap[key] = value
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_env
        if snap is not None and settings.snap is snap:
            snap[key] = old_snap


def packed(value: str):
    """PACKED=value for the calls inside (switch)."""
    return switch("PACKED", value)


def lane_rows(model, t, desc, lane: int, n: int, width: int | None = None) -> torch.Tensor:
    """The first ``n`` rows of lane ``lane`` of an OWN_ROWS tensor (each chip its own rows, chip-linear) -> host
    float32 [n, width] (only the lane's chips are read; ``width`` trims the columns, as own_rows_to_host)."""
    sp, tp = tuple(model.mesh.shape)
    chips = desc.lane_chips(sp, tp)[lane]
    dev = ttnn.get_device_tensors(t)
    w = t.shape[-1]
    parts = [ttnn.to_torch(dev[q]).reshape(-1, w) for q in chips]
    out = torch.cat(parts, dim=0)[:n].float()
    return out if width is None else out[:, :width]


def boundary_rows(model, h, desc, lane: int, n: int) -> torch.Tensor:
    """A layer boundary (DevH) -> lane ``lane``'s first ``n`` rows as the reference packs them [n, 4H + 4] (the
    streams, then the 4 used columns of the pre: TtDS41Model.to_host)."""
    from models.demos.deepseek_v41_flash_d_p.tt.layout import HC
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import as_run_desc, seg_index

    rd = as_run_desc(desc, model.mesh)
    if getattr(rd, "glued", False):  # PB.9G: the chunk ran as its glued chunk (chip-aligned lanes): the segment's rows
        return glued_boundary_rows(model, h, rd.lane_off(seg_index(desc, lane)), n)
    return torch.cat([lane_rows(model, h.streams, desc, lane, n), lane_rows(model, h.pre, desc, lane, n, HC)], dim=-1)


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def sample_idx(n: int, k: int = 64) -> torch.Tensor:
    return torch.unique(torch.linspace(0, max(0, n - 1), min(k, n)).long())


class Capture:
    """Per (segment, layer): the digest of the segment's rows of the layer output (streams and pre) and a float32
    copy of 64 sampled rows (the max abs diff where two digests differ)."""

    def __init__(self):
        self.dig, self.smp = {}, {}

    def add(self, key, rows: torch.Tensor) -> None:
        self.dig[key] = digest(rows)
        self.smp[key] = rows[sample_idx(rows.shape[0])].clone()

    @staticmethod
    def diff(a: "Capture", b: "Capture", keys) -> float:
        """0 when every digest matches; else the max abs diff over the sampled rows of the differing keys, at least
        1e-30 (the bytes differ somewhere even when the sampled rows agree)."""
        worst = 0.0
        for k in keys:
            if a.dig.get(k) == b.dig.get(k) and k in a.dig:
                continue
            if k not in a.smp or k not in b.smp:
                return float("inf")
            d = (a.smp[k] - b.smp[k]).abs()
            d = torch.nan_to_num(d, nan=float("inf"))
            worst = max(worst, float(d.max()) if d.numel() else 0.0, 1e-30)
        return worst


def served_digests(state, layers, length: int) -> dict:
    """{(layer, config name): sha256} of the slot's served records of positions [0, length) (window snapshot /
    window rows, compressed KV, index keys)."""
    out = {}
    for i in layers:
        for name, t in state.served(i, length).items():
            out[(i, name)] = digest(t)
        c = state.window[i]
        if getattr(c, "snap", None) is not None:  # (WINDOW_KV_KEEP=snapshot128: the served 128-row snapshot)
            out[(i, "snapshot")] = digest(torch.cat([x.reshape(-1) for x in c.snapshot_host()]))
    return out


# ---------------------------------------------------------------------------------------------- PB.7 glued chunks
def glued(value: str):
    """PACKED_GLUED=value for the calls inside (switch)."""
    return switch("PACKED_GLUED", value)


def model_rows(model, t, off: int, n: int, width: int | None = None) -> torch.Tensor:
    """Model rows [off, off + n) of an OWN_ROWS tensor (each chip its own rows [q R, (q + 1) R), chip-linear) -> host
    float32 [n, width] (only the chips holding them are read)."""
    dev = ttnn.get_device_tensors(t)
    w = t.shape[-1]
    per = int(t.shape[-2])
    q0, q1 = off // per, max(off // per, (off + n - 1) // per)
    parts = [ttnn.to_torch(dev[q]).reshape(-1, w) for q in range(q0, q1 + 1)]
    out = torch.cat(parts, dim=0)[off - q0 * per : off - q0 * per + n].float()
    return out if width is None else out[:, :width]


def glued_boundary_rows(model, h, off: int, n: int) -> torch.Tensor:
    """boundary_rows for a glued chunk's segment at model rows [off, off + n)."""
    from models.demos.deepseek_v41_flash_d_p.tt.layout import HC

    return torch.cat([model_rows(model, h.streams, off, n), model_rows(model, h.pre, off, n, HC)], dim=-1)
