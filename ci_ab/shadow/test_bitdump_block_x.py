# r3 unary datacopy, bit dump of the block paths against main's per tile path (not for merge).
# The datacopy and tilize functional runners of the tree under test run on fixed, exhaustive stimuli (every bf16 / fp16
# bit pattern; for Float32 every upper half with seeded lower halves), with the golden check off, and the raw result
# bytes are written to $BITDUMP_DIR/<case>.bin. On a branch tree the runners take the block call (unpack_block=True),
# or the per tile call in the _tile test, on main the per tile call; bitdump_cmp.py compares two directories byte for byte.
import inspect
import os

import numpy as np
import pytest
import torch

import test_eltwise_unary_datacopy as dc
import test_unpack_tilize as tz
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, Tilize
from helpers.stimuli_config import StimuliConfig

OUT = os.environ.get("BITDUMP_DIR", "/tmp/bitdump")
_STATE = {"case": None, "offset": 0}
_real_dc_gen = dc.generate_stimuli
_real_tz_gen = tz.generate_stimuli
_real_collect = StimuliConfig.collect_results


def _patterns(t, offset):
    n = t.numel()
    idx = (torch.arange(n, dtype=torch.int64) + offset) % 65536
    if t.dtype in (torch.bfloat16, torch.float16):
        return idx.to(torch.int16).view(t.dtype).reshape(t.shape)
    lo = torch.from_numpy(np.random.default_rng(offset + 7).integers(0, 65536, n, dtype=np.int64))
    bits = ((idx << 16) | lo).to(torch.int64)
    bits = torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)
    if t.dtype == torch.int64:  # UInt32
        return ((idx << 16) | lo).reshape(t.shape)
    if t.dtype == torch.int32:
        return bits.reshape(t.shape)
    return bits.view(torch.float32).reshape(t.shape)


def _gen(real):
    def gen(*args, **kwargs):
        a, ca, b, cb = real(*args, **kwargs)
        return _patterns(a, _STATE["offset"]), ca, b, cb

    return gen


def _collect(self, location="0,0"):
    raw = self.collect_raw_result_bytes(location)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, _STATE["case"] + ".bin"), "wb") as f:
        f.write(raw)
    return _real_collect(self, location)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(dc, "generate_stimuli", _gen(_real_dc_gen))
    monkeypatch.setattr(tz, "generate_stimuli", _gen(_real_tz_gen))
    monkeypatch.setattr(dc, "passed_test", lambda *a, **k: True)
    monkeypatch.setattr(tz, "passed_test", lambda *a, **k: True)
    monkeypatch.setattr(StimuliConfig, "collect_results", _collect)


def _has_block(fn):
    return "unpack_block" in inspect.signature(fn).parameters


F = DataFormat
# Datacopy stimuli are tile ordered and a tile uses its first nf * 256 datums: these offsets give every 16-bit pattern once
# to a used datum over the runs of a case (32 tiles of [128, 256] per run).
DC_OFFSETS = {4: [0, 32768], 2: [0, 512, 32768, 33280], 1: [o + h for h in (0, 32768) for o in (0, 256, 512, 768)]}
DC = [
    (F.Float16_b, F.Float16_b, DestAccumulation.No, 4),
    (F.Float16_b, F.Float16_b, DestAccumulation.No, 2),
    (F.Float16_b, F.Float16_b, DestAccumulation.No, 1),
    (F.Float16_b, F.Float16_b, DestAccumulation.Yes, 4),
    (F.Float16, F.Float16, DestAccumulation.No, 4),
    (F.Float32, F.Float32, DestAccumulation.Yes, 4),
    (F.Float32, F.Float16_b, DestAccumulation.Yes, 4),
    (F.Float16_b, F.Bfp8_b, DestAccumulation.No, 4),
    (F.Bfp8_b, F.Float16_b, DestAccumulation.No, 4),
]


def _dc(cfg, block):
    fin, fout, acc, nf = cfg
    kw = {"unpack_block": block} if _has_block(dc._run_unary_datacopy_test) else {}
    for r, off in enumerate(DC_OFFSETS[nf]):
        _STATE["case"] = f"dc_{fin.name}_{fout.name}_acc{acc.name}_nf{nf}_r{r}"
        _STATE["offset"] = off
        dc._run_unary_datacopy_test(InputOutputFormat(fin, fout), acc, nf, Tilize.No, [128, 256], **kw)


@pytest.mark.parametrize("cfg", DC, ids=lambda c: f"{c[0].name}-{c[1].name}-acc{c[2].name}-nf{c[3]}")
def test_bitdump_datacopy(cfg):
    _dc(cfg, True)


# The 32-bit unpack-to-dest copy, every upper half with seeded lower halves: Float32 (as above), Int32 and UInt32.
DC32 = [
    (F.Float32, F.Float32, DestAccumulation.Yes, 4),
    (F.Float32, F.Float16_b, DestAccumulation.Yes, 4),
    (F.Int32, F.Int32, DestAccumulation.Yes, 4),
    (F.UInt32, F.UInt32, DestAccumulation.Yes, 4),
]


@pytest.mark.parametrize("cfg", DC32, ids=lambda c: f"{c[0].name}-{c[1].name}-acc{c[2].name}-nf{c[3]}")
def test_bitdump_datacopy32_block(cfg):
    _dc(cfg, True)


# The same through the per tile call on every tree (the branch's per tile unpack-to-dest bracket).
@pytest.mark.parametrize("cfg", DC32, ids=lambda c: f"{c[0].name}-{c[1].name}-acc{c[2].name}-nf{c[3]}")
def test_bitdump_datacopy32_tile(cfg):
    _dc(cfg, False)


TZ = [
    (F.Float16_b, F.Float16_b, 4),
    (F.Float16_b, F.Float16_b, 2),
    (F.Float16, F.Float16, 4),
    (F.Float32, F.Float32, 4),
    (F.Float32, F.Float16_b, 4),
    (F.Float16_b, F.Bfp8_b, 4),
]


# Tilize stimuli are row major [64, 64]; with two faces a tile uses its top 16 rows.
TZ_OFFSETS = {4: [k * 4096 for k in range(16)], 2: [k * 4096 + h for k in range(16) for h in (0, 1024)]}


@pytest.mark.parametrize("cfg", TZ, ids=lambda c: f"{c[0].name}-{c[1].name}-nf{c[2]}")
def test_bitdump_tilize(cfg):
    fin, fout, nf = cfg
    kw = {"unpack_block": True} if _has_block(tz.unpack_tilize) else {}
    for r, off in enumerate(TZ_OFFSETS[nf]):
        _STATE["case"] = f"tz_{fin.name}_{fout.name}_nf{nf}_r{r}"
        _STATE["offset"] = off
        tz.unpack_tilize(InputOutputFormat(fin, fout), num_faces=nf, **kw)
