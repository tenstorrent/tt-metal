# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Inputs, references and checks shared by the projection tests."""

import torch
from loguru import logger

from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import frames

CFG = AttentionConfig()
# ProjProgram(fidelity=...) of the tests: "custom_mm" (default), "recommended" or a pass schedule (program.FIDELITY)
import os as _os

FID = _os.environ.get("HY4_PROJ_FIDELITY", "custom_mm")
FID_KW = {} if FID == "custom_mm" else {"fidelity": FID}
FID_TAG = "" if FID == "custom_mm" else f" [{FID}]"
# HY4_PROJ_PRE_ROUNDED=1: the streams are given pre-rounded terms (ProjProgram(pre_rounded=True)); not with the
# prepared caches, which hold the other values
PRE_ROUNDED = _os.environ.get("HY4_PROJ_PRE_ROUNDED") == "1"
if PRE_ROUNDED:
    FID_KW = dict(FID_KW, pre_rounded=True)
    FID_TAG += " [pre-rounded]"
POSITIONS = torch.tensor([3, 17, 1000, 4095, 65536, 131071, 500000, 1048575])  # one per user


def random_reference(seed=0, full=False):
    """A reference layer with random parameters at the scale a trained layer has (tests/test_attention.py
    _reference), the layer's input norm weight, and on a full layer its indexer."""
    ref = Hy4Attention(CFG).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in Hy4Attention.PARAMETER_NAMES:
            p = getattr(ref, name)
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5)
            elif name == "learnable_sink_param":
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=g))
    gamma = 1.0 + 0.25 * torch.randn(CFG.hidden_size, generator=g)
    indexer = None
    if full:
        indexer = Hy4Indexer(IndexerConfig()).eval()
        with torch.no_grad():
            for name in ("wq_b", "wk", "weights_proj"):
                p = getattr(indexer, name)
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5)
    return ref, gamma, indexer


def random_input(seed=1, bf16=True, seven_bits=False, rows=L.ROWS):
    """(y [8, 6144] the collapsed input before the norm, scale [8] = 1 / RMS(y) with the model's eps)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(rows, L.HIDDEN, generator=g) * (0.5 + torch.rand(rows, 1, generator=g) * 4)
    if bf16:
        y = y.to(torch.bfloat16).float()
    if seven_bits:
        y = L.round7(y)
    scale = torch.rsqrt(y.double().pow(2).mean(dim=1) + 1e-5).float()
    return y, scale


def tables(rows=L.ROWS):
    """(cos, sin) of POSITIONS in the HF frame and in the device's (interleaved), [8, 64] each (16 rows: the 8
    positions and 8 more)."""
    positions = POSITIONS if rows == L.ROWS else torch.cat((POSITIONS, POSITIONS // 3 + 7))
    cos, sin = rope_cos_sin(positions, CFG)
    d_cos, d_sin = frames.rope_tables(cos, sin)
    return cos, sin, d_cos, d_sin


def check(name, ref, dev, threshold=None, prefix="HY4_MK7"):
    """Log pcc, max abs error, relative rms error and gain of dev against ref; assert the pcc if a threshold is
    given. Returns (pcc, relative rms error, gain)."""
    ref, dev = ref.double(), dev.double()
    score = pcc(ref.float(), dev.float())
    err = (ref - dev).abs().max().item()
    rel = ((ref - dev).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    gain = ((ref * dev).sum() / ref.pow(2).sum()).item()
    logger.info(
        f"{prefix} {name}: pcc={score:.6f} max_abs_err={err:.3e} rel_rms_err={rel:.3e} gain={gain:.5f} "
        f"ref_rms={ref.pow(2).mean().sqrt().item():.3e}"
    )
    if threshold is not None:
        assert score >= threshold, f"{name}: pcc {score:.6f} < {threshold}"
    return score, rel, gain
