# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One chip, the reference's batched expert TP (mk_experts.ReaderExperts MKX_XSL): slice c read from the FULL expert
tensors (XSL_C = c, the reference's validated path) vs the same slice as its OWN tensors (xsl_sliced: the production /
hybrid layout) -- the two must agree bitwise; and the 8 slices' partials summed vs the plain EP program on the same
experts (the XSL design's numerics: m rounded to bfloat8_b with the routing weight folded in).

    GLM53_MESH_SHAPE=1,1 pytest .../test_mk_xsl_sliced.py -s        # E=8 experts, top-8 (one full group)
    GLM53_MK_XSL_E=16 ...                                            # 16 experts, top-8: partial groups
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import _nd_dram
from models.autoports.zai_org_glm_5_3_flash.tt.mk_experts import ReaderExperts
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
E = int(os.environ.get("GLM53_MK_XSL_E", "8"))
TOPK = 8
T = int(os.environ.get("GLM53_MK_T", "4"))
H, I = 4096, 2048
TP = 8
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1264512"))
BF16, FP32 = ttnn.bfloat16, ttnn.float32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _pack(w, dev, cfg):
    t = ttnn.from_torch(w.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_device(t, dev, memory_config=cfg)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, worker_l1_size=WORKER_L1)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_xsl_sliced(mesh_device):
    dev = mesh_device
    torch.manual_seed(0)
    gu = [
        torch.randn(H, 2 * I) * 0.02 for _ in range(E)
    ]  # the fused [g0|u0|g1|u1..] layout (any data: self-consistency)
    dn = [torch.randn(I, H) * 0.02 for _ in range(E)]
    gu_full = [_pack(w, dev, _nd_dram(H, 64)) for w in gu]
    dn_full = [_pack(w, dev, _nd_dram(I, 64)) for w in dn]
    x = ttnn.from_torch((torch.randn(1, 1, T, H) * 0.5).to(torch.bfloat16), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)  # fmt: skip
    ids_t = torch.stack([torch.randperm(E)[:TOPK] for _ in range(T)]).to(torch.int32)  # distinct experts per row
    ids = ttnn.from_torch(ids_t.reshape(1, 1, T, TOPK), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=dev,
                          memory_config=DRAM)  # fmt: skip
    sc_t = torch.rand(T, E) * 0.5 + 0.25
    scores = ttnn.from_torch(sc_t.reshape(1, 1, T, E).to(torch.bfloat16), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)  # fmt: skip
    kw = dict(top_k=TOPK, scaling=1.0, limit=10.0, intermediate=I, hidden=H, e_lo=0, num_experts=E)

    def run(experts):
        y = experts(x, ids, scores, t=T)
        out = ttnn.to_torch(y).float()[0, 0, :T]
        ttnn.deallocate(y)
        return out

    y_ep = run(ReaderExperts(dev, gu_full, dn_full, **kw))
    logger.info(f"EP y: absmax {y_ep.abs().max():.4f} std {y_ep.std():.4f}")
    w2, w1 = 2 * I // TP, I // TP
    y_sum_full, y_sum_sliced = torch.zeros_like(y_ep), torch.zeros_like(y_ep)
    worst = 1.0
    for c in range(TP):
        y_c_full = run(ReaderExperts(dev, gu_full, dn_full, xsl=c, xsl_tp=TP, **kw))  # slice c from the full tensors
        gu_s = [_pack(w[:, w2 * c : w2 * (c + 1)], dev, _nd_dram(H, 64)) for w in gu]
        dn_s = [_pack(w[w1 * c : w1 * (c + 1)], dev, _nd_dram(w1, 64)) for w in dn]
        y_c_sliced = run(ReaderExperts(dev, gu_s, dn_s, xsl=0, xsl_tp=TP, xsl_sliced=True, **kw))
        for t_ in gu_s + dn_s:
            ttnn.deallocate(t_)
        exact = bool(torch.equal(y_c_full, y_c_sliced))
        p = _pcc(y_c_full, y_c_sliced)
        worst = min(worst, p)
        logger.info(f"MK_XSL_SLICE {c}: full-tensor read vs sliced tensors: bitwise {exact}, pcc {p:.6f}, "
                    f"absmax full {y_c_full.abs().max():.4f} sliced {y_c_sliced.abs().max():.4f}")  # fmt: skip
        y_sum_full += y_c_full
        y_sum_sliced += y_c_sliced
    p_full, p_sliced = _pcc(y_sum_full, y_ep), _pcc(y_sum_sliced, y_ep)
    logger.info(f"MK_XSL_SUMMARY: E {E} T {T}: sum of 8 slices vs EP: full-read pcc {p_full:.6f}, sliced pcc "
                f"{p_sliced:.6f}; worst slice full-vs-sliced pcc {worst:.6f}")  # fmt: skip
    assert worst > 0.9999, f"the sliced read differs from the full-tensor read: worst pcc {worst}"
    assert p_full > 0.99, f"XSL (full read) vs EP: pcc {p_full}"
