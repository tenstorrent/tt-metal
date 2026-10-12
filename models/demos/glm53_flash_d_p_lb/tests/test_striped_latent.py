# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the striped (block-cyclic) MLA latent cache vs the replicated one, one DSA layer (SL_LAYER, default 3) with
real weights, SL_CHUNKS consecutive chunks of SL_CHUNK rows from position 0 (random residual input, same for both).
Checks per chunk: block outputs bit-identical, latent state (state_torch) identical. Then a prefix reload: the
replicated block's state after the chunks loaded into a fresh striped block (load_state -> pending stripe), one more
chunk on both, outputs bit-identical. Prints the MLA step device times of both."""

import os

import torch

import ttnn
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
LAYER = int(os.environ.get("SL_LAYER", "3"))
CHUNK = int(os.environ.get("SL_CHUNK", "2048"))
CHUNKS = int(os.environ.get("SL_CHUNKS", "3"))


def _same(a, b):
    a, b = a.float(), b.float()
    nan = torch.isnan(a) & torch.isnan(b)
    return bool(((a == b) | nan).all()), float((a - b).abs().nan_to_num(0).max())


@mesh_parametrize
def test_striped_latent(mesh_device):
    from models.demos.glm53_flash_d_p.tt import mla_attention
    from models.demos.glm53_flash_d_p.tt.common import residual_layout, split_from_host
    from models.demos.glm53_flash_d_p.tt.model import TtGlmBlock

    hooks = S.hooks()
    hooks.apply_device_settings(S)
    loader, cfg = hooks._loader_cfg(S)
    assert residual_layout() == "split"
    max_seq = CHUNK * (CHUNKS + 1)

    def build(striped):
        mla_attention.STRIPED = striped
        b = TtGlmBlock(mesh_device, cfg, loader, LAYER, max_seq, [CHUNK], layout="split", experts_dtype=hooks.experts_dtype(S))
        assert b.attn.striped == striped, (b.attn.striped, striped)
        return b

    rep, stp = build(False), build(True)
    torch.manual_seed(0)
    xs = [torch.randn(1, 1, CHUNK, cfg.hc_mult * cfg.hidden_size) * 0.5 for _ in range(CHUNKS + 1)]

    def run(blk, k):
        x = split_from_host(mesh_device, xs[k])
        y = blk(x, k * CHUNK)
        out = torch.cat([ttnn.to_torch(t).reshape(-1, t.shape[-1]) for t in ttnn.get_device_tensors(y)])
        ttnn.deallocate(y)
        ttnn.deallocate(x)
        return out

    ok = True
    for k in range(CHUNKS):
        a, b = run(rep, k), run(stp, k)
        same, d = _same(a, b)
        la, lb = rep.attn.state_torch()["kv_latent"], stp.attn.state_torch()["kv_latent"]
        lsame, ld = _same(la, lb)
        print(f"SL chunk {k} at {k * CHUNK}: output bit-identical {same} (max |d| {d:.3g}); latent state identical {lsame} (max |d| {ld:.3g})", flush=True)
        ok &= same and lsame

    # prefix reload into a fresh striped block, then one more chunk on both
    st = rep.state_torch(CHUNKS * CHUNK)
    stp2 = build(True)
    stp2.load_state(st, CHUNKS * CHUNK)
    a, b = run(rep, CHUNKS), run(stp2, CHUNKS)
    same, d = _same(a, b)
    print(f"SL reload + chunk {CHUNKS}: output bit-identical {same} (max |d| {d:.3g})", flush=True)
    ok &= same
    assert ok
