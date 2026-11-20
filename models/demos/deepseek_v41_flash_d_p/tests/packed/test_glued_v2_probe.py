# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7B diagnostic probe (not gated): where a glued segment first differs from the same segment alone inside layer
GLUED_PROBE_LAYER under PACKED_GLUED_ONECALL=on (tt/packed_glued.GluedCall). Runs GLUED_PROBE_SEG (default O) alone and
in GLUED_PROBE_COMP (default glued1) through DS41_PACKED_LAYERS and compares the segment's rows of: the layer's top-k
(the indexer's, combined), the attention input of the out-projection (the sparse_sdpa output, rows of every head) and
the layer output.

    BRINGUP_CHUNK=8192 <the gate's PB.7B env> DS41_PACKED_LAYERS=0,1,2 GLUED_PROBE_LAYER=2 \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_v2_probe.py
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

import ttnn
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tests.packed import test_glued_oracle as O

O = importlib.import_module(O.__name__)
_pc = importlib.reload(_pc)
S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("DS41_PACKED_LAYERS", "GLUED_PROBE_LAYER", "GLUED_PROBE_COMP", "GLUED_PROBE_SEG")


def _chip_rows(t: ttnn.Tensor, raw: bool = False) -> torch.Tensor:
    """[1, B, R, W] each chip its own rows -> host [B, chips R, W] (chip-linear); ``raw``: the bf16 bits as int16."""
    parts = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
    parts = [p.view(torch.int16) if raw else p.float() for p in parts]
    b, r, w = int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
    return torch.cat([p.reshape(b, r, w) for p in parts], dim=1)


@mesh_parametrize
def test_glued_v2_probe(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt import indexer as ixm
    from models.demos.deepseek_v41_flash_d_p.tt import packed_glued as pg
    from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk

    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    if model.skip_mode != "off":
        model.use_decoder_skip("off")
    layers = O._layers(model)
    L = int(os.environ.get("GLUED_PROBE_LAYER", "2"))
    comp = os.environ.get("GLUED_PROBE_COMP", "glued1")
    seg = os.environ.get("GLUED_PROBE_SEG", "O")
    names = O.COMPS[comp]
    run = O.Run(Golden.for_rung(S, "s56320").tokens())
    cap = {"alone": {}, "glued": {}}
    side = {"now": None, "layer": None}
    att_cls = type(model.blocks[L].attention)
    orig = dict(
        layer=type(model).layer,
        topk=ixm.topk_large_indices,
        out=att_cls._out,
        keep=pg.GluedCall._keep,
    )

    def layer(self, i, *a, **kw):
        side["layer"] = i
        try:
            return orig["layer"](self, i, *a, **kw)
        finally:
            side["layer"] = None

    ring0 = ttnn.bringup.ring_indexer_score_dsa

    def ring(*a, **kw):
        out = ring0(*a, **kw)
        if side["now"] is not None and side["layer"] == L:
            cap[side["now"]].setdefault("logits", []).append((_chip_rows(out), kw.get("kv_len"), kw.get("lane_starts")))
            cap[side["now"]].setdefault("bits", []).append(_chip_rows(out, raw=True))
            if side["now"] == "glued" and "topk_scalar" not in cap["glued"]:  # (the first call: pass 0)
                ix = model.blocks[L].indexer
                kv = int(os.environ.get("GLUED_PROBE_KV", "6144"))
                t1 = orig["topk"](out, k=ix.topk, valid_length=kv)
                cap["glued"]["topk_scalar"] = _chip_rows(t1)
                ttnn.deallocate(t1)
                vl = pg._chip_u32(mesh_device, tuple([kv] * (mesh_device.shape[0] * mesh_device.shape[1])))
                t2 = orig["topk"](out, k=ix.topk, valid_length_tensor=vl)
                cap["glued"]["topk_tensor"] = _chip_rows(t2)
                ttnn.deallocate(t2)
        return out

    def topk(x, *a, **kw):
        out = orig["topk"](x, *a, **kw)
        ix = model.blocks[L].indexer
        if side["now"] == "alone" and side["layer"] == L and kw.get("k") == ix.topk:
            cap["alone"]["topk"] = _chip_rows(out)
            cap["alone"]["ncalls"] = cap["alone"].get("ncalls", 0) + 1
            print(f"  alone topk call {cap['alone']['ncalls']}: kw {sorted(kw)} x {tuple(x.shape)}")
        return out

    def keep(self, key, t):
        if side["now"] == "glued" and key == ("top", L):
            cap["glued"]["topk"] = _chip_rows(t)
        return orig["keep"](self, key, t)

    def out(self, o, cos, sin):
        if side["now"] is not None and side["layer"] == L:
            cap[side["now"]]["sdpa"] = _chip_rows(o)
        return orig["out"](self, o, cos, sin)

    type(model).layer = layer
    ttnn.bringup.ring_indexer_score_dsa = ring
    ixm.topk_large_indices = topk
    att_cls._out = out
    pg.GluedCall._keep = keep
    try:
        st = O._fresh_state(model, layers, run, seg)
        side["now"] = "alone"
        a = O._chunk(model, layers, [(seg, run.seg(seg), run.toks(seg)[O.SEGS[seg]["start"] :])], [st], "on", "alone")
        st.free()
        side["now"] = None  # (the glued run's history prefixes are not captured)
        states = {k: O._fresh_state(model, layers, run, k) for k in names}
        desc = GluedChunk(O.CHUNK, tuple(run.seg(k) for k in names))
        k = names.index(seg)
        off, n = desc.lane_off(k), desc.rows(k)
        side["now"] = "glued"
        items = [(x, run.seg(x), run.toks(x)[O.SEGS[x]["start"] :]) for x in names]
        g = O._chunk(model, layers, items, [states[x] for x in names], "on", comp)
        for s2 in states.values():
            s2.free()
    finally:
        side["now"] = None
        type(model).layer = orig["layer"]
        ttnn.bringup.ring_indexer_score_dsa = ring0
        ixm.topk_large_indices = orig["topk"]
        att_cls._out = orig["out"]
        pg.GluedCall._keep = orig["keep"]
    R = 8192 // (mesh_device.shape[0] * mesh_device.shape[1])
    print(f"probe {comp} {seg} layer {L}: model rows [{off}, {off + n}), chips {off // R}..{(off + n - 1) // R}")
    for i in layers:
        if i > L:
            break
        print(f"  layer {i} output: {O.Capture.diff(g, a, [(seg, i)])}")
    la, lg = cap["alone"].get("logits", []), cap["glued"].get("logits", [])
    print(
        f"  logits calls alone {[(tuple(t.shape), kv) for t, kv, _ in la]} glued {[(tuple(t.shape), kv) for t, kv, _ in lg]}"
    )
    if la and lg:
        ta, kva, _ = la[-1]
        for c, (tg, kvg, st) in enumerate(lg):
            w = min(int(kva), int(tg.shape[-1]))
            va, vg = ta[:, :n, :w], tg[:, off : off + n, :w]
            fin = torch.isfinite(va) & torch.isfinite(vg)
            d = torch.where(fin, (va - vg).abs(), torch.zeros_like(va)).amax(dim=(0, 2))
            inf_d = (torch.isfinite(va) != torch.isfinite(vg)).any(dim=2).any(dim=0)
            rows = ((d > 0) | inf_d).nonzero().reshape(-1)
            print(f"  logits call {c} (kv_len {kvg}): rows differ {rows.numel()} first {rows[:12].tolist()}")
    ba, bg = cap["alone"].get("bits"), cap["glued"].get("bits")
    if ba and bg:
        w = 6144
        xa, xg = ba[-1][:, 192:n, :w], bg[0][:, off + 192 : off + n, :w]
        neq = xa != xg
        rows = neq.any(dim=2).any(dim=0).nonzero().reshape(-1)
        print(f"  logits bits (O rows 192.., cols < {w}): rows differ {rows.numel()}, values differ {int(neq.sum())}")
        if rows.numel():
            idx = neq.nonzero()[:6].tolist()
            for b, r, c in idx:
                print(
                    f"    row {192 + r} col {c}: alone 0x{int(xa[b, r, c]) & 0xFFFF:04x} glued 0x{int(xg[b, r, c]) & 0xFFFF:04x}"
                )
    ta = cap["alone"].get("topk")
    for kname in ("topk_scalar", "topk_tensor"):
        v = cap["glued"].get(kname)
        if v is not None and ta is not None:
            a7, g7 = ta[:, 192:n], v[:, off + 192 : off + n]  # (O's pass-0 rows: chips 7-9)
            rows = ((a7 - g7).abs().amax(dim=(0, 2)) > 0).nonzero().reshape(-1)
            print(f"  {kname} of the glued call-0 logits vs alone, O rows 192..: {rows.numel()} differ")
    for kname in ("topk", "sdpa"):
        va, vg = cap["alone"].get(kname), cap["glued"].get(kname)
        if va is None or vg is None:
            print(f"  {kname}: not captured (alone {va is not None}, glued {vg is not None})")
            continue
        sent = lambda v: float((v >= 4.29e9).float().mean())  # noqa: E731
        print(f"  {kname}: shapes alone {tuple(va.shape)} glued {tuple(vg.shape)}")
        va, vg = va[:, :n], vg[:, off : off + n]
        if kname == "topk":
            print(f"  topk sentinel fraction alone {sent(va):.3f} glued {sent(vg):.3f}")
        d = (va - vg).abs().amax(dim=(0, 2))  # per row
        rows = (d > 0).nonzero().reshape(-1)
        print(
            f"  {kname}: rows differ {rows.numel()} of {n}: first {rows[:16].tolist()} last {rows[-8:].tolist()}; "
            f"max {float(d.max())}"
        )
        if kname == "topk" and rows.numel():
            r0 = int(rows[0])
            sa, sg = set(va[0, r0].long().tolist()), set(vg[0, r0].long().tolist())
            print(f"    row {r0}: |alone - glued| {len(sa - sg)}, |glued - alone| {len(sg - sa)}")
            print(f"    alone {va[0, r0, :12].long().tolist()} glued {vg[0, r0, :12].long().tolist()}")
