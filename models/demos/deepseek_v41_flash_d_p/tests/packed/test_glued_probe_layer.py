# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7 diagnostic probe (not gated): where a glued segment first differs from the same segment alone, inside one
layer. Runs the oracle's composition GLUED_PROBE_COMP (default glued2) and GLUED_PROBE_SEG (default P) alone through
DS41_PACKED_LAYERS, and at layer GLUED_PROBE_LAYER captures the segment's rows of: the attention input (attn_norm, in
the segment's own [0, n) geometry after the GluedMover), every attention step's output that is an OWN_ROWS-like
tensor, the attention output, the MoE input and output (model rows) and the layer output. Prints the first that
differs.

    BRINGUP_CHUNK=8192 DS41_PACKED=on ... DS41_PACKED_GLUED=on DS41_PACKED_LAYERS=0,..,20 GLUED_PROBE_LAYER=20 \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_probe_layer.py
"""

from __future__ import annotations

import importlib
import os

import pytest

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


def _rows0(model, t, n):
    """rows [0, n) of a device tensor whose rows are the chips' own rows (chip-linear), else None."""
    try:
        if not isinstance(t, ttnn.Tensor) or len(t.shape) != 4:
            return None
        return _pc.model_rows(model, t, 0, n)
    except Exception:  # noqa: BLE001
        return None


@mesh_parametrize
def test_glued_probe_layer(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_v2_adapter as mba
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as pd
    from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk

    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    if model.skip_mode != "off":
        model.use_decoder_skip("off")
    layers = O._layers(model)
    L = int(os.environ.get("GLUED_PROBE_LAYER", "20"))
    comp = os.environ.get("GLUED_PROBE_COMP", "glued2")
    seg = os.environ.get("GLUED_PROBE_SEG", "P")
    names = O.COMPS[comp]
    run = O.Run(Golden.for_rung(S, "s56320").tokens())
    n_rows = {}
    cap = {"alone": {}, "glued": {}}
    side = {"now": None}
    orig_seg_steps, orig_moe = pd.PackedChunk._seg_steps, mba.TtMoEBlockV2Layer.__call__

    def seg_steps(self, block, r, x, names=pd.ATTN_STEPS):
        tgt = side["now"] is not None and block.layer == L and r.state is side["state"]
        if not tgt:
            return orig_seg_steps(self, block, r, x, names)
        n = r.run_end - r.start
        n_rows["n"] = n
        c = cap[side["now"]]
        c["attn_in"] = _rows0(model, x, n)
        fns = block.fns
        wrapped = {}
        for s in block.steps:
            if s.name in names:
                f = fns[s.name]

                def w(ctx, *a, _f=f, _nm=s.name):
                    out = _f(ctx, *a)
                    for k, t in enumerate(out if isinstance(out, (tuple, list)) else [out]):
                        v = _rows0(model, t, n)
                        if v is not None:
                            c[f"step:{_nm}[{k}]"] = v
                    return out

                wrapped[s.name] = w
        saved = dict(fns)
        fns.update(wrapped)
        try:
            y = orig_seg_steps(self, block, r, x, names)
        finally:
            fns.clear()
            fns.update(saved)
        c["attn_out"] = _rows0(model, y, n)
        return y

    def moe_call(self, x, *a, **kw):
        y = orig_moe(self, x, *a, **kw)
        if side["now"] is not None and self.layer == L:
            off, n = side["off"], n_rows.get("n", 0)
            if n:
                cap[side["now"]]["moe_in"] = _pc.model_rows(model, x, off, n)
                cap[side["now"]]["moe_out"] = _pc.model_rows(model, y, off, n)
        return y

    orig_lean = mba._lean_out
    cur_layer = {"i": None}

    def lean_out(mesh, yt, se):  # (the MoE's routed owner rows yt and the shared expert's se, before their add)
        if side["now"] is not None and cur_layer["i"] == L and n_rows.get("n"):
            off, n = side["off"], n_rows["n"]
            cap[side["now"]]["moe_routed"] = _pc.model_rows(model, yt, off, n)
            cap[side["now"]]["moe_shared"] = _pc.model_rows(model, se, off, n)
        return orig_lean(mesh, yt, se)

    def moe_call2(self, x, *a, **kw):
        cur_layer["i"] = self.layer
        try:
            return moe_call(self, x, *a, **kw)
        finally:
            cur_layer["i"] = None

    orig_add = O.Capture.add

    def cap_add(self, key, rows):
        orig_add(self, key, rows)
        if side["now"] is not None and key == (seg, L):
            cap[side["now"]]["layer_out_full"] = rows.clone()

    pd.PackedChunk._seg_steps = seg_steps
    mba.TtMoEBlockV2Layer.__call__ = moe_call2
    mba._lean_out = lean_out
    O.Capture.add = cap_add
    try:
        # alone
        st = O._fresh_state(model, layers, run, seg)
        side.update(now="alone", state=st, off=0)
        a = O._chunk(model, layers, [(seg, run.seg(seg), run.toks(seg)[O.SEGS[seg]["start"] :])], [st], "on", "alone")
        st.free()
        # glued
        states = {k: O._fresh_state(model, layers, run, k) for k in names}
        desc = GluedChunk(O.CHUNK, tuple(run.seg(k) for k in names))
        side.update(now="glued", state=states[seg], off=desc.lane_off(names.index(seg)))
        items = [(k, run.seg(k), run.toks(k)[O.SEGS[k]["start"] :]) for k in names]
        g = O._chunk(model, layers, items, [states[k] for k in names], "on", comp)
        for s2 in states.values():
            s2.free()
    finally:
        side["now"] = None
        pd.PackedChunk._seg_steps = orig_seg_steps
        mba.TtMoEBlockV2Layer.__call__ = orig_moe
        mba._lean_out = orig_lean
        O.Capture.add = orig_add
    print(f"probe {comp} {seg} layer {L} (model row {side['off']}), n {n_rows.get('n')}")
    for i in layers:
        if i > L:
            break
        d = O.Capture.diff(g, a, [(seg, i)])
        print(f"  layer {i} output: {d}")
    for k in cap["alone"]:
        va, vg = cap["alone"][k], cap["glued"].get(k)
        if va is None or vg is None or va.shape != vg.shape:
            print(
                f"  {k}: shapes {None if va is None else tuple(va.shape)} / {None if vg is None else tuple(vg.shape)}"
            )
            continue
        d = (va - vg).abs()
        rows = (d.amax(-1) > 0).nonzero().reshape(-1)
        cols = (d.amax(0) > 0).nonzero().reshape(-1)
        if rows.numel():
            print(
                f"    cols differ {cols.numel()} of {va.shape[1]} first {cols[:8].tolist()}; rows {rows[:32].tolist()}"
            )
        print(
            f"  {k}: max abs {float(d.max())}, rows differ {rows.numel()} of {va.shape[0]}"
            f"{' first ' + str(rows[:8].tolist()) if rows.numel() else ''}"
        )
