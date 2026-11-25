# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""tt/pf_tiles.py: the expert tensors rebuilt on the card from a PF-layout cache are the TILE cache's, byte for
byte. Needs the card and both caches of the run mode (skips without them)."""

import numpy as np
import pytest
import torch

import ttnn
from models.demos.inkling2_mini import modes
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.convert import stick
from models.demos.inkling2_mini.tt import pf_tiles
from models.demos.inkling2_mini.tt.device import open_device
from models.demos.inkling2_mini.tt.fused.moe_bank import pf_memory_config

PF, TILE = modes.roots()


def _tiles(t, path):
    ttnn.dump_tensor(str(path), t.cpu())
    dtype = "BFP4" if t.dtype == ttnn.bfloat4_b else "BFP8"
    return dtype, np.array(stick._read(path, dtype)[1])


@pytest.mark.skipif(
    not ((PF / "manifest.json").is_file() and (TILE / "manifest.json").is_file()), reason="needs the PF and TILE caches"
)
@pytest.mark.parametrize("layer", [0, 12])  # a BFP4 and a BFP8 gate_up in the default map
def test_rebuilt_experts_are_the_tile_cache(layer, tmp_path):
    assert wcache.layouts(PF, layer).get("experts.w_gate_up") == "PF", "the decode cache is not PF-layout"
    w = wcache.load_layer(PF, layer, None)
    dev = open_device()
    try:
        for name, fn in (("experts.w_gate_up", pf_tiles.gate_up), ("experts.w_down", pf_tiles.down)):
            main = ttnn.to_device(w[name], dev, pf_memory_config(w[name].shape[1], w[name].shape[2]))
            extra = ttnn.to_device(w[name + ".x"], dev, ttnn.DRAM_MEMORY_CONFIG)
            out = fn(main, extra)
            dtype, got = _tiles(out, tmp_path / "t.tensorbin")
            ref = stick._read(TILE / "layers" / f"L{layer:02d}" / f"{name}.tensorbin", dtype)[1]
            assert got.shape == ref.shape and np.array_equal(got, ref), f"L{layer} {name}"
            for t in (out, main, extra):
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)


@pytest.mark.skipif(not (PF / "manifest.json").is_file(), reason="needs the PF cache")
@pytest.mark.parametrize("layer", [0, 12])
def test_expert_op_reads_pf_in_place(layer):
    """unified_routed_expert_moe on the PF tensors in place (gate_up_extra / down_extra) gives the same output, bit
    for bit, as on the plain tensors rebuilt from them; 64 tokens routed to random top-8 experts."""
    from models.demos.inkling2_mini.tt import prefill_model as PM

    op = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe
    if "gate_up_extra" not in (op.__doc__ or ""):
        pytest.skip("this build's expert op has no PF-layout path")
    w = wcache.load_layer(PF, layer, None)
    dev = open_device()
    try:
        main = {n: ttnn.to_device(w[n], dev, pf_memory_config(w[n].shape[1], w[n].shape[2]))
                for n in ("experts.w_gate_up", "experts.w_down")}  # fmt: skip
        extra = {n: ttnn.to_device(w[n + ".x"], dev, ttnn.DRAM_MEMORY_CONFIG) for n in main}
        plain = (pf_tiles.gate_up(main["experts.w_gate_up"], extra["experts.w_gate_up"]),
                 pf_tiles.down(main["experts.w_down"], extra["experts.w_down"]))  # fmt: skip
        g = torch.Generator().manual_seed(layer)
        t, E, k = 64, PM.NUM_EXPERTS, PM.TOP_K
        flat = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(t)]).reshape(-1).numpy()
        counts = np.bincount(flat, minlength=E)
        cap = -(-counts // 32) * 32
        off = np.concatenate([[0], np.cumsum(cap)[:-1]])
        rows = -(-(t * k + E * 31) // 32) * 32 + 32
        order = np.argsort(flat, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size) - np.concatenate([[0], np.cumsum(counts)[:-1]])[flat[order]]
        xri = np.full(rows, t, dtype=np.int32)
        xri[off[flat] + rank] = np.arange(flat.size) // k
        rm = ttnn.ROW_MAJOR_LAYOUT

        def u32(v):
            return ttnn.from_torch(torch.as_tensor(np.asarray(v, dtype=np.int32)).view(1, -1), dtype=ttnn.uint32,
                                   layout=rm, device=dev)  # fmt: skip

        x = ttnn.from_torch(torch.randn(t, PM.HIDDEN, generator=g) * 0.5, dtype=ttnn.bfloat16, layout=rm, device=dev)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True
        )
        kw = PM._has_unified(dev, plain[0], plain[1], ttnn.MathFidelity.LoFi) or {}

        def run(gu, dn, **xkw):
            y = op(x, u32(off), u32(counts), u32(range(E)), [gu] * E, [gu] * E, [dn] * E, t,
                   compute_kernel_config=ckc, activation=ttnn.RoutedExpertActivation.SituGlu, x_row_index=u32(xri),
                   output_row_major=True, zero_output_row=rows - 32, **kw, **xkw)  # fmt: skip
            return ttnn.to_torch(y).float()

        want = run(*plain)
        got = run(main["experts.w_gate_up"], main["experts.w_down"], gate_up_extra=extra["experts.w_gate_up"],
                  down_extra=extra["experts.w_down"])  # fmt: skip
        used = torch.from_numpy(xri < t)  # the rows of the expert regions (the others are not written)
        want, got = want.reshape(rows, -1)[used], got.reshape(rows, -1)[used]
        assert torch.isfinite(want).all() and want.abs().max() > 0
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), float((got - want).abs().max())
    finally:
        ttnn.close_device(dev)
