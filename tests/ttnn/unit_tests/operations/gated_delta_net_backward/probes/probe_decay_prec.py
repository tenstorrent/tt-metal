import sys

import torch
import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

sys.path.insert(0, "/localdev/mnedeljkovic/tt-metal/tests/ttnn/unit_tests/operations/gated_delta_net_backward")
from test_gated_delta_net_backward_debug import device_algorithm, make_inputs, read_block

gdn = sys.modules["ttnn.operations.gated_delta_net_backward.gated_delta_net_backward"]


def test_decay_precision(device):
    for gs in (0.5, 8.0):
        shape, chunk = (1, 128, 2, 64, 64), 32
        ref = make_inputs(shape, g_scale=gs)

        def dev(t):
            return ttnn.from_torch(
                t.to(torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        gated_delta_net_backward(
            dev(ref["q"]),
            dev(ref["k"]),
            dev(ref["v"]),
            dev(ref["g"]),
            dev(ref["beta"]),
            dev(ref["do"]),
            chunk_size=chunk,
        )
        sc = gdn._LAST_SCRATCH
        flat = ttnn.to_torch(sc["sc"]).double()
        smap, geo = sc["map"], sc["geo"]
        Ct = geo.Ct
        _, P_, _ = device_algorithm(ref["q"], ref["k"], ref["v"], ref["g"], ref["beta"], ref["do"], chunk_size=chunk)
        worst_abs = 0.0
        worst_rel = 0.0
        for it in range(geo.NI):
            bh, i = it // geo.NC, it % geo.NC
            b, h = bh // shape[2], bh % shape[2]
            vecs = read_block(flat, smap.base_vec + it * smap.st_vec, 3, 1)
            got = vecs[: 32 * Ct, 0]
            exp = P_[(b, h, i)]["decay"]
            worst_abs = max(worst_abs, float((got - exp).abs().max()))
            worst_rel = max(worst_rel, float(((got - exp).abs() / exp.abs().clamp_min(1e-9)).max()))
        print(
            f"DECAY g_scale={gs}: |decay|max={float(exp.abs().max()):.3f} abs_err={worst_abs:.4g} rel_err={worst_rel:.3g}"
        )
        # beta is a PURE COPY of the same gathered column: if it is exact the
        # loss is in the matmul, not in the datacopy.
        vecs = read_block(flat, smap.base_vec, 3, 1)
        bgot = vecs[32 * Ct : 64 * Ct, 0]
        bexp = ref["beta"][0, : 32 * Ct, 0].double()
        print(f"BETA  g_scale={gs}: rel_err={float(((bgot - bexp).abs() / bexp.abs().clamp_min(1e-9)).max()):.3g}")
