import math, torch, ttnn
from models.common.utility_functions import comp_pcc
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, pytorch_genify_matmul, quantize
from ttnn.operations.genify_matmul import genify_matmul
import ttnn.operations.genify_matmul.genify_matmul_program_descriptor as pd

orig = pd._make_blocking
CAP = [None]
P32 = [False]


def capped(geom, block_b, bm, bn, bk_max):
    if CAP[0] is not None:
        bk_max = min(bk_max, CAP[0])
    return orig(geom, block_b, bm, bn, bk_max)


pd._make_blocking = capped
orig_geom = pd._geometry


def geom32(a, b, out, acc):
    g = orig_geom(a, b, out, acc)
    if P32[0]:
        import dataclasses

        g = dataclasses.replace(g, p_dtype=ttnn.float32, p_tile_bytes=ttnn.tile_size(ttnn.float32))
    return g


pd._geometry = geom32
device = ttnn.open_device(device_id=0)
try:
    for ash, bsh in [((1, 64, 16384), (16384, 256)), ((1, 64, 2048), (2048, 256))]:
        torch.manual_seed(4321)
        A = torch.randn(ash)
        B = torch.randn(bsh) / math.sqrt(bsh[-2])
        ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        exp = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, ttnn.bfloat16))
        for p32 in [False, True]:
            for cap in [None, 32, 8, 2]:
                CAP[0] = cap
                P32[0] = p32
                cfg = ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
                )
                out = genify_matmul(ta, tb, compute_kernel_config=cfg)
                got = ttnn.to_torch(out).float()
                _, pcc = comp_pcc(exp, got)
                rr = ((got - exp).pow(2).mean().sqrt() / exp.std()).item()
                bias = ((got - exp).mean() / exp.std()).item()
                print(f"PROBE {ash} p32={p32} cap={cap} pcc={pcc:.5f} rr={rr:.4f} meanerr/std={bias:.4f}")
finally:
    ttnn.close_device(device)
