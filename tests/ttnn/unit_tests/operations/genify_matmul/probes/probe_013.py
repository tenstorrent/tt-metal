import math, torch, ttnn
from models.common.utility_functions import comp_pcc
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, pytorch_genify_matmul, quantize
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul.genify_matmul_program_descriptor import plan


def stats(got, exp):
    _, pcc = comp_pcc(exp, got)
    rr = ((got - exp).pow(2).mean().sqrt() / exp.std()).item()
    mask = exp.abs() > 0.1 * exp.std()
    r = (got[mask] / exp[mask]).double()
    q = torch.quantile(r[: 1 << 24], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    big = exp.abs() > 2 * exp.std()
    rb = (got[big] / exp[big]).median().item()
    return f"pcc={pcc:.6f} rr={rr:.4f} ratio p5/50/95={q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f} ratio|big|={rb:.4f}"


device = ttnn.open_device(device_id=0)
try:
    for ash, bsh in [((1, 2048), (2048, 1536)), ((1, 384, 1536), (1536, 1024)), ((1, 64, 256), (256, 256))]:
        torch.manual_seed(4321)
        A = torch.randn(ash)
        B = torch.randn(bsh) / math.sqrt(bsh[-2])
        ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        exp = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, ttnn.bfloat16))
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
        )
        out = genify_matmul(ta, tb, compute_kernel_config=cfg)
        geom, grid, blk, _ = plan(ta, tb, out, False, device)
        print(
            f"PROBE {ash} genify nkb={blk.num_k_blocks} bk={blk.block_k_tiles} {stats(ttnn.to_torch(out).float(), exp)}"
        )
        for l1acc in [False, True]:
            ckc = ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=l1acc
            )
            s = ttnn.to_torch(ttnn.matmul(ta, tb, compute_kernel_config=ckc)).float()
            print(f"PROBE {ash} stock l1acc={l1acc} {stats(s, exp)}")
finally:
    ttnn.close_device(device)
