import math, torch, ttnn
from models.common.utility_functions import comp_pcc
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, pytorch_genify_matmul, quantize
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul.genify_matmul_program_descriptor import plan

device = ttnn.open_device(device_id=0)
try:
    for ash, bsh in [((1, 64, 16384), (16384, 256)), ((1, 64, 4096), (4096, 256)), ((1, 64, 2048), (2048, 256))]:
        torch.manual_seed(4321)
        A = torch.randn(ash)
        B = torch.randn(bsh) / math.sqrt(bsh[-2])
        ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        exp = pytorch_genify_matmul(quantize(A, ttnn.bfloat16), quantize(B, ttnn.bfloat16))
        for acc in [False, True]:
            cfg = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
            )
            out = genify_matmul(ta, tb, compute_kernel_config=cfg)
            got = ttnn.to_torch(out).float()
            _, pcc = comp_pcc(exp, got)
            rr = ((got - exp).pow(2).mean().sqrt() / exp.std()).item()
            geom, grid, blk, _ = plan(ta, tb, out, acc, device)
            ckc = ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, packer_l1_acc=True
            )
            s = ttnn.to_torch(ttnn.matmul(ta, tb, compute_kernel_config=ckc)).float()
            _, pcc_s = comp_pcc(exp, s)
            rr_s = ((s - exp).pow(2).mean().sqrt() / exp.std()).item()
            print(
                f"PROBE {ash} acc={acc} genify pcc={pcc:.5f} rr={rr:.4f} nkb={blk.num_k_blocks} bk={blk.block_k_tiles} | stock pcc={pcc_s:.5f} rr={rr_s:.4f}"
            )
finally:
    ttnn.close_device(device)
