import torch, ttnn, math, os
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd
from eval.metrics import check_output

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def run(ashape, bshape, dt, bdt, acc, bl1, **kw):
    A = torch.randn(*ashape)
    B = torch.randn(*bshape) / math.sqrt(ashape[-1])
    ta = create_ttnn_input_tensor(A, device, dtype=dt, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(
        B,
        device,
        dtype=bdt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG if bl1 else ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = ttnn.to_torch(genify_matmul(ta, tb, compute_kernel_config=cfg, **kw)).float()
    ref = torch.matmul(ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float())
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    rms = ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    return pcc, rms


for g, r in [
    ("32,1,3", "weight"),
    ("32,3,1", "act"),
    ("32,3,1", "weight"),
    ("32,1,3", "act"),
    ("", "weight"),
    ("", "act"),
]:
    os.environ["GENIFY_FORCE_GRID"] = g
    os.environ["GENIFY_RES"] = r
    if not g:
        del os.environ["GENIFY_FORCE_GRID"]
    print(g, r, run((1, 32, 512, 128), (1, 32, 128, 1024), ttnn.bfloat8_b, ttnn.bfloat8_b, False, True))
for r in ["weight", "act", "none"]:
    os.environ["GENIFY_RES"] = r
    os.environ.pop("GENIFY_FORCE_GRID", None)
    print("2048x128x3072", r, run((1, 1, 2048, 128), (128, 3072), ttnn.bfloat16, ttnn.bfloat16, True, False))
    print("ragged 300x96x200", r, run((1, 1, 300, 96), (96, 200), ttnn.bfloat16, ttnn.bfloat16, True, False))
ttnn.close_device(device)
