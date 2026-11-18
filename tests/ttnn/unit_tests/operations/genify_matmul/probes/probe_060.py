import sys, math, torch, ttnn

sys.path.insert(0, "ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_pair_placement")
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    A = torch.randn(1, 128, 5120)
    B = torch.randn(5120, 8704) / math.sqrt(5120)
    ta = create_ttnn_input_tensor(
        A, dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tb = create_ttnn_input_tensor(B, dev, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )

    def run():
        o = genify_matmul(
            ta,
            tb,
            dtype=ttnn.bfloat16,
            compute_kernel_config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activation="swiglu",
        )
        return ttnn.to_torch(o)

    base = run()
    import place_plugin, os

    os.environ["GMP_GRAD"] = "1"
    place_plugin.pytest_configure(None)
    grad = run()
    print(
        "BITCHECK equal=",
        torch.equal(base, grad),
        "maxdiff=",
        (base.float() - grad.float()).abs().max().item(),
        "shape",
        tuple(grad.shape),
    )
finally:
    ttnn.close_device(dev)
