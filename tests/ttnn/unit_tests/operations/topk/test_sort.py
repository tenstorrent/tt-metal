import ttnn
import torch
import tests.ttnn.unit_tests.operations.topk.utils as utils

# torch.set_printoptions(
#     threshold=torch.inf,  # print all elements
#     precision=6,          # decimal places
#     linewidth=200,        # characters per line before wrapping
# )

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten().to(torch.float64), y.flatten().to(torch.float64)
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


def test_sort():
    device = utils.DeviceGetter.get_device((1, 1))
    mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    # Load input
    input_tt = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        mem_cfg,
    )

    # Convert to pytorch tensor
    input_torch = ttnn.to_torch(input_tt)

    # TT inference: ttnn.sort
    _, tt_indices = ttnn.sort(
        input_tt, 1, True, False, memory_config=mem_cfg
    )
    tt_indices_torch = ttnn.to_torch(tt_indices)

    # CPU inference: torch.sort on the same tensor
    _, cpu_indices = torch.sort(input_torch, dim=1, descending=True, stable=False)

    # Comparison
    indices_pcc = compute_pcc(cpu_indices, tt_indices_torch)

    print("\n" + "=" * 60)
    print("  SORT OUTPUT COMPARISON")
    print("=" * 60)
    print(f"  {'Input shape':<25}: {input_torch.shape}")
    print(f"  {'Input dtype':<25}: {input_torch.dtype}")
    print("-" * 60)
    print(f"  {'CPU indices shape':<25}: {cpu_indices.shape}")
    print(f"  {'CPU indices dtype':<25}: {cpu_indices.dtype}")
    print(f"  {'TT  indices shape':<25}: {tt_indices_torch.shape}")
    print(f"  {'TT  indices dtype':<25}: {tt_indices_torch.dtype}")
    print("-" * 60)
    print(f"  {'Indices PCC':<25}: {indices_pcc}")
    print(f"  {'Indices torch.equal':<25}: {torch.equal(cpu_indices, tt_indices_torch)}")
    print("-" * 60)
    print(f"  CPU indices: {cpu_indices}")
    print(f"  TT  indices: {tt_indices_torch}")
    print("-" * 60)
    print(f"  Input: {input_torch}")
    print("=" * 60)

