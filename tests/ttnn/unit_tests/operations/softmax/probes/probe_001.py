import torch, ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.softmax import softmax

device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 32, 4096), (1, 1, 32, 8192), (1, 1, 128, 4096), (2, 1, 64, 4096)]:
        torch.manual_seed(42)
        torch_input = torch.randn(shape, dtype=torch.float32)
        for ns in [True, False]:
            torch_expected = torch.softmax(torch_input, dim=-1)
            ttnn_input = ttnn.from_torch(
                torch_input,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn_output = softmax(ttnn_input, dim=-1, numeric_stable=ns)
            torch_output = ttnn.to_torch(ttnn_output)
            # PCC
            ref = torch_expected.flatten().double()
            act = torch_output.flatten().double()
            ref_mean, act_mean = ref.mean(), act.mean()
            num = ((ref - ref_mean) * (act - act_mean)).sum()
            den = (((ref - ref_mean) ** 2).sum() * ((act - act_mean) ** 2).sum()).sqrt()
            pcc = (num / den).item() if den.item() > 0 else 1.0
            # RMS rel
            rms = ((torch_output - torch_expected) ** 2).mean().sqrt().item()
            ref_std = torch_expected.std().item()
            rms_rel = rms / ref_std if ref_std > 0 else 0.0
            max_abs = (torch_output - torch_expected).abs().max().item()
            sum_along = torch_output.sum(dim=-1)
            sum_dev = (sum_along - 1).abs().max().item()
            print(
                f"  shape={shape} ns={ns}: PCC={pcc:.7f} RMS_rel={rms_rel:.5f} max_abs={max_abs:.3e} sum_dev={sum_dev:.3e}"
            )
finally:
    ttnn.close_device(device)
