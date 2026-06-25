import torch
import ttnn

device = ttnn.open_device(device_id=0)

for dtype in [ttnn.bfloat16, ttnn.bfloat8_b]:
    for shape in [(1, 1, 32, 32), (1, 1, 64, 128)]:
        for dim in [-1, -2]:
            torch.manual_seed(42)
            torch_dtype = torch.bfloat16 if dtype != ttnn.float32 else torch.float32
            torch_input = torch.randn(shape, dtype=torch_dtype)

            ttnn_input = ttnn.from_torch(
                torch_input,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
            torch_output = ttnn.to_torch(ttnn_output)

            expected = torch.softmax(torch_input.float(), dim=dim)

            from tests.ttnn.utils_for_testing import check_with_pcc

            passed, msg = check_with_pcc(expected, torch_output, pcc=0.99)
            print(f"dtype={dtype}, shape={shape}, dim={dim}: PCC pass={passed}, msg={msg}")

ttnn.close_device(device)
