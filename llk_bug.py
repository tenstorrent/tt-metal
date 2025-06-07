import ttnn
import torch


def test_llk_repro():
    x = torch.randint(-100, 100, (1, 1, 32, 32), dtype=torch.bfloat16)

    with ttnn.manage_device(0) as dev:
        ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        ttnn_result = ttnn.lt(ttnn_x, 5, use_legacy=True)
        print("ttnn_result:", ttnn_result)


if __name__ == "__main__":
    test_llk_repro()
