import torch
import ttnn


def main():
    # Open Tenstorrent device
    device = ttnn.open_device(device_id=0)

    try:
        # Create two PyTorch tensors filled with 1s and 2s
        torch_tensor1 = torch.full((32, 32), 1.0, dtype=torch.float32)
        torch_tensor2 = torch.full((32, 32), 2.0, dtype=torch.float32)

        # Print input tensors
        print("Input tensors:")
        print(torch_tensor1)
        print(torch_tensor2)

        # Convert PyTorch tensors to TT-NN tensors with TILE_LAYOUT
        tt_tensor1 = ttnn.from_torch(torch_tensor1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_tensor2 = ttnn.from_torch(torch_tensor2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Perform eltwise addition on the device
        tt_result = ttnn.add(tt_tensor1, tt_tensor2)

        # Convert the result back to a PyTorch tensor for inspection
        torch_result = ttnn.to_torch(tt_result)

        # Print output tensor
        print("Output tensor:")
        print(torch_result)

    finally:
        # Close Tenstorrent device
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
