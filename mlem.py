import ttnn
import torch


def ttnn_convolution():
    print("=== TTNN 2D Convolution ===")

    device = ttnn.open_device(device_id=0, l1_small_size=10 * 1024)

    try:
        # input_torch = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
        #                             [5.0, 6.0, 7.0, 8.0],
        #                             [9.0, 10.0, 11.0, 12.0],
        #                             [13.0, 14.0, 15.0, 16.0]]]], dtype=torch.bfloat16)

        # weight_torch = torch.tensor([[[[0.0, 1.0, 2.0],
        #                                 [3.0, 4.0, 5.0],
        #                                 [6.0, 7.0, 8.0]]]], dtype=torch.bfloat16)

        input_torch = torch.randn([1, 1, 1000, 1000], dtype=torch.bfloat16)

        weight_torch = torch.randn([1, 1, 3, 3], dtype=torch.bfloat16)

        input_ttnn = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        weight_ttnn = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        output_ttnn = ttnn.conv2d(
            input_tensor=input_ttnn,
            weight_tensor=weight_ttnn,
            device=device,
            in_channels=1,
            out_channels=1,
            batch_size=1,
            input_height=1000,
            input_width=1000,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
        )

        output_torch = ttnn.to_torch(output_ttnn)
        # output_torch = torch.reshape(output_torch, (1, 1, 4, 4))

        print(f"\nOutput shape: {output_torch.shape}")
        # print(f"Output:\n{output_torch.squeeze()}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    ttnn_convolution()
