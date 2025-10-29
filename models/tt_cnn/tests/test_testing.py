# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.testing import create_random_input_tensor


class TestCreateRandomInputTensor:
    """Tests for the create_random_input_tensor function"""

    def test_tensor_creation_no_device(self):
        """Test basic tensor creation without device"""
        batch = 2
        groups = 1
        input_channels = 32
        input_height = 64
        input_width = 64

        torch_tensor, ttnn_tensor = create_random_input_tensor(
            batch=batch,
            groups=groups,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            fold=True,
            pad=True,
            device=None,
        )

        # Check torch tensor shape (NCHW)
        expected_torch_shape = (batch, input_channels * groups, input_height, input_width)
        assert torch_tensor.shape == expected_torch_shape, f"Expected {expected_torch_shape}, got {torch_tensor.shape}"

        # Check ttnn tensor - it should be folded by default with channel_order="last"
        ttnn_torch = ttnn.to_torch(ttnn_tensor)
        expected_ttnn_shape = (batch, 1, input_height * input_width, input_channels)
        assert ttnn_torch.shape == expected_ttnn_shape, f"Expected {expected_ttnn_shape}, got {ttnn_torch.shape}"

    def test_channel_order_first(self):
        """Test with channel_order='first'"""
        batch = 1
        groups = 1
        input_channels = 8
        input_height = 32
        input_width = 32

        torch_tensor, ttnn_tensor = create_random_input_tensor(
            batch=batch,
            groups=groups,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            channel_order="first",
            pad=False,
            device=None,
        )
        ttnn_torch = ttnn.to_torch(ttnn_tensor)

        # With fold=True and channel_order="first", shape should be (batch, 1, channels, H*W)
        expected_shape = (batch, 1, input_channels, input_height * input_width)
        assert ttnn_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_torch.shape}"

    def test_channel_order_last(self):
        """Test with channel_order='last'"""
        batch = 1
        groups = 1
        input_channels = 8
        input_height = 32
        input_width = 32

        torch_tensor, ttnn_tensor = create_random_input_tensor(
            batch=batch,
            groups=groups,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            channel_order="last",
            device=None,
        )
        ttnn_torch = ttnn.to_torch(ttnn_tensor)

        # With fold=True and channel_order="last", shape should be (batch, 1, H*W, channels)
        expected_shape = (batch, 1, input_height * input_width, 16)  # padded to 16
        assert ttnn_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_torch.shape}"

    def test_no_fold(self):
        """Test with fold=False"""
        batch = 2
        groups = 1
        input_channels = 16
        input_height = 64
        input_width = 64

        torch_tensor, ttnn_tensor = create_random_input_tensor(
            batch=batch,
            groups=groups,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            channel_order="last",
            fold=False,
            device=None,
        )
        ttnn_torch = ttnn.to_torch(ttnn_tensor)

        # With fold=False and channel_order="last", shape should be (B, H, W, C)
        expected_shape = (batch, input_height, input_width, input_channels)
        assert ttnn_torch.shape == expected_shape, f"Expected {expected_shape}, got {ttnn_torch.shape}"

    def test_groups_parameter(self):
        """Test with groups > 1"""
        batch = 1
        groups = 2
        input_channels = 8
        input_height = 32
        input_width = 32

        torch_tensor, ttnn_tensor = create_random_input_tensor(
            batch=batch,
            groups=groups,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            channel_order="first",
            device=None,
        )

        # Torch tensor should have channels = input_channels * groups
        expected_torch_shape = (batch, input_channels * groups, input_height, input_width)
        assert torch_tensor.shape == expected_torch_shape, f"Expected {expected_torch_shape}, got {torch_tensor.shape}"

    def test_different_input_sizes(self):
        """Test with different input dimensions"""
        test_sizes = [
            (32, 32),
            (64, 64),
            (128, 128),
            (224, 224),
            (480, 640),  # Non-square
        ]

        for height, width in test_sizes:
            torch_tensor, ttnn_tensor = create_random_input_tensor(
                batch=1,
                groups=1,
                input_channels=16,
                input_height=height,
                input_width=width,
                device=None,
            )

            assert torch_tensor.shape[2:] == (height, width), f"Expected height={height}, width={width}"
