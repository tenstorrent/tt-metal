# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, MaxPool2dConfiguration
from models.tt_cnn.tt.testing import (
    create_random_input_tensor,
    pad_channels_up_to_target,
    verify_conv2d_from_config,
    verify_maxpool2d_from_config,
)


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


class TestPadChannelsUpToTarget:
    """Tests for the pad_channels_up_to_target helper function"""

    def test_pad_channels_basic(self):
        """Test basic padding functionality"""
        input_tensor = torch.randn(1, 8, 32, 32)
        padded = pad_channels_up_to_target(input_tensor, target=16)

        assert padded.shape == (1, 16, 32, 32), f"Expected shape (1, 16, 32, 32), got {padded.shape}"
        assert torch.allclose(padded[:, :8, :, :], input_tensor), "Original channels should be preserved"
        assert torch.allclose(padded[:, 8:, :, :], torch.zeros(1, 8, 32, 32)), "Padded channels should be zeros"

    def test_pad_channels_no_padding_needed(self):
        """Test when channels already meet or exceed target"""
        input_tensor = torch.randn(2, 16, 64, 64)
        padded = pad_channels_up_to_target(input_tensor, target=16)
        assert torch.equal(padded, input_tensor), "Tensor should be unchanged when channels equal target"

        input_tensor = torch.randn(1, 32, 32, 32)
        padded = pad_channels_up_to_target(input_tensor, target=16)
        assert torch.equal(padded, input_tensor), "Tensor should be unchanged when channels exceed target"


class TestLayerUnitTestGenerators:
    """Tests for the layer unit test generator utilities"""

    @pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
    @pytest.mark.parametrize(
        "input_height,input_width,in_channels,out_channels,kernel_size,padding,stride",
        [
            (32, 32, 16, 32, (3, 3), (1, 1), (1, 1)),  # Basic 3x3 conv
            (56, 56, 64, 128, (1, 1), (0, 0), (1, 1)),  # 1x1 conv (ResNet-style)
            (64, 64, 32, 64, (3, 3), (1, 1), (2, 2)),  # Stride 2 (downsampling)
            (112, 112, 64, 128, (5, 5), (2, 2), (1, 1)),  # 5x5 kernel
        ],
    )
    def test_verify_conv2d_from_config(
        self, input_height, input_width, in_channels, out_channels, kernel_size, padding, stride, device
    ):
        """Test verify_conv2d_from_config with various configurations"""
        config = Conv2dConfiguration.with_random_weights(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=1,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        verify_conv2d_from_config(config, device)

    @pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
    @pytest.mark.parametrize(
        "input_height,input_width,channels,kernel_size,stride,ceil_mode",
        [
            (32, 32, 16, (2, 2), (2, 2), False),  # Basic 2x2 pool
            (64, 64, 32, (4, 4), (4, 4), False),  # Larger 4x4 kernel
            (33, 33, 16, (2, 2), (2, 2), True),  # Odd dimensions with ceil_mode
            (56, 56, 32, (3, 3), (2, 2), False),  # 3x3 kernel with stride 2
        ],
    )
    def test_verify_maxpool2d_from_config(
        self, input_height, input_width, channels, kernel_size, stride, ceil_mode, device
    ):
        """Test verify_maxpool2d_from_config with various configurations"""
        config = MaxPool2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            batch_size=1,
            kernel_size=kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
        )

        verify_maxpool2d_from_config(config, device)
