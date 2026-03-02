import torch
import torch.nn as nn
import math


def apply_channel_mask(tensor, mask):
    """Apply mask to tensor with channel dimension."""
    # tensor: (batch, channels, time, features)
    # mask: (batch, time, features)
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def calculate_conv_output_size(input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple[int, int]):
    """Calculate exact output size after convolution."""
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class MaskedConvSequential(nn.Sequential):
    """Sequential container with masking support for variable-length inputs."""

    def forward(self, x, lengths):
        # Convert input (batch, time, features) to conv format
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())

        # Process through each layer with mask propagation
        for i, layer in enumerate(self):
            # Apply current mask before layer
            x = apply_channel_mask(x, mask)

            # Apply layer
            x = layer(x)

            # Update lengths for stride operations with proper padding
            if hasattr(layer, "stride") and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (layer._left_padding, layer._right_padding)  # CausalConv2D
                else:
                    padding = layer.padding
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())

        # Final masking
        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        """Create mask matching tensor dimensions."""
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling with depthwise separable convolutions (dw-striding).

    Architecture matches:
    - Conv2d(1, 256, 3x3, stride=2, padding=1) + ReLU
    - Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256)
    - Conv2d(256, 256, 1x1) + ReLU
    - Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256)
    - Conv2d(256, 256, 1x1) + ReLU
    - Linear(4096, 1024)
    """

    def __init__(self, feat_in=80, feat_out=1024, conv_channels=256):
        super().__init__()

        self.feat_in = feat_in
        self.feat_out = feat_out
        self.conv_channels = conv_channels
        self.subsampling_factor = 8  # 2^3 for 3 striding layers
        self._sampling_num = int(math.log(self.subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2

        # Convolutional layers
        layers = []

        # Layer 0: Initial conv
        layers.append(nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=self._left_padding))
        layers.append(nn.ReLU(inplace=True))

        # Layer 1: First depthwise separable block
        layers.append(
            nn.Conv2d(
                conv_channels, conv_channels, kernel_size=3, stride=2, padding=self._left_padding, groups=conv_channels
            )
        )
        layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=1, stride=1))
        layers.append(nn.ReLU(inplace=True))

        # Layer 2: Second depthwise separable block
        layers.append(
            nn.Conv2d(
                conv_channels, conv_channels, kernel_size=3, stride=2, padding=self._left_padding, groups=conv_channels
            )
        )
        layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=1, stride=1))
        layers.append(nn.ReLU(inplace=True))

        self.conv = MaskedConvSequential(*layers)

        # Calculate output features after convolutions using calc_length like NeMo
        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.conv_out_features = conv_channels * int(out_length)
        self.out = nn.Linear(self.conv_out_features, feat_out)

        # Initialize weights following NeMo's depthwise conv initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following NeMo's depthwise conv initialization."""
        with torch.no_grad():
            # Initialize first conv
            scale = 1.0 / self._kernel_size
            nn.init.uniform_(self.conv[0].weight, -scale, scale)
            nn.init.uniform_(self.conv[0].bias, -scale, scale)

            # Initialize depthwise and pointwise convs
            dw_max = (self._kernel_size**2) ** -0.5
            pw_max = self.conv_channels**-0.5

            # First block (indices 2, 3)
            nn.init.uniform_(self.conv[2].weight, -dw_max, dw_max)
            nn.init.uniform_(self.conv[2].bias, -dw_max, dw_max)
            nn.init.uniform_(self.conv[3].weight, -pw_max, pw_max)
            nn.init.uniform_(self.conv[3].bias, -pw_max, pw_max)

            # Second block (indices 5, 6)
            nn.init.uniform_(self.conv[5].weight, -dw_max, dw_max)
            nn.init.uniform_(self.conv[5].bias, -dw_max, dw_max)
            nn.init.uniform_(self.conv[6].weight, -pw_max, pw_max)
            nn.init.uniform_(self.conv[6].bias, -pw_max, pw_max)

            # Initialize output linear
            fc_scale = (self.feat_out * self.feat_in / self._sampling_num) ** -0.5
            nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
            nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def forward(self, x, lengths):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, time, feat_in)
            lengths: Length tensor of shape (batch,)

        Returns:
            x: Output tensor of shape (batch, time//8, feat_out)
            lengths: Updated lengths tensor of shape (batch,)
        """
        # Apply convolutional subsampling
        x, lengths = self.conv(x, lengths)

        # Flatten channel and frequency axes
        batch_size, channels, time, freq = x.size()
        x = x.transpose(1, 2).reshape(batch_size, time, -1)

        # Apply final linear projection
        x = self.out(x)

        return x, lengths


# Example usage with NeMo's parameters
if __name__ == "__main__":
    # Create model with feat_in=128 to match NeMo's debug output
    standalone = ConvSubsampling(feat_in=128, feat_out=1024, conv_channels=256)

    # Test with exact input shape from NeMo debug traces
    x_test = torch.randn(1, 744, 128)  # Match NeMo's input shape
    lengths_test = torch.tensor([743])

    # Forward pass
    output, output_lengths = standalone(x_test, lengths_test)

    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input lengths: {lengths_test}")
    print(f"Output lengths: {output_lengths}")
    print(f"Expected output time: {744 // 8}")
    print(f"Linear input features: {standalone.conv_out_features}")
