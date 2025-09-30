import torch
import torch.nn as nn
from typing import Tuple, Callable, Type, Union


# Essential utility functions
def create_conv2d(
    in_channels, out_channels, kernel_size, stride=1, padding="", dilation=1, groups=1, bias=False, **kwargs
):
    if padding == "":
        padding = (kernel_size - 1) // 2 * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def get_act_layer(act_layer):
    if isinstance(act_layer, str):
        if act_layer.lower() == "relu":
            return nn.ReLU
        elif act_layer.lower() == "gelu":
            return nn.GELU
    return act_layer


def create_shortcut(
    downsample_type, in_chs, out_chs, kernel_size=1, stride=1, dilation=(1, 1), norm_layer=nn.BatchNorm2d
):
    if in_chs == out_chs and stride == 1:
        return nn.Identity()
    if downsample_type == "conv1x1":
        return nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size, stride, bias=False), norm_layer(out_chs))
    return None


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def create_act_layer(act_layer, inplace=False, **kwargs):
    if isinstance(act_layer, str):
        if act_layer.lower() == "sigmoid":
            return nn.Sigmoid()
        elif act_layer.lower() == "relu":
            return nn.ReLU(inplace=inplace)
    elif act_layer == nn.ReLU:
        return nn.ReLU(inplace=inplace)
    return act_layer(**kwargs) if act_layer else nn.Identity()


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# Type aliases
PadType = Union[str, int, Tuple[int, int]]
LayerType = Type[nn.Module]


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=False,
        apply_norm=True,
        apply_act=True,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        drop_layer=None,
        **kwargs,
    ):
        super().__init__()

        self.conv = create_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        if apply_norm:
            self.bn = norm_layer(out_channels)
            self.act = act_layer() if apply_act and act_layer else nn.Identity()
        else:
            self.bn = nn.Identity()
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# SE Module
class SEModule(nn.Module):
    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        add_maxpool=False,
        bias=True,
        act_layer=nn.ReLU,
        norm_layer=None,
        gate_layer="sigmoid",
    ):
        super().__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


# DropPath
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Bottleneck(nn.Module):
    """RegNet Bottleneck block.

    This is almost exactly the same as a ResNet Bottleneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        dilation: Tuple[int, int] = (1, 1),
        bottle_ratio: float = 1,
        group_size: int = 1,
        se_ratio: float = 0.25,
        downsample: str = "conv1x1",
        linear_out: bool = False,
        act_layer: Callable = nn.ReLU,
        norm_layer: Callable = nn.BatchNorm2d,
        drop_block=None,
        drop_path_rate: float = 0.0,
    ):
        super(Bottleneck, self).__init__()
        act_layer = get_act_layer(act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvNormAct(
            bottleneck_chs,
            bottleneck_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=groups,
            drop_layer=drop_block,
            **cargs,
        )
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.conv3 = ConvNormAct(bottleneck_chs, out_chs, kernel_size=1, apply_act=False, **cargs)
        self.act3 = nn.Identity() if linear_out else act_layer()
        self.downsample = create_shortcut(
            downsample,
            in_chs,
            out_chs,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self) -> None:
        """Zero-initialize the last batch norm in the block."""
        if hasattr(self.conv3, "bn") and hasattr(self.conv3.bn, "weight"):
            nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.downsample is not None:
            x = self.drop_path(x) + self.downsample(shortcut)
        x = self.act3(x)
        return x


# Example usage
# if __name__ == "__main__":
#    model = Bottleneck(in_chs=64, out_chs=256, stride=2, group_size=8)
#    x = torch.randn(1, 64, 56, 56)
#    output = model(x)
#    print(f"Input: {x.shape}, Output: {output.shape}")

# dimensions
if __name__ == "__main__":
    model = Bottleneck(in_chs=64, out_chs=256, stride=2, group_size=8)
    x = torch.randn(1, 64, 56, 56)

    print("=== RegNet Bottleneck Architecture Test ===")
    print(f"Input: {x.shape}")

    # Print model architecture
    print("\n=== Model Architecture ===")
    print(model)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with intermediate outputs
    print("\n=== Forward Pass Details ===")
    with torch.no_grad():
        shortcut = x
        print(f"Original input: {x.shape}")

        # Conv1 (1x1 reduction)
        x1 = model.conv1(x)
        print(f"After conv1 (1x1): {x1.shape}")

        # Conv2 (3x3 grouped conv)
        x2 = model.conv2(x1)
        print(f"After conv2 (3x3 grouped): {x2.shape}")

        # SE Module (RegNet's key difference - SE after conv2)
        x_se = model.se(x2)
        print(f"After SE module: {x_se.shape}")

        # Conv3 (1x1 expansion, no activation)
        x3 = model.conv3(x_se)
        print(f"After conv3 (1x1, no act): {x3.shape}")

        # Downsample shortcut if needed
        if model.downsample is not None:
            shortcut_out = model.downsample(shortcut)
            print(f"Shortcut downsampled: {shortcut_out.shape}")
        else:
            shortcut_out = shortcut
            print(f"Shortcut (identity): {shortcut_out.shape}")

        # Add residual connection
        x_residual = model.drop_path(x3) + shortcut_out
        print(f"After residual addition: {x_residual.shape}")

        # Final activation
        output = model.act3(x_residual)
        print(f"Final output: {output.shape}")

    # Verify RegNet-specific properties
    print("\n=== RegNet Architecture Verification ===")
    bottleneck_chs = int(round(256 * 1.0))  # bottle_ratio = 1.0
    groups = bottleneck_chs // 8  # group_size = 8
    print(f"Bottleneck channels: {bottleneck_chs}")
    print(f"Groups for conv2: {groups}")
    print(f"SE ratio: 0.25 (SE channels: {int(round(64 * 0.25))})")
    print(f"Stride: 2 (spatial downsampling)")

    # Test SE module specifically
    print(f"\n=== SE Module Test ===")
    if hasattr(model.se, "fc1"):
        print(f"SE fc1: {model.se.fc1.in_channels} -> {model.se.fc1.out_channels}")
        print(f"SE fc2: {model.se.fc2.in_channels} -> {model.se.fc2.out_channels}")
    else:
        print("SE module is Identity (se_ratio = 0)")
