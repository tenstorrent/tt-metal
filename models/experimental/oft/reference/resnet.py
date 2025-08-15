import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def print_tensor_shape(tensor, name):
    if tensor is not None:
        print(f"{name} shape: {tensor.shape=}")
    else:
        print(f"{name} is None")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes, stride), nn.GroupNorm(16, planes))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        # print_tensor_shape(identity, "identity")
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # print_tensor_shape(out, "out")
        out = self.bn2(self.conv2(out))
        # return out
        if self.downsample is not None:
            # print_tensor_shape(identity, "identity")
            identity = self.downsample(x)

        # print_tensor_shape(out, "out")
        # print_tensor_shape(identity, "identity")
        out += identity
        # print_tensor_shape(out, "out")
        out = F.relu(out, inplace=True)

        return out


class ResNetFeatures(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetFeatures, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(16, 64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        conv1 = F.max_pool2d(conv1, 3, stride=2, padding=1)

        feats4 = self.layer1(conv1)
        feats8 = self.layer2(feats4)
        feats16 = self.layer3(feats8)
        feats32 = self.layer4(feats16)

        return feats8, feats16, feats32


@pytest.mark.parametrize(
    "inplanes, planes, stride, input_shape",
    [
        (64, 64, 1, (1, 64, 56, 56)),  # identity path
        (64, 128, 2, (1, 64, 56, 56)),  # downsample path
        (128, 256, 2, (1, 128, 64, 80)),
    ],
)
def test_basicblock_forward(inplanes, planes, stride, input_shape):
    torch.manual_seed(0)
    block = BasicBlock(inplanes, planes, stride)
    x = torch.randn(*input_shape)
    out = block.forward(x)
    # Output shape should match expected ResNet block output
    expected_h = input_shape[2] // stride
    expected_w = input_shape[3] // stride
    assert out.shape == (input_shape[0], planes, expected_h, expected_w)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "input_shape, layers",
    [
        ((1, 3, 224, 224), [2, 2, 2, 2]),  # ResNet-18
        ((2, 3, 128, 128), [2, 2, 2, 2]),  # batch size 2
    ],
)
def test_resnetfeatures_forward(input_shape, layers):
    torch.manual_seed(0)
    print(f"Testing ResNetFeatures with input shape: {input_shape} and layers: {layers}")
    model = ResNetFeatures(BasicBlock, layers)
    x = torch.randn(*input_shape)
    feats8, feats16, feats32 = model(x)
    # Output shapes after each stage (approximate for stride=2 blocks)
    b, c, h, w = input_shape
    # After conv1 + maxpool: h//4, w//4
    h4, w4 = h // 4, w // 4
    # After layer2: h//8, w//8
    h8, w8 = h // 8, w // 8
    # After layer3: h//16, w//16
    h16, w16 = h // 16, w // 16
    # After layer4: h//32, w//32
    h32, w32 = h // 32, w // 32
    assert feats8.shape[2] == h8 and feats8.shape[3] == w8
    assert feats16.shape[2] == h16 and feats16.shape[3] == w16
    assert feats32.shape[2] == h32 and feats32.shape[3] == w32
    assert feats8.shape[0] == b and feats16.shape[0] == b and feats32.shape[0] == b
    print("ResNetFeatures forward test passed.")
