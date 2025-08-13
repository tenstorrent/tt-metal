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
        print_tensor_shape(identity, "identity")
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        print_tensor_shape(out, "out")
        out = self.bn2(self.conv2(out))
        # return out
        if self.downsample is not None:
            print_tensor_shape(identity, "identity")
            identity = self.downsample(x)

        print_tensor_shape(out, "out")
        print_tensor_shape(identity, "identity")
        out += identity
        print_tensor_shape(out, "out")
        out = F.relu(out, inplace=True)

        return out
