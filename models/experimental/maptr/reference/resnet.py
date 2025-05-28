import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style="pytorch"):
        super(BasicBlock, self).__init__()
        assert style in ["pytorch", "caffe"]
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)

        return out


class ResLayer(nn.Sequential):
    def __init__(self, inplanes, planes, stride=1, dilation=1, style="pytorch"):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        blocks = [
            BasicBlock(inplanes, planes, stride, dilation, downsample, style),
            BasicBlock(planes, planes, 1, dilation, style=style),
        ]

        super().__init__(*blocks)


class ResNet(nn.Module):
    def __init__(
        self,
        depth,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_eval=True,
        bn_frozen=False,
        style="pytorch",
    ):
        super(ResNet, self).__init__()

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = norm_eval
        self.bn_frozen = bn_frozen
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        stage_blocks = (2, 2, 2, 2)  # resnet18 model

        self.style = style
        self.res_layers = []

        for i, num_blocks in enumerate(stage_blocks):  # 4
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = ResLayer(self.inplanes, planes, stride=stride, dilation=dilation, style=self.style)
            self.inplanes = planes * 1
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        outs = []

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
