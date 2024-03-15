import torch
import torch.nn as nn
from collections import OrderedDict


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


def conv_bn_mish(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        Mish(),
    )


def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        Mish(),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        Mish(),
    )


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super(CSPBlock, self).__init__()
        hidden_channels = out_channels // 2
        self.conv1 = conv_bn_mish(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = conv_dw(hidden_channels, hidden_channels, 1)
        self.conv3 = conv_bn_mish(hidden_channels, hidden_channels, 1, 1, 0)
        self.conv4 = conv_dw(hidden_channels, hidden_channels, 1)
        self.concat = nn.Sequential(
            conv_bn_mish(hidden_channels * 2, hidden_channels, 1, 1, 0),
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            Mish(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        concat = torch.cat((x3, x4), dim=1)
        return self.concat(concat)


class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.stem = conv_bn_mish(3, 32, 3, 1, 1)
        self.layer1 = self._make_layer(32, 64, 1, 1)
        self.layer2 = self._make_layer(64, 128, 2, 1)
        self.layer3 = self._make_layer(128, 256, 8, 2)
        self.layer4 = self._make_layer(256, 512, 8, 2)
        self.layer5 = self._make_layer(512, 1024, 4, 2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [conv_bn_mish(in_channels, out_channels, 3, stride, 1)]
        for _ in range(blocks):
            layers.append(CSPBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class YOLOv4(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv4, self).__init__()
        self.backbone = CSPDarknet53()
        self.head = nn.Sequential(
            conv_bn_mish(1024, 512, 1, 1, 0),
            conv_bn_mish(512, 1024, 3, 1, 1),
            conv_bn_mish(1024, 512, 1, 1, 0),
            nn.Conv2d(512, 255, 1, 1, 0),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def custom_summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            # Parameters
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            # Operation parameters
            if isinstance(module, nn.Conv2d):
                summary[m_key]["kernel_size"] = list(module.kernel_size)
                summary[m_key]["stride"] = list(module.stride)
                summary[m_key]["padding"] = list(module.padding)
                summary[m_key]["dilation"] = list(module.dilation)
                summary[m_key]["groups"] = module.groups

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # Initialize summary dict
    summary = OrderedDict()
    # Register hook
    hooks = []
    model.apply(register_hook)
    # Forward pass
    model(torch.zeros(input_size).type(dtype))
    # Remove hooks
    for hook in hooks:
        hook.remove()
    # Print summary
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #", "Kernel Size", "Stride", "Padding", "Dilation", "Groups"
    )
    print(line_new)
    print("================================================================")
    total_params = 0
    trainable_params = 0
    for layer in summary:
        line = "{:>20}  {:>25} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            str(summary[layer].get("kernel_size", "")),
            str(summary[layer].get("stride", "")),
            str(summary[layer].get("padding", "")),
            str(summary[layer].get("dilation", "")),
            str(summary[layer].get("groups", "")),
        )
        # line2 = (1, , 64, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False), # resblock conv1
        total_params += summary[layer]["nb_params"]
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        print(line)
    print("================================================================")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("----------------------------------------------------------------")


model = YOLOv4()
input_size = (3, 480, 640)
dummy_input = torch.zeros((1,) + input_size)
custom_summary(model, dummy_input.size(), device="cpu")


# import csv
# def custom_summary_to_csv(model, input_size, csv_file, batch_size=-1, device="cuda"):
#    def register_hook(module):
#        def hook(module, input, output):
#            class_name = str(module.__class__).split(".")[-1].split("'")[0]
#            module_idx = len(summary)
#            m_key = f"{class_name}-{module_idx + 1}"
#            summary[m_key] = {}
#            summary[m_key]["input_shape"] = list(input[0].size())
#            summary[m_key]["input_shape"][0] = batch_size
#            if isinstance(output, (list, tuple)):
#                summary[m_key]["output_shape"] = [
#                    [-1] + list(o.size())[1:] for o in output
#                ]
#            else:
#                summary[m_key]["output_shape"] = list(output.size())
#                summary[m_key]["output_shape"][0] = batch_size
#            # Parameters
#            params = 0
#            if hasattr(module, "weight") and hasattr(module.weight, "size"):
#                params += torch.prod(torch.LongTensor(list(module.weight.size())))
#                summary[m_key]["trainable"] = module.weight.requires_grad
#            if hasattr(module, "bias") and hasattr(module.bias, "size"):
#                params += torch.prod(torch.LongTensor(list(module.bias.size())))
#            summary[m_key]["nb_params"] = params
#            # Operation parameters
#            if isinstance(module, nn.Conv2d):
#                summary[m_key]["kernel_size"] = list(module.kernel_size)
#                summary[m_key]["stride"] = list(module.stride)
#                summary[m_key]["padding"] = list(module.padding)
#                summary[m_key]["dilation"] = list(module.dilation)
#                summary[m_key]["groups"] = module.groups
#        if (
#            not isinstance(module, nn.Sequential)
#            and not isinstance(module, nn.ModuleList)
#        ):
#            hooks.append(module.register_forward_hook(hook))
#    device = device.lower()
#    assert device in [
#        "cuda",
#        "cpu",
#    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
#    if device == "cuda" and torch.cuda.is_available():
#        dtype = torch.cuda.FloatTensor
#    else:
#        dtype = torch.FloatTensor
#    # Initialize summary dict
#    summary = OrderedDict()
#    # Register hook
#    hooks = []
#    model.apply(register_hook)
#    # Forward pass
#    model(torch.zeros(input_size).type(dtype))
#    # Remove hooks
#    for hook in hooks:
#        hook.remove()
#    # Write summary to CSV
#    with open(csv_file, mode='w', newline='') as file:
#        writer = csv.writer(file)
#        writer.writerow(["Layer (type)", "Output Shape", "Param #", "Kernel Size", "Stride", "Padding", "Dilation", "Groups"])
#        for layer in summary:
#            writer.writerow([
#                layer,
#                str(summary[layer]["output_shape"]),
#                "{0:,}".format(summary[layer]["nb_params"]),
#                str(summary[layer].get("kernel_size", "")),
#                str(summary[layer].get("stride", "")),
#                str(summary[layer].get("padding", "")),
#                str(summary[layer].get("dilation", "")),
#                str(summary[layer].get("groups", ""))
#            ])

import csv


def custom_summary_to_csv(model, input_size, csv_file, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            # Parameters
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            # Operation parameters
            if isinstance(module, nn.Conv2d):
                summary[m_key]["kernel_size"] = list(module.kernel_size)
                summary[m_key]["stride"] = list(module.stride)
                summary[m_key]["padding"] = list(module.padding)
                summary[m_key]["dilation"] = list(module.dilation)
                summary[m_key]["groups"] = module.groups

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # Initialize summary dict
    summary = OrderedDict()
    # Register hook
    hooks = []
    model.apply(register_hook)
    # Forward pass
    model(torch.zeros(input_size).type(dtype))
    # Remove hooks
    for hook in hooks:
        hook.remove()
    # Write summary to CSV
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Layer (type)",
                "Input Shape",
                "Output Shape",
                "Param #",
                "Kernel Size",
                "Stride",
                "Padding",
                "Dilation",
                "Groups",
                "TTNN conv2d unit test inputs",
            ]
        )
        for layer in summary:
            try:
                if summary[layer].get("stride", "")[0] == 2:
                    print("\n\n yes! We do have stride 2 and the corresponding remaining parameters are: ")
                    print("input shape: ", summary[layer]["input_shape"])
                    print("output shape: ", summary[layer]["output_shape"])
                    print("kernel size: ", summary[layer].get("kernel_size", ""))
                    print(" padding: ", summary[layer].get("padding", ""))
            except Exception as E:
                print(E)
            try:
                writer.writerow(
                    [
                        layer,
                        str(summary[layer]["input_shape"]),
                        str(summary[layer]["output_shape"]),
                        "{0:,}".format(summary[layer]["nb_params"]),
                        str(summary[layer].get("kernel_size", "")),
                        str(summary[layer].get("stride", "")),
                        str(summary[layer].get("padding", "")),
                        str(summary[layer].get("dilation", "")),
                        str(summary[layer].get("groups", "")),
                        str(
                            (
                                1,
                                summary[layer]["output_shape"][1],
                                summary[layer]["input_shape"][1],
                                summary[layer]["input_shape"][2],
                                summary[layer]["input_shape"][3],
                                summary[layer].get("kernel_size", "")[0],
                                summary[layer].get("kernel_size", "")[1],
                                summary[layer].get("stride", "")[0],
                                summary[layer].get("stride", "")[1],
                                summary[layer].get("padding", "")[0],
                                summary[layer].get("padding", "")[1],
                                True,
                                None,
                                False,
                            )
                        ),
                    ]
                )
            except Exception as E:
                print(E)
                writer.writerow(
                    [
                        layer,
                        str(summary[layer]["input_shape"]),
                        str(summary[layer]["output_shape"]),
                        "{0:,}".format(summary[layer]["nb_params"]),
                        str(summary[layer].get("kernel_size", "")),
                        str(summary[layer].get("stride", "")),
                        str(summary[layer].get("padding", "")),
                        str(summary[layer].get("dilation", "")),
                        str(summary[layer].get("groups", "")),
                    ]
                )


# Create an instance of YOLOv4 model
model = YOLOv4()
input_size = (3, 480, 640)
dummy_input = torch.zeros((1,) + input_size)
custom_summary_to_csv(model, dummy_input.size(), "model_summary_full_myYolov4.csv", device="cpu")
