# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from collections import OrderedDict
from models.experimental.yolov4.reference.yolov4 import Yolov4


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


def test_generate_csv():
    model = Yolov4()
    input_size = (3, 480, 640)
    dummy_input = torch.zeros((1,) + input_size)
    custom_summary(model, dummy_input.size(), device="cpu")

    # Create an instance of YOLOv4 model
    custom_summary_to_csv(model, dummy_input.size(), "model_summary_full_myYolov4.csv", device="cpu")
