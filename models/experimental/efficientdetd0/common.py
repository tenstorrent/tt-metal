# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch


def load_partial_state(torch_model: torch.nn.Module, state_dict, layer_name: str = ""):
    partial_state_dict = {}
    layer_prefix = layer_name + "."
    for k, v in state_dict.items():
        if k.startswith(layer_prefix):
            partial_state_dict[k[len(layer_prefix) :]] = v
    torch_model.load_state_dict(partial_state_dict, strict=True)
    return torch_model


def load_torch_model_state(torch_model: torch.nn.Module = None, layer_name: str = "", model_location_generator=None):
    return torch_model.eval()


#     if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
#         model_path = "models"
#     else:
#         model_path = model_location_generator("vision-models/detr3d", model_subdir="", download_if_ci_v2=True)
#     if model_path == "models":
#         if not os.path.exists(
#             "models/experimental/detr3d/sunrgbd_masked_ep720.pth"
#         ):  # check if sunrgbd_masked_ep720.pth is available
#             os.system(
#                 "models/experimental/detr3d/resources/detr3d_weights_download.sh"
#             )  # execute the detr3d_weights_download.sh file
#         weights_path = "models/experimental/detr3d/sunrgbd_masked_ep720.pth"
#     else:
#         weights_path = os.path.join(model_path, "sunrgbd_masked_ep720.pth")

#     # Load checkpoint
#     state_dict = torch.load(weights_path, map_location="cpu")["model"]

#     if isinstance(
#         torch_model,
#         (
#             # SharedMLP,
#             # GenericMLP,
#             # PointnetSAModuleVotes,
#             # MaskedTransformerEncoder,
#             # TransformerEncoderLayer,
#             # TransformerDecoderLayer,
#             # TransformerDecoder,
#             # MultiheadAttention,
#         ),
#     ):
#         torch_model = load_partial_state(torch_model, state_dict, layer_name)
#     elif isinstance(torch_model, Model3DETR):
#         torch_model.load_state_dict(state_dict, strict=True)
#     else:
#         raise NotImplementedError("Unknown torch model. Weight loading not implemented")
#     logger.info(f"Successfully loaded weights: 3Detr {layer_name}")

#     return torch_model.eval()


import ttnn
from ttnn.torch_tracer import trace, visualize
from ttnn.dot_access import make_dot_access_dict
from ttnn.model_preprocessing import ModuleArgs, Conv2dArgs, ConvTranspose2dArgs, MaxPool2dArgs, GroupNormArgs


def infer_ttnn_module_args(*, model, run_model, device):
    if run_model is None:
        return None

    with trace():
        output = run_model(model)

    visualize(output, file_name=ttnn.CONFIG.tmp_dir / "model_graph.svg")

    def _infer_ttnn_module_args(graph):
        ttnn_module_args = {}
        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes["operation"]
            if isinstance(operation, ttnn.tracer.TorchModule):
                *_, module_name = operation.module.__ttnn_tracer_name__.split(".")
                (input_node, _, edge_data), *_ = graph.in_edges(node, data=True)
                input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]
                if isinstance(operation.module, torch.nn.Conv2d):
                    ttnn_module_args[module_name] = Conv2dArgs(
                        in_channels=operation.module.in_channels,
                        out_channels=operation.module.out_channels,
                        kernel_size=operation.module.kernel_size,
                        stride=operation.module.stride,
                        padding=operation.module.padding,
                        dilation=operation.module.dilation,
                        groups=operation.module.groups,
                        padding_mode=operation.module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    )
                elif isinstance(operation.module, torch.nn.ConvTranspose2d):
                    ttnn_module_args[module_name] = ConvTranspose2dArgs(
                        in_channels=operation.module.in_channels,
                        out_channels=operation.module.out_channels,
                        kernel_size=operation.module.kernel_size,
                        stride=operation.module.stride,
                        padding=operation.module.padding,
                        output_padding=operation.module.output_padding,
                        dilation=operation.module.dilation,
                        groups=operation.module.groups,
                        padding_mode=operation.module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    )
                elif isinstance(operation.module, torch.nn.MaxPool2d):
                    ttnn_module_args[module_name] = MaxPool2dArgs(
                        kernel_size=operation.module.kernel_size,
                        stride=operation.module.stride,
                        padding=operation.module.padding,
                        dilation=operation.module.dilation,
                        batch_size=input_shape[0],
                        input_channels=input_shape[1],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    )
                elif isinstance(operation.module, torch.nn.GroupNorm):
                    ttnn_module_args[module_name] = GroupNormArgs(
                        num_groups=operation.module.num_groups,
                        num_channels=operation.module.num_channels,
                        eps=operation.module.eps,
                        affine=operation.module.affine,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    )
                elif isinstance(operation.module, torch.nn.BatchNorm2d):
                    continue
                else:
                    ttnn_module_args[module_name] = _infer_ttnn_module_args(operation.graph)

                if module_name.isdigit():
                    ttnn_module_args[int(module_name)] = ttnn_module_args[module_name]

        return make_dot_access_dict(ttnn_module_args, ignore_types=(ModuleArgs,))

    ttnn_module_args = _infer_ttnn_module_args(ttnn.tracer.get_graph(output))
    return ttnn_module_args[""]
