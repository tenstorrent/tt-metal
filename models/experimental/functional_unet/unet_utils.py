# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn


def create_unet_models(device, groups, torch_input_tensor):
    assert device.arch() == ttnn.device.Arch.WORMHOLE_B0, "Expected device architecture to be WORMHOLE_B0"
    assert (device.core_grid.x, device.core_grid.y) == (
        8,
        8,
    ), "Expected 8x8 core grid initialize UNet. If using N300 - make sure to enable ethernet dispatch cores"

    torch_model = unet_shallow_torch.UNet(groups=groups)
    torch_model.eval()

    new_state_dict = {}
    for name, parameter in torch_model.state_dict().items():
        if isinstance(parameter, torch.FloatTensor):
            if "b1" or "b2" or "b3" or "b4" or "bnb" in name:
                new_state_dict[name] = parameter
            else:
                new_state_dict[name] = parameter + 1000

    torch_model.load_state_dict(new_state_dict)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device, groups=groups),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = unet_shallow_ttnn.UNet(parameters)

    return torch_model, ttnn_model


def create_unet_input_tensors(
    device, batch, groups, pad_input=True, input_channels=4, input_height=1056, input_width=160, collapse=False
):
    torch_input_tensor = torch.randn(batch, input_channels * groups, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    if collapse:
        ttnn_input_tensor = torch.reshape(ttnn_input_tensor, (1, 1, batch * input_height * input_width, input_channels))

    if pad_input:
        # Pad to 16 if grayskull run and 32 for wormhole
        ttnn_input_tensor = ttnn_input_tensor.reshape(
            ttnn_input_tensor.shape[0],
            1,
            ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
            ttnn_input_tensor.shape[3],
        )
        pad = 32 if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else 16
        hpad = 0  # 96*32*64
        if ttnn_input_tensor.shape[-1] < pad or ttnn_input_tensor.shape[-2] < hpad:
            ttnn_input_tensor = torch.nn.functional.pad(
                ttnn_input_tensor,
                (0, max(0, pad - ttnn_input_tensor.shape[-1]), 0, max(0, hpad - ttnn_input_tensor.shape[-2])),
            )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)

    return torch_input_tensor, ttnn_input_tensor


# Model processing functions
def update_ttnn_module_args(ttnn_module_args, groups):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256 or groups > 1


# Create custom preprocessor to be ingested
def create_custom_preprocessor(device, groups=1):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, unet_shallow_torch.UNet):
            print("\n\n\n")
            print("model output weights: ", type(model.output_layer.weight))
            print("model output weights: ", list(model.output_layer.weight))

            ttnn_module_args.p1["deallocate_activation"] = False
            ttnn_module_args.p2["deallocate_activation"] = False
            ttnn_module_args.p3["deallocate_activation"] = False
            ttnn_module_args.p4["deallocate_activation"] = False

            ttnn_module_args.c1["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c1["padded_input_channels"] = None if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else 16
            ttnn_module_args.c1["use_shallow_conv_variant"] = (
                False if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c1_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c1_2["use_shallow_conv_variant"] = (
                False if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c1["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c1_2["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c1["deallocate_activation"] = True
            ttnn_module_args.c1_2["deallocate_activation"] = True
            ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = (
                {"act_block_h": 32}
                if groups > 1
                else ({"act_block_h": 5 * 32} if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else {"act_block_h": 64})
            )
            ttnn_module_args.c1_2["conv_blocking_and_parallelization_config_override"] = (
                {"act_block_h": 32}
                if groups > 1
                else ({"act_block_h": 5 * 32} if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else {"act_block_h": 64})
            )

            ttnn_module_args.c2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c2_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2["activation"] = "relu"  # Fuse relu with conv2
            ttnn_module_args.c2_2["activation"] = "relu"  # Fuse relu with conv2
            ttnn_module_args.c2["deallocate_activation"] = True
            ttnn_module_args.c2_2["deallocate_activation"] = True
            ttnn_module_args.c2["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args.c2_2["conv_blocking_and_parallelization_config_override"] = None

            ttnn_module_args.c3["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c3["use_shallow_conv_variant"] = False
            ttnn_module_args.c3_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c3_2["use_shallow_conv_variant"] = False
            ttnn_module_args.c3["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3["activation"] = "relu"  # Fuse relu with conv3
            ttnn_module_args.c3_2["activation"] = "relu"  # Fuse relu with conv3
            ttnn_module_args.c3["deallocate_activation"] = True
            ttnn_module_args.c3_2["deallocate_activation"] = True
            ttnn_module_args.c3["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args.c3_2["conv_blocking_and_parallelization_config_override"] = None

            ttnn_module_args.c4["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c4["use_shallow_conv_variant"] = False
            ttnn_module_args.c4_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c4_2["use_shallow_conv_variant"] = False
            ttnn_module_args.c4["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4["activation"] = "relu"  # Fuse relu with conv4
            ttnn_module_args.c4_2["activation"] = "relu"  # Fuse relu with conv4
            ttnn_module_args.c4["deallocate_activation"] = True
            ttnn_module_args.c4_2["deallocate_activation"] = True
            ttnn_module_args.c4["conv_blocking_and_parallelization_config_override"] = (
                {"num_cores_nhw": 42} if groups > 1 else None
            )
            ttnn_module_args.c4_2["conv_blocking_and_parallelization_config_override"] = (
                {"num_cores_nhw": 42} if groups > 1 else None
            )

            ttnn_module_args.bnc["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.bnc_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.bnc["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.bnc_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.bnc["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.bnc_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.bnc["activation"] = "relu"  # Fuse relu with bottle neck conv
            ttnn_module_args.bnc_2["activation"] = "relu"  # Fuse relu with bottle neck conv
            ttnn_module_args.bnc["deallocate_activation"] = True
            ttnn_module_args.bnc_2["deallocate_activation"] = True
            ttnn_module_args.bnc["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args.bnc_2["conv_blocking_and_parallelization_config_override"] = None

            ttnn_module_args.c5["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c5_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c5_3["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c5["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5_3["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5_3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5["activation"] = "relu"  # Fuse relu with conv5
            ttnn_module_args.c5_2["activation"] = "relu"  # Fuse relu with conv5
            ttnn_module_args.c5_3["activation"] = "relu"  # Fuse relu with conv5
            ttnn_module_args.c5["deallocate_activation"] = True
            ttnn_module_args.c5_2["deallocate_activation"] = True
            ttnn_module_args.c5_3["deallocate_activation"] = True
            ttnn_module_args.c5["conv_blocking_and_parallelization_config_override"] = (
                {"num_cores_nhw": 42} if groups > 1 else None
            )
            ttnn_module_args.c5_2["conv_blocking_and_parallelization_config_override"] = (
                {"num_cores_nhw": 42} if groups > 1 else None
            )
            ttnn_module_args.c5_3["conv_blocking_and_parallelization_config_override"] = (
                {"num_cores_nhw": 42} if groups > 1 else None
            )

            ttnn_module_args.c6["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c6["use_shallow_conv_variant"] = (
                False if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c6_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c6_2["use_shallow_conv_variant"] = (
                False if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c6_3["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c6["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6_3["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6_3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6["activation"] = "relu"  # Fuse relu with conv6
            ttnn_module_args.c6_2["activation"] = "relu"  # Fuse relu with conv6
            ttnn_module_args.c6_3["activation"] = "relu"  # Fuse relu with conv6
            ttnn_module_args.c6["deallocate_activation"] = True
            ttnn_module_args.c6_2["deallocate_activation"] = True
            ttnn_module_args.c6_3["deallocate_activation"] = True
            ttnn_module_args.c6["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args.c6_2["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args.c6_3["conv_blocking_and_parallelization_config_override"] = None

            ttnn_module_args.c7["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c7_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c7_3["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c7["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7_3["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7_3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7["activation"] = "relu"  # Fuse relu with conv7
            ttnn_module_args.c7_2["activation"] = "relu"  # Fuse relu with conv7
            ttnn_module_args.c7_3["activation"] = "relu"  # Fuse relu with conv7
            ttnn_module_args.c7["deallocate_activation"] = True
            ttnn_module_args.c7_2["deallocate_activation"] = True
            ttnn_module_args.c7_3["deallocate_activation"] = True
            ttnn_module_args.c7["padded_input_channels"] = 48 * groups
            ttnn_module_args.c7["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args.c7_2["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args.c7_3["conv_blocking_and_parallelization_config_override"] = None

            ttnn_module_args.c8["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c8_2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c8_3["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c8["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8_2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8_3["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8_2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8_3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8["activation"] = "relu"  # Fuse relu with conv8
            ttnn_module_args.c8_2["activation"] = "relu"  # Fuse relu with conv8
            ttnn_module_args.c8_3["activation"] = "relu"  # Fuse relu with conv8
            ttnn_module_args.c8["deallocate_activation"] = True
            ttnn_module_args.c8_2["deallocate_activation"] = True
            ttnn_module_args.c8_3["deallocate_activation"] = True
            ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args.c8_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args.c8_3["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}

            ttnn_module_args.output_layer["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.output_layer["dtype"] = ttnn.bfloat16
            ttnn_module_args.output_layer["weights_dtype"] = ttnn.bfloat16
            ttnn_module_args.output_layer["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args.output_layer["activation"] = None
            ttnn_module_args.output_layer["deallocate_activation"] = True

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
            print("model output weights for c1: ", type(conv1_weight))
            conv1_2_weight, conv1_2_bias = fold_batch_norm2d_into_conv2d(model.c1_2, model.b1_2)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
            conv2_2_weight, conv2_2_bias = fold_batch_norm2d_into_conv2d(model.c2_2, model.b2_2)
            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
            conv3_2_weight, conv3_2_bias = fold_batch_norm2d_into_conv2d(model.c3_2, model.b3_2)
            conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
            conv4_2_weight, conv4_2_bias = fold_batch_norm2d_into_conv2d(model.c4_2, model.b4_2)
            convbn_weight, convbn_bias = fold_batch_norm2d_into_conv2d(model.bnc, model.bnb)
            convbn_2_weight, convbn_2_bias = fold_batch_norm2d_into_conv2d(model.bnc_2, model.bnb_2)
            conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
            conv5_2_weight, conv5_2_bias = fold_batch_norm2d_into_conv2d(model.c5_2, model.b5_2)
            conv5_3_weight, conv5_3_bias = fold_batch_norm2d_into_conv2d(model.c5_3, model.b5_3)
            conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
            conv6_2_weight, conv6_2_bias = fold_batch_norm2d_into_conv2d(model.c6_2, model.b6_2)
            conv6_3_weight, conv6_3_bias = fold_batch_norm2d_into_conv2d(model.c6_3, model.b6_3)
            conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
            conv7_2_weight, conv7_2_bias = fold_batch_norm2d_into_conv2d(model.c7_2, model.b7_2)
            conv7_3_weight, conv7_3_bias = fold_batch_norm2d_into_conv2d(model.c7_3, model.b7_3)
            conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
            conv8_2_weight, conv8_2_bias = fold_batch_norm2d_into_conv2d(model.c8_2, model.b8_2)
            conv8_3_weight, conv8_3_bias = fold_batch_norm2d_into_conv2d(model.c8_3, model.b8_3)

            update_ttnn_module_args(ttnn_module_args.c1, groups)
            update_ttnn_module_args(ttnn_module_args.c1_2, groups)
            update_ttnn_module_args(ttnn_module_args.c2, groups)
            update_ttnn_module_args(ttnn_module_args.c2_2, groups)
            update_ttnn_module_args(ttnn_module_args.c3, groups)
            update_ttnn_module_args(ttnn_module_args.c3_2, groups)
            update_ttnn_module_args(ttnn_module_args.c4, groups)
            update_ttnn_module_args(ttnn_module_args.c4_2, groups)
            update_ttnn_module_args(ttnn_module_args.bnc, groups)
            update_ttnn_module_args(ttnn_module_args.bnc_2, groups)
            update_ttnn_module_args(ttnn_module_args.c5, groups)
            update_ttnn_module_args(ttnn_module_args.c5_2, groups)
            update_ttnn_module_args(ttnn_module_args.c5_3, groups)
            update_ttnn_module_args(ttnn_module_args.c6, groups)
            update_ttnn_module_args(ttnn_module_args.c6_2, groups)
            update_ttnn_module_args(ttnn_module_args.c6_3, groups)
            update_ttnn_module_args(ttnn_module_args.c7, groups)
            update_ttnn_module_args(ttnn_module_args.c7_2, groups)
            update_ttnn_module_args(ttnn_module_args.c7_3, groups)
            update_ttnn_module_args(ttnn_module_args.c8, groups)
            update_ttnn_module_args(ttnn_module_args.c8_2, groups)
            update_ttnn_module_args(ttnn_module_args.c8_3, groups)
            update_ttnn_module_args(ttnn_module_args.output_layer, groups)

            parameters["c1"], c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
            )
            parameters["c1_2"] = preprocess_conv2d(conv1_2_weight, conv1_2_bias, ttnn_module_args.c1_2)
            parameters["p1"] = {}
            ttnn_module_args.p1["parallel_config_override"] = {
                "grid_size": (c1_parallel_config.grid_size.x, c1_parallel_config.grid_size.y),
                "num_cores_nhw": c1_parallel_config.num_cores_nhw,
            }
            parameters["c2"], c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
            )
            parameters["c2_2"] = preprocess_conv2d(conv2_2_weight, conv2_2_bias, ttnn_module_args.c2_2)
            parameters["p2"] = {}
            ttnn_module_args.p2["parallel_config_override"] = {
                "grid_size": (c2_parallel_config.grid_size.x, c2_parallel_config.grid_size.y),
                "num_cores_nhw": c2_parallel_config.num_cores_nhw,
            }
            parameters["c3"], c3_parallel_config = preprocess_conv2d(
                conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
            )
            parameters["c3_2"] = preprocess_conv2d(conv3_2_weight, conv3_2_bias, ttnn_module_args.c3_2)
            parameters["p3"] = {}
            ttnn_module_args.p3["parallel_config_override"] = {
                "grid_size": (c3_parallel_config.grid_size.x, c3_parallel_config.grid_size.y),
                "num_cores_nhw": c3_parallel_config.num_cores_nhw,
            }
            parameters["c4"], c4_parallel_config = preprocess_conv2d(
                conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
            )
            parameters["c4_2"] = preprocess_conv2d(conv4_2_weight, conv4_2_bias, ttnn_module_args.c4_2)
            parameters["p4"] = {}
            ttnn_module_args.p4["parallel_config_override"] = {
                "grid_size": (c4_parallel_config.grid_size.x, c4_parallel_config.grid_size.y),
                "num_cores_nhw": c4_parallel_config.num_cores_nhw,
            }
            parameters["bnc"] = preprocess_conv2d(convbn_weight, convbn_bias, ttnn_module_args.bnc)
            parameters["bnc_2"] = preprocess_conv2d(convbn_2_weight, convbn_2_bias, ttnn_module_args.bnc_2)
            parameters["c5"] = preprocess_conv2d(conv5_weight, conv5_bias, ttnn_module_args.c5)
            parameters["c5_2"] = preprocess_conv2d(conv5_2_weight, conv5_2_bias, ttnn_module_args.c5_2)
            parameters["c5_3"] = preprocess_conv2d(conv5_3_weight, conv5_3_bias, ttnn_module_args.c5_3)
            parameters["c6"] = preprocess_conv2d(conv6_weight, conv6_bias, ttnn_module_args.c6)
            parameters["c6_2"] = preprocess_conv2d(conv6_2_weight, conv6_2_bias, ttnn_module_args.c6_2)
            parameters["c6_3"] = preprocess_conv2d(conv6_3_weight, conv6_3_bias, ttnn_module_args.c6_3)
            parameters["c7"] = preprocess_conv2d(conv7_weight, conv7_bias, ttnn_module_args.c7)
            parameters["c7_2"] = preprocess_conv2d(conv7_2_weight, conv7_2_bias, ttnn_module_args.c7_2)
            parameters["c7_3"] = preprocess_conv2d(conv7_3_weight, conv7_3_bias, ttnn_module_args.c7_3)
            parameters["c8"] = preprocess_conv2d(conv8_weight, conv8_bias, ttnn_module_args.c8)
            parameters["c8_2"] = preprocess_conv2d(conv8_2_weight, conv8_2_bias, ttnn_module_args.c8_2)
            parameters["c8_3"] = preprocess_conv2d(conv8_3_weight, conv8_3_bias, ttnn_module_args.c8_3)
            parameters["output_layer"] = preprocess_conv2d(
                model.output_layer.weight, model.output_layer.bias, ttnn_module_args.output_layer
            )
            breakpoint()

        return parameters

    return custom_preprocessor
