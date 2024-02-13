# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0

from models.experimental.functional_unet.tt import ttnn_functional_CONVENTIONAL_unet_maxpoolFallback

import ttnn

from collections.abc import MutableMapping


def flatten_dict(dictionary, parent_key="", separator="."):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def creat_state_dict_from_torchhub():
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model_dict_orig = list(model.state_dict().items())

    keys = [
        "c1",
        "b1",
        "c1_2",
        "b1_2",
        "c2",
        "b2",
        "c2_2",
        "b2_2",
        "c3",
        "b3",
        "c3_2",
        "b3_2",
        "c4",
        "b4",
        "c4_2",
        "b4_2",
        "bnc",
        "bnb",
        "bnc_2",
        "bnb_2",
        "c5",
        "b5",
        "c5_2",
        "b5_2",
        "c6",
        "b6",
        "c6_2",
        "b6_2",
        "c7",
        "b7",
        "c7_2",
        "b7_2",
        "c8",
        "b8",
        "c8_2",
        "b8_2",
        "output_layer",
    ]
    """
    encoder1.enc1conv1.weight
    encoder1.enc1norm1.weight
    encoder1.enc1norm1.bias
    encoder1.enc1norm1.running_mean
    encoder1.enc1norm1.running_var
    encoder1.enc1norm1.num_batches_tracked
    encoder1.enc1conv2.weight
    encoder1.enc1norm2.weight
    encoder1.enc1norm2.bias
    encoder1.enc1norm2.running_mean
    encoder1.enc1norm2.running_var
    encoder1.enc1norm2.num_batches_tracked
    encoder2.enc2conv1.weight
    encoder2.enc2norm1.weight
    encoder2.enc2norm1.bias
    encoder2.enc2norm1.running_mean
    encoder2.enc2norm1.running_var
    encoder2.enc2norm1.num_batches_tracked
    encoder2.enc2conv2.weight
    encoder2.enc2norm2.weight
    encoder2.enc2norm2.bias
    encoder2.enc2norm2.running_mean
    encoder2.enc2norm2.running_var
    encoder2.enc2norm2.num_batches_tracked
    encoder3.enc3conv1.weight
    encoder3.enc3norm1.weight
    encoder3.enc3norm1.bias
    encoder3.enc3norm1.running_mean
    encoder3.enc3norm1.running_var
    encoder3.enc3norm1.num_batches_tracked
    encoder3.enc3conv2.weight
    encoder3.enc3norm2.weight
    encoder3.enc3norm2.bias
    encoder3.enc3norm2.running_mean
    encoder3.enc3norm2.running_var
    encoder3.enc3norm2.num_batches_tracked
    encoder4.enc4conv1.weight
    encoder4.enc4norm1.weight
    encoder4.enc4norm1.bias
    encoder4.enc4norm1.running_mean
    encoder4.enc4norm1.running_var
    encoder4.enc4norm1.num_batches_tracked
    encoder4.enc4conv2.weight
    encoder4.enc4norm2.weight
    encoder4.enc4norm2.bias
    encoder4.enc4norm2.running_mean
    encoder4.enc4norm2.running_var
    encoder4.enc4norm2.num_batches_tracked
    bottleneck.bottleneckconv1.weight
    bottleneck.bottlenecknorm1.weight
    bottleneck.bottlenecknorm1.bias
    bottleneck.bottlenecknorm1.running_mean
    bottleneck.bottlenecknorm1.running_var
    bottleneck.bottlenecknorm1.num_batches_tracked
    bottleneck.bottleneckconv2.weight
    bottleneck.bottlenecknorm2.weight
    bottleneck.bottlenecknorm2.bias
    bottleneck.bottlenecknorm2.running_mean
    bottleneck.bottlenecknorm2.running_var
    bottleneck.bottlenecknorm2.num_batches_tracked
    upconv4.weight
    upconv4.bias
    decoder4.dec4conv1.weight
    decoder4.dec4norm1.weight
    decoder4.dec4norm1.bias
    decoder4.dec4norm1.running_mean
    decoder4.dec4norm1.running_var
    decoder4.dec4norm1.num_batches_tracked
    decoder4.dec4conv2.weight
    decoder4.dec4norm2.weight
    decoder4.dec4norm2.bias
    decoder4.dec4norm2.running_mean
    decoder4.dec4norm2.running_var
    decoder4.dec4norm2.num_batches_tracked
    upconv3.weight
    upconv3.bias
    decoder3.dec3conv1.weight
    decoder3.dec3norm1.weight
    decoder3.dec3norm1.bias
    decoder3.dec3norm1.running_mean
    decoder3.dec3norm1.running_var
    decoder3.dec3norm1.num_batches_tracked
    decoder3.dec3conv2.weight
    decoder3.dec3norm2.weight
    decoder3.dec3norm2.bias
    decoder3.dec3norm2.running_mean
    decoder3.dec3norm2.running_var
    decoder3.dec3norm2.num_batches_tracked
    upconv2.weight
    upconv2.bias
    decoder2.dec2conv1.weight
    decoder2.dec2norm1.weight
    decoder2.dec2norm1.bias
    decoder2.dec2norm1.running_mean
    decoder2.dec2norm1.running_var
    decoder2.dec2norm1.num_batches_tracked
    decoder2.dec2conv2.weight
    decoder2.dec2norm2.weight
    decoder2.dec2norm2.bias
    decoder2.dec2norm2.running_mean
    decoder2.dec2norm2.running_var
    decoder2.dec2norm2.num_batches_tracked
    upconv1.weight
    upconv1.bias
    decoder1.dec1conv1.weight
    decoder1.dec1norm1.weight
    decoder1.dec1norm1.bias
    decoder1.dec1norm1.running_mean
    decoder1.dec1norm1.running_var
    decoder1.dec1norm1.num_batches_tracked
    decoder1.dec1conv2.weight
    decoder1.dec1norm2.weight
    decoder1.dec1norm2.bias
    decoder1.dec1norm2.running_mean
    decoder1.dec1norm2.running_var
    decoder1.dec1norm2.num_batches_tracked
    conv.weight
    conv.bias
    """

    model_dict = {}

    c1_dict = {}
    c1_dict["weight"] = model_dict_orig[0][1]
    c1_dict["bias"] = torch.zeros(c1_dict["weight"].shape[0])
    model_dict["c1"] = c1_dict
    b1_dict = {}
    b1_dict["weight"] = model_dict_orig[1][1]
    b1_dict["bias"] = model_dict_orig[2][1]
    b1_dict["running_mean"] = model_dict_orig[3][1]
    b1_dict["running_var"] = model_dict_orig[4][1]
    b1_dict["num_batches_tracked"] = model_dict_orig[5][1]
    model_dict["b1"] = b1_dict

    c1_2_dict = {}
    c1_2_dict["weight"] = model_dict_orig[6][1]
    c1_2_dict["bias"] = torch.zeros(c1_2_dict["weight"].shape[0])
    model_dict["c1_2"] = c1_2_dict
    b1_2_dict = {}
    b1_2_dict["weight"] = model_dict_orig[7][1]
    b1_2_dict["bias"] = model_dict_orig[8][1]
    b1_2_dict["running_mean"] = model_dict_orig[9][1]
    b1_2_dict["running_var"] = model_dict_orig[10][1]
    b1_2_dict["num_batches_tracked"] = model_dict_orig[11][1]
    model_dict["b1_2"] = b1_2_dict

    c2_dict = {}
    c2_dict["weight"] = model_dict_orig[12][1]
    c2_dict["bias"] = torch.zeros(c2_dict["weight"].shape[0])
    model_dict["c2"] = c2_dict
    b2_dict = {}
    b2_dict["weight"] = model_dict_orig[13][1]
    b2_dict["bias"] = model_dict_orig[14][1]
    b2_dict["running_mean"] = model_dict_orig[15][1]
    b2_dict["running_var"] = model_dict_orig[16][1]
    b2_dict["num_batches_tracked"] = model_dict_orig[17][1]
    model_dict["b2"] = b2_dict

    c2_2_dict = {}
    c2_2_dict["weight"] = model_dict_orig[18][1]
    c2_2_dict["bias"] = torch.zeros(c2_2_dict["weight"].shape[0])
    model_dict["c2_2"] = c2_2_dict
    b2_2_dict = {}
    b2_2_dict["weight"] = model_dict_orig[19][1]
    b2_2_dict["bias"] = model_dict_orig[20][1]
    b2_2_dict["running_mean"] = model_dict_orig[21][1]
    b2_2_dict["running_var"] = model_dict_orig[22][1]
    b2_2_dict["num_batches_tracked"] = model_dict_orig[23][1]
    model_dict["b2_2"] = b2_2_dict

    c3_dict = {}
    c3_dict["weight"] = model_dict_orig[24][1]
    c3_dict["bias"] = torch.zeros(c3_dict["weight"].shape[0])
    model_dict["c3"] = c3_dict
    b3_dict = {}
    b3_dict["weight"] = model_dict_orig[25][1]
    b3_dict["bias"] = model_dict_orig[26][1]
    b3_dict["running_mean"] = model_dict_orig[27][1]
    b3_dict["running_var"] = model_dict_orig[28][1]
    b3_dict["num_batches_tracked"] = model_dict_orig[29][1]
    model_dict["b3"] = b3_dict

    c3_2_dict = {}
    c3_2_dict["weight"] = model_dict_orig[30][1]
    c3_2_dict["bias"] = torch.zeros(c3_2_dict["weight"].shape[0])
    model_dict["c3_2"] = c3_2_dict
    b3_2_dict = {}
    b3_2_dict["weight"] = model_dict_orig[31][1]
    b3_2_dict["bias"] = model_dict_orig[32][1]
    b3_2_dict["running_mean"] = model_dict_orig[33][1]
    b3_2_dict["running_var"] = model_dict_orig[34][1]
    b3_2_dict["num_batches_tracked"] = model_dict_orig[35][1]
    model_dict["b3_2"] = b3_2_dict

    c4_dict = {}
    c4_dict["weight"] = model_dict_orig[36][1]
    c4_dict["bias"] = torch.zeros(c4_dict["weight"].shape[0])
    model_dict["c4"] = c4_dict
    b4_dict = {}
    b4_dict["weight"] = model_dict_orig[37][1]
    b4_dict["bias"] = model_dict_orig[38][1]
    b4_dict["running_mean"] = model_dict_orig[39][1]
    b4_dict["running_var"] = model_dict_orig[40][1]
    b4_dict["num_batches_tracked"] = model_dict_orig[41][1]
    model_dict["b4"] = b4_dict

    c4_2_dict = {}
    c4_2_dict["weight"] = model_dict_orig[42][1]
    c4_2_dict["bias"] = torch.zeros(c4_2_dict["weight"].shape[0])
    model_dict["c4_2"] = c4_2_dict
    b4_2_dict = {}
    b4_2_dict["weight"] = model_dict_orig[43][1]
    b4_2_dict["bias"] = model_dict_orig[44][1]
    b4_2_dict["running_mean"] = model_dict_orig[45][1]
    b4_2_dict["running_var"] = model_dict_orig[46][1]
    b4_2_dict["num_batches_tracked"] = model_dict_orig[47][1]
    model_dict["b4_2"] = b4_2_dict

    bnc_dict = {}
    bnc_dict["weight"] = model_dict_orig[48][1]
    bnc_dict["bias"] = torch.zeros(bnc_dict["weight"].shape[0])
    model_dict["bnc"] = bnc_dict
    bnb_dict = {}
    bnb_dict["weight"] = model_dict_orig[49][1]
    bnb_dict["bias"] = model_dict_orig[50][1]
    bnb_dict["running_mean"] = model_dict_orig[51][1]
    bnb_dict["running_var"] = model_dict_orig[52][1]
    bnb_dict["num_batches_tracked"] = model_dict_orig[53][1]
    model_dict["bnb"] = bnb_dict

    bnc_2_dict = {}
    bnc_2_dict["weight"] = model_dict_orig[54][1]
    bnc_2_dict["bias"] = torch.zeros(bnc_2_dict["weight"].shape[0])
    model_dict["bnc_2"] = bnc_2_dict
    bnb_2_dict = {}
    bnb_2_dict["weight"] = model_dict_orig[55][1]
    bnb_2_dict["bias"] = model_dict_orig[56][1]
    bnb_2_dict["running_mean"] = model_dict_orig[57][1]
    bnb_2_dict["running_var"] = model_dict_orig[58][1]
    bnb_2_dict["num_batches_tracked"] = model_dict_orig[59][1]
    model_dict["bnb_2"] = bnb_2_dict

    c5T_dict = {}
    c5T_dict["weight"] = model_dict_orig[60][1]
    c5T_dict["bias"] = model_dict_orig[61][1]
    model_dict["c5T"] = c5T_dict

    c5_dict = {}
    c5_dict["weight"] = model_dict_orig[62][1]
    c5_dict["bias"] = torch.zeros(c5_dict["weight"].shape[0])
    model_dict["c5"] = c5_dict
    b5_dict = {}
    b5_dict["weight"] = model_dict_orig[63][1]
    b5_dict["bias"] = model_dict_orig[64][1]
    b5_dict["running_mean"] = model_dict_orig[65][1]
    b5_dict["running_var"] = model_dict_orig[66][1]
    b5_dict["num_batches_tracked"] = model_dict_orig[67][1]
    model_dict["b5"] = b5_dict

    c5_2_dict = {}
    c5_2_dict["weight"] = model_dict_orig[68][1]
    c5_2_dict["bias"] = torch.zeros(c5_2_dict["weight"].shape[0])
    model_dict["c5_2"] = c5_2_dict
    b5_2_dict = {}
    b5_2_dict["weight"] = model_dict_orig[69][1]
    b5_2_dict["bias"] = model_dict_orig[70][1]
    b5_2_dict["running_mean"] = model_dict_orig[71][1]
    b5_2_dict["running_var"] = model_dict_orig[72][1]
    b5_2_dict["num_batches_tracked"] = model_dict_orig[73][1]
    model_dict["b5_2"] = b5_2_dict

    c6T_dict = {}
    c6T_dict["weight"] = model_dict_orig[74][1]
    c6T_dict["bias"] = model_dict_orig[75][1]
    model_dict["c6T"] = c6T_dict

    c6_dict = {}
    c6_dict["weight"] = model_dict_orig[76][1]
    c6_dict["bias"] = torch.zeros(c6_dict["weight"].shape[0])
    model_dict["c6"] = c6_dict
    b6_dict = {}
    b6_dict["weight"] = model_dict_orig[77][1]
    b6_dict["bias"] = model_dict_orig[78][1]
    b6_dict["running_mean"] = model_dict_orig[79][1]
    b6_dict["running_var"] = model_dict_orig[80][1]
    b6_dict["num_batches_tracked"] = model_dict_orig[81][1]
    model_dict["b6"] = b6_dict

    c6_2_dict = {}
    c6_2_dict["weight"] = model_dict_orig[82][1]
    c6_2_dict["bias"] = torch.zeros(c6_2_dict["weight"].shape[0])
    model_dict["c6_2"] = c6_2_dict
    b6_2_dict = {}
    b6_2_dict["weight"] = model_dict_orig[83][1]
    b6_2_dict["bias"] = model_dict_orig[84][1]
    b6_2_dict["running_mean"] = model_dict_orig[85][1]
    b6_2_dict["running_var"] = model_dict_orig[86][1]
    b6_2_dict["num_batches_tracked"] = model_dict_orig[87][1]
    model_dict["b6_2"] = b6_2_dict

    c7T_dict = {}
    c7T_dict["weight"] = model_dict_orig[88][1]
    c7T_dict["bias"] = model_dict_orig[89][1]
    model_dict["c7T"] = c7T_dict

    c7_dict = {}
    c7_dict["weight"] = model_dict_orig[90][1]
    c7_dict["bias"] = torch.zeros(c7_dict["weight"].shape[0])
    model_dict["c7"] = c7_dict
    b7_dict = {}
    b7_dict["weight"] = model_dict_orig[91][1]
    b7_dict["bias"] = model_dict_orig[92][1]
    b7_dict["running_mean"] = model_dict_orig[93][1]
    b7_dict["running_var"] = model_dict_orig[94][1]
    b7_dict["num_batches_tracked"] = model_dict_orig[95][1]
    model_dict["b7"] = b7_dict

    c7_2_dict = {}
    c7_2_dict["weight"] = model_dict_orig[96][1]
    c7_2_dict["bias"] = torch.zeros(c7_2_dict["weight"].shape[0])
    model_dict["c7_2"] = c7_2_dict
    b7_2_dict = {}
    b7_2_dict["weight"] = model_dict_orig[97][1]
    b7_2_dict["bias"] = model_dict_orig[98][1]
    b7_2_dict["running_mean"] = model_dict_orig[99][1]
    b7_2_dict["running_var"] = model_dict_orig[100][1]
    b7_2_dict["num_batches_tracked"] = model_dict_orig[101][1]
    model_dict["b7_2"] = b7_2_dict

    c8T_dict = {}
    c8T_dict["weight"] = model_dict_orig[102][1]
    c8T_dict["bias"] = model_dict_orig[103][1]
    model_dict["c8T"] = c8T_dict

    c8_dict = {}
    c8_dict["weight"] = model_dict_orig[104][1]
    c8_dict["bias"] = torch.zeros(c8_dict["weight"].shape[0])
    model_dict["c8"] = c8_dict
    b8_dict = {}
    b8_dict["weight"] = model_dict_orig[105][1]
    b8_dict["bias"] = model_dict_orig[106][1]
    b8_dict["running_mean"] = model_dict_orig[107][1]
    b8_dict["running_var"] = model_dict_orig[108][1]
    b8_dict["num_batches_tracked"] = model_dict_orig[109][1]
    model_dict["b8"] = b8_dict

    c8_2_dict = {}
    c8_2_dict["weight"] = model_dict_orig[110][1]
    c8_2_dict["bias"] = torch.zeros(c8_2_dict["weight"].shape[0])
    model_dict["c8_2"] = c8_2_dict
    b8_2_dict = {}
    b8_2_dict["weight"] = model_dict_orig[111][1]
    b8_2_dict["bias"] = model_dict_orig[112][1]
    b8_2_dict["running_mean"] = model_dict_orig[113][1]
    b8_2_dict["running_var"] = model_dict_orig[114][1]
    b8_2_dict["num_batches_tracked"] = model_dict_orig[115][1]
    model_dict["b8_2"] = b8_2_dict

    output_layer_dict = {}
    output_layer_dict["weight"] = model_dict_orig[116][1]
    output_layer_dict["bias"] = model_dict_orig[117][1]
    model_dict["output_layer"] = output_layer_dict

    return model_dict


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 511


def custom_preprocessor(model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, UNet):
        print("\n\n\n")
        print("model output weights type: ", type(model.output_layer.weight))
        # print("model output weights: ", list(model.output_layer.weight))

        ttnn_module_args.c1["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c1["padded_input_channels"] = 16
        # ttnn_module_args.c1["use_shallow_conv_variant"] = True
        ttnn_module_args.c1_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c1_2["use_shallow_conv_variant"] = True
        ttnn_module_args.c1["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1_2["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1_2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c1_2["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c1["deallocate_activation"] = True
        ttnn_module_args.c1_2["deallocate_activation"] = True
        ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
        ttnn_module_args.c1_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
        #        ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        #        ttnn_module_args.c1_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}

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
        # ttnn_module_args.c3["use_shallow_conv_variant"] = True
        ttnn_module_args.c3_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c3_2["use_shallow_conv_variant"] = True
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
        # ttnn_module_args.c4["use_shallow_conv_variant"] = False
        ttnn_module_args.c4_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c4_2["use_shallow_conv_variant"] = False
        ttnn_module_args.c4["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c4_2["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c4_2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c4["activation"] = "relu"  # Fuse relu with conv4
        ttnn_module_args.c4_2["activation"] = "relu"  # Fuse relu with conv4
        ttnn_module_args.c4["deallocate_activation"] = True
        ttnn_module_args.c4_2["deallocate_activation"] = True
        ttnn_module_args.c4["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c4_2["conv_blocking_and_parallelization_config_override"] = None
        # ttnn_module_args.c4_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}

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
        # ttnn_module_args.bnc_2["conv_blocking_and_parallelization_config_override"] = {"act_block_hnn": 64}

        ttnn_module_args.c5["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c5_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c5_3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c5["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c5_2["dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c5_3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c5_2["weights_dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c5_3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c5["activation"] = "relu"  # Fuse relu with conv5
        ttnn_module_args.c5_2["activation"] = "relu"  # Fuse relu with conv5
        # ttnn_module_args.c5_3["activation"] = "relu"  # Fuse relu with conv5
        ttnn_module_args.c5["deallocate_activation"] = True
        ttnn_module_args.c5_2["deallocate_activation"] = True
        # ttnn_module_args.c5_3["deallocate_activation"] = True
        ttnn_module_args.c5["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c5_2["conv_blocking_and_parallelization_config_override"] = None
        # ttnn_module_args.c5_3["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.c6["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c6["use_shallow_conv_variant"] = True
        ttnn_module_args.c6_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c6_2["use_shallow_conv_variant"] = True
        # ttnn_module_args.c6_3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c6["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c6_2["dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c6_3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c6_2["weights_dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c6_3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c6["activation"] = "relu"  # Fuse relu with conv6
        ttnn_module_args.c6_2["activation"] = "relu"  # Fuse relu with conv6
        # ttnn_module_args.c6_3["activation"] = "relu"  # Fuse relu with conv6
        ttnn_module_args.c6["deallocate_activation"] = True
        ttnn_module_args.c6_2["deallocate_activation"] = True
        # ttnn_module_args.c6_3["deallocate_activation"] = True
        ttnn_module_args.c6["conv_blocking_and_parallelization_config_override"] = None
        ttnn_module_args.c6_2["conv_blocking_and_parallelization_config_override"] = None
        # ttnn_module_args.c6_3["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.c7["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c7_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c7_3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c7["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_2["dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c7_3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_2["weights_dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c7_3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7["activation"] = "relu"  # Fuse relu with conv7
        ttnn_module_args.c7_2["activation"] = "relu"  # Fuse relu with conv7
        # ttnn_module_args.c7_3["activation"] = "relu"  # Fuse relu with conv7
        ttnn_module_args.c7["deallocate_activation"] = True
        ttnn_module_args.c7_2["deallocate_activation"] = True
        # ttnn_module_args.c7_3["deallocate_activation"] = True
        ttnn_module_args.c7["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        ttnn_module_args.c7_2["conv_blocking_and_parallelization_config_override"] = None
        # ttnn_module_args.c7_3["conv_blocking_and_parallelization_config_override"] = None

        ttnn_module_args.c8["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c8_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        # ttnn_module_args.c8_3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c8["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8_2["dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c8_3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8_2["weights_dtype"] = ttnn.bfloat8_b
        # ttnn_module_args.c8_3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["activation"] = "relu"  # Fuse relu with conv8
        ttnn_module_args.c8_2["activation"] = "relu"  # Fuse relu with conv8
        # ttnn_module_args.c8_3["activation"] = "relu"  # Fuse relu with conv8
        ttnn_module_args.c8["deallocate_activation"] = True
        ttnn_module_args.c8_2["deallocate_activation"] = True
        # ttnn_module_args.c8_3["deallocate_activation"] = True
        ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        ttnn_module_args.c8_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        # ttnn_module_args.c8_3["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}

        ttnn_module_args.output_layer["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.output_layer["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.output_layer["weights_dtype"] = ttnn.bfloat8_b
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
        # conv5_3_weight, conv5_3_bias = fold_batch_norm2d_into_conv2d(model.c5_3, model.b5_3)
        conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
        conv6_2_weight, conv6_2_bias = fold_batch_norm2d_into_conv2d(model.c6_2, model.b6_2)
        # conv6_3_weight, conv6_3_bias = fold_batch_norm2d_into_conv2d(model.c6_3, model.b6_3)
        conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
        conv7_2_weight, conv7_2_bias = fold_batch_norm2d_into_conv2d(model.c7_2, model.b7_2)
        # conv7_3_weight, conv7_3_bias = fold_batch_norm2d_into_conv2d(model.c7_3, model.b7_3)
        conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
        conv8_2_weight, conv8_2_bias = fold_batch_norm2d_into_conv2d(model.c8_2, model.b8_2)
        # conv8_3_weight, conv8_3_bias = fold_batch_norm2d_into_conv2d(model.c8_3, model.b8_3)

        update_ttnn_module_args(ttnn_module_args.c1)
        update_ttnn_module_args(ttnn_module_args.c1_2)
        update_ttnn_module_args(ttnn_module_args.c2)
        update_ttnn_module_args(ttnn_module_args.c2_2)
        update_ttnn_module_args(ttnn_module_args.c3)
        update_ttnn_module_args(ttnn_module_args.c3_2)
        update_ttnn_module_args(ttnn_module_args.c4)
        update_ttnn_module_args(ttnn_module_args.c4_2)
        update_ttnn_module_args(ttnn_module_args.bnc)
        update_ttnn_module_args(ttnn_module_args.bnc_2)
        update_ttnn_module_args(ttnn_module_args.c5)
        update_ttnn_module_args(ttnn_module_args.c5_2)
        # update_ttnn_module_args(ttnn_module_args.c5_3)
        update_ttnn_module_args(ttnn_module_args.c6)
        update_ttnn_module_args(ttnn_module_args.c6_2)
        # update_ttnn_module_args(ttnn_module_args.c6_3)
        update_ttnn_module_args(ttnn_module_args.c7)
        update_ttnn_module_args(ttnn_module_args.c7_2)
        # update_ttnn_module_args(ttnn_module_args.c7_3)
        update_ttnn_module_args(ttnn_module_args.c8)
        update_ttnn_module_args(ttnn_module_args.c8_2)
        # update_ttnn_module_args(ttnn_module_args.c8_3)
        update_ttnn_module_args(ttnn_module_args.output_layer)

        parameters["c1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.c1)
        parameters["c1_2"] = preprocess_conv2d(conv1_2_weight, conv1_2_bias, ttnn_module_args.c1_2)
        parameters["c2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.c2)
        parameters["c2_2"] = preprocess_conv2d(conv2_2_weight, conv2_2_bias, ttnn_module_args.c2_2)
        parameters["c3"] = preprocess_conv2d(conv3_weight, conv3_bias, ttnn_module_args.c3)
        parameters["c3_2"] = preprocess_conv2d(conv3_2_weight, conv3_2_bias, ttnn_module_args.c3_2)
        parameters["c4"] = preprocess_conv2d(conv4_weight, conv4_bias, ttnn_module_args.c4)
        parameters["c4_2"] = preprocess_conv2d(conv4_2_weight, conv4_2_bias, ttnn_module_args.c4_2)
        parameters["bnc"] = preprocess_conv2d(convbn_weight, convbn_bias, ttnn_module_args.bnc)
        parameters["bnc_2"] = preprocess_conv2d(convbn_2_weight, convbn_2_bias, ttnn_module_args.bnc_2)
        parameters["c5"] = preprocess_conv2d(conv5_weight, conv5_bias, ttnn_module_args.c5)
        parameters["c5_2"] = preprocess_conv2d(conv5_2_weight, conv5_2_bias, ttnn_module_args.c5_2)
        # parameters["c5_3"] = preprocess_conv2d(conv5_3_weight, conv5_3_bias, ttnn_module_args.c5_3)
        parameters["c6"] = preprocess_conv2d(conv6_weight, conv6_bias, ttnn_module_args.c6)
        parameters["c6_2"] = preprocess_conv2d(conv6_2_weight, conv6_2_bias, ttnn_module_args.c6_2)
        # parameters["c6_3"] = preprocess_conv2d(conv6_3_weight, conv6_3_bias, ttnn_module_args.c6_3)
        parameters["c7"] = preprocess_conv2d(conv7_weight, conv7_bias, ttnn_module_args.c7)
        parameters["c7_2"] = preprocess_conv2d(conv7_2_weight, conv7_2_bias, ttnn_module_args.c7_2)
        # parameters["c7_3"] = preprocess_conv2d(conv7_3_weight, conv7_3_bias, ttnn_module_args.c7_3)
        parameters["c8"] = preprocess_conv2d(conv8_weight, conv8_bias, ttnn_module_args.c8)
        parameters["c8_2"] = preprocess_conv2d(conv8_2_weight, conv8_2_bias, ttnn_module_args.c8_2)
        # parameters["c8_3"] = preprocess_conv2d(conv8_3_weight, conv8_3_bias, ttnn_module_args.c8_3)
        parameters["output_layer"] = preprocess_conv2d(
            model.output_layer.weight, model.output_layer.bias, ttnn_module_args.output_layer
        )

    return parameters


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Contracting Path
        self.c1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.r1 = nn.ReLU(inplace=True)
        self.c1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b1_2 = nn.BatchNorm2d(32)
        self.r1_2 = nn.ReLU(inplace=True)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.r2 = nn.ReLU(inplace=True)
        self.c2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b2_2 = nn.BatchNorm2d(64)
        self.r2_2 = nn.ReLU(inplace=True)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.r3 = nn.ReLU(inplace=True)
        self.c3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.b3_2 = nn.BatchNorm2d(128)
        self.r3_2 = nn.ReLU(inplace=True)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(256)
        self.r4 = nn.ReLU(inplace=True)
        self.c4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b4_2 = nn.BatchNorm2d(256)
        self.r4_2 = nn.ReLU(inplace=True)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bnc = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bnb = nn.BatchNorm2d(512)
        self.bnr = nn.ReLU(inplace=True)
        self.bnc_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bnb_2 = nn.BatchNorm2d(512)
        self.bnr_2 = nn.ReLU(inplace=True)

        # self.u4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c5T = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.c5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm2d(256)
        self.r5 = nn.ReLU(inplace=True)
        self.c5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b5_2 = nn.BatchNorm2d(256)
        self.r5_2 = nn.ReLU(inplace=True)
        self.c5_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b5_3 = nn.BatchNorm2d(256)
        self.r5_3 = nn.ReLU(inplace=True)
        # self.u3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c6T = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.c6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.b6 = nn.BatchNorm2d(128)
        self.r6 = nn.ReLU(inplace=True)
        self.c6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.b6_2 = nn.BatchNorm2d(128)
        self.r6_2 = nn.ReLU(inplace=True)
        self.c6_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.b6_3 = nn.BatchNorm2d(128)
        self.r6_3 = nn.ReLU(inplace=True)
        # self.u2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c7T = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.c7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.b7 = nn.BatchNorm2d(64)
        self.r7 = nn.ReLU(inplace=True)
        self.c7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b7_2 = nn.BatchNorm2d(64)
        self.r7_2 = nn.ReLU(inplace=True)
        self.c7_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.b7_3 = nn.BatchNorm2d(64)
        self.r7_3 = nn.ReLU(inplace=True)
        # self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.c8T = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.c8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.b8 = nn.BatchNorm2d(32)
        self.r8 = nn.ReLU(inplace=True)
        self.c8_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b8_2 = nn.BatchNorm2d(32)
        self.r8_2 = nn.ReLU(inplace=True)
        self.c8_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b8_3 = nn.BatchNorm2d(32)
        self.r8_3 = nn.ReLU(inplace=True)

        # Output layer
        self.output_layer = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        c1 = self.c1(x)
        b1 = self.b1(c1)
        r1 = self.r1(b1)
        c1_2 = self.c1_2(r1)
        b1_2 = self.b1_2(c1_2)
        r1_2 = self.r1_2(b1_2)

        p1 = self.p1(r1_2)

        c2 = self.c2(p1)
        b2 = self.b2(c2)
        r2 = self.r2(b2)
        c2_2 = self.c2_2(r2)
        b2_2 = self.b2_2(c2_2)
        r2_2 = self.r2_2(b2_2)
        p2 = self.p2(r2_2)

        c3 = self.c3(p2)
        b3 = self.b3(c3)
        r3 = self.r3(b3)
        c3_2 = self.c3_2(r3)
        b3_2 = self.b3_2(c3_2)
        r3_2 = self.r3_2(b3_2)
        p3 = self.p3(r3_2)

        c4 = self.c4(p3)
        b4 = self.b4(c4)
        r4 = self.r4(b4)
        c4_2 = self.c4_2(r4)
        b4_2 = self.b4_2(c4_2)
        r4_2 = self.r4_2(b4_2)
        p4 = self.p4(r4_2)

        bnc = self.bnc(p4)
        bnb = self.bnb(bnc)
        bnr = self.bnr(bnb)
        bnc_2 = self.bnc_2(bnr)
        bnb_2 = self.bnb_2(bnc_2)
        bnr_2 = self.bnr_2(bnb_2)
        # u4 = self.u4(bnr_2)
        c5T = self.c5T(bnr_2)
        print("the shape of c5T is: ", c5T.size())
        print("the shape of r4_2 is: ", r4_2.size())
        conc1 = torch.cat([c5T, r4_2], dim=1)

        c5 = self.c5(conc1)
        b5 = self.b5(c5)
        r5 = self.r5(b5)
        c5_2 = self.c5_2(r5)
        b5_2 = self.b5_2(c5_2)
        r5_2 = self.r5_2(b5_2)
        #        c5_3 = self.c5_3(r5_2)
        #        b5_3 = self.b5_3(c5_3)
        #        r5_3 = self.r5_3(b5_3)
        # u3 = self.u3(r5_3)
        c6T = self.c6T(r5_2)
        conc2 = torch.cat([c6T, r3_2], dim=1)

        c6 = self.c6(conc2)
        b6 = self.b6(c6)
        r6 = self.r6(b6)
        c6_2 = self.c6_2(r6)
        b6_2 = self.b6_2(c6_2)
        r6_2 = self.r6_2(b6_2)
        #        c6_3 = self.c6_3(r6_2)
        #        b6_3 = self.b6_3(c6_3)
        #        r6_3 = self.r6_3(b6_3)
        # u2 = self.u2(r6_3)
        c7T = self.c7T(r6_2)

        conc3 = torch.cat([c7T, r2_2], dim=1)

        c7 = self.c7(conc3)
        b7 = self.b7(c7)
        r7 = self.r7(b7)
        c7_2 = self.c7_2(r7)
        b7_2 = self.b7_2(c7_2)
        r7_2 = self.r7_2(b7_2)
        #        c7_3 = self.c7_3(r7_2)
        #        b7_3 = self.b7_3(c7_3)
        #        r7_3 = self.r7_3(b7_3)

        # u1 = self.u1(r7_3)
        c8T = self.c8T(r7_2)
        conc4 = torch.cat([c8T, r1_2], dim=1)

        c8 = self.c8(conc4)
        b8 = self.b8(c8)
        r8 = self.r8(b8)
        c8_2 = self.c8_2(r8)
        b8_2 = self.b8_2(c8_2)
        r8_2 = self.r8_2(b8_2)
        c8_3 = self.c8_3(r8_2)
        b8_3 = self.b8_3(c8_3)
        r8_3 = self.r8_3(b8_3)

        # Output layer
        output = self.output_layer(r8_3)

        return output
        # return r8_3


device_id = 0
device = ttnn.open_device(device_id=device_id)

torch.manual_seed(0)

torch_model = UNet()
for layer in torch_model.children():
    print(layer)

new_state_dict = {}
for name, parameter in torch_model.state_dict().items():
    if isinstance(parameter, torch.FloatTensor):
        new_state_dict[name] = parameter + 100.0

torch_model.load_state_dict(new_state_dict)

torch_input_tensor = torch.randn(1, 3, 256, 256)  # Batch size of 2, 3 channels (RGB), 1056x160 input
torch_output_tensor = torch_model(torch_input_tensor)


reader_patterns_cache = {}
parameters = preprocess_model(
    initialize_model=lambda: torch_model,
    run_model=lambda model: model(torch_input_tensor),
    custom_preprocessor=custom_preprocessor,
    reader_patterns_cache=reader_patterns_cache,
    device=device,
)

ttnn_model = ttnn_functional_CONVENTIONAL_unet_maxpoolFallback.UNet(parameters)
output_tensor = ttnn_model.torch_call(torch_input_tensor)
output_tensor = output_tensor[:, 0, :, :]
output_tensor = torch.reshape(
    output_tensor, (output_tensor.shape[0], 1, output_tensor.shape[1], output_tensor.shape[2])
)
assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9999)
ttnn.close_device(device)
