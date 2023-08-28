from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import pytest
import tt_lib
from models.utility_functions import comp_pcc
from tests.python_api_testing.models.resnet.metalResnetBlock50 import compute_conv_output_shape, resnet50_1x1_conv_as_matmul, resnet50_optimized_conv, _nearest_32, format_tensor

# hardcoding matmul config for 1x1 convs
# key: mm act height, mm act width, mm weight width
hardcoded_matmul_config_conv = {
    (3136, 64, 64) : {"compute_with_storage_grid_size" : (2,2),
                            "in0_block_w" : 2,
                            "out_subblock_h" : 1,
                            "out_subblock_w": 1,
                            "per_core_M": 49,
                            "per_core_N": 1,
                        },

    (3136, 64, 256) : {"compute_with_storage_grid_size" : (4,2),
                            "in0_block_w" : 2,
                            "out_subblock_h" : 1,
                            "out_subblock_w": 1,
                            "per_core_M": 49,
                            "per_core_N": 2,
                        },
    (3136, 256, 64) : {"compute_with_storage_grid_size" : (2,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 14,
                    "per_core_N": 1,
                },
    (3136, 256, 128) : {"compute_with_storage_grid_size" : (4,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 14,
                    "per_core_N": 1,
                },
    (800, 128, 512) : {"compute_with_storage_grid_size" : (4,2),
                    "in0_block_w" : 4,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 13,
                    "per_core_N": 4,
                },
    (800, 512, 128) : {"compute_with_storage_grid_size" : (4,4),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 7,
                    "per_core_N": 1,
                },
    (800, 512, 256) : {"compute_with_storage_grid_size" : (8,4),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 7,
                    "per_core_N": 1,
                },
    (224, 256, 1024) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 4,
                },
    (224, 1024, 256) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 32,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 1,
                },
    (224, 1024, 512) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 32,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 2,
                },
    (64, 512, 2048) : {"compute_with_storage_grid_size" : (8,2),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 8,
                },
    (64, 2048, 512) : {"compute_with_storage_grid_size" : (8,2),
                    "in0_block_w" : 64,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 2,
                },
}

hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv = {
    (3136, 64) : [128, 64, 128, 64] ,
    (800, 128) : [128, 128, 128, 64] ,
    (224, 256) : [64, 128, 64, 128],
    (64, 512) : [32, 64, 32, 64] ,
}

@pytest.mark.parametrize("use_new_matmul", (True,))
@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # 1x1 convs in rn50
        (64, 64, 56, 56, 1, 1, 1, 1, 0, 0),
        (256, 64, 56, 56, 1, 1, 1, 1, 0, 0), # slow with new_matmul but less than bias computation time
        (64, 256, 56, 56, 1, 1, 1, 1, 0, 0),
        (64, 256, 56, 56, 1, 1, 1, 1, 0, 0),
        (128, 256, 56, 56, 1, 1, 1, 1, 0, 0),
        (512, 128, 28, 28, 1, 1, 1, 1, 0, 0),
        (128, 512, 28, 28, 1, 1, 1, 1, 0, 0),
        (256, 512, 28, 28, 1, 1, 1, 1, 0, 0),
        (1024, 256, 14, 14, 1, 1, 1, 1, 0, 0),
        (256, 1024, 14, 14, 1, 1, 1, 1, 0, 0),
        (512, 1024, 14, 14, 1, 1, 1, 1, 0, 0),
        (2048, 512, 7, 7, 1, 1, 1, 1, 0, 0),
        (512, 2048, 7, 7, 1, 1, 1, 1, 0, 0), # slightly slower with new matmul but less than old matmul + bias computation time

        #3x3 convs in rn50 (not complete list)
        (64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
        (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        (512, 512, 14, 14, 3, 3, 2, 2, 1, 1),
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),

        # downsample convs in rn50 (not complete list)
        (128, 128, 56, 56, 1, 1, 2, 2, 0, 0),
        (256, 256, 28, 28, 3, 3, 2, 2, 1, 1),

    )
)
def test_resnet50_conv(use_program_cache, device, K,C,H,W,R,S,stride_h,stride_w,pad_h,pad_w, use_new_matmul):
    for i in range(1): # increase num of iterations to test op caching
        assert C % 32 == 0
        assert K % 32 == 0
        torch.manual_seed(0)
        memory_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)
        conv_input_shape = [1, C, H, W]
        conv_weight_shape = [K, C, R, S]
        conv_bias_shape = [1, 1, 1, K]
        conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
        conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
        conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
        conv_weight_pyt = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
        conv_bias_pyt = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
        out_golden = torch.nn.functional.conv2d(conv_input_pyt, conv_weight_pyt, bias=conv_bias_pyt.reshape(-1), stride=(stride_h, stride_w), padding=(pad_h, pad_w))

        is_1x1_conv = R == 1 and S == 1 and stride_h == 1 and stride_w == 1 and pad_h == 0 and pad_w == 0

        conv_params = [K, C, R, S, stride_h, stride_w, pad_h, pad_w, 1, 1]
        conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
        print("Conv output shape - ", conv_output_shape)
        conv_as_mm_padded_act_height = _nearest_32(conv_output_shape[1] * conv_output_shape[2])

        if (is_1x1_conv):
            matmul_config = None
            if (conv_as_mm_padded_act_height, C, K) in hardcoded_matmul_config_conv and use_new_matmul:
                print("Setting matmul config for 1x1 conv")
                matmul_config = hardcoded_matmul_config_conv[(conv_as_mm_padded_act_height, C, K)]
            # 1x1 conv with stride 1 padding 0 is run using regular matmul
            conv = resnet50_1x1_conv_as_matmul(conv_weight_pyt.reshape(-1).tolist(), conv_params, device, conv_bias_pyt.reshape(-1).tolist(), matmul_config)
        else:

            assert (conv_as_mm_padded_act_height, K) in hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv
            [act_block_h_datums, weight_block_w_datums, out_subblock_h_datums, out_subblock_w_datums] = hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv[(conv_as_mm_padded_act_height, K)]
            conv = resnet50_optimized_conv(conv_weight_pyt.reshape(-1).tolist(),
                                conv_params,
                                device,
                                [act_block_h_datums, C*S], [C*S, weight_block_w_datums],
                                [out_subblock_h_datums, out_subblock_w_datums],
                                conv_bias_pyt.reshape(-1).tolist())

        conv_input_on_device = tt_lib.tensor.Tensor(
                                conv_input_pyt_nhwc.reshape(-1).tolist(),
                                conv_input_pyt_nhwc.shape,
                                tt_lib.tensor.DataType.BFLOAT16,
                                tt_lib.tensor.Layout.ROW_MAJOR).to(device, memory_config)

        if (is_1x1_conv):
            # convert activation RM to tile layout
            conv_input_on_device = conv_input_on_device.reshape(1, 1, conv_input_shape_nhwc[1]*conv_input_shape_nhwc[2], conv_input_shape_nhwc[3])
            conv_input_on_device = format_tensor(conv_input_on_device, tt_lib.tensor.Layout.TILE, device, memory_config)

        output_on_device = conv(conv_input_on_device)

        # convert tiled output to RM
        assert(output_on_device.layout() == tt_lib.tensor.Layout.TILE)
        output_on_device = format_tensor(output_on_device, tt_lib.tensor.Layout.ROW_MAJOR, device, memory_config)
        output_on_device = output_on_device.reshape(conv_output_shape[0], conv_output_shape[1], conv_output_shape[2], conv_output_shape[3])

        # Copy to host and Compare against pytorch
        out = output_on_device.cpu()
        assert out.layout() == tt_lib.tensor.Layout.ROW_MAJOR

        out_result = out.to_torch()
        # NHWC to NCHW
        out_result = torch.transpose(out_result, 2, 3)
        out_result = torch.transpose(out_result, 1, 2)

        # Compare against golden
        assert out_result.shape == out_golden.shape
        passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
        assert comp_pcc(out_golden, out_result, 0.99)
        print("Passing=", passing_pcc)
        print("Output pcc=", output_pcc)
        assert passing_pcc
