# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ModelOptimisations:
    @staticmethod
    def process(parameters):
        # Topdown Network conv config
        for i in range(len(parameters.conv_args.topdown)):
            parameters.conv_args.topdown[i]["conv1"]["slice_type"] = ttnn.Conv2dDRAMSliceHeight
            parameters.conv_args.topdown[i]["conv1"]["num_slices"] = 2
            parameters.conv_args.topdown[i]["conv1"]["math_fidelity"] = ttnn.MathFidelity.HiFi2
            parameters.conv_args.topdown[i]["conv1"]["fp32_dest_acc_en"] = True
            parameters.conv_args.topdown[i]["conv1"]["output_layout"] = ttnn.ROW_MAJOR_LAYOUT
            parameters.conv_args.topdown[i]["conv1"]["weights_dtype"] = ttnn.bfloat8_b

            parameters.conv_args.topdown[i]["conv2"]["slice_type"] = ttnn.Conv2dDRAMSliceHeight
            parameters.conv_args.topdown[i]["conv2"]["num_slices"] = 2
            parameters.conv_args.topdown[i]["conv2"]["math_fidelity"] = ttnn.MathFidelity.HiFi2
            parameters.conv_args.topdown[i]["conv2"]["fp32_dest_acc_en"] = True
            parameters.conv_args.topdown[i]["conv2"]["output_layout"] = ttnn.ROW_MAJOR_LAYOUT
            parameters.conv_args.topdown[i]["conv2"]["weights_dtype"] = ttnn.bfloat8_b
        return parameters
