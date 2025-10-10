# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ModelOptimisations:
    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat8_b,
    ):
        self.conv_configs = {}
        self.conv_output_dtype = conv_act_dtype
        self.compute_configs = {}
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b

        # DEFAULT CONFIGURATION
        self.conv_configs["DEFAULT"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
            config_tensors_in_dram=True,
        )

        # COMPUTE KERNEL CONFIGURATION
        self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"] = ttnn.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Should be LoFi on N1
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # MM COMPUTE KERNEL CONFIGURATION
        self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"] = ttnn.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Should be LoFi on N1
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def get_conv_config(self, conv_path):
        if conv_path is None:
            return self.conv_configs["DEFAULT"]

        # Frontend ResNet feature extractor convolutions
        if conv_path == "frontend.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer1.0.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer1.0.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer1.1.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer1.1.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer2.0.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer2.0.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer2.0.downsample.0":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer2.1.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer2.1.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer3.0.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer3.0.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer3.0.downsample.0":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer3.1.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer3.1.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer4.0.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer4.0.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer4.0.downsample.0":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer4.1.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "frontend.layer4.1.conv2":
            return self.conv_configs["DEFAULT"]

        # Lateral connection convolutions
        if conv_path == "lat8":
            return self.conv_configs["DEFAULT"]
        if conv_path == "lat16":
            return self.conv_configs["DEFAULT"]
        if conv_path == "lat32":
            return self.conv_configs["DEFAULT"]

        # Top-down path convolutions
        if conv_path == "topdown.0.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.0.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.1.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.1.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.2.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.2.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.3.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.3.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.4.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.4.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.5.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.5.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.6.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.6.conv2":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.7.conv1":
            return self.conv_configs["DEFAULT"]
        if conv_path == "topdown.7.conv2":
            return self.conv_configs["DEFAULT"]

        # Head convolution
        if conv_path == "head":
            return self.conv_configs["DEFAULT"]

        # Default fallback
        return self.conv_configs["DEFAULT"]

    def get_matmul_config(self, path):
        if path is None:
            return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

        # OFT Linear layers
        if path == "oft8.conv3d":
            return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]
        if path == "oft16.conv3d":
            return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]
        if path == "oft32.conv3d":
            return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

        # Default fallback
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

    def get_conv_compute_config(self, module_path):
        # Return default compute config for all convolution paths
        return self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"]

    def get_mm_compute_config(self, module_path):
        # Return default compute config for all matmul/linear paths
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]
