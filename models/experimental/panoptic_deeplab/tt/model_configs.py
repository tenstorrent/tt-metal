# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ModelOptimisations:
    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat8_b,
        conv_w_dtype=ttnn.bfloat8_b,
    ):
        self.conv_configs = {}
        self.conv_output_dtype = conv_act_dtype
        self.compute_configs = {}
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b

        # DEFAULT CONFIGURATION (used by most layers)
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

        # COMPUTE KERNEL CONFIGURATION (used by all convolutions)
        self.compute_configs["CONV_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def get_conv_config(self, conv_path):
        """
        Get the appropriate Conv2dConfig for a given convolution path in Panoptic DeepLab.

        Args:
            conv_path: String path identifying the convolution layer (e.g., "stem.conv1", "res2.0.conv1")

        Returns:
            ttnn.Conv2dConfig: Configuration for the convolution layer
        """
        if conv_path is None:
            return self.conv_configs["DEFAULT"]

        # STEM CONVOLUTIONS
        if conv_path == "stem.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "stem.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "stem.conv3":
            return self.conv_configs["DEFAULT"]

        # RES2 LAYER CONVOLUTIONS
        elif conv_path == "res2.0.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.0.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.0.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.0.shortcut":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.1.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.1.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.1.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.2.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.2.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res2.2.conv3":
            return self.conv_configs["DEFAULT"]

        # RES3 LAYER CONVOLUTIONS
        elif conv_path == "res3.0.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.0.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.0.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.0.shortcut":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.1.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.1.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.1.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.2.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.2.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.2.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.3.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.3.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res3.3.conv3":
            return self.conv_configs["DEFAULT"]

        # RES4 LAYER CONVOLUTIONS
        elif conv_path == "res4.0.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.0.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.0.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.0.shortcut":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.1.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.1.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.1.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.2.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.2.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.2.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.3.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.3.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.3.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.4.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.4.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.4.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.5.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.5.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res4.5.conv3":
            return self.conv_configs["DEFAULT"]

        # RES5 LAYER CONVOLUTIONS
        elif conv_path == "res5.0.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.0.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.0.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.1.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.1.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.1.conv3":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.2.conv1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.2.conv2":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "res5.2.conv3":
            return self.conv_configs["DEFAULT"]

        # ASPP CONVOLUTIONS
        elif conv_path == "aspp.convs.0":  # 1x1 conv branch
            return self.conv_configs["DEFAULT"]
        elif conv_path == "aspp.convs.1":  # 3x3 dilated conv (dilation=6)
            return self.conv_configs["DEFAULT"]
        elif conv_path == "aspp.convs.2":  # 3x3 dilated conv (dilation=12)
            return self.conv_configs["DEFAULT"]
        elif conv_path == "aspp.convs.3":  # 3x3 dilated conv (dilation=18)
            return self.conv_configs["DEFAULT"]
        elif conv_path == "aspp.convs.4":  # Global pooling branch
            return self.conv_configs["DEFAULT"]
        elif conv_path == "aspp.project":  # Final projection conv
            return self.conv_configs["DEFAULT"]

        # DECODER CONVOLUTIONS
        elif conv_path == "decoder.res5.project_conv":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "decoder.res4.project_conv":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "decoder.res4.fuse_conv.0":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "decoder.res4.fuse_conv.1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "decoder.res3.project_conv":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "decoder.res3.fuse_conv.0":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "decoder.res3.fuse_conv.1":
            return self.conv_configs["DEFAULT"]

        # SEMANTIC SEGMENTATION HEAD CONVOLUTIONS
        elif conv_path == "semantic_head.head.0":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "semantic_head.head.1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "semantic_head.predictor":
            return self.conv_configs["DEFAULT"]

        # INSTANCE EMBEDDING HEAD CONVOLUTIONS
        elif conv_path == "instance_head.center_head.0":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "instance_head.center_head.1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "instance_head.center_predictor":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "instance_head.offset_head.0":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "instance_head.offset_head.1":
            return self.conv_configs["DEFAULT"]
        elif conv_path == "instance_head.offset_predictor":
            return self.conv_configs["DEFAULT"]

        # DEFAULT FALLBACK
        else:
            return self.conv_configs["DEFAULT"]

    def get_conv_compute_config(self, module_path):
        """
        Get the appropriate compute kernel configuration for a given module path.

        Args:
            module_path: String path identifying the module

        Returns:
            ttnn.WormholeComputeKernelConfig: Compute configuration for the module
        """
        return self.compute_configs["CONV_COMPUTE_CONFIG"]

    def get_conv_output_dtype(self):
        """Get the default output dtype for convolutions."""
        return self.conv_output_dtype

    def get_slice_config(self, conv_path):
        """
        Get the appropriate slicing configuration for a given convolution path.

        Args:
            conv_path: String path identifying the convolution layer

        Returns:
            dict: Slicing configuration with mode and num_slices
        """
        # STEM: Use width slicing for all conv layers
        if "stem" in conv_path:
            return {"mode": "width", "num_slices": 4}

        # RESNET BOTTLENECKS: Use width slicing for res3 shortcuts
        elif "res3" in conv_path and "shortcut" in conv_path:
            return {"mode": "width", "num_slices": 2}
        else:
            return {"mode": "none", "num_slices": 1}

    def get_aspp_slice_config(self, branch_index):
        """
        Get slicing configuration for ASPP branches.

        Args:
            branch_index: Index of the ASPP branch (0-4)

        Returns:
            dict: Slicing configuration for the branch
        """
        if branch_index == 0:  # 1x1 conv branch
            return {"mode": "none", "num_slices": 1}
        elif branch_index in [1, 2, 3]:  # Dilated conv branches
            channel_slices = [2, 4, 8]
            return {"mode": "channel", "num_slices": channel_slices[branch_index - 1]}
        elif branch_index == 4:  # Global pooling branch
            return {"mode": "none", "num_slices": 1}
        else:
            return {"mode": "none", "num_slices": 1}

    def get_decoder_slice_config(self, conv_path, iteration_index=0):
        """
        Get slicing configuration for decoder convolutions.

        Args:
            conv_path: String path identifying the decoder convolution
            iteration_index: Index of the decoder iteration (0 for first, 1+ for subsequent)

        Returns:
            dict: Slicing configuration for the decoder conv
        """
        if "fuse_conv.0" in conv_path:
            if iteration_index == 0:
                # Check if this is res3 stage (which has 160 channels after projection) - use channel slicing
                if "res3" in conv_path:
                    return {"mode": "channel", "num_slices": 5}
                else:
                    # Other stages use height slicing
                    return {"mode": "height", "num_slices": 4}
            else:
                # Use height slicing for subsequent iterations
                return {"mode": "height", "num_slices": 4}
        elif "fuse_conv.1" in conv_path:
            # Always use height slicing with num_slices=2 (matches original hardcoded logic)
            return {"mode": "height", "num_slices": 2}
        else:
            return {"mode": "none", "num_slices": 1}

    def get_head_slice_config(self, head_type):
        """
        Get slicing configuration for head convolutions.

        Args:
            head_type: Type of head ("semantic", "center", "offset")

        Returns:
            dict: Slicing configuration for the head
        """
        # All head convolutions use height slicing
        return {"mode": "height", "num_slices": 2}
