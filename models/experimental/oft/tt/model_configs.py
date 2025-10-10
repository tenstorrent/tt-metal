# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect
import ttnn


class ModelOptimizations:
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
        self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Should be LoFi on N1
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # MM COMPUTE KERNEL CONFIGURATION
        self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Should be LoFi on N1
            math_approx_mode=False,
            fp32_dest_acc_en=True,
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

        # not a conv
        return None

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

        # not a matmul
        return None

    def get_compute_config(self, module_path):
        # Return default compute config for all paths
        return self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"]

    def apply(self, layer_args, model_prefix=None):
        """Apply optimizations to the provided layer_args dictionary in place"""

        def walk_layer_args(layer_args, prefix=""):
            """Recursively walk through state dictionary and identify dictionaries containing leaf nodes"""
            all_conv_layers = {}
            all_mm_layers = {}

            for key, value in layer_args.items():
                current_path = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    # Check if this dictionary contains any non-dict values (leaf nodes))
                    has_leaf_nodes = any(not isinstance(v, dict) for v in value.values())

                    if has_leaf_nodes:
                        # Check if any of the leaf nodes are conv layers
                        if self._is_conv_layer(current_path):
                            all_conv_layers[current_path] = value
                        elif self._is_mm_layer(current_path):
                            all_mm_layers[current_path] = value

                    # Continue recursion for all dictionaries
                    conv_layers_rec, mm_layers_rec = walk_layer_args(value, current_path)
                    all_conv_layers.update(conv_layers_rec)
                    all_mm_layers.update(mm_layers_rec)

            return all_conv_layers, all_mm_layers

        # Walk through the state dictionary
        conv_layers, matmul_layers = walk_layer_args(layer_args, prefix=model_prefix)
        # No need to remove duplicates as dictionaries automatically handle unique keys

        # Print convolution settings for each layer
        print("=" * 80)
        print("CONVOLUTION LAYER SETTINGS")
        print("=" * 80)

        for path, args in conv_layers.items():
            print(f"\nLayer: {path}")

            # Get the convolution configuration for this layer
            conv_config = self.get_conv_config(path)
            assert conv_config is not None, f"Conv config not found for layer {path}"

            for name, value in inspect.getmembers(conv_config):
                if not name.startswith("__") and not inspect.ismethod(value):
                    args[f"{name}"] = value

            compute_config = self.get_compute_config(path)
            assert compute_config is not None, f"Compute config not found for layer {path}"

            for name, value in inspect.getmembers(compute_config):
                if not name.startswith("__") and not inspect.ismethod(value):
                    args[f"{name}"] = value

    def _is_conv_layer(self, path):
        """Check if a parameter path belongs to a convolution layer"""
        if self.get_conv_config(path) is not None:
            return True
        return False

    def _is_mm_layer(self, path):
        """Check if a parameter path belongs to a matmul layer"""
        if self.get_matmul_config(path) is not None:
            return True
        return False
