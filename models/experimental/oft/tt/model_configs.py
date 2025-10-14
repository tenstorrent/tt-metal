# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)


def create_sharding_strategy(conv2d_config):
    """Create appropriate sharding strategy from conv2d_args"""
    from models.tt_cnn.tt.builder import (
        AutoShardedStrategyConfiguration,
        BlockShardedStrategyConfiguration,
        HeightShardedStrategyConfiguration,
        WidthShardedStrategyConfiguration,
    )

    shard_layout = conv2d_config.shard_layout
    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return HeightShardedStrategyConfiguration(
            reshard_if_not_optimal=conv2d_config.reshard_if_not_optimal,
            act_block_h_override=conv2d_config.act_block_h_override,
        )
    elif shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        return BlockShardedStrategyConfiguration(
            reshard_if_not_optimal=conv2d_config.reshard_if_not_optimal,
        )
    elif shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return WidthShardedStrategyConfiguration(
            reshard_if_not_optimal=conv2d_config.reshard_if_not_optimal,
        )
    else:
        return AutoShardedStrategyConfiguration()


def create_slice_strategy(conv2d_slice_config):
    """Create appropriate slice strategy from conv2d_slice_config"""
    from models.tt_cnn.tt.builder import (
        L1FullSliceStrategyConfiguration,
        HeightSliceStrategyConfiguration,
        WidthSliceStrategyConfiguration,
    )

    slice_type = conv2d_slice_config.slice_type
    if slice_type == ttnn.Conv2dL1Full:
        return L1FullSliceStrategyConfiguration()
    if slice_type == ttnn.Conv2dDRAMSliceHeight:
        return HeightSliceStrategyConfiguration(conv2d_slice_config.num_slices)
    elif slice_type == ttnn.Conv2dDRAMSliceWidth:
        return WidthSliceStrategyConfiguration(conv2d_slice_config.num_slices)
    else:
        return None


def update_conv2d_configuration(
    conv2d_configuration: Conv2dConfiguration,
    output_dtype: ttnn.DataType,
    conv2d_config: ttnn.Conv2dConfig,
    conv2d_slice_config: ttnn.Conv2dSliceConfig,
    compute_config: ttnn.WormholeComputeKernelConfig,
):
    """Update conv2d_config with values from conv2d_config_override that are not inherited from torch model"""

    # Create a new Conv2dConfiguration instance with updated values
    return Conv2dConfiguration(
        # Keep original values from torch model
        input_height=conv2d_configuration.input_height,
        input_width=conv2d_configuration.input_width,
        in_channels=conv2d_configuration.in_channels,
        out_channels=conv2d_configuration.out_channels,
        batch_size=conv2d_configuration.batch_size,
        kernel_size=conv2d_configuration.kernel_size,
        weight=conv2d_configuration.weight,
        stride=conv2d_configuration.stride,
        padding=conv2d_configuration.padding,
        groups=conv2d_configuration.groups,
        dilation=conv2d_configuration.dilation,
        bias=conv2d_configuration.bias,
        # Update with new configuration values
        sharding_strategy=create_sharding_strategy(conv2d_config),
        slice_strategy=create_slice_strategy(conv2d_slice_config),
        weights_dtype=conv2d_config.weights_dtype,
        activation=conv2d_config.activation,
        output_dtype=output_dtype,
        output_layout=conv2d_config.output_layout,
        enable_act_double_buffer=conv2d_config.enable_act_double_buffer,
        enable_weights_double_buffer=conv2d_config.enable_weights_double_buffer,
        deallocate_activation=conv2d_config.deallocate_activation,
        reallocate_halo_output=conv2d_config.reallocate_halo_output,
        math_fidelity=compute_config.math_fidelity,
        fp32_dest_acc_en=compute_config.fp32_dest_acc_en,
        packer_l1_acc=compute_config.packer_l1_acc,
    )


class ModelOptimizations:
    def __init__(
        self,
        conv_output_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat8_b,
    ):
        self.conv_configs = {}
        self.conv_slice_configs = {}
        self.conv_output_dtype = conv_output_dtype
        self.compute_configs = {}
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b

        # CONFIGURATIONS CONV CONFIG
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

        self.conv_configs["HS_ABH_32_TILE"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32,
            config_tensors_in_dram=True,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.conv_configs["HS_ABH_32_TILE_DEALLOC"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32,
            config_tensors_in_dram=True,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.conv_configs["HS_ABH_64_RM"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=False,  # cannot deallocate if in dram
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32 * 2,
            config_tensors_in_dram=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.conv_configs["DEALLOC_RM"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32 * 4,
            config_tensors_in_dram=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # CONFIGURATION CONV SLICE CONFIG
        self.conv_slice_configs["DEFAULT"] = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dL1Full,
            num_slices=0,
        )
        self.conv_slice_configs["HSLICE_2"] = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
            num_slices=2,
        )
        self.conv_slice_configs["HSLICE_4"] = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
            num_slices=4,
        )

        # CONFIGURATION COMPUTE KERNEL
        self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Should be LoFi on N1
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # CONFIGURATION MM COMPUTE KERNEL
        self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,  # Should be LoFi on N1
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def get_conv_config(self, conv_path):
        """Return conv2d_config, conv2d_slice_config, compute_config, conv_output_dtype for the given conv_path"""

        if conv_path is None:
            assert False, "conv_path cannot be None"

        # Frontend ResNet feature extractor convolutions
        if conv_path == "frontend.conv1":
            return (
                self.conv_configs["DEFAULT"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer1.0.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer1.0.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer1.1.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer1.1.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer2.0.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer2.0.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer2.0.downsample.0":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer2.1.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer2.1.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer3.0.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer3.0.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer3.0.downsample.0":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer3.1.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer3.1.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer4.0.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer4.0.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer4.0.downsample.0":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer4.1.conv1":
            return (
                self.conv_configs["HS_ABH_32_TILE"],  # has residual in DRAM
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "frontend.layer4.1.conv2":
            return (
                self.conv_configs["HS_ABH_32_TILE_DEALLOC"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )

        # Lateral connection convolutions
        if conv_path == "lat8":
            return (
                self.conv_configs["DEFAULT"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "lat16":
            return (
                self.conv_configs["DEFAULT"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "lat32":
            return (
                self.conv_configs["DEFAULT"],
                self.conv_slice_configs["DEFAULT"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )

        # Top-down path convolutions
        if conv_path == "topdown.0.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.0.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.1.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.1.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.2.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.2.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.3.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.3.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.4.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.4.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.5.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.5.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.6.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.6.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.7.conv1":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )
        if conv_path == "topdown.7.conv2":
            return (
                self.conv_configs["HS_ABH_64_RM"],
                self.conv_slice_configs["HSLICE_4"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )

        # Head convolution
        if conv_path == "head":
            return (
                self.conv_configs["DEALLOC_RM"],
                self.conv_slice_configs["HSLICE_2"],
                self.compute_configs["DEFAULT_CONV_COMPUTE_CONFIG"],
                self.conv_output_dtype,
            )

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

    def apply(self, state_dict, model_prefix=None):
        """Apply optimizations to the provided layer_args dictionary in place"""

        # Get all weights and biases from state_dict

        def load_to_weights_and_biases(state_dict, all_conv_layers=None, all_mm_layers=None, prefix=""):
            """Recursively walk through state dictionary and map path to weights and biases"""
            all_conv_layers = all_conv_layers or {}
            all_mm_layers = all_mm_layers or {}

            for key, value in state_dict.items():
                current_path = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    # Check if this dictionary contains any non-dict values (leaf nodes))
                    has_leaf_nodes = any(not isinstance(v, dict) for v in value.values())
                    if has_leaf_nodes:
                        # Check if any of the leaf nodes are conv layers
                        if self._is_conv_layer(current_path):
                            all_conv_layers[current_path] = all_conv_layers.get(current_path) or {}
                            all_conv_layers[current_path]["bias"] = value.get("bias", None)
                            all_conv_layers[current_path]["weight"] = value.get("weight", None)
                        elif self._is_mm_layer(current_path):
                            all_mm_layers[current_path] = all_mm_layers.get(current_path) or {}
                            all_mm_layers[current_path]["bias"] = value.get("bias", None)
                            all_mm_layers[current_path]["weight"] = value.get("weight", None)

                    # Continue recursion for all dictionaries
                    conv_layers_rec, mm_layers_rec = load_to_weights_and_biases(
                        value, all_conv_layers=all_conv_layers, all_mm_layers=all_mm_layers, prefix=current_path
                    )
                    all_conv_layers.update(conv_layers_rec)
                    all_mm_layers.update(mm_layers_rec)

            return all_conv_layers, all_mm_layers

        def load_path_to_layer_args(layer_args, all_conv_layers=None, all_mm_layers=None, prefix=""):
            """Recursively walk through layer_args and map path to conv and matmul layer arguments"""
            all_conv_layers = all_conv_layers or {}
            all_mm_layers = all_mm_layers or {}

            for key, value in layer_args.items():
                current_path = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    # Check if this dictionary contains any non-dict values (leaf nodes))
                    # from ttnn.model_preprocessing import ModuleArgs, Conv2dArgs
                    # has_leaf_nodes = isinstance(value, ModuleArgs)
                    has_leaf_nodes = any(not isinstance(v, dict) for v in value.values())
                    if has_leaf_nodes:
                        # Check if any of the leaf nodes are conv layers
                        if self._is_conv_layer(current_path):
                            all_conv_layers[current_path] = all_conv_layers.get(current_path) or {}
                            all_conv_layers[current_path]["layer_args"] = value
                        elif self._is_mm_layer(current_path):
                            all_mm_layers[current_path] = all_mm_layers.get(current_path) or {}
                            all_mm_layers[current_path]["layer_args"] = value

                    # Continue recursion for all dictionaries
                    conv_layers_rec, mm_layers_rec = load_path_to_layer_args(
                        value, all_conv_layers=all_conv_layers, all_mm_layers=all_mm_layers, prefix=current_path
                    )
                    all_conv_layers.update(conv_layers_rec)
                    all_mm_layers.update(mm_layers_rec)

            return all_conv_layers, all_mm_layers

        def store_configuration_to_layer_args(layer_args, all_layers, is_layer, prefix=""):
            """ """
            for key, value in layer_args.items():
                current_path = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    # Check if this dictionary contains any non-dict values (leaf nodes))
                    # from ttnn.model_preprocessing import ModuleArgs, Conv2dArgs
                    # has_leaf_nodes = isinstance(value, ModuleArgs)
                    has_leaf_nodes = any(not isinstance(v, dict) for v in value.values())
                    if has_leaf_nodes:
                        # Check if any of the leaf nodes are conv layers
                        if is_layer(current_path):
                            value["optimized_configuration"] = all_layers.get(current_path, {}).get(
                                "conv2d_configuration", None
                            )

                    # Continue recursion for all dictionaries
                    store_configuration_to_layer_args(
                        value, all_layers=all_layers, is_layer=is_layer, prefix=current_path
                    )

        conv_layers, matmul_layers = {}, {}
        conv_layers, matmul_layers = load_path_to_layer_args(
            state_dict.layer_args, all_conv_layers=conv_layers, all_mm_layers=matmul_layers, prefix=model_prefix
        )
        conv_layers, matmul_layers = load_to_weights_and_biases(
            state_dict, all_conv_layers=conv_layers, all_mm_layers=matmul_layers, prefix=model_prefix
        )

        for path, conv_arg in conv_layers.items():
            conv2d_configuration = Conv2dConfiguration.from_model_args(
                conv_arg["layer_args"], conv_arg["weight"], conv_arg["bias"]
            )
            conv2d_config, conv2d_slice_config, compute_config, output_dtype = self.get_conv_config(path)
            conv_arg["conv2d_configuration"] = update_conv2d_configuration(
                conv2d_configuration, output_dtype, conv2d_config, conv2d_slice_config, compute_config
            )

        store_configuration_to_layer_args(
            state_dict.layer_args, all_layers=conv_layers, is_layer=self._is_conv_layer, prefix=model_prefix
        )

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
