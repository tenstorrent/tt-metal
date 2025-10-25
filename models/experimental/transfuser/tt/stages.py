# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck


optimization_dict = {
    "layer1": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer2": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer3": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # for image width
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # for image width
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer4": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
}


class Ttstages:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        stage_name,
    ) -> None:
        self.inplanes = 32
        stage_config = {
            "layer1": {"planes": 72, "groups": 3},
            "layer2": {"planes": 216, "groups": 9},
            "layer3": {"planes": 576, "groups": 24},
            "layer4": {"planes": 1512, "groups": 63},
        }
        config = stage_config[stage_name]
        self.layer = self._make_layer(
            parameters=parameters,
            planes=config["planes"],
            blocks=len(parameters.keys()),
            stride=stride,
            groups=config["groups"],
            model_config=model_config,
            stage_name=stage_name,
        )

    @staticmethod
    def _make_layer(
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
        stage_name=None,
    ) -> List[TTRegNetBottleneck]:
        """
        parameters:
        - Either a root dict that contains {layer1, layer2, ...} each with {b1,b2,...}
        - Or a stage dict that directly contains {b1,b2,...}
        stage_name:
        - Required if 'parameters' is the root dict (so we can pick the stage).
        - Ignored if 'parameters' already looks like a stage dict.
        """

        # ---- Resolve which stage dict to use ----
        def _resolve_stage_dict(params, stage_key):
            # If it already looks like a stage dict (has b1), just use it
            if isinstance(params, dict) and any(k.startswith("b") for k in params.keys()):
                return params
            # Otherwise expect a root dict with the stage_name present
            if not isinstance(params, dict) or stage_key not in params:
                available = list(params.keys()) if isinstance(params, dict) else []
                raise KeyError(
                    f"Expected a stage dict for '{stage_key}' or a root dict containing it. " f"Got keys: {available}"
                )
            return params[stage_key]

        stage_params = _resolve_stage_dict(parameters, stage_name)

        # ---- Choose paramters per stage ----
        layer_config = optimization_dict[stage_name]

        # ---- Validate available blocks ----
        # Expected names: b1, b2, ..., b{blocks}
        available_block_names = sorted(
            [k for k in stage_params.keys() if k.startswith("b")],
            key=lambda s: int(s[1:]) if s[1:].isdigit() else 0,
        )

        # If fewer blocks than requested, raise a descriptive error
        if len(available_block_names) < blocks:
            raise KeyError(
                f"Requested {blocks} blocks for {stage_name}, but only found blocks: "
                f"{available_block_names}. "
                f"Did you pass parameters for the wrong stage (e.g., layer1 for layer2)?"
            )

        layers = []

        # ---- First block (may have downsample) ----
        downsample = stride != 1 or inplanes != planes
        layers.append(
            TTRegNetBottleneck(
                parameters=stage_params["b1"],
                model_config=model_config,
                layer_config=layer_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
                # shard_layout=shard_layout,
            )
        )
        inplanes = planes

        # ---- Remaining blocks (stride=1, no downsample) ----
        # Build exactly the number requested, in order b2..b{blocks}
        for idx in range(2, blocks + 1):
            bname = f"b{idx}"
            if bname not in stage_params:
                # Extra guard (should have been caught above)
                raise KeyError(f"Missing block '{bname}' in {stage_name}. " f"Available: {available_block_names}")
            layers.append(
                TTRegNetBottleneck(
                    parameters=stage_params[bname],
                    model_config=model_config,
                    layer_config=layer_config,
                    stride=1,
                    downsample=False,
                    groups=groups,
                    # shard_layout=shard_layout,
                )
            )

        return layers

    def __call__(self, x, device, input_shape=None):
        shape = input_shape if input_shape is not None else x.shape
        # Process image input
        for block in self.layer:
            x, shape = block(x, device, shape)

        return x, shape
