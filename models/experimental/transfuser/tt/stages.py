# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List, Dict, Any
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck


optimization_dict: Dict[str, Dict[str, Dict[str, Any]]] = {
    "layer1": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer2": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer3": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 32,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
    "layer4": {
        "conv1": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
        },
        "conv2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc1": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "se_fc2": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "conv3": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        "downsample": {
            "shard_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            "act_block_h": 64,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
    },
}


class Ttstages:
    def __init__(
        self,
        parameters: Dict[str, Any],
        stride: int,
        model_config: Dict[str, Any],
        stage_name: str,
        torch_model=None,
        use_fallback: bool = False,
    ) -> None:
        """
        Builds a sequence of TTRegNetBottleneck blocks for a given stage (layer1..layer4).
        `parameters` can be either:
          - a root dict containing stage dicts (layer1/layer2/.. -> {b1,b2,...}), or
          - a stage dict directly containing {b1,b2,...}.
        """
        self.inplanes = 32
        stage_config = {
            "layer1": {"planes": 72, "groups": 3},
            "layer2": {"planes": 216, "groups": 9},
            "layer3": {"planes": 576, "groups": 24},
            "layer4": {"planes": 1512, "groups": 63},
        }
        if stage_name not in stage_config:
            raise KeyError(f"Unknown stage '{stage_name}'. Expected one of {list(stage_config.keys())}.")

        cfg = stage_config[stage_name]
        planes, groups = cfg["planes"], cfg["groups"]

        # number of blocks = number of entries under stage (b1..bN)
        blocks = (
            len(parameters.keys())
            if any(k.startswith("b") for k in parameters.keys())
            else len(parameters.get(stage_name, {}).keys())
        )

        self.layer = self._make_layer(
            parameters=parameters,
            planes=planes,
            blocks=blocks,
            stride=stride,
            groups=groups,
            model_config=model_config,
            stage_name=stage_name,
            torch_model=torch_model,
            use_fallback=use_fallback,
        )

    @staticmethod
    def _make_layer(
        parameters: Dict[str, Any],
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config: Dict[str, Any] = None,
        stage_name: str = None,
        torch_model=None,
        use_fallback: bool = False,
    ) -> List[TTRegNetBottleneck]:
        """
        Build TTRegNetBottleneck blocks for a stage.

        parameters:
          - Either a root dict that contains {layer1, layer2, ...} each with {b1,b2,...}
          - Or a stage dict that directly contains {b1,b2,...}

        stage_name:
          - Required if 'parameters' is the root dict (so we can pick the stage).
          - Ignored if 'parameters' already looks like a stage dict.
        """

        def _resolve_stage_dict(params: Dict[str, Any], stage_key: str) -> Dict[str, Any]:
            # Already a stage dict?
            if isinstance(params, dict) and any(k.startswith("b") for k in params.keys()):
                return params
            # Otherwise, pull stage dict from root
            if not isinstance(params, dict) or stage_key not in params:
                available = list(params.keys()) if isinstance(params, dict) else []
                raise KeyError(
                    f"Expected a stage dict for '{stage_key}' or a root dict containing it. " f"Got keys: {available}"
                )
            return params[stage_key]

        stage_params = _resolve_stage_dict(parameters, stage_name)
        layer_cfg = optimization_dict[stage_name]

        # Sort blocks b1..bN deterministically
        available_block_names = sorted(
            (k for k in stage_params.keys() if k.startswith("b")),
            key=lambda s: int(s[1:]) if s[1:].isdigit() else 0,
        )

        if len(available_block_names) < blocks:
            raise KeyError(
                f"Requested {blocks} blocks for {stage_name}, but only found "
                f"{available_block_names}. Check that you passed the correct stage params."
            )

        layers: List[TTRegNetBottleneck] = []

        # First block (downsample if stride != 1)
        first_block_downsample = stride != 1
        layers.append(
            TTRegNetBottleneck(
                parameters=stage_params["b1"],
                model_config=model_config,
                layer_config=layer_cfg,
                stride=stride,
                downsample=first_block_downsample,
                groups=groups,
                torch_model=torch_model,
                use_fallback=use_fallback,
                block_name="b1",
                stage_name=stage_name,
            )
        )

        # Remaining blocks (stride=1, no downsample)
        for idx in range(2, blocks + 1):
            bname = f"b{idx}"
            if bname not in stage_params:
                raise KeyError(f"Missing block '{bname}' in {stage_name}. Available: {available_block_names}")
            layers.append(
                TTRegNetBottleneck(
                    parameters=stage_params[bname],
                    model_config=model_config,
                    layer_config=layer_cfg,
                    stride=1,
                    downsample=False,
                    groups=groups,
                    torch_model=torch_model,
                    use_fallback=use_fallback,
                    block_name=bname,
                    stage_name=stage_name,
                )
            )

        return layers

    def __call__(self, x, device, input_shape=None):
        shape = input_shape if input_shape is not None else x.shape
        for block in self.layer:
            x, shape = block(x, device, shape)
        return x, shape
