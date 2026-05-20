# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA (Low-Rank Adaptation) utilities for Flux1 DiT fine-tuning.

Replaces target linear layers in DistributedFlux1Transformer with LoRA
wrappers that compute:  out = base_layer(x) + lora_B(lora_A(x)) * scaling

Supported base layer types:
  ColumnParallelLinear  → LoRAColumnParallelLinear
  RowParallelLinear     → LoRARowParallelLinear
  ReplicatedLinear      → LoRAReplicatedLinear

Call AFTER loading pretrained weights:
  load_weights_from_hf_distributed(model, state_dict, config, ...)
  inject_lora(model, lora_config)
"""

import math

import torch
import ttml
from ttml.modules import AbstractModuleBase, Parameter

from model_flux_distributed import (
    _make_replicated,
    _make_replicated_zeros,
    _make_sharded_zeros,
    _make_sharded_weight,
    _get_tp_size,
)

LORA_TARGETS_DOUBLE = [
    "to_qkv",
    "to_out",
    "add_qkv_proj",
    "to_add_out",
    "ff1",
    "ff2",
    "norm1_linear",
    "norm1_context_linear",
]

LORA_TARGETS_SINGLE = [
    "to_qkv",
    "proj_mlp",
    "proj_out",
    "time_embed",
]

LORA_TARGETS_ALL = sorted(set(LORA_TARGETS_DOUBLE + LORA_TARGETS_SINGLE))


def _make_replicated_weight_init(shape, std):
    data = (torch.randn(shape) * std).float().numpy()
    return _make_replicated(data)


class LoRAReplicatedLinear(AbstractModuleBase):
    """LoRA wrapper for ReplicatedLinear."""

    def __init__(self, base_layer, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        w_shape = base_layer.weight.tensor.shape()
        in_features = w_shape[-1]
        out_features = w_shape[-2]
        self.scaling = alpha / rank
        self.lora_A = Parameter(
            _make_replicated_weight_init((1, 1, rank, in_features), std=1.0 / math.sqrt(rank))
        )
        self.lora_B = Parameter(_make_replicated_zeros((1, 1, out_features, rank)))

    def forward(self, x):
        base_out = self.base_layer(x)
        h = ttml.ops.linear.linear(x, self.lora_A.tensor, None)
        lora_out = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.scaling != 1.0:
            lora_out = lora_out * self.scaling
        return ttml.ops.binary.add(base_out, lora_out)


class LoRAColumnParallelLinear(AbstractModuleBase):
    """LoRA wrapper for ColumnParallelLinear.

    lora_A: replicated  (1, 1, rank, in_features)
    lora_B: column-sharded on dim 2  (1, 1, out_features/tp, rank) per device
    """

    def __init__(self, base_layer, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        self.shard_dim = base_layer.shard_dim
        w_shape = base_layer.weight.tensor.shape()
        in_features = w_shape[-1]
        out_features = w_shape[-2] * _get_tp_size()
        self.scaling = alpha / rank
        self.lora_A = Parameter(
            _make_replicated_weight_init((1, 1, rank, in_features), std=1.0 / math.sqrt(rank))
        )
        self.lora_B = Parameter(_make_sharded_zeros((1, 1, out_features, rank), 2, self.shard_dim))

    def forward(self, x):
        x_b = ttml.ops.distributed.broadcast(x, self.shard_dim)
        bias_t = self.base_layer.col_bias.tensor if self.base_layer.col_bias is not None else None
        base_out = ttml.ops.linear.linear(x_b, self.base_layer.weight.tensor, bias_t)
        h = ttml.ops.linear.linear(x_b, self.lora_A.tensor, None)
        lora_out = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.scaling != 1.0:
            lora_out = lora_out * self.scaling
        combined = ttml.ops.binary.add(base_out, lora_out)
        if self.base_layer.gather_output:
            combined = ttml.ops.distributed.all_gather(
                combined, 3, self.shard_dim, ttml.ops.distributed.GradOutputType.REPLICATED
            )
        return combined


class LoRARowParallelLinear(AbstractModuleBase):
    """LoRA wrapper for RowParallelLinear (input_is_parallel=True).

    lora_A: row-sharded on dim 3  (1, 1, rank, in_features/tp) per device
    lora_B: replicated  (1, 1, out_features, rank)
    """

    def __init__(self, base_layer, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        self.shard_dim = base_layer.shard_dim
        w_shape = base_layer.weight.tensor.shape()
        out_features = w_shape[-2]
        in_features = w_shape[-1] * _get_tp_size()
        self.scaling = alpha / rank
        self.lora_A = Parameter(
            _make_sharded_weight((1, 1, rank, in_features), 3, self.shard_dim, std=1.0 / math.sqrt(rank))
        )
        self.lora_B = Parameter(_make_replicated_zeros((1, 1, out_features, rank)))

    def forward(self, x):
        base_out = self.base_layer(x)
        h = ttml.ops.linear.linear(x, self.lora_A.tensor, None)
        h = ttml.ops.distributed.all_reduce(h, True, self.shard_dim)
        lora_out = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.scaling != 1.0:
            lora_out = lora_out * self.scaling
        return ttml.ops.binary.add(base_out, lora_out)


def _make_lora_wrapper(module, rank: int, alpha: float):
    cls_name = type(module).__name__
    if cls_name == "ReplicatedLinear":
        return LoRAReplicatedLinear(module, rank, alpha)
    elif cls_name == "ColumnParallelLinear":
        return LoRAColumnParallelLinear(module, rank, alpha)
    elif cls_name == "RowParallelLinear":
        return LoRARowParallelLinear(module, rank, alpha)
    else:
        raise ValueError(
            f"Cannot inject LoRA into module type '{cls_name}'. "
            f"Supported: ReplicatedLinear, ColumnParallelLinear, RowParallelLinear"
        )


def inject_lora(model, lora_config: dict):
    """Replace target linear layers with LoRA wrappers.

    Traverses the model tree and replaces every sub-module whose attribute
    name appears in ``lora_config["targets"]`` with an appropriate LoRA
    wrapper.  Must be called AFTER loading pretrained weights.

    Args:
        model:       DistributedFlux1Transformer instance.
        lora_config: dict with keys:
                       ``targets`` – list of module names, e.g. ``["to_qkv", "to_out"]``
                       ``rank``    – LoRA rank r
                       ``alpha``   – LoRA scaling alpha
    Returns:
        The model, modified in-place.
    """
    from ttml.modules import Parameter as TtmlParameter, ModuleList

    targets = set(lora_config.get("targets", []))
    rank = lora_config["rank"]
    alpha = lora_config.get("alpha", rank)

    def _inject(module):
        module_vars = vars(module) if hasattr(module, "__dict__") else {}
        for name, child in list(module_vars.items()):
            if name.startswith("_") or isinstance(child, TtmlParameter):
                continue
            if isinstance(child, ModuleList):
                for i in range(len(child)):
                    _inject(child[i])
            elif hasattr(child, "forward"):
                if name in targets:
                    setattr(module, name, _make_lora_wrapper(child, rank, alpha))
                else:
                    _inject(child)

    _inject(model)
    return model
