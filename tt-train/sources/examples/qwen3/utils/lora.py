# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA (Low-Rank Adaptation) utilities for Qwen3 fine-tuning.

PEFT-style mixin pattern (mirrors HuggingFace PEFT BaseTunerLayer):

  inject_adapter_in_model(model, lora_config)

traverses the module tree and replaces every attribute whose name appears
in lora_config["targets"] with an appropriate LoRA wrapper:

  LinearProjection       → LoRALinearProjection       (single-device)
  ColumnParallelLinear   → LoRAColumnParallelLinear    (tensor-parallel)
  RowParallelLinear      → LoRARowParallelLinear       (tensor-parallel)

Each wrapper stores the original layer as ``self.base_layer`` and
computes:  out = base_layer(x) + lora_B(lora_A(x)) * scaling

Call AFTER loading pretrained weights:
  load_weights_from_hf(model, state_dict, config)
  inject_adapter_in_model(model, lora_config)

All LoRA weights are always explicitly initialized regardless of the
global empty_init context, because they have no corresponding pretrained
weights to load.
"""

import math

import torch
import ttml
from ttml.modules import AbstractModuleBase, Parameter

from .tensor_utils import (
    torch_to_ttml,
    get_tp_size,
    make_dist_replicated_weight,
    make_dist_replicated_zeros,
    make_dist_sharded_weight,
    make_dist_sharded_zeros,
)


LORA_TARGETS_ALL = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# =====================================================================
# PEFT-style LoRA wrapper modules
# =====================================================================


class LoRALinearProjection(AbstractModuleBase):
    """LoRA wrapper for single-device LinearProjection.

    Forward: base_layer(x) + lora_B(lora_A(x)) * scaling
    Dimensions inferred from base_layer.weight shape (1, 1, out, in).
    """

    def __init__(self, base_layer, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        w_shape = base_layer.weight.tensor.shape()
        in_features = w_shape[-1]
        out_features = w_shape[-2]
        self.scaling = alpha / rank
        self.lora_A = Parameter(
            torch_to_ttml(
                torch.randn(1, 1, rank, in_features) * (1.0 / math.sqrt(rank))
            )
        )
        self.lora_B = Parameter(torch_to_ttml(torch.zeros(1, 1, out_features, rank)))

    def forward(self, x):
        base_out = self.base_layer(x)
        h = ttml.ops.linear.linear(x, self.lora_A.tensor, None)
        lora_out = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.scaling != 1.0:
            lora_out = lora_out * self.scaling
        return ttml.ops.binary.add(base_out, lora_out)


class LoRAColumnParallelLinear(AbstractModuleBase):
    """LoRA wrapper for ColumnParallelLinear (tensor-parallel).

    lora_A: replicated  (1, 1, rank, in_features)
    lora_B: column-sharded on dim 2  (1, 1, out_features/tp, rank) per device

    A single broadcast is performed for both the base and LoRA paths.
    Dimensions inferred from base_layer.weight per-device shape.
    """

    def __init__(self, base_layer, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        self.shard_dim = base_layer.shard_dim
        w_shape = base_layer.weight.tensor.shape()
        in_features = w_shape[-1]
        out_features = w_shape[-2] * get_tp_size(self.shard_dim)
        self.scaling = alpha / rank
        self.lora_A = Parameter(
            make_dist_replicated_weight(
                (1, 1, rank, in_features), std=1.0 / math.sqrt(rank)
            )
        )
        self.lora_B = Parameter(
            make_dist_sharded_zeros((1, 1, out_features, rank), 2, self.shard_dim)
        )

    def forward(self, x):
        x_b = ttml.ops.distributed.broadcast(x, self.shard_dim)
        bias_t = (
            self.base_layer.col_bias.tensor
            if self.base_layer.col_bias is not None
            else None
        )
        base_out = ttml.ops.linear.linear(x_b, self.base_layer.weight.tensor, bias_t)
        if self.base_layer.gather_output:
            base_out = ttml.ops.distributed.all_gather(
                base_out,
                3,
                self.shard_dim,
                ttml.ops.distributed.GradOutputType.REPLICATED,
            )
        h = ttml.ops.linear.linear(x_b, self.lora_A.tensor, None)
        lora_out = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.scaling != 1.0:
            lora_out = lora_out * self.scaling
        return ttml.ops.binary.add(base_out, lora_out)


class LoRARowParallelLinear(AbstractModuleBase):
    """LoRA wrapper for RowParallelLinear (tensor-parallel, input_is_parallel=True).

    lora_A: row-sharded on dim 3  (1, 1, rank, in_features/tp) per device
    lora_B: replicated  (1, 1, out_features, rank)

    Input is already scattered. Partial lora_A products are all-reduced before lora_B.
    Dimensions inferred from base_layer.weight per-device shape.
    """

    def __init__(self, base_layer, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        self.shard_dim = base_layer.shard_dim
        w_shape = base_layer.weight.tensor.shape()
        out_features = w_shape[-2]
        in_features = w_shape[-1] * get_tp_size(self.shard_dim)
        self.scaling = alpha / rank
        self.lora_A = Parameter(
            make_dist_sharded_weight(
                (1, 1, rank, in_features), 3, self.shard_dim, std=1.0 / math.sqrt(rank)
            )
        )
        self.lora_B = Parameter(make_dist_replicated_zeros((1, 1, out_features, rank)))

    def forward(self, x):
        base_out = self.base_layer(x)
        h = ttml.ops.linear.linear(x, self.lora_A.tensor, None)
        h = ttml.ops.distributed.all_reduce(h, True, self.shard_dim)
        lora_out = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.scaling != 1.0:
            lora_out = lora_out * self.scaling
        return ttml.ops.binary.add(base_out, lora_out)


# =====================================================================
# PEFT-style adapter injection
# =====================================================================


def _make_lora_wrapper(module, rank: int, alpha: float):
    """Create the appropriate LoRA wrapper for the given base module."""
    cls_name = type(module).__name__
    if cls_name == "LinearProjection":
        return LoRALinearProjection(module, rank, alpha)
    elif cls_name == "ColumnParallelLinear":
        return LoRAColumnParallelLinear(module, rank, alpha)
    elif cls_name == "RowParallelLinear":
        return LoRARowParallelLinear(module, rank, alpha)
    else:
        raise ValueError(
            f"Cannot inject LoRA into module type '{cls_name}'. "
            f"Supported: LinearProjection, ColumnParallelLinear, RowParallelLinear"
        )


def inject_adapter_in_model(model, lora_config: dict):
    """Replace target linear layers with LoRA wrappers (PEFT mixin pattern).

    Traverses the model tree and replaces every sub-module whose attribute
    name appears in ``lora_config["targets"]`` with an appropriate LoRA
    wrapper.  The wrapper stores the original layer as ``self.base_layer``
    and computes::

        out = base_layer(x) + lora_B(lora_A(x)) * scaling

    Must be called AFTER loading pretrained weights.  LoRA parameters are
    initialized fresh (A ~ N(0, 1/r), B = 0) so the initial LoRA
    contribution is exactly zero — identical starting point to the base model.

    Trainable parameters can be selected by filtering for ``"lora"`` in
    the parameter path returned by ``model.parameters()``.

    Args:
        model:       ttml model (Qwen3ForCausalLM or DistributedQwen3ForCausalLM).
        lora_config: dict with keys:
                       ``targets`` – list of module names, e.g. ``["q_proj", "v_proj"]``
                       ``rank``    – LoRA rank *r*
                       ``alpha``   – LoRA scaling alpha (default: rank → scaling = 1.0)

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
