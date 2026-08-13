# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tt_dit-Module wrapper around tt-symbiote `TTNNLinearMeshShard`.

The Cosmos3 trunk is wrapped via tt-symbiote (Phase 1 MVP): each `nn.Linear`
is replaced with a `TTNNLinearMeshShard`. Those modules are NOT tt_dit
`Module`s — they don't expose `Parameter`s, so `tt_dit.utils.cache.load_model`
can't act on them.

This file adds the bridge. `CosmosLinearBank` is a tt_dit `Module` whose
children are `CachedLinearWeights` (one per `TTNNLinearMeshShard`), each
holding the sharded weight + bias as tt_dit `Parameter`s. After
`cache.load_model` populates the parameters (from disk on a warm hit, or from
the wrapped `nn.Linear` state dict on a miss), `sync_to_tt_modules()` patches
the `tt_weight`/`tt_bias` slots on the original tt-symbiote modules so their
existing forward path picks the cached tensors up. We also flip the
`_preprocessed_weight`/`_weights_on_device` flags so the manual
`preprocess_weights()` + `move_weights_to_device()` pump is skipped.

Cache keying piggybacks on `tt_dit/utils/cache.py`. The faked `parallel_config`
NamedTuple supplies the (factor, mesh_axis) tuple `config_id` expects.
"""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import ttnn
from models.tt_dit.layers.module import Module, Parameter

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from models.tt_dit.experimental.cosmos3_i2v.tt_modules.linear_mesh_shard import TTNNLinearMeshShard


# `cache.config_id` expects each non-None field to expose `.factor` and `.mesh_axis`.
_ParallelAxis = namedtuple("_ParallelAxis", ["factor", "mesh_axis"])
_CosmosParallel = namedtuple("_CosmosParallel", ["linear_shard"])


def build_parallel_config(mesh_shape: tuple[int, int]) -> _CosmosParallel:
    """Cache-key descriptor for the sharded-Linear placement on a 2D mesh.

    Cosmos3 currently shards every Linear's out-feature dim across mesh axis 1
    (see `TTNNLinearMeshShard`). The descriptor only feeds `cache.config_id`,
    so it encodes the choice as a single axis entry: factor = mesh axis 1
    size, mesh_axis = 1. If the placement strategy ever changes (e.g. 2D
    sharding) extend the NamedTuple here.
    """
    return _CosmosParallel(linear_shard=_ParallelAxis(factor=mesh_shape[1], mesh_axis=1))


def _dtype_key(weight_dtype: ttnn.DataType) -> str:
    """Short string for `cache.model_cache_dir`'s dtype slot."""
    name = str(weight_dtype).rsplit(".", 1)[-1].lower()
    return {"bfloat16": "bf16", "bfloat8_b": "bfp8", "bfloat4_b": "bfp4"}.get(name, name)


class CachedLinearWeights(Module):
    """Weight + bias for one `TTNNLinearMeshShard`, exposed as tt_dit `Parameter`s.

    The on-device layout matches what `TTNNLinearMeshShard.move_weights_to_device_impl`
    produces:

      - weight: shape `(in_features, out_features)`, sharded along axis 1
        (out-feature) across mesh axis 1, replicated along mesh axis 0.
      - bias  : shape `(1, out_features)`, same sharding. Bias stays at
        bfloat16 even when the weight is BFP8 — biases are tiny and benefit
        less from quantization (matches the tt-symbiote module's own choice).

    `_prepare_torch_state` transposes the incoming `weight` from torch's
    `(out, in)` layout to `(in, out)` before `load_torch_tensor` shapes-checks
    it. Bias is reshaped from `(out,)` to `(1, out)` for the same reason.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        has_bias: bool,
        device: ttnn.MeshDevice,
        weight_dtype: ttnn.DataType,
    ) -> None:
        super().__init__()
        self._tt_module: TTNNLinearMeshShard | None = None
        self.weight = Parameter(
            total_shape=(in_features, out_features),
            device=device,
            dtype=weight_dtype,
            mesh_axes=(None, 1),
        )
        if has_bias:
            self.bias = Parameter(
                total_shape=(1, out_features),
                device=device,
                dtype=ttnn.bfloat16,
                mesh_axes=(None, 1),
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.get("weight")
        if weight is not None and weight.dim() == 2:
            state["weight"] = weight.t().contiguous()
        bias = state.get("bias")
        if bias is not None and bias.dim() == 1:
            state["bias"] = bias.reshape(1, -1)

    def attach(self, tt_module: TTNNLinearMeshShard) -> None:
        """Remember which tt-symbiote module's `tt_weight/tt_bias` we should populate."""
        self._tt_module = tt_module

    def sync(self) -> None:
        """Push loaded `Parameter.data` into the tt-symbiote module's slots.

        Also flips the tt-symbiote lifecycle flags so a subsequent
        `preprocess_weights()` + `move_weights_to_device()` call is a no-op
        (the base class short-circuits on `_preprocessed_weight=True` /
        `_weights_on_device=True`).
        """
        if self._tt_module is None:
            return
        mod = self._tt_module
        mod.tt_weight = self.weight.data
        mod.tt_bias = self.bias.data if hasattr(self, "bias") else None
        mod._preprocessed_weight = True
        mod._weights_on_device = True

    def forward(self) -> None:
        msg = "CachedLinearWeights is a parameter container; call the wrapped TTNNLinearMeshShard instead"
        raise RuntimeError(msg)


class CosmosLinearBank(Module):
    """Container Module: one `CachedLinearWeights` child per replaced Linear.

    Child names are the tt-symbiote `module_name` of each linear (e.g.
    `layers.0.attention.to_q`). tt_dit's `add_module` is a dict insert, so
    dots in the name are fine — `pop_substate` does prefix-string matching,
    not dotted-path traversal.

    The forward method is not callable; this module exists only to delegate
    save/load to `cache.load_model`.
    """

    def __init__(
        self,
        all_modules: Mapping[str, TTNNLinearMeshShard],
        *,
        device: ttnn.MeshDevice,
        weight_dtype: ttnn.DataType,
        linear_cls: type,
    ) -> None:
        super().__init__()
        self._all_modules = all_modules

        for name, mod in all_modules.items():
            if not isinstance(mod, linear_cls):
                continue
            # `from_torch` made torch_layer point at the original nn.Linear; we
            # need it to know whether bias was present and to source weights on
            # a cache miss.
            torch_layer = mod.torch_layer
            has_bias = torch_layer is not None and torch_layer.bias is not None
            child = CachedLinearWeights(
                in_features=mod.in_features,
                out_features=mod.out_features,
                has_bias=has_bias,
                device=device,
                weight_dtype=weight_dtype,
            )
            child.attach(mod)
            self.add_module(name, child)

    def torch_state_dict(self) -> dict[str, torch.Tensor]:
        """Build a flat state dict from the wrapped `nn.Linear` weights.

        Used on a cache miss: `cache.load_model` calls this to populate the
        bank, then writes the resulting on-device tensors back to disk.
        """
        sd: dict[str, torch.Tensor] = {}
        for name, child in self.named_children():
            mod = child._tt_module
            if mod is None or mod.torch_layer is None:
                continue
            sd[f"{name}.weight"] = mod.torch_layer.weight
            if mod.torch_layer.bias is not None:
                sd[f"{name}.bias"] = mod.torch_layer.bias
        return sd

    def sync_to_tt_modules(self) -> None:
        """After load (cache hit or miss), patch every linked tt-symbiote module."""
        for _, child in self.named_children():
            child.sync()

    def forward(self) -> None:
        msg = "CosmosLinearBank is a cache container; it is not callable"
        raise RuntimeError(msg)
