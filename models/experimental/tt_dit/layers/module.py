# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import torch
import ttnn

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, MutableMapping
    from typing import Any


class IncompatibleKeys(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]


class Module:
    def __init__(self) -> None:
        self._children = {}
        self._parameters = {}

    def named_children(self) -> Iterator[tuple[str, Any]]:
        yield from self._children.items()

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        yield from self._parameters.items()

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        super().__setattr__(name, value)

        if name in ("_children", "_parameters"):
            return

        children = self.__dict__.get("_children")
        parameters = self.__dict__.get("_parameters")

        if isinstance(value, Module) or hasattr(value, "load_state_dict"):
            if children is None:
                msg = "cannot assign child module before Module.__init__() call"
                raise AttributeError(msg)
            self._children[name] = value
        elif isinstance(value, Parameter):
            if parameters is None:
                msg = "cannot assign parameter before Module.__init__() call"
                raise AttributeError(msg)
            self._parameters[name] = value
        elif children is not None:
            children.pop(name, None)
            parameters.pop(name, None)

    def __delattr__(self, name: str) -> None:
        children = self.__dict__.get("_children")
        parameters = self.__dict__.get("_parameters")

        if children is not None:
            children.pop(name, None)
        if parameters is not None:
            parameters.pop(name, None)

        super().__delattr__(name)

    def _prepare_torch_state(self, state: MutableMapping[str, Any]) -> None:
        """Prepare Torch state before loading.

        This method is meant to be overridden by the inheriting class to modify the Torch state
        before loading. The `state` argument is not a PyTorch state dict but a hierarchically nested
        dict of dicts containing the same data.
        """

    def load_torch_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True) -> IncompatibleKeys:
        state = _unflatten_state_dict(state_dict)

        self._prepare_torch_state(state)

        unexpected_keys = []
        missing_keys = []

        for name, child in self.named_children():
            child_state = state.pop(name, {})

            if isinstance(child, Module):
                child_missing, child_unexpected = child.load_torch_state_dict(child_state, strict=False)
                missing_keys.extend(f"{name}.{k}" for k in child_missing)
                unexpected_keys.extend(f"{name}.{k}" for k in child_unexpected)
            else:  # legacy
                child.load_state_dict(_flatten_state_dict(child_state))

        for name, parameter in self.named_parameters():
            if name in state:
                tensor = state.pop(name)
                assert isinstance(tensor, torch.Tensor)
                parameter.load_from_torch(tensor)
            else:
                missing_keys.append(name)

        unexpected_keys.extend(_flatten_state_dict(state).keys())

        error_msg = ""
        if strict and missing_keys:
            error_msg += "missing Torch state keys: " + ", ".join(missing_keys) + "; "
        if strict and unexpected_keys:
            error_msg += "unexpected Torch state keys: " + ", ".join(unexpected_keys) + "\n"
        if error_msg:
            raise ValueError(error_msg)

        return IncompatibleKeys(missing_keys, unexpected_keys)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.forward(*args, **kwargs)


class Parameter:
    def __init__(
        self,
        *,
        shape: Collection[int],
        device: ttnn.MeshDevice,
        layout: ttnn.Layout = ttnn.Layout.TILE,
        dtype: ttnn.DataType = ttnn.bfloat16,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        mesh_placements: Collection[ttnn.PlacementReplicate | ttnn.PlacementShard] | None = None,
        mesh_mapping: Mapping[int, int] | None = None,
        to_host: bool = False,
        init: torch.Tensor | bool = False,
    ) -> None:
        assert mesh_mapping is None or mesh_placements is None, (
            "only one of mesh_mapping and mesh_placement should be supplied"
        )

        mesh_rank = len(list(device.shape))

        if mesh_placements is None:
            mesh_placements = _mesh_placements_from_mapping(mesh_mapping or {}, mesh_rank=mesh_rank)
        else:
            assert len(mesh_placements) == mesh_rank

        self.shape = tuple(shape)
        self.device = device
        self.layout = layout
        self.dtype = dtype
        self.memory_config = memory_config
        self.mesh_placements = tuple(mesh_placements)
        self.to_host = to_host
        self.data = None

        if isinstance(init, torch.Tensor):
            self.load_from_torch(init)
        elif init is True:
            self.load_from_torch(torch.randn(self.shape))
        elif init is not False:
            msg = "init should be a Torch tensor or bool"
            raise ValueError(msg)

    def load_from_torch(self, torch_tensor: torch.Tensor, /) -> None:
        assert torch_tensor.shape == self.shape

        self.data = ttnn.from_torch(
            torch_tensor,
            layout=self.layout,
            dtype=self.dtype,
            memory_config=self.memory_config,
            device=None if self.to_host else self.device,
            mesh_mapper=ttnn.create_mesh_mapper(
                self.device,
                ttnn.MeshMapperConfig(self.mesh_placements),
            ),
        )


def _unflatten_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Turns a PyTorch state dict into a nested dict of dicts."""
    root = {}

    for key, value in state_dict.items():
        *parts, leaf = key.split(".")

        node = root
        for p in parts:
            node = node.setdefault(p, {})

        node[leaf] = value

    return root


def _flatten_state_dict(nested: Mapping[str, Any], *, prefix: str = "") -> dict[str, torch.Tensor]:
    """Turns a nested dict of dicts back into a PyTorch state dict."""
    state_dict = {}

    for k, v in nested.items():
        child_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            state_dict.update(_flatten_state_dict(v, prefix=child_key))
        elif v is not None:
            state_dict[child_key] = v

    return state_dict


def _mesh_placements_from_mapping(
    mapping: Mapping[int, int], *, mesh_rank: int
) -> list[ttnn.PlacementReplicate | ttnn.PlacementShard]:
    placements = [ttnn.PlacementReplicate()] * mesh_rank

    for k, v in mapping.items():
        if k is None:
            assert False, "success"
            continue
        assert k < mesh_rank
        placements[k] = ttnn.PlacementShard(v)

    return placements
