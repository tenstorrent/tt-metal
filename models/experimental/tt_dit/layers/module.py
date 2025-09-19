# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import torch
import ttnn

from ..utils.substate import pop_substate

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping
    from typing import Any


class IncompatibleKeys(NamedTuple):
    missing_keys: list[str]
    unexpected_keys: list[str]


class Module:
    def __init__(self) -> None:
        self._children = {}
        self._parameters = {}

    # TODO: change "Any" to "Module" as soon as all modules are migrated
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
        else:
            if children is not None:
                children.pop(name, None)
            if parameters is not None:
                parameters.pop(name, None)

    def __delattr__(self, name: str) -> None:
        children = self.__dict__.get("_children")
        parameters = self.__dict__.get("_parameters")

        if children is not None:
            children.pop(name, None)
        if parameters is not None:
            parameters.pop(name, None)

        super().__delattr__(name)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Prepare Torch state dict before loading."""

    def load_torch_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True) -> IncompatibleKeys:
        state_dict = dict(state_dict)
        self._prepare_torch_state(state_dict)

        unexpected_keys = []
        missing_keys = []

        for name, child in self.named_children():
            child_state = pop_substate(state_dict, name)

            if isinstance(child, Module):
                child_missing, child_unexpected = child.load_torch_state_dict(child_state, strict=False)
                missing_keys.extend(f"{name}.{k}" for k in child_missing)
                unexpected_keys.extend(f"{name}.{k}" for k in child_unexpected)
            else:  # legacy
                child.load_state_dict(child_state)

        for name, parameter in self.named_parameters():
            if name in state_dict:
                parameter.load_torch_tensor(state_dict.pop(name))
            else:
                missing_keys.append(name)

        unexpected_keys.extend(state_dict.keys())

        error_msg = ""
        if strict and missing_keys:
            error_msg += "missing Torch state keys: " + ", ".join(missing_keys) + "; "
        if strict and unexpected_keys:
            error_msg += "unexpected Torch state keys: " + ", ".join(unexpected_keys) + "\n"
        if error_msg:
            raise ValueError(error_msg)

        return IncompatibleKeys(missing_keys, unexpected_keys)

    def save_to_cache(self, path_prefix: str) -> None:
        for name, child in self.named_children():
            child.save_to_cache(f"{path_prefix}{name}.")

        for name, parameter in self.named_parameters():
            ttnn.dump_tensor(f"{path_prefix}{name}.ext", parameter.data)

    def load_from_cache(self, path_prefix: str) -> None:
        for name, child in self.named_children():
            child.load_from_cache(f"{path_prefix}{name}.")

        for name, parameter in self.named_parameters():
            parameter.data = ttnn.load_tensor(f"{path_prefix}{name}.ext")

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
            self.load_torch_tensor(init)
        elif init is True:
            self.load_torch_tensor(torch.randn(self.shape))
        elif init is not False:
            msg = "init should be a Torch tensor or bool"
            raise ValueError(msg)

    def load_torch_tensor(self, torch_tensor: torch.Tensor, /) -> None:
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


def _mesh_placements_from_mapping(
    mapping: Mapping[int, int], *, mesh_rank: int
) -> list[ttnn.PlacementReplicate | ttnn.PlacementShard]:
    placements = [ttnn.PlacementReplicate()] * mesh_rank

    for k, v in mapping.items():
        if k is None:
            continue
        assert k < mesh_rank
        placements[k] = ttnn.PlacementShard(v)

    return placements
