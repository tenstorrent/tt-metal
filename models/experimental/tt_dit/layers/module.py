# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

from ..utils.substate import pop_substate

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, MutableMapping
    from typing import Any

    import torch


class Module:
    def __init__(self) -> None:
        self._children = {}

    def named_children(self) -> Iterator[tuple[str, Any]]:
        yield from self._children.items()

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        super().__setattr__(name, value)

        children = self.__dict__.get("_children")

        if isinstance(value, Module) or hasattr(value, "load_state_dict"):
            if children is None:
                msg = "cannot assign child module before Module.__init__() call"
                raise AttributeError(msg)
            self._children[name] = value
        elif children is not None:
            children.pop(name, None)

    def __delattr__(self, name: str) -> None:
        children = self.__dict__.get("_children")
        if children is not None:
            children.pop(name, None)

        super().__delattr__(name)

    def _prepare_torch_state_dict(self, state_dict: MutableMapping[str, torch.Tensor]) -> None:
        pass

    def _load_local_torch_state_dict(self, state_dict: MutableMapping[str, torch.Tensor]) -> None:
        pass

    def load_torch_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, warn: bool = True) -> list[str]:
        state_dict = dict(state_dict)

        self._prepare_torch_state_dict(state_dict)
        self._load_local_torch_state_dict(state_dict)

        unexpected_keys = []

        for name, child in self.named_children():
            child_state_dict = pop_substate(state_dict, name)

            if isinstance(child, Module):
                child_unexpected_keys = child.load_torch_state_dict(child_state_dict, warn=False)
                unexpected_keys.extend(f"{name}.{k}" for k in child_unexpected_keys)
            else:  # legacy
                child.load_state_dict(child_state_dict)

        unexpected_keys.extend(state_dict.keys())

        if warn:
            for k in unexpected_keys:
                logger.warning("unexpected torch state key: {}", k)

        return unexpected_keys

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.forward(*args, **kwargs)
