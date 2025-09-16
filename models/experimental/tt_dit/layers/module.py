# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping
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

    def _prepare_torch_state(self, state: MutableMapping[str, Any]) -> None:
        pass

    def _load_local_torch_state(self, state: MutableMapping[str, Any]) -> None:
        pass

    def load_torch_state_dict(self, state_dict: Mapping[str, torch.Tensor], *, warn: bool = True) -> dict[str, Any]:
        state = _unflatten_state_dict(state_dict)

        self._prepare_torch_state(state)
        self._load_local_torch_state(state)

        unexpected = {}

        for name, child in self.named_children():
            child_state = state.pop(name, {})

            if isinstance(child, Module):
                child_unexpected = child.load_torch_state_dict(child_state, warn=False)
                if child_unexpected:
                    unexpected[name] = child_unexpected
            else:  # legacy
                child.load_state_dict(_flatten_state_dict(child_state))

        unexpected.update(state)

        if warn and unexpected:
            for k in _flatten_state_dict(unexpected):
                logger.warning("unexpected torch state entry: {}", k)

        return unexpected

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.forward(*args, **kwargs)


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
        else:
            state_dict[child_key] = v

    return state_dict
