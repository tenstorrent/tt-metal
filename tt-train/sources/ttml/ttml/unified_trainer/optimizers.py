# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Several optimizers over disjoint weights, stepped as one."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .configs import OptimizerConfig
from .providers.base import UnsupportedTrainingConfig


def assign_parameters(names: Iterable[str], optimizers: Sequence[OptimizerConfig]) -> list[list[str]]:
    """The trainable parameter names each optimizer trains, in the order of ``optimizers``.

    A parameter goes to the first optimizer whose ``modules`` lists a ``/``-separated part of its name, else to the
    one optimizer without ``modules``. Raises if a parameter gets no optimizer, or an optimizer no parameter.
    """
    rest = next((index for index, optimizer in enumerate(optimizers) if optimizer.modules is None), None)
    assigned: list[list[str]] = [[] for _ in optimizers]
    untrained = []
    for name in names:
        parts = set(name.split("/"))
        index = next(
            (
                i
                for i, optimizer in enumerate(optimizers)
                if optimizer.modules and parts.intersection(optimizer.modules)
            ),
            rest,
        )
        if index is None:
            untrained.append(name)
        else:
            assigned[index].append(name)

    if untrained:
        shown = ", ".join(untrained[:3]) + (f" and {len(untrained) - 3} more" if len(untrained) > 3 else "")
        raise UnsupportedTrainingConfig(
            f"optimizer: no optimizer trains {shown}; list their modules, or leave one optimizer's modules unset to "
            "train the remaining weights"
        )
    for index, (optimizer, parameters) in enumerate(zip(optimizers, assigned)):
        if not parameters:
            if optimizer.modules:
                problem = f"modules {optimizer.modules} match no trainable weight"
            else:
                problem = "the other optimizers train every weight, so this one, without modules, trains none"
            raise UnsupportedTrainingConfig(f"optimizer[{index}]: {problem}")
    return assigned


class CombinedOptimizer:
    """The optimizers of a run, each over its own weights, stepped and zeroed together.

    ``get_group_lrs`` lists every optimizer's groups in order, so its first is the first optimizer's. The state dict
    nests each optimizer's under its index, as ``ttml.checkpointing`` saves and loads composite optimizers.
    """

    def __init__(self, optimizers: Sequence[Any]) -> None:
        self.optimizers = list(optimizers)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def get_group_lrs(self) -> list[float]:
        return [lr for optimizer in self.optimizers for lr in optimizer.get_group_lrs()]

    def get_state_dict(self) -> dict:
        return {str(index): optimizer.get_state_dict() for index, optimizer in enumerate(self.optimizers)}

    def set_state_dict(self, state: dict) -> None:
        for index, optimizer in enumerate(self.optimizers):
            optimizer.set_state_dict(state[str(index)])
