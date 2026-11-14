# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from .euler import EulerSolver
from .unipc import UniPCSolver, UniPCVariant

if TYPE_CHECKING:
    from .base import Solver


class CustomSigmaScheduler:
    """Stand-in for a scheduler, for a pipeline that supplies its own sigmas.
    The sigmas are taken exactly as given, so any flow shift belongs to whatever builds them.
    Only Euler can be driven this way; UniPC needs its scheduler for the shift and its multistep bookkeeping.
    """


def solver_for_scheduler(scheduler: Any) -> Solver:
    """Return the on-device solver matching `scheduler`, bound to it.

    The scheduler determines the solver family, so a pipeline picks its solver by
    choosing a scheduler rather than by constructing one directly.
    """
    if isinstance(scheduler, CustomSigmaScheduler):
        # Resolved to a scheduler-less solver, so the marker never outlives this call.
        return EulerSolver()

    if isinstance(scheduler, UniPCMultistepScheduler):
        if not scheduler.config.use_flow_sigmas:
            msg = "UniPCSolver requires a scheduler configured with use_flow_sigmas=True"
            raise ValueError(msg)
        _check_unipc_config(scheduler.config)
        return UniPCSolver(
            order=scheduler.config.solver_order,
            variant=UniPCVariant(scheduler.config.solver_type),
            scheduler=scheduler,
        )

    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        if scheduler.config.get("stochastic_sampling", False):
            msg = "EulerSolver does not support stochastic_sampling=True"
            raise ValueError(msg)
        return EulerSolver(scheduler=scheduler)

    msg = f"no solver available for {type(scheduler).__name__}"
    raise ValueError(msg)


def _check_unipc_config(config: Any) -> None:
    """Reject UniPC options that UniPCSolver does not implement, instead of silently ignoring them."""
    # UniPCSolver always tapers the order at the end, always corrects, and predicts x0 from a flow velocity.
    unsupported = {
        "prediction_type": (config.prediction_type, "flow_prediction"),
        "predict_x0": (config.predict_x0, True),
        "lower_order_final": (config.lower_order_final, True),
        "disable_corrector": (list(config.disable_corrector), []),
        "thresholding": (config.thresholding, False),
        "solver_p": (config.solver_p, None),
    }
    for name, (value, expected) in unsupported.items():
        if value != expected:
            msg = f"UniPCSolver requires a scheduler configured with {name}={expected!r}, got {value!r}"
            raise ValueError(msg)
