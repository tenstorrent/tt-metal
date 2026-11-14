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
        config = scheduler.config
        if not config.use_flow_sigmas:
            msg = "UniPCSolver requires a scheduler configured with use_flow_sigmas=True"
            raise ValueError(msg)
        if config.prediction_type != "flow_prediction":
            msg = (
                f"UniPCSolver only supports prediction_type='flow_prediction', got '{config.prediction_type}'"
            )
            raise ValueError(msg)
        if config.disable_corrector:
            msg = "UniPCSolver does not support disable_corrector; the corrector always runs"
            raise ValueError(msg)
        if not config.lower_order_final:
            msg = "UniPCSolver does not support lower_order_final=False; order is always tapered"
            raise ValueError(msg)
        if config.thresholding:
            msg = "UniPCSolver does not support thresholding"
            raise ValueError(msg)
        if not config.predict_x0:
            msg = "UniPCSolver does not support predict_x0=False"
            raise ValueError(msg)
        if config.solver_p is not None:
            msg = "UniPCSolver does not support solver_p"
            raise ValueError(msg)
        return UniPCSolver(
            order=config.solver_order,
            variant=UniPCVariant(config.solver_type),
            scheduler=scheduler,
        )

    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return EulerSolver(scheduler=scheduler)

    msg = f"no solver available for {type(scheduler).__name__}"
    raise ValueError(msg)
