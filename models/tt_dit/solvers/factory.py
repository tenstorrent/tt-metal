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
        return UniPCSolver(
            order=scheduler.config.solver_order,
            variant=UniPCVariant(scheduler.config.solver_type),
            scheduler=scheduler,
        )

    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return EulerSolver(scheduler=scheduler)

    msg = f"no solver available for {type(scheduler).__name__}"
    raise ValueError(msg)
