"""Engine — the dumb dispatcher that walks the handler return-graph (PLAN 8.1).

There is NO loop construct here. The engine reads `state.json.state`, calls that
state's handler, and the handler RETURNS the name of the next state. A cycle
(e.g. the REPAIR self-heal loop) is an emergent property of handlers pointing
back at each other — the engine just keeps dispatching until a terminal.

Crash-resume falls out for free: state is checkpointed after every transition,
so restarting re-reads the last state and continues exactly there.
"""

from __future__ import annotations

from typing import Callable

from . import states
from .loop_context import LoopContext

Handler = Callable[[LoopContext], str]


class EngineError(Exception):
    """Unregistered state, illegal transition, or a non-terminating cycle."""


def run(ctx: LoopContext, handlers: dict[str, Handler], max_steps: int = 10_000, stop_after=None, log=None) -> str:
    """Drive the state machine from `ctx.state['state']` to a terminal state.

    `max_steps` is an infinite-loop backstop only — real termination comes from
    CHECK_EXIT (target/budget/max-iter/floor). `stop_after` (a set of state names)
    parks the engine right after those stages run, for inspecting the pipeline
    partway. Returns the terminal state, or the parked state if stop_after fired.
    """
    stop_after = set(stop_after or ())
    state = ctx.state["state"]
    steps = 0
    while state not in states.TERMINAL:
        steps += 1
        if steps > max_steps:
            raise EngineError(f"exceeded {max_steps} steps at {state!r} — non-terminating cycle?")
        handler = handlers.get(state)
        if handler is None:
            raise EngineError(f"no handler registered for state {state!r}")

        ctx.log_event(state, "start")
        next_state = handler(ctx)
        if next_state not in handlers and next_state not in states.TERMINAL:
            raise EngineError(f"handler {state!r} returned unknown state {next_state!r}")

        ctx.state["state"] = next_state
        ctx.save()  # WAL: every transition is durable before we move on
        ctx.log_event(state, "done", f"-> {next_state}")
        if log:
            log(ctx, state, next_state)
        if state in stop_after:
            return next_state  # `state` was the last stage to run; parked at next_state
        state = next_state
    return state
