"""Keep an MCP client informed while a tool works, so silence is never mistaken for a wedge.

A client aborts a call that "sent no response or progress" for its silence window. Every tool in
this repo's MCP servers goes completely quiet for the whole of a device step -- a profile, a gate,
a full-pipeline measurement -- and those run far longer than that window while emitting nothing. So
the client cannot tell work from a hang, and it guesses hang.

Observed cost: perf-mcp's profile_model was aborted mid-run on every attempt of a run started from
an unoptimized baseline, where each profile takes several times longer than on an already-optimized
one. It persists the roofline snapshot on its LAST statement, so the snapshot was never written and
the report silently dropped its per-stage roofline and fidelity ladder, reporting no failure.

Raising the silence window only moves the guess further out; the client still cannot tell the two
apart. Reporting progress does, and costs nothing when a call is quick. Judging a wedge stays with
the run's own forward-progress watchdog, which samples output, syscalls and stack and kills a call
whose signature stops moving -- this only stops the client pre-empting that judgement.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import os

_DEFAULT_EVERY_S = 30
_ENV_EVERY_S = "MCP_PROGRESS_EVERY_S"


def _every_s() -> int:
    try:
        return max(1, int(os.environ.get(_ENV_EVERY_S) or _DEFAULT_EVERY_S))
    except ValueError:
        return _DEFAULT_EVERY_S


def _context_class():
    """The server's Context type, whichever mcp generation is installed."""
    try:
        from mcp.server.fastmcp import Context

        return Context
    except ModuleNotFoundError:
        try:
            from mcp.server.mcpserver import Context

            return Context
        except ModuleNotFoundError:
            return None


def keep_client_informed(fn, context_cls, every_s: int):
    """Wrap one synchronous tool so it reports progress while it runs.

    The body is unchanged and still synchronous: it runs in a worker thread while this coroutine
    reports on it, so a call is announced as alive exactly as long as it is actually running and
    stops the moment it returns. An already-async tool is left alone.
    """
    if context_cls is None or inspect.iscoroutinefunction(fn):
        return fn

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn

    @functools.wraps(fn)
    async def _alive(*args, ctx=None, **kwargs):
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
        ticks = 0
        while True:
            done, _pending = await asyncio.wait({work}, timeout=every_s)
            if done:
                return work.result()
            ticks += 1
            if ctx is None:
                continue
            try:
                # No total: how long a device step will take is not knowable in advance, and
                # inventing one would report a completion percentage that is a guess.
                await ctx.report_progress(
                    progress=float(ticks),
                    message="%s: running (%ds)" % (fn.__name__, ticks * every_s),
                )
            except Exception:  # noqa: BLE001 -- a client that cannot be told must not fail the call
                pass

    try:
        ctx_param = inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, annotation=context_cls)
        _alive.__signature__ = sig.replace(parameters=[*sig.parameters.values(), ctx_param])
        # The ANNOTATION, not only the signature: functools.wraps copies the wrapped function's
        # __annotations__, and the server reads that dict to recognise the context parameter and
        # keep it out of the tool's public schema. Without this the injected `ctx` is advertised to
        # the agent as an argument of every tool.
        _alive.__annotations__ = {**getattr(fn, "__annotations__", {}), "ctx": context_cls}
    except (TypeError, ValueError):  # noqa: BLE001 -- registration must survive an odd signature
        return fn
    return _alive


def install(mcp) -> bool:
    """Make every tool registered on `mcp` from now on report progress. Idempotent.

    Applied where tools are registered rather than tool by tool, so no signature changes and no
    server has to remember to do it for each new tool. Returns False when the installed mcp offers
    no Context to report through, leaving the server exactly as it was.
    """
    if getattr(mcp, "_progress_installed", False):
        return True
    context_cls = _context_class()
    if context_cls is None:
        return False
    raw_tool = mcp.tool
    every_s = _every_s()

    def _tool(*d_args, **d_kwargs):
        register = raw_tool(*d_args, **d_kwargs)

        def _apply(fn):
            # REGISTER THE WRAPPER, RETURN THE ORIGINAL. The decorated name stays bound to the plain
            # synchronous function, because the servers' own code calls these tools in process --
            # the deterministic stop gate invokes termination_check directly, and the suites call
            # them as ordinary functions. Returning the coroutine rebound every one of those names
            # and they got a coroutine object back instead of a result. The MCP server holds the
            # wrapper, which is the only caller that needs the progress reports.
            register(keep_client_informed(fn, context_cls, every_s))
            return fn

        return _apply

    mcp.tool = _tool
    mcp._progress_installed = True
    return True
