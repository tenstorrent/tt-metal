"""An MCP client must be able to tell a working tool from a hung one.

A client aborts a call that "sent no response or progress" for its silence window. Every tool in
these servers went completely quiet for the whole of a device step, so the client could not tell
work from a hang and guessed hang: perf-mcp's profile_model was aborted mid-run on every attempt of
a run started from an unoptimized baseline, and because it persists the roofline snapshot on its
LAST statement, the snapshot was never written -- the report lost its per-stage roofline and its
fidelity ladder and said nothing had failed.

Raising the silence window was tried first and does not fix this: the client still cannot tell the
two apart, it just waits longer before guessing. Reporting progress is what distinguishes them.
"""

from __future__ import annotations

import asyncio
import time

from scripts.tt_hw_planner.mcp_progress import install, keep_client_informed


class _Ctx:
    def __init__(self):
        self.reports = []

    async def report_progress(self, progress=None, total=None, message=None):
        self.reports.append((progress, message))


class _Server:
    """Stands in for FastMCP/MCPServer: records what gets registered."""

    def __init__(self):
        self.registered = {}

    def tool(self, *_a, **_kw):
        def _register(fn):
            self.registered[fn.__name__] = fn
            return fn

        return _register


def _run(fn, **kw):
    return asyncio.run(fn(**kw))


def test_a_slow_tool_is_announced_as_alive_while_it_runs():
    ctx = _Ctx()

    def slow():
        time.sleep(2.2)
        return {"ok": True, "value": 42}

    started = time.time()
    result = _run(keep_client_informed(slow, _Ctx, every_s=1), ctx=ctx)
    elapsed = time.time() - started

    assert result == {"ok": True, "value": 42}, "the tool's own result must pass through untouched"
    assert len(ctx.reports) >= 2, "a call running for seconds reported nothing -- the client would abort it"
    assert elapsed < 4.0, "reporting must not slow the call down"


def test_reporting_stops_the_moment_the_call_ends():
    """A finished call that kept reporting would keep a dead call looking alive."""
    ctx = _Ctx()

    def quick():
        return {"quick": True}

    assert _run(keep_client_informed(quick, _Ctx, every_s=1), ctx=ctx) == {"quick": True}
    assert ctx.reports == [], "a call that returns immediately must not report at all"


def test_a_failure_still_reaches_the_caller():
    def boom():
        raise ValueError("tool failed")

    try:
        _run(keep_client_informed(boom, _Ctx, every_s=1), ctx=_Ctx())
    except ValueError as exc:
        assert str(exc) == "tool failed"
    else:  # pragma: no cover
        raise AssertionError("the wrapper swallowed a tool failure")


def test_a_client_that_cannot_be_told_does_not_fail_the_call():
    """Progress is best-effort: a broken sink must never cost the result."""

    class _Broken(_Ctx):
        async def report_progress(self, **_kw):
            raise RuntimeError("no transport")

    def slow():
        time.sleep(1.2)
        return "done"

    assert _run(keep_client_informed(slow, _Ctx, every_s=1), ctx=_Broken()) == "done"


def test_the_public_schema_is_unchanged():
    """ctx is injected, so it must stay out of the signature the agent is shown."""
    import inspect

    def real_tool(op_signature: str, measured_ms: float = 0.0) -> dict:
        return {}

    wrapped = keep_client_informed(real_tool, _Ctx, every_s=1)
    params = inspect.signature(wrapped).parameters
    assert "op_signature" in params and "measured_ms" in params, "a real argument was lost"
    assert params["ctx"].kind is inspect.Parameter.KEYWORD_ONLY
    assert wrapped.__annotations__["ctx"] is _Ctx, "the server reads __annotations__ to hide ctx"


def test_installing_covers_every_tool_and_is_idempotent():
    server = _Server()
    assert install(server) is True
    assert install(server) is True, "installing twice must not double-wrap"

    @server.tool()
    def one():
        return 1

    @server.tool()
    def two():
        return 2

    assert set(server.registered) == {"one", "two"}
    for fn in server.registered.values():
        assert asyncio.iscoroutinefunction(fn), "a registered tool was left unwrapped"
    # ...and the decorated NAME stays synchronous: the stop gate and the suites call these tools in
    # process, and handing them a coroutine broke every such caller.
    assert one() == 1 and two() == 2, "the module-level name must stay the plain function"


def test_an_async_tool_is_left_alone():
    async def already_async():
        return "async"

    assert keep_client_informed(already_async, _Ctx, every_s=1) is already_async
