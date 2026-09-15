"""Agentic single-component perf-test builder.

The one-shot generator + regex error-scraping (`_extract_error`) cannot parse the open-ended error
formats a component perf test hits (python tracebacks, ttnn TT_FATAL, C++ `unordered_map::at`,
unpack/shape errors). This builder instead drives the claude-code CLI (`claude -p`, login auth, no
SDK, no model tier — like the rest of the cc engine) to write the test, RUN it through an
out-of-process MCP tool (perf_test_mcp.py) that routes to `_run_perf_node` (device self-heal + output
bounding live there, and the agent never touches the device), READ the raw output itself, and iterate
until it passes or the module is genuinely eager-only. The agent's own conversation carries what it
built and what it already tried, so it does not repeat dead ends.

Returns False (never raises) when the claude CLI / MCP server is not wired, so the caller degrades to
the one-shot generator.
"""

from __future__ import annotations

import re
import os
import sys
from pathlib import Path

# ONE state directory for every durable temp artifact -- see cc_optimize/tmpstate.py.
# agent/state_dir.py loads cc_optimize/tmpstate.py by path, once, for the four modules that need it.
from .state_dir import state_dir


def _component_run_timeout() -> int:
    """Budget for one perf-test build run on device, scaled from observed build cost (BUG 4).
    The fixed 240 s was below llama's real 872 s build and ACE's 408 s."""
    try:
        from .probes import adaptive_op_timeout

        return adaptive_op_timeout("build", env_key="PERF_MCP_COMPONENT_RUN_TIMEOUT_S")
    except Exception:  # noqa: BLE001
        # 240 s was the defect: llama's real perf-test build takes ~872 s.
        from .device_budget import operator_ceiling_s

        return int(os.environ.get("PERF_MCP_COMPONENT_RUN_TIMEOUT_S", "") or operator_ceiling_s())


_TIMEOUT_CODES = {124, 137, 143, -9, -15}

_WEDGE_RETRY_GUIDANCE = (
    "The trace capture HUNG the device (timed out; the harness reset it). The code BETWEEN "
    "ttnn.begin_trace_capture and ttnn.end_trace_capture touched the HOST — a trace records only pure "
    "device-op dispatch. RESTRUCTURE so the captured region is host-free: build the module and ALL its "
    "inputs / masks / constants ONCE, resident on device, BEFORE begin_trace_capture; nothing between "
    "begin/end may call ttnn.from_torch / ttnn.to_torch / .item() / .cpu() / torch construction / python "
    "shape or control-flow. If one op inside is the host bit, move it OUT of the captured region or "
    "replace it with a device-only equivalent. Then call run_perf_test again to keep attempting the "
    "TRACE. There is NO eager fallback — eager is only ever used when the operator sets TT_PERF_TRACE=0."
)


def _eager_flag() -> bool:
    return os.environ.get("TT_PERF_TRACE") == "0"


def _judge_output(rc, out: str) -> str:
    from .perf_test_gen import _parse_trace_path

    text = out or ""
    if _eager_flag():
        return "PASS_EAGER" if (rc == 0 and "FORWARD_WALL_MS=" in text) else "FAIL"
    # Was `"WEDGE" in text`: the bare substring anywhere in pytest output -- a comment or a
    # string in the GENERATED test, and the agent is explicitly told about WEDGE verdicts --
    # classified every failing run as a device hang, triggering a board reset and the wrong
    # "your trace touched the host" guidance instead of the real error. Require our own marker.
    if rc in _TIMEOUT_CODES or re.search(r"^\s*(?:E\s+)?WEDGE(?:[:=]|\b)", text, re.MULTILINE):
        return "WEDGE"
    traced = ("TRACE_PER_TOKEN_MS=" in text) and bool(_parse_trace_path(text))
    if rc == 0 and traced:
        return "PASS_TRACE"
    return "FAIL"


def _bound_output(out: str, limit: int = 16000) -> str:
    if not out:
        return "(no output captured)"
    for marker in ("=== FAILURES ===", "=== ERRORS ==="):
        idx = out.rfind(marker)
        if idx != -1:
            return out[idx:][:limit]
    return out[-limit:]


def _run_and_format(node_abs: str, state: dict | None = None) -> str:
    from .perf_test_gen import _run_perf_node

    if state is None:
        state = {"wedges": 0, "passed": False}
    env = {"TT_PERF_TRACE": "0" if _eager_flag() else "1"}
    rc, out = _run_perf_node(node_abs, env, timeout_s=_component_run_timeout())
    verdict = _judge_output(rc, out)
    if verdict == "WEDGE":
        state["wedges"] += 1
        return f"VERDICT=WEDGE\nrc={rc}\n(trace hang {state['wedges']}) {_WEDGE_RETRY_GUIDANCE}"
    if verdict in ("PASS_TRACE", "PASS_EAGER"):
        state["passed"] = True
    return f"VERDICT={verdict}\nrc={rc}\n----- raw test output -----\n{_bound_output(out)}"


_PERF_TEST_MCP_REL = "models/experimental/perf_automation/cc_optimize/perf_test_mcp.py"
_PERF_RUN_TOOL = "mcp__perftest-mcp__run_perf_test"
# How often the agent's descendant tree is re-snapshotted while it runs. The snapshot is the ONLY
# record of a grandchild once the agent exits -- /proc PPID links die with the parent -- so this is
# the resolution at which a leaked worker can still be named. A /proc scan is microseconds; the agent
# runs for minutes.
_AGENT_REAP_POLL_S = float(os.environ.get("PERF_MCP_AGENT_REAP_POLL_S", "5"))


def _repo_root() -> Path:
    # .../models/experimental/perf_automation/agent/perf_test_agent.py -> repo root is parents[4]
    return Path(__file__).resolve().parents[4]


_SYSTEM = (
    "You write ONE single-component performance test for a TTNN model, then stop. Workflow: write "
    "the test file with Write/Edit, RUN it by calling the run_perf_test tool, READ the raw output "
    "(the real traceback AND the input the test built), fix your file, and repeat until run_perf_test "
    "returns VERDICT=PASS_TRACE — the ONLY success. VERDICT=WEDGE means the trace hung because the "
    "captured region touched the host: RESTRUCTURE it to be host-free (build inputs/constants resident "
    "once before the capture) and keep attempting the trace. There is NO eager fallback — do NOT print "
    "TRACE_NOT_TRACE_CAPABLE or drop the trace block; eager mode is only ever enabled by the operator via "
    "TT_PERF_TRACE=0 and is outside your control. Edit ONLY the one perf test file named in the task. "
    "NEVER run pytest, tt-smi, kill, fuser, or open/close a device yourself — the run_perf_test tool "
    "and the harness own all device execution and recovery; doing it yourself breaks the run. Do not "
    "repeat an approach that already failed the same way — change the approach. Keep your final "
    "message to one short line."
)


_MCP_IMPORT_PROBE = (
    "try:\n"
    "    from mcp.server.fastmcp import FastMCP\n"
    "except ModuleNotFoundError:\n"
    "    from mcp.server.mcpserver import MCPServer as FastMCP\n"
)


def build_component_perf_test(root: str | Path, task: str, out_rel: str, prompt_body: str, max_turns: int = 48) -> bool:
    """Author ONE single-component perf test with the claude-code CLI (`claude -p`), exactly like the
    rest of the cc engine: login auth, NO claude SDK, NO model tier / escalation ladder (claude's own
    default model). The one device tool (run_perf_test) is exposed out-of-process via --mcp-config to
    perf_test_mcp.py; success is signalled back through a status file the server updates each run."""
    import json as _json
    import subprocess
    import subprocess as _sp
    import tempfile
    import time

    root = Path(root)
    node_abs = f"{root / out_rel}::test_{task}_perf"
    repo_root = _repo_root()
    server = repo_root / _PERF_TEST_MCP_REL
    if not server.is_file():
        print(
            "      · agentic builder unavailable: no perf-test MCP server at %s" % server,
            file=sys.stderr,
            flush=True,
        )
        return False
    # SAY WHY IT IS UNAVAILABLE. Without run_perf_test the agent can write a test but never run it,
    # so it cannot converge -- and the caller reports "agentic builder did not converge", which reads
    # as the model being hard to generate for. On 2026-08-15 the real cause was a dependency: mcp
    # 2.0.0 moved FastMCP, so the server died at import and the tool silently fell back to the
    # one-shot generator for three runs. Import failures are cheap to check and expensive to guess.
    #
    # PROBE WHAT THE SERVERS DO, not one spelling of it. They now import FastMCP under mcp 1.x and
    # MCPServer under 2.x, so a probe hard-coded to `mcp.server.fastmcp` reports "unavailable" for a
    # server that would have started perfectly well -- the same silent fallback, caused by the check
    # rather than the dependency. Measured 2026-08-27 on a venv resolved to mcp 2.1.1.
    _probe = _sp.run(
        [sys.executable, "-c", _MCP_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if _probe.returncode != 0:
        print(
            "      · agentic builder unavailable: the perf-test MCP server cannot start, so the "
            "agent would have no way to RUN what it writes.\n        %s\n        fix: pip install "
            "-r models/experimental/perf_automation/requirements-agent.txt"
            % ((_probe.stderr or "").strip().splitlines() or ["import failed"])[-1],
            file=sys.stderr,
            flush=True,
        )
        return False
    try:
        from .agent_bin import resolve_claude_bin
    except Exception:  # noqa: BLE001
        return False

    status_fd, status_path = tempfile.mkstemp(prefix=f"perftest_status_{task}_", suffix=".json")
    os.close(status_fd)
    Path(status_path).write_text(_json.dumps({"passed": False}))

    server_env = {"PERF_TEST_NODE": node_abs, "PERF_TEST_STATUS_FILE": status_path}
    if "TT_PERF_TRACE" in os.environ:
        server_env["TT_PERF_TRACE"] = os.environ["TT_PERF_TRACE"]
    cfg = {
        "mcpServers": {
            "perftest-mcp": {
                "command": sys.executable or "python",
                "args": [str(server)],
                "env": server_env,
            }
        }
    }
    cfg_path = server.parent / f".perftest_mcp_config_{task}.json"
    cfg_path.write_text(_json.dumps(cfg, indent=2))

    prompt = (
        prompt_body + f"\n\nWrite the test file at `{out_rel}` (relative to your working directory) with the Write "
        "tool. Then CALL run_perf_test, read its raw output, and iterate (Edit -> run_perf_test) until it "
        "returns VERDICT=PASS_TRACE — the ONLY success. On VERDICT=WEDGE the trace hung — restructure the "
        "captured region to be host-free and keep attempting the trace; there is no eager fallback. Do NOT "
        "finish until run_perf_test returns PASS_TRACE."
    )

    env = dict(os.environ)
    for _k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(_k, None)

    overall_timeout = int(os.environ.get("PERF_TEST_AGENT_TIMEOUT_S", "") or max(3600, _component_run_timeout() * 8))
    log_path = state_dir() / f"perftest_{task}.agent.log"
    cmd = [
        resolve_claude_bin(),
        "-p",
        prompt,
        "--mcp-config",
        str(cfg_path),
        "--strict-mcp-config",
        "--system-prompt",
        _SYSTEM,
        "--allowedTools",
        "Read,Write,Edit,Glob,Grep," + _PERF_RUN_TOOL,
        "--permission-mode",
        "bypassPermissions",
        "--max-turns",
        str(max_turns),
        "--output-format",
        "text",
    ]
    try:
        _lf = open(log_path, "w", buffering=1, errors="ignore")
    except OSError:
        _lf = subprocess.DEVNULL
    proc = subprocess.Popen(cmd, cwd=str(root), env=env, start_new_session=True, stdout=_lf, stderr=subprocess.STDOUT)
    # THE TREE OUTLIVES THE AGENT, AND ONLY A LIVE ROOT CAN BE WALKED.
    #
    # This waited for the agent and killed its process GROUP on timeout. Both halves leaked. The
    # claude CLI spawns workers in their own sessions, so the group kill never reached them; and the
    # kill only ran on TimeoutExpired, so an agent that finished on its own -- exhausted --max-turns,
    # returned a verdict -- left everything it had spawned running, with nothing to kill it.
    #
    # Measured 2026-08-16: the agent and its parent were still alive 37 and 70 minutes after the run
    # gave up. They sat in their own sessions holding no device, so neither the group kill nor
    # _reclaim_device's device-holder sweep could see them, and the supervisor -- which treats its
    # own child exiting as the attempt being over -- started a second optimize run on the same board.
    # Two runs driving one board took the ARC cores down; `tt-smi -r` then failed with "ARC core
    # (8, 0) failed to start" until the tree was killed by hand, after which the same reset worked.
    #
    # So: snapshot the descendants WHILE the root is alive (once it exits they are reparented to init
    # and unreachable), and reap on EVERY exit, exactly as _run_device_proc already does for its own
    # subprocess -- "Reap any lingering group member on EVERY exit".
    _seen: set = set()
    _timed_out = False
    try:
        from .probes import _descendant_pids as _desc, _kill_tree as _reap
    except Exception:  # noqa: BLE001 -- reaping is best-effort; never fail the build over it
        _desc, _reap = (lambda _p: []), (lambda _p, extra=(): None)
    try:
        _deadline = time.monotonic() + float(overall_timeout or 0)
        while True:
            try:
                proc.wait(timeout=_AGENT_REAP_POLL_S)
                break
            except subprocess.TimeoutExpired:
                _seen.update(_desc(proc.pid))
                if overall_timeout and time.monotonic() >= _deadline:
                    _timed_out = True
                    break
    except Exception:  # noqa: BLE001
        pass
    finally:
        _seen.update(_desc(proc.pid))
        try:
            _reap(proc.pid, extra=_seen)
        except Exception:  # noqa: BLE001
            pass
        if _timed_out:
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
    if _lf not in (None, subprocess.DEVNULL):
        try:
            _lf.close()
        except Exception:  # noqa: BLE001
            pass
    try:
        cfg_path.unlink()
    except OSError:
        pass

    try:
        passed = bool(_json.loads(Path(status_path).read_text()).get("passed"))
    except Exception:  # noqa: BLE001
        passed = False
    try:
        Path(status_path).unlink()
    except OSError:
        pass
    return passed
