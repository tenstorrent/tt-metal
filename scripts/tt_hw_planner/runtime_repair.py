"""Runtime-repair loop for `tt_hw_planner up --auto`.

When ``cmd_up`` routes a model to one of the fast paths
(``ALREADY-SUPPORTED`` or ``COLD-START``) and the underlying
``cmd_prepare --execute`` returns non-zero with a Python error in
the repo's runtime code (e.g. ``models/tt_transformers/tt/``),
historically cmd_up just printed the canonical ``OUTCOME: FAIL rc=N``
banner and gave up. That left genuinely fixable model-specific gaps
(e.g. medgemma-4b-it's nested ``rope_scaling`` shape) un-repaired
even though they were exactly the kind of failure the iterate-loop
exists to handle.

This module provides the building blocks of a runtime-repair driver
that closes that gap. It exposes three pure helpers (no I/O, no
subprocess, no agent calls) so the parsing/classifier behaviour can
be invariant-tested without spinning up an LLM or a device:

  * :func:`parse_pytest_traceback` — extract the deepest in-repo
    frame, the exception type+message, and the rendered local
    variables from a pytest "short test summary" + traceback block.
  * :func:`is_repairable_failure` — yes/no classifier deciding
    whether the failure is the kind we should hand to an LLM agent
    (Python error in ``models/``) vs. one we should bail on
    (auth/network/hardware/timeout/pytest collection error).
  * :func:`build_repair_prompt` — render the focused prompt the
    LLM agent receives. Kept pure so the prompt template can be
    diffed in tests without invoking claude.

The actual loop driver (which runs pytest, calls the agent, retries
until success or max_iters) lives in ``cli.py`` because it needs to
share the existing ``_invoke_agent``/heartbeat/process-tree-kill
infrastructure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


_TB_FRAME_RE = re.compile(
    r"^(?P<file>[^\s:]+\.py):(?P<line>\d+):\s+(?P<exc>"
    r"Failed|"
    r"[A-Z][A-Za-z_0-9]*Error|"
    r"[A-Z][A-Za-z_0-9]*Warning|"
    r"[A-Z][A-Za-z_0-9]*Exception"
    r")\s*$",
    re.MULTILINE,
)


_TB_EXC_MSG_RE = re.compile(
    r"^E\s+(?P<exc>Failed|[A-Z][A-Za-z_0-9]*(?:Error|Warning|Exception)):\s*(?P<msg>.*)$",
    re.MULTILINE,
)


_NON_REPAIRABLE_PATH_PREFIXES: tuple = (
    "python_env/",
    "/usr/",
    "/opt/",
    "<frozen ",
    "_pytest/",
    "pytest/",
    "ttnn/ttnn/",
)


_REPAIRABLE_PATH_PREFIXES: tuple = (
    "models/",
    "scripts/",
)


_NON_REPAIRABLE_EXC_SUBSTRS: tuple = (
    "Failed: Timeout",
    "pytest-timeout",
    "TimeoutError",
    "ConnectionError",
    "HTTPError",
    "GatedRepoError",
    "RepositoryNotFoundError",
    "OSError: Can't load",
    "DEVICE_FATAL",
    "DispatchError",
    "out of memory",
)


@dataclass
class TracebackInfo:
    """Parsed information about a pytest failure.

    ``failure_file`` is the deepest in-repo file at which the
    exception was raised; ``failure_line`` is the line number; the
    other fields carry context for the LLM repair prompt.
    """

    failure_file: Optional[str] = None
    failure_line: Optional[int] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None

    excerpt: str = ""

    frames: List[str] = field(default_factory=list)

    @property
    def is_parseable(self) -> bool:
        return self.failure_file is not None and self.failure_line is not None and self.exception_type is not None


def parse_pytest_traceback(captured_output: str) -> TracebackInfo:
    """Parse a captured pytest stdout+stderr blob into a
    :class:`TracebackInfo`.

    Strategy:
      1. Find all ``<file>:<line>: <ExceptionType>`` matches (pytest's
         frame-summary lines).
      2. The deepest in-repo frame is the failure location.
      3. The ``E   ExceptionType: ...`` line, if present, gives the
         exception message.
      4. The traceback excerpt is the trailing ~80 lines of the
         output (which includes pytest's rendered local variables).

    Returns a :class:`TracebackInfo` with whatever could be parsed.
    Caller checks ``info.is_parseable`` before deciding what to do."""
    info = TracebackInfo()
    if not captured_output:
        return info

    frames_in_order: List[str] = []
    deepest_repo_frame: Optional[tuple] = None
    for m in _TB_FRAME_RE.finditer(captured_output):
        file_ = m.group("file")
        line = int(m.group("line"))
        exc = m.group("exc")
        frames_in_order.append(f"{file_}:{line}")

        if _is_repo_path(file_):
            deepest_repo_frame = (file_, line, exc)
    info.frames = frames_in_order
    if deepest_repo_frame is not None:
        info.failure_file = deepest_repo_frame[0]
        info.failure_line = deepest_repo_frame[1]
        info.exception_type = deepest_repo_frame[2]

    exc_msg_match = None
    for em in _TB_EXC_MSG_RE.finditer(captured_output):
        exc_msg_match = em
    if exc_msg_match is not None:
        if info.exception_type is None:
            info.exception_type = exc_msg_match.group("exc")
        info.exception_message = exc_msg_match.group("msg").strip()

    lines = captured_output.splitlines()
    info.excerpt = "\n".join(lines[-240:])

    return info


def _is_repo_path(path: str) -> bool:
    """True iff ``path`` looks like a path inside this repo's own
    source tree (not site-packages, not stdlib, not ttnn bindings)."""
    if not path:
        return False
    for prefix in _NON_REPAIRABLE_PATH_PREFIXES:
        if path.startswith(prefix):
            return False
    for prefix in _REPAIRABLE_PATH_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def is_repairable_failure(info: TracebackInfo) -> tuple:
    """Decide whether a failure is the kind that an LLM agent can
    plausibly repair by editing a file in ``models/``.

    Returns ``(yes, reason)`` where ``yes`` is a bool and ``reason``
    is a human-readable string explaining the decision. Even when
    ``yes`` is False, the reason is printed so the user understands
    why the repair loop didn't fire."""
    if not info.is_parseable:
        return (
            False,
            "could not parse a Python traceback out of the pytest "
            "output -- the failure may be a timeout, a hang, a "
            "subprocess kill, or a pre-collection import error",
        )
    if info.failure_file is None or not _is_repo_path(info.failure_file):
        return (
            False,
            f"the deepest frame ({info.failure_file}) is not in "
            f"`models/` or `scripts/`; refusing to ask the LLM to "
            f"patch site-packages or generated bindings",
        )
    combined = " ".join(
        [
            info.exception_type or "",
            info.exception_message or "",
        ]
    )
    for needle in _NON_REPAIRABLE_EXC_SUBSTRS:
        if needle in combined:
            return (
                False,
                f"exception class/message contains '{needle}', which "
                f"is not a code-patchable failure (try the explicit "
                f"recovery commands in the OUTCOME banner instead)",
            )
    return (True, "repairable: in-repo Python error in a model file")


def build_repair_prompt(
    *,
    model_id: str,
    info: TracebackInfo,
    iter_idx: int,
    max_iters: int,
    previous_attempts: Optional[List[str]] = None,
) -> str:
    """Render the focused LLM prompt for one repair iteration.

    The prompt:
      * names the model and the exact iteration of the loop,
      * shows the captured traceback excerpt (last ~240 lines),
      * names the file the LLM should patch first,
      * lists hard constraints (no edits to pytest.ini /
        scripts/tt_hw_planner / ttnn bindings; don't break already-
        supported sibling models),
      * if previous attempts didn't fix it, lists what was tried so
        the LLM doesn't loop on the same patch.

    Kept pure (string in, string out) so the template can be diffed
    in tests."""
    previous_attempts = previous_attempts or []
    prev_block = ""
    if previous_attempts:
        prev_block = (
            "\nPREVIOUS REPAIR ATTEMPTS (each one was tried in an "
            "earlier iteration of this loop and did NOT fix the "
            "failure; do not repeat the same fix):\n"
        )
        for i, p in enumerate(previous_attempts, 1):
            prev_block += f"  attempt {i}: {p}\n"

    return (
        f"You are an automated runtime-repair agent for the\n"
        f"`tt_hw_planner` model bring-up tool. The user ran:\n"
        f"    python -m scripts.tt_hw_planner up {model_id} --auto ...\n"
        f"The model is detected as compatible with an existing TT\n"
        f"backend (tt_transformers / simple_text_demo), but running\n"
        f"the demo pytest fails at iteration {iter_idx}/{max_iters}\n"
        f"of the runtime-repair loop.\n"
        f"\n"
        f"FAILURE\n"
        f"-------\n"
        f"  file       : {info.failure_file}\n"
        f"  line       : {info.failure_line}\n"
        f"  exception  : {info.exception_type}\n"
        f"  message    : {info.exception_message or '(see traceback)'}\n"
        f"\n"
        f"PYTEST OUTPUT (tail; includes traceback + locals)\n"
        f"--------------------------------------------------\n"
        f"{info.excerpt}\n"
        f"--------------------------------------------------\n"
        f"{prev_block}"
        f"\n"
        f"YOUR JOB\n"
        f"--------\n"
        f"Edit {info.failure_file} (and any other in-repo files that\n"
        f"the fix logically requires, e.g. its callers) so the demo\n"
        f"pytest passes for {model_id}. After your edit the demo\n"
        f"will be re-run automatically; if it still fails you'll be\n"
        f"called again with the new error.\n"
        f"\n"
        f"HARD CONSTRAINTS\n"
        f"----------------\n"
        f"  1. Do NOT edit any file under:\n"
        f"        scripts/tt_hw_planner/    (the planner tool)\n"
        f"        ttnn/ttnn/                (generated bindings)\n"
        f"        python_env/               (venv site-packages)\n"
        f"        tests/, conftest.py       (test infrastructure)\n"
        f"        pytest.ini, pyproject.toml\n"
        f"  2. Do NOT add or expand CPU-only fallbacks unless the\n"
        f"     code already documents the relevant block as PARTIAL\n"
        f"     and the comment explicitly invites a CPU path.\n"
        f"  3. Do NOT regress already-supported models. Any helper\n"
        f"     you change must still produce the same behaviour for\n"
        f"     the previous inputs (e.g. old flat rope_scaling shape\n"
        f"     must keep working when you teach it the new nested\n"
        f"     shape).\n"
        f"  4. Prefer minimal, surgical patches. A 10-line change\n"
        f"     that handles the new config shape is better than a\n"
        f"     150-line refactor.\n"
        f"  5. Reply with the SHORTEST sentence describing what you\n"
        f"     changed; the loop driver echoes that into a one-line\n"
        f"     audit log. The actual diff goes via your file-edit\n"
        f"     tool calls.\n"
    )
