"""LLM-driven environment-incompatibility resolver.

When the env-compat check returns a list of problems, this module
asks the LLM to look at the actual problem text + the current python
env and propose the right ``pip install`` command. The fix is then
executed and the caller re-execs the original argv.

Why LLM-driven and not hardcoded:
  * The codebase's target transformers version drifts over time (it
    targets 4.x today, will target 5.x once the migration lands).
    Hardcoding ``transformers<5.0`` becomes wrong the moment the
    migration ships.
  * The failure might not be transformers at all — could be torch,
    numpy, ttnn, or a missing dep like sentencepiece. The error
    messages name the right package; let the LLM read them.
  * Some incompatibilities have multiple valid fixes (downgrade vs
    apply a code patch). The LLM picks the cheaper path given context.

Safety:
  * The LLM's proposed command is parsed and validated — only
    ``pip install ...`` form is accepted. No arbitrary shell.
  * The proposed command is printed before execution so the operator
    sees what's about to happen.
  * Hard cap via ``_ENV_FIX_ATTEMPTED_FLAG`` prevents infinite
    re-exec loops.

Fallback:
  * If the LLM isn't reachable (agent_bin missing) or its response
    can't be parsed, the caller can fall back to a deterministic
    pin (``transformers<5.0``) or to the manual-options banner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional


# ─── Result schema ──────────────────────────────────────────────────


@dataclass
class EnvFixProposal:
    """One LLM-proposed environment fix.

    Fields:
      * ``pip_args`` — list of args after ``pip install``. The caller
        runs ``pip install <pip_args...>``. Validated to be a non-empty
        list of plain strings (no shell metacharacters).
      * ``reasoning`` — human-readable explanation the operator sees
        before the command runs.
      * ``raw_text`` — original verdict file body, kept for debugging
        when parsing fails / falls back.
    """

    pip_args: List[str]
    reasoning: str = ""
    raw_text: str = ""

    @property
    def pip_command_str(self) -> str:
        """Pretty rendering for log lines."""
        return "pip install " + " ".join(self.pip_args)


# ─── Prompt builder ──────────────────────────────────────────────────


def build_env_fix_prompt(
    *,
    env_problems: List[str],
    installed_packages_summary: str,
    python_version: str,
    verdict_path: Path,
) -> str:
    """Render the LLM-fix prompt.

    Pure function. The LLM gets:
      * The structured problem list from the env-compat check
      * Summary of currently-installed packages (`pip freeze` slice)
      * Python version
      * The verdict file path it must write to

    Returns a prompt string ready for ``_invoke_agent``.
    """
    problems_block = "\n".join(f"  - {p}" for p in env_problems) or "  (none)"
    return f"""You are an environment-fix advisor for the tt_hw_planner bring-up tool.

CONTEXT
-------
The bring-up's pre-flight check found incompatibilities between the
installed Python packages and the codebase's expected API surface.
Your job: propose a SINGLE pip install command that resolves the
specific problems below.

DO NOT modify any source files. ONLY write the verdict JSON.

PYTHON VERSION
--------------
{python_version}

INSTALLED PACKAGES (top of `pip freeze`)
----------------------------------------
{installed_packages_summary}

PROBLEMS DETECTED
-----------------
{problems_block}

DECISION RULES
--------------
  • The fix is ALWAYS a ``pip install ...`` command (no arbitrary
    shell, no source file edits).
  • Prefer the smallest change: pinning ONE package to a known-working
    constraint is better than reinstalling several. If the problems
    name a specific package (e.g. "transformers==5.8.1 incompatible"),
    target THAT package.
  • Use version-range constraints (``<5.0``, ``>=4.40,<5.0``) when
    possible — they survive minor patch bumps. Use exact pins only
    when the problem requires it.
  • If multiple packages need adjustment, list them in one command
    (pip install accepts multiple).

OUTPUT
------
Write this exact JSON shape to {verdict_path}:

{{
  "pip_args": ["package_spec_1", "package_spec_2", ...],
  "reasoning": "one-sentence justification"
}}

The ``pip_args`` field is what comes AFTER ``pip install``. Examples:
  - To downgrade transformers: ``["transformers<5.0"]``
  - To pin a range: ``["transformers>=4.40,<5.0"]``
  - To install a missing dep: ``["sentencepiece"]``
  - Multiple: ``["transformers<5.0", "tokenizers>=0.20"]``

DO NOT include the word "pip" or "install" in pip_args. The caller
prepends those automatically.

DO NOT propose code edits. Only pip install commands.
DO NOT make any edits to source files.
"""


# ─── Verdict parser ──────────────────────────────────────────────────


# Validation regex for individual pip args. Permits:
#   - package names (alphanumeric, hyphens, dots, underscores)
#   - version specifiers (<>=, .,!*) embedded in the spec
#   - bracketed extras [feat]
# Rejects anything with shell metacharacters: ;, &, |, $, `, <, >, etc.
# (except in the safe context inside a version specifier).
_SAFE_PIP_ARG = re.compile(r"^[A-Za-z0-9._\-\[\]<>=!~,*]+$")


def parse_env_fix_verdict(verdict_path: Path) -> Optional[EnvFixProposal]:
    """Read and validate the LLM's proposed pip command.

    Returns ``None`` on any failure: file missing, malformed JSON,
    missing fields, unsafe argument content. Strict validation here
    prevents an LLM hallucination from turning into ``rm -rf /``.

    Pure with respect to non-file inputs. Never raises.
    """
    if not verdict_path.is_file():
        return None
    try:
        raw_text = verdict_path.read_text(encoding="utf-8")
        blob = json.loads(raw_text)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    pip_args = blob.get("pip_args")
    if not isinstance(pip_args, list) or not pip_args:
        return None
    # Validate every arg is safe
    safe_args: List[str] = []
    for a in pip_args:
        if not isinstance(a, str) or not a.strip():
            return None
        a = a.strip()
        # No pip flags — only bare package specs. This blocks tricks
        # like --extra-index-url=evil.com which would let an attacker
        # redirect pip to a hostile package index.
        if a.startswith("-"):
            return None
        if not _SAFE_PIP_ARG.match(a):
            return None
        # Defensive: forbid sub-strings that smell like injection even
        # if the regex passed
        if any(bad in a for bad in (";", "&", "|", "$", "`", "\n", "\r", " ")):
            return None
        safe_args.append(a)
    reasoning = blob.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = ""
    return EnvFixProposal(pip_args=safe_args, reasoning=reasoning.strip(), raw_text=raw_text)


# ─── Env snapshot helpers ──────────────────────────────────────────


def installed_packages_summary(max_lines: int = 30) -> str:
    """Best-effort `pip freeze` summary for the prompt.

    Truncates to the first ``max_lines`` to keep the prompt bounded.
    Returns the string or ``"(unavailable)"`` on any failure. Pure
    apart from the subprocess call.
    """
    import subprocess
    import sys as _sys

    try:
        proc = subprocess.run(
            [_sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return "(unavailable)"
    if proc.returncode != 0:
        return "(unavailable: pip freeze failed)"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


def python_version_str() -> str:
    import sys as _sys

    return f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"


# ─── Pip executor ──────────────────────────────────────────────────


def ensure_pip_available(*, timeout_s: int = 120) -> "tuple[bool, str]":
    """Ensure ``sys.executable -m pip`` works, bootstrapping via ensurepip if absent."""
    import subprocess
    import sys as _sys

    probe = subprocess.run(
        [_sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return True, ""

    print("  pip not found in this interpreter — bootstrapping via ensurepip...")
    try:
        boot = subprocess.run(
            [_sys.executable, "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"ensurepip timed out after {timeout_s}s"
    except Exception as exc:
        return False, f"ensurepip invocation raised {type(exc).__name__}: {exc}"
    if boot.returncode != 0:
        tail = (boot.stdout + boot.stderr).strip().splitlines()[-20:]
        return False, "ensurepip failed:\n" + "\n".join(tail)

    reprobe = subprocess.run(
        [_sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if reprobe.returncode != 0:
        return False, "pip still unavailable after ensurepip bootstrap"
    print(f"  pip bootstrapped: {reprobe.stdout.strip()}")
    return True, ""


def run_pip_install(pip_args: List[str], *, timeout_s: int = 300) -> "tuple[bool, str]":
    """Execute ``pip install <pip_args>`` and return ``(ok, log_tail)``.

    Bounded subprocess. ``log_tail`` carries the last 20 lines of
    stdout+stderr on failure (informative for the operator). Never
    raises; exceptions become ``(False, error_str)``.
    """
    import subprocess
    import sys as _sys

    import importlib.util as _ilu

    ensure_pip_available()
    _ilu.invalidate_caches()
    if _ilu.find_spec("pip") is not None:
        cmd = [_sys.executable, "-m", "pip", "install", *pip_args]
    else:
        cmd = ["uv", "pip", "install", "--python", _sys.executable, *pip_args]
    print(f"  Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"pip install timed out after {timeout_s}s"
    except Exception as exc:
        return False, f"pip invocation raised {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-20:]
        return False, "\n".join(tail)
    return True, ""


# ─── Orchestrator ───────────────────────────────────────────────────


def run_llm_env_fix(
    *,
    env_problems: List[str],
    work_dir: Path,
    agent_invoker: Optional[Callable[..., int]] = None,
    agent_bin: str = "claude",
    agent_model: str = "haiku",
    timeout_s: int = 180,
) -> Optional[EnvFixProposal]:
    """Ask the LLM to propose an env-fix and return its parsed
    proposal.

    Orchestrates: snapshot env → build prompt → invoke agent → parse
    verdict. Same best-effort contract as Items 2-7: any failure mode
    returns ``None`` so the caller falls back to its deterministic
    safety net (typically the manual-options banner).

    The ``agent_invoker`` seam matches the pattern used elsewhere
    (Items 2/3/5). Tests inject mocks; real callers pass None and
    get the default _invoke_agent wrapper.
    """
    if not env_problems:
        return None

    verify_dir = work_dir / "_env_fix"
    try:
        verify_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    verdict_path = verify_dir / "verdict.json"
    try:
        if verdict_path.exists():
            verdict_path.unlink()
    except Exception:
        pass

    prompt = build_env_fix_prompt(
        env_problems=env_problems,
        installed_packages_summary=installed_packages_summary(),
        python_version=python_version_str(),
        verdict_path=verdict_path,
    )

    if agent_invoker is None:

        def _default(prompt_text, *, expected_deliverable_files, timeout_s, **_):
            from .agent import _invoke_agent

            return _invoke_agent(
                prompt_text,
                provider="claude",
                agent_bin=agent_bin,
                cwd=work_dir,
                model=agent_model,
                timeout_s=timeout_s,
                iter_tag="env_fix",
                expected_deliverable_files=list(expected_deliverable_files),
            )

        agent_invoker = _default

    try:
        rc = agent_invoker(prompt, expected_deliverable_files=[verdict_path], timeout_s=timeout_s)
    except Exception:
        return None
    if rc != 0:
        return None

    return parse_env_fix_verdict(verdict_path)


__all__ = [
    "EnvFixProposal",
    "build_env_fix_prompt",
    "installed_packages_summary",
    "parse_env_fix_verdict",
    "python_version_str",
    "run_llm_env_fix",
    "run_pip_install",
]
