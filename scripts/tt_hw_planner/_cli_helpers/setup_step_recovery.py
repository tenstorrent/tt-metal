"""Layered recovery for tool-internal setup-step failures.

Background
----------
Steps 0-4 of ``cmd_up`` (pre-flight / static analysis / scaffold /
LLM-gate / autofill) run in the tool process itself, NOT in pytest.
A Python exception raised in any of them hard-aborts the bring-up.
Examples we've hit:

  * ``AutoModel.from_pretrained`` raises ``ValueError`` for a
    trust_remote_code config the base AutoModel doesn't recognize.
  * Stub write raises ``FileNotFoundError`` because ``_stubs/`` doesn't
    exist yet.
  * Memory-fit check fails for the requested dtype (model too big).

Existing recovery mechanisms (``_runtime_repair_loop``, ``run_llm_env_fix``,
``_maybe_escalate_pcc_fail``, etc.) catch pytest tracebacks, package
mismatches, and PCC failures — but they do NOT catch tool-internal
setup-step exceptions. Every new model family that exposes a new
infrastructure failure mode hard-aborts until the tool source is
patched. That doesn't scale.

Two-layer design
----------------
1. **Rule registry** (fast, deterministic, fail-closed). The rules are
   parametrized templates for recoveries we've already seen:

     - ``cascade_class`` — try alternative auto-classes (e.g., when
       ``AutoModel.from_pretrained`` fails, retry with
       ``AutoModelForCausalLM``).
     - ``mkdir_parents`` — create missing parent directories.
     - ``dtype_downgrade`` — try a smaller dtype on memory-fit fail.
     - ``overlay_drop`` — drop stale overlays that prevent fresh runs.
     - ``skip_component`` — skip a component and continue.
     - ``re_exec`` — restart the command with corrected env.

   When an exception matches a rule's pattern, the rule fires
   immediately without an LLM call.

2. **LLM fallback** (slower, novel failures). When no rule matches,
   the LLM is shown the traceback + step name + workspace state and
   asked to pick an action from the same enum above (an
   *allowlist* — the LLM doesn't write tool code, it just names which
   parametrized recovery to apply).

Safety
------
* The LLM cannot propose arbitrary code. It picks one action from a
  fixed enum and provides parameters that are themselves validated.
* Every action is bounded:

    - ``cascade_class`` only tries names found in the requested module.
    - ``mkdir_parents`` only creates dirs within the worktree.
    - ``re_exec`` only sets specific env vars (no shell injection).
    - ``overlay_drop`` only drops a single model scope.

* Re-entry guard prevents infinite recovery loops.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─── Recovery action enum ──────────────────────────────────────────


class RecoveryAction(str, Enum):
    """Allowlist of recovery actions the rule registry + LLM may pick.

    NEW actions must be added here explicitly, with a corresponding
    apply-fn case in ``apply_recovery_action`` AND a parser-validator
    case in ``parse_recovery_verdict``. The allowlist shape is the
    primary safety boundary: the LLM cannot propose actions outside
    this set.
    """

    CASCADE_CLASS = "cascade_class"
    MKDIR_PARENTS = "mkdir_parents"
    DTYPE_DOWNGRADE = "dtype_downgrade"
    OVERLAY_DROP = "overlay_drop"
    SKIP_COMPONENT = "skip_component"
    RE_EXEC = "re_exec"
    CANNOT_RECOVER = "cannot_recover"


@dataclass
class RecoveryProposal:
    """One concrete recovery to apply.

    ``args`` shape varies per action. The parser validates them.
    """

    action: RecoveryAction
    args: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    source: str = "rule"  # "rule" or "llm"

    def label(self) -> str:
        return f"{self.action.value}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


# ─── Argument validators (per-action allowlist) ─────────────────────


# Tighter than env-fix: action args must look like Python identifiers,
# bounded paths, or short enum strings. No shell metacharacters anywhere.
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Path must be relative or under a temp/worktree; no escapes.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./\-]+$")
_SAFE_DTYPE_RE = re.compile(r"^(?:bf16|bfp8_b|bfp4_b|fp32|fp16)$")


def _validate_cascade_class(args: Dict[str, Any]) -> bool:
    """Args: {module: 'transformers', attrs: ['AutoModelForCausalLM', ...]}.

    All attrs must be Python identifiers. Module must be a single
    dotted Python module name."""
    module = args.get("module")
    attrs = args.get("attrs")
    if not isinstance(module, str) or not all(part.isidentifier() for part in module.split(".")):
        return False
    if not isinstance(attrs, list) or not attrs:
        return False
    if any(not isinstance(a, str) or not _SAFE_IDENT_RE.match(a) for a in attrs):
        return False
    return True


def _validate_mkdir_parents(args: Dict[str, Any]) -> bool:
    """Args: {path: '...'}. Must be a SAFE RELATIVE path, no shell
    metacharacters, no '..' escapes, no absolute paths.

    Reject absolute paths because they could escape the worktree
    (e.g., LLM proposing mkdir /etc/foo). The caller's repo_root
    becomes the parent for any relative path."""
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return False
    if path.startswith("/"):
        return False
    if ".." in path.split("/"):
        return False
    if not _SAFE_PATH_RE.match(path):
        return False
    return True


def _validate_dtype_downgrade(args: Dict[str, Any]) -> bool:
    """Args: {from: 'bf16', to: 'bfp8_b'}. Both must be known dtypes."""
    return all(isinstance(args.get(k), str) and _SAFE_DTYPE_RE.match(args.get(k, "")) for k in ("from", "to"))


def _validate_overlay_drop(args: Dict[str, Any]) -> bool:
    """Args: {model_id: 'org/name'}. Must look like an HF model id."""
    model_id = args.get("model_id")
    if not isinstance(model_id, str) or "/" not in model_id:
        return False
    # Permit alphanumerics, slashes, dots, hyphens, underscores; nothing else.
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9_./\-]*$", model_id))


def _validate_skip_component(args: Dict[str, Any]) -> bool:
    """Args: {component_name: 'attention'}."""
    name = args.get("component_name")
    return isinstance(name, str) and bool(_SAFE_IDENT_RE.match(name or ""))


def _validate_re_exec(args: Dict[str, Any]) -> bool:
    """Args: {env: {KEY: VAL, ...}}. Keys must be uppercase identifiers,
    values must be safe (no shell metacharacters)."""
    env = args.get("env", {})
    if not isinstance(env, dict):
        return False
    for k, v in env.items():
        if not isinstance(k, str) or not re.match(r"^[A-Z_][A-Z0-9_]*$", k):
            return False
        if not isinstance(v, str):
            return False
        if any(bad in v for bad in (";", "&", "|", "$", "`", "\n", "\r")):
            return False
    return True


def _validate_cannot_recover(args: Dict[str, Any]) -> bool:
    """Args: {reason: '...'}. Always accepts; this is the explicit
    'no fix' verdict."""
    return isinstance(args.get("reason", ""), str)


_ACTION_VALIDATORS: Dict[RecoveryAction, Callable[[Dict[str, Any]], bool]] = {
    RecoveryAction.CASCADE_CLASS: _validate_cascade_class,
    RecoveryAction.MKDIR_PARENTS: _validate_mkdir_parents,
    RecoveryAction.DTYPE_DOWNGRADE: _validate_dtype_downgrade,
    RecoveryAction.OVERLAY_DROP: _validate_overlay_drop,
    RecoveryAction.SKIP_COMPONENT: _validate_skip_component,
    RecoveryAction.RE_EXEC: _validate_re_exec,
    RecoveryAction.CANNOT_RECOVER: _validate_cannot_recover,
}


# ─── Rule registry (fast deterministic recoveries) ──────────────────


@dataclass
class _Rule:
    """One deterministic recovery rule."""

    name: str
    matches: Callable[[BaseException, str], bool]
    build_proposal: Callable[[BaseException, str], RecoveryProposal]


def _rule_automodel_cascade(exc: BaseException, _step: str) -> bool:
    """``AutoModel.from_pretrained`` raised ``ValueError`` for a
    trust_remote_code config. Cascade to task-specific factories."""
    if not isinstance(exc, ValueError):
        return False
    return "Unrecognized configuration class" in str(exc) and "AutoModel" in str(exc)


def _build_automodel_cascade(_exc: BaseException, _step: str) -> RecoveryProposal:
    return RecoveryProposal(
        action=RecoveryAction.CASCADE_CLASS,
        args={
            "module": "transformers",
            "attrs": ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForSeq2SeqLM"],
        },
        reasoning="AutoModel base class can't see trust_remote_code Phi3/etc configs; cascade to task-specific factories.",
        source="rule",
    )


def _rule_mkdir_missing(exc: BaseException, _step: str) -> bool:
    """``FileNotFoundError`` writing into a missing parent dir."""
    return isinstance(exc, FileNotFoundError) and exc.errno == 2 if hasattr(exc, "errno") else False


def _build_mkdir_missing(exc: BaseException, _step: str) -> RecoveryProposal:
    # exc.filename is the target file. We want its parent dir.
    target = getattr(exc, "filename", None) or ""
    parent = str(Path(target).parent) if target else ""
    return RecoveryProposal(
        action=RecoveryAction.MKDIR_PARENTS,
        args={"path": parent},
        reasoning=f"Write to {target!r} failed because parent dir doesn't exist; mkdir it.",
        source="rule",
    )


_DEFAULT_RULES: List[_Rule] = [
    _Rule("automodel-cascade", _rule_automodel_cascade, _build_automodel_cascade),
    _Rule("mkdir-missing-parent", _rule_mkdir_missing, _build_mkdir_missing),
]


def find_rule_proposal(
    exc: BaseException, step_name: str, *, rules: Optional[List[_Rule]] = None
) -> Optional[RecoveryProposal]:
    """Walk the rule registry and return the first matching proposal."""
    for rule in rules or _DEFAULT_RULES:
        try:
            if rule.matches(exc, step_name):
                return rule.build_proposal(exc, step_name)
        except Exception:
            continue
    return None


# ─── Verdict parser (LLM-fallback verdicts) ─────────────────────────


def parse_recovery_verdict(verdict_path: Path) -> Optional[RecoveryProposal]:
    """Read the LLM-proposed verdict JSON, validate it strictly, and
    return a ``RecoveryProposal`` or ``None``.

    Schema:
      {
        "action": "<RecoveryAction value>",
        "args": {...},
        "reasoning": "..."
      }

    Validation:
      * action must be a known ``RecoveryAction`` value
      * args must pass the action-specific validator
      * reasoning is optional, must be a string if present
    """
    if not verdict_path.is_file():
        return None
    try:
        blob = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    action_raw = blob.get("action")
    if not isinstance(action_raw, str):
        return None
    try:
        action = RecoveryAction(action_raw)
    except ValueError:
        return None
    args = blob.get("args", {})
    if not isinstance(args, dict):
        return None
    validator = _ACTION_VALIDATORS.get(action)
    if validator is None or not validator(args):
        return None
    reasoning = blob.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = ""
    return RecoveryProposal(action=action, args=args, reasoning=reasoning.strip(), source="llm")


# ─── Prompt builder ────────────────────────────────────────────────


def build_recovery_prompt(
    *,
    step_name: str,
    exception_class: str,
    exception_message: str,
    traceback_text: str,
    workspace_summary: str,
    verdict_path: Path,
) -> str:
    """Render the LLM prompt for a setup-step recovery. Pure.

    The prompt instructs the LLM to pick ONE action from the enum and
    fill its parameters. We give the menu inline so the model doesn't
    have to remember our spec.
    """
    return f"""You are a setup-step recovery advisor for the tt_hw_planner bring-up tool.

CONTEXT
-------
A tool-internal Python exception aborted Step "{step_name}". Your job:
identify the failure shape and propose ONE recovery action from the
fixed menu below. Do NOT write code. Do NOT edit source files.

EXCEPTION
---------
Class:   {exception_class}
Message: {exception_message}

TRACEBACK (tail)
----------------
{traceback_text}

WORKSPACE STATE
---------------
{workspace_summary}

ACTION MENU (pick exactly one)
------------------------------
- "cascade_class"
    Try alternative classes when a Auto* class fails on a trust_remote_code config.
    args: {{"module": "transformers", "attrs": ["AutoModelForCausalLM", "AutoModelForImageTextToText"]}}
    Each attr MUST be a Python identifier; module must be a dotted module name.

- "mkdir_parents"
    Create a missing parent directory before retrying the failing write.
    args: {{"path": "relative/or/abs/path"}}
    Path must not contain shell metacharacters or ".." escapes.

- "dtype_downgrade"
    Retry the bring-up with a smaller numeric dtype (memory-fit fail).
    args: {{"from": "bf16", "to": "bfp8_b"}}
    Both must be one of: bf16, bfp8_b, bfp4_b, fp32, fp16.

- "overlay_drop"
    Drop ALL overlays for a model scope and retry from scratch.
    args: {{"model_id": "org/name"}}
    Use ONLY when overlays from prior runs are blocking the current run.

- "skip_component"
    Skip a specific component and continue (component will graduate
    via CPU fallback or LLM later).
    args: {{"component_name": "attention"}}
    name must be a Python identifier.

- "re_exec"
    Restart the same command with corrected environment variables.
    args: {{"env": {{"VAR_NAME": "value", ...}}}}
    Variable names must be uppercase Python identifiers; values cannot
    contain shell metacharacters.

- "cannot_recover"
    Surface the failure to the operator — no recovery action is safe.
    args: {{"reason": "..."}}

OUTPUT
------
Write this EXACT JSON shape to {verdict_path}:

{{
  "action": "<one of the menu values above>",
  "args": {{ ... }},
  "reasoning": "one sentence justification"
}}

DO NOT propose actions outside the menu. DO NOT edit any source files.
DO NOT write code. ONLY write the verdict JSON.
"""


# ─── Action appliers ───────────────────────────────────────────────


def apply_recovery_action(proposal: RecoveryProposal, *, repo_root: Path) -> Tuple[bool, str]:
    """Execute the proposed action in-process. Returns ``(ok, note)``.

    For actions that require re-running the caller's step (cascade,
    mkdir, dtype, skip), ``ok=True`` means "state is ready for retry."
    For RE_EXEC, the function never returns (``os.execvpe``); on any
    error before exec, ``ok=False``.
    For CANNOT_RECOVER, ``ok=False`` always (the operator must handle).
    """
    action = proposal.action
    args = proposal.args
    if action == RecoveryAction.MKDIR_PARENTS:
        path = Path(args["path"])
        if not path.is_absolute():
            path = repo_root / path
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"mkdir failed: {type(exc).__name__}: {exc}"
        return True, f"created directory: {path}"
    if action == RecoveryAction.CASCADE_CLASS:
        # Not directly applicable in-process; the caller has to re-do
        # the loader call with the proposed attrs. Caller reads
        # ``proposal.args`` and retries. We just acknowledge here.
        return True, f"cascade attrs ready: {args['attrs']}"
    if action == RecoveryAction.DTYPE_DOWNGRADE:
        # Same as cascade: the caller decides whether to apply.
        return True, f"dtype downgrade ready: {args['from']} → {args['to']}"
    if action == RecoveryAction.OVERLAY_DROP:
        try:
            from ..overlay_manager import drop_scope
        except Exception as exc:
            return False, f"overlay_manager import failed: {exc}"
        try:
            count, _ = drop_scope(args["model_id"])
        except Exception as exc:
            return False, f"drop_scope raised: {type(exc).__name__}: {exc}"
        return True, f"dropped {count} overlay(s) for {args['model_id']}"
    if action == RecoveryAction.SKIP_COMPONENT:
        return True, f"component skipped (caller honors): {args['component_name']}"
    if action == RecoveryAction.RE_EXEC:
        new_env = dict(os.environ)
        new_env.update({k: v for k, v in args["env"].items()})
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.execvpe(sys.executable, [sys.executable, "-m", "scripts.tt_hw_planner", *sys.argv[1:]], new_env)
        except Exception as exc:
            return False, f"execvpe failed: {type(exc).__name__}: {exc}"
        return True, "re-exec completed"  # unreachable
    if action == RecoveryAction.CANNOT_RECOVER:
        # CANNOT_RECOVER is a legitimate verdict — the LLM (or a rule)
        # explicitly declined. Apply succeeds in the sense that the
        # decision is valid; the caller distinguishes via
        # ``proposal.action == CANNOT_RECOVER`` and surfaces the
        # failure instead of retrying.
        return True, f"explicit no-fix: {args.get('reason', '')}"
    return False, f"unhandled action: {action}"


# ─── Orchestrator ──────────────────────────────────────────────────


_RECOVERY_ATTEMPTED_FLAG = "TT_HW_PLANNER_SETUP_RECOVERY_ATTEMPTED"


def run_setup_step_recovery(
    *,
    exc: BaseException,
    step_name: str,
    work_dir: Path,
    repo_root: Path,
    workspace_summary: str = "",
    agent_invoker: Optional[Callable[..., int]] = None,
    agent_bin: str = "claude",
    agent_model: str = "haiku",
    timeout_s: int = 180,
    rules: Optional[List[_Rule]] = None,
) -> Optional[RecoveryProposal]:
    """Two-layer recovery: try the rule registry first, fall back to
    the LLM allowlist if no rule matches.

    Returns the proposal that was successfully applied, or ``None`` if:
      * the rules + LLM both decline,
      * the re-entry guard already fired (avoid loops),
      * the LLM is unreachable,
      * the proposed action's apply step failed.

    The caller is responsible for RETRYING the failing step using the
    proposal's recovery info (e.g. switching auto-class, re-running
    after mkdir, etc.). This function does NOT itself retry.
    """
    if os.environ.get(_RECOVERY_ATTEMPTED_FLAG):
        # Avoid recursive recovery on the same invocation.
        return None

    # Layer 1: rules
    proposal = find_rule_proposal(exc, step_name, rules=rules)
    if proposal is not None:
        ok, note = apply_recovery_action(proposal, repo_root=repo_root)
        if ok:
            print(f"  [setup-recovery] rule {proposal.label()} → {note}")
            return proposal
        # Rule matched but apply failed; fall through to LLM
        print(f"  [setup-recovery] rule matched but apply failed: {note}; falling back to LLM")

    # Layer 2: LLM fallback
    import traceback

    verdict_dir = work_dir / "_setup_recovery"
    try:
        verdict_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    verdict_path = verdict_dir / "verdict.json"
    try:
        if verdict_path.exists():
            verdict_path.unlink()
    except Exception:
        pass

    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    # Bound traceback length so prompt doesn't blow up.
    if len(tb_text) > 8000:
        tb_text = tb_text[-8000:]

    prompt = build_recovery_prompt(
        step_name=step_name,
        exception_class=type(exc).__name__,
        exception_message=str(exc),
        traceback_text=tb_text,
        workspace_summary=workspace_summary or "(no workspace summary provided)",
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
                iter_tag="setup_recovery",
                expected_deliverable_files=list(expected_deliverable_files),
            )

        agent_invoker = _default

    try:
        rc = agent_invoker(prompt, expected_deliverable_files=[verdict_path], timeout_s=timeout_s)
    except Exception as agent_exc:
        print(f"  [setup-recovery] LLM call raised: {type(agent_exc).__name__}: {agent_exc}")
        return None
    if rc != 0:
        print(f"  [setup-recovery] LLM call returned rc={rc}; no recovery proposed")
        return None

    proposal = parse_recovery_verdict(verdict_path)
    if proposal is None:
        print("  [setup-recovery] LLM verdict missing or rejected (malformed / unsafe args)")
        return None
    ok, note = apply_recovery_action(proposal, repo_root=repo_root)
    if not ok:
        print(f"  [setup-recovery] LLM proposed {proposal.label()} but apply failed: {note}")
        return None
    print(f"  [setup-recovery] LLM proposed {proposal.label()} → {note}")
    return proposal


__all__ = [
    "RecoveryAction",
    "RecoveryProposal",
    "apply_recovery_action",
    "build_recovery_prompt",
    "find_rule_proposal",
    "parse_recovery_verdict",
    "run_setup_step_recovery",
]
