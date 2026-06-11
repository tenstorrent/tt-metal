"""APPLY handler (PLAN 8.5) — REAL.

Record a clean git checkpoint, then the edit sub-agent applies the one selected
lever to the model files. The editor itself is injectable (ctx.deps["edit_runner"])
so this is testable without a key; the default is the live SDK editor.

In:  ctx.state["selected_lever"], manifest model_files, the model's git repo.
Out: ctx.state["git_sha_clean"] recorded, files edited, ctx.state["last_edit"]. -> VERIFY
"""

from __future__ import annotations

from .. import gitio, router, states


def apply(ctx) -> str:
    lever = ctx.state.get("selected_lever")
    repo = gitio.repo_root(ctx.model_root())
    ctx.state["git_sha_clean"] = gitio.head_sha(repo)  # record BEFORE editing (REVERT target)

    try:
        section = router.read_section(lever) if lever else ""
    except KeyError:
        section = ""  # lever id may not resolve to playbook text in tests; pass the id alone

    runner = ctx.deps.get("edit_runner") or _default_runner()
    result = runner(lever=lever, section=section, model_files=ctx.model_files())

    ctx.record_agent_call(states.APPLY, "edit", result.get("model", "?"), result.get("usage"))
    ctx.state["last_edit"] = {"files": result.get("files", []), "summary": result.get("summary", "")}
    return states.VERIFY


def _default_runner():
    from ..edit_agent import make_edit_runner

    return make_edit_runner()
