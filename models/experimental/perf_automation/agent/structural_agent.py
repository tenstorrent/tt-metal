"""structural-edit sub-agent — applies a COORDINATED multi-op optimization
(sharding / full-grid / trace) for levers tagged `lever_type: structural`, where a
single config kwarg is not enough. Gets the per-op fingerprints + op->source
attribution, is fenced to the executed files, and self-verifies by re-reading.
Ground truth stays the deterministic GATE_PCC + REMEASURE. Same SDK seam as
edit_agent.make_edit_runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

# When on-device tools (check_candidate_edit / measure_candidate) are attached, the agent makes
# MANY minutes-long device calls (a tracy profile is ~2-3 min; the kernel author made 8 in one
# session). The 300s hang-watchdog default then kills it mid-iteration AND retries from scratch.
# Use a far larger wall budget for device-tool agents — still finite, so a true hang is bounded.
_DEVICE_CALL_TIMEOUT_S = float(os.environ.get("AGENT_DEVICE_CALL_TIMEOUT_S", "3600"))

# An on-device-tool agent spends many TURNS per author->check->measure cycle (edit, tool call,
# read result, rethink); the default 60 ran out mid-iteration before it could emit final JSON
# (observed: 5 measure cycles exhausted 60 turns). Give it a larger turn budget when tools attach.
_DEVICE_MAX_TURNS = int(os.environ.get("AGENT_DEVICE_MAX_TURNS", "200"))

STRUCTURAL_TEMPLATE = (
    "You are applying ONE *structural* performance optimization to a TTNN model, then stopping.\n\n"
    "Optimization (lever '{lever}'):\n{section}\n\n"
    "Hottest individual ops in this bucket:\n{top_ops}\n\n"
    "WHERE the hot ops are EMITTED (op->source attribution — edit THESE exact source lines,\n"
    "they are ranked by how much matmul work they emit; the top one is where the dominant\n"
    "matmuls actually execute — NOT necessarily the 'obvious' FFN/attention stub):\n{hot_sources}\n\n"
    "Edit ONLY these files (the executed path; Read them first):\n{files}\n\n"
    "THIS IS A COORDINATED, MULTI-SITE CHANGE — a single config kwarg is NOT enough:\n"
    "  1. Convert the INPUT activation tensor to a sharded L1 `memory_config` BEFORE the op\n"
    "     (e.g. `ttnn.to_memory_config(x, sharded_cfg)` / `ttnn.create_sharded_memory_config(...)`).\n"
    "     Setting only a `program_config` on a tensor that is still DRAM_INTERLEAVED does NOTHING\n"
    "     — the kernel graph is unchanged and the edit is inert. Sharding is a property of the\n"
    "     TENSOR, not the matmul call.\n"
    "  2. Give the op the matching `program_config` + full core grid\n"
    "     (`device.compute_with_storage_grid_size()` — never hard-code the grid).\n"
    "  3. Keep the OUTPUT sharded so the next op consumes it without a reshard back to DRAM.\n\n"
    "FORCE-TRY: make the full change; do NOT ship a partial 'safe' version (that is the inert\n"
    "no-op above). The downstream PCC gate will catch any correctness regression — rely on it.\n\n"
    "SELF-VERIFY before finishing (Read-only — do NOT run the model/device):\n"
    "  - Re-Read each file you edited and CONFIRM your edit added a tensor `memory_config`\n"
    "    conversion (to_memory_config / create_sharded_memory_config / interleaved_to_sharded)\n"
    "    ON THE EXECUTED CALL PATH (the method the forward actually runs), not merely a\n"
    "    `program_config=` kwarg and not in a dead/unused helper. If it only added a\n"
    "    program_config, or sits in code the forward doesn't call, the edit is INERT — fix it.\n\n"
    "When done, output exactly ONE JSON object and nothing else:\n"
    '  {{"files": [<repo-relative paths you changed>], "summary": <one sentence on the coordinated change>}}'
)


def _format_top_ops(top_ops: list[dict] | None) -> str:
    if not top_ops:
        return "  (per-op detail unavailable — target the bucket's dominant op)"
    lines = []
    for o in top_ops:
        extra = ""
        if o.get("ideal_ms") is not None:  # ROOFLINE evidence: how far off the hardware floor + which wall
            extra = f"  ROOFLINE: ideal={o.get('ideal_ms')}ms GAP={o.get('gap_ms')}ms " f"bound_by={o.get('bound_by')}"
        lines.append(
            f"  - {o.get('op_code','?')} [{o.get('shape','?')}] ×{o.get('count','?')}: "
            f"{o.get('device_ms',0):.3f}ms total, {o.get('cores','?')} cores ({o.get('grid','?')}), "
            f"mem={o.get('memory','?')}, fidelity={o.get('fidelity','?')}{extra}"
        )
    return "\n".join(lines)


# FROM-PRINCIPLES template: used when a hot bucket has NO playbook lever (conv/scan/moe/
# other/kv_cache/etc.). No prescribed recipe — give the agent the roofline gap + a MENU of
# ttnn primitives and make it DIAGNOSE from the gap which wall it's hitting, then close it.
# This is the model-agnostic, think-don't-pattern-match path.
FROM_PRINCIPLES_TEMPLATE = (
    "You are optimizing ONE hot operation on Tenstorrent hardware from FIRST PRINCIPLES,"
    " then stopping. There is no pre-written recipe — you must diagnose and decide.\n\n"
    "Hottest ops in this bucket (with ROOFLINE gap = how far above the hardware floor, and"
    " bound_by = which wall they hit):\n{top_ops}\n\n"
    "WHERE they are emitted (op->source attribution — edit THESE exact lines):\n{hot_sources}\n\n"
    "Edit ONLY these files (the executed path; Read them first):\n{files}\n\n"
    "DIAGNOSE from the gap + bound_by, then apply the matching ttnn primitive(s):\n"
    "  - bound_by=memory (DRAM/L1 bandwidth): shrink bytes moved -> lower-precision dtype"
    " (bfloat8_b/bfloat4_b weights via dtype/weights_dtype), shard the input into L1"
    " (ttnn.to_memory_config to a sharded config), keep outputs sharded to avoid reshards,"
    " remove redundant tilize/untilize/typecast.\n"
    "  - bound_by=compute (FLOPs): the op is far from its FLOP ceiling because the grid is"
    " under-occupied (cores << full grid) or fidelity is higher than needed -> occupy the"
    " full core grid (device.compute_with_storage_grid_size(), matching program_config) and"
    " drop MATH_FIDELITY to the lowest the PCC gate tolerates.\n"
    "  - tiny op / huge gap with near-zero ideal (dispatch/launch bound): FUSE it into a"
    " neighbor (activation into the matmul, elementwise chains, norm+residual) so it stops"
    " being a separate dispatched kernel.\n"
    "  - if NO ttnn primitive closes the gap (the op library can't express the needed fusion or"
    " dataflow): author a custom tt-lang (ttl) kernel for THIS op — see the GUIDELINES"
    " '#tt-lang-kernel' section for the API — preserving the op's I/O dtype/layout contract so the"
    " rest of the graph still stitches. This is the deepest move; use it only when the ttnn"
    " primitives above are exhausted.\n\n"
    "RULES: change the TENSOR's memory_config/dtype, not merely a program_config kwarg on a"
    " DRAM tensor (that is a no-op). Make the real change on the EXECUTED call path; the PCC"
    " gate will catch any correctness regression so FORCE-TRY rather than shipping a safe"
    " no-op. Read your edit back to confirm it adds a real device op (a new memory_config /"
    " dtype / fused op), not just a kwarg.\n\n"
    "When done, output exactly ONE JSON object and nothing else:\n"
    '  {{"files": [<repo-relative paths you changed>], "summary": <one sentence on what you did and why>}}'
)


# KERNEL template: the tt-lang-kernel lever is lever_type:structural so it lands on THIS agent,
# but it must AUTHOR a ttl kernel — NOT do the sharding edit STRUCTURAL_TEMPLATE prescribes. (Found
# via an isolated test: with the sharding template the agent did ttnn.to_memory_config sharding,
# never a kernel.) The {section} carries the ttl API + a proven matmul template to ADAPT.
KERNEL_TEMPLATE = (
    "You are AUTHORING a custom tt-lang (`ttl`) kernel for ONE hot op, then stopping. This is NOT a "
    "sharding / memory_config / program_config edit — you write an actual ttl kernel and route the "
    "op through it.\n\n"
    "Kernel playbook (the ttl API + a PROVEN, PCC-correct template to ADAPT — do NOT write from a "
    "blank page):\n{section}\n\n"
    "Hottest ops in this bucket (author the kernel for the dominant one):\n{top_ops}\n\n"
    "WHERE the hot op executes (op->source attribution — replace THAT call site):\n{hot_sources}\n\n"
    "Edit ONLY these files (Read them first):\n{files}\n\n"
    "STEPS:\n"
    "  1. ADAPT the proven matmul template in the playbook to this op's shape (drop the bias/relu if "
    "the op has none; match dtype). OCCUPY THE GRID: a single-core grid=(1,1) kernel is correct but "
    "SLOWER than the stock op and WILL be rejected — distribute the output tiles across the device "
    "compute grid.\n"
    "  2. Replace the op's call site so the model executes YOUR ttl kernel, PRESERVING the op's I/O "
    "dtype/layout/memory_config contract (so the rest of the graph still stitches).\n"
    "  3. VALIDATE with `check_candidate_edit` before finishing: on FAIL (compile / crash / low PCC), "
    "fix it and re-check; only finish once it PASSES.\n\n"
    "When done, output exactly ONE JSON object: "
    '{{"files": [<repo-relative paths you changed>], "summary": <one sentence: the ttl kernel you wrote>}}'
)


# DECODE template: the structural-decode lever fires ONLY when the gate detects an
# autoregressive decode loop that re-runs a growing-sequence prefill every token (no
# cached decode_step / KV-cache) — the repeat_prefill signal. It is a multi-site
# restructure of the decode loop, not an op knob, so it lands on this structural agent
# but must AUTHOR a cached single-token decode path, NOT the sharding recipe.
DECODE_TEMPLATE = (
    "This model's decode is REPEAT-PREFILL: an autoregressive loop that re-runs the full "
    "growing-sequence forward for every generated token, with NO cached single-token "
    "`decode_step` and NO KV-cache. That is the dominant cost (host-dispatch-bound: most of "
    "the per-token time is re-slicing/re-computing the whole prefix). Your job is the "
    "STRUCTURAL decode rewrite — add a cached single-token decode path — then stop.\n\n"
    "STEPS:\n"
    "  1. Add a KV-cache + a single-token `decode_step`: after the one-time prefill of the "
    "prompt, each subsequent token must attend to CACHED keys/values and run attention on the "
    "ONE new token (seq_len=1), not re-prefill the whole sequence. Append the new K/V to the "
    "cache each step.\n"
    "  2. PRESERVE the decode loop's I/O contract exactly (same output tokens/logits, same "
    "dtype/layout at the boundary) so the rest of the pipeline still stitches and PCC holds.\n"
    "  3. VALIDATE with `check_candidate_edit` before finishing: it is kept ONLY if PCC-clean "
    "AND faster; on FAIL (wrong tokens / slower / crash) fix or revert, and re-check.\n\n"
    "WHERE the decode loop executes (op->source attribution — restructure THAT loop):\n{hot_sources}\n\n"
    "Hottest ops in the repeat-prefill decode:\n{top_ops}\n\n"
    "Edit ONLY these files (Read them first):\n{files}\n\n"
    "When done, output exactly ONE JSON object: "
    '{{"files": [<repo-relative paths you changed>], "summary": <one sentence: the KV-cache/decode_step you added>}}'
)


def build_structural_prompt(
    lever: str, section: str, model_files: list, top_ops: list[dict] | None, hot_sources: list[dict] | None = None
) -> str:
    from .op_attribution import format_hot_sources

    files = "\n".join(f"  - {f}" for f in model_files)
    if lever == "structural-decode":  # gate detected repeat_prefill -> add cached decode_step/KV-cache
        return DECODE_TEMPLATE.format(
            top_ops=_format_top_ops(top_ops),
            hot_sources=format_hot_sources(hot_sources or []),
            files=files,
        )
    if not (section or "").strip():  # FROM_PRINCIPLES: no playbook section -> think from the roofline gap
        return FROM_PRINCIPLES_TEMPLATE.format(
            top_ops=_format_top_ops(top_ops),
            hot_sources=format_hot_sources(hot_sources or []),
            files=files,
        )
    if lever == "tt-lang-kernel":  # AUTHOR a ttl kernel — NOT the sharding recipe below
        return KERNEL_TEMPLATE.format(
            section=section,
            top_ops=_format_top_ops(top_ops),
            hot_sources=format_hot_sources(hot_sources or []),
            files=files,
        )
    return STRUCTURAL_TEMPLATE.format(
        lever=lever or "(unspecified)",
        section=section,
        top_ops=_format_top_ops(top_ops),
        hot_sources=format_hot_sources(hot_sources or []),
        files=files,
    )


def make_structural_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 60,
) -> Callable[..., dict]:
    """Build the structural editor: runner(lever, section, model_files, top_ops) -> result.

    Mirrors edit_agent.make_edit_runner but (self-verify by re-reading, no device run),
    a bigger turn budget (coordinated edit + verify loop), and the structural model tier.
    Result parsing is LENIENT: APPLY uses git-diff as ground truth, so a missing/!JSON
    final message yields files=[] rather than raising.
    """
    from .config import agent_effort, apply_agent_env, get_edit_model

    resolved = apply_agent_env(env_agent_path)

    def runner(
        *,
        lever: str,
        section: str,
        model_files: list,
        top_ops: list[dict] | None = None,
        hot_sources: list[dict] | None = None,
        error: str | None = None,
        spec: dict | None = None,
        cwd: str | None = None,
        attempt: int = 0,
        validate: "Callable[[], dict] | None" = None,
        measure: "Callable[[], dict] | None" = None,
        baseline_ms: float | None = None,
    ) -> dict:
        # Escalation ladder: APPLY (attempt 0) -> haiku, repair 1 -> sonnet, repair 2+ -> opus.
        model = get_edit_model(attempt, resolved)

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        from .probes import _extract_json_object, _usage_summary

        files = [str(p) for p in model_files]
        prompt = build_structural_prompt(lever, section, files, top_ops, hot_sources)
        if error:  # REPAIR: prepend the failure, keep the structural protocol
            prompt = f"Your previous structural edit for '{lever}' FAILED:\n{error}\n\n" + prompt

        # Optional on-device validation tool (the verified gap): without it the structural agent
        # must self-verify by re-reading only; with it, it can actually TEST the edit (catches the
        # invalid core_grid / degenerate-capture crashes) before submitting.
        allowed = ["Read", "Edit", "Glob", "Grep"]
        mcp_servers: dict = {}
        check_server = None
        if validate is not None:
            from .edit_check import EDIT_CHECK_SERVER, EDIT_CHECK_TOOL, _PROMPT_NOTE, make_edit_check_server

            check_server = make_edit_check_server(validate)
            if check_server is not None:
                allowed = allowed + [EDIT_CHECK_TOOL]
                prompt = prompt + _PROMPT_NOTE
                mcp_servers[EDIT_CHECK_SERVER] = check_server

        # Perf-feedback tool — ONLY for the kernel lever: a kernel must be FASTER, not just
        # correct, so the agent needs to measure its kernel vs the baseline and iterate. Knob
        # edits skip this (REMEASURE already times them); profiling is too costly to run per-edit.
        if measure is not None and lever == "tt-lang-kernel":
            from .perf_check import (
                PERF_CHECK_SERVER,
                PERF_CHECK_TOOL,
                _PROMPT_NOTE as _PERF_NOTE,
                make_perf_check_server,
            )

            perf_server = make_perf_check_server(measure, baseline_ms)
            if perf_server is not None:
                allowed = allowed + [PERF_CHECK_TOOL]
                prompt = prompt + _PERF_NOTE
                mcp_servers[PERF_CHECK_SERVER] = perf_server
        verify_clause = (
            "validate the edit with `check_candidate_edit` (runs the correctness test safely)"
            if check_server is not None
            else "Self-verify by RE-READING (do NOT run the model or device — that deadlocks the profiler that holds it)"
        )

        opts: dict = dict(
            model=model,
            system_prompt=(
                "You apply exactly one STRUCTURAL optimization (sharding / grid / layout) to "
                "TTNN model source using Read, Edit, Grep, Glob. A structural edit is multi-site: "
                "you MUST change the tensor's memory_config, not just a program_config, and it must "
                f"land on the EXECUTED call path. {verify_clause}. Your FINAL message must be one "
                "JSON object, no prose. Stay inside the working directory."
            ),
            allowed_tools=allowed,
            permission_mode="bypassPermissions",
            setting_sources=[],
            # device-tool agents need many turns per author->check->measure cycle; bump from default
            max_turns=max(max_turns, _DEVICE_MAX_TURNS) if mcp_servers else max_turns,
            max_buffer_size=50 * 1024 * 1024,
            effort=agent_effort(resolved),  # cap reasoning so the structural editor doesn't think-for-minutes
        )
        if mcp_servers:
            opts["mcp_servers"] = mcp_servers
        if cwd:
            opts["cwd"] = cwd
        options = ClaudeAgentOptions(**opts)
        chunks: list[str] = []
        usage: dict = {}

        async def _go() -> None:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)
                elif isinstance(msg, ResultMessage):
                    usage["u"] = _usage_summary(msg)

        from .sdk_retry import run_with_retry

        # On-device tools make the agent legitimately long-running (each profile/PCC run is minutes);
        # give it a generous wall budget instead of the 300s hang default, which killed + retried it
        # mid-iteration. No device tools -> default (a true stall is still bounded).
        call_timeout = _DEVICE_CALL_TIMEOUT_S if mcp_servers else None
        run_with_retry(_go, lambda: (chunks.clear(), usage.clear()), timeout=call_timeout)

        response = "\n".join(chunks)
        files_out, summary = [], ""
        try:  # LENIENT: APPLY falls back to git-diff if files is empty
            import json

            obj = json.loads(_extract_json_object(response))
            if isinstance(obj, dict):
                files_out = [str(f) for f in (obj.get("files") or [])]
                summary = str(obj.get("summary", ""))
        except Exception:
            pass
        return {
            "files": files_out,
            "summary": summary,
            "model": model,
            "usage": usage.get("u"),
            "prompt": prompt,
            "response": response,
        }

    return runner
