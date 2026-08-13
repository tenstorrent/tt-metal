# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Give a multi-stack model one depth knob per stack, by asking the agent that is already here.

WHY A KNOB PER STACK, and why a missing one is expensive rather than merely untidy.

A model with no depth argument at all is refused by the contract before the device is opened. The
case this handles is worse because it looks fine: the factory accepts `layers`, the clause passes,
and the value reaches exactly ONE stack. Every other stack builds at full depth.

Measured on Voxtral-Mini-3B, 2026-08-11/12. `layers` capped the text decoder and nothing else, so a
"2-layer" profile built 2 text layers behind two 32-layer audio encoders: 18729 ops, 35.2M tracy
zones, and a baseline killed at its budget with no BEFORE number for the run that followed. Capping
every stack took the same profile to 2471 ops and the build from 30+ minutes to 7 seconds.

WHY THE AGENT AND NOT AN AST REWRITE. Adding parameters to a signature is mechanical; making them do
anything is not. Threading a depth into a stack means finding where that stack is constructed and
what else derives from its count -- on Voxtral that was five separate places (n_layers, the routed
layer loop, layer_range, and both encoder truncations) plus a shared base class so the capped
encoder stayed discoverable. A rewriter that adds parameters it cannot wire produces the exact
failure this exists to remove: a knob that is accepted and ignored. The first two hand-written
attempts also built cleanly and died on the first forward, which is what a PCC gate catches and a
syntax check does not.

So the agent edits, and the tool checks the result the only way that cannot be fooled: it caps and
re-measures the work signal. A repair that does not move the op count is reported INERT exactly as
an unrepaired model is.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

_SYSTEM = (
    "You are editing a TTNN model pipeline so its profiling depth can be capped per stack. "
    "Make the smallest change that works. Do not restructure the model, do not touch stub bodies, "
    "and do not change numerics at full depth."
)


def stage_names(model_root) -> list:
    """PIPELINE_STAGES as the model declares it -- the override names come from the model itself."""
    p = Path(model_root) / "tt" / "pipeline.py"
    if not p.is_file():
        return []
    try:
        tree = ast.parse(p.read_text(errors="ignore"))
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PIPELINE_STAGES" for t in node.targets):
            try:
                return [str(v.value) for v in node.value.elts]
            except Exception:  # noqa: BLE001
                return []
    return []


def factory_params(model_root) -> set:
    """Names build_pipeline can actually RECEIVE -- its parameters, plus any it forwards by name.

    **KWARGS ALONE DOES NOT COUNT; **KWARGS PLUS AN ALLOWLIST DOES. The original Voxtral factory was
    `build_pipeline(device, model=None, **kwargs)` filtering to
    {batch_size, prefill_capacity, kv_capacity} -- so `layers` passed in by the harness was dropped
    without a word, which is the defect the depth clause exists to catch.

    But the repair's natural fix is to add the name to that same set, and then the argument DOES
    arrive: build_pipeline(device, layers=2) survives the filter and reaches the pipeline. Reading
    only the signature scored that as a failure twice on 2026-08-12 -- the agent had made the model
    cappable, the tool said "added nothing", and the re-measure that would have proved it never ran.

    So a name mentioned in a string-literal set inside the factory counts as received. That is a
    filter the code applies, not a comment: a name absent from it is still dropped, and a factory
    with no filter at all still fails on the signature alone.
    """
    p = Path(model_root) / "tt" / "pipeline.py"
    if not p.is_file():
        return set()
    try:
        tree = ast.parse(p.read_text(errors="ignore"))
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_pipeline":
            names = {a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)}
            if node.args.kwarg is not None:  # has **kwargs: an allowlist can forward by name
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Set, ast.List, ast.Tuple)):
                        names |= {e.value for e in sub.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            return names
    return set()


def missing_knobs(model_root, n_stacks: int = 1, stage_map=None) -> list:
    """Depth arguments this model needs and does not accept.

    THE BASE KNOB COMES FIRST AND IS NOT OPTIONAL. Without `layers` the builder has nowhere to put a
    depth at all, so every profile builds the whole model however many stacks it has -- measured on
    Voxtral with the variable unset: n_layers=30, enc_a=32, enc_b=32, bulk=27. That is the failure
    this exists to remove, and it is independent of how many stacks were found. An earlier version
    returned [] for a single-stack model, which is exactly the case that most needs the knob.

    PER-STAGE OVERRIDES ARE AN OPTIMISATION ON TOP, and only when the run actually knows which stage
    each stack ran in. Named from that mapping, never by position: slicing PIPELINE_STAGES by stack
    count asked Voxtral for `prefill_layers` when both of its visible stacks run in encode. With no
    mapping the honest answer is the base knob alone -- one depth reaching every stack already meets
    the goal, and a wrong name is worse than a missing one.
    """
    params = factory_params(model_root)
    if not params:
        return []  # no factory to change; the build-pipeline clause owns that
    want = []
    if not (params & {"layers", "n_layers", "num_layers", "depth"}):
        want.append("layers")
    if n_stacks > 1 and stage_map:
        for stage in sorted(stage_map):
            if not stage_map[stage]:
                continue
            name = "%s_layers" % stage
            if name not in params:
                want.append(name)
    return want


def repair_prompt(model_root, stacks: list, missing: list) -> str:
    """The task, with the stack paths the walk already found so nothing has to be guessed."""
    where = "\n".join("  - %s  (%d blocks, element type %s)" % (p, n, t) for p, n, t in stacks)
    return (
        "This TTNN pipeline has %d independent repeated block stacks, found by walking the object "
        "build_pipeline returns:\n%s\n\n"
        "build_pipeline currently accepts a single depth argument, so every stack is forced to one "
        "depth -- or worse, the value reaches one stack and the others build at FULL depth. That is "
        "what makes profiling expensive: on this class of model an uncapped encoder is the "
        "difference between 2471 and 18729 dispatched ops.\n\n"
        "Add these keyword arguments to build_pipeline and thread each one into the construction of "
        "its own stack: %s.\n\n"
        "Rules:\n"
        "  * None means 'fall back to `layers`'; `layers=None` still means EVERY layer.\n"
        "  * Never treat 0 as a sentinel -- a zero-layer build has no KV cache and dies before any "
        "timing marker.\n"
        "  * A capped build must remain a RUNNABLE MODEL. If capping would leave an aggregate "
        "sub-block holding zero layers, keep it at one.\n"
        "  * Where a stack keeps graduated stubs at specific indices, cap from the END so those "
        "bodies survive.\n"
        "  * Do not change behaviour when every argument is None.\n\n"
        "Edit tt/pipeline.py only. When done, confirm the file parses." % (len(stacks), where, ", ".join(missing))
    )


def _shortfall(model_root, missing: list) -> list:
    """Which requested arguments the FACTORY still does not accept."""
    params = factory_params(model_root)
    return [m for m in missing if m not in params]


def _retry_feedback(model_root, still: list) -> str:
    """Tell the agent exactly what the tool measured, not that it 'failed'.

    THE FIRST ATTEMPT ON VOXTRAL IS THE REASON THIS EXISTS. The agent wrote 40 good lines: a
    _cap_stack helper, both encoders trimmed, the tail kept so the graduated bodies at 28..31
    survive, and it worked out unprompted that the two towers are the same stack and must share a
    depth. It added `layers` to VoxtralPipeline.__init__ -- and not to build_pipeline, which is the
    only entry point the harness can reach. Everything was right except the door.

    A retry that just says "try again" invites the same edit. This names the ONE remaining fact: the
    factory's parameter list, as parsed, without the argument in it.
    """
    params = sorted(factory_params(model_root))
    return (
        "The edit is not reachable yet. build_pipeline's parameters are now: %s -- and the harness "
        "calls build_pipeline(device, ...) directly, so anything it does not accept never arrives.\n\n"
        "Still missing from THAT function's signature: %s.\n\n"
        "Add them to `def build_pipeline(...)` itself and pass them through to whatever you wired "
        "them to inside the pipeline. If build_pipeline filters **kwargs to a known set, add the "
        "names to that set as well -- a filtered kwargs dict drops them silently, which is the exact "
        "defect this repair exists to remove.\n\n"
        "Do not redo the work already done inside the pipeline; only make it reachable."
        % (", ".join(params) or "(none parsed)", ", ".join(still))
    )


def repair(model_root, stacks: list, missing: list, timeout_s: int = 1800, attempts: int = 3) -> dict:
    """Ask the agent to add the knobs, and keep asking with what the tool MEASURED.

    Returns {"attempted", "added", "params", "rounds"}.

    ONE SHOT WAS NOT ENOUGH, measured on Voxtral 2026-08-12: the first attempt produced a correct
    _cap_stack, capped both encoder towers, kept the tail, and put `layers` on the pipeline class
    instead of on the factory. The tool reported "added nothing" -- correctly, since the harness can
    only call build_pipeline -- and then gave up, discarding an edit that was one signature away from
    working.

    So each round re-reads the factory's parameters and, if the argument still is not there, hands
    that back as the next instruction. The feedback is a parsed fact, never a verdict: "these are the
    parameters build_pipeline now has; these are still missing". An agent told "it failed" repeats
    itself; an agent told what the signature says fixes the signature.

    What this cannot decide is whether the knob CAPS anything -- only whether it is reachable. That
    is settled by the caller re-measuring the work signal, which no edit can talk its way past.
    """
    from .agent_bin import resolve_claude_bin

    root = Path(model_root)
    if not missing or not (root / "tt" / "pipeline.py").is_file():
        return {"attempted": False, "added": [], "params": sorted(factory_params(root)), "rounds": 0}

    env = dict(os.environ)
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    rounds = 0
    prompt = repair_prompt(root, stacks, missing)
    for i in range(max(1, int(attempts))):
        cmd = [
            resolve_claude_bin(),
            "-p",
            prompt,
            "--system-prompt",
            _SYSTEM,
            "--allowedTools",
            "Read,Write,Edit,Glob,Grep",
            "--permission-mode",
            "bypassPermissions",
            "--max-turns",
            os.environ.get("PERF_MCP_KNOB_REPAIR_TURNS", "40"),
            "--output-format",
            "text",
        ]
        try:
            subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=timeout_s)
        except Exception:  # noqa: BLE001 -- a round that cannot run leaves the model as it was
            break
        rounds = i + 1
        still = _shortfall(root, missing)
        if not still:
            break
        prompt = _retry_feedback(root, still)
    params = factory_params(root)
    return {
        "attempted": True,
        "added": sorted(m for m in missing if m in params),
        "params": sorted(params),
        "rounds": rounds,
    }
