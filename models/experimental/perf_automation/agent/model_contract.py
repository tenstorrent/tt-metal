# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a model must provide before it can be optimized -- checked by READING it, before anything runs.

THE TOOL HAS BEEN ADAPTING TO MODELS INSTEAD OF STATING WHAT IT NEEDS. Every requirement here was
already written down somewhere -- perf_adapter's docstring describes the per-stage hooks, emit_e2e's
prompt specifies build_pipeline's signature -- as PROSE, in the place that consumes it. Prose is
checked by whoever happens to read it, which is how a model reaches the device missing a clause and
the tool discovers it forty minutes in, as a crash with no obvious connection to the cause.

The case that produced this file: gemma-3's prefill decides its own traced-vs-eager, from an
allow-list inside the model, while decode is controlled by the harness. The tool asks for eager
before profiling -- the profiler measures per-op time from eager dispatch, and a traced region emits
none -- and prefill traced anyway. 194 fatals, no profiling data, no baseline, and a run that had
already spent minutes on the device. Nothing about that failure names its cause; the C++ says
"Event Synchronization is not supported during trace capture".

That clause is visible in the SOURCE. `can_enable_trace` consults `trace_prefill_supported_seq_lens`
and nothing else -- no harness signal reaches it. A reader can see that in seconds, and so can this.

WHY STATIC FIRST. The dynamic checks (does the depth cap actually reduce op count, does the census
complete, does batch reach the stage) need a built model and a device. These need a file. Running
them first means a model in the wrong shape is turned away before the perf test is generated, before
the weights load, before the board is touched -- and the answer is a list of what to change rather
than a stack trace.

WHAT THIS DELIBERATELY DOES NOT DO. It does not judge whether the model is FAST, or correct, or
whether its numbers are good. It answers one question: can this model be measured the way the tool
measures? A clause here must be something a model can comply with by construction, and something a
reader of the source can confirm -- otherwise it belongs in the dynamic tier, where behaviour is
observed rather than inferred.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import stage_seams as _seams

# A stage is measured BOTH ways: eager for per-op profiling, traced for end-to-end latency. So every
# stage needs both paths, and the CHOICE has to belong to the harness -- one authority, not one per
# stage. These are the names the harness sets; a model that reads any of them is participating.
HARNESS_TRACE_SIGNALS = ("TT_PERF_TRACE", "TT_METAL_DEVICE_PROFILER")


@dataclass(frozen=True)
class Finding:
    """One unmet clause. `remedy` is the porting task, stated so it can be acted on without
    rediscovering the reasoning -- it is the whole point of checking early."""

    clause: str
    detail: str
    remedy: str
    severity: str = "error"  # error = cannot be measured; warn = measurable, worse
    # COMPATIBILITY vs PORTING, and only the first can block.
    #
    # A model EMITTED by emit-e2e satisfies the porting clauses by construction -- PIPELINE_STAGES,
    # the per-stage hooks, the self-tests are its output. But optimize is also run DIRECTLY on
    # hand-written tt-metal models that never went through it, and those legitimately lack that
    # shape: gemma-3 and llama3_1_8b_p150 are both in that category. Refusing them for not looking
    # like emit-e2e's output would refuse the entire direct path.
    #
    # What is never acceptable, emitted or not, is a model that FIGHTS the harness: a trace gate the
    # harness cannot reach, a depth cap of 0, a factory that runs the model instead of returning it.
    # Those break the tool no matter how the model was written, and they are what may block.
    kind: str = "compatibility"

    @property
    def blocking(self) -> bool:
        """Only a COMPATIBILITY error stops a run. A porting gap is reported and stepped over."""
        return self.severity == "error" and self.kind == "compatibility"

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return "%s [%s] %s\n      -> %s" % (
            "FAIL" if self.blocking else ("port" if self.kind == "porting" else "warn"),
            self.clause,
            self.detail,
            self.remedy,
        )


@dataclass
class Source:
    """The model's python sources, parsed once. Files that do not parse are reported, not skipped:
    a clause that cannot be checked has not been met, and silence would read as compliance."""

    root: Path
    trees: dict = field(default_factory=dict)  # path -> ast.Module
    texts: dict = field(default_factory=dict)  # path -> str
    unparsed: list = field(default_factory=list)

    @classmethod
    def load(cls, model_root, max_files: int = 4000) -> "Source":
        s = cls(root=Path(model_root))
        for p in sorted(s.root.rglob("*.py"))[:max_files]:
            # Generated perf tests are the TOOL's output, not the model's: judging the model by a
            # file the tool wrote would make the contract self-satisfying.
            if p.name.startswith("test_main_perf") or "__pycache__" in p.parts:
                continue
            try:
                txt = p.read_text(errors="ignore")
                s.texts[p] = txt
                s.trees[p] = ast.parse(txt)
            except SyntaxError as exc:
                s.unparsed.append((p, "%s line %s" % (exc.msg, exc.lineno)))
            except OSError:
                continue
        return s

    def functions(self, name: str):
        """(path, node) for every def/async def with this name."""
        for p, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    yield p, node

    def assigns(self, name: str):
        for p, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    yield p, node

    def mentions(self, text: str) -> bool:
        return any(text in t for t in self.texts.values())


def _segment(src: Source, path: Path, node) -> str:
    """The source text of one function, for asking what it reads."""
    try:
        lines = src.texts[path].splitlines()
        return "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- the clauses


def _c_parses(src: Source) -> list:
    if not src.trees and not src.unparsed:
        return [
            Finding(
                "sources",
                "no python sources found under %s" % src.root,
                "point --model-dir at the directory holding the model's implementation",
            )
        ]
    return [
        Finding(
            "sources",
            "%s does not parse (%s)" % (p.name, why),
            "fix the syntax error; a file the tool cannot read is a clause it cannot check",
        )
        for p, why in src.unparsed
    ]


def _c_build_pipeline(src: Source) -> list:
    """The single entry point the harness builds through.

    `layers=None` means ALL layers and must never be 0: a literal 0 arrives truthy in an env var and
    has been read by builders as "build zero layers", which measures nothing and reports no markers.
    """
    found = list(src.functions("build_pipeline"))
    if not found:
        return [
            Finding(
                "build-pipeline",
                "no module-level build_pipeline(device, ...) found",
                "expose build_pipeline(device, model=None, layers=None, **kwargs) returning the "
                "pipeline object -- the harness opens the device ONCE and passes it in",
                kind="porting",
            )
        ]
    out = []
    for p, fn in found:
        args = [a.arg for a in fn.args.args]
        if not args or args[0] not in ("device", "dev", "mesh_device"):
            out.append(
                Finding(
                    "build-pipeline",
                    "%s: first parameter is %r, not the device" % (p.name, args[0] if args else "(none)"),
                    "take the device as the first positional parameter; the test fixture is the sole "
                    "device opener, and a second open has no memory left for its KV cache",
                )
            )
        if not fn.args.kwarg:
            out.append(
                Finding(
                    "build-pipeline",
                    "%s: no **kwargs" % p.name,
                    "accept **kwargs so the harness can pass knobs this model does not know about "
                    "without the call failing",
                    severity="warn",
                )
            )
        for a, d in zip(reversed(args), reversed(fn.args.defaults or [])):
            if a == "layers" and isinstance(d, ast.Constant) and d.value == 0:
                out.append(
                    Finding(
                        "build-pipeline",
                        "%s: layers defaults to 0" % p.name,
                        "default layers to None (= all layers). 0 arrives truthy from an env var and "
                        "has been read as 'build zero layers', which measures nothing",
                    )
                )
    return out


def _c_stages(src: Source) -> list:
    """PIPELINE_STAGES plus the per-stage hooks, because the tool measures each stage separately."""
    declared = None
    for _p, node in src.assigns("PIPELINE_STAGES"):
        if isinstance(node.value, (ast.List, ast.Tuple)):
            declared = [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            break
    if declared is None:
        if src.mentions("def decode_step"):
            return [
                Finding(
                    "stages",
                    "no PIPELINE_STAGES; only the legacy single-step decode contract",
                    "declare PIPELINE_STAGES = ['prefill', 'decode', ...]. The legacy path still "
                    "works but collapses to ONE stage, so prefill and decode cannot be told apart "
                    "-- and they are not the same currency: decode recurs per token, prefill once "
                    "per request",
                    severity="warn",
                    kind="porting",
                )
            ]
        return [
            Finding(
                "stages",
                "no PIPELINE_STAGES and no decode_step",
                "declare PIPELINE_STAGES and expose <stage>%s(inputs) / <stage>%s() " % _seams.REQUIRED
                + "for each; without them there is nothing the harness can measure",
                kind="porting",
            )
        ]
    out = []
    for st in declared:
        # THE ITEM COUNT IS THE COMPUTE CEILING'S ONLY INPUT, and its absence is silent. The roofline
        # prices a stage at 2 x params x items, and a stage that states nothing is priced at ONE item
        # -- which is right for a recurring step and 1500x wrong for an audio tower over 1500 frames,
        # with nothing to tell the two apart downstream. Measured on voxtral 2026-08-27: encode's
        # compute roof read 0.007 ms against a 238.79 ms measurement, so the stage was reported
        # memory-bound when it is compute-bound, and the reader had no way to see that the number was
        # a placeholder rather than a finding.
        #
        # WARN AND PORTING, never blocking. A hand-written model that never went through emit-e2e is
        # a first-class input here, and a missing optional seam must not refuse it -- it only means
        # that stage's arithmetic ceiling is a placeholder, which is worth saying out loud on every
        # run rather than discovering from an implausible utilisation figure weeks later.
        _items_hook = _seams.hook(st, _seams.ITEMS)
        if not (list(src.functions(_items_hook)) or src.mentions(_items_hook)):
            out.append(
                Finding(
                    "stage-items",
                    "stage %r states no item count (%s missing) -- its compute ceiling is a "
                    "placeholder of 1 item" % (st, _items_hook),
                    "expose %s(): ZERO-ARG, returning how many items ONE %s_trace_step call retires, "
                    "batch included -- count what the stage's repeated blocks process, not what it "
                    "returns. A stage that genuinely retires one item should return 1 explicitly, so "
                    "'one item' is a statement rather than a default." % (_items_hook, st),
                    severity="warn",
                    kind="porting",
                )
            )
        for hook in (_seams.hook(st, _s) for _s in _seams.REQUIRED):
            if not (list(src.functions(hook)) or src.mentions(hook)):
                out.append(
                    Finding(
                        "stages",
                        "stage %r declared but %s is missing" % (st, hook),
                        "expose %s -- setup does host prep OUTSIDE the trace, step is one "
                        "fixed-shape host-op-free call reading only resident buffers" % hook,
                    )
                )
    return out


def _c_trace_authority(src: Source) -> list:
    """THE CLAUSE THIS FILE EXISTS FOR: the harness chooses traced or eager, for every stage.

    A stage must support BOTH -- eager so the profiler can attribute per-op device time, traced so
    end-to-end latency is measured the way the model ships -- and the choice must come from outside.
    A model that decides for itself cannot be profiled: the tool asks for eager, the model traces,
    and the capture dies with "Event Synchronization is not supported during trace capture".

    Checked by asking what the model's own trace gate READS. gemma-3's consults an allow-list and
    nothing else, so no harness signal can reach it -- visible without running anything.
    """
    out = []
    gates = list(src.functions("can_enable_trace")) + list(src.functions("get_trace_prefill_supported_seq_lens"))
    if not gates:
        return out  # no model-side trace gate: the harness is the only authority, which is the goal
    reads_signal = False
    for p, fn in gates:
        seg = _segment(src, p, fn)
        if any(sig in seg for sig in HARNESS_TRACE_SIGNALS):
            reads_signal = True
    if not reads_signal:
        names = sorted({fn.name for _p, fn in gates})
        out.append(
            Finding(
                "trace-authority",
                "%s %s tracing from model state alone; no harness signal (%s) reaches %s"
                % (
                    ", ".join(names),
                    "decides" if len(names) == 1 else "decide",
                    " / ".join(HARNESS_TRACE_SIGNALS),
                    "it" if len(names) == 1 else "them",
                ),
                "have the trace gate consult the harness: return no traceable shapes when "
                "TT_METAL_DEVICE_PROFILER=1 or TT_PERF_TRACE=0. The profiler measures per-op time "
                "from EAGER dispatch -- a traced region emits none, and syncing inside a capture is "
                "fatal, so 'profiler on' and 'traced' cannot both be true",
            )
        )
    return out


def _c_depth_knob(src: Source) -> list:
    """A depth cap the FACTORY accepts, so coverage can be profiled at a fraction of the model.

    THE OLD CHECK WAS A REGEX FOR `layers=` ANYWHERE IN THE MODEL, and that is how Voxtral-Mini-3B
    passed it while the knob did nothing at all. The string occurs in any pipeline that writes
    `n_layers = cfg.num_hidden_layers`; meanwhile build_pipeline's signature was
    `(device, model=None, **kwargs)` and it filtered kwargs to {batch_size, prefill_capacity,
    kv_capacity}, dropping `layers` silently. The generated perf test recorded the consequence in
    its own comment -- "No depth argument on this builder" -- and every profile built all 32 layers.

    The factory's SIGNATURE is the checkable thing: emit-e2e specifies `layers` as a build argument,
    so a builder that does not accept one cannot honour a cap however many times the string appears.
    **kwargs alone does not count: it is what silently swallowed the argument.
    """
    fac = list(src.functions("build_pipeline"))
    if not fac:
        return []  # a missing factory is the build-pipeline clause's finding, not this one
    for _path, fn in fac:
        names = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
        if names & {"layers", "n_layers", "num_layers", "depth"}:
            return []
    return [
        Finding(
            "depth-knob",
            "build_pipeline does not accept a depth argument",
            "accept `layers` (None = every layer, never 0) and thread it into the block count of "
            "EVERY repeated stack. Without it every profile builds the whole model: on a 3B "
            "multimodal pipeline that is 35M tracy zones and a baseline killed at its budget. "
            "**kwargs does not satisfy this -- a filtered kwargs dict is what dropped it silently",
            kind="compatibility",
        )
    ]


def _c_selftests(src: Source) -> list:
    """The model's OWN proofs, which emit_e2e specifies and nothing was checking.

    trace_capture_selftest(device) is the answer to "does every stage work traced": for EACH stage it
    captures one step, executes it, releases the trace before the next (stage traces must not
    co-reside), and returns True only if every stage captured host-free AND matched its reference by
    PCC. That is precisely the question a harness cannot answer from outside -- it can observe that a
    stage traced, but not that the traced output is still correct.

    host_op_selftest() is the authoritative fully-on-device check: ttnn ops do not dispatch through
    torch, so a genuinely on-device forward fires ZERO host aten ops. Anything it does fire is host
    compute that the ttnn-crossing heuristics cannot see -- a host-built prefix embedding uploaded via
    as_tensor, an HF submodule forward, sampling left on the host.

    Both are specified in emit_e2e's contract and both are checkable by name here; whether they PASS
    is the dynamic tier's question.
    """
    out = []
    for fn, why, remedy in (
        (
            "trace_capture_selftest",
            "no per-stage trace/PCC self-test",
            "expose trace_capture_selftest(device): for EACH stage capture one step, execute_trace "
            "it, RELEASE it before the next, and return True only if every stage captured host-free "
            "and matched its reference by PCC. Without it, 'this stage can be traced correctly' is "
            "assumed rather than proven",
        ),
        (
            "host_op_selftest",
            "no fully-on-device check",
            "expose host_op_selftest(): run the forward under host_op_observer.observe_host_ops() "
            "with encoding and weight-build OUTSIDE the observed region, and return "
            "host_op_observer.verdict(ops). A truly on-device forward fires ZERO host aten ops",
        ),
    ):
        if not (list(src.functions(fn)) or src.mentions("def %s" % fn)):
            out.append(Finding("selftests", why, remedy, severity="warn", kind="porting"))
    return out


def declared_stage_names(model_root) -> list:
    """The stages this model says it has, from its own PIPELINE_STAGES. [] when it declares none.

    THE ONE READER. Three places walked the AST for this assignment independently, and anything that
    wanted the answer either copied the walk or made do with a guess. A stage list is the model's own
    statement about itself and is what every stage-keyed thing in the tool should be checked against
    -- which stages exist, which regime tags are legal, which depth knobs get set.
    """
    import ast as _ast

    try:
        src = Source.load(model_root)
    except Exception:  # noqa: BLE001 -- an unreadable source declares nothing, and says so as []
        return []
    for _p, node in src.assigns("PIPELINE_STAGES"):
        if isinstance(node.value, (_ast.List, _ast.Tuple)):
            got = [e.value for e in node.value.elts if isinstance(e, _ast.Constant) and isinstance(e.value, str)]
            if got:
                return [str(x) for x in got]
    return []


def _c_decode_contract(src: Source) -> list:
    """An autoregressive stage keeps the decode contract, or its per-token step cannot be measured.

    decode_prefill seeds the resident KV (self- and, for a seq2seq decoder, cross-attention);
    decode_step reads them and NEVER recomputes. A decode_step that re-derives its KV is not a decode
    step -- it prices a prefill on every token, and tok/s/u is then a number about the wrong work.
    """
    # WHO THIS CHECK IS FOR, derived from what the source states rather than from what a stage is
    # called. This selected the model by `"decode" in stage_name`, so a per-token loop named
    # `generate` was never checked for the contract it needs, and a non-autoregressive stage that
    # merely had `decode` in its name was checked for hooks it should not have.
    # Two statements can say a model retires one token per call, and both are the model's own:
    # PIPELINE_UNIT == "token" declares it, and `def decode_step` keeps the contract that means it.
    _unit = ""
    for _p, node in src.assigns("PIPELINE_UNIT"):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            _unit = node.value.value.strip().lower()
            break
    if _unit != "token" and not src.mentions("def decode_step"):
        return []
    out = []
    for hook in ("decode_prefill", "decode_step"):
        if not (list(src.functions(hook)) or src.mentions("def %s" % hook)):
            out.append(
                Finding(
                    "decode-contract",
                    "autoregressive stage present but %s is missing" % hook,
                    "expose decode_prefill (seeds resident self- and cross-attn KV) and decode_step "
                    "(reads them, never recomputes). A step that re-derives its KV prices a prefill "
                    "on every token",
                    kind="porting",
                )
            )
    return out


def _c_build_returns(src: Source) -> list:
    """build_pipeline RETURNS the resident object; it must not run the model.

    A factory that calls generate()/run() and hands back a result exposes none of the per-stage
    hooks, so the trace engine has nothing to capture and skips the model entirely -- while looking
    like it worked.
    """
    out = []
    for p, fn in src.functions("build_pipeline"):
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
        if not returns:
            out.append(
                Finding(
                    "build-pipeline",
                    "%s: returns nothing" % p.name,
                    "return the resident pipeline OBJECT -- the one carrying PIPELINE_STAGES and the "
                    "per-stage hooks. A factory that returns None gives the harness nothing to measure",
                )
            )
            continue
        seg = _segment(src, p, fn)
        if re.search(r"return\s+\w*\.?(generate|run_tts|run_demo|forward)\s*\(", seg):
            out.append(
                Finding(
                    "build-pipeline",
                    "%s: returns the RESULT of running the model, not the pipeline" % p.name,
                    "return the object, do not run it. A one-shot result exposes none of the hooks, "
                    "so the trace engine skips the model while appearing to succeed",
                )
            )
    return out


def _c_weights_present(src: Source) -> list:
    """The model's weights are on this machine, checked BEFORE the device is opened.

    A DEMO SHIPS CODE, NOT WEIGHTS. tt-metal model directories hold the pipeline, the stubs and the
    tests; `from_pretrained(<repo id>)` pulls several GB into a shared HF cache once and every demo
    reads it from there. Measured 2026-08-13: zero weight files under the Voxtral demo, ~9 GB in
    ~/.cache/huggingface/hub/models--mistralai--Voxtral-Mini-3B-2507/.

    Without them nothing can be optimized -- and the failure lands badly. The run gets through
    discovery, perf-test generation (~10 minutes of agent work) and a device open before the build
    tries to load weights, at which point it either dies far from the cause or silently downloads
    gigabytes in the middle of a profiling run, with whatever timing that produces recorded as a
    measurement.

    The repo id is read from the model's own source -- the from_pretrained call it already makes --
    so this needs no configuration and no network. Missing weights are a COMPATIBILITY defect: the
    model cannot be measured as this tool measures, and it must be said before the device is touched.

    NOT A DOWNLOAD TRIGGER. This only looks. Fetching several GB is the operator's decision, not a
    side effect of asking whether a model is ready.
    """
    try:
        from .checkpoint_sections import checkpoint_keys, hf_cache_dir
    except Exception:  # noqa: BLE001 -- never take the contract down over an optional witness
        return []
    if checkpoint_keys(src.root):
        return []  # weights sit beside the model

    # NOT EVERY MODEL NAMES A HUB REPO. tt-metal also has path-configured models that read a
    # directory out of the environment (LLAMA_DIR and the like). Checking the env vars the model
    # ITSELF reads keeps this general without a naming heuristic: if any of them currently points at
    # a directory holding weight files, the model is provisioned, whatever the variable is called.
    for var in _env_vars_read(src):
        val = os.environ.get(var)
        if val and Path(val).is_dir() and checkpoint_keys(val):
            return []

    ids = _hf_repo_ids(src)
    if not ids:
        # NOTHING IS CLAIMED, SO NOTHING IS OWED. No repo named and no configured path resolved: the
        # model may well take its weights from an argument the operator supplies. Reporting that as a
        # finding turns a readiness gate into noise -- it fires on every model whose provisioning
        # this cannot see, which is not the same as a model that is missing something. Only a stated
        # requirement (a repo the source names) can be checked, and only that is flagged.
        return []
    missing = [rid for rid in ids if hf_cache_dir(rid) is None]
    if not missing:
        return []
    return [
        Finding(
            "weights-present",
            "the model loads %s but no local weights were found for %s"
            % (", ".join(sorted(ids)), ", ".join(sorted(missing))),
            "fetch the weights before optimizing (they are not in the demo directory -- a tt-metal "
            "demo ships code only, and from_pretrained reads a shared HF cache). Without them the "
            "run reaches perf-test generation and a device open before failing, or downloads "
            "gigabytes mid-profile and records the result as a measurement.",
            severity="error",
            kind="compatibility",
        )
    ]


def _env_vars_read(src: Source) -> set:
    """Every environment variable the model's own source reads."""
    out = set()
    for _path, tree in src.trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ("getenv", "get"):
                continue
            if name == "get" and not (
                isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ"
            ):
                continue
            for arg in node.args[:1]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.add(arg.value)
    return out


def _hf_repo_ids(src: Source) -> set:
    """Every hub repo id the model's own source names. Static: no import, no network."""
    ids = set()
    for _path, tree in src.trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name != "from_pretrained":
                    continue
                for arg in node.args[:1]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "/" in arg.value:
                        ids.add(arg.value)
                    elif isinstance(arg, ast.Name):
                        ids |= _const_str(tree, arg.id)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if (
                        isinstance(t, ast.Name)
                        and "REPO" in t.id.upper()
                        or (isinstance(t, ast.Name) and "MODEL_ID" in t.id.upper())
                    ):
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            if "/" in node.value.value:
                                ids.add(node.value.value)
    return ids


def _const_str(tree, name: str) -> set:
    """Module-level `NAME = "..."` bindings, so from_pretrained(HF_REPO_ID) resolves."""
    out = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        out.add(node.value.value)
    return out


CLAUSES = (
    ("sources", _c_parses),
    ("build-pipeline", _c_build_pipeline),
    ("stages", _c_stages),
    ("trace-authority", _c_trace_authority),
    ("depth-knob", _c_depth_knob),
    ("weights-present", _c_weights_present),
    ("selftests", _c_selftests),
    ("decode-contract", _c_decode_contract),
    ("build-returns", _c_build_returns),
)


def check(model_root) -> list:
    """Every unmet clause, worst first. Empty means the model is measurable as the tool measures.

    Never raises: a contract check that takes the run down with it is worse than the gap it looks
    for. An internal failure is reported AS a finding, because a clause that could not be checked
    has not been met -- the same rule the rest of the tool applies to a guard that could not run.
    """
    try:
        src = Source.load(model_root)
    except Exception as exc:  # noqa: BLE001
        return [Finding("sources", "could not read %s (%s)" % (model_root, str(exc)[:120]), "check the path")]
    out = []
    for name, fn in CLAUSES:
        try:
            out.extend(fn(src))
        except Exception as exc:  # noqa: BLE001
            out.append(
                Finding(
                    name, "clause could not be checked (%s)" % str(exc)[:120], "report this: the check itself failed"
                )
            )
    return sorted(out, key=lambda f: (0 if f.blocking else 1, 0 if f.kind == "compatibility" else 1))


def report(findings, model_root) -> str:
    n_err = sum(1 for f in findings if f.blocking)
    if not findings:
        return "  [contract] %s meets all %d clauses" % (Path(model_root).name, len(CLAUSES))
    n_port = sum(1 for f in findings if f.kind == "porting")
    head = "  [contract] %s: %d unmet — %d BLOCKING, %d porting-shape (informational for a model "
    head = (head + "optimize runs on directly)") % (Path(model_root).name, len(findings), n_err, n_port)
    return "\n".join([head] + ["    " + str(f) for f in findings])
