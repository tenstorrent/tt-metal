# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gate 1: the 18 graduated TTNN stubs, loaded by name and proved unmodified.

What "graduated" means here
---------------------------
Source B (``models/tt_dit/pipelines/flux_2_klein_9b_transformer``) is the
bring-up workspace for this checkpoint. A component *graduated* when its live
``_stubs/<name>.py`` reached a passing per-component PCC on device and the runner
froze the file as a snapshot: ``.last_good_native`` for a body that runs
replicated, ``.last_good_sharded`` for one whose weights really are split across
the 8-chip mesh. All 18 rows of ``e2e_plan.json`` are in that state, at
per-component PCC 0.99991 .. 1.0 at TP=8.

The e2e pipeline's job is to *compose* those bodies, not to re-port them, and
Gate 1 is the evidence that it did: this module is where the composition asks
for a stub (:func:`stub_module`, :func:`build_stub`) and where the evidence is
collected (:func:`gate_1_report`). A sharded body counts as native and is never
rewritten to replication -- rewriting one to make the wiring easier is exactly
the failure Gate 1 exists to catch.

Why sha256 and not "it imported fine"
-------------------------------------
An import proves the file parses; it does not prove the file is still the body
that earned the PCC. A single edit -- swapping ``ttnn.matmul`` for a torch
fallback, dropping an ``all_reduce``, replacing a sharded ``TtLinear`` with a
replicated one -- would keep the import green and quietly turn a "native TTNN
port" claim into a lie. Comparing the live file byte-for-byte against its own
frozen snapshot, *and* against the sha256 the plan recorded when it certified
the 18, closes that hole. ``_flux2_ttnn.py`` (the bodies' shared primitives) is
checked the same way, because editing it would change all 18 at once without
touching any of their files.

Host-only
---------
Nothing in this module opens a device. :func:`gate_1_report` is pure filesystem
+ import work, so it can run in CI, on a box whose mesh is busy, or before the
pipeline is built. ``device`` is only ever passed *through* :func:`build_stub`
into the stub's own ``build()``.

The source scan
---------------
:func:`forbidden_source_scan` is the static half of Gate 1's "no torch compute
op on the hot path" claim (the dynamic half is ``host_op_selftest`` in
``tt/pipeline.py``). It looks for: ``.generate(`` (this checkpoint has no
``generate()`` at all -- the reference callable is ``forward``), assignment to a
``.forward`` attribute (monkey-patching HF), host torch/``torch.nn.functional``/
``F.`` compute ops, and the coverage-sweep helper names Gate 2 forbids. Shape and
dtype plumbing (``zeros``/``tensor``/``arange``/``cat``/``reshape``/``expand``/
``repeat_interleave``/``full_like``/``manual_seed``/``no_grad``/``randn``/
``Generator``/``.to()``) is not compute and is not reported.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import re
import tokenize
from pathlib import Path
from typing import Any, Iterable

# This file is <repo>/models/demos/flux_2_klein_9b_transformer/tt/stubs.py, so the
# repo root is four parents up. Everything else is expressed relative to it, which
# keeps the module importable from any cwd (demos are run as `python -m ...`).
REPO_ROOT = Path(__file__).resolve().parents[4]

# Resolved from the HuggingFace repo rather than a machine-local path.
# Override with TT_FLUX2_KLEIN_TRANSFORMER to point at an existing local snapshot
# (skips the Hub round-trip and makes runs deterministic).
_HF_REPO = "black-forest-labs/FLUX.2-klein-9B"
_HF_SUBFOLDER = "transformer"


def _resolve_checkpoint() -> str:
    override = os.environ.get("TT_FLUX2_KLEIN_TRANSFORMER")
    if override:
        return override
    from huggingface_hub import snapshot_download

    root = snapshot_download(_HF_REPO, allow_patterns=[f"{_HF_SUBFOLDER}/*"])
    return os.path.join(root, _HF_SUBFOLDER)


HF_MODEL_ID = _resolve_checkpoint()
SOURCE_B = "models/tt_dit/pipelines/flux_2_klein_9b_transformer"  # repo-relative

# The graduated bodies live in a directory without __init__.py; it resolves as an
# implicit namespace package under `models.tt_dit.pipelines` (which is a real
# package), which is exactly how the stubs import each other and how Source B's
# own PCC tests import them.
STUBS_PACKAGE = "models.demos.flux_2_klein_9b.transformer._stubs"

_STUBS_DIR = REPO_ROOT / SOURCE_B / "_stubs"
_PLAN_PATH = REPO_ROOT / "models/demos/flux_2_klein_9b_transformer/e2e_plan.json"
_FALLBACKS_PATH = REPO_ROOT / SOURCE_B / "_runtime_fallbacks.json"

# The shared TTNN primitives the graduated bodies are built on (TtLinear with
# column/row/regrouped schemes, all_gather/all_reduce/mesh_partition, TtRotary,
# TtLayerNorm, TtRmsNorm). Not a graduated component itself, but pinned by sha
# because a change here changes every body.
SHARED_IMPL = "_flux2_ttnn"

# The 18, sorted. This is a literal and not a directory glob on purpose: the
# _stubs/ directory also holds .bak / .preiter_native / .best_native files and
# could hold a body that has NOT graduated, and Gate 1 must fail loudly if the
# set it is asked about ever stops matching the set the plan certified.
GRADUATED: list[str] = [
    "ada_layer_norm_continuous",
    "decoder_head",
    "encoder_stack",
    "flux2_attention",
    "flux2_feed_forward",
    "flux2_modulation",
    "flux2_parallel_self_attention",
    "flux2_pos_embed",
    "flux2_single_transformer_block",
    "flux2_swi_g_l_u",
    "flux2_timestep_guidance_embeddings",
    "flux2_transformer_block",
    "layer",
    "mlp",
    "patch_embed",
    "self_attention",
    "timestep_embedding",
    "timesteps",
]

# Snapshot suffixes the bring-up runner writes, in the order they are looked for.
# A body has exactly one of them; which one it is records HOW it graduated
# (replicated-native vs genuinely mesh-sharded) and is reported as evidence.
_SNAPSHOT_SUFFIXES = (".last_good_native", ".last_good_sharded")
_SNAPSHOT_KINDS = {
    ".last_good_native": "last_good_native",
    ".last_good_sharded": "last_good_sharded",
}

# build_stub() appends here, so gate_1_report() can attribute every routed object
# to the stub module that made it even when the pipeline wrapped or subclassed it
# (the explicitly-assembled blocks subclass the graduated block types, so
# type(obj).__module__ alone is not always the stub).
_PROVENANCE: dict[str, list[dict[str, Any]]] = {}


# --------------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stub_path(name: str) -> Path:
    return _STUBS_DIR / f"{name}.py"


def _snapshot_path(name: str) -> tuple[Path | None, str | None]:
    """The frozen snapshot beside ``<name>.py``, and which kind it is."""
    live = _stub_path(name)
    for suffix in _SNAPSHOT_SUFFIXES:
        candidate = live.with_name(live.name + suffix)
        if candidate.is_file():
            return candidate, _SNAPSHOT_KINDS[suffix]
    return None, None


def _plan() -> dict:
    """``e2e_plan.json``, read lazily and cached on the function."""
    cached = getattr(_plan, "_cache", None)
    if cached is None:
        cached = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))
        _plan._cache = cached  # type: ignore[attr-defined]
    return cached


def _plan_sha256() -> dict[str, str]:
    return dict(_plan().get("gate_1_sha256", {}).get("sha256", {}))


def _check_name(name: str) -> str:
    if name not in GRADUATED:
        raise KeyError(
            f"{name!r} is not one of the {len(GRADUATED)} graduated stubs. " f"Available: {', '.join(GRADUATED)}"
        )
    return name


# ------------------------------------------------------------- stub access


def stub_module(name: str):
    """Import and return the graduated stub module ``<STUBS_PACKAGE>.<name>``.

    Plain ``importlib`` so the module object -- and therefore its ``__file__`` --
    is the real file under Source B's ``_stubs/``, which is what Gate 1 asserts
    the routed objects come from.
    """
    _check_name(name)
    return importlib.import_module(f"{STUBS_PACKAGE}.{name}")


def build_stub(name: str, device, torch_module):
    """Build the graduated stub ``name`` on ``device`` from ``torch_module``.

    Every stub exposes the same entry point -- ``build(device, torch_module)``,
    returning a callable object -- which is what makes routing them from one
    table possible. A missing ``build`` is a hard error rather than a fallback:
    silently hand-rolling a replacement is precisely what Gate 1 forbids.
    """
    module = stub_module(name)
    builder = getattr(module, "build", None)
    if not callable(builder):
        raise RuntimeError(
            f"graduated stub {name!r} ({module.__file__}) exposes no callable build(device, "
            "torch_module); it cannot be composed, and hand-rolling a replacement would "
            "break Gate 1"
        )

    obj = builder(device, torch_module)

    _PROVENANCE.setdefault(name, []).append(
        {
            "builder_module": module.__name__,
            "builder_file": module.__file__,
            "type_module": type(obj).__module__,
            "type_name": type(obj).__qualname__,
            # A pipeline subclass of a graduated type still resolves under _stubs/
            # through its MRO, so check the whole MRO, not just the leaf type.
            "from_stubs": any(getattr(base, "__module__", "").startswith(STUBS_PACKAGE) for base in type(obj).__mro__),
        }
    )
    return obj


def provenance() -> dict[str, list[dict[str, Any]]]:
    """What :func:`build_stub` has built so far, per graduated name (a copy)."""
    return {name: [dict(record) for record in records] for name, records in _PROVENANCE.items()}


# ------------------------------------------------------------------- gate 1


def gate_1_report() -> dict:
    """Collect Gate 1 evidence. Host-only: no device is opened or needed.

    ``ok`` is True only when all 18 live bodies are byte-identical to their own
    frozen snapshot AND to the sha256 the plan certified, all 18 expose
    ``build()``, ``_flux2_ttnn.py`` matches its recorded sha256, and Source B's
    ``_runtime_fallbacks.json`` is empty (a non-empty one means some component
    fell back to torch at runtime, which would void the "still real ttnn" claim).
    """
    problems: list[str] = []
    plan_shas = _plan_sha256()
    modules: dict[str, dict[str, Any]] = {}

    for name in GRADUATED:
        live = _stub_path(name)
        entry: dict[str, Any] = {
            "live_sha256": "",
            "snapshot": None,
            "snapshot_sha256": None,
            "identical": False,
            "has_build": False,
            "file": str(live),
        }
        modules[name] = entry

        if not live.is_file():
            problems.append(f"{name}: live stub is missing ({live})")
            continue
        entry["live_sha256"] = _sha256(live)

        snapshot, kind = _snapshot_path(name)
        if snapshot is None:
            # No snapshot => the component never graduated, so it must not be routed.
            problems.append(f"{name}: no .last_good_native/.last_good_sharded snapshot beside {live}")
        else:
            entry["snapshot"] = kind
            entry["snapshot_sha256"] = _sha256(snapshot)
            entry["identical"] = entry["snapshot_sha256"] == entry["live_sha256"]
            if not entry["identical"]:
                problems.append(
                    f"{name}: live body differs from its {kind} snapshot "
                    f"({entry['live_sha256'][:12]} != {entry['snapshot_sha256'][:12]}) -- "
                    "the graduated body was edited instead of composed as-is"
                )

        expected = plan_shas.get(name)
        if expected is None:
            problems.append(f"{name}: e2e_plan.json gate_1_sha256 records no sha for it")
        elif expected != entry["live_sha256"]:
            problems.append(
                f"{name}: live body differs from the sha256 e2e_plan.json certified "
                f"({entry['live_sha256'][:12]} != {expected[:12]})"
            )

        # Importing is also the only honest way to answer "does it expose build()":
        # the attribute has to exist on the module object the pipeline will use.
        try:
            entry["has_build"] = callable(getattr(stub_module(name), "build", None))
        except Exception as exc:  # noqa: BLE001 - an unimportable body is a Gate 1 failure, not a crash
            problems.append(f"{name}: stub module failed to import: {type(exc).__name__}: {exc}")
        if not entry["has_build"]:
            problems.append(f"{name}: exposes no callable build(device, torch_module)")

    shared_file = _STUBS_DIR / f"{SHARED_IMPL}.py"
    shared: dict[str, Any] = {
        "file": str(shared_file),
        "sha256": None,
        "identical_to_plan": False,
    }
    if not shared_file.is_file():
        problems.append(f"{SHARED_IMPL}.py is missing ({shared_file})")
    else:
        shared["sha256"] = _sha256(shared_file)
        expected_shared = plan_shas.get(SHARED_IMPL)
        shared["identical_to_plan"] = expected_shared == shared["sha256"]
        if not shared["identical_to_plan"]:
            problems.append(
                f"{SHARED_IMPL}.py differs from the sha256 e2e_plan.json certified "
                f"({shared['sha256'][:12]} != {(expected_shared or '<absent>')[:12]}) -- "
                "editing the shared primitives changes all 18 bodies at once"
            )

    fallbacks: dict[str, Any] = {}
    if not _FALLBACKS_PATH.is_file():
        problems.append(f"_runtime_fallbacks.json is missing ({_FALLBACKS_PATH})")
    else:
        try:
            fallbacks = json.loads(_FALLBACKS_PATH.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            problems.append(f"_runtime_fallbacks.json is not valid JSON: {exc}")
        if fallbacks:
            problems.append(
                f"_runtime_fallbacks.json is not empty ({sorted(fallbacks)}) -- at least one "
                "component fell back to torch at runtime"
            )

    return {
        "ok": not problems,
        "graduated": len(GRADUATED),
        "modules": modules,
        "shared_impl": shared,
        "runtime_fallbacks": fallbacks,
        "problems": problems,
        # Extra, additive: what build_stub() has actually been asked to build.
        # Empty on a host-only run, which is why `ok` does not depend on it.
        "provenance": provenance(),
    }


# ------------------------------------------------------- forbidden source scan

# Suppression marker. The tables below name the very tokens they hunt for, so
# without it this function would report itself. Checked against the RAW line.
NOQA_MARKER = "# noqa: tt-only"

# The torch compute ops that must never appear on the hot path. Elementwise
# arithmetic (add/mul, the Euler update) and shape/dtype plumbing are absent on
# purpose: they are not the "did you secretly run the model on the host?" tell.
_COMPUTE_OPS = (
    r"matmul|mm|bmm|einsum|softmax|log_softmax|layer_norm|rms_norm|batch_norm|group_norm"  # noqa: tt-only
    r"|embedding|embedding_bag|conv1d|conv2d|conv3d|conv_transpose\w*"  # noqa: tt-only
    r"|scaled_dot_product_attention|relu|gelu|silu|tanh|sigmoid|leaky_relu"  # noqa: tt-only
    r"|argmax|topk|multinomial|dropout"  # noqa: tt-only
)

# Coverage helpers Gate 2 forbids: a stub must be reached by the real forward, not
# by a sweep that calls everything to tick a counter. Matched as a call OR a def,
# because either one existing is the smell.
_SWEEP_NAMES = r"coverage_step|coverage_sweep|invoke_all_stubs|_touch_all_graduated"  # noqa: tt-only

_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\.generate\s*\("),  # noqa: tt-only
        # The reason text names the very call it hunts for, and a line that is a
        # bare string literal plus a comma is not "entirely" a string token, so
        # the docstring rule does not cover it. Marked explicitly.
        "calls .generate() -- this checkpoint ships no generation head; the reference callable is forward()",  # noqa: tt-only
    ),
    (
        re.compile(r"\.forward\s*="),  # noqa: tt-only
        "assigns to a .forward attribute -- monkey-patching the HF reference is forbidden",
    ),
    (
        re.compile(rf"\btorch\.(?:{_COMPUTE_OPS})\s*\("),
        "host torch compute op on the hot path",
    ),
    (
        re.compile(rf"\btorch\.nn\.functional\.(?:{_COMPUTE_OPS})\s*\("),
        "host torch.nn.functional compute op on the hot path",
    ),
    (
        re.compile(rf"(?<![\w.])F\.(?:{_COMPUTE_OPS})\s*\("),  # noqa: tt-only
        "host F.* compute op on the hot path",
    ),
    (
        re.compile(rf"\b(?:{_SWEEP_NAMES})\b"),
        "coverage-sweep helper -- every graduated stub must be reached by the real forward",
    ),
)


def _mask_strings_and_comments(text: str) -> tuple[list[str], set[int]]:
    """Blank out comments, and flag lines that are *entirely* string/comment.

    Returns ``(lines_with_comments_blanked, docstring_line_numbers)``.

    Tokenizing rather than regexing quote characters is what makes the "skip
    docstrings" rule reliable: this module's own docstring lists every forbidden
    op by name, and a naive scan would flag the documentation. Comments are
    blanked even mid-line (a trailing ``# torch.matmul(...) is illegal here`` note
    is prose, not code), while ordinary string literals stay visible so a
    violation smuggled through ``eval``/``exec`` is still reported -- that is the
    job the explicit NOQA marker exists for.
    """
    lines = text.splitlines()
    masked = [list(line) for line in lines]
    covered = [bytearray(len(line)) for line in lines]

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in (tokenize.STRING, tokenize.COMMENT):
                continue
            (start_row, start_col), (end_row, end_col) = tok.start, tok.end
            for row in range(start_row, min(end_row, len(lines)) + 1):
                index = row - 1
                if index < 0 or index >= len(lines):
                    continue
                lo = start_col if row == start_row else 0
                hi = end_col if row == end_row else len(lines[index])
                for col in range(lo, min(hi, len(lines[index]))):
                    covered[index][col] = 1
                    if tok.type == tokenize.COMMENT:
                        masked[index][col] = " "
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # An unparseable file cannot be masked safely; scan it verbatim rather
        # than pretend it is clean.
        return lines, set()

    docstring_lines = set()
    for index, line in enumerate(lines):
        content = [col for col, ch in enumerate(line) if not ch.isspace()]
        if content and all(covered[index][col] for col in content):
            docstring_lines.add(index + 1)

    return ["".join(chars) for chars in masked], docstring_lines


def forbidden_source_scan(paths: Iterable[str]) -> list[str]:
    """Scan python files for STRICT TT-ONLY violations.

    Returns ``'file:line: reason'`` strings; an empty list means clean. Lines
    inside a triple-quoted docstring, comment-only lines, and any line ending
    with ``# noqa: tt-only`` are skipped.
    """
    findings: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            findings.append(f"{path}:0: file not found (nothing scanned)")
            continue

        text = path.read_text(encoding="utf-8")
        code_lines, docstring_lines = _mask_strings_and_comments(text)
        raw_lines = text.splitlines()

        for lineno, code in enumerate(code_lines, start=1):
            if lineno in docstring_lines:
                continue
            if raw_lines[lineno - 1].rstrip().endswith(NOQA_MARKER):
                continue
            if not code.strip():
                continue
            for pattern, reason in _FORBIDDEN:
                match = pattern.search(code)
                if match:
                    findings.append(f"{path}:{lineno}: {reason} [{match.group(0).strip()}]")

    return findings
