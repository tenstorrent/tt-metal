# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The e2e gate for the FLUX.2 Klein 9B VAE (diffusers ``AutoencoderKLFlux2``).

READY = Gate 1 AND Gate 2 AND Gate 3, for all three Calls, in ONE on-device run of this file::

    ./python_env/bin/python -m pytest models/demos/flux_2_klein_9b_vae/tests/e2e/test_e2e_pipeline.py -s

Gate 1 (``test_gate1_stubs_are_native_ttnn``, NO device)
    (a) every live ``_stubs/<name>.py`` is byte-identical (sha256) to its
        ``.last_good_sharded`` snapshot, so no graduated stub was rewritten;
    (b) ``_runtime_fallbacks.json`` is ``{}``, so none is running on a torch fallback;
    (c) a static scan of the 12 routed stubs + ``tt/pipeline.py`` + ``demo/*.py`` finds no
        forbidden host-compute op and no HF orchestration;
    (d) no ``coverage_step`` / ``coverage_sweep`` / ``invoke_all_stubs`` /
        ``_touch_all_graduated`` function exists anywhere in the package.

Gate 2 + Gate 3 (``test_e2e_pipeline``, ON DEVICE)
    one pipeline build, all three Calls, the invocation ledger, and PCC >= 0.95 each.

Gate 3-chaining (``test_gate3_chaining_is_real``, ON DEVICE)
    the reconstruction is genuinely the TT numbers (not a copy of the golden) and it
    actually depends on its input.

NOTE ON THE SCAN'S HONESTY: the scan reads each file's TEXT, drops COMMENT and STRING
tokens (so a pattern named in a comment or an error message is not a hit) and then matches
the remaining CODE. ``tt/reference.py`` is Source-A only and is deliberately NOT scanned.
The only in-file allowlist is functions whose name starts with ``_hf_reference`` / ``_golden``
(explicit reference implementations) or ends with ``_trace_setup`` (where the plan permits
seeding trace constants from the HF reference).
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import pathlib
import re
import tokenize

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.flux_2_klein_9b.vae.tt import reference as R
from models.demos.flux_2_klein_9b.vae.tt.pipeline import GRADUATED_MODULES, build_pipeline

PCC_TARGET = 0.95

_TP = int(os.environ.get("TT_HW_PLANNER_SHARD_TP", "8"))
_DP = int(os.environ.get("TT_HW_PLANNER_SHARD_DP", "1"))
_MESH = (_DP, _TP) if _DP > 1 else _TP

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[2]


def _bringup_dir() -> pathlib.Path:
    """models/tt_dit/pipelines/flux_2_klein_9b_vae — from the frozen contract, with a fallback."""
    candidate = pathlib.Path(getattr(R, "BRINGUP_DIR", ""))
    if candidate.is_dir():
        return candidate
    return REPO_ROOT / "models" / "tt_dit" / "pipelines" / "flux_2_klein_9b_vae"


# --------------------------------------------------------------------------------------
# Gate 1 static machinery
# --------------------------------------------------------------------------------------
_FORBIDDEN_TORCH_COMPUTE = (
    r"torch\.matmul|torch\.mm|torch\.bmm|torch\.einsum|torch\.softmax|torch\.log_softmax|"
    r"torch\.layer_norm|torch\.rms_norm|torch\.batch_norm|torch\.group_norm|torch\.embedding|"
    r"torch\.conv[123]d|torch\.conv_transpose|torch\.scaled_dot_product_attention|"
    r"torch\.relu|torch\.gelu|torch\.silu|torch\.tanh|torch\.sigmoid|torch\.argmax|"
    r"torch\.topk|torch\.multinomial|torch\.nn\.functional\.|\bF\.[a-z]"
)
_FORBIDDEN_HF_ORCHESTRATION = r"\.generate\(|\.forward\s*="
_FORBIDDEN = re.compile(_FORBIDDEN_TORCH_COMPUTE + "|" + _FORBIDDEN_HF_ORCHESTRATION)

_ALLOWED_FUNCTION_PREFIXES = ("_hf_reference", "_golden")
_ALLOWED_FUNCTION_SUFFIXES = ("_trace_setup",)

_SHORTCUT_FUNCTION_NAMES = frozenset({"coverage_step", "coverage_sweep", "invoke_all_stubs", "_touch_all_graduated"})


def _code_by_line(src: str, path: pathlib.Path) -> dict:
    """Map line number -> that line's CODE text, with comments and string literals removed.

    Tokens on one line are joined with no separator so dotted attribute access
    (``torch`` ``.`` ``matmul``) is reconstructed exactly as written.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:  # pragma: no cover
        raise AssertionError(f"{path} does not tokenize: {exc}") from exc

    dropped = {
        tokenize.COMMENT,
        tokenize.STRING,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
    rows: dict = {}
    for tok in tokens:
        # FSTRING_START/MIDDLE/END (py>=3.12) carry the literal text of an f-string; the
        # embedded expressions arrive as ordinary tokens and are still scanned.
        if tok.type in dropped or tokenize.tok_name.get(tok.type, "").startswith("FSTRING"):
            continue
        rows.setdefault(tok.start[0], []).append(tok.string)
    return {row: "".join(parts) for row, parts in rows.items()}


def _allowlisted_lines(src: str, path: pathlib.Path) -> set:
    tree = ast.parse(src, filename=str(path))
    allowed: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith(_ALLOWED_FUNCTION_PREFIXES) or node.name.endswith(_ALLOWED_FUNCTION_SUFFIXES):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            allowed.update(range(start, int(getattr(node, "end_lineno", node.lineno)) + 1))
    return allowed


def _scan_for_forbidden(path: pathlib.Path) -> list:
    assert path.is_file(), f"cannot scan {path}: file does not exist"
    src = path.read_text(encoding="utf-8")
    allowed = _allowlisted_lines(src, path)
    code = _code_by_line(src, path)
    raw_lines = src.splitlines()
    hits = []
    for row in sorted(code):
        if row in allowed:
            continue
        match = _FORBIDDEN.search(code[row])
        if match:
            text = raw_lines[row - 1].strip() if row - 1 < len(raw_lines) else ""
            hits.append(f"{path}:{row}: forbidden {match.group(0)!r} in {text!r}")
    return hits


def _shortcut_functions() -> list:
    stubs_dir = _bringup_dir() / "_stubs"
    files = sorted(PACKAGE_ROOT.rglob("*.py")) + [stubs_dir / f"{n}.py" for n in sorted(GRADUATED_MODULES)]
    offenders = []
    for path in files:
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover
            raise AssertionError(f"{path} does not parse: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _SHORTCUT_FUNCTION_NAMES:
                offenders.append(f"{path}:{node.lineno}: def {node.name}(...)")
    return offenders


def test_gate1_stubs_are_native_ttnn():
    """Gate 1 — every routed graduated stub is still real, unmodified, native ttnn."""
    bringup = _bringup_dir()
    stubs_dir = bringup / "_stubs"
    assert stubs_dir.is_dir(), f"bring-up stubs directory not found: {stubs_dir}"

    # The plan's definition of GRADUATED is "has a .last_good_sharded snapshot". Derive the
    # set from disk and cross-check it against the pipeline's own GRADUATED_MODULES.
    suffix = ".py.last_good_sharded"
    from_disk = {p.name[: -len(suffix)] for p in stubs_dir.glob("*" + suffix)}
    declared = set(GRADUATED_MODULES)
    assert (
        len(declared) == 12
    ), f"expected 12 graduated modules, GRADUATED_MODULES has {len(declared)}: {sorted(declared)}"
    assert from_disk == declared, (
        "GRADUATED_MODULES disagrees with the .last_good_sharded snapshots on disk. "
        f"only in pipeline={sorted(declared - from_disk)} only on disk={sorted(from_disk - declared)}"
    )

    # (a) byte-identical to the graduated snapshot
    for name in sorted(declared):
        live = stubs_dir / f"{name}.py"
        snapshot = stubs_dir / f"{name}{suffix}"
        assert live.is_file(), f"routed stub missing: {live}"
        assert snapshot.is_file(), f"graduation snapshot missing: {snapshot}"
        live_sha = hashlib.sha256(live.read_bytes()).hexdigest()
        snap_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        print(f"[gate1a] {name}: sha256={live_sha}", flush=True)
        assert live_sha == snap_sha, (
            f"{name}.py was modified after graduation: live sha256={live_sha} != snapshot sha256={snap_sha}. "
            "The routed stub must be the byte-identical graduated (sharded, native ttnn) body."
        )

    # (b) nothing fell back to torch at runtime
    fallbacks_path = bringup / "_runtime_fallbacks.json"
    assert fallbacks_path.is_file(), f"missing {fallbacks_path}"
    fallbacks = json.loads(fallbacks_path.read_text(encoding="utf-8"))
    print(f"[gate1b] _runtime_fallbacks.json = {fallbacks}", flush=True)
    assert fallbacks == {}, f"a graduated stub is running on a torch fallback: {fallbacks}"

    # (c) static scan of the hot path
    scanned = [stubs_dir / f"{name}.py" for name in sorted(declared)]
    scanned.append(PACKAGE_ROOT / "tt" / "pipeline.py")
    scanned.extend(sorted((PACKAGE_ROOT / "demo").glob("*.py")))
    hits = []
    for path in scanned:
        hits.extend(_scan_for_forbidden(path))
    print(f"[gate1c] scanned {len(scanned)} hot-path files for forbidden host compute", flush=True)
    assert not hits, "forbidden host-compute / HF-orchestration in the hot path:\n  " + "\n  ".join(hits)

    # (d) no coverage shortcut anywhere in the package
    offenders = _shortcut_functions()
    print(f"[gate1d] coverage-shortcut functions found: {offenders}", flush=True)
    assert not offenders, "Gate 2 anti-shortcut violated — coverage sweep function(s) exist:\n  " + "\n  ".join(
        offenders
    )


# --------------------------------------------------------------------------------------
# Gates 2 + 3 — on device
# --------------------------------------------------------------------------------------
def _measure(name: str, run_tt, run_golden) -> dict:
    """Run one Call and its golden, recording the outcome instead of raising.

    Collecting all three before asserting anything is deliberate: one failing Call must not
    hide the other two, and every PCC must be printable on failure as well as on pass.
    """
    record = {"name": name, "pcc": float("nan"), "ok": False, "max_abs_err": float("nan"), "shape": None, "error": None}
    try:
        tt = run_tt()
        golden = run_golden()
        tt32 = tt.detach().float()
        golden32 = golden.detach().float()
        record["shape"] = tuple(tt32.shape)
        record["golden_shape"] = tuple(golden32.shape)
        assert (
            record["shape"] == record["golden_shape"]
        ), f"{name}: TT output shape {record['shape']} != golden shape {record['golden_shape']}"
        ok, pcc = comp_pcc(golden32, tt32, PCC_TARGET)
        record["ok"] = bool(ok)
        record["pcc"] = float(pcc)
        record["max_abs_err"] = float((golden32 - tt32).abs().max().item())
    except Exception as exc:  # recorded, printed, then asserted below
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_e2e_pipeline(mesh_device):
    """Gates 2 + 3 — one pipeline, all three Calls, the ledger, and PCC >= 0.95 each."""
    torch.manual_seed(int(os.environ.get("TT_PLANNER_TEST_SEED", "0") or "0"))

    pipeline = build_pipeline(mesh_device)
    # Only the three real Calls may populate the ledger — a build-time warm-up must not.
    pipeline.reset_invocations()

    pixel_values = R.preprocess_image(R.load_input_image())
    # The decode Call's STAGE INPUT is a REAL latent for this image, produced by the HF
    # reference encoder. That is the head's own input, not a tensor injected at a joint
    # inside the chain -- and both sides (TT and golden) start from the same tensor.
    #
    # `_captured/decoder/args.pt` is deliberately NOT used here: the capture hooked the
    # `decoder` submodule, so that tensor is post_quant_conv's OUTPUT, whereas this head --
    # like `AutoencoderKLFlux2.decode` -- applies post_quant_conv itself. Feeding it would
    # still compare like with like, but it would not be a latent. (It stays the right shape
    # for the trace contract, which only pins shapes, and that is where it is used.)
    latent = R.hf_reference_encode(pixel_values)

    records = [
        _measure("encode", lambda: pipeline.run_encode(pixel_values), lambda: R.hf_reference_encode(pixel_values)),
        _measure("decode", lambda: pipeline.run_decode(latent), lambda: R.hf_reference_decode(latent)),
        _measure(
            "reconstruct",
            lambda: pipeline.run_reconstruct(pixel_values),
            lambda: R.hf_reference_reconstruct(pixel_values),
        ),
    ]

    # ---- print EVERYTHING first, so every number is visible on failure too -------------
    for record in records:
        if record["error"] is not None:
            print(f"{record['name']}: e2e PCC=FAILED ({record['error']})", flush=True)
            continue
        print(f"{record['name']}: e2e PCC={record['pcc']}", flush=True)
        print(
            f"[gate3] {record['name']}: shape={record['shape']} max_abs_err={record['max_abs_err']} "
            f"target={PCC_TARGET} pass={record['ok']}",
            flush=True,
        )

    broken = [f"{r['name']}: {r['error']}" for r in records if r["error"] is not None]
    assert not broken, "Call(s) raised before a PCC could be computed:\n  " + "\n  ".join(broken)

    # ---- Gate 2: the passive invocation ledger ---------------------------------------
    ledger = dict(pipeline.invoked_modules())
    print(f"[gate2] invoked_modules() = {ledger}", flush=True)
    missing = sorted(set(GRADUATED_MODULES) - set(ledger))
    extra = sorted(set(ledger) - set(GRADUATED_MODULES))
    assert not missing, f"graduated module(s) never invoked in any real forward path: {missing}"
    assert not extra, f"ledger contains names that are not graduated modules: {extra}"
    zero = sorted(name for name, count in ledger.items() if int(count) < 1)
    assert not zero, f"graduated module(s) recorded with a zero invocation count: {zero}"
    assert (
        int(ledger["down_encoder_block2_d"]) >= 4
    ), f"encoder has 4 down_blocks, so down_encoder_block2_d must run >= 4 times, got {ledger['down_encoder_block2_d']}"
    assert (
        int(ledger["up_decoder_block2_d"]) >= 4
    ), f"decoder has 4 up_blocks, so up_decoder_block2_d must run >= 4 times, got {ledger['up_decoder_block2_d']}"

    # ---- Gate 3: the headline numbers -------------------------------------------------
    by_name = {r["name"]: r["pcc"] for r in records}
    print(
        f"FINAL_PCC encode={by_name['encode']} decode={by_name['decode']} reconstruct={by_name['reconstruct']}",
        flush=True,
    )
    min_pcc = min(by_name.values())
    print(f"e2e PCC={min_pcc}", flush=True)
    assert min_pcc >= PCC_TARGET, f"e2e PCC {min_pcc} below target {PCC_TARGET}; per-Call: " + ", ".join(
        f"{k}={v}" for k, v in by_name.items()
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_gate3_chaining_is_real(mesh_device):
    """The reconstruction is the TT numbers, produced from ITS OWN latent, and it moves with its input."""
    torch.manual_seed(int(os.environ.get("TT_PLANNER_TEST_SEED", "0") or "0"))

    pipeline = build_pipeline(mesh_device)
    pixel_values = R.preprocess_image(R.load_input_image())

    tt_recon = pipeline.run_reconstruct(pixel_values).detach().float()
    hf_chain = R.hf_reference_decode(R.hf_reference_encode(pixel_values)).detach().float()

    ok, pcc = comp_pcc(hf_chain, tt_recon, PCC_TARGET)
    print(f"[gate3-chain] reconstruct vs hf_decode(hf_encode(x)): e2e PCC={pcc}", flush=True)

    # Not a copy of the golden: a device chain in bf16 can never be bit-identical to the
    # fp32 torch chain. Equality here would mean the reference tensor was handed back.
    assert not torch.equal(tt_recon, hf_chain), (
        "run_reconstruct returned a tensor bit-identical to the HF golden chain — the TT chain is not "
        "producing its own numbers (a reference tensor is being injected or returned)."
    )
    assert ok, f"reconstruct vs the HF chain PCC {pcc} below {PCC_TARGET} — the chain is wired wrong"

    # And it is not short-circuited / constant: a different input must give a different output.
    perturbed = torch.clamp(pixel_values * 0.5 - 0.25, -1.0, 1.0)
    tt_perturbed = pipeline.run_reconstruct(perturbed).detach().float()
    delta = float((tt_recon - tt_perturbed).abs().max().item())
    print(f"[gate3-chain] max|recon(x) - recon(0.5x-0.25)| = {delta}", flush=True)
    assert tt_perturbed is not tt_recon, "run_reconstruct returned the same object for a different input"
    assert delta > 1e-3, (
        f"perturbing the input moved the output by only {delta} — the pipeline output does not depend on "
        "its input (constant or short-circuited chain)."
    )
