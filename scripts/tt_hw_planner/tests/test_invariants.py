"""Invariant tests for `scripts.tt_hw_planner`.

These tests pin the load-bearing behaviors of the auto-iterate bring-up
loop. They are intentionally low-level and unit-scoped — they exercise the
pure functions and do NOT hit hardware, the network, or the LLM. The point
is to fail loudly during local development when an edit accidentally
relaxes one of the invariants the previous painful debug sessions
established.

Invariants pinned here (Tier 1 of the auto-loop improvement plan):

1. `_classify_failure` recognises the four new specific TTNN error patterns
   (L1_SMALL_ZERO, EMBEDDING_DTYPE, CONCAT_INCOMPATIBLE) BEFORE falling
   through to the broader L1_OOM / API_SIGNATURE buckets. Without that
   ordering, the LLM gets the wrong directive and burns iterations on the
   wrong root cause.

2. `_strategy_directive_for_failure` returns a non-empty, substantive hint
   for every failure class. A regression that returns the default
   one-liner for one of the named classes silently degrades convergence.

3. `_extract_pcc_from_failure` parses the PCC value out of the canonical
   "PCC X.YZ below target 0.99" line, which is what powers the
   PCC-improvement progress detector.

4. The PCC progress logic recognises a strict improvement (e.g. 0.88 ->
   0.99) as progress, exactly the case that triggered today's
   `mask_decoder_config` cap-out.

Run from the repo root with:

    python -m pytest scripts/tt_hw_planner/tests/test_invariants.py -v

These tests do not require any tt-metal device or HuggingFace network
access; they should pass on any host with Python 3.10+ and pytest.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Optional


def _planner_source() -> str:
    """Concatenate cli.py + all _cli_helpers/ + commands/ modules so
    source-grep invariant tests still work after the cli.py refactor."""
    base = Path(cli.__file__).parent
    out = [Path(cli.__file__).read_text()]
    for sub in ("_cli_helpers", "commands"):
        d = base / sub
        if d.is_dir():
            for f in sorted(d.glob("*.py")):
                out.append(f.read_text())
    return "\n".join(out)


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

cli = importlib.import_module("scripts.tt_hw_planner.cli")


def test_classify_l1_small_zero_beats_l1_oom() -> None:
    """`bank size is 0 B` in an L1_SMALL allocator failure is a device-init
    bug, not a stub bug. Must classify as L1_SMALL_ZERO so the LLM does not
    waste attempts rewriting an already-correct stub. Without this, the
    pattern would match the broader `L1_OOM` rule and the LLM would receive
    a "shrink your buffer" directive that cannot fix the real problem."""
    msg = (
        "TT_FATAL @ /home/ttuser/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:462: false\n"
        "Out of Memory: Not enough space to allocate 1760 B L1_SMALL buffer "
        "across 110 banks, where each bank needs to store 16 B, but bank "
        "size is 0 B (allocated: 0 B, free: 0 B, largest free block: 0 B)"
    )
    assert cli._classify_failure("", msg) == "L1_SMALL_ZERO"


def test_classify_embedding_dtype() -> None:
    """`ttnn.embedding` rejects INT32/INT64 index tensors with the
    "Input must be UINT32 or BFLOAT16" assertion. Must classify as
    EMBEDDING_DTYPE so the directive points the LLM at the surgical
    `ttnn.typecast` fix instead of treating it as a generic OTHER error."""
    msg = "RuntimeError: TT_FATAL: Input must be UINT32 or BFLOAT16 " "(at ttnn::embedding)"
    assert cli._classify_failure("", msg) == "EMBEDDING_DTYPE"


def test_classify_concat_incompatible_beats_api_signature() -> None:
    """`ttnn.concat()` with a mixed-type list raises `TypeError:
    incompatible function arguments`. The broader `API_SIGNATURE` class
    would match this too, but the directive for that class is generic
    ("use keyword-only signatures"). CONCAT_INCOMPATIBLE specifically
    points the LLM at the ttnn-vs-torch tensor mismatch in the list."""
    msg = (
        "TypeError: ttnn.concat(): incompatible function arguments. " "The following argument types are supported: ..."
    )
    assert cli._classify_failure("", msg) == "CONCAT_INCOMPATIBLE"


def test_classify_falls_through_to_existing_buckets() -> None:
    """The new specific classes must NOT swallow unrelated failures. Spot
    check the existing buckets still classify correctly."""
    assert cli._classify_failure("", "AssertionError: PCC 0.93 below target 0.99") == "PCC_ONLY"
    assert cli._classify_failure("", "TypeError: ttnn.matmul(): incompatible function arguments") == "API_SIGNATURE"
    assert cli._classify_failure("", "shape (1,2,3) does not match (1,2,4)") == "SHAPE"
    assert cli._classify_failure("", "Some unrelated error") == "OTHER"


def test_classify_device_reset_wins_over_oom() -> None:
    """Device-reset detection must precede the OOM rule, because a stale
    IOMMU mapping after an orphan kill can look like an allocator failure
    but is environmental, not a code defect. The directive for
    DEVICE_NEEDS_RESET tells the LLM "do NOT rewrite the stub"."""
    msg = (
        "TT_FATAL @ /home/ttuser/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:462: false\n"
        "pin_or_map_sysmem_to_device failed; Proceeding could lead to "
        "undefined behavior"
    )
    cls = cli._classify_failure("", msg)
    assert cls == "DEVICE_NEEDS_RESET", f"device-reset detection regressed; got {cls}"


_KNOWN_CLASSES = [
    "DEVICE_NEEDS_RESET",
    "HANG",
    "L1_SMALL_ZERO",
    "EMBEDDING_DTYPE",
    "CONCAT_INCOMPATIBLE",
    "L1_OOM",
    "API_SIGNATURE",
    "SHAPE",
    "PCC_ONLY",
    "OTHER",
]


def test_every_known_class_has_substantive_directive() -> None:
    """A failure class without a meaningful directive degrades the LLM
    feedback loop. Each NAMED class must return substantive guidance
    (>=80 chars); only the catch-all OTHER class is allowed to be terse
    (~70 chars), since it is the intentional generic fallback used when
    the failure cannot be diagnosed. A regression that returns the OTHER
    one-liner for a named class is caught by the uniqueness check below."""
    seen = set()
    for cls_name in _KNOWN_CLASSES:
        directive = cli._strategy_directive_for_failure(cls_name)
        assert directive, f"empty directive for {cls_name}"
        min_len = 60 if cls_name == "OTHER" else 80
        assert (
            len(directive) >= min_len
        ), f"directive for {cls_name} is suspiciously short ({len(directive)} chars): {directive!r}"
        seen.add(directive)

    assert len(seen) == len(_KNOWN_CLASSES), "two distinct failure classes returned identical directive text"


def test_l1_small_zero_directive_says_dont_rewrite_stub() -> None:
    """The L1_SMALL_ZERO directive must explicitly tell the LLM NOT to
    rewrite the stub, because the bug is in device init, not in the code."""
    text = cli._strategy_directive_for_failure("L1_SMALL_ZERO")
    assert "NOT" in text or "not" in text
    assert (
        "device" in text.lower() and "init" in text.lower() or "l1_small_size" in text.lower()
    ), f"L1_SMALL_ZERO directive must reference device init / l1_small_size; got: {text!r}"


def test_embedding_dtype_directive_names_uint32() -> None:
    """The EMBEDDING_DTYPE directive must name the surgical fix (cast
    indices to uint32) explicitly, otherwise the LLM will guess."""
    text = cli._strategy_directive_for_failure("EMBEDDING_DTYPE")
    lower = text.lower()
    assert "uint32" in lower
    assert "typecast" in lower or "from_torch" in lower


def test_concat_directive_names_ttnn_tensor_requirement() -> None:
    """The CONCAT_INCOMPATIBLE directive must name the root cause: every
    element of the concat list must be a ttnn.Tensor on the device."""
    text = cli._strategy_directive_for_failure("CONCAT_INCOMPATIBLE")
    assert "ttnn.Tensor" in text or "ttnn.from_torch" in text


def test_extract_pcc_from_pytest_assertion_line() -> None:
    """The most common form: pytest's `AssertionError: PCC <x> below target
    0.99 for <comp> of <model> ...`"""
    msg = (
        "AssertionError: PCC 0.9877806828998408 below target 0.99 for "
        "mask_decoder_config of facebook/sam2-hiera-small "
        "(primary arg `image_embeddings`)"
    )
    pcc = cli._extract_pcc_from_failure("", msg)
    assert pcc is not None
    assert abs(pcc - 0.9877806828998408) < 1e-12


def test_extract_pcc_from_negative_value() -> None:
    """PCC can be negative when the ttnn output is uncorrelated or
    anti-correlated with the torch ref. Must still extract correctly."""
    msg = "AssertionError: PCC -0.0006025997490610422 below target 0.99 for vision_config of ..."
    pcc = cli._extract_pcc_from_failure("", msg)
    assert pcc is not None
    assert pcc < 0


def test_extract_pcc_returns_none_when_no_match() -> None:
    """No PCC in the message at all (e.g. an OOM trace) must return None,
    not 0.0; the progress detector treats None as "no PCC observation"."""
    msg = "TypeError: ttnn.matmul(): incompatible function arguments"
    assert cli._extract_pcc_from_failure("", msg) is None


def test_extract_pcc_from_pcc_achieved_line() -> None:
    """Some traces emit `pcc achieved : 0.8845  (target >= 0.99)` instead
    of the assertion form. The extractor must handle both."""
    msg = "    pcc achieved      : 0.8845  (target >= 0.99)"
    pcc = cli._extract_pcc_from_failure("", msg)
    assert pcc is not None
    assert abs(pcc - 0.8845) < 1e-4


def test_progress_detector_inputs_for_mask_decoder_case() -> None:
    """End-to-end sanity check on the inputs to the Tier 1 #2 progress
    detector for the real-world `mask_decoder_config` case from the
    2026-05-21 sam2-hiera-small run:

      iter 5: PCC = 0.8845
      iter 6: PCC = 0.9878

    Both iterations classify as PCC_ONLY, but PCC strictly improved by
    >0.001, so the progress detector must reset the consecutive-same-class
    counter to 1 instead of incrementing it to 2 (which would cap the
    component at max=2 and lose all the progress)."""
    iter5_msg = "AssertionError: PCC 0.8845300534571463 below target 0.99 for mask_decoder_config"
    iter6_msg = "AssertionError: PCC 0.9877806828998408 below target 0.99 for mask_decoder_config"

    assert cli._classify_failure("", iter5_msg) == "PCC_ONLY"
    assert cli._classify_failure("", iter6_msg) == "PCC_ONLY"
    pcc5 = cli._extract_pcc_from_failure("", iter5_msg)
    pcc6 = cli._extract_pcc_from_failure("", iter6_msg)
    assert pcc5 is not None and pcc6 is not None

    assert pcc6 > pcc5 + 0.001, f"this case should be detected as PCC progress; pcc5={pcc5} pcc6={pcc6}"


def test_up_has_op_synth_and_no_op_synth_flags() -> None:
    """`up` must support both --op-synth (explicit on) and --no-op-synth
    (explicit off). The auto-iterate loop relies on the default-on
    behavior when --auto is set; --no-op-synth is the documented escape
    hatch."""

    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            cli.main(["up", "--help"])
        except SystemExit:
            pass
    help_text = buf.getvalue()
    assert "--op-synth" in help_text, "up subcommand is missing --op-synth"
    assert "--no-op-synth" in help_text, "up subcommand is missing --no-op-synth"


def test_promote_has_op_synth_and_no_op_synth_flags() -> None:
    """`promote` carries the same flag surface as `up` so scripted
    pipelines can pass the flag through unconditionally."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            cli.main(["promote", "--help"])
        except SystemExit:
            pass
    help_text = buf.getvalue()
    assert "--op-synth" in help_text, "promote subcommand is missing --op-synth"
    assert "--no-op-synth" in help_text, "promote subcommand is missing --no-op-synth"


def test_autofill_stubs_source_carries_op_synth_regen_logic() -> None:
    """`autofill_stubs` must carry the `_needs_op_synth_regen` branch and
    the matching `_skip_preserve_branches` flag, so that
    `promote --op-synth` can replace a plain Phase-1 torch wrapper with an
    op-synth partial port (it does NOT pass `overwrite=True`, because
    that would also clobber native ttnn stubs and lose LLM work). A
    regression that drops either of those names from autofill_stubs would
    silently revert promote-side op-synth back to the "preserved:already-
    autofilled" no-op behavior observed in the 2026-05-22 sam2-hiera-small
    run."""
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    src_path = Path(bl.__file__)
    text = src_path.read_text()
    assert "_needs_op_synth_regen" in text, (
        "bringup_loop.autofill_stubs is missing the op-synth regen detection. "
        "Without it, plain torch-wrapper stubs cannot be upgraded to op-synth "
        "partial ports via `promote --op-synth`."
    )
    assert "_skip_preserve_branches" in text, (
        "bringup_loop.autofill_stubs is missing the regen-path skip flag. "
        "Without it, the regen branch falls through into the user-edited-"
        "native preserve branch (which has its own `continue`), so the "
        "op-emitter write never runs."
    )

    assert ".opplan.json" in text, (
        "bringup_loop.autofill_stubs must use the .opplan.json sidecar as "
        "the signal that a stub is already an op-synth partial port."
    )


def test_capture_real_inputs_pins_rng() -> None:
    """`capture_real_inputs` MUST seed `torch.manual_seed` before any
    `torch.randn` call. Without this, the captured "ground-truth" tensors
    differ between `up`/`promote` invocations on the same model, which
    makes PCC values for an identical TTNN stub fluctuate across runs
    (observed: decoder_head PCC 0.766 -> 0.671 between two pre-flight
    sweeps on identical native code) and corrupts the PCC-progress signal
    that the cap-rule (Tier 1 #2) relies on to decide "the LLM is
    actually making headway, grant more attempts"."""
    cap = importlib.import_module("scripts.tt_hw_planner.capture_inputs")
    src = Path(cap.__file__).read_text()
    assert "torch.manual_seed(" in src, (
        "capture_real_inputs is missing torch.manual_seed; captured " "tensors will be non-deterministic across runs."
    )
    assert "TT_PLANNER_CAPTURE_SEED" in src, (
        "capture_real_inputs should expose TT_PLANNER_CAPTURE_SEED so "
        "robustness probes can intentionally vary the captures."
    )

    body_start = src.find("def capture_real_inputs(")
    assert body_start != -1, "capture_real_inputs function not found"
    body = src[body_start:]
    seed_idx = body.find("torch.manual_seed(")
    primary_randn_idx = body.find("torch.randn(1, 3,")
    assert seed_idx != -1, "torch.manual_seed missing inside capture_real_inputs"
    assert primary_randn_idx != -1, (
        "primary `torch.randn(1, 3, ...)` pixel_values draw not found " "inside capture_real_inputs"
    )
    assert seed_idx < primary_randn_idx, (
        "torch.manual_seed must be called BEFORE the primary "
        "`torch.randn(1, 3, ...)` pixel_values draw inside "
        "capture_real_inputs (otherwise the captured ground-truth "
        "input is still drawn from an unseeded RNG)."
    )


def test_emitted_pcc_test_pins_rng() -> None:
    """The emitted Phase-2 PCC test scaffold (in `bringup_loop._make_arg_for`
    wrapper) MUST also seed `torch.manual_seed` at the top of the
    `test_<component>` function, so that the synthetic-input fallback path
    (used when `_captured/<comp>.pt` is missing) is also deterministic
    across runs. Mirrors the capture-inputs seed for consistency."""
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    src = Path(bl.__file__).read_text()
    assert "TT_PLANNER_TEST_SEED" in src, (
        "bringup_loop's emitted test template should expose "
        "TT_PLANNER_TEST_SEED so the synthetic-fallback path is "
        "deterministic across runs."
    )
    assert "torch.manual_seed(_seed)" in src, (
        "bringup_loop's emitted test template must call " "torch.manual_seed(_seed) at the top of the test function."
    )


def test_cli_has_preiter_native_snapshot_helper() -> None:
    """`cli.py` must define `_snapshot_preiter_native_stub` and call it
    from the per-iter pre-apply state capture loop.

    This guards the "cap-out clobbers session-start native code" bug
    observed in the 2026-05-22 sam2-hiera-small run: `decoder_head` came
    into the session already-native (PCC=0.671, imperfect but on-device),
    the LLM failed to perfect it in 2 attempts, and the cap-out
    `_skip_component_to_fallback` restored from `.py.bak` (the original
    op-synth torch-wrapper template) — silently regressing it. The
    pre-iter snapshot captures the session-start native state so the
    worst-case cap-out is "rollback to where we started" rather than
    "rollback to a stale template"."""
    src = _planner_source()
    assert "_snapshot_preiter_native_stub" in src, (
        "cli.py must define _snapshot_preiter_native_stub (the pre-iter "
        "snapshot helper that prevents cap-out from silently regressing "
        "session-start native code)."
    )
    assert ".py.preiter_native" in src, (
        "cli.py must use the .py.preiter_native sidecar as the pre-iter " "snapshot file."
    )

    assert src.count("_snapshot_preiter_native_stub(comp)") >= 1, (
        "cli.py must call _snapshot_preiter_native_stub(comp) inside the "
        "per-iter pre-apply state loop, before the LLM is invoked."
    )


def test_skip_component_to_fallback_prefers_preiter_native() -> None:
    """`_skip_component_to_fallback` must try `.py.preiter_native` BEFORE
    falling back to `.py.bak`. Without this ordering, a cap-out on a
    component that entered the session already-native silently throws
    away that native code in favour of the original torch-wrapper
    template — strictly worse than the session start."""
    src = _planner_source()

    body_start = src.find("def _skip_component_to_fallback(")
    assert body_start != -1, "_skip_component_to_fallback function not found"

    body = src[body_start : body_start + 4000]
    preiter_idx = body.find(".py.preiter_native")
    bak_idx = body.find(".py.bak")
    assert preiter_idx != -1, "_skip_component_to_fallback must consult .py.preiter_native"
    assert bak_idx != -1, "_skip_component_to_fallback must consult .py.bak"
    assert preiter_idx < bak_idx, (
        ".py.preiter_native must be tried BEFORE .py.bak; otherwise a "
        "cap-out on an already-native component silently regresses it "
        "to the torch-wrapper template."
    )


def test_snapshot_preiter_skips_torch_wrappers() -> None:
    """The pre-iter snapshot must be a no-op when the stub is currently
    a torch wrapper. In that case the .bak IS the correct floor, and
    pinning the torch-wrapper as `.preiter_native` would just create a
    misleading duplicate of `.bak` that masquerades as 'native'."""
    src = _planner_source()
    body_start = src.find("def _snapshot_preiter_native_stub(")
    assert body_start != -1
    body = src[body_start : body_start + 2000]
    assert "_stub_uses_torch_wrapper(" in body, (
        "_snapshot_preiter_native_stub must check _stub_uses_torch_wrapper "
        "and bail out if the stub is currently a torch fallback (the .bak "
        "is the right floor in that case)."
    )


def test_emitted_pcc_test_guards_torch_tensor_input() -> None:
    """The emitted Phase-2 PCC test scaffold's `_ttnn_to_torch_mesh_safe`
    helper must early-return when given a `torch.Tensor`, because
    CPU-fallback `__call__` paths (e.g. after a cap-out restore from
    `.bak`) legitimately produce torch tensors. Without this guard, the
    test crashes with the unrelated:
        AttributeError: 'Tensor' object has no attribute 'to_torch'
    masking the actual PCC delta. Observed in the 2026-05-22
    sam2-hiera-small final pytest after `decoder_head` was restored."""
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    src = Path(bl.__file__).read_text()
    body_start = src.find("def _ttnn_to_torch_mesh_safe(")
    assert body_start != -1, "bringup_loop emitter template missing _ttnn_to_torch_mesh_safe"
    body = src[body_start : body_start + 2000]
    assert "isinstance(ttnn_tensor, torch.Tensor)" in body, (
        "emitted _ttnn_to_torch_mesh_safe must isinstance-check "
        "torch.Tensor and short-circuit return; otherwise CPU-fallback "
        "outputs crash the test with AttributeError on .to_torch()."
    )

    guard_idx = body.find("isinstance(ttnn_tensor, torch.Tensor)")
    sync_idx = body.find("ttnn.synchronize_device")
    if sync_idx != -1:
        assert guard_idx < sync_idx, (
            "torch.Tensor type-guard must be checked BEFORE any ttnn "
            "device sync / to_torch call inside _ttnn_to_torch_mesh_safe."
        )


def test_activation_diff_module_exists_and_exports_localize() -> None:
    """The `activation_diff` module must exist and expose the
    `localize_pcc_divergence` and `format_localization_hint_block`
    entry points. Tier-2 Improvement A relies on both being importable
    by the auto-iter loop."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    assert hasattr(mod, "localize_pcc_divergence"), "activation_diff must expose localize_pcc_divergence"
    assert hasattr(mod, "format_localization_hint_block"), "activation_diff must expose format_localization_hint_block"
    assert hasattr(mod, "HelperDivergence"), "activation_diff must expose HelperDivergence dataclass"
    assert hasattr(mod, "LocalizationResult"), "activation_diff must expose LocalizationResult dataclass"


def test_activation_diff_handles_missing_inputs_gracefully() -> None:
    """`localize_pcc_divergence` MUST return None (not raise) when the
    target component has no op-synth manifest. The auto-iter prompt
    assembly relies on this graceful degradation — any exception would
    break the entire iteration."""
    import tempfile

    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_dir = Path(tmpdir)
        (demo_dir / "_stubs").mkdir()
        result = mod.localize_pcc_divergence(demo_dir, "nonexistent_component")
        assert result is None, (
            "localize_pcc_divergence must degrade to None (not raise) "
            "when no op-synth manifest exists for the component"
        )


def test_activation_diff_format_returns_empty_on_none() -> None:
    """`format_localization_hint_block` must return "" when given None
    (the no-signal case). Caller is `prompt += format(...)` so an empty
    string is the correct no-op."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    assert mod.format_localization_hint_block("decoder_head", None) == ""


def test_activation_diff_pcc_helper_handles_torch_tensors() -> None:
    """The internal PCC helper must:
      - return 1.0 (within 1e-3) for two identical tensors
      - return <0.95 for orthogonal-ish tensors
      - return None for None inputs
    This is the numerical heart of the localization signal — a
    regression here returns nonsense PCC values and the LLM gets
    pointed at the wrong helper as "first divergence"."""
    import torch

    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    a = torch.randn(64, 64)
    assert mod._pcc(a, a.clone()) is not None
    assert mod._pcc(a, a.clone()) > 0.999, "_pcc must report ~1.0 for two identical tensors"

    torch.manual_seed(0)
    b = torch.randn(64, 64)
    pcc_ab = mod._pcc(a, b)
    assert pcc_ab is not None and pcc_ab < 0.99
    assert mod._pcc(None, a) is None
    assert mod._pcc(a, None) is None


def test_full_hf_reference_source_helper_exists() -> None:
    """Tier-2 Improvement B: `cli.py` must define
    `_full_hf_reference_source` and call it from the prompt assembly
    inside the PCC_ONLY branch. Without that, the LLM sees only the
    truncated `_torch_ref_summary` forward source — which is exactly
    what made decoder_head intractable (its true bug was in the
    constructor's residual chain, not the forward)."""
    src = _planner_source()
    assert "def _full_hf_reference_source(" in src, (
        "cli.py must define _full_hf_reference_source for PCC_ONLY " "prompts"
    )

    inv_idx = src.find("_full_hf_reference_source(\n")
    if inv_idx == -1:
        inv_idx = src.find("_full_hf_reference_source(stub_path")
    assert inv_idx != -1, (
        "cli.py must invoke _full_hf_reference_source(stub_path, ...) " "inside the per-component prompt assembly"
    )


def test_prompt_emits_localization_and_full_hf_only_on_pcc_only() -> None:
    """The LOCALIZATION HINT and FULL HF REFERENCE SOURCE blocks must
    be GUARDED so they only fire on PCC_ONLY (plus a narrow OTHER+
    crash escape hatch for localize). Emitting them on every failure
    class blows up prompt size on shape/API errors where they add
    zero signal.

    2026-05-31 update: after Path 2→Path 1 escalation rewire and
    pcc_repair.py deletion, the prompt-assembly path moved into
    auto_iterate.py. The structure is now TWO adjacent guards:
      (a) `if failure_class in _localize_classes ...:` calling
          `localize_pcc_divergence` (PCC_ONLY is in _localize_classes).
      (b) `if failure_class == "PCC_ONLY":` calling
          `_full_hf_reference_source` (the heavier enrichment).
    Both must be present and BOTH gates must be PCC-guarded (no
    accidental fire on shape/API errors)."""
    src = _planner_source()

    # Guard (a): localize_pcc_divergence must be inside a guard that
    # includes PCC_ONLY (either == "PCC_ONLY" or `in _localize_classes`
    # where _localize_classes is the PCC-bearing set).
    loc_idx = src.find("localize_pcc_divergence(")
    assert loc_idx >= 0, "localize_pcc_divergence call must exist"
    # Search backward for the controlling guard within 2000 chars.
    pre_window = src[max(0, loc_idx - 2000) : loc_idx]
    assert "_localize_classes" in pre_window or 'failure_class == "PCC_ONLY"' in pre_window, (
        "localize_pcc_divergence must be guarded by either the "
        '_localize_classes membership check or `failure_class == "PCC_ONLY"` '
        "— without a guard it fires on shape/API errors (prompt blowout)."
    )

    # Guard (b): _full_hf_reference_source CALL must be inside
    # `failure_class == "PCC_ONLY"` (heavier enrichment, PCC-only).
    # Search for assignment-form which is the call site (not the def).
    call_marker = "full_hf_source = _full_hf_reference_source("
    full_hf_idx = src.find(call_marker)
    assert full_hf_idx >= 0, "_full_hf_reference_source call site must exist"
    pre_window_b = src[max(0, full_hf_idx - 2000) : full_hf_idx]
    assert 'failure_class == "PCC_ONLY"' in pre_window_b, (
        "_full_hf_reference_source call must be guarded by "
        '`if failure_class == "PCC_ONLY":` — without that guard the '
        "heavy HF source dump fires on non-PCC failures."
    )


def test_full_hf_reference_returns_empty_on_missing_stub() -> None:
    """`_full_hf_reference_source` must NEVER raise — every failure
    path returns "". The auto-iter prompt assembly catches general
    exceptions defensively, but the in-function fast-path makes the
    common "stub doesn't exist" case zero-cost."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        bogus = Path(tmpdir) / "does_not_exist.py"
        result = cli._full_hf_reference_source(bogus)
        assert result == "", (
            "_full_hf_reference_source must return '' for missing stub " "(not raise FileNotFoundError)"
        )


def test_effective_attempt_cap_helper_exists() -> None:
    """`cli.py` must define `_effective_attempt_cap` as a NESTED helper
    inside `_run_auto_iterate_loop` (it closes over the per-loop
    state dicts). The function is the cap-relaxation lever for the
    structural-but-stuck PCC regime."""
    src = _planner_source()
    assert "def _effective_attempt_cap(" in src, "cli.py must define _effective_attempt_cap for Tier-2 C"

    assert "PCC_STUCK_THRESHOLD" in src, "PCC_STUCK_THRESHOLD constant must be defined in cli.py"
    assert "PCC_STUCK_EXTRA_ATTEMPTS" in src, "PCC_STUCK_EXTRA_ATTEMPTS constant must be defined in cli.py"


def test_is_at_cap_uses_effective_attempt_cap() -> None:
    """`_is_at_cap` must consult `_effective_attempt_cap(comp)`, NOT
    the raw `max_attempts_per_component`. Otherwise the relaxation
    is dead code."""
    src = _planner_source()
    is_at_cap_idx = src.find("def _is_at_cap(")
    assert is_at_cap_idx != -1

    body = src[is_at_cap_idx : is_at_cap_idx + 3000]
    assert "_effective_attempt_cap(" in body, (
        "_is_at_cap must call _effective_attempt_cap(comp) so the " "Tier-2 C relaxation actually takes effect"
    )


def test_effective_cap_clamps_to_hard_total() -> None:
    """The relaxed cap must NEVER exceed `hard_total_attempt_cap`.
    A bug that returned `base + extras` unconditionally would let
    a chronically-stuck PCC component loop indefinitely. We pin the
    `min(..., hard_total_attempt_cap)` clamp."""
    src = _planner_source()
    cap_idx = src.find("def _effective_attempt_cap(")
    assert cap_idx != -1
    body = src[cap_idx : cap_idx + 2000]
    assert "hard_total_attempt_cap" in body, (
        "_effective_attempt_cap must reference hard_total_attempt_cap " "to clamp the relaxation"
    )
    assert "min(" in body, "_effective_attempt_cap must use min(...) to clamp the " "relaxed cap to the hard ceiling"


def test_adaptive_cap_only_triggers_on_pcc_only_failure_class() -> None:
    """The relaxation must require the LAST failure class to be
    PCC_ONLY. Granting extra attempts on, e.g., L1_OOM or HANG
    would burn budget on unrecoverable structural bugs."""
    src = _planner_source()
    cap_idx = src.find("def _effective_attempt_cap(")
    assert cap_idx != -1
    body = src[cap_idx : cap_idx + 2000]
    assert 'last_class == "PCC_ONLY"' in body, (
        "_effective_attempt_cap must gate the relaxation on " 'last_class == "PCC_ONLY"'
    )
    assert "last_pcc >= PCC_STUCK_THRESHOLD" in body, (
        "_effective_attempt_cap must require last_pcc >= " "PCC_STUCK_THRESHOLD (the 'structural but stuck' regime)"
    )


def _write_junit_with_message_body(target: Path, message_body: str) -> None:
    """Write a minimal but real-shaped JUnit XML where `<failure
    message="..."/>` carries a multi-line message (the first line is the
    short exception, subsequent lines are the diagnostic info)."""
    import xml.sax.saxutils as _sx

    msg_attr = _sx.quoteattr(message_body)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites>\n"
        '  <testsuite name="pytest" tests="1" failures="1">\n'
        '    <testcase classname="models.demos.foo.tests.pcc.test_bar"\n'
        '              name="test_bar[device_params0]">\n'
        f'      <failure type="RuntimeError" message={msg_attr}>'
        "test source body (no diagnostics in here)"
        "</failure>\n"
        "    </testcase>\n"
        "  </testsuite>\n"
        "</testsuites>\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(xml)


def test_parse_pytest_report_preserves_multiline_message_body(tmp_path, monkeypatch) -> None:
    """Real bug observed on 2026-05-22:
    JUnit's `<failure message="...">` carried the full TT_FATAL block
    (`L1_SMALL buffer ... bank size is 0 B`) across multiple lines, but
    `_parse_pytest_report` only kept the first line of `message` and
    dropped the rest. As a result `_classify_failure` saw a text body
    that contained neither "L1_SMALL" nor "bank size is 0", so it fell
    through to L1_OOM and the LLM got the wrong directive. This test
    pins the parser to preserve lines 2-60 of the message in the
    details blob the classifier consumes."""
    msg = (
        "RuntimeError: TT_FATAL @ "
        "/home/ttuser/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:462: false\n"
        "info:\n"
        "Out of Memory: Not enough space to allocate 18128 B L1_SMALL "
        "buffer across 103 banks, where each bank needs to store 176 B, "
        "but bank size is 0 B (allocated: 0 B, free: 0 B)\n"
        "backtrace:\n"
        " --- tt::tt_metal::BankManager::allocate_buffer(...)\n"
    )
    junit_path = tmp_path / "generated" / "test_reports" / "most_recent_tests.xml"
    _write_junit_with_message_body(junit_path, msg)

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    report = cli._parse_pytest_report()
    details = str(report.get("details", ""))

    assert "L1_SMALL" in details, (
        "details must preserve the line 'L1_SMALL buffer ...' from the "
        "JUnit message body, otherwise _classify_failure will mis-bucket "
        "L1_SMALL_ZERO failures as L1_OOM. Got details (truncated):\n"
        f"{details[:400]}"
    )
    assert "bank size is 0" in details, (
        "details must preserve the discriminating phrase 'bank size is " "0 B' from the JUnit message body."
    )

    summary = str(report.get("summary", ""))
    assert cli._classify_failure(summary, details) == "L1_SMALL_ZERO", (
        "End-to-end: the (summary, details) pair the loop feeds to "
        "_classify_failure must classify this real-shaped failure as "
        "L1_SMALL_ZERO, NOT L1_OOM."
    )


def test_hang_synthesis_prints_compute_split() -> None:
    """User-visible regression observed on 2026-05-22:
    A run that consists entirely of HANG iterations (which can happen
    when every Claude-generated stub hangs at the same harness stage)
    used to display ZERO `compute split` blocks because the HANG branch
    `continue`s before the normal post-pytest print at the bottom of
    the iter. The user lost visibility into the on-device vs on-CPU
    percentages between iters. Pin: the HANG branch must also call
    `_format_compute_split` / `_format_op_split` before its `continue`."""
    src = _planner_source()
    hang_idx = src.find("HANG detected (")
    assert hang_idx != -1
    cont_idx = src.find("continue", hang_idx)
    assert cont_idx != -1
    hang_body = src[hang_idx:cont_idx]
    assert "_format_compute_split(" in hang_body, (
        "HANG synthesis branch must call _format_compute_split so the "
        "user sees the components on-device/on-CPU percentage between "
        "consecutive HANG iters (previously only the normal-failure "
        "path printed this, leaving a run-of-HANGs visually empty)."
    )
    assert "_format_op_split(" in hang_body, (
        "HANG synthesis branch must also call _format_op_split so the "
        "operation-level percentage stays visible between iters."
    )


def test_hang_synthesis_records_failure_for_consec_counter() -> None:
    """Real bug observed on 2026-05-22:
    The auto-iterate HANG synthesis path (when pytest is killed by the
    wall-clock timeout) writes its synthetic failure record and then
    `continue`s the iter loop. Pre-fix, that `continue` jumped OVER the
    `_record_failure_for_component(...)` call that lives in the normal
    XML-parsing path, so the consecutive-same-class counter stayed at 0
    across N consecutive HANG iterations. The cap-out never fired, the
    loop kept invoking the LLM against the same hanging stub, and
    decoder_head went from `attempt 1/2` to `attempt 3/2` to ... until
    the total `max_iters` ceiling killed the run.

    Pin: the HANG synthesis branch must call
    `_record_failure_for_component(comp, \"HANG\", ...)` for each hung
    component BEFORE the `continue`."""
    src = _planner_source()

    hang_idx = src.find("HANG detected (")
    assert hang_idx != -1, "could not locate HANG-detection branch in cli.py"

    cont_idx = src.find("continue", hang_idx)
    assert cont_idx != -1, "could not locate the continue at end of HANG branch"

    hang_body = src[hang_idx:cont_idx]
    assert "_record_failure_for_component(" in hang_body and '"HANG"' in hang_body, (
        "HANG synthesis branch must call "
        '`_record_failure_for_component(comp, "HANG", ...)` so the '
        "consec-same-class counter advances; otherwise consecutive "
        "wall-clock timeouts NEVER trigger cap-out and the loop "
        "iterates against the same hanging stub indefinitely."
    )


def test_scope_report_preserves_multiline_message_body() -> None:
    """Real bug observed on 2026-05-22:
    `_parse_pytest_report` was fixed to keep lines 2-60 of the JUnit
    `message` (where TT_FATAL discriminators live), but
    `_scope_report_to_demo` then rebuilt `summary`/`details` from only
    `message.splitlines()[0]` + pytest body, silently re-truncating the
    same lines on EVERY auto-iterate path that scopes the report. The
    classifier saw the truncated text and re-buckets L1_SMALL_ZERO ->
    L1_OOM, EMBEDDING_DTYPE -> OTHER, etc.

    Pin: both parser AND scoper must preserve msg_lines[1:60].
    """
    src = _planner_source()
    scope_idx = src.find("def _scope_report_to_demo(")
    assert scope_idx != -1
    next_def = src.find("\ndef ", scope_idx + 1)
    body = src[scope_idx : next_def if next_def != -1 else len(src)]
    assert "msg_lines[1:60]" in body or "splitlines()[1:60]" in body, (
        "_scope_report_to_demo must preserve JUnit message lines "
        "[1:60] in `details`, otherwise it silently undoes the fix in "
        "`_parse_pytest_report` on every scoped path the loop uses."
    )


def test_no_op_continue_path_records_failure_for_consec_counter() -> None:
    """Same counter-bypass shape as the HANG bug, found by audit: the
    op-synth byte-identical no-op `continue` path used to skip
    `_record_failure_for_component`, so repeated no-ops never advanced
    the per-class cap-out counter. Pin: the NO_OP branch must record."""
    src = _planner_source()

    idx = src.find("agent wrote byte-identical output")
    assert idx != -1, "could not locate the NO_OP branch banner text"

    m = re.search(r"^\s+continue\s*$", src[idx:], flags=re.MULTILINE)
    assert m is not None
    body = src[idx : idx + m.end()]
    assert "_record_failure_for_component(" in body and '"NO_OP"' in body, (
        "NO_OP branch must call `_record_failure_for_component(comp, "
        '"NO_OP", None)` so repeated no-ops trigger cap-out; without '
        "this the consec-same-class counter stays at 0 and the loop "
        "wastes iters until the hard total cap."
    )


def test_wrapper_regression_continue_path_records_failure() -> None:
    """The strict-native still-wrappers branch had the same bug: it
    `continue`d before failure-recording, so wrapper-regression
    iterations never advanced the cap counter."""
    src = _planner_source()
    idx = src.find("agent wrote torch wrapper(s) again")
    assert idx != -1
    m = re.search(r"^\s+continue\s*$", src[idx:], flags=re.MULTILINE)
    assert m is not None
    body = src[idx : idx + m.end()]
    assert "_record_failure_for_component(" in body and '"WRAPPER"' in body, (
        "Still-wrappers branch must call `_record_failure_for_component"
        '(comp, "WRAPPER", None)` so wrapper-regression cap-out fires.'
    )


def test_empty_agent_continue_path_records_failure() -> None:
    """Empty-agent branch was the third counter-bypass site found by
    the audit. Repeated empty-agent invocations (e.g. credentials
    misconfigured, CLI version mismatch) should hit the per-class cap
    and skip the component to fallback, not just the global hard cap."""
    src = _planner_source()
    idx = src.find("agent produced zero usable response files")
    assert idx != -1

    matches = list(re.finditer(r"^\s+continue\s*$", src[idx:], flags=re.MULTILINE))
    assert matches, "could not find the loop-around continue after empty-agent banner"
    body = src[idx : idx + matches[-1].end()]
    assert "_record_failure_for_component(" in body and '"EMPTY_AGENT"' in body, (
        "Empty-agent branch must call `_record_failure_for_component"
        '(target, "EMPTY_AGENT", None)` so cap-out fires on repeated '
        "empty agent invocations."
    )


def test_stub_has_graduated_returns_false_for_missing_file(tmp_path) -> None:
    """`_stub_has_graduated_from_autofill` returning True for a missing
    stub file silently inflated the compute split (component reported
    as on-device) and removed the component from `_auto_iteration_blockers`'s
    ungraduated list (so the loop never re-attempted it). Pin:
    pessimistic on missing/unreadable stubs."""
    import importlib

    bringup_loop = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    missing = tmp_path / "absent_stub.py"
    assert not missing.is_file()
    assert bringup_loop._stub_has_graduated_from_autofill(missing) is False


def test_stub_has_graduated_requires_pcc_graduation_snapshot(tmp_path) -> None:
    """Pin: a stub WITHOUT a `.py.last_good_native` snapshot is NOT
    classified as graduated, even if its body looks like native ttnn
    (no torch-fallback markers).

    Why this matters: op-synth scaffolded stubs ship with pre-bound
    `_apply_*` helpers and no torch fallback. Before this fix, they were
    misclassified as graduated, causing final_categorization to bucket
    them as ON_DEVICE and hiding the work-remaining signal. Only
    PCC-validated stubs (which write the `.py.last_good_native` snapshot
    at graduation time) should be marked graduated."""
    import importlib

    bringup_loop = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    stub = tmp_path / "scaffold.py"
    # Scaffold-style body: native-looking, no torch fallback, NO snapshot.
    stub.write_text("class Stub:\n" "    def __call__(self, x):\n" "        return self._apply_op(x)\n")
    assert not stub.with_suffix(".py.last_good_native").is_file()
    assert bringup_loop._stub_has_graduated_from_autofill(stub) is False, (
        "stub with no .py.last_good_native snapshot must be classified " "as NOT graduated, regardless of body content"
    )


def test_stub_has_graduated_true_only_with_snapshot_and_no_fallback(tmp_path) -> None:
    """The positive case: snapshot exists AND current stub has no
    torch-fallback markers → graduated."""
    import importlib

    bringup_loop = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    stub = tmp_path / "graduated.py"
    stub.write_text(
        "import ttnn\n" "class Stub:\n" "    def __call__(self, x):\n" "        return ttnn.matmul(x, self.w)\n"
    )
    # Simulate _snapshot_native_stub having run: snapshot file present.
    stub.with_suffix(".py.last_good_native").write_text(stub.read_text())
    assert bringup_loop._stub_has_graduated_from_autofill(stub) is True


def test_stub_has_graduated_false_when_snapshot_but_current_rolled_back(tmp_path) -> None:
    """Edge case: snapshot exists (was graduated at some point), but
    the current stub was rolled back to torch fallback. Should return
    False because the current body delegates to torch."""
    import importlib

    bringup_loop = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    stub = tmp_path / "regressed.py"
    stub.write_text("class Stub:\n" "    def __call__(self, x):\n" "        return self._get_torch_submodule()(x)\n")
    stub.with_suffix(".py.last_good_native").write_text("# stale snapshot\n")
    assert bringup_loop._stub_has_graduated_from_autofill(stub) is False


def test_skip_component_restore_priority_prefers_best_native() -> None:
    """Pin the cap-out restore priority introduced by the audit fixes:
    last_good_native > best_native > preiter_native > bak.
    Without best_native, in-session PCC improvements (e.g. 0.67 -> 0.88
    that still failed) were thrown away on cap-out, restoring the
    older preiter snapshot. The priority list must explicitly mention
    best_native in the restoration chain."""
    src = _planner_source()
    skip_idx = src.find("def _skip_component_to_fallback(")
    assert skip_idx != -1
    next_def = src.find("\n    def ", skip_idx + 1)
    body = src[skip_idx : next_def if next_def != -1 else len(src)]

    for kind in (".py.last_good_native", ".py.best_native", ".py.preiter_native", ".py.bak"):
        assert kind in body, (
            f"_skip_component_to_fallback must reference `{kind}` in "
            f"its restoration chain; the audit-introduced order is "
            f"last_good_native > best_native > preiter_native > bak."
        )

    last_good_pos = body.find(".py.last_good_native")
    best_native_pos = body.find(".py.best_native")
    preiter_pos = body.find(".py.preiter_native")
    bak_pos = body.find(".py.bak")
    assert last_good_pos < best_native_pos < preiter_pos < bak_pos, (
        "_skip_component_to_fallback restore order must be "
        "last_good_native -> best_native -> preiter_native -> bak. "
        f"Got positions: last_good={last_good_pos}, "
        f"best={best_native_pos}, preiter={preiter_pos}, bak={bak_pos}"
    )


def test_up_auto_scopes_seed_report_to_demo_before_already_done_check() -> None:
    """Real bug observed on 2026-05-22:
    `up --auto` read `_parse_pytest_report().all_passed` directly when
    deciding whether to short-circuit with `already_done = True`. Any
    stale `most_recent_tests.xml` from an unrelated pytest run (e.g.
    user running `pytest scripts/tt_hw_planner/tests/test_invariants.py`)
    would be read as "all passed" and the entire auto loop was skipped,
    even though zero of THIS demo's PCC tests had run.

    Pin: the `--auto` block in `up` must scope the seed report to the
    demo via `_scope_report_to_demo(...)` and ensure `seed_all_passed`
    only fires when this demo's tests actually appear in the report."""
    src = _planner_source()

    idx = src.find("stale `most_recent_tests.xml`")
    assert idx != -1, (
        "could not find the stale-XML scoping comment in cli.py; "
        "if you intentionally removed it, please add a regression "
        "test for the new shape of the seed-report scoping."
    )

    fix_window = src[idx : idx + 4000]
    assert "_scope_report_to_demo(" in fix_window, (
        "the `up --auto` seed-report block must call "
        "`_scope_report_to_demo(...)` so unrelated pytest runs don't "
        "trigger the `already_done` short-circuit."
    )
    assert 'seed_report.get("per_test")' in fix_window or "saw_any_demo_test" in fix_window, (
        "the `up --auto` block must verify the scoped report saw at "
        "least one of THIS demo's tests before honoring "
        "`seed_all_passed=True`; otherwise an empty/stale XML still "
        "trips the short-circuit."
    )


def test_promote_auto_revalidates_when_structural_says_done() -> None:
    """The `promote --auto` command short-circuits with
    \"nothing to do — every NEW component is already native TTNN\"
    purely from `_auto_iteration_blockers` (a structural check on the
    on-disk stub). A stub from a killed prior session can be
    structurally native yet hang at runtime, so the structural-only
    check is a false-green under `--auto`. Pin: under `--auto`, the
    promote command must still hand off to the auto-iterate loop so
    its pre-flight pytest sweep can verify each native stub actually
    works (re-targeting failures)."""
    src = _planner_source()
    idx = src.find("nothing to do \u2014 every NEW component is already native TTNN")
    if idx == -1:
        idx = src.find("nothing to do -- every NEW component is already native TTNN")
    if idx == -1:
        idx = src.find("every NEW component is already native TTNN")
    assert idx != -1, "could not locate the promote 'nothing to do' banner"

    region = src[max(0, idx - 3000) : idx + 500]
    assert 'getattr(args, "auto", False)' in region or "args.auto" in region, (
        "the promote 'nothing to do' branch must gate the early-return "
        "on `--auto`; under --auto, structurally-native stubs must "
        "still be re-validated via pytest before declaring done."
    )


def test_regression_guard_uses_preiter_native_snapshot() -> None:
    """The strict-native regression guard (where the agent rewrites a
    native component as a torch wrapper) used to restore ONLY from
    `.py.bak`. For a NEW component, `.bak` IS a torch wrapper so the
    restore quietly did nothing — the regressed wrapper stayed on
    disk. Pin: the regression guard must consult `.py.preiter_native`
    and `.py.last_good_native` BEFORE falling back to `.bak`."""
    src = _planner_source()
    idx = src.find("REGRESSION GUARD: agent rewrote")
    assert idx != -1

    region = src[max(0, idx - 3000) : idx + 500]
    assert ".py.preiter_native" in region, (
        "Regression guard region must consider .py.preiter_native as "
        "a restore source; without it native code can be silently lost "
        "on agent regressions when .bak is itself a torch wrapper."
    )
    assert ".py.last_good_native" in region, "Regression guard region must also consider .py.last_good_native."


def test_parse_pytest_report_classifies_embedding_dtype_from_message_body(tmp_path, monkeypatch) -> None:
    """Same regression as L1_SMALL_ZERO, but for EMBEDDING_DTYPE: the
    discriminator 'Input must be UINT32 or BFLOAT16' lives on line 2+
    of the JUnit message. Pin both ends of the pipeline."""
    msg = (
        "RuntimeError: TT_FATAL @ "
        "/home/ttuser/tt-metal/.../embedding.cpp:NN: false\n"
        "info:\n"
        "Input must be UINT32 or BFLOAT16\n"
    )
    junit_path = tmp_path / "generated" / "test_reports" / "most_recent_tests.xml"
    _write_junit_with_message_body(junit_path, msg)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    report = cli._parse_pytest_report()
    details = str(report.get("details", ""))
    summary = str(report.get("summary", ""))
    assert "Input must be UINT32 or BFLOAT16" in details
    assert cli._classify_failure(summary, details) == "EMBEDDING_DTYPE"


def test_edit_scope_table_has_l1_small_zero_entry() -> None:
    """The L1_SMALL_ZERO failure class is the canonical example of a
    failure that cannot be fixed inside the per-component stub alone
    — it needs `l1_small_size=...` to be added to the device fixture
    in `conftest.py`. The edit-scope permission table must include
    `L1_SMALL_ZERO` -> `conftest.py` (or its tests/pcc variant);
    without this, the LLM has no way to fix the root cause."""
    table = cli._EDIT_SCOPE_FOR_FAILURE_CLASS
    assert "L1_SMALL_ZERO" in table, (
        "_EDIT_SCOPE_FOR_FAILURE_CLASS must whitelist L1_SMALL_ZERO; "
        "without it the LLM can only fall back to CPU on conv2d OOMs"
    )
    paths = table["L1_SMALL_ZERO"]
    assert any("conftest.py" in p for p in paths), "L1_SMALL_ZERO must unlock at least one conftest.py path"


def test_resolve_extra_edit_paths_unknown_class_returns_empty(
    tmp_path,
) -> None:
    """Unknown / unmapped failure classes must return an empty
    extra-paths list — the default scope (the stub itself) is the
    safe fallback. A regression that returned a wildcard here would
    let the LLM edit arbitrary repo files for any failure class."""
    result = cli._resolve_extra_edit_paths(tmp_path, "FAKE_CLASS_XYZ")
    assert result == [], "unknown failure classes must not unlock any extra edit paths"


def test_format_escalated_edit_scope_block_is_empty_when_no_escalation(
    tmp_path,
) -> None:
    """When the failure class is mapped to NO extra files (or maps
    to files that don't exist on disk), the prompt block must be the
    empty string. Otherwise we'd pollute every prompt with an empty
    ESCALATED EDIT SCOPE header."""
    block = cli._format_escalated_edit_scope_block(tmp_path, "FAKE_CLASS_XYZ")
    assert block == ""
    block_other = cli._format_escalated_edit_scope_block(tmp_path, "OTHER")
    assert block_other == ""


def test_format_escalated_edit_scope_block_renders_l1_small_zero(
    tmp_path,
) -> None:
    """When L1_SMALL_ZERO is the failure class AND a matching conftest.py
    exists under the demo dir, the prompt block must (a) include the
    'ESCALATED EDIT SCOPE' header, (b) name the conftest.py file, and
    (c) explicitly tell the LLM to add `l1_small_size`. A regression
    that drops any of these falls back to the (failed) blind-rewrite
    behavior."""
    (tmp_path / "conftest.py").write_text("# fake conftest\n")
    block = cli._format_escalated_edit_scope_block(tmp_path, "L1_SMALL_ZERO")
    assert "ESCALATED EDIT SCOPE" in block, (
        "L1_SMALL_ZERO with a discoverable conftest.py must produce " "the ESCALATED EDIT SCOPE prompt section"
    )
    assert "conftest.py" in block
    assert "l1_small_size" in block, "the L1_SMALL_ZERO directive must name the specific kwarg " "the LLM should add"


def test_prompt_assembly_includes_escalated_scope_block() -> None:
    """The prompt-assembly path must call _format_escalated_edit_scope_block
    and concatenate its output into the final prompt string. Without
    this wiring, the table is dead code."""
    src = _planner_source()
    assert "_format_escalated_edit_scope_block(" in src, "_format_escalated_edit_scope_block must be invoked in cli.py"
    assert "escalated_scope_block" in src, (
        "the prompt assembly must thread the escalated-scope block " "into the final prompt"
    )


def test_component_complexity_bonus_helper_exists() -> None:
    """The complexity-bonus helper must be defined and called by
    `_effective_attempt_cap` so the capability is observable from
    the cap calculation. We verify by inspecting source — the
    function is a closure inside cmd_promote/cmd_up, not at module
    level, so we can't import it directly."""
    src = _planner_source()
    assert "_component_complexity_bonus" in src, "complexity-bonus helper must be defined"
    assert "_op_synth_manifest(" in src
    cap_idx = src.find("def _effective_attempt_cap")
    assert cap_idx >= 0
    cap_window = src[cap_idx : cap_idx + 4000]
    assert "_component_complexity_bonus" in cap_window, "_effective_attempt_cap must call _component_complexity_bonus"
    assert "complexity" in cap_window, "cap function must integrate complexity bonus"


def test_complexity_bonus_uses_palette_and_gaps() -> None:
    """Complexity bucketing must be driven by palette length AND
    llm_gaps length so it captures both 'many ops' and 'many novel
    ops'. Without the gaps component, a refactored model with a
    small palette but many op-NEW gaps would receive no extra
    budget despite being harder to port."""
    src = _planner_source()

    bonus_idx = src.find("def _component_complexity_bonus")
    assert bonus_idx >= 0, "_component_complexity_bonus definition must exist"
    bonus_window = src[bonus_idx : bonus_idx + 2000]
    assert "palette" in bonus_window
    assert (
        "llm_gaps" in bonus_window or "gaps" in bonus_window
    ), "complexity bonus must read llm_gaps from manifest, not just palette"


def test_complexity_bonus_is_bounded() -> None:
    """The bonus must be capped (e.g. min(4, ...)) so a single huge
    component cannot starve the rest of the run by exhausting the
    hard total attempt cap. The historical bug was an unbounded
    cap that let one component eat all of `hard_total_attempt_cap`."""
    src = _planner_source()
    bonus_idx = src.find("def _component_complexity_bonus")
    assert bonus_idx >= 0
    bonus_window = src[bonus_idx : bonus_idx + 2000]
    assert re.search(r"min\(\s*\d+\s*,", bonus_window), "_component_complexity_bonus must clamp its output via min(...)"


def test_systemic_pattern_tracker_helpers_exist() -> None:
    """The systemic-pattern tracker must define
    `_record_systemic_failure` and `_format_systemic_pattern_block`
    and feed those into the prompt assembly. Without these, the
    `repeat_error_counts` map remains L1_OOM-specific and we silently
    burn iters on shared-path bugs (e.g. ttnn_to_torch hangs)."""
    src = _planner_source()
    assert "_record_systemic_failure" in src, "systemic-pattern tracker must define _record_systemic_failure"
    assert (
        "_format_systemic_pattern_block" in src
    ), "systemic-pattern tracker must define _format_systemic_pattern_block"
    assert "systemic_failure_counts" in src, "systemic-pattern tracker must maintain a counts map"
    assert "SYSTEMIC_THRESHOLD" in src, (
        "systemic-pattern tracker must declare an explicit threshold " "constant (currently 3 distinct components)"
    )


def test_systemic_pattern_threshold_is_at_least_two() -> None:
    """The threshold must be >=2 — a single component failing with a
    given signature is NOT systemic; that's just a stub bug. Pin
    the threshold so a future edit doesn't accidentally collapse it
    to 1 (which would fire the systemic banner on every iter)."""
    src = _planner_source()
    m = re.search(r"SYSTEMIC_THRESHOLD\s*=\s*(\d+)", src)
    assert m is not None
    assert int(m.group(1)) >= 2, (
        "SYSTEMIC_THRESHOLD must be at least 2; a single component " "failing is not a systemic pattern"
    )


def test_systemic_block_threaded_into_prompt() -> None:
    """The systemic-pattern block must be concatenated into the
    final prompt string. Verifies the wiring, not just the helper."""
    src = _planner_source()
    assert "systemic_block" in src, "prompt assembly must thread the systemic-pattern block"


def test_systemic_record_skips_environmental_classes() -> None:
    """`_record_systemic_failure` must NOT count DEVICE_NEEDS_RESET
    (host-hygiene events) toward the systemic threshold, otherwise
    a flaky device on a real environment would trip the banner."""
    src = _planner_source()
    rec_idx = src.find("def _record_systemic_failure")
    assert rec_idx >= 0
    window = src[rec_idx : rec_idx + 1500]
    assert "DEVICE_NEEDS_RESET" in window, (
        "_record_systemic_failure must explicitly skip "
        "DEVICE_NEEDS_RESET so device flakiness doesn't false-trip "
        "the systemic-pattern banner"
    )


def test_classify_failure_recognizes_tt_fatal_opaque() -> None:
    """A bare `TT_FATAL @ <op>.cpp:N: false` with no allocator /
    API / shape signature must classify as `TT_FATAL_OPAQUE`, not
    `OTHER`. Without this, the next-iter prompt routes through the
    generic OTHER directive and never tells the LLM to inject
    shape probes."""
    summary = "RuntimeError: TT_FATAL @ softmax.cpp:42: false"
    details = ""
    assert cli._classify_failure(summary, details) == "TT_FATAL_OPAQUE", (
        "bare TT_FATAL with no allocator/API/shape clue must classify "
        "as TT_FATAL_OPAQUE for the shape-probe directive to fire"
    )


def test_strategy_directive_tt_fatal_opaque_mentions_shape_probe() -> None:
    """The TT_FATAL_OPAQUE strategy directive must explicitly teach
    the LLM the SHAPE_PROBE template — without the literal probe
    tag string the LLM will improvise and the harvester won't pick
    up the prints."""
    directive = cli._strategy_directive_for_failure("TT_FATAL_OPAQUE", strict_native=True)
    assert "SHAPE_PROBE" in directive, (
        "TT_FATAL_OPAQUE directive must mention SHAPE_PROBE so the "
        "LLM's probe lines are parseable by _extract_shape_probes"
    )
    assert "Read" in directive and "Edit" in directive, "TT_FATAL_OPAQUE directive must invite Read+Edit tool use"


def test_extract_shape_probes_parses_canonical_lines() -> None:
    """The harvester must recognise the exact line format produced
    by the template in the TT_FATAL_OPAQUE directive. A drift here
    means the LLM instruments, runs pytest, and the next-iter
    prompt drops its observations on the floor."""
    text = (
        "some pytest stderr...\n"
        "[SHAPE_PROBE my-tag-1] arg0: shape=(1, 16, 64, 64) dtype=DataType.BFLOAT16 layout=Layout.TILE mem=DRAM\n"
        "[SHAPE_PROBE my-tag-1] arg1: shape=(64, 64) dtype=DataType.BFLOAT16 layout=Layout.ROW_MAJOR mem=L1\n"
        "[SHAPE_PROBE my-tag-2] x: shape=(8,) dtype=DataType.UINT32 layout=Layout.ROW_MAJOR mem=DRAM\n"
        "more stderr\n"
    )
    probes = cli._extract_shape_probes(text)
    assert len(probes) == 3
    assert probes[0]["tag"] == "my-tag-1"
    assert probes[0]["name"] == "arg0"
    assert "shape=(1, 16, 64, 64)" in probes[0]["payload"]
    assert probes[2]["tag"] == "my-tag-2"


def test_extract_shape_probes_empty_on_no_probes() -> None:
    """No probe lines -> empty list. The prompt formatter relies on
    this for the "empty string when no probes" property."""
    assert cli._extract_shape_probes("") == []
    assert cli._extract_shape_probes("some traceback with no probe markers") == []


def test_extract_shape_probes_from_report_reads_full_body_pre_truncation() -> None:
    """Regression: the harvester used to read the pre-truncated
    `details` string from the iter loop, which caps each failure
    body at 60 lines. A stub with many probed call sites can easily
    push probes past that cap, where they'd be silently dropped.
    Pin: the report-walking harvester reads the FULL `body` field
    stored in `per_test[*]` (no truncation)."""
    body_lines = ["traceback line %d" % i for i in range(80)]
    body_lines.append("[SHAPE_PROBE deep-probe] x: shape=(1,16,64,64) dtype=bf16 layout=TILE mem=DRAM")
    body_lines.append("E    RuntimeError: TT_FATAL @ softmax.cpp:42: false")
    body_text = "\n".join(body_lines)
    report = {
        "per_test": {
            "tests/pcc/test_decoder.py::test_decoder": {
                "test_id": "test_decoder",
                "message": "RuntimeError: TT_FATAL @ softmax.cpp:42: false",
                "body": body_text,
            }
        }
    }
    probes = cli._extract_shape_probes_from_report(report)
    assert len(probes) == 1, (
        "report-walking harvest must find probes past line 60 of "
        "the body; the old details-string harvest dropped them"
    )
    assert probes[0]["tag"] == "deep-probe"
    assert "shape=(1,16,64,64)" in probes[0]["payload"]


def test_extract_shape_probes_from_report_deduplicates_message_and_body() -> None:
    """The JUnit XML's `message` attribute and `body` element can
    BOTH contain the same probe line (pytest sometimes echoes the
    captured stderr into both). The report-walking harvest must
    dedupe so the LLM doesn't see double-prints in its prompt."""
    line = "[SHAPE_PROBE p1] arg0: shape=(8,) dtype=uint32 layout=ROW_MAJOR mem=DRAM"
    report = {
        "per_test": {
            "tests/pcc/test_x.py::test_x": {
                "test_id": "test_x",
                "message": line,
                "body": "traceback...\n" + line,
            }
        }
    }
    probes = cli._extract_shape_probes_from_report(report)
    assert len(probes) == 1, "duplicate probe lines must be deduped"


def test_extract_shape_probes_from_report_empty_on_no_per_test() -> None:
    """Defensive: an empty / malformed report must return [] not
    raise, since the auto-iterate loop unconditionally invokes the
    harvester every iter."""
    assert cli._extract_shape_probes_from_report({}) == []
    assert cli._extract_shape_probes_from_report({"per_test": None}) == []
    assert cli._extract_shape_probes_from_report({"per_test": {}}) == []


def test_auto_loop_uses_report_walking_shape_probe_harvest() -> None:
    """Pin the wiring: the iter loop must call
    `_extract_shape_probes_from_report(report)`, NOT the legacy
    `_extract_shape_probes(last_failures + last_failure_details)`
    which reads from the pre-truncated 60-line details. A regression
    that flips this back to the string harvest would silently drop
    probes past line 60."""
    src = _planner_source()
    assert "_extract_shape_probes_from_report(report)" in src, (
        "auto-iterate loop must use the report-walking harvest so "
        "probes past line 60 of a failure body are not dropped"
    )


def test_hang_branch_feeds_systemic_tracker() -> None:
    """Regression: the HANG synthesis branch updates the per-
    component progress tracker (`_record_failure_for_component`)
    but used to SKIP `_record_systemic_failure`, so a run where
    every Claude stub hangs at the same harness stage never tripped
    the systemic-pattern banner — even though 3+ components hung
    with the SAME signature. The systemic detector exists exactly
    to catch hang-storms like this. Pin: the HANG branch must also
    feed the systemic tracker."""
    src = _planner_source()
    hang_idx = src.find("HANG detected (")
    assert hang_idx != -1
    cont_idx = src.find("continue", hang_idx)
    assert cont_idx != -1
    hang_body = src[hang_idx:cont_idx]
    assert "_record_systemic_failure(" in hang_body, (
        "HANG synthesis branch must call _record_systemic_failure "
        "so a hang-storm (3+ components hung with same signature) "
        "trips the systemic-pattern banner. Without this, the loop "
        "silently iterates on hung stubs instead of redirecting the "
        "LLM to fix the shared harness path."
    )


def test_scaffolder_templates_default_l1_small_size_nonzero() -> None:
    """Regression observed 2026-05-22 on SAM2-hiera-small: the
    scaffolder's `@pytest.mark.parametrize('device_params', [{}],
    indirect=True)` template opens the device with `l1_small_size=0`.
    Every `ttnn.conv2d` then raises `TT_FATAL: bank size is 0 B` and
    the autofill stub falls back to torch-on-host — turning a 30s
    PCC test into a 5-15min one, which blows the 10-min auto-iterate
    pre-flight pytest budget before a SINGLE test completes.

    Pin: BOTH scaffolder pytest templates (the regular and the
    op-synth variants) in `bringup_loop.py` MUST default to
    `l1_small_size=24576` (or any other non-zero value the repo's
    vision demos use). A regression to `{}` reintroduces the storm."""
    from pathlib import Path

    src = (Path(cli.__file__).parent / "bringup_loop.py").read_text()
    forbidden = '@pytest.mark.parametrize("device_params", [{{}}], indirect=True)'
    required = '@pytest.mark.parametrize("device_params", [{{"l1_small_size":'
    assert forbidden not in src, (
        "scaffolder templates must NOT use the empty `[{}]` device_params "
        "form (opens device with l1_small_size=0 -> conv2d storm). "
        'Use `[{{"l1_small_size": 24576}}]` instead.'
    )

    assert src.count(required) >= 2, (
        "expected the l1_small_size=24576 default in BOTH scaffolder "
        "templates (regular pytest template at ~line 420 and op-synth "
        "template at ~line 957), got fewer occurrences"
    )


def test_upgrade_test_to_set_l1_small_size_helper_exists() -> None:
    """The in-place fixer for stale `[{}]` device_params on
    already-scaffolded test files must exist and be exported from
    `capture_inputs`. Without it, demos created with the OLD template
    silently keep blowing the budget forever — the user's only
    recovery would be `rm -rf demo_dir` + re-scaffold."""
    import importlib

    capture_inputs = importlib.import_module("scripts.tt_hw_planner.capture_inputs")
    assert hasattr(capture_inputs, "upgrade_test_to_set_l1_small_size"), (
        "capture_inputs.upgrade_test_to_set_l1_small_size must be "
        "defined so the auto-iterate pre-flight repairs stale demos"
    )


def test_upgrade_test_to_set_l1_small_size_is_idempotent(tmp_path) -> None:
    """The in-place fixer must be idempotent: running it twice in a
    row must leave the file unchanged on the second call (returns
    False). The auto-iterate pre-flight invokes it on every run, so
    a non-idempotent fixer would corrupt the file on iter 2."""
    import importlib

    capture_inputs = importlib.import_module("scripts.tt_hw_planner.capture_inputs")
    test_file = tmp_path / "test_x.py"
    test_file.write_text(
        "import pytest\n\n"
        '@pytest.mark.parametrize("device_params", [{}], indirect=True)\n'
        "def test_x(device_params, device):\n    pass\n"
    )
    changed1 = capture_inputs.upgrade_test_to_set_l1_small_size(test_file)
    assert changed1 is True, "first call on a stale `[{}]` file must report modified=True"
    assert '"l1_small_size": 24576' in test_file.read_text()
    assert "[{}]" not in test_file.read_text()
    changed2 = capture_inputs.upgrade_test_to_set_l1_small_size(test_file)
    assert changed2 is False, "second call on the now-patched file must report " "modified=False (idempotent)"


def test_upgrade_test_to_set_l1_small_size_no_op_on_clean_files(
    tmp_path,
) -> None:
    """A test file that already has a non-empty device_params must
    NOT be touched — the user may have intentionally set a different
    `l1_small_size` (e.g. 79104 for full segmentation) and we must
    not overwrite that."""
    import importlib

    capture_inputs = importlib.import_module("scripts.tt_hw_planner.capture_inputs")
    test_file = tmp_path / "test_x.py"
    original_content = (
        "import pytest\n\n"
        '@pytest.mark.parametrize("device_params", '
        '[{"l1_small_size": 79104, "trace_region_size": 6434816}], '
        "indirect=True)\n"
        "def test_x(device_params, device):\n    pass\n"
    )
    test_file.write_text(original_content)
    changed = capture_inputs.upgrade_test_to_set_l1_small_size(test_file)
    assert changed is False, "must not touch already-configured files"
    assert test_file.read_text() == original_content, (
        "in-place fixer must NEVER overwrite a user-configured "
        "device_params (could clobber l1_small_size=79104 etc.)"
    )


def test_upgrade_all_tests_in_demo_chains_both_upgraders() -> None:
    """`upgrade_all_tests_in_demo` must invoke BOTH
    `upgrade_test_to_use_captured_inputs` AND
    `upgrade_test_to_set_l1_small_size` for every test file.
    Source-level pin so a refactor doesn't accidentally drop the
    l1_small_size fixer back out of the pre-flight loop."""
    from pathlib import Path

    src = (Path(cli.__file__).parent / "capture_inputs.py").read_text()
    upgrade_all_idx = src.find("def upgrade_all_tests_in_demo")
    assert upgrade_all_idx >= 0
    fn_body = src[upgrade_all_idx : upgrade_all_idx + 3000]
    assert "upgrade_test_to_use_captured_inputs" in fn_body, (
        "upgrade_all_tests_in_demo must still call the captured-" "inputs upgrader (preserves existing behavior)"
    )
    assert "upgrade_test_to_set_l1_small_size" in fn_body, (
        "upgrade_all_tests_in_demo must ALSO call the new "
        "l1_small_size fixer so stale demos are auto-repaired on "
        "every auto-iterate pre-flight"
    )


def test_preflight_hang_targeting_block_exists_in_source() -> None:
    """Regression observed 2026-05-22 on SAM2-hiera-small: when the
    pre-flight pytest sweep timed out (rc=124), the loop's per-component
    handling (lines 4988-5056) was guarded by `if _preflight_report:`,
    which is empty on timeout because pytest gets SIGKILL'd before its
    JUnit finalizer runs. The iter loop then entered iter 1 with
    `last_failed_components == []`, and `_pick_target` fell back to
    alphabetical-by-name — so a hang in `vision_config` (last
    alphabetically) made the LLM spend its first 15-min budget on
    `decoder_head` (first alphabetically).

    Pin: a block must exist between the pre-flight pytest call and the
    `if _preflight_report:` block that reads `_LAST_PYTEST_STAGES`,
    extracts the LAST inserted component (the one mid-execution at
    SIGKILL time), and sets `last_failed_components` so iter 1 targets
    the actual culprit.

    Source-level test because the auto-iterate loop is a single 500+
    line function with many local-state mutations that aren't easy to
    exercise without a real pytest run."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    assert fn_idx > 0, "_run_auto_iterate_loop must exist"

    fn_slice = src[fn_idx : fn_idx + 200000]

    hang_idx = fn_slice.find("if _preflight_rc == 124")
    assert hang_idx > 0, (
        "the pre-flight HANG targeting fix must detect a timeout "
        "(rc == 124) before falling through to the report-handler "
        "(which is skipped on timeout because XML never wrote)"
    )

    hang_block = fn_slice[hang_idx : hang_idx + 4000]
    assert "_LAST_PYTEST_STAGES" in hang_block, (
        "the pre-flight HANG targeting fix must read "
        "_LAST_PYTEST_STAGES to infer the hung component (otherwise "
        "iter 1 falls back to alphabetical targeting and burns a "
        "Claude budget on the wrong component)"
    )
    assert "last_failed_components" in hang_block, (
        "the pre-flight HANG targeting fix must SET "
        "last_failed_components so `_pick_target` picks the inferred "
        "culprit on iter 1"
    )


def test_preflight_hang_target_is_last_stage_entry() -> None:
    """The pre-flight HANG targeting fix must pick the LAST insertion-
    order entry from `_LAST_PYTEST_STAGES`, not the first. pytest runs
    tests sequentially in cmd-line order, and earlier tests either
    PASSED or FAILED before the timeout — only the LAST one is
    mid-execution at SIGKILL time. Picking the first would target a
    test that already completed, which is a useless target.

    Source-level pin guarding the [-1] indexing."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]

    hang_idx = fn_slice.find("if _preflight_rc == 124")
    assert hang_idx > 0, "HANG targeting block must guard on `_preflight_rc == 124`"

    between = fn_slice[hang_idx : hang_idx + 4000]
    assert "_pf_ordered_dedup[-1]" in between, (
        "the pre-flight HANG targeting must select [-1] (last entry) "
        "from the deduped ordered stage list — that's the test mid-"
        "execution at SIGKILL. Selecting [0] would target an "
        "already-completed test."
    )


def test_preflight_compute_split_printed_before_iter_loop() -> None:
    """Regression observed 2026-05-22 on SAM2-hiera-small: when the
    pre-flight pytest sweep hung (10 min wall-clock), the very first
    "% on device / % on CPU" number the user saw was AFTER iter 1's
    pytest — i.e. ~25 minutes into the run. That made the early phase
    of a bring-up feel blind, especially for users investigating
    whether the autofilled stubs were graduating or still on CPU
    fallback. The per-iter compute split prints (at lines ~6444 and
    ~6817) cover the iter loop but NOT the pre-flight.

    Pin: a `_format_compute_split` call must exist between the
    `if _preflight_report:` block and the `for it in range(1,
    max_iters + 1):` iter loop opener, so the pre-flight settles its
    baseline split before iter 1 starts."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]

    pf_block_idx = fn_slice.find("AUTO-ITERATE pre-flight:")
    iter_loop_idx = fn_slice.find("while True:")
    assert pf_block_idx > 0, "pre-flight banner must exist"
    assert iter_loop_idx > pf_block_idx, "the iter loop must come AFTER the pre-flight (sanity check)"
    pre_iter_slice = fn_slice[pf_block_idx:iter_loop_idx]
    assert "Pre-flight compute split" in pre_iter_slice, (
        "the pre-flight block must print a 'Pre-flight compute split' "
        "banner BEFORE the iter loop opens, so the user sees the "
        "baseline % on device / % on CPU number without waiting for "
        "iter 1 to finish"
    )
    assert "_format_compute_split(" in pre_iter_slice, (
        "the pre-flight compute split must call _format_compute_split "
        "(the same helper the per-iter prints use, for consistent "
        "formatting)"
    )
    assert "_format_op_split(" in pre_iter_slice, (
        "the pre-flight compute split must ALSO call _format_op_split "
        "so the user sees BOTH the component-level and op-level "
        "breakdown (same as per-iter prints)"
    )


def test_preflight_compute_split_is_exception_safe() -> None:
    """A bug in `_format_compute_split` / `_format_op_split` must
    NOT abort the auto-iterate loop — the compute-split is purely
    informational. The pre-flight wrapper must catch exceptions and
    fall through to the iter loop with a diagnostic, never propagate."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]
    pf_block_idx = fn_slice.find("AUTO-ITERATE pre-flight:")
    iter_loop_idx = fn_slice.find("while True:")
    pre_iter_slice = fn_slice[pf_block_idx:iter_loop_idx]

    split_idx = pre_iter_slice.find("Pre-flight compute split")
    assert split_idx > 0
    pre_split = pre_iter_slice[:split_idx]

    last_try = pre_split.rfind("try:")
    assert last_try > 0, (
        "the pre-flight compute split must be inside a try/except so " "a formatting bug never aborts the bring-up loop"
    )

    post_split = pre_iter_slice[split_idx:]
    assert "except Exception" in post_split, (
        "the pre-flight compute split try block must have an "
        "`except Exception` handler that logs and continues, never "
        "re-raises"
    )


def test_preflight_hang_handles_zero_stage_lines_gracefully() -> None:
    """If the pre-flight pytest times out BEFORE printing any
    `[bringup] stage=` line (e.g. hang in module import or scaffold
    load), the inference can't identify a culprit. The fix must NOT
    crash with an IndexError; it must fall through to the default
    iter-loop behavior with a clear log message.

    Source-level pin: the [-1] indexing must be inside a non-empty
    guard."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]

    hang_idx = fn_slice.find("if _preflight_rc == 124")
    assert hang_idx > 0
    between = fn_slice[hang_idx : hang_idx + 4000]

    assert "if _pf_ordered_dedup:" in between, (
        "the LAST-entry selection must be inside an `if "
        "_pf_ordered_dedup:` guard — otherwise an empty stage map "
        "(hang before build_torch_reference) raises IndexError"
    )

    assert "before ANY" in between or "[bringup] stage line" in between, (
        "the else-branch for the empty-stage-map case must emit a "
        "diagnostic log so the user knows targeting fell back to "
        "alphabetical"
    )


def test_agent_complexity_timeout_preserves_base_when_no_bonus() -> None:
    """Without a complexity bonus the user's `--auto-agent-timeout`
    value must be returned unchanged. The bonus only ADDS budget for
    heavier components; it never reduces it. Pinning so a refactor
    that accidentally clamps to a lower ceiling can't slip in."""
    f = cli._agent_complexity_timeout
    assert f(900, 0) == 900
    assert f(1500, 0) == 1500
    assert f(600, 0) == 600


def test_agent_complexity_timeout_adds_5min_per_unit() -> None:
    """Each complexity unit (1..4 from
    `_component_complexity_bonus`) must add exactly 5 minutes = 300s
    of budget so the increment is predictable and matches the
    documented behavior in the function's docstring."""
    f = cli._agent_complexity_timeout
    assert f(900, 1) == 900 + 300
    assert f(900, 2) == 900 + 600
    assert f(900, 3) == 900 + 900
    assert f(900, 4) == 900 + 1200


def test_agent_complexity_timeout_clamps_at_20min_extra() -> None:
    """The bonus must be capped at +20 min (1200s) total regardless
    of the requested bonus. Defends against a future refactor of
    `_component_complexity_bonus` that returns >4 (it currently
    clamps to 4 itself, but the agent timeout has its own ceiling
    as a belt-and-suspenders measure to prevent runaway budgets)."""
    f = cli._agent_complexity_timeout
    assert f(900, 4) == 2100, "expected base 900 + cap 1200"
    assert f(900, 5) == 2100, "expected cap to apply at bonus=5"
    assert f(900, 99) == 2100, "expected cap to apply at bonus=99"


def test_agent_complexity_timeout_unbounded_passthrough() -> None:
    """If the user passes `timeout_s=0` (unbounded), the complexity
    bonus must NOT magically introduce a finite ceiling. Bonus is
    additive on top of the base; `0 + N` is still 0 in our
    contract."""
    f = cli._agent_complexity_timeout
    assert f(0, 0) == 0
    assert f(0, 4) == 0
    assert f(-1, 4) == -1, (
        "negative timeout (defensive, shouldn't happen) must " "passthrough so the caller can detect its own bad input"
    )


def test_invoke_agent_uses_stream_json_for_claude() -> None:
    """Regression observed 2026-05-22 on SAM2-hiera-small: with
    `--output-format text`, claude produces zero stdout until its
    FINAL response, so the heartbeat loop sees `log=quiet` for the
    entire 15-min budget even while claude is actively reading the
    codebase via its `Read` tool. The loop then kills claude
    mid-investigation. Switching to `stream-json` emits one NDJSON
    event per tool call / model message, which the heartbeat parses
    to detect real progress.

    Pin: the claude branch of `_invoke_agent` MUST use
    `stream-json`, NOT `text`."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _invoke_agent")
    assert fn_idx > 0
    fn_end = src.find("def ", fn_idx + 100)

    fn_slice = src[fn_idx : fn_end if fn_end > fn_idx else fn_idx + 30000]

    claude_idx = fn_slice.find('elif provider == "claude":')
    assert claude_idx > 0, "claude branch must exist"
    claude_branch = fn_slice[claude_idx : claude_idx + 2000]
    assert '"stream-json"' in claude_branch, (
        "claude branch must use --output-format stream-json so the "
        "heartbeat can detect tool-call progress; --output-format "
        "text produces zero stdout until the final response"
    )
    assert '"--verbose"' in claude_branch, (
        "claude CLI requires --verbose when -p + stream-json are "
        "combined; omitting it causes the CLI to reject the args"
    )


def test_invoke_agent_collects_descendants_before_kill() -> None:
    """The claude CLI internally fork()+exit()s its wrapper,
    leaving the actual worker reparented to init the moment we
    SIGTERM. `killpg` only signals the SAME process group, so the
    worker escapes — observed 2026-05-22 as a 40-minute orphan
    still making Anthropic API calls.

    `_kill_agent_tree` must call `_collect_agent_descendant_pids`
    BEFORE signaling so the descendant list is captured while the
    parent is still alive (after kill, descendants get reparented
    to init and the link is lost). Pin via source inspection."""
    assert hasattr(cli, "_collect_agent_descendant_pids"), "the descendant-walking helper must exist"
    assert hasattr(cli, "_kill_agent_tree"), "the kill helper must exist"
    from pathlib import Path

    src = _planner_source()

    kill_idx = src.find("def _kill_process_tree")
    assert kill_idx > 0, "the generalized helper `_kill_process_tree` must exist"
    next_def = src.find("\ndef ", kill_idx + 10)
    kill_body = src[kill_idx : next_def if next_def > 0 else kill_idx + 5000]

    collect_idx = kill_body.find("_collect_agent_descendant_pids(")
    signal_term_idx = kill_body.find("signal.SIGTERM")
    assert collect_idx > 0, "must call _collect_agent_descendant_pids"
    assert signal_term_idx > 0, (
        "must reference `signal.SIGTERM` in the actual signal-send " "call (not just in the docstring)"
    )
    assert collect_idx < signal_term_idx, (
        "_collect_agent_descendant_pids MUST be called BEFORE the "
        "SIGTERM is sent — otherwise the immediate child dies first "
        "and its descendants get reparented to init, making them "
        "untraceable from our point of view"
    )


def test_invoke_agent_signals_descendants_directly() -> None:
    """The kill helper must signal each descendant PID with
    `os.kill(pid, sig)` directly — not just rely on the process
    group via `killpg`. The whole point of the /proc walk is to
    catch detached processes that `killpg` misses."""
    from pathlib import Path

    src = _planner_source()
    kill_idx = src.find("def _kill_process_tree")
    next_def = src.find("\ndef ", kill_idx + 10)
    kill_body = src[kill_idx : next_def if next_def > 0 else kill_idx + 5000]

    assert "os.kill(" in kill_body, (
        "_kill_process_tree must use `os.kill(pid, sig)` to signal "
        "individual descendant PIDs (killpg alone misses processes "
        "that left their parent's group via setsid())"
    )
    assert "os.killpg(" in kill_body, (
        "_kill_process_tree should ALSO call killpg as a secondary "
        "channel — covers in-group siblings of the immediate child"
    )


def test_parse_stream_json_event_robust_to_garbage() -> None:
    """`stream-json` log lines are mixed with the occasional plain
    text diagnostic line (claude CLI prints some warnings via
    stderr which gets merged into the same log). The parser must
    cleanly return None on anything that isn't a JSON object — not
    raise, not return a non-dict."""
    f = cli._parse_stream_json_event
    assert f("") is None
    assert f("   ") is None
    assert f("not json") is None
    assert f("[1,2,3]") is None, "arrays must be rejected"
    assert f("{invalid") is None, "malformed JSON must be rejected"
    assert f('"plain string"') is None, "strings must be rejected"
    parsed = f('{"type": "assistant", "message": null}')
    assert parsed == {"type": "assistant", "message": None}


def test_summarize_stream_json_event_counts_tool_use_blocks() -> None:
    """Claude `stream-json` emits an `assistant` event whose message
    `content` is a list of blocks; each block has a `type` field
    that can be `text` or `tool_use`. Counting just the assistant
    events undercounts progress because one assistant message can
    contain many tool calls (parallel tools). The summary must
    count `tool_use` blocks inside the assistant message
    separately."""
    counts: Dict[str, int] = {}
    cli._summarize_stream_json_event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read"},
                    {"type": "tool_use", "name": "Grep"},
                    {"type": "text", "text": "..."},
                ],
            },
        },
        counts,
    )
    assert counts.get("assistant") == 1
    assert counts.get("tool_use") == 2, "must count EACH tool_use block, not just the parent " "assistant message"


def test_summarize_stream_json_event_handles_missing_fields() -> None:
    """The parser must NOT crash on events with unexpected shapes
    (missing `message`, missing `content`, non-list `content`,
    non-dict blocks). Real-world stream-json from different claude
    CLI versions has varying schemas."""
    counts: Dict[str, int] = {}

    cli._summarize_stream_json_event({"type": "assistant"}, counts)
    assert counts.get("assistant") == 1

    cli._summarize_stream_json_event(
        {"type": "assistant", "message": {"content": "not a list"}},
        counts,
    )

    cli._summarize_stream_json_event(
        {"type": "assistant", "message": {"content": ["raw string"]}},
        counts,
    )

    cli._summarize_stream_json_event({"type": "result"}, counts)
    assert counts.get("result") == 1

    cli._summarize_stream_json_event({"type": "blizzard"}, counts)
    assert counts.get("other") == 1


def test_read_proc_rchar_returns_int_for_live_process() -> None:
    """The /proc/io rchar reader is a critical progress signal
    (catches "claude's Read tool is actively reading files" even
    before any stream-json event lands). Must return a positive int
    for any live Linux process with task IO accounting enabled."""
    import os

    rchar = cli._read_proc_rchar(os.getpid())
    if rchar is None:
        import pytest

        pytest.skip("kernel built without CONFIG_TASK_IO_ACCOUNTING")
    assert isinstance(rchar, int) and rchar > 0


def test_read_proc_rchar_none_for_dead_process() -> None:
    """Reading rchar for a non-existent PID must return None, not
    raise. The heartbeat treats None as "no progress signal from
    this source" and falls through to the other signals."""

    result = cli._read_proc_rchar(999999)
    assert result is None


def test_collect_agent_descendant_pids_returns_list() -> None:
    """The descendant collector must return a list of ints
    (possibly empty). Pinning so a refactor that switches to
    `set` or `Iterator` would break callers that index/length-
    check."""
    import os

    result = cli._collect_agent_descendant_pids(os.getpid())
    assert isinstance(result, list)
    for pid in result:
        assert isinstance(pid, int)


def test_invoke_agent_signature_accepts_complexity_and_iter_tag() -> None:
    """The new kwargs `complexity_bonus` and `iter_tag` must be
    keyword-only AND have defaults that preserve existing behavior.
    A caller that doesn't pass them must get the same default
    behavior as before."""
    import inspect

    sig = inspect.signature(cli._invoke_agent)
    params = sig.parameters
    assert "complexity_bonus" in params
    assert params["complexity_bonus"].default == 0, (
        "complexity_bonus must default to 0 so existing callers " "get the unchanged base timeout"
    )
    assert params["complexity_bonus"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "iter_tag" in params
    assert params["iter_tag"].default is None, (
        "iter_tag must default to None so existing callers fall "
        "back to the legacy `<provider>_last_run.log` filename"
    )


def test_invoke_agent_call_site_passes_complexity_bonus() -> None:
    """The call site inside `_run_auto_iterate_loop` must compute
    the complexity bonus for the iter target component and pass it
    to `_invoke_agent` — otherwise the new bumped-timeout feature
    is dead code."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 300000]
    call_idx = fn_slice.find("rc = _invoke_agent(")
    assert call_idx > 0, "call site `rc = _invoke_agent(` must exist"

    call_block = fn_slice[max(0, call_idx - 3000) : call_idx + 2500]
    assert "complexity_bonus=" in call_block, (
        "call site must pass complexity_bonus kwarg so heavy " "components get the bumped timeout"
    )
    assert "iter_tag=" in call_block, (
        "call site must pass iter_tag so each iter's log is " "preserved (otherwise iter 2 overwrites iter 1's log)"
    )
    assert "_component_complexity_bonus(" in call_block, (
        "call site must compute the bonus via "
        "`_component_complexity_bonus` for consistency with the "
        "attempt-cap logic"
    )


def test_invoke_agent_uses_stall_detection() -> None:
    """The heartbeat loop must detect a TRUE stall (no progress
    signal of any kind for `stall_budget_s`) and kill early,
    rather than waiting for the full wall-clock budget. Without
    this, a hung agent burns the entire 15-30 minute budget for
    nothing."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _invoke_agent")
    next_def = src.find("\ndef ", fn_idx + 10)
    fn_body = src[fn_idx : next_def if next_def > 0 else fn_idx + 50000]
    assert "stall_budget_s" in fn_body, (
        "_invoke_agent must define a `stall_budget_s` to bound " "no-progress time before kill"
    )
    assert "STALL DETECTED" in fn_body, (
        "stall path must log a banner so the user can distinguish "
        "stall-kill from wall-clock-kill in the iteration log"
    )
    assert "last_progress_t" in fn_body, (
        "stall path must track a `last_progress_t` timestamp that "
        "gets refreshed by ANY of: log growth, /proc rchar growth, "
        "or new stream-json events"
    )


def test_invoke_agent_async_stdin_write_does_not_block_heartbeat() -> None:
    """A 50-100 KiB prompt is larger than the default Linux pipe
    buffer (64 KiB). A synchronous `proc.stdin.write(prompt)` in
    `_invoke_agent` would block the heartbeat loop for as long as
    the kernel takes to drain the pipe. Use a daemon thread so the
    write and the heartbeat are decoupled."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _invoke_agent")
    next_def = src.find("\ndef ", fn_idx + 10)
    fn_body = src[fn_idx : next_def if next_def > 0 else fn_idx + 50000]

    assert "_write_prompt_async" in fn_body, "_invoke_agent must define a helper for async stdin write"
    assert "threading.Thread(" in fn_body, (
        "_invoke_agent must start the prompt writer in a Thread " "so the heartbeat loop is decoupled from a slow stdin"
    )
    assert "daemon=True" in fn_body, (
        "the prompt-writer thread must be daemon so it doesn't " "block process exit if the subprocess crashes early"
    )


def test_strategy_directive_for_partial_cpu_fallback_exists() -> None:
    """PARTIAL_CPU_FALLBACK is a new failure class introduced to handle
    components whose PCC test passes but still have one or more
    `_apply_*` helpers running on CPU at runtime. The strategy
    directive must (a) acknowledge PCC is passing, (b) tell the LLM
    to ONLY touch the named helpers (not the whole __call__), and
    (c) require PCC to remain passing after the rewrite."""
    directive = cli._strategy_directive_for_failure("PARTIAL_CPU_FALLBACK")
    assert "PARTIAL_CPU_FALLBACK" in directive

    assert "PCC" in directive and "PASSES" in directive

    assert "Do NOT" in directive

    assert "still pass" in directive.lower() or "must" in directive.lower()

    assert "_apply_" in directive


def test_partial_cpu_components_helper_reads_compute_split() -> None:
    """The candidate-pool gating logic must use a single source of
    truth (the compute-split's `new_native_partial_cpu_names`) to
    decide which components are partial-CPU. Otherwise the compute
    split (what the user SEES) and the iteration loop (what the user
    GETS) can disagree — e.g. UI says '99% on device' but the loop
    declares done."""
    assert hasattr(cli, "_partial_cpu_components"), "_partial_cpu_components helper must exist"

    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _partial_cpu_components")
    assert fn_idx > 0
    next_def = src.find("\ndef ", fn_idx + 10)
    body = src[fn_idx : next_def if next_def > 0 else fn_idx + 2000]
    assert "_compute_split(" in body, (
        "_partial_cpu_components must read from _compute_split so the "
        "UI and the iteration loop never disagree about who is "
        "partial-CPU"
    )
    assert "new_native_partial_cpu_names" in body, (
        "_partial_cpu_components must reference the canonical key "
        "(`new_native_partial_cpu_names`) that the compute split "
        "populates"
    )


def test_runtime_fallback_details_returns_helpers_and_kinds() -> None:
    """The LLM prompt for PARTIAL_CPU_FALLBACK needs to know both
    WHICH helpers fall back AND WHAT op kind each one wraps (so it
    can choose the right ttnn op as a replacement). The detail
    helper must surface both."""
    assert hasattr(cli, "_runtime_fallback_details"), "_runtime_fallback_details helper must exist"

    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _runtime_fallback_details")
    next_def = src.find("\ndef ", fn_idx + 10)
    body = src[fn_idx : next_def if next_def > 0 else fn_idx + 2000]
    assert '"helpers"' in body and '"kinds"' in body, (
        "_runtime_fallback_details must return both `helpers` and "
        "`kinds` so the LLM knows what op kind to convert to ttnn"
    )


def test_iterate_loop_adds_partial_cpu_to_candidate_pool() -> None:
    """The auto-iterate loop must add partial-CPU components back into
    the candidate pool. Without this, the loop exits the moment PCC
    passes — even if 1/409 ops is still on CPU — defeating the
    purpose of an autonomous bring-up tool.

    Source-pin: the candidate_pool construction must reference both
    `partial_cpu_set` (the union side) and subtract only
    `graduated_this_run - partial_cpu_set` (so graduated components
    that ARE partial-CPU remain in the pool)."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]
    pool_idx = fn_slice.find("candidate_pool = sorted(")
    assert pool_idx > 0, "candidate_pool construction must exist"
    pool_block = fn_slice[pool_idx : pool_idx + 2000]
    assert "partial_cpu_set" in pool_block, (
        "candidate_pool must union in `partial_cpu_set` so the loop "
        "keeps targeting components with runtime CPU fallbacks"
    )
    assert "- (set(graduated_this_run) - partial_cpu_set)" in pool_block, (
        "candidate_pool must subtract only the graduated components "
        "that are NOT also partial-CPU — otherwise the pool drops a "
        "partial-CPU component the moment it first passes PCC"
    )


def test_partial_cpu_override_guarded_by_prev_failure_state() -> None:
    """Bug-1 regression: the PARTIAL_CPU_FALLBACK override must NOT
    fire if the previous iteration's pytest failed on this component.
    If it does, the prompt lies to the LLM ("PCC passes, just fix
    the helper") while the on-disk stub is actually broken (e.g. a
    prior partial-CPU rewrite regressed PCC, or the test hung).

    Pinned via source inspection so a refactor can't accidentally
    drop the guard.
    """
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]
    override_idx = fn_slice.find('failure_class = "PARTIAL_CPU_FALLBACK"')
    assert override_idx > 0, "PARTIAL_CPU_FALLBACK override must exist"

    guard_window = fn_slice[max(0, override_idx - 2000) : override_idx]
    assert "_target_known_broken" in guard_window, (
        "PARTIAL_CPU_FALLBACK override must be guarded by "
        "`_target_known_broken` to avoid lying to the LLM when the "
        "stub is currently broken (prev iter PCC fail / hang / OOM)"
    )
    assert "verified_fail" in guard_window, (
        "`_target_known_broken` must consult `verified_fail` "
        "(persistent PCC-fail flag) so the guard survives an iter "
        "boundary"
    )
    assert "_prev_iter_failed" in guard_window, (
        "`_target_known_broken` must also consult `_prev_iter_failed` "
        "(snapshot of `last_failed_components` BEFORE it is "
        "clobbered with `[iter_target_component]`) so the guard "
        "catches HANG / L1_OOM (which don't populate `verified_fail`)"
    )


def test_persist_runtime_fallbacks_clears_stale_entries() -> None:
    """Bug-2 regression: when a PCC test passes with ZERO fallback
    events, `_persist_runtime_fallbacks` must clear the prior stale
    entry for that component. Without this, a successful LLM rewrite
    that removes a CPU fallback is invisible to the loop, which keeps
    targeting the component as still partial-CPU forever."""
    import inspect

    sig = inspect.signature(cli._persist_runtime_fallbacks)
    assert "tested_components" in sig.parameters, (
        "_persist_runtime_fallbacks must accept `tested_components` "
        "so callers can declare which components were actually "
        "exercised by the pytest run that just produced the drain"
    )
    assert sig.parameters["tested_components"].default is None, (
        "`tested_components` must default to None for backward " "compatibility (drain-only-overlay behavior preserved)"
    )

    import tempfile
    from pathlib import Path
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        from unittest.mock import patch

        demo_dir = Path(tmpdir)
        persisted = demo_dir / "_runtime_fallbacks.json"
        persisted.write_text(
            json.dumps(
                {
                    "vision_config": {
                        "conv2d": ["_apply_x"],
                        "helpers": ["_apply_x"],
                        "kinds": ["conv2d"],
                    },
                    "other_comp": {
                        "matmul": ["_apply_y"],
                        "helpers": ["_apply_y"],
                        "kinds": ["matmul"],
                    },
                }
            )
        )

        with patch.object(
            cli,
            "_runtime_fallback_paths",
            return_value=(demo_dir / "_runtime_fallbacks.jsonl", persisted),
        ):
            cli._persist_runtime_fallbacks(
                "fake-model",
                drained={},
                tested_components=["vision_config"],
            )
        after = json.loads(persisted.read_text())
        assert "vision_config" not in after, (
            "tested-but-no-events component MUST be cleared from the "
            "persisted file — otherwise the loop loops forever on a "
            "fallback that has actually been fixed"
        )
        assert "other_comp" in after, (
            "untested components MUST be preserved (focused pytest "
            "reruns must not zero out unrelated components' state)"
        )


def test_persist_runtime_fallbacks_preserves_when_drain_has_events() -> None:
    """When the drain DOES have events for a tested component, those
    events overlay the prior state (not clear it). Lock this so a
    future refactor doesn't accidentally turn the clear-on-clean
    logic into clear-always."""
    import tempfile
    from pathlib import Path
    import json
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_dir = Path(tmpdir)
        persisted = demo_dir / "_runtime_fallbacks.json"
        persisted.write_text("{}")
        with patch.object(
            cli,
            "_runtime_fallback_paths",
            return_value=(demo_dir / "_runtime_fallbacks.jsonl", persisted),
        ):
            cli._persist_runtime_fallbacks(
                "fake-model",
                drained={
                    "vision_config": {
                        "conv2d": ["_apply_x"],
                        "helpers": ["_apply_x"],
                        "kinds": ["conv2d"],
                    },
                },
                tested_components=["vision_config"],
            )
        after = json.loads(persisted.read_text())
        assert "vision_config" in after, (
            "drain events must overlay (not be cleared by) the " "tested-component clear logic"
        )
        assert "_apply_x" in after["vision_config"]["helpers"]


def test_convergence_check_blocked_by_live_partial_cpu() -> None:
    """Bug-3 regression: convergence (`converged_native` and
    `converged_partial`) must consult `_partial_cpu_block_convergence`.
    Without this gate, the loop hits the legacy "all PCC pass" exit
    BEFORE the candidate-pool fix has a chance to push the loop into
    another iteration on partial-CPU targets — observed 2026-05-22
    20:00 UTC on sam2-hiera-small (the loop printed "model runs
    natively on QB2 (no CPU fallback)" with NEW-partial-CPU=1 still
    in the compute split above it).

    Pinned via source inspection so a refactor can't drop the gate.
    """
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]
    gate_idx = fn_slice.find("_partial_cpu_block_convergence")
    assert gate_idx > 0, "convergence path must define `_partial_cpu_block_convergence`"

    count = fn_slice.count("_partial_cpu_block_convergence")
    assert count >= 4, (
        f"`_partial_cpu_block_convergence` must be used in BOTH "
        f"convergence-eval sites (main + post-sweep). Found "
        f"{count} occurrence(s) — need at least 4 (2 defs + 2 uses "
        f"each in converged_native and converged_partial)"
    )


def test_partial_cpu_block_convergence_filters_capped_out_components() -> None:
    """The block-convergence gate must exclude `permanently_skipped`
    components. A partial-CPU component that hit its attempt cap
    has been restored to its last working version by
    `_skip_component_to_fallback` — it must NOT keep the loop alive
    indefinitely, otherwise the loop spins forever on a component
    whose remaining op genuinely cannot be expressed in native TTNN.
    """
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]

    set_idx = fn_slice.find("partial_cpu_set = ")
    assert set_idx > 0, "partial_cpu_set construction must exist"
    set_block = fn_slice[set_idx : set_idx + 300]
    assert "set(permanently_skipped)" in set_block, (
        "`partial_cpu_set` must subtract `permanently_skipped` so "
        "cap'd-out partial-CPU components don't keep the loop alive"
    )
    assert "set(partial_cpu_pool)" in set_block, "`partial_cpu_set` must derive from `partial_cpu_pool`"


def test_convergence_msg_describes_partial_cpu_when_allowed() -> None:
    """When `--allow-partial-cpu` short-circuits the gate and the
    loop exits with partial-CPU still on disk, the convergence
    banner must NOT falsely claim "no CPU fallback". Operator
    visibility: the user passed `--allow-partial-cpu` intending to
    halt at PCC-pass; the banner should make it explicit that some
    ops are still on CPU."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]

    msg_idx = fn_slice.find('(no CPU fallback)"')
    assert msg_idx > 0, "converge_msg `no CPU fallback` branch must exist"

    surrounding = fn_slice[max(0, msg_idx - 2000) : msg_idx]
    assert "_live_partial_cpu" in surrounding, (
        "converge_msg construction must branch on whether "
        "`_live_partial_cpu` is non-empty so we don't print "
        "'no CPU fallback' when there IS a CPU fallback"
    )

    assert "--allow-partial-cpu" in surrounding, (
        "the partial-CPU converge_msg branch must reference "
        "--allow-partial-cpu so the user understands WHY the loop "
        "exited with CPU fallbacks still present"
    )


def test_partial_cpu_diagnostic_banner_emitted_before_iter_continues() -> None:
    """When convergence is blocked SOLELY by partial-CPU (not by
    failed components or ungraduated stubs), the loop must print a
    visible diagnostic explaining why it's continuing. Without this,
    a user sees compute split saying "100% on device" then a
    confusing iter N+1 banner with no explanation."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]

    diag_idx = fn_slice.find("PCC tests pass but")
    assert diag_idx > 0, "diagnostic banner explaining partial-CPU continue must exist"
    diag_block = fn_slice[max(0, diag_idx - 800) : diag_idx + 800]
    assert "--allow-partial-cpu" in diag_block, (
        "diagnostic must mention --allow-partial-cpu as the user's " "escape hatch"
    )
    assert "continuing to push toward 100%" in diag_block, (
        "diagnostic must state that the loop is pushing for 100% "
        "on-device, so the user knows convergence is intentional"
    )


def test_drain_now_passes_tested_components_to_persist() -> None:
    """The `_drain_now` closure inside `_run_focused_pytest` must
    derive tested components from `test_files` and forward them to
    `_persist_runtime_fallbacks`. Pinned at the source level so a
    refactor can't drop the wiring."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_focused_pytest")
    fn_slice = src[fn_idx : fn_idx + 50000]
    drain_idx = fn_slice.find("def _drain_now")
    assert drain_idx > 0, "_drain_now closure must exist"
    drain_block = fn_slice[drain_idx : drain_idx + 3000]
    assert "test_files" in drain_block, "_drain_now must inspect test_files to derive tested " "components"
    assert "tested_components=" in drain_block, (
        "_drain_now must call _persist_runtime_fallbacks with the " "`tested_components=` kwarg"
    )


def test_iterate_loop_overrides_failure_class_for_partial_cpu() -> None:
    """When the iter target is a partial-CPU component, the loop must
    override `failure_class` to PARTIAL_CPU_FALLBACK so the LLM gets
    the partial-CPU directive (not the generic PCC/HANG/OOM ones).
    Without this override the directive comes from whatever
    `_classify_failure` returned for the LAST iter's failure — which
    might be 'PCC_ONLY' or 'OTHER' and is unhelpful here."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]
    override_idx = fn_slice.find('failure_class = "PARTIAL_CPU_FALLBACK"')
    assert override_idx > 0, (
        "iter loop must override failure_class to PARTIAL_CPU_FALLBACK " "when targeting a partial-CPU component"
    )

    surrounding = fn_slice[max(0, override_idx - 500) : override_idx + 100]
    assert "_target_partial_cpu_info" in surrounding, (
        "the override must be guarded by a check on whether the " "target is actually a partial-CPU component"
    )


def test_iterate_loop_allow_partial_cpu_flag_kwarg() -> None:
    """The new --allow-partial-cpu CLI flag must be threaded through
    to `_run_auto_iterate_loop` as a kwarg with default False (i.e.
    the loop pushes to 100% on device by default — the user must
    opt out)."""
    import inspect

    sig = inspect.signature(cli._run_auto_iterate_loop)
    assert "allow_partial_cpu" in sig.parameters, "_run_auto_iterate_loop must accept `allow_partial_cpu` kwarg"
    assert sig.parameters["allow_partial_cpu"].default is False, (
        "`allow_partial_cpu` must default to False so the loop " "continues iterating until 100% on device by default"
    )


def test_iterate_loop_partial_cpu_set_respects_flag() -> None:
    """When --allow-partial-cpu is True, `partial_cpu_pool` must be
    empty so the loop doesn't add partial-CPU components to the
    candidate pool. Pinned via source inspection so a future
    refactor can't drop this gate."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _run_auto_iterate_loop")
    fn_slice = src[fn_idx : fn_idx + 200000]
    pool_idx = fn_slice.find("partial_cpu_pool: List[str] = ")
    assert pool_idx > 0, "partial_cpu_pool definition must exist"
    block = fn_slice[pool_idx : pool_idx + 400]
    assert "allow_partial_cpu" in block, "partial_cpu_pool must be gated on the allow_partial_cpu flag"
    assert "[] if allow_partial_cpu else" in block, "when allow_partial_cpu is True, partial_cpu_pool must be []"


def test_classify_failure_recognizes_no_hardware() -> None:
    """The 'No chips detected' / 'num_chips > 0' signature must be
    classified as NO_HARDWARE so the auto-iterate loop's pre-flight
    bail-out can find it. Pinned 2026-05-22 after observing a
    SAM2-hiera-small bring-up burn through 45 min of Opus budget on
    a missing-kernel-driver issue that was misclassified as
    TT_FATAL_OPAQUE (because the error came packaged inside a
    TT_FATAL string)."""
    for summary, details in [
        (
            "RuntimeError: TT_FATAL ... num_chips > 0",
            "info:\nNo chips detected in the cluster\nbacktrace:",
        ),
        ("", "No chips detected in the cluster"),
        ("", "TT_FATAL @ tt_cluster.cpp:119: num_chips > 0"),
    ]:
        cls = cli._classify_failure(summary, details)
        assert cls == "NO_HARDWARE", (
            f"expected NO_HARDWARE for summary={summary!r} " f"details={details[:60]!r}, got {cls!r}"
        )


def test_classify_failure_no_hardware_beats_other_classes() -> None:
    """When the same failure text matches multiple patterns (e.g.
    contains both `TT_FATAL` and `No chips detected`), NO_HARDWARE
    must always win. Otherwise the fallback to TT_FATAL_OPAQUE
    would still trigger LLM invocation on a host-env issue."""
    text_with_many_patterns = (
        "TT_FATAL @ /home/ttuser/tt-metal/tt_metal/llrt/tt_cluster.cpp:119: "
        "num_chips > 0\n"
        "info:\n"
        "No chips detected in the cluster\n"
        "AssertionError: PCC 0.5 below target 0.99\n"
        "L1_SMALL bank size is 0 B\n"
    )
    cls = cli._classify_failure("", text_with_many_patterns)
    assert cls == "NO_HARDWARE", (
        "NO_HARDWARE must win over PCC_ONLY, L1_SMALL_ZERO, "
        "TT_FATAL_OPAQUE, etc., to prevent LLM invocation on a "
        "host-environment problem"
    )


def test_detect_no_hardware_failure_reads_per_test_messages() -> None:
    """The pre-flight bail-out helper must look at per-test error
    messages (where TT_FATAL diagnostics actually land in the JUnit
    XML), not just the `summary` / `details` top-level keys.
    Otherwise it misses the very case it's designed to catch."""
    report = {
        "summary": "  - ERROR test_vision_config",
        "details": "(no failure traceback parsed)",
        "per_test": {
            "test_vision_config[device_params0]": {
                "test_id": "test_vision_config[device_params0]",
                "message": (
                    "RuntimeError: TT_FATAL @ tt_cluster.cpp:119: "
                    "num_chips > 0\n"
                    "info:\nNo chips detected in the cluster"
                ),
                "body": "",
            }
        },
    }
    msg = cli._detect_no_hardware_failure(report)
    assert msg is not None
    assert "No chips detected" in msg


def test_detect_no_hardware_failure_returns_none_for_clean_report() -> None:
    """A clean pre-flight report (no failures, or failures that aren't
    hardware-related) must NOT trigger the bail-out. False positives
    here would prevent the LLM loop from ever running."""
    clean = {
        "summary": "(no failures)",
        "details": "",
        "per_test": {},
    }
    assert cli._detect_no_hardware_failure(clean) is None

    pcc_failure = {
        "summary": "AssertionError: PCC 0.5 below target 0.99",
        "details": "PCC 0.5 below target 0.99",
        "per_test": {"x": {"message": "AssertionError: PCC 0.5", "body": ""}},
    }
    assert cli._detect_no_hardware_failure(pcc_failure) is None


def test_format_no_hardware_banner_contains_actionable_steps() -> None:
    """The bail-out banner must contain specific shell commands the
    user can copy-paste (lsmod, ls /dev/tenstorrent*, lspci,
    modprobe). Without these, the user is left guessing what
    'host environment problem' actually means."""
    lines = cli._format_no_hardware_diagnostic_banner("No chips detected")
    text = "\n".join(lines)
    assert "lsmod | grep tenstorrent" in text
    assert "/dev/tenstorrent" in text
    assert "lspci" in text
    assert "modprobe tenstorrent" in text
    assert "No chips detected" in text, (
        "banner must echo the actual error message so the user can " "correlate it with their pytest output"
    )


def test_preflight_no_hardware_bail_block_exists() -> None:
    """The auto-iterate loop must call `_detect_no_hardware_failure`
    BEFORE entering the iter loop, and must `return 2` immediately
    when it fires. Pinned via source inspection so a future
    refactor can't silently remove the early-bail."""
    from pathlib import Path

    src = _planner_source()

    bail_idx = src.find("_no_hw_msg = _detect_no_hardware_failure(")
    assert bail_idx > 0, "pre-flight no-hardware bail block must exist"
    bail_block = src[bail_idx : bail_idx + 2000]
    assert "_format_no_hardware_diagnostic_banner" in bail_block, "bail block must print the actionable banner"
    assert "return 2" in bail_block, (
        "bail block must return exit code 2 (configuration error) "
        "so wrappers can distinguish this from PCC convergence "
        "failures (return 1)"
    )


def test_kill_process_tree_generalized_helper_exists() -> None:
    """The kill logic must be exposed as a general-purpose helper
    that takes a `label` parameter, NOT just as the
    agent-specific `_kill_agent_tree`. This lets other subprocess-
    spawning sites (notably the pytest kill in
    `_run_focused_pytest`) reuse the same robust orphan-killing
    logic. The agent-specific wrapper must still exist for
    backward compatibility."""
    assert hasattr(cli, "_kill_process_tree"), (
        "the generalized helper `_kill_process_tree(proc, label=...)` "
        "must exist so other subprocess sites can reuse it"
    )
    assert hasattr(cli, "_kill_agent_tree"), "the agent-specific wrapper must remain for back-compat"
    import inspect

    sig = inspect.signature(cli._kill_process_tree)
    assert "label" in sig.parameters, "_kill_process_tree must accept `label` for log attribution"


def test_pytest_kill_uses_kill_process_tree() -> None:
    """The pytest hang-timeout kill path in `_run_focused_pytest`
    must use `_kill_process_tree` instead of inline
    killpg(SIGTERM)/wait/killpg(SIGKILL). pytest itself doesn't
    call setsid(), but the TT-Metal C++ runtime (loaded via
    `import ttnn`) may spawn dispatcher / device-poll subprocesses
    that escape the parent's pg. Using the same robust helper
    here defensively prevents the same orphan-class of bug we
    observed in `_invoke_agent`."""
    from pathlib import Path

    src = _planner_source()

    stages_idx = src.find('"  Last reported stage(s) before hang:"')
    assert stages_idx > 0, "pytest hang-timeout block (anchored on 'Last reported " "stage(s) before hang:') must exist"

    kill_block = src[stages_idx : stages_idx + 1200]
    assert "_kill_process_tree(" in kill_block, (
        "pytest hang-timeout block must call `_kill_process_tree` "
        "to apply the same /proc-walking kill logic as the agent "
        "kill site"
    )
    assert 'label="pytest"' in kill_block, (
        "pytest kill must use label='pytest' so its log lines are " "attributable in post-mortem"
    )


def test_invoke_agent_non_stdin_path_uses_devnull() -> None:
    """When prompt_via_stdin is False (cursor branch), the subprocess
    must NOT inherit the parent's stdin. Otherwise a CLI that
    accidentally tries to read stdin (e.g. an interactive auth
    prompt) will block waiting for the parent terminal."""
    from pathlib import Path

    src = _planner_source()
    fn_idx = src.find("def _invoke_agent")
    next_def = src.find("\ndef ", fn_idx + 10)
    fn_body = src[fn_idx : next_def if next_def > 0 else fn_idx + 50000]

    else_idx = fn_body.find("else:\n        proc = subprocess.Popen(")
    assert else_idx > 0, "non-stdin Popen branch must exist"
    else_block = fn_body[else_idx : else_idx + 1500]
    assert "stdin=subprocess.DEVNULL" in else_block, (
        "non-stdin Popen branch must close stdin via DEVNULL — "
        "inheriting the parent's stdin lets a misbehaving CLI "
        "hang waiting for a manual prompt"
    )


def test_hang_branch_comments_do_not_contain_word_continue() -> None:
    """Defensive: `test_hang_synthesis_prints_compute_split` and
    `test_hang_synthesis_records_failure_for_consec_counter` both
    delimit the HANG body by searching for the first occurrence of
    the bare word `continue` after `HANG detected (`. If a NEW
    comment we add inside the HANG branch contains the bare word
    `continue`, those tests will silently truncate the searched
    window and either pass for the wrong reason or fail with a
    confusing message. Pin: no bare word `continue` in any HANG-
    branch comment between the banner and the actual statement."""
    src = _planner_source()
    hang_idx = src.find("HANG detected (")
    assert hang_idx != -1
    cont_idx = src.find("\n            continue\n", hang_idx)
    assert cont_idx != -1, "expected the actual `continue` statement at one indent " "level inside the HANG branch"
    hang_body = src[hang_idx:cont_idx]
    for line_no, line in enumerate(hang_body.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            assert not re.search(r"\bcontinue\b", stripped), (
                f"HANG-branch comment line {line_no} contains the "
                f"bare word `continue`, which breaks the body-search "
                f"in test_hang_synthesis_*. Rephrase. Offending line:"
                f"\n  {line!r}"
            )


def test_format_shape_probe_block_empty_when_no_probes() -> None:
    """When no probes were captured, the block must be the empty
    string so we don't pollute every prompt with an empty
    SHAPE_PROBE OBSERVATIONS header."""
    assert cli._format_shape_probe_block([]) == ""


def test_format_shape_probe_block_groups_by_tag() -> None:
    """When probes ARE present, the block must (a) include the
    SHAPE_PROBE OBSERVATIONS header, (b) group lines by tag (since
    the LLM may have instrumented multiple call sites with distinct
    tags). Without grouping the LLM can't tell which probe came
    from which call site."""
    probes = [
        {"tag": "softmax-probe", "name": "arg0", "payload": "shape=(1,16,64,64)"},
        {"tag": "concat-probe", "name": "x[0]", "payload": "shape=(8,)"},
        {"tag": "softmax-probe", "name": "arg1", "payload": "shape=(64,64)"},
    ]
    block = cli._format_shape_probe_block(probes)
    assert "SHAPE_PROBE OBSERVATIONS" in block
    assert "softmax-probe" in block
    assert "concat-probe" in block
    softmax_idx = block.find("softmax-probe")
    concat_idx = block.find("concat-probe")
    softmax_arg1_idx = block.rfind("arg1")
    assert softmax_arg1_idx > softmax_idx, (
        "probe lines must be grouped by tag (arg1 must appear in the " "softmax-probe section, not interleaved)"
    )


def test_invoke_agent_includes_grep_in_claude_tools() -> None:
    """The LLM's tool list passed to `claude -p` must include Grep,
    otherwise the agentic-affordances directive that tells it to
    'Grep for class <RefModule>' is nonsensical (the tool isn't
    available)."""
    src = _planner_source()
    invoke_idx = src.find("def _invoke_agent")
    assert invoke_idx >= 0

    window = src[invoke_idx : invoke_idx + 20000]
    assert "claude" in window
    claude_branch_idx = window.find('provider == "claude"')
    assert claude_branch_idx >= 0
    branch = window[claude_branch_idx : claude_branch_idx + 2000]
    assert '"Grep"' in branch, (
        "the claude branch of _invoke_agent must pass Grep in --tools "
        "so the LLM can search the repo for exemplars / shared paths"
    )


def test_agentic_affordances_block_helper_exists() -> None:
    """The agentic-affordances block helper must be defined."""
    assert hasattr(cli, "_format_agentic_affordances_block"), "_format_agentic_affordances_block must be defined"
    assert hasattr(cli, "_AGENTIC_INVESTIGATION_CLASSES"), (
        "the set of classes that trigger the affordances block must " "be a named module-level constant for visibility"
    )


def test_agentic_affordances_empty_on_easy_class() -> None:
    """For easy failure classes (NO_OP, EMPTY_AGENT, simple
    API_SIGNATURE) with a low consec counter and no systemic
    pattern, the block must be empty — adding it to every prompt
    would just bloat tokens and waste budget on cases that have
    historically converged in one shot."""
    block = cli._format_agentic_affordances_block("NO_OP", consec_count=0, has_systemic_pattern=False)
    assert block == "", "NO_OP with consec=0 and no systemic pattern must not emit " "the agentic-affordances block"


def test_agentic_affordances_fires_on_hard_class() -> None:
    """For hard classes (TT_FATAL_OPAQUE, PCC_ONLY, HANG, L1_OOM,
    DEVICE_NEEDS_RESET), the block must be non-empty and explicitly
    name the available tools so the LLM knows to use them."""
    for failure_class in [
        "TT_FATAL_OPAQUE",
        "PCC_ONLY",
        "HANG",
        "L1_OOM",
        "DEVICE_NEEDS_RESET",
    ]:
        block = cli._format_agentic_affordances_block(failure_class, consec_count=0, has_systemic_pattern=False)
        assert block, f"agentic-affordances block must be non-empty for {failure_class}"
        assert "AGENTIC AFFORDANCES" in block
        assert (
            "Read" in block and "Grep" in block and "Edit" in block
        ), "affordances block must explicitly name Read/Grep/Edit tools"


def test_agentic_affordances_fires_when_stuck_regardless_of_class() -> None:
    """The block must also fire when the LLM has already failed the
    same way >=2 times, even for an 'easy' class. After repeat
    failures, blind regen has demonstrably not worked — switch to
    investigation mode."""
    block = cli._format_agentic_affordances_block(
        "API_SIGNATURE",
        consec_count=2,
        has_systemic_pattern=False,
    )
    assert block, (
        "agentic-affordances block must fire on consec >= 2 even " "for classes not in _AGENTIC_INVESTIGATION_CLASSES"
    )


def test_agentic_affordances_fires_on_systemic_pattern() -> None:
    """The block must also fire when the cross-component systemic
    detector has flipped, since the fix requires Grep-driven
    discovery of the shared path."""
    block = cli._format_agentic_affordances_block(
        "OTHER",
        consec_count=0,
        has_systemic_pattern=True,
    )
    assert block, "agentic-affordances block must fire when a systemic pattern " "has been detected for THIS iteration"


def test_agentic_block_threaded_into_prompt() -> None:
    """The block must be concatenated into the final prompt string.
    Pins the wiring."""
    src = _planner_source()
    assert "agentic_block" in src, "prompt assembly must thread the agentic-affordances block " "into the final prompt"
    assert "_format_agentic_affordances_block(" in src


def test_pick_target_picks_least_attempted_failed_component() -> None:
    """Bug A: when multiple components in `last_failed_components` are
    under-cap, `_pick_target` must pick the one with the FEWEST total
    attempts so far, NOT the alphabetically first. The sam2-hiera-tiny
    symptom was `_pick_target` always returning `encoder_stack` because
    it sorted alphabetically, while `vision_config` and `self_attention`
    never got an attempt across 6 iters."""
    src = _planner_source()
    pick_idx = src.find("def _pick_target() -> str:")
    assert pick_idx >= 0
    window = src[pick_idx : pick_idx + 4000]
    assert "attempts_per_component.get(c, 0)" in window, "_pick_target must use attempts_per_component as primary key"
    assert (
        "consecutive_same_class_attempts.get(c, 0)" in window
    ), "_pick_target must use consecutive_same_class as secondary key"

    sorted_call = window[window.find("min(") : window.find(",\n            )", window.find("min(")) + 30]
    assert sorted_call.index("attempts_per_component") < sorted_call.index("consecutive_same_class_attempts"), (
        "fair-rotation key must put attempts FIRST so the loop picks "
        "the least-tried component before considering same-class streak"
    )


def test_failure_class_severity_helper_exists() -> None:
    """Bug B foundation: the severity helper must be a real callable
    with the expected ordering. Without this `_record_failure_for_component`
    cannot distinguish "real progress" from "class shift that is actually
    a regression"."""
    assert hasattr(cli, "_failure_class_severity"), "_failure_class_severity must be defined"
    sev = cli._failure_class_severity

    assert sev("PCC_ONLY") < sev("SHAPE"), (
        "PCC_ONLY (structurally correct, numerically off) must rank " "as less severe than SHAPE (structural error)"
    )

    assert sev("SHAPE") < sev("HANG"), (
        "SHAPE must rank as less severe than HANG (which is a total " "test failure / no observable progress)"
    )

    assert sev("PARTIAL_CPU_FALLBACK") < sev("PCC_ONLY"), (
        "PARTIAL_CPU_FALLBACK (PCC passes, one helper still on CPU) " "must be the closest-to-pass severity"
    )

    assert sev("BOGUS_NEVER_USED") >= 100, "unknown classes must rank worse than anything known"


def test_record_failure_only_resets_on_severity_decrease() -> None:
    """Bug B core: when a component shifts from PCC_ONLY (severity 3) to
    HANG (severity 10), that is a REGRESSION, not progress. The counter
    must NOT reset. Previously any class change reset the counter,
    letting components cycle SHAPE -> HANG -> SHAPE -> HANG... and
    bypass the cap entirely."""
    src = _planner_source()
    rec_idx = src.find("def _record_failure_for_component(")
    assert rec_idx >= 0
    window = src[rec_idx : rec_idx + 8000]
    assert "_failure_class_severity(failure_class)" in window, (
        "_record_failure_for_component must consult severity for the " "new class"
    )
    assert "_failure_class_severity(prev_class)" in window, (
        "_record_failure_for_component must compare new severity AGAINST "
        "previous severity so REGRESSIONS don't count as progress"
    )
    assert "<" in window[window.find("_failure_class_severity") : window.find("_failure_class_severity") + 500], (
        "severity comparison must use '<' (new must be STRICTLY lower " "than prev to count as progress)"
    )


def test_record_failure_rejects_other_oscillation_as_progress() -> None:
    """Bug #3 from audit: the OTHER->SHAPE oscillation. `_classify_failure`
    sometimes returns OTHER when the traceback parse comes up short.
    Iter N: SHAPE -> iter N+1: OTHER (parse glitch) -> iter N+2: SHAPE
    must NOT reset the counter (OTHER -> SHAPE = severity 100 -> 7 looks
    like progress but is just parse noise on the same bug)."""
    src = _planner_source()
    rec_idx = src.find("def _record_failure_for_component(")
    assert rec_idx >= 0
    window = src[rec_idx : rec_idx + 8000]

    assert "prev_class not in" in window, (
        "_record_failure_for_component must guard against OTHER/UNKNOWN "
        "as the PREVIOUS class so the oscillation OTHER->SHAPE doesn't "
        "count as progress"
    )
    assert '"OTHER"' in window or "'OTHER'" in window
    assert '"UNKNOWN"' in window or "'UNKNOWN'" in window


def test_hard_total_attempt_cap_tightened() -> None:
    """Bug C: the hard cap formula `max(5, max*3)` was too loose — for
    `--auto-max-attempts-per-component 2` it yielded 6, letting encoder_
    stack accumulate 4 attempts (2x the user's intent). New formula
    `max(3, max*2)` gives 4 for max=2, matching user expectation."""
    src = _planner_source()

    assert "max(3, max_attempts_per_component * 2)" in src, (
        "hard_total_attempt_cap must use the tightened max(3, max*2) "
        "formula so the user's `--auto-max-attempts-per-component` "
        "value is actually respected (within a 2x ceiling)"
    )
    assert "max(5, max_attempts_per_component * 3)" not in src, (
        "the old loose formula max(5, max*3) must be removed entirely " "so no code path falls back to it"
    )


def test_effective_attempt_cap_always_clamps_to_hard_ceiling() -> None:
    """Bug C: previously the non-PCC-stuck branch of `_effective_attempt_
    cap` returned `base` directly (no clamp), so a complexity bonus of 4
    + max_attempts_per_component=2 yielded an effective cap of 6 while
    the hard ceiling was supposed to be 4. The clamp must apply on EVERY
    return path."""
    src = _planner_source()
    cap_idx = src.find("def _effective_attempt_cap(comp: str) -> int:")
    assert cap_idx >= 0
    window = src[cap_idx : cap_idx + 3000]

    clamp_count = window.count("hard_total_attempt_cap")
    assert clamp_count >= 2, (
        "_effective_attempt_cap must clamp to hard_total_attempt_cap "
        "on EVERY return path (the PCC-stuck branch AND the default "
        "branch). Found "
        f"{clamp_count} references — expected >= 2."
    )


def test_invoke_agent_accepts_deliverable_dirs() -> None:
    """Bug D foundation: `_invoke_agent` must accept the deliverable
    tracking parameters. Without them the heartbeat loop has no way to
    detect "claude has burned 80% of budget without writing any file"
    early-kill condition."""
    import inspect

    sig = inspect.signature(cli._invoke_agent)
    assert "deliverable_dirs" in sig.parameters, "_invoke_agent must accept `deliverable_dirs` for Bug D early-kill"
    assert "expected_deliverable_files" in sig.parameters, (
        "_invoke_agent must accept `expected_deliverable_files` for "
        "Bug #5 target-specific detection (wrong-filename writes "
        "MUST NOT satisfy the deadline)"
    )


def test_snapshot_deliverable_state_with_expected_files_handles_missing(tmp_path) -> None:
    """Bug #5 detail: when an expected file does NOT yet exist, the
    snapshot must record a sentinel value (not skip the path). Otherwise
    the agent could write the file and we wouldn't detect the change
    because the path wasn't in the baseline."""
    expected = tmp_path / "_synth_responses" / "target.py"
    state = cli._snapshot_deliverable_state(
        [tmp_path / "_synth_responses"],
        expected_files=[expected],
    )
    assert str(expected) in state, (
        "non-existent expected file must be in the baseline snapshot "
        "(as a sentinel) so its later creation is detected"
    )
    assert state[str(expected)] == (-1.0, -1), (
        "missing-file sentinel must be (-1.0, -1) so a real (mtime, " "size) tuple is always different"
    )


def test_deliverable_changed_detects_creation(tmp_path) -> None:
    """Bug #5: creating a previously-missing expected file must be
    detected as a change. The original bug was that any *.py write in
    `_synth_responses/` satisfied the deadline, so the agent could
    write `decoder_head.py` while the target was `vision_config` and
    look like it delivered."""
    expected = tmp_path / "target.py"
    baseline = {str(expected): (-1.0, -1)}
    expected.write_text("# stub\n")
    current = cli._snapshot_deliverable_state([], expected_files=[expected])
    assert cli._deliverable_changed(baseline, current), "creation of an expected file must register as a change"


def test_deliverable_changed_ignores_wrong_filename(tmp_path) -> None:
    """Bug #5 inverse: writing to a NON-expected file must NOT trigger
    the deliverable flag when `expected_files` is supplied."""
    expected = tmp_path / "target.py"
    baseline = cli._snapshot_deliverable_state([], expected_files=[expected])

    (tmp_path / "wrong.py").write_text("# this is not the target\n")

    current = cli._snapshot_deliverable_state([], expected_files=[expected])
    assert not cli._deliverable_changed(baseline, current), (
        "writing to a non-expected file must NOT satisfy the " "deliverable check (this is the core of Bug #5)"
    )


def test_deliverable_deadline_wired_into_invoke_agent_body() -> None:
    """Bug D: the heartbeat loop must actually USE the deliverable_dirs
    parameter — early-kill at 80% of budget if no file appeared. Pin
    the wiring via source inspection."""
    src = _planner_source()
    invoke_idx = src.find("def _invoke_agent(")
    assert invoke_idx >= 0

    next_def = src.find("\ndef ", invoke_idx + 100)
    body = src[invoke_idx:next_def]
    assert "DELIVERABLE DEADLINE" in body, (
        "_invoke_agent body must include the DELIVERABLE DEADLINE " "kill branch (Bug D early-kill)"
    )
    assert "deliverable_deadline_s" in body, "deadline state variable must exist in the heartbeat loop"
    assert "deliverable_written" in body, "deliverable_written flag must be tracked across heartbeat ticks"


def test_prompt_includes_deliverable_directive_when_budgeted() -> None:
    """Bug D: the LLM-facing prompt must tell Claude about the
    deliverable deadline so it can plan its time. Without this, claude
    treats the timeout as soft and gets killed mid-investigation."""
    src = _planner_source()
    assert "DELIVERABLE DEADLINE:" in src, "prompt builder must emit the DELIVERABLE DEADLINE clause"
    assert "PARTIAL stub" in src, (
        "the directive must tell the LLM to fall back to a partial "
        "stub if it runs out of time (rather than writing nothing)"
    )


def test_prompt_deliverable_deadline_uses_effective_timeout() -> None:
    """Bug #6 from audit: the prompt's stated deadline must use the
    SAME formula the heartbeat loop uses (`_agent_complexity_timeout`
    applied to the user's flat budget). Otherwise complexity-bumped
    iters tell the LLM the wrong number."""
    src = _planner_source()

    bc_idx = src.find("budget_clause = (")
    assert bc_idx >= 0

    window = src[max(0, bc_idx - 2000) : bc_idx + 2000]
    assert "_agent_complexity_timeout(" in window, (
        "budget_min for the prompt must be derived via " "_agent_complexity_timeout so it matches the runtime kill"
    )


def test_cmd_bringup_handoff_to_chat_respects_quiet_flag() -> None:
    """Bug E: when called from the auto-iterate loop with
    `quiet_handoff=True`, `cmd_bringup --handoff-to-chat` must NOT
    print the manual-flow walkthrough (which tells users to paste
    something into Cursor chat — irrelevant in auto mode)."""
    src = _planner_source()
    assert 'getattr(args, "quiet_handoff", False)' in src, (
        "cmd_bringup must read `quiet_handoff` from the argv namespace "
        "and gate the manual-flow walkthrough on it being False"
    )


def test_auto_loop_passes_quiet_handoff_true() -> None:
    """Bug E: the auto-loop's handoff_argv namespace must include
    `quiet_handoff=True` so the walkthrough is suppressed."""
    src = _planner_source()

    h_idx = src.find("handoff_argv = argparse.Namespace(")
    assert h_idx >= 0
    block = src[h_idx : h_idx + 2000]
    assert "quiet_handoff=True" in block, (
        "auto-loop's handoff_argv must set quiet_handoff=True so the " "manual-flow walkthrough is suppressed"
    )


def test_partial_cpu_override_not_clobbered_by_reclassify() -> None:
    """Bug #1: the partial-CPU override sets failure_class =
    PARTIAL_CPU_FALLBACK, but `_classify_failure` was called
    unconditionally three lines later and returned OTHER (no
    PARTIAL_CPU rule). That silently destroyed the override on every
    partial-CPU iter."""
    src = _planner_source()

    assert 'failure_class != "PARTIAL_CPU_FALLBACK"' in src, (
        "auto-loop must guard the re-classify call so the PARTIAL_CPU " "override survives (Bug #1)"
    )


def test_failure_class_initialized_before_partial_cpu_override() -> None:
    """Live-run regression pin (2026-05-23 sam2-hiera-tiny @ iter 1):
    the Bug #1 guard `if failure_class != "PARTIAL_CPU_FALLBACK"`
    reads `failure_class` BEFORE the partial-CPU override block
    runs. On a non-partial-CPU iter (e.g. HANG-targeted
    `encoder_stack`) that override never fires, leaving the
    variable undefined and raising `UnboundLocalError`. Pre-
    initialize the variable to "" before the override branch so
    the guard's comparison is always safe."""
    src = _planner_source()

    block_idx = src.find("_target_partial_cpu_info: Dict[str, List[str]] = (")
    assert block_idx >= 0

    preamble = src[max(0, block_idx - 800) : block_idx]
    assert "failure_class" in preamble, (
        "must initialize failure_class BEFORE the partial-CPU "
        "override block so the Bug #1 guard's comparison doesn't "
        "hit UnboundLocalError on a non-partial-CPU iter"
    )

    assert "failure_class: str =" in preamble or 'failure_class = ""' in preamble or "failure_class = ''" in preamble, (
        "failure_class initialization must use an empty-string "
        "sentinel that is distinguishable from PARTIAL_CPU_FALLBACK; "
        "found preamble = " + preamble[-300:]
    )


def test_regression_guard_records_failure_before_continue() -> None:
    """Bug #2: the strict-native regression guard restores the
    previous version and `continue`s. Without recording the failure,
    `consecutive_same_class_attempts` never advances and the cap is
    bypassed indefinitely."""
    src = _planner_source()

    rg_idx = src.find('"REGRESSION GUARD:')
    if rg_idx < 0:
        rg_idx = src.find("REGRESSION GUARD: agent rewrote")
    assert rg_idx >= 0
    block = src[rg_idx : rg_idx + 6000]
    cont_idx = block.find("\n                continue")
    assert cont_idx >= 0, "could not find regression-guard continue"
    pre_continue = block[:cont_idx]
    assert "_record_failure_for_component(" in pre_continue, (
        "regression-guard path must call _record_failure_for_component "
        "BEFORE its continue so the per-class cap actually advances"
    )
    assert '"WRAPPER"' in pre_continue or "'WRAPPER'" in pre_continue, (
        "regression-guard failure class must be WRAPPER (matches the "
        "actual symptom — agent regressed to a torch wrapper)"
    )


def test_regression_pop_clears_all_per_component_state() -> None:
    """Bug #7: popping `attempts_per_component[r]` without also clearing
    `consecutive_same_class_attempts`, `last_failure_class_per_component`,
    and `last_pcc_per_component` leaves stale state that can both under-
    cap (stale low consec on a restored bad component) and over-cap
    (stale at-cap consec on a restored good component)."""
    src = _planner_source()

    pop_idx = src.find("attempts_per_component.pop(r, None)")
    assert pop_idx >= 0
    block = src[pop_idx : pop_idx + 2000]
    assert (
        "consecutive_same_class_attempts.pop(r, None)" in block
    ), "regression-pop must also clear consecutive_same_class_attempts"
    assert (
        "last_failure_class_per_component.pop(r, None)" in block
    ), "regression-pop must also clear last_failure_class_per_component"
    assert "last_pcc_per_component.pop(r, None)" in block, "regression-pop must also clear last_pcc_per_component"


def test_empty_agent_records_before_cap_check() -> None:
    """Bug #8: the EMPTY_AGENT path checked `_is_at_cap` BEFORE calling
    `_record_failure_for_component`, so the failure that would push the
    consec counter to cap didn't trigger same-iter fallback — wasting
    one extra iter."""
    src = _planner_source()

    ec_idx = src.find("consecutive_empty_agent_calls += 1")
    assert ec_idx >= 0

    block = src[ec_idx : ec_idx + 3000]
    rec_pos = block.find('_record_failure_for_component(\n                    iter_target_component, "EMPTY_AGENT"')
    cap_pos = block.find("_is_at_cap(iter_target_component)")
    assert rec_pos >= 0 and cap_pos >= 0, "could not locate EMPTY_AGENT record / cap-check lines"
    assert rec_pos < cap_pos, (
        "_record_failure_for_component must run BEFORE _is_at_cap in "
        "the EMPTY_AGENT branch so the cap-out can fire same-iter"
    )


def test_preflight_hang_seeds_record_failure() -> None:
    """Bug #9: when pre-flight pytest hangs, the inferred target gets
    seeded into `last_failed_components` and `verified_fail`, but
    previously NOT into the per-component progress trackers. First iter
    would then record the actual failure with `prev_class=""` (first
    failure observed = progress!), resetting the counter to 1 instead
    of accumulating the HANG that already cost the pre-flight budget."""
    src = _planner_source()

    pfh_idx = src.find("pre-flight HANG: targeting")
    assert pfh_idx >= 0

    block = src[max(0, pfh_idx - 2000) : pfh_idx + 500]
    assert '_record_failure_for_component(_pf_target, "HANG", None)' in block, (
        "pre-flight HANG inference must seed _record_failure_for_component "
        "with HANG so the consec counter doesn't reset on iter 1"
    )


def test_pick_target_all_at_cap_fallback_not_alphabetic() -> None:
    """Bug #10: the all-at-cap fallback was `return sorted(
    candidate_pool)[0]` — duplicating the alphabetical fixation
    pattern Bug A fixed elsewhere. Should use the same fair-rotation
    key (least attempts, then consec, then alphabetic)."""
    src = _planner_source()
    pick_idx = src.find("def _pick_target() -> str:")
    assert pick_idx >= 0
    window = src[pick_idx : pick_idx + 4000]

    assert "return sorted(candidate_pool)[0]" not in window, (
        "_pick_target all-at-cap fallback must not use a bare "
        "`return sorted(candidate_pool)[0]` — that's the fixation "
        "pattern Bug A fixed"
    )


def test_auto_loop_captures_apply_all_responses_return_code() -> None:
    """Bug #11: `cmd_bringup(apply_argv)` can return 3 on syntax-error
    / empty-apply. Previously the loop ignored this and ran pytest on
    whatever was on disk, wasting another 30-60s of TT hardware time."""
    src = _planner_source()

    assert "apply_rc = cmd_bringup(apply_argv)" in src, (
        "auto-loop must capture cmd_bringup's return code from " "apply_all_responses (Bug #11)"
    )

    arc_idx = src.find("apply_rc = cmd_bringup(apply_argv)")
    assert arc_idx >= 0
    block = src[arc_idx : arc_idx + 2000]
    assert "apply_rc not in (0, None)" in block, (
        "auto-loop must check apply_rc and route non-zero through the " "failure-recording path"
    )
    assert "EMPTY_AGENT" in block, (
        "apply-fail handling must record EMPTY_AGENT against the " "targets so cap accounting catches it"
    )


def test_regression_sweep_hang_records_hang() -> None:
    """Bug #12: when the regression sweep hangs (rc=124), previously
    the loop silently continued with no record of failure and no
    rollback — leaving any partial regression state on disk and
    proceeding as if nothing happened."""
    src = _planner_source()

    rs_idx = src.find("regression sweep hung")
    assert rs_idx >= 0
    block = src[rs_idx : rs_idx + 1500]
    assert "_record_failure_for_component" in block, (
        "regression-sweep hang branch must record HANG against " "previously-graduated components"
    )
    assert '"HANG"' in block or "'HANG'" in block


def test_record_failure_no_cap_advance_before_llm_invocation() -> None:
    """Bug X: the validation sweep records pytest failures for every
    component in `last_failed_components`. Before this fix, the
    consecutive-same-class counter was incremented even when
    `attempts_per_component[c] == 0` (LLM never invoked), so a
    Phase-1 autofill failure capped out the component without
    Claude ever getting to try. This test pins the guard that makes
    the cap-counter strictly an LLM-iteration counter."""
    src = _planner_source()

    def_idx = src.find("def _record_failure_for_component(")
    assert def_idx >= 0

    drn_idx = src.find('if failure_class == "DEVICE_NEEDS_RESET":', def_idx)
    assert drn_idx >= 0

    body = src[def_idx : def_idx + 8000]
    assert "attempts_per_component.get(comp, 0) == 0" in body, (
        "_record_failure_for_component must guard against advancing "
        "the cap counter when the LLM has not yet been invoked on "
        "this component (Bug X)"
    )

    guard_idx = body.find("attempts_per_component.get(comp, 0) == 0")
    inc_idx = body.find("consecutive_same_class_attempts[comp] =")
    assert inc_idx > 0, "test window must capture the consec increment line"
    assert guard_idx < inc_idx, (
        "the Bug-X guard must execute BEFORE the consec counter " "increment so a return short-circuits cap advancement"
    )

    guard_block = body[guard_idx : guard_idx + 600]
    assert "return False" in guard_block, (
        "the Bug-X guard branch must return False so the caller's " "PROGRESS print and counter logic both skip"
    )


def test_record_failure_pre_llm_still_records_class_and_pcc() -> None:
    """Bug X corollary: even though the cap counter must NOT advance
    pre-LLM, we still want the failure class and PCC value snapshotted
    so the NEXT real LLM iter can detect progress relative to the
    pre-LLM baseline. Without this, the very first LLM iter has no
    `prev_class` to compare against and the progress detector defaults
    to "first failure" (counter=1) — which is fine, but loses the
    diagnostic continuity."""
    src = _planner_source()
    def_idx = src.find("def _record_failure_for_component(")
    body = src[def_idx : def_idx + 8000]
    guard_idx = body.find("attempts_per_component.get(comp, 0) == 0")
    assert guard_idx >= 0
    guard_block = body[guard_idx : guard_idx + 600]
    assert "last_failure_class_per_component[comp] = failure_class" in guard_block, (
        "pre-LLM guard must still snapshot failure_class for later " "progress comparison (Bug X corollary)"
    )
    assert "last_pcc_per_component[comp] = pcc_value" in guard_block, (
        "pre-LLM guard must still snapshot pcc_value for later " "progress comparison (Bug X corollary)"
    )


def test_autofill_template_coerce_to_torch_promotes_to_fp32() -> None:
    """Bug Y: the autofill template's `_coerce_to_torch` returned the
    ttnn->torch converted tensor directly, which inherits the device
    dtype (typically bf16 on Blackhole). HF reference modules have
    fp32 weights, so calling them with bf16 inputs raises:
        RuntimeError: mat1 and mat2 must have the same dtype,
        but got BFloat16 and Float
    The validation sweep then classified this as WRAPPER and the
    component was permanently skipped (compounded by Bug X). Fix:
    promote float tensors to fp32 inside `_coerce_to_torch`."""
    from scripts.tt_hw_planner import bringup_loop

    template = bringup_loop._AUTOFILL_TORCH_TEMPLATE

    assert "is_floating_point()" in template, (
        "autofill template must promote float tensors to fp32 in " "_coerce_to_torch (Bug Y)"
    )
    assert "torch.float32" in template or "_torch.float32" in template, (
        "autofill template must reference torch.float32 explicitly " "as the promotion target (Bug Y)"
    )

    assert bringup_loop._AUTOFILL_DTYPE_FIX_MARKER in template, (
        "autofill template must embed the dtype-fix marker so the "
        "preserve branch can detect pre-fix stubs and force-regen"
    )


def test_autofill_stub_missing_dtype_fix_detector() -> None:
    """Bug Y part 2: pre-fix autofill stubs on disk must be detected
    so the preserve branch regenerates them instead of leaving the
    broken bf16-handling `_coerce_to_torch` in place."""
    import tempfile
    from pathlib import Path

    from scripts.tt_hw_planner import bringup_loop

    old_stub = """# autogen
\"\"\"HF reference fallback for `foo`.

Generated by `tt_hw_planner bringup --autofill`. ...
\"\"\"
import transformers

def _get_torch_submodule():
    pass

def _coerce_to_torch(x):
    import ttnn as _ttnn
    if isinstance(x, _ttnn.Tensor):
        return _ttnn.to_torch(x)
    return x
"""
    new_stub = old_stub.replace(
        "return _ttnn.to_torch(x)",
        "# " + bringup_loop._AUTOFILL_DTYPE_FIX_MARKER + "\n        return _ttnn.to_torch(x).to(_torch.float32)",
    )
    user_native = """import ttnn
class FooNative:
    def __call__(self, x):
        return ttnn.gelu(x)
"""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        old_path = tmp_path / "old.py"
        new_path = tmp_path / "new.py"
        native_path = tmp_path / "native.py"
        old_path.write_text(old_stub)
        new_path.write_text(new_stub)
        native_path.write_text(user_native)

        assert bringup_loop._autofill_stub_missing_dtype_fix(old_path), (
            "old-template autofill stub must be detected as needing " "the dtype-fix regen"
        )

        assert not bringup_loop._autofill_stub_missing_dtype_fix(
            new_path
        ), "new-template autofill stub must NOT be flagged for regen"

        assert not bringup_loop._autofill_stub_missing_dtype_fix(native_path), (
            "user-edited native stubs (no autofill markers) must "
            "NEVER be flagged for regen — that would destroy the "
            "LLM's work"
        )


def test_autofill_regen_branch_wired_into_preserve_chain() -> None:
    """Bug Y part 2: the regen-on-missing-dtype-fix branch must be
    correctly threaded between the op-synth-regen branch and the
    preserve-already-autofilled branch in `autofill_stubs`. Verify
    the elif chain ordering (any other order either misses regen
    cases or clobbers user-native code)."""
    src = Path(__import__("scripts.tt_hw_planner.bringup_loop", fromlist=["bringup_loop"]).__file__).read_text()

    fn_idx = src.find("def autofill_stubs(")
    assert fn_idx >= 0
    body = src[fn_idx : fn_idx + 60000]

    assert '"regen:autofill-missing-dtype-fix"' in body, "preserve chain must register the new regen action label"

    op_synth_idx = body.find('"regen-as-op-synth:torch-wrapper-upgrade"')
    dtype_fix_idx = body.find('"regen:autofill-missing-dtype-fix"')
    preserved_idx = body.find('"preserved:already-autofilled"')
    assert op_synth_idx >= 0 and dtype_fix_idx >= 0 and preserved_idx >= 0
    assert op_synth_idx < dtype_fix_idx < preserved_idx, (
        "preserve elif chain must be: op-synth-regen, dtype-fix-regen, "
        "preserve-already-autofilled (any other order misses or "
        "clobbers cases)"
    )


def test_auto_emit_runnable_demo_uses_captured_inputs() -> None:
    """Tool contract (2026-05-23 live-run sam2-hiera-tiny): the scaffolder's
    `demo.py` is a stale copy from the sibling model and never matches the
    LLM-written `__call__` signatures (e.g. sibling encoder returned a 3-
    tuple but the new model's port returns a single tensor). A green
    convergence banner used to be undermined by `pytest demo.py::test_demo`
    crashing with `ValueError: not enough values to unpack`. Fix: emit a
    deterministic captured-inputs-driven `demo.py` after convergence and
    verify it via pytest. This test pins that the emitter exists, is
    exported, and reads from `_captured/<comp>/`."""
    from scripts.tt_hw_planner import bringup_loop

    assert hasattr(bringup_loop, "emit_runnable_demo"), (
        "bringup_loop must export `emit_runnable_demo` so the auto-loop "
        "convergence hook can regenerate `demo.py` post-graduation"
    )
    src = Path(bringup_loop.__file__).read_text()

    # The new template loads captured inputs at runtime from
    # `_captured/<safe>/` using the wired-component table; the literal
    # `_captured` substring must appear in the template source.
    assert "_captured" in src, (
        "demo template must load captured inputs from `_captured/<safe>/` "
        "so it matches exactly what the PCC test validated"
    )

    assert "mod.build(device, torch_module)" in src, (
        "demo template must use `build(device, torch_module)` to "
        "construct the TTNN port — that's the universal contract"
    )

    assert "_normalize_out" in src, (
        "demo template must normalize the output via _normalize_out "
        "(handles dict / tuple / ttnn.Tensor return types)"
    )


def test_auto_emit_demo_wires_all_graduated_components() -> None:
    """End-to-end demo must wire EVERY graduated component into the HF
    reference model, not pick a single primary. Components in the
    KERNEL_MISSING bucket stay on CPU as HF reference automatically.
    The maximal-antichain selector ensures parents subsume children when
    both graduate (the parent's TT forward already contains the child)."""
    from scripts.tt_hw_planner import bringup_loop, demo_wiring

    # The single-primary picker must be GONE — replaced by antichain
    # selection that wires ALL graduated components.
    bl_src = Path(bringup_loop.__file__).read_text()
    assert "def _pick_primary_component(" not in bl_src, (
        "single-primary picker must be removed in favor of multi-component "
        "wiring via demo_wiring.select_maximal_antichain"
    )

    # The demo template must iterate over WIRED_COMPONENTS, not load a
    # single PRIMARY_COMPONENT.
    assert "WIRED_COMPONENTS" in bl_src, (
        "demo template must declare WIRED_COMPONENTS table — the list of "
        "(submodule_path, stub_import_path, name) tuples for every wired "
        "graduated component"
    )

    # The demo template must replace each graduated submodule on the HF
    # model so the forward path exercises ON-DEVICE.
    assert "_set_submodule(hf_model" in bl_src, (
        "demo template must install each TT port onto the HF model via "
        "_set_submodule — that's how mixed CPU/device execution composes"
    )

    # The wiring module must exist and expose the public API the
    # bringup_loop uses.
    for fn in (
        "collect_graduated_components",
        "select_maximal_antichain",
        "build_wiring_specs",
        "format_wiring_literal",
    ):
        assert hasattr(demo_wiring, fn), f"demo_wiring must export `{fn}`"


def test_auto_emit_op_count_reads_counts_dict() -> None:
    """Bug pin (2026-05-23): the first version of `_component_op_count`
    read a non-existent `op_count` field and always returned 0, causing
    the primary picker to fall back to pure alphabetic and pick whatever
    component sorted first lexically (e.g. `decoder_head`) regardless of
    actual size. The correct schema (per op_emitter.py) is
    `counts: {op-REUSE, op-ADAPT, op-NEW}` and `palette: [ops]`. The
    reader must sum the counts dict, falling back to len(palette)."""
    from scripts.tt_hw_planner import bringup_loop

    src = Path(bringup_loop.__file__).read_text()
    rdr_idx = src.find("def _component_op_count(")
    assert rdr_idx >= 0
    body = src[rdr_idx : rdr_idx + 1500]
    assert '"counts"' in body or "'counts'" in body, (
        "op-count reader must consult the `counts` dict from the "
        "opplan manifest, not a non-existent `op_count` scalar"
    )
    assert '"palette"' in body or "'palette'" in body, (
        "op-count reader must fall back to len(palette) when `counts` " "is missing"
    )


def test_convergence_path_emits_and_verifies_demo() -> None:
    """A green AUTO-ITERATE converge banner must imply
    `pytest demo.py::test_demo` is also green.

    2026-05-31: this hook has moved out of the converged branch in
    auto_iterate.py and into cmd_up's post-Phase-2 gate, so the demo
    only emits when every HOT component is graduated (better signal).
    Test pins both: (a) the auto-loop convergence branch carries the
    move-out NOTE so the architectural decision is documented, and
    (b) cmd_up actually calls `_emit_and_verify_runnable_demo(MODEL)`
    after the iterate-loop graduates."""
    src = _planner_source()

    conv_idx = src.find('banner(f"AUTO-ITERATE converged after')
    assert conv_idx >= 0, "auto-loop must have a converged banner"
    conv_block = src[conv_idx : conv_idx + 3000]
    assert "end-to-end demo emission has moved OUT" in conv_block, (
        "converged branch must carry the NOTE documenting that emit " "moved to cmd_up's post-Phase-2 gate"
    )

    # The actual emit+verify hook now lives in cmd_up.
    cmd_up_idx = src.find("def cmd_up(")
    assert cmd_up_idx >= 0
    cmd_up_block = src[cmd_up_idx : cmd_up_idx + 60000]
    assert "_emit_and_verify_runnable_demo(MODEL" in cmd_up_block, (
        "cmd_up must call _emit_and_verify_runnable_demo(MODEL) after "
        "the iterate-loop graduates so the green banner implies a "
        "green demo.py"
    )


def test_emit_and_verify_helper_uses_subprocess_pytest() -> None:
    """The convergence-path verifier must actually run pytest as a
    subprocess (not just emit and trust). Without verification we'd be
    back to "convergence banner lies about the demo working". Pins:
      - subprocess.run is invoked.
      - The args contain `pytest` and `::test_demo`.
      - There's a non-zero timeout so a hung pytest doesn't wedge the
        whole convergence path."""
    src = _planner_source()
    helper_idx = src.find("def _emit_and_verify_runnable_demo(")
    assert helper_idx >= 0, "helper must exist"
    body = src[helper_idx : helper_idx + 4000]
    assert "subprocess.run" in body, (
        "helper must actually run pytest as a subprocess to verify " "the auto-emitted demo passes"
    )
    assert "::test_demo" in body, (
        "helper must target the `test_demo` function specifically " "(not just the whole demo.py module)"
    )
    assert "timeout=" in body, (
        "helper must specify a timeout so a hung pytest can't wedge " "the convergence path forever"
    )


def test_up_and_promote_have_regen_demo_only_flag() -> None:
    """Both `up` and `promote` must accept `--regen-demo-only` for the
    case where the user has an already-converged model and wants to
    regenerate + verify the demo without re-iterating Claude (e.g.
    after a git pull clobbered demo.py)."""
    src = _planner_source()

    occurrences = src.count('"--regen-demo-only"')
    assert occurrences >= 2, (
        f"--regen-demo-only must be registered on both `up` and " f"`promote` parsers; found {occurrences} occurrences"
    )

    for fn in ("def cmd_up(", "def cmd_promote("):
        fn_idx = src.find(fn)
        assert fn_idx >= 0, f"{fn} must exist"

        block = src[fn_idx : fn_idx + 3000]
        assert "regen_demo_only" in block and "_emit_and_verify_runnable_demo" in block, (
            f"{fn} must short-circuit on `args.regen_demo_only` and "
            f"call `_emit_and_verify_runnable_demo` directly, skipping "
            f"the full pipeline"
        )


def test_op_emitter_coerce_to_torch_also_promotes_to_fp32() -> None:
    """Bug Y part 3: the op-synth emitter (op_emitter.py) generates
    a separate `_coerce_to_torch` body via string concatenation. It
    must carry the same fp32 promotion as the autofill template,
    otherwise op-synth stubs hit the same bf16/fp32 mismatch."""
    from scripts.tt_hw_planner import op_emitter as _op_emitter

    src = Path(_op_emitter.__file__).read_text()

    coerce_idx = src.find("'def _coerce_to_torch(x):")
    assert coerce_idx >= 0, "op_emitter must emit a _coerce_to_torch fn"
    block = src[coerce_idx : coerce_idx + 3000]

    assert "is_floating_point()" in block, (
        "op-synth _coerce_to_torch must promote float tensors to " "fp32 (Bug Y part 3)"
    )
    assert "float32" in block, "op-synth _coerce_to_torch must reference float32 as the " "promotion target"


def test_memory_fit_gate_helper_exists_and_classifies_three_outcomes() -> None:
    """2026-05-23 user-flagged bug: `up --auto` would burn LLM tokens
    on (model, box, mesh) triples that the planner verdict already
    rejected (e.g. 70B model on a 1x1 mesh). The
    `_check_memory_fit_before_llm` helper must exist and must
    classify outcomes into exactly three buckets:
      * "fit"     — requested target has headroom; proceed.
      * "no-fit"  — abort BEFORE scaffold/autofill/LLM iteration.
      * "unknown" — no LLM-style memory model (vision / multi-modal);
                    let the compat gate be the source of truth."""
    src = _planner_source()
    assert "def _check_memory_fit_before_llm(" in src, (
        "_check_memory_fit_before_llm helper must be defined in cli.py " "to pre-gate LLM iteration on planner verdicts"
    )
    fn_idx = src.find("def _check_memory_fit_before_llm(")

    body = src[fn_idx : fn_idx + 6000]
    for status in ('"fit"', '"no-fit"', '"unknown"'):
        assert status in body, f"_check_memory_fit_before_llm must produce {status} as one " f"of its return statuses"

    assert "memory_model is None" in body, (
        "_check_memory_fit_before_llm must skip the gate when probe "
        "has no LLM memory model (vision / multi-modal path)"
    )

    assert "evaluate_all(" in body, (
        "_check_memory_fit_before_llm must reuse evaluate_all from " "verdict.py — duplicating the math would drift"
    )


def test_memory_fit_gate_enforcer_short_circuits_on_no_fit() -> None:
    """The `_enforce_memory_fit_or_abort` helper must:
      1. Return non-None (an exit code) when the underlying helper
         returns 'no-fit', so callers can `return` immediately to
         abort the pipeline.
      2. Return None for the 'fit' and 'unknown' cases (proceed).
    The gate has no opt-out flag: memory budgets are hardware-
    determined, and an LLM rewrite cannot shrink a too-large model."""
    src = _planner_source()
    assert "def _enforce_memory_fit_or_abort(" in src, "_enforce_memory_fit_or_abort wrapper must be defined in cli.py"
    fn_idx = src.find("def _enforce_memory_fit_or_abort(")
    body = src[fn_idx : fn_idx + 5000]

    assert "_check_memory_fit_before_llm(" in body, (
        "_enforce_memory_fit_or_abort must call the underlying " "_check_memory_fit_before_llm to get the verdict"
    )

    assert "return 2" in body, (
        "_enforce_memory_fit_or_abort must return exit code 2 on " "no-fit so callers can abort before LLM iteration"
    )


def test_cmd_up_and_promote_call_memory_fit_gate_before_llm() -> None:
    """Wiring test: cmd_up and cmd_promote must both invoke the
    memory-fit gate before any path that could lead to LLM iteration
    (scaffold for cmd_up; the auto-iterate loop for cmd_promote).
    Otherwise the helper exists but is dead code."""
    src = _planner_source()
    for fn_name in ("def cmd_up(", "def cmd_promote("):
        fn_idx = src.find(fn_name)
        assert fn_idx >= 0, f"{fn_name} must exist in cli.py"

        block = src[fn_idx : fn_idx + 60000]
        assert "_enforce_memory_fit_or_abort(" in block, (
            f"{fn_name.rstrip('(')} must invoke "
            f"_enforce_memory_fit_or_abort before LLM iteration so "
            f"unsupported (model, box, mesh) triples abort early "
            f"without burning LLM tokens"
        )


def test_rope_hf_check_warns_when_runtime_falls_back_to_mllama() -> None:
    """2026-05-23 compat bug fix: `check_rope_hf` used to BLOCKER on
    `head_dim % 64 != 0` even though the runtime's default RoPE kernel
    (`rotary_embedding_llama`) only needs `head_dim % 32 == 0`. That
    produced false-positives like "Phi-3.5-mini cannot run on QB2"
    for a model that runs out of the box. Pinned behaviors:

      - head_dim % 64 == 0 -> [ok], BLOCKER severity (no fallback needed)
      - head_dim % 32 == 0 (but not 64) -> [warn], NOT a BLOCKER
      - head_dim % 32 != 0 -> [FAIL] BLOCKER (neither kernel accepts)
    """
    from scripts.tt_hw_planner.kernel_constraints import (
        Severity,
        check_rope_hf,
    )

    findings = check_rope_hf({"head_dim": 128}, 1)
    assert findings and findings[0].passes
    assert findings[0].severity == Severity.BLOCKER, "the constraint itself is BLOCKER-class; it just passes here"

    findings = check_rope_hf({"head_dim": 96}, 1)
    assert findings and not findings[0].passes
    assert findings[0].severity == Severity.WARN, (
        "head_dim=96 must downgrade to WARN since runtime falls back "
        "to rotary_embedding_llama (mllama RoPE, % 32 == 0)"
    )

    findings = check_rope_hf({"head_dim": 80}, 1)
    assert findings and not findings[0].passes
    assert findings[0].severity == Severity.BLOCKER, (
        "head_dim=80 is NOT divisible by 32; neither RoPE kernel " "accepts it; must remain BLOCKER"
    )


def test_rope_scaling_dynamic_is_blocked_not_silently_allowed() -> None:
    """2026-05-23 compat bug fix: `check_rope_scaling` previously listed
    `dynamic` in the supported-set, but `tt_transformers/tt/rope.py:314`
    only branches on linear/llama3/yarn/longrope and raises ValueError
    on everything else (including `dynamic`). Pin: `dynamic` is rejected."""
    from scripts.tt_hw_planner.kernel_constraints import (
        Severity,
        check_rope_scaling,
    )

    findings = check_rope_scaling({"rope_scaling": {"type": "dynamic", "factor": 8.0}}, 1)
    assert findings and not findings[0].passes, "rope_type=dynamic must FAIL compat; runtime raises ValueError"
    assert findings[0].severity == Severity.BLOCKER, "unsupported rope_type must be a hard BLOCKER (runtime crash)"


def test_rope_scaling_warns_on_rope_parameters_migration() -> None:
    """2026-05-23 compat bug fix: tt_transformers/tt/model_config.py:2736
    reads `cfg.rope_scaling` only, NOT `cfg.rope_parameters` (the newer
    HF field). If a model migrates to `rope_parameters` (HF's
    deprecation warning recommends this), the runtime will silently
    treat it as no-scaling. Pin: WARN on this case so users notice."""
    from scripts.tt_hw_planner.kernel_constraints import (
        Severity,
        check_rope_scaling,
    )

    findings = check_rope_scaling({"rope_parameters": {"rope_type": "llama3", "factor": 8.0}}, 1)
    assert findings, "rope_parameters without rope_scaling must produce a finding " "(was silently ignored before)"
    assert findings[0].severity == Severity.WARN, (
        "rope_parameters migration is a soft incompat (runtime is " "wrong, not crashing); WARN is the right severity"
    )
    assert "rope_parameters" in findings[0].constraint.lower(), (
        "constraint message must call out the field name so users " "know what to fix"
    )


def test_rope_scaling_warns_when_type_field_missing() -> None:
    """2026-05-23 compat bug fix: if `rope_scaling` is set as a dict
    but has no `type` / `rope_type` field, the previous check
    silently returned [] -- hiding a misconfigured scaling. Pin: WARN."""
    from scripts.tt_hw_planner.kernel_constraints import (
        Severity,
        check_rope_scaling,
    )

    findings = check_rope_scaling({"rope_scaling": {"factor": 8.0}}, 1)
    assert findings, "rope_scaling dict without `type` must produce a finding " "(was silently ignored before)"
    assert findings[0].severity == Severity.WARN


def test_rope_scaling_passes_cleanly_for_supported_types() -> None:
    """Sanity: all four supported rope_types must pass cleanly."""
    from scripts.tt_hw_planner.kernel_constraints import (
        Severity,
        check_rope_scaling,
    )

    for rtype in ("linear", "llama3", "yarn", "longrope"):
        findings = check_rope_scaling({"rope_scaling": {"type": rtype}}, 1)
        assert findings and findings[0].passes, f"rope_type={rtype!r} must pass (it's in the supported set)"

    findings = check_rope_scaling({}, 1)
    assert findings == [], "no rope_scaling -> no findings"


def test_env_compat_passes_when_patches_applied() -> None:
    """2026-05-23 environment pre-check: when the three transformers-5.x
    patches are in place (try/except AutoModelForVision2Seq import,
    rope_parameters fallback, Encoding-to-list normalization), the
    `_check_demo_environment_compat` helper must return ok=True
    against the current repo state. Pin this so we notice if a
    future regression removes one of the patches."""
    from scripts.tt_hw_planner.cli import _check_demo_environment_compat

    ok, problems = _check_demo_environment_compat()
    if not ok:
        msg = "\n".join(problems)
        raise AssertionError(f"env-check unexpectedly reported problems with the " f"patches in place:\n{msg}")


def test_env_compat_detects_unpatched_transformers_5x() -> None:
    """The env-check must FIRE if the transformers-5.x patches are
    removed. We can't actually remove the patches in CI without
    breaking everyone, so this test mocks the file reads to simulate
    a fresh checkout against transformers 5.x."""
    import unittest.mock
    import sys
    from scripts.tt_hw_planner.cli import _check_demo_environment_compat

    fake_files: Dict[str, str] = {
        "models/common/llama_models.py": (
            "from transformers import AutoModelForVision2Seq, " "AutoProcessor, pipeline\n"
        ),
        "models/tt_transformers/tt/model_config.py": (
            'self.rope_theta = text_config.get("rope_theta")\n' 'self.rope_scaling = text_config.get("rope_scaling")\n'
        ),
        "models/tt_transformers/tt/common.py": (
            "def encode_prompt_hf(tokenizer, prompt_text):\n"
            "    result = tokenizer.apply_chat_template(chat, "
            "add_generation_prompt=True, tokenize=True)\n"
            "    return result\n"
        ),
    }
    real_read_text = Path.read_text
    real_is_file = Path.is_file

    def fake_read_text(self, *args, **kwargs):
        for suffix, text in fake_files.items():
            if str(self).endswith(suffix):
                return text
        return real_read_text(self, *args, **kwargs)

    def fake_is_file(self):
        for suffix in fake_files:
            if str(self).endswith(suffix):
                return True
        return real_is_file(self)

    fake_tf = unittest.mock.MagicMock()
    fake_tf.__version__ = "5.8.1"
    with unittest.mock.patch.object(Path, "read_text", fake_read_text), unittest.mock.patch.object(
        Path, "is_file", fake_is_file
    ), unittest.mock.patch.dict(sys.modules, {"transformers": fake_tf}):
        ok, problems = _check_demo_environment_compat()
    assert not ok, "env-check must fail on unpatched transformers 5.x repo"
    joined = "\n".join(problems).lower()
    assert "automodelforvision2seq" in joined, "must call out the broken vision import"
    assert "rope" in joined, "must call out the rope_parameters migration"
    assert "tokenizers.encoding" in joined or "encoding" in joined, "must call out the Encoding-to-list issue"


def test_cmd_up_runs_env_check_for_supported_models() -> None:
    """Wiring test: cmd_up's supported-model branch must call
    `_check_demo_environment_compat` BEFORE invoking cmd_prepare.
    Otherwise environment incompatibilities only surface as runtime
    crashes inside the demo (the exact bad UX this pre-check is
    designed to prevent)."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0
    block = src[fn_idx : fn_idx + 40000]
    env_check_call = block.find("_check_demo_environment_compat(")
    cmd_prepare_call = block.find("cmd_prepare(prepare_argv)")
    assert env_check_call >= 0, "cmd_up must call _check_demo_environment_compat in the " "supported-model branch"
    assert cmd_prepare_call >= 0
    assert env_check_call < cmd_prepare_call, (
        "env-check must run BEFORE cmd_prepare, otherwise the demo " "still crashes at runtime with a cryptic error"
    )


def test_cmd_up_skips_meta_plan_for_already_supported_models() -> None:
    """2026-05-23 UX fix: meta-plan must not run for already-supported
    models. Running it wastes ~5 min loading the full HF model plus
    a Claude LLM call asking how to bring up a model that doesn't
    need any bring-up loop. Pinned behaviors:

      - `_already_supported` check is computed BEFORE the meta-plan
      - the meta-plan call is guarded by `not _already_supported`
    """
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0
    block = src[fn_idx : fn_idx + 40000]
    early_check = block.find("_already_supported = (")
    meta_plan_call = block.find("_run_advisory_meta_plan(")
    assert early_check >= 0 and meta_plan_call >= 0, "both the supported-check and meta-plan call must be present"
    assert early_check < meta_plan_call, (
        "_already_supported must be computed BEFORE the meta-plan, "
        "otherwise supported models still pay the meta-plan cost"
    )

    guard_window = block[max(0, meta_plan_call - 300) : meta_plan_call]
    assert "_already_supported" in guard_window, (
        "the meta-plan call must be guarded by `not _already_supported` " "so supported models skip it entirely"
    )


def test_cmd_up_prepare_argv_uses_sane_defaults_not_none() -> None:
    """2026-05-23 bug fix: the supported-model branch in cmd_up was
    passing `max_seq_len=None` / `max_generated_tokens=None` to
    cmd_prepare. cmd_prepare rendered the values into the pytest
    invocation as the literal strings `--max_seq_len None`, which
    pytest then rejected with `invalid int value: 'None'`. Pin:
    use the same defaults the `prepare` parser registers (1024
    and 200)."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0

    block_full = src[fn_idx:]
    cursor = 0
    supported_branch: str = ""
    while True:
        i = block_full.find("if _already_supported:", cursor)
        if i < 0:
            break
        window = block_full[i : i + 14000]
        if "cmd_prepare(prepare_argv)" in window or "cmd_prepare(\n" in window:
            supported_branch = window
            break
        cursor = i + 1
    assert supported_branch, "could not locate the `if _already_supported:` branch that " "leads to cmd_prepare"
    assert "max_seq_len=1024" in supported_branch, (
        "supported-model branch must pass max_seq_len=1024 (the " "registered default for the `prepare` parser)"
    )
    assert "max_generated_tokens=200" in supported_branch, (
        "supported-model branch must pass max_generated_tokens=200 " "(the registered default for the `prepare` parser)"
    )
    assert "max_seq_len=None" not in supported_branch, (
        "must NOT pass max_seq_len=None -- that gets rendered as the " "literal string 'None' into the pytest cmdline"
    )


def test_cmd_up_auto_routes_generic_llm_backends_to_prepare_execute() -> None:
    """2026-05-23 bug fix: when a model has compat verdict READY or
    FEASIBLE WITH WORK with no MISSING blocks AND the picked backend
    is `routing_mode="generic"` (i.e. tt_transformers/simple_text_demo),
    cmd_up must skip the scaffold step and route directly to
    prepare+execute. Without this, the scaffold step aborts with
    "no closest already-ported sibling found" - a contradiction
    because Step 1 already announced "Backend match: GENERIC".
    This hit Gemma-2-27b after ~70 minutes of weight download."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0
    block = src[fn_idx : fn_idx + 30000]

    assert "_route_via_generic_llm" in block, (
        "cmd_up must define a `_route_via_generic_llm` flag to " "handle generic LLM backends without a tuning row"
    )

    assert "routing_mode" in block and '"generic"' in block, "must inspect the picked backend's routing_mode"
    assert "FEASIBLE WITH WORK" in block, "must allow FEASIBLE WITH WORK compat verdict (not just READY)"

    route_idx = block.find("_route_via_generic_llm")
    after_route = block[route_idx : route_idx + 8000]
    assert "_already_supported = True" in after_route, (
        "the generic-LLM branch must flip _already_supported so the " "existing supported-model routing applies"
    )


def test_audit_bug_1_generic_routing_uses_correct_probe_import() -> None:
    """2026-05-23 audit bug #1: the original generic-LLM fast-path
    fix imported from `.hf_probe` (which does not exist), and the
    surrounding try/except silently swallowed the ImportError, so
    `_generic_backend_picked` was ALWAYS False. Result: Gemma-2-27b
    still hit the scaffold abort despite the fix. Pin the correct
    import name."""
    src = _planner_source()
    assert "from .hf_probe import" not in src, (
        "cmd_up must NOT import from .hf_probe (module doesn't exist); " "the correct module is .probe"
    )
    fn_idx = src.find("def cmd_up(")
    block = src[fn_idx : fn_idx + 40000]
    assert (
        "from .probe import probe_model as _probe_model" in block
    ), "the generic-LLM detection must import probe_model from .probe"


def test_audit_bug_2_generic_fast_path_rejects_partial_blocks() -> None:
    """2026-05-23 audit bug #2: the generic-LLM fast-path used to fire
    on `FEASIBLE WITH WORK` + zero MISSING, allowing PARTIAL blocks
    (sliding-window+chunked-prefill, MoE adaptations, vision tower)
    through to `prepare --execute` where they crash at runtime. Pin
    the tighter gate: only `READY` with zero PARTIAL too."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    block = src[fn_idx : fn_idx + 40000]

    assert "_CompatStatus.PARTIAL" in block, "generic-LLM fast-path must reject components with PARTIAL status"

    route_block_idx = block.find("_route_via_generic_llm = True")
    assert route_block_idx >= 0
    pre_block = block[max(0, route_block_idx - 1500) : route_block_idx]
    assert '_early_compat.overall == "READY"' in pre_block, (
        "generic-LLM fast-path must require compat overall == 'READY' " "(not the looser 'FEASIBLE WITH WORK')"
    )


def test_audit_bug_3_module_tree_discovery_does_not_load_weights_by_default() -> None:
    """2026-05-23 audit bug #3: `discover_components_from_hf_id` used
    to call `AutoModel.from_pretrained(...)`, which downloads + loads
    full weights. For 27B models that's 54 GB + 30 minutes BEFORE
    the LLM call. Pin the config-only default."""
    src = (Path(cli.__file__).parent / "module_tree.py").read_text()
    assert "AutoModel.from_config" in src, (
        "discover_components_from_hf_id must use AutoModel.from_config " "(config-only, no weight load) by default"
    )
    assert "load_weights: bool = False" in src, "load_weights parameter must default to False"


def test_audit_bug_4_closest_supported_covers_common_llama_family() -> None:
    """2026-05-23 audit bug #4: many Llama-family model_types had no
    sibling mapping in `closest_supported_model()`, causing scaffold
    to abort even though compat said FEASIBLE/READY. Pin coverage
    for the common ones."""
    src = (Path(cli.__file__).parent / "compatibility.py").read_text()

    required = [
        "gemma2",
        "gemma",
        "olmo",
        "olmo2",
        "cohere",
        "granite",
        "internlm",
        "starcoder2",
        "llama4",
        "ministral",
        "mistral3",
    ]
    for mt in required:
        assert f'"{mt}":' in src, (
            f"closest_supported_model() candidates dict is missing "
            f"`{mt}` -> sibling mapping; scaffold will abort on this "
            f"model_type even though it's a known Llama-family arch"
        )


def test_audit_bug_5_meta_plan_skipped_for_generic_llm_route() -> None:
    """2026-05-23 audit bug #5: meta-plan ran for generic-LLM models
    that would skip scaffold entirely. That meant a 5+ minute waste
    (full weight load + LLM call) for a model that goes straight to
    `prepare --execute`. Pin that meta-plan is skipped when
    `_route_via_generic_llm`."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    block = src[fn_idx : fn_idx + 40000]
    meta_idx = block.find("_run_advisory_meta_plan(")
    assert meta_idx >= 0

    guard_block = block[max(0, meta_idx - 800) : meta_idx]
    assert "not _route_via_generic_llm" in guard_block, "meta-plan call must be guarded by `not _route_via_generic_llm`"


def test_audit_bug_6_env_compat_check_fires_before_scaffold_path() -> None:
    """2026-05-23 audit bug #6: env compat check only fired in the
    supported-model fast-path, so NEW models going through scaffold
    -> autofill -> prepare -> pytest still crashed at runtime
    instead of aborting upfront with the clear banner. Pin the
    early call."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    block = src[fn_idx : fn_idx + 40000]

    env_check_calls = [i for i in range(len(block)) if block.startswith("_check_demo_environment_compat()", i)]
    assert len(env_check_calls) >= 1, (
        "cmd_up must call _check_demo_environment_compat() at least "
        "once in the main body (not just the supported-model branch)"
    )

    scaffold_idx = block.find("Step 2/6  Scaffold")
    assert scaffold_idx > env_check_calls[0], (
        "env compat check must run BEFORE Step 2/6 (scaffold) so the "
        "abort happens upfront, not after LLM iteration starts"
    )


def test_audit_bug_9_rope_scaling_type_reads_rope_parameters_too() -> None:
    """2026-05-23 audit bug #9: compat `_rope_scaling_type` only read
    `rope_scaling`, missing `rope_parameters` (the newer field used
    by transformers 5.x for Phi-3.5 etc.). Models with scaling only
    in `rope_parameters` would silently report READY here."""
    src = (Path(cli.__file__).parent / "compatibility.py").read_text()
    fn_idx = src.find("def _rope_scaling_type(")
    assert fn_idx >= 0
    body = src[fn_idx : fn_idx + 1500]
    assert "rope_parameters" in body, (
        "_rope_scaling_type must also check `rope_parameters` for " "the type field, not just `rope_scaling`"
    )


def test_audit_bug_11_unknown_load_errors_do_not_trigger_pip_upgrade() -> None:
    """2026-05-23 audit bug #11: load failures classified as
    "unknown" (network timeouts, trust_remote_code import errors,
    OOM, etc.) still ran `pip install -U transformers`. That's
    wasteful and misleading. Pin that only `version` errors
    trigger the upgrade."""
    src = _planner_source()
    fn_idx = src.find("def _preflight_load_with_autofix(")
    assert fn_idx >= 0
    body = src[fn_idx : fn_idx + 4000]

    unknown_idx = body.find('if kind == "unknown":')
    pip_idx = body.find("Attempting automatic fix")
    assert unknown_idx >= 0 and pip_idx >= 0
    assert unknown_idx < pip_idx, (
        "the `unknown` classification must short-circuit BEFORE the " "pip install -U transformers code path"
    )


def test_audit_bug_13_sliding_window_check_does_not_lie_about_passing() -> None:
    """2026-05-23 audit bug #13: `check_sliding_window` set
    `passes=True` for a known incompatibility (sliding-window +
    chunked prefill), so the kernel table showed `[ ok ]`.
    Pin that the finding now reports passes=False."""
    from scripts.tt_hw_planner.kernel_constraints import check_sliding_window

    findings = check_sliding_window({"sliding_window": 4096}, 1)
    assert len(findings) == 1
    assert findings[0].passes is False, (
        "check_sliding_window must report passes=False when a " "sliding_window is present (chunked prefill incompat)"
    )


def test_audit_bug_14_sliding_window_check_detects_layer_types() -> None:
    """2026-05-23 audit bug #14: Gemma-2 / hybrid models use
    `layer_types: [...]` containing `sliding_attention` entries
    rather than the scalar `sliding_window` field. The kernel check
    used to silently miss those configs."""
    from scripts.tt_hw_planner.kernel_constraints import check_sliding_window

    cfg = {
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
    }
    findings = check_sliding_window(cfg, 1)
    assert len(findings) == 1, (
        "check_sliding_window must detect sliding layers expressed " "via layer_types[*]='sliding_attention'"
    )


def test_audit_bug_17_scaffold_distinguishes_generic_backend_error() -> None:
    """2026-05-23 audit bug #17: direct `scaffold` subcommand always
    asked for a sibling, even for generic backends where simple_text_demo
    is portable and no per-model tt/ folder is needed. Pin the
    differentiated error message that points to `prepare --execute`."""
    src = (Path(cli.__file__).parent / "scaffold.py").read_text()

    assert "routing_mode" in src and '"generic"' in src, "scaffold must inspect the picked backend's routing_mode"
    assert "prepare " in src and "--execute" in src, (
        "scaffold's generic-backend error must point users at " "`prepare --execute` instead of demanding a sibling"
    )


def test_audit_bug_20_encode_prompt_fallback_normalizes_tokens() -> None:
    """2026-05-23 audit bug #20: `ModelConfig.encode_prompt` fallback
    path called `self.tokenizer.encode(...)` without normalizing
    the return type. On transformers 5.x TokenizersBackend that
    can return a `tokenizers.Encoding` instead of List[int],
    breaking downstream `torch.tensor(...)`. Pin normalization."""
    mc_src = (_REPO_ROOT / "models" / "tt_transformers" / "tt" / "model_config.py").read_text()

    fn_idx = mc_src.find("def encode_prompt(self, prompt_text")
    assert fn_idx >= 0
    body = mc_src[fn_idx : fn_idx + 1500]
    assert "_normalize_token_result_to_list" in body, (
        "encode_prompt fallback must normalize the tokenizer.encode() " "return type for transformers 5.x compat"
    )


def test_learning_module_extends_backend_model_type_keys_idempotently() -> None:
    """2026-05-23 learning loop: `learning._extend_backend_model_type_keys`
    must:
      - Add a new model_type to the target backend's `model_type_keys`.
      - Be idempotent: re-adding the same key returns (True, "(no-op...)").
      - Not touch other backends in the same file.
      - Add a `model_type_keys=[...]` field if the backend block doesn't
        have one (e.g. the generic LLM/VLM catch-alls).
    """
    import tempfile
    from scripts.tt_hw_planner.learning import _extend_backend_model_type_keys

    src = """from typing import List

_BACKENDS: List[object] = [
    FamilyBackend(
        category="STT",
        name="Whisper (distil-large-v3)",
        demo_path="...",
        routing_mode="template",
        canonical_hf_id="distil-whisper/distil-large-v3",
        notes="...",
        model_type_keys=["whisper"],
        pipeline_tags=["automatic-speech-recognition"],
    ),
    FamilyBackend(
        category="LLM",
        name="tt_transformers / simple_text_demo",
        demo_path="...",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="...",
    ),
]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        path = Path(f.name)
    try:
        ok, msg = _extend_backend_model_type_keys(
            backend_name="Whisper (distil-large-v3)",
            new_model_type="speecht5",
            backends_file=path,
        )
        assert ok, msg
        new_src = path.read_text()
        assert (
            "'speecht5'" in new_src or '"speecht5"' in new_src
        ), "speecht5 must appear in the file after the extension"

        assert "whisper" in new_src

        ok2, msg2 = _extend_backend_model_type_keys(
            backend_name="Whisper (distil-large-v3)",
            new_model_type="speecht5",
            backends_file=path,
        )
        assert ok2, msg2
        assert "no-op" in msg2

        ok3, msg3 = _extend_backend_model_type_keys(
            backend_name="tt_transformers / simple_text_demo",
            new_model_type="brand_new_arch",
            backends_file=path,
        )
        assert ok3, msg3
        new_src = path.read_text()
        assert "brand_new_arch" in new_src

        assert "speecht5" in new_src
        assert "whisper" in new_src
    finally:
        path.unlink(missing_ok=True)


def test_learning_module_extends_closest_supported_model_idempotently() -> None:
    """2026-05-23 learning loop: `learning._add_to_closest_supported_model_map`
    injects the new mapping into `compatibility.closest_supported_model`'s
    candidates dict. Idempotent."""
    import tempfile
    from scripts.tt_hw_planner.learning import (
        _add_to_closest_supported_model_map,
    )

    src = """def closest_supported_model(model_id, cfg):
    if model_id in SUPPORTED_HF_MODELS:
        return model_id
    mt = (cfg.get("model_type") or "").lower()
    candidates = {
        "qwen2": "Qwen/Qwen2.5-32B",
        "llama": "meta-llama/Llama-3.1-8B",
    }
    if mt in candidates:
        return candidates[mt]
    return None
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        path = Path(f.name)
    try:
        ok, msg = _add_to_closest_supported_model_map(
            new_model_type="speecht5",
            sibling_model_id="microsoft/speecht5_tts",
            compat_file=path,
        )
        assert ok, msg
        new_src = path.read_text()
        assert '"speecht5"' in new_src
        assert "microsoft/speecht5_tts" in new_src

        assert '"qwen2"' in new_src
        assert '"llama"' in new_src

        ok2, msg2 = _add_to_closest_supported_model_map(
            new_model_type="speecht5",
            sibling_model_id="microsoft/speecht5_tts",
            compat_file=path,
        )
        assert ok2, msg2
        assert "no-op" in msg2
    finally:
        path.unlink(missing_ok=True)


def test_register_successful_bringup_is_best_effort_and_never_raises() -> None:
    """2026-05-23 learning loop: `register_successful_bringup` MUST NOT
    raise on any failure mode. The contract is "best-effort write +
    list-of-messages return"; a write failure here can't turn a
    successful bring-up into a failed return code. Test with all-bad
    paths to ensure we get a graceful (not-OK) result."""
    from scripts.tt_hw_planner.learning import register_successful_bringup

    msgs = register_successful_bringup(
        model_id="test/never-going-to-exist",
        model_type="brand_new_xyz",
        category="LLM",
        backend_name="this-backend-does-not-exist-anywhere",
        sibling_model_id="test/never-going-to-exist",
        path="X. test path",
        notes="invariant pin",
    )
    assert isinstance(msgs, list)

    assert any(m.startswith(("OK", "FAIL")) for m in msgs), (
        f"register_successful_bringup should return OK/FAIL annotated " f"messages, got: {msgs}"
    )


def test_cmd_up_calls_register_bringup_success_on_cold_start_success() -> None:
    """2026-05-23 learning loop: `cmd_up`'s cold-start branch MUST
    call `_register_bringup_success` after `cmd_prepare` returns 0.
    Without this, the next time the same arch is requested it still
    has to go through inline auto-onboard."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0
    block = src[fn_idx : fn_idx + 50000]

    cold_idx = block.find("_cold_start_signal is not None")
    assert cold_idx >= 0

    cold_branch = block[cold_idx : cold_idx + 8000]
    assert "_register_bringup_success(" in cold_branch, (
        "cmd_up cold-start branch must call _register_bringup_success " "after cmd_prepare returns 0"
    )

    supp_idx = block.find("if _already_supported:")

    while supp_idx >= 0:
        tail = block[supp_idx : supp_idx + 12000]
        if "cmd_prepare(prepare_argv)" in tail:
            break
        next_idx = block.find("if _already_supported:", supp_idx + 1)
        if next_idx == supp_idx:
            break
        supp_idx = next_idx
    assert supp_idx >= 0
    supp_branch = block[supp_idx : supp_idx + 12000]
    assert "_register_bringup_success(" in supp_branch, (
        "cmd_up supported-model fast path must also call " "_register_bringup_success after cmd_prepare returns 0"
    )


def test_cmd_up_calls_register_bringup_success_on_iterate_loop_graduation() -> None:
    """2026-05-23 learning loop: cmd_up's iterate-loop exit must also
    call _register_bringup_success when the loop graduates (rc == 0).
    Without this, hours of iteration are wasted because the next
    similar model has to go through inline auto-onboard AGAIN."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    block = src[fn_idx : fn_idx + 60000]
    loop_idx = block.find("_run_auto_iterate_loop(")
    assert loop_idx >= 0
    loop_branch = block[loop_idx : loop_idx + 4000]
    assert "_register_bringup_success(" in loop_branch, (
        "cmd_up must call _register_bringup_success after the auto-"
        "iterate loop returns 0 so the per-component port is "
        "persisted as a sibling for the next similar model"
    )


def test_every_known_category_has_at_least_one_generic_backend() -> None:
    """2026-05-23 universal-generic coverage: per user request "make
    it generic for ALL categories not just LLM/VLM", every well-known
    HF category must have at least one `routing_mode="generic"`
    backend registered. This way the cold-start path (Pattern B) is
    not just for LLM/VLM -- it's the universal floor.

    The list below mirrors `probe.PIPELINE_CATEGORY`'s value-set plus
    `Unknown` (the catch-all when probe can't classify)."""
    from scripts.tt_hw_planner.family_backends import backends_for_category

    expected_categories = (
        "LLM",
        "VLM",
        "STT",
        "TTS",
        "Image",
        "Video",
        "CNN",
        "Embed",
        "NLP",
        "Unknown",
    )
    for cat in expected_categories:
        cands = backends_for_category(cat)
        assert cands, f"category {cat!r} has NO registered backend at all"
        generics = [b for b in cands if getattr(b, "routing_mode", "") == "generic"]
        assert generics, (
            f"category {cat!r} has no generic-routing backend; the "
            f"cold-start path is unreachable for this category. Add a "
            f"`hf_eager universal ({cat})` entry pointing at "
            f"`models/demos/hf_eager/demo.py`."
        )


def test_pick_backend_category_default_prefers_generic_over_template() -> None:
    """2026-05-23: when there's no exact / pipeline match,
    `pick_backend_with_quality` MUST prefer a `routing_mode="generic"`
    backend over a template-routing one. Otherwise an unknown audio
    arch (e.g. SpeechT5) lands on Whisper's template and the bring-up
    loop wastes iterations against the wrong skeleton.

    Pin this for every category where BOTH a template AND a generic
    backend are registered."""
    from scripts.tt_hw_planner.family_backends import pick_backend_with_quality

    cases = [
        ("STT", "speecht5_tts"),
        ("CNN", "totally_made_up_vision_arch"),
        ("Image", "flux"),
        ("Embed", "some_new_embedding_arch"),
        ("NLP", "some_new_nlp_arch"),
    ]
    for category, made_up_mt in cases:
        b, q = pick_backend_with_quality(category=category, model_type=made_up_mt)
        assert q == "category-default", f"{category}/{made_up_mt} should fall through to " f"category-default; got {q}"
        assert getattr(b, "routing_mode", "") == "generic", (
            f"category-default fallback for {category}/{made_up_mt} "
            f"must prefer a generic backend over the template; got "
            f"`{b.name}` (routing={b.routing_mode}). The new fallback "
            f"policy is: unknown arch -> hf_eager generic, NOT silent "
            f"wrong-template."
        )


def test_hf_eager_universal_demo_exists_and_is_runnable() -> None:
    """2026-05-23: the universal HF-eager demo MUST exist at the path
    the generic backends register against. Without this file, every
    generic backend would point at a 404 demo path and the bring-up
    loop would fail with FileNotFoundError before iterating."""
    demo = _REPO_ROOT / "models" / "demos" / "hf_eager" / "demo.py"
    assert demo.is_file(), (
        f"models/demos/hf_eager/demo.py is missing. This is the " f"target every generic FamilyBackend points at."
    )
    src = demo.read_text()

    assert "os.environ" in src and "HF_MODEL" in src, "hf_eager demo must read HF_MODEL from the environment"

    assert "_classify_arch_to_automodel" in src or "AutoModelFor" in src, (
        "hf_eager demo must dispatch on the model's HF config so it "
        "can handle every category (LLM, VLM, STT, vision, ...)"
    )

    assert "CPU" in src and "FALLBACK" in src.upper(), (
        "hf_eager demo must include a CPU-fallback banner so users " "aren't misled into thinking it runs on tt-metal"
    )


def test_cold_start_scaffold_error_exists_and_subclasses_scaffold_error() -> None:
    """2026-05-23 cold-start feature: when there is no closest sibling
    AND/OR the model uses a generic architecture-portable demo, scaffold
    must raise `ColdStartScaffoldError` (a `ScaffoldError` subclass) so
    `cmd_up` can fall through to `prepare --execute` instead of aborting.
    Pin the type, fields, and inheritance."""
    from scripts.tt_hw_planner.scaffold import (
        ColdStartScaffoldError,
        ScaffoldError,
    )

    assert issubclass(ColdStartScaffoldError, ScaffoldError), (
        "ColdStartScaffoldError must subclass ScaffoldError so existing "
        "`except ScaffoldError:` call sites keep working"
    )
    err = ColdStartScaffoldError(
        "test/foo",
        "no sibling found",
        suggested_cmd="python -m scripts.tt_hw_planner prepare test/foo --execute",
    )
    assert err.model_id == "test/foo"
    assert "no sibling found" in str(err)
    assert "cold-start" in str(err).lower()
    assert "prepare test/foo --execute" in str(err)


def test_plan_scaffold_raises_cold_start_for_no_sibling() -> None:
    """2026-05-23 cold-start feature: when an LLM/VLM model has no
    `closest_supported_model()` sibling, plan_scaffold MUST raise
    `ColdStartScaffoldError` (not plain `ScaffoldError`), so `cmd_up`
    falls through to `prepare --execute` ("bring things from scratch").

    We can't easily mock plan_scaffold without spinning up the HF probe,
    so this is a source-grep invariant. Pin both code sites."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "scaffold.py").read_text()

    no_sibling_idx = src.find("if not sibling_id:")
    assert no_sibling_idx >= 0
    body = src[no_sibling_idx : no_sibling_idx + 1500]
    assert "ColdStartScaffoldError" in body, (
        "the `if not sibling_id:` branch must raise ColdStartScaffoldError "
        "so cmd_up can fall through to prepare --execute"
    )

    generic_idx = src.find('routing_mode", "") == "generic"')
    assert generic_idx >= 0
    body2 = src[generic_idx : generic_idx + 1500]
    assert "ColdStartScaffoldError" in body2, (
        "the generic-backend branch must raise ColdStartScaffoldError " "(not plain ScaffoldError) for the same reason"
    )


def test_cmd_up_catches_cold_start_and_routes_to_prepare_execute() -> None:
    """2026-05-23 cold-start feature: cmd_up must catch
    `ColdStartScaffoldError` BEFORE invoking cmd_scaffold, print a
    cold-start banner, and call cmd_prepare(execute=True). Without this
    wiring the new ColdStartScaffoldError just bubbles up as a plain
    error and the user sees an abort -- defeating the whole feature."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0
    block = src[fn_idx : fn_idx + 60000]

    assert "ColdStartScaffoldError" in block, "cmd_up must import ColdStartScaffoldError and probe for it"
    assert "_cold_start_signal" in block, (
        "cmd_up must capture the ColdStartScaffoldError into a signal "
        "variable so it can branch on it BEFORE cmd_scaffold runs"
    )

    cold_branch_idx = block.find("_cold_start_signal is not None")
    assert cold_branch_idx >= 0
    cold_branch = block[cold_branch_idx : cold_branch_idx + 4000]
    assert "cmd_prepare(prepare_argv)" in cold_branch, "the cold-start branch must invoke cmd_prepare directly"
    assert "execute=True" in cold_branch, "the cold-start branch must call cmd_prepare with execute=True"

    scaffold_call = block.find("cmd_scaffold(scaffold_argv)")
    assert scaffold_call >= 0
    assert cold_branch_idx < scaffold_call, (
        "cold-start branch must run BEFORE cmd_scaffold, otherwise the "
        "cold-start signal would be consumed by cmd_scaffold returning 2"
    )


def test_demo_folder_scaffold_uses_cold_start_when_no_backend_registered() -> None:
    """2026-05-23 cold-start feature: even non-LLM categories (vision,
    audio, embedding, ...) should not hard-abort when no
    `FamilyBackend` is registered. They should raise
    ColdStartScaffoldError pointing at `auto-onboard` so the LLM can
    draft a backend entry."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "scaffold.py").read_text()
    fn_idx = src.find("def _plan_demo_folder_scaffold(")
    assert fn_idx >= 0
    body = src[fn_idx : fn_idx + 2000]
    assert "ColdStartScaffoldError" in body, (
        "_plan_demo_folder_scaffold must use ColdStartScaffoldError for "
        "the no-backend-registered case so cmd_up can hand off to "
        "auto-onboard"
    )
    assert "auto-onboard" in body, (
        "the cold-start message must direct the user to `auto-onboard` " "(the LLM-drafted FamilyBackend feature)"
    )


def test_audit_bug_10_moe_block_fires_for_all_non_mla_moe_models() -> None:
    """2026-05-23 audit bug #10: MoE building block previously only
    fired for `mixtral` / `qwen*` / `phi*` / `gemma*` model_types.
    OLMoE, Jamba, GraniteMoE, and other non-MLA MoE families slipped
    past as `READY` even though the runtime mixtral_moe.py would
    hard-code num_devices=8 / top-2 and silently mis-route. Verify
    that an `olmoe`-like config now triggers the MoE block as
    PARTIAL (needed=True), not silently passes."""
    from scripts.tt_hw_planner.compatibility import BUILDING_BLOCKS, Status

    moe_block = next(
        (b for b in BUILDING_BLOCKS if b.name == "MoE routing (Mixtral-style)"),
        None,
    )
    assert moe_block is not None
    olmoe_like = {
        "model_type": "olmoe",
        "num_local_experts": 64,
        "num_experts_per_tok": 8,
        "hidden_size": 2048,
    }
    granite_moe_like = {
        "model_type": "granitemoe",
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    jamba_like = {
        "model_type": "jamba",
        "num_experts": 16,
        "moe_intermediate_size": 1024,
    }
    for cfg in (olmoe_like, granite_moe_like, jamba_like):
        assert moe_block.needed_when(cfg), (
            f"MoE block must report needed=True for {cfg['model_type']} "
            f"(audit bug #10). Currently this family slips past as READY "
            f"and crashes/mis-routes at runtime."
        )
        assert moe_block.status_when_needed == Status.PARTIAL


def test_audit_bug_19_env_compat_check_catches_hard_coded_grid_literals() -> None:
    """2026-05-23 audit bug #19: `_check_demo_environment_compat` only
    checked `find_grid`. Verify that the check ALSO grep-scans for
    `compute_with_storage_grid_size = ttnn.CoreCoord(...)` literals
    without an adjacent `_dispatch_safe_grid`, and surfaces them as
    problems. This stops future regressions where someone reverts a
    SDPA/QKV/MLP migration."""
    cli_src = _planner_source()
    fn_idx = cli_src.find("def _check_demo_environment_compat(")
    assert fn_idx >= 0
    body = cli_src[fn_idx : fn_idx + 8000]
    assert "_dispatch_safe_grid" in body, (
        "env compat check must verify model_config.py defines the " "`_dispatch_safe_grid` helper"
    )
    assert "compute_with_storage_grid_size" in body, (
        "env compat check must grep model_config.py for any remaining "
        "hard-coded `compute_with_storage_grid_size = ...` grid literals"
    )


def test_audit_bug_21_lm_head_prefetcher_clamps_to_actual_grid_height() -> None:
    """2026-05-23 audit bug #21: `get_lm_head_reshard_mem_config`
    used hard-coded y=7 in CoreCoord ranges, which would place
    kernels on dispatch cores for meshes whose compute grid is
    less than 8 rows tall. Pin that the function clamps to
    `self.max_grid_size.y - 1` when available."""
    mc_src = (_REPO_ROOT / "models" / "tt_transformers" / "tt" / "model_config.py").read_text()
    fn_idx = mc_src.find("def get_lm_head_reshard_mem_config(")
    assert fn_idx >= 0
    body = mc_src[fn_idx : fn_idx + 1500]
    assert "max_grid_size" in body, (
        "get_lm_head_reshard_mem_config must clamp the prefetcher core "
        "range to the actual mesh compute grid height (audit bug #21)"
    )

    assert "min(7" in body, (
        "Expected `min(7, gs.y - 1)` clamp pattern; the literal `7` "
        "remains as an upper bound (since the prefetcher only uses "
        "rows 0..7) but it must not be the *only* value"
    )


def test_audit_bug_22_cmd_compat_returns_nonzero_for_targeted_with_missing_blocks() -> None:
    """2026-05-23 audit bug #22: `cmd_compat` returned exit 0 for
    TARGETED models even when they had MISSING needed building
    blocks (i.e. architecturally impossible). `up --auto` would
    then proceed to scaffold + waste an LLM iteration. Pin that
    cmd_compat now returns non-zero when ANY needed block is MISSING,
    regardless of `report.overall` text."""
    src = _planner_source()
    fn_idx = src.find("def cmd_compat(")
    assert fn_idx >= 0
    body = src[fn_idx : fn_idx + 6000]

    assert "audit bug #22" in body, "cmd_compat must reference the audit-bug-22 fix for traceability"
    assert "MISSING" in body and "needed" in body, (
        "cmd_compat must check for needed&MISSING blocks beyond the "
        "BLOCKED overall string, so TARGETED models with missing blocks "
        "still exit non-zero"
    )


def test_audit_bug_24_qk_rmsnorm_block_fires_for_phi4_and_olmo2() -> None:
    """2026-05-23 audit bug #24: Q/K RMSNorm block previously only
    fired for `qwen3*`. The description and runtime both support
    Phi-4 / Olmo2 / OLMoE which also use q_norm/k_norm tensors.
    Pin that the predicate now also matches these."""
    from scripts.tt_hw_planner.compatibility import BUILDING_BLOCKS

    qk_block = next(
        (b for b in BUILDING_BLOCKS if b.name == "Q/K RMSNorm"),
        None,
    )
    assert qk_block is not None
    for mt in ("qwen3", "phi4", "olmo2", "olmoe"):
        assert qk_block.needed_when({"model_type": mt}), (
            f"Q/K RMSNorm block must report needed=True for {mt} " f"(audit bug #24)"
        )

    assert qk_block.needed_when(
        {"model_type": "internvl", "text_config": {"model_type": "phi4"}}
    ), "Q/K RMSNorm block must recurse into text_config for VLMs"

    assert not qk_block.needed_when({"model_type": "llama"})


def test_audit_bug_7_8_model_config_uses_dispatch_safe_grid_helper() -> None:
    """2026-05-23 audit bugs #7, #8: SDPA and QKV program configs
    used hard-coded `(8, 8)` / `(8, 10)` grids instead of clamping
    to actual `compute_with_storage_grid_size`. On BH QB2 1x4 the
    real compute grid is 11x10 (not 12x10) due to dispatch core
    reservations; hard-coded `(8, 10)` may still be safe today,
    but the pattern of NOT clamping is the bug class we already hit
    with `find_grid`. Pin that the helper exists AND that SDPA/QKV
    use it."""
    mc_src = (_REPO_ROOT / "models" / "tt_transformers" / "tt" / "model_config.py").read_text()
    assert "def _dispatch_safe_grid(" in mc_src, (
        "model_config.py must define `_dispatch_safe_grid` helper " "to clamp grids to the device's actual compute grid"
    )

    for fn_name in (
        "get_attn_sdpa_decode_program_config",
        "get_attn_qkv_program_config",
    ):
        idx = mc_src.find(f"def {fn_name}(")
        if idx < 0:
            continue
        body = mc_src[idx : idx + 4000]
        assert "_dispatch_safe_grid(" in body, (
            f"{fn_name} must call self._dispatch_safe_grid(...) " f"instead of using hard-coded grid literals"
        )


def test_meta_plan_timeout_is_at_least_180s() -> None:
    """2026-05-23 bug fix: meta-plan timed out at 120s on Gemma-2-27b
    (large config, the LLM had more context to chew on). Bump the
    default to >= 180s so it doesn't fail spuriously on big models.
    Note: it's advisory-only so a timeout doesn't block bring-up,
    but it does mean the bring-up runs WITHOUT meta-plan context,
    which is a regression in capability."""
    src = (Path(cli.__file__).parent / "meta_plan.py").read_text()

    import re

    matches = re.findall(r"timeout_s:\s*int\s*=\s*(\d+)", src)
    assert matches, "no timeout_s default found in meta_plan.py"
    for val in matches:
        assert int(val) >= 180, (
            f"meta_plan.py timeout_s default is {val}s; must be >= 180s " f"so large models don't spuriously time out"
        )


def test_cmd_up_auto_routes_already_supported_to_prepare_execute() -> None:
    """2026-05-23 UX fix: `up --auto` must be the universal command.
    For NEW architectures it scaffolds + iterates; for ALREADY
    SUPPORTED models it auto-routes to `prepare --execute`. The user
    should never need to know which path applies -- the tool decides.

    Pinned behaviors:
      - cmd_up detects the supported case BEFORE calling cmd_scaffold
      - It calls cmd_prepare with execute=True (not just printing a
        next-step hint)
      - cmd_scaffold is NOT called in the supported path
    """
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0

    block = src[fn_idx : fn_idx + 60000]
    assert "_already_supported" in block, "cmd_up must detect 'already supported' case before scaffold"

    scaffold_call = block.find("cmd_scaffold(scaffold_argv)")
    early_check = block.find("_already_supported")
    assert early_check >= 0 and scaffold_call >= 0 and early_check < scaffold_call, (
        "the supported-model branch must run BEFORE cmd_scaffold, "
        "otherwise we still hit the scaffold ScaffoldError path"
    )

    supported_branch = block[early_check:scaffold_call]
    assert "cmd_prepare(prepare_argv)" in supported_branch, (
        "supported-model branch must invoke cmd_prepare directly, "
        "not just print the command for the user to copy/paste"
    )
    assert "execute=True" in supported_branch, (
        "supported-model branch must call cmd_prepare with " "execute=True so the demo actually runs"
    )


def test_cross_component_context_block_produces_position_signatures_failures() -> None:
    """Improvement 1 (cross-component context): the helper
    `_build_cross_component_context_block` must produce a non-empty
    block with three sections:
      1. position summary  (X graduated / Y in-progress / Z untouched)
      2. other components' __call__ signatures
      3. cross-component failure patterns

    When applied to a real demo dir (sam2_hiera_tiny is the only
    fully-converged one we ship), all three sections must appear."""
    from scripts.tt_hw_planner.cli import _build_cross_component_context_block

    block = _build_cross_component_context_block(
        Path("models/demos/vision/segmentation/sam2_hiera_tiny").resolve(),
        current_target="self_attention",
        attempts_per_component={"self_attention": 2},
        last_failure_class_per_component={
            "self_attention": "PCC",
            "encoder_stack": "WRAPPER",
        },
    )
    assert block, "block must be non-empty for an existing demo dir"
    assert "CROSS-COMPONENT CONTEXT" in block, "section header must be present"
    assert "position:" in block, "must include position summary"
    assert "signatures" in block.lower(), "must include signatures section"
    assert "failure pattern" in block.lower(), "must include failure pattern section"

    sigs_section = block.split("signatures")[1].split("failure pattern")[0]
    assert "self_attention:" not in sigs_section, (
        "the current target must NOT appear in the 'other components' "
        "signatures section -- it's the in-progress component, not context"
    )


def test_cross_component_context_block_safe_when_no_status_json() -> None:
    """The helper must return an empty string (NOT raise) when
    bringup_status.json is missing -- it's strictly additive context
    and must never block the prompt assembly."""
    from scripts.tt_hw_planner.cli import _build_cross_component_context_block
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        block = _build_cross_component_context_block(
            Path(tmp),
            current_target=None,
        )
    assert block == "", (
        "must return '' when no bringup_status.json (additive context " "must never block prompt assembly)"
    )


def test_meta_plan_skip_llm_returns_valid_unknown_verdict() -> None:
    """The meta-plan helper must produce a valid `MetaPlanVerdict`
    even when `skip_llm=True` (offline tests). The verdict's
    feasibility must be `UNKNOWN`, parse_error must indicate the
    skip reason, and `format_verdict_banner` must produce a
    non-empty string that mentions 'advisory only'."""
    from scripts.tt_hw_planner.meta_plan import (
        run_meta_plan,
        format_verdict_banner,
    )

    v = run_meta_plan(
        model_id="fake/test",
        category="CNN",
        model_type="fake",
        backend_name="(none)",
        match_quality="none",
        box="QB2",
        mesh=None,
        components=[],
        skip_llm=True,
    )
    assert v.feasibility == "UNKNOWN"
    assert v.parse_error and "skip_llm" in v.parse_error
    banner = format_verdict_banner(v)
    assert "META-PLAN" in banner
    assert "advisory only" in banner.lower(), (
        "banner must communicate that meta-plan is advisory only -- " "users must know it doesn't gate the bring-up"
    )


def test_meta_plan_extract_json_handles_markdown_fences() -> None:
    """LLMs sometimes wrap JSON in ```json ... ``` fences even when
    the prompt asks for raw JSON. `_extract_json` must strip both
    sides cleanly."""
    from scripts.tt_hw_planner.meta_plan import _extract_json

    obj = _extract_json('```json\n{"feasibility": "HIGH"}\n```')
    assert obj == {"feasibility": "HIGH"}

    obj = _extract_json('```\n{"feasibility": "MEDIUM"}\n```')
    assert obj == {"feasibility": "MEDIUM"}

    obj = _extract_json('   {"feasibility": "LOW"}   ')
    assert obj == {"feasibility": "LOW"}

    obj = _extract_json("not json at all")
    assert obj is None


def test_cmd_up_runs_meta_plan_by_default_and_no_meta_plan_disables_it() -> None:
    """Wiring test: cmd_up must call `_run_advisory_meta_plan` after
    the backend gate, AND must honor `--no-meta-plan` to disable."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0
    block = src[fn_idx : fn_idx + 40000]
    assert "_run_advisory_meta_plan(" in block, "cmd_up must call _run_advisory_meta_plan before scaffold"
    assert "no_meta_plan" in block, "cmd_up must check args.no_meta_plan to honor the disable flag"

    backend_gate_idx = block.find("_enforce_backend_match_quality_or_abort")
    meta_idx = block.find("_run_advisory_meta_plan")
    assert backend_gate_idx >= 0 and meta_idx > backend_gate_idx, (
        "meta-plan must run AFTER the backend gate so a backend-" "mismatch abort doesn't waste a meta-plan LLM call"
    )

    scaffold_idx = block.find("Step 2/6  Scaffold")
    assert scaffold_idx > meta_idx, (
        "meta-plan must run BEFORE scaffold so the user sees the " "advisory verdict before the demo dir is created"
    )


def test_up_has_no_meta_plan_flag() -> None:
    """The `--no-meta-plan` flag must be registered on `up` so users
    have a documented opt-out."""
    src = _planner_source()
    assert '"--no-meta-plan"' in src, "--no-meta-plan must be registered on the `up` parser"


def test_auto_onboard_skip_llm_produces_valid_proposal() -> None:
    """The `auto_onboard()` orchestrator must produce a valid
    `AutoOnboardProposal` even when `skip_llm=True` (offline tests
    and dry-runs). The deterministic stub:
      - MUST have zero validation errors
      - MUST have use_module_tree=True
      - MUST have model_type_keys containing the probe's model_type
      - MUST have a non-empty backend_dataclass_source ready to splice
      - SHOULD reflect at least one discovered component when the
        model has any architectural structure"""
    from unittest.mock import patch
    from types import SimpleNamespace
    from scripts.tt_hw_planner import auto_onboard as _ao
    from scripts.tt_hw_planner.module_tree import DiscoveredComponent

    fake_probe = SimpleNamespace(
        model_id="fake/test-model",
        category="CNN",
        saved_dtype="float32",
        memory_model=None,
        raw_config={"model_type": "fake_test_arch"},
        pipeline_tag="image-segmentation",
    )
    fake_components = [
        DiscoveredComponent(
            name="vision_encoder",
            submodule_path="vision_encoder",
            class_name="FakeVisionEncoder",
            occurrences=1,
            leaf_op_count=10,
            sample_paths=["vision_encoder"],
            status_hint="NEW",
        ),
        DiscoveredComponent(
            name="block",
            submodule_path="vision_encoder.blocks",
            class_name="FakeBlock",
            occurrences=4,
            leaf_op_count=16,
            sample_paths=["vision_encoder.blocks.0"],
            status_hint="NEW",
        ),
    ]
    with patch.object(_ao, "probe_model", return_value=fake_probe), patch.object(
        _ao, "discover_components_from_hf_id", return_value=fake_components
    ):
        proposal = _ao.auto_onboard("fake/test-model", skip_llm=True)

    assert proposal.validation_errors == [], f"skip-llm stub must pass validation; got {proposal.validation_errors}"
    assert proposal.backend_python_repr["use_module_tree"] is True, (
        "auto-onboarded backends MUST set use_module_tree=True so they "
        "work without a hand-written sibling template demo"
    )
    keys_lower = [str(k).lower() for k in proposal.backend_python_repr["model_type_keys"]]
    assert "fake_test_arch" in keys_lower, (
        f"model_type_keys must include the probe's model_type; got "
        f"{proposal.backend_python_repr['model_type_keys']}"
    )
    assert proposal.backend_dataclass_source.strip().startswith(
        "FamilyBackend("
    ), "backend_dataclass_source must be ready to splice as a Python expression"
    assert len(proposal.discovered_components) == 2, "discovered_components must reflect the actual module-tree walk"


def test_auto_onboard_proposal_rejects_invalid_drafts() -> None:
    """`_validate_proposal` must reject drafts that are missing
    required fields, have a wrong category, or fail to include the
    new model's model_type in their keys. This is the guardrail that
    keeps a sloppy LLM response from polluting family_backends.py."""
    from scripts.tt_hw_planner.auto_onboard import _validate_proposal

    errs = _validate_proposal({"category": "CNN"}, new_model_type="x")
    assert any("missing required field" in e for e in errs), "should reject incomplete drafts"

    errs = _validate_proposal(
        {
            "category": "NotARealCategory",
            "name": "X",
            "demo_path": "p",
            "routing_mode": "template",
            "canonical_hf_id": None,
            "model_type_keys": ["x"],
            "use_module_tree": True,
        },
        new_model_type="x",
    )
    assert any("category must be" in e for e in errs), "should reject invalid categories"

    errs = _validate_proposal(
        {
            "category": "CNN",
            "name": "X",
            "demo_path": "p",
            "routing_mode": "generic",
            "canonical_hf_id": None,
            "model_type_keys": ["x"],
            "use_module_tree": True,
        },
        new_model_type="x",
    )
    assert any("routing_mode must be 'template'" in e for e in errs)

    errs = _validate_proposal(
        {
            "category": "CNN",
            "name": "X",
            "demo_path": "p",
            "routing_mode": "template",
            "canonical_hf_id": None,
            "model_type_keys": ["x"],
            "use_module_tree": False,
        },
        new_model_type="x",
    )
    assert any("use_module_tree must be true" in e for e in errs)

    errs = _validate_proposal(
        {
            "category": "CNN",
            "name": "X",
            "demo_path": "p",
            "routing_mode": "template",
            "canonical_hf_id": None,
            "model_type_keys": ["something_else"],
            "use_module_tree": True,
        },
        new_model_type="brand_new_arch",
    )
    assert any("must include" in e for e in errs), "should reject drafts that don't claim the new model_type"

    errs = _validate_proposal(
        {
            "category": "CNN",
            "name": "BrandNew (new arch)",
            "demo_path": "models/demos/x",
            "routing_mode": "template",
            "canonical_hf_id": "x/y",
            "model_type_keys": ["brand_new_arch"],
            "use_module_tree": True,
        },
        new_model_type="brand_new_arch",
    )
    assert errs == [], f"valid draft should pass; got {errs}"


def test_auto_onboard_write_refuses_invalid_proposal() -> None:
    """`write_backend_into_registry` must refuse to splice an invalid
    proposal (validation_errors non-empty) into family_backends.py.
    This is the final guardrail before disk."""
    from scripts.tt_hw_planner.auto_onboard import (
        AutoOnboardProposal,
        write_backend_into_registry,
    )

    invalid = AutoOnboardProposal(
        model_id="fake/test",
        new_model_type="fake",
        new_pipeline_tag=None,
        inferred_category="CNN",
        validation_errors=["some error"],
        backend_dataclass_source="",
    )
    ok, msg = write_backend_into_registry(invalid)
    assert ok is False, "must refuse invalid proposals"
    assert "validation errors" in msg


def test_auto_onboard_subcommand_registered() -> None:
    """The `auto-onboard` subcommand must be wired into the CLI so
    `python -m scripts.tt_hw_planner auto-onboard <model_id>` works.
    Also must accept `--accept` (write to disk) and `--skip-llm`
    (offline mode)."""
    src = _planner_source()

    assert '"auto-onboard"' in src, "auto-onboard must be in SUBCOMMANDS allowlist"

    assert (
        'sub.add_parser(\n        "auto-onboard"' in src
    ), "auto-onboard sub-parser must be created with `sub.add_parser`"

    assert '"--accept"' in src and '"--skip-llm"' in src, (
        "auto-onboard must expose --accept (write to disk) and " "--skip-llm (offline dry-run) flags"
    )

    from scripts.tt_hw_planner.commands import auto_onboard as _ao_cmd_mod

    assert (
        "def cmd_auto_onboard(" in Path(_ao_cmd_mod.__file__).read_text()
    ), "cmd_auto_onboard must be defined in scripts/tt_hw_planner/commands/auto_onboard.py"


def test_module_tree_discover_finds_high_level_components() -> None:
    """Module-tree decomposition (2026-05-23 audit defect 2):
    `discover_components` must walk a torch.nn.Module tree, cluster
    by `type(m).__name__`, and emit one component per architectural
    cluster -- WITHOUT depending on a sibling template demo's
    filenames. It must:

      - Recognize high-level architectural suffixes (*Block, *Layer,
        *Head, *Attention, *Encoder, *Decoder, *MLP, *PatchEmbed) and
        emit a component even for singleton occurrences.
      - Strip the per-model class-name prefix (Sam2HieraBlock ->
        hiera_block) so component names are clean snake_case.
      - Record the real `named_modules()` path on each component,
        collapsing ModuleList(*) numeric suffixes (the four
        Sam2HieraBlock children at `vision_encoder.blocks.{0,1,2,3}`
        collapse to a single `vision_encoder.blocks` path).
      - Hint NEW status for every discovered component (ADAPT was
        removed 2026-05-31; trichotomy collapsed to REUSE/NEW. The
        reuse_registry is the only source of REUSE; class-name alone
        is no longer enough evidence for any other label)."""
    import torch.nn as nn
    from scripts.tt_hw_planner.module_tree import discover_components

    class Sam2HieraBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(64, 64)
            self.norm = nn.LayerNorm(64)

    class Sam2HieraSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(64, 192)

    class Sam2PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 64, 16)

    class Sam2MaskDecoderHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.out = nn.Linear(64, 16)

    class Sam2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Sam2HieraBlock() for _ in range(4)])
            self.self_attention = Sam2HieraSelfAttention()
            self.patch_embed = Sam2PatchEmbed()
            self.head = Sam2MaskDecoderHead()

    comps = discover_components(Sam2Model())
    names = [c.name for c in comps]
    paths = {c.name: c.submodule_path for c in comps}
    occ = {c.name: c.occurrences for c in comps}
    hints = {c.name: c.status_hint for c in comps}

    assert "hiera_block" in names, f"Sam2HieraBlock (4 instances) must be promoted; got names {names}"

    assert paths["hiera_block"] == "blocks", (
        f"Sam2HieraBlock children at blocks.0..3 must collapse to "
        f"parent path 'blocks'; got {paths['hiera_block']!r}"
    )
    assert occ["hiera_block"] == 4, "occurrence count must reflect all 4 children"

    assert (
        "mask_decoder_head" in names or "head" in names
    ), f"Singleton high-level Head class must still be emitted; got {names}"
    assert "patch_embed" in names, f"Singleton high-level PatchEmbed class must be emitted; got {names}"

    # Post-ADAPT-removal (2026-05-31): every discovered component now
    # defaults to NEW. The reuse_registry is the only source of REUSE.
    assert (
        hints.get("patch_embed") == "NEW"
    ), f"PatchEmbed should default to NEW post-ADAPT-removal, got {hints.get('patch_embed')}"

    for n in names:
        assert not n.startswith("sam2_"), f"Per-model prefix 'Sam2' must be stripped; got {n!r}"


def test_module_tree_handles_single_class_model() -> None:
    """Edge case: a model consisting of a single nn.Linear or similar
    leaf should NOT silently produce zero components -- callers
    expect at least one entry to drive the bring-up loop. The wrapping
    function `_extract_components_from_module_tree` emits a single
    `model_root` Component in that case (tested separately); raw
    `discover_components` simply returns []."""
    import torch.nn as nn
    from scripts.tt_hw_planner.module_tree import discover_components

    class SingleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

    comps = discover_components(SingleLinear())

    assert comps == [], f"single-leaf model should not produce module-tree components; " f"got {comps}"


def test_family_backend_has_use_module_tree_flag() -> None:
    """`FamilyBackend.use_module_tree` must exist as an opt-in flag
    (default False). Backends that set True use the module-tree
    decomposition; existing hand-written backends default to False
    (back-compat with the filename-grep path)."""
    from scripts.tt_hw_planner.family_backends import FamilyBackend

    backend = FamilyBackend(
        category="CNN",
        name="test",
        demo_path="x/y/z",
        routing_mode="template",
        canonical_hf_id=None,
    )
    assert hasattr(backend, "use_module_tree"), "FamilyBackend must expose use_module_tree as a field"
    assert backend.use_module_tree is False, "FamilyBackend.use_module_tree must default to False for back-compat"


def test_use_module_tree_only_on_targeted_backends() -> None:
    """A backend with `use_module_tree=True` must also be narrowly
    targeted (template routing with explicit `model_type_keys`), so
    its module-tree-derived components only apply to the intended
    model_types. The original invariant (no hand-written backend
    may set use_module_tree=True) was too strict: Qwen3-Embedding
    and other decoder-only-as-encoder backends legitimately need
    module-tree discovery + the reuse_registry. The actual hazard
    we're guarding against is a GENERIC (catch-all) backend
    accidentally flipping the flag, which would apply module-tree
    discovery to every model in the category."""
    from scripts.tt_hw_planner.family_backends import all_backends

    for b in all_backends():
        if not b.use_module_tree:
            continue
        assert b.routing_mode == "template", (
            f"backend {b.name!r} has use_module_tree=True but "
            f"routing_mode={b.routing_mode!r}; module-tree may only "
            f"be enabled on template-routing backends"
        )
        assert b.model_type_keys, (
            f"backend {b.name!r} has use_module_tree=True but no "
            f"model_type_keys; module-tree is too expensive to run "
            f"on every model in a category"
        )


def test_component_has_submodule_path_field() -> None:
    """`Component.submodule_path` must exist so the bring-up plan can
    carry the discovered HF module path forward to autofill / op-
    synth / PCC capture, bypassing COMPONENT_SUBMODULE_HINTS for
    module-tree-discovered components."""
    from scripts.tt_hw_planner.bringup_plan import Component

    c = Component(name="x", kind="x", status="NEW")
    assert hasattr(c, "submodule_path"), (
        "Component must expose submodule_path so the module-tree path " "can record the actual HF named_modules() path"
    )
    assert c.submodule_path is None, "default must be None"


def test_autofill_resolver_uses_discovered_submodule_path_first() -> None:
    """The autofill resolver `_resolve_torch_submodule_for_autofill`
    must take an optional `discovered_submodule_path` and try it
    FIRST before falling back to the legacy hint dict. This is the
    audit defect 3 fix -- without this, module-tree-discovered
    components would still be at the mercy of the static hint dict
    being right."""
    src = Path("scripts/tt_hw_planner/bringup_loop.py").resolve().read_text()
    fn_idx = src.find("def _resolve_torch_submodule_for_autofill(")
    assert fn_idx >= 0
    body = src[fn_idx : fn_idx + 3500]
    assert "discovered_submodule_path" in body, (
        "_resolve_torch_submodule_for_autofill must accept "
        "discovered_submodule_path so module-tree components bypass "
        "COMPONENT_SUBMODULE_HINTS"
    )

    discovered_idx = body.find("discovered_submodule_path")
    hints_idx = body.find("COMPONENT_SUBMODULE_HINTS.get(")
    assert discovered_idx < hints_idx, (
        "discovered_submodule_path must be checked BEFORE the legacy "
        "COMPONENT_SUBMODULE_HINTS dict so module-tree decomposition "
        "doesn't pay for the hint dict's inaccuracies"
    )


def test_pick_backend_with_quality_classifies_four_outcomes() -> None:
    """Loud-fallback foundation (2026-05-23 audit defect 1):
    `pick_backend_with_quality` MUST distinguish four cases so the CLI
    can refuse to silently scaffold against the wrong template.

      - "exact"            -- model_type matched a backend's keys
      - "pipeline"         -- pipeline_tag matched (model_type missed)
      - "category-default" -- both missed; closest-by-category was used
      - "none"             -- no backend at all (no category)

    The legacy `pick_backend()` function MUST still work and return
    just the backend (back-compat for existing callers)."""
    from scripts.tt_hw_planner.family_backends import (
        pick_backend,
        pick_backend_with_quality,
    )

    b, q = pick_backend_with_quality(category="CNN", model_type="sam2", pipeline_tag="image-segmentation")
    assert q == "exact", f"sam2 should be exact CNN match, got {q}"
    assert b is not None and "segformer" in b.name.lower()

    b, q = pick_backend_with_quality(
        category="CNN",
        model_type="brand_new_segmentation_model",
        pipeline_tag="image-segmentation",
    )
    assert q == "pipeline", f"unknown model_type + known pipeline_tag should be pipeline, got {q}"
    assert b is not None

    b, q = pick_backend_with_quality(
        category="CNN",
        model_type="totally_made_up_xyz",
        pipeline_tag="totally-made-up-tag",
    )
    assert q == "category-default", f"total miss should be category-default, got {q}"
    assert b is not None

    b, q = pick_backend_with_quality(
        category="ZZZ-fabricated-category-that-cannot-exist",
        model_type="vivit",
        pipeline_tag="video-classification",
    )
    assert q == "none", f"fabricated category should be 'none'; got {q}"
    assert b is None

    legacy = pick_backend(category="CNN", model_type="sam2", pipeline_tag="image-segmentation")

    new_b, _ = pick_backend_with_quality(category="CNN", model_type="sam2")
    assert legacy is new_b, "legacy pick_backend() must agree with pick_backend_with_quality"


def test_loud_fallback_gate_never_aborts_uses_auto_onboard_inline() -> None:
    """2026-05-23: per user requirement "any model from any category,
    zero user intervention", `_enforce_backend_match_quality_or_abort`
    no longer ABORTS the bring-up on category-default template
    fallbacks. Instead it:
      - PROCEEDs on quality=exact (silent success).
      - PROCEEDs on quality=pipeline (print INFO, proceed).
      - PROCEEDs on quality=category-default IF the backend is
        `routing_mode="generic"` (LLM/VLM catch-all by design).
      - On quality=category-default AND routing_mode="template",
        calls `_try_auto_onboard_inline` to LLM-draft a tailored
        backend; if that succeeds, proceeds with the EXACT-match
        new backend. If auto-onboard fails (no agent / validation
        errors), the gate STILL proceeds with the closest template
        + a loud warning -- never aborts.
      - On backend==None (no category match at all), also calls
        auto-onboard inline; if it fails, defers to cmd_up's
        ColdStartScaffoldError handler.
      - PROCEEDs if `accept_closest=True` regardless (explicit
        user override).
    """
    src = _planner_source()
    fn_idx = src.find("def _enforce_backend_match_quality_or_abort(")
    assert fn_idx >= 0, "_enforce_backend_match_quality_or_abort must be defined in cli.py"
    body = src[fn_idx : fn_idx + 60000]

    assert "pick_backend_with_quality(" in body, (
        "loud-fallback gate must call pick_backend_with_quality (the "
        "quality-aware variant), not the legacy single-return form"
    )

    for q in ('"exact"', '"pipeline"', '"category-default"'):
        assert q.strip('"') in body, f"loud-fallback gate must handle quality={q}"

    assert "routing_mode" in body and "generic" in body, (
        "loud-fallback gate must special-case generic-mode backends "
        "so the LLM/VLM tt_transformers catch-all doesn't get auto-"
        "onboarded"
    )

    assert "_try_auto_onboard_inline(" in body, (
        "category-default template branch must auto-invoke "
        "`_try_auto_onboard_inline` so the tool drafts a backend "
        "without asking the user for `--accept-closest-backend`"
    )

    assert "return 2" not in body, (
        "loud-fallback gate must not abort with exit 2; per user "
        "requirement (zero user intervention) it must always either "
        "auto-onboard or fall back to closest-template with a warning"
    )


def test_try_auto_onboard_inline_helper_exists_and_is_safe() -> None:
    """2026-05-23: `_try_auto_onboard_inline` must:
    - Be defined in cli.py.
    - Probe for an LLM agent (claude/codex) via `shutil.which`
      BEFORE calling auto_onboard, so we don't hang for 4 min
      on a missing binary.
    - Return None on ANY failure (no agent, draft validation
      errors, registry write fail, re-pick still category-
      default) instead of raising -- the caller relies on None
      meaning "try the closest-template fallback instead".
    - On success, return (backend, quality) re-picked from
      `pick_backend_with_quality`."""
    src = _planner_source()
    fn_idx = src.find("def _try_auto_onboard_inline(")
    assert fn_idx >= 0, "cli.py must define `_try_auto_onboard_inline` " "(the inline auto-onboard helper)"
    body = src[fn_idx : fn_idx + 6000]

    assert "which(" in body or "shutil.which" in body or "from shutil import which" in body, (
        "inline auto-onboard must probe for `claude`/`codex` via " "shutil.which before calling auto_onboard"
    )

    assert "auto_onboard(" in body, "must call `auto_onboard(model_id, ...)` to LLM-draft the entry"
    assert "write_backend_into_registry(" in body, (
        "must call `write_backend_into_registry(proposal)` to splice " "the drafted backend into family_backends.py"
    )

    assert "pick_backend_with_quality" in body, (
        "must re-pick the backend after writing it into the registry "
        "so the caller can use the newly-registered exact match"
    )


def test_cmd_up_calls_loud_fallback_gate() -> None:
    """Wiring test: `cmd_up` must invoke the loud-fallback gate after
    the memory-fit gate and before scaffold. Otherwise the helper
    exists but is dead code."""
    src = _planner_source()
    fn_idx = src.find("def cmd_up(")
    assert fn_idx >= 0
    block = src[fn_idx : fn_idx + 60000]
    assert "_enforce_backend_match_quality_or_abort(" in block, (
        "cmd_up must call _enforce_backend_match_quality_or_abort so "
        "silent-template-fallbacks are caught before scaffold"
    )

    assert "accept_closest_backend" in block, (
        "cmd_up must thread args.accept_closest_backend through to the " "loud-fallback gate"
    )


def test_up_has_accept_closest_backend_flag() -> None:
    """The `--accept-closest-backend` flag must be registered on `up`
    so users have a documented escape hatch when they've manually
    verified that the closest-by-category template backend is
    structurally similar enough to their new architecture."""
    src = _planner_source()
    assert '"--accept-closest-backend"' in src, "--accept-closest-backend must be registered on the `up` parser"


def test_memory_fit_gate_has_no_user_facing_opt_out() -> None:
    """The memory-fit gate is unconditional by design (2026-05-23
    user decision: an LLM cannot rewrite away a hardware memory
    budget, so there is no legitimate "let me try anyway" mode).

    This test guards against accidentally re-adding either
    `--allow-overcommit` (an override flag) or `--check-fit-only`
    (a dry-run flag). If a future workflow genuinely needs an
    escape hatch, the right move is to make the planner's memory
    model less conservative, not to add a bypass."""
    src = _planner_source()
    assert '"--allow-overcommit"' not in src, (
        "memory-fit gate must NOT have an --allow-overcommit override; " "if the planner is wrong, fix the planner"
    )
    assert '"--check-fit-only"' not in src, (
        "memory-fit gate must NOT expose a --check-fit-only dry-run "
        "flag; the gate prints its verdict as part of normal `up --auto` "
        "output, which is enough signal"
    )


def test_memory_fit_gate_skips_vision_models_without_memory_model() -> None:
    """Live behaviour test: a vision model whose probe yields
    `memory_model = None` (e.g. sam2-hiera-tiny) MUST return
    'unknown' from the gate, NOT 'no-fit'. Otherwise every vision
    bring-up would falsely abort before scaffold."""

    from types import SimpleNamespace
    from unittest.mock import patch

    fake_probe = SimpleNamespace(
        model_id="fake/vision-model",
        category="multimodal",
        saved_dtype="float32",
        memory_model=None,
    )
    with patch.object(cli, "probe_model", return_value=fake_probe):
        status, msg = cli._check_memory_fit_before_llm(
            "fake/vision-model",
            box_name="QB2",
            mesh_str="1x4",
            dtype_override=None,
        )
    assert status == "unknown", (
        f"vision models without a memory model must return 'unknown' " f"to bypass the gate; got {status} (msg={msg!r})"
    )
    assert "vision" in msg.lower() or "multi-modal" in msg.lower(), (
        "skip diagnostic should mention vision/multi-modal so users " "understand why the gate didn't fire"
    )


def _DELETED_test_decode_divergence_module_exports_new_symbols() -> None:
    """`activation_diff` must expose the new end-to-end decode-mode
    probe entry points so the cli can call them. A regression here
    silently disables the only signal the agent has for per-layer
    divergence on ALREADY-SUPPORTED model bring-ups (the legacy
    op-synth localizer doesn't run for those)."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    for sym in (
        "localize_decode_divergence",
        "format_decode_localization_hint_block",
        "DecodeLayerStats",
        "DecodeLocalizationResult",
    ):
        assert hasattr(mod, sym), f"activation_diff must expose {sym} for the decode-mode " f"per-layer probe"


def test_decode_divergence_blank_model_id_returns_none() -> None:
    """Blank model id is a configuration error -- the probe must
    return None (not raise) so the caller can fall back to the
    pre-probe prompt path. The auto-iter loop swallows None as a
    no-op."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    assert (
        mod.localize_decode_divergence(
            model_id="",
            prompt_text="anything",
        )
        is None
    )


def test_decode_divergence_format_returns_empty_on_none() -> None:
    """Mirroring the legacy formatter contract: None result = empty
    string, so `prompt = block + prompt` is a clean no-op."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    assert mod.format_decode_localization_hint_block("any/model", None) == ""


def test_decode_divergence_format_emits_skip_line_on_failure() -> None:
    """When the probe FAILS (timeout, model load error, missing
    decoder path) we still want a single audit line in the prompt so
    the next iteration's review can see that the probe was
    attempted. Empty result body but a real failure note triggers
    the skip-line path."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    failed = mod.DecodeLocalizationResult(
        hf_model_id="any/model",
        note="layers-not-resolved",
        prompt_text="anything",
    )
    rendered = mod.format_decode_localization_hint_block("any/model", failed)
    assert "DECODE-LAYER PROBE" in rendered, (
        "format_decode_localization_hint_block must surface failed "
        "probes as a one-line audit entry, not silently swallow them"
    )
    assert "layers-not-resolved" in rendered


def test_decode_divergence_format_renders_populated_table() -> None:
    """Full happy-path render: populated table -> block contains the
    reference banner, the (layer, decode_step) rows in execution
    order, and the FIRST-LAYER signal the LLM uses as a checklist."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")
    rows = [
        mod.DecodeLayerStats(
            layer_idx=li,
            decode_step=step,
            shape=(1, 1, 2560),
            dtype="torch.bfloat16",
            mean=0.01 * (li + 1),
            std=0.1 + 0.01 * li,
            l2=1.0 + 0.05 * li,
            abs_max=1.5 + 0.1 * li,
        )
        for step in (0, 10, 11, 15)
        for li in range(4)
    ]
    res = mod.DecodeLocalizationResult(
        table=rows,
        probed_decode_steps=[0, 10, 11, 15],
        hf_model_id="any/model",
        num_layers=4,
        decoder_path="model.layers",
        note="ok",
        prompt_text="how do I get to the moon",
        forward_succeeded=True,
        collapse_position=42,
        prefix_match_count=11,
    )
    rendered = mod.format_decode_localization_hint_block("any/model", res)
    assert "DECODE-LAYER ACTIVATION REFERENCE" in rendered
    assert "model.layers" in rendered
    assert "decode_step=0" in rendered
    assert "decode_step=11" in rendered

    assert "mean=" in rendered
    assert "std=" in rendered
    assert "l2=" in rendered
    assert "abs_max=" in rendered

    assert "collapse position" in rendered
    assert "42" in rendered


def test_resolve_decoder_layers_handles_unknown_models() -> None:
    """`_resolve_decoder_layers` MUST return (None, None) when the
    candidate paths don't match. This is the safety net for
    architectures we haven't enumerated yet -- without it the probe
    would raise AttributeError mid-loop."""
    mod = importlib.import_module("scripts.tt_hw_planner.activation_diff")

    class _Dummy:
        pass

    path, layers = mod._resolve_decoder_layers(_Dummy())
    assert path is None and layers is None


def test_cli_exposes_pcc_probe_layers_flag() -> None:
    """The cli must register `--pcc-probe` as a choice argument that
    accepts at least `none` and `layers`. A missing flag means
    --auto users have no way to opt into the new probe; a missing
    `layers` choice removes the only mode that fires the HF
    reference forward pass."""
    src = _planner_source()
    assert '"--pcc-probe"' in src, (
        "cli.py must register the --pcc-probe argparse flag for the " "decode-mode per-layer probe"
    )

    assert 'choices=("none", "layers")' in src, "--pcc-probe must accept the 'none' and 'layers' choices"


# NOTE (2026-05-31): pcc_repair.py was deleted; the two former
# `test_pcc_repair_loop_*` invariants — which pinned that the
# --pcc-probe flag was threaded through `_pcc_repair_loop(...)`
# call sites — have been removed with it. The --pcc-probe flag
# itself still exists and is consumed by _maybe_run_decode_layer_probe
# (covered by test_maybe_run_decode_layer_probe_*).


def test_maybe_run_decode_layer_probe_returns_empty_on_disabled_flag() -> None:
    """When `pcc_probe != "layers"`, the helper must short-circuit
    BEFORE doing any HF model load. We can't invoke the cli function
    directly under pytest collection (the cli module is heavy and
    has top-level torch imports), so we verify the structural
    guard via source inspection."""
    src = _planner_source()
    helper_anchor = src.find("def _maybe_run_decode_layer_probe(")
    assert helper_anchor != -1, (
        "cli.py must define _maybe_run_decode_layer_probe as the " "wrapper that decides whether to load the HF model"
    )

    body_end = src.find("\ndef ", helper_anchor + 1)
    body = src[helper_anchor : body_end if body_end != -1 else len(src)]

    guard_idx = body.find('pcc_probe != "layers"')
    return_idx = body.find('return ""')
    assert guard_idx != -1 and return_idx != -1 and return_idx > guard_idx, (
        "_maybe_run_decode_layer_probe must guard with "
        '`if pcc_probe != "layers": return ""` BEFORE any HF '
        "model load -- otherwise the probe pays a ~30s cost on "
        "every iteration even when --pcc-probe=none"
    )


def test_decode_layer_probe_cache_present_for_iteration_reuse() -> None:
    """The probe is cached by `(model_id, collapse_position)` so the
    HF forward isn't repeated on consecutive PCC-repair iterations
    that see the same collapse signature. A regression that drops the
    cache forces the HF model load on every iteration (~30s wasted
    per iter)."""
    src = _planner_source()
    assert "_DECODE_LAYER_PROBE_CACHE" in src, (
        "cli.py must define _DECODE_LAYER_PROBE_CACHE so consecutive "
        "PCC-repair iterations with the same collapse signature reuse "
        "the HF reference forward pass"
    )
