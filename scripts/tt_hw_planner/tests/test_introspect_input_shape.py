"""Pin the 2026-06-04 Tier-1b introspection enhancements inside
``_PCC_TEST_TEMPLATE`` (the string template in ``bringup_loop.py`` that
gets emitted into every generated PCC test file).

The Tier-1b additions to ``_detect_hidden_shape`` bundle the property
probes that past hand-fixes added per-component (feature_projection,
variance_predictor, hifi_gan_residual_block). Tier-1b additions to
``_make_arg_for`` add introspection-based synthesis for REQUIRED args
not in the well-known names list (catches missing args like hifi_gan's
``input_embeds`` and code_hifi_gan's ``spkr_id``/``lang_id``).

Since these helpers live inside a string template (so they get emitted
into every generated test file), the tests verify the template TEXT
contains the right probe patterns — same convention as
``test_invariants.py`` uses for template assertions.
"""

from __future__ import annotations

from pathlib import Path


_BRINGUP_LOOP = Path(__file__).resolve().parent.parent / "bringup_loop.py"


def _source() -> str:
    return _BRINGUP_LOOP.read_text(encoding="utf-8")


def _template_body() -> str:
    """Extract the body of the _PCC_TEST_TEMPLATE string for inspection."""
    src = _source()
    start = src.find("_PCC_TEST_TEMPLATE = '''")
    assert start != -1, "_PCC_TEST_TEMPLATE not found in bringup_loop.py"
    body_start = src.index("'''", start) + 3
    body_end = src.index("'''", body_start)
    return src[body_start:body_end]


# ─── _detect_hidden_shape Tier-1b probes ─────────────────────────────


def test_template_has_detect_hidden_shape_function() -> None:
    """Baseline: the template defines _detect_hidden_shape."""
    body = _template_body()
    assert "def _detect_hidden_shape(torch_module, model=None):" in body


def test_detect_hidden_shape_probes_top_level_in_features() -> None:
    """Tier-1b: probe ``module.in_features`` directly (catches the case
    where the component IS a Linear, e.g. naked projection layer)."""
    body = _template_body()
    detect_start = body.find("def _detect_hidden_shape")
    detect_end = body.find("\ndef ", detect_start + 1)
    detect = body[detect_start:detect_end]
    assert 'hasattr(torch_module, "in_features")' in detect, "Tier-1b probe missing: top-level .in_features"


def test_detect_hidden_shape_probes_top_level_in_channels() -> None:
    """Tier-1b: probe ``module.in_channels`` directly (catches Conv components)."""
    body = _template_body()
    detect_start = body.find("def _detect_hidden_shape")
    detect_end = body.find("\ndef ", detect_start + 1)
    detect = body[detect_start:detect_end]
    assert 'hasattr(torch_module, "in_channels")' in detect, "Tier-1b probe missing: top-level .in_channels"


def test_detect_hidden_shape_probes_normalized_shape() -> None:
    """Tier-1b: probe ``module.normalized_shape`` (LayerNorm/RMSNorm)."""
    body = _template_body()
    detect_start = body.find("def _detect_hidden_shape")
    detect_end = body.find("\ndef ", detect_start + 1)
    detect = body[detect_start:detect_end]
    assert 'hasattr(torch_module, "normalized_shape")' in detect, "Tier-1b probe missing: .normalized_shape"


def test_detect_hidden_shape_probes_component_specific_names() -> None:
    """Tier-1b bundles past hand-fixes:
    - conv1.in_channels (variance_predictor)
    - projection.in_features (feature_projection)
    - layer_norm.normalized_shape (feature_projection)."""
    body = _template_body()
    detect_start = body.find("def _detect_hidden_shape")
    detect_end = body.find("\ndef ", detect_start + 1)
    detect = body[detect_start:detect_end]
    assert '"conv1"' in detect, "Tier-1b probe missing: conv1.in_channels (variance_predictor pattern)"
    assert '"projection"' in detect, "Tier-1b probe missing: projection.in_features (feature_projection pattern)"
    assert '"layer_norm"' in detect, "Tier-1b probe missing: layer_norm pattern"


def test_detect_hidden_shape_probes_modulelist_first() -> None:
    """Tier-1b: ModuleList container → probe module[0]'s submodules
    (hifi_gan_residual_block pattern with varying channel counts)."""
    body = _template_body()
    detect_start = body.find("def _detect_hidden_shape")
    detect_end = body.find("\ndef ", detect_start + 1)
    detect = body[detect_start:detect_end]
    assert "ModuleList" in detect, "Tier-1b probe missing: ModuleList handling"
    assert (
        "torch_module[0]" in detect or "first = torch_module" in detect
    ), "Tier-1b probe missing: ModuleList[0] indexing"


def test_detect_hidden_shape_legacy_probes_still_present() -> None:
    """Tier-1b additions must NOT remove existing probes (qkv, q_proj
    chain, weight.shape fallback, config.hidden_size). Pre-existing
    working models still need these."""
    body = _template_body()
    detect_start = body.find("def _detect_hidden_shape")
    detect_end = body.find("\ndef ", detect_start + 1)
    detect = body[detect_start:detect_end]
    # Legacy probes
    assert '"qkv"' in detect, "HIERA qkv probe removed (regression)"
    assert '"q_proj"' in detect, "Standard attention q_proj probe removed (regression)"
    assert "config" in detect and "hidden_size" in detect, "config.hidden_size fallback removed (regression)"


# ─── _make_arg_for Tier-1b required-arg introspection ────────────────


def test_template_has_make_arg_for_function() -> None:
    """Baseline: the template defines _make_arg_for."""
    body = _template_body()
    assert "def _make_arg_for(arg_name, *, model, torch_module):" in body


def test_make_arg_for_introspects_required_args_via_signature() -> None:
    """Tier-1b: when arg_name is not in well-known list, use
    inspect.signature(torch_module.forward) to determine if it's
    REQUIRED, and synthesize accordingly."""
    body = _template_body()
    make_arg_start = body.find("def _make_arg_for(arg_name")
    # End at the _Omit class (defined right after _make_arg_for)
    make_arg_end = body.find("class _Omit", make_arg_start)
    make_arg = body[make_arg_start:make_arg_end]

    assert "inspect" in make_arg, "Tier-1b: must import inspect for signature inspection"
    assert "signature" in make_arg, "Tier-1b: must call inspect.signature"
    assert "Parameter.empty" in make_arg, "Tier-1b: must check default is Parameter.empty (required arg test)"


def test_make_arg_for_synthesizes_int_ids_for_id_args() -> None:
    """Tier-1b: required args ending in _id / _ids / containing
    'spkr'/'lang' get torch.long tensors (catches code_hifi_gan's
    spkr_id, lang_id)."""
    body = _template_body()
    make_arg_start = body.find("def _make_arg_for(arg_name")
    make_arg_end = body.find("class _Omit", make_arg_start)
    make_arg = body[make_arg_start:make_arg_end]
    # The introspection branch must check for id-like names
    assert (
        "spkr" in make_arg or "lang" in make_arg or '"_id"' in make_arg or "endswith" in make_arg
    ), "Tier-1b: must detect ID-like arg names to generate int tensors"
    # And generate torch.long
    assert "torch.long" in make_arg, "Tier-1b: ID-arg branch must produce torch.long tensors"


def test_make_arg_for_synthesizes_tensor_for_unknown_required_args() -> None:
    """Tier-1b: required args not in well-known list get a tensor
    sized by _detect_hidden_shape (catches hifi_gan's input_embeds)."""
    body = _template_body()
    make_arg_start = body.find("def _make_arg_for(arg_name")
    make_arg_end = body.find("class _Omit", make_arg_start)
    make_arg = body[make_arg_start:make_arg_end]
    # Tier-1b's tensor-synth fallback uses _detect_hidden_shape
    introspect_block = make_arg[make_arg.find("inspect.signature") :] if "inspect.signature" in make_arg else ""
    assert (
        "_detect_hidden_shape" in introspect_block
    ), "Tier-1b: required-arg fallback must call _detect_hidden_shape for tensor synthesis"


def test_make_arg_for_introspection_is_best_effort() -> None:
    """Tier-1b introspection must be wrapped in try/except so a weird
    edge case (e.g., builtin without useful signature) doesn't break
    test emission entirely."""
    body = _template_body()
    make_arg_start = body.find("def _make_arg_for(arg_name")
    make_arg_end = body.find("class _Omit", make_arg_start)
    make_arg = body[make_arg_start:make_arg_end]
    # Must have try/except around the introspection block
    introspect_idx = make_arg.find("inspect.signature")
    if introspect_idx != -1:
        # Check there's a 'try:' before introspect.signature within the
        # same function — i.e., the introspection lives inside a try block.
        before_introspect = make_arg[:introspect_idx]
        assert "try:" in before_introspect, "Tier-1b introspection must be guarded by try/except"


def test_make_arg_for_returns_omit_for_optional_unknown() -> None:
    """Optional args (with defaults) not in well-known list should
    still fall through to _OMIT — let HF apply its own default."""
    body = _template_body()
    make_arg_start = body.find("def _make_arg_for(arg_name")
    make_arg_end = body.find("class _Omit", make_arg_start)
    make_arg = body[make_arg_start:make_arg_end]
    # The function still ends with `return _OMIT`
    assert "return _OMIT" in make_arg, "Tier-1b: _OMIT fallback must remain for non-required unknown args"
