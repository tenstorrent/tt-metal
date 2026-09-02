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

    # The signature read itself lives in `_forward_param` so the "has a default" and "is required"
    # questions cannot drift apart; `_make_arg_for` must go through it rather than re-reading.
    assert "_forward_param(" in make_arg, "Tier-1b: must consult the forward signature helper"
    assert "def _forward_param(" in body, "Tier-1b: the signature helper must be defined in the test"
    assert "signature" in body[: body.find("def _make_arg_for(arg_name")], "helper must read the signature"
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
    # Tier-1b's tensor-synth fallback uses _detect_hidden_shape. Anchored on the required-arg
    # branch (`param.kind`) rather than the signature call, which now lives in `_forward_param`.
    assert "param.kind" in make_arg, "Tier-1b: must branch on the parameter kind for required args"
    introspect_block = make_arg[make_arg.find("param.kind") :]
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
    # Must have try/except around the introspection block. Asserted unconditionally: this used to
    # be skipped when the anchor was missing, so a refactor that moved the block turned the test
    # green by finding nothing rather than by the guard still being there.
    introspect_idx = make_arg.find("param.kind")
    assert introspect_idx != -1, "Tier-1b: required-arg introspection branch must exist"
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


def test_defaulted_args_are_left_to_the_module_not_a_typed_list() -> None:
    """Args the module gives a default to must be omitted, decided from the signature.

    This was a literal tuple of names -- past_key_values, cache_position, use_cache, return_dict,
    head_mask, encoder_hidden_states, encoder_attention_mask, labels -- forced to None. A typed
    list only covers the models whoever wrote it had in mind: a model that spells its cache
    differently got a value forced on an arg it would have defaulted better itself, and nothing
    said so. Reading the default off the signature covers those names and survives a rename.
    """
    body = _template_body()
    make_arg_start = body.find("def _make_arg_for(arg_name")
    make_arg_end = body.find("class _Omit", make_arg_start)
    make_arg = body[make_arg_start:make_arg_end]

    # The decision is made from the signature, before the required-arg synthesis branch.
    assert "param.default is not inspect.Parameter.empty" in make_arg, "must omit args that have defaults"
    omit_idx = make_arg.find("param.default is not inspect.Parameter.empty")
    assert omit_idx < make_arg.find("param.kind"), "the default check must come before synthesis"

    # None of the old names may be back as a forced-None branch.
    for name in (
        "past_key_values",
        "cache_position",
        "use_cache",
        "return_dict",
        "head_mask",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "labels",
    ):
        assert f'"{name}"' not in make_arg, f"{name} is decided by the signature now, not by name"


def test_argument_type_comes_from_what_the_module_declares() -> None:
    """The module's own annotation decides the dtype, before any guess from the argument's name.

    Guessing from the name ("ends with _id, so integer") only worked for the spellings someone had
    already met: a required integer argument named anything else got a float tensor and the module
    raised on it. Modules annotate their forward args, so for most of them this is stated outright.
    """
    body = _template_body()
    assert "def _declared_dtype(param):" in body, "the test must be able to read declared types"

    make_arg_start = body.find("def _make_arg_for(arg_name")
    make_arg = body[make_arg_start : body.find("class _Omit", make_arg_start)]
    declared_idx = make_arg.find("_declared_dtype(param)")
    name_guess_idx = make_arg.find('_lc.endswith("_id")')
    assert declared_idx != -1, "_make_arg_for must consult the declared type"
    assert name_guess_idx != -1, "the name heuristic stays for modules that annotate nothing"
    assert declared_idx < name_guess_idx, "what the module declares must win over guessing by name"


def test_declared_dtype_reads_annotations_rather_than_a_name_table() -> None:
    """Exercised for real: the helper is pulled out of the template and run."""
    import inspect as _inspect

    import torch

    body = _template_body()
    start = body.find("def _declared_dtype(param):")
    end = body.find("\ndef ", start + 1)
    ns = {"inspect": _inspect, "torch": torch}
    exec(body[start:end], ns)  # noqa: S102 -- this source is what lands in the generated test
    declared = ns["_declared_dtype"]

    def _param(annotation):
        return _inspect.Parameter("x", _inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)

    assert declared(_param("Optional[torch.LongTensor]")) is torch.long
    assert declared(_param("bool")) is torch.bool
    assert declared(_param("torch.FloatTensor")) is None, "floats fall through to shape detection"
    assert declared(_param(_inspect.Parameter.empty)) is None, "no annotation means no claim"
    assert declared(None) is None
