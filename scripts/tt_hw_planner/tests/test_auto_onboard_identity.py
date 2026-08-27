"""A persisted backend entry must not carry machine-local identity.

Regression pin: a composite's component is addressed by directory path, and an
auto-onboard run wrote that absolute path into family_backends.py as the entry's
canonical_hf_id -- a value that names a directory on exactly one machine. The same
entry carried model_type_keys=[''], which would match every model whose config has
no model_type.
"""

from __future__ import annotations

from scripts.tt_hw_planner.auto_onboard import _validate_proposal


def _obj(**over):
    base = {
        "category": "Image",
        "name": "some unique backend name for tests",
        "demo_path": "models/demos/whatever",
        "routing_mode": "template",
        "canonical_hf_id": "some-org/some-model",
        "model_type_keys": ["some_key"],
        "use_module_tree": True,
    }
    base.update(over)
    return base


def test_absolute_path_is_not_persisted_as_a_model_identity(tmp_path) -> None:
    obj = _obj(canonical_hf_id=str(tmp_path))
    _validate_proposal(obj, new_model_type=None)
    assert obj["canonical_hf_id"] is None


def test_a_real_hf_id_is_kept() -> None:
    obj = _obj(canonical_hf_id="some-org/some-model")
    _validate_proposal(obj, new_model_type=None)
    assert obj["canonical_hf_id"] == "some-org/some-model"


def test_blank_model_type_key_is_rejected() -> None:
    """[''] would match every model whose config has no model_type."""
    errors = _validate_proposal(_obj(model_type_keys=[""]), new_model_type=None)
    assert any("model_type_keys" in e for e in errors)


def test_blank_keys_are_stripped_before_validation() -> None:
    obj = _obj(model_type_keys=["", "real_key", "  "])
    _validate_proposal(obj, new_model_type=None)
    assert obj["model_type_keys"] == ["real_key"]


def test_none_identity_is_allowed() -> None:
    obj = _obj(canonical_hf_id=None)
    _validate_proposal(obj, new_model_type=None)
    assert obj["canonical_hf_id"] is None
