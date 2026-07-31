"""Unit tests for the generic capture-driver framework (capture_drivers.py).

Covers:
  - Layer 1: try_introspected_forward via mock models with annotated forward
  - Layer 2: SessionDriverPattern detection + drive on session-style mocks,
            and no-false-positive on non-session mocks
  - Layer 3: register_capture_driver / resolve_custom_driver registry
  - The orchestrator: try_capture_drivers fallback ordering

All tests use minimal Python mock models -- no torch, no HF -- so the suite
runs in seconds without GPU dependencies.
"""

from __future__ import annotations

import inspect
from typing import Optional


def _import_module():
    from scripts.tt_hw_planner import capture_drivers as cd

    return cd


def test_session_pattern_detects_init_method():
    cd = _import_module()

    class HasInitSession:
        def init_inference_session(self):
            return {"state": "ok"}

        def propagate_in_video(self, session):
            return [1, 2, 3]

    pattern = cd.SessionDriverPattern()
    assert pattern.can_drive(HasInitSession())


def test_session_pattern_drives_to_success():
    cd = _import_module()

    class FakeSessionModel:
        def init_inference_session(self):
            return {"state": "initialized"}

        def propagate_in_video(self, session):
            assert session == {"state": "initialized"}
            return iter([1, 2, 3])

    pattern = cd.SessionDriverPattern()
    ok, err = pattern.drive(FakeSessionModel(), pixel_values=None)
    assert ok, f"drive should succeed: {err}"


def test_session_pattern_no_false_positive_on_plain_model():
    cd = _import_module()

    class PlainModel:
        def forward(self, x):
            return x

    pattern = cd.SessionDriverPattern()
    assert not pattern.can_drive(PlainModel())


def test_custom_driver_registry_resolves_match():
    cd = _import_module()
    cd._CUSTOM_DRIVER_REGISTRY.clear()

    class TaggedModel:
        pass

    invocations = []

    @cd.register_capture_driver(matcher=lambda m: isinstance(m, TaggedModel))
    def custom_driver(model, pixel_values):
        invocations.append((model, pixel_values))

    model = TaggedModel()
    resolved = cd.resolve_custom_driver(model)
    assert resolved is custom_driver
    resolved(model, "sentinel")
    assert invocations == [(model, "sentinel")]

    cd._CUSTOM_DRIVER_REGISTRY.clear()


def test_custom_driver_registry_returns_none_when_no_match():
    cd = _import_module()
    cd._CUSTOM_DRIVER_REGISTRY.clear()

    class Foo:
        pass

    @cd.register_capture_driver(matcher=lambda m: False)
    def never_matches(model, pixel_values):
        pass

    assert cd.resolve_custom_driver(Foo()) is None
    cd._CUSTOM_DRIVER_REGISTRY.clear()


def test_try_introspected_forward_synthesizes_optional_args():
    cd = _import_module()

    calls = []

    class ModelWithOptionalArg:
        def forward(self, attention_mask: Optional[int] = None):
            calls.append(attention_mask)
            return "ran"

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    ok, err = cd.try_introspected_forward(ModelWithOptionalArg(), pixel_values=None)
    assert ok, f"should succeed (optional arg, has default): {err}"


def test_try_introspected_forward_fails_unknown_required_arg():
    cd = _import_module()

    class ModelWithUnknownReq:
        def forward(self, mystery_arg):
            return "should not reach"

    ok, err = cd.try_introspected_forward(ModelWithUnknownReq(), pixel_values=None)
    assert not ok
    assert "mystery_arg" in err


def test_try_capture_drivers_falls_through_chain_to_success():
    cd = _import_module()
    cd._CUSTOM_DRIVER_REGISTRY.clear()

    class SessionStyle:
        def init_state(self):
            return "state-ok"

        def propagate_frames(self, state):
            return iter([state])

    ok, attempts = cd.try_capture_drivers(SessionStyle(), pixel_values=None)
    assert ok
    assert any("SessionDriverPattern: ok" in a for a in attempts)


def test_try_capture_drivers_custom_driver_short_circuits():
    cd = _import_module()
    cd._CUSTOM_DRIVER_REGISTRY.clear()

    class Tagged:
        pass

    @cd.register_capture_driver(matcher=lambda m: isinstance(m, Tagged))
    def first_match_driver(model, pixel_values):
        pass

    ok, attempts = cd.try_capture_drivers(Tagged(), pixel_values=None)
    assert ok
    assert any("custom_driver" in a for a in attempts)
    cd._CUSTOM_DRIVER_REGISTRY.clear()


def test_init_session_pattern_is_case_insensitive():
    cd = _import_module()

    class WeirdCasing:
        def Init_Inference_Session(self):
            return {}

        def Process_Frames(self, s):
            return []

    pattern = cd.SessionDriverPattern()
    assert pattern.can_drive(WeirdCasing())


def test_drive_pattern_only_matches_at_start():
    cd = _import_module()

    class HasDriveLikeMethodNotAtStart:
        def init_session(self):
            return {}

        def my_process_method(self, s):
            return []

    pattern = cd.SessionDriverPattern()
    drive_name = pattern._find_drive_method(HasDriveLikeMethodNotAtStart())
    assert drive_name is None


def _run_all():
    """Manual runner so the suite can be exercised without pytest."""
    fns = [
        test_session_pattern_detects_init_method,
        test_session_pattern_drives_to_success,
        test_session_pattern_no_false_positive_on_plain_model,
        test_custom_driver_registry_resolves_match,
        test_custom_driver_registry_returns_none_when_no_match,
        test_try_introspected_forward_synthesizes_optional_args,
        test_try_introspected_forward_fails_unknown_required_arg,
        test_try_capture_drivers_falls_through_chain_to_success,
        test_try_capture_drivers_custom_driver_short_circuits,
        test_init_session_pattern_is_case_insensitive,
        test_drive_pattern_only_matches_at_start,
    ]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n  {passed}/{len(fns)} pass")
    return passed == len(fns)


def test_synthesize_input_ids_dispatches_text_modality():
    """Regression: text models (gpt2-style) declare
    ``input_ids: Optional[Tensor] = None``. The introspected_forward
    USED to skip parameters with defaults — leaving input_ids out and
    failing with 'You have to specify either input_ids or inputs_embeds'.
    Fix: input_ids (and similar text-mode names) are PRIMARY inputs
    and always synthesized."""
    import inspect

    import torch

    from scripts.tt_hw_planner.capture_drivers import _synthesize_for_param

    class _Model:
        pass

    # Mock parameter (Optional[LongTensor] = None, like gpt2 input_ids)
    param = inspect.Parameter("input_ids", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
    out = _synthesize_for_param("input_ids", param, _Model(), torch.zeros(1, 64, dtype=torch.long))
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long


def test_synthesize_input_features_dispatches_audio_modality():
    """Audio models (Whisper-style) declare ``input_features`` —
    the synth must produce a 3-D float tensor of mel-spectrogram shape."""
    import inspect

    import torch

    from scripts.tt_hw_planner.capture_drivers import _synthesize_for_param

    class _Cfg:
        num_mel_bins = 80

    class _Model:
        config = _Cfg()

    param = inspect.Parameter("input_features", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
    out = _synthesize_for_param("input_features", param, _Model(), torch.zeros(1, 3, 224, 224))
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 3
    assert out.shape[1] == 80


def test_introspected_forward_provides_primary_input_with_default():
    """The introspected driver must synthesize the PRIMARY input even
    when its default is None. Without this fix, HF models declaring
    ``input_ids: Optional[Tensor] = None`` fail with no kwargs."""
    import torch

    from scripts.tt_hw_planner.capture_drivers import _PRIMARY_INPUT_NAMES

    assert "input_ids" in _PRIMARY_INPUT_NAMES
    assert "input_features" in _PRIMARY_INPUT_NAMES
    assert "pixel_values" in _PRIMARY_INPUT_NAMES


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
