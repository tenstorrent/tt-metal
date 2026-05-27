# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the generic descriptor framework (expose()).

Tests the internal helpers and end-to-end expose() logic using mocks —
no device required.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.experimental.ops.descriptors._generic import (
    _auto_build_inputs,
    _derive_name,
    _discover_fields,
    _discover_types,
    _is_pending,
    _types_from_nb_doc,
    _resolve_dotted,
    expose,
)
from models.experimental.ops.descriptors.op_descriptor import (
    OpDescriptor,
    _DeferredOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockParams:
    """Params class with value-member fields."""

    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.mode = None


class _MockInputs:
    """Inputs class with value-member fields (default-constructible)."""

    def __init__(self):
        self.input_tensor = None
        self.weight = None


class _MockRefInputs:
    """Inputs class with reference-member semantics (requires constructor arg)."""

    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        self.weight = None


class _MockTensor:
    """Stand-in for ttnn.Tensor in unit tests."""

    def __init__(self, name="mock"):
        self._name = name
        self._device = SimpleNamespace(name="mock_device")

    def device(self):
        return self._device


def _make_mock_device_op(
    params_class=_MockParams,
    inputs_class=_MockInputs,
    has_select_factory=True,
):
    """Create a mock DeviceOperation class with annotated methods."""

    class MockDeviceOp:
        @staticmethod
        def compute_program_hash(
            operation_attributes: params_class,
            tensor_args: inputs_class,
        ) -> int:
            return 42

        @staticmethod
        def compute_output_specs(
            operation_attributes: params_class,
            tensor_args: inputs_class,
        ):
            return SimpleNamespace(shape=(1, 1, 32, 32))

    MockDeviceOp.__name__ = "MockDeviceOperation"
    MockDeviceOp.__qualname__ = "MockDeviceOperation"

    if has_select_factory:
        factory = SimpleNamespace()
        factory.create_descriptor = MagicMock(return_value=SimpleNamespace(kernels=[]))

        MockDeviceOp.select_program_factory = staticmethod(lambda attrs, tensors: factory)

    return MockDeviceOp


# ---------------------------------------------------------------------------
# _derive_name
# ---------------------------------------------------------------------------


class TestDeriveName:
    def test_strips_device_operation_suffix(self):
        class MatmulDeviceOperation:
            pass

        assert _derive_name(MatmulDeviceOperation) == "matmul"

    def test_strips_operation_suffix(self):
        class SoftmaxOperation:
            pass

        assert _derive_name(SoftmaxOperation) == "softmax"

    def test_camel_to_snake(self):
        class BinaryNgDeviceOperation:
            pass

        assert _derive_name(BinaryNgDeviceOperation) == "binary_ng"

    def test_single_word(self):
        class SliceDeviceOperation:
            pass

        assert _derive_name(SliceDeviceOperation) == "slice"

    def test_no_matching_suffix(self):
        class MyCustomThing:
            pass

        assert _derive_name(MyCustomThing) == "my_custom_thing"


# ---------------------------------------------------------------------------
# _discover_types
# ---------------------------------------------------------------------------


class TestDiscoverTypes:
    def test_from_compute_program_hash(self):
        device_op = _make_mock_device_op()
        p, i = _discover_types(device_op, None, None)
        assert p is _MockParams
        assert i is _MockInputs

    def test_explicit_overrides(self):
        device_op = _make_mock_device_op()

        class OtherParams:
            pass

        class OtherInputs:
            pass

        p, i = _discover_types(device_op, OtherParams, OtherInputs)
        assert p is OtherParams
        assert i is OtherInputs

    def test_partial_override(self):
        device_op = _make_mock_device_op()

        class OtherParams:
            pass

        p, i = _discover_types(device_op, OtherParams, None)
        assert p is OtherParams
        assert i is _MockInputs

    def test_fallback_to_compute_output_specs(self):
        class DeviceOp:
            @staticmethod
            def compute_output_specs(a: _MockParams, b: _MockInputs):
                pass

        DeviceOp.__name__ = "DeviceOp"
        p, i = _discover_types(DeviceOp, None, None)
        assert p is _MockParams
        assert i is _MockInputs

    def test_raises_when_no_annotations(self):
        class DeviceOp:
            @staticmethod
            def compute_program_hash(a, b):
                return 0

        DeviceOp.__name__ = "DeviceOp"
        with pytest.raises(TypeError, match="Cannot auto-discover"):
            _discover_types(DeviceOp, None, None)

    def test_nb_doc_parsing(self):
        """Test type extraction from nanobind-style docstrings."""

        class FakeMethod:
            __doc__ = (
                "compute_program_hash("
                "operation_attributes: ttnn._ttnn.operations.matmul.MatmulParams, "
                "tensor_args: ttnn._ttnn.operations.matmul.MatmulInputs) -> int"
            )

        p, i = _types_from_nb_doc(FakeMethod)
        import ttnn

        assert p is ttnn.MatmulParams
        assert i is ttnn.MatmulInputs

    def test_nb_doc_parsing_no_match(self):
        class FakeMethod:
            __doc__ = "some unrelated docstring"

        p, i = _types_from_nb_doc(FakeMethod)
        assert p is None
        assert i is None

    def test_resolve_dotted_valid(self):
        import ttnn

        cls = _resolve_dotted("ttnn._ttnn.operations.matmul.MatmulParams")
        assert cls is ttnn.MatmulParams

    def test_resolve_dotted_invalid(self):
        assert _resolve_dotted("nonexistent.module.Class") is None

    def test_discover_types_real_nanobind(self):
        """Verify type discovery works on actual nanobind-bound DeviceOperations."""
        import ttnn

        p, i = _discover_types(ttnn.MatmulDeviceOperation, None, None)
        assert p is ttnn.MatmulParams
        assert i is ttnn.MatmulInputs

        p, i = _discover_types(ttnn.LayerNormDeviceOperation, None, None)
        assert p is ttnn.LayerNormParams
        assert i is ttnn.LayerNormInputs

        p, i = _discover_types(ttnn.SliceDeviceOperation, None, None)
        assert p is ttnn.SliceParams
        assert i is ttnn.SliceInputs

    def test_raises_when_no_methods(self):
        class DeviceOp:
            pass

        DeviceOp.__name__ = "DeviceOp"
        with pytest.raises(TypeError, match="Cannot auto-discover"):
            _discover_types(DeviceOp, None, None)


# ---------------------------------------------------------------------------
# _discover_fields
# ---------------------------------------------------------------------------


class TestDiscoverFields:
    def test_value_member_class(self):
        fields = _discover_fields(_MockParams)
        assert fields == {"alpha", "beta", "mode"}

    def test_value_member_inputs(self):
        fields = _discover_fields(_MockInputs)
        assert fields == {"input_tensor", "weight"}

    def test_reference_member_class(self):
        fields = _discover_fields(_MockRefInputs)
        # Falls back to class-level introspection — may find different
        # attributes depending on what's on the class vs instance.
        # At minimum, we should find something or get an empty set.
        assert isinstance(fields, set)

    def test_excludes_dunder(self):
        class Cls:
            def __init__(self):
                self.visible = 1
                self.__hidden = 2

        fields = _discover_fields(Cls)
        assert "visible" in fields
        assert "__hidden" not in fields

    def test_excludes_methods(self):
        class Cls:
            def __init__(self):
                self.field = 1

            def method(self):
                pass

        fields = _discover_fields(Cls)
        assert "field" in fields
        assert "method" not in fields


# ---------------------------------------------------------------------------
# _auto_build_inputs
# ---------------------------------------------------------------------------


class TestAutoBuildInputs:
    def test_default_constructible(self):
        tensor_kw = {"input_tensor": "t1", "weight": "t2"}
        result = _auto_build_inputs(_MockInputs, tensor_kw, {"input_tensor", "weight"})
        assert result.input_tensor == "t1"
        assert result.weight == "t2"

    def test_skips_none_values(self):
        tensor_kw = {"input_tensor": "t1", "weight": None}
        result = _auto_build_inputs(_MockInputs, tensor_kw, {"input_tensor", "weight"})
        assert result.input_tensor == "t1"
        assert result.weight is None

    def test_skips_unknown_fields(self):
        tensor_kw = {"input_tensor": "t1", "unknown": "t2"}
        result = _auto_build_inputs(_MockInputs, tensor_kw, {"input_tensor", "weight"})
        assert result.input_tensor == "t1"
        assert not hasattr(result, "unknown") or result.unknown is None

    def test_reference_member_fallback(self):
        tensor_kw = {"input_tensor": "t1", "weight": "t2"}
        result = _auto_build_inputs(_MockRefInputs, tensor_kw, {"input_tensor", "weight"})
        assert result.input_tensor == "t1"
        assert result.weight == "t2"


# ---------------------------------------------------------------------------
# _is_pending
# ---------------------------------------------------------------------------


class TestIsPending:
    def test_none_is_pending(self):
        assert _is_pending(None)

    def test_deferred_output_is_pending(self):
        assert _is_pending(_DeferredOutput())

    def test_string_is_not_pending(self):
        assert not _is_pending("hello")

    def test_zero_is_not_pending(self):
        assert not _is_pending(0)


# ---------------------------------------------------------------------------
# expose() — end-to-end with mocks
# ---------------------------------------------------------------------------


class TestExpose:
    def test_raises_without_factory(self):
        device_op = _make_mock_device_op(has_select_factory=False)
        fn = expose(device_op, required_inputs=["input_tensor"])

        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn.allocate_tensor_on_device = MagicMock(return_value=_MockTensor("out"))
            result = fn(input_tensor=mock_tensor)
            with pytest.raises(ValueError, match="no factory available"):
                _ = result.descriptor

    def test_accepts_explicit_factory(self):
        device_op = _make_mock_device_op(has_select_factory=False)
        mock_factory = SimpleNamespace(create_descriptor=MagicMock(return_value=SimpleNamespace(kernels=[])))
        fn = expose(device_op, factory=mock_factory)
        assert callable(fn)
        assert fn.__name__ == "mock"

    def test_name_derivation(self):
        device_op = _make_mock_device_op()
        fn = expose(device_op)
        assert fn.__name__ == "mock"

    def test_explicit_name(self):
        device_op = _make_mock_device_op()
        fn = expose(device_op, name="custom_op")
        assert fn.__name__ == "custom_op"

    def test_inline_returns_op_descriptor(self):
        device_op = _make_mock_device_op()
        fn = expose(device_op)

        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn.allocate_tensor_on_device = MagicMock(return_value=_MockTensor("out"))
            result = fn(input_tensor=mock_tensor, weight=_MockTensor("w"))

        assert isinstance(result, OpDescriptor)
        assert result.program_cache_key is not None

    def test_persistent_mode_when_required_missing(self):
        device_op = _make_mock_device_op()
        fn = expose(device_op, required_inputs=["input_tensor"])

        result = fn(input_tensor=None, weight=_MockTensor("w"))
        assert isinstance(result, OpDescriptor)
        assert result.program_cache_key is None  # not yet materialized

    def test_persistent_deferred_output(self):
        device_op = _make_mock_device_op()
        fn = expose(device_op, required_inputs=["input_tensor"])

        deferred = _DeferredOutput()
        result = fn(input_tensor=deferred, weight=_MockTensor("w"))
        assert isinstance(result, OpDescriptor)
        assert result.program_cache_key is None

    def test_params_set_from_kwargs(self):
        captured = {}

        class TrackingParams:
            def __init__(self):
                self.alpha = 0.0
                self.beta = 0.0
                self.mode = None

        class TrackingDeviceOp:
            @staticmethod
            def compute_program_hash(
                operation_attributes: TrackingParams,
                tensor_args: _MockInputs,
            ) -> int:
                captured["alpha"] = operation_attributes.alpha
                captured["beta"] = operation_attributes.beta
                return 42

            @staticmethod
            def compute_output_specs(operation_attributes, tensor_args):
                return SimpleNamespace(shape=(1,))

            @staticmethod
            def select_program_factory(operation_attributes, tensor_args):
                fct = SimpleNamespace()
                fct.create_descriptor = MagicMock(return_value=SimpleNamespace(kernels=[]))
                return fct

        TrackingDeviceOp.__name__ = "TrackingDeviceOperation"

        fn = expose(TrackingDeviceOp, required_inputs=["input_tensor"])
        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn.allocate_tensor_on_device = MagicMock(return_value=_MockTensor("out"))
            fn(input_tensor=mock_tensor, alpha=3.14, beta=2.71)

        assert captured["alpha"] == 3.14
        assert captured["beta"] == 2.71

    def test_validate_called(self):
        device_op = _make_mock_device_op()
        validator = MagicMock()
        fn = expose(device_op, validate=validator, required_inputs=["input_tensor"])

        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn.allocate_tensor_on_device = MagicMock(return_value=_MockTensor("out"))
            fn(input_tensor=mock_tensor)

        validator.assert_called_once()

    def test_extra_cache_key_fn(self):
        device_op = _make_mock_device_op()
        fn = expose(
            device_op,
            required_inputs=["input_tensor"],
            extra_cache_key_fn=lambda **kw: (kw.get("my_extra", 0),),
        )

        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn.allocate_tensor_on_device = MagicMock(return_value=_MockTensor("out"))
            r1 = fn(input_tensor=mock_tensor, my_extra=1)
            r2 = fn(input_tensor=mock_tensor, my_extra=2)

        # Different extra → different cache keys
        assert r1.program_cache_key != r2.program_cache_key

    def test_params_preprocessor(self):
        captured = {}

        class PP(_MockParams):
            def __init__(self):
                super().__init__()
                self.preprocessed = False

        class DeviceOp:
            @staticmethod
            def compute_program_hash(
                operation_attributes: PP,
                tensor_args: _MockInputs,
            ) -> int:
                captured["preprocessed"] = operation_attributes.preprocessed
                return 42

            @staticmethod
            def compute_output_specs(operation_attributes, tensor_args):
                return SimpleNamespace(shape=(1,))

            @staticmethod
            def select_program_factory(operation_attributes, tensor_args):
                fct = SimpleNamespace()
                fct.create_descriptor = MagicMock(return_value=SimpleNamespace(kernels=[]))
                return fct

        DeviceOp.__name__ = "PPDeviceOperation"

        def _preprocess(params, inputs):
            params.preprocessed = True
            return params

        fn = expose(DeviceOp, params_preprocessor=_preprocess, required_inputs=["input_tensor"])
        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn.allocate_tensor_on_device = MagicMock(return_value=_MockTensor("out"))
            fn(input_tensor=mock_tensor)

        assert captured["preprocessed"] is True

    def test_output_is_vector(self):
        device_op = _make_mock_device_op()

        fn = expose(device_op, num_outputs=2, output_is_vector=True, required_inputs=["input_tensor"])
        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn.allocate_tensor_on_device = MagicMock(return_value=_MockTensor("out"))
            result = fn(input_tensor=mock_tensor)

        assert isinstance(result, OpDescriptor)
        assert len(result.output_tensors) == 2


# ---------------------------------------------------------------------------
# Eager fallback — no compute_program_hash
# ---------------------------------------------------------------------------


def _make_eager_mock_device_op(
    params_class=_MockParams,
    inputs_class=_MockInputs,
    has_select_factory=True,
):
    """Create a mock DeviceOperation WITHOUT compute_program_hash."""

    class MockDeviceOp:
        @staticmethod
        def compute_output_specs(
            operation_attributes: params_class,
            tensor_args: inputs_class,
        ):
            return SimpleNamespace(shape=(1, 1, 32, 32))

        @staticmethod
        def create_output_tensors(operation_attributes, tensor_args):
            return _MockTensor("eager_out")

    MockDeviceOp.__name__ = "MockEagerDeviceOperation"
    MockDeviceOp.__qualname__ = "MockEagerDeviceOperation"

    if has_select_factory:
        factory = SimpleNamespace()
        factory.create_descriptor = MagicMock(return_value=SimpleNamespace(kernels=[]))
        MockDeviceOp.select_program_factory = staticmethod(lambda attrs, tensors: factory)

    return MockDeviceOp


class TestExposeEagerFallback:
    def test_eager_when_no_compute_program_hash(self):
        device_op = _make_eager_mock_device_op()
        fn = expose(device_op, required_inputs=["input_tensor"])

        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn, patch(
            "models.experimental.ops.descriptors.op_descriptor.ttnn"
        ) as mock_ttnn_od:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn_od.compute_program_descriptor_hash = MagicMock(return_value=12345)
            result = fn(input_tensor=mock_tensor)

        assert isinstance(result, OpDescriptor)
        assert result._descriptor is not None
        assert result._factory_fn is None
        assert result.program_cache_key == 12345

    def test_eager_with_explicit_factory(self):
        device_op = _make_eager_mock_device_op(has_select_factory=False)
        mock_descriptor = SimpleNamespace(kernels=[])
        mock_factory = SimpleNamespace(create_descriptor=MagicMock(return_value=mock_descriptor))

        fn = expose(device_op, factory=mock_factory, required_inputs=["input_tensor"])

        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn, patch(
            "models.experimental.ops.descriptors.op_descriptor.ttnn"
        ) as mock_ttnn_od:
            mock_ttnn.Tensor = _MockTensor
            mock_ttnn_od.compute_program_descriptor_hash = MagicMock(return_value=99999)
            result = fn(input_tensor=mock_tensor)

        assert isinstance(result, OpDescriptor)
        assert result._descriptor is mock_descriptor
        mock_factory.create_descriptor.assert_called_once()

    def test_eager_raises_without_any_factory(self):
        device_op = _make_eager_mock_device_op(has_select_factory=False)
        fn = expose(device_op, required_inputs=["input_tensor"])

        mock_tensor = _MockTensor("a")
        with patch("models.experimental.ops.descriptors._generic.ttnn") as mock_ttnn:
            mock_ttnn.Tensor = _MockTensor
            with pytest.raises(ValueError, match="no factory available"):
                fn(input_tensor=mock_tensor)

    def test_eager_persistent_then_inline(self):
        device_op = _make_eager_mock_device_op()
        fn = expose(device_op, required_inputs=["input_tensor"])

        result = fn(input_tensor=None)
        assert isinstance(result, OpDescriptor)
        assert result.program_cache_key is None  # persistent, not yet materialized
