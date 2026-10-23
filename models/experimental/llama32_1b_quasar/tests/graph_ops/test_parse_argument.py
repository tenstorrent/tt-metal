# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ``generate_from_graph_capture.parse_argument`` — the consumer side of
``ttnn.graph._safe_arg_str``.

``tests/ttnn/unit_tests/base_functionality/test_graph_report.py`` covers the producer: what
``_safe_arg_str`` writes into a capture. These cover what this generator makes of it, which is
where a mismatch between the two actually costs coverage: an argument spelling the parser does not
recognize takes the whole call down with it (``{"k": "skip"}`` -> ``drop_unreconstructible``), so a
silent regression here shows up as ops quietly missing from a generated suite rather than as a
failure.

No device and no ttnn: the generator is pure stdlib, so these run in a bare checkout.

    pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_parse_argument.py
"""

import pytest

from models.experimental.llama32_1b_quasar.tests.graph_ops.generate_from_graph_capture import parse_argument

_INTERLEAVED_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _summary(shape="1, 2", dtype="BFLOAT16", layout="TILE", tensor_id=1, buffer_type="DRAM"):
    """One element as ``ttnn.graph._ttnn_tensor_summary`` writes it."""
    return (
        f"ttnn.Tensor(shape=Shape([{shape}]), dtype=DataType.{dtype}, layout=Layout.{layout}, "
        f"memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,"
        f"buffer_type=BufferType::{buffer_type},shard_spec=std::nullopt,nd_shard_spec=std::nullopt,"
        f"created_with_nd_shard_spec=0,per_core_allocation=0), storage_type=StorageType.DEVICE, "
        f"tensor_id={tensor_id}, is_allocated=True)"
    )


class TestTensorSequences:
    """The summarized-sequence spelling, and refusing anything that is not all of it."""

    def test_summary_list_keeps_every_element_and_its_memory_config(self):
        spec = parse_argument(f"[{_summary(tensor_id=1)}, {_summary(shape='3, 4', tensor_id=2)}]")

        assert spec["k"] == "tlist", spec
        assert [t["shape"] for t in spec["tensors"]] == [[1, 2], [3, 4]]
        # The memory config per element is the whole point of the summarized spelling: the legacy
        # one carried none, so list operands were rebuilt DRAM-interleaved whatever they had been.
        assert all(t["mem"] == _INTERLEAVED_DRAM for t in spec["tensors"]), spec

    def test_tuple_sequence_parses(self):
        spec = parse_argument(f"({_summary()}, {_summary()})")

        assert spec["k"] == "tlist" and len(spec["tensors"]) == 2, spec

    def test_legacy_value_dumping_spelling_still_parses(self):
        """Captures taken before _safe_arg_str summarized sequences must keep working."""
        element = "ttnn.Tensor([[1.0, 2.0]], shape=Shape([1, 2]), dtype=DataType::BFLOAT16, layout=Layout::TILE)"
        spec = parse_argument(f"[{element}, {element}]")

        assert spec["k"] == "tlist" and len(spec["tensors"]) == 2, spec
        assert all(t["mem"] is None for t in spec["tensors"]), "legacy elements carry no memory config"

    @pytest.mark.parametrize(
        "tail",
        ["... +8 more", "... 5 element(s) below the summary depth limit"],
        ids=["element-cap", "depth-cap"],
    )
    def test_elided_sequence_is_refused(self, tail):
        """A partially recorded sequence must not become a case with the operands that fit.

        _safe_arg_str summarizes at most _MAX_SEQUENCE_ELEMENTS entries; the per-element count
        cannot tell a complete list from a truncated one, so the marker is the only signal.
        """
        spec = parse_argument(f"[{_summary()}, {tail}]")

        assert spec["k"] == "skip", spec

    def test_nested_sequence_is_refused(self):
        """graph_case rebuilds a flat tlist, so a nested one would call the op differently."""
        spec = parse_argument(f"[[{_summary()}], [{_summary()}]]")

        assert spec["k"] == "skip", spec

    def test_unparseable_element_refuses_the_whole_sequence(self):
        """One element in a spelling we cannot read must not silently shrink the operand list."""
        spec = parse_argument(f"[{_summary()}, ttnn.Tensor(<unrecognized>)]")

        assert spec["k"] == "skip", spec


class TestScalarArguments:
    """The literal branches, including the bare-identifier one added for fused activations."""

    def test_bare_identifier_is_a_string(self):
        """ttnn.linear(activation="gelu") reaches the capture as `gelu`, unquoted."""
        assert parse_argument("gelu") == {"k": "lit", "v": "gelu"}

    @pytest.mark.parametrize("text", ["nan", "inf", "infinity", "-inf"])
    def test_float_spellings_stay_floats(self, text):
        """nan/inf are float() spellings literal_eval rejects; they must not become strings."""
        spec = parse_argument(text)

        assert spec["k"] == "lit", spec
        assert isinstance(spec["v"], float), f"{text} reconstructed as {type(spec['v']).__name__}"

    @pytest.mark.parametrize(
        "text, expected",
        [("True", True), ("False", False), ("None", None), ("37984", 37984), ("1e-05", 1e-05), ("(1, 1)", (1, 1))],
    )
    def test_literals_are_unaffected_by_the_string_branch(self, text, expected):
        assert parse_argument(text) == {"k": "lit", "v": expected}

    @pytest.mark.parametrize(
        "text",
        ["model_cache/Qwen/tok_embeddings.weight", "<WormholeComputeKernelConfig object at 0x7f00>"],
        ids=["path", "object-repr"],
    )
    def test_unreconstructible_scalars_are_refused(self, text):
        assert parse_argument(text)["k"] == "skip", text

    def test_object_repr_drops_the_heap_address(self):
        """Two identical calls from different layers must not look like different cases."""
        a = parse_argument("<WormholeComputeKernelConfig object at 0x7f0000000000>")
        b = parse_argument("<WormholeComputeKernelConfig object at 0x7f1111111111>")

        assert a == b, (a, b)


class TestTypedArguments:
    """Spellings the parser turns into ttnn objects rather than literals."""

    @pytest.mark.parametrize(
        "text, kind, value",
        [
            ("DataType.BFLOAT16", "dtype", "BFLOAT16"),
            ("Layout.TILE", "layout", "TILE"),
            ("[UnaryOpType.SILU]", "acts", ["SILU"]),
        ],
    )
    def test_enum_spellings(self, text, kind, value):
        spec = parse_argument(text)

        assert spec["k"] == kind, spec
        assert spec["v"] == value, spec

    def test_mesh_device(self):
        assert parse_argument("MeshDevice(1x1 grid, 1 devices)")["k"] == "device"
