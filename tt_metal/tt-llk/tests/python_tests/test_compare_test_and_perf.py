# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pairing and ignore-set checks for compare_test_and_perf.

These do not import SFPU test modules. The comparer itself loads those, and
that import needs a device stack this host does not have.
"""

from pathlib import Path

import compare_test_and_perf as cmp


def test_normalize_strips_prefix_and_quasar_suffix():
    assert (
        cmp.normalize("test_perf_eltwise_binary_sfpu_float_quasar")
        == "eltwise_binary_sfpu_float"
    )
    assert cmp.normalize("test_perf_eltwise_binary_sfpu_float") == (
        "eltwise_binary_sfpu_float"
    )
    assert cmp.normalize("test_eltwise_binary_sfpu_float") == (
        "eltwise_binary_sfpu_float"
    )


def test_ignore_set_includes_implied_math_format_separately_from_measurement():
    assert "implied_math_format" in cmp.IGNORED_AXES
    assert "implied_math_format" not in cmp.MEASUREMENT_AXES
    for axis in ("iterations", "loop_factor", "run_types", "is_perf"):
        assert axis in cmp.IGNORED_AXES
        assert axis in cmp.MEASUREMENT_AXES
    assert cmp.ignored_reason("loop_factor") == "ignored measurement axis"
    assert cmp.ignored_reason("implied_math_format") == "ignored"


def test_cross_arch_pairs_quasar_suffix(tmp_path: Path):
    (tmp_path / "perf_matmul.py").write_text("x = 1\n")
    (tmp_path / "perf_only_bh.py").write_text("x = 1\n")
    quasar = tmp_path / "quasar"
    quasar.mkdir()
    (quasar / "perf_matmul_quasar.py").write_text("x = 1\n")
    (quasar / "perf_only_qsr_quasar.py").write_text("x = 1\n")

    matched, left_only, right_only = cmp.discover_cross_arch(
        tmp_path, "blackhole", "quasar", "perf"
    )

    assert [(key, left.name, right.name) for key, left, right in matched] == [
        ("matmul", "perf_matmul.py", "perf_matmul_quasar.py")
    ]
    assert [path.name for path in left_only] == ["perf_only_bh.py"]
    assert [path.name for path in right_only] == ["perf_only_qsr_quasar.py"]


def test_cross_arch_same_tree_pairs_a_file_with_itself(tmp_path: Path):
    (tmp_path / "test_matmul.py").write_text("x = 1\n")
    (tmp_path / "perf_matmul.py").write_text("x = 1\n")

    matched, left_only, right_only = cmp.discover_cross_arch(
        tmp_path, "wormhole", "blackhole", "func"
    )

    assert len(matched) == 1
    assert matched[0][0] == "matmul"
    assert matched[0][1] == matched[0][2]
    assert left_only == []
    assert right_only == []


def test_cross_arch_verdict_uses_architecture_names():
    bucket, headline = cmp.verdict(
        "mathop",
        ["add", "mul"],
        ["add"],
        True,
        True,
        "axis",
        cmp.arch_sides("blackhole", "quasar"),
    )
    assert bucket == "diff"
    assert headline.startswith("[~] mathop: Q subset of B")

    _bucket, only = cmp.verdict(
        "bcast_dim",
        ["none"],
        [],
        True,
        False,
        "axis",
        cmp.arch_sides("blackhole", "quasar"),
    )
    assert only == "[B] bcast_dim: B-only axis"


def test_math_op_joins_mathop_without_renaming_when_both_sides_agree():
    left = cmp.Sweep(
        axes={"math_op": ["MathOperation.Elwadd", "MathOperation.Elwsub"]},
        rows=[{"math_op": "MathOperation.Elwadd"}],
        raw={"math_op": ["add", "sub"]},
        raw_rows=[{"math_op": "add"}],
    )
    right = cmp.Sweep(
        axes={"mathop": ["MathOperation.Elwadd"]},
        rows=[{"mathop": "MathOperation.Elwadd"}],
        raw={"mathop": ["add"]},
        raw_rows=[{"mathop": "add"}],
    )
    both = cmp.Sweep(
        axes={"math_op": ["MathOperation.Elwadd"]},
        rows=[{"math_op": "MathOperation.Elwadd"}],
        raw={"math_op": ["add"]},
        raw_rows=[{"math_op": "add"}],
    )
    other = cmp.Sweep(
        axes={"math_op": ["MathOperation.Elwsub"]},
        rows=[{"math_op": "MathOperation.Elwsub"}],
        raw={"math_op": ["sub"]},
        raw_rows=[{"math_op": "sub"}],
    )

    cmp.align_sweeps(left, right)
    cmp.align_sweeps(both, other)

    assert list(left.axes) == ["mathop"]
    assert left.rows[0]["mathop"] == "MathOperation.Elwadd"
    assert "math_op" in both.axes
    assert "mathop" not in both.axes


def test_dest_sync_dest_acc_splits_onto_the_other_sides_axes():
    import enum

    class DestSync(enum.Enum):
        Half = 1

    class DestAcc(enum.Enum):
        No = 0
        Yes = 1

    left = cmp.Sweep(
        axes={
            "dest_sync": ["DestSync.Half"],
            "dest_acc": ["DestAcc.No", "DestAcc.Yes"],
        },
        rows=[{"dest_sync": "DestSync.Half", "dest_acc": "DestAcc.No"}],
        raw={"dest_sync": [DestSync.Half], "dest_acc": [DestAcc.No, DestAcc.Yes]},
        raw_rows=[{"dest_sync": DestSync.Half, "dest_acc": DestAcc.No}],
    )
    right = cmp.Sweep(
        axes={"dest_sync_dest_acc": ["packed"], "enable_direct_indexing": ["False"]},
        rows=[
            {
                "dest_sync_dest_acc": "packed",
                "enable_direct_indexing": "False",
            }
        ],
        raw={
            "dest_sync_dest_acc": [
                (DestSync.Half, DestAcc.No),
                (DestSync.Half, DestAcc.Yes),
            ],
            "enable_direct_indexing": [False],
        },
        raw_rows=[
            {
                "dest_sync_dest_acc": (DestSync.Half, DestAcc.No),
                "enable_direct_indexing": False,
            }
        ],
    )

    cmp.align_sweeps(left, right)

    assert list(right.axes) == ["dest_sync", "dest_acc", "enable_direct_indexing"]
    assert right.axes["dest_acc"] == ["DestAcc.No", "DestAcc.Yes"]
    assert right.rows[0]["dest_sync"] == "DestSync.Half"
    assert "dest_sync_dest_acc" not in right.axes
    assert list(left.axes) == ["dest_sync", "dest_acc"]


def test_bundle_stays_packed_when_the_other_side_lacks_those_axes():
    import enum

    class DestSync(enum.Enum):
        Half = 1

    class DestAcc(enum.Enum):
        No = 0

    sweep = cmp.Sweep(
        axes={"dest_sync_dest_acc": ["packed"]},
        rows=[{"dest_sync_dest_acc": "packed"}],
        raw={"dest_sync_dest_acc": [(DestSync.Half, DestAcc.No)]},
        raw_rows=[{"dest_sync_dest_acc": (DestSync.Half, DestAcc.No)}],
    )
    other = cmp.Sweep(
        axes={"formats": ["Float16"]},
        rows=[{"formats": "Float16"}],
        raw={"formats": ["Float16"]},
        raw_rows=[{"formats": "Float16"}],
    )

    cmp.align_sweeps(sweep, other)

    assert list(sweep.axes) == ["dest_sync_dest_acc"]


def test_dest_reuse_exception_uses_the_separate_quasar_module():
    left = {"eltwise_binary": Path("perf_eltwise_binary.py")}
    right = {
        "eltwise_binary": Path("quasar/perf_eltwise_binary_quasar.py"),
        "eltwise_binary_reuse_dest": Path(
            "quasar/perf_eltwise_binary_reuse_dest_quasar.py"
        ),
    }

    extras, consumed = cmp.apply_cross_arch_exceptions(left, right)

    assert consumed == {"eltwise_binary_reuse_dest"}
    path, left_function, right_function = extras["eltwise_binary"][0]
    assert path == right["eltwise_binary_reuse_dest"]
    assert left_function == "eltwise_binary_dest_reuse"
    assert right_function == "eltwise_binary_reuse_dest"


def test_format_values_list_shared_entries_before_side_only_entries():
    left, right = cmp.order_format_values(
        [
            "DataFormat.Bfp4_b",
            "DataFormat.Float16",
            "DataFormat.Int8",
            "DataFormat.Bfp8_b",
            "DataFormat.Float16_b",
        ],
        [
            "DataFormat.MxFp4",
            "DataFormat.Int8",
            "DataFormat.Float16_b",
            "DataFormat.Float16",
        ],
    )
    assert left == [
        "DataFormat.Float16",
        "DataFormat.Float16_b",
        "DataFormat.Int8",
        "DataFormat.Bfp4_b",
        "DataFormat.Bfp8_b",
    ]
    assert right == [
        "DataFormat.Float16",
        "DataFormat.Float16_b",
        "DataFormat.Int8",
        "DataFormat.MxFp4",
    ]
    assert cmp._is_format_name("formats.input")
    assert cmp._is_format_name("format.output")
    assert cmp._is_format_name("InputOutputFormat.input_B")
    assert not cmp._is_format_name("mathop")


def test_same_arch_verdict_keeps_functional_perf_labels():
    _bucket, headline = cmp.verdict(
        "approx_mode", [], ["No"], False, True, "axis", cmp.FUNCTIONAL_PERF
    )
    assert headline == "[P] approx_mode: PERF-ONLY axis"
