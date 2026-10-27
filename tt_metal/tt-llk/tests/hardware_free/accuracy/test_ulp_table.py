# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_table.py``: the budget table as typed rows, the
one serializer that writes it, and the measurement records."""

import json
import math

import pytest
import yaml
from conftest import TANH_DOMAINS, problem_id, refused
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.ulp_domains import Domain, Problem
from helpers.ulp_table import (
    TABLE_PATH,
    AccuracyContract,
    BudgetKey,
    Measurement,
    Metric,
    Reading,
    Row,
    dump_table,
    from_plain,
    load_table,
    read_table,
    write_table_file,
)

INF = math.inf
F16B, F16, F32 = DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32


@pytest.mark.parametrize(
    "fields, problem",
    [
        (dict(output=F16B), Problem.MISSING_FIGURE),
        (dict(measured=-1), Problem.BAD_NUMBER),
        (dict(measured=True), Problem.BAD_NUMBER),
        (dict(metric=Metric.TOLERANCE, nonfinite=0), Problem.BAD_NUMBER),
        (dict(measured=1, nonfinite=3), Problem.NONFINITE_ON_BUDGET),
        (dict(measured=1, atol=0.1), Problem.MIXED_METRIC),
        (dict(metric=Metric.TOLERANCE, near_zero_atol=0.5), Problem.MIXED_METRIC),
        (dict(input=DataFormat.Bfp8_b, measured=1), Problem.NOT_A_COMPUTE_FORMAT),
        (
            dict(input=F16B, output=F16B, measured=10, domains=TANH_DOMAINS),
            Problem.BESIDE_DOMAINS,
        ),
        (dict(output=F16B, domains=TANH_DOMAINS), Problem.UNPINNED_DOMAINS),
        (
            dict(input=F16B, output=F16B, domains=TANH_DOMAINS[:1]),
            Problem.NOT_COVERING,
        ),
    ],
    ids=problem_id,
)
def test_a_row_the_table_cannot_hold_is_refused(fields, problem):
    with refused(problem):
        Row(**fields)


def test_a_rows_contract_is_its_figure_or_what_its_list_implies():
    assert Row(output=F16B, measured=2).contract == AccuracyContract(max_ulp=2)
    tolerance = Row(metric=Metric.TOLERANCE, measured=300, atol=0.1).contract
    assert tolerance == AccuracyContract(metric=Metric.TOLERANCE, atol=0.1)
    listed = Row(input=F16B, output=F16B, domains=TANH_DOMAINS).contract
    assert (listed.metric, listed.max_ulp) == (Metric.TOLERANCE, None)  # 10 > bf16's 6
    fits = Row(
        input=F16B, output=F32, near_zero_atol=1e-7, domains=TANH_DOMAINS
    ).contract
    assert (fits.metric, fits.max_ulp, fits.near_zero_atol) == (Metric.ULP, 10, 1e-7)
    assert Row(output=F16B, measured=2).key == BudgetKey(output_format=F16B)


@pytest.mark.parametrize(
    "plain, problem",
    [
        ({"measured": 1, "colour": "red"}, Problem.UNKNOWN_FIELD),
        ({"output": "Float17", "measured": 1}, Problem.NOT_A_MEMBER),
        ({"approx": True, "measured": 1}, Problem.NOT_A_MEMBER),
        ({"measured": 1.5}, Problem.BAD_NUMBER),
        ({"domains": {"low": 0}}, Problem.WRONG_SHAPE),
        (["measured", 1], Problem.WRONG_SHAPE),
    ],
    ids=[
        "unknown-field",
        "unknown-format",
        "bare-yes",
        "fraction",
        "list-shape",
        "row-shape",
    ],
)
def test_a_plain_value_that_does_not_fit_its_field_is_refused(plain, problem):
    with refused(problem):
        from_plain(Row, plain)


def test_a_record_missing_a_required_field_is_refused():
    """A dataclass built without a required field raises ``TypeError``, which is not a
    refusal; the reader names the fields instead."""
    with refused(Problem.MISSING_FIELD):
        from_plain(Measurement, {"op": "Abs", "key": {}})
    with refused(Problem.MISSING_FIELD):
        from_plain(Domain, {"low": 0})


@pytest.mark.parametrize("entry", [[151, 128, 128], [151, 128, 128, 3, 0]])
def test_a_bins_entry_of_the_wrong_length_is_refused(entry):
    """`bins` is `(binade, present, ranked, worst)`: a fixed shape, so three or five
    values are refused rather than read as a variadic tuple."""
    record = {"op": "Abs", "key": {}, "worst": 1, "lanes": 1, "bins": [entry]}
    with refused(Problem.WRONG_SHAPE):
        from_plain(Measurement, record)


@pytest.mark.parametrize(
    "data, problem",
    [
        ({"Nope": [{"measured": 1}]}, Problem.UNKNOWN_OP),
        ({"Abs": []}, Problem.NO_ROWS),
        ({"Abs": [{"output": "Float16_b", "measured": 1}] * 2}, Problem.REPEATED_KEY),
    ],
    ids=problem_id,
)
def test_a_table_the_registry_cannot_load_is_refused(data, problem):
    with refused(problem):
        read_table(yaml.safe_dump(data))


@pytest.mark.parametrize("text", ["Abs: [", "Abs:\n- &a {measured: 1}\n- {<<: *a}\n"])
def test_text_that_is_not_the_tables_yaml_is_refused(text):
    """A syntax error, and a merge key (the one writer never produces one)."""
    with refused(Problem.BAD_YAML):
        read_table(text)


def test_a_repeated_yaml_key_is_refused_not_overwritten():
    """PyYAML keeps the last of two equal keys, which would drop a whole op."""
    text = yaml.safe_dump({"Abs": [{"measured": 1}]}) * 2
    with refused(Problem.DUPLICATE_YAML_KEY):
        read_table(text)


@pytest.mark.parametrize(
    "text",
    ["Abs:\n- {{measured: 1}}\n", "Abs: &rows\n- *rows\n"],
    ids=["doubled-braces", "self-containing-anchor"],
)
def test_a_row_that_is_not_a_mapping_of_names_is_refused_not_a_type_error(text):
    """A mapping used as a key (doubled braces pasted from an f-string), or a row that
    is the op's own row list (an anchor that contains itself), raised a bare
    ``TypeError`` from the loader, past the guard's "cannot load" handling."""
    with refused(Problem.WRONG_SHAPE):
        read_table(text)


def test_a_typed_table_round_trips_through_the_one_serializer():
    table = {
        MathOperation.Tanh: (
            Row(
                input=F16B,
                output=F16B,
                approx=ApproximationMode.Yes,
                dest=DestAccumulation.No,
                near_zero_atol=2.4e-7,
                domains=TANH_DOMAINS,
            ),
            Row(
                output=F32, metric=Metric.TOLERANCE, measured=9, nonfinite=2, note="#1"
            ),
            Row(input=F16, arch=ChipArchitecture.BLACKHOLE, measured=4),
        ),
        MathOperation.Abs: (Row(measured=0),),
    }
    back = read_table(dump_table(table))
    assert {op: set(rows) for op, rows in back.items()} == {
        op: set(rows) for op, rows in table.items()
    }


def test_a_table_the_loader_would_refuse_is_never_written(tmp_path):
    """`write_table_file` reads its dump back before writing, so a refused rewrite --
    two rows of one op on one key -- leaves the file as it was."""
    path = tmp_path / "budget.yaml"
    before = dump_table({MathOperation.Abs: (Row(output=F16B, measured=1),)})
    path.write_text(before, encoding="utf-8")
    twice = (Row(output=F16B, measured=1), Row(output=F16B, measured=2))
    with refused(Problem.REPEATED_KEY):
        write_table_file({MathOperation.Abs: twice}, path)
    assert path.read_text(encoding="utf-8") == before


def test_a_table_no_driver_could_resolve_is_never_written(tmp_path):
    """Two rows that are equally specific for Float16 -> Float16_b each load alone, and
    each is a valid table to read back; the query is what is ambiguous."""
    path = tmp_path / "budget.yaml"
    before = dump_table({MathOperation.Abs: (Row(output=F16B, measured=1),)})
    path.write_text(before, encoding="utf-8")
    overlapping = (Row(input=F16, measured=1), Row(output=F16B, measured=1))
    with refused(Problem.AMBIGUOUS):
        write_table_file({MathOperation.Abs: overlapping}, path)
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("text", ["false", "0", "[]", "''"], ids=repr)
def test_a_falsy_document_that_is_not_a_mapping_is_not_an_empty_table(text):
    with refused(Problem.WRONG_SHAPE):
        read_table(text)


@pytest.mark.parametrize("text", ["", "# only a comment\n", "null"], ids=repr)
def test_an_empty_document_is_an_empty_table(text):
    assert read_table(text) == {}


def test_the_table_on_disk_is_exactly_what_the_serializer_writes():
    """No hand formatting survives a re-emit, so none may be in the file."""
    assert dump_table(load_table()) == TABLE_PATH.read_text(encoding="utf-8")


def test_a_measurement_round_trips_through_its_json_line():
    measurement = Measurement(
        op=MathOperation.Tanh,
        key=BudgetKey(
            input_format=F16B,
            output_format=F16B,
            approx_mode=ApproximationMode.No,
            dest_acc=DestAccumulation.No,
            arch=ChipArchitecture.WORMHOLE,
        ),
        worst=10,
        lanes=65025,
        nonfinite=2,
        bins=((-151, 128, 128, 3), (151, 128, 128, 3)),
        readings=(Reading(-INF, -1.0, 3, 16000), Reading(1.0, INF, None, 0)),
    )
    text = measurement.to_json()
    assert Measurement.from_json(text) == measurement
    # JSON has no infinity: `json.dumps` would write `-Infinity`, which only Python
    # reads back; an infinite edge is spelt as a string instead.
    assert "Infinity" not in text
    assert json.loads(text)["readings"][0]["low"] == "-inf"
