import itertools
from dataclasses import dataclass
from typing import List

from helpers.format_config import DataFormat

from .._spec import BaseOpSpec


@dataclass(frozen=True)
class TernaryInputOutputFormat:
    input_a_format: DataFormat
    input_b_format: DataFormat
    input_c_format: DataFormat
    output_format: DataFormat


def ternary_input_output_formats(
    inputs_a: List[DataFormat],
    inputs_b: List[DataFormat],
    inputs_c: List[DataFormat],
    outputs: List[DataFormat],
) -> List[TernaryInputOutputFormat]:
    return [
        TernaryInputOutputFormat(a, b, c, o)
        for a, b, c, o in itertools.product(inputs_a, inputs_b, inputs_c, outputs)
    ]


def ternary_input_output_formats_matched(
    formats: List[DataFormat],
) -> List[TernaryInputOutputFormat]:
    return [TernaryInputOutputFormat(f, f, f, f) for f in formats]


@dataclass(kw_only=True)
class TernaryOpSpec(BaseOpSpec):
    formats: List[TernaryInputOutputFormat]
