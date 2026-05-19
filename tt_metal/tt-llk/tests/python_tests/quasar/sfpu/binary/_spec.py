import itertools
from dataclasses import dataclass
from typing import List

from helpers.format_config import DataFormat

from .._spec import BaseOpSpec


@dataclass(frozen=True)
class BinaryInputOutputFormat:
    input_a_format: DataFormat
    input_b_format: DataFormat
    output_format: DataFormat


def binary_input_output_formats(
    inputs_a: List[DataFormat],
    inputs_b: List[DataFormat],
    outputs: List[DataFormat],
) -> List[BinaryInputOutputFormat]:
    return [
        BinaryInputOutputFormat(a, b, o)
        for a, b, o in itertools.product(inputs_a, inputs_b, outputs)
    ]


@dataclass(kw_only=True)
class BinaryOpSpec(BaseOpSpec):
    formats: List[BinaryInputOutputFormat]
