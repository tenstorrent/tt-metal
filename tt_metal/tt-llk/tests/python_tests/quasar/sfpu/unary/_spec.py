import itertools
from dataclasses import dataclass
from typing import List

from helpers.format_config import DataFormat

from .._spec import BaseOpSpec


@dataclass(frozen=True)
class InputOutputFormat:
    input_format: DataFormat
    output_format: DataFormat


def input_output_formats(
    inputs: List[DataFormat],
    outputs: List[DataFormat] = None,
) -> List[InputOutputFormat]:
    if outputs is None:
        outputs = inputs
    return [InputOutputFormat(i, o) for i, o in itertools.product(inputs, outputs)]


@dataclass(kw_only=True)
class UnaryOpSpec(BaseOpSpec):
    formats: List[InputOutputFormat]
