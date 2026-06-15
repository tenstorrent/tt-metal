import itertools
from dataclasses import dataclass, field
from typing import List

from helpers.format_config import DataFormat
from helpers.llk_params import VectorMode
from helpers.test_variant_parameters import VECTOR_MODE, TemplateParameter

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
    # The ternary SFPU dispatch is the only arity whose kernel call takes a VECTOR_MODE
    # face selector, so it lives here as a declared template param rather than a driver
    # hardcode. Default is all faces (RC); give the field a list (e.g.
    # ``VECTOR_MODE([VectorMode.RC, VectorMode.R, VectorMode.C])``) to sweep it into
    # separate variants — see BaseOpSpec.extra_templates. A swept mode needs a
    # mode-aware ``golden`` (the default golden compares all faces).
    extra_templates: List[TemplateParameter] = field(
        default_factory=lambda: [VECTOR_MODE(VectorMode.RC)]
    )
    arity_macro: str = "SFPU_TERNARY_OP"
