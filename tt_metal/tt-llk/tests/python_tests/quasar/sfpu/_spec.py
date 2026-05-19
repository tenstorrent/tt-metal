from collections.abc import Callable
from dataclasses import dataclass, field

from helpers.format_config import FormatConfig, List
from helpers.llk_params import MathOperation
from helpers.test_variant_parameters import TemplateParameter

DispatchDefinesFn = Callable[[FormatConfig], dict]


@dataclass
class BaseOpSpec:
    name: str
    math_op: MathOperation
    header: str
    sfpu_defines: DispatchDefinesFn
    extra_templates: List[TemplateParameter] = field(default_factory=list)

    def __post_init__(self):
        if type(self) is BaseOpSpec:
            raise TypeError(
                "BaseOpSpec is not a concrete implementation; use a subclass"
            )
