import itertools
from dataclasses import fields, replace
from typing import Iterator, List, Tuple

from helpers.test_variant_parameters import TemplateParameter


def _expand_one(tp: TemplateParameter) -> Iterator[TemplateParameter]:
    """One TemplateParameter -> N TemplateParameters by exploding any list-valued
    fields. Scalar fields pass through. The TemplateParameter dataclass field
    types stay scalar; lists are tolerated at runtime as a spec-author shortcut."""
    list_fields = [f for f in fields(tp) if isinstance(getattr(tp, f.name), list)]
    if not list_fields:
        yield tp
        return
    value_lists = [getattr(tp, f.name) for f in list_fields]
    for combo in itertools.product(*value_lists):
        yield replace(tp, **{f.name: v for f, v in zip(list_fields, combo)})


def expand_extra_templates(
    extras: List[TemplateParameter],
) -> List[Tuple[TemplateParameter, ...]]:
    """Return one tuple of fully-resolved (scalar-field) TemplateParameter
    instances per cross-axis combination. Empty input -> [()]; the driver
    splices the tuple into TestConfig.templates so a no-extras spec emits
    nothing extra."""
    if not extras:
        return [()]
    axes = [list(_expand_one(t)) for t in extras]
    return [tuple(combo) for combo in itertools.product(*axes)]
