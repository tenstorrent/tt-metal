"""Constraint catalog + recipe library for ttnn ops.

When the LLM driver is asked to port an HF submodule to ttnn, it sees
the stub + opplan + captures + failure trace — but NOT the universe of
shape/dtype quirks that ttnn ops impose. This package fills that gap:

  - ``catalog``: declarative rules ("if this component looks like X
    AND its input shape has property Y, ttnn op Z will reject it").
  - ``checker``: given a component's captures + opplan, evaluate every
    rule and return matching violations.
  - ``recipes``: per-violation workaround recipe (markdown blocks the
    LLM can paste into its prompt and copy patterns from).
  - ``prompt_injection``: format violations + recipes as a markdown
    block to splice into the LLM prompt.

This is shape/dtype/quirk knowledge about ttnn, independent of any
model. Encoded once, reused on every bringup the tool does.
"""

from .catalog import Catalog, Constraint, OpCallContext, Violation
from .checker import check_component
from .prompt_injection import format_constraint_hints
from .recipes import RECIPES, Recipe

__all__ = [
    "Catalog",
    "Constraint",
    "OpCallContext",
    "RECIPES",
    "Recipe",
    "Violation",
    "check_component",
    "format_constraint_hints",
]
