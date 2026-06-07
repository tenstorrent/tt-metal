# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Task-template package for emit-e2e production-demo emission.

Public surface:
  * ``TaskTemplate`` — ABC every concrete template subclasses
  * ``TemplateContext`` — input dataclass passed to every emit_*
  * ``EmittedFiles`` — output dataclass returned by emit_all
  * ``register_template``, ``lookup_template`` — registry seams

Concrete templates live in sibling modules (``s2tt_template.py`` etc.)
and register themselves at import time. Importing this package triggers
those side-effect registrations.
"""

from ._base import (
    INPUT_AUDIO,
    INPUT_CSV,
    INPUT_IMAGE,
    INPUT_TEXT,
    INPUT_VIDEO,
    OUTPUT_AUDIO,
    OUTPUT_IMAGE,
    OUTPUT_SCALAR,
    OUTPUT_TEXT,
    CompositionTree,
    EmittedFiles,
    Quirk,
    TaskTemplate,
    TemplateContext,
)
from ._registry import (
    all_task_names,
    lookup_template,
    lookup_template_by_name,
    multi_task_tasks_for,
    register_multi_task,
    register_template,
    registered_classes,
)

# Concrete templates — import for register_template() side effects
from . import s2tt_template  # noqa: F401
from . import t2t_template  # noqa: F401
from . import t2s_template  # noqa: F401
from . import segmentation_template  # noqa: F401

__all__ = [
    "INPUT_AUDIO",
    "INPUT_CSV",
    "INPUT_IMAGE",
    "INPUT_TEXT",
    "INPUT_VIDEO",
    "OUTPUT_AUDIO",
    "OUTPUT_IMAGE",
    "OUTPUT_SCALAR",
    "OUTPUT_TEXT",
    "CompositionTree",
    "EmittedFiles",
    "Quirk",
    "TaskTemplate",
    "TemplateContext",
    "all_task_names",
    "lookup_template",
    "lookup_template_by_name",
    "multi_task_tasks_for",
    "register_multi_task",
    "register_template",
    "registered_classes",
]
