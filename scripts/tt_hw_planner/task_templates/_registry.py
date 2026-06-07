# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TaskTemplate registry — maps HF AutoModel class -> concrete TaskTemplate.

Templates register themselves at import time. The orchestrator uses
``lookup_template(task_class_name)`` to dispatch. Lookup returns ``None``
if no template exists for the task class — emit-e2e then refuses
cleanly with a "no template for HF class X" message pointing at
``task_templates/`` for the author to add one.

This is the single point that decides which task pipeline gets emitted
for a given model. Adding support for a new task category =
authoring one new TaskTemplate subclass and adding one register() call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from ._base import TaskTemplate


# Internal registry — populated by ``register_template`` calls in each
# concrete template module's import side effects.
_REGISTRY: Dict[str, Type[TaskTemplate]] = {}

# Multi-task models map ONE model_id to MULTIPLE task templates.
# Keyed by HF model_type (from config.model_type) instead of task class,
# since one model file declares multiple task heads.
# e.g. {"seamless_m4_t": ["s2tt", "t2t", "t2s"]}
_MULTI_TASK_MAP: Dict[str, List[str]] = {}


def register_template(template_cls: Type[TaskTemplate]) -> Type[TaskTemplate]:
    """Decorator-friendly registration. Idempotent — silently replaces
    a previous registration with the same HF_TASK_CLASS."""
    if not template_cls.HF_TASK_CLASS:
        raise ValueError(f"{template_cls.__name__} must declare HF_TASK_CLASS before register_template")
    if not template_cls.TASK_NAME:
        raise ValueError(f"{template_cls.__name__} must declare TASK_NAME (used in emitted filenames)")
    _REGISTRY[template_cls.HF_TASK_CLASS] = template_cls
    return template_cls


def register_multi_task(model_type: str, task_names: List[str]) -> None:
    """Declare that an HF ``model_type`` (from config.model_type)
    supports multiple tasks. emit-e2e --all-tasks expands to all of them.

    Example: ``register_multi_task("seamless_m4_t", ["s2tt", "t2t", "t2s"])``.
    """
    _MULTI_TASK_MAP[model_type] = list(task_names)


def lookup_template(task_class_name: str) -> Optional[Type[TaskTemplate]]:
    """Return the TaskTemplate subclass for an HF AutoModel class name,
    or None if no template is registered."""
    return _REGISTRY.get(task_class_name)


def lookup_template_by_name(task_name: str) -> Optional[Type[TaskTemplate]]:
    """Return the template whose TASK_NAME matches (e.g. 's2tt' -> S2TTTemplate).
    Used by the --task CLI flag to pick a specific task on multi-task models."""
    for cls in _REGISTRY.values():
        if cls.TASK_NAME == task_name:
            return cls
    return None


def all_task_names() -> List[str]:
    """List of every registered TASK_NAME. Used for CLI --task choices."""
    return sorted({cls.TASK_NAME for cls in _REGISTRY.values()})


def multi_task_tasks_for(model_type: str) -> List[str]:
    """If model_type has multi-task heads registered, return their task names.
    Otherwise return empty list."""
    return list(_MULTI_TASK_MAP.get(model_type, []))


def registered_classes() -> Dict[str, Type[TaskTemplate]]:
    """Read-only view of the registry — for diagnostics."""
    return dict(_REGISTRY)


__all__ = [
    "register_template",
    "register_multi_task",
    "lookup_template",
    "lookup_template_by_name",
    "all_task_names",
    "multi_task_tasks_for",
    "registered_classes",
]
