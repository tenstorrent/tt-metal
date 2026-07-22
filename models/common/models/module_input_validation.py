# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runtime validation helpers for declared module input configs."""

from __future__ import annotations

import contextlib
import contextvars
import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from loguru import logger

import ttnn

NamedModuleIterator = Callable[[object], Iterable[tuple[str, object]]]
_validation_suspend_depth = contextvars.ContextVar("module_input_validation_suspend_depth", default=0)


@dataclass
class ConfigMismatch:
    """Record of a module config mismatch detected during validation."""

    module_name: str
    expected_memcfg: ttnn.MemoryConfig
    actual_memcfg: ttnn.MemoryConfig
    # todo)) check for more details like dtype --> example: somebody messed up and run compile with float32 but actual is bfloat16 --> meaning unexpected compilation time included in the actual run


@contextlib.contextmanager
def suspend_module_input_validation():
    """Temporarily bypass validation checks while keeping wrapped methods installed."""
    token = _validation_suspend_depth.set(_validation_suspend_depth.get() + 1)
    try:
        yield
    finally:
        _validation_suspend_depth.reset(token)


@contextlib.contextmanager
def validate_module_input_configs(
    *,
    model: object,
    iter_named_modules: NamedModuleIterator | None,
    mode: str,
):
    """Instrument one forward pass to check actual vs declared input memcfgs."""
    if iter_named_modules is None:
        # Generator-based context managers must yield once even for a no-op branch.
        yield
        return
    if mode not in {"prefill", "decode"}:
        raise ValueError(f"Unsupported validation mode: {mode}")

    mismatches: list[ConfigMismatch] = []
    originals: dict[str, tuple[object, str, Callable]] = {}

    for name, module in iter_named_modules(model):
        if not hasattr(module, "config"):
            continue

        expected_attr = f"{mode}_input_memcfg"
        expected = getattr(module.config, expected_attr, None)
        if expected is None:
            continue

        method_name = f"{mode}_forward"
        if not hasattr(module, method_name):
            raise AttributeError(f"Module {name} has {expected_attr} but no {method_name} method")

        original_method = getattr(module, method_name)
        originals[name] = (module, method_name, original_method)

        def make_wrapper(orig, mod_name, exp_memcfg):
            @functools.wraps(orig)
            def wrapper(x, *args, **kwargs):
                # TODO: replace monkey-patched method interception with explicit module-level validation hooks.
                if _validation_suspend_depth.get() <= 0 and isinstance(x, ttnn.Tensor) and x.is_allocated():
                    actual = x.memory_config()
                    logger.info(
                        f"[validate_module_configs] Validating module configs in {mode} mode: "
                        f"{mod_name} expected {exp_memcfg}, actual {actual}"
                    )
                    if actual != exp_memcfg:
                        mismatches.append(
                            ConfigMismatch(
                                module_name=mod_name,
                                expected_memcfg=exp_memcfg,
                                actual_memcfg=actual,
                            )
                        )
                return orig(x, *args, **kwargs)

            return wrapper

        setattr(module, method_name, make_wrapper(original_method, name, expected))

    try:
        yield
    finally:
        for name, (module, method_name, original) in originals.items():
            setattr(module, method_name, original)

        for mismatch in mismatches:
            logger.warning(
                f"Config mismatch at {mismatch.module_name}: "
                f"declared {mismatch.expected_memcfg}, actual {mismatch.actual_memcfg}"
            )
