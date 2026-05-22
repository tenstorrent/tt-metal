#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Script-API surface for tt-triage.

Defines the symbols every triage script needs (`TriageScript`, `ScriptConfig`,
`triage_field`, ...), separate from the runtime in `triage.py` (CLI, `main`,
`run_script`, console / serializer wiring). `triage.py` re-exports everything
here so existing scripts can keep `from triage import ScriptConfig, ...`.
"""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Callable, TypeVar
import importlib
import inspect
import os
import re
import sys
import traceback

from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.elf import ElfVariable
from ttexalens.umd_device import TimeoutDeviceRegisterError


class TTTriageError(Exception):
    """Base class for TT Triage errors."""

    pass


class ScriptPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class ScriptConfig:
    data_provider: bool = False
    disabled: bool = False
    depends: list[str] = field(default_factory=list)
    priority: ScriptPriority = ScriptPriority.MEDIUM


class ScriptArguments:
    def __init__(self, args: dict[str, Any]):
        self.args = args

    def __getitem__(self, item: str) -> Any:
        return self.args.get(item, None)


# Type variables for the decorator
T = TypeVar("T")


def triage_singleton(run_method: Callable[[ScriptArguments, Context], T], /) -> Callable[[ScriptArguments, Context], T]:
    # Check that run method has two arguments (args, context)
    assert callable(run_method), "run_method must be a callable function."
    signature = inspect.signature(run_method)
    assert (
        len(signature.parameters) == 2 and "args" in signature.parameters and "context" in signature.parameters
    ), "run_method must have two arguments (args, context)."

    # Cache results and exceptions so repeat calls don't re-run on already-failed setup.
    cache: dict[tuple[int, int], tuple[bool, Any]] = {}

    def cache_wrapper(args: ScriptArguments, context: Context) -> T:
        cache_key = (id(args), id(context))
        if cache_key not in cache:
            try:
                cache[cache_key] = (True, run_method(args, context))
            except Exception as e:
                cache[cache_key] = (False, e)
        ok, payload = cache[cache_key]
        if not ok:
            raise payload
        return payload

    return cache_wrapper


# Data serialization
def serialize_collection(items: Iterable[Any], separator: str = ", ") -> str:
    return separator.join(default_serializer(item) for item in items)


def default_serializer(value) -> str:
    """Default serializer for fields."""
    if value is None:
        return "N/A"
    if isinstance(value, Device):
        return str(value.id)
    elif isinstance(value, OnChipCoordinate):
        return value.to_user_str()
    elif isinstance(value, ElfVariable):
        try:
            return serialize_collection(value.as_list())
        except:
            return str(value)
    elif isinstance(value, Iterable) and not isinstance(value, str):
        return serialize_collection(value)
    else:
        return str(value)


def hex_serializer(value: int | None) -> str:
    if value is None:
        return "N/A"
    return hex(value)


def collection_serializer(separator: str):
    def serializer(item):
        return serialize_collection(item, separator)

    return serializer


def triage_field(serialized_name: str | None = None, serializer: Callable[[Any], str] | None = None, verbose: int = 0):
    if serializer is None:
        serializer = default_serializer
    return field(
        metadata={"recurse": False, "serialized_name": serialized_name, "serializer": serializer, "verbose": verbose}
    )


def recurse_field(verbose: int = 0):
    return field(metadata={"recurse": True, "verbose": verbose})


@dataclass
class TriageScript:
    name: str
    path: str
    config: ScriptConfig
    module: ModuleType
    run_method: Callable[..., Any]
    documentation: str
    depends: list["TriageScript"] = field(default_factory=list)
    failed: bool = False
    failure_message: str | None = None
    to_raw_data_method: Callable[[Any], Any] | None = None
    merge_method: Callable[[list[Any]], Any] | None = None

    def run(self, args: ScriptArguments, context: Context, log_error: bool = True) -> Any:
        try:
            result = self.run_method(args=args, context=context)
            if self.config.data_provider and result is None:
                if log_error:
                    self.failed = True
                    self.failure_message = "Data provider script did not return any data."
                else:
                    raise TTTriageError("Data provider script did not return any data.")
            return result
        except TimeoutDeviceRegisterError:
            raise
        except (ValueError, TTTriageError) as e:
            # User-facing exceptions: surface the message only, no traceback noise.
            if log_error:
                self.failed = True
                self.failure_message = f"{e}"
                return None
            else:
                raise
        except Exception as e:
            if log_error:
                self.failed = True
                self.failure_message = traceback.format_exc()
                return None
            else:
                raise

    def to_raw_data(self, result: Any) -> Any:
        if self.to_raw_data_method is not None:
            return self.to_raw_data_method(result)
        return result

    def merge(self, parts: list[Any]) -> Any:
        if self.merge_method is not None:
            return self.merge_method(parts)
        from aggregator import default_merge

        return default_merge(parts)

    @staticmethod
    def load(script_path: str) -> "TriageScript":
        script_path = os.path.abspath(script_path)
        base_path = os.path.dirname(script_path)
        appended = False
        if not base_path in sys.path:
            sys.path.append(base_path)
            appended = True
        try:
            script_name = os.path.splitext(os.path.basename(script_path))[0]
            script_module = importlib.import_module(script_name)

            # Check if script has a configuration
            script_config: ScriptConfig = script_module.script_config
            if script_config is None:
                # This script does not have a configuration, which means it is not tt-triage script, skipping...
                raise ValueError(f"Script {script_path} does not have script_config.")

            # Check if script has a docstring and an owner
            if not script_module.__doc__:
                raise ValueError(f"Script {script_path} must have a docstring, see relevant scripts for examples.\n")

            if not re.search(r"^Owner:\s*\S+", script_module.__doc__, re.MULTILINE):
                raise ValueError(
                    f"Script {script_path} docstring must include an 'Owner:' field with the corresponding owner of the script.\n"
                )

            # Check if script has a run method with two arguments (args and context)
            run_method = script_module.run if hasattr(script_module, "run") and callable(script_module.run) else None
            if run_method is not None:
                signature = inspect.signature(run_method)
                if (
                    len(signature.parameters) != 2
                    or "args" not in signature.parameters
                    or "context" not in signature.parameters
                ):
                    run_method = None
            if run_method is None:
                raise ValueError(
                    f"Script {script_path} does not have a valid run method with two arguments (args, context)."
                )

            to_raw_data_method = getattr(script_module, "to_raw_data", None)
            if to_raw_data_method is not None and not callable(to_raw_data_method):
                to_raw_data_method = None
            merge_method = getattr(script_module, "merge", None)
            if merge_method is not None and not callable(merge_method):
                merge_method = None

            triage_script = TriageScript(
                name=os.path.basename(script_path),
                path=script_path,
                config=deepcopy(script_config),
                module=script_module,
                run_method=run_method,
                documentation=script_module.__doc__,
                to_raw_data_method=to_raw_data_method,
                merge_method=merge_method,
            )

            if triage_script.config.depends is None:
                # If script does not have dependencies, set it to empty list
                triage_script.config.depends = []
            else:
                triage_script.config.depends = [
                    dep if isinstance(dep, str) and dep.endswith(".py") else f"{dep}.py"
                    for dep in triage_script.config.depends
                ]
                triage_script.config.depends = [os.path.join(base_path, dep) for dep in triage_script.config.depends]

            return triage_script
        finally:
            if appended:
                sys.path.remove(base_path)
