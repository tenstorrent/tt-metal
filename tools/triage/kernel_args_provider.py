#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    kernel_args_provider

Description:
    Data provider for the arguments of the kernel each RISC runs: its compile-time args, its runtime
    args on this core and its common runtime args, for dump_callstacks and dump_running_operations.

    Compile-time args come from Inspector, which copies them when the kernel is compiled; they never
    change. Runtime args are read from L1, where the kernel reads them: kernel_config_base plus the
    RISC's rta_offset (crta_offset for common args) from the launch message. The host's copies can be
    older or newer than what a hung kernel uses, so L1 is the only exact source. L1 holds where each
    list starts but not how long it is, so Inspector gives the number of words to read. With watcher
    asserts on, each list starts with a count word, which is skipped.

    Each value carries the name metal gives it: Metal 2.0 arg names, tensor::<accessor> and
    scratch::<accessor> slots, and tt-blaze's named args. Other values are known by position only.
    A RISC not enabled by the launch, or running a kernel Inspector does not know, has no args (None).

Owner:
    onenezicTT
"""

from dataclasses import dataclass
from functools import cache

from rich.markup import escape

from configuration_provider import run as get_configuration_provider
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from inspector_capnp import KernelData
from triage import ScriptConfig, run_script, triage_field, triage_singleton
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.hardware.risc_debug import RiscLocation

script_config = ScriptConfig(
    data_provider=True,
    depends=["dispatcher_data", "configuration_provider"],
)


@dataclass
class KernelArg:
    name: str | None
    value: int


def serialize_kernel_args(args: list[KernelArg] | None) -> str:
    """Values in word order, in hex, as name=value when the word has a name."""
    if args is None:
        return "N/A"
    return ", ".join(f"{escape(arg.name)}={arg.value:#x}" if arg.name else f"{arg.value:#x}" for arg in args)


@dataclass
class KernelArgsData:
    compile_time_args: list[KernelArg] | None = triage_field("CT Args", serialize_kernel_args, verbose=1)
    runtime_args: list[KernelArg] | None = triage_field("RT Args", serialize_kernel_args, verbose=1)
    common_runtime_args: list[KernelArg] | None = triage_field("Common RT Args", serialize_kernel_args, verbose=1)


def get_compile_time_args(kernel: KernelData) -> list[KernelArg]:
    """Positional args in order, then the named ones."""
    positional = [
        KernelArg(name or None, value) for name, value in zip(kernel.compileTimeArgNames, kernel.compileTimeArgs)
    ]
    return positional + [KernelArg(arg.name, arg.value) for arg in kernel.namedCompileTimeArgs]


class KernelArgsProvider:
    def __init__(self, dispatcher_data: DispatcherData, watcher_asserts: bool):
        self.dispatcher_data = dispatcher_data
        self.count_word_bytes = 4 if watcher_asserts else 0

    @cache
    def get_cached_kernel_args(self, risc_location: RiscLocation) -> KernelArgsData:
        return self.get_kernel_args(risc_location)

    def get_kernel_args(self, risc_location: RiscLocation) -> KernelArgsData:
        core = self.dispatcher_data.get_cached_core_data(risc_location)
        kernel = core.kernel
        if kernel is None or core.risc_enabled_by_kernel is False:
            return KernelArgsData(None, None, None)
        location = risc_location.location
        (x, y), _ = location.to("logical")
        runtime_arg_count = 0
        for group in kernel.runtimeArgCounts:
            if any(r.start.x <= x <= r.end.x and r.start.y <= y <= r.end.y for r in group.coreRanges):
                runtime_arg_count = group.count
        base = core.kernel_config_base + self.count_word_bytes
        return KernelArgsData(
            compile_time_args=get_compile_time_args(kernel),
            runtime_args=self._read_args(
                location, base + core.rta_offset, runtime_arg_count, list(kernel.runtimeArgNames)
            ),
            common_runtime_args=self._read_args(
                location, base + core.crta_offset, kernel.commonRuntimeArgCount, list(kernel.commonRuntimeArgNames)
            ),
        )

    def _read_args(self, location: OnChipCoordinate, address: int, count: int, names: list[str]) -> list[KernelArg]:
        if count == 0:
            return []
        data = bytearray(4 * count)
        self.dispatcher_data.l1_memory_access(location).read(address, data)
        values = [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
        return [KernelArg(names[i] if i < len(names) and names[i] else None, value) for i, value in enumerate(values)]


@triage_singleton
def run(args, context: Context) -> KernelArgsProvider:
    configuration = get_configuration_provider(args, context)
    # Metal's rule for the count word: is_watcher_assert_enabled in tt_metal/impl/program/dispatch.cpp
    disabled_features = (configuration.get("watcher_disabled_features") or "").split(", ")
    watcher_asserts = configuration.get_bool("watcher_enabled") and "ASSERT" not in disabled_features
    return KernelArgsProvider(get_dispatcher_data(args, context), watcher_asserts)


if __name__ == "__main__":
    run_script()
