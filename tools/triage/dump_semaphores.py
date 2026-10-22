#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_semaphores [--include-global-unchanged]

Options:
    --include-global-unchanged   Also show global semaphores whose Value equals Initial.

Description:
    Semaphore values on Tensix and Ethernet cores, one row per semaphore per core.

    Kind is program for a semaphore of the program running on the core (CreateSemaphore). It is read
    from the kernel config buffer at the core type's sem_offset, and only on cores its core range
    covers: the host initializes a slot on those cores only, so any other core holds a stale value
    there. Kind is global for a GlobalSemaphore, one word at the same L1 address on every Tensix core
    of its range, shown whether or not the core is running.

    Initial is the value the host last wrote: at launch for a program semaphore, at creation or the
    latest reset_semaphore_value for a global one, N/A if it never wrote one. A kernel resetting a
    semaphore on device is not reflected.

    Without --include-global-unchanged, global rows still at Initial (idle or never signaled) are
    left out, since models can declare ~100 GlobalSemaphores per core. A global semaphore missing
    from a core it covers is therefore at Initial. Program rows are always shown.


Owner:
    onenezicTT
"""

from run_checks import run as get_run_checks
from semaphore_provider import SemaphoreData, SemaphoreProvider, run as get_semaphore_provider
from triage import ScriptConfig, run_script
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate

script_config = ScriptConfig(
    depends=["run_checks", "semaphore_provider"],
)


def shown_semaphores(
    location: OnChipCoordinate, provider: SemaphoreProvider, include_unchanged: bool
) -> list[SemaphoreData]:
    return [
        s
        for s in provider.get_cached_semaphores(location)
        if include_unchanged or s.kind == "program" or s.value != s.initial
    ]


def run(args, context: Context):
    include_unchanged: bool = args["--include-global-unchanged"]
    run_checks = get_run_checks(args, context)
    provider = get_semaphore_provider(args, context)
    return run_checks.run_per_block_check(
        lambda location: shown_semaphores(location, provider, include_unchanged),
        block_filter=["tensix", "active_eth", "idle_eth"],
    )


if __name__ == "__main__":
    run_script()
