# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared measurement axes for Blaze-promoted experimental perf twins."""

from helpers.llk_params import PerfRunType
from helpers.perf.core import ALL_PERF_RUN_TYPES, create_test_or_perf_config
from helpers.test_variant_parameters import LOOP_FACTOR, PERF_RUN_TYPE

PERF_LOOP_FACTOR = 32

PERF_MEASUREMENT_AXES = dict(
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)


def with_functional_defaults(test_config_kwargs):
    """Copy templates and runtimes, appending the functional perf symbols when absent.

    Direct ``TestConfig`` construction skips ``create_test_or_perf_config``, which is
    what injects ``PERF_RUN_TYPE`` and ``LOOP_FACTOR``. Kernels that branch on those
    symbols then fail to compile. Callers that already pass either symbol keep it.
    """
    templates = list(test_config_kwargs.get("templates") or [])
    if not any(isinstance(param, PERF_RUN_TYPE) for param in templates):
        templates.append(PERF_RUN_TYPE(PerfRunType.L1_TO_L1))
    runtimes = list(test_config_kwargs.get("runtimes") or [])
    if not any(isinstance(param, LOOP_FACTOR) for param in runtimes):
        runtimes.append(LOOP_FACTOR(1))
    return {**test_config_kwargs, "templates": templates, "runtimes": runtimes}


def execute_llk(
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
    boot_mode=None,
    test_config_kwargs,
):
    """Build TestConfig or PerfConfig, inject LOOP_FACTOR, and run.

    Returns the functional ``run()`` outcome, or ``None`` after a perf run.
    """
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    runtimes = [
        param
        for param in list(test_config_kwargs.get("runtimes") or [])
        if not isinstance(param, LOOP_FACTOR)
    ]
    runtimes.append(LOOP_FACTOR(loop_factor))
    kwargs = {**test_config_kwargs, "runtimes": runtimes}
    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types or [PerfRunType.L1_TO_L1],
        test_config_kwargs=kwargs,
        boot_mode=boot_mode,
    )
    if is_perf:
        configuration.run(perf_report)
        return None
    return configuration.run()
