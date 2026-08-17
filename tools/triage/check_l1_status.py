#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_l1_status

Description:
    Checks that each worker L1 launch value matches the expected firmware launch value from Inspector.

Owner:
    anashTT
"""

from inspector_data import run as get_inspector_data
from metal_device_id_mapping import run as get_metal_device_id_mapping
from run_checks import run as get_run_checks
from triage import ScriptConfig, log_check_location, run_script
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_word_from_device


script_config = ScriptConfig(
    depends=["run_checks", "inspector_data", "metal_device_id_mapping"],
)


def check_worker_l1_launch_value(location: OnChipCoordinate, tensix_fw_launch_values: dict[int, int]) -> None:
    expected_fw_launch_value = tensix_fw_launch_values.get(location.device.unique_id)
    if expected_fw_launch_value is None:
        raise RuntimeError(f"Missing Inspector build env data for {location}")

    actual_value = read_word_from_device(location, 0)
    log_check_location(
        location,
        actual_value == expected_fw_launch_value,
        f"Worker L1[0] mismatch: read 0x{actual_value:08x}, expected 0x{expected_fw_launch_value:08x}",
    )


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    inspector_data = get_inspector_data(args, context)
    metal_device_id_mapping = get_metal_device_id_mapping(args, context)
    tensix_fw_launch_values = {
        metal_device_id_mapping.get_unique_id(build_env.metalDeviceId): build_env.buildInfo.tensixFwLaunchAddrValue
        for build_env in inspector_data.getAllBuildEnvs().buildEnvs
    }

    run_checks.run_per_block_check(
        lambda location: check_worker_l1_launch_value(location, tensix_fw_launch_values),
        block_filter="tensix",
    )


if __name__ == "__main__":
    run_script()
