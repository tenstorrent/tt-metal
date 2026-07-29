# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess


def get_smbus_telemetry():
    result = subprocess.run(
        ["tt-smi", "-s", "--snapshot_no_tty"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None


def get_ddr_speed():
    data = get_smbus_telemetry()
    if data and data.get("device_info"):
        smbus_telem = data["device_info"][0].get("smbus_telem", {})
        ddr_speed_hex = smbus_telem.get("DDR_SPEED") or smbus_telem.get("SMBUS_TX_DDR_SPEED")
        if ddr_speed_hex:
            return int(ddr_speed_hex, 16)
    return None


def get_board_type():
    """Board type string reported by tt-smi (e.g. ``"p150b"``, ``"p100"``, ``"n300"``), or None."""
    data = get_smbus_telemetry()
    if data and data.get("device_info"):
        return data["device_info"][0].get("board_info", {}).get("board_type")
    return None


def get_tdp_limit_max():
    """Max TDP limit (watts) from smbus_telem, or None.

    Read like get_ddr_speed: smbus_telem values are hex strings, with a
    SMBUS_TX_-prefixed fallback key. Returns the decoded integer watts.
    """
    data = get_smbus_telemetry()
    if data and data.get("device_info"):
        smbus_telem = data["device_info"][0].get("smbus_telem", {})
        tdp = smbus_telem.get("TDP_LIMIT_MAX") or smbus_telem.get("SMBUS_TX_TDP_LIMIT_MAX")
        if tdp is not None:
            return tdp if isinstance(tdp, int) else int(str(tdp), 16)
    return None


def is_high_power(min_tdp_watts=130):
    """True iff the host's max TDP limit (smbus_telem TDP_LIMIT_MAX) is >= min_tdp_watts.

    Guards perf-sensitive CI jobs that must run on high-power galaxy hosts. Like is_p150(),
    it reads tt-smi only (no compute-device open / CHIP_IN_USE lock), so it is safe to call
    from a pytest ``skipif`` at collection time. Returns False on any error (tt-smi
    missing/failed/timeout, or field absent) so callers skip rather than error.
    """
    try:
        tdp = get_tdp_limit_max()
    except Exception:
        return False
    return tdp is not None and tdp >= min_tdp_watts


def is_p150():
    """True iff the host's board is a P150 (any revision, e.g. ``p150``/``p150b``).

    Reads board type via tt-smi (SMBus/sysfs), which does NOT open the compute device
    or take the CHIP_IN_USE lock — so, unlike ``ttnn.cluster.get_cluster_type()``, it is
    safe to call from a pytest ``skipif`` at collection time. Returns False on any error
    (tt-smi missing/failed/timeout) so callers degrade to skipping rather than erroring.
    """
    try:
        board_type = get_board_type()
    except Exception:
        return False
    return board_type is not None and board_type.lower().startswith("p150")
