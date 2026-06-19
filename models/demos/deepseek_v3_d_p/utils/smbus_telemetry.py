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
