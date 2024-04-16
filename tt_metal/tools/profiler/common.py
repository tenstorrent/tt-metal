# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path

ENVS = dict(os.environ)
TT_METAL_HOME = ""
if "TT_METAL_HOME" in ENVS.keys():
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
else:
    logger.error("TT_METAL_HOME environment variable is not set up properly.")
    sys.exit(1)

PROFILER_DEVICE_SIDE_LOG = "profile_log_device.csv"
PROFILER_HOST_SIDE_LOG = "profile_log_host.csv"
PROFILER_HOST_DEVICE_SYNC_INFO = "sync_device_info.csv"

PROFILER_SCRIPTS_ROOT = TT_METAL_HOME / "tt_metal/tools/profiler"
PROFILER_ARTIFACTS_DIR = TT_METAL_HOME / "generated/profiler"

PROFILER_BIN_DIR = TT_METAL_HOME / "build/tools/profiler/bin"
PROFILER_LOGS_DIR = PROFILER_ARTIFACTS_DIR / ".logs"
PROFILER_OUTPUT_DIR = PROFILER_ARTIFACTS_DIR / "reports"
PROFILER_OPS_LOGS_DIR = PROFILER_LOGS_DIR / "ops"
PROFILER_LOG_LOCATIONS_RECORD = PROFILER_LOGS_DIR / ".locations.log"

TRACY_OPS_TIMES_FILE_NAME = "tracy_ops_times.csv"
TRACY_OPS_DATA_FILE_NAME = "tracy_ops_data.csv"
TRACY_MODULE_PATH = TT_METAL_HOME / "tt_metal/third_party/tracy"
TRACY_FILE_NAME = "tracy_profile_log_host.tracy"

TRACY_CAPTURE_TOOL = "capture-release"
TRACY_CSVEXPROT_TOOL = "csvexport-release"


def rm(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def clear_profiler_runtime_artifacts():
    rm(PROFILER_ARTIFACTS_DIR)


def get_log_locations():
    logLocations = set()
    deviceLogLocations = set()
    if os.path.isfile(PROFILER_LOG_LOCATIONS_RECORD):
        with open(PROFILER_LOG_LOCATIONS_RECORD, "r") as recordFile:
            for line in recordFile.readlines():
                logLocation = line.strip()
                if os.path.isdir(f"{logLocation}") or os.path.isdir(f"{TT_METAL_HOME}/{logLocation}"):
                    logLocations.add(logLocation)
                    tmpSplit = logLocation.rsplit("_", 1)
                    if tmpSplit[-1] == "device":
                        deviceLogLocations.add(tmpSplit[0])
    for logLocation in deviceLogLocations:
        if logLocation in logLocations:
            logLocations.remove(f"{logLocation}_device")

    return list(logLocations)
