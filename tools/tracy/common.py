# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
from pathlib import Path

from loguru import logger

ENVS = dict(os.environ)
TT_METAL_HOME = Path(ENVS.get("TT_METAL_HOME", Path.cwd()))

PROFILER_DEVICE_SIDE_LOG = "profile_log_device.csv"
PROFILER_HOST_SIDE_LOG = "profile_log_host.csv"
PROFILER_HOST_DEVICE_SYNC_INFO = "sync_device_info.csv"
PROFILER_CPP_DEVICE_PERF_REPORT = "cpp_device_perf_report.csv"

PROFILER_SCRIPTS_ROOT = TT_METAL_HOME / "tools/tracy"
PROFILER_ARTIFACTS_DIR = TT_METAL_HOME / "generated/profiler"
if "TT_METAL_PROFILER_DIR" in ENVS.keys():
    PROFILER_ARTIFACTS_DIR = Path(ENVS["TT_METAL_PROFILER_DIR"])


PROFILER_BIN_DIR = TT_METAL_HOME / "build/tools/profiler/bin"

TRACY_OPS_TIMES_FILE_NAME = "tracy_ops_times.csv"
TRACY_OPS_DATA_FILE_NAME = "tracy_ops_data.csv"
TRACY_FILE_NAME = "tracy_profile_log_host.tracy"

TRACY_CAPTURE_TOOL = "capture-release"
TRACY_CSVEXPROT_TOOL = "csvexport-release"


def find_all(start_dir: str, filename: str):
    for root, _dirs, files in os.walk(start_dir, followlinks=True):
        if filename in files:
            yield Path(root) / filename


def find_profiler_binaries():
    logger.info(f"Searching for profiler binaries in {TT_METAL_HOME}...")
    global PROFILER_BIN_DIR
    for binary in find_all(TT_METAL_HOME, TRACY_CAPTURE_TOOL):
        if binary.is_file():
            capture_tool = binary
            logger.info(f"Found profiler capture tool: {capture_tool}")

            PROFILER_BIN_DIR = binary.parent

            csvexport_tool = PROFILER_BIN_DIR / TRACY_CSVEXPROT_TOOL
            if not csvexport_tool.is_file():
                logger.error(f"CSV export tool {csvexport_tool} not found in the same directory as capture tool.")
                sys.exit(1)

            logger.info(f"Found profiler CSV export tool: {csvexport_tool}")
            return

    logger.error(
        "Profiler binaries not found. Please ensure that the TT_METAL_HOME environment variable is set correctly."
    )
    sys.exit(1)


find_profiler_binaries()


def generate_logs_folder(outFolder):
    return Path(outFolder) / ".logs"


def generate_reports_folder(outFolder):
    return Path(outFolder) / "reports"


PROFILER_LOGS_DIR = generate_logs_folder(PROFILER_ARTIFACTS_DIR)
PROFILER_OUTPUT_DIR = generate_reports_folder(PROFILER_ARTIFACTS_DIR)


def rm(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def clear_profiler_runtime_artifacts():
    rm(PROFILER_ARTIFACTS_DIR)
