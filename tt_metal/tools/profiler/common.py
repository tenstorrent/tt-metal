# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path
from loguru import logger
import sys

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
if "TT_METAL_PROFILER_DIR" in ENVS.keys():
    PROFILER_ARTIFACTS_DIR = Path(ENVS["TT_METAL_PROFILER_DIR"])


PROFILER_BIN_DIR = TT_METAL_HOME / "build/tools/profiler/bin"

TRACY_OPS_TIMES_FILE_NAME = "tracy_ops_times.csv"
TRACY_OPS_DATA_FILE_NAME = "tracy_ops_data.csv"
TRACY_MODULE_PATH = TT_METAL_HOME / "tt_metal/third_party/tracy"
TRACY_FILE_NAME = "tracy_profile_log_host.tracy"

TRACY_CAPTURE_TOOL = "capture-release"
TRACY_CSVEXPROT_TOOL = "csvexport-release"


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
