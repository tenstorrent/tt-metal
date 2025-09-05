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

PROFILER_SCRIPTS_ROOT = TT_METAL_HOME / "tools/tracy"
PROFILER_ARTIFACTS_DIR = TT_METAL_HOME / "generated/profiler"
if "TT_METAL_PROFILER_DIR" in ENVS.keys():
    PROFILER_ARTIFACTS_DIR = Path(ENVS["TT_METAL_PROFILER_DIR"])


PROFILER_BIN_DIR = TT_METAL_HOME / "build/bin"
PROFILER_WASM_DIR = TT_METAL_HOME / "build/profiler/build_wasm"
PROFILER_WASM_TRACE_FILE_NAME = "embed.tracy"

TRACY_OPS_TIMES_FILE_NAME = "tracy_ops_times.csv"
TRACY_OPS_DATA_FILE_NAME = "tracy_ops_data.csv"
TRACY_MODULE_PATH = TT_METAL_HOME / "tt_metal/third_party/tracy"
TRACY_FILE_NAME = "tracy_profile_log_host.tracy"

TRACY_CAPTURE_TOOL = "tracy-capture"
TRACY_CSVEXPROT_TOOL = "tracy-csvexport"

LD_LIBRARY_PATH = TT_METAL_HOME / "build/_deps/capstone-build"


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
