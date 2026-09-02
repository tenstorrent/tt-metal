# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import sys
import signal
import os
import io
import csv
import re
import shutil
import subprocess
import tempfile
import time
import socket

from loguru import logger

from .process_ops_logs import process_ops
from .common import (
    TT_METAL_HOME,
    PROFILER_BIN_DIR,
    PROFILER_LOGS_DIR,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_WASM_DIR,
    PROFILER_WASM_TRACE_FILE_NAME,
    PROFILER_WASM_TRACES_DIR,
    TRACY_MODULE_PATH,
    TRACY_FILE_NAME,
    TRACY_OPS_TIMES_FILE_NAME,
    TRACY_OPS_DATA_FILE_NAME,
    TRACY_CAPTURE_TOOL,
    TRACY_CSVEXPROT_TOOL,
    generate_logs_folder,
    resolve_tracy_tool_path,
)

import tracy.tracy_state

DEFAULT_CHILD_CALLS = ["CompileProgram", "HWCommandQueue_write_buffer"]
TTNN_SESSION_ID_MESSAGE_PREFIX = "TTNN_SESSION_ID:"
PROFILE_LOG_SESSION_ID_PATTERN = re.compile(r"(?:^|,\s*)SESSION_ID:\s*([^,\r\n]+)")
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def validate_session_id(session_id):
    """Require a compact identifier safe in Tracy and CSV metadata."""
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(f"Session ID must be 1-128 ASCII letters, digits, '.', '_', ':', or '-'; got {session_id!r}")
    return session_id


def extract_ttnn_session_ids(messages_file):
    """Return the non-empty TTNN session IDs in a Tracy message export."""
    session_ids = set()
    with open(messages_file, newline="") as csv_file:
        for row in csv.reader(csv_file, delimiter=";"):
            if not row:
                continue
            message = row[0].strip()
            if message.startswith(TTNN_SESSION_ID_MESSAGE_PREFIX):
                session_id = message.removeprefix(TTNN_SESSION_ID_MESSAGE_PREFIX).strip()
                if session_id:
                    session_ids.add(validate_session_id(session_id))
    return session_ids


def annotate_profile_log_session_id(profile_log, session_id):
    """Append SESSION_ID to the device-log preamble, or verify an existing value.

    Returns ``True`` when the preamble was changed and ``False`` when it already
    contained the requested ID. A different existing ID is an error: silently
    retaining it would pair this performance report with the wrong memory report.
    """
    session_id = validate_session_id(session_id)
    profile_log = os.fspath(profile_log)
    with open(profile_log, "r", newline="") as source:
        preamble = source.readline()
        match = PROFILE_LOG_SESSION_ID_PATTERN.search(preamble)
        if match:
            existing_session_id = validate_session_id(match.group(1).strip())
            if existing_session_id != session_id:
                raise ValueError(
                    f"{profile_log} already contains SESSION_ID: {existing_session_id}, "
                    f"but Tracy contains TTNN_SESSION_ID: {session_id}"
                )
            return False

        with tempfile.NamedTemporaryFile(
            "w", newline="", dir=os.path.dirname(profile_log) or ".", delete=False
        ) as destination:
            temporary_path = destination.name
            destination.write(f"{preamble.rstrip()}, SESSION_ID: {session_id}\n")
            shutil.copyfileobj(source, destination)

    os.chmod(temporary_path, os.stat(profile_log).st_mode)
    os.replace(temporary_path, profile_log)
    return True


def annotate_profile_log_from_tracy_messages(messages_file, profile_log):
    """Stamp a device log when its Tracy export has exactly one TTNN session ID."""
    session_ids = extract_ttnn_session_ids(messages_file)
    if not session_ids:
        logger.warning("No TTNN_SESSION_ID metadata found; device profile log will not be annotated")
        return None
    if len(session_ids) != 1:
        logger.warning(
            "Found multiple TTNN session IDs in one Tracy capture ({}); device profile log will not be annotated",
            ", ".join(sorted(session_ids)),
        )
        return None

    session_id = next(iter(session_ids))
    changed = annotate_profile_log_session_id(profile_log, session_id)
    if changed:
        logger.info(f"Added SESSION_ID: {session_id} to {profile_log}")
    else:
        logger.info(f"Verified existing SESSION_ID: {session_id} in {profile_log}")
    return session_id


def signpost(header, message=None):
    import ttnn

    if message:
        ttnn.tracy_message(f"`TT_SIGNPOST: {header}\n{message}`")
        logger.info(f"{header} : {message} ")
    else:
        ttnn.tracy_message(f"`TT_SIGNPOST: {header}`")
        logger.info(f"{header}")


class Profiler:
    def __init__(self):
        from tracy.tracy_ttnn import tracy_marker_func, tracy_marker_line, finish_all_zones

        self.doProfile = tracy_state.doPartial and sys.gettrace() is None and sys.getprofile() is None
        self.doLine = tracy_state.doLine

        self.lineMarker = tracy_marker_line
        self.funcMarker = tracy_marker_func
        self.finishZones = finish_all_zones

    def enable(self):
        if self.doProfile:
            if self.doLine:
                sys.settrace(self.lineMarker)
            else:
                sys.setprofile(self.funcMarker)

    def disable(self):
        if self.doProfile:
            sys.settrace(None)
            sys.setprofile(None)
            self.finishZones()


def runctx(cmd, globals, locals, partialProfile):
    from tracy.tracy_ttnn import tracy_marker_func, finish_all_zones

    if not partialProfile:
        sys.setprofile(tracy_marker_func)

    try:
        exec(cmd, globals, locals)
    finally:
        sys.setprofile(None)
        finish_all_zones()


def run_report_setup(verbose, outputFolder, binFolder, port):
    logger.info("Verifying tracy profiling tools")
    capture_exe = resolve_tracy_tool_path(binFolder, TRACY_CAPTURE_TOOL)
    csvexport_exe = resolve_tracy_tool_path(binFolder, TRACY_CSVEXPROT_TOOL)
    toolsReady = capture_exe is not None and csvexport_exe is not None

    logsFolder = generate_logs_folder(outputFolder)
    captureProcess = None
    if toolsReady:
        subprocess.run(
            f"rm -rf {logsFolder}; mkdir -p {logsFolder}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        options = ""
        if port:
            options += f"-p {port}"

        captureCommand = (f"{capture_exe} -o {logsFolder / TRACY_FILE_NAME} -f {options}",)
        if verbose:
            logger.info(f"Capture command: {captureCommand}")
            captureProcess = subprocess.Popen(captureCommand, shell=True)
        else:
            captureProcess = subprocess.Popen(
                captureCommand, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    else:
        logger.error(f"Tracy tools were not found. Please make sure you are on a Tracy-enabled build (default).")
        sys.exit(1)

    return captureProcess


def generate_report(
    outputFolder, binFolder, nameAppend, childCalls, collect_noc_traces=False, device_analysis_types=[]
):
    logsFolder = generate_logs_folder(outputFolder)
    tracyOutFile = logsFolder / TRACY_FILE_NAME
    timeOut = 15
    timeCount = 0
    while not os.path.exists(tracyOutFile):
        logger.warning(
            f"tracy capture out not found, will try again in 1 second. Run in verbose (-v) mode to see tracy capture info"
        )
        if timeCount > timeOut:
            logger.error(
                f"tracy capture output file {tracyOutFile} was not generated. Run in verbose (-v) mode to see tracy capture info"
            )
            sys.exit(1)
        timeCount += 1
        time.sleep(1)
    csvexport_exe = resolve_tracy_tool_path(binFolder, TRACY_CSVEXPROT_TOOL)
    if csvexport_exe is None:
        logger.error(f"tracy-csvexport was not found under {binFolder}")
        sys.exit(1)
    with open(logsFolder / TRACY_OPS_TIMES_FILE_NAME, "w") as csvFile:
        childCallStr = ""
        childCallsList = DEFAULT_CHILD_CALLS
        if childCalls:
            childCallsList = list(set(childCalls + DEFAULT_CHILD_CALLS))
        if childCallsList:
            childCallStr = f"-x {','.join(childCallsList)}"
        subprocess.run(
            f"{csvexport_exe} -u -t TT_ {childCallStr} {logsFolder / TRACY_FILE_NAME}",
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    logger.info(f"Host side ops time report generated at {logsFolder / TRACY_OPS_TIMES_FILE_NAME}")

    with open(logsFolder / TRACY_OPS_DATA_FILE_NAME, "w") as csvFile:
        subprocess.run(
            f'{csvexport_exe} -m -s ";" {logsFolder / TRACY_FILE_NAME}',
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    logger.info(f"Host side ops data report generated at {logsFolder / TRACY_OPS_DATA_FILE_NAME}")

    profile_log = logsFolder / PROFILER_DEVICE_SIDE_LOG
    if profile_log.is_file():
        annotate_profile_log_from_tracy_messages(logsFolder / TRACY_OPS_DATA_FILE_NAME, profile_log)

    process_ops(
        outputFolder,
        nameAppend,
        True,
        device_only=False,
        analyze_noc_traces=collect_noc_traces,
        device_analysis_types=device_analysis_types,
        force_legacy_device_logs=False,
    )


def get_available_port():
    ip = socket.gethostbyname(socket.gethostname())

    for port in range(8086, 8500):
        try:
            serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serv.bind((ip, port))
            return str(port)
        except PermissionError as e:
            pass
        except OSError as e:
            pass
    return None


def split_comma_list(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(","))
