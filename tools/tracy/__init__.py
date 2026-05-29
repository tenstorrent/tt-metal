# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pathlib import Path

from loguru import logger

from .process_ops_logs import process_ops
from .common import (
    TT_METAL_HOME,
    PROFILER_BIN_DIR,
    PROFILER_LOGS_DIR,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_WASM_DIR,
    PROFILER_WASM_TRACE_FILE_NAME,
    PROFILER_WASM_TRACES_DIR,
    PROFILER_CPP_DEVICE_PERF_REPORT,
    PROFILER_DEVICE_SIDE_LOG,
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

        # Write the Tracy binary to a RAM-backed tmpfs (/dev/shm) so it does not
        # compete with the C++ device log (profile_log_device.csv, which can be
        # 10–20 GB) for disk space.  generate_report() reads it there and deletes
        # it immediately after exporting the CSV files.
        shm_binary = Path(f"/dev/shm/tracy_capture_{os.getpid()}.tracy")
        captureCommand = (f"{capture_exe} -o {shm_binary} -f {options}",)
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


def _generate_report_from_cpp_device_perf(logsFolder: Path, outputFolder: Path, nameAppend) -> None:
    """Fast path: generate ops_perf_results_*.csv directly from cpp_device_perf_report.csv.

    When TT_METAL_PROFILER_CPP_POST_PROCESS=1 the TTNN op profiler does not emit
    Tracy messages, so TRACY_CSVEXPROT_TOOL -m produces an empty file and
    import_tracy_op_logs returns no ops.  Rather than exporting the Tracy binary
    (which can be 1+ GB → 20+ GB CSV that takes 35 min to read), we read the
    compact C++ report directly and write a minimal ops_perf_results CSV that
    device_perf_utils.post_process_ops_log can consume.
    """
    import csv as _csv
    import pandas as pd
    from datetime import datetime
    from .common import generate_reports_folder

    cpp_report = logsFolder / PROFILER_CPP_DEVICE_PERF_REPORT
    if not cpp_report.is_file():
        logger.warning(f"cpp_device_perf_report.csv not found at {logsFolder} — cannot generate fast report")
        return

    df = pd.read_csv(cpp_report)

    reportFolder = generate_reports_folder(outputFolder)
    dateStr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    outDir = reportFolder / dateStr
    outDir.mkdir(parents=True, exist_ok=True)
    name = "ops_perf_results"
    if nameAppend:
        name += f"_{nameAppend}"
    name += f"_{dateStr}"
    outPath = outDir / f"{name}.csv"

    # Direct pass-through columns from cpp_device_perf_report.csv → ops_perf_results.csv
    # (source name → output name; same where identical).
    col_map = {
        "OP NAME": "OP CODE",
        "DEVICE ID": "DEVICE ID",
        "DEVICE ARCH": "DEVICE ARCH",
        "CORE COUNT": "CORE COUNT",
        "AVAILABLE WORKER CORE COUNT": "AVAILABLE WORKER CORE COUNT",
        "DEVICE FW DURATION [ns]": "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]": "DEVICE KERNEL DURATION [ns]",
        "DEVICE KERNEL FIRST TO LAST START [ns]": "DEVICE KERNEL FIRST TO LAST START [ns]",
        "DEVICE BRISC KERNEL DURATION [ns]": "DEVICE BRISC KERNEL DURATION [ns]",
        "DEVICE NCRISC KERNEL DURATION [ns]": "DEVICE NCRISC KERNEL DURATION [ns]",
        "DEVICE TRISC0 KERNEL DURATION [ns]": "DEVICE TRISC0 KERNEL DURATION [ns]",
        "DEVICE TRISC1 KERNEL DURATION [ns]": "DEVICE TRISC1 KERNEL DURATION [ns]",
        "DEVICE TRISC2 KERNEL DURATION [ns]": "DEVICE TRISC2 KERNEL DURATION [ns]",
        "DEVICE ERISC KERNEL DURATION [ns]": "DEVICE ERISC KERNEL DURATION [ns]",
        "OP TO OP LATENCY [ns]": "OP TO OP LATENCY [ns]",
        "OP TO OP LATENCY BR/NRISC START [ns]": "OP TO OP LATENCY BR/NRISC START [ns]",
        "METAL TRACE ID": "METAL TRACE ID",
        "METAL TRACE REPLAY SESSION ID": "METAL TRACE REPLAY SESSION ID",
    }
    out_rows = []
    for _, row in df.iterrows():
        # Use "tt_dnn_device" so tt-perf-report recognises these as device ops.
        out_row = {"OP TYPE": "tt_dnn_device", "GLOBAL CALL COUNT": int(row.get("GLOBAL CALL COUNT", 0))}
        for src, dst in col_map.items():
            val = row.get(src)
            out_row[dst] = val if (val is not None and pd.notna(val) and val != "") else "-"

        # String placeholder columns (tt-perf-report uses `in` operator or pd.notna guard)
        for col in [
            "ATTRIBUTES",
            "PARALLELIZATION STRATEGY",
            "HOST START TS",
            "HOST END TS",
            "HOST DURATION [ns]",
            "DEVICE FW START CYCLE",
            "DEVICE FW END CYCLE",
            "DEVICE KERNEL DURATION DM START [ns]",
            "DEVICE KERNEL DURATION PER CORE MIN [ns]",
            "DEVICE KERNEL DURATION PER CORE MAX [ns]",
            "DEVICE KERNEL DURATION PER CORE AVG [ns]",
            "DEVICE COMPUTE CB WAIT FRONT [ns]",
            "DEVICE COMPUTE CB RESERVE BACK [ns]",
            "DISPATCH TOTAL CQ CMD OP TIME [ns]",
            "DISPATCH GO SEND WAIT TIME [ns]",
            "COMPUTE KERNEL SOURCE",
            "COMPUTE KERNEL HASH",
            "DATA MOVEMENT KERNEL SOURCE",
            "DATA MOVEMENT KERNEL HASH",
            "PROGRAM HASH",
            "PROGRAM CACHE HIT",
            "INPUT_0_DATATYPE",
            "INPUT_1_DATATYPE",
            "OUTPUT_0_DATATYPE",
        ]:
            out_row[col] = "-"

        # MATH FIDELITY must be "HiFi4", "HiFi2", or "LoFi" for matmul analysis.
        # Use "HiFi4" as a conservative default (actual fidelity not in device report).
        out_row["MATH FIDELITY"] = "HiFi4"
        # Memory strings: "DRAM" in check and split('_')[-2] both need a string with underscores.
        out_row["INPUT_0_MEMORY"] = "L1_INTERLEAVED"
        out_row["INPUT_1_MEMORY"] = "L1_INTERLEAVED"
        out_row["OUTPUT_0_MEMORY"] = "L1_INTERLEAVED"

        # Shape columns: V2.1 format uses _PAD[LOGICAL] suffix; get_value_physical_logical()
        # calls int() on the value, so use 0 (renders as "0 x 0 x 0" for matmul size).
        for col in [
            "INPUT_0_W_PAD[LOGICAL]",
            "INPUT_0_Z_PAD[LOGICAL]",
            "INPUT_0_Y_PAD[LOGICAL]",
            "INPUT_0_X_PAD[LOGICAL]",
            "INPUT_1_W_PAD[LOGICAL]",
            "INPUT_1_Z_PAD[LOGICAL]",
            "INPUT_1_Y_PAD[LOGICAL]",
            "INPUT_1_X_PAD[LOGICAL]",
            "OUTPUT_0_W_PAD[LOGICAL]",
            "OUTPUT_0_Z_PAD[LOGICAL]",
            "OUTPUT_0_Y_PAD[LOGICAL]",
            "OUTPUT_0_X_PAD[LOGICAL]",
        ]:
            out_row[col] = 0
        out_rows.append(out_row)

    if out_rows:
        fieldnames = list(out_rows[0].keys())
        with open(outPath, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        logger.info(f"Fast C++ perf report generated at {outPath} ({len(out_rows)} ops)")
    else:
        logger.warning("cpp_device_perf_report.csv was empty — no ops_perf_results CSV written")


def generate_report(
    outputFolder, binFolder, nameAppend, childCalls, collect_noc_traces=False, device_analysis_types=[]
):
    logsFolder = generate_logs_folder(outputFolder)
    logsFolder.mkdir(parents=True, exist_ok=True)

    # Delete the raw per-RISC device log immediately — it can be 15–25 GB and is
    # never needed once cpp_device_perf_report.csv exists.
    _raw_log = logsFolder / PROFILER_DEVICE_SIDE_LOG
    _cpp_report = logsFolder / PROFILER_CPP_DEVICE_PERF_REPORT
    if _raw_log.is_file() and _cpp_report.is_file():
        _raw_log.unlink()
        logger.info("Deleted raw device log (cpp_device_perf_report.csv present)")

    if _cpp_report.is_file():
        # When the compact C++ device report is present, the full Tracy binary export
        # would produce a 20 GB timing-zones CSV (171 M rows) that takes 35 min to
        # read.  Instead we export ONLY the Tracy user messages (-m), which contain
        # the TTNN op metadata JSON (shapes, dtypes, math_fidelity, memory layout).
        # These messages are typically a few MB and parse in seconds.
        # The timing-zones export (-u -p TT_) is skipped entirely; device kernel
        # durations come from cpp_device_perf_report.csv instead.
        shm_binary = Path(f"/dev/shm/tracy_capture_{os.getpid()}.tracy")
        tracyOutFile = shm_binary if shm_binary.exists() else logsFolder / TRACY_FILE_NAME

        # Wait briefly for capturetool to finalise the binary.
        timeOut = 15
        timeCount = 0
        while not tracyOutFile.exists():
            logger.warning("Tracy binary not found yet, retrying in 1 s...")
            if timeCount > timeOut:
                logger.error(f"Tracy binary {tracyOutFile} was not generated.")
                break
            timeCount += 1
            time.sleep(1)

        if tracyOutFile.exists():
            # Export only messages (op metadata JSON) — fast, typically KB–MB.
            msgs_path = logsFolder / TRACY_OPS_DATA_FILE_NAME
            with open(msgs_path, "w") as f:
                subprocess.run(
                    f'{binFolder / TRACY_CSVEXPROT_TOOL} -m -s ";" {tracyOutFile}',
                    shell=True,
                    stdout=f,
                    stderr=subprocess.DEVNULL,
                )
            logger.info(f"Op metadata messages exported ({msgs_path.stat().st_size} bytes)")

            # Write an empty times file so import_tracy_op_logs does not return early.
            # Host timing is not needed; device timing comes from cpp_device_perf_report.csv.
            times_path = logsFolder / TRACY_OPS_TIMES_FILE_NAME
            with open(times_path, "w") as f:
                f.write(
                    "name,src_file,src_line,zone_name,zone_text,ns_since_start,exec_time_ns,thread,special_parent_text\n"
                )

            # Delete the binary — it is no longer needed and occupies RAM.
            tracyOutFile.unlink()
            logger.info(f"Deleted Tracy binary {tracyOutFile} after message export")

            # Run the standard pipeline:
            #  import_tracy_op_logs reads the messages → builds ops dict with real metadata
            #  _enrich_ops_from_perf_csv merges with cpp_device_perf_report.csv for timing
            #  generate_reports writes ops_perf_results_*.csv with full shape/dtype/fidelity
            process_ops(
                outputFolder,
                nameAppend,
                True,
                device_only=False,
                analyze_noc_traces=collect_noc_traces,
                device_analysis_types=device_analysis_types,
                force_legacy_device_logs=False,
            )

            # Check if process_ops produced a report; if not (messages were empty),
            # fall back to the fast path that uses placeholders.
            from .common import generate_reports_folder

            report_folder = generate_reports_folder(Path(outputFolder))
            if report_folder.exists() and any(report_folder.iterdir()):
                return  # Success — real metadata in the CSV.

        logger.warning(
            "Tracy messages were empty or binary not found — "
            "falling back to fast C++ report (shapes/dtypes will be placeholders)."
        )
        _generate_report_from_cpp_device_perf(logsFolder, Path(outputFolder), nameAppend)
        return

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
