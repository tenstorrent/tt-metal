# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from shutil import copyfile


from tracy import *
from tracy.serve_wasm import launch_server_subprocess, point_embed_at_trace


# Bit positions match PROFILE_PERF_COUNTERS_* in tt_metal/tools/profiler/perf_counters.hpp.
# l1_2/3/4 are Blackhole-only (its 2-NOC L1 has more client ports).
PERF_COUNTER_GROUP_BITS = {
    "fpu": 0,
    "pack": 1,
    "unpack": 2,
    "l1_0": 3,
    "l1_1": 4,
    "instrn": 5,
    "l1_2": 6,
    "l1_3": 7,
    "l1_4": 8,
}
PERF_COUNTER_L1_GROUPS = {"l1_0", "l1_1", "l1_2", "l1_3", "l1_4"}
# Max counter groups whose readout code fits in one BRISC firmware image. Measured on Blackhole:
# 3 groups fit, 4 overflow .text by 4B, 5 by 16B (each group's readout is ~12B of code). Conservative
# default that is safe on every arch; the L1 mux separately allows at most one L1 bank per pass anyway.
PERF_COUNTER_MAX_GROUPS_PER_PASS = 3
# marker id the firmware tags perf-counter rows with (PERF_COUNTER_PROFILER_ID in perf_counters.hpp).
PERF_COUNTER_MARKER_ID = "9090"
# device-side profiler log filename (PROFILER_DEVICE_SIDE_LOG in tt_metal.tools.profiler.common).
PERF_COUNTER_DEVICE_LOG_NAME = "profile_log_device.csv"


def schedule_perf_counter_passes(requested_groups, max_groups_per_pass=PERF_COUNTER_MAX_GROUPS_PER_PASS):
    """Partition requested counter groups into capture passes (replays of the workload).

    Two hardware/firmware constraints per pass:
      - at most ONE L1 bank (l1_0..l1_4 share a single count-time mux)
      - at most max_groups_per_pass groups (BRISC firmware .text fit)
    Returns a list of passes, each an ordered list of group names.
    """
    import math

    seen = list(dict.fromkeys(g.lower() for g in requested_groups))  # dedup, preserve order
    l1 = [g for g in seen if g in PERF_COUNTER_L1_GROUPS]
    non_l1 = [g for g in seen if g not in PERF_COUNTER_L1_GROUPS]
    total = len(l1) + len(non_l1)
    if total == 0:
        return []
    # Enough passes to give every L1 bank its own pass AND keep each pass within the group cap.
    num_passes = max(len(l1), math.ceil(total / max_groups_per_pass))
    passes = [[] for _ in range(num_passes)]
    for i, g in enumerate(l1):  # one L1 bank per pass
        passes[i].append(g)
    for g in non_l1:  # fill remaining slots, least-full pass first
        target = min((p for p in passes if len(p) < max_groups_per_pass), key=len)
        target.append(g)
    return [p for p in passes if p]


def perf_counter_groups_to_bitfield(groups):
    """OR the PROFILE_PERF_COUNTERS_* bits for a list of group names."""
    bits = 0
    for g in groups:
        bits |= 1 << PERF_COUNTER_GROUP_BITS[g.lower()]
    return bits


def merge_perf_counter_device_logs(pass_csvs, out_csv, base_dir):
    """Merge per-pass device profiler CSVs into one.

    Each pass replayed the same workload, so zone-timing rows are equivalent and ops align by
    (run host ID, trace id, core) via the deterministic global_call_count. Take pass 0 as the base
    (zones + its counter rows) and append ONLY the perf-counter rows (marker id 9090) from later
    passes, so the merged log carries every group's counters against one set of zones.

    All reads/writes are confined to ``base_dir`` (the profiler logs folder): every path is resolved
    to an absolute path and rejected if it escapes ``base_dir``, so a crafted path can never read or
    write outside the profiler output tree.
    """
    base = Path(base_dir).resolve()

    def _confined(p):
        resolved = Path(p).resolve()
        if resolved != base and base not in resolved.parents:
            raise ValueError(f"perf-counter log path escapes {base}: {resolved}")
        return resolved

    merged = list(_confined(pass_csvs[0]).read_text().splitlines(keepends=True))
    for extra in pass_csvs[1:]:
        for line in _confined(extra).read_text().splitlines(keepends=True):
            # column 4 (0-indexed) is timer_id; perf-counter rows use PERF_COUNTER_MARKER_ID.
            fields = line.split(",")
            if len(fields) > 4 and fields[4].strip() == PERF_COUNTER_MARKER_ID:
                merged.append(line)
    _confined(out_csv).write_text("".join(merged))


def main():
    from optparse import OptionParser

    usage = "python3 -m tracy [-m module | scriptfile] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option("-m", dest="module", action="store_true", help="Profile a library module.", default=False)
    parser.add_option("-p", dest="partial", action="store_true", help="Only profile enabled zones", default=False)
    parser.add_option("-l", dest="lines", action="store_true", help="Profile every line of python code", default=False)
    parser.add_option("-r", dest="report", action="store_true", help="Generate ops report", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", help="More info is printed to stdout", default=False)
    parser.add_option(
        "--no-device", dest="device", action="store_false", help="Do not include device data", default=True
    )
    parser.add_option(
        "-o", "--output-folder", action="store", help="Profiler artifacts folder", type="string", dest="output_folder"
    )
    parser.add_option(
        "-n",
        "--name-append",
        action="store",
        help="Custom name to be added to report name",
        type="string",
        dest="name_append",
    )
    parser.add_option(
        "-t", "--port", action="store", help="Internal port used by the script", type="string", dest="port"
    )
    parser.add_option(
        "--web-app-port",
        action="store",
        type="int",
        dest="web_app_port",
        default=None,
        help="HTTP port for the Tracy WASM web UI after capture (default: 8080, or TRACY_WASM_HTTP_PORT if set). WebSocket uses this port + 1.",
    )
    parser.add_option(
        "--no-op-info-cache",
        dest="opInfoCache",
        action="store_false",
        help="Show full op info for cached ops as well",
        default=True,
    )
    parser.add_option(
        "--op-support-count",
        dest="op_support_count",
        action="store",
        help="Maximum number of ops that can be supported by the profiler",
        type="int",
    )
    parser.add_option(
        "--child-functions",
        type="string",
        help="Comma separated list of child function to have their duration included for parent OPs",
        action="callback",
        callback=split_comma_list,
    )
    parser.add_option(
        "--process-logs-only",
        dest="processLogsOnly",
        action="store_true",
        help="Only process the logs available in the default logs folder",
        default=False,
    )
    parser.add_option(
        "--profile-dispatch-cores",
        dest="profile_dispatch_cores",
        action="store_true",
        help="Collect dispatch cores profiling data",
        default=False,
    )
    parser.add_option(
        "--enable-sum-profiling",
        dest="do_sum",
        action="store_true",
        help="Enable sum profiling",
        default=False,
    )
    parser.add_option(
        "--no-runtime-analysis",
        dest="no_runtime_analysis",
        action="store_true",
        help="Disable C++ post-processing of profiling data (enabled by default)",
        default=False,
    )
    parser.add_option(
        "--sync-host-device",
        dest="sync_host_device",
        action="store_true",
        help="Sync host with all devices",
        default=False,
    )
    parser.add_option(
        "--device-trace-profiler",
        dest="device_trace_profiler",
        action="store_true",
        help="Profile device side trace durations",
        default=[],
    )
    parser.add_option(
        "--device-memory-profiler",
        dest="device_memory_profiler",
        action="store_true",
        help="Profile allocated device L1 and DRAM memory buffers",
        default=False,
    )
    parser.add_option(
        "--dump-device-data-mid-run",
        dest="mid_run_device_data",
        action="store_true",
        help="Dump collected device data to files and push to Tracy GUI mid-run",
        default=False,
    )
    parser.add_option(
        "--disable-device-data-dump-to-files",
        dest="disable_device_data_dump_to_files",
        action="store_true",
        help="Disable dumping collected device data to files",
        default=False,
    )
    parser.add_option(
        "--disable-device-data-push-to-tracy",
        dest="disable_device_data_push_to_tracy",
        action="store_true",
        help="Disable pushing collected device data to Tracy GUI",
        default=False,
    )
    parser.add_option(
        "--collect-noc-traces",
        dest="collect_noc_traces",
        action="store_true",
        help="Collect noc event traces when profiling",
        default=False,
    )
    parser.add_option(
        "--check-exit-code",
        dest="check_exit_code",
        action="store_true",
        help="Exit the run and do not attempt post processing if the test command fails",
        default=False,
    )
    parser.add_option(
        "-a",
        "--device-analysis-types",
        dest="device_analysis_types",
        action="append",
        help="List of device analysis types",
        default=[],
    )
    parser.add_option(
        "--tracy-tools-folder", dest="binary_folder", action="store", help="Tracy tools folder", type="string"
    )
    parser.add_option(
        "--profiler-capture-perf-counters",
        type="string",
        help="Comma-separated list of performance counter groups to capture: fpu, pack, unpack, l1_0..l1_4, instrn, all",
        action="callback",
        callback=split_comma_list,
        dest="perf_counter_groups",
    )
    parser.add_option(
        "--perf-counter-multipass",
        dest="perf_counter_multipass",
        action="store_true",
        default=False,
        help="When the requested counter groups don't fit one pass (>1 L1 bank, or too many groups for "
        "BRISC firmware), replay the workload once per scheduled pass and merge results. Without this, "
        "such a request errors with the required pass plan.",
    )
    parser.add_option(
        "--no-capture-tool", dest="noCapture", action="store_true", help="Do not run Tracy capture tool", default=False
    )

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    originalArgs = sys.argv.copy()

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    outputFolderEnvStr = "TT_METAL_PROFILER_DIR"
    outputFolder = PROFILER_ARTIFACTS_DIR
    if options.output_folder:
        logger.info(f"Setting profiler artifacts folder to {options.output_folder}")
        Path(options.output_folder).mkdir(parents=True, exist_ok=True)
        os.environ["TT_METAL_PROFILER_DIR"] = options.output_folder
        outputFolder = Path(options.output_folder)

    binaryFolder = PROFILER_BIN_DIR
    if options.binary_folder:
        logger.info(f"Setting tracy tool folder to {options.binary_folder}")
        binaryFolder = Path(options.binary_folder)
        if not binaryFolder.exists():
            logger.error(f"Tracy tools folder {options.binary_folder} does not exist")
            sys.exit(1)

    if options.processLogsOnly:
        generate_report(generate_logs_folder(outputFolder), binaryFolder, "", None, options.collect_noc_traces)
        sys.exit(0)

    if options.port:
        port = options.port
    else:
        port = get_available_port()

    if options.mid_run_device_data:
        if options.device_trace_profiler:
            logger.error("Cannot use --dump-device-data-mid-run and --device-trace-profiler together")
            sys.exit(1)
        if options.profile_dispatch_cores:
            logger.error("Cannot use --dump-device-data-mid-run and --profile-dispatch-cores together")
            sys.exit(1)

    opInfoCacheStr = "TT_METAL_PROFILER_NO_CACHE_OP_INFO"
    if options.opInfoCache:
        if opInfoCacheStr in os.environ.keys():
            del os.environ[opInfoCacheStr]
    else:
        os.environ[opInfoCacheStr] = "1"

    if options.profile_dispatch_cores:
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "1"

    if options.do_sum:
        os.environ["TT_METAL_PROFILER_SUM"] = "1"

    if options.mid_run_device_data:
        os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"

    if options.disable_device_data_dump_to_files:
        os.environ["TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES"] = "1"

    if options.disable_device_data_push_to_tracy:
        os.environ["TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY"] = "1"

    if options.sync_host_device:
        os.environ["TT_METAL_PROFILER_SYNC"] = "1"

    if options.device_trace_profiler:
        os.environ["TT_METAL_TRACE_PROFILER"] = "1"

    if options.collect_noc_traces:
        os.environ["TT_METAL_DEVICE_PROFILER_NOC_EVENTS"] = "1"
        os.environ["TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH"] = str(
            generate_logs_folder(os.path.abspath(outputFolder))
        )

    # Only the outer (capture) process schedules/validates. The inner "--no-capture-tool" run that
    # executes the workload just honors the TT_METAL_PROFILE_PERF_COUNTERS mask inherited via env
    # (set per-pass by the multi-pass loop below), so it must not re-run this block.
    if options.perf_counter_groups and not options.noCapture:
        # Detect device arch: Blackhole has L1 banks 2-4 (2-NOC), WH/GS only 0-1.
        declared_arch = next(
            (os.environ.get(v) for v in ("TT_METAL_DEVICE_ARCH", "TT_ARCH_NAME", "ARCH_NAME") if os.environ.get(v)),
            None,
        )
        if declared_arch is None:
            try:
                import ttnn

                device = ttnn.open_device(device_id=0)
                declared_arch = str(device.arch()).split(".")[-1]
                ttnn.close_device(device)
            except Exception:
                logger.debug("Failed to detect device arch via ttnn")
        is_blackhole = declared_arch is not None and declared_arch.strip().lower() == "blackhole"

        # Resolve requested group names; "all" expands to the arch's full set.
        arch_l1 = ["l1_0", "l1_1", "l1_2", "l1_3", "l1_4"] if is_blackhole else ["l1_0", "l1_1"]
        resolved = []
        for group in options.perf_counter_groups:
            g = group.lower()
            if g == "all":
                resolved = ["fpu", "pack", "unpack", "instrn"] + arch_l1
                break
            elif g in PERF_COUNTER_GROUP_BITS:
                resolved.append(g)
            else:
                logger.warning(
                    f"Unknown counter group '{group}'. " f"Valid groups: {', '.join(PERF_COUNTER_GROUP_BITS)}, all"
                )
        resolved = list(dict.fromkeys(resolved))

        # Reject BH-only groups on non-BH architectures.
        bh_only = sorted(set(resolved) & {"l1_2", "l1_3", "l1_4"})
        if bh_only and not is_blackhole:
            raise ValueError(
                f"Performance counter groups {', '.join(bh_only)} are supported only on Blackhole, "
                f"but device arch is {declared_arch or 'undeclared'}."
            )

        # Schedule into passes: L1 banks share one mux (<=1 L1/pass), BRISC firmware fits a limited
        # number of groups/pass. A single pass sets the mask directly; multiple passes need opt-in.
        passes = schedule_perf_counter_passes(resolved)
        if len(passes) <= 1:
            bitfield = perf_counter_groups_to_bitfield(resolved)
            if bitfield > 0:
                os.environ["TT_METAL_PROFILE_PERF_COUNTERS"] = str(bitfield)
                logger.info(f"Setting performance counter groups: {resolved} (bitfield: {bitfield})")
        else:
            plan = "\n".join(
                f"  pass {i + 1}: {', '.join(p)}  (bitfield {perf_counter_groups_to_bitfield(p)})"
                for i, p in enumerate(passes)
            )
            if not options.perf_counter_multipass:
                raise ValueError(
                    f"Requested counter groups {resolved} need {len(passes)} capture passes "
                    f"(L1 banks share one mux; BRISC firmware fits <= {PERF_COUNTER_MAX_GROUPS_PER_PASS} "
                    f"groups/pass):\n{plan}\n"
                    "Re-run with --perf-counter-multipass to replay the workload once per pass and merge "
                    "the results, or request fewer groups."
                )
            options.perf_counter_pass_bitfields = [perf_counter_groups_to_bitfield(p) for p in passes]
            logger.info(f"Multi-pass perf-counter capture ({len(passes)} passes):\n{plan}")

    if not (
        options.no_runtime_analysis or options.do_sum or options.profile_dispatch_cores or options.perf_counter_groups
    ):
        os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"
    else:
        reasons = []
        if options.no_runtime_analysis:
            reasons.append("--no-runtime-analysis")
        if options.do_sum:
            reasons.append("--enable-sum-profiling")
        if options.profile_dispatch_cores:
            reasons.append("--profile-dispatch-cores")
        if options.perf_counter_groups:
            reasons.append("--profiler-capture-perf-counters")

        reason_str = ", ".join(reasons)
        logger.warning(
            f"Skipping runtime analysis (C++ post-processing) due to conflicting options ({reason_str}). Falling back to legacy Python processing."
        )

    if options.device_memory_profiler:
        os.environ["TT_METAL_MEM_PROFILER"] = "1"

    if options.op_support_count:
        os.environ["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = str(options.op_support_count)

    if len(args) > 0:
        if options.noCapture:
            code = None
            if options.report:
                os.environ["TTNN_OP_PROFILER"] = "1"
                os.environ["TT_METAL_PROFILER_TRACE_TRACKING"] = "1"
            if options.module:
                import runpy

                code = "run_module(modname, run_name='__main__')"
                globs = {
                    "run_module": runpy.run_module,
                    "modname": args[0],
                }
            else:
                trySystem = False
                try:
                    progname = args[0]
                    sys.path.insert(0, os.path.dirname(progname))
                    with io.open_code(progname) as fp:
                        code = compile(fp.read(), progname, "exec")
                    spec = importlib.machinery.ModuleSpec(name="__main__", loader=None, origin=progname)
                    globs = {
                        "__spec__": spec,
                        "__file__": spec.origin,
                        "__name__": spec.name,
                        "__package__": None,
                        "__cached__": None,
                    }
                except (ValueError, SyntaxError) as exc:
                    trySystem = True
                if trySystem:
                    subprocess.run(" ".join(args), shell=True, check=True)

            if options.partial:
                tracy_state.doPartial = True

            if options.lines:
                tracy_state.doLine = True

            try:
                if code:
                    runctx(code, globs, None, options.partial)
            except BrokenPipeError as exc:
                # Prevent "Exception ignored" during interpreter shutdown.
                sys.stdout = None
                sys.exit(exc.errno)
        else:
            if not port:
                logger.error("No available port found")
                sys.exit(1)
            logger.info(f"Using port {port}")
            captureProcess = run_report_setup(options.verbose, outputFolder, binaryFolder, port)

            originalArgs = ["--no-capture-tool"] + originalArgs[1:]
            osCmd = " ".join(originalArgs)

            testCommand = f"{sys.executable} -m tracy {osCmd}"

            envVars = dict(os.environ)
            if options.device:
                envVars["TT_METAL_DEVICE_PROFILER"] = "1"
            elif "TT_METAL_DEVICE_PROFILER" in envVars.keys():
                del envVars["TT_METAL_DEVICE_PROFILER"]

            if port:
                envVars["TRACY_PORT"] = port

            # Multi-pass perf-counter capture replays the workload once per scheduled pass (each with
            # its own group mask) and merges the per-pass device logs. Single pass runs once as before.
            pass_bitfields = getattr(options, "perf_counter_pass_bitfields", None)
            proc_holder = {"p": None}

            def signal_handler(sig, frame):
                if proc_holder["p"] is not None:
                    os.killpg(os.getpgid(proc_holder["p"].pid), signal.SIGTERM)
                captureProcess.terminate()
                captureProcess.communicate()
                sys.exit(3)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            def run_workload(env):
                proc = subprocess.Popen([testCommand], shell=True, env=env, preexec_fn=os.setsid)
                proc_holder["p"] = proc
                logger.info("Test process started")
                proc.communicate()
                if options.check_exit_code and proc.returncode != 0:
                    logger.error(f"{testCommand} exited with a non-zero return code")
                    sys.exit(4)

            if pass_bitfields and len(pass_bitfields) > 1:
                device_log = generate_logs_folder(outputFolder) / PERF_COUNTER_DEVICE_LOG_NAME
                pass_dir = generate_logs_folder(outputFolder) / "perf_counter_passes"
                pass_dir.mkdir(parents=True, exist_ok=True)
                pass_logs = []
                for i, bitfield in enumerate(pass_bitfields):
                    logger.info(f"Perf-counter pass {i + 1}/{len(pass_bitfields)} (bitfield {bitfield})")
                    if device_log.is_file():
                        device_log.unlink()  # fresh per pass so each snapshot holds only that pass
                    pass_env = dict(envVars)
                    pass_env["TT_METAL_PROFILE_PERF_COUNTERS"] = str(bitfield)
                    run_workload(pass_env)
                    if device_log.is_file():
                        snap = pass_dir / f"pass_{i}.csv"
                        copyfile(device_log, snap)
                        pass_logs.append(snap)
                    else:
                        logger.warning(f"Device log missing after perf-counter pass {i + 1}: {device_log}")
                if pass_logs:
                    merge_perf_counter_device_logs(pass_logs, device_log, generate_logs_folder(outputFolder))
                    logger.info(f"Merged {len(pass_logs)} perf-counter pass logs into {device_log}")
            else:
                run_workload(envVars)

            try:
                captureProcess.communicate(timeout=15)
                # Copy the generated .tracy file to the server's traces folder with a unique name
                import datetime

                tracy_src = PROFILER_LOGS_DIR / TRACY_FILE_NAME
                traces_dir = PROFILER_WASM_TRACES_DIR
                # Use timestamp, optional name_append, and a short form of the tested command for uniqueness
                timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
                name_part = f"_{options.name_append}" if options.name_append else ""
                # Short form of the command being tested (first arg, basename, no extension)
                cmd_short = ""
                if len(args) > 0:
                    if os.path.basename(args[0]) == "pytest" and len(args) > 1:
                        # Use the next argument after pytest for the name
                        cmd_base = os.path.basename(args[-1])
                    else:
                        cmd_base = os.path.basename(args[0])
                    cmd_short = os.path.splitext(cmd_base)[0]
                    # Sanitize for filename
                    cmd_short = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cmd_short)
                    cmd_short = f"{cmd_short}"
                tracy_dst = traces_dir / f"{cmd_short}{name_part}{timestamp}.tracy"
                logger.info(f"Copying {tracy_src} to {tracy_dst}")
                try:
                    copyfile(tracy_src, tracy_dst)
                    logger.info(f"Copied {tracy_src} to {tracy_dst}")
                except Exception as e:
                    logger.warning(f"Could not copy {tracy_src} to {tracy_dst}: {e}")
                # Point embed.tracy (always a relative symlink into traces/) at the new capture so
                # the GUI loads it by default. Symlink-only: the live-reload watcher follows it and
                # DELETE relies on it to detect/advance the active trace. On the rare FS without
                # symlink support this warns and skips; the trace is still reachable via ?trace=.
                try:
                    point_embed_at_trace(tracy_dst.name)
                    logger.info(f"embed.tracy -> traces/{tracy_dst.name}")
                except Exception as e:
                    logger.warning(f"Could not update embed.tracy: {e}")
                launch_server_subprocess(port=options.web_app_port)
                # Start the WASM server as a daemon with defaults
                if options.report:
                    generate_report(
                        outputFolder,
                        binaryFolder,
                        options.name_append,
                        options.child_functions,
                        options.collect_noc_traces,
                        options.device_analysis_types,
                    )
            except subprocess.TimeoutExpired as e:
                captureProcess.terminate()
                captureProcess.communicate()
                logger.error(
                    f"No profiling data could be captured. Please make sure you are on a Tracy-enabled build (default)."
                )
                sys.exit(1)

    else:
        parser.print_usage()
    return parser


# When invoked as main program, invoke the profiler on a script
if __name__ == "__main__":
    main()
