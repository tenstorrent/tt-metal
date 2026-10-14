"""
Usage:
    dump_craq_router [--craq-streams]

Options:
    --craq-streams   Also dump non-zero stream scratch / space-available registers (streams 0..63).

Description:
    Dumps the CRAQ fabric router kernel's file-scope diagnostic counters (diag_*) from every active-eth erisc
    that is running a fabric_erisc_router_* kernel, plus its waypoint. Used to localize lost credits/packets
    when a fabric test hangs (callstacks alone show idle routers).

Owner:
    snijjar
"""

import os

from ttexalens.context import Context
from ttexalens.hardware.risc_debug import RiscDebug
from ttexalens.memory_access import create_memory_access
from ttexalens.tt_exalens_lib import read_word_from_device
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache, ElfsCache
from run_checks import run as get_run_checks
from triage import ScriptConfig, log_check_location, run_script

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)

DIAG_VARS = [
    "diag_ch1_sends",
    "diag_ch1_completions",
    "diag_ch1_credit_returns",
    "diag_forwards",
    "diag_src1_arrivals",
    "diag_arrivals",
    "g_my_x",
    "g_my_y",
]

NOC_OVERLAY_START_ADDR = 0xFFB40000
NOC_STREAM_REG_SPACE_SIZE = 0x1000
STREAM_REMOTE_SRC_REG_INDEX = 4  # scratch register (read_stream_scratch_register)
STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX = 297  # get_ptr_val


def stream_reg_addr(stream_id: int, reg_index: int) -> int:
    return NOC_OVERLAY_START_ADDR + stream_id * NOC_STREAM_REG_SPACE_SIZE + (reg_index << 2)


def dump_router(risc_debug: RiscDebug, dispatcher_data: DispatcherData, elfs_cache: ElfsCache, want_streams: bool, context: Context):
    risc_name = risc_debug.risc_location.risc_name
    location = risc_debug.risc_location.location
    try:
        core_data = dispatcher_data.get_cached_core_data(risc_debug.risc_location)
    except Exception as e:  # noqa: BLE001
        return
    kernel_path = core_data.kernel_path
    if kernel_path is None or "fabric_erisc_router" not in kernel_path:
        return
    message = f"{risc_name} kernel={core_data.kernel_name} waypoint={core_data.waypoint}: "
    try:
        kernel_elf = elfs_cache[kernel_path]
        mem = create_memory_access(risc_debug)
        vals = []
        for var in DIAG_VARS:
            try:
                vals.append(f"{var}={int(kernel_elf.get_global(var, mem).read_value())}")
            except Exception as e:  # noqa: BLE001
                vals.append(f"{var}=?({type(e).__name__})")
        message += " ".join(vals)
    except Exception as e:  # noqa: BLE001
        message += f"elf/mem error: {e}"
    if want_streams:
        regs = []
        for sid in range(64):
            try:
                scratch = read_word_from_device(location, stream_reg_addr(sid, STREAM_REMOTE_SRC_REG_INDEX), context=context)
                avail = read_word_from_device(
                    location, stream_reg_addr(sid, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX), context=context
                )
            except Exception as e:  # noqa: BLE001
                regs.append(f"s{sid}:err")
                break
            if scratch or avail:
                regs.append(f"s{sid}:scratch={scratch:#x},avail={avail}")
        message += "\n    streams: " + (" ".join(regs) if regs else "(all zero)")
    log_check_location(location, True, message)


def run(args, context: Context):
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    want_streams = os.environ.get("CRAQ_TRIAGE_STREAMS", "1") != "0"
    try:
        want_streams = want_streams or bool(args["--craq-streams"])
    except Exception:  # noqa: BLE001
        pass
    run_checks.run_per_core_check(
        lambda risc_debug: dump_router(risc_debug, dispatcher_data, elfs_cache, want_streams, context),
        block_filter=["active_eth"],
        core_filter=["erisc0", "erisc1"],
    )


if __name__ == "__main__":
    run_script()
