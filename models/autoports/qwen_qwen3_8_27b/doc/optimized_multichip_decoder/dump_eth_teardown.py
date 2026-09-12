#!/usr/bin/env python3
"""Parent-owned post-abort capture; initializes ExaLens and reads device state.

No Inspector RPC is required. Never run concurrently with a device workload.
No reset, halt, resume, firmware load, or private-memory access is performed.
PC sampling uses the debug-bus selector registers; L1 is only read.
"""

import argparse
import json
from pathlib import Path

from ttexalens.coordinate import OnChipCoordinate
from ttexalens.elf import read_elf
from ttexalens.memory_access import create_l1_memory_access
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_from_device, read_word_from_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cores", default="29,24,28,22", help="Virtual x coordinates at y25")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context = init_ttexalens()
    device = context.devices[args.device]
    firmware = read_elf(context.file_api, str(args.firmware))
    for x in [int(value) for value in args.cores.split(",")]:
        location = OnChipCoordinate(x, 25, "translated", device, "eth")
        result = {"device": args.device, "virtual_core": [x, 25]}

        def sample(name, callback):
            try:
                result[name] = callback()
            except Exception as exc:
                result[name] = {"error": str(exc)}

        # Preserve raw mailbox and router metadata before decoding fields.
        for name, address, size in [("mailbox", 256, 12768), ("router_metadata", 88896, 1024)]:
            try:
                data = read_from_device(location, address, num_bytes=size, context=context)
                (args.output / f"d{args.device}_eth{x}-25_{name}.bin").write_bytes(data)
                result[name + "_capture"] = {"address": address, "bytes": len(data)}
            except Exception as exc:
                result[name + "_capture"] = {"error": str(exc)}

        for name, address in [
            ("heartbeat", 0x7CC70),
            ("termination", 89072),
            ("edm_status", 89104),
            ("retrain_sync_stream30", 0xFFB40000 + 30 * 0x1000 + 4 * 4),
            ("teardown_sync_stream31", 0xFFB40000 + 31 * 0x1000 + 4 * 4),
        ]:
            sample(name, lambda address=address: hex(read_word_from_device(location, address, context=context)))

        # This calls the explicit debug-bus path, never get_pc's halt fallback.
        for risc_name in ["erisc0", "erisc1"]:
            sample(
                risc_name + "_pc_samples",
                lambda risc_name=risc_name: [
                    hex(int(location.noc_block.debug_bus.read_signal(risc_name + "_pc"))) for _ in range(3)
                ],
            )

        try:
            mailbox = firmware.read_global("mailboxes", create_l1_memory_access(location))
            sample("assert_tripped", lambda: int(mailbox.watcher.assert_status.tripped))
            sample("assert_line", lambda: int(mailbox.watcher.assert_status.line_num))
            sample("assert_risc", lambda: int(mailbox.watcher.assert_status.which))
            sample("aerisc_run_flag", lambda: int(mailbox.aerisc_run_flag))
            sample("subordinate_sync", lambda: [int(mailbox.subordinate_sync.map[i]) for i in range(4)])
            sample("go_signal", lambda: int(mailbox.go_messages[0].signal))
            sample("launch_read_pointer", lambda: int(mailbox.launch_msg_rd_ptr))
            sample(
                "waypoints",
                lambda: [mailbox.watcher.debug_waypoint[i].waypoint.read_bytes().hex() for i in range(2)],
            )
            pointer = int(mailbox.launch_msg_rd_ptr)
            launch = mailbox.launch[pointer].kernel_config
            sample("exit_erisc_kernel", lambda: int(launch.exit_erisc_kernel))
            sample("config_base", lambda: int(launch.kernel_config_base[1]))
            sample("kernel_text_offsets", lambda: [int(launch.kernel_text_offset[i]) for i in range(2)])
            sample("watcher_kernel_ids", lambda: [int(launch.watcher_kernel_ids[i]) for i in range(2)])
        except Exception as exc:
            result["mailbox_decode_error"] = str(exc)
        (args.output / f"d{args.device}_eth{x}-25.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
