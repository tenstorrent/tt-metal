"""Decode this capture offline: stdlib and addr2line only, with no device imports.

Mailbox offsets are from the exact watcher_noinline_stage5 firmware DWARF:
mailboxes_t@0x100, watcher@1232, assert_status@80, launch@16 (144 bytes),
aerisc_run_flag@2312, subordinate_sync@8. These are capture-specific offsets,
not a general mailbox ABI. Kernel text relocation uses saved launch metadata.
"""

import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "tt_metal/hw").is_dir())
ADDR2LINE = ROOT / "runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
metadata = (DOC / "watcher_full_eth_noinline/generated/inspector/kernels.yaml").read_text()
kernel_paths = {
    int(kid): Path(path) for kid, path in re.findall(r"watcher_kernel_id: (\d+)\n.*?\n    path: (.*?)\n", metadata)
}
pc_records = json.loads((HERE / "pc_samples.json").read_text())
result = []
for capture in json.loads((HERE / "l1_manifest.json").read_text()):
    raw = (HERE / capture["path"]).read_bytes()
    assert len(raw) == capture["bytes"]
    assert hashlib.sha256(raw).hexdigest() == capture["sha256"]

    def u32(address):
        return struct.unpack_from("<I", raw, address)[0]

    launch_index = u32(268)
    assert launch_index < 8
    launch = 272 + 144 * launch_index
    ids = struct.unpack_from("<5H", raw, launch + 124)
    bases = struct.unpack_from("<4I", raw, launch)
    text_offsets = struct.unpack_from("<5I", raw, launch + 52)
    line, tripped, which, claim, hw_fault = struct.unpack_from("<HBBIQ", raw, 1568)
    item = {
        "device": capture["device"],
        "translated": capture["translated"],
        "assert": {"line": line, "type": tripped, "which": which, "claim": claim, "hw_fault": hw_fault},
        "waypoints": [raw[1492 + 4 * i : 1496 + 4 * i].rstrip(b"\0").decode() for i in range(5)],
        "aerisc_run_flag": u32(2568),
        "subordinate_sync": u32(264),
        "go_signal": raw[1427],
        "kernel_ids": list(ids[:2]),
        "launch_index": launch_index,
        "erisc": [],
    }
    for risc in range(2):
        kernel = kernel_paths[ids[risc]]
        args = {
            key: int(value)
            for key, value in re.findall(r'\{"([^"]+)",(\d+)\}', (kernel / "named_ct_arg_map_generated.h").read_text())
        }
        assert args["MY_ERISC_ID"] == risc and args["NUM_ACTIVE_ERISCS"] == 2
        used_nocs = set()
        for n in range(args["NUM_SENDER_CHANNELS"]):
            if args[f"IS_SENDER_CHANNEL_{n}_SERVICED"]:
                used_nocs.add(args[f"SENDER_CH_{n}_ACK_NOC_ID"])
        for n in range(args["NUM_RECEIVER_CHANNELS"]):
            if args[f"IS_RECEIVER_CHANNEL_{n}_SERVICED"]:
                used_nocs.update([args[f"RX_CH_{n}_FWD_NOC_ID"], args[f"RX_CH_{n}_LOCAL_WRITE_NOC_ID"]])
        assert used_nocs == {risc}
        pcs = next(
            entry["pc"]
            for entry in pc_records
            if entry["device"] == capture["device"]
            and entry["translated"] == capture["translated"]
            and entry["risc"] == f"erisc{risc}"
        )
        kind = "active_erisc" if risc == 0 else "subordinate_active_erisc"
        if risc == 0:
            elf = kernel.parents[2] / "firmware/active_erisc/active_erisc.elf"
            translated_pcs = sorted(set(pcs))
        else:
            elf = kernel / kind / f"{kind}.elf"
            data = elf.read_bytes()
            assert data[:6] == b"\x7fELF\x01\x01"
            entry = struct.unpack_from("<I", data, 24)[0]
            translated_pcs = sorted({pc - bases[1] - text_offsets[risc] + entry for pc in pcs})
        symbols = subprocess.check_output(
            [str(ADDR2LINE), "-e", str(elf), "-fiaC", *map(hex, translated_pcs)], text=True
        )
        item["erisc"].append(
            {
                "id": risc,
                "owned_nocs": sorted(used_nocs),
                "pcs": list(map(hex, pcs)),
                "elf": str(elf),
                "elf_pcs": list(map(hex, translated_pcs)),
                "symbols": symbols,
            }
        )
        if risc == 0:
            item["termination_signal"] = u32(args["TERMINATION_SIGNAL_ADDR"])
            item["edm_status"] = hex(u32(args["EDM_STATUS_PTR_ADDR"]))
    result.append(item)

(HERE / "decoded_state.json").write_text(json.dumps(result, indent=2) + "\n")
for item in result:
    print(
        item["device"],
        item["translated"],
        item["assert"],
        item["waypoints"][:2],
        "subsync",
        hex(item["subordinate_sync"]),
        "termination",
        item["termination_signal"],
        "status",
        item["edm_status"],
    )
