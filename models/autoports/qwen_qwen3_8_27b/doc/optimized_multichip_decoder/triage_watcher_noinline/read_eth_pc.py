"""Sample Ethernet debug-bus PCs without halting/resetting the RISC cores."""
import json
from pathlib import Path

from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_init import init_ttexalens

out = Path(__file__).resolve().parent
context = init_ttexalens()
records = []
for entry in json.loads((out / "l1_manifest.json").read_text()):
    device = context.devices[entry["device"]]
    coord = OnChipCoordinate(*entry["translated"], "translated", device)
    block = device.get_block(coord)
    for risc in block.all_riscs:
        assert risc.debug_bus_pc_signal is not None, "Do not fall back to halting"
        records.append(
            {
                "device": entry["device"],
                "translated": entry["translated"],
                "risc": risc.risc_info.risc_name,
                "pc": [risc.get_pc() for _ in range(4)],
            }
        )
        print(records[-1], flush=True)
(out / "pc_samples.json").write_text(json.dumps(records, indent=2) + "\n")
