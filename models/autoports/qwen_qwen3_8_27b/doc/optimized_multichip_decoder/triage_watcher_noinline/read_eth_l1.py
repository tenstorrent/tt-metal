"""Post-abort read-only Ethernet L1 capture; no reset or kernel launch."""
import hashlib
import json
import re
from pathlib import Path

from ttexalens.context import NocId
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_init import init_ttexalens

out = Path(__file__).resolve().parent
log = out.parent / "watcher_full_eth_noinline/generated/watcher/watcher.log"
coords = sorted(set(re.findall(r"Device (\d+) acteth .*?virtual\(x=\s*(\d+),y=\s*(\d+)\)", log.read_text())))
context = init_ttexalens(noc_id=NocId.NOC1)
records = []
for chip, x, y in coords:
    device = context.devices[int(chip)]
    coord = OnChipCoordinate(int(x), int(y), "translated", device)
    data = bytearray(128 * 1024)
    device.noc_read(coord, 0, data, noc_id=NocId.NOC1)
    path = out / f"device{chip}_eth{x}_{y}_l1.bin"
    path.write_bytes(data)
    records.append(
        {
            "device": int(chip),
            "translated": [int(x), int(y)],
            "path": path.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    print(records[-1], flush=True)
(out / "l1_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
