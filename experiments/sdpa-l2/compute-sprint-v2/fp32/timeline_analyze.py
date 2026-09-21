# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sparse RISC issue endpoints; neither unit-retirement nor exclusive costs."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

STEPS = ["CHUNK_ENTRY", "QK_ENTRY", "QK_EXIT", "DENOM_ENTRY", "DENOM_EXIT", "CHUNK_EXIT"]
path = Path(sys.argv[1])
groups = defaultdict(dict)
with path.open() as stream:
    metadata = next(stream).strip()
    for raw in csv.DictReader(stream):
        row = {k.strip(): v.strip() for k, v in raw.items()}
        if row["zone name"].startswith("FP32_"):
            groups[(row["run host ID"], row["RISC processor type"])][row["zone name"][5:]] = int(row["time[cycles since reset]"])
records = []
for (run, risc), times in groups.items():
    assert set(times) == set(STEPS), (run, risc, times)
    records.append(dict(run=run, risc=risc, cycles={a+"_to_"+b:times[b]-times[a] for a,b in zip(STEPS,STEPS[1:])},
                        total_cycles=times["CHUNK_EXIT"]-times["CHUNK_ENTRY"]))
print(json.dumps(dict(metadata=metadata, warning=__doc__, records=records), indent=2))
