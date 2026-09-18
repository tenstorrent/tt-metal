"""Lifecycle proof for the direct native readiness gate, which does not use pytest fixtures."""

import re


def clean_lifecycle(log):
    lines = log.splitlines()
    drivers = [i for i, l in enumerate(lines) if "Opening user mode device driver" in l]
    inventories = []
    fabrics = []
    closes = [i for i, l in enumerate(lines) if "Closing devices in cluster completed." in l]
    for i, line in enumerate(lines):
        match = re.search(r"Opening local chip ids/PCIe ids: \{([^}]*)\}", line)
        if match:
            values = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
            if sorted(values) != list(range(32)):
                return None
            inventories.append(i)
        match = re.search(r"Fabric initialized on (\d+) devices", line)
        if match:
            if int(match.group(1)) != 32:
                return None
            fabrics.append(i)
    if not all(len(rows) == 1 for rows in [drivers, inventories, fabrics, closes]):
        return None
    if not drivers[0] < inventories[0] < fabrics[0] < closes[0]:
        return None
    opens = [
        i
        for i, l in enumerate(lines)
        if any(
            x in l
            for x in [
                "Opening user mode device driver",
                "Opening local chip ids/PCIe ids:",
                "multidevice with",
                "Fabric initialized on",
            ]
        )
    ]
    if max(opens) >= closes[0]:
        return None
    return lines[closes[0]]
