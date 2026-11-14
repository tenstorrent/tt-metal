#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/tdp_cfg.py: read-only, for run.sh --tdp (lib/tdp.sh). One line per chip:
#   "<bdf> <tdp_limit W> <max_tdp_limit W>" from each chip's SPI firmware table (cmfwcfg chip_limits), the same read as
# tt-dc-metal-infra tdp-hardcap/cap_max_tdp.py --show. It only reads; it writes nothing to the chip.
import sys

try:
    import pyluwen
except Exception as e:
    sys.exit(f"no pyluwen: {e}")
for c in pyluwen.detect_chips():
    bdf = c.get_pci_bdf()
    bdf = bdf[5:] if bdf.count(":") == 2 else bdf
    try:
        cl = c.as_bh().decode_boot_fs_table("cmfwcfg")["chip_limits"]
        print(f"{bdf} {cl.get('tdp_limit', '-')} {cl.get('max_tdp_limit', '-')}", flush=True)
    except Exception:
        print(f"{bdf} - -", flush=True)
