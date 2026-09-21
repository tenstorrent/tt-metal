# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only selected candidate ELF inspection; never imports TTNN or opens a device."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[4]
objdump = root / "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
cache = Path("/localdev/cglagovich/compute-sprint-20260918/jit-cache")
records = []
for kernel in cache.glob("*/kernels/compute/*"):
    source, defines = kernel / "chlkc_unpack.cpp", kernel / "defines_generated.h"
    if not source.exists() or not defines.exists():
        continue
    if "/compute-sprint-v3/compensated/compute.cpp" not in source.read_text():
        continue
    if "compensated/group2/compute_streaming.hpp" not in defines.read_text():
        continue
    for elf in kernel.glob("trisc2/*.elf"):
        if ".xip." in elf.name:
            continue
        asm = subprocess.check_output([str(objdump), "-d", "-C", str(elf)], text=True)
        functions = []
        lines = asm.splitlines()
        instructions = [line for line in lines if "\t" in line]
        mad_round_windows = ["\n".join(instructions[max(0, i-8):i+10])
                             for i, line in enumerate(instructions[:-1])
                             if "sfpmad" in line and "sfpstochrnd" in instructions[i+1]]
        for i, line in enumerate(lines):
            if line.endswith(">:") and "group2" in line:
                j = i + 1
                while j < len(lines) and not lines[j].endswith(">:"):
                    j += 1
                functions.append("\n".join(lines[i:j]))
        records.append(dict(elf=str(elf), sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                            defines=defines.read_text(), functions=functions,
                            mad_round_windows=mad_round_windows))
print(json.dumps(records, indent=2))
