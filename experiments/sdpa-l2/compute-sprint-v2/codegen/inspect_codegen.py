# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only binding/toolchain/ELF reconnaissance; never opens a device."""
import json
import hashlib
from pathlib import Path
import subprocess
import ttnn

root = Path(__file__).resolve().parents[4]
size = root / "runtime/sfpi/compiler/bin/riscv-tt-elf-size"
cache = Path("/localdev/cglagovich/compute-sprint-20260918/jit-cache")
result = {"bindings": {}, "kernels": []}
for name, module in [("ttnn", ttnn), ("ttnn._ttnn", ttnn._ttnn), ("program_descriptor", ttnn._ttnn.program_descriptor)]:
    result["bindings"][name] = [x for x in dir(module) if "Opt" in x or "Build" in x]
    if hasattr(module, "types"):
        result["bindings"][name + ".types"] = [x for x in dir(module.types) if "Opt" in x or "Build" in x]
result["compiler"] = subprocess.check_output([str(size.parent / "riscv-tt-elf-g++"), "--version"], text=True)
driver = size.parent / "riscv-tt-elf-g++"
cc1 = Path(subprocess.check_output([str(driver), "-print-prog-name=cc1plus"], text=True).strip())
result["compiler_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (driver, cc1)}
required = ["SDPA_SPRINT_IDENTITY4", "SDPA_SPRINT_STATE_UNPACK_BATCH", "SDPA_SPRINT_LAZY_INIT", "SDPA_SPRINT_MAX_SCAN"]
for kernel in cache.glob("*/kernels/resident/*"):
    source = kernel / "chlkc_unpack.cpp"
    defines = kernel / "defines_generated.h"
    if not source.exists() or not defines.exists():
        continue
    src, defs = source.read_text(), defines.read_text()
    if not any(x in src for x in ("/compute-sprint-v1/fp32/resident.cpp", "/compute-sprint-v2/codegen/resident.cpp")) or not all(x in defs for x in required):
        continue
    if "SDPA_SPRINT_CANONICAL" in defs:
        continue
    elfs = list(kernel.glob("trisc?/*.elf"))
    elfs = [p for p in elfs if ".xip." not in p.name]
    if len(elfs) != 3:
        continue
    record = dict(path=str(kernel), source=src, defines=defs,
                  variant="C" if "SDPA_SPRINT_C_REFINE_HOIST" in defs else "D",
                  sizes=subprocess.check_output([str(size), *map(str, elfs)], text=True),
                  elf_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in elfs})
    if "SDPA_CODEGEN_OPT" in defs:
        record["largest_symbols"] = {}
        for elf in elfs:
            symbols = subprocess.check_output([str(size.parent / "riscv-tt-elf-nm"), "-S", "--size-sort", "-C", str(elf)], text=True)
            record["largest_symbols"][elf.parent.name] = [line[:240] for line in symbols.splitlines()[-12:]]
    result["kernels"].append(record)
print(json.dumps(result, indent=2))
