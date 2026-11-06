# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the router bring-up tools (temporary: drop before merge; see README.md).

- toolchain_prefix(): the RISC-V binutils/gcc prefix, as the JIT build finds it.
- router_images(dir, kernel): the router ELFs in a kernel cache or an offline build.
"""
import glob
import os
import shutil
import sys

# The ERISC kernel images a router build produces, by architecture (the HAL's JIT target names):
#   Wormhole: "erisc" (one ERISC per Ethernet core; the router runs sender and receiver in one loop).
#   Blackhole: "active_erisc" (ERISC0) and "subordinate_active_erisc" (ERISC1); one or two ERISCs per router.
ERISC_IMAGES = ("erisc", "active_erisc", "subordinate_active_erisc")
ERISC_TAG = {"erisc": "ERISC", "active_erisc": "E0", "subordinate_active_erisc": "E1"}
DEFAULT_KERNEL = "fabric_erisc_router"


def repo_root():
    """The tt-metal tree these tools sit in (tt_metal/fabric/tools/router_bringup)."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))


def toolchain_prefix():
    """The RISC-V tool prefix (".../riscv-tt-elf-"): $ROUTER_TOOLS_TOOLCHAIN if set, else the JIT build's order
    ($TT_METAL_HOME or this tree's runtime/sfpi, then /opt/tenstorrent/sfpi), else riscv-tt-elf-objdump on PATH."""
    p = os.environ.get("ROUTER_TOOLS_TOOLCHAIN")
    if p:
        return p
    roots = [r for r in (os.environ.get("TT_METAL_HOME"), repo_root()) if r]
    for r in roots + ["/opt/tenstorrent"]:
        for sub in ("runtime/sfpi", "sfpi"):
            cand = os.path.join(r, sub, "compiler/bin/riscv-tt-elf-")
            if os.path.exists(cand + "objdump"):
                return cand
    w = shutil.which("riscv-tt-elf-objdump")
    if w:
        return w[: -len("objdump")]
    sys.exit("no RISC-V toolchain found: set ROUTER_TOOLS_TOOLCHAIN=<sfpi>/compiler/bin/riscv-tt-elf-")


def router_images(root, kernel=DEFAULT_KERNEL, procs=ERISC_IMAGES):
    """[(config hash, image name, elf path)] under ROOT: a kernel cache (.../kernels/KERNEL/<hash>/<proc>/<proc>.elf)
    or an offline build (<hash>/<proc>/<proc>.elf)."""
    out = []
    for proc in procs:
        pats = [f"{root}/**/kernels/{kernel}/*/{proc}/{proc}.elf", f"{root}/*/{proc}/{proc}.elf"]
        seen = set()
        for pat in pats:
            for elf in glob.glob(pat, recursive=True):
                if elf in seen:
                    continue
                seen.add(elf)
                h = os.path.basename(os.path.dirname(os.path.dirname(elf)))
                out.append((h, proc, elf))
    return sorted(out)
