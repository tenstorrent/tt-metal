# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path


def _check__C_so_rpath():
    directory = Path(__file__).parent
    check_f = directory / ".rpath_checked"
    if os.path.exists(check_f):
        return
    target_so = None
    for f in os.listdir(directory):
        if f.startswith("_C") and f.endswith(".so"):
            target_so = directory / f
            break
    if not target_so:
        return
    import subprocess

    def has_not_found(target_so):
        if not os.path.exists(target_so):
            return False
        cmd = f"ldd {target_so}"
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return "not found" in result.stdout

    if has_not_found(target_so):
        new_rpath = directory / "build" / "lib"
        subprocess.check_call(f"patchelf --set-rpath {new_rpath} {target_so}", shell=True)
    subprocess.check_call(f"touch {check_f}", shell=True)


def _check_so_rpath_in_build_lib():
    directory = Path(__file__).parent / "build/lib"
    check_f = directory / ".rpath_checked"
    if not os.path.exists(directory):
        return
    if os.path.exists(check_f):
        return
    import subprocess

    eager_so = directory / "libtt_eager.so"
    metal_so = directory / "libtt_metal.so"

    def has_not_found(target_so):
        if not os.path.exists(target_so):
            return False
        cmd = f"ldd {target_so}"
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return "not found" in result.stdout

    new_rpath = directory
    if has_not_found(eager_so):
        subprocess.check_call(f"patchelf --set-rpath {new_rpath} {eager_so}", shell=True)
    if has_not_found(metal_so):
        subprocess.check_call(f"patchelf --set-rpath {new_rpath} {metal_so}", shell=True)
    subprocess.check_call(f"touch {check_f}", shell=True)


sit_pkgs_tt_lib = Path(__file__).parent

if "ARCH_NAME" not in os.environ or os.environ["ARCH_NAME"] == "":
    os.environ["ARCH_NAME"] = "grayskull"
if "TT_METAL_HOME" not in os.environ or os.environ["TT_METAL_HOME"] == "":
    # Workaround: treat $SITE_PACKAGES/tt_lib as TT_METAL_HOME
    os.environ["TT_METAL_HOME"] = str(sit_pkgs_tt_lib)
tt_metal_soft_link = sit_pkgs_tt_lib / "tt_metal"
tt_metal_soft_link_src = sit_pkgs_tt_lib / ".." / "tt_metal"
if not os.path.exists(tt_metal_soft_link) and os.path.exists(tt_metal_soft_link_src):
    os.symlink(tt_metal_soft_link_src, tt_metal_soft_link)
tt_eager_soft_link = sit_pkgs_tt_lib / "tt_eager"
tt_eager_soft_link_src = sit_pkgs_tt_lib / ".." / "tt_eager"
if not os.path.exists(tt_eager_soft_link) and os.path.exists(tt_eager_soft_link_src):
    os.symlink(tt_eager_soft_link_src, tt_eager_soft_link)

_check_so_rpath_in_build_lib()
_check__C_so_rpath()
from ._C import tensor, device, profiler, operations
