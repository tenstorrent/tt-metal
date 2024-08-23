# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Huge pages is a linux feature that enables support for memory pages greater than the default 4kb.
Running this script with `python setup_hugepages.py enable` will enable hugepages up to a size of
1GB, and set iommu=pt. If run with `python setup_hugepages.py check` this script will check if
hugepages is configured as required.
"""
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Union

HUGEPAGE_SIZE = 1 << 30

proc_mounts = Path("/proc/mounts")
proc_cmdline = Path("/proc/cmdline")
sys_kernel = Path("/sys/kernel")
sys_kernel_mm_hugepages = Path("/sys/kernel/mm/hugepages")
etc_default_grub = Path("/etc/default/grub")
etc_default_grub_cloud_vm = Path("/etc/default/grub.d/50-cloudimg-settings.cfg")
dev_hugepages_1G = Path("/dev/hugepages-1G")
dev_hugepages = Path("/dev/hugepages")
grub_d_curtin_settings = Path("/etc/default/grub.d/50-curtin-settings.cfg")


def sanity_checks() -> Optional[str]:
    if not proc_mounts.is_file():
        return f"/proc must be mounted ({proc_mounts} is not a file)."

    if not sys_kernel.is_dir():
        return f"sysfs must be mounted on /sys ({sys_kernel} is not a directory)."

    if not sys_kernel_mm_hugepages.is_dir():
        return f"kernel does not appear to support huge pages ({sys_kernel_mm_hugepages} is not a directory)."

    if not etc_default_grub.is_file():
        return f"{etc_default_grub} not found."

    seen_hugepagesz_option = False
    seen_hugepagesz = set()
    get_kernel_cmdline_options()
    for k, v in get_kernel_cmdline_options():
        if k == "hugepagesz":
            size = parse_scaled_value(v)
            if size in seen_hugepagesz:
                return f"hugepagesz {size} (written {v}), seen twice in kernel command line."
            seen_hugepagesz.add(parse_scaled_value(v))

            seen_hugepagesz_option = True
        elif k == "hugepages":
            if not seen_hugepagesz_option:
                return f"hugepages option on kernel command line without preceding hugepagesz."
            seen_hugepagesz_option = False

    return None


def maybe_int(x: str) -> Union[int, str]:
    try:
        return int(x)
    except ValueError:
        return x


# # Returns dict pagesize (B) -> dict item name -> value, where value is converted to int if possible
def read_kernel_hugepage_info() -> Dict[int, Dict[str, Union[int, str]]]:
    try:
        results = {}

        for page_size_dir in sys_kernel_mm_hugepages.glob("hugepages-*kB"):
            items = {item.name: maybe_int(item.read_text()) for item in page_size_dir.iterdir() if item.name != 'demote'}

            page_size_bytes = int(page_size_dir.name[10:-2]) * 1024
            results[page_size_bytes] = items
        return results
    except Exception as e:
        raise RuntimeError("Reading kernel hugepage config from sysfs failed.", e)


def parse_scaled_value(value: str) -> int:
    match = re.fullmatch(r"(\d+)([kmgt]?)([b]?)?", value, re.IGNORECASE | re.ASCII)

    quantity = int(match[1])

    scale_letter = match[2]
    if scale_letter == "":
        pass
    elif scale_letter in "Tt":
        quantity <<= 40
    elif scale_letter in "Gg":
        quantity <<= 30
    elif scale_letter in "Mm":
        quantity <<= 20
    elif scale_letter in "Kk":
        quantity <<= 10

    return quantity


def find_hugepage_mounts() -> Dict[int, Path]:
    hugetlbfs_mount_re = re.compile(r"^hugetlbfs (?P<mountpoint>/[^ ]+) hugetlbfs (?P<options>[^ ]+) 0 0$")
    pagesize_re = re.compile(r"(?:^|,)pagesize=(?P<pagesize>\d+[KMGT])(?:,|$)")

    results = {}
    for line in open(proc_mounts):
        mount_match = hugetlbfs_mount_re.match(line)
        if mount_match:
            # print(line, "in", proc_mounts)
            options = mount_match["options"]
            pagesize_match = pagesize_re.search(options)
            if pagesize_match:
                mount_page_size = parse_scaled_value(pagesize_match["pagesize"])
                results[mount_page_size] = Path(mount_match["mountpoint"])
    return results


def add_systemctl_mount():
    dev_hugepages_1G_mount = """#  SPDX-License-Identifier: LGPL-2.1+
#
#  This file is derived from systemd's dev-hugepages.mount.
#
#  systemd is free software; you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 2.1 of the License, or
#  (at your option) any later version.

[Unit]
Description=1GB Huge Pages File System
Documentation=https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt
Documentation=https://www.freedesktop.org/wiki/Software/systemd/APIFileSystems
DefaultDependencies=no
Before=sysinit.target
ConditionPathExists=/sys/kernel/mm/hugepages
ConditionCapability=CAP_SYS_ADMIN
ConditionVirtualization=!private-users

[Install]
WantedBy=sysinit.target

[Mount]
What=hugetlbfs
Where=/dev/hugepages-1G
Type=hugetlbfs
Options=pagesize=1G,mode=0777
"""

    etc_systemd_system_dev_hugepages_1G_mount = Path("/etc/systemd/system/dev-hugepages\\x2d1G.mount")

    if not etc_systemd_system_dev_hugepages_1G_mount.parent.is_dir():
        raise RuntimeError(
            f"systemd configuration isn't present ({etc_systemd_system_dev_hugepages_1G_mount.parent} is not a directory)."
        )

    try:
        try:
            etc_systemd_system_dev_hugepages_1G_mount.unlink()
        except FileNotFoundError:  # unlink(missing_ok=True) available in 3.8
            pass

        etc_systemd_system_dev_hugepages_1G_mount.write_text(dev_hugepages_1G_mount)
    except Exception as e:
        raise RuntimeError(
            f"writing systemd mount unit failed ({etc_systemd_system_dev_hugepages_1G_mount}).",
            e,
        )

    try:
        subprocess.check_call(
            [
                "/bin/systemctl",
                "enable",
                "--system",
                "--now",
                etc_systemd_system_dev_hugepages_1G_mount.name,
            ]
        )
    except Exception as e:
        raise RuntimeError("Starting hugepage mount unit failed", e)


def get_kernel_cmdline_options():
    try:
        cmdline = proc_cmdline.read_text().strip()
    except Exception as e:
        raise RuntimeError(f"Reading {proc_cmdline} failed", e)

    for option in re.split(r"\s+", cmdline):
        kv = re.split("=", option, maxsplit=1)  # ROOT=UUID=1234 => ROOT, UUID=1234
        key = kv[0]
        value = kv[1] if len(kv) > 1 else ""
        yield (key, value)


def patch_etc_default_grub(is_cloud_vm):
    if grub_d_curtin_settings.is_file():
        for line in open(grub_d_curtin_settings):
            if "GRUB_CMDLINE_LINUX_DEFAULT" in line and line[0] != "#":
                raise RuntimeError(
                    "Please remove the GRUB_CMDLINE_LINUX_DEFAULT override from /etc/default/grub.d/50-curtin-settings.cfg"
                )

    etc_default_grub_txt = ""
    etc_grub_path = etc_default_grub_cloud_vm if is_cloud_vm else etc_default_grub
    for line in open(etc_grub_path):
        if "GRUB_CMDLINE_LINUX_DEFAULT" in line:
            # Get GRUB_CMDLINE_LINUX_DEFAULT line without hugepages/iommu args
            GRUB_re = re.compile(r'(?<=")(.*?)(?=")')
            GRUB_search = GRUB_re.search(line)
            GRUB_args = []
            if GRUB_search:
                GRUB_args = GRUB_search.group(0).split(" ")
                GRUB_args = list(
                    filter(
                        lambda i: "hugepages" not in i and "iommu" not in i,
                        GRUB_args,
                    )
                )

            # Add hugepages to GRUB_CMDLINE_LINUX_DEFAULT
            num_required_hugepages = get_num_required_hugepages()
            GRUB_args.extend(["hugepagesz=1G", f"hugepages={num_required_hugepages}", "iommu=pt"])
            line = line.split("=")[0] + '="' + " ".join(GRUB_args).strip() + '"\n'
        etc_default_grub_txt += line

    try:
        try:
            etc_grub_path.unlink()
        except FileNotFoundError:  # unlink(missing_ok=True) available in 3.8
            pass

        etc_grub_path.write_text(etc_default_grub_txt)
    except Exception as e:
        raise RuntimeError(f"writing etc default grub failed ({etc_grub_path}).", e)

    try:
        update_grub_cmd = ["update-grub2"] if is_cloud_vm else ["update-grub"]

        subprocess.check_call(update_grub_cmd)
    except Exception as e:
        raise RuntimeError("failed to run update grub cmd", e)


def get_host_has_grayskull():
    lspci_output = subprocess.check_output("lspci -vv -n", shell=True, stderr=subprocess.STDOUT).decode("utf-8")

    return "1e52:faca" in lspci_output


# Get the number of TT devices. Number of hugepages will need to match this now.
def get_num_required_hugepages(assert_devs_exist=True):
    tt_devices = []
    for device_path in sorted(glob.glob("/dev/tenstorrent/*")):
        # Filter out and make sure they are integers only
        device_id = os.path.basename(device_path)
        if device_id.isnumeric():
            tt_devices.append(int(device_id))

    num_tt_devices = len(tt_devices)
    if assert_devs_exist:
        assert num_tt_devices > 0, "Did not find any tt devices."

    host_has_grayskull = get_host_has_grayskull()

    num_required_hugepages = num_tt_devices if host_has_grayskull else num_tt_devices * 2

    return num_required_hugepages


def is_proc_cmdline_set():
    num_required_hugepages = get_num_required_hugepages()
    seen_hugepagesz = False
    seen_hugepages = False
    seen_iommu = False
    for k, v in get_kernel_cmdline_options():
        if "hugepagesz" in k:
            size = parse_scaled_value(v)
            seen_hugepagesz = size == HUGEPAGE_SIZE
        elif k == "hugepages":
            seen_hugepages = int(v) >= num_required_hugepages
        elif k == "iommu":
            seen_iommu = v == "pt"

    hugepage_info = read_kernel_hugepage_info()
    correct_nr_hugepages = hugepage_info[HUGEPAGE_SIZE]["nr_hugepages"] >= 1

    if not seen_hugepagesz:
        print("hugepagesz not in /proc/cmdline")
    if not seen_hugepages:
        print(f"hugepages={num_required_hugepages} not in /proc/cmdline")
    if not seen_iommu:
        print("iommu=pt not in /proc/cmdline")
    if not correct_nr_hugepages:
        print(
            "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages should be set to at least 1:\n",
            hugepage_info,
        )
    return seen_hugepagesz and seen_hugepages and seen_iommu and correct_nr_hugepages


def is_hugepages_set():
    if not dev_hugepages.is_dir():
        print(f"{dev_hugepages} does not exist or is not a directory")
        return False
    if not dev_hugepages_1G.is_dir():
        print(f"{dev_hugepages_1G} does not exist or is not a directory")
        return False
    return True


def is_hugepages_mounted():
    hugepage_mounts = find_hugepage_mounts()
    if len(hugepage_mounts) != 2:
        print("Hugepage mounts length not equal to 2:\n", hugepage_mounts)
        return False

    if HUGEPAGE_SIZE not in hugepage_mounts.keys():
        print(
            f"HUGEPAGE_SIZE {HUGEPAGE_SIZE} not in huge page mount:\n",
            hugepage_mounts,
        )
        return False

    if dev_hugepages not in hugepage_mounts.values():
        print("/dev/hugepages is not in huge page mount:\n", hugepage_mounts)
        return False

    if dev_hugepages_1G not in hugepage_mounts.values():
        print("/dev/hugepages_1G is not in huge page mount:\n", hugepage_mounts)
        return False

    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "operation",
        choices=["check", "enable", "first_pass"],
        help="Operation: check whether enabled, enable, first_pass.",
    )
    args = ap.parse_args()

    insane = sanity_checks()
    if insane:
        print("Sanity check failed:\n", insane)
        return 1

    def enable_or_first_pass():
        if os.geteuid() != 0:
            print(
                "To enable huge pages you need to run with sudo:\nsudo -E env PATH=$PATH python3 scripts/setup_hugepages.py enable"
            )
            return 1
        # Check if /proc/cmdline has hugepagesz=1G hugepages=N iommu=pt
        if not is_proc_cmdline_set():
            print("/proc/cmdline not satisfactory, updating GRUB_CMDLINE_LINUX_DEFAULT...")

            def get_is_cloud_vm():
                return etc_default_grub_cloud_vm.is_file()

            is_cloud_vm = get_is_cloud_vm()
            patch_etc_default_grub(is_cloud_vm=is_cloud_vm)
            print("=" * 100)
            print(
                "PLEASE REBOOT AND RESTART THE SCRIPT WITH `sudo -E env PATH=$PATH python3 scripts/hugepagessetup.py enable`"
            )
            print("If stuck on this step, something may be overriding GRUB_CMDLINE_LINUX_DEFAULT")
            return 0
        else:
            print("/proc/cmdline is satisfactory, continuing with installation")
            add_systemctl_mount()

            if not is_hugepages_set():
                return 1
            if not is_hugepages_mounted():
                return 1

            print("Huge pages is now set up")
            return 0

    if args.operation == "check":
        if get_num_required_hugepages(assert_devs_exist=False) == 0:
            print("WARNING: No tt devices detected. Cannot run check for huge page being enabled")
            return 0

        if not is_proc_cmdline_set() or not is_hugepages_set() or not is_hugepages_mounted():
            print("\nCheck failed - huge pages is not enabled")
            return 1

        print("Hugepage check passed!")
    elif args.operation == "enable":
        return enable_or_first_pass()
    elif args.operation == "first_pass":
        enable_or_first_pass()
        return 0
    else:
        pass

    return 0


if __name__ == "__main__":
    returncode = main()

    if returncode != 0:
        raise Exception("Hugepages is not setup")

    sys.exit(returncode)
