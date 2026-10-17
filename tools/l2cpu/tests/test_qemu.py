# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Runs the firmware QEMU suite (make -C tools/l2cpu/fw clean all qemu-test). Needs riscv64-unknown-elf-gcc and
qemu-system-riscv64 (Ubuntu: gcc-riscv64-unknown-elf, qemu-system-misc); skipped when they are missing."""
import os
import shutil
import subprocess

import pytest

FW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fw")


@pytest.mark.skipif(
    not (shutil.which("riscv64-unknown-elf-gcc") and shutil.which("qemu-system-riscv64")),
    reason="RISC-V toolchain or QEMU not installed",
)
def test_firmware_qemu():
    r = subprocess.run(["make", "-C", FW, "clean", "all", "qemu-test"], capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert r.stdout.count("qemu-test irq PASS") == 1 and r.stdout.count("qemu-test poll PASS") == 1
