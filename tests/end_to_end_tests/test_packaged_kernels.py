# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib
import re

import pytest

import ttnn

# tensor_prefetcher_manager.cpp hands this exact string to CreateKernel. kernel.cpp
# resolves a relative kernel path against the root dir last, after the cwd, so any run
# that still has a source checkout finds it there and never touches the wheel's copy.
PREFETCHER_KERNEL = "tt_metal/impl/buffers/kernels/tensor_prefetcher.cpp"


def packaged_root():
    """The directory a relative kernel path resolves against.

    Mirrors the SetRootDir call in ttnn/__init__.py: an installed wheel roots at the
    package directory, an editable install two levels up at the source tree.
    """
    package_dir = pathlib.Path(ttnn.__file__).resolve().parent
    if "site-packages" in str(package_dir) or "dist-packages" in str(package_dir):
        return package_dir
    return package_dir.parent.parent


@pytest.mark.eager_host_side
def test_prefetcher_kernel_and_its_headers_are_packaged():
    root = packaged_root()

    kernel = root / PREFETCHER_KERNEL
    assert kernel.is_file(), f"{PREFETCHER_KERNEL} is missing under {root}; add it to tt_metal_patterns in setup.py"

    # The JIT compiler resolves the kernel's own tt_metal includes against the same
    # root. A header left out of the wheel fails only once a kernel build runs on
    # hardware, so check the include closure here rather than just the kernel file.
    includes = re.findall(r'^#include\s+"(tt_metal/[^"]+)"', kernel.read_text(), re.MULTILINE)
    assert includes, f"expected {PREFETCHER_KERNEL} to include tt_metal headers; the check below proves nothing"

    missing = sorted(include for include in includes if not (root / include).is_file())
    assert not missing, f"{PREFETCHER_KERNEL} includes headers missing under {root}: {missing}"
