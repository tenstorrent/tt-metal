# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Setup script for TT-SMI Python package."""

from setuptools import setup, Extension, find_packages
import os
import sys

# Native C++ bindings are required
ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("\n" + "=" * 80)
    print("ERROR: pybind11 is required to build tt-smi-ui")
    print("Install with: pip install pybind11")
    print("=" * 80 + "\n")
    sys.exit(1)

# Get TT-Metal include paths
tt_metal_root = os.environ.get("TT_METAL_HOME", os.path.abspath("../../../.."))
build_dir = os.environ.get("TT_METAL_BUILD_DIR", f"{tt_metal_root}/build_Release")

# Check if build directory exists
if not os.path.exists(build_dir):
    print("\n" + "=" * 80)
    print(f"ERROR: TT-Metal build directory not found: {build_dir}")
    print("Build TT-Metal first with: ./build_metal_with_flags.sh")
    print("Or set TT_METAL_BUILD_DIR environment variable to the correct path")
    print("=" * 80 + "\n")
    sys.exit(1)

# Find CPM cache for dependencies (fmt, yaml-cpp, etc.)
import glob

fmt_include = glob.glob(f"{tt_metal_root}/.cpmcache/fmt/*/include")
yaml_include = glob.glob(f"{tt_metal_root}/.cpmcache/yaml-cpp/*/include")

include_dirs = [
    "tt_smi_ui",
    f"{tt_metal_root}",
    f"{tt_metal_root}/tt_metal",
    f"{tt_metal_root}/tt_metal/third_party/umd/device/api",  # UMD headers
]

# Add CPM dependencies if found
if fmt_include:
    include_dirs.extend(fmt_include)
if yaml_include:
    include_dirs.extend(yaml_include)

# Build directory exists, configure native extension
native_module = Pybind11Extension(
    "tt_smi_ui.bindings.native",
    sources=[
        "tt_smi_ui/bindings/native.cpp",
        "tt_smi_ui/tt_smi_backend.cpp",
    ],
    include_dirs=include_dirs,
    library_dirs=[
        f"{build_dir}/lib",
    ],
    libraries=[
        "device",
    ],
    runtime_library_dirs=[
        f"{build_dir}/lib",
    ],
    extra_compile_args=["-std=c++20", "-O3"],
    cxx_std=20,
)
ext_modules.append(native_module)
cmdclass = {"build_ext": build_ext}
print(f"Building with C++ bindings (TT_METAL_HOME={tt_metal_root})\n")

setup(
    name="tt-smi-ui",
    version="0.1.0",
    author="Tenstorrent",
    description="Python UI for Tenstorrent System Management Interface",
    long_description=open("SHM_TRACKING_README.md").read() if os.path.exists("SHM_TRACKING_README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        "rich>=13.0.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "tt-smi-ui=tt_smi_ui.cli:main",
            "ttsmi-ui=tt_smi_ui.cli:main",  # Short alias
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
)
