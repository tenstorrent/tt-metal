# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import glob
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from collections import namedtuple

from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

readme = None

# Read README.md file from project root
readme_path = Path(__file__).absolute().parent / "README.md"
readme = readme_path.read_text(encoding="utf-8")


# Get the platform-specific lib directory name
def get_lib_dir():
    if sys.platform == "win32":
        return "bin"  # Windows DLLs go in bin directory
    elif sys.platform.startswith("linux"):
        return "lib64" if os.path.exists("/usr/lib64") else "lib"
    else:  # macOS and others
        return "lib"


BUNDLE_SFPI = False


def expand_patterns(patterns):
    """
    Given a list of glob patterns with brace expansion (e.g. `*.{h,hpp}`),
    return a flat list of glob patterns with the braces expanded.
    """
    expanded = []

    for pattern in patterns:
        if "{" in pattern and "}" in pattern:
            pre = pattern[: pattern.find("{")]
            post = pattern[pattern.find("}") + 1 :]
            options = pattern[pattern.find("{") + 1 : pattern.find("}")].split(",")

            for opt in options:
                expanded.append(f"{pre}{opt}{post}")
        else:
            expanded.append(pattern)

    return expanded


def copy_tree_with_patterns(src_dir, dst_dir, patterns, exclude_files=[]):
    """Copy only files matching glob patterns from src_dir into dst_dir, excluding specified files"""
    # Convert exclude_files to a set for faster lookups if there are files to exclude
    exclude_files = set(exclude_files) if exclude_files else None

    for pattern in expand_patterns(patterns):
        full_pattern = os.path.join(src_dir, pattern)
        matched_files = glob.glob(full_pattern, recursive=True)
        print(f"copying matched_files: {matched_files}")
        for src_path in matched_files:
            if os.path.isdir(src_path):
                continue
            rel_path = os.path.relpath(src_path, src_dir)
            # Only check for exclusions if we have files to exclude
            if exclude_files is not None:
                filename = os.path.basename(rel_path)
                if filename in exclude_files:
                    print(f"excluding file: {rel_path}")
                    continue
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)


class EnvVarNotFoundException(Exception):
    pass


def attempt_get_env_var(env_var_name):
    if env_var_name not in os.environ:
        raise EnvVarNotFoundException(f"{env_var_name} is not provided")
    return os.environ[env_var_name]


def get_is_srcdir_build():
    build_dir = CMakeBuild.get_working_dir()
    assert build_dir.is_dir()
    git_dir = build_dir / ".git"
    return git_dir.exists()


def get_metal_local_version_scheme(metal_build_config, version):
    if version.dirty:
        return f"+g{version.node}"
    else:
        return ""


def get_metal_main_version_scheme(metal_build_config, version):
    is_release_version = version.distance is None or version.distance == 0
    is_dirty = version.dirty
    is_clean_prod_build = (not is_dirty) and is_release_version

    if is_clean_prod_build:
        return version.format_with("{tag}")
    elif is_dirty and not is_release_version:
        return version.format_with("{tag}.dev{distance}")
    elif is_dirty and is_release_version:
        return version.format_with("{tag}")
    else:
        assert not is_dirty and not is_release_version
        return version.format_with("{tag}.dev{distance}")


def get_version(metal_build_config):
    return {
        "version_scheme": partial(get_metal_main_version_scheme, metal_build_config),
        "local_scheme": partial(get_metal_local_version_scheme, metal_build_config),
    }


def get_from_precompiled_dir():
    """Additional option if the precompiled C++ libs are already in-place."""
    precompiled_dir = os.environ.get("TT_FROM_PRECOMPILED_DIR", None)
    return Path(precompiled_dir) if precompiled_dir else None


@dataclass(frozen=True)
class MetalliumBuildConfig:
    from_precompiled_dir = get_from_precompiled_dir()


metal_build_config = MetalliumBuildConfig()


class CMakeBuild(build_ext):
    @staticmethod
    def get_build_env():
        return {
            **os.environ.copy(),
            "CXX": "clang++-17",
        }

    @staticmethod
    def get_working_dir():
        working_dir = Path(__file__).parent
        assert working_dir.is_dir()
        return working_dir

    # This should only run when building the wheel. Should not be running for any dev flow
    # Taking advantage of the fact devs run editable pip install -> "pip install -e ."
    def run(self) -> None:
        if self.is_editable_install_():
            assert get_is_srcdir_build(), f"Editable install detected in a non-srcdir environment, aborting"
            return

        build_env = CMakeBuild.get_build_env()
        source_dir = (
            metal_build_config.from_precompiled_dir
            if metal_build_config.from_precompiled_dir
            else CMakeBuild.get_working_dir()
        )
        assert source_dir.is_dir(), f"Source dir {source_dir} seems to not exist"

        if metal_build_config.from_precompiled_dir:
            build_dir = source_dir / "build"
            assert (build_dir / "lib").exists() and (
                source_dir / "runtime"
            ).exists(), "The precompiled option is selected via `TT_FROM_PRECOMPILED` \
            env var. Please place files into `build/lib` and `runtime` folders."
        else:
            build_dir = source_dir / "build_Release"
            # We indirectly set a wheel build for our CMake build by using BUILD_SHARED_LIBS. This does the following things:
            # - Bundles (most) of our libraries into a static library to deal with a potential singleton bug error with tt_cluster (to fix)
            build_script_args = ["--build-static-libs", "--release"]

            if "CIBUILDWHEEL" in os.environ:
                cmake_args = [
                    "cmake",
                    "-B",
                    build_dir,
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_INSTALL_PREFIX=build_Release",
                    "-DBUILD_SHARED_LIBS=OFF",
                    "-DTT_INSTALL=OFF",
                    "-DTT_UNITY_BUILDS=ON",
                    "-DTT_ENABLE_LIGHT_METAL_TRACE=ON",
                    "-DWITH_PYTHON_BINDINGS=ON",
                    "-DTT_USE_SYSTEM_SFPI=ON",
                    "-DENABLE_CCACHE=TRUE",
                ]

                # Add Tracy flags if enabled
                if os.environ.get("CIBW_ENABLE_TRACY") == "ON":
                    cmake_args.extend(
                        [
                            "-DENABLE_TRACY=ON",
                        ]
                    )

                cmake_args.extend(["-S", source_dir])

                subprocess.check_call(cmake_args)
                subprocess.check_call(
                    [
                        "cmake",
                        "--build",
                        build_dir,
                    ]
                )
                subprocess.check_call(
                    [
                        "cmake",
                        "--install",
                        build_dir,
                    ]
                )
            else:
                subprocess.check_call(["./build_metal.sh", *build_script_args], cwd=source_dir, env=build_env)

        # Some verbose sanity logging to see what files exist in the outputs
        subprocess.check_call(["ls", "-hal"], cwd=source_dir, env=build_env)
        subprocess.check_call(["ls", "-hal", str(build_dir / "lib")], cwd=source_dir, env=build_env)
        subprocess.check_call(["ls", "-hal", "runtime"], cwd=source_dir, env=build_env)

        # Copy needed C++ shared libraries and runtime assets into wheel (sfpi, FW etc)
        lib_patterns = [
            "_ttnn.so",
            "libtt_metal.so",
            "libdevice.so",
        ]
        runtime_patterns = [
            "hw/**/*",
        ]
        runtime_exclude_files = []
        if BUNDLE_SFPI:
            runtime_patterns.append("sfpi/**/*")
            runtime_exclude_files = [
                "riscv32-unknown-elf-lto-dump",
                "riscv32-unknown-elf-gdb",
                "riscv32-unknown-elf-objdump",
                "riscv32-unknown-elf-run",
                "riscv32-unknown-elf-ranlib",
                "riscv32-unknown-elf-gprof",
                "riscv32-unknown-elf-strings",
                "riscv32-unknown-elf-size",
                "riscv32-unknown-elf-readelf",
                "riscv32-unknown-elf-nm",
                "riscv32-unknown-elf-c++filt",
                "riscv32-unknown-elf-addr2line",
                "riscv32-unknown-elf-gcov",
                "riscv32-unknown-elf-gcov-tool",
                "riscv32-unknown-elf-gcov-dump",
                "riscv32-unknown-elf-elfedit",
                "riscv32-unknown-elf-gcc-ranlib",
                "riscv32-unknown-elf-gcc-nm",
                "riscv32-unknown-elf-gdb-add-index",
            ]
        ttnn_patterns = [
            # These weren't supposed to be in the JIT API, but one file currently is
            "api/ttnn/tensor/enum_types.hpp",
        ]
        ttnn_cpp_patterns = [
            "ttnn/deprecated/**/kernels/**/*",
            "ttnn/operations/**/kernels/**/*",
            "ttnn/operations/ccl/**/*",
            "ttnn/operations/data_movement/**/*",
            "ttnn/operations/moreh/**/*",
        ]
        tt_metal_patterns = [
            "api/tt-metalium/buffer_constants.hpp",
            "api/tt-metalium/buffer_types.hpp",
            "api/tt-metalium/circular_buffer_constants.h",
            "api/tt-metalium/constants.hpp",
            "api/tt-metalium/dev_msgs.h",
            "api/tt-metalium/fabric_host_interface.h",
            "api/tt-metalium/fabric_edm_types.hpp",
            "api/tt-metalium/fabric_edm_packet_header.hpp",
            "api/tt-metalium/edm_fabric_counters.hpp",
            "core_descriptors/*.yaml",
            "fabric/hw/**/*",
            "fabric/mesh_graph_descriptors/*.yaml",
            "hw/**/*",
            "hostdevcommon/api/hostdevcommon/**/*",
            "impl/dispatch/kernels/**/*",
            "include/**/*",
            "kernels/**/*",
            "third_party/tt_llk/**/*",
            "tools/profiler/*",
            "soc_descriptors/*.yaml",
        ]
        copy_tree_with_patterns(build_dir / get_lib_dir(), self.build_lib + f"/ttnn/build/lib", lib_patterns)
        copy_tree_with_patterns(build_dir, self.build_lib + "/ttnn/build/lib", ["sfpi-version.json"])
        copy_tree_with_patterns(
            source_dir / "runtime", self.build_lib + "/ttnn/runtime", runtime_patterns, runtime_exclude_files
        )
        copy_tree_with_patterns(source_dir / "ttnn", self.build_lib + "/ttnn", ttnn_patterns)
        copy_tree_with_patterns(source_dir / "ttnn/cpp", self.build_lib + "/ttnn/cpp", ttnn_cpp_patterns)
        copy_tree_with_patterns(source_dir / "tt_metal", self.build_lib + "/ttnn/tt_metal", tt_metal_patterns)

        # Move built final built _ttnn SO into appropriate location in ttnn Python tree in wheel
        assert len(self.extensions) == 1, f"Detected {len(self.extensions)} extensions, but should be only 1: ttnn"
        ext = list(self.extensions)[0]
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)

        build_lib = self.build_lib
        full_lib_path = build_lib + "/" + filename

        dir_path = os.path.dirname(full_lib_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        dest_ttnn_build_dir = self.build_lib + "/ttnn/build"
        src = os.path.join(dest_ttnn_build_dir, build_constants_lookup[ext].so_src_location)
        self.copy_file(src, full_lib_path)
        os.remove(src)

    def is_editable_install_(self):
        return self.inplace


packages = find_packages(where="ttnn", exclude=["ttnn.examples", "ttnn.examples.*"])

print(("packaging: ", packages))

# Empty sources in order to force extension executions
ttnn_lib_C = Extension("ttnn._ttnn", sources=[])

ext_modules = [ttnn_lib_C]

BuildConstants = namedtuple("BuildConstants", ["so_src_location"])

build_constants_lookup = {
    ttnn_lib_C: BuildConstants(so_src_location="lib/_ttnn.so"),
}


setup(
    url="http://www.tenstorrent.com",
    use_scm_version=get_version(metal_build_config),
    packages=packages,
    package_dir={
        "": "ttnn",
    },
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
)
