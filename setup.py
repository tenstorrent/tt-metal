# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import sysconfig
import platform
import subprocess
from dataclasses import dataclass
from functools import partial
from collections import namedtuple

from pathlib import Path
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext


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


def get_arch_name():
    return attempt_get_env_var("ARCH_NAME")


def get_metal_local_version_scheme(metal_build_config, version):
    from setuptools_scm.version import ScmVersion, guess_next_version

    arch_name = metal_build_config.arch_name

    if version.dirty:
        return f"+g{version.node}.{arch_name}"
    else:
        return ""


def get_metal_main_version_scheme(metal_build_config, version):
    from setuptools_scm.version import ScmVersion, guess_next_version

    is_release_version = version.distance is None or version.distance == 0
    is_dirty = version.dirty
    is_clean_prod_build = (not is_dirty) and is_release_version

    arch_name = metal_build_config.arch_name

    if is_clean_prod_build:
        return version.format_with("{tag}+{arch_name}", arch_name=arch_name)
    elif is_dirty and not is_release_version:
        return version.format_with("{tag}.dev{distance}", arch_name=arch_name)
    elif is_dirty and is_release_version:
        return version.format_with("{tag}", arch_name=arch_name)
    else:
        assert not is_dirty and not is_release_version
        return version.format_with("{tag}.dev{distance}+{arch_name}", arch_name=arch_name)


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
    arch_name = get_arch_name()
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
        build_dir = source_dir / "build"

        if metal_build_config.from_precompiled_dir:
            assert (build_dir / "lib").exists() and (
                source_dir / "runtime"
            ).exists(), "The precompiled option is selected via `TT_FROM_PRECOMPILED` \
            env var. Please place files into `build/lib` and `runtime` folders."
        else:
            # We indirectly set a wheel build for our CMake build by using BUILD_SHARED_LIBS. This does two things:
            # - Set the right rpath ($ORIGIN) for our Python bindings so it can find the extra libraries it needs at runtime
            # - Bundles (most) of our libraries into a static library to deal with a potential singleton bug error with tt_cluster (to fix)
            cmake_args = ["-DBUILD_SHARED_LIBS=OFF"]

            nproc = subprocess.check_output(["nproc"]).decode().strip()
            build_args = [f"-j{nproc}"]

            subprocess.check_call(["cmake", "-G", "Ninja", source_dir, *cmake_args], cwd=build_dir, env=build_env)
            subprocess.check_call(
                ["cmake", "--build", ".", "--target", "install", *build_args], cwd=build_dir, env=build_env
            )

        # Some verbose sanity logging to see what files exist in the outputs
        subprocess.check_call(["ls", "-hal"], cwd=source_dir, env=build_env)
        subprocess.check_call(["ls", "-hal", "build/lib"], cwd=source_dir, env=build_env)
        subprocess.check_call(["ls", "-hal", "runtime"], cwd=source_dir, env=build_env)

        # Copy needed C++ shared libraries and runtime assets into wheel (sfpi, FW etc)
        dest_ttnn_build_dir = self.build_lib + "/ttnn/build"
        os.makedirs(dest_ttnn_build_dir, exist_ok=True)
        self.copy_tree(source_dir / "build/lib", dest_ttnn_build_dir + "/lib")
        self.copy_tree(source_dir / "runtime", self.build_lib + "/runtime")

        # Encode ARCH_NAME into package for later use so user doesn't have to provide
        arch_name_file = self.build_lib + "/ttnn/.ARCH_NAME"
        # should probably change to Python calls to write to a file descriptor instead of calling Linux tools
        subprocess.check_call(f"echo {metal_build_config.arch_name} > {arch_name_file}", shell=True)

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

        src = os.path.join(dest_ttnn_build_dir, build_constants_lookup[ext].so_src_location)
        self.copy_file(src, full_lib_path)
        os.remove(src)

    def is_editable_install_(self):
        return self.inplace


# Include tt_metal_C for kernels and src/ and tools
# And any kernels inside `tt_eager/tt_dnn. We must keep all ops kernels inside tt_dnn
packages = ["tt_lib", "tt_metal", "tt_lib.models", "ttnn", "ttnn.cpp", "tracy"]

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
        "": "ttnn",  # only this is relevant in case of editable install mode
        "tracy": "ttnn/tracy",
        "tt_metal": "tt_metal",  # kernels depend on headers here
        "ttnn.cpp": "ttnn/cpp",
        "tt_lib.models": "models",  # make sure ttnn does not depend on model and remove!!!
    },
    include_package_data=True,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
