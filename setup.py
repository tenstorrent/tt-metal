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

from pathlib import Path
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext


class BUDAEagerBuildConstants:
    BUDA_EAGER_SO_SRC_LOCATION = "build/lib/libtt_lib_csrc.so"


class EnvVarNotFoundException(Exception):
    pass


def attempt_get_env_var(env_var_name):
    if env_var_name not in os.environ:
        raise EnvVarNotFoundException(f"{env_var_name} is not provided")
    return os.environ[env_var_name]


def get_is_srcdir_build():
    build_dir = Path(__file__).parent
    assert build_dir.is_dir()
    git_dir = build_dir / ".git"
    return git_dir.exists()


def get_is_dev_build():
    try:
        is_dev_build = attempt_get_env_var("TT_METAL_ENV") == "dev"
    except EnvVarNotFoundException as e:
        is_dev_build = False

    return is_dev_build


def get_arch_name():
    return attempt_get_env_var("ARCH_NAME")


def get_buda_eager_local_version_scheme(buda_eager_build_config, version):
    from setuptools_scm.version import ScmVersion, guess_next_version

    arch_name = buda_eager_build_config.arch_name

    if version.dirty:
        return f"+g{version.node}.{arch_name}"
    else:
        return ""


def get_buda_eager_main_version_scheme(buda_eager_build_config, version):
    from setuptools_scm.version import ScmVersion, guess_next_version

    is_release_version = version.distance is None or version.distance == 0
    is_dirty = version.dirty
    is_clean_prod_build = (not is_dirty) and is_release_version

    arch_name = buda_eager_build_config.arch_name

    if is_clean_prod_build:
        return version.format_with("{tag}+{arch_name}", arch_name=arch_name)
    elif is_dirty and not is_release_version:
        return version.format_with("{tag}.dev{distance}", arch_name=arch_name)
    elif is_dirty and is_release_version:
        return version.format_with("{tag}", arch_name=arch_name)
    else:
        assert not is_dirty and not is_release_version
        return version.format_with("{tag}.dev{distance}+{arch_name}", arch_name=arch_name)


def get_version(buda_eager_build_config):
    return {
        "version_scheme": partial(get_buda_eager_main_version_scheme, buda_eager_build_config),
        "local_scheme": partial(get_buda_eager_local_version_scheme, buda_eager_build_config),
    }


@dataclass(frozen=True)
class BUDAEagerBuildConfig:
    is_dev_build = get_is_dev_build()
    is_srcdir_build = get_is_srcdir_build()
    arch_name = get_arch_name()


buda_eager_build_config = BUDAEagerBuildConfig()


class BUDAEagerBuild(build_ext):
    @staticmethod
    def get_buda_eager_build_env():
        """
        Force production environment when creating the wheel because there's
        a lot of extra stuff that's added to the environment in dev that the
        wheel doesn't need
        """
        return {
            **os.environ.copy(),
            "TT_METAL_HOME": Path(__file__).parent,
            "TT_METAL_ENV": "production",
            # Need to create static lib for tt_metal runtime because currently
            # we package it with the wheel at the moment
            "TT_METAL_CREATE_STATIC_LIB": "1",
        }

    def run(self):
        assert (
            len(self.extensions) == 1
        ), f"Detected more than 1 extension module - aborting because we shouldn't be doing more yet"

        ext = self.extensions[0]
        if self.is_editable_install_():
            assert (
                buda_eager_build_config.is_srcdir_build
            ), f"Editable install detected in a non-srcdir environment, aborting"
            return

        build_env = BUDAEagerBuild.get_buda_eager_build_env()
        subprocess.check_call(["make", "build"], env=build_env)
        subprocess.check_call(["ls", "-hal", "build/lib"], env=build_env)

        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)

        build_lib = self.build_lib
        full_lib_path = build_lib + "/" + filename

        dir_path = os.path.dirname(full_lib_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        src = BUDAEagerBuildConstants.BUDA_EAGER_SO_SRC_LOCATION
        self.copy_file(src, full_lib_path)

    def is_editable_install_(self):
        return not os.path.exists(self.build_lib)


# Include tt_metal_C for kernels and src/ and tools
# And any kernels inside `tt_eager/tt_dnn. We must keep all ops kernels inside
# tt_dnn
packages = ["tt_lib", "tt_metal", "tt_lib.models", "tt_eager.tt_dnn"]

# Empty sources in order to force a BUDAEagerBuild execution
buda_eager_lib_C = Extension("tt_lib._C", sources=[])

ext_modules = [buda_eager_lib_C]

setup(
    url="http://www.tenstorrent.com",
    use_scm_version=get_version(buda_eager_build_config),
    packages=packages,
    package_dir={
        "": "tt_eager",
        "tt_metal": "tt_metal",
        "tt_lib.models": "models",
        "tt_eager.tt_dnn": "tt_eager/tt_dnn",
    },
    include_package_data=True,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=BUDAEagerBuild),
    zip_safe=False,
)
