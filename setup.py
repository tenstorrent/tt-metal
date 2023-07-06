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

    is_srcdir_build = git_dir.exists()

    if is_srcdir_build:
        assert git_dir.is_dir(), f"{git_dir} is named .git/ but is not a directory"

    return is_srcdir_build


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
    is_release_version = version.distance == 0
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
        "version_scheme": partial(
            get_buda_eager_main_version_scheme, buda_eager_build_config
        ),
        "local_scheme": partial(
            get_buda_eager_local_version_scheme, buda_eager_build_config
        ),
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
        return {
            **os.environ.copy(),
            "TT_METAL_HOME": Path(__file__).parent,
            "TT_METAL_ENV": "production",
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

        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)

        build_lib = self.build_lib
        full_lib_path = build_lib + "/" + filename

        src = BUDAEagerBuildConstants.BUDA_EAGER_SO_SRC_LOCATION
        self.copy_file(src, full_lib_path)

    def is_editable_install_(self):
        return not os.path.exists(self.build_lib)


# Empty sources in order to force a BUDAEagerBuild execution
buda_eager_lib_C = Extension("tt_lib._C", sources=[])

ext_modules = [buda_eager_lib_C]

packages = find_namespace_packages(
    where="libs",
)

setup(
    url="http://www.tenstorrent.com",
    use_scm_version=get_version(buda_eager_build_config),
    packages=packages,
    package_dir={"": "libs"},
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=BUDAEagerBuild),
    zip_safe=False,
)
