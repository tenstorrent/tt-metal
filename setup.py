import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext


def is_editable_install(build_extension):
    return not os.path.exists(build_extension.build_lib)


class BudaEagerBuildConstants:
    BUDA_EAGER_SO_SRC_LOCATION = "build/lib/libtt_lib_csrc.so"


class BudaEagerBuild(build_ext):
    def run(self):
        assert (
            len(self.extensions) == 1
        ), f"Detected more than 1 extension module - aborting"

        for ext in self.extensions:
            if is_editable_install(self):
                continue

            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)

            build_lib = self.build_lib
            full_lib_path = build_lib + "/" + filename

            src = BudaEagerBuildConstants.BUDA_EAGER_SO_SRC_LOCATION
            self.copy_file(src, full_lib_path)


short_hash = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    .decode("ascii")
    .strip()
)
date = (
    subprocess.check_output(
        ["git", "show", "-s", "--format=%cd", "--date=format:%y%m%d", "HEAD"]
    )
    .decode("ascii")
    .strip()
)
version = "0.1." + date + "+dev.gs." + short_hash

# Empty sources in order to force a BudaEagerBuild execution
buda_eager_lib_C = Extension("tt_lib._C", sources=[])

ext_modules = [buda_eager_lib_C]

packages = find_namespace_packages(
    where="libs",
    exclude=["*csrc"],
)

setup(
    name="tt_lib",
    version=version,
    author="Tenstorrent",
    url="http://www.tenstorrent.com",
    author_email="info@tenstorrent.com",
    description="General compute framework for Tenstorrent devices",
    python_requires=">=3.8,<3.9",
    packages=packages,
    package_dir={"": "libs"},
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=BudaEagerBuild),
    zip_safe=False,
)
