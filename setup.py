# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import glob
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from collections import namedtuple

from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.editable_wheel import editable_wheel
from setuptools_scm.version import guess_next_dev_version as _guess_next_dev
from wheel.wheelfile import WheelFile

readme = None

# Read README.md file from project root
readme_path = Path(__file__).absolute().parent / "README.md"
readme = readme_path.read_text(encoding="utf-8")


def get_lib_dir() -> str:
    """
    Inspired by GNUInstallDirs logic:
    default = 'lib'
    upgrade to 'lib64' only on 64-bit Linux that is not Debian/Arch/Alpine.
    """
    libdir = "lib"

    if platform.system() == "Linux":
        # skip lib64 on Debian/Arch/Alpine
        if not (
            Path("/etc/debian_version").exists()
            or Path("/etc/arch-release").exists()
            or Path("/etc/alpine-release").exists()
        ):
            if platform.architecture()[0] == "64bit":
                libdir = "lib64"

    return libdir


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


def get_metal_local_version_scheme(metal_build_config, version):
    if version.dirty:
        return f"+g{version.node}"
    else:
        return ""


def get_metal_main_version_scheme(metal_build_config, version):
    # Safety net
    if version is None:
        return "0.0.0.dev0"

    if getattr(version, "exact", False):
        # Exact tag (release/rc/dev*) already normalized by packaging
        return version.format_with("{tag}")

    # Untagged commit → let setuptools_scm choose X.Y.Z.devN
    return _guess_next_dev(version)


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
class MetaliumBuildConfig:
    from_precompiled_dir = get_from_precompiled_dir()


metal_build_config = MetaliumBuildConfig()


# WORKAROUND: make editable installation work
#
# The setuptools generates `MetaPathFinder` and hooks them to the python import machinery (via `sys.meta_path`),
# to be able to resolve imports of packages in editable installation. These finders are used as a fallback
# when python isn't able to find the package from the `sys.path`.
#
# However, their logic isn't able to resolve `import ttnn` properly. The problem is that the `ttnn` package
# is contained in the `ttnn` directory, which is a subdirectory of the root of the repository. If we execute
# `import ttnn` from the root of the repository, the `importlib` will find the top-level directory `ttnn` and
# won't fallback to the `MetaPathFinder` logic.
#
# To workaround this, we create our `.pth` file and add it to the editable wheel. Python will automatically
# load this file and populate the `sys.path` with the paths specified in the `.pth` file.
#
# NOTE: Needs `wheel` to be installed.
class EditableWheel(editable_wheel):
    def run(self):
        # Build the editable wheel first.
        super().run()

        # Create a .pth file with paths to the repo root, ttnn and tools directories.
        # This file gets loaded automatically by the python interpreter and its content gets populated into `sys.path`;
        # i.e. as if these paths were added to the PYTHONPATH.
        pth_filename = "ttnn-custom.pth"
        pth_content = f"{Path(__file__).parent}\n{Path(__file__).parent / 'ttnn'}\n{Path(__file__).parent / 'tools'}\n"

        print(f"EditableWheel.run: adding {pth_filename} to the wheel")

        # Find .whl file in the dist_dir (e.g. `ttnn-0.59.0rc42.dev21+gg66363d962a-0.editable-cp310-cp310-linux_x86_64.whl`)
        wheel = next((f for f in os.listdir(self.dist_dir) if f.endswith(".whl") and "editable" in f), None)

        assert wheel, f"Expected to see editable wheel in dist dir: {self.dist_dir}, but didn't find one"

        # Add the .pth file to the wheel archive.
        WheelFile(os.path.join(self.dist_dir, wheel), mode="a").writestr(pth_filename, pth_content)


packages = find_packages(where="ttnn", exclude=["ttnn.examples", "ttnn.examples.*"])
packages += find_packages("tools")

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
        "tracy": "tools/tracy",
        "triage": "tools/triage",
    },
    ext_modules=ext_modules,
    cmdclass=dict(editable_wheel=EditableWheel),
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["tt-triage = triage.triage:main"]},
)
