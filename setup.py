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
class MetaliumBuildConfig:
    from_precompiled_dir = get_from_precompiled_dir()


metal_build_config = MetaliumBuildConfig()


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
    zip_safe=False,
    long_description=readme,
    long_description_content_type="text/markdown",
)
