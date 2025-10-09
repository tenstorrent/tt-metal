# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def is_editable_install(package_name):
    import importlib.metadata
    import json

    try:
        # Get the distribution object for the package
        dist = importlib.metadata.distribution(package_name)

        # Check for direct_url.json in the metadata
        direct_url_json = dist.read_text("direct_url.json")
        if direct_url_json:
            data = json.loads(direct_url_json).get("dir_info")
            # Look for the 'editable' key
            return data and data.get("editable") or False

    except importlib.metadata.PackageNotFoundError:
        # Package not found, so it's not installed
        pass
    except FileNotFoundError:
        # direct_url.json not found, likely not an editable install
        pass
    except json.JSONDecodeError:
        # Error parsing direct_url.json
        pass
    return False


if is_editable_install(__name__):
    import os

    ld_library_path = "LD_LIBRARY_PATH"
    tt_metal_home = os.getenv("TT_METAL_HOME") or f'{os.getenv("HOME")/tt-metal}'

    if ld_library_path in os.environ.keys():
        os.environ[ld_library_path] = (
            f"{tt_metal_home}/build/lib" + os.pathsep + os.environ[ld_library_path]
        )
    else:
        os.environ[ld_library_path] = f"{tt_metal_home}/build/lib"

    import sys

    os.execl(sys.executable, sys.executable, *sys.argv)

from ._ttml import *
