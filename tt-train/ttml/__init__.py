# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def _preload_from_tt_metal_home_if_editable():
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

    def preload_dev_tt_metal_libraries():
        import os

        tt_metal_home = os.getenv("TT_METAL_HOME") or os.path.join(
            os.getenv("HOME"), "tt-metal"
        )

        import ctypes

        for filename in ["libtt_metal.so", "_ttnncpp.so", "_ttnn.so"]:
            file = os.path.join(tt_metal_home, "build", "lib", filename)
            ctypes.cdll.LoadLibrary(file)

    """Check if this module is editable, which indicates a dev environment"""
    if is_editable_install(__name__):
        """Expect a dev build dir, load libraries from there to avoid mixing dev and installed environments"""
        preload_dev_tt_metal_libraries()


_preload_from_tt_metal_home_if_editable()
from ._ttml import *
