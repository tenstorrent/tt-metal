# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def _preload_if_editable():
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

        tt_metal_home = os.getenv("TT_METAL_HOME") or f'{os.getenv("HOME")/tt-metal}'

        import ctypes

        for filename in ["libtt_metal.so", "_ttnncpp.so", "_ttnn.so"]:
            file = f"{tt_metal_home}/build/lib/{filename}"
            try:
                ctypes.cdll.LoadLibrary(file)
            except:
                pass


_preload_if_editable()
from ._ttml import *
