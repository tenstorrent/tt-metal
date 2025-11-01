# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def _initialize():
    """
    Initialize the ttml module.

    If installed in editable mode, preloads TT Metal libraries from
    the development build directory to ensure consistency.
    """

    def _is_editable_install(package_name):
        """
        Check if a package is installed in editable mode.

        Args:
            package_name: Name of the package to check

        Returns:
            bool: True if the package is installed in editable mode, False otherwise
        """
        import importlib.metadata
        import json

        try:
            dist = importlib.metadata.distribution(package_name)
            direct_url_json = dist.read_text("direct_url.json")

            if direct_url_json:
                data = json.loads(direct_url_json).get("dir_info", {})
                return data.get("editable", False)

        except (
            importlib.metadata.PackageNotFoundError,
            FileNotFoundError,
            json.JSONDecodeError,
        ):
            pass

        return False

    def _preload_dev_tt_metal_libraries():
        """
        Preload TT Metal libraries from development build directory.

        This ensures that in development environments, libraries are loaded
        from TT_METAL_HOME to avoid mixing dev and installed environments.

        Raises:
            EnvironmentError: If TT_METAL_HOME is not set and default path doesn't exist
            FileNotFoundError: If required libraries are not found
            RuntimeError: If library loading fails
        """
        import ctypes
        import os
        from pathlib import Path

        # Determine TT Metal home directory
        tt_metal_home = os.getenv("TT_METAL_HOME")
        if not tt_metal_home:
            tt_metal_home = Path.home() / "tt-metal"
            if not tt_metal_home.exists():
                raise EnvironmentError(
                    "TT_METAL_HOME environment variable is not set and "
                    f"default path {tt_metal_home} does not exist. "
                    "Please set TT_METAL_HOME to your tt-metal installation directory."
                )
        else:
            tt_metal_home = Path(tt_metal_home)

        lib_dir = tt_metal_home / "build" / "lib"

        if not lib_dir.exists():
            raise FileNotFoundError(
                f"Library directory not found: {lib_dir}\n"
                f"Make sure TT Metal libraries are built in {tt_metal_home}"
            )

        # Load required libraries
        required_libs = ["libtt_metal.so", "_ttnncpp.so", "_ttnn.so"]

        for filename in required_libs:
            lib_path = lib_dir / filename

            if not lib_path.exists():
                raise FileNotFoundError(
                    f"Required library not found: {lib_path}\n"
                    f"Make sure TT Metal is built correctly.\n"
                    f"TT_METAL_HOME: {tt_metal_home}"
                )

            try:
                ctypes.cdll.LoadLibrary(str(lib_path))
            except OSError as e:
                raise RuntimeError(
                    f"Failed to load library {filename}: {e}\n" f"Path: {lib_path}"
                ) from e

    if _is_editable_install(__name__):
        _preload_dev_tt_metal_libraries()


# Initialize the module
_initialize()

# Import all symbols from the compiled extension
from ._ttml import *  # noqa: F401, F403
