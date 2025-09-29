# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def try_setup_ttml_so():
    import os
    import sys

    TT_METAL_HOME = os.environ["TT_METAL_HOME"]
    TT_TRAIN_BUILD_HOME = f"{TT_METAL_HOME}/tt-train/build/sources/ttml"
    TT_METAL_BUILD_HOME = f"{TT_METAL_HOME}/build/tt-train/sources/ttml"
    paths = (TT_TRAIN_BUILD_HOME, TT_METAL_BUILD_HOME)

    for path in paths:
        try:
            sys.path.append(path)
            import _ttml

            return True
        except ModuleNotFoundError:
            pass

    print(f"_ttml .so file not found in {paths}")
    return False


if try_setup_ttml_so():
    from _ttml import *
