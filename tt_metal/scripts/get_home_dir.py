# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser(prog="BUDA-Eager env var setter")
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()

    short = args.short

    home_dir = str(Path(__file__).parent.parent.parent.resolve())

    if short:
        print(home_dir)
    else:
        raise Exception("Long form output of get_home_dir not completed yet")
