# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import argparse

from sweeps import print_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", action="store_true")
    detailed = parser.parse_args().detailed

    print_report(detailed=detailed)


if __name__ == "__main__":
    main()
