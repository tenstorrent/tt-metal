# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from tests.ttnn.sweep_tests.sweep import run_sweeps, check_sweeps


def main():
    run_sweeps()
    check_sweeps()


if __name__ == "__main__":
    main()
