# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from tests.ttnn.sweep_tests.sweep import run_all_tests, print_report


def main():
    run_all_tests()
    print_report()


if __name__ == "__main__":
    main()
