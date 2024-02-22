# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tests.ttnn.sweep_tests.sweep import run_all_tests, print_report


def main():
    device = ttnn.open_device(device_id=0)
    run_all_tests(device=device)
    ttnn.close_device(device)
    print_report()


if __name__ == "__main__":
    main()
