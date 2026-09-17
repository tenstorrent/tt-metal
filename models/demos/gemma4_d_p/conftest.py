# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


from models.demos.gemma4_d_p.tests.kv_pcc.data import DEFAULT_DATASET


def pytest_addoption(parser):
    parser.addoption(
        "--kv-pcc-data",
        default=str(DEFAULT_DATASET),
        help=(
            "Dataset directory path (absolute or relative to the working directory); "
            "contains input.txt, GPU traces, and baseline.json."
        ),
    )
