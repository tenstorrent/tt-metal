# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


from models.demos.gemma4_d_p.demo.kv_pcc_data import DEFAULT_DATASET


def pytest_addoption(parser):
    parser.addoption(
        "--kv-pcc",
        action="store_true",
        default=False,
        help="Check traced prefill K/V accuracy against an approved baseline instead of reporting performance.",
    )
    parser.addoption(
        "--kv-pcc-data",
        default=str(DEFAULT_DATASET),
        help=(
            "Dataset directory path (absolute or relative to the working directory); "
            "contains input.txt, GPU traces, and baseline.json."
        ),
    )
