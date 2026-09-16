# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--kv-pcc",
        action="store_true",
        default=False,
        help="Check traced prefill K/V accuracy against an approved baseline instead of reporting performance.",
    )
    parser.addoption(
        "--kv-pcc-baseline",
        help="Markdown baseline override; defaults to tests/kv_pcc_baselines/baseline.md.",
    )
