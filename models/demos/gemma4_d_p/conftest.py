# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--kv-pcc",
        action="store_true",
        default=False,
        help="Measure traced prefill K/V PCC using the workspace gemma_gpu_traces reference instead of performance.",
    )
