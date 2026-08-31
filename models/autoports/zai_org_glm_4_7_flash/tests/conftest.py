# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    config.addinivalue_line("markers", "real_weights: needs the local GLM-4.7-Flash HF snapshot")
    config.addinivalue_line("markers", "long: long-running long-context evidence tests")
