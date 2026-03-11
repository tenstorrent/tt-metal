# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for TADA tests.

The tada path and dac mock setup is done via pytest_plugins in the test file itself,
since conftest.py may not be loaded early enough with --import-mode=importlib.
"""
