#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

pip install -q --upgrade pip
pip install -q --no-cache-dir uv

# Install to the active venv (if VIRTUAL_ENV is set) or system Python otherwise.
# The tt-metalium base image sets VIRTUAL_ENV=/opt/venv and activates it by default.
if [ -n "$VIRTUAL_ENV" ]; then
    uv pip install -q --index-strategy unsafe-best-match --no-cache-dir -r /tmp/requirements_tests.txt
else
    uv pip install -q --system --index-strategy unsafe-best-match --no-cache-dir -r /tmp/requirements_tests.txt
fi
