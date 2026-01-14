#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

pip install -q --upgrade pip
pip install -q --no-cache-dir -r /tmp/requirements_tests.txt
