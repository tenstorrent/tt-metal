#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Source centralized version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/versions.sh
source "${SCRIPT_DIR}/versions.sh"

wget -O "${EXALENS_WHEEL}" "${EXALENS_URL}" || exit 1
pip install --no-cache-dir "${EXALENS_WHEEL}"
rm "${EXALENS_WHEEL}"
