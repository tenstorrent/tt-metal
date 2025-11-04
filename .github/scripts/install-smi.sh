#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Source centralized version configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/versions.sh
source "${SCRIPT_DIR}/versions.sh"

wget -O "${TT_SMI_WHEEL}" "${TT_SMI_URL}" || exit 1
pip install --no-cache-dir "${TT_SMI_WHEEL}"
rm "${TT_SMI_WHEEL}"
