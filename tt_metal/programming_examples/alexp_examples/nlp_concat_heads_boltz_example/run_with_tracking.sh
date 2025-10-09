#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Enable allocation tracking
export TT_ALLOC_TRACKING_ENABLED=1

echo "🔍 Allocation tracking ENABLED"
echo "   Make sure allocation_server_poc is running!"
echo ""

# Run the Python script
python3 python_nlp_concat_heads_boltz.py
