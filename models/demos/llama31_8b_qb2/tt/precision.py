# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The checked-in precision policy used by every layer and both inference phases."""

import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[1] / "config/precision.json"


def load_precision_config():
    policy = json.loads(POLICY_PATH.read_text())
    if policy["schema_version"] != 1 or policy["layer_exceptions"]:
        raise ValueError("Expected a version-1 precision policy shared by all 32 layers")
    return policy
