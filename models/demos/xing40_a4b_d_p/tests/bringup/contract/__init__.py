# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 serving-contract tests (bringup/serving_contract.md, bringup/contract_tests.yaml)."""

import os
from pathlib import Path

# Gates set BRINGUP_SPEC; a run by hand gets this model's spec.
os.environ.setdefault("BRINGUP_SPEC", str(Path(__file__).resolve().parents[3] / "bringup" / "spec.yaml"))
