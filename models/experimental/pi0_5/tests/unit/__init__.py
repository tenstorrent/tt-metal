# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
No-device CPU unit / smoke tests for Pi0.5 reference modules.

Everything under ``tests/unit`` runs on CPU with random weights — no
Tenstorrent device and no pretrained checkpoint required — so these gate
cheaply in CI before the heavier on-device PCC/perf suites.
"""
