# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Traffic-calibrated static expert placement for expert-parallel MoE models.

The core (this package) is model-independent and runs on the host: it builds a diverse request corpus,
splits requests into calibration and held-out halves, calibrates a lossless expert permutation per MoE
layer, validates it, screens whether it is worth device time, and gates it with an on-device A/B.
Everything that touches a specific model implementation (routing capture, on-device remap, placed weight
cache, timed and golden runs) lives behind a ``PlacementAdapter`` in ``adapters/``.

Entry point: ``python -m models.common.moe_placement --help``. See README.md.
"""
