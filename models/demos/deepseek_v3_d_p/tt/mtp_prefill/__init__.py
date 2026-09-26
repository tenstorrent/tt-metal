# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tenstorrent device modules for GLM-5.2 multi-token prediction during prefill.

``TtFusedMTP`` is the input projection, ``TtMTPModule`` wraps it with one decoder layer and
``shared_head.norm``, and ``TtMTPPredictor`` replays that module across the prediction levels.
"""
