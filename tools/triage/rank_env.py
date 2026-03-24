#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared rank-environment helpers for triage tooling."""

from __future__ import annotations

import os

# Keep this order aligned with the MPI/launcher rank detection used by TT-Metal.
RANK_ENV_VAR_PRECEDENCE = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "SLURM_PROCID",
    "PMIX_RANK",
    "TT_MESH_HOST_RANK",
)


def get_rank_from_env() -> int | None:
    """Return the first valid non-negative rank from the environment, or None."""

    for var in RANK_ENV_VAR_PRECEDENCE:
        val = os.environ.get(var)
        if val is None:
            continue
        try:
            rank = int(val)
        except (ValueError, OverflowError):
            continue
        if rank >= 0:
            return rank
    return None
