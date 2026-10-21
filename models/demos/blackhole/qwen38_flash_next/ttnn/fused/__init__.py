# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused decode kernels over ``ttnn.generic_op``.

``program`` builds and runs one program from Python (kernels, CBs, semaphores, the rows contract); ``registry`` names
each fused kernel with the composed ttnn chain it replaces; the proven kernels serve by default (``QWEN38_FUSED_OFF``
falls back to the chains, ``QWEN38_FUSED`` switches an opt-in kernel on).  Each kernel is a sub-package ``<name>/``
with its ``kernels/*.cpp`` and registers itself on import; a few sub-packages only hold kernels that other kernels
run.  Add new ones to the import list below.
"""

from . import program
from .registry import (
    ALL,
    BITWISE,
    COMPONENT,
    DEFAULT_ON,
    ENV,
    OFF_ENV,
    TOLERANCE_CLASSES,
    ULP,
    AdmittedStep,
    FusedKernel,
    GateSpec,
    default_names,
    enabled,
    enabled_names,
    kernel,
    kernels,
    register,
    resolve,
    resolve_admitted,
)
from . import (
    gdn_post_rows,
    gdn_pre_rows,
    gdn_rows_wrap,
    gdn_step,
    gr_read,
    gr_write,
    greedy_tail,
    moe_combine,
    moe_post,
    mtp_accept,
    ple,
    position_derive,
    qsa_block,
    router_tail,
    sampler_tail,
    shared_expert,
    sparse_sdpa_tiled,
)
from . import gr_fold  # the line transports gr_recip_last runs on
from . import gr_recip_last  # after gr_fold: it composes gr_fold's transports over gr_read's programs
from . import moe_dense  # after router_tail, shared_expert and gr_read: it hosts their kernels and untilize_rows'
from . import gdn_prefill_rows  # after the two rows programs: it wires them together for the prefill slab
from . import qsa_rows  # after qsa_block: it composes its score merge over the verify tile's rows
from . import gdn_rows_scan  # after gdn_rows_wrap and gdn_step: the verify rows' fold composes both

__all__ = [
    "ALL",
    "BITWISE",
    "COMPONENT",
    "DEFAULT_ON",
    "ENV",
    "OFF_ENV",
    "TOLERANCE_CLASSES",
    "ULP",
    "FusedKernel",
    "GateSpec",
    "default_names",
    "enabled",
    "enabled_names",
    "gdn_post_rows",
    "gdn_pre_rows",
    "gdn_prefill_rows",
    "gdn_rows_scan",
    "gdn_rows_wrap",
    "gdn_step",
    "gr_read",
    "gr_recip_last",
    "gr_write",
    "greedy_tail",
    "kernel",
    "kernels",
    "moe_combine",
    "moe_dense",
    "moe_post",
    "ple",
    "position_derive",
    "program",
    "qsa_block",
    "qsa_rows",
    "register",
    "resolve",
    "router_tail",
    "sampler_tail",
    "shared_expert",
    "sparse_sdpa_tiled",
    "AdmittedStep",
    "resolve_admitted",
]
