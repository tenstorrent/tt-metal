# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shared ``device_params`` dicts for the Flux2 test suite.

Every Flux2 test parametrizes the repo-global ``device_params`` fixture (indirect) with one of
these. They build on the fabric/topology base dicts in ``models/tt_dit/utils/test.py`` and layer on
the three knobs Flux2 needs: an L1_SMALL region for the VAE conv2d, a trace region for the
transformer denoising loop, and ``require_exact_physical_num_devices`` so one CI command self-selects
the right mesh per SKU. Centralizing them here keeps the L1_SMALL value and the req-exact overlay
defined once instead of copy-pasted across the five test modules.
"""

from ....utils.test import line_params_8k, line_params_req_exact_devices, ring_params_8k

# Applied to a row so it self-skips unless the mesh it asks for is exactly what the machine has.
REQ_EXACT = {"require_exact_physical_num_devices": True}
# The Flux2 VAE uses conv2d, which needs L1_SMALL buffers on every mesh.
L1_SMALL_SIZE = 65536
# The transformer traces the denoising loop.
TRACE_REGION_SIZE = 31_000_000

# Base fabric/topology params + L1_SMALL for the VAE conv2d. line_params_flux2 already carries
# require_exact (via line_params_req_exact_devices).
line_params_flux2 = {**line_params_req_exact_devices, "l1_small_size": L1_SMALL_SIZE}
line_params_8k_flux2 = {**line_params_8k, "l1_small_size": L1_SMALL_SIZE}
ring_params_8k_flux2 = {**ring_params_8k, "l1_small_size": L1_SMALL_SIZE}
ring_params_8k_flux2_req_exact = {**ring_params_8k_flux2, **REQ_EXACT}

# Transformer: base + trace region.
line_params_flux2_transformer = {**line_params_flux2, "trace_region_size": TRACE_REGION_SIZE}

# Performance: base + req-exact so each row self-selects its mesh.
line_params_flux2_perf = {**line_params_flux2, **REQ_EXACT}
line_params_8k_flux2_perf = {**line_params_8k_flux2, **REQ_EXACT}
ring_params_8k_flux2_perf = {**ring_params_8k_flux2, **REQ_EXACT}

# VAE single-device row: L1_SMALL only, no fabric and no require_exact.
# Don't set require_exact so it runs on every machine rather than only on a literal one-chip host, which doesn't exist.

# No fabric because FABRIC_1D on a 1-device submesh of a multi-chip box doesn't work: fabric gets initialized
# on 1 device, but router sync still waits on a router
# on another chip (that isn't configured), so the handshake stays at STARTED and times
# out in fabric_firmware_initializer.cpp.

# Even if the bug above were fixed, CCLManager.all_gather returns early when shape[mesh_axis] == 1, _all_gather_hw
# when both factors are 1, vae_all_gather when the cluster axis is 1 -- so no CCL op is ever
# issued and fabric is redundant.
single_device_params_flux2 = {"l1_small_size": L1_SMALL_SIZE}

# Prompt encoder: fabric + req-exact, no L1_SMALL (it has no conv2d). Same shape as
# line_params_req_exact_devices; aliased here so all Flux2 device params have one home.
prompt_encoder_params_flux2 = line_params_req_exact_devices
