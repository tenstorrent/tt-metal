# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shared ``device_params`` dicts for the FLUX.2 Klein test suite (``tests/models/flux3_klein/device_params.py``).

Every device test parametrizes the repo-global ``device_params`` fixture (indirect) with one of these. They build on
the fabric/topology base dicts in ``models/tt_dit/utils/test.py`` and layer on the knobs this model needs: an
L1_SMALL region for the VAE conv2d, a trace region for the traced denoising step and the VAE decode, and
``require_exact_physical_num_devices`` so one CI command self-selects the right mesh per SKU. The host-only rows
never import this module (it pulls in ``ttnn``); they share ``common.py`` instead.

The gate is the Blackhole Galaxy 4x8 at SP8 x TP4 Ring with 8 KB packets (the port's default layout,
``pipelines/flux2_klein/demo_flux2_klein.py`` ``open_mesh``); the values are the pipeline's own.
"""

from ....utils.test import (
    line_params_8k,
    line_params_req_exact_devices,
    ring_params_8k,
    ring_params_8k_req_exact_devices,
)

# Applied to a row so it self-skips unless the mesh it asks for is exactly what the machine has.
REQ_EXACT = {"require_exact_physical_num_devices": True}
# The FLUX.2 VAE (decoder and encoder) uses conv2d, which needs L1_SMALL buffers on every mesh.
L1_SMALL_SIZE = 65536
# The denoiser traces one step (both CFG forwards + the Euler update) and the VAE decode of the canvas size; the
# value is the TOTAL over the DRAM banks (demo_flux2_klein.py TRACE_REGION_SIZE).
TRACE_REGION_SIZE = 256_000_000

# Base fabric/topology params + L1_SMALL for the VAE conv2d.
line_params_flux2_klein = {**line_params_req_exact_devices, "l1_small_size": L1_SMALL_SIZE}
line_params_8k_flux2_klein = {**line_params_8k, "l1_small_size": L1_SMALL_SIZE}
ring_params_8k_flux2_klein = {**ring_params_8k, "l1_small_size": L1_SMALL_SIZE}
ring_params_8k_flux2_klein_req_exact = {**ring_params_8k_req_exact_devices, "l1_small_size": L1_SMALL_SIZE}

# Transformer / pipeline: base + trace region.
line_params_flux2_klein_transformer = {**line_params_flux2_klein, "trace_region_size": TRACE_REGION_SIZE}
ring_params_8k_flux2_klein_transformer = {**ring_params_8k_flux2_klein, "trace_region_size": TRACE_REGION_SIZE}

# Prompt encoder: fabric only, no L1_SMALL (it has no conv2d); the 4x8 rows take the ring dict above.
prompt_encoder_params_flux2_klein = line_params_req_exact_devices
