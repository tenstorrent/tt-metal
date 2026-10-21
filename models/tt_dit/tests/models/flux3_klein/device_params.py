# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shared ``device_params`` dicts for the FLUX 3 Image Klein test suite.

Every device test parametrizes the repo-global ``device_params`` fixture (indirect) with one of these. They
build on the fabric/topology base dicts in ``models/tt_dit/utils/test.py`` and layer on the knobs this model
needs: an L1_SMALL region for the VAE conv2d, a trace region for the traced denoising step, and
``require_exact_physical_num_devices`` so one CI command self-selects the right mesh per SKU. The host-only
tests never import this module (it pulls in ``ttnn``); they share ``common.py`` instead.

The gate is the Blackhole Galaxy 4x8 at SP4 x TP8 Ring with 8 KB packets (PLAN.md section 0, decision 2);
the line rows exist for submesh iteration.
"""

from ....utils.test import (
    line_params_8k,
    line_params_req_exact_devices,
    ring_params_8k,
    ring_params_8k_req_exact_devices,
)

# Applied to a row so it self-skips unless the mesh it asks for is exactly what the machine has.
REQ_EXACT = {"require_exact_physical_num_devices": True}
# The Flux2 VAE decoder (reused unchanged) uses conv2d, which needs L1_SMALL buffers on every mesh.
L1_SMALL_SIZE = 65536
# The transformer traces one denoising step (both CFG forwards + the Euler update, M4). The value is the TOTAL over
# the 8 DRAM banks (64 MB -> 8,003,584 B per bank). Measured per bank after one captured step (M4 design probes,
# 2026-10-17): 4x8 ring base CFG T2I 3,203,072 B (25.6 MB total, 82 % of the old 31 MB
# budget), ring distilled 1,622,016 B, (2,2) submesh base 1,523,712 B, distilled 778,240 B. One ring base trace plus
# any second trace (the other model, a 1-ref layout, a second length bucket) does not fit in 31 MB.
TRACE_REGION_SIZE = 256_000_000  # 2026-10-21: base 4096x4096 DiT trace 64.27 MB + the VAE trace exceeded 64 MB

# Base fabric/topology params + L1_SMALL for the VAE conv2d. line_params_flux3_klein already carries
# require_exact (via line_params_req_exact_devices).
line_params_flux3_klein = {**line_params_req_exact_devices, "l1_small_size": L1_SMALL_SIZE}
line_params_8k_flux3_klein = {**line_params_8k, "l1_small_size": L1_SMALL_SIZE}
ring_params_8k_flux3_klein = {**ring_params_8k, "l1_small_size": L1_SMALL_SIZE}
ring_params_8k_flux3_klein_req_exact = {**ring_params_8k_req_exact_devices, "l1_small_size": L1_SMALL_SIZE}

# Transformer: base + trace region.
line_params_flux3_klein_transformer = {**line_params_flux3_klein, "trace_region_size": TRACE_REGION_SIZE}
ring_params_8k_flux3_klein_transformer = {**ring_params_8k_flux3_klein, "trace_region_size": TRACE_REGION_SIZE}

# Performance: base + req-exact so each row self-selects its mesh.
line_params_flux3_klein_perf = {**line_params_flux3_klein, **REQ_EXACT}
line_params_8k_flux3_klein_perf = {**line_params_8k_flux3_klein, **REQ_EXACT}
ring_params_8k_flux3_klein_perf = {**ring_params_8k_flux3_klein, **REQ_EXACT}

# VAE single-device row: L1_SMALL only, no fabric and no require_exact (see flux2/device_params.py for why a
# 1-device submesh of a multi-chip box must not initialize fabric, and why no CCL op is issued on it anyway).
single_device_params_flux3_klein = {"l1_small_size": L1_SMALL_SIZE}

# Prompt encoder: fabric + req-exact, no L1_SMALL (it has no conv2d).
prompt_encoder_params_flux3_klein = line_params_req_exact_devices

# Prompt upsampler (M9): ONE chip, the qwen36 single-device parameters (demo/vision_demo.py DEVICE_PARAMS): L1_SMALL
# 24576 for the Gated DeltaNet prefill conv1d, two command queues, a 256 MiB trace region for the prefill chunk trace
# + the decode trace; NO fabric (a 1-device mesh of a multi-chip box must not initialise it) and no require_exact. The
# same dict is ``pipelines/flux3_klein/upsampler.py::PUP_DEVICE_PARAMS`` (asserted equal by the upsampler test).
PUP_DEVICE_PARAMS = {"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 268435456}
