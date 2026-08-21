# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free PCC gate and device-perf profile for the Qwen3.5 / 3.6 VISION TOWER.

``test_wrapped_model.py`` is the tower's REAL-WEIGHT PCC test. It needs ``dummy_weights=True``, which
routes the CONFIG through ``ModelArgs.LOCAL_HF_PARAMS``; that table used to have no ``Qwen3.5-9B``
entry, so the 9B raised ``KeyError: 'Qwen3.5-9B'`` before reaching the device and had no runnable
gate at all. It now has one (``model_params/Qwen3.5-9B/config.json``), so PREFER IT for numerics.

This test builds the HF reference from ``vision_config`` alone (``Qwen3_5VisionModel(vcfg)``, random
weights), which needs no checkpoint and makes it cheap to run anywhere.

WHAT CONFIG-INIT WEIGHTS CANNOT SEE
-----------------------------------
"Weight *values* do not matter to PCC, only that both sides use the same ones" is FALSE for any error
term that scales with the dynamic range of the activations. The trained tower develops MASSIVE
ACTIVATIONS from block 9 on -- on the 9B, absmax 354 against an rms of 0.65, and absmax 12032 at the
last block -- while config-init weights (iid, initializer_range 0.02) produce none. Anything that
quantizes an activation in a block-float format is therefore invisible here and severe in the real
model: this test read 0.99921 on the 9B while the real-weight tower was at 0.96875, because the
attention out-projection was writing the residual stream in bfloat8_b, whose 16-channel shared
exponent is set by the outlier. So treat this case as a CHEAP SMOKE TEST plus the profiling harness,
and treat ``test_wrapped_model.py`` as the gate.

TWO CASES, EACH AT THE DEPTH THAT SUITS IT
------------------------------------------
``oneblock``  -- ONE block, signpost-bounded, warmed up. This is the profiling case: with a single
    block every block op appears exactly once, so the perf report needs no dividing and no "read the
    second instance" caveat. A window is ``head + depth x block + tail``, so window totals are only
    comparable at equal depth -- keeping the profiled depth pinned at 1 is what makes the numbers in
    ``../VISION_TOWER_PERF.md`` mean the same thing run to run.

``fulldepth`` -- ALL ``vision_config.depth`` blocks, no warmup, no signposts. Depth matters for
    numerics: error compounds block over block, so a shallow check flatters the tower -- on the 9B,
    0.99977 at depth 1 against 0.99929 at the real depth of 27. (With REAL weights the same spread is
    0.99981 -> 0.98850, an order of magnitude wider; see above.) The host reference costs ~0.8 TFLOP
    per block, which is ~30 s for the whole tower at this grid.

Only the perf case emits signposts, so profiling the whole file still yields exactly ONE
``start``/``stop`` window -- but prefer ``-k oneblock`` so the full-depth reference is not computed just
to be thrown away.

Run the numerical gate (full depth)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s -k fulldepth

Profile one block::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B python -m tracy -p -v -r -m pytest \\
        models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s -k oneblock
    tt-perf-report --start-signpost start --end-signpost stop <ops_perf_results_*.csv>

For the 27B tower, point ``HF_MODEL`` at the LOCAL config dir -- ``ModelArgs`` takes
``CKPT_DIR = HF_MODEL`` and ``model_name`` from its basename, so no checkpoint or hub fetch is
needed (this tower's reference weights are config-init either way)::

    MESH_DEVICE=T3K HF_MODEL=$PWD/models/tt_transformers/model_params/Qwen3.6-27B pytest \\
        models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s -k fulldepth

``QWEN36_VISION_MM_TUNING=0`` runs either case with the Wormhole tuning gated off, for A/B.
"""

from __future__ import annotations

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

# (grid, depth, pcc_required, profile). depth=None means the config's full depth.
#
# The thresholds are MEASURED values with a little margin, not aspirations. Demo grid, config-init
# weights:
#
#              depth 1    depth 27 (full)
#   9B  / N300  0.99977    0.99929
#   27B / T3K   0.99965    0.99903
#
# These are HIGH because config-init weights have no activation outliers, not because the tower is
# accurate to 4 nines on real input -- the real-weight numbers are ~0.988 (9B) and ~0.998 (27B), in
# test_wrapped_model.py. A drop below these floors still means a real numerical regression, so they
# are worth keeping; they just cannot be the only gate.
#
# The full-depth number was once ~0.985 here, and that was NOT bfloat8_b weight error as previously
# assumed -- it was the sequence padding: the tower ran SDPA with `is_causal=False` and no
# `attn_mask`, so the pad rows acted as unmasked keys and every real query summed `exp(0)` over each
# of them. Tightening the pad from a 2048 multiple to the 128 the tower actually requires (see
# `DropInVisionTransformer.forward`) fixed it. Note depth 1 barely moved: that error only appears
# once compounded over depth, which is why the shallow case cannot be the gate.
CASES = [
    ((1, 86, 128), 1, 0.999, True),
    ((1, 86, 128), None, 0.998, False),
]

WEIGHT_DTYPE = ttnn.bfloat8_b


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}]


@torch.no_grad()
@pytest.mark.timeout(3600)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "grid, depth, pcc_required, profile",
    CASES,
    # Selector strings deliberately avoid "pcc" and "vision_tower": `-k` also matches the module and
    # function names, so `-k pcc` would select BOTH cases.
    ids=[
        f"{'oneblock' if prof else 'fulldepth'}_patches{math.prod(g)}_depth{d or 'full'}"
        for g, d, _, prof in CASES  # noqa: B023
    ],
)
def test_vision_tower_pcc(mesh_device, device_params, grid, depth, pcc_required, profile, tmp_path, reset_seeds):
    del device_params
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    mesh_device.enable_program_cache()
    n_patches = math.prod(grid)
    seq_len = ((n_patches // 2048) + 1) * 2048

    model_args = VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=seq_len)
    vcfg = model_args.hf_config.vision_config
    if depth is not None:
        vcfg.depth = depth
    merge = vcfg.spatial_merge_size
    assert grid[1] % merge == 0 and grid[2] % merge == 0, f"grid h,w must divide by {merge}"

    torch.manual_seed(0)
    reference_model = Qwen3_5VisionModel(vcfg).eval()
    tt_model = DropInVisionTransformer(
        reference_model,
        model_args,
        dtype=WEIGHT_DTYPE,
        debug=False,
        tt_ccl=TT_CCL(mesh_device),
        # Random weights must never land in the production ttnn cache (keyed by name only).
        weight_cache_path=tmp_path / "vision_pcc_weights",
    )

    pixel_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
    grid_thw = torch.tensor([grid], dtype=torch.long)
    pixel_values = torch.randn(n_patches, pixel_dim)

    # `seq_len` above only sizes `max_seq_len`; the tower pads rows to a multiple of 128, so report
    # what it will actually run at rather than the (larger) buffer bound.
    tower_rows = -(-n_patches // 128) * 128
    logger.info(
        f"{'PROFILE' if profile else 'PCC'} case: depth={vcfg.depth}, grid={grid} "
        f"({n_patches} patches -> {tower_rows} rows, max_seq_len {seq_len})"
    )
    reference_output = reference_model(pixel_values, grid_thw).pooler_output

    signpost = None
    if profile:
        # Warm up outside the signposts so kernel compilation and first-touch allocation are not
        # measured, then drain the on-device profiler buffer so those markers neither accumulate nor
        # land in the start..stop window. Both are no-ops without a profiler build.
        ttnn.deallocate(tt_model(pixel_values, grid_thw))
        read_profiler = getattr(ttnn, "ReadDeviceProfiler", None)
        if read_profiler is not None:
            read_profiler(mesh_device)
        try:
            from tracy import signpost
        except ImportError:
            logger.info("tracy.signpost unavailable; running without signpost markers.")

    if signpost is not None:
        signpost("start")
    tt_output = tt_model(pixel_values, grid_thw)
    ttnn.synchronize_device(mesh_device)
    if signpost is not None:
        signpost("stop")
        read_profiler = getattr(ttnn, "ReadDeviceProfiler", None)
        if read_profiler is not None:
            read_profiler(mesh_device)

    # The merger output is fractured along dim=3 (out_hidden_size/TP per device).
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)[:, : vcfg.out_hidden_size]
    ttnn.deallocate(tt_output)

    assert tt_output_torch.shape == reference_output.shape, f"{tt_output_torch.shape} != {reference_output.shape}"
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"vision tower depth={vcfg.depth} grid={grid}: PCC {pcc_message} (required {pcc_required})")
    assert passing, f"vision tower PCC below {pcc_required}: {pcc_message}"
