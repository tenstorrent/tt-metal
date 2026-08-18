# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-layer GDN DECODE workload for Tracy / device-perf profiling of Qwen3.5/3.6.

Runs **one** Gated DeltaNet (``linear_attention``) decoder layer's ``forward(mode="decode")``
for a single recurrent step, over a sweep of batch sizes. Same pattern as
``test_profile_single_layer_prefill.py``: each measured iteration puts ``start``/``stop``
signposts around the ``forward`` call ONLY. Model build, weight load, embedding and teardown all
happen outside the signpost window, and a warmup iteration runs first without signposts so kernel
compilation and first-touch allocation are not measured.

WHY GDN DECODE IS ITS OWN TEST (NOT THE PREFILL FILE, NOT FULL-ATTENTION)
--------------------------------------------------------------------------
GDN decode is a single O(1) recurrent step (conv shift-register + rank-1 state update) --
completely different cost shape from the chunk-scan used in prefill (``test_profile_single_layer_
prefill.py``), whose cost grows with sequence length. Decode has no such length axis: the GDN
layer carries no KV cache and the recurrent step's cost is independent of how many tokens came
before it (state is a fixed-size ``[B, Nv, Dk, Dv]`` tensor). The dimension that DOES matter for
decode is batch size, so this test sweeps ``DECODE_BATCH_SIZES`` instead of sequence length. It is
also GDN-only, unlike the prefill file's GDN-vs-full-attention pair: full-attention decode needs
a paged KV cache and a running position, and mixing that setup in here would just add unused
plumbing since GDN's forward path never touches ``cos``/``sin``/``page_table``.

WHY THESE BATCH SIZES
----------------------
1 is the latency floor (single in-flight request). 8 and 32 bracket the batch sizes actually used
by continuous-batching decode in ``model.py`` (e.g. ``test_gdn_tp_peruser_state`` batches of
8/32), showing how the per-token recurrence amortizes as more user rows share the same kernel
launch.

Standalone Tracy capture::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B \\
      python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_single_layer_gdn_decode.py::test_profile_single_layer_gdn_decode[wormhole_b0-batch8-mesh_device0-device_params0]"

    Note the ``-m`` (profile a library module) and the full node id with NO trailing pytest flags:
    tracy parses argv with optparse and does not call disable_interspersed_args, so a later ``-v``
    is taken as tracy's own verbose and ``-k`` fails as an unknown option. Without ``-m``, argv[0]
    is opened as a *script path* and you get "FileNotFoundError: 'pytest'".

Plain run (no profiler; sanity-checks the workload itself)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_single_layer_gdn_decode.py -v -s
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.tests.perf.perf_signposts import install_attention_signposts, install_mlp_signposts

# Layer 0 is always GDN (``linear_attention``) in the hybrid G,G,G,F,... pattern regardless of how
# many layers the model is truncated to, so a 1-layer build is enough to isolate it.
NUM_LAYERS = 1
GDN_LAYER_IDX = 0
DECODE_BATCH_SIZES = [1, 8, 32]
NUM_WARMUP_ITERS = 1


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)

# Multi-device needs fabric_config: the TP all-gather / reduce-scatter run over the ETH links and
# without FABRIC_1D they are never discovered, after which the first CCL hangs and wedges the ETH
# cores. num_command_queues=2 + trace_region match demo/text_demo.py so the profile reflects the
# served configuration.
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1024 * 1024 * 1024} if _MULTI else {}),
    }
]


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


class _LayerPerfFixtures(NamedTuple):
    model: object
    layer: object
    layer_idx: int
    batch_size: int
    x: ttnn.Tensor


def _setup(mesh_device, batch_size: int) -> _LayerPerfFixtures:
    """Build the model + the single decode-step input the GDN layer needs. Nothing here is profiled."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=batch_size, max_seq_len=256, n_layers=NUM_LAYERS)
    layer = model.layers[GDN_LAYER_IDX]
    assert not layer.is_full_attention, f"layer {GDN_LAYER_IDX} expected GDN (linear_attention)"

    # One decode token per batch row: [1, B] -> embed -> [1, 1, B, dim_frac], matching
    # Qwen36Model.decode_tp's per-step input shape.
    torch.manual_seed(0)
    tokens = torch.randint(0, 2000, (1, batch_size), dtype=torch.long)
    tok = ttnn.from_torch(
        tokens.to(torch.int32),
        dtype=ttnn.uint32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x = model.embd(tok)
    x = ttnn.reshape(x, (1, 1, batch_size, x.shape[-1]))
    ttnn.deallocate(tok)

    logger.info(f"profiling layer {GDN_LAYER_IDX} (GDN decode), batch_size={batch_size}, mesh={MESH_SHAPE}")
    return _LayerPerfFixtures(model=model, layer=layer, layer_idx=GDN_LAYER_IDX, batch_size=batch_size, x=x)


def _run_decode_step(mesh_device, f: _LayerPerfFixtures, *, use_signpost: bool = False) -> None:
    """One recurrent decode forward on the single layer. Only the forward sits inside the signposts."""
    if use_signpost:
        from tracy import signpost

        signpost("start")

    out = f.layer.forward(f.x, mode="decode")

    if use_signpost:
        # Inside the window on purpose: without it the clock stops on dispatch, not execution.
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    else:
        ttnn.synchronize_device(mesh_device)

    ttnn.deallocate(out)


@pytest.mark.timeout(3600)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("batch_size", DECODE_BATCH_SIZES, ids=[f"batch{b}" for b in DECODE_BATCH_SIZES])
def test_profile_single_layer_gdn_decode(mesh_device, device_params, batch_size):
    """One decode step through ONE Qwen3.5/3.6 GDN decoder layer (Tracy profile target)."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, batch_size)

    for _ in range(NUM_WARMUP_ITERS):
        _run_decode_step(mesh_device, f)

    # Nested MLP signposts only on the measured iteration, so the warmup stays unmarked.
    restores = [g(f.layer) for g in (install_attention_signposts, install_mlp_signposts)] if use_signpost else []
    try:
        _run_decode_step(mesh_device, f, use_signpost=use_signpost)
    finally:
        for r in restores:
            r()

    ttnn.deallocate(f.x)

    logger.info(
        f"Profile workload complete: layer={f.layer_idx} (GDN decode), batch_size={f.batch_size}, "
        f"signposts={'on' if use_signpost else 'off'}"
    )
