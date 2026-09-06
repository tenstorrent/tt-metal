# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final RMSNorm + LM-head workload for Tracy / device-perf reports of Qwen3.5-9B.

Runs the same end-of-model block ``Qwen36Model`` uses after the last decoder layer, with
``start``/``stop`` signposts around that pair ONLY. Weight construction, residual upload and
teardown sit outside the window.

WHY THIS IS ITS OWN TEST
------------------------
Layer profile files stop at the decoder stack. Token embedding and MLP have isolated Tracy
targets; the final RMSNorm + vocab-sharded LM head (plus the logits all-gather) do not appear
in those captures. This file is the Tracy target for that block, so a ``tt-perf-report`` of
this test is a final-norm / LM-head report, not a layer report with an extra gather mixed in.

WHAT IS INSIDE THE WINDOW
-------------------------
Decode (``decode`` / ``decode_paged`` / ``_forward_decode``): ``_final_norm_decode`` then
``_lm_head``. On TP that is DistributedNorm with the framework ``lm_head`` sharded-norm config
(output forced back to DRAM) then ``ttnn.linear`` on a vocab-sharded BFP8 weight and
``tt_all_gather`` of the logit row.

Prefill: the production TP / chunked path selects the last token *before* this block, so the
logits matmul is always one tile row. Short single-device ``prefill()`` (T<=1024) instead runs
final-norm over the full sequence and then slices. Prefill cases here follow that short path
(full-seq DistributedNorm, last-row LM head) so the report can show how the pre-norm gather
scales with T. The LM head itself does not.

Dummy weights of the real 9B shapes (``norm.weight`` + ``output.weight`` [dim, vocab]). Matmul
cost does not depend on the values. PCC lives in ``tests/unit/test_lm_head.py``.

SHAPES
------
Input is the last-layer residual: TILE DRAM, fractured on the hidden dim under TP
(``[1,1,B or T, dim/tp]``). Decode uses the same batches as the embedding / MLP captures
(B=1, B=32). Prefill uses 128 / 2048 / 4096 for the final-norm gather.

Standalone Tracy capture (N300 9B decode B=32)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B \\
      python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_lm_head.py::test_profile_lm_head[wormhole_b0-decode-B32-mesh_device0-device_params0]"

    Note the ``-m`` and the full node id with NO trailing pytest flags: tracy parses argv
    with optparse and a later ``-v`` is taken as tracy's own verbose. Without ``-m``,
    argv[0] is opened as a script path and you get "FileNotFoundError: 'pytest'".

Then, from the generated Tracy CSV::

    tt-perf-report generated/profiler/reports/<run>/ops_perf_results_*.csv \\
      --start-signpost start --end-signpost stop --no-color

    If stacked FLOPs summary crashes (empty groupby), add ``--no-summary``.

Plain run (no profiler; sanity-checks the workload itself)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_lm_head.py -v -s
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

NUM_WARMUP_ITERS = 1

# (mode, length): decode length is batch, prefill length is sequence (final-norm rows).
CASES = (
    ("decode", 1),
    ("decode", 32),
    ("prefill", 128),
    ("prefill", 2048),
    ("prefill", 4096),
)


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)

# fabric_config on multi-device: final-norm all-gather + LM-head vocab all-gather run over ETH.
# Without FABRIC_1D a 2-chip N300 can fail ETH discovery (same as the other profile files).
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


class _LmHeadFixtures(NamedTuple):
    norm: object
    lm_head_weight: ttnn.Tensor
    vocab_sharded: bool
    tt_ccl: object
    x: ttnn.Tensor
    args: object
    mode: str
    length: int
    dim: int
    vocab: int
    num_devices: int


def _make_final_norm(mesh_device, args, state_dict, tt_ccl):
    """Build the production final RMSNorm (DistributedNorm under TP).

    Mirrors ``Qwen36Model.__init__`` — plain DistributedNorm, not PrefillTunedDistributedNorm
    (that wrapper is only for the per-layer attn/ff norms).
    """
    from models.common.rmsnorm import RMSNorm

    num_devices = getattr(args, "num_devices", 1)
    norm = RMSNorm(
        device=mesh_device,
        dim=args.dim,
        state_dict=state_dict,
        weight_key="norm",
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        add_unit_offset=True,
        eps=args.norm_eps,
        **(
            dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
            if num_devices > 1
            else {}
        ),
    )
    if num_devices > 1:
        from models.tt_transformers.tt.distributed_norm import DistributedNorm

        return DistributedNorm(norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy)
    return norm


def _final_norm(f: _LmHeadFixtures, x):
    """Same call as ``Qwen36Model._final_norm_decode`` (decode) or ``self.norm(..., PREFILL)``."""
    from models.tt_transformers.tt.common import Mode

    if f.mode == "decode":
        if f.num_devices > 1:
            nc = dict(f.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            return f.norm(x, mode=Mode.DECODE, norm_config=nc)
        return f.norm(x, mode=Mode.DECODE)
    return f.norm(x, mode=Mode.PREFILL)


def _lm_head(mesh_device, f: _LmHeadFixtures, x):
    """Same as ``Qwen36Model._lm_head``: linear, then vocab all-gather when the weight is sharded."""
    logits = ttnn.linear(x, f.lm_head_weight)
    if f.vocab_sharded:
        from models.tt_transformers.tt.ccl import tt_all_gather

        logits = tt_all_gather(
            logits,
            mesh_device,
            f.tt_ccl,
            cluster_axis=None,
            dim=len(logits.shape) - 1,
            topology=f.args.ccl_topology(),
            num_workers_per_link=4,
            chunks_per_sync=25,
        )
    return logits


def _setup(mesh_device, mode: str, length: int) -> _LmHeadFixtures:
    """Build final-norm + LM-head weight + one residual tensor. Nothing here is profiled."""
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    max_seq = max(length, 2048) if mode == "prefill" else 2048
    max_batch = length if mode == "decode" else 1
    # Construct Qwen36ModelArgs normally, then flip dummy_weights (same LOCAL_HF_PARAMS trap as
    # test_profile_token_embedding.py / test_profile_mlp.py).
    args = Qwen36ModelArgs(mesh_device=mesh_device, max_batch_size=max_batch, max_seq_len=max_seq)
    args.dummy_weights = True
    dim, vocab = args.dim, args.vocab_size
    num_devices = getattr(args, "num_devices", 1)

    torch.manual_seed(0)
    norm_state = {"norm.weight": torch.randn(dim, dtype=torch.bfloat16)}
    # Values unused; empty avoids a 1.2GB randn of [dim, vocab] on the host.
    lm_w = torch.empty(dim, vocab, dtype=torch.bfloat16)

    tt_ccl = TT_CCL(mesh_device) if num_devices > 1 else None
    norm = _make_final_norm(mesh_device, args, norm_state, tt_ccl)

    vocab_sharded = num_devices > 1 and vocab % num_devices == 0
    if vocab_sharded:
        lm_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    else:
        lm_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 else None
    lm_head_weight = ttnn.as_tensor(
        lm_w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **(dict(mesh_mapper=lm_mapper) if lm_mapper is not None else {}),
    )

    rows = length
    x_torch = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16)
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if num_devices > 1 else None
    x = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    logger.info(
        f"profiling final-norm+LM-head mode={mode} length={length} dim={dim} vocab={vocab} "
        f"vocab_sharded={vocab_sharded} mesh={MESH_SHAPE} x_last={x.shape[-1]}"
    )
    return _LmHeadFixtures(
        norm=norm,
        lm_head_weight=lm_head_weight,
        vocab_sharded=vocab_sharded,
        tt_ccl=tt_ccl,
        x=x,
        args=args,
        mode=mode,
        length=length,
        dim=dim,
        vocab=vocab,
        num_devices=num_devices,
    )


def _run_lm_head(mesh_device, f: _LmHeadFixtures, *, use_signpost: bool = False) -> None:
    """One final-norm + LM-head. Only those calls sit inside the signposts."""
    if use_signpost:
        from tracy import signpost

        signpost("start")

    xn = _final_norm(f, f.x)
    # Prefill: LM head is last-token only (production TP/chunked selects before the norm; short
    # prefill() norms the full sequence then slices). Decode runs the head on the whole batch.
    if f.mode == "prefill" and f.length > 1:
        x_last = xn[:, :, -1:, :]
        ttnn.deallocate(xn)
    else:
        x_last = xn
    logits = _lm_head(mesh_device, f, x_last)
    ttnn.deallocate(x_last)

    if use_signpost:
        # Inside the window on purpose: without it the clock stops on dispatch, not execution.
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    else:
        ttnn.synchronize_device(mesh_device)

    ttnn.deallocate(logits)


@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "mode,length",
    CASES,
    ids=[f"{mode}-{'B' if mode == 'decode' else 'T'}{n}" for mode, n in CASES],
)
def test_profile_lm_head(mesh_device, device_params, mode, length):
    """One final RMSNorm + LM-head at a production decode-batch or prefill-seq (Tracy target)."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, mode, length)

    for _ in range(NUM_WARMUP_ITERS):
        _run_lm_head(mesh_device, f)

    _run_lm_head(mesh_device, f, use_signpost=use_signpost)

    ttnn.deallocate(f.x)
    ttnn.deallocate(f.lm_head_weight)

    logger.info(
        f"Profile workload complete: final-norm+LM-head mode={f.mode} length={f.length} "
        f"dim={f.dim} vocab={f.vocab} signposts={'on' if use_signpost else 'off'}"
    )
