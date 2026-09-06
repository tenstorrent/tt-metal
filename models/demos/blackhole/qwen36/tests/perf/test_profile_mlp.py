# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLP / FFN workload for Tracy / device-perf reports of Qwen3.5-9B.

Runs the same post-attention FFN block the decoder uses after every attention residual
(``ffn_norm`` then ``Qwen36MLP.forward``), with ``start``/``stop`` signposts around that
pair ONLY. Weight construction, residual upload and teardown sit outside the window.

WHY THIS IS ITS OWN TEST
------------------------
A ``tt-perf-report`` of the single-layer attention/GDN profile captures is a full-layer
report: attention + both residuals + MLP. This file is the Tracy target for the FFN block
alone, so a report of this test is an MLP/FFN report, not a layer report with the MLP
mixed in.

The measured region is ``ffn_norm`` (pre-AG stats / gather / RMSNorm on 9B; post-AG on
27B) then gate/up, SwiGLU multiply, down-proj, and the MLP reduce-scatter. The trailing
residual add (``h + ff_output`` in ``layer.forward``) is outside the window.

Dummy weights of the real 9B shapes (gate/up/down + post-attention RMSNorm). Matmul cost
does not depend on the values, so loading the checkpoint would only add setup time. PCC
lives in ``tests/unit/test_mlp.py`` and ``tests/test_mlp_tp.py``.

SHAPES
------
The residual ``h`` is what ``ttnn.add(x, attn_out)`` produces in ``layer.forward``: TILE
DRAM, fractured on the hidden dim under TP (``[1,1,B or T, dim/tp]``). Decode uses the
same batches the attention-decode / embedding captures use (B=1, B=32). Prefill uses the
production chunk (2048) plus a short and a 2-chunk point so the report can show how the
2D matmuls scale with T.

Standalone Tracy capture (N300 9B decode B=32)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B \\
      python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_mlp.py::test_profile_mlp[wormhole_b0-decode-B32-mesh_device0-device_params0]"

    Note the ``-m`` and the full node id with NO trailing pytest flags: tracy parses argv
    with optparse and a later ``-v`` is taken as tracy's own verbose. Without ``-m``,
    argv[0] is opened as a script path and you get "FileNotFoundError: 'pytest'".

Then, from the generated Tracy CSV::

    tt-perf-report generated/profiler/reports/<run>/ops_perf_results_*.csv \\
      --start-signpost start --end-signpost stop --no-color

    If stacked FLOPs summary crashes (empty groupby), add ``--no-summary``.

Plain run (no profiler; sanity-checks the workload itself)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_mlp.py -v -s
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, run_for_wormhole_b0_or_blackhole

NUM_WARMUP_ITERS = 1

# (mode, length): decode length is batch, prefill length is sequence.
# ids must stay unique across the parametrize so tracy node ids are unambiguous.
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

# fabric_config on multi-device: ff_norm all-gather + MLP reduce-scatter run over ETH, and
# without FABRIC_1D a 2-chip N300 can fail ETH discovery (same as the other profile files).
# num_command_queues + trace_region match demo/text_demo.py.
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


class _MlpFixtures(NamedTuple):
    ffn_norm: object
    mlp: object
    h: ttnn.Tensor
    args: object
    mode: str
    length: int
    dim: int
    hidden_dim: int
    num_devices: int


def _ff_norm_config(args, mode: str, num_devices: int):
    """Same per-mode ff_norm memory config ``layer.forward`` passes."""
    from models.tt_transformers.tt.common import Mode

    _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
    if num_devices > 1:
        if _norm_mode == Mode.DECODE:
            return args.get_norm_config("attn", _norm_mode)
        return args.get_norm_config("ff", _norm_mode)
    return {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None


def _make_ffn_norm(mesh_device, args, state_dict, tt_ccl):
    """Build the production post-attention RMSNorm (PrefillTunedDistributedNorm under TP).

    Mirrors ``Qwen36DecoderLayer._make_norm`` for ``post_attention_layernorm`` so this
    capture cannot drift from the layer's FFN block without a matching change there.
    """
    from models.common.rmsnorm import RMSNorm
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    num_devices = getattr(args, "num_devices", 1)
    fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(num_devices)
    ff_gather_dtype = ttnn.bfloat8_b if (args.dim > 4096 and not is_blackhole()) else None

    norm = RMSNorm(
        device=mesh_device,
        dim=args.dim,
        state_dict=state_dict,
        weight_key="post_attention_layernorm",
        state_dict_prefix="layers.0.",
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
        from models.demos.blackhole.qwen36.tt.prefill_norm_tuned import PrefillTunedDistributedNorm

        return PrefillTunedDistributedNorm(
            norm,
            args,
            tt_ccl=tt_ccl,
            TG=args.is_galaxy,
            ag_config_key="ff_norm",
            enable_all_gather=not fuse_ff_agmm,
            prefill_gather_dtype=ff_gather_dtype,
        )
    return norm


def _setup(mesh_device, mode: str, length: int) -> _MlpFixtures:
    """Build ffn_norm + MLP + one residual tensor. Nothing here is profiled."""
    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    max_seq = max(length, 2048) if mode == "prefill" else 2048
    max_batch = length if mode == "decode" else 1
    # Load HF config from the real snapshot first. Passing dummy_weights=True into Qwen36ModelArgs
    # makes ModelArgs._set_hf_params look up LOCAL_HF_PARAMS[model_name], and model_name is the
    # hashed snapshot dir -- KeyError. Same pattern as tests/perf/test_profile_token_embedding.py:
    # construct normally, then flip dummy_weights so loaders skip the weight-cache path.
    args = Qwen36ModelArgs(mesh_device=mesh_device, max_batch_size=max_batch, max_seq_len=max_seq)
    args.dummy_weights = True
    dim, hidden_dim = args.dim, args.hidden_dim
    num_devices = getattr(args, "num_devices", 1)

    torch.manual_seed(0)
    norm_state = {"layers.0.post_attention_layernorm.weight": torch.randn(dim, dtype=torch.bfloat16)}
    mlp_state = {
        "gate_proj.weight": torch.randn(hidden_dim, dim, dtype=torch.bfloat16),
        "up_proj.weight": torch.randn(hidden_dim, dim, dtype=torch.bfloat16),
        "down_proj.weight": torch.randn(dim, hidden_dim, dtype=torch.bfloat16),
    }

    tt_ccl = TT_CCL(mesh_device) if num_devices > 1 else None
    ffn_norm = _make_ffn_norm(mesh_device, args, norm_state, tt_ccl)
    mlp = Qwen36MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)

    rows = length
    h_torch = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16)
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if num_devices > 1 else None
    h = ttnn.from_torch(
        h_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    logger.info(
        f"profiling MLP/FFN mode={mode} length={length} dim={dim} hidden_dim={hidden_dim} "
        f"mesh={MESH_SHAPE} h_last={h.shape[-1]}"
    )
    return _MlpFixtures(
        ffn_norm=ffn_norm,
        mlp=mlp,
        h=h,
        args=args,
        mode=mode,
        length=length,
        dim=dim,
        hidden_dim=hidden_dim,
        num_devices=num_devices,
    )


def _run_mlp(mesh_device, f: _MlpFixtures, *, use_signpost: bool = False) -> None:
    """One FFN block (ffn_norm + MLP). Only those two calls sit inside the signposts."""
    from models.tt_transformers.tt.common import Mode

    _norm_mode = Mode.PREFILL if f.mode == "prefill" else Mode.DECODE
    nc = _ff_norm_config(f.args, f.mode, f.num_devices)

    if use_signpost:
        from tracy import signpost

        signpost("start")

    ff_in = f.ffn_norm(f.h, mode=_norm_mode, norm_config=nc)
    ff_out = f.mlp.forward(ff_in)
    ttnn.deallocate(ff_in)

    if use_signpost:
        # Inside the window on purpose: without it the clock stops on dispatch, not execution.
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    else:
        ttnn.synchronize_device(mesh_device)

    ttnn.deallocate(ff_out)


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
def test_profile_mlp(mesh_device, device_params, mode, length):
    """One post-attention FFN block at a production decode-batch or prefill-seq (Tracy target)."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, mode, length)

    for _ in range(NUM_WARMUP_ITERS):
        _run_mlp(mesh_device, f)

    _run_mlp(mesh_device, f, use_signpost=use_signpost)

    ttnn.deallocate(f.h)

    logger.info(
        f"Profile workload complete: MLP/FFN mode={f.mode} length={f.length} "
        f"dim={f.dim} hidden_dim={f.hidden_dim} signposts={'on' if use_signpost else 'off'}"
    )
