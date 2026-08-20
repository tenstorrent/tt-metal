# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every LLM ``LayerNormDeviceOperation`` for Tracy / device-perf reports of Qwen3.5-9B.

Each case runs ONE production ``ttnn.rms_norm`` (or the DistributedNorm wrapper that ends in one)
with ``start``/``stop`` signposts around that call ONLY. Weight construction, residual upload and
teardown sit outside the window. Identical kernels across the decoder stack are represented by one
instance of each kind; the report is per-call, not 32× stacked.

WHY THIS IS ITS OWN TEST
------------------------
MLP and LM-head profile files each include a hidden-dim norm mixed with a matmul. Layer captures
mix every RMSNorm with attention/GDN/MLP. This file is the Tracy target for the layernorm kernel
itself, so a ``tt-perf-report`` of one node id is that op's report -- the input layout, last-dim,
and memory config the production forward actually uses.

KINDS (every LLM ``LayerNormDeviceOperation`` on Qwen3.5-9B)
-----------------------------------------------------------
Hidden-dim residual (DistributedNorm / PrefillTunedDistributedNorm; 9B is gather-then-rms_norm):

``attn``     -- every-layer ``attention_norm``. Decode: 32-core attn sharded config. Prefill 9B: L1 out.
``ff``       -- every-layer ``ffn_norm``. Decode reuses attn layout; prefill DRAM.
``final``    -- once-per-forward ``Qwen36Model.norm``. Decode: ``lm_head`` grid, output forced DRAM.

Full-attention per-head (raw ``ttnn.rms_norm``, L1 interleaved, last dim ``head_dim``):

``q_norm``   -- ``TPAttention._qk_norm`` on Q. Decode ``[1,B,NH,HD]``; prefill ``[1,NH,S,HD]``.
``k_norm``   -- same on K. Decode ``[1,B,NKV,HD]``; prefill ``[1,NKV,S,HD]``.
               Wormhole fuses (1+w) into the rms_norm weight; Blackhole keeps rms_norm + multiply
               (only the rms_norm sits in the window).

GDN (raw ``ttnn.rms_norm``; GDN output has no +1):

``gdn_l2``   -- decode Q/K L2. Same kernel for q and k after GQA expand: ``[B,1,Nv,Dk]``,
               unweighted, ``epsilon=1e-6/Dk``. Prefill L2 is in-kernel inside
               ``chunk_gated_delta_rule`` (``flat_qkv_enabled``) -- not a host LayerNormDeviceOperation,
               so there is no prefill case.
``gdn_out``  -- GDN gated output norm (``norm.weight``). Decode (WH tile-opt): ``[B,Nv,1,Dv]`` L1.
               Prefill: head-major ``[Nv,T,Dv]``; DRAM once ``Nv*T*Dv*4 > 8MB`` (fp32-byte threshold).

Vision ``DistributedLayerNorm`` (mean+var) is a different pipeline and is not in this file.

Dummy weights of the real 9B shapes. PCC lives in ``tests/unit/test_rms_norm.py``.

SHAPES
------
Decode B=1/32; prefill T=128/2048/4096. Hidden-dim input is fractured TILE DRAM
(``[1,1,B or T, dim/tp]``). Per-head tensors are replicated (local NH/NKV/Nv already).

Standalone Tracy capture (N300 9B, q_norm decode B=32)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B \\
      python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_rmsnorm.py::test_profile_rmsnorm[wormhole_b0-q_norm-decode-B32-mesh_device0-device_params0]"

    Note the ``-m`` and the full node id with NO trailing pytest flags: tracy parses argv
    with optparse and a later ``-v`` is taken as tracy's own verbose. Without ``-m``,
    argv[0] is opened as a script path and you get "FileNotFoundError: 'pytest'".

Then, from the generated Tracy CSV::

    tt-perf-report generated/profiler/reports/<run>/ops_perf_results_*.csv \\
      --start-signpost start --end-signpost stop --no-color

    If stacked FLOPs summary crashes (empty groupby), add ``--no-summary``.

Plain run (no profiler; sanity-checks the workload itself)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_rmsnorm.py -v -s
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

HIDDEN_KINDS = ("attn", "ff", "final")
# (mode, length): decode length is batch, prefill length is sequence.
LENGTHS = (
    ("decode", 1),
    ("decode", 32),
    ("prefill", 128),
    ("prefill", 2048),
    ("prefill", 4096),
)

# One node id per production LayerNormDeviceOperation × shape. gdn_l2 is decode-only (see docstring).
CASES = (
    tuple((kind, mode, n) for kind in HIDDEN_KINDS for mode, n in LENGTHS)
    + tuple((kind, mode, n) for kind in ("q_norm", "k_norm") for mode, n in LENGTHS)
    + tuple(("gdn_l2", "decode", n) for _, n in LENGTHS if _ == "decode")
    + tuple(("gdn_out", mode, n) for mode, n in LENGTHS)
)
CASE_IDS = [f"{kind}-{mode}-{'B' if mode == 'decode' else 'T'}{n}" for kind, mode, n in CASES]


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)

# fabric_config on multi-device: DistributedNorm all-gather runs over ETH.
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


class _NormFixtures(NamedTuple):
    kind: str
    mode: str
    length: int
    dim: int
    num_devices: int
    args: object
    x: ttnn.Tensor
    norm: object = None
    weight: object = None
    epsilon: float = 1e-6
    memory_config: object = None


def _replicate_weight(mesh_device, w_1d: torch.Tensor):
    """Same as ``tpc.replicate`` for a 1-D norm weight (no cache)."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    return tpc.replicate(w_1d, mesh_device, None)


def _upload(mesh_device, x_torch, *, shard_last: bool, memory_config):
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if shard_last else ttnn.ReplicateTensorToMesh(mesh_device)
    x = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    if memory_config is not None and memory_config != ttnn.DRAM_MEMORY_CONFIG:
        x_l1 = ttnn.to_memory_config(x, memory_config)
        ttnn.deallocate(x)
        return x_l1
    return x


def _norm_config(kind: str, args, mode: str, num_devices: int):
    """Same per-kind memory config the production forward passes (hidden-dim only)."""
    from models.tt_transformers.tt.common import Mode

    _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
    if kind == "final":
        if mode == "decode" and num_devices > 1:
            nc = dict(args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            return nc
        return None
    if num_devices > 1:
        if kind == "attn":
            nc = args.get_norm_config("attn", _norm_mode)
            if _norm_mode == Mode.PREFILL and (args.dim <= 4096 or is_blackhole()):
                nc = {**nc, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            return nc
        if _norm_mode == Mode.DECODE:
            return args.get_norm_config("attn", _norm_mode)
        return args.get_norm_config("ff", _norm_mode)
    return {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None


def _make_hidden_norm(mesh_device, args, kind: str, tt_ccl):
    """Build production residual RMSNorm. Mirrors layer._make_norm / Qwen36Model.norm."""
    from models.common.rmsnorm import RMSNorm
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    num_devices = getattr(args, "num_devices", 1)
    dim = args.dim

    if kind == "final":
        state = {"norm.weight": torch.randn(dim, dtype=torch.bfloat16)}
        norm = RMSNorm(
            device=mesh_device,
            dim=dim,
            state_dict=state,
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

    if kind == "attn":
        weight_key, ag_key, prefix = "input_layernorm", "attention_norm", "layers.0."
        fuse_agmm = (
            num_devices > 1 and is_blackhole() and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
        )
        gather_dtype = None
    else:
        weight_key, ag_key, prefix = "post_attention_layernorm", "ff_norm", "layers.0."
        fuse_agmm = tpc.mlp_gateup_agmm_enabled(num_devices)
        gather_dtype = ttnn.bfloat8_b if (dim > 4096 and not is_blackhole()) else None

    state = {f"{prefix}{weight_key}.weight": torch.randn(dim, dtype=torch.bfloat16)}
    norm = RMSNorm(
        device=mesh_device,
        dim=dim,
        state_dict=state,
        weight_key=weight_key,
        state_dict_prefix=prefix,
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
            ag_config_key=ag_key,
            enable_all_gather=not fuse_agmm,
            prefill_gather_dtype=gather_dtype,
        )
    return norm


def _gdn_out_prefill_mem_config(nv: int, seq: int, dv: int):
    """Same L1/DRAM split as ``TPGatedDeltaNet.forward_prefill`` (fp32-byte threshold)."""
    if is_blackhole() or nv * seq * dv * 4 <= (8 << 20):
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def _setup(mesh_device, kind: str, mode: str, length: int) -> _NormFixtures:
    """Build one production RMSNorm + matching input. Nothing here is profiled."""
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    max_seq = max(length, 2048) if mode == "prefill" else 2048
    max_batch = length if mode == "decode" else 1
    # Load HF config from the real snapshot first. Passing dummy_weights=True into Qwen36ModelArgs
    # makes ModelArgs._set_hf_params look up LOCAL_HF_PARAMS[model_name], and model_name is the
    # hashed snapshot dir -- KeyError. Same pattern as the other profile files: construct normally,
    # then flip dummy_weights so loaders skip the weight-cache path.
    args = Qwen36ModelArgs(mesh_device=mesh_device, max_batch_size=max_batch, max_seq_len=max_seq)
    args.dummy_weights = True
    num_devices = getattr(args, "num_devices", 1)

    torch.manual_seed(0)
    tt_ccl = TT_CCL(mesh_device) if num_devices > 1 else None
    shard_hidden = num_devices > 1

    if kind in HIDDEN_KINDS:
        norm = _make_hidden_norm(mesh_device, args, kind, tt_ccl)
        x = _upload(
            mesh_device,
            torch.randn(1, 1, length, args.dim, dtype=torch.bfloat16),
            shard_last=shard_hidden,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.info(f"profiling RMSNorm kind={kind} mode={mode} length={length} x={tuple(x.shape)}")
        return _NormFixtures(
            kind=kind, mode=mode, length=length, dim=args.dim, num_devices=num_devices, args=args, x=x, norm=norm
        )

    _L1 = ttnn.L1_MEMORY_CONFIG
    if kind in ("q_norm", "k_norm"):
        nh = (
            getattr(args, "n_local_heads", args.n_heads)
            if kind == "q_norm"
            else getattr(args, "n_local_kv_heads", args.n_kv_heads)
        )
        hd = args.head_dim
        # Decode: nlp_create_qkv_heads_decode -> [1,B,NH,HD]. Prefill: nlp_create_qkv_heads -> [1,NH,S,HD].
        x_t = (
            torch.randn(1, length, nh, hd, dtype=torch.bfloat16)
            if mode == "decode"
            else torch.randn(1, nh, length, hd, dtype=torch.bfloat16)
        )
        x = _upload(mesh_device, x_t, shard_last=False, memory_config=_L1)
        # Production loads (raw + 1.0); Wormhole fuses that into rms_norm's weight.
        weight = _replicate_weight(mesh_device, torch.randn(hd, dtype=torch.float32) + 1.0)
        logger.info(f"profiling RMSNorm kind={kind} mode={mode} length={length} x={tuple(x.shape)} hd={hd}")
        return _NormFixtures(
            kind=kind,
            mode=mode,
            length=length,
            dim=hd,
            num_devices=num_devices,
            args=args,
            x=x,
            weight=weight,
            epsilon=1e-6,
            memory_config=_L1,
        )

    nv = getattr(args, "gdn_nv_tp", args.linear_num_value_heads)
    dk = getattr(args, "gdn_dk", args.linear_key_head_dim)
    dv = getattr(args, "gdn_dv", args.linear_value_head_dim)
    if kind == "gdn_l2":
        # WH 9B tile_opt: after GQA expand, q and k are both [B,1,Nv,Dk] in L1. Same rms_norm kernel.
        x = _upload(
            mesh_device,
            torch.randn(length, 1, nv, dk, dtype=torch.bfloat16),
            shard_last=False,
            memory_config=_L1,
        )
        logger.info(f"profiling RMSNorm kind={kind} mode={mode} length={length} x={tuple(x.shape)} dk={dk}")
        return _NormFixtures(
            kind=kind,
            mode=mode,
            length=length,
            dim=dk,
            num_devices=num_devices,
            args=args,
            x=x,
            epsilon=1e-6 / dk,
            memory_config=None,
        )

    # gdn_out
    weight = _replicate_weight(mesh_device, torch.randn(dv, dtype=torch.float32))
    if mode == "decode":
        # WH tile_opt recurrence returns [B,H,1,V]; rms_norm reduces last dim Dv.
        x = _upload(
            mesh_device,
            torch.randn(length, nv, 1, dv, dtype=torch.bfloat16),
            shard_last=False,
            memory_config=_L1,
        )
        out_mc = _L1
    else:
        # Fused chunk return_o_bh: [B*Nv,T,Dv] with B=1. Output mem config matches production.
        x = _upload(
            mesh_device,
            torch.randn(nv, length, dv, dtype=torch.bfloat16),
            shard_last=False,
            memory_config=_L1,
        )
        out_mc = _gdn_out_prefill_mem_config(nv, length, dv)
    logger.info(f"profiling RMSNorm kind={kind} mode={mode} length={length} x={tuple(x.shape)} dv={dv} out={out_mc}")
    return _NormFixtures(
        kind=kind,
        mode=mode,
        length=length,
        dim=dv,
        num_devices=num_devices,
        args=args,
        x=x,
        weight=weight,
        epsilon=1e-6,
        memory_config=out_mc,
    )


def _run_hidden(f: _NormFixtures):
    from models.tt_transformers.tt.common import Mode

    _norm_mode = Mode.PREFILL if f.mode == "prefill" else Mode.DECODE
    nc = _norm_config(f.kind, f.args, f.mode, f.num_devices)
    return f.norm(f.x, mode=_norm_mode, **({"norm_config": nc} if nc is not None else {}))


def _run_qk_norm(f: _NormFixtures):
    """Same as ``TPAttention._qk_norm``: WH fuses weight into rms_norm; BH is rms_norm then multiply."""
    if is_blackhole():
        return ttnn.rms_norm(f.x, epsilon=f.epsilon, memory_config=f.memory_config)
    return ttnn.rms_norm(f.x, weight=f.weight, epsilon=f.epsilon, memory_config=f.memory_config)


def _run_raw(f: _NormFixtures):
    kwargs = {"epsilon": f.epsilon}
    if f.memory_config is not None:
        kwargs["memory_config"] = f.memory_config
    if f.weight is not None:
        kwargs["weight"] = f.weight
    return ttnn.rms_norm(f.x, **kwargs)


def _run_norm(mesh_device, f: _NormFixtures, *, use_signpost: bool = False) -> None:
    """One LayerNormDeviceOperation. Only that call sits inside the signposts."""
    if use_signpost:
        from tracy import signpost

        signpost("start")

    if f.kind in HIDDEN_KINDS:
        out = _run_hidden(f)
    elif f.kind in ("q_norm", "k_norm"):
        out = _run_qk_norm(f)
    else:
        out = _run_raw(f)

    if use_signpost:
        # Inside the window on purpose: without it the clock stops on dispatch, not execution.
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    else:
        ttnn.synchronize_device(mesh_device)

    ttnn.deallocate(out)


@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("kind,mode,length", CASES, ids=CASE_IDS)
def test_profile_rmsnorm(mesh_device, device_params, kind, mode, length):
    """One production LayerNormDeviceOperation at a decode-batch or prefill-seq."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, kind, mode, length)

    for _ in range(NUM_WARMUP_ITERS):
        _run_norm(mesh_device, f)

    _run_norm(mesh_device, f, use_signpost=use_signpost)

    ttnn.deallocate(f.x)
    if f.weight is not None:
        ttnn.deallocate(f.weight)

    logger.info(
        f"Profile workload complete: RMSNorm kind={f.kind} mode={f.mode} length={f.length} "
        f"dim={f.dim} signposts={'on' if use_signpost else 'off'}"
    )
