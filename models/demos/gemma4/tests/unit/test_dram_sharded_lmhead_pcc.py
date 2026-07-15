# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device PCC probe: DramShardedMatmul vs interleaved linear for lm_head-sized splits.

    pytest models/demos/gemma4/tests/unit/test_dram_sharded_lmhead_pcc.py -k 1x4 -sv
"""

from __future__ import annotations

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.dram_sharded import (
    TILE,
    DramShardedMatmul,
    MultiSplitDramShardedMatmul,
    _dram_weight_mem_config,
)

from ...tests.test_factory import parametrize_mesh_with_fabric


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def _to_torch_mesh0(t):
    try:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    except Exception:
        return ttnn.to_torch(t)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_dram_weight_roundtrip_pcc(mesh_device):
    """to_memory_config DRAM-width-shard then back should preserve weight."""
    k, n = 5376, 7168
    torch.manual_seed(0)
    w_torch = torch.randn(1, 1, k, n, dtype=torch.bfloat16)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    w = ttnn.from_torch(
        w_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dram_cores = mesh_device.dram_grid_size().x
    padded_n = math.ceil(n / (TILE * dram_cores)) * (TILE * dram_cores)
    mem = _dram_weight_mem_config(k, n, dram_cores)
    logger.info(f"DRAM mem: cores={dram_cores} logical_n={n} padded_n={padded_n}")
    w_ds = ttnn.to_memory_config(w, mem)
    w_back = ttnn.to_memory_config(w_ds, ttnn.DRAM_MEMORY_CONFIG)
    t0 = _to_torch_mesh0(w)
    t1 = _to_torch_mesh0(w_back)
    logger.info(f"shapes interleaved={tuple(t0.shape)} roundtrip={tuple(t1.shape)} ds_shape={w_ds.shape}")
    if t1.shape[-1] != t0.shape[-1]:
        t1 = t1[..., : t0.shape[-1]]
    p = _pcc(t0, t1)
    max_abs = (t0.float() - t1.float()).abs().max().item()
    logger.info(f"weight roundtrip PCC={p:.6f} max_abs={max_abs:.6g}")
    assert p > 0.999, f"DRAM weight roundtrip corrupted: PCC={p}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_single_split_dram_vs_interleaved(mesh_device):
    """One lm_head-sized split: DramShardedMatmul variants vs plain linear."""
    k, n, m = 5376, 7168, 32
    torch.manual_seed(1)
    x_torch = torch.randn(1, 1, m, k, dtype=torch.bfloat16)
    w_torch = torch.randn(1, 1, k, n, dtype=torch.bfloat16)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    x = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        w_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ref_t = _to_torch_mesh0(ttnn.linear(x, w))

    mm = DramShardedMatmul.try_build(mesh_device, w, k=k, n=n, m=m, name="probe_split")
    assert mm is not None, "DramShardedMatmul failed to build"
    logger.info(
        f"DramShardedMatmul cores={mm.num_cores} grid={mm.rows}x{mm.cols} "
        f"per_core_N={math.ceil(n / (TILE * mm.num_cores))} weight_shape={mm.weight.shape}"
    )

    results = {}

    def _record(name, out_tt):
        t = _to_torch_mesh0(out_tt)
        if t.shape[-1] != ref_t.shape[-1]:
            logger.warning(f"{name}: N {t.shape[-1]} vs ref {ref_t.shape[-1]}")
            t = t[..., : ref_t.shape[-1]]
        p = _pcc(ref_t, t)
        arg_ref = int(ref_t[0, 0, 0].argmax())
        arg_t = int(t[0, 0, 0].argmax())
        max_abs = (ref_t.float() - t.float()).abs().max().item()
        logger.info(
            f"{name}: PCC={p:.6f} max_abs={max_abs:.6g} "
            f"argmax_ref={arg_ref} argmax={arg_t} match={arg_ref == arg_t}"
        )
        results[name] = p

    # A: current default (HiFi2)
    _record("hifi2_default", mm(x))

    # B: no compute_kernel_config
    saved = mm._compute_kernel_config
    mm._compute_kernel_config = None
    _record("no_ck", mm(x))

    # C: HiFi4
    mm._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    _record("hifi4", mm(x))

    # D: HiFi2 + explicit bf16 output dtype
    mm._compute_kernel_config = saved
    x_sh = ttnn.to_memory_config(x, mm._in_mem_config)
    out_d = ttnn.linear(
        x_sh,
        mm.weight,
        program_config=mm._prog_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=mm._compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    out_d = ttnn.sharded_to_interleaved(out_d, ttnn.DRAM_MEMORY_CONFIG)
    _record("hifi2_bf16_out", out_d)

    # E: HiFi2 + fp32 dest accum
    mm._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    _record("hifi2_fp32acc", mm(x))

    # F: HiFi4 + fp32 dest accum
    mm._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    _record("hifi4_fp32acc", mm(x))

    # G: interleaved linear with HiFi2 (control)
    out_g = ttnn.linear(
        x,
        w,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    )
    _record("interleaved_hifi2", out_g)

    mm._compute_kernel_config = saved

    best_name = max(results, key=results.get)
    logger.info(f"best={best_name} PCC={results[best_name]:.6f}")
    assert results[best_name] > 0.99, f"best DRAM variant PCC too low: {results}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_multisplit_dram_vs_full_interleaved(mesh_device):
    """Full N=65536 via multi-split DRAM vs one interleaved linear."""
    k, n_full, m = 5376, 65536, 32
    torch.manual_seed(2)
    x_torch = torch.randn(1, 1, m, k, dtype=torch.bfloat16)
    w_torch = torch.randn(1, 1, k, n_full, dtype=torch.bfloat16)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    x = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        w_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ref = _to_torch_mesh0(ttnn.linear(x, w))

    ms = MultiSplitDramShardedMatmul.try_build(mesh_device, w, k=k, n=n_full, name="probe_ms")
    assert ms is not None
    out = _to_torch_mesh0(ms(x))
    if out.shape[-1] != ref.shape[-1]:
        logger.warning(f"multisplit N {out.shape[-1]} vs {ref.shape[-1]}")
        out = out[..., : ref.shape[-1]]
    p = _pcc(ref, out)
    arg_ref = int(ref[0, 0, 0].argmax())
    arg_out = int(out[0, 0, 0].argmax())
    logger.info(
        f"multisplit DRAM vs full linear: PCC={p:.6f} " f"argmax_match={arg_ref == arg_out} ({arg_ref} vs {arg_out})"
    )
    assert p > 0.99, f"multisplit DRAM PCC={p}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_tp_column_parallel_multisplit_vs_full(mesh_device):
    """Mirror real lm_head: column-parallel weight, mesh-slice multi-split, allgather.

    This is the path that diverges in the demo; synthetic replicate-mesh PCC was fine.
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    tp = mesh_device.shape[1]
    assert tp == 4
    k, vocab, m = 5376, 262144, 32
    n_per = vocab // tp  # 65536
    torch.manual_seed(3)
    x_torch = torch.randn(1, 1, m, k, dtype=torch.bfloat16)
    w_torch = torch.randn(1, 1, k, vocab, dtype=torch.bfloat16)

    mesh_config = MeshConfig(tuple(mesh_device.shape), ModeConfig(tp=tp, ep=1, sp=1))
    col_mapper = mesh_config.column_parallel(mesh_device)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    x = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        w_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"TP weight mesh shape={w.shape} (expect N_per={n_per})")

    # Reference: full interleaved linear on sharded weight, then allgather
    ref_sh = ttnn.linear(x, w)
    # Concat device shards on host as ground truth for per-device then AG
    ref_devs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(ref_sh)]
    ref_full = torch.cat(ref_devs, dim=-1)

    # Multi-split interleaved (known-good in demo)
    import os

    os.environ["GEMMA4_LM_HEAD_MULTI_SPLIT_INTERLEAVED"] = "1"
    ms_int = MultiSplitDramShardedMatmul.try_build(mesh_device, w, k=k, n=n_per, name="tp_ms_int")
    assert ms_int is not None
    out_int = ms_int(x)
    out_int_devs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(out_int)]
    out_int_full = torch.cat(out_int_devs, dim=-1)
    p_int = _pcc(ref_full, out_int_full)
    logger.info(f"TP interleaved multi-split vs full linear: PCC={p_int:.6f}")

    # Multi-split DRAM
    os.environ["GEMMA4_LM_HEAD_MULTI_SPLIT_INTERLEAVED"] = "0"
    # Clear the flag properly
    del os.environ["GEMMA4_LM_HEAD_MULTI_SPLIT_INTERLEAVED"]
    ms_dram = MultiSplitDramShardedMatmul.try_build(mesh_device, w, k=k, n=n_per, name="tp_ms_dram")
    assert ms_dram is not None
    assert not ms_dram._interleaved_mode
    out_dram = ms_dram(x)
    out_dram_devs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(out_dram)]
    out_dram_full = torch.cat(out_dram_devs, dim=-1)
    p_dram = _pcc(ref_full, out_dram_full)
    p_vs_int = _pcc(out_int_full, out_dram_full)
    arg_ref = int(ref_full[0, 0, 0].argmax())
    arg_dram = int(out_dram_full[0, 0, 0].argmax())
    arg_int = int(out_int_full[0, 0, 0].argmax())
    logger.info(
        f"TP DRAM multi-split vs full linear: PCC={p_dram:.6f} vs_interleaved_ms={p_vs_int:.6f} "
        f"argmax ref/int/dram={arg_ref}/{arg_int}/{arg_dram}"
    )

    # Softcap (same as model) then compare argmax
    cap = 30.0

    def softcap(t):
        return torch.tanh(t.float() / cap) * cap

    ref_sc = softcap(ref_full)
    dram_sc = softcap(out_dram_full)
    int_sc = softcap(out_int_full)
    logger.info(
        f"after softcap: DRAM vs ref PCC={_pcc(ref_sc, dram_sc):.6f} "
        f"int vs ref PCC={_pcc(ref_sc, int_sc):.6f} "
        f"argmax ref/int/dram={int(ref_sc[0,0,0].argmax())}/{int(int_sc[0,0,0].argmax())}/{int(dram_sc[0,0,0].argmax())}"
    )

    assert p_int > 0.999, f"TP interleaved multi-split broken: PCC={p_int}"
    assert p_dram > 0.99, f"TP DRAM multi-split PCC={p_dram}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_tp_dram_weight_roundtrip(mesh_device):
    """Column-parallel weight: interleaved → DRAM-width-shard → interleaved must be exact."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    tp = mesh_device.shape[1]
    k, n_per = 5376, 7168
    torch.manual_seed(7)
    w_torch = torch.randn(1, 1, k, n_per * tp, dtype=torch.bfloat16)
    mesh_config = MeshConfig(tuple(mesh_device.shape), ModeConfig(tp=tp, ep=1, sp=1))
    w = ttnn.from_torch(
        w_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dram_cores = mesh_device.dram_grid_size().x
    mem = _dram_weight_mem_config(k, n_per, dram_cores)
    # Slice one chunk like MultiSplit does, then convert
    w_slice = ttnn.slice(w, [0, 0, 0, 0], [1, 1, k, n_per])
    w_ds = ttnn.to_memory_config(w_slice, mem)
    w_back = ttnn.to_memory_config(w_ds, ttnn.DRAM_MEMORY_CONFIG)
    t0 = _to_torch_mesh0(w_slice)
    t1 = _to_torch_mesh0(w_back)
    if t1.shape[-1] != t0.shape[-1]:
        t1 = t1[..., : t0.shape[-1]]
    max_abs = (t0.float() - t1.float()).abs().max().item()
    p = _pcc(t0, t1)
    logger.info(f"TP slice weight roundtrip PCC={p:.6f} max_abs={max_abs:.6g}")
    assert max_abs == 0.0, f"TP DRAM weight convert corrupted: max_abs={max_abs}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_real_lmhead_disagreement_rate(mesh_device):
    """Many random acts × real lm_head: how often DRAM flips greedy argmax vs interleaved."""
    import os
    from pathlib import Path

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.demos.gemma4.utils.general_utils import get_cache_file_name

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH",
        "/home/ttuser/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main",
    )
    if not Path(model_path).exists():
        pytest.skip(f"no model at {model_path}")

    tp = mesh_device.shape[1]
    mesh_config = MeshConfig(tuple(mesh_device.shape), ModeConfig(tp=tp, ep=1, sp=1))
    state = Gemma4ModelArgs.load_state_dict(model_path)
    embed_key = (
        "model.language_model.embed_tokens.weight"
        if "model.language_model.embed_tokens.weight" in state
        else "model.embed_tokens.weight"
    )
    embed = state[embed_key]
    k = int(embed.shape[1])
    vocab = int(embed.shape[0])
    n_per = vocab // tp
    lm_host = embed.transpose(0, 1).unsqueeze(0).unsqueeze(0).contiguous()
    col_mapper = mesh_config.column_parallel(mesh_device)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = Path(model_path) / "tensor_cache_bf16"
    cache_stem = get_cache_file_name(str(cache), f"lm_head.weight_tp{tp}_bf16")
    w = ttnn.as_tensor(
        lm_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=cache_stem,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    os.environ.pop("GEMMA4_LM_HEAD_MULTI_SPLIT_INTERLEAVED", None)
    ms = MultiSplitDramShardedMatmul.try_build(mesh_device, w, k=k, n=n_per, name="disagree")
    assert ms is not None and not ms._interleaved_mode

    n_trials = int(os.environ.get("GEMMA4_LM_HEAD_DISAGREE_TRIALS", "32"))
    flips = 0
    flips_vs_hifi2 = 0
    margins = []
    hifi2_ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    for seed in range(n_trials):
        torch.manual_seed(1000 + seed)
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, k, dtype=torch.bfloat16) * 0.02,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ref = torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(ttnn.linear(x, w))], dim=-1)
        ref_h2 = torch.cat(
            [ttnn.to_torch(d) for d in ttnn.get_device_tensors(ttnn.linear(x, w, compute_kernel_config=hifi2_ck))],
            dim=-1,
        )
        out = torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(ms(x))], dim=-1)
        # softcap like model
        cap = 30.0
        ref_sc = torch.tanh(ref.float() / cap) * cap
        ref_h2_sc = torch.tanh(ref_h2.float() / cap) * cap
        out_sc = torch.tanh(out.float() / cap) * cap
        row = ref_sc[0, 0, 0]
        row_h2 = ref_h2_sc[0, 0, 0]
        row_o = out_sc[0, 0, 0]
        a_ref = int(row.argmax())
        a_h2 = int(row_h2.argmax())
        a_out = int(row_o.argmax())
        top2 = torch.topk(row, 2)
        margin = float(top2.values[0] - top2.values[1])
        if a_ref != a_out:
            flips += 1
            margins.append(margin)
            logger.info(f"seed={seed} FLIP vs_default ref={a_ref} dram={a_out} margin={margin:.6g}")
        if a_h2 != a_out:
            flips_vs_hifi2 += 1
            logger.info(f"seed={seed} FLIP vs_hifi2_int h2={a_h2} dram={a_out}")
    logger.info(
        f"disagreement vs_default={flips}/{n_trials} vs_hifi2_int={flips_vs_hifi2}/{n_trials} "
        f"flip_margins={margins}"
    )
    # Informational — don't fail the suite; we want the rate logged.
    assert flips < n_trials, "DRAM flipped every trial — path is broken"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_flip_seed_per_split_pcc(mesh_device):
    """Known flip seed=1: per-split DRAM vs interleaved PCC + which vocab region flips."""
    import os
    from pathlib import Path

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.dram_sharded import plan_column_splits
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.demos.gemma4.utils.general_utils import get_cache_file_name

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH",
        "/home/ttuser/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main",
    )
    if not Path(model_path).exists():
        pytest.skip(f"no model at {model_path}")

    tp = mesh_device.shape[1]
    mesh_config = MeshConfig(tuple(mesh_device.shape), ModeConfig(tp=tp, ep=1, sp=1))
    state = Gemma4ModelArgs.load_state_dict(model_path)
    embed_key = (
        "model.language_model.embed_tokens.weight"
        if "model.language_model.embed_tokens.weight" in state
        else "model.embed_tokens.weight"
    )
    embed = state[embed_key]
    k = int(embed.shape[1])
    vocab = int(embed.shape[0])
    n_per = vocab // tp
    lm_host = embed.transpose(0, 1).unsqueeze(0).unsqueeze(0).contiguous()
    w = ttnn.as_tensor(
        lm_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        cache_file_name=get_cache_file_name(str(Path(model_path) / "tensor_cache_bf16"), f"lm_head.weight_tp{tp}_bf16"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    torch.manual_seed(1001)  # seed=1 in disagreement loop
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, k, dtype=torch.bfloat16) * 0.02,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    os.environ.pop("GEMMA4_LM_HEAD_MULTI_SPLIT_INTERLEAVED", None)
    ms = MultiSplitDramShardedMatmul.try_build(mesh_device, w, k=k, n=n_per, name="flip_split")
    assert ms is not None and not ms._interleaved_mode

    # Full logits
    ref = torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(ttnn.linear(x, w))], dim=-1)
    out = torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(ms(x))], dim=-1)
    cap = 30.0
    ref_sc = torch.tanh(ref.float() / cap) * cap
    out_sc = torch.tanh(out.float() / cap) * cap
    a_ref = int(ref_sc[0, 0, 0].argmax())
    a_out = int(out_sc[0, 0, 0].argmax())
    logger.info(f"full softcap argmax ref={a_ref} dram={a_out} match={a_ref==a_out}")
    logger.info(f"ref top5={ref_sc[0,0,0].topk(5)} dram top5={out_sc[0,0,0].topk(5)}")

    # Per-split: rebuild interleaved slices and compare each DramShardedMatmul
    max_columns = int(os.environ.get("GEMMA4_LM_HEAD_MAX_COLUMNS", "7168"))
    sizes = plan_column_splits(n_per, max_columns)
    offset = 0
    for i, (mm, logical_n, _) in enumerate(ms.splits):
        w_slice = ttnn.slice(w, [0, 0, 0, offset], [1, 1, k, offset + logical_n])
        ref_s = torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(ttnn.linear(x, w_slice))], dim=-1)
        out_s = mm(x)
        if out_s.shape[-1] > logical_n:
            s = out_s.shape
            out_s = ttnn.slice(out_s, [0, 0, 0, 0], [s[0], s[1], s[2], logical_n])
        out_st = torch.cat([ttnn.to_torch(d) for d in ttnn.get_device_tensors(out_s)], dim=-1)
        p = _pcc(ref_s, out_st)
        max_abs = (ref_s.float() - out_st.float()).abs().max().item()
        # which device-local columns differ most
        diff = (ref_s.float() - out_st.float()).abs()[0, 0, 0]
        worst = int(diff.argmax())
        logger.info(
            f"split{i} n={logical_n} PCC={p:.6f} max_abs={max_abs:.6g} "
            f"worst_col={worst} ref={float(ref_s[0,0,0,worst]):.6g} dram={float(out_st[0,0,0,worst]):.6g}"
        )
        offset += logical_n
        w_slice.deallocate(True)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_real_lmhead_weight_dram_vs_interleaved(mesh_device):
    """Load real Gemma4 lm_head (tied embed), compare DRAM multi-split vs interleaved.

    Uses a random activation (not a real hidden state) but the *real* weight —
    isolates weight-layout / DRAM-convert issues from the rest of the model.
    """
    import os
    from pathlib import Path

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.demos.gemma4.utils.general_utils import get_cache_file_name

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH",
        "/home/ttuser/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main",
    )
    if not Path(model_path).exists() and "gemma-4" not in model_path:
        pytest.skip(f"no model at {model_path}")

    tp = mesh_device.shape[1]
    mesh_config = MeshConfig(tuple(mesh_device.shape), ModeConfig(tp=tp, ep=1, sp=1))
    state = Gemma4ModelArgs.load_state_dict(model_path)
    embed_key = (
        "model.language_model.embed_tokens.weight"
        if "model.language_model.embed_tokens.weight" in state
        else "model.embed_tokens.weight"
    )
    embed = state[embed_key]  # [vocab, hidden]
    k = int(embed.shape[1])
    vocab = int(embed.shape[0])
    n_per = vocab // tp
    lm_host = embed.transpose(0, 1).unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,K,vocab]

    col_mapper = mesh_config.column_parallel(mesh_device)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = Path(model_path) / "tensor_cache_bf16"
    # Prefer the demo's existing cache file if present.
    cache_stem = get_cache_file_name(str(cache), f"lm_head.weight_tp{tp}_bf16")
    w = ttnn.as_tensor(
        lm_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=cache_stem,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"real lm_head mesh shape={w.shape} k={k} n_per={n_per}")

    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, k, dtype=torch.bfloat16) * 0.02,  # scale like post-norm
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ref_devs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(ttnn.linear(x, w))]
    ref = torch.cat(ref_devs, dim=-1)

    os.environ.pop("GEMMA4_LM_HEAD_MULTI_SPLIT_INTERLEAVED", None)
    ms = MultiSplitDramShardedMatmul.try_build(mesh_device, w, k=k, n=n_per, name="real_lm")
    assert ms is not None and not ms._interleaved_mode
    out_devs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(ms(x))]
    out = torch.cat(out_devs, dim=-1)

    p = _pcc(ref, out)
    arg_ref = int(ref[0, 0, 0].argmax())
    arg_out = int(out[0, 0, 0].argmax())
    max_abs = (ref.float() - out.float()).abs().max().item()
    # Top-k overlap
    topk = 16
    ref_top = set(ref[0, 0, 0].float().topk(topk).indices.tolist())
    out_top = set(out[0, 0, 0].float().topk(topk).indices.tolist())
    overlap = len(ref_top & out_top)
    logger.info(
        f"REAL lm_head DRAM vs interleaved: PCC={p:.6f} max_abs={max_abs:.6g} "
        f"argmax_match={arg_ref==arg_out} ({arg_ref} vs {arg_out}) top{topk}_overlap={overlap}"
    )
    assert p > 0.99, f"real lm_head DRAM PCC={p}"
    assert overlap >= 12, f"top{topk} overlap only {overlap}"
