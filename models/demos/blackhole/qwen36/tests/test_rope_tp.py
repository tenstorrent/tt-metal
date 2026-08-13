# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partial-RoPE PCC for the Qwen3.6-27B full-attention layer, TP path.

Two tests (prefill / decode), same shape as the GDN suite: the **torch implementation
of the model** is the reference and the **exact rope path the text demo drives** is the
TTNN device-under-test.

Qwen3.6 uses PARTIAL rotary (partial_rotary_factor=0.25): only the first
``rope_head_dim = head_dim * 0.25`` dims of each head are rotated (HF split-halves
format); the remaining dims pass through unchanged.

* reference — HF ``apply_rotary_pos_emb`` from ``modeling_qwen3_5`` (the model's own
  function; it handles the partial split + pass-through concat).
* TTNN — ``rope_tp.rot_mats_prefill/rot_mats_decode`` (cos/sin generation) +
  ``apply_partial_rope_prefill/apply_partial_rope_decode`` (the slice→rotate→concat the
  demo's TPAttention calls at ``attention/tp.py:155-156`` prefill / ``:236-237`` decode).

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_rope_tp.py -v -s
"""
import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    get_pcc_threshold,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
)
from models.demos.blackhole.qwen36.tt.attention.rope_tp import (
    apply_partial_rope_decode,
    apply_partial_rope_prefill,
    rot_mats_decode,
    rot_mats_prefill,
)
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def _cos_sin(positions, rope_dim, theta):
    """HF split-halves cos/sin for the given positions. positions: [L] -> [L, rope_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    freqs = torch.outer(positions.float(), inv_freq)  # [L, rope_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [L, rope_dim]
    return emb.cos(), emb.sin()


def _read0(mesh_device, x):
    """Read device-0's copy of a replicated tensor (rope is per-head, identical per device)."""
    t = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return t[0].float()


def _rope_params(mesh_device, B):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=B, max_seq_len=256)
    HD = args.head_dim
    rope_dim = args.rope_head_dim
    theta = args.rope_theta
    # Per-device head shards (matches TPAttention); fall back to full counts on 1 device.
    NH = getattr(args, "n_local_heads", args.n_heads)
    NKV = getattr(args, "n_local_kv_heads", args.n_kv_heads)
    logger.info(f"HD={HD} rope_dim={rope_dim} (partial={rope_dim < HD}) theta={theta} NH={NH} NKV={NKV}")
    assert rope_dim < HD, "expected partial rotary (rope_dim < head_dim)"
    return args, HD, rope_dim, theta, NH, NKV


@torch.no_grad()
@parametrize_mesh_tp()
def test_partial_rope_prefill(mesh_device, reset_seeds, ensure_gc, request):
    """Prefill partial-RoPE: TTNN apply_partial_rope_prefill vs HF apply_rotary_pos_emb."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    S = 128
    _, HD, rope_dim, theta, NH, NKV = _rope_params(mesh_device, B=1)

    # q: [1, NH, S, HD], k: [1, NKV, S, HD]  (post q_norm/k_norm layout, heads in dim 1)
    q = torch.randn(1, NH, S, HD, dtype=torch.bfloat16)
    k = torch.randn(1, NKV, S, HD, dtype=torch.bfloat16)

    # ---- TTNN (demo path) ----
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, rope_dim, S, theta)
    q_tt = apply_partial_rope_prefill(replicate_to_device(mesh_device, q), cos_tt, sin_tt, NH, rope_dim)
    k_tt = apply_partial_rope_prefill(replicate_to_device(mesh_device, k), cos_tt, sin_tt, NKV, rope_dim)
    q_out, k_out = _read0(mesh_device, q_tt), _read0(mesh_device, k_tt)

    # ---- torch reference (model's apply_rotary_pos_emb, positions 0..S-1, unsqueeze_dim=1) ----
    cos, sin = _cos_sin(torch.arange(S), rope_dim, theta)  # [S, rope_dim]
    q_ref, k_ref = apply_rotary_pos_emb(q.float(), k.float(), cos.unsqueeze(0), sin.unsqueeze(0), unsqueeze_dim=1)

    thr = get_pcc_threshold(request)
    pq, pccq = comp_pcc(q_ref, q_out, thr)
    pk, pcck = comp_pcc(k_ref, k_out, thr)
    logger.info(f"partial-RoPE prefill PCC  q={pccq}  k={pcck}")
    assert pq and pk, f"partial-RoPE prefill PCC too low: q={pccq} k={pcck}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_partial_rope_decode(mesh_device, reset_seeds, ensure_gc, request):
    """Decode partial-RoPE: per-user positions. TTNN apply_partial_rope_decode vs HF reference."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    B = 8
    _, HD, rope_dim, theta, NH, NKV = _rope_params(mesh_device, B=B)
    positions = torch.arange(B, dtype=torch.int32) * 7 + 3  # distinct per-user positions

    # q: [1, B, NH, HD], k: [1, B, NKV, HD]  (decode layout, heads in dim 2)
    q = torch.randn(1, B, NH, HD, dtype=torch.bfloat16)
    k = torch.randn(1, B, NKV, HD, dtype=torch.bfloat16)

    # ---- TTNN (demo path) ----
    cos_tt, sin_tt = rot_mats_decode(mesh_device, rope_dim, 256, theta, positions)
    q_tt = apply_partial_rope_decode(replicate_to_device(mesh_device, q), cos_tt, sin_tt, NH, B, rope_dim)
    k_tt = apply_partial_rope_decode(replicate_to_device(mesh_device, k), cos_tt, sin_tt, NKV, B, rope_dim)
    q_out, k_out = _read0(mesh_device, q_tt), _read0(mesh_device, k_tt)

    # ---- torch reference (per-user positions, heads in dim 2 -> unsqueeze_dim=2) ----
    cos, sin = _cos_sin(positions, rope_dim, theta)  # [B, rope_dim]
    q_ref, k_ref = apply_rotary_pos_emb(q.float(), k.float(), cos.unsqueeze(0), sin.unsqueeze(0), unsqueeze_dim=2)

    thr = get_pcc_threshold(request)
    pq, pccq = comp_pcc(q_ref, q_out, thr)
    pk, pcck = comp_pcc(k_ref, k_out, thr)
    logger.info(f"partial-RoPE decode PCC  q={pccq}  k={pcck}")
    assert pq and pk, f"partial-RoPE decode PCC too low: q={pccq} k={pcck}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_decode_rope_device_gather(mesh_device, reset_seeds, ensure_gc, request):
    """Device-gathered decode cos/sin == the host-computed pack it replaces.

    ``Qwen36Model._gather_decode_rope`` indexes device-resident cos/sin tables with the per-user
    rope positions instead of running the trig on host every token. This checks both halves of
    that substitution: the numerics against ``rot_mats_decode`` (the host path), and the output
    RANK, which differs between TP ([1, B, 1, rope_dim]) and single device ([1, B, rope_dim]) and
    is what the downstream apply_partial_rope_decode expects.
    """
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.demos.blackhole.qwen36.tt.rope import compute_rope_freqs

    B = 8
    max_seq = 256
    args, _, rope_dim, theta, _, _ = _rope_params(mesh_device, B=B)
    positions = torch.arange(B, dtype=torch.int32) * 7 + 3

    cos_cpu, sin_cpu = compute_rope_freqs(head_dim=rope_dim, max_seq_len=max_seq, theta=theta)

    # Minimal stand-in carrying only what _gather_decode_rope reads, so the method under test runs
    # without materializing the 27B weights.
    class _Shim:
        pass

    shim = _Shim()
    shim.device = mesh_device
    shim.num_devices = mesh_device.get_num_devices()
    shim.args = args
    shim.rope = _Shim()
    shim.rope.cos_cpu, shim.rope.sin_cpu = cos_cpu, sin_cpu
    shim._rope_tbl = None
    # _gather_decode_rope calls this on self; bind the real implementation so the table build is
    # exercised too.
    shim._rope_gather_tables = lambda: Qwen36Model._rope_gather_tables(shim)

    pos_tt = ttnn.from_torch(
        positions.reshape(1, B).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos_dev, sin_dev = Qwen36Model._gather_decode_rope(shim, pos_tt)

    expected_rank = 4 if shim.num_devices > 1 else 3
    assert len(cos_dev.shape) == expected_rank, f"rank {len(cos_dev.shape)} != {expected_rank}"

    cos_ref_tt, sin_ref_tt = rot_mats_decode(mesh_device, rope_dim, max_seq, theta, positions)
    cos_ref, sin_ref = _read0(mesh_device, cos_ref_tt), _read0(mesh_device, sin_ref_tt)
    cos_out, sin_out = _read0(mesh_device, cos_dev), _read0(mesh_device, sin_dev)

    thr = get_pcc_threshold(request)
    pc, pccc = comp_pcc(cos_ref.reshape(B, rope_dim), cos_out.reshape(B, rope_dim), thr)
    ps, pccs = comp_pcc(sin_ref.reshape(B, rope_dim), sin_out.reshape(B, rope_dim), thr)
    logger.info(f"device-gathered decode rope PCC  cos={pccc}  sin={pccs}")
    assert pc and ps, f"device rope gather PCC too low: cos={pccc} sin={pccs}"
