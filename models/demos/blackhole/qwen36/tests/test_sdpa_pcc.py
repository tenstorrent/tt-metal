# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit: scaled_dot_product_attention_decode kernel fidelity vs a torch reference.

Decode uses ttnn.transformer.scaled_dot_product_attention_decode reading a KV cache. This measures
that op's fidelity against exact softmax(qKᵀ·scale)V computed on the SAME bf16-rounded q/K/V (so the
PCC gap is the KERNEL's, not input quantization). We sweep the two axes that could explain why the
user's branch (yito/qwen36_27b_p300x2_tp) had better decode accuracy than main:
  - program_config: main (grid (8,8), q/k_chunk=0) vs branch (q/k_chunk=32, max_cores_per_head_batch=16)
  - compute_kernel_config: HiFi2 (main default, math_approx_mode=True) vs HiFi4 (math_approx_mode=False)
Both MHA (NKV=NH) and GQA (NKV=1, kernel broadcasts) are tested.

Correct API shapes (from model.forward_decode + branch attn()):
  q          [1, B, NH, HD]            TILE
  K/V cache  [B, NKV, max_seq, HD]     TILE, bf16
  cur_pos    [B] int32                 ROW_MAJOR, replicated   <-- was [NH]=8 (wrong) → hard crash

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_sdpa_pcc.py -s
"""
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.tp_common import COMPUTE_HIFI2, COMPUTE_HIFI4


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _torch_ref(q, K, V, scale):
    # exact attention on the (already bf16-rounded) tensors, upcast to fp32 for the matmul.
    # q [1,NH,1,HD]; K/V [1,NH,S,HD] (GQA: NKV<NH keys repeated). Returns [NH, HD].
    NH = q.shape[1]
    NKV = K.shape[1]
    rep = NH // NKV
    Kr = K.repeat_interleave(rep, dim=1).float()
    Vr = V.repeat_interleave(rep, dim=1).float()
    attn = torch.softmax((q.float() @ Kr.transpose(-1, -2)) * scale, dim=-1)  # [1,NH,1,S]
    return (attn @ Vr)[0, :, 0, :]


@parametrize_mesh_tp()
def test_sdpa_pcc(mesh_device):
    from loguru import logger

    NH, HD, S, MAXS = 8, 128, 128, 256
    scale = HD**-0.5
    B = 1
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    main_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
    )
    branch_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=32,
        max_cores_per_head_batch=16,
    )

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    # (NKV, cache_len): full-cache (MAXS==S, matches branch's cur_pos=ctx-1) vs partial-fill (MAXS>S,
    # the real decode case where positions > cur_pos are still zero).
    for NKV, MAXS in ((NH, S), (NH, 256), (1, S), (1, 256)):
        # bf16-round the reference inputs so the PCC gap reflects the kernel, not quantization.
        q = (torch.randn(1, NH, 1, HD) * 0.5).to(torch.bfloat16).float()
        K = (torch.randn(1, NKV, S, HD) * 0.5).to(torch.bfloat16).float()
        V = (torch.randn(1, NKV, S, HD) * 0.5).to(torch.bfloat16).float()
        o_ref = _torch_ref(q, K, V, scale)  # [NH, HD]

        # device tensors: q [1,B,NH,HD]; cache [B,NKV,MAXS,HD] with [:S] filled, rest zero
        q_dec = q.permute(0, 2, 1, 3).contiguous()  # [1,1,NH,HD]
        Kc = torch.zeros(B, NKV, MAXS, HD)
        Vc = torch.zeros(B, NKV, MAXS, HD)
        Kc[:, :, :S] = K[0]
        Vc[:, :, :S] = V[0]
        cur = ttnn.from_torch(
            torch.tensor([S - 1] * B, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )

        def run(prog, comp):
            qd, kd, vd = dev(q_dec), dev(Kc), dev(Vc)
            o = ttnn.transformer.scaled_dot_product_attention_decode(
                qd, kd, vd, cur_pos_tensor=cur, scale=scale, program_config=prog, compute_kernel_config=comp
            )
            ot = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(NH, HD)
            for t in (qd, kd, vd, o):
                ttnn.deallocate(t)
            return ot

        variants = [
            ("main_cfg+HiFi2", main_cfg, COMPUTE_HIFI2),
            ("main_cfg+HiFi4", main_cfg, COMPUTE_HIFI4),
            ("branch_cfg+HiFi2", branch_cfg, COMPUTE_HIFI2),
            ("branch_cfg+HiFi4", branch_cfg, COMPUTE_HIFI4),
        ]
        head = "MHA" if NKV == NH else f"GQA(NKV={NKV})"
        kind = f"{head} cache={'full' if MAXS == S else 'partial'}"
        for name, prog, comp in variants:
            try:
                o = run(prog, comp)
                logger.info(f"SDPA_DECODE {kind} {name}: PCC_vs_torch={_pcc(o, o_ref):.6f}")
            except Exception as e:
                logger.info(f"SDPA_DECODE {kind} {name}: FAILED {type(e).__name__}: {str(e)[:120]}")
