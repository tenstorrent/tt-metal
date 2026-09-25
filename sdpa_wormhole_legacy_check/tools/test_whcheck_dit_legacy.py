# SPDX-License-Identifier: Apache-2.0
"""whcheck: tt_dit legacy (non-Blackhole) SDPA path, identical on BASE and HEAD trees.
Random weights; saves tt outputs to $WHCHECK_OUT for bitwise BASE-vs-HEAD comparison,
and checks PCC against the diffusers torch module."""
import os
import diffusers.models.attention_processor
import pytest
import torch
import ttnn

from ...blocks import attention as flux1_attention
from ...models.transformers.wan2_2.attention_wan import WanAttention
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ...utils.tensor import bf16_tensor, from_torch

OUT = os.environ.get("WHCHECK_OUT", "/tmp/whcheck_out")
LINE = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1].item()


def _save(name, t):
    os.makedirs(OUT, exist_ok=True)
    torch.save(t.contiguous(), os.path.join(OUT, name + ".pt"))


def _parallel(mesh_device, sp_axis, tp_axis):
    shape = tuple(mesh_device.shape)
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=shape[tp_axis], mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=shape[sp_axis], mesh_axis=sp_axis),
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params"),
    [
        pytest.param((1, 1), 0, 1, {}, id="1x1"),
        pytest.param((1, 2), 1, 0, LINE, id="1x2sp1"),
        pytest.param((2, 4), 1, 0, LINE, id="2x4sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("prompt_seq_len", [128, 0], ids=["p128", "p0"])
def test_flux1_attention_legacy(mesh_device, sp_axis, tp_axis, prompt_seq_len):
    torch.manual_seed(0)
    heads, head_dim = 8, 128
    dim = heads * head_dim
    seq = 2048
    parallel_config = _parallel(mesh_device, sp_axis, tp_axis)
    sp = tuple(mesh_device.shape)[sp_axis]
    joint = prompt_seq_len > 0
    tm = diffusers.models.attention_processor.Attention(
        query_dim=dim, added_kv_proj_dim=dim if joint else None, dim_head=head_dim, heads=heads, out_dim=dim,
        context_pre_only=not joint, pre_only=not joint, bias=True, qk_norm="rms_norm", eps=1e-6,
        processor=diffusers.models.attention_processor.FluxAttnProcessor2_0(),
    ).eval()
    spatial = torch.randn((1, seq, dim))
    prompt = torch.randn((1, prompt_seq_len, dim)) if joint else None
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    th = torch.arange(prompt_seq_len + seq).float()[:, None] * inv[None, :]
    cos, sin = th.cos().repeat_interleave(2, -1), th.sin().repeat_interleave(2, -1)
    with torch.no_grad():
        out = tm.forward(spatial, prompt, image_rotary_emb=(cos, sin))
    ref_s, ref_p = out if joint else (out, None)

    ccl = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    A = flux1_attention.Attention
    tt_model = A(query_dim=dim, head_dim=head_dim, heads=heads, out_dim=dim, added_kv_proj_dim=dim if joint else 0,
                 context_pre_only=not joint, pre_only=not joint, eps=1e-6, mesh_device=mesh_device, ccl_manager=ccl,
                 parallel_config=parallel_config, padding_config=None)
    print("WHCHECK flux1 cfg", tt_model.sdpa_program_config.q_chunk_size, tt_model.sdpa_program_config.k_chunk_size,
          tt_model.sdpa_program_config.exp_approx_mode, tt_model.sdpa_compute_kernel_config)
    tt_model.load_torch_state_dict(dict(tm.state_dict()))
    tt_spatial = bf16_tensor(spatial, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_prompt = bf16_tensor(prompt, device=mesh_device) if joint else None
    srope = tuple(bf16_tensor(t[prompt_seq_len:], device=mesh_device, mesh_axis=sp_axis, shard_dim=-2) for t in (cos, sin))
    prope = tuple(bf16_tensor(t[:prompt_seq_len], device=mesh_device) for t in (cos, sin)) if joint else None
    s_out, p_out = tt_model.forward(spatial=tt_spatial, prompt=tt_prompt, spatial_rope=srope, prompt_rope=prope,
                                    spatial_sequence_length=seq)
    s = tensor.to_torch(s_out, mesh_axes=[..., sp_axis, tp_axis]).reshape(-1, dim)[:seq].float()
    tag = f"flux1_{tuple(mesh_device.shape)}_p{prompt_seq_len}".replace(" ", "")
    _save(tag + "_spatial", s)
    ps = _pcc(s, ref_s.reshape(-1, dim))
    print("WHCHECK", tag, "spatial pcc", ps)
    assert ps > 0.99
    if joint:
        p = tensor.to_torch(p_out, mesh_axes=[..., None, tp_axis]).reshape(-1, dim).float()
        _save(tag + "_prompt", p)
        pp = _pcc(p, ref_p.reshape(-1, dim))
        print("WHCHECK", tag, "prompt pcc", pp)
        assert pp > 0.99


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params"),
    [pytest.param((1, 1), 0, 1, {}, id="1x1"), pytest.param((1, 2), 1, 0, LINE, id="1x2sp1"),
     pytest.param((2, 4), 1, 0, LINE, id="2x4sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
def test_wan_self_attention_legacy(mesh_device, sp_axis, tp_axis):
    from diffusers.models.transformers.transformer_wan import WanAttention as TW, WanAttnProcessor
    torch.manual_seed(0)
    heads, head_dim = 8, 128
    dim = heads * head_dim
    seq = 2048
    tm = TW(dim=dim, heads=heads, dim_head=head_dim, eps=1e-6, cross_attention_dim_head=None, processor=WanAttnProcessor())
    with torch.no_grad():
        for p in tm.parameters():
            p.copy_(p.bfloat16().float())
    tm.eval()
    spatial = torch.randn(1, seq, dim).bfloat16().float()
    theta = torch.rand(1, seq, 1, head_dim // 2) * 2 * torch.pi
    tc, ts = stack_cos_sin(theta.cos(), theta.sin())
    with torch.no_grad():
        ref = tm(hidden_states=spatial, encoder_hidden_states=None, rotary_emb=[tc, ts])
    shape = tuple(mesh_device.shape)
    ccl = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = _parallel(mesh_device, sp_axis, tp_axis)
    dims = [None, None]; dims[sp_axis] = 2
    tt_spatial = ttnn.from_torch(spatial.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                 mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=dims))
    cdims = [None, None]; cdims[sp_axis] = 2
    mk = lambda t: ttnn.from_torch(t.permute(0, 2, 1, 3), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                   mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=cdims))
    tt_cos, tt_sin = mk(tc), mk(ts)
    tt_tm = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    m = WanAttention(dim=dim, num_heads=heads, qk_norm=True, eps=1e-6, mesh_device=mesh_device, ccl_manager=ccl,
                     parallel_config=pc, is_self=True)
    print("WHCHECK wan cfg", m.sdpa_program_config.q_chunk_size, m.sdpa_program_config.k_chunk_size,
          m.sdpa_program_config.exp_approx_mode, m.sdpa_compute_kernel_config)
    m.load_torch_state_dict(dict(tm.state_dict()))
    out = m(tt_spatial, N=seq, rope_cos=tt_cos, rope_sin=tt_sin, trans_mat=tt_tm)
    gd = [None, None]; gd[sp_axis] = 2; gd[tp_axis] = 3
    o = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=gd, mesh_shape=shape))
    o = o[0][:, :seq, :].float()
    tag = f"wan_self_{shape}".replace(" ", "")
    _save(tag, o)
    p = _pcc(o, ref)
    print("WHCHECK", tag, "pcc", p)
    assert p > 0.99
