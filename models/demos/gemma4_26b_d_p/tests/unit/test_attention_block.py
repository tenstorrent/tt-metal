# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 attention block (real weights) in chunked SP prefill vs the torch reference.

Layer 0 (sliding, 8x256 KV) and layer 5 (full, 2x512 KV, K=V). The TT side processes ``n_chunks``
chunks through the block-cyclic cache; the reference runs one-shot (chunk-invariant in fp32).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.gemma4_26b_d_p.bringup.registry import record_result
from models.demos.gemma4_26b_d_p.reference.blocks import Attention, RMSNorm, attention_mask, rope_cos_sin
from models.demos.gemma4_26b_d_p.reference.config import Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader
from models.demos.gemma4_26b_d_p.tests.mesh import MESH_PARAMS, mesh_id, sp_tp
from models.demos.gemma4_26b_d_p.tt.attention.attention import TtAttention
from models.demos.gemma4_26b_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.gemma4_26b_d_p.tt.ccl import CCLManager
from models.demos.gemma4_26b_d_p.tt.rope import build_indexed_rope, build_transformation_mat


def bc_index(kv_actual, sp, C):
    pos = rotated_chip_positions(kv_actual, sp, C)
    return torch.tensor([pos[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)


@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["L0-sliding", "L5-full"])
@pytest.mark.parametrize("n_chunks,chunk", [(2, 4096)], ids=["2x4k"])
def test_attention_block(mesh_device, device_params, layer_idx, n_chunks, chunk):
    cfg = Gemma4TextConfig.from_json()
    r = CheckpointReader()
    sd = r.substate(f"layers.{layer_idx}.self_attn")
    sp, tp, sp_axis, _ = sp_tp(mesh_device)
    C = chunk // sp
    S = n_chunks * chunk
    lt = cfg.layer_types[layer_idx]
    spec = cfg.rope_spec(lt)
    window = cfg.sliding_window if cfg.is_sliding(layer_idx) else None

    # Reference (fp32) on realistic inputs: input_layernorm(randn).
    torch.manual_seed(0)
    norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    norm.weight.data = r.get(f"layers.{layer_idx}.input_layernorm.weight").float()
    x = norm(torch.randn(1, S, cfg.hidden_size)).bfloat16().float()
    ref = Attention(cfg, layer_idx).float()
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    pos = torch.arange(S)
    cos, sin = rope_cos_sin(pos, spec.theta, spec.head_dim, spec.rotated_pairs)
    n_kv, hd = cfg.layer_kv_heads(layer_idx), cfg.layer_head_dim(layer_idx)
    empty = torch.zeros(1, n_kv, 0, hd)
    with torch.no_grad():
        ref_out, _, _ = ref(x, cos, sin, empty, empty, attention_mask(pos, pos, window))

    sp_topo, tp_topo = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=1, topology=sp_topo)
    tt_attn = TtAttention(mesh_device, cfg, layer_idx, sd, ccl, tp_topo)
    kv = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=S, sp_axis=sp_axis, n_kv_local=tt_attn.nkv_l, head_dim=hd)
    rope = build_indexed_rope(mesh_device, spec, max_seq_len=S, chunk_size=chunk, sp_axis=sp_axis)
    trans = build_transformation_mat(mesh_device)
    x_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None))

    got = torch.zeros_like(ref_out)
    for c in range(n_chunks):
        kv_actual = c * chunk
        idx = bc_index(kv_actual, sp, C)
        tt_x = ttnn.from_torch(x[:, idx][None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=x_map)
        out = tt_attn(tt_x, rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
        dts = ttnn.get_device_tensors(out)
        rows = [ttnn.to_torch(dts[s * tp]).float()[0, 0] for s in range(sp)]  # TP col 0 of every SP row
        got[0, idx] = torch.cat(rows, 0)
        out.deallocate(True)

    ok, pcc = comp_pcc(ref_out, got, 0.99)
    per_chunk = [comp_pcc(ref_out[:, i * chunk : (i + 1) * chunk], got[:, i * chunk : (i + 1) * chunk])[1] for i in range(n_chunks)]
    logger.info(f"attention L{layer_idx} ({lt}) mesh={mesh_id(mesh_device)}: PCC {pcc} per-chunk {per_chunk}")
    record_result("attention_layer", mesh_id(mesh_device), float(pcc), bool(ok), f"L{layer_idx} {n_chunks}x{chunk}", layer_variant="sliding" if window else "full")
    assert ok, pcc
