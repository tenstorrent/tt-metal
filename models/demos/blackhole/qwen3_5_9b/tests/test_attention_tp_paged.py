# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD: TPAttention paged-KV path (vLLM contract) must match the concat path.

B=1 on P150x4. Prefill a short prompt and take one decode step via BOTH:
  - the internal concat KV path (validated by test_attention_tp + the demo), and
  - the external paged KV path (forward_prefill_paged + paged forward_decode),
then assert the prefill and decode outputs match (PCC). The concat path is the
oracle, so this proves the paged code added in Phase 1a is correct.

Run:
  source python_env/bin/activate
  MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
    pytest -svq models/demos/blackhole/qwen3_5_9b/tests/test_attention_tp_paged.py
"""
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.tp_common import dequant_fp8_block


def _mp():
    return os.path.expanduser(os.environ.get("HF_MODEL", "Qwen/Qwen3.6-27B"))


def _load_attn_layer(model_path, layer_idx):
    from safetensors import safe_open

    model_path = Path(model_path)
    wm = json.load(open(model_path / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        base = next(k for k in wm if k.endswith(f"layers.{layer_idx}.self_attn.{name}.weight"))
        with safe_open(str(model_path / wm[base]), framework="pt") as sf:
            w = sf.get_tensor(base)
            sk = base + "_scale_inv"
            if wm.get(sk):
                with safe_open(str(model_path / wm[sk]), framework="pt") as sf2:
                    w = dequant_fp8_block(w, sf2.get_tensor(sk))
            else:
                w = w.to(torch.bfloat16)
        out[f"{name}.weight"] = w
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"P150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attention_tp_paged(mesh_device, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", _mp())
    args = Qwen35ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NKV, HD = args.n_local_kv_heads, args.head_dim
    block_size, S, num_blocks = 64, 64, 4
    logger.info(f"devices={nd} layer={li} NKV_local={NKV} HD={HD} S={S} num_blocks={num_blocks}")

    # args.CKPT_DIR is the resolved local snapshot dir (Qwen35ModelArgs downloads the hub id).
    sd = _load_attn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_attention_weights_tp(mesh_device, sd, args)

    def to_dev(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def rm_pt(rows):
        return ttnn.from_torch(
            torch.tensor(rows, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )

    def mk_cache():
        return ttnn.from_torch(
            torch.zeros(num_blocks, NKV, block_size, HD, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    xp = torch.randn(1, 1, S, args.dim, dtype=torch.bfloat16)
    xd = torch.randn(1, 1, 1, args.dim, dtype=torch.bfloat16)
    cos_p, sin_p = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)
    cos_d, sin_d = rot_mats_decode(
        mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, torch.tensor([S], dtype=torch.int32)
    )
    cur_tt = ttnn.from_torch(
        torch.tensor([S], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)

    # ---- concat reference (the oracle) ----
    a_ref = TPAttention(mesh_device, args, tw, tt_ccl)
    a_ref.reset_state()
    pre_ref = ttnn.to_torch(a_ref.forward_prefill(to_dev(xp), cos_p, sin_p), mesh_composer=comp).float()
    dec_ref = ttnn.to_torch(a_ref.forward_decode(to_dev(xd), cur_tt, cos_d, sin_d), mesh_composer=comp).float()

    # ---- paged path ----
    a_pag = TPAttention(mesh_device, args, tw, tt_ccl)
    a_pag.set_paged_kv_cache(mk_cache(), mk_cache())
    pre_pag = ttnn.to_torch(
        a_pag.forward_prefill_paged(
            to_dev(xp), cos_p, sin_p, rm_pt([[0]]), chunk_page_table=rm_pt([[0]]), chunk_start_idx=0
        ),
        mesh_composer=comp,
    ).float()
    dec_pag = ttnn.to_torch(
        a_pag.forward_decode(to_dev(xd), cur_tt, cos_d, sin_d, page_table=rm_pt([list(range(num_blocks))])),
        mesh_composer=comp,
    ).float()

    ok_p, pcc_p = comp_pcc(pre_ref, pre_pag, 0.97)
    ok_d, pcc_d = comp_pcc(dec_ref, dec_pag, 0.97)
    logger.info(f"PREFILL paged-vs-concat PCC = {pcc_p}")
    logger.info(f"DECODE  paged-vs-concat PCC = {pcc_d}")
    assert ok_p, f"prefill paged PCC too low: {pcc_p}"
    assert ok_d, f"decode paged PCC too low: {pcc_d}"
    logger.info("PASSED: TPAttention paged path matches concat path (B=1)")
