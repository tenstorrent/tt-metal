# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: is the cached column-sharded out_proj weight (tensor cache 'out_col') identical to a fresh shard?"""
import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import load_gdn_layer, model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
def test_outcol_cache(mesh_device, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    os.environ["QWEN36_GDN_OUT_MODE"] = "agmm"
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    cache_dir = args.weight_cache_path() / f"layers.{li}" / "tp"
    logger.info(
        f"cache dir {cache_dir} exists={cache_dir.exists()} out_col files: {[p.name for p in cache_dir.glob('out_col*')] if cache_dir.exists() else None}"
    )
    tw_cached = load_gdn_weights_tp(mesh_device, sd, args, cache_dir=cache_dir)
    tw_fresh = load_gdn_weights_tp(mesh_device, sd, args, cache_dir=None)
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    for name in ("out_col", "out", "qkvz"):
        a = ttnn.to_torch(tw_cached[name], mesh_composer=comp).float()
        b = ttnn.to_torch(tw_fresh[name], mesh_composer=comp).float()
        _, p = comp_pcc(a, b, 0.999)
        logger.info(
            f"OUTCOL_CACHE {name}: cached vs fresh PCC={p} max|d|={float((a-b).abs().max()):.4f} shape={tuple(a.shape)}"
        )
    # and the torch ground truth for out_col: W^T = [in=6144, out=5120]
    wt = sd["linear_attn.out_proj.weight"].float().T
    a = ttnn.to_torch(tw_cached["out_col"], mesh_composer=comp).float()
    _, p = comp_pcc(a, wt, 0.99)
    logger.info(f"OUTCOL_CACHE out_col cached vs torch W^T PCC={p} shapes {tuple(a.shape)} {tuple(wt.shape)}")
    # per-device shard order check: device d should hold columns [d*1280,(d+1)*1280)
    per_dev = ttnn.get_device_tensors(tw_cached["out_col"])
    for d, t in enumerate(per_dev):
        td = ttnn.to_torch(t).float()
        _, p = comp_pcc(td, wt[:, d * 1280 : (d + 1) * 1280], 0.99)
        logger.info(f"OUTCOL_CACHE device {d} shard vs torch cols PCC={p}")
