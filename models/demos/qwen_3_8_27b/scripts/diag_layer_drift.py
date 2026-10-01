# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DIAGNOSTIC (reduced sequence length — never a graded result): per-layer residual-stream drift of the
device model vs the fp32 streamed reference on the real checkpoint, over the first N trace tokens.

    python models/demos/qwen_3_8_27b/scripts/diag_layer_drift.py [N]   # N default 2048

Prints, per layer: hidden PCC, relative max error, and the top outlier channels' magnitude, so an
accuracy loss can be pinned to the layer (and block type) where it enters.
"""

import os
import sys
from pathlib import Path

import torch

from models.demos.qwen_3_8_27b.config import PrefillSpec, Qwen38Config
from models.demos.qwen_3_8_27b.reference import golden
from models.demos.qwen_3_8_27b.reference.checkpoint import CheckpointReader, checkpoint_dir
from models.demos.qwen_3_8_27b.tests.common import pcc
from models.demos.qwen_3_8_27b.tt.common import weight_cache_dir
from models.demos.qwen_3_8_27b.tt.context import PrefillCtx
from models.demos.qwen_3_8_27b.tt.kv_cache import cache_capacity
from models.demos.qwen_3_8_27b.tt.mesh import MeshConfig, close_mesh, make_ccl_manager, open_mesh
from models.demos.qwen_3_8_27b.tt.model import CheckpointWeights, TtQwen38Model, gather_hidden


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    spec = PrefillSpec.load().validate()
    cfg = Qwen38Config.from_hf_json(checkpoint_dir() / "config.json")
    ids = golden.trace_token_ids(os.environ["PREFILL_TRACE_DIR"])[:N]
    ref_cache = Path(os.environ.get("QWEN38_DIAG_CACHE", "/tmp/qwen38_diag")) / f"ref_hidden_{N}.pt"
    if ref_cache.exists():
        ref_h = torch.load(ref_cache)
    else:
        ref_h = []
        golden.stream_forward(
            cfg,
            ids[None],
            reader=CheckpointReader(),
            dtype=torch.float32,
            on_layer=lambda i, x: ref_h.append(x.clone()),
        )
        ref_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ref_h, ref_cache)

    mesh = open_mesh(spec.mesh_shape)
    try:
        mc = MeshConfig(mesh, spec.sp, spec.tp)
        model = TtQwen38Model(
            mc, make_ccl_manager(mesh), cfg, spec, CheckpointWeights(CheckpointReader()), cache=weight_cache_dir(mesh)
        )
        caches = model.allocate_caches(cache_capacity(spec.max_seq_len, [spec.chunk_size, N]))
        dev_h = {}
        model.forward(
            model.embedding.make_tokens(ids),
            PrefillCtx(caches, 0, 0, N, N),
            on_layer=lambda i, x: dev_h.__setitem__(i, gather_hidden(x, mc)),
        )
        if os.environ.get("QWEN38_DIAG_ISOLATED") == "1":
            # each device layer fed the REFERENCE input: its own error, free of upstream drift
            from models.demos.qwen_3_8_27b.tests.common import to_sp

            print(f"layer  type               isolated-PCC  (N={N}, REDUCED, reference input per layer)")
            emb = CheckpointReader().text("embed_tokens.weight")[ids].float()[None]
            for i in range(cfg.num_hidden_layers):
                caches.reset_gdn()
                ctx = PrefillCtx(caches, 0, 0, N, N)
                ctx.cos, ctx.sin = model.rope.tables(0, N // mc.sp)
                x_in = emb if i == 0 else ref_h[i - 1]
                y = model.layers[i](to_sp(x_in[None].to(torch.bfloat16), mc), ctx, model.rope)
                print(f"{i:4d}  {cfg.layer_types[i]:17s}  {pcc(gather_hidden(y, mc)[0], ref_h[i][0].float()):.6f}")
            return
        print(f"layer  type               PCC       max|ref|   rel_maxerr  top-channel |ref|  (N={N}, REDUCED)")
        for i in range(cfg.num_hidden_layers):
            r, d = ref_h[i][0].float(), dev_h[i][0]
            err = (r - d).abs()
            ch = r.abs().amax(0).topk(3)
            print(
                f"{i:4d}  {cfg.layer_types[i]:17s}  {pcc(d, r):.6f}  {r.abs().max():9.1f}  {(err.max() / r.abs().max()):.4f}"
                f"      {ch.values.tolist()} @ {ch.indices.tolist()}"
            )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
