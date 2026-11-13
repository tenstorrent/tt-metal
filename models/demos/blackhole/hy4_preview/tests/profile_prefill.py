# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage traced latency of ONE prefill chunk (512 tokens at positions P..P+512, context built by earlier chunks) of a Hy4 attention layer on the 4x8 mesh. Env: LAYER, KVFMT, P0 (2560), GOLD.
Stages: front (q_a gather + norm, kv proj, lat norm, k_pe rope), indexer (select incl. causal fix), qpath, sdpa (all_to_all heads->seq, op, all_to_all back), finish. Prints RES lines."""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.profile_decode import tr
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/gold/golden")
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
L = int(os.environ.get("LAYER", "1"))
KVFMT = os.environ.get("KVFMT", "bf16")
P0 = int(os.environ.get("P0", "2560"))
N = 512


def main():
    import models.demos.blackhole.hy4_preview.tests.profile_decode as pd

    pd.L, pd.KVFMT, pd.V1, pd.P = L, KVFMT, False, P0
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=800_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        a = Hy4Attention(
            md,
            mesh_4x8(),
            ccl,
            cfg,
            load_sd(L),
            L,
            kv_fmt=KVFMT,
            rope_interleave=g["conv"]["rope_interleave"],
            idx_rope_interleave=g["conv"]["idx_rope_interleave"],
        )
        h = None
        for p0 in range(0, P0 + N, N):
            h = upload(md, g["h"][p0 : p0 + N].reshape(1, 1, N, -1).to(torch.bfloat16))
            ids_in = (
                upload(
                    md, g["idx"][p0 : p0 + N].to(torch.int32).reshape(1, 1, N, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                )
                if not a.has_indexer
                else None
            )
            _, ids = a.prefill_chunk(h, p0, ids_in)
        ttnn.synchronize_device(md)
        cos, sin, icos, isin = a.rope_tabs(P0, N)
        c = a.cfg
        qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
        qa = a._rms(a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3), a.q_a_norm)
        kvc = a.cache.rows

        def front():
            qa_c = ttnn.linear(h, a.w_q_a, compute_kernel_config=a.ckc)
            qn = a._rms(a.mesh_config.allgather(qa_c, a.ccl, axis=1, dim=3), a.q_a_norm)
            kv = ttnn.linear(h, a.w_kv_a, compute_kernel_config=a.ckc)
            lat = a._rms(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, N, 512]), a.kv_a_norm)
            kpe = a.rope.apply(ttnn.slice(kv, [0, 0, 0, 512], [1, 1, N, 576]), cos, sin, a.ckc)
            return qn, lat, kpe

        tr(md, "front", front)
        if a.has_indexer:
            tr(
                md,
                "indexer(select incl. causal fix)",
                lambda: a.indexer.select(h, qa, P0, icos, isin, None, append=False),
            )

        def qpath():
            q = ttnn.linear(qa, a.w_q_b, compute_kernel_config=a.ckc)
            qn, qp = ttnn.experimental.nlp_create_q_heads_split(
                q, num_heads=a.hc, split_head_dim=192, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            qabs = ttnn.matmul(qn, a.w_uk, compute_kernel_config=a.ckc)
            qp = a.rope.apply(qp, cos, sin, a.ckc)
            return ttnn.concat([qabs, qp], dim=3)

        tr(md, "qpath", qpath)
        qfull = qpath()
        tr(md, "sdpa(a2a heads->seq, op, a2a back)", lambda: a._sparse_prefill(qfull, kvc, ids))
        o_lat = a._sparse_prefill(qfull, kvc, ids)
        tr(md, "finish", lambda: a._finish(o_lat, h))
        tr(
            md,
            "ALL prefill_chunk (append-free)",
            lambda: a.prefill_chunk(h, P0, ids_in, append=False, tabs=(cos, sin, icos, isin)),
        )
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
