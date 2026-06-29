# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Traced TP2(VLM) -> cross-mesh KV socket -> PP2(denoise) pipeline, random weights/input.

End-to-end proof on 4 chips (2x2 parent from the mesh_device fixture, FABRIC_1D): a tensor-parallel
VLM layer (replicated QKV -> per-layer KV; gate/up column-parallel + down row-parallel + all-reduce
CCL) is captured on the VLM-mesh trace with its KV socket-send as the trace tail; each PP (1x1)
denoise stage's recv + real TTNNPi05DenoiseExpertBlock is captured on its trace (PP0->PP1 hidden hop
socketed). Correctness is eager-vs-traced PCC parity (random weights => "traced == eager").

Run:  pytest models/experimental/pi0_5/tt/tt_pipeline/test_xmesh_traced_pipeline.py -s
"""
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, "models/experimental/pi0_5/tests/perf")
from test_perf_pi05_pipeline_stage_isolated import _build_isolated_stage, _expert_block_w, _suffix_w
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock
from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
from models.experimental.pi0_5.tt.tt_pipeline import euler_schedule, perf_suffix_len
from models.experimental.pi0_5.tt.tt_pipeline import denoise_block as _db
from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import _bind_prefix_kv

W_VLM, MLP_VLM, HD, NKV, PREFIX, PAGE = 2048, 4096, 256, 1, 1024, 4096
_L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

pytestmark = pytest.mark.skipif(ttnn.get_num_devices() < 4, reason="cross-mesh TP2->PP2 needs 4 chips")


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 134_217_728}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_xmesh_traced_pipeline(mesh_device):
    _db.DECODE_ALL = True
    # ---- PP denoise reference (2 layers) + TP VLM random weights (host) ----
    cfg = Pi0_5ModelConfig(action_horizon=10, num_denoising_steps=5)
    ec = cfg.expert_config
    W, hd, suf = ec.width, ec.head_dim, perf_suffix_len(10)
    sc = SuffixConfig(action_dim=32, action_horizon=10, expert_width=W, pi05=True)
    torch.manual_seed(42)
    rb = [
        AdaRMSGemmaBlock(ec, _expert_block_w(W, ec.mlp_dim, hd, ec.num_heads, ec.num_kv_heads), 4 + i) for i in range(2)
    ]
    rs = Pi0_5SuffixEmbedding(sc, _suffix_w(W, 32))
    fw, fb = torch.randn(3 * W, W) * 0.02, torch.randn(3 * W) * 0.02
    ts, _ = euler_schedule(5)
    cond = rs.embed_timestep_adarms(torch.tensor([ts[0]]))
    mask = torch.zeros(1, 1, suf, PREFIX + suf)
    zkv = [(torch.zeros(1, NKV, PREFIX, hd), torch.zeros(1, NKV, PREFIX, hd)) for _ in range(2)]  # build-only skeleton
    inp_t = torch.randn(1, suf, W) * 0.5
    torch.manual_seed(11)
    Wk = [torch.randn(W_VLM, NKV * HD) * 0.02 for _ in range(2)]
    Wv = [torch.randn(W_VLM, NKV * HD) * 0.02 for _ in range(2)]
    Wg, Wu, Wd = (
        torch.randn(W_VLM, MLP_VLM) * 0.02,
        torch.randn(W_VLM, MLP_VLM) * 0.02,
        torch.randn(MLP_VLM, W_VLM) * 0.02,
    )
    x_vlm = torch.randn(1, PREFIX, W_VLM) * 0.1

    parent = mesh_device
    subs = []
    try:
        vlm = parent.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(0, 0))
        pp = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, c)) for c in (0, 1)]
        subs += [vlm, *pp]

        # TP weights: QKV replicated; gate/up column-parallel; down row-parallel.
        REP, COL, ROW = (
            ttnn.ReplicateTensorToMesh(vlm),
            ttnn.ShardTensorToMesh(vlm, dim=-1),
            ttnn.ShardTensorToMesh(vlm, dim=0),
        )
        f16 = lambda t, mm: ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=vlm, memory_config=_DRAM, mesh_mapper=mm
        )
        wk, wv = [f16(w, REP) for w in Wk], [f16(w, REP) for w in Wv]
        wg, wu, wd, x_v = f16(Wg, COL), f16(Wu, COL), f16(Wd, ROW), f16(x_vlm, REP)

        def tp_vlm():
            """Replicated QKV -> per-PP-layer (K,V); tensor-parallel MLP (all-reduce) to exercise CCL in-trace."""
            kvs = []
            for j in range(2):
                k = ttnn.permute(
                    ttnn.reshape(ttnn.linear(x_v, wk[j], memory_config=_DRAM), (1, PREFIX, NKV, HD)), (0, 2, 1, 3)
                )
                v = ttnn.permute(
                    ttnn.reshape(ttnn.linear(x_v, wv[j], memory_config=_DRAM), (1, PREFIX, NKV, HD)), (0, 2, 1, 3)
                )
                kvs.append((ttnn.typecast(k, ttnn.bfloat8_b), ttnn.typecast(v, ttnn.bfloat8_b)))
            h = ttnn.multiply(
                ttnn.silu(ttnn.linear(x_v, wg, memory_config=_DRAM)),
                ttnn.linear(x_v, wu, memory_config=_DRAM),
                memory_config=_DRAM,
            )
            ttnn.all_reduce(ttnn.linear(h, wd, memory_config=_DRAM), memory_config=_DRAM)
            return kvs

        # KV sockets: (K,V) per PP stage, from the same-column (collinear) VLM chip.
        mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
        sk, rv = [], []
        for c in (0, 1):
            sp, rb_ = [], []
            for w in (0, 1):
                conn = ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, c), ttnn.CoreCoord(0, 0)),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1 + w)),
                )
                sp.append(ttnn.create_socket_pair(vlm, pp[c], ttnn.SocketConfig([conn], mem)))
                rb_.append(
                    ttnn.from_torch(
                        torch.zeros(1, NKV, PREFIX, hd),
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=pp[c],
                        memory_config=_DRAM,
                    )
                )
            sk.append(sp)
            rv.append(rb_)

        # PP denoise blocks (real), prefix-KV bound to the socket recv buffers.
        stg = [
            _build_isolated_stage(
                pp[i],
                rb[i : i + 1],
                rs,
                fw,
                fb,
                cond,
                zkv[i : i + 1],
                mask,
                cfg,
                sc,
                is_first=False,
                is_last=False,
                position_offset=PREFIX,
                prefix_len=PREFIX,
                suffix_len=suf,
            )
            for i in range(2)
        ]
        for i in range(2):
            stg[i]._prefix_kv = [
                (
                    _bind_prefix_kv(rv[i][0], pp[i], ttnn.bfloat8_b, _L1),
                    _bind_prefix_kv(rv[i][1], pp[i], ttnn.bfloat8_b, _L1),
                )
            ]

        # PP0 -> PP1 hidden hop (collinear, row 1).
        hc = ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 3)),
        )
        hs, hr = ttnn.create_socket_pair(pp[0], pp[1], ttnn.SocketConfig([hc], mem))
        hbuf = ttnn.from_torch(
            torch.zeros(1, suf, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pp[1], memory_config=_L1
        )
        inp0 = ttnn.from_torch(inp_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pp[0], memory_config=_L1)

        def send_kv(kvs):
            for c in (0, 1):
                for w in (0, 1):
                    ttnn.experimental.send_direct_async(kvs[c][w], sk[c][w][0])

        def recv_kv(c):
            for w in (0, 1):
                ttnn.experimental.recv_direct_async(rv[c][w], sk[c][w][1])

        # ---- eager ----
        send_kv(tp_vlm())
        for c in (0, 1):
            recv_kv(c)
        ttnn.synchronize_device(pp[1])
        ttnn.experimental.send_direct_async(ttnn.to_memory_config(stg[0].forward(inp0), _L1), hs)
        ttnn.experimental.recv_direct_async(hbuf, hr)
        ttnn.synchronize_device(pp[1])
        out_eager = ttnn.to_torch(stg[1].forward(hbuf))

        # ---- trace: VLM(compute+CCL+KV send) ; PP0(recv KV + denoise + hop send) ; PP1(recv KV + recv hop + denoise) ----
        tv = ttnn.begin_trace_capture(vlm, cq_id=0)
        send_kv(tp_vlm())
        ttnn.end_trace_capture(vlm, tv, cq_id=0)
        t0 = ttnn.begin_trace_capture(pp[0], cq_id=0)
        recv_kv(0)
        ttnn.experimental.send_direct_async(ttnn.to_memory_config(stg[0].forward(inp0), _L1), hs)
        ttnn.end_trace_capture(pp[0], t0, cq_id=0)
        t1 = ttnn.begin_trace_capture(pp[1], cq_id=0)
        recv_kv(1)
        ttnn.experimental.recv_direct_async(hbuf, hr)
        out_t = stg[1].forward(hbuf)
        ttnn.end_trace_capture(pp[1], t1, cq_id=0)

        for r in range(4):
            ttnn.execute_trace(vlm, tv, cq_id=0, blocking=False)
            ttnn.execute_trace(pp[0], t0, cq_id=0, blocking=False)
            ttnn.execute_trace(pp[1], t1, cq_id=0, blocking=False)
            ttnn.synchronize_device(pp[1])
            assert _pcc(out_eager, ttnn.to_torch(out_t)) > 0.99, f"replay {r}: traced != eager"

        ttnn.release_trace(vlm, tv)
        ttnn.release_trace(pp[0], t0)
        ttnn.release_trace(pp[1], t1)
    finally:
        for sm in reversed(subs):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
