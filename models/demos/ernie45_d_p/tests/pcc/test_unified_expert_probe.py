# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P3.1: is the fused unified_routed_expert_moe kernel worth integrating for ERNIE on 1x4?

The DeepSeek EP dispatch/combine ops cannot run with a 1-chip dispatch group (dispatch needs fabric
neighbours on the dispatch axis; only cluster_axis=0 is supported), so this probe bypasses them: the
per-chip dispatched buffer (tokens grouped by local expert, 32-aligned regions, device-resident counts /
offsets) is built on the host from REAL golden routing (layer L of the 55k golden's last 5120-token chunk).
It then checks the kernel's per-expert outputs vs torch and times it against the current dense-EP routed
experts (tt/moe.py TtMoE.routed_partial) on the same tokens.
"""

import time

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.reference.ernie_ref import pcc
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import CACHE_ROOT, COMPUTE_HIFI2, Golden, to_mesh_activation
from models.demos.ernie45_d_p.tt.moe import TtMoE

TASK = "P3.1"
N = 4


def _sync_time(fn, mesh, reps=3, setup=None):
    """Best-of-reps wall time. `setup()` (untimed) builds fresh inputs per rep: unified_routed_expert_moe
    writes its output IN PLACE into a TILE dispatched buffer, so re-running on the same buffer is invalid."""
    out, best = None, 1e9
    for _ in range(reps):  # first rep compiles
        args = setup() if setup else ()
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        out = fn(*args)
        ttnn.synchronize_device(mesh)
        best = min(best, time.time() - t0)
    return out, best


def build_dispatch(x, idx, E, epc):
    """Per chip c: buffer of tokens routed to experts 16c..16c+15, each region 32-aligned.
    Returns stacked [N, 1, rows, H] buffers, counts/offsets [N, epc] and per-chip (expert, rows, token ids)."""
    S, H = x.shape
    per_chip, rows_max = [], 0
    for c in range(N):
        regions, off = [], 0
        for j in range(epc):
            e = c * epc + j
            tok = torch.nonzero((idx == e).any(-1)).flatten()
            regions.append((e, off, tok))
            off += ((len(tok) + 31) // 32) * 32
        per_chip.append(regions)
        rows_max = max(rows_max, off)
    bufs = torch.zeros(N, 1, rows_max, H)
    # Indexed by GLOBAL expert id (as offset_cumsum produces them); the kernel maps local->global itself.
    counts = torch.zeros(N, E, dtype=torch.int32)
    offsets = torch.zeros(N, E, dtype=torch.int32)
    for c, regions in enumerate(per_chip):
        for j, (e, off, tok) in enumerate(regions):
            bufs[c, 0, off : off + len(tok)] = x[tok]
            counts[c, e], offsets[c, e] = len(tok), off
    return bufs, counts, offsets, per_chip, rows_max


@mesh_1x4
@pytest.mark.parametrize("layer", [1, 14])
def test_unified_expert_probe(mesh_device, cfg, layer_weights, record, layer):
    G = Golden(56320, 5120)
    last = G.seq // G.chunk - 1
    g = G.layer(last, layer)
    x, idx = g["ffn_norm"].float(), g["topk_idx"].long()  # [S, H], [S, 6]
    S, H, E, I = x.shape[0], cfg.hidden_size, cfg.moe_num_experts, cfg.moe_intermediate_size
    epc = E // N
    w = layer_weights(layer)
    bufs, counts, offsets, per_chip, rows = build_dispatch(x, idx, E, epc)
    tok_counts = [int(counts[c].sum()) for c in range(N)]
    assert sum(tok_counts) == S * cfg.moe_k
    print(
        f"L{layer}: S={S}, routed rows per chip {tok_counts} (dense-EP computes {S * epc} per chip), buffer rows {rows}"
    )

    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    tt_buf = ttnn.from_torch(bufs, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=shard)
    tt_buf = ttnn.squeeze(ttnn.squeeze(tt_buf, 0), 0)  # per chip [rows, H]
    u32 = lambda t: ttnn.squeeze(  # noqa: E731
        ttnn.from_torch(
            t[:, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=shard
        ),
        0,
    )
    tt_counts, tt_offsets = u32(counts), u32(offsets)  # per chip [1, E]
    gidx = ttnn.from_torch(
        ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=epc, dispatch_group_size=1, num_dispatch_groups=N
        ),
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)
    routed = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=epc,
        global_expert_idx_table=gidx,
        emb_dim=H,
        hidden_dim=I,
        max_tokens=S,
        torch_weights=[{"gate_proj": w.e_gate[e], "up_proj": w.e_up[e], "down_proj": w.e_down[e]} for e in range(E)],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        compute_kernel_config=COMPUTE_HIFI2,
        weight_cache_path=CACHE_ROOT / "unified_moe",
        cache_name_prefix=f"layer_{layer}.routed_expert.bf16",
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    pre_x = ttnn.to_torch(ttnn.get_device_tensors(tt_buf)[0]).float().reshape(-1, H)
    pre_counts = ttnn.to_torch(ttnn.get_device_tensors(tt_counts)[0]).flatten()
    pre_off = ttnn.to_torch(ttnn.get_device_tensors(tt_offsets)[0]).flatten()
    print(
        f"PRE chip0: buffer pcc vs host {pcc(pre_x, bufs[0, 0]):.5f}; counts[0:4] dev {pre_counts[:4].tolist()} host "
        f"{counts[0, :4].tolist()}; offsets[0:4] dev {pre_off[:4].tolist()} host {offsets[0, :4].tolist()}; "
        f"shapes buf {list(tt_buf.shape)} counts {list(tt_counts.shape)} offs {list(tt_offsets.shape)} gidx {list(gidx.shape)} "
        f"gidx chip0 {ttnn.to_torch(ttnn.get_device_tensors(gidx)[0]).flatten()[:4].tolist()}"
    )
    out, t_unified = _sync_time(
        lambda b: routed(b, tt_counts, tt_offsets), mesh_device, setup=lambda: (ttnn.clone(tt_buf),)
    )
    outs = [ttnn.to_torch(t).float().reshape(-1, H) for t in ttnn.get_device_tensors(out)]

    # Diagnostics on chip 0's first non-empty region: magnitude, and which expert's FFN the output matches.
    c0 = next((e, off, tok) for e, off, tok in per_chip[0] if len(tok))
    e0, off0, tok0 = c0
    got0 = outs[0][off0 : off0 + len(tok0)]
    ffn = lambda e, xs: F.linear(
        F.silu(F.linear(xs, w.e_gate[e].float())) * F.linear(xs, w.e_up[e].float()), w.e_down[e].float()
    )  # noqa: E731
    ref0 = ffn(e0, x[tok0])
    best = max(range(E), key=lambda e: pcc(got0, ffn(e, x[tok0])))
    xin = pre_x[off0 : off0 + len(tok0)]
    print(
        f"DIAG chip0 expert {e0} rows {len(tok0)} @ {off0}: |out| {got0.norm():.3e} |ref| {ref0.norm():.3e} "
        f"pcc {pcc(got0, ref0):.4f}; best-matching expert {best} pcc {pcc(got0, ffn(best, x[tok0])):.4f}; "
        f"input rows pcc {pcc(xin, x[tok0]):.5f}; out rows beyond buffer used: |tail| {outs[0][rows - 32:].norm():.3e}"
    )
    worst = 1.0
    for c, regions in enumerate(per_chip):
        for e, off, tok in regions:
            if len(tok) == 0:
                continue
            ref = F.linear(
                F.silu(F.linear(x[tok], w.e_gate[e].float())) * F.linear(x[tok], w.e_up[e].float()), w.e_down[e].float()
            )
            worst = min(worst, pcc(outs[c][off : off + len(tok)], ref))
    metrics.record(record.task, f"pcc_unified_expert_min_L{layer:02d}", worst)

    # Baseline: current dense-EP routed experts on the same tokens (routing from the same golden idx/weights).
    dense = TtMoE(mesh_device, cfg, layer, w)
    R = torch.zeros(S, E).scatter(-1, idx, g["topk_w"].float())
    tt_R = to_mesh_activation(mesh_device, R)
    tt_x = to_mesh_activation(mesh_device, x)
    _, t_dense = _sync_time(lambda: dense.routed_partial(tt_x, tt_R), mesh_device)

    speed = t_dense / t_unified
    print(
        f"L{layer}: unified experts {t_unified * 1e3:.1f} ms vs dense-EP {t_dense * 1e3:.1f} ms -> {speed:.2f}x; worst expert PCC {worst:.5f}"
    )
    metrics.record(record.task, f"unified_expert_ms_L{layer:02d}", round(t_unified * 1e3, 2))
    metrics.record(record.task, f"dense_ep_expert_ms_L{layer:02d}", round(t_dense * 1e3, 2))
    metrics.record(record.task, f"speedup_L{layer:02d}", round(speed, 2))
    assert worst >= 0.99
