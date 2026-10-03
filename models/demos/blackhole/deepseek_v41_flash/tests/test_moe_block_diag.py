# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test: DSV41MoEBlock on a Blackhole Galaxy vs the torch golden on real layer weights/activations."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_weights_vs_reference import golden_moe
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 500_000,
            },
            id="fabric_1D_ring",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer_id", [2, 20])
@pytest.mark.timeout(1800)
@torch.no_grad()
def test_moe_block_diag(mesh_device, layer_id):
    torch.manual_seed(1234)
    rows, cols = tuple(mesh_device.shape)
    per_dev = 4
    batch = per_dev * rows

    # real activations: the FFN input of the checkpoint's own layer, for `batch` distinct tokens
    blk = R.build_layer(layer_id, max_batch_size=2, max_seq_len=256)
    tok = torch.randint(1000, 100000, (2, batch // 2))
    h, pm = R.embed_tokens(tok)
    cap = {}
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
    blk(h, 0, pm, None)
    x = cap["x"].reshape(batch, 1, 1, -1).to(torch.bfloat16)  # [batch, 1, 1, hidden]

    w = load_moe_layer(layer_id)
    golden, _ = golden_moe(x.reshape(batch, -1), w)  # [batch, hidden]

    from models.demos.blackhole.deepseek_v41_flash.reference.calibrate import calibrate_gate_cutoff

    K = calibrate_gate_cutoff(layer_id, seed=99)  # different tokens than the test batch (seed 1234)
    print("ROUTER calibrated cutoff K = %.3f" % K)
    moe = DSV41MoEBlock(mesh_device, w, topology=ttnn.Topology.Linear, batch_per_device=per_dev, gate_bias_shift=K)
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(rows, cols))
    tt_tok = ttnn.from_torch(
        x,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    shard_gate = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=(rows, cols))
    tt_gate_in = ttnn.from_torch(
        x.reshape(1, 1, batch, -1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard_gate,
    )

    # ---- router vs golden ----
    tt_s, tt_i = moe.gate.forward(tt_gate_in)
    devs_s, devs_i = ttnn.get_device_tensors(tt_s), ttnn.get_device_tensors(tt_i)
    dev_scores = torch.cat([ttnn.to_torch(devs_s[r * cols]).reshape(-1, 6) for r in range(rows)]).float()[:batch]
    dev_idx = torch.cat([ttnn.to_torch(devs_i[r * cols]).reshape(-1, 6) for r in range(rows)]).long()[:batch]
    xf = x.reshape(batch, -1).float()
    s = torch.nn.functional.softplus(xf @ w["gate_weight"]).sqrt()
    g_idx = (s + w["gate_bias"]).topk(6, dim=-1).indices
    g_w = s.gather(1, g_idx)
    g_w = g_w / (g_w.sum(-1, keepdim=True) + 1e-20) * 1.5
    same = [(set(dev_idx[t].tolist()) == set(g_idx[t].tolist())) for t in range(batch)]
    print("ROUTER same expert set per token:", sum(same), "/", batch)
    # margin between 6th and 7th ranked (score+bias) in fp32, and bf16-emulated selection agreement
    sb = s + w["gate_bias"]
    top7 = sb.topk(7, dim=-1).values
    margin = top7[:, 5] - top7[:, 6]
    print("ROUTER margin(6th-7th) per token:", [round(float(m), 4) for m in margin])
    print("ROUTER mismatch tokens margin:", [round(float(margin[t]), 4) for t in range(batch) if not same[t]])
    print(
        "ROUTER score range tok0 (fp32): min %.3f max %.3f; bias absmax %.3f"
        % (s[0].min(), s[0].max(), w["gate_bias"].abs().max())
    )
    s16 = (torch.nn.functional.softplus((xf @ w["gate_weight"]).bfloat16()).sqrt()).bfloat16()
    g16_idx = (s16 + w["gate_bias"].bfloat16()).topk(6, dim=-1).indices
    same16 = [(set(dev_idx[t].tolist()) == set(g16_idx[t].tolist())) for t in range(batch)]
    print("ROUTER same set vs bf16-emulated golden:", sum(same16), "/", batch)
    # compare weights by expert id
    errs = []
    for t in range(batch):
        dmap = {int(e): float(v) for e, v in zip(dev_idx[t], dev_scores[t])}
        for e, v in zip(g_idx[t].tolist(), g_w[t].tolist()):
            if e in dmap:
                errs.append(abs(dmap[e] - v) / max(abs(v), 1e-6))
    print(
        "ROUTER weight rel-err (shared experts) mean %.4f max %.4f" % (sum(errs) / max(len(errs), 1), max(errs or [0]))
    )
    print("ROUTER sample tok0 dev:", dev_idx[0].tolist(), [round(v, 3) for v in dev_scores[0].tolist()])
    print("ROUTER sample tok0 gold:", g_idx[0].tolist(), [round(v, 3) for v in g_w[0].tolist()])

    # ---- full block, split by component ----
    out = moe.forward(tt_gate_in, tt_tok)
    ttnn.synchronize_device(mesh_device)
    got = ttnn.to_torch(
        ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=(rows, cols)),
    )
    got = got.reshape(batch, -1).float()

    def run_golden(use_shared):
        o = torch.zeros(batch, 5120)
        for t in range(batch):
            xt = xf[t : t + 1]
            for j in range(6):
                e = int(g_idx[t, j])
                h = torch.nn.functional.silu(xt @ w["w0"][0, e].float()) * (xt @ w["w1"][0, e].float())
                o[t] += g_w[t, j] * (h @ w["w2"][0, e].float())[0]
            if use_shared:
                h = torch.nn.functional.silu(xt @ w["shared_w0"][384][0, 0].float()) * (
                    xt @ w["shared_w1"][384][0, 0].float()
                )
                o[t] += (h @ w["shared_w2"][384][0, 0].float())[0]
        return o

    routed, shared_only = run_golden(False), None
    full = run_golden(True)
    shared_only = full - routed
    print("PCC got vs full golden   : %.5f" % R.pcc(got, full))
    print("PCC got vs routed-only   : %.5f" % R.pcc(got, routed))
    print("PCC got vs shared-only   : %.5f" % R.pcc(got, shared_only))
    print("norms  got %.3f routed %.3f shared %.3f" % (got.norm(), routed.norm(), shared_only.norm()))
    print("per-token PCC:", [round(R.pcc(got[t], full[t]), 3) for t in range(batch)])
    # best scalar fit of got onto [routed, shared]
    A = torch.stack([routed.flatten().float(), shared_only.flatten().float()], dim=1)
    coef = torch.linalg.lstsq(A, got.flatten().unsqueeze(1).float()).solution.flatten()
    print("lstsq coefficients (routed, shared):", [round(float(c), 4) for c in coef])
