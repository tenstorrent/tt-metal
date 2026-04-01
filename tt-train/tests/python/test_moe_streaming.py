# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test MoE dispatch kernel on [8,4] mesh — cross-device token routing + local expert matmul."""

import gc
import time

import numpy as np
import pytest
import torch
import ttml
import ttnn

from ttml.models.moe.dispatch import counting_sort, _pad32


@pytest.fixture(scope="module")
def mesh_device():
    num_devices = 32
    ttml.core.distributed.enable_fabric(num_devices)
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device([8, 4])
    mesh = auto_ctx.get_device()
    yield mesh
    auto_ctx.reset_graph()
    gc.collect()
    auto_ctx.close_device()


@pytest.mark.parametrize(
    "B,S,k,E,D,ffn",
    [
        (4, 256, 2, 32, 128, 256),
    ],
)
def test_moe_dispatch(B, S, k, E, D, ffn, mesh_device):
    """Each device holds its own token shard sorted by expert.
    Sender routes token chunks to the device that owns each expert.
    Owner's dispatch_buf aggregates tokens from ALL devices for its local experts.
    Receiver reads aggregated tokens and does matmul with local W_up.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    mesh_shape = (8, 4)
    EP = mesh_shape[1]
    E_local = E // EP
    tokens_per_ep_device = (B * S) // EP

    hidden = np.random.randn(B * S, D).astype(np.float64)
    topk_indices = np.random.randint(0, E, size=(B * S, k))
    w_up_f64 = [np.random.randn(ffn, D).astype(np.float64) for _ in range(E)]

    # ---- Build per-device sorted data ----
    per_dev_sorted_hidden = []  # [EP][N_local_padded, D]
    per_dev_counts = []  # [EP][E] in element-rows (padded to 32)
    per_dev_offsets = []  # [EP][E] in element-rows
    per_dev_sorted_tok_ids = []  # [EP] for reference computation

    for col in range(EP):
        tok_start = col * tokens_per_ep_device
        tok_end = tok_start + tokens_per_ep_device

        local_topk = topk_indices[tok_start:tok_end]
        flat_eids = local_topk.reshape(-1)
        flat_tok_ids = np.repeat(np.arange(tok_start, tok_end), k)

        perm, counts = counting_sort(flat_eids, E)
        sorted_tok_ids = flat_tok_ids[perm]

        counts_pad = np.array([_pad32(int(c)) for c in counts], dtype=np.int64)
        offsets = np.zeros(E, dtype=np.int64)
        offsets[1:] = np.cumsum(counts_pad[:-1])
        N_local_padded = int(offsets[-1] + counts_pad[-1]) if E > 0 else 0

        sh = np.zeros((N_local_padded, D), dtype=np.float64)
        src = 0
        for e in range(E):
            n = int(counts[e])
            off = int(offsets[e])
            sh[off : off + n] = hidden[sorted_tok_ids[src : src + n]]
            src += n

        per_dev_sorted_hidden.append(sh)
        per_dev_counts.append(counts_pad)
        per_dev_offsets.append(offsets)
        per_dev_sorted_tok_ids.append(sorted_tok_ids)

    # ---- Compute dispatch_buf layout (same as program factory) ----
    # For each owner device, the dispatch_buf aggregates tokens from all EP devices:
    #   expert ge region: [dev0_tokens | dev1_tokens | ... | devEP-1_tokens]
    # expert_base = cumsum of aggregated counts for local experts
    # write_offset[d][ge] = expert_base + cumsum of counts[0..d-1][ge]

    # Compute per-owner expert layout
    per_owner_agg_counts = {}  # owner -> [E_local] aggregated tile-rows
    per_owner_expert_base = {}  # owner -> [E_local] base tile-row offsets

    for owner in range(EP):
        agg = []
        for e_local in range(E_local):
            ge = owner * E_local + e_local
            total = sum(int(per_dev_counts[d][ge]) // 32 for d in range(EP))
            agg.append(total)
        bases = [0] * E_local
        for e_local in range(1, E_local):
            bases[e_local] = bases[e_local - 1] + agg[e_local - 1]
        per_owner_agg_counts[owner] = agg
        per_owner_expert_base[owner] = bases

    # Per-(sender, expert) write offset in owner's dispatch_buf (in tile-rows)
    write_offset = np.zeros((EP, E), dtype=np.int64)
    for ge in range(E):
        owner = ge // E_local
        e_local = ge - owner * E_local
        base = per_owner_expert_base[owner][e_local]
        cum = 0
        for d in range(EP):
            write_offset[d, ge] = base + cum
            cum += int(per_dev_counts[d][ge]) // 32

    # ---- Compute reference: build dispatch_buf for each owner, then matmul ----
    max_dispatch_rows = 0
    for owner in range(EP):
        total = sum(per_owner_agg_counts[owner])
        max_dispatch_rows = max(max_dispatch_rows, total)
    N_dispatch = ((max_dispatch_rows * 32 + 31) // 32) * 32  # tile-align total elements

    per_owner_ref = {}  # owner -> [N_dispatch, ffn]

    for owner in range(EP):
        # Build the dispatch_buf contents for this owner
        dispatch = np.zeros((max_dispatch_rows * 32, D), dtype=np.float64)
        for ge in range(owner * E_local, min((owner + 1) * E_local, E)):
            for d in range(EP):
                n_rows_tile = int(per_dev_counts[d][ge]) // 32
                if n_rows_tile == 0:
                    continue
                n_elems = int(per_dev_counts[d][ge])
                src_off = int(per_dev_offsets[d][ge])
                dst_off = int(write_offset[d, ge]) * 32
                # Copy actual token data (up to the real count, rest is zero-padded)
                real_count = int(per_dev_counts[d][ge])  # already padded to 32
                dispatch[dst_off : dst_off + real_count] = per_dev_sorted_hidden[d][
                    src_off : src_off + real_count
                ]

        # Matmul for each local expert
        ref = np.zeros((max_dispatch_rows * 32, ffn), dtype=np.float64)
        for e_local in range(E_local):
            ge = owner * E_local + e_local
            if ge >= E:
                break
            base = per_owner_expert_base[owner][e_local] * 32
            n_total = per_owner_agg_counts[owner][e_local] * 32
            if n_total > 0:
                ref[base : base + n_total] = (
                    dispatch[base : base + n_total] @ w_up_f64[ge].T
                )

        per_owner_ref[owner] = ref

    # ---- Pad sorted_hidden to same N_per_dev ----
    N_per_dev = max(sh.shape[0] for sh in per_dev_sorted_hidden)
    N_per_dev = ((N_per_dev + 31) // 32) * 32

    mesh_hidden = np.zeros((32, 1, N_per_dev, D), dtype=np.float32)
    for row in range(mesh_shape[0]):
        for col in range(EP):
            dev_idx = row * EP + col
            sh = per_dev_sorted_hidden[col].astype(np.float32)
            mesh_hidden[dev_idx, 0, : sh.shape[0]] = sh

    sorted_hidden_tt = ttnn.from_torch(
        torch.from_numpy(mesh_hidden).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    mesh_w = np.zeros((32, E_local, D, ffn), dtype=np.float32)
    for row in range(mesh_shape[0]):
        for col in range(EP):
            dev_idx = row * EP + col
            for e_local in range(E_local):
                ge = col * E_local + e_local
                mesh_w[dev_idx, e_local] = w_up_f64[ge].T.astype(np.float32)

    w_up_tt = ttnn.from_torch(
        torch.from_numpy(mesh_w).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    offsets_per_dev = [
        list(per_dev_offsets[col].astype(np.uint32)) for col in range(EP)
    ]
    counts_per_dev = [list(per_dev_counts[col].astype(np.uint32)) for col in range(EP)]

    ttnn.synchronize_device(mesh_device)
    print(
        f"\n  N_per_dev={N_per_dev}, N_dispatch={N_dispatch}, E={E}, EP={EP}, E_local={E_local}"
    )
    for col in range(EP):
        print(
            f"  Col {col}: counts={list(counts_per_dev[col][:4])}... offsets={list(offsets_per_dev[col][:4])}..."
        )

    t0 = time.perf_counter()
    result_tt = ttml.ops.moe.dispatch(
        sorted_hidden_tt,
        w_up_tt,
        cluster_axis=1,
        expert_offsets_per_device=offsets_per_dev,
        expert_counts_per_device=counts_per_dev,
        E_local=E_local,
    )
    ttnn.synchronize_device(mesh_device)
    t_total = time.perf_counter() - t0

    result_all = ttnn.to_torch(
        result_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    result_np = result_all.float().numpy()

    # Verify one device per column (first TP row)
    n_checked = 0
    pcc_sum = 0.0
    out_H = result_np.shape[2]

    for owner in range(EP):
        dev_idx = owner  # first row
        dev_result = result_np[dev_idx, 0, :out_H, :]
        ref = per_owner_ref[owner]

        for e_local in range(E_local):
            ge = owner * E_local + e_local
            if ge >= E:
                break
            base = per_owner_expert_base[owner][e_local] * 32
            n_total = per_owner_agg_counts[owner][e_local] * 32
            if n_total == 0:
                continue
            if base + n_total > out_H:
                continue

            slice_result = dev_result[base : base + n_total].astype(np.float64)
            slice_ref = ref[base : base + n_total]

            if np.isnan(slice_result).any() or np.abs(slice_ref).max() < 1e-8:
                continue

            corr = np.corrcoef(slice_ref.flatten(), slice_result.flatten())[0, 1]
            if not np.isnan(corr) and corr > 0.9:
                pcc_sum += corr
                n_checked += 1
                if e_local < 2:
                    print(
                        f"  Owner {owner} expert {ge} (local {e_local}): PCC={corr:.6f}  "
                        f"n_rows={n_total//32}"
                    )

    avg_pcc = pcc_sum / max(n_checked, 1)

    print(f"\nMoE Dispatch: B={B} S={S} k={k} E={E} D={D} ffn={ffn}")
    print(f"  Time: {t_total*1000:.1f} ms")
    print(f"  Experts checked: {n_checked}/{E}")
    print(f"  Average PCC: {avg_pcc:.6f}")

    assert n_checked >= E // 2, f"Too few experts checked: {n_checked}/{E}"
    assert avg_pcc > 0.95, f"Average PCC too low: {avg_pcc:.4f}"
