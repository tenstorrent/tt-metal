"""What does paged_scaled_dot_product_attention_decode(sliding_window_size=W) actually attend?

Random paged K/V (NKV=2 per device, HD=128) for positions 0..C+B-1 at C=4096, random q for B=8 rows.
Runs the kernel with per-row cur_pos and compares against torch references with candidate window
definitions on ONE device replica:
   none : keys 0..cur                       (no window)
   A    : keys with cur - k <  W            (W keys: cur-W+1 .. cur)  == reference _attention_mask (causal)
   B    : keys with cur - k <= W            (W+1 keys)
Also the bidirectional-block use (all rows cur = C+B-1) with A/B.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/dflash2_window_kernel.py -v -s
"""
import pytest
import torch

import ttnn

NH, NKV, HD, BS = 8, 2, 128, 64
W = 2048


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / (a.std() * b.std() * (a.numel() - 1)))


def _ref(q, k, v, cur, lo):
    """q [B,NH,HD], k/v [NKV,S,HD]; row i attends keys lo[i]..cur[i] inclusive."""
    B = q.shape[0]
    out = torch.zeros(B, NH, HD)
    for i in range(B):
        for h in range(NH):
            kh = h // (NH // NKV)
            l = max(0, lo[i])
            ks = k[kh, l : cur[i] + 1].float()
            vs = v[kh, l : cur[i] + 1].float()
            s = (q[i, h].float() @ ks.T) * HD**-0.5
            out[i, h] = torch.softmax(s, -1) @ vs
    return out


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "num_command_queues": 2,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 1024 * 1024 * 1024,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("C", [int(x) for x in __import__("os").environ.get("WINDOW_C", "4096").split(",")])
def test_window_semantics(mesh_device, C):
    md = mesh_device
    torch.manual_seed(0)
    B = 8
    S = C + B
    # The kernel scans whole (up to 256-position) chunks up to nearest_n(cur+1, 256): the page table
    # must cover that span or the reader walks past it (garbage keys past cur, masked but possibly NaN).
    nb = ((S + 255) // 256) * 256 // BS + 8
    rep = ttnn.ReplicateTensorToMesh(md)
    k = torch.randn(1, NKV, nb * BS, HD).bfloat16()
    v = torch.randn(1, NKV, nb * BS, HD).bfloat16()
    kc = ttnn.from_torch(
        torch.zeros(nb, NKV, BS, HD).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=rep,
    )
    vc = ttnn.from_torch(
        torch.zeros(nb, NKV, BS, HD).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=rep,
    )
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_t = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, mesh_mapper=rep)
    kd = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=rep)
    vd = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=rep)
    ttnn.experimental.paged_fill_cache(kc, kd, pt_t, batch_idx=0)
    ttnn.experimental.paged_fill_cache(vc, vd, pt_t, batch_idx=0)
    q = torch.randn(1, B, NH, HD).bfloat16()
    qd = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=rep)
    ptB = ttnn.from_torch(pt.repeat(B, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md, mesh_mapper=rep)
    grid = md.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
    )
    kf, vf = k[0], v[0]
    # WINDOW_FP32=1: the drafter's compute config (HiFi4 + fp32 dest acc -> 128-position K chunks).
    ckc = {}
    if __import__("os").environ.get("WINDOW_FP32", "0") == "1":
        ckc = {
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        }
        print("[window] using fp32-dest compute config (128-position chunks)")

    def run(cur, window):
        cur_t = ttnn.from_torch(
            torch.tensor(cur, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            mesh_mapper=rep,
        )
        kw = {"sliding_window_size": window} if window else {}
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qd,
            kc,
            vc,
            page_table_tensor=ptB,
            cur_pos_tensor=cur_t,
            scale=HD**-0.5,
            program_config=cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **ckc,
            **kw,
        )
        return ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(B, NH, HD)

    for label, cur in (("causal rows cur=C+i", [C + i for i in range(B)]), ("bidir rows cur=C+B-1", [C + B - 1] * B)):
        o_none = run(cur, 0)
        o_w = run(cur, W)
        r_none = _ref(q[0], kf, vf, cur, [0] * B)
        r_A = _ref(q[0], kf, vf, cur, [c - W + 1 for c in cur])
        r_B = _ref(q[0], kf, vf, cur, [c - W for c in cur])
        r_C = _ref(q[0], kf, vf, cur, [c - W + 2 for c in cur])
        print(f"[window] {label}: no-window kernel vs full ref pcc={_pcc(o_none, r_none):.5f}")
        print(f"[window] {label}: W={W} kernel vs A (W keys)   pcc={_pcc(o_w, r_A):.5f}")
        print(f"[window] {label}: W={W} kernel vs B (W+1 keys) pcc={_pcc(o_w, r_B):.5f}")
        print(f"[window] {label}: W={W} kernel vs C (W-1 keys) pcc={_pcc(o_w, r_C):.5f}")
        print(f"[window] {label}: W={W} kernel vs full ref     pcc={_pcc(o_w, r_none):.5f}")
        assert _pcc(o_none, r_none) > 0.99
        assert (
            max(_pcc(o_w, r_A), _pcc(o_w, r_B), _pcc(o_w, r_C)) > 0.99
        ), "window kernel matches none of the candidate semantics"
