"""Quick smoke test for gated_delta_attn_seq Path A API."""
import torch
import ttnn
import time


def chunk_seq_ref(v_cor, k_cum, q_decay, intra_attn, k_decay_t, dl_exp):
    BH, NC, C, Dv = v_cor.shape
    Dk = k_cum.shape[-1]
    S = torch.zeros(BH, Dk, Dv)
    outs = []
    for c in range(NC):
        v_prime = torch.bmm(k_cum[:, c], S)
        v_new = v_cor[:, c] - v_prime
        o_inter = torch.bmm(q_decay[:, c], S)
        intra_v = torch.bmm(intra_attn[:, c], v_new)
        out = o_inter + intra_v
        s_upd = torch.bmm(k_decay_t[:, c], v_new)
        scl = dl_exp[:, c, 0, 0].view(BH, 1, 1)
        S = S * scl + s_upd
        outs.append(out)
    return torch.stack(outs, dim=1), S


def make_L_unit_identity(BH, NC, C):
    eye_32 = torch.eye(32)
    L = torch.zeros(BH, NC, C, C)
    for b in range(C // 32):
        L[:, :, b * 32 : (b + 1) * 32, b * 32 : (b + 1) * 32] = eye_32
    return L


def make_L_inv(BH, NC, C):
    eye_32 = torch.eye(32)
    L_inv = torch.zeros(BH, NC, C, 32)
    for b in range(C // 32):
        L_inv[:, :, b * 32 : (b + 1) * 32, :] = eye_32
    return L_inv


def to_tt(t, mesh):
    return ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def to_torch(t, mesh):
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    n = 4
    return full[: full.shape[0] // n]


def run_test(mesh, BH, NC, C=128, Dk=128, Dv=128, label=""):
    torch.manual_seed(42)
    v_beta_sc = torch.randn(BH, NC, C, Dv) * 0.3
    k_bd_sc = torch.randn(BH, NC, C, Dk) * 0.1
    q_decay = torch.randn(BH, NC, C, Dk) * 0.1
    intra_attn = torch.randn(BH, NC, C, C) * 0.01
    k_decay_t = torch.randn(BH, NC, Dk, C) * 0.1
    dl_exp = torch.ones(BH, NC, 1, 1) * 0.95
    L_unit = make_L_unit_identity(BH, NC, C)
    L_inv = make_L_inv(BH, NC, C)

    # With L_unit=I, v_cor=v_beta_sc and k_cum=k_bd_sc so ref matches exactly
    ref_out, ref_state = chunk_seq_ref(v_beta_sc, k_bd_sc, q_decay, intra_attn, k_decay_t, dl_exp)

    t0 = time.perf_counter()
    out_tt, state_tt = ttnn.transformer.gated_delta_attn_seq(
        to_tt(L_unit, mesh),
        to_tt(v_beta_sc, mesh),
        to_tt(k_bd_sc, mesh),
        to_tt(intra_attn, mesh),
        to_tt(q_decay, mesh),
        to_tt(k_decay_t, mesh),
        to_tt(dl_exp, mesh),
        to_tt(L_inv, mesh),
    )
    ttnn.synchronize_device(mesh)
    elapsed = (time.perf_counter() - t0) * 1000

    out_t = to_torch(out_tt, mesh)
    state_t = to_torch(state_tt, mesh)
    ttnn.deallocate(out_tt)
    ttnn.deallocate(state_tt)

    ref_f = ref_out.flatten().float()
    out_f = out_t.flatten().float()
    pcc_out = torch.corrcoef(torch.stack([ref_f, out_f]))[0, 1].item()
    ref_s = ref_state.flatten().float()
    out_s = state_t.flatten().float()
    pcc_st = torch.corrcoef(torch.stack([ref_s, out_s]))[0, 1].item()

    err_out = (ref_out - out_t).abs().max().item()
    err_state = (ref_state - state_t).abs().max().item()

    status = "PASS" if pcc_out >= 0.99 and pcc_st >= 0.99 else "FAIL"
    print(
        f"  {label:20s}  out_pcc={pcc_out:.4f} (max_err={err_out:.3f})  "
        f"state_pcc={pcc_st:.4f} (max_err={err_state:.3f})  {elapsed:.0f}ms  [{status}]"
    )

    if pcc_out < 0.99 or pcc_st < 0.99:
        print(f"    ref_out[0,0,0,:4]  = {ref_out[0,0,0,:4].tolist()}")
        print(f"    out_t[0,0,0,:4]   = {out_t[0,0,0,:4].tolist()}")
        print(f"    ref_state[0,:4,0] = {ref_state[0,:4,0].tolist()}")
        print(f"    state_t[0,:4,0]   = {state_t[0,:4,0].tolist()}")
        print(f"    out min/max: {out_t.min().item():.4f} / {out_t.max().item():.4f}")
        print(f"    state min/max: {state_t.min().item():.4f} / {state_t.max().item():.4f}")

    return pcc_out >= 0.99 and pcc_st >= 0.99


if __name__ == "__main__":
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        passed = 0
        failed = 0
        for bh, nc in [(1, 1), (1, 2), (2, 2), (4, 2), (12, 4), (12, 32)]:
            ok = run_test(mesh, bh, nc, label=f"BH={bh} NC={nc}")
            if ok:
                passed += 1
            else:
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
    finally:
        ttnn.close_mesh_device(mesh)
