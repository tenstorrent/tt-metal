# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end PCC test for chunk_gated_delta_rule (gdn_chunk_ops.py).

Compares device-side chunkwise parallel DeltaNet (no CPU roundtrip for
triangular solve) against the sequential PyTorch reference implementation.

Weight-free: uses Qwen3.5-27B GDN dimensions with synthetic random inputs.

Run:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    MESH_DEVICE=P150x4 pytest models/demos/qwen35_27b/tt/tests/test_chunk_gated_delta_rule.py -v -s
"""

import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

# Qwen3.5-27B GDN constants (TP=4, P150x4)
GDN_Nk = 16
GDN_Nv = 48
GDN_Dk = 128
GDN_Dv = 128
TP = 4

Nk_TP = GDN_Nk // TP  # 4
Nv_TP = GDN_Nv // TP  # 12
Dk = GDN_Dk  # 128
Dv = GDN_Dv  # 128
repeat_factor = Nv_TP // Nk_TP  # 3
key_dim_tp = Nk_TP * Dk  # 512
value_dim_tp = Nv_TP * Dv  # 1536
qkv_dim_tp = 2 * key_dim_tp + value_dim_tp  # 2560
BH = Nv_TP  # 12 heads on device (B=1)
scale = Dk**-0.5


def _to_device(t, mesh_device, dtype=ttnn.float32):
    return ttnn.from_torch(
        t.float(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(t, mesh_device, leading_dim):
    """Collect from mesh, return first leading_dim rows (device 0 copy)."""
    all_data = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return all_data[:leading_dim].float()


def _compute_pcc(ref, test):
    r = ref.float().flatten()
    t = test.float().flatten()
    vr = r - r.mean()
    vt = t - t.mean()
    num = (vr * vt).sum()
    den = (vr.norm() * vt.norm()) + 1e-12
    return (num / den).item()


def _preprocess_inputs(conv_cpu, a_cpu, b_cpu, neg_exp_A_cpu, dt_bias_cpu):
    """Extract and preprocess Q, K, V, beta, g for chunk_gated_delta_rule.

    Returns:
        q:    [BH, N, Dk]  L2-normed (NOT scaled; scale applied inside chunk_gated_delta_rule)
        k:    [BH, N, Dk]  L2-normed
        v:    [BH, N, Dv]
        beta: [BH, N, 1]   sigmoid(b)
        g:    [BH, N]      neg_exp_A * softplus(a + dt_bias)
    """
    N = conv_cpu.shape[1]
    conv_2d = conv_cpu[0]  # [N, qkv_dim_tp]
    a_2d = a_cpu[0]  # [N, Nv_TP]
    b_2d = b_cpu[0]  # [N, Nv_TP]
    neg_A = neg_exp_A_cpu[0, 0]  # [Nv_TP]
    dt_b = dt_bias_cpu[0, 0]  # [Nv_TP]

    q_raw = conv_2d[:, :key_dim_tp].view(N, Nk_TP, Dk)
    k_raw = conv_2d[:, key_dim_tp : 2 * key_dim_tp].view(N, Nk_TP, Dk)
    v_raw = conv_2d[:, 2 * key_dim_tp :].view(N, Nv_TP, Dv)

    q_n = F.normalize(q_raw, p=2, dim=-1)  # [N, Nk_TP, Dk]
    k_n = F.normalize(k_raw, p=2, dim=-1)  # [N, Nk_TP, Dk]

    q_exp = q_n.repeat_interleave(repeat_factor, dim=1)  # [N, Nv_TP, Dk]
    k_exp = k_n.repeat_interleave(repeat_factor, dim=1)

    # Transpose to [BH, N, D]
    q_bht = q_exp.permute(1, 0, 2).contiguous()  # [Nv_TP, N, Dk]
    k_bht = k_exp.permute(1, 0, 2).contiguous()
    v_bht = v_raw.permute(1, 0, 2).contiguous()  # [Nv_TP, N, Dv]

    beta_cpu = torch.sigmoid(b_2d).permute(1, 0).unsqueeze(-1)  # [Nv_TP, N, 1]
    g_cpu = (neg_A * F.softplus(a_2d + dt_b)).permute(1, 0)  # [Nv_TP, N]

    return q_bht, k_bht, v_bht, beta_cpu, g_cpu


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "num_tokens,chunk_size",
    [
        (32, 32),  # single chunk
        (64, 32),  # two chunks, C=32
        (128, 32),  # four chunks, C=32
        (128, 64),  # two chunks, C=64
        (256, 64),  # four chunks, C=64
        (256, 128),  # two chunks, C=128 (6 Neumann steps)
        (512, 128),  # four chunks, C=128
        (1024, 128),  # eight chunks, C=128 (demo ISL=1k)
        (256, 256),  # single chunk, C=256 (7 Neumann steps)
        (512, 256),  # two chunks, C=256
        (4096, 4096),  # single chunk, full ISL (C=N, zero sequential loop)
    ],
)
def test_chunk_gated_delta_rule_pcc(mesh_device, reset_seeds, ensure_gc, num_tokens, chunk_size):
    """chunk_gated_delta_rule (device-side triangular solve) must match PyTorch reference."""
    from models.demos.qwen35_27b.reference.functional import gdn_prefill_ref
    from models.demos.qwen35_27b.tt.gdn_chunk_ops import chunk_gated_delta_rule, create_chunk_masks

    if mesh_device.get_num_devices() < 4:
        pytest.skip("P150x4 TP=4 required")

    N = num_tokens
    logger.info(f"N={N}, chunk_size={chunk_size}, BH={BH}, Dk={Dk}, Dv={Dv}")

    # ---- Synthetic inputs ----
    torch.manual_seed(42)
    conv_cpu = torch.randn(1, N, qkv_dim_tp) * 0.1
    a_cpu = torch.randn(1, N, Nv_TP) * 0.1
    b_cpu = torch.randn(1, N, Nv_TP) * 0.1
    neg_exp_A_cpu = torch.full((1, 1, Nv_TP), -0.1)
    dt_bias_cpu = torch.zeros(1, 1, Nv_TP)

    # ---- Reference: sequential PyTorch ----
    ref_out, ref_state = gdn_prefill_ref(
        conv_cpu,
        a_cpu,
        b_cpu,
        neg_exp_A_cpu,
        dt_bias_cpu,
        scale=scale,
        Dk=Dk,
        Dv=Dv,
        Nk_TP=Nk_TP,
        Nv_TP=Nv_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )
    # ref_out: [Nv_TP, N, Dv], ref_state: [Nv_TP, Dk, Dv]

    # ---- Device: chunk_gated_delta_rule ----
    q_bht, k_bht, v_bht, beta_bht, g_bht = _preprocess_inputs(conv_cpu, a_cpu, b_cpu, neg_exp_A_cpu, dt_bias_cpu)

    q_tt = _to_device(q_bht, mesh_device)
    k_tt = _to_device(k_bht, mesh_device)
    v_tt = _to_device(v_bht, mesh_device)
    beta_tt = _to_device(beta_bht, mesh_device)
    g_tt = _to_device(g_bht, mesh_device)

    cached_masks = create_chunk_masks(chunk_size, mesh_device)

    # scale is applied internally inside chunk_gated_delta_rule (q *= scale)
    o_tt, state_tt = chunk_gated_delta_rule(
        q_tt,
        k_tt,
        v_tt,
        beta_tt,
        g_tt,
        chunk_size=chunk_size,
        scale=scale,
        mesh_device=mesh_device,
        cached_masks=cached_masks,
    )

    out_dev = _from_device(o_tt, mesh_device, BH)  # [BH, N, Dv]
    state_dev = _from_device(state_tt, mesh_device, BH)  # [BH, Dk, Dv]

    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)
    ttnn.deallocate(beta_tt)
    ttnn.deallocate(g_tt)
    ttnn.deallocate(o_tt)
    ttnn.deallocate(state_tt)
    for v in cached_masks.values():
        ttnn.deallocate(v)

    # ---- Compare ----
    out_pcc = _compute_pcc(ref_out, out_dev)
    state_pcc = _compute_pcc(ref_state, state_dev)

    logger.info(f"  Output PCC: {out_pcc:.6f}  (max|err|={( ref_out - out_dev).abs().max():.4e})")
    logger.info(f"  State  PCC: {state_pcc:.6f}  (max|err|={(ref_state - state_dev).abs().max():.4e})")

    assert out_pcc > 0.99, f"N={N} chunk={chunk_size}: output PCC {out_pcc:.6f} < 0.99"
    assert state_pcc > 0.99, f"N={N} chunk={chunk_size}: state PCC {state_pcc:.6f} < 0.99"
    logger.info(f"PASS N={N} chunk={chunk_size}: out={out_pcc:.6f}, state={state_pcc:.6f}")


_MESH_SHAPE = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8)))


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "batch,C",
    [
        (12, 32),  # BH=12, C=32 (1 chunk at ISL=32)
        (96, 128),  # BH=12 * 8 chunks, C=128 (ISL=1024)
        (384, 128),  # BH=12 * 32 chunks, C=128 (ISL=4096)
    ],
    ids=["b12_c32", "b96_c128", "b384_c128"],
)
def test_solve_triangular_ttnn(mesh_device, reset_seeds, ensure_gc, batch, C):
    """_solve_lower_triangular_ttnn must match torch.linalg.solve_triangular (PCC > 0.99).

    Constructs L = I + kk * tril_ones (same structure as in chunk_gated_delta_rule
    where L = I - attn_raw = I + kk*L_mask). Diagonal entries are > 1 (well-conditioned).
    """
    from models.demos.qwen35_27b.tt.gdn_chunk_ops import _solve_lower_triangular_ttnn

    torch.manual_seed(7)
    # Build realistic lower triangular L: I + kk*tril, kk = k_beta @ k.T
    # Use L2-normed k so diagonal kk[i,i] in (0, 1]
    k = torch.randn(batch, C, Dk)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    kb = k * torch.sigmoid(torch.randn(batch, C, 1)) * 0.3  # k_beta
    kk = kb @ k.transpose(1, 2)  # [batch, C, C]
    tril_mask = torch.tril(torch.ones(C, C))
    kk_lower = kk * tril_mask
    L = torch.eye(C).unsqueeze(0) + kk_lower  # [batch, C, C] lower triangular

    # --- CPU reference ---
    eye_C = torch.eye(C).unsqueeze(0).expand(batch, -1, -1)
    L_inv_ref = torch.linalg.solve_triangular(L, eye_C, upper=False)  # [batch, C, C]

    # --- Device ---
    L_tt = ttnn.from_torch(
        L.float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    eye_1cc = ttnn.from_torch(
        torch.eye(C, dtype=torch.float32).unsqueeze(0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    L_inv_tt = _solve_lower_triangular_ttnn(L_tt, eye_1cc, mesh_device)
    ttnn.deallocate(L_tt)
    ttnn.deallocate(eye_1cc)

    L_inv_dev = ttnn.to_torch(L_inv_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:batch].float()
    ttnn.deallocate(L_inv_tt)

    pcc = _compute_pcc(L_inv_ref, L_inv_dev)
    max_err = (L_inv_ref - L_inv_dev).abs().max().item()
    logger.info(f"  batch={batch} C={C}: PCC={pcc:.6f}  max_err={max_err:.4e}")
    assert pcc > 0.99, f"solve_triangular PCC {pcc:.6f} < 0.99  (batch={batch}, C={C})"
    logger.info(f"PASS  batch={batch} C={C}: PCC={pcc:.6f}")


def _build_realistic_L(batch, C, seed, high_coherence=False):
    """Build L = I + kk * tril_with_decay — same structure as in chunk_gated_delta_rule.

    high_coherence=True: k vectors all point in nearly the same direction.
    This is the adversarial case for plain Neumann (spectral_norm(N) >> 1)
    that motivated blocked forward substitution.
    """
    torch.manual_seed(seed)
    K = 128
    if high_coherence:
        # All k ≈ same direction: e_0 + small noise
        dominant = torch.zeros(K)
        dominant[0] = 1.0
        k = dominant.unsqueeze(0).unsqueeze(0).expand(batch, C, K) + torch.randn(batch, C, K) * 0.05
    else:
        k = torch.randn(batch, C, K)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    beta = torch.sigmoid(torch.randn(batch, C, 1)) * 0.5 + 0.5  # ∈ (0.5, 1.0)

    # Realistic decay: cumsum of negative g values (log-space decay per token)
    g = -torch.rand(batch, C) * 0.3 - 0.05  # ∈ (-0.35, -0.05) per step
    triu_ones = torch.triu(torch.ones(C, C))
    decay = (g.unsqueeze(1) @ triu_ones.unsqueeze(0)).squeeze(1)  # [batch, C]
    decay_col = decay.unsqueeze(-1)
    decay_row = decay.unsqueeze(-2)
    L_mask = torch.exp(torch.clamp(decay_col - decay_row, max=0.0)) * torch.tril(torch.ones(C, C))

    kk = torch.einsum("bik,bjk->bij", k * beta, k)  # [batch, C, C]
    L = torch.eye(C).unsqueeze(0) + kk * L_mask  # lower triangular, positive diagonal
    return L.float()


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "batch,C,high_coherence",
    [
        (12, 32, False),  # single block, random k
        (96, 128, False),  # ISL=1024, random k
        (384, 128, False),  # ISL=4096, random k
        (96, 128, True),  # ISL=1024, adversarial high-coherence k (breaks plain Neumann)
        (384, 128, True),  # ISL=4096, adversarial high-coherence k
    ],
    ids=["b12_c32_rand", "b96_c128_rand", "b384_c128_rand", "b96_c128_hico", "b384_c128_hico"],
)
def test_solve_triangular_blocked_ttnn(mesh_device, reset_seeds, ensure_gc, batch, C, high_coherence):
    """_solve_lower_triangular_blocked_ttnn must match torch.linalg.solve_triangular (PCC > 0.99).

    Tests both random k-vectors (easy case) and high-coherence k-vectors (adversarial case
    where plain Neumann on the full 128×128 matrix fails due to catastrophic cancellation).
    The blocked approach inverts 32×32 diagonal sub-blocks where the decay mask suppresses
    off-diagonal entries, keeping Neumann stable.
    """
    from models.demos.qwen35_27b.tt.gdn_chunk_ops import _solve_lower_triangular_blocked_ttnn

    L = _build_realistic_L(batch, C, seed=42, high_coherence=high_coherence)

    # --- CPU reference (numerically exact via LAPACK forward substitution) ---
    eye_C = torch.eye(C).unsqueeze(0).expand(batch, -1, -1)
    L_inv_ref = torch.linalg.solve_triangular(L, eye_C.float(), upper=False)  # [batch, C, C]

    # --- Device ---
    L_tt = ttnn.from_torch(
        L,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    eye_1cc = ttnn.from_torch(
        torch.eye(C, dtype=torch.float32).unsqueeze(0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    L_inv_tt = _solve_lower_triangular_blocked_ttnn(L_tt, eye_1cc, mesh_device)
    ttnn.deallocate(L_tt)
    ttnn.deallocate(eye_1cc)

    L_inv_dev = ttnn.to_torch(L_inv_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:batch].float()
    ttnn.deallocate(L_inv_tt)

    pcc = _compute_pcc(L_inv_ref, L_inv_dev)
    max_err = (L_inv_ref - L_inv_dev).abs().max().item()
    tag = "hico" if high_coherence else "rand"
    logger.info(f"  blocked solve [{tag}] batch={batch} C={C}: PCC={pcc:.6f}  max_err={max_err:.4e}")
    assert pcc > 0.99, f"blocked solve PCC {pcc:.6f} < 0.99  (batch={batch}, C={C}, high_coherence={high_coherence})"
    logger.info(f"PASS  blocked [{tag}] batch={batch} C={C}: PCC={pcc:.6f}")


def _time_fn(fn, mesh_device, warmup=2, repeats=3):
    """Run fn() warmup+repeats times, return (min_ms, median_ms) of the timed runs."""
    import statistics
    import time as _t

    for _ in range(warmup):
        fn()
        ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(repeats):
        t0 = _t.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)
        times.append((_t.perf_counter() - t0) * 1000)
    return min(times), statistics.median(times)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_chunk_gated_delta_rule_perf_sweep(mesh_device, reset_seeds, ensure_gc):
    """Sweep chunk_size=[32,64,128,256,512,1024,2048,4096] at N=4096.

    Single device open for the entire sweep — one setup/teardown.
    Prints a result row as each chunk size completes.
    """
    from models.demos.qwen35_27b.tt.gdn_chunk_ops import chunk_gated_delta_rule, create_chunk_masks
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused

    if mesh_device.get_num_devices() < 4:
        pytest.skip("P150x4 TP=4 required")

    N = 4096
    num_pairs = BH
    chunk_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    torch.manual_seed(42)
    conv_cpu = torch.randn(1, N, qkv_dim_tp) * 0.1
    a_cpu = torch.randn(1, N, Nv_TP) * 0.1
    b_cpu = torch.randn(1, N, Nv_TP) * 0.1
    neg_exp_A_cpu = torch.full((1, 1, Nv_TP), -0.1)
    dt_bias_cpu = torch.zeros(1, 1, Nv_TP)

    q_bht, k_bht, v_bht, beta_bht, g_bht = _preprocess_inputs(conv_cpu, a_cpu, b_cpu, neg_exp_A_cpu, dt_bias_cpu)
    q_tt = _to_device(q_bht, mesh_device)
    k_tt = _to_device(k_bht, mesh_device)
    v_tt = _to_device(v_bht, mesh_device)
    beta_tt = _to_device(beta_bht, mesh_device)
    g_tt = _to_device(g_bht, mesh_device)

    def _to_mesh_bf16(t):
        return ttnn.from_torch(
            t.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def _unshard(t):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        return t

    conv_3d = _unshard(_to_mesh_bf16(conv_cpu.float()))
    a_3d = _unshard(_to_mesh_bf16(a_cpu.float()))
    b_3d = _unshard(_to_mesh_bf16(b_cpu.float()))
    neg_exp_A = _to_mesh_bf16(neg_exp_A_cpu.float())
    dt_bias_tt = _to_mesh_bf16(dt_bias_cpu.float())
    norm_w = _to_mesh_bf16(torch.ones(1, 1, value_dim_tp))
    scale_tt = _to_mesh_bf16(torch.full((1, 1, 1), scale))
    rms_scale_tt = _to_mesh_bf16(torch.full((1, 1, 1), float(Dv**0.5)))
    rms_eps_tt = _to_mesh_bf16(torch.full((1, 1, 1), Dv * 1e-6))
    rec_states = _to_mesh_bf16(torch.zeros(num_pairs, Dk, Dv))
    fused_out = _to_mesh_bf16(torch.zeros(num_pairs * N, 1, Dv))

    def _run_fused():
        gdn_prefill_fused(
            conv_3d,
            a_3d,
            b_3d,
            neg_exp_A,
            dt_bias_tt,
            norm_w,
            scale_tt,
            rms_scale_tt,
            rms_eps_tt,
            rec_states,
            fused_out,
            num_pairs=num_pairs,
            num_tokens=N,
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )

    # Time fused baseline once (shared across all chunk sizes)
    fused_min, _ = _time_fn(_run_fused, mesh_device)

    logger.info(f"\n{'chunk_size':>12} {'steps':>6} {'chunk_ms':>10} {'fused_ms':>10} {'speedup':>8}")
    logger.info(f"{'----------':>12} {'-----':>6} {'--------':>10} {'--------':>10} {'-------':>8}")

    results = []
    for C in chunk_sizes:
        cached_masks = create_chunk_masks(C, mesh_device)

        def _run_chunk(C=C, cached_masks=cached_masks):
            o, s = chunk_gated_delta_rule(
                q_tt,
                k_tt,
                v_tt,
                beta_tt,
                g_tt,
                chunk_size=C,
                scale=scale,
                mesh_device=mesh_device,
                cached_masks=cached_masks,
            )
            ttnn.deallocate(o)
            ttnn.deallocate(s)

        chunk_min, _ = _time_fn(_run_chunk, mesh_device)
        speedup = fused_min / chunk_min
        steps = N // C

        logger.info(f"{C:>12} {steps:>6} {chunk_min:>10.1f} {fused_min:>10.1f} {speedup:>8.2f}×")
        results.append((C, chunk_min, speedup))

        for v in cached_masks.values():
            ttnn.deallocate(v)

    for t in (
        q_tt,
        k_tt,
        v_tt,
        beta_tt,
        g_tt,
        conv_3d,
        a_3d,
        b_3d,
        neg_exp_A,
        dt_bias_tt,
        norm_w,
        scale_tt,
        rms_scale_tt,
        rms_eps_tt,
        rec_states,
        fused_out,
    ):
        ttnn.deallocate(t)

    best_C, best_ms, best_speedup = min(results, key=lambda x: x[1])
    logger.info(f"\n  Best chunk_size={best_C}: {best_ms:.1f} ms ({best_speedup:.2f}× vs fused {fused_min:.1f} ms)")
    assert best_speedup > 1.0, f"Best chunk config should beat fused kernel, got {best_speedup:.2f}×"
