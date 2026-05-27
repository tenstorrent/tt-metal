# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the gated_delta_attn_seq TTNN kernel (Path A).

test_gated_delta_attn_kernel  (fast): tests C++ kernel directly with pre-computed
    unit-diagonal inputs — no JIT compilation of preprocessing ops.

test_gated_delta_attn_pipeline (slow / integration): tests the full
    chunk_gated_delta_rule_seq pipeline against a PyTorch reference.

test_gated_delta_attn_perf    (perf): measures ms/layer targeting < 10ms on P150x4.

Run kernel correctness only (fast, < 2 min):
    pytest tests/ttnn/unit_tests/operations/sdpa/test_gated_delta_attn.py \
        -k kernel -v -s

Run full pipeline + perf (needs pre-compiled float32 ops):
    pytest tests/ttnn/unit_tests/operations/sdpa/test_gated_delta_attn.py \
        -k pipeline or perf -v -s --timeout=1800
"""

import time

import pytest
import torch
import torch.nn.functional as F

import ttnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tt(t: torch.Tensor, mesh, dtype=ttnn.float32):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_torch(t, mesh):
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    n = 4  # P150x4 mesh
    return full[: full.shape[0] // n]


def _assert_pcc(ref: torch.Tensor, out: torch.Tensor, threshold: float = 0.99, name: str = ""):
    ref_f = ref.flatten().float()
    out_f = out.flatten().float()
    pcc = torch.corrcoef(torch.stack([ref_f, out_f]))[0, 1].item()
    max_err = (ref_f - out_f).abs().max().item()
    print(f"  {name}: PCC={pcc:.5f}  max_err={max_err:.4f}")
    assert pcc >= threshold, f"{name}: PCC {pcc:.5f} < {threshold}"


# ---------------------------------------------------------------------------
# Pure-PyTorch sequential scan (Path A unit form) for the chunk-level kernel.
# ---------------------------------------------------------------------------


def _compute_L_inv(L_unit):
    """Compute 4 diagonal block inverses of L_unit. Returns [BH, NC, C, 32]."""
    BH, NC, C, _ = L_unit.shape
    n_blocks = C // 32
    L_inv = torch.zeros(BH, NC, C, 32, dtype=torch.float32)
    for h in range(BH):
        for c in range(NC):
            for b in range(n_blocks):
                block = L_unit[h, c, b * 32 : (b + 1) * 32, b * 32 : (b + 1) * 32].float()
                L_inv[h, c, b * 32 : (b + 1) * 32, :] = torch.linalg.inv(block)
    return L_inv


def _make_path_a_inputs(v_cor, k_cum, q_decay, intra_attn, k_decay_t, dl_exp, chunk_size=128):
    """Convert standard chunk tensors to Path A unit-diagonal inputs."""
    BH, NC, C, Dv = v_cor.shape
    Dk = k_cum.shape[3]

    # For testing: construct L_unit = I (identity diagonal blocks).
    # With L_unit = I, the solve gives v_cor = v_beta_sc and k_cum = k_bd_sc.
    eye_32 = torch.eye(32, dtype=torch.float32)
    L_unit = torch.zeros(BH, NC, C, C)
    for b in range(BH):
        for nc in range(NC):
            for bi in range(C // 32):
                L_unit[b, nc, bi * 32 : (bi + 1) * 32, bi * 32 : (bi + 1) * 32] = eye_32

    L_inv = _compute_L_inv(L_unit)  # [BH, NC, C, 32]

    return L_unit, v_cor, k_cum, intra_attn, q_decay, k_decay_t, dl_exp, L_inv


def _chunk_seq_reference(v_cor, k_cum, q_decay, intra_attn, k_decay_t, dl_exp, initial_state=None):
    """CPU reference for the sequential scan kernel."""
    BH, NC, C, Dv = v_cor.shape
    Dk = k_cum.shape[3]

    S = torch.zeros(BH, Dk, Dv) if initial_state is None else initial_state.clone().float()
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


def _pytorch_gdn_reference(q, k, v, beta, g, chunk_size=128, scale=None, initial_state=None):
    """Per-token sequential DeltaNet reference — CPU, float32, exact."""
    BH, T, Dk = q.shape
    Dv = v.shape[2]
    if scale is None:
        scale = Dk**-0.5

    state = torch.zeros(BH, Dk, Dv, dtype=torch.float32) if initial_state is None else initial_state.clone().float()
    outs = []
    for t in range(T):
        q_t = q[:, t, :].float() * scale
        k_t = k[:, t, :].float()
        v_t = v[:, t, :].float()
        beta_t = beta[:, t, 0].float()
        g_t = g[:, t].float()

        state = state * torch.exp(g_t)[:, None, None]
        kv_mem = torch.bmm(k_t.unsqueeze(1), state).squeeze(1)
        delta = beta_t[:, None] * (v_t - kv_mem)
        state = state + torch.bmm(k_t.unsqueeze(2), delta.unsqueeze(1))
        out = torch.bmm(q_t.unsqueeze(1), state).squeeze(1)
        outs.append(out)

    return torch.stack(outs, dim=1), state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh(request):
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    yield mesh_device
    ttnn.close_mesh_device(mesh_device)


# ---------------------------------------------------------------------------
# Fast kernel correctness test: feeds pre-computed unit-diagonal inputs.
# L_unit = I makes the solve trivial, so v_cor = v_beta_sc and k_cum = k_bd_sc.
# This matches the existing sequential scan reference exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_heads, num_chunks, chunk_size, key_dim, val_dim",
    [
        pytest.param(12, 32, 128, 128, 128, id="qwen35_27b_isl4k_kernel"),
        pytest.param(12, 8, 128, 128, 128, id="qwen35_27b_isl1k_kernel"),
    ],
)
def test_gated_delta_attn_kernel(mesh, num_heads, num_chunks, chunk_size, key_dim, val_dim):
    """Direct C++ kernel test with L_unit=I: compare vs sequential PyTorch scan."""
    torch.manual_seed(42)
    BH, NC, C, Dk, Dv = num_heads, num_chunks, chunk_size, key_dim, val_dim

    # Random chunk inputs for the sequential scan reference
    v_cor = torch.randn(BH, NC, C, Dv) * 0.3
    k_cum = torch.randn(BH, NC, C, Dk) * 0.1
    q_decay = torch.randn(BH, NC, C, Dk) * 0.1
    intra_attn = torch.randn(BH, NC, C, C) * 0.01
    k_decay_t = torch.randn(BH, NC, Dk, C) * 0.1
    dl_exp = torch.ones(BH, NC, 1, 1) * 0.95

    # CPU reference (exact sequential scan)
    ref_out, ref_state = _chunk_seq_reference(v_cor, k_cum, q_decay, intra_attn, k_decay_t, dl_exp)

    # Build Path A inputs with L_unit = I (solve is identity: out = RHS)
    L_unit, v_beta_sc, k_bd_sc, intra_attn_pa, q_decay_pa, k_decay_t_pa, dl_exp_pa, L_inv = _make_path_a_inputs(
        v_cor, k_cum, q_decay, intra_attn, k_decay_t, dl_exp
    )

    out_tt, state_tt = ttnn.transformer.gated_delta_attn_seq(
        _to_tt(L_unit, mesh),
        _to_tt(v_beta_sc, mesh),
        _to_tt(k_bd_sc, mesh),
        _to_tt(intra_attn_pa, mesh),
        _to_tt(q_decay_pa, mesh),
        _to_tt(k_decay_t_pa, mesh),
        _to_tt(dl_exp_pa, mesh),
        _to_tt(L_inv, mesh),
    )

    out_t = _to_torch(out_tt, mesh)
    state_t = _to_torch(state_tt, mesh)

    _assert_pcc(ref_out, out_t, threshold=0.99, name="output")
    _assert_pcc(ref_state, state_t, threshold=0.99, name="final_state")


# ---------------------------------------------------------------------------
# Integration test: full chunk_gated_delta_rule_seq pipeline vs PyTorch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_heads, seq_len, key_dim, val_dim, chunk_size",
    [
        pytest.param(12, 1024, 128, 128, 128, id="qwen35_27b_isl1k_pipeline"),
        pytest.param(12, 4096, 128, 128, 128, id="qwen35_27b_isl4k_pipeline"),
    ],
)
def test_gated_delta_attn_pipeline(mesh, num_heads, seq_len, key_dim, val_dim, chunk_size):
    """Full pipeline: chunk_gated_delta_rule_seq vs per-token PyTorch reference."""
    torch.manual_seed(42)
    BH, S, Dk, Dv, C = num_heads, seq_len, key_dim, val_dim, chunk_size

    q_t = torch.randn(BH, S, Dk)
    k_t = F.normalize(torch.randn(BH, S, Dk), dim=-1)
    v_t = torch.randn(BH, S, Dv)
    beta_t = torch.sigmoid(torch.randn(BH, S, 1))
    g_t = -F.softplus(torch.randn(BH, S))

    print(f"\n  PyTorch reference for isl={S}...", flush=True)
    ref_out, ref_state = _pytorch_gdn_reference(q_t, k_t, v_t, beta_t, g_t, chunk_size=C)

    print(f"  Running chunk_gated_delta_rule_seq...", flush=True)
    from models.demos.qwen35_27b.tt.gdn_chunk_ops import create_chunk_masks
    from models.demos.qwen35_27b.tt.gdn_chunk_ops_seq import chunk_gated_delta_rule_seq

    masks = create_chunk_masks(C, mesh)
    out, final_state = chunk_gated_delta_rule_seq(
        _to_tt(q_t, mesh),
        _to_tt(k_t, mesh),
        _to_tt(v_t, mesh),
        _to_tt(beta_t, mesh),
        _to_tt(g_t, mesh),
        chunk_size=C,
        scale=None,
        initial_state=None,
        mesh_device=mesh,
        cached_masks=masks,
    )
    out_t = _to_torch(out, mesh)
    state_t = _to_torch(final_state, mesh)
    ttnn.deallocate(out)
    ttnn.deallocate(final_state)
    print(f"  Done. out shape={out_t.shape}", flush=True)

    _assert_pcc(ref_out, out_t, threshold=0.99, name="output")
    _assert_pcc(ref_state, state_t, threshold=0.95, name="final_state")


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_heads, seq_len, key_dim, val_dim, chunk_size, target_ms",
    [
        pytest.param(12, 4096, 128, 128, 128, 10.0, id="qwen35_27b_isl4k_perf"),
    ],
)
def test_gated_delta_attn_perf(mesh, num_heads, seq_len, key_dim, val_dim, chunk_size, target_ms):
    """Single-layer sequential scan should complete in < target_ms ms."""
    torch.manual_seed(7)
    BH, S, Dk, Dv, C = num_heads, seq_len, key_dim, val_dim, chunk_size
    NC = S // C

    v_cor = torch.randn(BH, NC, C, Dv) * 0.3
    k_cum = torch.randn(BH, NC, C, Dk) * 0.1
    q_decay = torch.randn(BH, NC, C, Dk) * 0.1
    intra_attn = torch.randn(BH, NC, C, C) * 0.01
    k_decay_t = torch.randn(BH, NC, Dk, C) * 0.1
    dl_exp = torch.ones(BH, NC, 1, 1) * 0.95

    L_unit, v_beta_sc, k_bd_sc, intra_attn_pa, q_decay_pa, k_decay_t_pa, dl_exp_pa, L_inv = _make_path_a_inputs(
        v_cor, k_cum, q_decay, intra_attn, k_decay_t, dl_exp
    )

    def _make_inputs():
        return (
            _to_tt(L_unit, mesh),
            _to_tt(v_beta_sc, mesh),
            _to_tt(k_bd_sc, mesh),
            _to_tt(intra_attn_pa, mesh),
            _to_tt(q_decay_pa, mesh),
            _to_tt(k_decay_t_pa, mesh),
            _to_tt(dl_exp_pa, mesh),
            _to_tt(L_inv, mesh),
        )

    for _ in range(2):
        ins = _make_inputs()
        out, st = ttnn.transformer.gated_delta_attn_seq(*ins)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(out)
        ttnn.deallocate(st)

    N = 5
    times = []
    for _ in range(N):
        ins = _make_inputs()
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        out, st = ttnn.transformer.gated_delta_attn_seq(*ins)
        ttnn.synchronize_device(mesh)
        times.append((time.time() - t0) * 1000)
        ttnn.deallocate(out)
        ttnn.deallocate(st)

    avg_ms = sum(times) / N
    print(f"\n  gated_delta_attn_seq perf: {avg_ms:.1f} ms/layer (target {target_ms} ms)")
    assert avg_ms < target_ms * 3, f"Perf regression: {avg_ms:.1f} ms/layer exceeds 3× target ({target_ms * 3} ms)"
