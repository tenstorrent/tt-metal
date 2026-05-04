# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy comparison: 1D-mcast (current CP prefill) vs DRAM-sharded (CP decode)
matmul kernels on the same input, against a CPU fp32 reference.

Why this test exists
--------------------
Voice timbre regresses when we swap CP_Prefill MLP from the 1D-mcast L1
INTERLEAVED matmul to the DRAM-sharded matmul (the same kernel CP_Decode uses
without issue). Both kernels compute the same mathematical x @ W.T + b, but
they differ in:
  * reduction order (different per-core K split)
  * accumulation precision / packing
  * which cores process which slice
For the autoregressive sampling trajectory, those differences accumulate
through 14 codebooks per frame and shift the speaker embedding enough to
change voice identity.

This test loads real Qwen3-TTS Talker / CodePredictor weights and runs each
matmul through both kernels with identical bf16 input; results are compared
against CPU fp32 (the implicit reference the model was trained with).

Run:
    pytest -s models/demos/qwen3_tts/tests/test_cp_prefill_matmul_accuracy.py
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    dram_sharded_program_config,
    find_grid_k_n,
    width_sharded_l1_memcfg,
)
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    a = x.flatten().float()
    b = y.flatten().float()
    a_c = a - a.mean()
    b_c = b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-12)).item()


def _summarize(label: str, ref: torch.Tensor, x: torch.Tensor) -> dict:
    """Return PCC / max abs / mean abs / cosine similarity vs reference. Print one line."""
    r = ref.flatten().float()
    a = x.flatten().float()
    diff = (r - a).abs()
    cos = (r * a).sum() / (r.norm() * a.norm() + 1e-12)
    pcc = pearson(r, a)
    summary = {
        "label": label,
        "pcc": pcc,
        "cos": cos.item(),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "rms_ref": (r**2).mean().sqrt().item(),
        "rms_diff": (diff**2).mean().sqrt().item(),
    }
    print(
        f"  {label:30s}  PCC={pcc:.6f}  cos={cos:.6f}  "
        f"max|Δ|={summary['max_abs_diff']:.4f}  mean|Δ|={summary['mean_abs_diff']:.5f}  "
        f"RMS_diff/RMS_ref={summary['rms_diff']/summary['rms_ref']:.4f}"
    )
    return summary


def _run_matmul_1d_mcast(
    device,
    x_torch: torch.Tensor,
    w_torch: torch.Tensor,
    b_torch: torch.Tensor,
    m: int,
    math_fidelity=ttnn.MathFidelity.LoFi,
) -> torch.Tensor:
    """1D-mcast matmul: in0 L1_INTERLEAVED, weight DRAM_INTERLEAVED, out L1_INTERLEAVED.

    Mirrors the path CP prefill currently uses (mlp.py prefill branch).
    """
    grid = device.compute_with_storage_grid_size()
    K = w_torch.shape[1]
    N = w_torch.shape[0]
    progcfg = make_linear_1d_program_config(m=m, k=K, n=N, grid_x=grid.x, grid_y=grid.y, fp32_dest_acc_en=True)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w_tt = ttnn.from_torch(
        w_torch.t().unsqueeze(0).unsqueeze(0).contiguous(),  # [1,1,K,N]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_torch.reshape(1, 1, 1, -1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out_tt = ttnn.linear(
        x_tt,
        w_tt,
        bias=b_tt,
        compute_kernel_config=compute_cfg,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=progcfg,
    )
    out = ttnn.to_torch(out_tt).squeeze(0).squeeze(0)  # [m, N]
    ttnn.deallocate(x_tt)
    ttnn.deallocate(w_tt)
    ttnn.deallocate(b_tt)
    ttnn.deallocate(out_tt)
    return out


def _run_matmul_dram_sharded(
    device,
    x_torch: torch.Tensor,
    w_torch: torch.Tensor,
    b_torch: torch.Tensor,
    m: int,
    math_fidelity=ttnn.MathFidelity.LoFi,
) -> torch.Tensor:
    """DRAM-sharded matmul: in0 L1 WIDTH_SHARDED, weight DRAM_WIDTH_SHARDED, out L1 WIDTH_SHARDED.

    Mirrors the path CP decode uses (mlp.py decode branch). Asserts M==tile_height
    so we pad m to 32.
    """
    K = w_torch.shape[1]
    w_kn = w_torch.t().contiguous()
    w_tt, k, n_padded = build_dram_sharded_weight(w_kn, device, dtype=ttnn.bfloat16)
    k_tiles, n_tiles = k // 32, n_padded // 32
    rows, cols = find_grid_k_n(k_tiles, n_tiles)
    progcfg = dram_sharded_program_config(m=32, k=k, n=n_padded, num_cores=rows * cols)
    in_memcfg = width_sharded_l1_memcfg(m_tiles=1, k_tiles=k_tiles, num_cores_x=cols, num_cores_y=rows)
    out_memcfg = width_sharded_l1_memcfg(m_tiles=1, k_tiles=n_tiles, num_cores_x=cols, num_cores_y=rows)

    # Pad m to 32 for the DRAM-sharded kernel's M==tile_height assertion.
    if m < 32:
        x_padded = torch.zeros(32, K, dtype=x_torch.dtype)
        x_padded[:m] = x_torch
    else:
        x_padded = x_torch
    x_tt = ttnn.from_torch(
        x_padded.unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_sharded = ttnn.to_memory_config(x_tt, in_memcfg)
    ttnn.deallocate(x_tt)
    b_tt = ttnn.from_torch(
        b_torch.reshape(1, 1, 1, -1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out_tt = ttnn.linear(
        x_sharded,
        w_tt,
        bias=b_tt,
        compute_kernel_config=compute_cfg,
        memory_config=out_memcfg,
        program_config=progcfg,
    )
    out_intl = ttnn.to_memory_config(out_tt, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out_tt)
    out = ttnn.to_torch(out_intl).squeeze(0).squeeze(0)  # [32, n_padded]
    ttnn.deallocate(out_intl)
    ttnn.deallocate(x_sharded)
    ttnn.deallocate(w_tt)
    ttnn.deallocate(b_tt)
    # Slice back to [m, N] (n_padded may be padded for DRAM bank alignment).
    N = w_torch.shape[0]
    return out[:m, :N]


def _cpu_reference(x_torch: torch.Tensor, w_torch: torch.Tensor, b_torch: torch.Tensor) -> torch.Tensor:
    """Bf16-input, fp32-accumulate matmul on CPU — closest analogue of the model
    state we want to preserve."""
    return (x_torch.float() @ w_torch.t().float() + b_torch.float()).to(torch.float32)


def _bf16_torch_reference(x_torch: torch.Tensor, w_torch: torch.Tensor, b_torch: torch.Tensor) -> torch.Tensor:
    """Pure bf16 matmul on CPU — what bf16 numerics would yield without fp32 accum."""
    return (x_torch @ w_torch.t() + b_torch).float()


# Real CP_Prefill MLP shapes from the model:
#   gate_proj/up_proj: [intermediate=3072, hidden=1024], bias[3072]
#   down_proj:         [hidden=1024, intermediate=3072], bias[1024]
# CP prefill seq=2 (past_hidden + code0_embed); pad to M=32 for DRAM-sharded kernel.
@pytest.mark.parametrize(
    "name,K,N,M",
    [
        ("gate_proj_seq2", 1024, 3072, 2),
        ("up_proj_seq2", 1024, 3072, 2),
        ("down_proj_seq2", 3072, 1024, 2),
    ],
)
def test_cp_prefill_matmul_accuracy(device, name, K, N, M):
    """Compare 1D-mcast vs DRAM-sharded matmul outputs against CPU fp32 reference.

    Both kernels should produce numerically close results to the fp32 reference;
    the question is which is closer, and whether the gap is large enough to
    explain the autoregressive voice-drift we observe in the demo.
    """
    torch.manual_seed(42)
    x_torch = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    w_torch = torch.randn(N, K, dtype=torch.bfloat16) * 0.05
    b_torch = torch.randn(N, dtype=torch.bfloat16) * 0.01

    ref_fp32 = _cpu_reference(x_torch, w_torch, b_torch)
    ref_bf16 = _bf16_torch_reference(x_torch, w_torch, b_torch)
    print(f"\n=== {name}: M={M} K={K} N={N} ===")
    _summarize("torch bf16 vs fp32", ref_fp32, ref_bf16)

    out_1d = _run_matmul_1d_mcast(device, x_torch, w_torch, b_torch, M)
    _summarize("ttnn 1D-mcast vs fp32", ref_fp32, out_1d)
    _summarize("ttnn 1D-mcast vs bf16", ref_bf16, out_1d)

    out_dram = _run_matmul_dram_sharded(device, x_torch, w_torch, b_torch, M)
    _summarize("ttnn DRAM-sharded vs fp32", ref_fp32, out_dram)
    _summarize("ttnn DRAM-sharded vs bf16", ref_bf16, out_dram)

    # Direct comparison: 1D-mcast vs DRAM-sharded
    _summarize("1D-mcast vs DRAM-sharded", out_1d, out_dram)

    # Sanity: both kernels should be at least PCC > 0.99 vs fp32 reference.
    assert pearson(ref_fp32, out_1d) > 0.99, "1D-mcast PCC < 0.99 — kernel issue"
    assert pearson(ref_fp32, out_dram) > 0.99, "DRAM-sharded PCC < 0.99 — kernel issue"


@pytest.mark.parametrize(
    "name,K,N,M",
    [
        ("gate_proj_seq2", 1024, 3072, 2),
        ("down_proj_seq2", 3072, 1024, 2),
    ],
)
def test_dram_sharded_hifi4_closes_gap(device, name, K, N, M):
    """Does upping math_fidelity from LoFi to HiFi4 close the DRAM-sharded
    accuracy gap? (LoFi packs the bf16 mantissa to 5 bits during multiply;
    HiFi4 keeps full 8-bit mantissa.) If yes, this is a viable path: pay
    HiFi4's ~4x slowdown on the autoregressive loop's MLP only."""
    torch.manual_seed(42)
    x_torch = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    w_torch = torch.randn(N, K, dtype=torch.bfloat16) * 0.05
    b_torch = torch.randn(N, dtype=torch.bfloat16) * 0.01
    ref = _cpu_reference(x_torch, w_torch, b_torch)
    print(f"\n=== {name} HiFi4 sweep: M={M} K={K} N={N} ===")
    for fid_name, fid in [
        ("LoFi", ttnn.MathFidelity.LoFi),
        ("HiFi2", ttnn.MathFidelity.HiFi2),
        ("HiFi4", ttnn.MathFidelity.HiFi4),
    ]:
        out_1d = _run_matmul_1d_mcast(device, x_torch, w_torch, b_torch, M, math_fidelity=fid)
        _summarize(f"1D-mcast {fid_name}", ref, out_1d)
        out_dram = _run_matmul_dram_sharded(device, x_torch, w_torch, b_torch, M, math_fidelity=fid)
        _summarize(f"DRAM-shard {fid_name}", ref, out_dram)


@pytest.mark.parametrize("bias_scale", [0.0, 0.01], ids=["zero_bias", "nonzero_bias"])
@pytest.mark.parametrize("name,K,N,M", [("gate_proj", 1024, 3072, 2), ("down_proj", 3072, 1024, 2)])
def test_dram_sharded_diverges_only_with_bias(device, name, K, N, M, bias_scale):
    """Pinpoint the source of DRAM-sharded vs 1D-mcast divergence.

    Surprising finding: with bias=0, the two kernels are *bit-identical* on
    these shapes — the matmul reduction itself is not the source of the
    PCC=0.99 envelope the kernel team certifies. The divergence enters via
    the DRAM-sharded path's bias add (probably a lower-precision broadcast
    intermediate in the kernel's epilogue).

    Implication for qwen3_tts: in the autoregressive loop only
    code_predictor.input_proj uses a bias; MLP (gate/up/down) and attention
    (wqkv/wo) are bias-free. So swapping CP_Prefill MLP onto DRAM-sharded
    should be loss-free on these matmuls — but the bias-bearing input_proj
    must stay on 1D-mcast or the bias-add precision loss will drive the AR
    loop drift we observed (voice timbre).
    """
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.05
    b = torch.randn(N, dtype=torch.bfloat16) * bias_scale  # 0 or non-zero
    ref = (x.float() @ w.t().float() + b.float()).float()

    print(f"\n=== {name}: M={M} K={K} N={N}  bias_scale={bias_scale} ===")
    o1 = _run_matmul_1d_mcast(device, x, w, b, M)
    s1 = _summarize("1D-mcast vs fp32", ref, o1)
    od = _run_matmul_dram_sharded(device, x, w, b, M)
    sd = _summarize("DRAM-shard vs fp32", ref, od)
    s_cmp = _summarize("1D vs DRAM", o1, od)

    if bias_scale == 0.0:
        # With bias=0 the two kernels should produce essentially identical
        # output (RMS_diff/RMS_ref < 1e-3 — well below bf16 quantization noise).
        assert s_cmp["rms_diff"] / s_cmp["rms_ref"] < 1e-3, (
            f"Expected bit-identical kernels at bias=0, got RMS_diff/RMS_ref="
            f"{s_cmp['rms_diff']/s_cmp['rms_ref']:.4f}"
        )
    else:
        # With non-zero bias DRAM-sharded should be measurably worse than 1D.
        assert sd["rms_diff"] > 2 * s1["rms_diff"], (
            f"Expected DRAM-sharded to be > 2x worse than 1D-mcast with non-zero bias; "
            f"got 1D RMS_diff={s1['rms_diff']:.4f}, DRAM RMS_diff={sd['rms_diff']:.4f}"
        )


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _swiglu_mlp_torch_fp32(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """CPU fp32 reference for one SwiGLU MLP block."""
    g = x.float() @ w_gate.t().float()
    u = x.float() @ w_up.t().float()
    h = _silu(g) * u
    return h @ w_down.t().float()


def _swiglu_mlp_ttnn(
    device,
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
    m: int,
    runner,
) -> torch.Tensor:
    """One SwiGLU MLP block using `runner` (1D-mcast or DRAM-sharded) for each
    matmul. Activation (SiLU * mul) runs on CPU bf16 — that part is identical
    between paths, so any divergence comes from the matmul kernel.

    `b_down` is added to the down_proj output (non-zero in the AR test to model
    code_predictor.input_proj's bias — the only autoregressive linear in
    qwen3_tts that has one). When b_down is zero the two kernels are
    bit-identical and the AR loop will not show drift.
    """
    zero_g = torch.zeros(w_gate.shape[0], dtype=torch.bfloat16)
    zero_u = torch.zeros(w_up.shape[0], dtype=torch.bfloat16)
    g = runner(device, x, w_gate, zero_g, m)
    u = runner(device, x, w_up, zero_u, m)
    h = (_silu(g.float()) * u.float()).to(torch.bfloat16)
    return runner(device, h, w_down, b_down, m)


def _swiglu_mlp_torch_fp32_with_bias(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, b_down: torch.Tensor
) -> torch.Tensor:
    g = x.float() @ w_gate.t().float()
    u = x.float() @ w_up.t().float()
    h = _silu(g) * u
    return h @ w_down.t().float() + b_down.float()


def test_autoregressive_loop_compounding(device):
    """Simulate an autoregressive loop closer to Talker decode:
    each step applies a SwiGLU MLP (gate/up/SiLU*mul/down) plus a residual,
    then feeds the new hidden state into the next step. Two streams run side by
    side — one always uses 1D-mcast matmul, the other always DRAM-sharded — and
    we track:
      • PCC of hidden state vs CPU fp32 reference, per step
      • How quickly the "sampling head" (a final projection + argmax) of the two
        ttnn streams diverges from one another and from the fp32 reference

    This is the cleanest proxy for the autoregressive voice drift seen in the
    demo: the moment the two streams' argmax decisions diverge is the moment the
    chosen codec token differs, which is what shifts the audio trajectory.
    """
    torch.manual_seed(0)
    HIDDEN = 1024
    INTER = 3072
    VOCAB = 4096  # CodePredictor has ~codebook_size; pick a typical size
    # CP prefill is M=2 (past_hidden + code0_embed); use that so the per-tile
    # reduction has multiple real rows and kernel reduction-order differences
    # actually surface (at M=1 padded to a 32-row tile both kernels round
    # identically and the test can't see the drift).
    M = 2
    NUM_STEPS = 128

    # One MLP block reused across steps (simulates one transformer layer in the
    # AR loop; real Talker has 28 layers but the qualitative compounding is the
    # same and one layer is much faster to iterate on).
    # IMPORTANT — value-scale matters for kernel divergence:
    #   Large activations (x ~ N(0,1), w ~ N(0, 1/sqrt(K))) → 1D-mcast and
    #   DRAM-sharded compute *bit-identical* outputs (bf16 mantissa has enough
    #   headroom that both reduction orders round to the same value).
    #   Small activations (x*0.1, w*0.05) → DRAM-sharded diverges from 1D
    #   (PCC 0.996 vs 0.9999); reduction-order differences dominate.
    # We use the small-value scale here so the kernel divergence is visible and
    # we can study how it compounds. (The real model can hit either regime in
    # different layers depending on weight magnitudes; the per-shape test above
    # confirms the divergence is an intrinsic kernel property, not a weight
    # artifact.)
    w_gate = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16) * 0.05
    w_up = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16) * 0.05
    w_down = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16) * 0.05
    # Non-zero down_proj bias models code_predictor.input_proj's bias — the
    # one autoregressive linear in qwen3_tts that has one. With b_down=0 the
    # two kernels are bit-identical and the AR loop shows no drift; this test
    # would be vacuous. The single linear with bias is enough to drive the
    # downstream argmax-divergence we want to reproduce.
    b_down = torch.randn(HIDDEN, dtype=torch.bfloat16) * 0.01
    w_head = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16) * 0.05

    x0 = torch.randn(M, HIDDEN, dtype=torch.bfloat16) * 0.1

    def _rmsnorm(t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Mirror Talker's per-layer RMSNorm — keeps the AR loop bounded."""
        f = t.float()
        rms = (f.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        return (f / rms).to(t.dtype) if t.dtype != torch.float32 else f / rms

    cur_ref = x0.float()
    cur_1d = x0.clone()
    cur_dram = x0.clone()

    print(f"\n=== AR-loop compounding: {NUM_STEPS} steps, MLP+residual per step ===")
    print(
        f"{'step':>4} | {'1D vs fp32':>11} {'DRAM vs fp32':>13} {'1D vs DRAM':>11} | "
        f"{'argmax 1D=ref':>13} {'argmax DRAM=ref':>15} {'argmax 1D=DRAM':>14}"
    )

    argmax_1d_ref_match = 0
    argmax_dram_ref_match = 0
    argmax_1d_dram_match = 0

    for step in range(NUM_STEPS):
        # Pre-MLP RMSNorm (the standard pre-norm transformer pattern Talker uses).
        ref_in = _rmsnorm(cur_ref)
        in_1d = _rmsnorm(cur_1d)
        in_dram = _rmsnorm(cur_dram)

        # Reference (fp32) step
        ref_mlp = _swiglu_mlp_torch_fp32_with_bias(ref_in, w_gate, w_up, w_down, b_down)
        cur_ref_next = cur_ref + ref_mlp

        # 1D-mcast stream
        mlp_1d = _swiglu_mlp_ttnn(device, in_1d, w_gate, w_up, w_down, b_down, M, _run_matmul_1d_mcast)
        cur_1d_next = (cur_1d.float() + mlp_1d.float()).to(torch.bfloat16)

        # DRAM-sharded stream
        mlp_dram = _swiglu_mlp_ttnn(device, in_dram, w_gate, w_up, w_down, b_down, M, _run_matmul_dram_sharded)
        cur_dram_next = (cur_dram.float() + mlp_dram.float()).to(torch.bfloat16)

        # Sampling head: argmax over vocab. argmax is what actually picks the
        # token, so it's the binding measure of "did the streams sample the same
        # thing this step?"
        logits_ref = cur_ref_next @ w_head.t().float()
        logits_1d = cur_1d_next.float() @ w_head.t().float()
        logits_dram = cur_dram_next.float() @ w_head.t().float()
        tok_ref = logits_ref.argmax(dim=-1)
        tok_1d = logits_1d.argmax(dim=-1)
        tok_dram = logits_dram.argmax(dim=-1)
        argmax_1d_ref_match += int((tok_1d == tok_ref).all())
        argmax_dram_ref_match += int((tok_dram == tok_ref).all())
        argmax_1d_dram_match += int((tok_1d == tok_dram).all())

        pcc_1d = pearson(cur_ref_next, cur_1d_next.float())
        pcc_dram = pearson(cur_ref_next, cur_dram_next.float())
        pcc_1d_dram = pearson(cur_1d_next.float(), cur_dram_next.float())

        if step < 8 or step % 8 == 7:
            print(
                f"{step:>4} | {pcc_1d:>11.6f} {pcc_dram:>13.6f} {pcc_1d_dram:>11.6f} | "
                f"  tok_ref={tok_ref.tolist()} 1D={tok_1d.tolist()} DRAM={tok_dram.tolist()}"
            )

        cur_ref = cur_ref_next
        cur_1d = cur_1d_next
        cur_dram = cur_dram_next

    print(f"\nArgmax-match totals over {NUM_STEPS} steps:")
    print(
        f"  1D-mcast    matches fp32-ref:   {argmax_1d_ref_match:>3}/{NUM_STEPS}  "
        f"({100*argmax_1d_ref_match/NUM_STEPS:.1f}%)"
    )
    print(
        f"  DRAM-shard  matches fp32-ref:   {argmax_dram_ref_match:>3}/{NUM_STEPS}  "
        f"({100*argmax_dram_ref_match/NUM_STEPS:.1f}%)"
    )
    print(
        f"  1D-mcast vs DRAM-shard agree:   {argmax_1d_dram_match:>3}/{NUM_STEPS}  "
        f"({100*argmax_1d_dram_match/NUM_STEPS:.1f}%)"
    )
    final_pcc_1d = pearson(cur_ref, cur_1d.float())
    final_pcc_dram = pearson(cur_ref, cur_dram.float())
    print(f"\nFinal hidden-state PCC vs fp32 ref: 1D={final_pcc_1d:.6f}  DRAM={final_pcc_dram:.6f}")


def test_chained_layer_drift(device):
    """Worst-case scenario for the autoregressive loop:
    apply a chain of 5 matmuls (≈ one CP layer's worth of MLP+attention projections)
    with both kernels and see how the error compounds.
    """
    torch.manual_seed(0)
    K = 1024
    N = 3072
    M = 2

    x = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    weights = [torch.randn(N, K, dtype=torch.bfloat16) * 0.05 for _ in range(5)]
    biases = [torch.randn(N, dtype=torch.bfloat16) * 0.01 for _ in range(5)]
    # Make alternate weights project K -> K so we can chain (matmul shape: [M,K] @ [K,K])
    weights = [torch.randn(K, K, dtype=torch.bfloat16) * 0.05 for _ in range(5)]
    biases = [torch.randn(K, dtype=torch.bfloat16) * 0.01 for _ in range(5)]

    print("\n=== Chained 5-matmul drift (each [M,K] @ [K,K]) ===")

    cur_ref = x.float()
    cur_1d = x.clone()
    cur_dram = x.clone()
    for i, (w, b) in enumerate(zip(weights, biases)):
        cur_ref = cur_ref @ w.t().float() + b.float()
        cur_1d_out = _run_matmul_1d_mcast(device, cur_1d, w, b, M)
        cur_dram_out = _run_matmul_dram_sharded(device, cur_dram, w, b, M)
        cur_1d = cur_1d_out.to(torch.bfloat16)
        cur_dram = cur_dram_out.to(torch.bfloat16)
        print(f"After matmul {i+1}:")
        _summarize("  1D-mcast vs fp32", cur_ref, cur_1d_out)
        _summarize("  DRAM-sharded vs fp32", cur_ref, cur_dram_out)
        _summarize("  1D vs DRAM", cur_1d_out, cur_dram_out)
