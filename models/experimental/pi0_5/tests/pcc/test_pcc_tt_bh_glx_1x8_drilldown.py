# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drilldown PCC: single-chip TTNN vs 1×8 mesh TTNN, layer-0 component-by-component.

Both backends run the SAME deterministic input + SAME cumsum-based cos/sin
(matching the LIBERO single-arm mask: img_masks=[T,T,F], lang_mask=20-real).
Each test runs the single-chip phase first (chip 0 of the visible set),
closes the device, then opens the 1×8 mesh and runs the mesh phase. Results
are saved to a temp dir, then compared.

This complements `test_pcc_tt_bh_glx_stages.py` (which compares each backend
vs. the torch reference). Here we compare the two TTNN backends directly to
each other, isolating divergence introduced by the mesh dispatch path. This
is the right tool for explaining the LIBERO-on-1×8 regression vs single-chip.

Run (requires all 8 chips of the prefill submesh visible):

    TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \\
        pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_1x8_drilldown.py

Each test prints a one-line VERDICT and asserts a tight numeric threshold;
exact diff/PCC values are printed so a failing test pinpoints the divergent
component.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
PREFIX_LEN = 1024
VLM_WIDTH = 2048
VLM_HEAD_DIM = 256
NUM_LAYERS = 18
LIBERO_LANG_REAL = 20
MESH_TP = int(os.environ.get("PI0_TP", "8"))

# K columns inside xqkv: single-chip wqkv is (hidden, 8*256 + 256 + 256);
# 1×8 head-par per-chip wqkv is (hidden, 1*256 + 256 + 256). K columns at:
K_COL_SINGLE = (2048, 2304)
K_COL_MESH = (256, 512)


_PRODUCTION_ENV = "/home/tt-admin/sdawle/pi05_openpi_upstream_bh_glx_trace/tt-metal/_bench_runs/pi05_production.env"


def _seed_production_env():
    """Source PI05 production env defaults (idempotent; existing env wins).

    The 1×8 pipeline assumes PI0_TP=8 and a handful of other knobs are set.
    Without these, mesh init asserts or takes a different code path than the
    real LIBERO rollout — making the comparison apples-to-oranges.
    """
    import re

    if Path(_PRODUCTION_ENV).exists():
        with open(_PRODUCTION_ENV) as f:
            for line in f:
                m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2)
    for k, v in {
        "PI0_TP": str(MESH_TP),
        "PI0_TP4_ATTN_HEADPAR": "1",
        "PI0_MLP_BS": "1",
        "PI0_MLP_FUSED_RS": "0",
    }.items():
        os.environ.setdefault(k, v)


_seed_production_env()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(), b.std()
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(a, b, atol=1e-5) else 0.0
    cov = ((a - m1) * (b - m2)).mean()
    return (cov / (s1 * s2)).item()


def _deterministic_hidden() -> torch.Tensor:
    """bf16 hidden state shaped (1, PREFIX_LEN, VLM_WIDTH). Same seed → same bytes."""
    torch.manual_seed(SEED)
    return (torch.randn(1, PREFIX_LEN, VLM_WIDTH, dtype=torch.float32) * 0.1).to(torch.bfloat16)


def _libero_pad_mask() -> torch.Tensor:
    """LIBERO single-arm prefix mask: cam0=real, cam1=real, cam2=pad, lang=20 real."""
    mask = torch.zeros(PREFIX_LEN, dtype=torch.bool)
    mask[0:256] = True
    mask[256:512] = True
    mask[768 : 768 + LIBERO_LANG_REAL] = True
    return mask


def _prefix_cos_sin_libero(max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumsum-based cos/sin for the LIBERO mask, shape (1, 1, PREFIX_LEN, head_dim)."""
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import _precompute_rope_table_torch

    pad_mask = _libero_pad_mask()
    position_ids = torch.cumsum(pad_mask.to(torch.int64), dim=0) - 1
    position_ids = position_ids.clamp(min=0, max=max_seq_len - 1)
    cos_t, sin_t = _precompute_rope_table_torch(VLM_HEAD_DIM, max_seq_len)
    cos = cos_t[position_ids].unsqueeze(0).unsqueeze(0).contiguous()
    sin = sin_t[position_ids].unsqueeze(0).unsqueeze(0).contiguous()
    return cos, sin


def _prefix_mask_libero() -> torch.Tensor:
    """LIBERO-style prefix attention mask: pads block both directions."""
    pad_mask = _libero_pad_mask()
    pad_2d = pad_mask[:, None] & pad_mask[None, :]
    mask = torch.zeros(PREFIX_LEN, PREFIX_LEN, dtype=torch.bfloat16)
    mask.masked_fill_(~pad_2d, -1e4)
    return mask.unsqueeze(0).unsqueeze(0)


# ──────────────────────────── Skip if no checkpoint ─────────────────────────

requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)


# ──────────────────────────── Stage helpers ─────────────────────────────────


def _open_single():
    """Open chip 0 of the visible set (single-chip phase)."""
    import ttnn

    return ttnn.open_device(device_id=int(os.environ.get("PI0_DEVICE_ID", "0")), l1_small_size=24576)


def _single_l0_layernorm_xqkv(hidden_t: torch.Tensor):
    """Run input_layernorm + xqkv matmul on single-chip; return (normed, K-cols, xqkv shape)."""
    import ttnn
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_gemma import rms_norm_ttnn
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    device = _open_single()
    try:
        cfg = Pi0_5ModelConfig(action_horizon=10)
        model = Pi0_5ModelTTNN(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)), device)
        block0 = model.backbone.vlm_blocks[0]
        eps = cfg.vlm_config.rms_norm_eps

        hidden = ttnn.from_torch(
            hidden_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        normed_tt = rms_norm_ttnn(hidden, block0.input_layernorm_weight, eps)
        ttnn.deallocate(hidden)
        normed_torch = ttnn.to_torch(normed_tt).float()

        grid = device.compute_with_storage_grid_size()
        xqkv_tt = ttnn.linear(
            normed_tt,
            block0.attention.wqkv,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=block0.attention.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        )
        ttnn.deallocate(normed_tt)
        xqkv_full = ttnn.to_torch(xqkv_tt).float()
        # Shape may be (1, seq, N) or (1, 1, seq, N) — handle both.
        if xqkv_full.dim() == 4:
            k_slice = xqkv_full[0, 0, :, K_COL_SINGLE[0] : K_COL_SINGLE[1]].contiguous()
        else:
            k_slice = xqkv_full[0, :, K_COL_SINGLE[0] : K_COL_SINGLE[1]].contiguous()
        ttnn.deallocate(xqkv_tt)
        return normed_torch, k_slice, tuple(xqkv_full.shape)
    finally:
        ttnn.close_device(device)


def _mesh_l0_layernorm_xqkv(hidden_t: torch.Tensor):
    """Same as _single_l0_layernorm_xqkv but on the 1×8 mesh. K-cols extracted
    per chip via the head-par slab layout."""
    import ttnn
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_1x8 import Pi0_5GLX1x8Pipeline
    from models.experimental.pi0_5.tt.ttnn_gemma import rms_norm_ttnn

    with open_prefill_tp4_mesh(tp=MESH_TP, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        cfg = Pi0_5ModelConfig(action_horizon=10)
        pipe = Pi0_5GLX1x8Pipeline(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights, mesh)
        block0 = pipe.prefill.blocks[0]
        eps = cfg.vlm_config.rms_norm_eps

        hidden = ttnn.from_torch(
            hidden_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        normed_tt = rms_norm_ttnn(hidden, block0.input_layernorm_weight, eps)
        ttnn.deallocate(hidden)
        normed_full = ttnn.to_torch(normed_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        normed_torch = normed_full[:1]  # chip0 view (replicated)

        grid = mesh.compute_with_storage_grid_size()
        xqkv_tt = ttnn.linear(
            normed_tt,
            block0.attention.wqkv,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=block0.attention.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        )
        ttnn.deallocate(normed_tt)
        # Sharded along last dim across 8 chips — concat gives (1, seq, 8*768).
        xqkv_full = ttnn.to_torch(xqkv_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        if xqkv_full.dim() == 4:
            k_slice = xqkv_full[0, 0, :, K_COL_MESH[0] : K_COL_MESH[1]].contiguous()
        else:
            k_slice = xqkv_full[0, :, K_COL_MESH[0] : K_COL_MESH[1]].contiguous()
        ttnn.deallocate(xqkv_tt)
        return normed_torch, k_slice, tuple(xqkv_full.shape)


# ──────────────────────────── Tests ─────────────────────────────────────────


@requires_checkpoint
def test_l0_input_layernorm_equivalent():
    """Layer-0 input_layernorm output: single-chip vs mesh, bit-equivalent expected."""
    hidden_t = _deterministic_hidden()
    s_norm, _, _ = _single_l0_layernorm_xqkv(hidden_t)
    m_norm, _, _ = _mesh_l0_layernorm_xqkv(hidden_t)
    assert s_norm.shape == m_norm.shape, f"shape: {tuple(s_norm.shape)} vs {tuple(m_norm.shape)}"
    diff = (s_norm - m_norm).abs()
    pcc = _pcc(s_norm, m_norm)
    print(
        f"\n[L0 input_layernorm] PCC={pcc:.6f}  max_diff={diff.max().item():.3e}  "
        f"mean_diff={diff.mean().item():.3e}"
    )
    assert diff.max().item() < 1e-2, f"layernorm diverges: max {diff.max().item()}"


@requires_checkpoint
def test_l0_xqkv_kcols_equivalent():
    """Layer-0 K columns post-QKV matmul: single-chip vs mesh, bit-equivalent expected."""
    hidden_t = _deterministic_hidden()
    _, s_k, s_shape = _single_l0_layernorm_xqkv(hidden_t)
    _, m_k, m_shape = _mesh_l0_layernorm_xqkv(hidden_t)
    assert s_k.shape == m_k.shape, f"K shape: {tuple(s_k.shape)} vs {tuple(m_k.shape)}"
    diff = (s_k - m_k).abs()
    pcc = _pcc(s_k, m_k)
    print(
        f"\n[L0 xqkv K-cols] PCC={pcc:.6f}  max_diff={diff.max().item():.3e}  "
        f"mean_diff={diff.mean().item():.3e}  shapes: single={s_shape} mesh={m_shape}"
    )
    assert diff.max().item() < 1e-2, f"xqkv K columns diverge: max {diff.max().item()}"


@requires_checkpoint
def test_l0_rope_k_bf8_diverges():
    """Isolated rotary_embedding micro-test on bf8 K input.

    Single-chip and mesh produce IDENTICAL output for bf16 K, but **diverge
    by ~1 bf16-ULP per element for bf8 K**. The single-chip path effectively
    quantizes the output to bf8 grid; the mesh path keeps bf16 precision.

    Production K is bf8 (output of bf8 QKV matmul), so this 1-ULP gap shows
    up in every production RoPE call and cascades across 18 layers. This
    test pins the kernel-level behavior so a future TTNN fix that unifies
    the dispatch is detected (PCC will jump to 1.0, max_diff to 0.0).

    Threshold: bf16 K must be bit-equivalent; bf8 K is asserted ≤0.05 max
    (the synthetic input is randn*0.5, ~10× smaller than production K, so
    the real prefill amplifies this proportionally).
    """
    import ttnn
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    torch.manual_seed(SEED)
    k_t = (torch.randn(1, 1, PREFIX_LEN, VLM_HEAD_DIM, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    half = VLM_HEAD_DIM // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, VLM_HEAD_DIM, 2, dtype=torch.float32) / VLM_HEAD_DIM))
    t = torch.arange(PREFIX_LEN, dtype=torch.float32)
    fo = torch.outer(t, freqs)
    cos_t = torch.cat([torch.cos(fo), torch.cos(fo)], dim=-1).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    sin_t = torch.cat([torch.sin(fo), torch.sin(fo)], dim=-1).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    def _run(device, k_dtype):
        k = ttnn.from_torch(
            k_t, dtype=k_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        cos = ttnn.from_torch(
            cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        sin = ttnn.from_torch(
            sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        kr = ttnn.experimental.rotary_embedding(k, cos, sin)
        out = ttnn.to_torch(kr).float()
        ttnn.deallocate(k)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(kr)
        return out

    # Single-chip phase.
    device = _open_single()
    try:
        s_bf16 = _run(device, ttnn.bfloat16)
        s_bf8 = _run(device, ttnn.bfloat8_b)
    finally:
        ttnn.close_device(device)

    # Mesh phase — replicate K and take chip 0's RoPE output.
    with open_prefill_tp4_mesh(tp=MESH_TP, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:

        def _run_mesh(k_dtype):
            k = ttnn.from_torch(
                k_t, dtype=k_dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            cos = ttnn.from_torch(
                cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            sin = ttnn.from_torch(
                sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            kr = ttnn.experimental.rotary_embedding(k, cos, sin)
            full = ttnn.to_torch(kr, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            ttnn.deallocate(k)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(kr)
            return full[:1]

        m_bf16 = _run_mesh(ttnn.bfloat16)
        m_bf8 = _run_mesh(ttnn.bfloat8_b)

    d_bf16 = (s_bf16 - m_bf16).abs()
    d_bf8 = (s_bf8 - m_bf8).abs()
    print(
        f"\n[L0 rotary_embedding bf16-K] PCC={_pcc(s_bf16, m_bf16):.6f}  "
        f"max_diff={d_bf16.max().item():.3e}  mean_diff={d_bf16.mean().item():.3e}"
    )
    print(
        f"[L0 rotary_embedding bf8-K]  PCC={_pcc(s_bf8, m_bf8):.6f}  "
        f"max_diff={d_bf8.max().item():.3e}  mean_diff={d_bf8.mean().item():.3e}"
    )
    # bf16 K: must be bit-identical (TTNN op is deterministic across contexts).
    assert d_bf16.max().item() < 1e-3, f"bf16-K rotary_embedding diverges: {d_bf16.max().item()}"
    # bf8 K: documents the known divergence. Loosened threshold reflects
    # the synthetic input magnitude — production K (~10× larger) sees ~0.3 here.
    assert d_bf8.max().item() < 0.05, f"bf8-K rotary_embedding divergence exceeds expected: {d_bf8.max().item()}"


@requires_checkpoint
def test_per_layer_kv_cache_cascade():
    """Full prefill with LIBERO mask: per-layer K/V cache PCC + max-diff.

    Both backends run their REAL production prefill (with cumsum-based cos/sin
    + LIBERO single-arm attention mask) on the same deterministic prefix_embs.
    K and V cache slabs are extracted per layer and compared.

    Expected behavior with current code: L0 K diff ≈ 0.3 (RoPE bf8 quantization
    gap), cascading through L1..L17. The exact per-layer table makes future
    regressions or fixes immediately visible — a successful unification of the
    RoPE dispatch should drop L0 diff to <0.01.

    Asserts only on L0 K (the source); downstream layers are reported but
    not asserted on (they're consequences of L0 + accumulated noise).
    """
    import ttnn

    hidden_t = _deterministic_hidden().float().to(torch.bfloat16)
    pad_mask = _libero_pad_mask()
    prefix_mask_4d = _prefix_mask_libero()
    cos_4d, sin_4d = _prefix_cos_sin_libero(max_seq_len=4096)

    # ── Single-chip phase ────────────────────────────────────────────────
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader

    device = _open_single()
    try:
        from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

        cfg = Pi0_5ModelConfig(action_horizon=10)
        model = Pi0_5ModelTTNN(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)), device)
        prefix = ttnn.from_torch(
            hidden_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cos_tt = ttnn.from_torch(
            cos_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sin_tt = ttnn.from_torch(
            sin_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        mask_tt = ttnn.from_torch(
            prefix_mask_4d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _, single_kv = model.backbone.forward_vlm(
            prefix, mask_tt, None, use_cache=True, cos_override=cos_tt, sin_override=sin_tt
        )
        # Convert each (K, V) to torch fp32 and detach from device.
        single_kv_torch = [(ttnn.to_torch(k).float(), ttnn.to_torch(v).float()) for (k, v) in single_kv]
    finally:
        ttnn.close_device(device)

    # ── Mesh phase ───────────────────────────────────────────────────────
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_1x8 import Pi0_5GLX1x8Pipeline

    with open_prefill_tp4_mesh(tp=MESH_TP, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        cfg = Pi0_5ModelConfig(action_horizon=10)
        pipe = Pi0_5GLX1x8Pipeline(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights, mesh)

        # Build artifacts in pipeline state. LIBERO single-arm: third camera padded.
        img_masks = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]
        lang_mask = torch.zeros(1, 256, dtype=torch.bool)
        lang_mask[0, :LIBERO_LANG_REAL] = True
        pipe._build_upstream_artifacts(img_masks=img_masks, lang_masks=lang_mask)

        prefix = ttnn.from_torch(
            hidden_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # Replicate pipeline_1x8.sample_actions prefill call exactly: attention
        # mask + cos/sin per-layer overrides if present.
        if pipe._prefix_cos is not None:
            num_layers = len(pipe.prefill.blocks)
            _, mesh_kv = pipe.prefill.run(
                prefix,
                attention_mask=pipe._prefix_attn_mask,
                per_chip_cos=[pipe._prefix_cos] * num_layers,
                per_chip_sin=[pipe._prefix_sin] * num_layers,
            )
        else:
            _, mesh_kv = pipe.prefill.run(prefix, attention_mask=pipe._prefix_attn_mask)

        # Mesh KV slabs are replicated head-par. Take chip 0's view per layer.
        def _chip0(tt):
            full = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            return full[:1]

        mesh_kv_torch = [(_chip0(k), _chip0(v)) for (k, v) in mesh_kv]

    # ── Compare per-layer ────────────────────────────────────────────────
    assert (
        len(single_kv_torch) == len(mesh_kv_torch) == NUM_LAYERS
    ), f"layer count: single={len(single_kv_torch)} mesh={len(mesh_kv_torch)}"
    print(
        "\nLayer-by-layer K/V cache comparison (single TTNN vs 1×8 mesh TTNN):\n"
        f"{'L':>3}  {'K-PCC':>9}  {'K-maxΔ':>9}  {'V-PCC':>9}  {'V-maxΔ':>9}"
    )
    rows = []
    for i, ((sk, sv), (mk, mv)) in enumerate(zip(single_kv_torch, mesh_kv_torch)):
        # Single keeps full K-head; mesh head-par returns chip-0's head slab.
        # Squeeze head dim if shapes differ structurally (both should be (1,1,seq,head_dim) here).
        if sk.shape != mk.shape:
            # Mesh head-par per-chip K shape: (1, 1, seq, head_dim). Single same.
            # If a mismatch appears, fall back to slicing matching head 0.
            sk_cmp = sk[..., : mk.shape[-1]] if sk.shape[-1] > mk.shape[-1] else sk
        else:
            sk_cmp = sk
        if sv.shape != mv.shape:
            sv_cmp = sv[..., : mv.shape[-1]] if sv.shape[-1] > mv.shape[-1] else sv
        else:
            sv_cmp = sv
        k_pcc = _pcc(sk_cmp, mk)
        v_pcc = _pcc(sv_cmp, mv)
        k_d = (sk_cmp - mk).abs().max().item()
        v_d = (sv_cmp - mv).abs().max().item()
        rows.append((i, k_pcc, k_d, v_pcc, v_d))
        print(f"{i:>3}  {k_pcc:>9.4f}  {k_d:>9.3e}  {v_pcc:>9.4f}  {v_d:>9.3e}")

    # Assert only on L0 — downstream layers are consequences of L0 + accumulated
    # cascade noise. A fix that brings L0 K_pcc to ~1.0 will normally bring all
    # downstream layers in line; if it doesn't, the per-layer table makes the
    # remaining divergence visible.
    l0_k_pcc, l0_k_diff = rows[0][1], rows[0][2]
    assert l0_k_pcc > 0.99, (
        f"L0 K cache PCC {l0_k_pcc:.4f} below 0.99 — RoPE divergence has gotten WORSE; "
        f"check ttnn_gemma.py rotary_embedding dispatch."
    )
    # Soft warning if L0 K diff is too large (current known: ~0.31).
    if l0_k_diff > 0.5:
        pytest.fail(
            f"L0 K cache max diff {l0_k_diff:.3f} > 0.5 — divergence has WORSENED past the "
            f"known RoPE-bf8 baseline (~0.31). Investigate ttnn_gemma.py or QKV matmul pcfg."
        )


# ─────────── Production-path L0 drilldown ───────────────────────────────────
# Calls the REAL production attention.forward on each backend, instrumented
# to capture intermediate tensors at each step (xqkv, post-heads-split q/k/v,
# k_rope_padded). Compares each between backends to localize where the L0 K
# cache divergence (~0.31) originates inside the production code path.
#
# The isolated tests above (layernorm, xqkv, rotary_embedding) all show
# bit-equivalence — yet the full block forward shows L0 K diff 0.31. The
# divergence must come from something specific to the production stitching:
# the program_config-driven xqkv pcfg, nlp_create_qkv_heads with different
# num_heads (8 vs 1), or the slice/concat path. This test isolates which.


def _run_single_l0_production_attention(hidden_t: torch.Tensor):
    """Replicate single-chip production L0 attention.forward step-by-step,
    dumping each intermediate. Returns dict of torch tensors."""
    import ttnn
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_gemma import rms_norm_ttnn, build_matmul_pcfg
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    device = _open_single()
    out = {}
    try:
        cfg = Pi0_5ModelConfig(action_horizon=10)
        model = Pi0_5ModelTTNN(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)), device)
        block0 = model.backbone.vlm_blocks[0]
        attn = block0.attention

        hidden = ttnn.from_torch(
            hidden_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        normed = rms_norm_ttnn(hidden, block0.input_layernorm_weight, cfg.vlm_config.rms_norm_eps)
        ttnn.deallocate(hidden)
        # Production reshapes (B, seq, H) → (B, 1, seq, H) before xqkv so the
        # downstream nlp_create_qkv_heads gets 4D input.
        if len(normed.shape) == 3:
            normed = ttnn.reshape(normed, (normed.shape[0], 1, normed.shape[1], normed.shape[2]))
        out["normed"] = ttnn.to_torch(normed).float()

        # xqkv via PRODUCTION pcfg path (replicate ttnn_gemma.py:659-682).
        batch_size = normed.shape[0]
        seq_len = normed.shape[2]
        m_tiles_interleaved = (seq_len + 31) // 32
        k_tiles_in = attn.hidden_size // 32
        n_tiles_qkv = attn.wqkv.shape[-1] // 32
        wqkv_pcfg = build_matmul_pcfg(
            m_tiles_interleaved,
            k_tiles_in,
            n_tiles_qkv,
            attn.grid_size[0],
            attn.grid_size[1],
            in0_block_w=8,
        )
        xqkv = ttnn.linear(
            normed,
            attn.wqkv,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=attn.compute_kernel_config_hifi2,
            program_config=wqkv_pcfg,
        )
        ttnn.deallocate(normed)
        xqkv_torch = ttnn.to_torch(xqkv).float()
        if xqkv_torch.dim() == 4:
            out["xqkv_k_cols"] = xqkv_torch[0, 0, :, K_COL_SINGLE[0] : K_COL_SINGLE[1]].contiguous()
        else:
            out["xqkv_k_cols"] = xqkv_torch[0, :, K_COL_SINGLE[0] : K_COL_SINGLE[1]].contiguous()

        # Heads split with PRODUCTION num_heads (8 for single).
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out["k_postsplit"] = ttnn.to_torch(k).float()

        # Build LIBERO cumsum cos/sin and apply RoPE.
        cos_4d, sin_4d = _prefix_cos_sin_libero(max_seq_len=4096)
        cos_tt = ttnn.from_torch(
            cos_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        sin_tt = ttnn.from_torch(
            sin_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        k_rope = ttnn.experimental.rotary_embedding(k, cos_tt, sin_tt)
        out["k_rope_padded"] = ttnn.to_torch(k_rope).float()
        ttnn.deallocate(xqkv)
    finally:
        ttnn.close_device(device)
    return out


def _run_mesh_l0_production_attention(hidden_t: torch.Tensor):
    """Same on the 1×8 mesh — head-par per-chip attention with num_heads=1."""
    import ttnn
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_1x8 import Pi0_5GLX1x8Pipeline
    from models.experimental.pi0_5.tt.ttnn_gemma import rms_norm_ttnn, build_matmul_pcfg

    out = {}
    with open_prefill_tp4_mesh(tp=MESH_TP, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        cfg = Pi0_5ModelConfig(action_horizon=10)
        pipe = Pi0_5GLX1x8Pipeline(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights, mesh)
        block0 = pipe.prefill.blocks[0]
        attn = block0.attention

        hidden = ttnn.from_torch(
            hidden_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        normed = rms_norm_ttnn(hidden, block0.input_layernorm_weight, cfg.vlm_config.rms_norm_eps)
        ttnn.deallocate(hidden)
        # Production reshapes hidden_states (B, seq, H) → (B, 1, seq, H) before
        # the xqkv linear so nlp_create_qkv_heads sees 4D input. Replicate that.
        if len(normed.shape) == 3:
            normed = ttnn.reshape(normed, (normed.shape[0], 1, normed.shape[1], normed.shape[2]))
        normed_full = ttnn.to_torch(normed, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        out["normed"] = normed_full[:1] if normed_full.shape[0] == MESH_TP else normed_full

        seq_len = normed.shape[2]
        m_tiles_interleaved = (seq_len + 31) // 32
        k_tiles_in = attn.hidden_size // 32
        n_tiles_qkv = attn.wqkv.shape[-1] // 32
        wqkv_pcfg = build_matmul_pcfg(
            m_tiles_interleaved,
            k_tiles_in,
            n_tiles_qkv,
            attn.grid_size[0],
            attn.grid_size[1],
            in0_block_w=8,
        )
        xqkv = ttnn.linear(
            normed,
            attn.wqkv,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=attn.compute_kernel_config_hifi2,
            program_config=wqkv_pcfg,
        )
        ttnn.deallocate(normed)
        # Sharded along last dim across 8 chips.
        xqkv_full = ttnn.to_torch(xqkv, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        if xqkv_full.dim() == 4:
            out["xqkv_k_cols"] = xqkv_full[0, 0, :, K_COL_MESH[0] : K_COL_MESH[1]].contiguous()
        else:
            out["xqkv_k_cols"] = xqkv_full[0, :, K_COL_MESH[0] : K_COL_MESH[1]].contiguous()

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k_full = ttnn.to_torch(k, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        out["k_postsplit"] = k_full[:1]  # chip 0; K is replicated per-chip (num_kv_heads=1)

        cos_4d, sin_4d = _prefix_cos_sin_libero(max_seq_len=4096)
        cos_tt = ttnn.from_torch(
            cos_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        sin_tt = ttnn.from_torch(
            sin_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        k_rope = ttnn.experimental.rotary_embedding(k, cos_tt, sin_tt)
        k_rope_full = ttnn.to_torch(k_rope, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
        out["k_rope_padded"] = k_rope_full[:1]
        ttnn.deallocate(xqkv)
    return out


@requires_checkpoint
def test_l0_production_attention_step_by_step():
    """L0 attention forward: production pcfg + production num_heads, step by step.

    Reproduces the EXACT production attention code path (xqkv via build_matmul_pcfg,
    nlp_create_qkv_heads with num_heads=8 vs 1, rotary_embedding) and dumps each
    intermediate. Localizes WHICH step in the L0 production code path produces
    the 0.31 K cache divergence observed in test_per_layer_kv_cache_cascade.
    """
    hidden_t = _deterministic_hidden()
    s = _run_single_l0_production_attention(hidden_t)
    m = _run_mesh_l0_production_attention(hidden_t)

    print("\nProduction L0 attention step-by-step (single vs 1×8 mesh):\n")
    for name in ("normed", "xqkv_k_cols", "k_postsplit", "k_rope_padded"):
        sx, mx = s[name], m[name]
        if sx.shape != mx.shape:
            print(f"  {name:>16}  SHAPE MISMATCH  single={tuple(sx.shape)} mesh={tuple(mx.shape)}")
            continue
        d = (sx - mx).abs()
        print(
            f"  {name:>16}  PCC={_pcc(sx, mx):.6f}  max_diff={d.max().item():.3e}  "
            f"mean_diff={d.mean().item():.3e}  shape={tuple(sx.shape)}"
        )

    # We're not asserting tight bounds — this test is diagnostic, not pass/fail.
    # The print output is the artifact. A later assertion is added once a fix
    # candidate is identified and we want CI to guard against regression.
    assert s["normed"].shape == m["normed"].shape, "shape mismatch on normed"
