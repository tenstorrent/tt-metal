# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fuse C contract test: fused tt-lang beta/g kernel vs the 6-op TTNN chain.

This validates the *module-level* fused beta/g path (``QWEN36_TT_LANG_BETA_G=1``)
as wired into ``TtQwen36DeltaAttention``:

1. **galaxy viability** — does the 1×1-tile ``ttnn.generic_op`` beta/g kernel
   actually launch on the 32-chip BH Galaxy (the recurrent v2/v3 tt-lang
   kernels SIGSEGV here; this 1-tile generic_op might too).
2. **PCC contract** — the fused kernel's ``beta``/``g`` match the 6-op chain
   (``sigmoid``/``add``/``softplus``/``exp``/``neg``/``multiply``) at PCC > 0.99.
3. **layout reconciliation** — under ``QWEN36_DN_FUSED_HEADS=1`` the recurrent
   core consumes beta/g as ``[B, n_v, T] = [1, 6, 1]`` (pre_transposed). The
   fused kernel emits ``[1, 1, 6]``; a single ``reshape`` to ``[1, 6, 1]`` must
   preserve the per-head order (verified vs a torch transpose at T=1).

Built via the ``perf_deltanet_decode_unit`` 1-layer-model scaffold so the test
exercises the REAL module method (``_compute_beta_g`` + ``_compute_beta_g_tt_lang``
+ the ``_beta_g_kernel_state`` built at ``__init__``).

Run (full galaxy):
    source python_env/bin/activate
    QWEN36_TT_LANG_BETA_G=1 python -m pytest -s --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_dn_beta_g.py
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tests.perf_deltanet_decode_unit import MESH_SHAPE, _build_one_layer_model

_PCC_BAR = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _make_ba(mesh, attn):
    """Random b/a in the decode call-site layout: per-device [B=1, T=1, n_v=6],
    sharded full n_v=48 along dim 2 across the 8 mesh-rows (6 per row) — the
    same ShardTensor2dMesh pattern the projection ``ba`` uses."""
    n_v = attn.n_v_heads  # 48
    torch.manual_seed(0xC0FFEE)
    b_t = torch.randn(1, 1, n_v, dtype=torch.bfloat16)
    a_t = torch.randn(1, 1, n_v, dtype=torch.bfloat16)
    row_shard = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=attn.cluster_shape)

    def _to(x):
        return ttnn.from_torch(
            x,
            device=mesh,
            dtype=ttnn.bfloat8_b,  # projection ba is bf8_b at the call site
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_shard,
        )

    return _to(b_t), _to(a_t), b_t, a_t


def _first_row_shard(t):
    """Read the per-device [1,1,6] (or [1,6,1]) shard from the first device."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def test_fused_beta_g_matches_chain(bh_glx_mesh):
    """Gate 1: the module's fused beta/g kernel runs on galaxy AND its output
    matches the 6-op chain at PCC > 0.99 (per-device [1,1,6] shards).

    QWEN36_TT_LANG_BETA_G is default-on (Fuse C), so the kernel state is built
    by default; the asserts below check the effective state rather than the raw
    env var (which need not be set explicitly)."""
    model, _ = _build_one_layer_model(bh_glx_mesh, 0)
    attn = model.layers[0].attention
    assert attn.use_tt_lang_beta_g, "use_tt_lang_beta_g not set (QWEN36_TT_LANG_BETA_G=0?)"
    assert attn._beta_g_kernel_state is not None, "kernel state not built"

    b, a, _, _ = _make_ba(bh_glx_mesh, attn)

    # --- fused kernel (the galaxy-viability check; may SIGSEGV) ---
    beta_f, g_f = attn._compute_beta_g_tt_lang(b, a)
    ttnn.synchronize_device(bh_glx_mesh)
    beta_fused = _first_row_shard(beta_f).clone()
    g_fused = _first_row_shard(g_f).clone()

    # --- 6-op chain reference (force the fallback path) ---
    saved = attn.use_tt_lang_beta_g
    attn.use_tt_lang_beta_g = False
    try:
        beta_c, g_c = attn._compute_beta_g(b, a, 1, 1)
    finally:
        attn.use_tt_lang_beta_g = saved
    ttnn.synchronize_device(bh_glx_mesh)
    beta_chain = _first_row_shard(beta_c)
    g_chain = _first_row_shard(g_c)

    beta_pcc = _pcc(beta_fused, beta_chain)
    g_pcc = _pcc(g_fused, g_chain)
    print(f"[fuse-C] beta PCC = {beta_pcc:.6f}  g PCC = {g_pcc:.6f}")
    print(
        f"[fuse-C] beta max|err| = {(beta_fused.float()-beta_chain.float()).abs().max():.6f}  "
        f"g max|err| = {(g_fused.float()-g_chain.float()).abs().max():.6f}"
    )
    assert beta_pcc > _PCC_BAR, f"beta PCC {beta_pcc:.5f} < {_PCC_BAR}"
    assert g_pcc > _PCC_BAR, f"g PCC {g_pcc:.5f} < {_PCC_BAR}"


def test_fused_beta_g_reshape_to_bht_layout(bh_glx_mesh):
    """Gate 1 (layout): the [1,1,6] fused output reshaped to [1,6,1] preserves
    the per-head order, matching what the fused-heads recurrent core (pre_transposed)
    expects. Compares the reshaped fused beta/g to the torch chain transposed
    [1,1,6]->[1,6,1] (bit-identical move at B=T=1)."""
    model, _ = _build_one_layer_model(bh_glx_mesh, 0)
    attn = model.layers[0].attention
    n_v_per_row = attn.n_v_per_row  # 6

    b, a, b_host, a_host = _make_ba(bh_glx_mesh, attn)

    beta_f, g_f = attn._compute_beta_g_tt_lang(b, a)
    beta_r = ttnn.reshape(beta_f, [1, n_v_per_row, 1])
    g_r = ttnn.reshape(g_f, [1, n_v_per_row, 1])
    ttnn.synchronize_device(bh_glx_mesh)

    assert list(beta_r.shape) == [1, n_v_per_row, 1], f"beta reshape {list(beta_r.shape)}"
    assert list(g_r.shape) == [1, n_v_per_row, 1], f"g reshape {list(g_r.shape)}"

    # HARDENING (final-review Important): the [1,1,6]->[1,6,1] reshape MUST return a
    # fresh COPY, not a view aliasing the PERSISTENT beta_out/g_out buffers. The
    # fused-heads decode path deallocates the reshaped beta/g every step (beta_g_owns_buffer);
    # if the reshape aliased the persistent buffers that dealloc would be a use-after-free
    # (the kernel writes into a freed buffer next step). Lock it loudly so a future ttnn
    # change that makes this reshape a view fails here instead of silently corrupting state.
    assert (
        beta_r.buffer_address() != beta_f.buffer_address()
    ), "beta [1,1,6]->[1,6,1] reshape aliases the persistent beta_out buffer — UAF risk (ttnn made it a view)"
    assert (
        g_r.buffer_address() != g_f.buffer_address()
    ), "g [1,1,6]->[1,6,1] reshape aliases the persistent g_out buffer — UAF risk (ttnn made it a view)"

    beta_bht = _first_row_shard(beta_r)  # [1,6,1]
    g_bht = _first_row_shard(g_r)

    # torch reference: 6-op chain on the FIRST row's 6 heads, transposed [1,1,6]->[1,6,1]
    A_log = attn.A_log_host_first_row if hasattr(attn, "A_log_host_first_row") else None
    # Recompute the chain in torch from the same host b/a (first 6 heads = first row).
    b6 = b_host[..., :n_v_per_row].float()  # [1,1,6]
    a6 = a_host[..., :n_v_per_row].float()
    # Pull dt_bias / A_log for the first row's 6 heads off the device.
    dt_bias_row = _first_row_shard(attn._beta_g_kernel_state["dt_bias_bf16"]).float()[..., :n_v_per_row]
    A_log_row = _first_row_shard(attn._beta_g_kernel_state["A_log_bf16"]).float()[..., :n_v_per_row]
    beta_ref = torch.sigmoid(b6)
    g_ref = -torch.exp(A_log_row) * torch.nn.functional.softplus(a6 + dt_bias_row)
    beta_ref_t = beta_ref.reshape(1, n_v_per_row, 1)
    g_ref_t = g_ref.reshape(1, n_v_per_row, 1)

    beta_pcc = _pcc(beta_bht, beta_ref_t)
    g_pcc = _pcc(g_bht, g_ref_t)
    print(f"[fuse-C][layout] beta[1,6,1] PCC={beta_pcc:.6f}  g[1,6,1] PCC={g_pcc:.6f}")
    assert beta_pcc > _PCC_BAR, f"beta layout PCC {beta_pcc:.5f} < {_PCC_BAR}"
    assert g_pcc > _PCC_BAR, f"g layout PCC {g_pcc:.5f} < {_PCC_BAR}"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-x"])
