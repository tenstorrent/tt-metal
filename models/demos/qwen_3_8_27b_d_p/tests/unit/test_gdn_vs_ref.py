# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet vs the torch reference, at the model's real dims and target SP x TP.

This block gets more scrutiny than the rest of the decoder, for two reasons. It is a **recurrent**
mixer, so an error does not stay local — it enters the carried state and contaminates every later
token; and every one of its pieces has a plausible-looking wrong version: a sigmoid gate instead
of silu, ``repeat`` instead of ``repeat_interleave`` on the GQA head map, the L2 norm's epsilon
outside the rsqrt instead of inside, a conv tap order reversed, a channel permutation that does
not match the one the projection used.

So each stage is measured on its own before the block, the block is measured before the chunked
run, and the chunked run is measured against a one-shot run of the same module. The stage tests
are per-TP-column: a permutation that shuffles heads *between* columns leaves the composed tensor
looking right and only shows up column by column.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_3_8_27b_d_p.reference.modeling import Qwen35GatedDeltaNet, init_random_weights
from models.demos.qwen_3_8_27b_d_p.tt.caches import allocate_gdn_state
from models.demos.qwen_3_8_27b_d_p.tt.gdn.prefill import GatedDeltaNet

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import WEIGHT_DTYPE, assert_tp_replicated, check_pcc, from_sp_sharded, randn, to_sp_sharded

S_LOCAL = 128  # per SP row; the chunk is S_LOCAL * sp, and must be a multiple of the scan's 32
LAYER_IDX = 0  # the first linear-attention layer


def _reference(cfg, seed: int = 41) -> Qwen35GatedDeltaNet:
    ref = Qwen35GatedDeltaNet(cfg, LAYER_IDX).eval()
    init_random_weights(ref, seed=seed)
    return ref


def _tt_gdn(mesh, cfg, mesh_config, ccl, ref) -> GatedDeltaNet:
    return GatedDeltaNet(
        mesh,
        cfg,
        ref.state_dict(),
        mesh_config=mesh_config,
        ccl_manager=ccl,
        layer_idx=LAYER_IDX,
        weight_dtype=WEIGHT_DTYPE,
    )


def _sp_col(t: ttnn.Tensor, mesh_config, column: int, seq_dim: int = 2) -> torch.Tensor:
    """Compose one TP column's shards across the SP rows."""
    dev = ttnn.get_device_tensors(t)
    return torch.cat([ttnn.to_torch(dev[r * mesh_config.tp + column]) for r in range(mesh_config.sp)], dim=seq_dim)


def _col0(t: ttnn.Tensor, column: int, mesh_config) -> torch.Tensor:
    """One TP column's row-0 shard, for tensors already gathered over SP."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[column])


@parametrize_mesh()
def test_gdn_gates_vs_ref(mesh, submesh_shape, device_params):
    """``beta = sigmoid(b)`` and ``g = -exp(A_log) * softplus(a + dt_bias)``, per TP column.

    ``-exp(A_log)`` is folded on the host in fp32; the device path must keep the rest fp32 too,
    because ``exp`` of a fp16 ``A_log`` reaches inf and the scan's state becomes NaN for the whole
    rest of the sequence rather than merely inaccurate.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=42, scale=0.5)
    with torch.no_grad():
        ref_beta, ref_g = ref.gates(x)

    gdn = _tt_gdn(mesh, cfg, mesh_config, ccl, ref)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    beta, g = gdn.gates(tt_x)
    hv = gdn.n_v_local
    for c in range(mesh_config.tp):
        check_pcc(
            f"gdn_beta[col{c}]",
            ref_beta[:, :, c * hv : (c + 1) * hv].reshape(1, 1, total, hv),
            _sp_col(beta, mesh_config, c).reshape(1, 1, total, hv),
        )
        check_pcc(
            f"gdn_g[col{c}]",
            ref_g[:, :, c * hv : (c + 1) * hv].reshape(1, 1, total, hv),
            _sp_col(g, mesh_config, c).reshape(1, 1, total, hv),
        )


@parametrize_mesh()
def test_gdn_causal_conv_vs_ref(mesh, submesh_shape, device_params):
    """Projection -> SP all-gather -> 4-tap causal conv + SiLU -> q/k/v split, per TP column.

    Also the test for the rank-block-major channel permutation: the conv is depthwise, so the
    projection's channel order and the taps' channel order must agree exactly. They differ from
    the HF order, and a mismatch is a channel-wise scramble of a still perfectly smooth tensor.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=43, scale=0.5)
    with torch.no_grad():
        mixed = ref.in_proj_qkv(x).transpose(1, 2)
        conv_out = ref.causal_conv(mixed, None)  # [1, conv_dim, T]
        ref_q_flat, ref_k_flat, ref_v_flat = torch.split(
            conv_out.transpose(1, 2), [cfg.gdn_key_dim, cfg.gdn_key_dim, cfg.gdn_value_dim], dim=-1
        )

    gdn = _tt_gdn(mesh, cfg, mesh_config, ccl, ref)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    mixed_local = ttnn.linear(tt_x, gdn.weights.in_proj_qkv, dtype=ttnn.bfloat16)
    mixed_full = gdn.gather_sequence(mixed_local)
    q, k, v, next_hist = gdn.causal_conv(mixed_full, gdn._zero_history())

    kdim, vdim = gdn.q_width, gdn.v_width
    for c in range(mesh_config.tp):
        check_pcc(
            f"gdn_conv_q[col{c}]",
            ref_q_flat[:, :, c * kdim : (c + 1) * kdim].reshape(1, total, kdim),
            _col0(q, c, mesh_config).reshape(1, total, kdim),
        )
        check_pcc(
            f"gdn_conv_k[col{c}]",
            ref_k_flat[:, :, c * kdim : (c + 1) * kdim].reshape(1, total, kdim),
            _col0(k, c, mesh_config).reshape(1, total, kdim),
        )
        check_pcc(
            f"gdn_conv_v[col{c}]",
            ref_v_flat[:, :, c * vdim : (c + 1) * vdim].reshape(1, total, vdim),
            _col0(v, c, mesh_config).reshape(1, total, vdim),
        )
    assert tuple(next_hist.shape) == (1, cfg.linear_conv_kernel_dim - 1, gdn.conv_width)


@parametrize_mesh()
def test_gdn_scan_vs_ref(mesh, submesh_shape, device_params):
    """The chunked gated delta rule itself, fed the reference's own q/k/v/g/beta.

    Decoupled from the projections so a failure here is the scan and not the plumbing. Checks the
    output AND the final recurrent state — the state is what a later chunk continues from, so an
    output-only check would pass a scan that leaves the wrong state behind.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    gdn = _tt_gdn(mesh, cfg, mesh_config, ccl, ref)

    hv_local, dk, dv = gdn.n_v_local, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    nk_local = gdn.n_k_local
    q = randn(1, total, nk_local * dk, seed=44)
    k = randn(1, total, nk_local * dk, seed=45)
    v = randn(1, total, hv_local * dv, seed=46)
    beta = torch.rand(1, total, hv_local)
    g = -torch.rand(1, total, hv_local) * 0.5

    from models.demos.qwen_3_8_27b_d_p.reference.modeling import torch_chunk_gated_delta_rule

    groups = cfg.gdn_num_value_groups
    with torch.no_grad():
        ref_o, ref_state = torch_chunk_gated_delta_rule(
            q.reshape(1, total, nk_local, dk).repeat_interleave(groups, dim=2),
            k.reshape(1, total, nk_local, dk).repeat_interleave(groups, dim=2),
            v.reshape(1, total, hv_local, dv),
            g=g,
            beta=beta,
            chunk_size=32,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    def _rep(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.float(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.replicate(mesh),
        )

    o, state = gdn.scan(_rep(q), _rep(k), _rep(v), _rep(g, ttnn.float32), _rep(beta, ttnn.float32), None)
    # o is head-major [hv_local, T, dv]; the reference is token-major [1, T, hv_local, dv].
    got_o = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(hv_local, total, dv)
    check_pcc(
        "gdn_scan_out", ref_o.reshape(1, total, hv_local, dv).permute(0, 2, 1, 3).reshape(hv_local, total, dv), got_o
    )
    got_state = ttnn.to_torch(ttnn.get_device_tensors(state)[0]).reshape(1, hv_local, dk, dv)
    check_pcc("gdn_scan_final_state", ref_state.reshape(1, hv_local, dk, dv), got_state)


@parametrize_mesh()
def test_gdn_gated_norm_vs_ref(mesh, submesh_shape, device_params):
    """Per-head RMSNorm then the **silu** gate, built as ``sigmoid_gated_rms_norm(...) * z``.

    The identity being relied on is ``silu(z) = z * sigmoid(z)``. Dropping the trailing multiply
    leaves a sigmoid-gated norm: smooth, well-scaled, and wrong — so this compares against the
    reference's silu form directly rather than trusting the algebra.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    gdn = _tt_gdn(mesh, cfg, mesh_config, ccl, ref)
    hv_local, dv = gdn.n_v_local, cfg.linear_value_head_dim

    core = randn(1, total, hv_local, dv, seed=47)
    z = randn(1, total, hv_local * dv, seed=48)
    with torch.no_grad():
        expected = ref.norm(core.reshape(-1, dv), z.reshape(-1, dv)).reshape(1, total, hv_local * dv)

    o_hm = ttnn.from_torch(
        core.permute(0, 2, 1, 3).reshape(hv_local, total, dv).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.replicate(mesh),
    )
    tt_z = ttnn.from_torch(
        z.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.replicate(mesh),
    )
    got = ttnn.to_torch(ttnn.get_device_tensors(gdn.gated_norm(o_hm, tt_z))[0])
    check_pcc("gdn_gated_norm", expected, got.reshape(1, total, hv_local * dv))


@parametrize_mesh()
def test_gated_deltanet_vs_ref(mesh, submesh_shape, device_params):
    """The whole block, one shot: projections -> gather -> conv -> scan -> gated norm -> out_proj."""
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=49, scale=0.5)
    with torch.no_grad():
        expected, ref_state = ref(x)

    gdn = _tt_gdn(mesh, cfg, mesh_config, ccl, ref)
    state = allocate_gdn_state(mesh, cfg, mesh_config=mesh_config)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    out = gdn(tt_x, state=state)
    assert_tp_replicated(out, mesh_config, "gated deltanet output")
    got = from_sp_sharded(out, mesh_config)
    check_pcc("gated_deltanet", expected.reshape(1, 1, total, cfg.hidden_size), got)

    # BOTH carried states are what a later chunk continues from, so both are graded.
    hv_local = gdn.n_v_local
    for c in range(mesh_config.tp):
        check_pcc(
            f"gdn_recurrent_state[col{c}]",
            ref_state.recurrent_state[:, c * hv_local : (c + 1) * hv_local],
            ttnn.to_torch(ttnn.get_device_tensors(state.recurrent)[c]).reshape(
                1, hv_local, cfg.linear_key_head_dim, cfg.linear_value_head_dim
            ),
        )

    # The conv history needs BOTH a transpose and the inverse channel permutation to line up with
    # the reference. Checked here rather than only in the P1 harness: a wrong conversion there
    # would read as a model bug at full depth instead of a layout bug in one helper.
    from models.demos.qwen_3_8_27b_d_p.tt.gdn.weights import device_conv_state_to_hf

    conv_dev = ttnn.get_device_tensors(state.conv_state)
    conv = torch.cat([ttnn.to_torch(conv_dev[c]) for c in range(mesh_config.tp)], dim=-1)
    conv = conv.reshape(1, cfg.linear_conv_kernel_dim - 1, cfg.gdn_conv_dim)
    conv_hf = device_conv_state_to_hf(conv, cfg, mesh_config.tp)
    assert tuple(conv_hf.shape) == tuple(ref_state.conv_state.shape)
    check_pcc("gdn_conv_state", ref_state.conv_state, conv_hf)


@parametrize_mesh()
def test_gated_deltanet_chunked_vs_ref(mesh, submesh_shape, device_params):
    """Two chunks through the SAME module, threading conv + recurrent state, vs a one-shot run.

    This is the GDN analogue of "chunk N attends the prefix left in the KV cache", and it needs
    BOTH carried states to be right: drop the conv history and only the first 3 tokens of each
    chunk are wrong (a PCC dent, easy to miss); drop the recurrent state and everything after the
    first chunk is wrong.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    chunk = S_LOCAL * mesh_config.sp
    total = 2 * chunk
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=50, scale=0.5)
    with torch.no_grad():
        expected, _ = ref(x)

    gdn = _tt_gdn(mesh, cfg, mesh_config, ccl, ref)
    state = allocate_gdn_state(mesh, cfg, mesh_config=mesh_config)
    outs = []
    for c in range(2):
        tt_x = to_sp_sharded(x[:, c * chunk : (c + 1) * chunk].reshape(1, 1, chunk, cfg.hidden_size), mesh, mesh_config)
        outs.append(from_sp_sharded(gdn(tt_x, state=state), mesh_config))
    got = torch.cat(outs, dim=2)
    check_pcc("gated_deltanet_chunked", expected.reshape(1, 1, total, cfg.hidden_size), got)


@parametrize_mesh(graded_only=True)
def test_gdn_conv_history_is_load_bearing(mesh, submesh_shape, device_params):
    """Guard-rail: a second chunk run with a ZEROED conv history must be measurably worse than
    one run with the carried history. Without this, ``test_gated_deltanet_chunked_vs_ref`` could
    pass on a 3-token error in 1024 and nobody would know the history was being dropped."""
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    chunk = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, 2 * chunk, cfg.hidden_size, seed=51, scale=0.5)
    with torch.no_grad():
        expected, _ = ref(x)
    expected_second = expected[:, chunk:].reshape(1, 1, chunk, cfg.hidden_size)

    gdn = _tt_gdn(mesh, cfg, mesh_config, ccl, ref)
    state = allocate_gdn_state(mesh, cfg, mesh_config=mesh_config)
    first = to_sp_sharded(x[:, :chunk].reshape(1, 1, chunk, cfg.hidden_size), mesh, mesh_config)
    gdn(first, state=state)

    second = to_sp_sharded(x[:, chunk:].reshape(1, 1, chunk, cfg.hidden_size), mesh, mesh_config)
    carried = from_sp_sharded(gdn(second, state=state), mesh_config)

    from models.common.utility_functions import comp_pcc

    _, pcc_carried = comp_pcc(expected_second.float(), carried.float(), 0.0)

    # Re-run the second chunk with the conv history zeroed but the recurrent state kept.
    state2 = allocate_gdn_state(mesh, cfg, mesh_config=mesh_config)
    first2 = to_sp_sharded(x[:, :chunk].reshape(1, 1, chunk, cfg.hidden_size), mesh, mesh_config)
    gdn(first2, state=state2)
    state2.conv_state.deallocate(True)
    state2.conv_state = gdn._zero_history()
    second2 = to_sp_sharded(x[:, chunk:].reshape(1, 1, chunk, cfg.hidden_size), mesh, mesh_config)
    dropped = from_sp_sharded(gdn(second2, state=state2), mesh_config)
    _, pcc_dropped = comp_pcc(expected_second.float(), dropped.float(), 0.0)

    assert float(pcc_carried) > float(pcc_dropped), (
        f"carrying the conv history ({pcc_carried}) is not better than dropping it "
        f"({pcc_dropped}) — the history is not actually being used"
    )
