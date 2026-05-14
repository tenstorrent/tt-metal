# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Math-level parity for S2V segmented timestep modulation.

Compares ``WanS2VTransformer3DModel._s2v_segmented_block_forward`` against
the **actual** reference ``WanS2VAttentionBlock.forward`` from the wan repo
(``wan/modules/s2v/model_s2v.py``). The reference's ``flash_attention``
requires CUDA, so we monkey-patch ``flash_attention`` to use
:func:`torch.nn.functional.scaled_dot_product_attention` — same math
(``softmax(QK/sqrt(d))V``), CPU-compatible.

Test bar: PCC ≥ 0.99 (``feedback_wan_pcc_bar.md``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.transformer_wan_s2v import WanS2VTransformer3DModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.padding import get_padded_vision_seq_len
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, float32_tensor, from_torch, local_device_to_torch
from ....utils.test import line_params

_REF_REPO = Path("/home/kevinmi/wan2_2_ref")


# ---------------------------------------------------------------------------
# Small synthetic config. Constraints:
#   * head_dim ≥ TILE*tp = 128 (DistributedRMSNorm divisibility).
#   * dim divisible by num_heads and tp.
#   * ffn_dim divisible by tp.
#   * N_noisy and N_const each tile-aligned (TILE*sp = 256).
# ---------------------------------------------------------------------------
DIM = 512
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS  # 128
FFN_DIM = 2048
TEXT_DIM = 4096
FREQ_DIM = 256

B = 1
N_NOISY = 512
N_REF = 256
N_MOTION = 256
N_CONST = N_REF + N_MOTION  # 512
EPS = 1e-6


def _install_wan_ref_stubs() -> None:
    """Stub the wan repo's CUDA/flash_attn dependencies for CPU execution.

    The reference's ``wan/__init__.py`` eagerly imports ``flash_attn``,
    ``decord``, and probes ``torch.cuda.current_device()``. None of these
    are available in the BH test env, so we substitute trivial replacements
    before importing ``wan.modules.s2v.model_s2v``.
    """
    if "flash_attn" not in sys.modules:
        mod = types.ModuleType("flash_attn")
        mod.flash_attn_func = None  # type: ignore[attr-defined]
        mod.flash_attn_qkvpacked_func = None  # type: ignore[attr-defined]
        sys.modules["flash_attn"] = mod
    if "decord" not in sys.modules:
        mod = types.ModuleType("decord")
        mod.VideoReader = None  # type: ignore[attr-defined]
        mod.cpu = lambda x=0: None  # type: ignore[attr-defined]
        sys.modules["decord"] = mod
    # ``rope_params`` and a few constructors call ``torch.cuda.current_device``;
    # patch it to a safe no-op for CPU.
    if not hasattr(torch.cuda, "_orig_current_device"):
        torch.cuda._orig_current_device = torch.cuda.current_device  # type: ignore[attr-defined]
        torch.cuda.current_device = lambda: 0  # type: ignore[assignment]


def _cpu_flash_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):  # noqa: ARG001
    """CPU-compatible replacement for ``wan.modules.attention.flash_attention``.

    Same math (``softmax(QK^T / sqrt(d)) V``), uses
    :func:`torch.nn.functional.scaled_dot_product_attention`. Ignores ``q_lens``/
    ``k_lens`` since our test inputs have no padding.

    Reference signature: ``q,k,v`` shape ``[B, L, H, D]`` → output same shape.
    SDPA wants ``[B, H, L, D]``, so we transpose around the call.
    """
    q_t = q.transpose(1, 2)  # [B, H, L_q, D]
    k_t = k.transpose(1, 2)  # [B, H, L_k, D]
    v_t = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
    return out.transpose(1, 2).contiguous()  # back to [B, L_q, H, D]


def _patch_wan_flash_attention() -> None:
    """Substitute the wan repo's CUDA-only ``flash_attention`` with CPU SDPA
    everywhere it's been imported (``wan.modules.attention``,
    ``wan.modules.model``, ``wan.modules.s2v.model_s2v``).
    """
    import wan.modules.attention as _attn
    import wan.modules.model as _model
    from wan.modules.s2v import model_s2v as _s2v_model

    _attn.flash_attention = _cpu_flash_attention
    _model.flash_attention = _cpu_flash_attention
    _s2v_model.flash_attention = _cpu_flash_attention


# Block-level ref → tt translation. Mirrors ``wan_s2v_weight_map.py`` for one block.
_REF_TO_TT_ATTN = {
    "q.weight": "to_q.weight",
    "q.bias": "to_q.bias",
    "k.weight": "to_k.weight",
    "k.bias": "to_k.bias",
    "v.weight": "to_v.weight",
    "v.bias": "to_v.bias",
    "o.weight": "to_out.0.weight",
    "o.bias": "to_out.0.bias",
    "norm_q.weight": "norm_q.weight",
    "norm_k.weight": "norm_k.weight",
}


def _translate_block_state(ref_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate a reference WanS2VAttentionBlock state_dict into the TT
    block's naming. Mirrors ``wan_s2v_weight_map.py`` per-block rules,
    without the ``blocks.0.`` prefix (we load directly into ``tt_model.blocks[0]``).
    """
    tt: dict[str, torch.Tensor] = {}
    for key, tensor in ref_state.items():
        if key.startswith("self_attn."):
            sub = key[len("self_attn.") :]
            if sub in _REF_TO_TT_ATTN:
                tt[f"attn1.{_REF_TO_TT_ATTN[sub]}"] = tensor
        elif key.startswith("cross_attn."):
            sub = key[len("cross_attn.") :]
            if sub in _REF_TO_TT_ATTN:
                tt[f"attn2.{_REF_TO_TT_ATTN[sub]}"] = tensor
        elif key in ("norm3.weight", "norm3.bias"):
            # ref.norm3 (cross-attn pre-norm with affine) → tt.norm2.
            tt[f"norm2.{key.split('.', 1)[1]}"] = tensor
        elif key == "modulation":
            # Pass through raw [1, 6, dim]; TT block's _prepare_torch_state
            # adds the leading unsqueeze to [1, 1, 6, dim].
            tt["scale_shift_table"] = tensor
        elif key.startswith("ffn.0."):
            tt[f"ffn.net.0.proj.{key.split('.', 2)[2]}"] = tensor
        elif key.startswith("ffn.2."):
            tt[f"ffn.net.2.{key.split('.', 2)[2]}"] = tensor
    return tt


def _upload_spatial_2dshard(spatial_1BND_torch, mesh_device, sp_axis, tp_axis, padded_N_noisy):
    """Upload spatial as ``concat([noisy_2dshard, const_2dshard], dim=-2)`` —
    matches the in-pipeline per-device concat layout that the mask alignment
    fix depends on.
    """
    noisy_part = spatial_1BND_torch[:, :, :padded_N_noisy, :].contiguous()
    const_part = spatial_1BND_torch[:, :, padded_N_noisy:, :].contiguous()
    shard_mapping = {sp_axis: 2, tp_axis: 3}
    noisy_tt = bf16_tensor_2dshard(noisy_part, mesh_device, shard_mapping=shard_mapping, layout=ttnn.TILE_LAYOUT)
    const_tt = bf16_tensor_2dshard(const_part, mesh_device, shard_mapping=shard_mapping, layout=ttnn.TILE_LAYOUT)
    return ttnn.concat([noisy_tt, const_tt], dim=-2)


def _gather_to_torch(tt_tensor, ccl_manager, sp_axis, tp_axis):
    """All-gather across SP (dim 2) then TP (dim 3); read device 0's view."""
    gathered = ccl_manager.all_gather_persistent_buffer(tt_tensor, dim=2, mesh_axis=sp_axis)
    gathered = ccl_manager.all_gather_persistent_buffer(gathered, dim=3, mesh_axis=tp_axis)
    return local_device_to_torch(gathered)


def _unpermute_concat_per_device(
    gathered: torch.Tensor, sp: int, padded_N_noisy: int, padded_const: int
) -> torch.Tensor:
    """Un-permute the SP-gathered output of a ``concat([noisy_SP, const_SP], dim=-2)``
    spatial back to contiguous global order ``[noisy_global | const_global]``.

    The per-device layout after the in-pipeline concat is
    ``[noisy_local | const_local]`` (sizes ``padded_N_noisy/sp`` and ``padded_const/sp``).
    After SP all-gather, dim 2 holds ``[dev0_noisy, dev0_const, dev1_noisy, dev1_const, ...]``
    — interleaved by device, not contiguous global. CPU references operate on
    contiguous global order, so we reshape and split here.
    """
    # gathered shape: [1, B, padded_N_total, dim] where padded_N_total = sp*(NN+NC)
    NN = padded_N_noisy // sp
    NC = padded_const // sp
    # Reshape to [..., sp, NN+NC, dim] to expose the per-device chunk.
    g = gathered.view(*gathered.shape[:2], sp, NN + NC, gathered.shape[-1])
    noisy_chunks = g[..., :NN, :].reshape(*gathered.shape[:2], sp * NN, gathered.shape[-1])
    const_chunks = g[..., NN:, :].reshape(*gathered.shape[:2], sp * NC, gathered.shape[-1])
    return torch.cat([noisy_chunks, const_chunks], dim=-2)


def _make_rope_freqs(N: int, head_dim: int) -> torch.Tensor:
    """Reference ``rope_params(N, head_dim)`` — outer product of position
    indices with inverse-frequency, converted to complex via :func:`torch.polar`.
    Returns ``[1, N, 1, head_dim/2]`` complex (with a head-dim of 1 that
    broadcasts across heads in the reference's ``rope_apply``).
    """
    theta = 10000.0
    freqs = torch.outer(
        torch.arange(N),
        1.0 / torch.pow(theta, torch.arange(0, head_dim, 2).to(torch.float64) / head_dim),
    )
    # [N, d/2] → [1, N, 1, d/2]
    return torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0).unsqueeze(2)


@pytest.mark.skipif(
    not (_REF_REPO / "wan" / "modules" / "s2v" / "model_s2v.py").exists(),
    reason="Wan-Video/Wan2.2 reference repo not at /home/kevinmi/wan2_2_ref",
)
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8),
            (4, 8),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_4x8sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_s2v_segmented_block_math_parity(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Per-block PCC against the **actual** reference WanS2VAttentionBlock.

    Drives the reference block (with ``flash_attention`` monkey-patched to
    CPU SDPA) and the TT ``_s2v_segmented_block_forward`` on the same random
    inputs. Bar: PCC ≥ 0.99.
    """
    torch.manual_seed(0)

    # --- Set up reference: stub imports, monkey-patch flash_attention. ---
    if str(_REF_REPO) not in sys.path:
        sys.path.insert(0, str(_REF_REPO))
    _install_wan_ref_stubs()
    _patch_wan_flash_attention()

    from wan.modules.s2v.model_s2v import WanS2VAttentionBlock  # noqa: WPS433

    ref_block = (
        WanS2VAttentionBlock(
            dim=DIM,
            ffn_dim=FFN_DIM,
            num_heads=NUM_HEADS,
            cross_attn_norm=True,
            eps=EPS,
        )
        .eval()
        .to(torch.float32)
    )
    logger.info(f"Reference block built: dim={DIM} ffn_dim={FFN_DIM} heads={NUM_HEADS}")

    # --- Build TT model with matching small config. Only block 0 is driven. ---
    sp = tuple(mesh_device.shape)[sp_axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    tt_model = WanS2VTransformer3DModel(
        patch_size=(1, 2, 2),
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=16,
        out_channels=16,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        ffn_dim=FFN_DIM,
        num_layers=1,
        cross_attn_norm=True,
        eps=EPS,
        rope_max_seq_len=1024,
        audio_dim=1024,
        num_audio_layers=25,
        num_audio_token=4,
        audio_inject_layers=(),  # disable; not driven here
        enable_adain=False,
        enable_motioner=False,
        enable_framepack=True,
        motion_token_num=1024,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        model_type="s2v",
    )

    # --- Transfer ref block weights → tt_model.blocks[0]. ---
    block_sd = _translate_block_state(ref_block.state_dict())
    incompatible = tt_model.blocks[0].load_torch_state_dict(block_sd, strict=False)
    logger.info(
        f"TT block 0 load: missing={len(incompatible.missing_keys)} " f"unexpected={len(incompatible.unexpected_keys)}"
    )

    # --- Inputs (no pad: sizes are tile-aligned to TILE*sp = 256). ---
    N_total = N_NOISY + N_CONST  # 1024
    padded_N_noisy = get_padded_vision_seq_len(N_NOISY, sp)
    padded_const = get_padded_vision_seq_len(N_CONST, sp)
    padded_N_total = padded_N_noisy + padded_const
    assert N_NOISY == padded_N_noisy and N_CONST == padded_const, "config should be tile-aligned"
    logger.info(f"Sizes: N_noisy={N_NOISY} N_const={N_CONST} padded_N_total={padded_N_total}")

    spatial_BND = torch.randn(B, N_total, DIM, dtype=torch.float32)
    context_BLD = torch.randn(B, 32, DIM, dtype=torch.float32)  # post-text-embedding
    e0_real = torch.randn(B, 6, DIM, dtype=torch.float32)
    e0_zero = torch.randn(B, 6, DIM, dtype=torch.float32)

    # Rope freqs (per-token, complex) — same on both sides.
    freqs_complex = _make_rope_freqs(N_total, HEAD_DIM)

    # --- Reference forward ---
    # Build the ``e`` payload in the format ``WanS2VAttentionBlock.forward`` expects:
    # ``[e0_stacked, seg_idx]`` where e0_stacked has shape [B, 6, 2, dim] holding
    # [real-t, zero-t] along dim 2 (see WanModel_S2V.forward, ``zero_timestep=True`` branch).
    e0_stacked = torch.cat([e0_real.unsqueeze(2), e0_zero.unsqueeze(2)], dim=2)  # [B, 6, 2, dim]
    seg_idx_tensor = torch.tensor(N_NOISY, dtype=torch.long)
    seq_lens = torch.tensor([N_total], dtype=torch.long)
    # grid_sizes: WanS2VSelfAttention.forward signature passes it but the S2V
    # rope_apply (model_s2v.py:62) uses ``freqs`` directly and ignores grid_sizes.
    grid_sizes = torch.tensor([[1, 1, N_total]], dtype=torch.long)
    with torch.no_grad():
        ref_out = ref_block(
            spatial_BND,
            [e0_stacked, seg_idx_tensor],
            seq_lens,
            grid_sizes,
            freqs_complex,
            context_BLD,
            None,  # context_lens
        )
    logger.info(f"Reference block output: {tuple(ref_out.shape)}")

    # --- TT-side input upload ---
    spatial_1BND = spatial_BND.unsqueeze(0)
    spatial_tt = _upload_spatial_2dshard(spatial_1BND, mesh_device, sp_axis, tp_axis, padded_N_noisy)
    prompt_tt = bf16_tensor(context_BLD.unsqueeze(0), device=mesh_device, layout=ttnn.TILE_LAYOUT)

    timestep_proj_real = float32_tensor(
        e0_real.unsqueeze(0),
        device=mesh_device,
        mesh_axis=tp_axis,
        shard_dim=3,
        layout=ttnn.TILE_LAYOUT,
    )
    timestep_proj_zero = float32_tensor(
        e0_zero.unsqueeze(0),
        device=mesh_device,
        mesh_axis=tp_axis,
        shard_dim=3,
        layout=ttnn.TILE_LAYOUT,
    )

    # Masks (concat-per-device layout matching the spatial).
    def _upload_mask_seg(t: torch.Tensor) -> ttnn.Tensor:
        return float32_tensor(
            t.contiguous(),
            device=mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
            layout=ttnn.TILE_LAYOUT,
        )

    m_n_noisy = torch.zeros(1, 1, padded_N_noisy, 1, dtype=torch.float32)
    m_n_noisy[:, :, :N_NOISY, :] = 1.0
    m_n_const = torch.zeros(1, 1, padded_const, 1, dtype=torch.float32)
    mask_noisy_tt = ttnn.concat([_upload_mask_seg(m_n_noisy), _upload_mask_seg(m_n_const)], dim=-2)

    m_c_noisy = torch.zeros(1, 1, padded_N_noisy, 1, dtype=torch.float32)
    m_c_const = torch.zeros(1, 1, padded_const, 1, dtype=torch.float32)
    m_c_const[:, :, :N_CONST, :] = 1.0
    mask_constant_tt = ttnn.concat([_upload_mask_seg(m_c_noisy), _upload_mask_seg(m_c_const)], dim=-2)

    # Rope cos/sin (real pair from complex freqs).
    # freqs_complex shape: [1, N, 1, head_dim/2] complex.
    # Permute to [1, 1, N, head_dim/2] then repeat_interleave to [1, 1, N, head_dim].
    cos_half = freqs_complex.real.permute(0, 2, 1, 3).float()  # [1, 1, N, head_dim/2]
    sin_half = freqs_complex.imag.permute(0, 2, 1, 3).float()
    cos_full = cos_half.repeat_interleave(2, dim=-1)
    sin_full = sin_half.repeat_interleave(2, dim=-1)
    rope_cos_tt = from_torch(
        cos_full.to(torch.float32),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[None, None, sp_axis, None],
        layout=ttnn.TILE_LAYOUT,
    )
    rope_sin_tt = from_torch(
        sin_full.to(torch.float32),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[None, None, sp_axis, None],
        layout=ttnn.TILE_LAYOUT,
    )
    from ....utils.mochi import get_rot_transformation_mat

    trans_mat_tt = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # --- TT segmented forward ---
    out_tt = tt_model._s2v_segmented_block_forward(  # noqa: SLF001
        tt_model.blocks[0],
        spatial_1BND=spatial_tt,
        prompt_1BLP=prompt_tt,
        N=padded_N_total,
        rope_cos=rope_cos_tt,
        rope_sin=rope_sin_tt,
        trans_mat=trans_mat_tt,
        timestep_proj_real=timestep_proj_real,
        timestep_proj_zero=timestep_proj_zero,
        mask_noisy=mask_noisy_tt,
        mask_constant=mask_constant_tt,
    )

    out_full = _gather_to_torch(out_tt, ccl_manager, sp_axis, tp_axis)
    # Un-permute the concat-per-device layout back to contiguous [noisy | const].
    out_full = _unpermute_concat_per_device(out_full, sp, padded_N_noisy, padded_const)
    out_BND = out_full.squeeze(0)[:, :N_total, :]
    logger.info(f"TT output: {tuple(out_BND.shape)}")

    assert_quality(out_BND.float(), ref_out.float(), pcc=0.99)
