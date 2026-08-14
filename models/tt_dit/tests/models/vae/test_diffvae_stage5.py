# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the LTX-2.5 DiffVAE stage-5 stack.

The reference is upstream's own ``ltx_core`` stage-5 modules with synthetic seeded
weights -- the shipped checkpoint lives on a gated HF repo and is not needed to prove
the port. Set ``LTX_CORE_SRC`` to a ``LTX-2/packages/ltx-core/src`` checkout if
``ltx_core`` is not already importable::

    git clone --depth 1 --filter=blob:none --sparse https://github.com/Lightricks/LTX-2.git
    cd LTX-2 && git sparse-checkout set packages
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.diffvae_ltx_stage5 import (
    NUM_ADALN_CHUNKS,
    DiffVAEStage5,
    DiffVAEStage5Config,
    Grid,
    _slice_last,
    default_rope_dim_split,
    patchify,
    unpatchify,
)
from models.tt_dit.utils.check import assert_quality

_LTX_CORE_SRC = os.environ.get("LTX_CORE_SRC")
if _LTX_CORE_SRC and _LTX_CORE_SRC not in sys.path:
    sys.path.insert(0, _LTX_CORE_SRC)

ltx_blocks = pytest.importorskip(
    "ltx_core.model.video_vae.transformer.combined.block",
    reason="ltx_core is the parity reference; set LTX_CORE_SRC to an LTX-2 checkout",
)
ltx_layers = pytest.importorskip("ltx_core.model.video_vae.transformer.layers")
ltx_fallback = pytest.importorskip("ltx_core.model.video_vae.transformer.fallback_na")
ltx_ops = pytest.importorskip("ltx_core.model.video_vae.ops")
ltx_rope = pytest.importorskip("ltx_core.model.video_vae.transformer.rope")
ltx_timestep = pytest.importorskip("ltx_core.model.transformer.timestep_embedding")

# Small enough to keep the host NA3D shim quick, deep enough in T that the 11-frame
# window is not just NATTEN's boundary shift everywhere.
GRID = Grid(batch=1, t=12, h=16, w=16)

# Emulates the static gates upstream folds into these projections at export. Without
# it the eight residual adds compound into activations far outside the trained range,
# which tells us nothing about the port.
FOLDED_GATE_SCALE = 0.1


class TorchStage5(torch.nn.Module):
    """``DiffusionVideoDecoder``'s stage-5 members and ``forward_diff_step``."""

    def __init__(self, config: DiffVAEStage5Config) -> None:
        super().__init__()
        self.config = config
        self.conv_in_x_t = torch.nn.Linear(config.patch_channels, config.dim, bias=True)
        self.t_embedder = ltx_timestep.PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim=config.t_emb_dim, size_emb_dim=0
        )
        self.shared_adaln = ltx_layers.AdaLNZero(dim=config.dim, t_emb_dim=config.t_emb_dim)
        self.diff_blocks = torch.nn.ModuleList(
            ltx_blocks.CombinedDiffusionNABlock(
                dim=config.dim,
                kernel_size=config.kernel_size,
                context_channels=config.context_channels,
                head_dim=config.head_dim,
            )
            for _ in range(config.num_blocks)
        )
        self.norm_out = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.conv_out = torch.nn.Linear(config.dim, config.patch_channels, bias=True)

        # natten is not installed on TT hosts; the eager tiled-SDPA backend is upstream's
        # own fallback and shares the window geometry with the ttnn module's host shim.
        eager = ltx_fallback.EagerSdpaAttention()
        for block in self.diff_blocks:
            block.attn.attention_function = eager

    def build_buffer(self, context: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        patched = ltx_ops.patchify(x_t, patch_size_hw=self.config.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(patched.permute(0, 2, 3, 4, 1))
        return torch.cat([context, x], dim=-1)

    def forward_diff_step(self, context_and_x: torch.Tensor, t: torch.Tensor) -> tuple[list, torch.Tensor]:
        cfg = self.config
        x = context_and_x[..., cfg.context_channels :]
        context = context_and_x[..., : cfg.context_channels]
        t_emb = self.t_embedder(cfg.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)

        per_block = []
        for block in self.diff_blocks:
            x = block.forward_combined(torch.cat([context, x], dim=-1), modulation)
            per_block.append(x)

        out = self.conv_out(self.norm_out(x))
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        return per_block, ltx_ops.unpatchify(out, patch_size_hw=cfg.patch_size, patch_size_t=1)


def randomize(model: TorchStage5, seed: int) -> None:
    """Seeded synthetic weights.

    AdaLN-Zero zero-inits its projection and ``scale_shift_table``, so an untouched
    reference has identity modulation everywhere and the test would not exercise it at all.
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    for name, param in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        with torch.no_grad():
            if name.endswith("scale_shift_table"):
                param.copy_(0.1 * randn(*param.shape))
            elif "norm" in name.split(".")[-2] and leaf == "weight" and param.ndim == 1:
                param.copy_(1.0 + 0.05 * randn(*param.shape))
            elif leaf == "weight" and param.ndim == 2:
                scale = param.shape[1] ** -0.5
                if name.endswith(("attn.proj.weight", "mlp.w_down.weight", "context_proj.weight")):
                    scale *= FOLDED_GATE_SCALE
                param.copy_(scale * randn(*param.shape))
            else:
                param.copy_(0.02 * randn(*param.shape))


def checkpoint_state(model: TorchStage5) -> dict[str, torch.Tensor]:
    """Re-spell the reference parameters into shipped-checkpoint key/shape form."""
    src = model.state_dict()
    out: dict[str, torch.Tensor] = {
        "conv_in_x_t.weight": src["conv_in_x_t.weight"],
        "conv_in_x_t.bias": src["conv_in_x_t.bias"],
        "shared_adaln.proj.weight": src["shared_adaln.proj.weight"],
        "shared_adaln.proj.bias": src["shared_adaln.proj.bias"],
        "norm_out.weight": src["norm_out.weight"],
        "conv_out.weight": src["conv_out.weight"],
        "conv_out.bias": src["conv_out.bias"],
    }
    for ckpt_idx, ref_leaf in ((0, "linear_1"), (2, "linear_2")):
        for leaf in ("weight", "bias"):
            out[f"t_embedder.mlp.{ckpt_idx}.{leaf}"] = src[f"t_embedder.timestep_embedder.{ref_leaf}.{leaf}"]

    for i in range(len(model.diff_blocks)):
        p = f"diff_blocks.{i}."
        for leaf in ("weight", "bias"):
            out[f"{p}attn.qkv.{leaf}"] = torch.cat([src[f"{p}attn.qkv.to_{n}.{leaf}"] for n in ("q", "k", "v")], dim=0)
            out[f"{p}attn.proj.{leaf}"] = src[f"{p}attn.proj.{leaf}"]
            out[f"{p}context_proj.{leaf}"] = src[f"{p}context_proj.{leaf}"]
        for key in (
            "attn.q_norm.weight",
            "attn.k_norm.weight",
            "mlp.w_gate.weight",
            "mlp.w_up.weight",
            "mlp.w_down.weight",
            "norm1.weight",
            "norm2.weight",
            "scale_shift_table",
        ):
            out[f"{p}{key}"] = src[f"{p}{key}"]
    return out


def make_inputs(config: DiffVAEStage5Config, grid: Grid, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    context = torch.randn(grid.batch, grid.t, grid.h, grid.w, config.context_channels, generator=generator)
    x_t = torch.randn(
        grid.batch,
        config.out_channels,
        grid.t,
        grid.h * config.patch_size,
        grid.w * config.patch_size,
        generator=generator,
    )
    timestep = torch.tensor([0.7] * grid.batch)
    return context, x_t, timestep


def tt_timestep(timestep: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(
        timestep.reshape(1, 1, -1, 1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )


def flat(x: torch.Tensor, channels: int) -> torch.Tensor:
    """``(B, T, H, W, C)`` -> the module's ``(1, B, sites, C)`` layout."""
    return x.reshape(1, x.shape[0], -1, channels)


# ---------------------------------------------------------------------------


def test_patch_packing_matches_upstream():
    """Guards the (c, w_sub, h_sub) channel order -- upstream's einops puts the W
    sub-index outside the H one, which reads backwards from the axis names."""
    config = DiffVAEStage5Config()
    x = torch.randn(2, config.out_channels, 5, 12, 16)

    ours = patchify(x, config.patch_size)
    theirs = ltx_ops.patchify(x, patch_size_hw=config.patch_size, patch_size_t=1)
    assert ours.shape == theirs.shape
    torch.testing.assert_close(ours, theirs)

    torch.testing.assert_close(unpatchify(ours, config.patch_size), x)
    torch.testing.assert_close(
        unpatchify(ours, config.patch_size),
        ltx_ops.unpatchify(theirs, patch_size_hw=config.patch_size, patch_size_t=1),
    )


def test_rope_dim_split_matches_upstream():
    from ltx_core.model.video_vae.transformer.rope_math import default_rope_dim_split as upstream

    for head_dim in (32, 64, 128, 256):
        assert default_rope_dim_split(head_dim) == upstream(head_dim)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_rope_matches_upstream(mesh_device: ttnn.MeshDevice):
    """The RoPE prelude in isolation: pair-swap matmul + fused cos/sin table against
    upstream's per-axis W-slabbed rotation."""
    from models.tt_dit.models.vae.diffvae_ltx_stage5 import (
        _build_rope_tables,
        _reshape_retiled,
        _rope_inv_freqs,
        _rope_pair_swap_matrix,
    )

    config = DiffVAEStage5Config()
    grid = GRID
    split = config.resolved_rope_dim_split
    torch.manual_seed(0)
    x = torch.randn(grid.batch, grid.t, grid.h, grid.w, config.num_heads, config.head_dim)

    expected = ltx_rope.apply_abs_rope(
        x,
        split,
        tuple(_rope_inv_freqs(d, config.rope_base) for d in split),
        num_tiles=4,
        compute_dtype=torch.float32,
    )

    tables = _build_rope_tables(
        grid,
        dim_split=split,
        base=config.rope_base,
        num_heads=config.num_heads,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    swap = ttnn.from_torch(
        _rope_pair_swap_matrix(config.head_dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    flat_shape = (1, grid.batch, grid.sites * config.num_heads, config.head_dim)
    tt_x = ttnn.from_torch(
        x.reshape(flat_shape).contiguous(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32
    )
    swapped = ttnn.matmul(tt_x, swap, compute_kernel_config=compute)
    got = ttnn.add(ttnn.multiply(tt_x, tables.cos), ttnn.multiply(swapped, tables.sin))
    got = _reshape_retiled(got, tuple(x.shape))

    assert_quality(expected, ttnn.to_torch(got), pcc=0.9999)


def test_checkpoint_keys_match_shipped_names():
    """The key set and shapes the loader must accept, spelled out independently of
    how the reference module happens to name its own parameters."""
    config = DiffVAEStage5Config()
    model = TorchStage5(config)
    state = checkpoint_state(model)

    dim, ctx, hidden, temb = config.dim, config.context_channels, config.mlp_hidden, config.t_emb_dim
    expected = {
        "conv_in_x_t.weight": (dim, config.patch_channels),
        "conv_in_x_t.bias": (dim,),
        "t_embedder.mlp.0.weight": (temb, 256),
        "t_embedder.mlp.0.bias": (temb,),
        "t_embedder.mlp.2.weight": (temb, temb),
        "t_embedder.mlp.2.bias": (temb,),
        "shared_adaln.proj.weight": (NUM_ADALN_CHUNKS * dim, temb),
        "shared_adaln.proj.bias": (NUM_ADALN_CHUNKS * dim,),
        "norm_out.weight": (dim,),
        "conv_out.weight": (config.patch_channels, dim),
        "conv_out.bias": (config.patch_channels,),
    }
    for i in range(config.num_blocks):
        p = f"diff_blocks.{i}."
        expected |= {
            f"{p}attn.qkv.weight": (3 * dim, dim),
            f"{p}attn.qkv.bias": (3 * dim,),
            f"{p}attn.proj.weight": (dim, dim),
            f"{p}attn.proj.bias": (dim,),
            f"{p}attn.q_norm.weight": (config.head_dim,),
            f"{p}attn.k_norm.weight": (config.head_dim,),
            f"{p}context_proj.weight": (dim, ctx),
            f"{p}context_proj.bias": (dim,),
            f"{p}mlp.w_gate.weight": (hidden, dim),
            f"{p}mlp.w_up.weight": (hidden, dim),
            f"{p}mlp.w_down.weight": (dim, hidden),
            f"{p}norm1.weight": (dim,),
            f"{p}norm2.weight": (dim,),
            f"{p}scale_shift_table": (NUM_ADALN_CHUNKS, dim),
        }

    assert set(state) == set(expected)
    assert {k: tuple(v.shape) for k, v in state.items()} == expected
    assert not [k for k in state if k.split(".")[-1].startswith("gate_")]


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
def test_unfolded_gates_are_rejected(mesh_device: ttnn.MeshDevice, expect_error):
    """The shipped checkpoint ships no gate tensors, but assuming that silently would
    decode a gated checkpoint wrong."""
    config = DiffVAEStage5Config(num_blocks=1)
    model = DiffVAEStage5(config, mesh_device=mesh_device)
    state = checkpoint_state(TorchStage5(config))
    state["diff_blocks.0.gate_msa"] = torch.ones(config.dim)

    with expect_error(ValueError, "unfolded static gates"):
        model.load_torch_state_dict(state)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("pcc", [0.999], ids=["pcc999"])
def test_stage5_parity(mesh_device: ttnn.MeshDevice, dtype: ttnn.DataType, pcc: float):
    config = DiffVAEStage5Config()
    grid = GRID

    reference = TorchStage5(config)
    reference.eval()
    randomize(reference, seed=1234)
    state = checkpoint_state(reference)

    model = DiffVAEStage5(config, mesh_device=mesh_device, dtype=dtype)
    model.load_torch_state_dict(state)

    context, x_t, timestep = make_inputs(config, grid, seed=99)
    ref_blocks, ref_pixels = reference.forward_diff_step(reference.build_buffer(context, x_t), timestep)

    tt_context = ttnn.from_torch(
        flat(context, config.context_channels).contiguous(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )
    tt_t = tt_timestep(timestep, mesh_device)

    buffer = ttnn.concat([tt_context, model.embed_x_t(x_t)], dim=-1)
    scaled_t = ttnn.multiply(tt_t, config.timestep_scale_multiplier)
    modulation = model.shared_adaln(model.t_embedder(scaled_t), grid.batch)
    tables = model.rope_tables(grid)

    tt_ctx_half = _slice_last(buffer, 0, config.context_channels)
    x = _slice_last(buffer, config.context_channels, config.context_channels + config.dim)

    failures = []
    for i, block in enumerate(model.diff_blocks):
        x = block(x, tt_ctx_half, modulation, grid, tables)
        logger.info(f"block {i}")
        try:
            assert_quality(flat(ref_blocks[i], config.dim), ttnn.to_torch(x), pcc=pcc)
        except Exception as err:  # noqa: BLE001
            failures.append(f"block {i}: {err}")

    logger.info("context half unchanged across all blocks")
    torch.testing.assert_close(
        ttnn.to_torch(tt_ctx_half).reshape(context.shape).to(torch.float32),
        ttnn.to_torch(tt_context).reshape(context.shape).to(torch.float32),
    )

    logger.info("final pixels")
    tt_pixels = model.forward(tt_context, x_t, tt_t, grid)
    try:
        assert_quality(ref_pixels, tt_pixels, pcc=pcc)
    except Exception as err:  # noqa: BLE001
        failures.append(f"pixels: {err}")

    assert not failures, "\n".join(failures)
