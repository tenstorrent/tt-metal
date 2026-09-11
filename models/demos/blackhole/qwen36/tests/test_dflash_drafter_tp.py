# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: the ttnn DFlash drafter (:class:`TtDFlashDrafter`) vs the host reference ``DFlashDraftModel``.

The host drafter is already validated end-to-end against the real 27B, so it is the oracle here.
Both sides get the SAME real ``z-lab/Qwen3.6-27B-DFlash`` weights and the same inputs.

The tests form a ladder, so a regression localises instead of just saying "the drafter is wrong":

1. ``test_fc_projection`` — the ``fc`` + ``hidden_norm`` tap projection alone. This is the only
   place the drafter consumes a *fractured* input, and the row permutation that makes a plain
   ``dim=0`` shard line up with the taps is the subtlest part of the port.
2. ``test_single_layer`` — one layer, parametrized over both layer types. Separates the causal
   sliding mask from the bidirectional (unmasked) one, which are easy to swap by accident.
3. ``test_full_drafter`` — all 5 layers.
4. ``test_context_accumulates`` — two consecutive steps, which is what exercises the drafter's KV
   history (and would catch the context-starvation bug the host path had).

Run:
    MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      pytest -svq models/demos/blackhole/qwen36/tests/test_dflash_drafter_tp.py
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.config import (
    DFlashDrafterConfig,
    load_drafter_state_dict,
    resolve_drafter_path,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter

# bf8 projection weights against an fp32 host reference; the target's own TP attention prefill test
# lands at ~0.999 on the same arithmetic, so 0.99 is a loose-but-meaningful gate.
PCC = 0.99


@pytest.fixture(scope="module")
def drafter_bits():
    """``(cfg, state_dict)`` for the real drafter checkpoint; skips when it cannot be resolved."""
    try:
        path = resolve_drafter_path()
    except Exception as e:  # noqa: BLE001 — a missing checkpoint is a skip, not a failure
        pytest.skip(f"drafter checkpoint unavailable ({type(e).__name__}: {e})")
    return DFlashDrafterConfig.from_pretrained(path), load_drafter_state_dict(path)


def _trim(cfg: DFlashDrafterConfig, state_dict: dict, n_layers: int, layer_types=None):
    """Cut the drafter down to its first ``n_layers``, optionally forcing their types."""
    types = tuple(layer_types) if layer_types is not None else cfg.layer_types[:n_layers]
    assert len(types) == n_layers
    small = replace(cfg, num_hidden_layers=n_layers, layer_types=types)
    kept = {k: v for k, v in state_dict.items() if not k.startswith("layers.") or int(k.split(".")[1]) < n_layers}
    return small, kept


def _host_drafter(cfg: DFlashDrafterConfig, state_dict: dict):
    """The reference ``DFlashDraftModel`` for this (possibly trimmed) config, fp32 on CPU."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from models.demos.blackhole.qwen36.reference.dflash.dflash import DFlashDraftModel

    hf = Qwen3Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        use_sliding_window=True,
        sliding_window=cfg.sliding_window,
        layer_types=list(cfg.layer_types),
    )
    hf.num_target_layers = cfg.num_target_layers
    hf.dflash_config = {"target_layer_ids": list(cfg.target_layer_ids), "mask_token_id": cfg.mask_token_id}
    hf.block_size = cfg.block_size
    hf._attn_implementation = "sdpa"

    model = DFlashDraftModel(hf)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected drafter tensors: {sorted(unexpected)[:5]}"
    assert not missing, f"missing drafter tensors: {sorted(missing)[:5]}"
    return model.float().eval()


def _upload_taps(mesh_device, ctx: torch.Tensor, cfg: DFlashDrafterConfig):
    """Split the ``[1, S, n*H]`` tap concat into ``n`` per-tap tensors, each fractured on hidden.

    This is what the target hands over: its residual stream at each tap layer, already sharded on
    the hidden dim across the mesh.
    """
    hidden = cfg.hidden_size
    multi = mesh_device.get_num_devices() > 1
    taps = []
    for j in range(len(cfg.target_layer_ids)):
        piece = ctx[:, :, j * hidden : (j + 1) * hidden].reshape(1, 1, ctx.shape[1], hidden)
        taps.append(
            ttnn.from_torch(
                piece.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **(dict(mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1)) if multi else {}),
            )
        )
    return taps


def _upload_replicated(mesh_device, x: torch.Tensor):
    multi = mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        x.reshape(1, 1, x.shape[-2], x.shape[-1]).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if multi else {}),
    )


def _download(mesh_device, t: ttnn.Tensor) -> torch.Tensor:
    """Read a replicated device tensor back as ``[1, S, W]`` fp32 (replica 0)."""
    if mesh_device.get_num_devices() > 1:
        host = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    else:
        host = ttnn.to_torch(t)
    return host.reshape(-1, host.shape[-1]).float().unsqueeze(0)


def _inputs(cfg: DFlashDrafterConfig, ctx_len: int, q_len: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    ctx = torch.randn(1, ctx_len, cfg.target_feature_size, generator=gen) * 0.05
    noise = torch.randn(1, q_len, cfg.hidden_size, generator=gen) * 0.05
    return ctx, noise


# ------------------------------------------------------------------------------------------------


@torch.no_grad()
@parametrize_mesh_tp()
def test_fc_projection(mesh_device, drafter_bits, reset_seeds, ensure_gc):
    """``project_taps`` must match ``hidden_norm(fc(concat(taps)))``.

    The device consumes the taps already fractured on the hidden dim, so ``fc``'s rows are permuted
    into device-major order (see :func:`~...tt.dflash.weights.reorder_fc_rows`). Get that permutation
    wrong and this is the only test that says so clearly — everything downstream just degrades.
    """
    cfg, sd = drafter_bits
    cfg1, sd1 = _trim(cfg, sd, 1)
    host = _host_drafter(cfg1, sd1)
    tt = TtDFlashDrafter(mesh_device, cfg1, sd1)

    ctx, _ = _inputs(cfg, ctx_len=64, q_len=cfg.block_size)
    golden = host.hidden_norm(host.fc(ctx))

    taps = _upload_taps(mesh_device, ctx, cfg)
    got = _download(mesh_device, tt.project_taps(taps))

    passing, out = comp_pcc(golden, got, PCC)
    logger.info(f"fc + hidden_norm: {out}")
    assert passing, f"tap projection diverged — suspect reorder_fc_rows: {out}"


@torch.no_grad()
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "bidirectional"])
@parametrize_mesh_tp()
def test_single_layer(mesh_device, drafter_bits, layer_type, reset_seeds, ensure_gc):
    """One layer, each attention regime separately.

    ``sliding_attention`` is causal within a 2048 window; ``full_attention`` is **bidirectional** —
    that is the layer that lets the masked slots see each other. Swapping the two is an easy and
    completely silent mistake, so they are checked apart.
    """
    cfg, sd = drafter_bits
    cfg1, sd1 = _trim(cfg, sd, 1, layer_types=(layer_type,))
    host = _host_drafter(cfg1, sd1)
    tt = TtDFlashDrafter(mesh_device, cfg1, sd1)

    ctx_len, q_len = 64, cfg.block_size
    ctx, noise = _inputs(cfg, ctx_len, q_len)
    golden = host(
        target_hidden=ctx,
        noise_embedding=noise,
        position_ids=torch.arange(ctx_len + q_len)[None],
    )

    kv = tt.project_taps(_upload_taps(mesh_device, ctx, cfg))
    got = _download(mesh_device, tt.forward(kv, _upload_replicated(mesh_device, noise), start=ctx_len))

    passing, out = comp_pcc(golden, got, PCC)
    logger.info(f"single {layer_type} layer: {out}")
    assert passing, f"{layer_type} layer diverged: {out}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_full_drafter(mesh_device, drafter_bits, reset_seeds, ensure_gc):
    """All 5 layers: 4 sliding then 1 bidirectional, the real checkpoint's own ``layer_types``."""
    cfg, sd = drafter_bits
    host = _host_drafter(cfg, sd)
    tt = TtDFlashDrafter(mesh_device, cfg, sd)

    ctx_len, q_len = 64, cfg.block_size
    ctx, noise = _inputs(cfg, ctx_len, q_len)
    golden = host(
        target_hidden=ctx,
        noise_embedding=noise,
        position_ids=torch.arange(ctx_len + q_len)[None],
    )

    kv = tt.project_taps(_upload_taps(mesh_device, ctx, cfg))
    got = _download(mesh_device, tt.forward(kv, _upload_replicated(mesh_device, noise), start=ctx_len))

    passing, out = comp_pcc(golden, got, PCC)
    logger.info(f"full {cfg.num_hidden_layers}-layer drafter: {out}")
    assert passing, f"full drafter diverged: {out}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_context_accumulates(mesh_device, drafter_bits, reset_seeds, ensure_gc):
    """Two consecutive steps: the drafter must carry context, not just the newest taps.

    Step 2 sees only 4 new tap rows but must still attend over all 68 context positions. On host
    that means a KV cache truncated to ``start``; on device it is the per-layer K/V history. Feeding
    step 2 without the first step's context is the exact bug the host path shipped with, and it
    fails nothing else — the drafter simply drafts worse.
    """
    from transformers import DynamicCache

    from models.demos.blackhole.qwen36.reference.dflash.generate import truncate_kv

    cfg, sd = drafter_bits
    host = _host_drafter(cfg, sd)
    tt = TtDFlashDrafter(mesh_device, cfg, sd)

    q_len = cfg.block_size
    first_ctx, first_noise = _inputs(cfg, 64, q_len, seed=0)
    second_ctx, second_noise = _inputs(cfg, 4, q_len, seed=1)
    start1, start2 = 64, 68

    cache = DynamicCache(config=host.config)
    host(
        target_hidden=first_ctx,
        noise_embedding=first_noise,
        position_ids=torch.arange(start1 + q_len)[None],
        past_key_values=cache,
        use_cache=True,
    )
    truncate_kv(cache, start1)
    golden = host(
        target_hidden=second_ctx,
        noise_embedding=second_noise,
        position_ids=torch.arange(start1, start2 + q_len)[None],
        past_key_values=cache,
        use_cache=True,
    )

    tt.reset()
    tt.forward(
        tt.project_taps(_upload_taps(mesh_device, first_ctx, cfg)),
        _upload_replicated(mesh_device, first_noise),
        start=start1,
    )
    assert tt.context_len == start1, f"drafter committed {tt.context_len} context rows, expected {start1}"
    got = _download(
        mesh_device,
        tt.forward(
            tt.project_taps(_upload_taps(mesh_device, second_ctx, cfg)),
            _upload_replicated(mesh_device, second_noise),
            start=start2,
        ),
    )
    assert tt.context_len == start2

    passing, out = comp_pcc(golden, got, PCC)
    logger.info(f"step 2 with carried context: {out}")
    assert passing, f"drafter context accumulation diverged: {out}"
