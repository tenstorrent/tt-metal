# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Weight loading for the device DFlash drafter: replicated, not tensor-parallel.

Every drafter tensor is replicated on every device and each device computes the whole drafter
redundantly. The drafter is small (1.73B params) and runs over a 16-token block, so it is bound by
collectives and dispatch rather than compute; replicating removes the per-layer all-reduces a
row-parallel ``o_proj`` / ``down_proj`` would need. The only collective left is one all-gather of
the target's taps, which arrive fractured on the hidden dim (see :func:`reorder_fc_rows`).

The one thing that stays TP-shaped is ``fc``'s row order: the tap all-gather produces device-major
columns, so ``fc``'s rows are permuted to match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig

#: bf8 for the attention projections and ``fc``; norms stay bf16 because they are tiny and a
#: per-channel gain is the last place to spend precision.
PROJ_DTYPE = ttnn.bfloat8_b
NORM_DTYPE = ttnn.bfloat16

#: bf4 for the MLP gate/up. The MLP is pure weight streaming, so halving the bytes halves its time.
#: Same choice the 27B target makes for its own gate/up.
MLP_DTYPE = ttnn.bfloat4_b

#: ``down_proj`` stays bf8: it projects back into the residual stream, so its error lands on the
#: stream every layer. bf4 here takes the drafter below the PCC gate in
#: ``tests/test_dflash_drafter_tp.py``. The target's ``load_mlp_weights`` makes the same split.
#: Greedy verification accepts only exact argmax matches, so weight precision can change acceptance
#: but never greedy output ids.
MLP_DOWN_DTYPE = ttnn.bfloat8_b


def reorder_fc_rows(fc_weight: torch.Tensor, cfg: DFlashDrafterConfig, tp: int) -> torch.Tensor:
    """Permute ``fc``'s rows into device-major order to match an all-gathered tap concat.

    ``fc`` consumes the concatenation of ``n`` target taps, each ``hidden_size`` wide, in tap-major
    order: ``[tap0(5120), tap1(5120), ...]``. But the taps arrive as the target's residual stream,
    already fractured on the hidden dim — device ``d`` holds columns ``[d*hf, (d+1)*hf)`` of every
    tap. The drafter concatenates its local slices (``[tap0_d, tap1_d, ..., tap4_d]``, ``n*hf`` wide)
    and all-gathers, which yields device-major columns:

        ``[tap0_d0 ... tap4_d0 | tap0_d1 ... tap4_d1 | ...]``

    That is not the order ``fc`` was trained on, so permute the weight's rows the same way once at
    load time rather than shuffling activations every step.

    Args:
        fc_weight: ``fc.weight`` as stored, ``[hidden_size, n*hidden_size]`` (out, in).

    Returns:
        ``[n*hidden_size, hidden_size]`` (in, out), rows in device-major order.
    """
    n_taps = len(cfg.target_layer_ids)
    hidden, hf = cfg.hidden_size, cfg.local_hidden(tp)
    assert fc_weight.shape == (hidden, n_taps * hidden), (
        f"fc.weight is {tuple(fc_weight.shape)}, expected {(hidden, n_taps * hidden)} — the tap count "
        f"({n_taps}) or hidden size disagrees with the checkpoint"
    )
    fc_in_out = fc_weight.T.contiguous()  # [n*hidden, hidden]
    order = torch.cat(
        [
            torch.cat([torch.arange(j * hidden + d * hf, j * hidden + (d + 1) * hf) for j in range(n_taps)])
            for d in range(tp)
        ]
    )
    return fc_in_out[order].contiguous()


@dataclass
class DrafterLayerWeights:
    """One drafter layer's uploaded tensors. All replicated, all full-width."""

    input_layernorm: ttnn.Tensor
    post_attention_layernorm: ttnn.Tensor
    q_proj: ttnn.Tensor
    #: ``k_proj`` and ``v_proj`` fused on the output dim, ``[hidden, 2*num_kv_heads*head_dim]``.
    #: Both read the same activation (``kv_src``), so one matmul serves both, and the resulting
    #: ``[k | v]`` column layout is exactly what ``nlp_create_qkv_heads(kv_tied=True)`` splits into
    #: head-major K and V in a single op — see :meth:`~..drafter.TtDFlashDrafter._kv_heads`.
    kv_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor
    k_norm: ttnn.Tensor
    gate_proj: ttnn.Tensor
    up_proj: ttnn.Tensor
    down_proj: ttnn.Tensor


@dataclass
class DrafterWeights:
    """Every drafter tensor, on device."""

    fc: ttnn.Tensor
    hidden_norm: ttnn.Tensor
    norm: ttnn.Tensor
    layers: list[DrafterLayerWeights] = field(default_factory=list)


def load_drafter_weights(
    mesh_device,
    state_dict: dict,
    cfg: DFlashDrafterConfig,
    *,
    cache_path=None,
) -> DrafterWeights:
    """Upload the drafter checkpoint, replicated across the mesh.

    ``state_dict`` is the raw checkpoint (see :func:`~.config.load_drafter_state_dict`) — key names
    are used verbatim, so a mismatch surfaces here as a ``KeyError`` rather than as silently wrong
    drafts.
    """
    tp = mesh_device.get_num_devices()
    multi = tp > 1

    def _upload(tensor, *, dtype, name=None):
        return ttnn.as_tensor(
            tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(str(cache_path / name) if (cache_path is not None and name) else None),
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if multi else {}),
        )

    def _norm(key):
        # Standard RMSNorm, not the target's zero-centered "+1 fold": the drafter is a Qwen3 model
        # and its gains are used as-is. Pre-offsetting these by +1 would be a silent accuracy bug.
        return _upload(state_dict[key], dtype=NORM_DTYPE, name=f"dflash.{key}")

    def _proj(key, name, dtype=None):
        # ttnn.linear wants [in, out]; the checkpoint stores [out, in].
        return _upload(state_dict[key].T.contiguous(), dtype=dtype or PROJ_DTYPE, name=name)

    def _kv_proj(p):
        # K then V on the output dim, the order the tied head-split reads them in. Both consume the
        # same kv_src rows, so one matmul serves both.
        kv = torch.cat(
            [state_dict[f"{p}self_attn.k_proj.weight"].T, state_dict[f"{p}self_attn.v_proj.weight"].T], dim=-1
        )
        return _upload(kv.contiguous(), dtype=PROJ_DTYPE, name=f"dflash.{p}kv_proj")

    weights = DrafterWeights(
        # Rows permuted to match the gathered tap order, then replicated like everything else.
        fc=_upload(
            reorder_fc_rows(state_dict["fc.weight"], cfg, tp),
            dtype=PROJ_DTYPE,
            name=f"dflash.fc.gathered.tp{tp}",
        ),
        hidden_norm=_norm("hidden_norm.weight"),
        norm=_norm("norm.weight"),
    )

    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        weights.layers.append(
            DrafterLayerWeights(
                input_layernorm=_norm(f"{p}input_layernorm.weight"),
                post_attention_layernorm=_norm(f"{p}post_attention_layernorm.weight"),
                q_proj=_proj(f"{p}self_attn.q_proj.weight", f"dflash.{p}q_proj"),
                kv_proj=_kv_proj(p),
                o_proj=_proj(f"{p}self_attn.o_proj.weight", f"dflash.{p}o_proj"),
                q_norm=_norm(f"{p}self_attn.q_norm.weight"),
                k_norm=_norm(f"{p}self_attn.k_norm.weight"),
                gate_proj=_proj(f"{p}mlp.gate_proj.weight", f"dflash.{p}gate", dtype=MLP_DTYPE),
                up_proj=_proj(f"{p}mlp.up_proj.weight", f"dflash.{p}up", dtype=MLP_DTYPE),
                down_proj=_proj(f"{p}mlp.down_proj.weight", f"dflash.{p}down", dtype=MLP_DOWN_DTYPE),
            )
        )
    return weights
