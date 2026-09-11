# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Weight loading for the device DFlash drafter: **replicated, not tensor-parallel**.

Every drafter tensor is replicated on every device and each device computes the whole drafter
redundantly. That sounds wasteful and is the opposite of what the 27B target does, so it is worth
saying why.

The drafter is 1.73B params — ~1.8 GB per device in bf8 — and a speculative step runs it over a
16-token block: roughly 55 GFLOP of arithmetic. It is nowhere near compute- or memory-bound. What
it *was* bound by, when this was tensor-parallel, is **collectives and dispatch**: a row-parallel
``o_proj`` and ``down_proj`` need an all-reduce each, and on a ``(1, N)`` mesh
``tt_all_reduce`` is a reduce-scatter *plus* a gather — so 4 CCL ops per layer, 22 per step,
against arithmetic that a CPU does in 0.38 s. Measured on T3K, the TP drafter spent 45% of a
speculative run's wall clock in ``propose`` and came out at 0.70x the host PyTorch drafter.

Replicating removes every one of those collectives except a single all-gather of the target's taps,
which the drafter needs anyway because the taps arrive fractured (see :func:`reorder_fc_rows`).
Redundant compute is free here; latency is not.

The one thing that stays TP-shaped is that ``fc`` row order: the tap all-gather produces
**device-major** columns, so ``fc``'s rows have to be permuted to match.
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

#: bf4 for the MLP gate/up. The MLP was **55 % of the drafter's step** and is pure weight streaming,
#: so half the bytes is half the time. Isolated
#: (``tests/perf/test_dflash_mlp_matmul_sweep.py``; see ``_layer_mlp``'s docstring for why no
#: program config can do this instead), then in the real 5-layer profile:
#:
#:                isolated        in-model (T3K, ctx64)
#:     gate    490 -> 278 us      490 -> 264 us
#:     up      469 -> 260 us      469 -> 266 us
#:     down    447 -> 358 us      NOT TAKEN, see below
#:
#: Whole step: **12.86 -> 10.68 ms (-2,180 us, -17 %)**, and the MLP from 55 % to 46 % of it. This
#: is the same choice the 27B target makes for its own gate/up. It costs accuracy — see
#: :data:`MLP_DOWN_DTYPE` for the one weight where that cost was not worth paying.
MLP_DTYPE = ttnn.bfloat4_b

#: ``down_proj`` stays **bf8**. It is the projection back into the residual stream, so its error
#: lands on the stream every layer rather than inside the SwiGLU, and it is the cheapest of the
#: three to leave alone (bf4 buys only -20 % there, against -43/-45 % on gate/up).
#:
#: MEASURED on the full 5-layer drafter against the fp32 host reference
#: (``tests/test_dflash_drafter_tp.py``, which gates at 0.99):
#:
#:     all bf8                  0.9978   (single layer 0.9989, step-2 0.9982)
#:     gate/up bf4, down bf8    0.9942   (single layer 0.9960, step-2 0.9949)   <- shipped
#:     all three bf4            0.9891   (single layer 0.9916, step-2 0.9901)   <- FAILS the gate
#:
#: So bf4 on ``down`` is not a judgement call, it is out: it alone costs 0.0051 PCC and takes the
#: drafter under the gate. The target model's ``load_mlp_weights`` makes the same split.
#:
#: The drafter's real metric is ACCEPTANCE, not PCC, and it HAS been measured
#: (``tests/reference/test_dflash_acceptance.py``, full 27B on T3K, greedy, 3 prompts x 96 tokens):
#:
#:     all bf8                 5.089 tok/step   (56 steps)
#:     gate/up bf4 (shipped)   5.089 tok/step   (56 steps)
#:
#: i.e. **no measurable acceptance cost** — and greedy output ids are bit-identical, which they must
#: be, since the target verifies every slot. So the PCC drop 0.9978 -> 0.9942 buys -2,180 us of a
#: 12,860 us step for nothing measurable. Caveats on that number are in the test's docstring: it is
#: 56 steps (a +/-1-step swing per prompt is the resolution) and GREEDY only. Under sampling,
#: rejection sampling reads the draft *probabilities* rather than just the argmax, so bf4 could
#: move both acceptance and the output distribution there; that is unmeasured.
MLP_DOWN_DTYPE = ttnn.bfloat8_b


def reorder_fc_rows(fc_weight: torch.Tensor, cfg: DFlashDrafterConfig, tp: int) -> torch.Tensor:
    """Permute ``fc``'s rows into device-major order to match an all-gathered tap concat.

    ``fc`` consumes the concatenation of ``n`` target taps, each ``hidden_size`` wide, in tap-major
    order: ``[tap0(5120), tap1(5120), ...]``. But the taps arrive as the target's residual stream,
    already fractured on the hidden dim — device ``d`` holds columns ``[d*hf, (d+1)*hf)`` of *every*
    tap. The drafter concatenates its local slices (``[tap0_d, tap1_d, ..., tap4_d]``, ``n*hf`` wide)
    and all-gathers, which yields **device-major** columns:

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
    proj_dtype=PROJ_DTYPE,
    mlp_dtype=MLP_DTYPE,
    mlp_down_dtype=MLP_DOWN_DTYPE,
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
        # Standard RMSNorm, NOT the target's zero-centered "+1 fold": the drafter is a Qwen3 model
        # and its gains are used as-is. Pre-offsetting these by +1 would be a silent accuracy bug.
        return _upload(state_dict[key], dtype=NORM_DTYPE, name=f"dflash.{key}")

    def _proj(key, name, dtype=None):
        # ttnn.linear wants [in, out]; the checkpoint stores [out, in].
        return _upload(state_dict[key].T.contiguous(), dtype=dtype or proj_dtype, name=name)

    def _kv_proj(p):
        # K then V on the output dim, the order the tied head-split reads them in. Fusing is free
        # here: both consume the same kv_src rows, so this is one launch instead of two.
        kv = torch.cat(
            [state_dict[f"{p}self_attn.k_proj.weight"].T, state_dict[f"{p}self_attn.v_proj.weight"].T], dim=-1
        )
        return _upload(kv.contiguous(), dtype=proj_dtype, name=f"dflash.{p}kv_proj")

    weights = DrafterWeights(
        # Rows permuted to match the gathered tap order, then replicated like everything else.
        fc=_upload(
            reorder_fc_rows(state_dict["fc.weight"], cfg, tp),
            dtype=proj_dtype,
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
                gate_proj=_proj(f"{p}mlp.gate_proj.weight", f"dflash.{p}gate", dtype=mlp_dtype),
                up_proj=_proj(f"{p}mlp.up_proj.weight", f"dflash.{p}up", dtype=mlp_dtype),
                down_proj=_proj(f"{p}mlp.down_proj.weight", f"dflash.{p}down", dtype=mlp_down_dtype),
            )
        )
    return weights
