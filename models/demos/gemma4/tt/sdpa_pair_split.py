# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GEMMA4_SDPA_PAIR_SPLIT``: the decode attention of a global layer split over the two chips that hold one KV head
-- the device-free part (pure torch, no ttnn import, so the host tests load it alone).

On a one-row mesh with KV heads replicated across TP (the 31B on the Galaxy (1,8) row: 32 q / 4 KV heads, TP=8,
device column ``i`` holds KV head ``i // 2`` and q heads ``[4i, 4i + 4)``), columns ``2g`` and ``2g + 1`` hold the
same KV head and today both read its whole KV sequence every decode step. Under the pair split each chip of the
pair ``g`` runs the attention of ALL ``2 * Hq / tp`` q heads of the KV group (``8g .. 8g + 7``) over HALF the key
chunks (half ``c % 2``, ``SdpaDecodeLse(pair_split=True)``), normalised over that half, with its row statistics.
The two halves are merged through their log-sum-exp: ``w_c = sigmoid(lse_c - lse_{c^1})``, and each chip feeds
``w_c * O_c`` through the o_proj rows of the group's q heads; the pair's two products add up to the group's
attention @ o_proj, the ``tp / 2`` pairs cover every q head, and the row all-reduce that already follows the o_proj
sums them (``pair_merge_reference`` below emulates the device graph).

Layouts (``pair_qkv_weight`` / ``pair_o_proj_weight``): device column ``i`` (pair ``g = i // 2``) gets the fused
``[Wq of q heads g*G .. g*G + G - 1 (natural order) | Wk of head g | Wv of head g]`` (``G = Hq / Hkv``; under K=V
tying ``Wv = Wk``), width ``(G + 2) * D`` (5120 for the 31B), and the o_proj rows of the same ``G`` heads,
``[G * D, hidden]`` (4096 x 5376): the TP=4 per-chip shapes of the 31B. The 4-head weights stay (prefill, the spec
verify and the MTP drafter use them); the pair copies cost ``(5376 x 5120 + 4096 x 5376)`` bfloat8_b per global layer
per chip, ~52 MB x 10 global layers on the 31B.
"""

from typing import Optional

import torch
from loguru import logger

from models.demos.gemma4.tt.mesh_orientation import SDPA_PAIR_SPLIT_ENV, decode_flag

PAIR_SPLIT_ENV = SDPA_PAIR_SPLIT_ENV
# What an empty half writes (SdpaDecodeLse pair mode): O = 0, m = EMPTY_HALF_M, l = 1 -- finite, so its merge weight
# is ~0 and no NaN reaches the merge.
EMPTY_HALF_M = -30000.0
# |lse_self - lse_partner| is clamped to this before the sigmoid: an empty half puts the difference at ~3e4, and
# exp(3e4) overflows fp32 inside the sigmoid's 1 / (1 + exp(-x)). sigmoid(64) = 1 - 1.6e-28.
LOGIT_CLAMP = 64.0
# The stats tensor of SdpaDecodeLse: [1, B, PNH, 64] with m at column 0 and l at column 32.
STATS_M_COL = 0
STATS_L_COL = 32

_LOGGED = set()


def log_once(message: str):
    """One INFO line per distinct message (the build and the decode call sites log why a layer keeps the 4-head
    path)."""
    if message not in _LOGGED:
        _LOGGED.add(message)
        logger.info(f"{PAIR_SPLIT_ENV}: {message}")


def pair_split_enabled() -> bool:
    """The knob: ``GEMMA4_SDPA_PAIR_SPLIT=1`` (``decode_flag``: the env when set, else the build profile; no profile
    turns it on, so it is off by default)."""
    return decode_flag(PAIR_SPLIT_ENV) == "1"


def pair_split_mesh(mesh_shape) -> bool:
    """A mesh the pair split can run on: one row with an even number of columns."""
    shape = tuple(int(d) for d in mesh_shape) if mesh_shape is not None else ()
    return len(shape) == 2 and shape[0] == 1 and shape[1] >= 2 and shape[1] % 2 == 0


def pair_split_reason(
    *,
    is_sliding: bool,
    num_q_heads: int,
    num_kv_heads: int,
    tp: int,
    mesh_shape,
    tp_axis: int = 1,
    kv_replicated: bool,
    fractured: bool = False,
) -> Optional[str]:
    """None when a layer's decode attention can take the pair split, else why not: a global layer on a one-row mesh
    with even columns and TP over them, KV heads replicated across TP, and the two chips of a pair together owning
    exactly one KV group (``Hq / Hkv == 2 * Hq / tp``)."""
    if is_sliding:
        return "sliding layer"
    shape = tuple(int(d) for d in mesh_shape) if mesh_shape is not None else ()
    if not pair_split_mesh(shape):
        return f"mesh {shape} is not one row with an even number of columns"
    if fractured:
        return "weight_fracture build"
    if int(tp_axis) != 1 or int(tp) != shape[1]:
        return f"TP {tp} on axis {tp_axis} is not the row's {shape[1]} columns"
    if not kv_replicated:
        return "KV heads are not replicated across TP"
    nq, nkv, tp = int(num_q_heads), int(num_kv_heads), int(tp)
    if nkv <= 0 or nq % nkv or nq % tp:
        return f"{nq} q heads do not split over {nkv} KV heads and TP {tp}"
    if nq // nkv != 2 * (nq // tp):
        return f"a pair holds {2 * (nq // tp)} q heads, a KV group has {nq // nkv}"
    return None


def pair_heads(num_q_heads: int, num_kv_heads: int) -> int:
    """q heads per chip under the pair split: the whole KV group."""
    return int(num_q_heads) // int(num_kv_heads)


def pair_qkv_weight(q_w, k_w, v_w, *, num_q_heads: int, num_kv_heads: int, head_dim: int, tp: int) -> torch.Tensor:
    """The fused pair QKV weight ``[hidden, tp * (G + 2) * D]`` from the HF ``q_proj [Hq*D, hidden]``, ``k_proj`` /
    ``v_proj [Hkv*D, hidden]``: column block ``i`` (sharded to device column ``i`` by the column-parallel mapper) is
    ``[Wq of the G q heads of KV group i // 2 | Wk of head i // 2 | Wv of head i // 2]``, transposed."""
    group = pair_heads(num_q_heads, num_kv_heads)
    d = int(head_dim)
    blocks = []
    for i in range(int(tp)):
        g = i // 2
        wq = q_w[g * group * d : (g + 1) * group * d].transpose(-2, -1)
        wk = k_w[g * d : (g + 1) * d].transpose(-2, -1)
        wv = v_w[g * d : (g + 1) * d].transpose(-2, -1)
        blocks.append(torch.cat([wq, wk, wv], dim=-1))
    return torch.cat(blocks, dim=-1)


def pair_o_proj_weight(o_w_t, *, num_q_heads: int, num_kv_heads: int, head_dim: int, tp: int) -> torch.Tensor:
    """The pair o_proj weight ``[tp * G * D, N]`` from the transposed HF ``o_proj`` ``[Hq*D, N]`` (N = hidden, padded
    as the caller pads it): row block ``i`` (sharded to device column ``i`` by the row-parallel mapper) is the rows of
    the G q heads of KV group ``i // 2``."""
    group = pair_heads(num_q_heads, num_kv_heads)
    d = int(head_dim)
    return torch.cat([o_w_t[(i // 2) * group * d : (i // 2 + 1) * group * d] for i in range(int(tp))], dim=-2)


def pair_selector(tp: int) -> torch.Tensor:
    """``[1, tp * tp, 1, 1]`` fp32, sharded on dim 1 over the row's columns: device column ``c`` holds the row
    ``e_c - e_{c^1}`` (``+1`` at itself, ``-1`` at its partner ``c ^ 1``), so the sum over the gathered device dim of
    ``gathered x selector`` is ``lse_self - lse_partner`` in one product and one fp32 reduction."""
    tp = int(tp)
    sel = torch.zeros(tp, tp, dtype=torch.float32)
    for c in range(tp):
        sel[c, c] = 1.0
        sel[c, c ^ 1] = -1.0
    return sel.reshape(1, tp * tp, 1, 1)


def lse_from_m_l(m, l, scale: float = 1.0):
    """``scale * m + ln(l)`` in fp32 (``m``: the raw-score row max, ``l``: the row sum at that max)."""
    return float(scale) * m.float() + torch.log(l.float())


def pair_merge_reference(out, m, l, *, scale: float = 1.0) -> torch.Tensor:
    """Torch emulation of the device merge over the row: ``out [tp, B, H, D]`` each device's half-normalised output,
    ``m`` / ``l [tp, B, H]`` its statistics. Returns ``[tp, B, H, D]``, what each device feeds its pair o_proj:
    ``lse`` per device, the all-gather (every device sees all ``tp``), ``sum_j gathered_j * selector_c[j]`` =
    ``lse_c - lse_{c^1}``, clamped, ``w_c = sigmoid(.)``, ``w_c * out_c``."""
    tp = int(out.shape[0])
    lse = lse_from_m_l(m, l, scale)  # [tp, B, H]
    sel = pair_selector(tp).reshape(tp, tp)  # row c: device c's selector
    d = torch.einsum("cj,jbh->cbh", sel, lse)
    w = torch.sigmoid(d.clamp(-LOGIT_CLAMP, LOGIT_CLAMP))
    return out.float() * w[..., None]
