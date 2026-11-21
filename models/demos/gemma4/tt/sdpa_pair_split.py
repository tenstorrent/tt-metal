# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GEMMA4_SDPA_PAIR_SPLIT``: the decode attention of a global layer split over the two chips that hold one KV head
-- the device-free part (pure torch, no ttnn import, so the host tests load it alone).

On a one-row mesh with KV heads replicated across TP (the 31B on the Galaxy (1,8) row: 32 q / 4 KV heads, TP=8,
device column ``i`` holds KV head ``i // 2`` and q heads ``[4i, 4i + 4)``), columns ``2g`` and ``2g + 1`` hold the
same KV head and, without the split, both read its whole KV sequence every decode step. Under the pair split each chip of the
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

``GEMMA4_SDPA_GROUP_SPLIT`` (default off): the same view over the ``G = tp / Hkv`` chips that hold each global KV
head of a line mesh (one row or one column, TP over its devices, ``group_split_parts``): the 12B on the (1,4) row (one
global KV head, 16 q heads: G = 4, every chip the whole q-head set) and the 26B-A4B on the (4,1) column (two KV heads:
G = 2 over mesh axis 0, chips (0, 1) and (2, 3)). Device ``i`` of the TP axis holds KV group ``i // G`` and attends
part ``i % G`` of every row's key chunks (``SdpaDecodeLse(pair_split=True, split_parts=G, split_axis=tp_axis)``);
its view's QKV / o_proj are ``pair_qkv_weight`` / ``pair_o_proj_weight`` (the chip's whole KV group: Hq / Hkv q
heads). The merge: G = 2 is the pair's sigmoid (the selector picks ``lse_self - lse_partner``); G > 2 gives each chip
``w_c = 1 / sum_{j in group(c)} exp(clamp(lse_j - lse_c))`` (``group_merge_reference``), the gathered LSEs masked to
the chip's group by ``group_mask`` (no mask when G = tp). The 12B's copies are its TP=1 shapes: (3840 x 9216 + 8192
x 3840) bfloat8_b = ~71 MB per global layer per chip (+50 MB over the 4-head pair), 8 global layers.
"""

from typing import Optional

import torch
from loguru import logger

from models.demos.gemma4.tt.mesh_orientation import SDPA_GROUP_SPLIT_ENV, SDPA_PAIR_SPLIT_ENV, decode_flag

PAIR_SPLIT_ENV = SDPA_PAIR_SPLIT_ENV
GROUP_SPLIT_ENV = SDPA_GROUP_SPLIT_ENV
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


def log_once(message: str, env: str = PAIR_SPLIT_ENV):
    """One INFO line per distinct message (the build and the decode call sites log why a layer keeps the 4-head
    path), prefixed with the knob (``env``) that asked for the split."""
    if (env, message) not in _LOGGED:
        _LOGGED.add((env, message))
        logger.info(f"{env}: {message}")


def pair_split_enabled() -> bool:
    """The knob: ``GEMMA4_SDPA_PAIR_SPLIT=1`` (``decode_flag``: the env when set, else the build profile: on in
    ``GALAXY_LINE_PAIR_ON``, the 31B on the (1,8) line; off elsewhere)."""
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


def group_split_enabled() -> bool:
    """The knob: ``GEMMA4_SDPA_GROUP_SPLIT=1`` (``decode_flag``: the env when set, else the build profile; off in
    every profile)."""
    return decode_flag(GROUP_SPLIT_ENV) == "1"


def group_split_mesh(mesh_shape) -> bool:
    """A mesh the group split can run on: a line of at least two devices (one row or one column)."""
    shape = tuple(int(d) for d in mesh_shape) if mesh_shape is not None else ()
    return len(shape) == 2 and min(shape) == 1 and max(shape) >= 2


def group_split_parts(
    *,
    is_sliding: bool,
    num_q_heads: int,
    num_kv_heads: int,
    tp: int,
    mesh_shape,
    tp_axis: int = 1,
    kv_replicated: bool,
    fractured: bool = False,
):
    """``(G, None)`` when a layer's decode attention can take the group split over the ``G = tp / Hkv >= 2`` chips
    holding each of its KV heads, else ``(0, why not)``: a global layer on a line mesh with TP over the line's devices
    (either axis), KV heads replicated across TP and ``Hkv`` dividing both ``Hq`` and ``tp``."""
    if is_sliding:
        return 0, "sliding layer"
    shape = tuple(int(d) for d in mesh_shape) if mesh_shape is not None else ()
    if not group_split_mesh(shape):
        return 0, f"mesh {shape} is not a line (one row or one column) of at least 2 devices"
    if fractured:
        return 0, "weight_fracture build"
    axis = int(tp_axis)
    if axis not in (0, 1) or int(tp) != shape[axis]:
        return 0, f"TP {tp} on axis {tp_axis} is not the line's {max(shape)} devices"
    if not kv_replicated:
        return 0, "KV heads are not replicated across TP"
    nq, nkv, tp = int(num_q_heads), int(num_kv_heads), int(tp)
    if nkv <= 0 or nq % nkv or nq % tp or tp % nkv:
        return 0, f"{nq} q heads, {nkv} KV heads and TP {tp} do not nest"
    parts = tp // nkv
    if parts < 2:
        return 0, f"each KV head sits on {parts} chip"
    return parts, None


def group_mask(tp: int, parts: int) -> torch.Tensor:
    """``[1, tp * tp, 1, 1]`` fp32, sharded on dim 1 over the TP axis: device ``c`` holds the row with 1 at the
    ``parts`` devices of its KV group (``c // parts``) and 0 elsewhere, the mask of the gathered LSE terms in the
    group merge (all ones when ``parts == tp``: the merge then takes no mask)."""
    tp, parts = int(tp), int(parts)
    mask = torch.zeros(tp, tp, dtype=torch.float32)
    for c in range(tp):
        g = c // parts
        mask[c, g * parts : (g + 1) * parts] = 1.0
    return mask.reshape(1, tp * tp, 1, 1)


def group_merge_reference(out, m, l, *, parts: int, scale: float = 1.0) -> torch.Tensor:
    """Torch emulation of the device group merge (G = ``parts`` > 2) over the TP axis: ``out [tp, B, H, D]`` each
    device's part-normalised output, ``m`` / ``l [tp, B, H]`` its statistics. Returns ``[tp, B, H, D]``, what each
    device feeds its group o_proj: ``lse`` per device, the all-gather, ``d_cj = clamp(lse_j - lse_c)``, ``S_c =
    sum_j mask_cj exp(d_cj)``, ``w_c = 1 / S_c`` (= ``exp(lse_c - logsumexp_{j in group} lse_j)``), ``w_c * out_c``."""
    tp = int(out.shape[0])
    lse = lse_from_m_l(m, l, scale)  # [tp, B, H]
    mask = group_mask(tp, parts).reshape(tp, tp)
    d = (lse[None, :] - lse[:, None]).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)  # [c, j, B, H] = lse_j - lse_c
    s = (torch.exp(d) * mask[:, :, None, None]).sum(dim=1)  # [tp, B, H]
    return out.float() * (1.0 / s)[..., None]


def pair_heads(num_q_heads: int, num_kv_heads: int) -> int:
    """q heads per chip under the pair split: the whole KV group."""
    return int(num_q_heads) // int(num_kv_heads)


def pair_qkv_weight(q_w, k_w, v_w, *, num_q_heads: int, num_kv_heads: int, head_dim: int, tp: int) -> torch.Tensor:
    """The fused pair QKV weight ``[hidden, tp * (G + 2) * D]`` from the HF ``q_proj [Hq*D, hidden]``, ``k_proj`` /
    ``v_proj [Hkv*D, hidden]``: column block ``i`` (sharded to device column ``i`` by the column-parallel mapper) is
    ``[Wq of the G q heads of KV group i // 2 | Wk of head i // 2 | Wv of head i // 2]``, transposed. The group split
    (``tp / Hkv`` chips per KV head) takes the same layout with block ``i`` on KV group ``i * Hkv // tp``."""
    group = pair_heads(num_q_heads, num_kv_heads)
    d = int(head_dim)
    blocks = []
    for i in range(int(tp)):
        g = i * int(num_kv_heads) // int(tp)
        wq = q_w[g * group * d : (g + 1) * group * d].transpose(-2, -1)
        wk = k_w[g * d : (g + 1) * d].transpose(-2, -1)
        wv = v_w[g * d : (g + 1) * d].transpose(-2, -1)
        blocks.append(torch.cat([wq, wk, wv], dim=-1))
    return torch.cat(blocks, dim=-1)


def pair_o_proj_weight(o_w_t, *, num_q_heads: int, num_kv_heads: int, head_dim: int, tp: int) -> torch.Tensor:
    """The pair o_proj weight ``[tp * G * D, N]`` from the transposed HF ``o_proj`` ``[Hq*D, N]`` (N = hidden, padded
    as the caller pads it): row block ``i`` (sharded to device column ``i`` by the row-parallel mapper) is the rows of
    the G q heads of KV group ``i // 2`` (the group split: ``i * Hkv // tp``)."""
    group = pair_heads(num_q_heads, num_kv_heads)
    d = int(head_dim)
    tp, nkv = int(tp), int(num_kv_heads)
    return torch.cat([o_w_t[(i * nkv // tp) * group * d : (i * nkv // tp + 1) * group * d] for i in range(tp)], dim=-2)


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
