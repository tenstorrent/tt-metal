# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Load MiniMax H3 transformer weights stored in ComfyUI's native layout.

ComfyUI files (Comfy-Org/MiniMax-H3 `diffusion_models/*.safetensors`, and fine-tunes built on them such
as 10Eros-Max) use the original MiniMax names. This module inverts that layout onto our diffusers names,
the same mapping as diffusers' `convert_minimax_h3_transformer_checkpoint_to_diffusers`.

The "pruned" builds drop the time embedder: `adaln_t_table` [G, R] holds the time embedding sampled at G
evenly spaced timesteps in [0, 1], already folded into an R-dimensional basis (R = 8), and every AdaLN
linear takes R inputs instead of 2688. ComfyUI serves it by interpolating the table at the timestep and
feeding the result straight into the AdaLN linears in fp32, with no SiLU (comfy/ldm/minimax/model.py).
`adaln_curve_embedding` reproduces that lookup; `CURVE_PAD` pads R to one tile with zero columns so tile
padding cannot change the result.
"""

from __future__ import annotations

import torch

CURVE_TABLE_KEY = "adaln_t_table"
CURVE_PAD = 32

_PREFIX_RENAMES = (("blocks.", "transformer_blocks."),)
_RENAMES = (
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("time_embedder.proj_in.", "time_embedder.linear_1."),
    ("time_embedder.proj_out.", "time_embedder.linear_2."),
    ("video_patch_proj.", "proj_in."),
    ("audio_patch_proj.", "audio_proj_in."),
    ("condition_proj.", "context_embedder."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    ("final_layer.video_out.", "proj_out."),
    ("final_layer.audio_out.", "audio_proj_out."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
    (".attn.out_proj.", ".attn.to_out.0."),
    (".mlp.fc1.", ".ff.net.0.proj."),
    (".mlp.fc2.", ".ff.net.2."),
)
_DROPPED = ("rope.inv_freq",)


def is_comfy_layout(keys) -> bool:
    return "video_patch_proj.weight" in keys


def is_pruned_layout(keys) -> bool:
    return CURVE_TABLE_KEY in keys


def comfy_key_to_diffusers(key: str) -> str:
    for old, new in _PREFIX_RENAMES:
        if key.startswith(old):
            key = new + key[len(old) :]
    for old, new in _RENAMES:
        key = key.replace(old, new)
    return key


def _qkv_names(key: str) -> list[str]:
    stem = key[: -len("qkv_proj.weight")]
    return [f"{stem}to_{p}.weight" for p in "qkv"]


def map_comfy_shapes(header: dict[str, list[int]]) -> dict[str, list[int]]:
    """Diffusers name -> shape from a Comfy header (name -> shape), without reading any data.

    `adaln_t_table` keeps its name; dropped tensors are left out.
    """
    out: dict[str, list[int]] = {}
    for key, shape in header.items():
        if key in _DROPPED or key == "__metadata__":
            continue
        if key == CURVE_TABLE_KEY:
            out[key] = list(shape)
            continue
        new = comfy_key_to_diffusers(key)
        if new.endswith("qkv_proj.weight"):
            third = [shape[0] // 3, *shape[1:]]
            for name in _qkv_names(new):
                out[name] = third
        else:
            out[new] = list(shape)
    return out


def convert_comfy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A Comfy-layout state dict in diffusers names. Consumes `state` to bound peak memory.

    qkv is stored stacked [q; k; v] in Comfy files; fc1 is [gate; value] and diffusers wants [value; gate].
    """
    out: dict[str, torch.Tensor] = {}
    for key in list(state):
        tensor = state.pop(key)
        if key in _DROPPED:
            continue
        if key == CURVE_TABLE_KEY:
            out[key] = tensor
            continue
        new = comfy_key_to_diffusers(key)
        if new.endswith("qkv_proj.weight"):
            for name, part in zip(_qkv_names(new), tensor.chunk(3, dim=0)):
                out[name] = part.contiguous()
        elif new.endswith(".ff.net.0.proj.weight") or new.endswith(".ff.net.0.proj.bias"):
            gate, value = tensor.chunk(2, dim=0)
            out[new] = torch.cat([value, gate], dim=0)
        else:
            out[new] = tensor
    return out


def adaln_curve_embedding(table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """ComfyUI's lookup: linear interpolation of `table` [G, R] at unscaled timesteps `t` in [0, 1]. fp32."""
    table = table.float()
    grid = table.shape[0]
    pos = t.float().clamp(0, 1) * (grid - 1)
    i0 = pos.floor().long().clamp(max=grid - 2)
    frac = (pos - i0.float()).unsqueeze(-1)
    return torch.lerp(table[i0], table[i0 + 1], frac)


def pad_curve(tensor: torch.Tensor, dim: int = -1, size: int = CURVE_PAD) -> torch.Tensor:
    """Zero-pad the curve basis dimension of an embedding or AdaLN weight to `size`."""
    extra = size - tensor.shape[dim]
    if extra < 0:
        raise ValueError(f"curve rank {tensor.shape[dim]} exceeds pad {size}")
    if extra == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = extra
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


# Per-step output heads (progressive-distillation grafts such as 10Eros-Max beta7_fix1, built from FastH3):
# `final_layer.video_out` / `audio_out` hold P heads stacked on the output rows ([P * C, hidden]); head 0
# is stored in full and heads 1..P-1 as offsets from it (metadata `pdd_head_storage`). A run of N forwards
# uses head `pdd_step_indices[i]` at forward i (metadata `source_inference_contract`).
STEP_HEAD_KEYS = ("proj_out", "audio_proj_out")
_OFFSET_STORAGE = "heads1-31-offset-from-head0"


def step_head_contract(metadata: dict[str, str] | None, num_heads: int) -> list[int]:
    """Head index used at each forward, from the file's inference contract (default: P/8 apart over 8 forwards)."""
    import json

    indices = None
    raw = (metadata or {}).get("source_inference_contract")
    if raw:
        contract = json.loads(raw)
        indices = contract.get("pdd_step_indices")
        forwards = contract.get("num_inference_steps") or contract.get("transformer_forwards")
        if indices is not None and forwards is not None:
            indices = list(indices)[: int(forwards)]
    if indices is None:
        indices = list(range(0, num_heads, max(num_heads // 8, 1)))[:8]
    if not indices or any(not 0 <= int(h) < num_heads for h in indices):
        raise ValueError(f"step heads {indices} out of range for {num_heads} heads")
    return [int(h) for h in indices]


def num_step_heads(rows: int, channels: int) -> int:
    """How many heads an output projection with `rows` output rows stacks (1: a plain single head)."""
    if rows % channels:
        raise ValueError(f"output rows {rows} are not a multiple of {channels} channels")
    return rows // channels


def unfold_step_heads(tensor: torch.Tensor, channels: int, heads: list[int], *, offsets: bool = True) -> torch.Tensor:
    """`[P * C, ...]` stacked heads -> the heads in `heads` order, stacked `[len(heads) * C, ...]`.

    With `offsets` (the beta7 storage), head k >= 1 is `head0 + stored_k`, summed in fp32 and cast back.
    """
    count = num_step_heads(tensor.shape[0], channels)
    stacked = tensor.float().reshape(count, channels, *tensor.shape[1:])
    out = []
    for h in heads:
        head = stacked[h] if h == 0 or not offsets else stacked[0] + stacked[h]
        out.append(head)
    return torch.cat(out, dim=0).to(tensor.dtype)


def select_step_heads(
    state: dict[str, torch.Tensor], channels: dict[str, int], heads: list[int], metadata: dict[str, str] | None
) -> int:
    """Rewrite multi-head `proj_out` / `audio_proj_out` in a diffusers-named state dict to the heads in
    `heads` order. `channels` maps each projection to its per-head width (video: in_channels * prod(patch_size)).
    Returns the number of heads left per projection (1 when the file has single heads)."""
    storage = (metadata or {}).get("pdd_head_storage", "")
    offsets = _OFFSET_STORAGE in storage
    counts = set()
    for name in STEP_HEAD_KEYS:
        c = channels[name]
        weight = state[f"{name}.weight"]
        count = num_step_heads(weight.shape[0], c)
        counts.add(count)
        if count == 1:
            continue
        if not storage:
            raise ValueError(f"{name} stacks {count} heads but the file names no pdd_head_storage")
        for part in ("weight", "bias"):
            state[f"{name}.{part}"] = unfold_step_heads(state[f"{name}.{part}"], c, heads, offsets=offsets)
    if counts == {1}:
        return 1
    if 1 in counts:
        raise ValueError("video and audio output heads disagree: one is single-head, the other stacked")
    return len(heads)


def step_head_slots(sigmas: list[float], heads: list[int], metadata: dict[str, str] | None) -> list[int]:
    """Slot in `heads` for each forward, picked by its sigma: the head whose point on the contract's pdd grid
    (`1 - k / pdd_steps` under its video shift) is nearest. With the contract's own schedule this is forward
    i -> slot i; another step count still runs each forward on the head trained nearest its noise level."""
    import json

    raw = (metadata or {}).get("source_inference_contract")
    contract = json.loads(raw) if raw else {}
    pdd_steps, shift = contract.get("pdd_steps"), contract.get("video_scheduler_shift")
    if not pdd_steps or not shift:
        if len(sigmas) != len(heads):
            raise ValueError(
                f"{len(sigmas)} forwards on a {len(heads)}-head contract with no pdd_steps/video_scheduler_shift"
            )
        return list(range(len(heads)))
    grid = [1.0 - h / float(pdd_steps) for h in heads]
    grid = [shift * g / (1 + (shift - 1) * g) for g in grid]
    return [min(range(len(grid)), key=lambda k: abs(grid[k] - float(s))) for s in sigmas]


def step_head_selector(slot: int, num_slots: int, channels: int) -> torch.Tensor:
    """One-hot `[num_slots * C, C]` that picks head `slot` out of the stacked projection output."""
    if not 0 <= slot < num_slots:
        raise ValueError(f"head slot {slot} out of range for {num_slots} slots")
    sel = torch.zeros(num_slots * channels, channels)
    sel[slot * channels : (slot + 1) * channels] = torch.eye(channels)
    return sel
