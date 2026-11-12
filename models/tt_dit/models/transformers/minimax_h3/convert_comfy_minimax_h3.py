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


# ---------------------------------------------------------------------------------------------------
# Parallel Decoding Distillation (PDD) heads
#
# FastH3-Omni-style students, and fine-tunes grafted onto them (10Eros-Max h3 8stepTURBO beta7 fix1),
# widen `final_layer.video_out` / `audio_out` to `pdd_steps` heads, one per interval of a fixed fine
# time grid on [0, grid_max_t]. A sampler step over the block [start, end) of that grid applies the
# integration-weighted mean of the block's heads, and the block's total weight is the node-sigma
# difference, so a plain Euler step over the node sigmas with the fused head is exact. Layout is
# FastVideo's "channel_major": head j is rows [j*O, (j+1)*O) (fastvideo/layers/pdd.py). Some merges
# store heads 1.. as offsets from head 0 (metadata `pdd_head_storage: head0-full;heads1-31-offset-...`).
# The contract (step indices, grid size, grid_max_t, shifts) is in the file's metadata
# `source_inference_contract`.

PDD_HEADS = {"video": ("final_layer.video_out.", "proj_out"), "audio": ("final_layer.audio_out.", "audio_proj_out")}


def pdd_contract(metadata: dict | None) -> dict | None:
    """The PDD sampling contract a file carries in its safetensors metadata; None for a plain file."""
    import json

    metadata = metadata or {}
    raw = metadata.get("source_inference_contract")
    if not raw:
        return None
    try:
        contract = dict(json.loads(raw))
    except ValueError:
        return None
    if "pdd_step_indices" not in contract:
        return None
    contract.setdefault("pdd_steps", int(contract["pdd_step_indices"][-1]))
    contract["heads_offset_from_head0"] = "offset-from-head0" in metadata.get("pdd_head_storage", "")
    return contract


def pdd_full_heads(
    weight: torch.Tensor, bias: torch.Tensor, grid: int, offsets: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[grid, O, I]` weights and `[grid, O]` biases in fp32, with offset-stored heads made absolute."""
    w = weight.float().reshape(grid, -1, weight.shape[-1]).clone()
    b = bias.float().reshape(grid, -1).clone()
    if offsets:
        w[1:] += w[0]
        b[1:] += b[0]
    return w, b


def _shifted(u: torch.Tensor, shift: float, max_t: float) -> torch.Tensor:
    """Rational shift `s*u*M / (u*(s-1) + M)` on [0, M] (FastVideo `shifted_noise_amount`)."""
    if shift == 1.0:
        return u
    return shift * u * max_t / (u * (shift - 1.0) + max_t)


def pdd_step_indices(contract: dict, num_steps: int) -> list[int]:
    """Fine-grid node indices for `num_steps` model evaluations: the trained partition when the count
    matches it, else FastVideo's balanced subset `arange(n + 1) * N // n`."""
    grid = int(contract["pdd_steps"])
    configured = [int(i) for i in contract["pdd_step_indices"]]
    if num_steps == len(configured) - 1:
        return configured
    if not 1 <= num_steps <= grid:
        raise ValueError(f"a {grid}-interval PDD grid cannot run {num_steps} steps")
    return [k * grid // num_steps for k in range(num_steps + 1)]


def pdd_plan(contract: dict, indices: list[int], shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`(node_sigmas, interval_weights)` for one modality: descending node sigmas ending at 0, and
    `sigma(u_{j+1}) - sigma(u_j)` for every fine-grid interval (float64)."""
    max_t = float(contract.get("grid_max_t", 0.999))
    grid = int(contract["pdd_steps"])
    u = torch.linspace(max_t, 0.0, grid + 1, dtype=torch.float64).clamp(max=max_t)
    sigma = _shifted(u, float(shift), max_t)
    sigma[-1] = 0.0
    return sigma[torch.tensor(indices)], sigma[1:] - sigma[:-1]


def pdd_fuse(
    w_heads: torch.Tensor, b_heads: torch.Tensor, weights: torch.Tensor, start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `[O, I]` weight and `[O]` bias of fine-grid block `[start, end)`: the weighted mean of its heads."""
    block = weights[start:end]
    normalized = (block / block.sum()).to(torch.float32)
    return (
        torch.einsum("n,noi->oi", normalized, w_heads[start:end]),
        torch.einsum("n,no->o", normalized, b_heads[start:end]),
    )
