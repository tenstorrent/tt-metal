# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 AdaLN modulation precompute, on host.

The model card licenses this outright: "approximately 13B parameters residing in AdaLN-related
branches. Because the AdaLN modulation outputs can be precomputed and cached, these parameters do
not need to be loaded for inference-only deployment."

Each block's ``adaln_proj`` is ``Linear(2688 -> 96768)`` -- 260.1M parameters; 50 of them plus
``norm_out.linear`` and ``time_embedder`` are ~26 GB, about 40% of the checkpoint. Their only input
is ``SiLU(time_embedder(t))``, and the full set of ``t`` a request will ever use is fixed by its
sigma schedules before the denoise loop starts. So the whole branch collapses to a table built once,
and those weights never reach the device. On a single 32 GB Blackhole that is not an optimization:
the bf16 DiT is ~66 GB with the AdaLN branches resident and ~40 GB without, so the table is what
makes the p150 profile arithmetically possible at all.

Rows follow the pipeline's **slot** framing (`packing.build_slot_routing` / `packing.slot_levels`):
a request pins an ordered tuple of roles -- ``("video", "audio", "condition_video")`` for t2va and
fl2va -- and every denoise step evaluates exactly those slots, in that order, with no deduplication.
So step ``i``'s slot ``s`` is row ``i * num_slots + s``, and the per-row index tensors the pipeline
already builds (`adaln_indices(token_tags, row_slot)`, `row_slot`) only need ``i * num_slots`` added
to them. Nothing about the addressing changes.

Two orderings below are load-bearing, both for the same reason -- 50 blocks read one ``temb``, so a
difference in it biases every block identically at every step and accumulates along the trajectory
instead of averaging out:

* **Batch.** ``adaln_proj`` is row-independent (projecting one row beside two others or beside 97 is
  bitwise identical, measured), but ``time_embedder`` is not: its fp32 GEMM picks a different kernel
  and accumulation order at a different batch size. ``temb`` is therefore computed **per step**, at
  the ``num_slots`` batch the device path uses, not once over the concatenated set.
* **Rounding.** ``time_embedder`` is fp32 while ``adaln_proj`` is bf16, and the reference applies
  SiLU at ``temb``'s own fp32 precision, casting only the *result* down. Hoisting the cast before
  the activation shifts values by 7.8e-3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

# diffusers' shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp order.
MINIMAX_H3_ADALN_PARAM_NAMES = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)
MINIMAX_H3_ADALN_PARAMS = len(MINIMAX_H3_ADALN_PARAM_NAMES)
MINIMAX_H3_MODALITY_NUM = 3

# Checkpoint keys the table replaces. `_read_safetensors` drops them before the loader ever sees
# them, which is where the 26 GB of host read and device residency actually disappears.
MINIMAX_H3_ADALN_DROP_KEYS = (".adaln_proj.", "time_embedder.", "norm_out.linear")


def is_adaln_key(key: str) -> bool:
    """True for a checkpoint key the precomputed table makes unnecessary."""
    return any(marker in key for marker in MINIMAX_H3_ADALN_DROP_KEYS)


def timestep_frequency_embedding(timesteps: torch.Tensor, freq_dim: int = 256) -> torch.Tensor:
    """Sinusoidal embedding, **cosine before sine**, in fp32.

    Matches ``Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)``. The sin/cos order
    is a checkpoint contract, not a convention.
    """
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = timesteps.to(torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def time_embedding(
    timesteps: torch.Tensor,
    proj_in_weight: torch.Tensor,
    proj_in_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    freq_dim: int = 256,
) -> torch.Tensor:
    """``temb`` for one step's slot levels, fp32 throughout.

    Stays fp32 because every AdaLN projection reads this same tensor and applies its own activation
    and cast afterwards.
    """
    hidden = torch.nn.functional.linear(
        timestep_frequency_embedding(timesteps, freq_dim).to(proj_in_weight.dtype),
        proj_in_weight,
        proj_in_bias,
    )
    hidden = torch.nn.functional.silu(hidden)
    return torch.nn.functional.linear(hidden, proj_out_weight, proj_out_bias)


def _activate_and_project(temb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """SiLU at ``temb``'s precision, cast only the result to the projection dtype."""
    return torch.nn.functional.linear(torch.nn.functional.silu(temb).to(weight.dtype), weight, bias)


def project_block_adaln(
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    """One block's ``adaln_proj`` for one step, as ``[num_slots * MODALITY_NUM, 6, hidden]``.

    The reference lays the output dim out as ``[modality][param][hidden]``, so the view splits it
    into one row per (slot, modality) pair -- row ``slot * MODALITY_NUM + tag``, which is exactly
    what ``packing.adaln_indices`` addresses within a step.
    """
    projected = _activate_and_project(temb, weight, bias)
    projected = projected.view(-1, MINIMAX_H3_ADALN_PARAMS * hidden_size)
    return torch.stack(projected.chunk(MINIMAX_H3_ADALN_PARAMS, dim=-1), dim=1)


def project_final_adaln(
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm_out.linear``: ``2 * hidden``, **shift then scale**.

    One modality, so rows are addressed by slot alone.
    """
    shift, scale = _activate_and_project(temb, weight, bias).chunk(2, dim=-1)
    return shift.contiguous(), scale.contiguous()


@dataclass
class MiniMaxH3AdalnTable:
    """Precomputed AdaLN modulation for every (step, slot) of one schedule.

    ``block_params`` is ``[layers, num_steps * num_slots * MODALITY_NUM, 6, hidden]`` and
    ``final_shift`` / ``final_scale`` are ``[num_steps * num_slots, hidden]`` -- the final layer has
    no modality axis. ``levels`` keeps the timesteps the table was built from so a mismatched reuse
    can be caught rather than silently modulating every step slightly wrong.
    """

    levels: torch.Tensor  # [num_steps, num_slots] float32
    block_params: torch.Tensor
    final_shift: torch.Tensor
    final_scale: torch.Tensor

    @property
    def num_layers(self) -> int:
        return int(self.block_params.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.block_params.shape[-1])

    @property
    def num_steps(self) -> int:
        return int(self.levels.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.levels.shape[1])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.block_params, self.final_shift, self.final_scale))

    def step_offset(self, step: int) -> int:
        """First absolute slot row of ``step``. Callers add this to their per-row slot indices."""
        if not 0 <= step < self.num_steps:
            raise ValueError(f"step {step} outside the table's {self.num_steps} steps")
        return step * self.num_slots


def _open_checkpoint(checkpoint_dir: str | Path):
    """``(get_any, close)`` over every safetensors shard under ``checkpoint_dir``."""
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    # Both layouts in circulation: the original MiniMax release names its shards `model-*`, the
    # diffusers conversion `diffusion_pytorch_model-*`. Single-file variants of each too. Globbing
    # `model-*` alone would silently miss the diffusers snapshot this pipeline loads.
    shards = sorted(
        {
            shard
            for pattern in (
                "model-*.safetensors",
                "diffusion_pytorch_model-*.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.safetensors",
            )
            for shard in checkpoint_dir.glob(pattern)
        }
    )
    if not shards:
        raise FileNotFoundError(
            f"no model-*.safetensors or diffusion_pytorch_model-*.safetensors under {checkpoint_dir}"
        )

    location: dict[str, Path] = {}
    handles = {}
    for shard in shards:
        handle = safe_open(shard, framework="pt", device="cpu")
        handles[shard] = handle
        for key in handle.keys():
            location[key] = shard

    def get_any(*candidates: str) -> torch.Tensor:
        """First candidate key that exists.

        The two checkpoint layouts name every AdaLN surface differently: the MiniMax release uses
        `time_embedder.proj_in` / `blocks.N` / `final_layer.adaln_proj`, the diffusers conversion
        this pipeline loads uses `time_embedder.linear_1` / `transformer_blocks.N` /
        `norm_out.linear`. Resolving by candidate keeps one builder for both.
        """
        for candidate in candidates:
            if candidate in location:
                return handles[location[candidate]].get_tensor(candidate)
        raise KeyError(f"none of {candidates} present in {checkpoint_dir}")

    def close() -> None:
        handles.clear()

    return get_any, close


def precompute_adaln_table(
    checkpoint_dir: str | Path,
    step_levels: list[torch.Tensor],
    num_layers: int = 50,
    hidden_size: int = 5376,
    freq_dim: int = 256,
) -> MiniMaxH3AdalnTable:
    """Build the modulation table, reading each projection exactly once.

    ``step_levels[i]`` is the ``[num_slots]`` float32 vector `packing.slot_levels` returns for
    denoise step ``i`` -- same order, same length, no deduplication.

    Each ``adaln_proj.linear.weight`` is 520 MB. They are read and released one at a time, and every
    step is projected while that block's weight is resident, so the 26 GB streams past once and is
    never held.
    """
    if not step_levels:
        raise ValueError("step_levels is empty; a table needs at least one denoise step")
    num_slots = int(step_levels[0].numel())
    if any(int(levels.numel()) != num_slots for levels in step_levels):
        raise ValueError("every step must carry the same number of slots (the role tuple is pinned per request)")

    get_any, close = _open_checkpoint(checkpoint_dir)
    try:
        proj_in_weight = get_any("time_embedder.proj_in.weight", "time_embedder.linear_1.weight")
        proj_in_bias = get_any("time_embedder.proj_in.bias", "time_embedder.linear_1.bias")
        proj_out_weight = get_any("time_embedder.proj_out.weight", "time_embedder.linear_2.weight")
        proj_out_bias = get_any("time_embedder.proj_out.bias", "time_embedder.linear_2.bias")
        step_temb = [
            time_embedding(levels, proj_in_weight, proj_in_bias, proj_out_weight, proj_out_bias, freq_dim=freq_dim)
            for levels in step_levels
        ]

        block_params = None
        for layer in range(num_layers):
            prefixes = (f"blocks.{layer}.adaln_proj.linear", f"transformer_blocks.{layer}.adaln_proj.linear")
            weight = get_any(*(f"{prefix}.weight" for prefix in prefixes))
            bias = get_any(*(f"{prefix}.bias" for prefix in prefixes))
            params = torch.cat([project_block_adaln(temb, weight, bias, hidden_size) for temb in step_temb], dim=0)
            del weight, bias
            if block_params is None:
                block_params = torch.empty((num_layers, *params.shape), dtype=params.dtype)
            block_params[layer] = params

        final_weight = get_any("final_layer.adaln_proj.linear.weight", "norm_out.linear.weight")
        final_bias = get_any("final_layer.adaln_proj.linear.bias", "norm_out.linear.bias")
        finals = [project_final_adaln(temb, final_weight, final_bias) for temb in step_temb]
        shift = torch.cat([pair[0] for pair in finals], dim=0)
        scale = torch.cat([pair[1] for pair in finals], dim=0)
    finally:
        close()

    return MiniMaxH3AdalnTable(
        levels=torch.stack([levels.to(torch.float32) for levels in step_levels]),
        block_params=block_params,
        final_shift=shift,
        final_scale=scale,
    )
