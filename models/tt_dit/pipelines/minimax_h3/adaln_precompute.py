# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 AdaLN modulation precompute.

The model card licenses this outright: "approximately 13B parameters residing in
AdaLN-related branches. Because the AdaLN modulation outputs can be precomputed
and cached, these parameters do not need to be loaded for inference-only
deployment."

Each block's ``adaln_proj`` is ``Linear(2688 -> 96768)`` -- 260.1M parameters, and
50 of them are 13B, about 40% of the checkpoint. Its only input is
``SiLU(time_embedder(t))``, and the full set of ``t`` a request will ever use is
fixed by its sigma schedules before the denoise loop starts. So the whole thing
collapses to a table built once, and those 13B weights never reach the device.

The table is built **per denoise step**, not over the union of all timesteps, and
that is not an arbitrary choice. ``adaln_proj`` is row-independent -- projecting
one row alongside 2 others or alongside 97 others gives bitwise-identical output,
measured. But ``time_embedder`` is *not*: its fp32 GEMM picks a different kernel
and accumulation order at batch 98 than at batch 2, and the resulting 5e-6
difference in ``temb`` amplifies through the bf16 projection to roughly one bf16
ULP on ~0.7% of modulation values. Computing ``temb`` per step, at the same batch
size the reference uses, reproduces it exactly. The cost is a table of ~1.4 GB
instead of ~0.95 GB and 49 tiny GEMMs per block; the expensive part is reading
the weights, which still happens once.

Rounding order is load-bearing for the same reason, more coarsely.
``time_embedder`` is fp32 while ``adaln_proj`` is bf16, and the reference applies
SiLU at ``temb``'s own fp32 precision, casting only the *result* down to bf16.
Hoisting the activation before the cast shifts values by 7.8e-3 -- and does so
identically for every block at every step, so it accumulates coherently along the
trajectory rather than averaging out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...lora.keys import AdapterEntry

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


def timestep_frequency_embedding(timesteps: torch.Tensor, freq_dim: int = 256) -> torch.Tensor:
    """Sinusoidal embedding, **cosine before sine**, in fp32.

    Matches ``Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)``.
    The sin/cos order is a checkpoint contract, not a convention.
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
    """``temb`` for a set of timesteps, fp32 throughout.

    Stays fp32 because every AdaLN projection reads this same tensor and applies
    its own activation and cast afterwards.
    """
    hidden = torch.nn.functional.linear(
        timestep_frequency_embedding(timesteps, freq_dim).to(proj_in_weight.dtype),
        proj_in_weight,
        proj_in_bias,
    )
    hidden = torch.nn.functional.silu(hidden)
    return torch.nn.functional.linear(hidden, proj_out_weight, proj_out_bias)


def _activate_and_project(temb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """SiLU at ``temb``'s precision, cast only the result to the projection dtype.

    See the module docstring on why that ordering matters.
    """
    return torch.nn.functional.linear(torch.nn.functional.silu(temb).to(weight.dtype), weight, bias)


def project_block_adaln(
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    hidden_size: int,
    modality_num: int = MINIMAX_H3_MODALITY_NUM,
) -> torch.Tensor:
    """One block's ``adaln_proj``, returning ``[T * modality_num, 6, hidden]``.

    The view splits the ``modality_num * 6 * hidden`` output into one row per
    (timestep, modality) pair, so row ``t * modality_num + tag`` is what
    ``timestep_indices * modality_num + token_tags`` addresses.
    """
    projected = _activate_and_project(temb, weight, bias)
    projected = projected.view(-1, MINIMAX_H3_ADALN_PARAMS * hidden_size)
    return torch.stack(projected.chunk(MINIMAX_H3_ADALN_PARAMS, dim=-1), dim=1)


def project_final_adaln(
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The final layer's ``adaln_proj``: ``2 * hidden``, **shift then scale**.

    One modality, so rows are addressed by timestep alone. The halves are ordered
    shift-then-scale, as in the LTX-2 and Wan output layers.
    """
    shift, scale = _activate_and_project(temb, weight, bias).chunk(2, dim=-1)
    return shift.contiguous(), scale.contiguous()


@dataclass
class MiniMaxH3AdalnTable:
    """Precomputed AdaLN modulation for every step of one request.

    Rows are grouped by denoise step. ``step_offsets[i]`` is where step ``i``'s
    timesteps begin in :attr:`timesteps`, so a row at step ``i`` with local
    timestep index ``m`` and modality tag ``tag`` lives at
    ``(step_offsets[i] + m) * modality_num + tag`` along ``block_params``' second
    axis. ``param`` is ordered per :data:`MINIMAX_H3_ADALN_PARAM_NAMES`.
    ``final_shift`` / ``final_scale`` are indexed by ``step_offsets[i] + m``
    alone -- the final layer has one modality.
    """

    timesteps: torch.Tensor
    step_offsets: torch.Tensor
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
        return int(self.step_offsets.numel()) - 1

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.block_params, self.final_shift, self.final_scale))

    def step_timesteps(self, step: int) -> torch.Tensor:
        """The sorted distinct timesteps of one step, as the reference builds them."""
        return self.timesteps[int(self.step_offsets[step]) : int(self.step_offsets[step + 1])]

    def step_rows(self, step: int, timestep_indices: torch.Tensor) -> torch.Tensor:
        """Map a step's local ``timestep_indices`` to rows in :attr:`final_shift`.

        ``timestep_indices`` is what ``build_row_timesteps`` returns, so this is
        the only translation a caller needs.
        """
        offset = int(self.step_offsets[step])
        span = int(self.step_offsets[step + 1]) - offset
        indices = timestep_indices.reshape(-1)
        if int(indices.numel()) and int(indices.max()) >= span:
            raise ValueError(f"timestep index {int(indices.max())} exceeds step {step}'s {span} timesteps")
        return offset + indices

    def adaln_indices(self, step: int, timestep_indices: torch.Tensor, token_tags: torch.Tensor) -> torch.Tensor:
        """Per-row index into ``block_params``' second axis.

        ``clamp(min=0)`` mirrors the reference for padding rows, whose outputs are
        discarded and whose tag is -1.
        """
        rows = self.step_rows(step, timestep_indices)
        return rows * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0).reshape(-1)


def request_step_timesteps(
    video_sigmas: torch.Tensor,
    audio_sigmas: torch.Tensor,
    condition_noise_aug: float | None = None,
    audio_condition_timestep: float | None = None,
) -> list[torch.Tensor]:
    """The sorted distinct timesteps of each denoise step.

    Both modalities step their own schedule inside one forward and video conditioning
    rows sit at ``max(video_t, noise_aug)``, so a step carries two or three levels.
    ``t = 1 - sigma``, and the terminal sigma has no evaluation.

    ``audio_condition_timestep`` adds a **fourth** level, and only ``ref2va`` needs it:
    a reference soundtrack's rows are clean and run at a literal ``t = 1.0`` at every
    step. Left out for ``t2va`` and ``fl2va``, which have no audio
    conditioning rows -- so their tables are unchanged, and the pipeline's table cache
    key already separates the two partitions.

    A missing level fails loudly: the table is addressed by matching a row's timestep
    *by value*, so a level it does not carry raises ``IndexError`` in the caller's lookup
    rather than silently modulating with a neighbouring row.
    """
    video = 1.0 - video_sigmas[:-1].to(torch.float32)
    audio = 1.0 - audio_sigmas[:-1].to(torch.float32)
    if video.numel() != audio.numel():
        raise ValueError(f"video and audio schedules must have equal length, got {video.numel()} and {audio.numel()}")

    steps = []
    for index in range(int(video.numel())):
        levels = [video[index], audio[index]]
        if condition_noise_aug is not None:
            levels.append(torch.clamp(video[index], min=float(condition_noise_aug)))
        if audio_condition_timestep is not None:
            levels.append(torch.tensor(float(audio_condition_timestep), dtype=video.dtype))
        steps.append(torch.unique(torch.stack(levels), sorted=True))
    return steps


def precompute_adaln_table(
    checkpoint_dir: str | Path,
    step_timesteps: list[torch.Tensor],
    num_layers: int = 50,
    hidden_size: int = 5376,
    freq_dim: int = 256,
    device: torch.device | str = "cpu",
    weight_hook: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
) -> MiniMaxH3AdalnTable:
    """Build the modulation table, reading each block's projection once.

    Each ``adaln_proj.linear.weight`` is 520 MB. They are read and released one by
    one, and every step's projection is evaluated while that block's weight is
    resident, so the 26 GB is streamed exactly once and never loaded again.

    ``temb`` is computed per step at the reference's own batch size -- see the
    module docstring; batching it changes its fp32 GEMM.

    ``weight_hook`` sees every tensor as it is read and may return a modified one. That is where a
    LoRA adapter's AdaLN half is applied: these weights never reach the device, so the on-device
    adapter path cannot touch them, and folding here keeps the streaming property -- one block
    resident at a time -- that reading the whole 26 GB to patch it would destroy. Anything it
    changes must also change :meth:`MiniMaxH3Pipeline._adaln_cache_path`'s key, or a later run
    silently loads the unadapted table.
    """
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    # Both layouts in circulation: the original MiniMax release names its shards `model-*` while the
    # diffusers conversion names them `diffusion_pytorch_model-*`. Single-file variants of each are
    # accepted too. Globbing `model-*` alone would silently miss the diffusers snapshot this
    # pipeline loads.
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
    try:
        for shard in shards:
            handle = safe_open(shard, framework="pt", device="cpu")
            handles[shard] = handle
            for key in handle.keys():
                location[key] = shard

        def get(key: str) -> torch.Tensor:
            if key not in location:
                raise KeyError(f"{key} not present in {checkpoint_dir}")
            tensor = handles[location[key]].get_tensor(key)
            return tensor if weight_hook is None else weight_hook(key, tensor)

        def get_any(*candidates: str) -> torch.Tensor:
            """First candidate key that exists.

            The two checkpoint layouts name every AdaLN surface differently: the original MiniMax
            release uses `time_embedder.proj_in` / `blocks.N` / `final_layer.adaln_proj`, while the
            diffusers conversion this pipeline loads uses `time_embedder.linear_1` /
            `transformer_blocks.N` / `norm_out.linear`. Resolving by candidate keeps one builder for
            both instead of a layout flag threaded through every caller.
            """
            for candidate in candidates:
                if candidate in location:
                    return get(candidate)
            raise KeyError(f"none of {candidates} present in {checkpoint_dir}")

        proj_in_weight = get_any("time_embedder.proj_in.weight", "time_embedder.linear_1.weight").to(device)
        proj_in_bias = get_any("time_embedder.proj_in.bias", "time_embedder.linear_1.bias").to(device)
        proj_out_weight = get_any("time_embedder.proj_out.weight", "time_embedder.linear_2.weight").to(device)
        proj_out_bias = get_any("time_embedder.proj_out.bias", "time_embedder.linear_2.bias").to(device)
        step_temb = [
            time_embedding(
                levels.to(device), proj_in_weight, proj_in_bias, proj_out_weight, proj_out_bias, freq_dim=freq_dim
            )
            for levels in step_timesteps
        ]

        block_params = None
        for layer in range(num_layers):
            prefixes = (f"blocks.{layer}.adaln_proj.linear", f"transformer_blocks.{layer}.adaln_proj.linear")
            weight = get_any(*(f"{prefix}.weight" for prefix in prefixes)).to(device)
            bias = get_any(*(f"{prefix}.bias" for prefix in prefixes)).to(device)
            params = torch.cat(
                [project_block_adaln(temb, weight, bias, hidden_size) for temb in step_temb],
                dim=0,
            )
            del weight, bias
            if block_params is None:
                block_params = torch.empty((num_layers, *params.shape), dtype=params.dtype, device=device)
            block_params[layer] = params

        final_weight = get_any("final_layer.adaln_proj.linear.weight", "norm_out.linear.weight").to(device)
        final_bias = get_any("final_layer.adaln_proj.linear.bias", "norm_out.linear.bias").to(device)
        finals = [project_final_adaln(temb, final_weight, final_bias) for temb in step_temb]
        shift = torch.cat([pair[0] for pair in finals], dim=0)
        scale = torch.cat([pair[1] for pair in finals], dim=0)
    finally:
        for handle in handles.values():
            del handle
        handles.clear()

    counts = torch.tensor([int(levels.numel()) for levels in step_timesteps], dtype=torch.long)
    step_offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return MiniMaxH3AdalnTable(
        timesteps=torch.cat([levels.to(torch.float32) for levels in step_timesteps]).to(device),
        step_offsets=step_offsets,
        block_params=block_params,
        final_shift=shift,
        final_scale=scale,
    )


# The two checkpoint layouts name every AdaLN surface differently, and an adapter is published
# against the diffusers one. Both spellings map to the same fold entry so a hook built from one
# adapter serves either checkpoint.
_ADALN_KEY_ALIASES: tuple[tuple[str, str], ...] = (
    ("time_embedder.linear_1", "time_embedder.proj_in"),
    ("time_embedder.linear_2", "time_embedder.proj_out"),
    ("norm_out.linear", "final_layer.adaln_proj.linear"),
)


class MiniMaxH3AdalnLoraFold:
    """Applies an adapter's AdaLN half while :func:`precompute_adaln_table` streams the checkpoint.

    ``adaln_proj``, ``time_embedder`` and ``norm_out.linear`` hold about 40% of the checkpoint's
    parameters and, under ``precomputed_adaln``, never become device modules -- the pipeline
    projects them on host into a table instead. Their adapter entries therefore have no Linear to
    bind to. Folding them in as the weights stream past is the only point at which they exist.

    :meth:`unapplied` exists because the failure mode here is silence: a spelling this fold does not
    recognise produces a perfectly valid table built from unadapted weights, and every downstream
    check passes. Assert it is empty.
    """

    def __init__(self, entries: Sequence[AdapterEntry], *, strength: float = 1.0) -> None:
        self._strength = float(strength)
        self._deltas: dict[str, torch.Tensor] = {}
        self._seen: set[str] = set()
        for entry in entries:
            suffix = "bias" if entry.kind == "diff_b" else "weight"
            if entry.kind == "lora":
                delta = entry.B.to(torch.float32) @ entry.A.to(torch.float32)
            elif entry.kind in ("diff", "diff_b"):
                delta = entry.delta.to(torch.float32)
            else:
                msg = f"{entry.path}: {entry.kind} has no meaning for a host-folded AdaLN weight"
                raise ValueError(msg)
            for name in self._aliases(entry.path):
                self._deltas[f"{name}.{suffix}"] = delta

    @staticmethod
    def _aliases(path: str) -> tuple[str, ...]:
        for diffusers_name, original_name in _ADALN_KEY_ALIASES:
            if path == diffusers_name:
                return (diffusers_name, original_name)
        if ".adaln_proj.linear" in path:
            layer = path.split(".")[1]
            return (f"transformer_blocks.{layer}.adaln_proj.linear", f"blocks.{layer}.adaln_proj.linear")
        return (path,)

    def __call__(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        delta = self._deltas.get(key)
        if delta is None:
            return tensor
        self._seen.add(key)
        if delta.shape != tensor.shape:
            msg = f"{key}: adapter delta {tuple(delta.shape)} does not match checkpoint {tuple(tensor.shape)}"
            raise ValueError(msg)
        # fp32 accumulate, then back to the checkpoint's dtype -- the same single rounding a
        # host-fused checkpoint would carry, and the ordering this module's docstring calls
        # load-bearing for `time_embedder`.
        return (tensor.to(torch.float32) + self._strength * delta).to(tensor.dtype)

    def unapplied(self) -> list[str]:
        """Fold entries no checkpoint key ever matched. Non-empty means part of the adapter is lost."""
        # Aliases double every entry; report the diffusers spelling only.
        return sorted(
            {key for key in self._deltas if key not in self._seen}
            & {key for key in self._deltas if not key.startswith(("blocks.", "final_layer.", "time_embedder.proj_"))}
        )
