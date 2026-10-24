# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein DiT hyper-parameters: the two released sizes and the per-mesh shape checks.

Mirrors the diffusers ``Flux2Transformer2DModel`` constructor arguments (``transformer_flux2.py:1151-1172``: patch
size 1, ``in_channels`` 128, ``out_channels`` None = 128, ``attention_head_dim`` 128, ``joint_attention_dim``,
``timestep_guidance_channels`` 256, ``mlp_ratio`` 3.0, ``axes_dims_rope`` (32, 32, 32, 32), ``rope_theta`` 2000,
``eps`` 1e-6, ``guidance_embeds`` False) and the BFL ``Klein4BParams`` / ``Klein9BParams`` (``model.py:24-49``). The
4B row is the snapshot's ``transformer/config.json``; the 9B row is ``Klein9BParams`` (hidden 4096, 32 heads, 8 + 24
blocks, ``context_in_dim`` 12288 = 3 x 4096, the Qwen3-8B hidden size) and is validated by :meth:`from_snapshot`
when that snapshot is on disk. The text side of the row (``text_hidden``, ``text_taps``, ``text_length``) is the
prompt encoder's contract (``pipeline_flux2_klein.py:204, 233-239, 257-261``): hidden states 9, 18 and 27 of the
Qwen3 encoder concatenated per token over 512 tokens give the ``joint_dim``-wide prompt rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


class Flux2KleinSizeRow(NamedTuple):
    """One released size: the DiT width and depth and the text encoder's hidden size."""

    hidden: int
    num_heads: int
    num_layers: int
    num_single_layers: int
    joint_dim: int
    text_hidden: int


FLUX2_KLEIN_SIZES: dict[str, Flux2KleinSizeRow] = {
    # transformer/config.json of black-forest-labs/FLUX.2-klein-4B; text_encoder/config.json hidden_size 2560.
    "4b": Flux2KleinSizeRow(
        hidden=3072, num_heads=24, num_layers=5, num_single_layers=20, joint_dim=7680, text_hidden=2560
    ),
    # model.py:24-35 Klein9BParams; text hidden = context_in_dim / 3.
    "9b": Flux2KleinSizeRow(
        hidden=4096, num_heads=32, num_layers=8, num_single_layers=24, joint_dim=12288, text_hidden=4096
    ),
}

# Largest ring-SDPA q chunk whose fp32-accumulating kernel fits L1 next to k chunk 512
# (block_flux2_klein.apply_exact_attention_profile).
FLUX2_KLEIN_SDPA_Q_CHUNK_MAX = 128

_TRANSFORMER_CONFIG = "transformer/config.json"
_TEXT_ENCODER_CONFIG = "text_encoder/config.json"
_TILE = 32


@dataclass(frozen=True)
class Flux2KleinConfig:
    """Hyper-parameters of one FLUX.2 Klein DiT (see the module docstring for the sources)."""

    size: str
    hidden: int
    num_heads: int
    num_layers: int
    num_single_layers: int
    joint_dim: int
    text_hidden: int
    head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    mlp_ratio: int = 3
    rope_theta: int = 2000
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    eps: float = 1e-6
    time_channels: int = 256
    text_taps: tuple[int, ...] = (9, 18, 27)
    text_length: int = 512

    def __post_init__(self) -> None:
        if self.size not in FLUX2_KLEIN_SIZES:
            raise ValueError(f"size must be one of {tuple(FLUX2_KLEIN_SIZES)}, got {self.size!r}")
        if self.hidden != self.num_heads * self.head_dim:
            raise ValueError(f"hidden {self.hidden} != num_heads {self.num_heads} x head_dim {self.head_dim}")
        if sum(self.axes_dim) != self.head_dim:
            raise ValueError(f"axes_dim {self.axes_dim} must sum to head_dim {self.head_dim}")
        if self.joint_dim != len(self.text_taps) * self.text_hidden:
            raise ValueError(
                f"joint_dim {self.joint_dim} != {len(self.text_taps)} text taps x text_hidden {self.text_hidden}"
            )

    @property
    def mlp_hidden(self) -> int:
        """Width of the SwiGLU hidden layer (``Flux2FeedForward`` / ``Flux2ParallelSelfAttention.mlp_hidden_dim``)."""
        return self.mlp_ratio * self.hidden

    @classmethod
    def for_size(cls, size: str) -> Flux2KleinConfig:
        """The configuration of a released size (``"4b"`` | ``"9b"``) from :data:`FLUX2_KLEIN_SIZES`."""
        if size not in FLUX2_KLEIN_SIZES:
            raise ValueError(f"size must be one of {tuple(FLUX2_KLEIN_SIZES)}, got {size!r}")
        return cls(size=size, **FLUX2_KLEIN_SIZES[size]._asdict())

    @classmethod
    def from_snapshot(cls, snapshot: Path | str) -> Flux2KleinConfig:
        """Read ``transformer/config.json`` (and ``text_encoder/config.json`` for the text hidden size) of a
        diffusers snapshot, identify the size row it matches and check every other field against this port's
        assumptions (no guidance embedder, patch size 1, 128 channels, 4 x 32 RoPE axes with theta 2000)."""
        snapshot = Path(snapshot)
        with (snapshot / _TRANSFORMER_CONFIG).open() as f:
            cfg = json.load(f)
        with (snapshot / _TEXT_ENCODER_CONFIG).open() as f:
            text_cfg = json.load(f)

        hidden = int(cfg["num_attention_heads"]) * int(cfg["attention_head_dim"])
        row = Flux2KleinSizeRow(
            hidden=hidden,
            num_heads=int(cfg["num_attention_heads"]),
            num_layers=int(cfg["num_layers"]),
            num_single_layers=int(cfg["num_single_layers"]),
            joint_dim=int(cfg["joint_attention_dim"]),
            text_hidden=int(text_cfg["hidden_size"]),
        )
        sizes = [name for name, known in FLUX2_KLEIN_SIZES.items() if known == row]
        if not sizes:
            raise ValueError(f"{snapshot / _TRANSFORMER_CONFIG} matches no known size: {row}")
        config = cls(
            size=sizes[0],
            **row._asdict(),
            head_dim=int(cfg["attention_head_dim"]),
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg["out_channels"]) if cfg.get("out_channels") is not None else int(cfg["in_channels"]),
            mlp_ratio=int(cfg["mlp_ratio"]),
            rope_theta=int(cfg["rope_theta"]),
            axes_dim=tuple(int(d) for d in cfg["axes_dims_rope"]),
            eps=float(cfg["eps"]),
            time_channels=int(cfg["timestep_guidance_channels"]),
        )
        expected = cls.for_size(config.size)
        if config != expected:
            raise ValueError(
                f"{snapshot / _TRANSFORMER_CONFIG} deviates from the {config.size} row: {config} != {expected}"
            )
        if float(cfg["mlp_ratio"]) != config.mlp_ratio:
            raise ValueError(f"mlp_ratio {cfg['mlp_ratio']} is not the integer {config.mlp_ratio}")
        if cfg.get("guidance_embeds", False):
            raise ValueError("guidance_embeds must be false: this port has no guidance embedder")
        if int(cfg.get("patch_size", 1)) != 1:
            raise ValueError(f"patch_size must be 1, got {cfg['patch_size']}")
        return config

    def local_shapes(self, tp_factor: int) -> tuple[int, int]:
        """``(heads_local, hidden_local)`` per device under tensor parallelism ``tp_factor``: the heads must split
        evenly (``attention_opt.py:112-113``) and the fused distributed norms need ``hidden % (32 * tp) == 0``
        (``normalization.py:184, 342``). 4b: 6 / 768 at TP4, 3 / 384 at TP8; 9b: 8 / 1024 and 4 / 512."""
        if tp_factor < 1:
            raise ValueError(f"tp_factor must be positive, got {tp_factor}")
        if self.num_heads % tp_factor:
            raise ValueError(f"{self.num_heads} heads do not split over TP {tp_factor}")
        if self.hidden % (_TILE * tp_factor):
            raise ValueError(f"hidden {self.hidden} is not a multiple of {_TILE} x TP {tp_factor}")
        return self.num_heads // tp_factor, self.hidden // tp_factor
