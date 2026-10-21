# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein prompt encoder: the vendored ``reference/flux3_klein/text_encoder.py::TextEncoder.encode`` on a
tt_dit mesh (``Qwen3VlTextEncoder`` built by ``encoders/qwen3vl/loader_flux3_klein.py``).

Host side, exactly the reference recipe (``text_encoder.py:30-38``): ``AutoProcessor`` of ``snapshot/text_encoder``
with the tokenizer's ``padding_side = "right"``; ``render`` = ``apply_chat_template([{"role": "user", "content":
prompt}], tokenize=False, add_generation_prompt=True)``; ``real`` = the token count of the rendered text truncated at
4096; ``L = min(ceil(real / 80) * 80, 4096)`` (``packing.padded_text_length``); ``tokenizer(text, padding="max_length",
truncation=True, max_length=L)``; positions ``(mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)``. Never
``apply_chat_template(tokenize=True)`` (not equal to ``tokenizer(text)`` in transformers 5.12.1).

RoPE tables are built here with HuggingFace's exact ``Qwen3VLTextRotaryEmbedding`` arithmetic
(:func:`flux3_klein_rope_tables`): ``inv_freq = 1 / theta ** (arange(0, 128, 2) / 128)`` in fp32, ``freqs = pos x
inv_freq``, ``cat(freqs, freqs)``, ``cos`` / ``sin`` cast to bf16 -- bit-equal to HF at every length up to 4096. For a
text-only prompt the three M-RoPE axes carry the same position, so the interleaved section layout is a no-op.
``model_qwen3vl.create_rope_tensors`` is NOT used: its ``theta ** -x`` form lands 6-692 bf16 table entries on the other
side of a rounding boundary at L >= 320.

Device side: ``encoder.forward(ids, attention_mask=mask, pos_embeds=(cos, sin))`` with the right-padding mask
(HF semantics: every query row, pad rows included, attends keys ``< R`` causally; the DiT consumes ALL ``L`` rows, so
the pad rows are part of the output and must match the reference), the eight taps concatenated LAYER-MAJOR on device
(``ttnn.concat(taps, dim=-1)`` in increasing layer order) into ``[1, L, 20480]`` bf16 = per token ``[tap4 | tap8 | ...
| tap32]``, the reference's ``stack(dim=1).permute(0, 2, 1, 3).reshape(L, -1)`` layout.

DiT handoff (``transformer_flux3_klein.py`` contract: ``[1, L_pad, 20480]`` bf16 ``mesh_axes=[None, sp, None]`` with
ZERO rows beyond L, ``L_pad = packing.padded_length(L, SP)``): with ``output_sp_axis`` set (the encoder lives on the
DiT's own mesh), :meth:`encode` pads the rows with ``ttnn.pad(value=0.0)`` and shards them with ``ttnn.mesh_partition(
dim=1, cluster_axis=output_sp_axis)``; the result goes straight into ``Flux3KleinTransformer.embed_text`` with no host
round trip. On another mesh: ``rows, ids = Flux3KleinPromptEncoder.to_host(enc)``; ``padded, _ = packing.pad_rows(rows,
packing.padded_length(len(rows), dit_sp))``; ``tensor.from_torch(padded[None], device=dit_mesh, mesh_axes=[None,
dit_sp_axis, None])`` (3.3 MB at L 80, 13 MB at 320, 168 MB at 4096). ``ids = packing.text_ids(L)`` in both cases.
:meth:`encode_host` returns the host ``(L, 20480) bf16, ids`` pair the M4 sampler and the golden comparisons take.

Batch of one prompt; classifier-free guidance is two calls, ``encode("")`` and ``encode(prompt)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import FLUX3_KLEIN_TEXT_SUBFOLDER, flux3_klein_rope_theta
from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .packing import (
    FLUX3_KLEIN_MAX_TEXT_LENGTH,
    FLUX3_KLEIN_TEXT_PAD_MULTIPLE,
    padded_length,
    padded_text_length,
    text_ids,
)


def flux3_klein_rope_tables(
    positions: torch.Tensor, *, head_dim: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``Qwen3VLTextRotaryEmbedding``'s bf16 tables for identical t/h/w positions: ``positions`` ``(L,)`` or
    ``(1, L)`` int64 -> ``(cos, sin)`` each ``(1, 1, L, head_dim)`` bf16. The arithmetic is HF's (fp32 ``1 / theta ** x``
    inverse frequencies, fp32 outer product, ``cat(freqs, freqs)``, one bf16 rounding of cos / sin); bit-equal to HF at
    L 80..4096 (``docs/M5_DESIGN.md`` decision 4)."""
    positions = positions.reshape(-1)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :].float()
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16)[None, None], emb.sin().to(torch.bfloat16)[None, None]


@dataclass(frozen=True)
class Flux3KleinTextEncoding:
    """One encoded prompt. ``tokens`` is a device tensor: ``[1, padded_length, 20480]`` bf16 ``mesh_axes=[None, sp_axis,
    None]`` with zero rows beyond ``length`` when the encoder was built with ``output_sp_axis`` (the DiT contract);
    otherwise ``[1, length, 20480]`` bf16 replicated on the encoder mesh. ``ids = packing.text_ids(length)`` ``(length,
    4)`` int64; ``length`` = L (a multiple of 80, <= 4096; every row is a DiT token); ``real_length`` = R, the
    attention-mask sum: rows ``[R, L)`` are the encoder's pad rows."""

    tokens: ttnn.Tensor
    ids: torch.Tensor
    length: int
    padded_length: int
    real_length: int
    sp_axis: int | None = None


class Flux3KleinPromptEncoder:
    MAX_LENGTH = FLUX3_KLEIN_MAX_TEXT_LENGTH
    PAD_MULTIPLE = FLUX3_KLEIN_TEXT_PAD_MULTIPLE

    def __init__(
        self,
        encoder: Qwen3VlTextEncoder,
        snapshot: str | os.PathLike,
        device: ttnn.MeshDevice,
        *,
        text_config: dict,
        output_sp_axis: int | None = None,
    ) -> None:
        """``encoder`` lives on ``device``; ``text_config`` is ``flux3_klein_text_config(snapshot)`` (head_dim and
        rope_theta); ``output_sp_axis`` is the DiT's sequence-parallel mesh axis when ``device`` IS the DiT mesh.
        ``encoder=None, device=None`` gives the host half alone (render / tokenize / positions / rope_tables), which
        the host tests use; ``encode`` then has no device to run on."""
        from transformers import AutoProcessor

        self._encoder = encoder
        self._device = device
        self._processor = AutoProcessor.from_pretrained(str(Path(snapshot) / FLUX3_KLEIN_TEXT_SUBFOLDER))
        self._processor.tokenizer.padding_side = "right"
        self._head_dim = int(text_config["head_dim"])
        self._rope_theta = flux3_klein_rope_theta(text_config)
        if output_sp_axis is not None and output_sp_axis not in range(len(tuple(device.shape))):
            raise ValueError(f"output_sp_axis {output_sp_axis} is not an axis of the mesh {tuple(device.shape)}")
        self._output_sp_axis = output_sp_axis

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    @property
    def output_sp_axis(self) -> int | None:
        return self._output_sp_axis

    # --- host ---------------------------------------------------------------------------------------------------------

    def render(self, prompt: str) -> str:
        """The chat-templated text (``text_encoder.py:30-32``), byte for byte."""
        return self._processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        """``(input_ids [1, L] int64, attention_mask [1, L] int64, L)`` by the reference recipe (``text_encoder.py:33-36``)."""
        text = self.render(prompt)
        tokenizer = self._processor.tokenizer
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.MAX_LENGTH)["input_ids"].shape[1]
        length = padded_text_length(real)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        return inputs["input_ids"], inputs["attention_mask"], length

    @staticmethod
    def positions(attention_mask: torch.Tensor) -> torch.Tensor:
        """``text_encoder.py:38``: positions ``0..R-1`` for the real rows, 1 for every pad row; ``[1, L]`` int64."""
        return (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)

    def rope_tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return flux3_klein_rope_tables(positions, head_dim=self._head_dim, rope_theta=self._rope_theta)

    # --- device -------------------------------------------------------------------------------------------------------

    def encode(self, prompt: str) -> Flux3KleinTextEncoding:
        input_ids, attention_mask, length = self.tokenize(prompt)
        real_length = int(attention_mask.sum())
        cos, sin = self.rope_tables(self.positions(attention_mask))

        tt_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        tt_mask = tensor.from_torch(attention_mask.bool(), device=self._device)
        pos_embeds = (bf16_tensor(cos, device=self._device), bf16_tensor(sin, device=self._device))

        taps = self._encoder.forward(tt_ids, attention_mask=tt_mask, pos_embeds=pos_embeds)
        tokens = ttnn.concat(taps, dim=-1)  # [1, L, 8 * hidden], layer-major per token
        for tap in taps:
            ttnn.deallocate(tap)
        for t in (tt_ids, tt_mask, *pos_embeds):
            ttnn.deallocate(t)

        out_length = length
        if self._output_sp_axis is not None:
            out_length = padded_length(length, tuple(self._device.shape)[self._output_sp_axis])
            if out_length != length:
                padded = ttnn.pad(tokens, [(0, 0), (0, out_length - length), (0, 0)], value=0.0)
                ttnn.deallocate(tokens)
                tokens = padded
            sharded = ttnn.mesh_partition(tokens, dim=1, cluster_axis=self._output_sp_axis)
            ttnn.deallocate(tokens)
            tokens = sharded

        return Flux3KleinTextEncoding(
            tokens=tokens,
            ids=text_ids(length),
            length=length,
            padded_length=out_length,
            real_length=real_length,
            sp_axis=self._output_sp_axis,
        )

    def encode_host(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``((L, 20480) bf16 host tensor, ids)``: the reference's return pair (the M4 sampler / golden-parity path)."""
        encoding = self.encode(prompt)
        rows, ids = self.to_host(encoding)
        ttnn.deallocate(encoding.tokens)
        return rows, ids

    @staticmethod
    def to_host(encoding: Flux3KleinTextEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        """Read ``encoding.tokens`` back: the real L rows only (the DiT pad rows beyond L are dropped).

        The tokens are sharded over ``sp_axis`` and REPLICATED over the other mesh axis (TP8): ``tensor.to_torch``
        with a composer read every replica (32 shards, host untilize of each) and kept one; the per-op profile of
        2026-10-21 put 2.3 s of a 4K render in that read. Here one shard per SP position is read (4 shards on the
        4x8 ring) and concatenated on the host; bit-identical rows (``test_text_readback_one_replica``)."""
        rows = read_sp_sharded_one_replica(encoding.tokens, encoding.sp_axis)[0, : encoding.length]
        return rows, encoding.ids


def read_sp_sharded_one_replica(x: ttnn.Tensor, sp_axis: int | None) -> torch.Tensor:
    """Host copy of a device tensor sharded over mesh axis ``sp_axis`` on tensor dim 1 and replicated over the other
    axis: reads ONE device per SP position (the first along the replicated axis) and concatenates on dim 1.
    ``sp_axis None``: a replicated tensor, one device read."""
    coords = [tuple(int(v) for v in c) for c in x.tensor_topology().mesh_coords()]
    shards = ttnn.get_device_tensors(x)
    if sp_axis is None:
        return ttnn.to_torch(shards[0])
    other = 1 - sp_axis
    picked = sorted(((c[sp_axis], i) for i, c in enumerate(coords) if c[other] == 0), key=lambda t: t[0])
    return torch.cat([ttnn.to_torch(shards[i]) for _, i in picked], dim=1)
