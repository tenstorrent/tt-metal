# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Serve Sana's packed MiniMax-H3 DiT contract with the TT transformer.

Sana's `MiniMaxH3X0Model` stays on the host: it maps noise levels to H3 timesteps, noises clean context rows and
converts velocity to x0. Only its inner `dit.forward(**kwargs)` is replaced by `StreamRenderDiT.__call__`, which
takes the same packed rows (`x`, `audio_x`, `prompt_embeds` scattered by the `*_pos_info` row maps) and returns the
same `(video_logits, audio_logits)`.

Three calls reach it, told apart by the cache arguments Sana's streaming-prefix mixin passes:

* no cache -- the bootstrap window, one dense document;
* `update_past_key_values=True` with a `PrefixCapture` -- the one-way prefix (Qwen context, picture, sink rows)
  attending only to itself; every layer's K/V is kept as a `PrefixBank`;
* a `PrefixBank` -- the dynamic rows of a continuation, attending to `[bank ; themselves]`.

Picture, reference and target video rows all enter through `proj_in`, so the TT source table is
`[text | audio | video]` with the video stream holding every video-like row in packed order.

With `trace`, continuations are traced, one trace per bank, from each bank's second use: by then both banks of the
session (one per denoising time) and their masks exist, so nothing long-lived is allocated after a capture. A bank's
mask is one buffer rewritten in place. `release_traces` must run before the next session allocates its banks. A
continuation is device-bound, so the trace is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import KVBankSlot, prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....utils.tensor import from_torch, local_device_to_torch
from ..packing import MINIMAX_H3_MODALITY_NUM, build_rope_tables

_TILE = 32
# Pad keys sit at -1e9 rather than -inf so a softmax row can never be all -inf.
_MASKED = -1e9
_TIMESTEP_SLOTS = 4
_SPARSE_BLOCK = 64


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def prefix_bank_mask(bank_rows: int, bank_padded_rows: int, rows: int, pad_to: int, sp_factor: int) -> torch.Tensor:
    """`[1, 1, pad_to / sp, bank_padded_rows + pad_to]` additive mask hiding both blocks' pad keys."""
    mask = torch.zeros(1, 1, pad_to // sp_factor, bank_padded_rows + pad_to)
    mask[..., bank_rows:bank_padded_rows] = _MASKED
    mask[..., bank_padded_rows + rows :] = _MASKED
    return mask


def sparse_block_tables(
    bank_rows: int, bank_padded_rows: int, rows: int, pad_to: int, sp_factor: int, heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(block_mask, block_counts)` of the kept-only `sparse_ring_sdpa` over `[own ; bank]` shards (`KVBankSlot`).

    Each device's ring shard is its own `pad_to / sp` rows then its `bank_padded_rows / sp` bank rows, real rows
    first in each part. Own-row Q blocks keep every block that has a real token; the bank's Q blocks keep none.
    """
    own_local, bank_local = pad_to // sp_factor, bank_padded_rows // sp_factor
    blocks = -(-(own_local + bank_local) // _SPARSE_BLOCK)
    nb_pad = _round_up(blocks, _TILE)
    counts = torch.zeros(sp_factor * nb_pad, dtype=torch.int32)
    for shard in range(sp_factor):
        real = torch.zeros(blocks * _SPARSE_BLOCK, dtype=torch.int32)
        real[: max(0, min(own_local, rows - shard * own_local))] = 1
        real[own_local : own_local + max(0, min(bank_local, bank_rows - shard * bank_local))] = 1
        counts[shard * nb_pad : shard * nb_pad + blocks] = real.view(blocks, _SPARSE_BLOCK).sum(-1)
    block_mask = torch.zeros(1, heads, nb_pad, sp_factor * nb_pad, dtype=torch.int32)
    block_mask[:, :, : own_local // _SPARSE_BLOCK] = (counts > 0).int()
    return block_mask, counts.reshape(1, 1, 1, -1)


@dataclass
class PrefixCapture:
    """Passed as `past_key_values` to a prefix fill; `bank` is set by the forward."""

    bank: PrefixBank | None = None


@dataclass
class PrefixBank:
    """Per-layer prefix K/V of one (branch, timestep), each device's SP shard, plus its row counts.

    `sparse` holds the kept-only `sparse_ring_sdpa` block tables per continuation length; `mask` the masked
    fallback's.
    """

    slots: list[KVBankSlot]
    rows: int
    padded_rows: int
    mask: ttnn.Tensor | None = None
    mask_key: tuple[int, int] | None = None
    sparse: dict[tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]] = field(default_factory=dict)
    uses: int = 0

    def release(self) -> None:
        for slot in self.slots:
            for tensor in (slot.key, slot.value):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            slot.key = slot.value = slot.mask = slot.block_mask = slot.block_counts = slot.query_pad = None
        if self.mask is not None:
            ttnn.deallocate(self.mask)
        self.mask = self.mask_key = None
        for tables in self.sparse.values():
            for tensor in tables:
                ttnn.deallocate(tensor)
        self.sparse.clear()


def _positions(info: dict) -> torch.Tensor:
    return info["position_ids"].reshape(-1).to(torch.long).cpu()


class StreamRenderDiT:
    def __init__(
        self,
        transformer: MiniMaxH3Transformer3DModel,
        *,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int,
        sp_factor: int,
        rope_freq_dim: int,
        rope_theta: float,
        head_dim: int,
        trace: bool = False,
        kv_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ) -> None:
        self.transformer = transformer
        self.trace = trace
        self.kv_dtype = kv_dtype
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.sp_factor = sp_factor
        self.rope_freq_dim = rope_freq_dim
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.prompt_cap = 0
        self.forwards = {"full": 0, "prefix": 0, "suffix": 0}
        self._query_pad: ttnn.Tensor | None = None

    # ------------------------------------------------------------------ uploads

    def _replicated(self, t: torch.Tensor, dtype: ttnn.DataType, layout=ttnn.Layout.TILE) -> ttnn.Tensor:
        return from_torch(t, device=self.mesh_device, dtype=dtype, layout=layout, mesh_axes=[..., None, None])

    def _sp_rows(self, values: torch.Tensor) -> ttnn.Tensor:
        return from_torch(
            values.to(torch.int32).reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, self.sp_axis],
        )

    def _indices(self, values: torch.Tensor) -> ttnn.Tensor:
        return self._replicated(values.to(torch.int32).reshape(1, 1, 1, -1), ttnn.uint32, ttnn.Layout.ROW_MAJOR)

    def _rope(self, position_ids: torch.Tensor, pad_to: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        pad = pad_to - position_ids.shape[0]
        position_ids = torch.cat([position_ids, torch.zeros(pad, 3, dtype=position_ids.dtype)])
        cos, sin = build_rope_tables(position_ids, rope_freq_dim=self.rope_freq_dim, rope_theta=self.rope_theta)
        cos, sin = prepare_rope_tables(cos, sin, self.head_dim)
        return tuple(
            from_torch(
                t.reshape(1, 1, pad_to, -1),
                device=self.mesh_device,
                dtype=ttnn.float32,
                mesh_axes=[..., self.sp_axis, None],
            )
            for t in (cos, sin)
        )

    def _bank_mask(self, bank: PrefixBank, rows: int, pad_to: int) -> ttnn.Tensor:
        key = (rows, pad_to)
        if bank.mask_key != key:
            # 0 and -1e9 survive block-float exactly, and a bfp4 mask is a quarter of the SDPA's mask traffic.
            mask = self._replicated(
                prefix_bank_mask(bank.rows, bank.padded_rows, rows, pad_to, self.sp_factor), ttnn.bfloat4_b
            )
            if bank.mask is None:
                bank.mask = mask
            elif tuple(bank.mask.shape) == tuple(mask.shape):
                ttnn.copy(mask, bank.mask)
                ttnn.deallocate(mask)
            else:
                raise ValueError(f"a bank's continuation length changed its padding: {bank.mask_key} -> {key}")
            bank.mask_key = key
        return bank.mask

    def _bank_sparse(self, bank: PrefixBank, rows: int, pad_to: int):
        """`(block_mask, block_counts, query_pad)` for the kept-only sparse ring attention, or None when this
        continuation's shard is not whole 64-row blocks (a block would straddle own rows and bank rows)."""
        sp = self.sp_factor
        own_local, bank_local = pad_to // sp, bank.padded_rows // sp
        if own_local % _SPARSE_BLOCK:
            return None
        key = (rows, pad_to)
        heads = self.transformer.transformer_blocks[0].attn.n_local_heads
        if key not in bank.sparse:
            tables = sparse_block_tables(bank.rows, bank.padded_rows, rows, pad_to, sp, heads)
            bank.sparse[key] = tuple(self._replicated(t, ttnn.uint32, ttnn.Layout.ROW_MAJOR) for t in tables)
        if self._query_pad is None or self._query_pad.shape[2] != bank_local:
            self._query_pad = self._replicated(torch.zeros(1, heads, bank_local, self.head_dim), ttnn.bfloat16)
        return (*bank.sparse[key], self._query_pad)

    def release_traces(self) -> None:
        self.transformer.release_traces()

    def _bind_bank(self, slots: list[KVBankSlot] | None) -> None:
        blocks = self.transformer.transformer_blocks
        for index, block in enumerate(blocks):
            block.attn.kv_bank = None if slots is None else slots[index]

    # ------------------------------------------------------------------ forward

    def __call__(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        cache = kwargs.get("past_key_values")
        if isinstance(cache, PrefixCapture):
            if not kwargs.get("update_past_key_values"):
                raise ValueError("a prefix capture must be a cache update")
            mode = "prefix"
        elif isinstance(cache, PrefixBank):
            mode = "suffix"
        elif cache is None:
            mode = "full"
            ranges = kwargs.get("q_ranges")
            if ranges is not None and len(ranges) != 1:
                raise ValueError("a cache-free forward must be one dense document (the bootstrap window)")
        else:
            raise TypeError(f"unsupported past_key_values {type(cache).__name__}")

        x = kwargs["x"]
        audio_x = kwargs["audio_x"]
        seq_len = int(x.shape[1])
        img_pos = _positions(kwargs["img_pos_info"])
        audio_pos = _positions(kwargs["audio_pos_info"])
        text_pos = _positions(kwargs["text_pos_info"])
        out_pos = _positions(kwargs["img_pos_for_infer_output_info"])
        if text_pos.numel() + img_pos.numel() + audio_pos.numel() != seq_len:
            raise ValueError("every packed row must be a text, video or audio row")

        tt = self.transformer
        text_rows = text_pos.numel()
        if text_rows:
            prompt = kwargs["prompt_embeds"][:text_rows].to(torch.bfloat16).cpu()
            self.prompt_cap = max(self.prompt_cap, _round_up(text_rows, _TILE))
            prompt = torch.cat([prompt, prompt.new_zeros(self.prompt_cap - text_rows, prompt.shape[-1])])
            tt.prepare_static_sources(
                prompt_1BLP=self._replicated(prompt.reshape(1, 1, self.prompt_cap, -1), ttnn.bfloat16),
                prompt_len=text_rows,
                prompt_cap=self.prompt_cap,
            )
        elif self.prompt_cap == 0:
            raise RuntimeError("the first forward of a session must carry the text rows")

        v_rows, a_rows = img_pos.numel(), audio_pos.numel()
        v_cap, a_cap = _round_up(max(v_rows, 1), _TILE), _round_up(max(a_rows, 1), _TILE)
        video = x[0].index_select(0, img_pos).to(torch.bfloat16).cpu()
        audio = audio_x[0].index_select(0, audio_pos).to(torch.bfloat16).cpu()
        video = torch.cat([video, video.new_zeros(v_cap - v_rows, video.shape[-1])])
        audio = torch.cat([audio, audio.new_zeros(a_cap - a_rows, audio.shape[-1])])

        # Source table `[text (prompt_cap) | audio (a_cap) | video (v_cap)]`; see `forward`.
        pad_to = _round_up(seq_len, self.sp_factor * _TILE)
        assembly = torch.zeros(pad_to, dtype=torch.int64)
        assembly[text_pos] = torch.arange(text_rows)
        assembly[audio_pos] = self.prompt_cap + torch.arange(a_rows)
        assembly[img_pos] = self.prompt_cap + a_cap + torch.arange(v_rows)

        inverse = kwargs["inverse_indices"].reshape(-1).to(torch.long).cpu()
        tags = kwargs["token_tags"].reshape(-1).to(torch.long).cpu().clamp(min=0)
        timestep_idx = torch.zeros(pad_to, dtype=torch.int64)
        timestep_idx[:seq_len] = inverse
        adaln = torch.zeros(pad_to, dtype=torch.int64)
        adaln[:seq_len] = inverse * MINIMAX_H3_MODALITY_NUM + tags
        timesteps = kwargs["unique_timesteps"].reshape(-1).to(torch.float32).cpu()
        if timesteps.numel() > _TIMESTEP_SLOTS:
            raise ValueError(f"{timesteps.numel()} distinct timesteps exceed the {_TIMESTEP_SLOTS} AdaLN slots")
        # A fixed slot count keeps the time embedding's shape, and so every traced buffer, the same for every forward;
        # no row's AdaLN index points at a pad slot.
        timesteps = torch.cat([timesteps, timesteps[:1].expand(_TIMESTEP_SLOTS - timesteps.numel())])
        timesteps = timesteps.reshape(1, 1, -1, 1)

        video_out = torch.zeros(v_cap, dtype=torch.int64)
        video_out[:v_rows] = img_pos
        audio_out = torch.zeros(a_cap, dtype=torch.int64)
        audio_out[:a_rows] = audio_pos

        rope_cos, rope_sin = self._rope(kwargs["img_position_ids"][0].cpu(), pad_to)

        traced = False
        if mode == "prefix":
            slots = [KVBankSlot(capture=True, kv_dtype=self.kv_dtype) for _ in tt.transformer_blocks]
        elif mode == "suffix":
            traced = self.trace and cache.uses > 0
            cache.uses += 1
            sparse = self._bank_sparse(cache, seq_len, pad_to)
            mask = None if sparse is not None else self._bank_mask(cache, seq_len, pad_to)
            block_mask, block_counts, query_pad = sparse if sparse is not None else (None, None, None)
            slots = cache.slots
            for slot in slots:
                slot.capture = False
                slot.mask = mask
                slot.block_mask, slot.block_counts, slot.query_pad = block_mask, block_counts, query_pad
        else:
            slots = None
        self._bind_bank(slots)
        try:
            video_out_tt, audio_out_tt = tt.forward(
                video_1BVC=self._replicated(video.reshape(1, 1, v_cap, -1), ttnn.bfloat16),
                audio_1BAC=self._replicated(audio.reshape(1, 1, a_cap, -1), ttnn.bfloat16),
                assembly_indices=self._indices(assembly),
                video_out_indices=self._indices(video_out),
                audio_out_indices=self._indices(audio_out),
                timestep=self._replicated(timesteps, ttnn.float32),
                adaln_indices=self._sp_rows(adaln),
                timestep_indices=self._sp_rows(timestep_idx),
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                logical_n=self._replicated(
                    torch.tensor([seq_len], dtype=torch.int64).reshape(1, 1, 1, 1), ttnn.uint32, ttnn.Layout.ROW_MAJOR
                ),
                pad_to=pad_to,
                traced=traced,
                trace_key=id(cache) if traced else None,
                # The schedule repeats a few timestep sets every chunk, so the blocks keep their adaLN tables per set.
                modulation_key=tuple(timesteps.flatten().tolist()),
            )
        finally:
            self._bind_bank(None)
        self.forwards[mode] += 1
        if mode == "prefix":
            cache.bank = PrefixBank(slots=slots, rows=seq_len, padded_rows=pad_to)

        video_rows = local_device_to_torch(video_out_tt).reshape(v_cap, -1)[:v_rows].float()
        audio_rows = local_device_to_torch(audio_out_tt).reshape(a_cap, -1)[:a_rows].float()
        ttnn.deallocate(video_out_tt)
        ttnn.deallocate(audio_out_tt)

        # Video logits come back for the output rows only, in `img_pos_for_infer_output_info` order.
        row_of = torch.full((seq_len,), -1, dtype=torch.long)
        row_of[img_pos] = torch.arange(v_rows)
        selected = row_of[out_pos]
        if (selected < 0).any():
            raise ValueError("an output row is not a video row")
        video_logits = video_rows.index_select(0, selected)
        audio_logits = audio_rows
        if not kwargs.get("skip_mask_out_condition", False):
            video_logits = video_logits * kwargs["update_mask"].reshape(-1, 1).to(video_logits).cpu()
            update_audio = kwargs.get("update_audio_mask")
            if update_audio is not None:
                audio_logits = audio_logits * update_audio.reshape(-1, 1).to(audio_logits).cpu()
        device = x.device
        return video_logits.to(device), audio_logits.to(device)
