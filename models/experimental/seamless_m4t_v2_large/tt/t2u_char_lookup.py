# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precomputed T2U character tables (torch CPU + optional device-resident gather)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import ttnn

from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import from_torch_uint32_rm

TokenIds = Union[Sequence[int], torch.Tensor]


@dataclass
class T2UCharLookupTables:
    """Per-token char ids and metadata; CPU tables + optional device upload at ``bind_device``."""

    pad_token_id: int
    unk_token_id: int
    vocab_size: int
    max_token_chars: int
    char_ids_flat: torch.Tensor  # int32 [total_chars]
    char_ids_offset: torch.Tensor  # int64 [vocab_size + 1]
    char_dense: torch.Tensor  # int32 [vocab_size, max_token_chars]
    token_char_len: torch.Tensor  # int16 [vocab_size]
    subword_len: torch.Tensor  # int16 [vocab_size]
    is_punc: torch.Tensor  # bool [vocab_size]
    starts_with_space: torch.Tensor  # bool [vocab_size]
    _unk_scalar: torch.Tensor  # int32 [1]
    _device: Optional[ttnn.Device] = field(default=None, repr=False)
    _char_row_buf: Optional[torch.Tensor] = field(default=None, repr=False)
    _char_row_cap: int = field(default=0, repr=False)

    def bind_device(self, device: ttnn.Device) -> None:
        """Record device for pinned char-row uploads (tables stay on CPU for gather)."""
        self._device = device

    def _ids_tensor(self, token_ids: TokenIds) -> torch.Tensor:
        if isinstance(token_ids, torch.Tensor):
            return token_ids.to(dtype=torch.long).reshape(-1)
        return torch.tensor(token_ids, dtype=torch.long)

    def _active_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.numel() == 0:
            return ids
        pad_pos = (ids == self.pad_token_id).nonzero(as_tuple=False)
        if pad_pos.numel():
            return ids[: int(pad_pos[0].item())]
        return ids

    def t2u_ids_from_seq(
        self,
        seq_host: Sequence[int],
        *,
        eos_id: int,
        slice_start: int = 2,
        slice_end: int = -1,
    ) -> torch.Tensor:
        """``sequences[:, 2:-1]`` with EOS→pad, as a 1-D long tensor (no Python char walks)."""
        row = seq_host[slice_start:slice_end]
        if not row:
            return torch.empty(0, dtype=torch.long)
        ids = torch.tensor(row, dtype=torch.long)
        if eos_id >= 0:
            ids = torch.where(ids == eos_id, torch.tensor(self.pad_token_id, dtype=torch.long), ids)
        return ids

    def char_count_tensor(self, token_ids: TokenIds) -> torch.Tensor:
        ids = self._ids_tensor(token_ids)
        return self._char_count_from_ids(ids, self._active_ids(ids))

    def char_count_per_id(self, token_ids: TokenIds) -> List[int]:
        return self.char_count_tensor(token_ids).tolist()

    def char_ids_tensor(self, token_ids: TokenIds) -> torch.Tensor:
        return self._char_ids_from_active(self._active_ids(self._ids_tensor(token_ids)))

    def char_ids_for_sequence(self, token_ids: TokenIds) -> List[int]:
        return self.char_ids_tensor(token_ids).tolist()

    def prepare_speech(self, token_ids: TokenIds) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(char_ids int32 [C], char_counts int32 [T])`` on CPU."""
        ids = self._ids_tensor(token_ids)
        active = self._active_ids(ids)
        return self._char_ids_from_active(active), self._char_count_from_ids(ids, active)

    def prepare_speech_for_upload(
        self,
        token_ids: TokenIds,
        *,
        device: ttnn.Device,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """CPU gather + pinned row buffer + ``ttnn.from_torch`` (no Python list roundtrip)."""
        char_ids_t, cc_t = self.prepare_speech(token_ids)
        char_tt = self._char_ids_to_device(char_ids_t, device)
        return char_tt, cc_t

    def prepare_speech_from_seq(
        self,
        seq_host: Sequence[int],
        *,
        eos_id: int,
        device: ttnn.Device,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """Full fast path from decode ``seq_host`` → device char tensor + CPU char counts."""
        ids = self.t2u_ids_from_seq(seq_host, eos_id=eos_id)
        return self.prepare_speech_for_upload(ids, device=device)

    def _char_ids_to_device(self, char_ids_t: torch.Tensor, device: ttnn.Device) -> ttnn.Tensor:
        n = int(char_ids_t.numel())
        if n == 0:
            return from_torch_uint32_rm(
                device,
                torch.zeros(1, 1, dtype=torch.int32),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._char_row_buf is None or self._char_row_cap < n:
            cap = max(n, self._char_row_cap * 2 if self._char_row_cap else 256)
            try:
                self._char_row_buf = torch.empty(cap, dtype=torch.int32, pin_memory=True)
            except RuntimeError:
                self._char_row_buf = torch.empty(cap, dtype=torch.int32)
            self._char_row_cap = cap
        row = self._char_row_buf[:n]
        row.copy_(char_ids_t)
        return from_torch_uint32_rm(
            device,
            row.unsqueeze(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _char_count_from_ids(self, ids: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        n = int(ids.numel())
        out = torch.zeros(n, dtype=torch.int32)
        active_n = int(active.numel())
        if active_n == 0:
            return out

        vs = self.vocab_size
        unk = self.unk_token_id

        oov = (active < 0) | (active >= vs)
        unk_row = active == unk

        lens = self.subword_len[active.clamp(0, vs - 1)].to(torch.int32)
        punc = self.is_punc[active.clamp(0, vs - 1)]
        sws = self.starts_with_space[active.clamp(0, vs - 1)]

        next_sws = torch.zeros(active_n, dtype=torch.bool)
        if active_n > 1:
            next_sws[:-1] = sws[1:]

        inc = punc & next_sws
        dec = torch.zeros(active_n, dtype=torch.bool)
        if active_n > 1:
            dec[1:] = punc[:-1] & sws[1:]

        counts = lens + inc.to(torch.int32) - (dec & ~inc).to(torch.int32)
        special = oov | unk_row
        counts = torch.where(special, torch.ones_like(counts), counts)
        out[:active_n] = counts
        return out

    def _char_ids_from_active(self, active: torch.Tensor) -> torch.Tensor:
        n = int(active.numel())
        if n == 0:
            return self.char_ids_flat.new_empty(0)

        vs = self.vocab_size
        unk = self._unk_scalar
        ids = active

        oov = (ids < 0) | (ids >= vs)
        if oov.any():
            parts: list[torch.Tensor] = []
            off = self.char_ids_offset
            flat = self.char_ids_flat
            for sid in ids.tolist():
                sid = int(sid)
                if sid < 0 or sid >= vs:
                    parts.append(unk)
                else:
                    lo, hi = int(off[sid].item()), int(off[sid + 1].item())
                    parts.append(flat[lo:hi] if hi > lo else unk)
            return torch.cat(parts)

        dense = self.char_dense[ids.clamp(0, vs - 1)]
        lens = self.token_char_len[ids.clamp(0, vs - 1)].to(torch.long)
        if int(lens.max().item()) == 0:
            return unk.expand(n)

        max_l = int(lens.max().item())
        pos = torch.arange(max_l, dtype=torch.long)
        valid = pos.unsqueeze(0) < lens.unsqueeze(1)
        return dense[:, :max_l][valid]

    def memory_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.char_ids_flat,
                self.char_ids_offset,
                self.char_dense,
                self.token_char_len,
                self.subword_len,
                self.is_punc,
                self.starts_with_space,
                self._unk_scalar,
            )
        )


def build_t2u_char_lookup_tables(
    generation_config: Any,
    *,
    vocab_size: int,
    pad_token_id: int,
    unk_token_id: int = 1,
    space: str = "▁",
) -> T2UCharLookupTables:
    """Build lookup tables once at model init from ``generation_config`` dicts."""
    if not hasattr(generation_config, "id_to_text"):
        raise ValueError("generation_config.id_to_text required for T2U char lookup.")
    if not hasattr(generation_config, "char_to_id"):
        raise ValueError("generation_config.char_to_id required for T2U char lookup.")

    id_to_text = generation_config.id_to_text
    char_to_id = generation_config.char_to_id
    vs = int(vocab_size)

    per_token_chars: list[list[int]] = [[unk_token_id] for _ in range(vs)]
    subword_len = torch.zeros(vs, dtype=torch.int16)
    is_punc = torch.zeros(vs, dtype=torch.bool)
    starts_with_space = torch.zeros(vs, dtype=torch.bool)
    max_token_chars = 1

    for key, raw_sub in id_to_text.items():
        sid = int(key)
        if sid < 0 or sid >= vs:
            continue
        sub = str(raw_sub)
        subword_len[sid] = len(sub)
        is_punc[sid] = len(sub) == 1 and not sub.isalpha() and not sub.isnumeric() and sub != space
        starts_with_space[sid] = len(sub) > 1 and sub[0] == space
        if sid == unk_token_id:
            per_token_chars[sid] = [unk_token_id]
        else:
            per_token_chars[sid] = [int(char_to_id.get(ch, unk_token_id)) for ch in sub]
        max_token_chars = max(max_token_chars, len(per_token_chars[sid]))

    char_dense = torch.full((vs, max_token_chars), unk_token_id, dtype=torch.int32)
    token_char_len = torch.zeros(vs, dtype=torch.int16)
    offsets = torch.zeros(vs + 1, dtype=torch.int64)
    running = 0
    for sid in range(vs):
        chunk = per_token_chars[sid]
        n = len(chunk)
        token_char_len[sid] = n
        offsets[sid] = running
        running += n
        if n:
            char_dense[sid, :n] = torch.tensor(chunk, dtype=torch.int32)
    offsets[vs] = running

    if running == 0:
        char_flat = torch.tensor([unk_token_id], dtype=torch.int32)
    else:
        char_flat = torch.empty(running, dtype=torch.int32)
        cursor = 0
        for sid in range(vs):
            chunk = per_token_chars[sid]
            n = len(chunk)
            if n:
                char_flat[cursor : cursor + n] = char_dense[sid, :n]
                cursor += n

    return T2UCharLookupTables(
        pad_token_id=int(pad_token_id),
        unk_token_id=int(unk_token_id),
        vocab_size=vs,
        max_token_chars=max_token_chars,
        char_ids_flat=char_flat,
        char_ids_offset=offsets,
        char_dense=char_dense,
        token_char_len=token_char_len,
        subword_len=subword_len,
        is_punc=is_punc,
        starts_with_space=starts_with_space,
        _unk_scalar=torch.tensor([unk_token_id], dtype=torch.int32),
    )
