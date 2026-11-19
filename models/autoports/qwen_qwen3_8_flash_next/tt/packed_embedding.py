# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-only fixed four-bit-codebook lookup with tiled integer payload storage.

The table packs eight four-bit codes into a UINT32 word. Linear words are
striped over 32 tile rows, avoiding per-embedding DRAM page padding. Setup
quantization is deliberately separate from lookup. NF4 absmax/row-scale is the
legacy default. Completed grouped caches declare their fixed FLOAT32 codebook,
group width and BF16 or packed eight-bit LUT scales explicitly in the manifest. This module does not
compute the target model's n-gram hashes.
"""

import ttnn

from .packed_embedding_cpu import NF4_VALUES, _quantize_rows, _round32, prepare_checkpoint_cache, quantization_spec
from .striped_gather import striped_gather


class PackedEmbedding:
    """Packed lookup building block; all forward arguments are device tensors."""

    def __init__(
        self,
        table,
        scales,
        codebook,
        *,
        rows=None,
        lanes=None,
        spec=None,
        scale_lanes=None,
        scale_indices=None,
        scale_codebook=None,
    ):
        self.table = table
        self.scales = scales
        self.codebook = codebook
        self.rows = rows
        self.lanes = lanes
        self.spec = quantization_spec() if spec is None else spec
        self.group_size = self.spec["group_size"]
        self.scales_per_row = self.spec["scales_per_row"]
        self.scale_lanes = scale_lanes
        self.scale_indices = scale_indices
        self.scale_codebook = scale_codebook
        self.scale_storage_per_row = self.spec.get("scale_words_per_row", self.scales_per_row)

    @classmethod
    def _from_stripes(
        cls,
        words,
        scales,
        *,
        mesh_device,
        rows,
        quantization="absmax",
        group_size=160,
        codebook_values=None,
        scale_codebook_values=None,
    ):
        import torch

        spec = quantization_spec(
            quantization=quantization,
            group_size=group_size,
            codebook_values=codebook_values,
            scale_codebook_values=scale_codebook_values,
        )
        table = ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        encoded_scales = spec.get("scale_encoding") == "uint8_fixed_log_lut"
        scale_table = ttnn.from_torch(
            scales,
            dtype=ttnn.uint32 if encoded_scales else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        scale_codebook = None
        if encoded_scales:
            levels = torch.tensor(spec["scale_codebook_values"], dtype=torch.bfloat16).reshape(1, 256)
            scale_codebook = ttnn.from_torch(
                levels.expand(32, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
        # FLOAT32 codebook preserves the exact CPU quantizer reconstruction;
        # only its final product with the BF16 scale is rounded to BF16.
        values = torch.tensor(tuple(spec["codebook_values"]) + (0.0,) * 16).reshape(1, 32).expand(32, -1).contiguous()
        codebook = ttnn.from_torch(values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        lanes = ttnn.from_torch(
            torch.arange(20).reshape(1, 20), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        scale_lanes = scale_indices = None
        if spec["scales_per_row"] > 1:
            scale_lanes = ttnn.from_torch(
                torch.arange(spec.get("scale_words_per_row", spec["scales_per_row"])).reshape(1, -1),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            scale_indices = ttnn.from_torch(
                (torch.arange(160) // group_size).reshape(1, 160),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
        return cls(
            table,
            scale_table,
            codebook,
            rows=rows,
            lanes=lanes,
            spec=spec,
            scale_lanes=scale_lanes,
            scale_indices=scale_indices,
            scale_codebook=scale_codebook,
        )

    @classmethod
    def from_torch(
        cls,
        weight,
        *,
        mesh_device,
        quantization="absmax",
        group_size=160,
        codebook_values=None,
        scale_codebook_values=None,
    ):
        """Setup only; weight is [rows,160], with manifest-defined group scales."""
        import torch

        if weight.ndim != 2 or weight.shape[1] != 160 or weight.shape[0] < 1:
            raise ValueError("Expected [rows,160] embedding table")
        rows = weight.shape[0]
        spec = quantization_spec(
            quantization=quantization,
            group_size=group_size,
            codebook_values=codebook_values,
            scale_codebook_values=scale_codebook_values,
        )
        packed, scales = _quantize_rows(
            weight,
            quantization=quantization,
            group_size=group_size,
            codebook_values=spec["codebook_values"],
            scale_codebook_values=spec.get("scale_codebook_values"),
        )
        word_storage = torch.zeros(32 * _round32((packed.numel() + 31) // 32), dtype=torch.uint32)
        scale_storage = torch.zeros(32 * _round32((scales.numel() + 31) // 32), dtype=scales.dtype)
        word_storage[: packed.numel()] = packed
        scale_storage[: scales.numel()] = scales
        return cls._from_stripes(
            word_storage.reshape(-1, 32).T.contiguous(),
            scale_storage.reshape(-1, 32).T.contiguous(),
            mesh_device=mesh_device,
            rows=rows,
            quantization=quantization,
            group_size=group_size,
            codebook_values=spec["codebook_values"],
            scale_codebook_values=spec.get("scale_codebook_values"),
        )

    @classmethod
    def from_checkpoint(
        cls,
        snapshot,
        *,
        cache_dir,
        mesh_device,
        chunk_rows=65536,
        quantization=None,
        group_size=None,
        codebook_values=None,
        scale_codebook_values=None,
    ):
        """Setup only: quantize all checkpoint rows, then upload their packed representation."""
        import json
        from pathlib import Path

        import numpy as np
        import torch

        # Existing completed caches own their quantizer identity. Explicit
        # overrides are still checked and cannot silently reuse another format.
        manifest_path = Path(cache_dir) / "manifest.json"
        metadata = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        quantization = metadata.get("quantization", "absmax") if quantization is None else quantization
        group_size = metadata.get("group_size", 160) if group_size is None else group_size
        codebook_values = metadata.get("codebook_values", NF4_VALUES) if codebook_values is None else codebook_values
        scale_codebook_values = (
            metadata.get("scale_codebook_values") if scale_codebook_values is None else scale_codebook_values
        )
        manifest = prepare_checkpoint_cache(
            snapshot,
            cache_dir,
            chunk_rows=chunk_rows,
            quantization=quantization,
            group_size=group_size,
            codebook_values=codebook_values,
            scale_codebook_values=scale_codebook_values,
        )
        words = np.memmap(
            Path(cache_dir) / "words.bin", mode="c", dtype=np.uint32, shape=(32, manifest["words_columns"])
        )
        encoded_scales = manifest.get("scale_encoding") == "uint8_fixed_log_lut"
        scales = np.memmap(
            Path(cache_dir) / "scales.bin",
            mode="c",
            dtype=np.uint32 if encoded_scales else np.uint16,
            shape=(32, manifest["scales_columns"]),
        )
        return cls._from_stripes(
            torch.from_numpy(words),
            torch.from_numpy(scales) if encoded_scales else torch.from_numpy(scales).view(torch.bfloat16),
            mesh_device=mesh_device,
            rows=manifest["rows"],
            quantization=quantization,
            group_size=group_size,
            codebook_values=codebook_values,
            scale_codebook_values=scale_codebook_values,
        )

    def lookup_words(self, *, columns, stripes):
        """columns [32,N], stripes [N,1], UINT32 tile; returns [N,1]."""
        selected = ttnn.gather(self.table, 1, columns)
        return ttnn.gather(ttnn.transpose(selected, 0, 1), 1, stripes)

    def unpack_words(self, words):
        """Return eight UINT32 code tensors, one per nibble lane."""
        return tuple(ttnn.bitwise_and(ttnn.logical_right_shift(words, shift), 15) for shift in range(0, 32, 4))

    @staticmethod
    def _lookup_striped(table, columns, stripes):
        # Direct indexed NoC reads avoid native gather's full-source scan.
        return striped_gather(table, columns, stripes)

    def __call__(self, row_ids):
        """Decode arbitrary device global row IDs [1,1,S,16], 1<=S<=32.

        Input IDs must be valid global checkpoint rows. The device hash owns
        this invariant; no host read or content-dependent Python runs here.
        Returns BF16 TILE [1,1,S,2560]. Inputs may be RM or TILE UINT32.
        """
        if len(row_ids.shape) != 4 or tuple(row_ids.shape)[:2] != (1, 1) or row_ids.shape[-1] != 16:
            raise ValueError("row_ids must have shape [1,1,S,16]")
        sequence = row_ids.shape[-2]
        if not 1 <= sequence <= 32 or row_ids.dtype != ttnn.uint32:
            raise ValueError("Expected UINT32 IDs with 1 <= S <= 32")
        count = sequence * 16
        rows = ttnn.reshape(ttnn.to_layout(row_ids, ttnn.TILE_LAYOUT), (count, 1))
        columns, stripes = self.word_addresses(rows)
        words = self._lookup_striped(self.table, columns, stripes)
        nibbles = [ttnn.reshape(value, (count, 20, 1)) for value in self.unpack_words(words)]
        codes = ttnn.reshape(ttnn.concat(nibbles, dim=2), (1, count * 160))
        indices = ttnn.repeat(codes, (32, 1))
        values = ttnn.gather(self.codebook, 1, indices)[0:1, :]
        values = ttnn.reshape(values, (count, 160))
        scale_columns, scale_stripes = self.scale_addresses(rows)
        compact = self._lookup_striped(self.scales, scale_columns, scale_stripes)
        if self.scale_codebook is not None:
            # Four UINT32 words carry sixteen 8-bit scale IDs. Only selected
            # words and a fixed tiny LUT are decoded; the full table stays put.
            byte_lanes = [
                ttnn.reshape(
                    ttnn.bitwise_and(ttnn.logical_right_shift(compact, shift), 255),
                    (count, self.scale_storage_per_row, 1),
                )
                for shift in (0, 8, 16, 24)
            ]
            scale_ids = ttnn.reshape(ttnn.concat(byte_lanes, dim=2), (1, count * self.scales_per_row))
            selected_scales = ttnn.gather(self.scale_codebook, 1, ttnn.repeat(scale_ids, (32, 1)))[0:1, :]
            compact = ttnn.reshape(selected_scales, (count, self.scales_per_row))
        if self.scales_per_row == 1:
            scales = compact
        else:
            # Native gather scans only the tiny[N,2..16] selected scale tensor,
            # never the resident full table. Each value receives its own group.
            scales = ttnn.gather(compact, 1, ttnn.repeat(self.scale_indices, (count, 1)))
        result = ttnn.typecast(ttnn.multiply(values, ttnn.typecast(scales, ttnn.float32)), ttnn.bfloat16)
        return ttnn.reshape(result, (1, 1, sequence, 2560))

    def scale_addresses(self, rows):
        """Exact UINT32[N,1] rows -> compact group-scale columns and stripes."""
        if self.scale_storage_per_row == 1:
            linear = rows
        else:
            # Even320001536 rows *10 scales stays within UINT32. Use integer
            # shifts/adds: UINT32 multiplication may dispatch through floating
            # arithmetic and lose large row addresses.
            if self.scale_storage_per_row in (2, 4, 8):
                base = ttnn.logical_left_shift(rows, self.scale_storage_per_row.bit_length() - 1)
            elif self.scale_storage_per_row == 5:
                base = ttnn.add(ttnn.logical_left_shift(rows, 2), rows)
            else:  # ten groups of16
                base = ttnn.add(ttnn.logical_left_shift(rows, 3), ttnn.logical_left_shift(rows, 1))
            linear = ttnn.add(base, self.scale_lanes)
        return ttnn.logical_right_shift(linear, 5), ttnn.bitwise_and(linear, 31)

    def word_addresses(self, rows):
        """Exact UINT32 [N,1] global rows -> [N,20] columns and stripes."""
        quotient = ttnn.logical_right_shift(rows, 3)
        remainder = ttnn.bitwise_and(rows, 7)
        # Never materialize row*20: it exceeds UINT32 over the real table.
        base = ttnn.add(ttnn.logical_left_shift(quotient, 2), quotient)
        offsets = ttnn.add(
            ttnn.add(ttnn.logical_left_shift(remainder, 4), ttnn.logical_left_shift(remainder, 2)), self.lanes
        )
        columns = ttnn.add(base, ttnn.logical_right_shift(offsets, 5))
        stripes = ttnn.bitwise_and(offsets, 31)
        return columns, stripes
