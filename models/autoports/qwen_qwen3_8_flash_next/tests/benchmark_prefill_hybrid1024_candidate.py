# SPDX-License-Identifier: Apache-2.0
"""Isolated two512 non-MoE pieces followed by one qualified MoE1024 group.

Each attention/PLE/HC piece keeps its original512-or-tail layout, fidelity and
state order. Only independent MoE rows are joined; no production modifications.
"""

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_moe1024_candidate as moe
from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_view import _decoder, _PrefillDecoder


class Hybrid1024Decoder(_PrefillDecoder):
    def _chunk_impl(self, x, *, positions, slots, tokens, page_table, decode):
        length = x.shape[2]
        if decode or length <= 512:
            return super()._chunk_impl(
                x, positions=positions, slots=slots, tokens=tokens, page_table=page_table, decode=decode
            )
        if x.shape[1] != 1 or length > 1024:
            raise ValueError("Hybrid experiment requires one prefill row through1024")
        old_large, old_l1 = self.ops.large_prefill, self.ops.l1_prefill
        pieces, mixtures, injections = [], [], []
        try:
            for start in range(0, length, 512):
                stop = min(start + 512, length)
                size = stop - start
                self.ops.large_prefill, self.ops.l1_prefill = size >= 256, True
                value = x[:, :, start:stop, :]
                position = positions if start == 0 else ttnn.add(positions, start)
                ids = None if tokens is None else tokens[:, start:stop]
                if self.ple is not None:
                    value = ttnn.add(
                        value,
                        self.ple.forward(
                            value,
                            tokens=ids,
                            positions=position,
                            slots=slots,
                            valid_lengths=self.lengths[size],
                            cache=self.ple_cache,
                        ),
                    )
                mixed, inject = self.attn_hyper.read(value, decode=False)
                if self.kind == "linear_attention":
                    update = self.attention.prefill_forward(mixed, current_positions=position, slots=slots)
                else:
                    update = self.attention.prefill_forward(
                        mixed, current_pos=position, page_table=page_table, slots=slots
                    )
                value = self.attn_hyper.write(value, update, inject)
                mixed, inject = self.mlp_hyper.read(value, decode=False)
                pieces.append(value)
                mixtures.append(mixed)
                injections.append(inject)
            # The already-qualified standalone1024 component uses DRAM matmul
            # intermediates. Keep that policy;512 non-MoE work remains in L1.
            self.ops.large_prefill, self.ops.l1_prefill = True, False
            joined = ttnn.concat(mixtures, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            update = self.ops.scatter_reduce(moe.group1024(self.mlp, joined))
            del joined, mixtures, mixed, inject
            outputs = []
            for start, value, injection in zip(range(0, length, 512), pieces, injections, strict=True):
                stop = min(start + 512, length)
                self.ops.large_prefill, self.ops.l1_prefill = stop - start >= 256, True
                out = self.mlp_hyper.write(value, update[:, :, start:stop, :], injection)
                outputs.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
            return ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        finally:
            self.ops.large_prefill, self.ops.l1_prefill = old_large, old_l1


def make_candidate(owner, lengths):
    if owner.batch_size != 32 or max(lengths) != 1024:
        raise ValueError("Expected original32-slot owner and outer1024 preparation")
    # Only native512-or-tail shapes reach attention, PLE or HC. Do not expand
    # QSA indexer offsets, PLE workspaces, norm guards or GDN RM-QK guards.
    piece_lengths = {1, 512}
    for length in lengths:
        piece_lengths.update(min(512, length - start) for start in range(0, length, 512))
    result = _decoder(owner, {}, sorted(piece_lengths))
    result.__class__ = Hybrid1024Decoder
    result.chunk_size = 1024
    for length in lengths:
        if length not in result.lengths:
            # Public prefill validation requires every outer shape prepared.
            # Correct length metadata is retained; PLE uses only piece entries.
            host = torch.zeros(1, 32, dtype=torch.int32)
            host[0, 0] = length
            result.lengths[length] = result.ops.upload(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    result.mlp.offsets1024 = result.ops.upload(
        torch.arange(128, dtype=torch.int32) * 1024, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )
    return result
