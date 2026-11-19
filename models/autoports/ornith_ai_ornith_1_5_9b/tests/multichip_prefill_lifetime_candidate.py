# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminate large-chunk L1 lifetime, norm staging, and minimal-matmul CB costs.

The default changes only reference lifetimes in the existing prefill loop.
Optional --norm-quantize-first and --minimal-k-block apply separately. The
frozen reference and geometry modules are patched only inside this process.
"""

import argparse
import sys

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_large_chunk_geometry as geometry


class PrefillLifetimeCandidate(geometry.LargeChunkGeometry):
    norm_quantize_first = False
    minimal_k_block = None

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        layer = super().from_state_dict(*args, **kwargs)
        if cls.minimal_k_block is not None:
            # This is the actual config used by the control's minimal_matmul
            # calls, including the direct packed-SwiGLU call inside finish().
            layer.prefill_program = ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=cls.minimal_k_block,
                N_block_size=8,
                subblock_h=1,
                subblock_w=2,
                compute_with_storage_grid_size=(11, 10),
            )
        print(
            "PREFILL_LIFETIME_POLICY",
            dict(
                release_chunk_locals=True,
                norm_quantize_first=cls.norm_quantize_first,
                minimal_k_block=cls.minimal_k_block if cls.minimal_k_block is not None else 16,
            ),
            flush=True,
        )
        return layer

    def norm(self, x, name):
        if not (
            self.norm_quantize_first
            and self.residual_layout == "replicated"
            and x.shape[2] >= 512
            and name in ("input_layernorm", "post_attention_layernorm")
        ):
            return super().norm(x, name)
        out = ttnn.rms_norm(
            x,
            weight=self.weights[name + ".weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.typecast(out, ttnn.bfloat8_b)
        return ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)

    def prefill_forward(self, x, *, state, start_position=0, page_table=None):
        """Consume a logical [1,1,S,4096] segment and update its request state.

        For continuation the caller supplies state from exactly start_position.
        Padded rows are never part of the logical recurrent sequence.
        """
        n = x.shape[2]
        assert x.shape[-1] == self.hidden_width
        assert 0 < n and 0 <= start_position and start_position + n <= self.config.max_position_embeddings
        outputs = []
        offset = 0
        while offset < n:
            absolute = start_position + offset
            if self.kind == "full_attention" and absolute % 32:
                chunk = x[:, :, offset : offset + 1, :]
                outputs.append(
                    ttnn.to_memory_config(
                        self.decode_forward(
                            chunk,
                            state=state,
                            page_table=page_table,
                            current_position=self.positions[absolute : absolute + 1],
                        ),
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                del chunk
                offset += 1
                continue
            count = min(n - offset, self.chunk_size)
            chunk = x[:, :, offset : offset + count, :]
            if self.kind == "full_attention" and count % 32:
                chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, 32 - count % 32), (0, 0)], 0.0)
            h = self.norm(chunk, "input_layernorm")
            mixed = (
                self.linear_attention(h, state, decode=False)
                if self.kind == "linear_attention"
                else self.full_attention(h, state, page_table=page_table, position=absolute, decode=False)
            )
            del h
            outputs.append(ttnn.to_memory_config(self.finish(chunk, mixed), ttnn.DRAM_MEMORY_CONFIG)[:, :, :count, :])
            del mixed, chunk
            offset += count
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--norm-quantize-first", action="store_true")
    parser.add_argument("--minimal-k-block", type=int, choices=(4, 8, 16))
    args, remaining = parser.parse_known_args()
    PrefillLifetimeCandidate.norm_quantize_first = args.norm_quantize_first
    PrefillLifetimeCandidate.minimal_k_block = args.minimal_k_block
    original = geometry.LargeChunkGeometry
    original_argv = sys.argv[:]
    try:
        geometry.LargeChunkGeometry = PrefillLifetimeCandidate
        sys.argv = [sys.argv[0], *remaining]
        geometry.main()
    finally:
        geometry.LargeChunkGeometry = original
        sys.argv = original_argv


if __name__ == "__main__":
    main()
