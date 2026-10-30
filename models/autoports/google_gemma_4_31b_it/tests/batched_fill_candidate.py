"""Batched paged-fill experiment; indices are prepared outside measured runtime."""
import argparse

import torch

import ttnn
from models.autoports.google_gemma_4_31b_it.tests import run_fused_decoder, time_fusion_candidate
from models.autoports.google_gemma_4_31b_it.tests.fusion_pre_batched_fill import FusedDecoder


class BatchedFillCandidate(FusedDecoder):
    batch = 1

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj.batch_indices = ttnn.from_torch(
            torch.arange(cls.batch, dtype=torch.int32),
            device=obj.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return obj

    def prefill_chunk_forward(self, x, *, cos, sin, kv_cache, page_table, chunk_page_table, chunk_start, valid_tokens):
        """x [B,1,S,H]; RoPE [B,1,S,D]; returns [B,1,valid_tokens,H].

        S is the padded physical length; 0 < valid_tokens <= S. Each request
        owns distinct cache pages. Resume retains all prefix pages and starts at
        a page boundary. The final partial page is private to this request.
        """
        if not 0 < valid_tokens <= x.shape[2] or chunk_start % kv_cache[0].shape[2]:
            raise ValueError("Invalid logical length or non-page-aligned chunk start")
        if chunk_start + valid_tokens > self.config.max_position_embeddings:
            raise ValueError("Context exceeds pinned HF contract")
        q, k, v = self._qkv(x, cos, sin, decode=False)
        ttnn.experimental.paged_fill_cache(kv_cache[0], k, chunk_page_table, batch_idx_tensor=self.batch_indices)
        ttnn.experimental.paged_fill_cache(kv_cache[1], v, chunk_page_table, batch_idx_tensor=self.batch_indices)
        a = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            page_table,
            chunk_start_idx=chunk_start,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=32
            ),
            compute_kernel_config=self.kernel,
        )
        out = self._finish(x, a, decode=False)
        return self._sequence_view(out, valid_tokens)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=1025)
    p.add_argument("--result", required=True)
    p.add_argument("--timing", action="store_true")
    a = p.parse_args()
    BatchedFillCandidate.batch = a.batch
    if a.timing:
        time_fusion_candidate.FusedDecoder = BatchedFillCandidate
        time_fusion_candidate.run(a.layer, a.batch, a.result)
    else:
        run_fused_decoder.FusedDecoder = BatchedFillCandidate
        a.synthetic = a.extended = a.new_bucket_case = a.skip_exact_maximum = False
        run_fused_decoder.run(a)
