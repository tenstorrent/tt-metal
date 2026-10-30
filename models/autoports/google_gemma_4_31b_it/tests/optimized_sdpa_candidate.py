"""Prefill SDPA accumulator/streaming comparison on the integrated runtime."""

from ttnn.types import BlackholeComputeKernelConfig

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder


class SDPAPrefillCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, *args, geometry, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj.rebase = geometry.get("rebase", True)
        obj.prefill_sdpa_kernel = BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=geometry["final_sdpa_fp32"],
            packer_l1_acc=True,
        )
        if geometry.get("final_decode_k"):
            for large_grid in (False, True):
                for chunk in (32, 64, 128):
                    obj.decode_programs[(large_grid, chunk)] = obj.decode_programs[
                        (large_grid, min(chunk, geometry["final_decode_k"]))
                    ]
        return obj

    def _prefill_attention_view(self, x, kv_cache, page_table, chunk_start):
        if not self.rebase:
            return page_table, chunk_start
        return super()._prefill_attention_view(x, kv_cache, page_table, chunk_start)
