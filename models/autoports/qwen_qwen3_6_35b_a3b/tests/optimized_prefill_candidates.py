# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit prefill projection kernels; decode uses the unchanged parent path.

The production plan presents linear-attention M<=64 and full-attention M<=256.
Logical requests of 1024 rows therefore test repeated chunks, not an M1024
matmul. This candidate never changes those numerical/cache chunk contracts.
"""

import argparse
import math

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.optimized_projection_candidates import ProjectionCandidateDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import DRAM


class PrefillCandidateDecoder(ProjectionCandidateDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        prefill_family="auto",
        prefill_grid=(8, 8),
        prefill_block_w=4,
        prefill_subblock=(1, 1),
        prefill_geometry=None,
        **kwargs,
    ):
        """Override family/grid/block_w/subblock per role or via ``default``.

        Optional ``m_block``/``n_block`` set minimal-matmul block sizes; 2D and
        1D use output blocks derived from the actual padded shape and grid.
        ``rows`` maps physical M strings to geometry overrides, applied last.
        All block dimensions are tile counts. Subblocks must fit FP32's four
        destination tiles. Invalid configs fail explicitly without fallback.
        """
        if prefill_family not in ("auto", "1d", "2d", "minimal"):
            raise ValueError("prefill_family must be auto, 1d, 2d, or minimal")
        self = super().from_state_dict(state_dict, prefill_block_w=None, **kwargs)
        self.prefill_spec = dict(
            family=prefill_family, grid=list(prefill_grid), block_w=prefill_block_w, subblock=list(prefill_subblock)
        )
        self.prefill_geometry = prefill_geometry or {}
        self.prefill_effective = {}
        self.precision_policy["prefill_candidate"] = dict(
            default=self.prefill_spec, overrides=self.prefill_geometry, effective=self.prefill_effective
        )
        return self

    def _candidate_project(self, x, role, dtype):
        if x.shape[-2] == 1:
            return super()._candidate_project(x, role, dtype)
        spec = dict(self.prefill_spec)
        spec.update(self.prefill_geometry.get("default", {}))
        spec.update(self.prefill_geometry.get(role, {}))
        spec.update(spec.pop("rows", {}).get(str(x.padded_shape[-2]), {}))
        family = spec["family"]
        if family == "auto":
            return super()._candidate_project(x, role, dtype)
        weight = self.candidate_weights[role]
        mt, kt, nt = x.padded_shape[-2] // 32, weight.padded_shape[-2] // 32, weight.padded_shape[-1] // 32
        gx, gy = spec["grid"]
        sh, sw = spec["subblock"]
        kw = spec["block_w"]
        grid = self.mesh_device.compute_with_storage_grid_size()
        if not (0 < gx <= grid.x and 0 < gy <= grid.y):
            raise ValueError("prefill grid exceeds device compute/storage grid")
        if sh <= 0 or sw <= 0 or sh * sw > 4 or kw <= 0 or kt % kw:
            raise ValueError("invalid FP32 subblock or nondividing K block")
        if family == "1d":
            pm, pn = mt, math.ceil(nt / (gx * gy))
        else:
            pm, pn = math.ceil(mt / gy), math.ceil(nt / gx)
        pm, pn = math.ceil(pm / sh) * sh, math.ceil(pn / sw) * sw
        common = dict(dtype=dtype, memory_config=DRAM, compute_kernel_config=self.candidate_compute[role])
        self.prefill_effective[f"{role}:M{x.padded_shape[-2]}"] = dict(
            spec, padded_mkn=[mt * 32, kt * 32, nt * 32], per_core_m=pm, per_core_n=pn
        )
        if family == "minimal":
            # The local C++ validator accepts mixed activation/weight dtypes;
            # the Python doc's older same-dtype sentence is not a restriction.
            mb, nb = spec.get("m_block", min(pm, 4)), spec.get("n_block", min(pn, 8))
            mb, nb = math.ceil(mb / sh) * sh, math.ceil(nb / sw) * sw
            config = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kw,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            )
            self.prefill_effective[f"{role}:M{x.padded_shape[-2]}"].update(m_block=mb, n_block=nb)
            return ttnn.experimental.minimal_matmul(x, weight, config=config, **common)
        fields = dict(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=kw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=pm,
            per_core_N=pn,
            fused_activation=None,
        )
        if family == "1d":
            config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**fields, fuse_batch=False, mcast_in0=True)
        elif family == "2d":
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                **fields, out_block_h=pm, out_block_w=pn, transpose_mcast=False
            )
        else:
            raise ValueError(f"unknown prefill family {family}")
        return ttnn.linear(x, weight, program_config=config, **common)


def main():
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import run_decoder

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-kwargs", default="{}")
    p.add_argument("--activation-file")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output")
    p.add_argument("--compare-functional", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--prefill-repeats", type=int, default=3)
    p.add_argument("--decode-repeats", type=int, default=30)
    run_decoder.FunctionalDecoder = PrefillCandidateDecoder
    run_decoder.run(p.parse_args())


if __name__ == "__main__":
    main()
