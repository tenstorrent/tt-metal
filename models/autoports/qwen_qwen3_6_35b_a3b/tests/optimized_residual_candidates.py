# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-attention decode residual-layout experiments; no runtime selection claim."""

import argparse

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.optimized_projection_candidates import ProjectionCandidateDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import DRAM


class ResidualCandidateDecoder(ProjectionCandidateDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        residual_mode="sharded",
        residual_return_dram=False,
        residual_moe_reshard=False,
        projection_geometry=None,
        **kwargs,
    ):
        if residual_mode not in ("sharded", "control"):
            raise ValueError("residual_mode must be sharded or control")
        geometry = {role: dict(spec) for role, spec in (projection_geometry or {}).items()}
        if residual_mode == "sharded" and hf_config.layer_types[layer_idx] == "full_attention":
            # Output N=2048 at two tiles/core gives exactly the existing
            # 32-core, 64-element-wide RMSNorm/residual layout.
            geometry.setdefault("attention_out", {})["per_core_n"] = 2
        self = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            projection_geometry=geometry,
            **kwargs,
        )
        self.residual_mode = residual_mode
        self.residual_return_dram = residual_return_dram
        self.residual_moe_reshard = residual_moe_reshard
        self._residual_decode_active = False
        if residual_mode == "sharded" and self.kind == "full_attention":
            w, im, _, pc = self.candidate_decode["attention_out"]
            self.candidate_decode["attention_out"] = (w, im, self.norm_memory, pc)
        self.precision_policy["residual_candidate"] = dict(
            mode=residual_mode,
            applicable=self.kind == "full_attention",
            residual_layout="32-core width-sharded L1, shard 32x64" if residual_mode == "sharded" else "inherited",
            attention_out_per_core_n=2 if residual_mode == "sharded" else None,
            return_dram=residual_return_dram,
            moe_output_reshard=residual_moe_reshard,
            prefill_singletons="inherited interleaved-return path via cache_boundary=False",
        )
        return self

    def _candidate_project(self, x, role, dtype):
        if self._residual_decode_active and role == "attention_out" and x.shape[-2] == 1:
            w, im, om, pc = self.candidate_decode[role]
            return ttnn.linear(
                ttnn.to_memory_config(x, im),
                w,
                dtype=dtype,
                program_config=pc,
                compute_kernel_config=self.candidate_compute[role],
                memory_config=om,
            )
        return super()._candidate_project(x, role, dtype)

    def _forward(
        self, hidden_states, *, state, decode, positions=None, start_pos=0, cache_boundary=True, projection_k=512
    ):
        if self.kind != "full_attention" or not decode or not cache_boundary or self.residual_mode != "sharded":
            return super()._forward(
                hidden_states,
                state=state,
                decode=decode,
                positions=positions,
                start_pos=start_pos,
                cache_boundary=cache_boundary,
                projection_k=projection_k,
            )
        # The flag changes dispatch only during this synchronous Python call;
        # captured TT commands contain no host-dependent data or callback.
        self._residual_decode_active = True
        try:
            residual = ttnn.to_memory_config(hidden_states, self.norm_memory)
            normalized = self.norm(residual, "input_layernorm.weight")
            attention = self._full_attention(normalized, state, positions, decode=True, start_pos=start_pos)
            residual = ttnn.add(residual, attention, memory_config=self.norm_memory)
            normalized = self.norm(residual, "post_attention_layernorm.weight")
            # Router/sparse expert consumers retain their proven interleaved
            # contract. Only this branch input changes memory configuration.
            moe = self._moe(ttnn.to_memory_config(normalized, DRAM))
            if self.residual_moe_reshard:
                moe = ttnn.to_memory_config(moe, self.norm_memory)
            result = ttnn.add(residual, moe, memory_config=self.norm_memory)
            return ttnn.to_memory_config(result, DRAM) if self.residual_return_dram else result
        finally:
            self._residual_decode_active = False


def main():
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import run_decoder

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-kwargs", default="{}")
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
    run_decoder.FunctionalDecoder = ResidualCandidateDecoder
    run_decoder.run(p.parse_args())


if __name__ == "__main__":
    main()
