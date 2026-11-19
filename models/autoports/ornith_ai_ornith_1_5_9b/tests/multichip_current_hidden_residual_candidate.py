# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Current-production hidden residual families, including persistent norm/activation AG.

The public residual is TP4 column-sharded BF16 [1,1,B,4096] globally, width1024
per rank. All decoder boundaries retain that contract. Only the normalization
consumer gathers full activations; no post-RS restore to replicated residuals.
"""

import argparse
import json
import sys
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

POLICIES = {
    "separate": (False, False, False, False, False),
    "persistent_all": (True, False, False, False, False),
    "bf8": (False, False, False, True, True),
    "bf8_persistent_all": (True, False, False, True, True),
    "gather": (True, True, False, False, False),
    "scatter": (True, False, True, False, False),
    "fused_all": (True, True, True, False, False),
    "mixed_fused_all": (True, True, True, True, False),
}


class CurrentHiddenResidualCandidate(MultichipDecoder):
    packed_hidden = True
    batch_size = 32
    persistent = False
    fused_gather_enabled = False
    fused_scatter_enabled = False
    communication_dtype = ttnn.bfloat16
    norm_communication_dtype = ttnn.bfloat16

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        kwargs.update(native_local=False, residual_layout="sharded", collective="separate")
        self = super().from_state_dict(*args, **kwargs)
        self.prefill_headroom = ttnn.get_memory_view(self.device, ttnn.BufferType.L1).total_bytes_per_bank - 64 * 1024
        self.rs_index = 0
        self.persistent_rs, self.persistent_stats, self.persistent_activation = [], {}, {}
        self.persistent_fused_rs = {}
        if self.persistent:
            n = self.batch_size

            def empty(shape, dtype, mem=ttnn.L1_MEMORY_CONFIG):
                return ttnn.empty(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mem)

            # Separate roles AND layers own disjoint banks. Stats are always BF16.
            for role in ("input_layernorm", "post_attention_layernorm"):
                self.persistent_stats[role] = empty((1, 1, n, 128), ttnn.bfloat16)
                self.persistent_activation[role] = empty(
                    (1, 1, 32 if self.fused_gather_enabled else n, 4096),
                    self.norm_communication_dtype,
                    ttnn.DRAM_MEMORY_CONFIG if self.fused_gather_enabled else ttnn.L1_MEMORY_CONFIG,
                )
            if self.fused_scatter_enabled:
                projection = "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"
                self.persistent_fused_rs = {
                    role: empty((1, 1, 32, 1024), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)
                    for role in (projection, "mlp.down_proj")
                }
            else:
                # Linear-topology RS doubles padded dimension0 for bidirectional staging.
                self.persistent_rs = [
                    [
                        empty((2, 1, 32, 4096), self.communication_dtype),
                        empty((1, 1, n, 1024), self.communication_dtype),
                    ]
                    for _ in range(2)
                ]
        return self

    def family_gather(self, x, bank=None):
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=self.device,
            persistent_output_tensor=bank,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def decode_norm(self, x, name):
        assert name in ("input_layernorm", "post_attention_layernorm")
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute, dtype=ttnn.bfloat16)
        stats = self.family_gather(stats, self.persistent_stats.get(name))
        result = ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self.config.rms_norm_eps,
            weight=self.weights[name + ".weight"],
            compute_kernel_config=self.compute,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        result = ttnn.typecast(result, self.norm_communication_dtype)
        return result if self.fused_gather_enabled else self.family_gather(result, self.persistent_activation.get(name))

    def fused_gather(self, x, name, role, swiglu=False):
        n = x.shape[2]
        if n % 32:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        # Gather staging has the input dtype. Current fused policies use BF16 norm
        # activations; optional BF8 policies use separate AG, avoiding mixed-format staging.
        weight = self.weights[name + ".weight"]
        out = ttnn.experimental.all_gather_minimal_matmul_async(
            x,
            ttnn.reshape(weight, (1, 1, *weight.shape)),
            fuse_swiglu=swiglu,
            persistent_output_buffer=self.persistent_activation.get(role),
            config=ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=16,
                N_block_size=4,
                subblock_h=1,
                subblock_w=2,
                compute_with_storage_grid_size=(8, 4),
            ),
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            topology=self.ccl.topology,
            cluster_axis=1,
            num_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.matmul_compute(name),
            force_transpose=True,
            num_workers_per_link=4,
        )[0]
        return out[:, :, :n, :]

    def decode_linear(self, x, name):
        if self.fused_gather_enabled and name == "packed_input":
            return self.fused_gather(x, name, "input_layernorm")
        if self.fused_scatter_enabled and name in ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"):
            n = x.shape[2]
            x = ttnn.typecast(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), self.communication_dtype)
            if n % 32:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            # The fused API accepts a preallocated BF16 output matching padded M.
            # A distinct bank per row projection keeps the attention result live.
            result = ttnn.experimental.matmul_reduce_scatter_prefill(
                x,
                self.weights[name + ".weight"],
                cluster_axis=1,
                scatter_dim=-1,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.mlp_fidelity if name.startswith("mlp.") else self.attention_fidelity,
                    math_approx_mode=False,
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                l1_headroom=self.prefill_headroom,
                output_tensor=self.persistent_fused_rs.get(name),
            )
            return ttnn.to_memory_config(result[:, :, :n, :], ttnn.L1_MEMORY_CONFIG)
        im, om, config = self.dram_layouts[name]
        result = ttnn.linear(
            ttnn.to_memory_config(x, im),
            self.dram_weights[name],
            program_config=config,
            compute_kernel_config=self.matmul_compute(name),
            memory_config=om,
            dtype=ttnn.bfloat16,
        )
        return ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)

    def decode_reduce(self, x):
        if self.fused_scatter_enabled:
            assert x.shape[-1] == 1024
            return ttnn.typecast(x, ttnn.bfloat16)
        assert x.shape[-1] == 4096
        bank = self.rs_index
        self.rs_index = 1 - bank
        x = ttnn.typecast(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), self.communication_dtype)
        result = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=self.persistent_rs[bank] if self.persistent else None,
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.typecast(result, ttnn.bfloat16)

    def decode_finish(self, x, mixed):
        residual = ttnn.add(x, mixed)
        h = self.decode_norm(residual, "post_attention_layernorm")
        if self.fused_gather_enabled:
            h = self.fused_gather(h, "packed_mlp", "post_attention_layernorm", True)
        else:
            gate = self.decode_linear(h, "mlp.gate_proj")
            up = self.decode_linear(h, "mlp.up_proj")
            h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual, self.decode_reduce(self.decode_linear(h, "mlp.down_proj")))

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        """TP4 BF16 [1,1,B,1024] per rank, 1<=B<=32, local L1 residual across layers.

        Request attention/cache state stays independent. Dense projections, norms,
        residual adds and reductions share the physical tile. The caller packs at
        stack entry and unpacks only at its exit, outside decoder layer boundaries.
        """
        n = x.shape[2]
        assert self.residual_layout == "sharded" and 1 <= n <= 32
        assert n == self.batch_size
        self.rs_index = 0
        assert n == len(states) == len(current_positions) == len(page_tables)
        regions = [
            (
                (id(st.conv), id(st.recurrent), st.state_row)
                if self.kind == "linear_attention"
                else (id(st.key), id(st.value), st.page_offset)
            )
            for st in states
        ]
        assert len(set(regions)) == n, "A decode batch must contain distinct request state regions"
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        packed = self.decode_linear(self.decode_norm(x, "input_layernorm"), "packed_input")
        pooled_linear = (
            self.kind == "linear_attention"
            and all(
                st.conv is states[0].conv
                and st.recurrent is states[0].recurrent
                and getattr(st, "state_row", None) == i
                for i, st in enumerate(states)
            )
            and tuple(states[0].conv.shape) == (3, 1, n, 2048)
            and states[0].conv.layout == ttnn.TILE_LAYOUT
            and states[0].recurrent.shape[0] == n
        )
        independent_linear = all(getattr(st, "state_row", None) is None for st in states)
        if n > 1 and (self.kind == "full_attention" or pooled_linear or independent_linear):
            if self.kind == "full_attention":
                local = self.full_attention_decode_batch(packed, states, current_positions, page_tables)
            elif pooled_linear:
                local = self.pooled_linear_attention_decode_batch(packed, states)
            else:
                local = self.linear_attention_decode_batch(packed, states)
        else:
            local = []
            for i, (state, position, table) in enumerate(zip(states, current_positions, page_tables)):
                row = packed[:, :, i : i + 1, :]
                local.append(
                    self.linear_attention(row, state, decode=True, packed=row, project=False)
                    if self.kind == "linear_attention"
                    else self.full_attention(
                        row, state, page_table=table, position=position, decode=True, packed=row, project=False
                    )
                )
            local = ttnn.concat(local, dim=2) if n > 1 else local[0]
        name = "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"
        return self.decode_finish(x, self.decode_reduce(self.decode_linear(local, name)))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", required=True, choices=tuple(POLICIES))
    args, rest = parser.parse_known_args()
    assert "--layout" in rest and rest[rest.index("--layout") + 1] == "sharded"
    batch = int(rest[rest.index("--batch") + 1])
    assert 1 <= batch <= 32
    # Other stack flags can replace the selected class with historical candidates.
    incompatible = {
        "--baseline",
        "--stage4",
        "--grouped",
        "--recurrent",
        "--attention-batch",
        "--pooled",
        "--full-pool",
        "--chunk-gdn",
        "--folded-conv",
        "--precise-ccl",
        "--table-offsets",
    }
    assert not incompatible.intersection(rest), incompatible.intersection(rest)
    persistent, gather, scatter, bf8, norm_bf8 = POLICIES[args.family]
    candidate = type(
        "SelectedHiddenResidualCandidate",
        (CurrentHiddenResidualCandidate,),
        {
            "batch_size": batch,
            "persistent": persistent,
            "fused_gather_enabled": gather,
            "fused_scatter_enabled": scatter,
            "communication_dtype": ttnn.bfloat8_b if bf8 else ttnn.bfloat16,
            "norm_communication_dtype": ttnn.bfloat8_b if norm_bf8 else ttnn.bfloat16,
        },
    )
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack

    original, original_argv = stack.MultichipDecoder, sys.argv
    try:
        stack.MultichipDecoder = candidate
        sys.argv = [sys.argv[0], *rest]
        stack.main()
    finally:
        stack.MultichipDecoder, sys.argv = original, original_argv
    output = Path(rest[rest.index("--output") + 1])
    data = json.loads(output.read_text())
    data["hidden_residual_family"] = {
        "family": args.family,
        "current_production_base": True,
        "persistent": persistent,
        "residual_contract": "TP4 BF16 TILE L1 interleaved local [1,1,B,1024]; no boundary collective",
        "persistent_stats_roles": 2 if persistent else 0,
        "persistent_activation_roles": 2 if persistent else 0,
        "persistent_rs_banks": 2 if persistent and not scatter else 0,
        "persistent_fused_rs_banks": 2 if persistent and scatter else 0,
        "fused_scatter_persistent_output_supported": True,
        "bf8_row_projection_inputs": bf8 and scatter,
    }
    output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
