# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Current native residual path: packed gate/up or minimal fused SwiGLU.

All production precision and PCC gates remain unchanged. Packed projections use
BF8 weights and BF16 outputs/activation stages. Minimal SwiGLU is a separate
arithmetic candidate, not assumed bitwise equivalent to separate BF16 packs.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


class NativePackedMlp(MultichipDecoder):
    mlp_family = "packed_sharded"
    packed_readers = 8
    minimal_k = 16

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        assert self.native_local and self.residual_layout == "replicated"
        self.packed_mlp_policy = dict(
            family=cls.mlp_family,
            readers=cls.packed_readers,
            minimal_k=cls.minimal_k,
            weight_dtype=str(self.weight_dtype("mlp.gate_proj")),
            output_dtype="BF16",
        )
        if cls.mlp_family == "minimal":
            # Existing packed_mlp weights already alternate gate/up 32-column
            # tiles, exactly the minimal_matmul fuse_swiglu producer contract.
            self.decode_swiglu_program = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=cls.minimal_k,
                N_block_size=8,
                subblock_h=1,
                subblock_w=2,
                compute_with_storage_grid_size=(11, 10),
            )
        else:
            prefix = f'model.language_model.layers.{kwargs["layer_idx"]}.'
            sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict
            pieces = [
                torch.cat(
                    [
                        sd["mlp.gate_proj.weight"][rank * 3072 : (rank + 1) * 3072].T,
                        sd["mlp.up_proj.weight"][rank * 3072 : (rank + 1) * 3072].T,
                    ],
                    dim=1,
                )
                for rank in range(4)
            ]
            banks = self.device.dram_grid_size().x
            assert banks == 8, "This measured target family uses the eight-bank Blackhole layout"
            # N=6144 -> 24 tiles/bank, exactly divisible by readers 1/2/4/8.
            bank_tiles = math.ceil(6144 / (32 * banks))
            assert bank_tiles % cls.packed_readers == 0
            dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            weight_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (4096, bank_tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.native_gate_up_weight = ttnn.from_torch(
                torch.cat(pieces, dim=0).contiguous(),
                device=self.device,
                dtype=self.weight_dtype("mlp.gate_proj"),
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_mem,
                mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
            )
            # Native family: 96 x 64 columns. Each half has48 x64 columns, just
            # like the existing gate/up/down-consumer layout. Alternate control:
            # 64 x96 columns then explicit interleaved slicing and legal reshard.
            shard_width = 64 if cls.mlp_family == "packed_sharded" else 96
            output_cores = 6144 // shard_width
            grid = ttnn.num_cores_to_corerangeset(output_cores, self.device.compute_with_storage_grid_size(), True)
            self.native_gate_up_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (32, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.native_gate_up_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=2,
                per_core_M=1,
                per_core_N=shard_width // 32,
                num_workers_per_dram_bank=cls.packed_readers,
                use_multicore_pipeline=cls.packed_readers > 1,
            )
            self.packed_mlp_policy.update(
                bank_tiles=bank_tiles,
                output_cores=output_cores,
                output_shard_width=shard_width,
                extra_weight_bytes_per_chip=26738688,
            )
        print("NATIVE_PACKED_MLP_POLICY", self.packed_mlp_policy, flush=True)
        return self

    def decode_mlp(self, h):
        down_input = self.dram_layouts["mlp.down_proj"][0]
        if self.mlp_family == "minimal":
            activated = ttnn.experimental.minimal_matmul(
                ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG),
                self.weights["packed_mlp.weight"],
                fuse_swiglu=True,
                config=self.decode_swiglu_program,
                compute_kernel_config=self.mlp_compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            return ttnn.to_memory_config(activated, down_input)
        packed = ttnn.linear(
            ttnn.to_memory_config(h, self.ar_mem),
            self.native_gate_up_weight,
            program_config=self.native_gate_up_program,
            compute_kernel_config=self.mlp_compute,
            memory_config=self.native_gate_up_mem,
            dtype=ttnn.bfloat16,
        )
        n = h.shape[2]
        if self.mlp_family == "packed_sharded":
            # Native TILE slice uses TensorAccessor for source/destination shards.
            # Each result owns storage on the down-input grid; no raw tensor view
            # or assumed contiguous shard address crosses the gate/up boundary.
            gate = ttnn.slice(packed, (0, 0, 0, 0), (1, 1, n, 3072), (1, 1, 1, 1), memory_config=down_input)
            up = ttnn.slice(packed, (0, 0, 0, 3072), (1, 1, n, 6144), (1, 1, 1, 1), memory_config=down_input)
        else:
            packed = ttnn.to_memory_config(packed, ttnn.L1_MEMORY_CONFIG)
            gate = ttnn.to_memory_config(packed[:, :, :, :3072], down_input)
            up = ttnn.to_memory_config(packed[:, :, :, 3072:], down_input)
        return ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=down_input)

    def decode_finish(self, x, mixed):
        residual = ttnn.add(x, mixed)
        h = self.decode_mlp(self.decode_norm(residual, "post_attention_layernorm"))
        return ttnn.add(residual, self.decode_reduce(self.decode_linear(h, "mlp.down_proj")))

    def finish(self, x, mixed):
        if x.shape[2] == 1 and self.native_local:
            return self.decode_finish(ttnn.to_memory_config(x, self.ar_mem), mixed)
        return super().finish(x, mixed)


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mlp-family", choices=("packed_sharded", "packed_interleaved", "minimal"), required=True)
    parser.add_argument("--packed-readers", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--minimal-k", type=int, choices=(8, 16), default=16)
    args, remaining = parser.parse_known_args()
    NativePackedMlp.mlp_family = args.mlp_family
    NativePackedMlp.packed_readers = args.packed_readers
    NativePackedMlp.minimal_k = args.minimal_k
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as harness

    original, original_argv = harness.MultichipDecoder, sys.argv[:]
    try:
        harness.MultichipDecoder = NativePackedMlp
        sys.argv = [sys.argv[0], *remaining]
        harness.main()
        output = Path(remaining[remaining.index("--output") + 1])
        data = json.loads(output.read_text())
        data["native_packed_mlp"] = vars(args)
        output.write_text(json.dumps(data, indent=2) + "\n")
    finally:
        harness.MultichipDecoder = original
        sys.argv[:] = original_argv


if __name__ == "__main__":
    main()
