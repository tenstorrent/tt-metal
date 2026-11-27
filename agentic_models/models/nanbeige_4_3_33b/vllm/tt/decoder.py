"""TP4 dense decoder. Host conversion is confined to construction."""

import math

import ttnn
from models.common.lightweightmodule import LightweightModule

from .cache import fill_partial, mask_positions
from .precision import compute_config, dtype, load_policy


# ttnn.experimental.all_reduce_async gained residual_tensor / one_shot after the pinned tt-metal.
FUSED_RESIDUAL_ALL_REDUCE = False


def _f32(value):
    """The pinned ttnn bindings take C++ float arguments only when the Python float is exactly a float32."""
    import struct

    return struct.unpack("f", struct.pack("f", value))[0]


class Decoder(LightweightModule):
    PAGE = 32
    CHUNK = 1024

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        num_pages=8192,
        residual="replicated",
        dram_weights=False,
        policy=None,
        collective_semaphores=None,
    ):
        import torch

        self = cls()
        self.device = mesh_device
        self.policy = policy or load_policy()
        self.mode = "prefill"
        self.group_compute = {
            mode: {group: compute_config(spec) for group, spec in groups.items()}
            for mode, groups in self.policy["compute"].items()
        }
        assert residual in ("replicated", "sharded")
        self.residual = residual
        self.dram_weights = dram_weights
        self.pipeline_layouts = {}
        self.config = hf_config
        self.layer_idx = layer_idx
        self.h = hf_config.hidden_size
        self.nq = hf_config.num_attention_heads // 4
        self.nkv = hf_config.num_key_value_heads // 4
        self.d = hf_config.head_dim
        self.i = hf_config.intermediate_size // 4
        self.eps = hf_config.rms_norm_eps
        assert mesh_device.get_num_devices() == 4
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        norm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        self.decode_norm_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(norm_cores, [32, 896], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1), subblock_w=4, block_h=1, block_w=28, inplace=False
        )
        self.collective_semaphores = collective_semaphores
        if residual == "replicated" and self.h == 7168:
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))})
            self.decode_collective_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.decode_collective_scratch_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [32, 512], ttnn.ShardOrientation.ROW_MAJOR),
            )
            if self.collective_semaphores is None:
                self.collective_semaphores = self.make_collective_semaphores(mesh_device)
        prefix = f"model.layers.{layer_idx}."

        def w(name):
            group = (
                "attention"
                if name.startswith("self_attn.")
                else (
                    "mlp_gate_up"
                    if name in ("mlp.gate_proj", "mlp.up_proj")
                    else "mlp_down"
                    if name == "mlp.down_proj"
                    else None
                )
            )
            if group and self.policy.get("gptq_weights") and self.policy["weights"][group] == "bfp4":
                from pathlib import Path

                from safetensors import safe_open

                with safe_open(
                    Path(self.policy["gptq_weights"]) / (prefix + name + ".safetensors"),
                    framework="pt",
                ) as f:
                    return f.get_tensor("weight")
            return state_dict[prefix + name + ".weight"]

        def put(t, dim=None, layout=ttnn.TILE_LAYOUT, group=None):
            return ttnn.from_torch(
                t.contiguous(),
                device=mesh_device,
                dtype=dtype(self.policy["weights"][group]) if group else ttnn.bfloat16,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=(
                    ttnn.ReplicateTensorToMesh(mesh_device) if dim is None else ttnn.ShardTensorToMesh(mesh_device, dim)
                ),
            )

        pieces = [
            torch.cat(
                [w("self_attn." + n).chunk(4, 0)[r].T for n in ("q_proj", "k_proj", "v_proj")],
                -1,
            )
            for r in range(4)
        ]
        self.qkv = put(torch.cat(pieces, -1), -1, group="attention")
        self.o = put(w("self_attn.o_proj").T, 0, group="attention")
        pieces = [torch.cat([w("mlp." + n).chunk(4, 0)[r].T for n in ("gate_proj", "up_proj")], -1) for r in range(4)]
        self.gu = put(torch.cat(pieces, -1), -1, group="mlp_gate_up")
        down = w("mlp.down_proj").T
        if dram_weights:
            down = torch.cat(
                [torch.nn.functional.pad(part, (0, 0, 0, 128)) for part in down.chunk(4, 0)],
                dim=0,
            )
        self.down = put(down, 0, group="mlp_down")
        if dram_weights:
            banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
            for name in ("qkv", "o", "gu", "down"):
                weight = getattr(self, name)
                k, n = weight.shape
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        banks,
                        [k, ((n + 255) // 256) * 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                setattr(self, name, ttnn.to_memory_config(weight, mem))
        elif residual == "replicated":
            for name in ("qkv", "o", "gu", "down"):
                setattr(self, name, self.prepare_linear(getattr(self, name), name))
        self.norm1 = put(
            w("input_layernorm").reshape(1, 1, -1, 32),
            dim=2 if residual == "sharded" else None,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.norm2 = put(
            w("post_attention_layernorm").reshape(1, 1, -1, 32),
            dim=2 if residual == "sharded" else None,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.norm1_tiled = put(w("input_layernorm").reshape(1, 1, 1, -1).expand(1, 1, 32, -1))
        self.norm2_tiled = put(w("post_attention_layernorm").reshape(1, 1, 1, -1).expand(1, 1, 32, -1))
        if num_pages is not None:
            self.allocate_cache(num_pages)
        return self

    @staticmethod
    def make_collective_semaphores(mesh_device):
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]

    def prepare_linear(self, weight, role):
        """One measured DRAM-sharded copy shared by prefill and compact decode."""
        if weight.dtype != ttnn.bfloat16 and not (weight.dtype == ttnn.bfloat8_b and role in ("qkv", "gu", "head")):
            return weight
        workers, block, storage_tiles, input_cores = {
            "qkv": (3, 14, None, 8),
            "o": (2, 7, None, 8),
            "gu": (3, 7, None, 8),
            "down": (2, 7, None, 8),
            "head": (3, 7, 16, 14),
        }[role]
        k, n = weight.shape
        padded_k = (k + 255) // 256 * 256 if role == "down" else k
        bank_tiles = (n + 256 * workers - 1) // (256 * workers) * workers
        padded_n = bank_tiles * 256
        # Padding the local tensor pads every TP rank independently. Mere shard
        # storage padding is insufficient for the shared prefill reader.
        if (padded_k, padded_n) != (k, n):
            weight = ttnn.pad(weight, [(0, padded_k - k), (0, padded_n - n)], value=0.0)
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(banks, [padded_k, bank_tiles * 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        input_grid = (
            banks
            if input_cores == 8
            else ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 1))})
        )
        self.pipeline_layouts[role] = dict(
            k=padded_k,
            n=n,
            input_memory=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(input_grid, [32, padded_k // input_cores], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            program=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=storage_tiles or bank_tiles,
                fused_activation=None,
                num_workers_per_dram_bank=workers,
                use_multicore_pipeline=True,
            ),
        )
        return ttnn.to_memory_config(weight, memory)

    def allocate_cache(self, num_pages):
        self.key, self.value = [
            ttnn.zeros(
                [num_pages, self.nkv, self.PAGE, self.d],
                device=self.device,
                dtype=dtype(self.policy["kv_dtype"]),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]

    def _linear(self, x, w, *, group=None, mode=None):
        role = (
            "qkv"
            if w is self.qkv
            else "o"
            if w is self.o
            else "gu"
            if w is self.gu
            else "down"
            if w is self.down
            else "head"
        )
        group = group or (
            "attention"
            if w is self.qkv or w is self.o
            else ("mlp_gate_up" if w is self.gu else "mlp_down" if w is self.down else "lm_head")
        )
        compute = self.group_compute[mode or self.mode][group]
        if role in self.pipeline_layouts:
            layout = self.pipeline_layouts[role]
            original_shape = tuple(x.shape)
            physical_shape = tuple(x.padded_shape)
            flatten_batch = math.prod(physical_shape[:-2]) > 1
            if flatten_batch:
                # Preserve each batch row's tile padding while folding batch
                # into M; restore the logical rows after the projection.
                if original_shape != physical_shape:
                    x = ttnn.pad(x, [(0, p - n) for n, p in zip(original_shape, physical_shape)], value=0.0)
                x = ttnn.reshape(x, (1, 1, math.prod(physical_shape[:-1]), physical_shape[-1]))
            if x.shape[-1] < layout["k"]:
                x = ttnn.pad(x, [(0, 0)] * (len(x.shape) - 1) + [(0, layout["k"] - x.shape[-1])], value=0.0)
            rows = x.padded_shape[-2]
            if rows == 32:
                x = ttnn.to_memory_config(x, layout["input_memory"])
                program = layout["program"]
                output_memory = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
            else:
                pm = (rows // 32 + 7) // 8
                pn = (w.shape[-1] // 32 + 10) // 11
                program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(11, 8),
                    in0_block_w=1 if role == "head" else 4,
                    out_subblock_h=2 if pm % 2 == 0 else 1,
                    out_subblock_w=2 if pn % 2 == 0 else 1,
                    per_core_M=pm,
                    per_core_N=pn,
                    transpose_mcast=False,
                    fused_activation=None,
                )
                output_memory = ttnn.DRAM_MEMORY_CONFIG
            out = ttnn.linear(
                x,
                w,
                program_config=program,
                memory_config=output_memory,
                dtype=ttnn.float32 if role in ("o", "down") else ttnn.bfloat16,
                compute_kernel_config=compute,
            )
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            if w.shape[-1] != layout["n"]:
                out = out[..., : layout["n"]]
            if flatten_batch:
                out = ttnn.reshape(out, (*physical_shape[:-1], layout["n"]))
                out = out[tuple(slice(0, n) for n in original_shape[:-1]) + (slice(None),)]
            return out
        if self.dram_weights:
            k, n = w.shape
            if x.shape[-1] < k:
                x = ttnn.pad(x, [(0, 0)] * (len(x.shape) - 1) + [(0, k - x.shape[-1])], value=0.0)
            rows = x.padded_shape[-2]
            if rows == 32:
                cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
                imem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, [32, k // 8], ttnn.ShardOrientation.ROW_MAJOR),
                )
                omem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        cores,
                        [32, ((n + 255) // 256) * 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                block = 1 if n > 10000 else {7168: 14, 1792: 7, 5632: 11}[k]
                config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=(n + 255) // 256,
                    fused_activation=None,
                    num_workers_per_dram_bank=2 if ((n + 255) // 256) % 2 == 0 else 1,
                )
                out = ttnn.linear(
                    ttnn.to_memory_config(x, imem),
                    w,
                    program_config=config,
                    memory_config=omem,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=compute,
                )
                return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            pm = (rows // 32 + 7) // 8
            pn = (n // 32 + 10) // 11
            sw = 2 if pn % 2 == 0 else 1
            sh = 2 if pm % 2 == 0 else 1
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(11, 8),
                in0_block_w=4,
                out_subblock_h=sh,
                out_subblock_w=sw,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
            )
            return ttnn.linear(
                x,
                w,
                program_config=config,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute,
            )
        return ttnn.linear(
            x,
            w,
            dtype=ttnn.float32 if w is self.o or w is self.down else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute,
        )

    def _reduce(self, x):
        if len(x.shape) == 3:
            x = ttnn.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])
        if self.residual == "sharded":
            y = ttnn.reduce_scatter(x, dim=3, num_links=1, topology=ttnn.Topology.Linear, cluster_axis=1)
        else:
            y = ttnn.all_reduce(x, topology=ttnn.Topology.Linear, cluster_axis=1)
        return y

    def _norm(self, x, w):
        if x.dtype == ttnn.float32:
            if (
                self.residual == "replicated"
                and self.h == 7168
                and x.padded_shape[0] * x.padded_shape[1] * x.padded_shape[2] == 32
                and (w is self.norm1 or w is self.norm2)
            ):
                # The BH sharded SFPU kernel retains FP32 residual/statistic math.
                # All compact decode buckets and one-tile prefill share this path.
                y = ttnn.rms_norm(
                    ttnn.to_memory_config(x, self.decode_norm_memory),
                    weight=w,
                    epsilon=self.eps,
                    program_config=self.decode_norm_program,
                    compute_kernel_config=self.compute,
                )
                return ttnn.typecast(ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG), ttnn.bfloat16)
            w = self.norm1_tiled if w is self.norm1 else self.norm2_tiled if w is self.norm2 else w
            y = ttnn.rms_norm(
                x,
                weight=w,
                epsilon=self.eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute,
            )
            return ttnn.typecast(y, ttnn.bfloat16)
        if self.residual == "sharded":
            stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute)
            stats = ttnn.all_gather(stats, dim=3, num_links=1, topology=ttnn.Topology.Linear, cluster_axis=1)
            norm = ttnn.rms_norm_post_all_gather(
                x, stats, weight=w, epsilon=self.eps, compute_kernel_config=self.compute
            )
            return ttnn.all_gather(norm, dim=3, num_links=1, topology=ttnn.Topology.Linear, cluster_axis=1)
        rows = x.padded_shape[0] * x.padded_shape[1] * x.padded_shape[2]
        gy = min(8, rows // 32)
        sh = ((rows + gy * 32 - 1) // (gy * 32)) * 32
        gy = (rows + sh - 1) // sh
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, gy - 1))})
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [sh, self.h // 8], ttnn.ShardOrientation.ROW_MAJOR),
        )
        y = ttnn.rms_norm(
            ttnn.to_memory_config(x, mem),
            weight=w,
            epsilon=self.eps,
            program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, gy),
                subblock_w=4,
                block_h=sh // 32,
                block_w=self.h // 8 // 32,
                inplace=True,
            ),
            compute_kernel_config=self.compute,
        )
        y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.typecast(y, ttnn.bfloat16) if y.dtype == ttnn.float32 else y

    def _add_reduced(self, residual, partial, index):
        # The fused one-shot all_reduce_async(residual_tensor=...) the bring-up used in decode is not in the
        # pinned tt-metal: every layer adds the composite all_reduce result (FP32 operands on both paths).
        if FUSED_RESIDUAL_ALL_REDUCE is False or not (
            self.mode == "decode"
            and self.residual == "replicated"
            and self.h == 7168
            and residual.dtype == partial.dtype == ttnn.float32
            and math.prod(list(residual.padded_shape)[:-1]) == 32
        ):
            return ttnn.add(residual, self._reduce(partial))
        if len(partial.shape) == 3:
            partial = ttnn.reshape(partial, [partial.shape[0], 1, partial.shape[1], partial.shape[2]])
        partial = ttnn.to_memory_config(partial, self.decode_collective_memory)
        residual = ttnn.to_memory_config(residual, self.decode_collective_memory)
        # One-shot completion consumes every owner's scratch before the next
        # operation. Keep scratch local so matmuls can reuse this L1 allocation;
        # trace capture records the same ordered lifetime. Semaphores persist.
        scratch = ttnn.empty(
            [1, 1, 32, 28672],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.decode_collective_scratch_memory,
        )
        return ttnn.experimental.all_reduce_async(
            partial,
            scratch,
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.collective_semaphores[index],
            dtype=ttnn.float32,
            memory_config=self.decode_collective_memory,
            topology=ttnn.Topology.Linear,
            num_links=2,
            fp32_dest_acc=True,
            one_shot=True,
            residual_tensor=residual,
        )

    def _finish(self, x, attention):
        x = self._add_reduced(x, self._linear(attention, self.o), 0)
        gu = self._linear(self._norm(x, self.norm2), self.gu)
        gate, up = gu[..., : self.i], gu[..., self.i :]
        mlp = ttnn.mul(ttnn.silu(gate), up)
        return self._add_reduced(x, self._linear(mlp, self.down), 1)

    def prefill_forward(self, x, *, start_pos, valid_len, page_table, cos, sin, fill_metadata=None):
        """Any logical length, with bounded native chunks and exact prefix preservation.

        RoPE covers [floor(start/32)*32, ceil((start+valid_len)/32)*32).
        Partial fills take device metadata [aligned_start, start%32, valid_len].
        Host chunk planning is outside decode traces; all tensor work stays on device.
        """
        self.mode = "prefill"
        if isinstance(start_pos, (list, tuple)):
            return self._prefill_rows(x, start_pos, valid_len, page_table, cos, sin, fill_metadata)
        skip = start_pos % 32
        assert 0 <= start_pos and start_pos + valid_len <= page_table.shape[1] * self.PAGE
        assert 0 < valid_len <= x.shape[2]
        aligned_start = start_pos - skip
        physical = ((skip + valid_len + 31) // 32) * 32
        if skip:
            x = ttnn.to_layout(
                ttnn.pad(
                    ttnn.to_layout(x[:, :, :valid_len, :], ttnn.ROW_MAJOR_LAYOUT),
                    [(0, 0), (0, 0), (skip, physical - skip - valid_len), (0, 0)],
                    value=0.0,
                ),
                ttnn.TILE_LAYOUT,
            )
        elif x.shape[2] != physical:
            x = x[:, :, :valid_len, :]
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - valid_len), (0, 0)], value=0.0)
        if (skip or valid_len != physical) and fill_metadata is None:
            raise ValueError("Partial fill requires device [aligned_start, skip, valid_len] metadata")
        outputs = []
        for offset in range(0, physical, self.CHUNK):
            count = min(self.CHUNK, physical - offset)
            part = x[:, :, offset : offset + count, :]
            partial = offset < skip or offset + count > skip + valid_len
            outputs.append(
                self._prefill_chunk(
                    part,
                    start=aligned_start + offset,
                    page_table=page_table,
                    cos=cos[:, :, offset : offset + count, :],
                    sin=sin[:, :, offset : offset + count, :],
                    metadata=fill_metadata if partial else None,
                    chunk_offset=offset,
                )
            )
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        return output[:, :, skip : skip + valid_len, :]

    def _prefill_rows(self, x, starts, lengths, page_table, cos, sin, metadata):
        """One dense projection per chunk across rows with independent paged attention."""
        assert len(starts) == len(lengths) == x.shape[0] and metadata is not None
        assert all(
            0 <= s and 0 < n <= x.shape[2] and s + n <= page_table.shape[1] * self.PAGE for s, n in zip(starts, lengths)
        )
        skips = [s % 32 for s in starts]
        physical = ((max(n + k for n, k in zip(lengths, skips)) + 31) // 32) * 32
        rows = []
        for row, (n, k) in enumerate(zip(lengths, skips)):
            part = ttnn.to_layout(x[row : row + 1, :, :n, :], ttnn.ROW_MAJOR_LAYOUT)
            rows.append(
                ttnn.to_layout(
                    ttnn.pad(part, [(0, 0), (0, 0), (k, physical - k - n), (0, 0)], value=0.0),
                    ttnn.TILE_LAYOUT,
                )
            )
        x = ttnn.concat(rows, dim=0)
        pieces = []
        per_row_chunk = max(32, (self.CHUNK // len(starts) // 32) * 32)
        for offset in range(0, physical, per_row_chunk):
            count = min(per_row_chunk, physical - offset)
            inp = x[:, :, offset : offset + count, :]
            packed = self._linear(self._norm(inp, self.norm1), self.qkv)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                packed,
                num_heads=self.nq,
                num_kv_heads=self.nkv,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            qs = []
            ks = []
            for row in range(len(starts)):
                cs = cos[row][:, :, offset : offset + count, :]
                sn = sin[row][:, :, offset : offset + count, :]
                qs.append(
                    ttnn.experimental.rotary_embedding_hf(q[row : row + 1], cs, sn, compute_kernel_config=self.compute)
                )
                ks.append(
                    ttnn.experimental.rotary_embedding_hf(k[row : row + 1], cs, sn, compute_kernel_config=self.compute)
                )
            k = ttnn.concat(ks, dim=0)
            fill_partial(self.key, k, page_table, metadata, offset)
            fill_partial(self.value, v, page_table, metadata, offset)
            attention = []
            for row, start in enumerate(starts):
                row_start = start - skips[row] + offset
                available = max(0, min(count, page_table.shape[1] * self.PAGE - row_start))
                if available == 0:
                    attention.append(ttnn.zeros_like(inp[row : row + 1, :, :, : self.nq * self.d]))
                    continue
                chunk = 128 if row_start % 128 == 0 and available % 128 == 0 else 32
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    qs[row][:, :, :available, :],
                    self.key,
                    self.value,
                    page_table[row : row + 1],
                    chunk_start_idx=row_start,
                    scale=_f32(self.d**-0.5),
                    program_config=ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=(8, 8),
                        q_chunk_size=chunk,
                        k_chunk_size=chunk,
                        exp_approx_mode=False,
                    ),
                    compute_kernel_config=self.compute,
                )
                joined = ttnn.transformer.concatenate_heads(attn)
                joined = ttnn.reshape(joined, [1, 1, available, self.nq * self.d])
                if available < count:
                    joined = ttnn.pad(
                        joined,
                        [(0, 0), (0, 0), (0, count - available), (0, 0)],
                        value=0.0,
                    )
                attention.append(joined)
            pieces.append(self._finish(inp, ttnn.concat(attention, dim=0)))
        output = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2)
        cropped = []
        last = ((max(lengths) + 31) // 32) * 32
        for row, (n, k) in enumerate(zip(lengths, skips)):
            part = output[row : row + 1, :, k : k + n, :]
            cropped.append(ttnn.pad(part, [(0, 0), (0, 0), (0, last - n), (0, 0)], value=0.0))
        return ttnn.concat(cropped, dim=0)[:, :, : max(lengths), :]

    def prefill_prepared(self, x, *, chunk_start, page_table, cos, sin, fill_metadata, sdpa_chunks=(32, 32)):
        """Traceable physical bucket; planning/padding and last-token selection are external.

        x has <=1024 tile-aligned rows, including front/tail padding. Refresh x,
        RoPE, page_table, INT32 chunk_start [1], and fill_metadata [start,skip,n]
        in place before replay. The physical extent must fit the page-table row.
        Returns the physical bucket; only [skip:skip+n] is logically valid.
        """
        assert x.shape[0] == 1 and x.shape[2] % 32 == 0 and x.shape[2] <= self.CHUNK
        return self._prefill_chunk(
            x,
            start=0,
            page_table=page_table,
            cos=cos,
            sin=sin,
            metadata=fill_metadata,
            start_tensor=chunk_start,
            sdpa_chunks=sdpa_chunks,
        )

    def _prefill_chunk(
        self,
        x,
        *,
        start,
        page_table,
        cos,
        sin,
        metadata=None,
        chunk_offset=0,
        start_tensor=None,
        sdpa_chunks=None,
    ):
        self.mode = "prefill"
        qkv = self._linear(self._norm(x, self.norm1), self.qkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.nq,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q, k = [ttnn.experimental.rotary_embedding_hf(t, cos, sin, compute_kernel_config=self.compute) for t in (q, k)]
        if metadata is not None:
            fill_partial(self.key, k, page_table, metadata, chunk_offset)
            fill_partial(self.value, v, page_table, metadata, chunk_offset)
        else:
            fill_table = page_table[:, start // 32 : (start + x.shape[2]) // 32]
            ttnn.experimental.paged_fill_cache(self.key, ttnn.typecast(k, self.key.dtype), fill_table)
            ttnn.experimental.paged_fill_cache(self.value, ttnn.typecast(v, self.value.dtype), fill_table)
        chunk = 128 if start_tensor is None and start % 128 == 0 and x.shape[2] % 128 == 0 else 32
        q_chunk, k_chunk = (chunk, chunk) if sdpa_chunks is None else sdpa_chunks
        # The odd seven-chunk Q extent assigns one task to each of 98 cores.
        # Other shapes retain their measured grid, including unaligned resumes.
        prefill_grid = (11, 10) if q_chunk == 128 and q.shape[2] == 896 else (8, 8)
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.key,
            self.value,
            page_table,
            **({"chunk_start_idx": start} if start_tensor is None else {"chunk_start_idx_tensor": start_tensor}),
            scale=_f32(self.d**-0.5),
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=prefill_grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute,
        )
        return self._finish(x, ttnn.transformer.concatenate_heads(attn))

    def prefill_batch_prepared(
        self, x, *, batch, chunk_start, page_table, cos, sin, fill_metadata, sdpa_chunks=(32, 32)
    ):
        """Native multi-row prefill: one dense tensor and one batched paged SDPA."""
        total = x.shape[2]
        width = total // batch
        self.mode = "prefill"
        qkv = self._linear(self._norm(x, self.norm1), self.qkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.nq,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q, k = [ttnn.experimental.rotary_embedding_hf(t, cos, sin, compute_kernel_config=self.compute) for t in (q, k)]

        def rows(t, heads):
            return ttnn.permute(ttnn.reshape(t, [heads, batch, width, self.d]), (1, 0, 2, 3))

        q, k, v = rows(q, self.nq), rows(k, self.nkv), rows(v, self.nkv)
        fill_partial(self.key, k, page_table, fill_metadata)
        fill_partial(self.value, v, page_table, fill_metadata)
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.key,
            self.value,
            page_table,
            chunk_start_idx_tensor=chunk_start,
            scale=_f32(self.d**-0.5),
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=sdpa_chunks[0],
                k_chunk_size=sdpa_chunks[1],
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute,
        )
        joined = ttnn.reshape(ttnn.permute(attn, (1, 0, 2, 3)), [1, self.nq, total, self.d])
        return self._finish(x, ttnn.transformer.concatenate_heads(joined))

    def decode_forward(self, x, *, cur_pos, page_table, active_rows, cos, sin):
        """One batched tensor; inactive cur_pos=-1 skips cache writes in-kernel."""
        self.mode = "decode"
        if active_rows is not None:
            cur_pos = mask_positions(cur_pos, active_rows)
        b = x.shape[2]
        cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(min(b, 8) - 1, (b + 7) // 8 - 1),
                )
            }
        )
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [32, self.d], ttnn.ShardOrientation.ROW_MAJOR),
        )
        qkv = ttnn.to_memory_config(self._linear(self._norm(x, self.norm1), self.qkv), ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=self.nq, num_kv_heads=self.nkv, memory_config=mem
        )
        cos, sin = [ttnn.to_memory_config(t, mem) for t in (cos, sin)]
        q, k = [
            ttnn.experimental.rotary_embedding_hf(t, cos, sin, is_decode_mode=True, compute_kernel_config=self.compute)
            for t in (q, k)
        ]
        ttnn.experimental.paged_update_cache(self.key, k, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(self.value, v, update_idxs_tensor=cur_pos, page_table=page_table)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.key,
            self.value,
            cur_pos_tensor=cur_pos,
            page_table_tensor=page_table,
            scale=_f32(self.d**-0.5),
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=32,
                k_chunk_size=256,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute,
        )
        joined = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(attn, mem), num_heads=self.nq)
        return self._finish(x, ttnn.to_memory_config(joined, ttnn.DRAM_MEMORY_CONFIG)[:, :, :b, :])
