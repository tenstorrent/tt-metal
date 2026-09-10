# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Llama 3.1 8B decoder on a four-device Blackhole ring.

Prefill carries [B, 1, S, 1024] per device; decode carries [1, 1, B, 1024].
Each device owns a contiguous hidden quarter, eight query heads and two KV
heads. Page tables and positions are replicated; KV storage is local paged
[B_pages, 2, 128, 128]. Folded RMSNorm affine weights preserve the selected
FP32-multiply, BF16-rounding order before block-float quantization.

The caller owns page allocation and must populate all prefix positions before
decode. Decode buffers are shared by sequential layers and must outlive every
trace that references them. Prefill starts at position zero and chunks logical
prompts internally without reducing the 131072-token context contract.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule

PROJECTION_GEOMETRY = {
    "qkv": {"cores": 8, "block": 16, "readers": 1},
    "o": {"cores": 8, "block": 4, "readers": 1},
    "gate_up": {"cores": 8, "block": 8, "readers": 2},
    "down": {"cores": 16, "block": 7, "readers": 1},
}


def validate_qb2_mesh(mesh_device):
    """Reject unqualified hardware before checkpoint conversion or device allocation."""
    arch = mesh_device.arch()
    cluster_type = ttnn.cluster.get_cluster_type()
    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)
    if (
        arch != ttnn.device.Arch.BLACKHOLE
        or cluster_type != ttnn.cluster.ClusterType.P300_X2
        or num_devices != 4
        or mesh_shape != (1, 4)
    ):
        raise ValueError(
            "Requires a Blackhole P300_X2 QB2 with four devices in a (1, 4) mesh; "
            f"got arch={arch}, cluster_type={cluster_type}, num_devices={num_devices}, mesh_shape={mesh_shape}"
        )


@dataclass(frozen=True)
class DecodeWorkspace:
    batch: int
    mesh: ttnn.MeshDevice
    dtype: ttnn.DataType
    buffers: dict[str, ttnn.Tensor]


class LlamaDecoder(LightweightModule):
    page_size = 128
    prefill_chunk_size = 1024
    supported_context = 131072

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, precision_policy, ccl, rope_state=None):
        """Load one layer with the selected precision on a 1×4 ring.

        Before mesh open configure FABRIC_1D_RING with 8192-byte router packets
        and reserve 16 KiB L1_SMALL for native collective semaphores. Call
        prepare_decode before warmup; share its workspace between stacked layers.
        """
        validate_qb2_mesh(mesh_device)

        import torch
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        dims = (
            hf_config.hidden_size,
            hf_config.intermediate_size,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            hf_config.max_position_embeddings,
        )
        if dims != (4096, 14336, 32, 8, 128, 131072):
            raise ValueError(f"Requires exact Llama-3.1-8B dimensions: {dims}")
        if (
            hf_config.attention_bias
            or hf_config.mlp_bias
            or hf_config.hidden_act != "silu"
            or hf_config.rope_parameters["rope_type"] != "llama3"
        ):
            raise ValueError("Unsupported architecture")
        if not 0 <= layer_idx < hf_config.num_hidden_layers:
            raise ValueError("Invalid layer index")
        self = cls()
        self.mesh_device = mesh_device
        p = precision_policy
        self.eps = hf_config.rms_norm_eps
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid.x - 1, self.grid.y - 1))}
        )
        self.ccl = ccl
        self.decode_workspace = None
        self.hidden = 1024
        self.residual_memcfg = self._width_memcfg(4096, 8)
        self.local_residual_memcfg = self._width_memcfg(self.hidden, 8)
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=self.grid,
            subblock_w=4,
            block_h=1,
            block_w=16,
            inplace=False,
        )
        self.decode_sdpa_program = None
        self.prefill_program = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=8,
            N_block_size=16,
            subblock_h=2,
            subblock_w=4,
            compute_with_storage_grid_size=self.grid,
        )
        self.prefill_mlp_program = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=8,
            N_block_size=24,
            subblock_h=2,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
        )
        self.prefill_weights = {}
        prefix = f"model.layers.{layer_idx}."

        def w(name):
            return state_dict[prefix + name + ".weight"].detach()

        def folded(name, norm):
            return (w(name).float() * w(norm).float()[None, :]).bfloat16()

        q, k, v = [folded("self_attn." + name, "input_layernorm").T for name in ("q_proj", "k_proj", "v_proj")]
        gate = folded("mlp.gate_proj", "post_attention_layernorm").T
        up = folded("mlp.up_proj", "post_attention_layernorm").T
        tensors = {
            "qkv": [
                torch.cat(
                    [q[:, i * 1024 : (i + 1) * 1024], k[:, i * 256 : (i + 1) * 256], v[:, i * 256 : (i + 1) * 256]],
                    dim=1,
                )
                for i in range(4)
            ],
            "o": list(w("self_attn.o_proj").T.chunk(4, dim=0)),
            "gate": list(gate.chunk(4, dim=1)),
            "up": list(up.chunk(4, dim=1)),
            "down": list(w("mlp.down_proj").T.chunk(4, dim=0)),
        }

        def convert(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def shards(ts, dtype, mem=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                torch.cat(ts, dim=0).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mem,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )

        for role, local in tensors.items():
            group = "gate_up" if role in ("gate", "up") else role
            self.prefill_weights[role] = shards(local, getattr(ttnn, p["weight_groups"][group]))

        decode_tensors = {role: tensors[role] for role in ("qkv", "o", "down")}
        decode_tensors["gate_up"] = [torch.cat([a, b], dim=1) for a, b in zip(tensors["gate"], tensors["up"])]
        self.decode_weights, self.decode_programs, self.decode_computes, self.decode_inputs = {}, {}, {}, {}
        self.decode_logical_widths = {}
        banks = mesh_device.dram_grid_size().x
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        for role, local in decode_tensors.items():
            opts = PROJECTION_GEOMETRY[role]
            kdim, ndim = local[0].shape
            alignment = 32 * banks * opts["readers"]
            padded_n = ((ndim + alignment - 1) // alignment) * alignment
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [kdim, padded_n // banks], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.decode_weights[role] = shards(
                [torch.nn.functional.pad(t, (0, padded_n - ndim)) for t in local],
                getattr(ttnn, p["weight_groups"][role]),
                mem,
            )
            self.decode_logical_widths[role] = ndim
            self.decode_inputs[role] = self._width_memcfg(kdim, opts["cores"])
            self.decode_programs[role] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=opts["block"],
                per_core_M=1,
                per_core_N=(padded_n + 32 * opts["cores"] - 1) // (32 * opts["cores"]),
                num_workers_per_dram_bank=opts["readers"],
            )
        paired = [
            torch.stack([a.reshape(4096, 112, 32), b.reshape(4096, 112, 32)], dim=2).reshape(4096, 7168)
            for a, b in zip(tensors["gate"], tensors["up"])
        ]
        # Prefill SwiGLU interleaves gate/up tiles; decode concatenates halves.
        self.prefill_weights["gate_up"] = shards(paired, getattr(ttnn, p["weight_groups"]["gate_up"]))
        if rope_state is not None:
            for name in ("cos_chunks", "sin_chunks", "cos_embedding", "sin_embedding", "chunk_starts"):
                setattr(self, name, getattr(rope_state, name))
            self._configure_precision(precision_policy)
            return self
        rope = LlamaRotaryEmbedding(hf_config)
        cos, sin = rope(torch.empty(1, dtype=torch.bfloat16), torch.arange(131072)[None])
        self.cos_chunks = tuple(convert(cos[:, s : s + 1024].reshape(1, 1, 1024, 128)) for s in range(0, 131072, 1024))
        self.sin_chunks = tuple(convert(sin[:, s : s + 1024].reshape(1, 1, 1024, 128)) for s in range(0, 131072, 1024))
        self.cos_embedding = convert(cos.reshape(131072, 128), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sin_embedding = convert(sin.reshape(131072, 128), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.chunk_starts = tuple(
            convert(torch.tensor([s], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for s in range(0, 131072, 1024)
        )
        self._configure_precision(precision_policy)
        return self

    def _configure_precision(self, precision_policy):
        p = precision_policy
        self.precision_policy = p
        self.ccl_dtype = getattr(ttnn, p["ccl_dtype"])
        self.cache_dtype = getattr(ttnn, p["kv_cache_dtype"])
        self.activation_dtype = getattr(ttnn, p["activation_dtype"])
        self.activation_output_dtype = getattr(ttnn, p["activation_output_dtype"])
        self.residual_dtype = getattr(ttnn, p["residual_dtype"])

        def compute(fidelity, fp32, packer=True, **extra):
            return ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=p["accumulation"]["math_approx_mode"],
                fp32_dest_acc_en=fp32,
                packer_l1_acc=packer,
                **extra,
            )

        self.compute = compute(p["compute_fidelities"]["norm_rope"], p["accumulation"]["norm_rope_fp32"])
        self.sdpa_compute = compute(p["compute_fidelities"]["sdpa"], p["accumulation"]["sdpa_fp32"], False)
        self.decode_sdpa_compute = compute(
            p["compute_fidelities"]["sdpa"], p["accumulation"]["sdpa_fp32"], False, dst_full_sync_en=True
        )
        self.prefill_computes = {
            g: compute(f, p["accumulation"]["matmul_fp32"]) for g, f in p["compute_fidelities"]["prefill"].items()
        }
        for role in self.decode_weights:
            self.decode_computes[role] = compute(
                p["compute_fidelities"]["decode"][role], p["accumulation"]["matmul_fp32"]
            )

    def allocate_cache(self, *, num_physical_pages):
        if not isinstance(num_physical_pages, int) or num_physical_pages < 1:
            raise ValueError("Cache requires a positive physical page count")
        return tuple(
            ttnn.zeros(
                (num_physical_pages, 2, 128, 128),
                dtype=self.cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _validate_cache(self, page_table, kv_cache):
        if len(page_table.shape) != 2 or page_table.shape[1] < 1 or page_table.dtype != ttnn.int32:
            raise ValueError("Page table must be nonempty replicated int32 [batch,pages]")
        if len(kv_cache) != 2 or kv_cache[0].shape != kv_cache[1].shape:
            raise ValueError("K and V require matching physical page shapes")
        for item in kv_cache:
            if (
                tuple(item.shape)[1:] != (2, 128, 128)
                or item.dtype != self.cache_dtype
                or item.layout != ttnn.TILE_LAYOUT
            ):
                raise ValueError("Each local cache must be BFP8 TILE [pages,2,128,128]")

    def prepare_decode(self, batch, *, workspace=None):
        """Bind one fixed-batch workspace shared by sequential decoder layers.

        Buffers must remain alive until their traces are released. Concurrent
        model instances require separate workspaces; rebinding is forbidden.
        """
        if not 1 <= batch <= 32:
            raise ValueError("Decode batch must be 1..32")
        if self.decode_workspace is not None:
            if self.decode_workspace.batch != batch or (
                workspace is not None and workspace is not self.decode_workspace
            ):
                raise ValueError("Release traces before rebinding a decoder workspace")
            return self.decode_workspace
        # The SDPA query-core prefix must match head splitting, not just contain it.
        grid = [8, 4] if batch <= 8 else [self.grid.x, self.grid.y]
        self.decode_sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=256, exp_approx_mode=False
        )
        if workspace is not None:
            if workspace.batch != batch or workspace.mesh is not self.mesh_device or workspace.dtype != self.ccl_dtype:
                raise ValueError("Shared workspace must match batch, mesh and collective dtype")
            self.decode_workspace = workspace
            return workspace
        buffers = {}
        for site in ("attn", "mlp", "o", "down"):
            width = 4096 if site in ("attn", "mlp") else 1024
            buffers[site] = ttnn.empty(
                (1, 1, batch, width),
                dtype=self.ccl_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=self._width_memcfg(width, 8),
            )
        self.decode_workspace = DecodeWorkspace(batch, self.mesh_device, self.ccl_dtype, buffers)
        return self.decode_workspace

    def _width_memcfg(self, width, cores):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(cores, self.grid, True),
                [32, ((width + cores * 32 - 1) // (cores * 32)) * 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _decode_norm(self, x):
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            memory_config=self.residual_memcfg,
            program_config=self.norm_program,
            compute_kernel_config=self.compute,
        )

    def _decode_rope(self, x, cos, sin):
        return ttnn.experimental.rotary_embedding_hf(
            x,
            cos,
            sin,
            is_decode_mode=True,
            compute_kernel_config=self.compute,
        )

    def _update_cache(self, k, v, current_pos, page_table, kv_cache):
        batch = v.shape[1]
        if batch <= 4:
            # Paired cache update requires disjoint input grids. One V reshard
            # plus one paired update beats two updates only for the small batches.
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(batch - 1, 5))})
            memcfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            v = ttnn.to_memory_config(v, memcfg)
            ttnn.experimental.paged_fused_update_cache(
                kv_cache[0],
                k,
                kv_cache[1],
                v,
                update_idxs_tensor=current_pos,
                page_table=page_table,
            )
        else:
            ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table)

    def _rope_tables(self, current_pos, batch, memcfg):
        # Logical/padded-shape views remove copies of unused tile rows. HF RoPE
        # broadcasts only row zero of each cos/sin tile; other rows are unused.
        direct = batch <= 8
        indices = ttnn.typecast(ttnn.reshape(current_pos, (batch, 1) if direct else (1, batch)), ttnn.uint32)
        padding = 31 if direct else 32 - batch
        if padding:
            indices = ttnn.pad(indices, ((0, 0), (0, padding)), value=0)
        outputs = []
        for table in (self.cos_embedding, self.sin_embedding):
            embedded = ttnn.embedding(
                indices, table, layout=ttnn.TILE_LAYOUT, memory_config=memcfg if direct else ttnn.DRAM_MEMORY_CONFIG
            )
            if direct:
                selected = ttnn.reshape(
                    embedded, ttnn.Shape((1, batch, 1, 128)), ttnn.Shape((1, batch, 32, 128)), skip_padding_fill=True
                )
            else:
                selected = ttnn.reshape(
                    embedded, ttnn.Shape((1, batch, 128)), ttnn.Shape((1, 32, 128)), skip_padding_fill=True
                )
                selected = ttnn.reshape(selected, (1, batch, 1, 128), skip_padding_fill=True)
                selected = ttnn.to_memory_config(selected, memcfg)
            outputs.append(selected)
        return outputs

    def _linear(self, x, role):
        group = "gate_up" if role in ("gate", "up") else role
        x = ttnn.typecast(x, self.activation_dtype) if x.dtype != self.activation_dtype else x
        return ttnn.experimental.minimal_matmul(
            x,
            self.prefill_weights[role],
            dtype=self.activation_output_dtype,
            config=self.prefill_program,
            compute_kernel_config=self.prefill_computes[group],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _decode_linear(self, x, role):
        x = ttnn.typecast(x, self.activation_dtype) if x.dtype != self.activation_dtype else x
        output = ttnn.linear(
            ttnn.to_memory_config(x, self.decode_inputs[role]),
            self.decode_weights[role],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.decode_programs[role],
            compute_kernel_config=self.decode_computes[role],
        )
        # Reader padding is physical storage, not additional model features.
        width = self.decode_logical_widths[role]
        if output.shape[-1] != width:
            output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)[:, :, :, :width]
        return output

    def _collective_input(self, x):
        return ttnn.typecast(x, self.ccl_dtype) if x.dtype != self.ccl_dtype else x

    def _ag(self, x, *, decode, site):
        memory = self.residual_memcfg if decode else ttnn.DRAM_MEMORY_CONFIG
        if not decode:
            x = ttnn.to_memory_config(x, memory)
        output = ttnn.experimental.all_gather_async(
            self._collective_input(x),
            dim=3,
            persistent_output_buffer=self.decode_workspace.buffers[site] if decode else None,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=memory,
            chunks_per_sync=1 if decode else 10,
            num_workers_per_link=1 if decode else 2,
            num_buffers_per_channel=2,
        )
        return ttnn.typecast(output, ttnn.bfloat16) if output.dtype != ttnn.bfloat16 else output

    def _rs(self, x, *, decode, site):
        memory = self.local_residual_memcfg if decode else ttnn.DRAM_MEMORY_CONFIG
        if not decode:
            x = ttnn.to_memory_config(x, memory)
        output = ttnn.reduce_scatter(
            self._collective_input(x),
            dim=3,
            memory_config=memory,
            output_tensor=self.decode_workspace.buffers[site] if decode else None,
            num_links=2,
            topology=ttnn.Topology.Ring,
        )
        return ttnn.typecast(output, ttnn.bfloat16) if output.dtype != ttnn.bfloat16 else output

    def _output_projection(self, x, role):
        return self._rs(self._decode_linear(x, role), decode=True, site=role)

    def _norm_input(self, x, *, decode, site):
        gathered = self._ag(x, decode=decode, site=site)
        if decode:
            return self._decode_norm(ttnn.to_memory_config(gathered, self.residual_memcfg))
        return self._prefill_norm(gathered)

    def _prefill_norm(self, x):
        rows = x.padded_shape[2]
        # At 224 rows the eight-core norm CBs overlap live L1 allocations.
        # Preserve logical lengths and the original larger-prefill path.
        if rows > 192:
            return ttnn.rms_norm(
                x, epsilon=self.eps, compute_kernel_config=self.compute, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]),
                [rows, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
            subblock_w=4,
            block_h=rows // 32,
            block_w=16,
            inplace=False,
        )
        value = ttnn.rms_norm(
            ttnn.to_memory_config(x, memory),
            epsilon=self.eps,
            program_config=config,
            memory_config=memory,
            compute_kernel_config=self.compute,
        )
        return ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)

    def _finish(self, x, attention):
        projected = self._rs(self._linear(attention, "o"), decode=False, site="o")
        residual = ttnn.add(x, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        normalized = self._norm_input(residual, decode=False, site="mlp")
        rows = x.padded_shape[2]
        if rows <= 256 or x.shape[2] >= 512:
            normalized = (
                ttnn.typecast(normalized, self.activation_dtype)
                if normalized.dtype != self.activation_dtype
                else normalized
            )
            config = (
                ttnn.MinimalMatmulConfig(
                    M_block_size=1,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
                )
                if rows <= 256
                else self.prefill_mlp_program
            )
            product = ttnn.experimental.minimal_matmul(
                normalized,
                self.prefill_weights["gate_up"],
                dtype=self.activation_output_dtype,
                fuse_swiglu=True,
                config=config,
                compute_kernel_config=self.prefill_computes["gate_up"],
                memory_config=ttnn.L1_MEMORY_CONFIG if rows <= 256 else ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            product = ttnn.mul(
                self._linear(normalized, "gate"),
                self._linear(normalized, "up"),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
        down = self._rs(self._linear(product, "down"), decode=False, site="down")
        result = ttnn.add(residual, down, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.typecast(result, self.residual_dtype) if result.dtype != self.residual_dtype else result

    def _create_decode_heads(self, packed, batch):
        return ttnn.experimental.nlp_create_qkv_heads_decode(
            ttnn.to_memory_config(packed, ttnn.L1_MEMORY_CONFIG),
            num_heads=8,
            num_kv_heads=2,
            overlap_qk_coregrid=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

    def _concat_decode(self, attention, memcfg, batch):
        if batch <= 4:
            return ttnn.reshape(attention, (1, 1, batch, 1024), skip_padding_fill=True)
        attention = ttnn.to_memory_config(attention, memcfg)
        result = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=8, sub_core_grids=self.core_grid)
        result = ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)
        return ttnn.reshape(
            result, ttnn.Shape((1, 1, batch, 1024)), ttnn.Shape((1, 1, 32, 1024)), skip_padding_fill=True
        )

    def _decode_finish(self, residual, attention):
        projected = self._output_projection(attention, "o")
        residual = ttnn.add(
            residual,
            ttnn.to_memory_config(projected, self.local_residual_memcfg),
            memory_config=self.local_residual_memcfg,
        )
        normalized = self._norm_input(residual, decode=True, site="mlp")
        normalized = ttnn.to_memory_config(normalized, self.decode_inputs["gate_up"])
        packed = ttnn.to_memory_config(self._decode_linear(normalized, "gate_up"), ttnn.L1_MEMORY_CONFIG)
        gate, up = packed[:, :, :, :3584], packed[:, :, :, 3584:]
        product = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=gate.memory_config(),
            dtype=ttnn.bfloat16,
        )
        down = self._output_projection(product, "down")
        result = ttnn.add(
            residual,
            ttnn.to_memory_config(down, self.local_residual_memcfg),
            memory_config=self.local_residual_memcfg,
        )
        return ttnn.typecast(result, self.residual_dtype) if result.dtype != self.residual_dtype else result

    def prefill_forward(self, x, *, page_table, kv_cache):
        """Run new paged prompts, with the logical shape and cache contract above."""
        self._validate_cache(page_table, kv_cache)
        batch, one, seq_len, hidden = x.shape
        if not (1 <= batch <= 32 and one == 1 and hidden == self.hidden and 1 <= seq_len <= self.supported_context):
            raise ValueError(f"Invalid prefill shape {x.shape}")
        if page_table.shape[0] != batch or page_table.shape[1] * self.page_size < seq_len:
            raise ValueError("Page table must cover each logical prompt")
        outputs = []
        for user in range(batch):
            table = page_table[user : user + 1, :]
            # Flexible SDPA requires a 32-byte-aligned page-table row. Its valid
            # Q bound prevents reads from these padding entries.
            if table.shape[1] % 8:
                table = ttnn.pad(table, ((0, 0), (0, (-table.shape[1]) % 8)), value=0)
            chunks = []
            for start in range(0, seq_len, self.prefill_chunk_size):
                length = min(self.prefill_chunk_size, seq_len - start)
                padded = (length + 31) // 32 * 32
                part = x[user : user + 1, :, start : start + length, :]
                part = (
                    ttnn.pad(part, ((0, 0), (0, 0), (0, padded - length), (0, 0)), value=0.0)
                    if padded != length
                    else part
                )
                n = self._norm_input(part, decode=False, site="attn")
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    self._linear(n, "qkv"),
                    num_heads=8,
                    num_kv_heads=2,
                    transpose_k_heads=False,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                cos = self.cos_chunks[start // self.prefill_chunk_size]
                sin = self.sin_chunks[start // self.prefill_chunk_size]
                q = ttnn.experimental.rotary_embedding(q, cos, sin, compute_kernel_config=self.compute)
                k = ttnn.experimental.rotary_embedding(k, cos, sin, compute_kernel_config=self.compute)
                fill_table = table[:, start // 128 :]
                ttnn.experimental.paged_fill_cache(
                    kv_cache[0], ttnn.typecast(k, self.cache_dtype), fill_table, batch_idx=0
                )
                ttnn.experimental.paged_fill_cache(
                    kv_cache[1], ttnn.typecast(v, self.cache_dtype), fill_table, batch_idx=0
                )
                attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q,
                    kv_cache[0],
                    kv_cache[1],
                    table,
                    chunk_start_idx_tensor=self.chunk_starts[start // self.prefill_chunk_size],
                    program_config=ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=self.grid,
                        q_chunk_size=next(c for c in (256, 128, 64, 32) if c <= 128 and padded % c == 0),
                        # Fewer BF16 online-output accumulation steps are required
                        # for accurate 128K attention.
                        k_chunk_size=256 if seq_len <= 256 else 512,
                        exp_approx_mode=False,
                    ),
                    compute_kernel_config=self.sdpa_compute,
                )
                attention = ttnn.experimental.nlp_concat_heads(
                    attention,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                chunks.append(self._finish(part, attention)[:, :, :length, :])
                # The next chunk can use the short-tail norm. Its sharded CBs
                # need the L1 space held by this chunk's norm/attention outputs.
                # Drop Python references; part may alias x, so do not explicitly
                # deallocate tensor storage shared with the full prompt.
                del n, q, k, v, attention, part, fill_table
            outputs.append(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))
        return outputs[0] if batch == 1 else ttnn.concat(outputs, dim=0)

    def decode_forward(self, x, *, current_pos, page_table, kv_cache, rotary_pos=None):
        """Trace-safe paged decode; positions and page mappings are device inputs."""
        self._validate_cache(page_table, kv_cache)
        batch = x.shape[2]
        if tuple(x.shape) != (1, 1, batch, self.hidden) or not 1 <= batch <= 32:
            raise ValueError(f"Invalid decode shape {x.shape}")
        if self.decode_workspace is None or self.decode_workspace.batch != batch:
            raise ValueError("Call prepare_decode(batch) before decode warmup/capture")
        if tuple(current_pos.shape) != (batch,) or current_pos.dtype != ttnn.int32 or page_table.shape[0] != batch:
            raise ValueError("One current position and page-table row per decode lane required")
        residual = ttnn.to_memory_config(x, self.local_residual_memcfg)
        n = self._norm_input(residual, decode=True, site="attn")
        q, k, v = self._create_decode_heads(self._decode_linear(n, "qkv"), batch)
        cos, sin = self._rope_tables(current_pos if rotary_pos is None else rotary_pos, batch, q.memory_config())
        q = self._decode_rope(q, cos, sin)
        k = self._decode_rope(k, cos, sin)
        self._update_cache(k, v, current_pos, page_table, kv_cache)
        # Decode SDPA reads whole K chunks before causal masking. A logical
        # page table can end within the final chunk; map its masked tail to a
        # valid physical page instead of letting the kernel read past the row.
        pages_per_chunk = max(1, 256 // self.page_size)
        tail_pages = (-page_table.shape[1]) % pages_per_chunk
        attention_table = ttnn.pad(page_table, ((0, 0), (0, tail_pages)), value=0) if tail_pages else page_table
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            cur_pos_tensor=current_pos,
            page_table_tensor=attention_table,
            program_config=self.decode_sdpa_program,
            compute_kernel_config=self.decode_sdpa_compute,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attention = self._concat_decode(attention, q.memory_config(), batch)
        return self._decode_finish(residual, attention)
