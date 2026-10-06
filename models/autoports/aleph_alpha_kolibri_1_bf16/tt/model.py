# SPDX-License-Identifier: Apache-2.0
"""TP4 Kolibri autoregressive model. Runtime methods operate only on TT tensors."""

import gc

import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _create_dram_sharded_mem_config

from .checkpoint import CONTEXT, config, load_weights
from .multichip_decoder import CollectiveWorkspace, MultichipDecoder
from .precision import layer_policy, load_precision_config

DRAM = ttnn.DRAM_MEMORY_CONFIG


class KolibriModel:
    def __init__(
        self,
        mesh_device,
        *,
        layer_indices=None,
        policy=None,
        precision_config=None,
        rope_capacity=CONTEXT,
        sharded_terminal_norm=True,
    ):
        self.device = mesh_device
        self.sharded_terminal_norm = sharded_terminal_norm
        self.config = config()
        self.precision_config = load_precision_config(precision_config)
        self.runtime_precision = self.precision_config["runtime"]
        self.policy = policy or layer_policy(self.precision_config, 0)
        self.workspace = CollectiveWorkspace(mesh_device, self.policy)
        self.layer_indices = list(range(50)) if layer_indices is None else list(layer_indices)
        self.layers = []
        for idx in self.layer_indices:
            w = load_weights(f"model.layers.{idx}.")
            layer = MultichipDecoder.from_state_dict(
                w,
                hf_config=self.config,
                layer_idx=idx,
                mesh_device=mesh_device,
                policy=policy or layer_policy(self.precision_config, idx),
                collective_workspace=self.workspace,
                sliding_cache_tokens=8704 if self.config.layer_types[idx] == "sliding_attention" else None,
            )
            self.layers.append(layer)
            del w
            gc.collect()
            print(f"LOADED_LAYER {idx}", flush=True)
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.runtime_precision["norm_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Own the upload buffer: the checkpoint's file-backed mmap stalls in
        # automatic pinned transfer on this KMD/runtime. An anonymous copy
        # passes even with identical pointer alignment (embedding_transfer_probe).
        embedding = load_weights("model.embed_tokens.")["model.embed_tokens.weight"].clone()
        self.embedding = self.tensor(
            embedding, dtype=getattr(ttnn, self.runtime_precision["embedding_dtype"]), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        del embedding
        print("LOADED_EMBEDDING", flush=True)
        self.norm = self.tensor(
            load_weights("model.norm.")["model.norm.weight"].reshape(1, 1, 1, -1),
            dtype=getattr(ttnn, self.runtime_precision["norm_dtype"]),
        )
        self.head_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.runtime_precision["head_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=self.runtime_precision["head_fp32"],
            packer_l1_acc=True,
        )
        head = load_weights("lm_head.")["lm_head.weight"]
        # Two 16000-column vocabulary pieces per rank, padded independently
        # to the eight-bank DRAM grid. LMHead1D trims padding before concat,
        # retaining the sampler's real 32000-column rank offsets.
        head = head.T.contiguous()
        dram = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
        self.head_input_memory = ttnn.create_sharded_memory_config(
            (32, 64),
            ttnn.CoreGrid(x=8, y=5),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        head_weights = []
        for split in range(2):
            pieces = [
                torch.nn.functional.pad(
                    head[:, rank * 32000 + split * 16000 : rank * 32000 + (split + 1) * 16000], (0, 128)
                )
                for rank in range(4)
            ]
            head_weights.append(
                LazyWeight(
                    source=torch.cat(pieces, dim=-1), dtype=getattr(ttnn, self.runtime_precision["head_weight_dtype"])
                )
            )
        weight_memory = _create_dram_sharded_mem_config(2560, 16128, dram_grid, dram_cores=dram.x)
        self.head = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=head_weights,
                mesh_device=mesh_device,
                dim=2560,
                lm_head_dtype=getattr(ttnn, self.runtime_precision["head_output_dtype"]),
                compute_kernel_config=self.head_compute,
                output_memcfg=DRAM,
                input_memcfg=self.head_input_memory,
                weights_memcfgs=[weight_memory] * 2,
                output_split_sizes=[16000] * 2,
                program_configs=[
                    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=1, per_core_M=1, per_core_N=13, num_workers_per_dram_bank=1
                    )
                    for _ in range(2)
                ],
            )
        )
        self.head.load_device_weights()
        del head
        # Absolute RoPE lookup is shared by all 40 sliding layers. No host
        # trigonometry or position rebuild is needed during token feedback.
        phase = torch.arange(rope_capacity).float()[:, None] / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
        phase = torch.cat([phase, phase], -1)
        self.cos_table = self.tensor(phase.cos().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sin_table = self.tensor(phase.sin().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.rope_capacity = rope_capacity

    def tensor(self, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous(),
            device=self.device,
            dtype=dtype,
            layout=layout,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def embed(self, tokens):
        return ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)

    def rope(self, positions, batch):
        # Keep inactive slots at -1 indefinitely while mapping their ignored
        # RoPE lookup to row0; no out-of-range unsigned lookup or host mask.
        tiled = ttnn.to_layout(ttnn.reshape(positions, (1, batch)), ttnn.TILE_LAYOUT)
        safe = ttnn.clamp(tiled, min=0, max=self.rope_capacity - 1)
        ids = ttnn.to_layout(ttnn.typecast(safe, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        return tuple(
            ttnn.reshape(ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT, memory_config=DRAM), (1, 1, batch, 128))
            for table in (self.cos_table, self.sin_table)
        )

    def activation(self, hidden):
        dtype = getattr(ttnn, self.runtime_precision["activation_dtype"])
        return ttnn.typecast(hidden, dtype) if hidden.dtype != dtype else hidden

    def residual(self, hidden):
        dtype = getattr(ttnn, self.runtime_precision["residual_dtype"])
        return ttnn.typecast(hidden, dtype) if hidden.dtype != dtype else hidden

    def terminal(self, hidden):
        hidden = self.activation(hidden)
        if hidden.shape[-2] > 32:
            # Direct model-prefill callers receive every logical logit row;
            # generator token-out prefill normally projects only its last row.
            pieces = [
                self.terminal(ttnn.slice(hidden, (0, 0, start, 0), (1, 1, min(start + 32, hidden.shape[-2]), 2560)))
                for start in range(0, hidden.shape[-2], 32)
            ]
            return ttnn.concat(pieces, dim=2, memory_config=DRAM)
        if self.sharded_terminal_norm:
            # Keep the selected decoder residual in L1 through the final norm.
            memory = self.layers[0].residual_mem
            hidden = ttnn.to_memory_config(hidden, memory)
            hidden = ttnn.rms_norm(
                hidden,
                epsilon=self.config.rms_norm_eps,
                weight=self.norm,
                compute_kernel_config=self.compute,
                program_config=self.layers[0].norm_program,
                memory_config=memory,
            )
        else:
            hidden = ttnn.to_memory_config(hidden, DRAM)
            hidden = ttnn.rms_norm(
                hidden, epsilon=self.config.rms_norm_eps, weight=self.norm, compute_kernel_config=self.compute
            )
        return self.head(ttnn.to_memory_config(hidden, self.head_input_memory))

    def decode_forward(self, tokens, *, current_pos, page_tables, kv_cache, rope_positions=None):
        batch = current_pos.shape[0]
        hidden = ttnn.reshape(self.embed(ttnn.reshape(tokens, (1, batch))), (1, 1, batch, 2560))
        cos, sin = self.rope(current_pos if rope_positions is None else rope_positions, batch)
        for layer_index, (layer, cache) in enumerate(zip(self.layers, kv_cache)):
            hidden = layer.decode_forward(
                self.activation(hidden),
                kv_cache=cache,
                page_table=page_tables[
                    layer_index if layer_index in page_tables else ("sliding" if layer.sliding else "full")
                ],
                current_pos=current_pos,
                cos=cos,
                sin=sin,
            )
            hidden = self.residual(hidden)
        return self.terminal(hidden)

    def prefill_chunk_forward(
        self,
        tokens,
        *,
        kv_cache,
        page_tables,
        chunk_page_tables,
        chunk_start,
        cos,
        sin,
        context_bound,
        chunk_start_alignment=32,
        return_hidden=False,
    ):
        hidden = ttnn.reshape(self.embed(tokens), (1, 1, tokens.shape[-1], 2560))
        for layer_index, (layer, cache) in enumerate(zip(self.layers, kv_cache)):
            key = layer_index if layer_index in page_tables else ("sliding" if layer.sliding else "full")
            hidden = layer.prefill_chunk_forward(
                self.activation(hidden),
                kv_cache=cache,
                page_table=page_tables[key],
                chunk_page_table=chunk_page_tables[key],
                chunk_start=chunk_start,
                chunk_start_alignment=chunk_start_alignment,
                cos=cos,
                sin=sin,
                context_length_bound=context_bound,
            )
            hidden = self.residual(hidden)
        return hidden if return_hidden else self.terminal(hidden)
