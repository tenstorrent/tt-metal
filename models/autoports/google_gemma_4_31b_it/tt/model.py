# SPDX-License-Identifier: Apache-2.0
"""Pinned text model built from this run's optimized TP4 decoder.

All HF conversion is construction-time. Forward methods consume device tensors.
The decoder stack shares its two gather workspaces on one serialized mesh stream.
"""

import math

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
from ttnn.types import BlackholeComputeKernelConfig

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from .multichip_decoder import MultichipDecoder
from .precision import POLICIES, load_precision
from .weights import HF_MODEL, HF_REVISION, PinnedWeights


class Gemma4Model:
    def __init__(self, mesh_device, *, layer_indices=None, head_mode="dram", precision_config=None):
        self.precision_config = load_precision(precision_config)
        policy = self.precision_config
        self.mesh = mesh_device
        self.head_mode = head_mode
        self.config = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
        c = self.config
        self.layer_indices = tuple(range(c.num_hidden_layers) if layer_indices is None else layer_indices)
        self.allocation_bytes = {}

        def record_memory(name):
            view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
            self.allocation_bytes[name] = int(view.num_banks * view.total_bytes_allocated_per_bank)

        record_memory("before_model")
        self.ccl = TT_CCL(mesh_device)
        record_memory("terminal_ccl")
        self.gather_pool = {}
        loader = PinnedWeights()
        self.layers = []
        for index in self.layer_indices:
            print("LOAD_LAYER", index, flush=True)
            groups = dict(policy["weight_groups"])
            groups.update(policy["layer_exceptions"].get(str(index), {}))
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    loader.layer(index),
                    hf_config=c,
                    layer_idx=index,
                    mesh_device=mesh_device,
                    decode_gather_buffers=self.gather_pool,
                    precision=groups["mlp"],
                    attention_precision=groups["attention"],
                    down_precision=groups["down"],
                    cache_dtype=getattr(ttnn, policy["kv_cache_dtype"]),
                    residual_dtype=getattr(ttnn, policy["residual_dtype"]),
                    norm_dtype=getattr(ttnn, policy["norm_dtype"]),
                    decode_ccl_dtype=getattr(ttnn, policy["ccl_dtype"]["decode"]),
                    activation_dtype=getattr(ttnn, policy["activation_dtype"]),
                    prefill_ccl_bfp8=policy["ccl_dtype"]["prefill"] == "bfloat8_b",
                    sliding_cache_tokens=2048 if c.layer_types[index] == "sliding_attention" else None,
                )
            )
        record_memory("decoder_stack")
        names = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]
        terminal = loader.load(names)
        embedding = terminal[names[0]]
        # Lookup shards hidden width; LM head shards vocabulary. They represent the
        # same HF tied weight but need distinct physical orientations and layouts.
        self.embedding = self.upload(
            embedding, dtype=getattr(ttnn, policy["embedding_dtype"]), layout=ttnn.ROW_MAJOR_LAYOUT, shard=1
        )
        record_memory("embedding")
        self.lm_head = self.upload(
            embedding.T.contiguous(), dtype=getattr(ttnn, POLICIES[policy["weight_groups"]["lm_head"]][0]), shard=1
        )
        if head_mode == "dram":
            full_head = self.lm_head
            dram = mesh_device.dram_grid_size()
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (c.hidden_size, 1024), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.lm_head = []
            for start in range(0, 65536, 8192):
                chunk = ttnn.slice(full_head, (0, start), (c.hidden_size, start + 8192))
                self.lm_head.append(ttnn.to_memory_config(chunk, memory))
            del chunk, full_head
        elif head_mode != "interleaved":
            raise ValueError("Unknown head mode")
        record_memory("lm_head")
        self.norm_weight = self.upload(
            terminal[names[1]].to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=getattr(ttnn, policy["norm_dtype"])
        )
        record_memory("final_norm")
        self.embed_scale = float(torch.tensor(math.sqrt(c.hidden_size), dtype=torch.bfloat16))
        self.kernel = BlackholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, policy["compute_fidelities"]["lm_head"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.rope_tables = {}
        rope = Gemma4TextRotaryEmbedding(c)
        for kind in sorted(set(c.layer_types)):
            cos, sin = rope(
                torch.empty(1, dtype=torch.bfloat16),
                torch.arange(c.max_position_embeddings).reshape(1, -1),
                layer_type=kind,
            )
            self.rope_tables[kind] = tuple(self.upload(t.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for t in (cos, sin))

        record_memory("rope_tables")

    def upload(self, value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
        return ttnn.from_torch(
            value,
            device=self.mesh,
            dtype=dtype,
            layout=layout,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.mesh) if shard is None else ttnn.ShardTensorToMesh(self.mesh, dim=shard)
            ),
        )

    def gather(self, x, dim):
        return ttnn.experimental.all_gather_async(
            x,
            dim=dim,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def embed(self, tokens):
        x = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT)
        return ttnn.mul(self.gather(x, len(x.shape) - 1), self.embed_scale)

    def rope(self, positions, *, decode):
        if decode:
            positions = ttnn.mul(positions, ttnn.ge(positions, 0))
        positions = ttnn.typecast(positions, ttnn.uint32)
        result = {}
        for kind, tables in self.rope_tables.items():
            values = []
            for table in tables:
                value = ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT)
                # Input indices are [1,B] for decode and [B,S] for prefill.
                b, s, d = value.shape
                values.append(ttnn.reshape(value, (1, s, 1, d) if decode else (b, 1, s, d)))
            result[kind] = values
        return result

    def terminal(self, hidden):
        x = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.rms_norm(
            x, epsilon=self.config.rms_norm_eps, weight=self.norm_weight, compute_kernel_config=self.layers[0].kernel
        )
        if self.head_mode == "interleaved":
            logits = ttnn.linear(
                x,
                self.lm_head,
                dtype=getattr(ttnn, self.precision_config["logits_dtype"]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.kernel,
                core_grid=ttnn.CoreGrid(x=8, y=8),
            )
        else:
            if x.shape[-2] <= 32:
                x = ttnn.to_memory_config(x, self.layers[0]._working_memory(x, 8))
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=3, per_core_M=1, per_core_N=32, num_workers_per_dram_bank=2
                )
                output_memory = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            else:
                rows = math.ceil(x.shape[-2] / 32)
                gy = min(8, rows)
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, gy),
                    in0_block_w=3,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=math.ceil(rows / gy),
                    per_core_N=32,
                    transpose_mcast=False,
                    fused_activation=None,
                )
                output_memory = ttnn.DRAM_MEMORY_CONFIG
            parts = []
            for weight in self.lm_head:
                part = ttnn.linear(
                    x,
                    weight,
                    dtype=getattr(ttnn, self.precision_config["logits_dtype"]),
                    memory_config=output_memory,
                    program_config=pc,
                    compute_kernel_config=self.kernel,
                )
                parts.append(ttnn.to_memory_config(part, ttnn.DRAM_MEMORY_CONFIG))
                del part
            logits = ttnn.concat(parts, dim=-1)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = ttnn.mul(ttnn.tanh(ttnn.mul(logits, 1.0 / cap)), cap)
        return logits

    def allocate_cache(self, *, batch, context, pool_tokens=None):
        if batch not in (1, 8, 16, 32) or not 1 <= context <= self.config.max_position_embeddings:
            raise ValueError("Invalid batch or context")
        # Decode SDPA rounds reads to chunk boundaries; reserve a full 128-token window.
        full_blocks = math.ceil(context / 128) * 4
        caches, tables = [], []
        self.host_page_tables = []
        self.pool_blocks = batch * full_blocks if pool_tokens is None else math.ceil(pool_tokens / 128) * 4
        if self.pool_blocks < batch * 4:
            raise ValueError("Pool needs at least one rounded read window per slot")
        for layer in self.layers:
            physical = 64 if layer.sliding_window else self.pool_blocks // batch
            caches.append(layer.allocate_cache(num_blocks=batch * 64 if layer.sliding_window else self.pool_blocks))
            page = torch.arange(batch * physical).reshape(batch, physical)
            page = page[:, torch.arange(max(full_blocks, 64) if layer.sliding_window else full_blocks) % physical]
            self.host_page_tables.append(page.to(torch.int32))
            tables.append(self.upload(page.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT))
        return caches, tables

    def prefill_chunk(
        self,
        tokens,
        positions,
        *,
        kv_cache,
        page_tables,
        chunk_tables,
        chunk_start,
        valid_tokens,
        attention_tables=None,
        start_tensors=None,
        return_hidden=False,
    ):
        x = self.embed(tokens)
        x = ttnn.reshape(x, (x.shape[0], 1, x.shape[1], self.config.hidden_size))
        ropes = self.rope(positions, decode=False)
        for index, (layer, cache, page, chunk_page) in enumerate(zip(self.layers, kv_cache, page_tables, chunk_tables)):
            cos, sin = ropes[layer.layer_type]
            x = layer.prefill_chunk_forward(
                x,
                cos=cos,
                sin=sin,
                kv_cache=cache,
                page_table=page,
                chunk_page_table=chunk_page,
                chunk_start=chunk_start,
                valid_tokens=valid_tokens,
                **(
                    {"attention_page_table": attention_tables[index], "chunk_start_idx_tensor": start_tensors[index]}
                    if attention_tables is not None
                    else {}
                ),
            )
        return x if return_hidden else self.terminal(x)

    def decode(self, tokens, positions, *, kv_cache, page_tables):
        x = self.embed(tokens)
        x = ttnn.reshape(x, (1, 1, tokens.shape[-1], self.config.hidden_size))
        ropes = self.rope(positions, decode=True)
        current_pos = ttnn.reshape(positions, (positions.shape[-1],))
        for layer, cache, page in zip(self.layers, kv_cache, page_tables):
            cos, sin = ropes[layer.layer_type]
            x = layer.decode_forward(x, cos=cos, sin=sin, kv_cache=cache, page_table=page, current_pos=current_pos)
        return self.terminal(x)
