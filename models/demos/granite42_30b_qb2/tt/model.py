# SPDX-License-Identifier: Apache-2.0
"""Granite autoregressive model with tensor parallelism across four devices.

All forwards are device-only. The generator owns persistent request state and
host API boundaries. Weights are loaded one layer at a time at construction.
"""

import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

import ttnn

from .decoder import GraniteDecoder
from .precision import layer_policy, load_precision

MODEL_ID = "ibm-granite/granite-4.2-30b"
REVISION = "9e668ce1c538387ef24d3644e9b0606647762636"


class GraniteModel:
    def __init__(self, mesh_device, *, override_num_layers=None, precision_config=None):
        self.mesh = mesh_device
        self.precision = load_precision(precision_config)
        self.cache_dtype = getattr(ttnn, self.precision["kv_cache_dtype"])
        self.activation_dtype = getattr(ttnn, self.precision["activation_dtype"])
        self.logits_dtype = getattr(ttnn, self.precision["logits_dtype"])
        self.config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        self.checkpoint = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
        self.index = json.loads((self.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        self.layers = []
        count = self.config.num_hidden_layers if override_num_layers is None else override_num_layers
        if not 1 <= count <= self.config.num_hidden_layers:
            raise ValueError("Invalid layer count; reduced stacks are debugging only")
        for i in range(count):
            prefix = f"model.layers.{i}."
            weights = self.load([k for k in self.index if k.startswith(prefix)])
            self.layers.append(
                GraniteDecoder.from_state_dict(
                    weights,
                    hf_config=self.config,
                    layer_idx=i,
                    policy=layer_policy(self.precision, i),
                    mesh_device=mesh_device,
                    collective_resources=self.layers[0].collective_resources if self.layers else None,
                )
            )
            print(f"GRANITE_LAYER_READY {i+1}/{count}", flush=True)
        del weights
        terminal = self.load(
            ["model.embed_tokens.weight", "model.norm.weight"]
            + ([] if self.config.tie_word_embeddings else ["lm_head.weight"])
        )
        # Use owned source storage: the mapped-buffer upload stalls on this host;
        # the matched clone control passes. The native transfer cause is unresolved.
        self.embedding = self.put(
            terminal["model.embed_tokens.weight"].clone(),
            dtype=getattr(ttnn, self.precision["weight_groups"]["embedding"]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # Fold final norm gamma and logits scaling into the terminal projection.
        head = terminal["model.embed_tokens.weight"] if self.config.tie_word_embeddings else terminal["lm_head.weight"]
        head = (head.float() * terminal["model.norm.weight"].float() / self.config.logits_scaling).T.bfloat16()
        self.lm_head = self.put(head, shard=1, dtype=getattr(ttnn, self.precision["weight_groups"]["lm_head"]))
        self.head_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["compute_fidelities"]["lm_head"]),
            math_approx_mode=False,
            fp32_dest_acc_en=self.precision["fp32_accumulation"]["lm_head"],
            packer_l1_acc=True,
        )
        # Precision-locked measured decode head: two readers per DRAM bank.
        dg = mesh_device.dram_grid_size()
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
        head_width = self.config.vocab_size // 4
        head_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(banks, [4096, math.ceil(head_width / (dg.x * 64)) * 64], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_head = ttnn.to_memory_config(self.lm_head, head_mem)
        self.head_output_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(16, mesh_device.compute_with_storage_grid_size(), row_wise=True),
                [32, head_width // 16],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.head_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=2,
            per_core_M=1,
            per_core_N=head_width // 16 // 32,
            num_workers_per_dram_bank=2,
        )
        positions = torch.arange(self.config.max_position_embeddings + 1024, dtype=torch.float32)
        inv = 1 / (self.config.rope_parameters["rope_theta"] ** (torch.arange(0, 128, 2).float() / 128))
        angles = positions[:, None] * inv[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        self.cos_table = self.put(angles.cos().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sin_table = self.put(angles.sin().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.rope_mem = {}
        for b in (1, 8, 16):
            grid = ttnn.num_cores_to_corerangeset(b, mesh_device.compute_with_storage_grid_size(), row_wise=True)
            self.rope_mem[b] = ttnn.create_sharded_memory_config(
                (32, 128), grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )

    def load(self, names):
        result = {}
        for file in sorted({self.index[n] for n in names}):
            with safe_open(self.checkpoint / file, framework="pt") as f:
                for n in names:
                    if self.index[n] == file:
                        result[n] = f.get_tensor(n)
        return result

    def put(self, value, *, dtype=None, layout=ttnn.TILE_LAYOUT, shard=None):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype or self.activation_dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.mesh) if shard is None else ttnn.ShardTensorToMesh(self.mesh, dim=shard)
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def allocate_cache(self, pages):
        # Each rank owns two distinct KV heads. zeros need no host sharding.
        return [
            [
                ttnn.zeros([pages, 2, 32, 128], dtype=self.cache_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh)
                for _ in range(2)
            ]
            for _ in self.layers
        ]

    def embed(self, tokens):
        ids = ttnn.reshape(tokens, [1, tokens.volume()])
        x = ttnn.embedding(ids, self.embedding, layout=ttnn.TILE_LAYOUT)
        x = ttnn.unsqueeze_to_4D(x)
        return x if self.config.embedding_multiplier == 1 else ttnn.multiply(x, self.config.embedding_multiplier)

    def terminal(self, x, *, decode=False):
        n = self.layers[0].norm(x)
        if decode:
            logits = ttnn.linear(
                n,
                self.decode_head,
                program_config=self.head_program,
                compute_kernel_config=self.head_compute,
                memory_config=self.head_output_mem,
                dtype=self.logits_dtype,
            )
            return ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
        n = ttnn.to_memory_config(n, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            n,
            self.lm_head,
            compute_kernel_config=self.head_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(x=8, y=8),
            dtype=self.logits_dtype,
        )

    def prefill_forward(self, tokens, *, entry, kv_cache, last_index=None, valid_seq_len=None):
        x = self.embed(tokens)
        for layer, cache in zip(self.layers, kv_cache):
            x = layer.prefill_chunk(
                x,
                page_table=entry.page_table,
                chunk_page_table=entry.write_page_table,
                kv_cache=cache,
                cos=entry.cos,
                sin=entry.sin,
                start_pos=entry.position,
                valid_seq_len=valid_seq_len,
            )
        if last_index is not None:
            rows = ttnn.reshape(ttnn.untilize(x, use_multicore=True), [x.shape[2], 4096])
            x = ttnn.unsqueeze_to_4D(ttnn.embedding(last_index, rows, layout=ttnn.TILE_LAYOUT))
        # Last-token prefill has the same one-row terminal shape as decode.
        return self.terminal(x, decode=last_index is not None)

    def decode_forward(self, tokens, *, current_pos, rope_indices, page_table, kv_cache, advance=True):
        batch = current_pos.shape[0]
        x = self.embed(tokens)
        rope = []
        for table in (self.cos_table, self.sin_table):
            r = ttnn.embedding(rope_indices, table, layout=ttnn.TILE_LAYOUT)
            r = ttnn.transpose(ttnn.unsqueeze_to_4D(r), 1, 2)
            rope.append(ttnn.to_memory_config(r, self.rope_mem[batch]))
        for layer, cache in zip(self.layers, kv_cache):
            x = layer.decode_forward(
                x, current_pos=current_pos, page_table=page_table, kv_cache=cache, cos=rope[0], sin=rope[1]
            )
        logits = self.terminal(x, decode=True)
        if advance:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rope_indices)
        return logits
