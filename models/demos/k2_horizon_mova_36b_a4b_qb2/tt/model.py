# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full K2 Horizon model around the accepted TP4/EP4 decoder stack.

Only construction touches checkpoint/host tensors. Forward methods operate on
device tensors; the generator owns logical lengths, request state and tracing.
"""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.common.tt_ccl import get_tt_ccl

from .multichip_decoder import MultichipDecoder
from .precision import layer_options, load_precision

HF_MODEL = "IFM/K2-Horizon-MoVA-36B-A4B"
HF_REVISION = "e9ce717642b2074082875be233c02c729ba47960"


def _mounted_snapshot():
    """Serving harnesses (TTI) mount the checkpoint and export MODEL_WEIGHTS_DIR (or HF_MODEL) as a local snapshot
    directory. Prefer it when it holds config.json and the safetensors, so the container needs no hub access."""
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        value = os.environ.get(name)
        if value and (Path(value) / "config.json").is_file() and any(Path(value).glob("*.safetensors")):
            return Path(value)
    return None


MOUNTED_SNAPSHOT = _mounted_snapshot()


def pretrained_source():
    """(name_or_path, revision) for HF config / tokenizer loading: the mounted snapshot, else the pinned hub revision."""
    if MOUNTED_SNAPSHOT is not None:
        return str(MOUNTED_SNAPSHOT), None
    return HF_MODEL, HF_REVISION


class Checkpoint:
    def __init__(self, path=None):
        if path is None:
            path = MOUNTED_SNAPSHOT or snapshot_download(HF_MODEL, revision=HF_REVISION, local_files_only=True)
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def load(self, prefix):
        result = {}
        for shard in sorted({v for k, v in self.index.items() if k.startswith(prefix)}):
            with safe_open(self.path / shard, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        result[key] = f.get_tensor(key)
        return result


class K2HorizonModel:
    def __init__(
        self,
        mesh_device,
        *,
        checkpoint=None,
        layer_indices=None,
        max_seq_len=524288,
        chunk_size=4096,
        precision_config=None,
    ):
        self.mesh_device = mesh_device
        self.precision = load_precision(precision_config)
        self.kv_cache_dtype = getattr(ttnn, self.precision["kv_cache_dtype"])
        source, revision = pretrained_source()
        self.config = AutoConfig.from_pretrained(
            source, revision=revision, trust_remote_code=True, local_files_only=True
        )
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("The accepted model requires a 1x4 Blackhole mesh")
        if not 1 <= max_seq_len <= min(self.config.max_position_embeddings, self.precision["max_context"]):
            raise ValueError("Invalid context capacity")
        self.max_seq_len, self.chunk_size = max_seq_len, chunk_size
        self.layer_indices = tuple(range(48) if layer_indices is None else layer_indices)
        self.ccl = get_tt_ccl(mesh_device)
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["norm_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.head_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["lm_head_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        ckpt = Checkpoint(checkpoint)
        embedding = ckpt.load("model.embed_tokens.")["model.embed_tokens.weight"]
        # Safetensors returns a file-backed mmap. Long-term DMA pinning of this
        # mapping loops in this host kernel's file-THP/GUP path. A private setup
        # copy preserves every BF16 bit and avoids pinning checkpoint pages.
        self.embedding = self.put(
            embedding.clone().reshape(1, 1, *embedding.shape),
            dtype=getattr(ttnn, self.precision["embedding_dtype"]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        head = embedding if self.config.tie_word_embeddings else ckpt.load("lm_head.")["lm_head.weight"]
        self.vocab_size = self.config.vocab_size
        self.padded_vocab_size = 262144
        head = torch.nn.functional.pad(head, (0, 0, 0, self.padded_vocab_size - self.vocab_size))
        self.lm_head = self.put(
            head.T.reshape(1, 1, 2560, self.padded_vocab_size),
            dtype=getattr(ttnn, self.precision["lm_head_dtype"]),
            shard=-1,
        )
        # Stream the selected-precision decode head from all DRAM banks.
        # Two local32768-column chunks keep output/circular buffers
        # bounded. The interleaved bank remains for multi-tile prefill logits.
        banks = mesh_device.dram_grid_size().x
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        head_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (2560, 32768 // banks), ttnn.ShardOrientation.ROW_MAJOR),
        )
        rank_heads = head.reshape(4, 65536, 2560)
        self.decode_lm_heads = [
            ttnn.from_torch(
                rank_heads[:, start : start + 32768].reshape(131072, 2560).T.contiguous()[None, None],
                device=mesh_device,
                dtype=getattr(ttnn, self.precision["lm_head_dtype"]),
                layout=ttnn.TILE_LAYOUT,
                memory_config=head_memory,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            for start in (0, 32768)
        ]
        self.decode_head_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.precision.get("lm_head_in0_block_w", 1),
            per_core_M=1,
            per_core_N=128,
            num_workers_per_dram_bank=4,
            use_multicore_pipeline=True,
        )
        gamma = ckpt.load("model.norm.")["model.norm.weight"]
        self.norm_weights = [
            self.put(g.reshape(1, 1, 1, 1280), dtype=getattr(ttnn, self.precision["norm_dtype"]))
            for g in gamma.chunk(2)
        ]
        del head, gamma, embedding
        # HF default RoPE computes angles in float32 before BF16 conversion.
        inv = 1.0 / (self.config.rope_parameters["rope_theta"] ** (torch.arange(0, 128, 2).float() / 128))
        angles = torch.arange(max_seq_len).float()[:, None] * inv[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        self.rope_tables = [
            self.put(f(angles).bfloat16()[None, None], layout=ttnn.ROW_MAJOR_LAYOUT) for f in (torch.cos, torch.sin)
        ]
        del angles
        self.layers = []
        for layer_idx in self.layer_indices:
            print(f"LOAD_LAYER {layer_idx}", flush=True)
            state = ckpt.load(f"model.layers.{layer_idx}.")
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    state,
                    hf_config=self.config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    chunk_size=chunk_size,
                    ccl=self.ccl,
                    options=layer_options(self.precision, layer_idx),
                )
            )
            del state

    def put(self, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
        mapper = (
            ttnn.ReplicateTensorToMesh(self.mesh_device)
            if shard is None
            else ttnn.ShardTensorToMesh(self.mesh_device, dim=shard)
        )
        return ttnn.from_torch(
            value.contiguous(),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def allocate_kv_cache(self, num_blocks, page_size=32, dtype=None):
        dtype = self.kv_cache_dtype if dtype is None else dtype
        if page_size not in (32, 64, 128, 256):
            raise ValueError("Supported page sizes are 32, 64, 128 and 256")
        # Each rank owns two heads. Identical zero allocation is intentional;
        # the per-rank K/V values written later are different TP head shards.
        shape = (num_blocks, 2, page_size, 128)
        return [
            [
                ttnn.zeros(
                    shape,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
            for _ in self.layers
        ]

    def embed(self, tokens):
        return self.lookup(tokens, self.embedding)

    def lookup(self, indices, table):
        result = ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(result, (1, 1, indices.shape[-1], table.shape[-1]))

    def terminal(self, hidden):
        # Final normalization has the same two complete RMS groups as HF.
        hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        groups = [
            ttnn.rms_norm(
                hidden[..., i * 1280 : (i + 1) * 1280],
                weight=self.norm_weights[i],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for i in range(2)
        ]
        normed = ttnn.concat(groups, dim=-1)
        if normed.shape[2] <= 32:
            activation = ttnn.interleaved_to_sharded(
                normed, (8, 1), (32, 320), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
            )
            chunks = [
                ttnn.to_memory_config(
                    ttnn.linear(
                        activation,
                        weight,
                        dtype=getattr(ttnn, self.precision["logits_dtype"]),
                        compute_kernel_config=self.head_compute,
                        program_config=self.decode_head_program,
                        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
                    ),
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for weight in self.decode_lm_heads
            ]
            return ttnn.concat(chunks, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            normed,
            self.lm_head,
            dtype=getattr(ttnn, self.precision["logits_dtype"]),
            compute_kernel_config=self.head_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

    def decode_forward(self, tokens, *, current_pos, rope_indices, page_table, kv_cache, batch_size):
        hidden = self.embed(tokens)[:, :, :batch_size, :]
        rotations = [self.lookup(rope_indices, table)[:, :, :batch_size, :] for table in self.rope_tables]
        cos, sin = [ttnn.transpose(r, 1, 2) for r in rotations]
        for layer, cache in zip(self.layers, kv_cache):
            hidden = layer.decode_forward(
                hidden, cos=cos, sin=sin, current_pos=current_pos, page_table=page_table, kv_cache=cache
            )
        # Canonical sampler consumes a physical tile of rows, vocab-sharded.
        logits = self.terminal(ttnn.pad(hidden, [(0, 0), (0, 0), (0, 32 - batch_size), (0, 0)], 0))
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rope_indices)
        return logits

    def prefill_chunk(
        self,
        tokens,
        *,
        position_ids,
        current_pos,
        page_table,
        fill_page_table,
        kv_cache,
        long_context=False,
        tokenwise=False,
        output="last",
    ):
        """Bounded physical bucket with runtime absolute positions/page mappings.

        Aligned buckets contain only real tokens. Awkward tails use the accepted
        BF16 paged-update continuation path; no padded row writes another slot.
        """
        hidden = self.embed(tokens)
        cos, sin = [self.lookup(position_ids, table) for table in self.rope_tables]
        for layer, cache in zip(self.layers, kv_cache):
            if tokenwise:
                hidden = layer.decode_forward(
                    hidden[:, :, :1, :],
                    cos=cos[:, :, :1, :],
                    sin=sin[:, :, :1, :],
                    current_pos=current_pos,
                    page_table=page_table,
                    kv_cache=cache,
                    _prefill=True,
                )
            else:
                grid = self.mesh_device.compute_with_storage_grid_size()
                hidden = layer._prefill_chunk(
                    hidden,
                    cos=cos,
                    sin=sin,
                    page_table=page_table,
                    kv_cache=cache,
                    start_pos=0,
                    fill_page_table=fill_page_table,
                    chunk_start_tensor=current_pos,
                    attention_grid_override=(grid.x, grid.y) if long_context else (8, 8),
                )
        if output == "none":
            return None
        if output in ("last", "hidden"):
            hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
            hidden = hidden[:, :, hidden.shape[2] - 1 : hidden.shape[2], :]
        if output == "hidden":
            return ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        return self.terminal(hidden)
