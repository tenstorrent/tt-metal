# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full TP4 Llama-3.1-8B path; all runtime model arithmetic executes in TTNN."""

import json
import os
from copy import copy
from dataclasses import replace
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.modules.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def checkpoint_path():
    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if model_path is not None:
        return Path(model_path)
    return Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))


class Checkpoint:
    """Read only the currently converted layer from the pinned safetensors."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.index = json.loads((self.folder / "model.safetensors.index.json").read_text())["weight_map"]

    def load(self, names):
        result = {}
        for shard in sorted({self.index[name] for name in names}):
            with safe_open(self.folder / shard, framework="pt", device="cpu") as handle:
                for name in names:
                    if self.index[name] == shard:
                        result[name] = handle.get_tensor(name)
        return result


class LlamaModel:
    supported_context = 131072
    page_size = 128
    vocab_size = 128256
    padded_vocab_size = 131072

    def __init__(
        self,
        mesh_device,
        *,
        max_batch_size=1,
    ):
        if tuple(mesh_device.shape) != (1, 4) or not 1 <= max_batch_size <= 32:
            raise ValueError("Requires the reserved TP4 mesh and 1..32 fixed batch slots")
        self.precision_policy = load_precision_config()
        p = self.precision_policy
        self.supported_context = p["runtime"]["supported_context"]
        self.page_size = p["runtime"]["page_size"]
        self.cache_dtype = getattr(ttnn, p["kv_cache_dtype"])
        self.logits_dtype = getattr(ttnn, p["logits_dtype"])
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.folder = checkpoint_path()
        self.config = AutoConfig.from_pretrained(self.folder, local_files_only=True)
        self.num_layers = self.config.num_hidden_layers
        if self.num_layers != 32:
            raise ValueError("Llama 3.1 8B requires all 32 decoder layers")
        self.ccl = TT_CCL(mesh_device)
        checkpoint = Checkpoint(self.folder)
        self.layers = []
        workspace = None
        for index in range(self.num_layers):
            names = [n for n in checkpoint.index if n.startswith(f"model.layers.{index}.")]
            layer = LlamaDecoder.from_state_dict(
                checkpoint.load(names),
                hf_config=self.config,
                layer_idx=index,
                mesh_device=mesh_device,
                ccl=self.ccl,
                precision_policy=p,
                rope_state=self.layers[0] if self.layers else None,
            )
            workspace = layer.prepare_decode(max_batch_size, workspace=workspace)
            self.layers.append(layer)
            logger.debug("Loaded decoder layer {}/{}", index + 1, self.num_layers)
        self.decode_families = {max_batch_size: self.layers}
        terminal_names = ["model.embed_tokens.weight", "model.norm.weight"]
        if not self.config.tie_word_embeddings:
            terminal_names.append("lm_head.weight")
        weights = checkpoint.load(terminal_names)
        embedding = weights["model.embed_tokens.weight"]
        self.embedding_weight = ttnn.from_torch(
            embedding,
            dtype=getattr(ttnn, p["weight_groups"]["embedding"]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        head = embedding if self.config.tie_word_embeddings else weights["lm_head.weight"]
        # Fold the final RMSNorm affine into the column-parallel projection.
        head = (head.float() * weights["model.norm.weight"].float()[None, :]).bfloat16().T.contiguous()
        head = torch.nn.functional.pad(head, (0, self.padded_vocab_size - self.vocab_size))
        opts = {
            **p["lm_head_geometry"],
            "dtype": p["weight_groups"]["lm_head"],
            "fidelity": p["compute_fidelities"]["decode"]["lm_head"],
        }
        self.lm_prefill_weight = ttnn.from_torch(
            head,
            dtype=getattr(ttnn, opts["dtype"]),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        local_width = self.padded_vocab_size // 4
        width = local_width // opts["splits"]
        banks = mesh_device.dram_grid_size().x
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [4096, width // banks], ttnn.ShardOrientation.ROW_MAJOR),
        )
        output_weights = []
        for split in range(opts["splits"]):
            source = torch.cat(
                [head[:, d * local_width + split * width : d * local_width + (split + 1) * width] for d in range(4)],
                dim=1,
            ).contiguous()
            output_weights.append(
                LazyWeight(
                    source=source,
                    dtype=getattr(ttnn, opts["dtype"]),
                    device=mesh_device,
                    mesh_mapper_config=ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementShard(-1)], mesh_shape_override=ttnn.MeshShape([4])
                    ),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory,
                )
            )
        program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=opts["block"],
            per_core_M=1,
            per_core_N=width // 32 // banks // opts["readers"],
            num_workers_per_dram_bank=opts["readers"],
        )
        self.lm_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, opts["fidelity"]),
            math_approx_mode=p["accumulation"]["math_approx_mode"],
            fp32_dest_acc_en=p["accumulation"]["matmul_fp32"],
            packer_l1_acc=True,
        )
        self.lm_head = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=output_weights,
                mesh_device=mesh_device,
                dim=4096,
                max_batch_size=max_batch_size,
                program_configs=[program] * opts["splits"],
                compute_kernel_config=self.lm_compute,
                output_split_sizes=[width] * opts["splits"],
                lm_head_dtype=self.logits_dtype,
                output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                input_memcfg=self.layers[0]._width_memcfg(4096, opts["cores"]),
                weights_memcfgs=[memory] * opts["splits"],
            )
        )
        self.lm_head.load_device_weights()
        self.prefill_head_program = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=8,
            N_block_size=16,
            subblock_h=2,
            subblock_w=4,
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        )
        logger.info("Llama 3.1-8B ready: {} layers, precision {}", self.num_layers, p["config_id"])

    def allocate_cache(self, num_physical_pages):
        return [layer.allocate_cache(num_physical_pages=num_physical_pages) for layer in self.layers]

    def prefill_hidden(self, tokens, *, page_table, kv_cache):
        """One logical prompt; padding/chunking stays within the TTNN decoder."""
        x = ttnn.embedding(
            tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.reshape(x, (1, 1, tokens.shape[-1], 1024))
        for layer, cache in zip(self.layers, kv_cache):
            x = layer.prefill_forward(x, page_table=page_table, kv_cache=cache)
        return x

    def prefill_head(self, hidden):
        norm = self.layers[0]._norm_input(hidden, decode=False, site="attn")
        if hidden.shape[2] == 1:
            norm = ttnn.to_memory_config(norm, self.lm_head.config.input_memcfg)
            return self.lm_head(norm)
        return ttnn.experimental.minimal_matmul(
            norm,
            self.lm_prefill_weight,
            config=self.prefill_head_program,
            compute_kernel_config=self.lm_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prepare_single_user_decode(self):
        """Share physical decode workspace and weights between serial CQ0 graphs."""
        if 1 in self.decode_families:
            return

        workspace = self.layers[0].decode_workspace
        workspace = replace(
            workspace,
            batch=1,
            buffers={
                key: ttnn.reshape(
                    value, ttnn.Shape((1, 1, 1, value.shape[-1])), value.padded_shape, skip_padding_fill=True
                )
                for key, value in workspace.buffers.items()
            },
        )
        family = []
        for layer in self.layers:
            alias = copy(layer)
            alias.decode_workspace = None
            alias.prepare_decode(1, workspace=workspace)
            family.append(alias)
        self.decode_families[1] = family

    def decode(self, tokens, *, current_pos, rotary_pos, page_table, kv_cache, execution_batch=None):
        """Device-only fixed-slot forward. tokens is the sampler output allocation."""
        batch = self.max_batch_size if execution_batch is None else execution_batch
        layers = self.decode_families[batch]
        indices = ttnn.reshape(tokens, (1, 32))
        x = ttnn.embedding(
            indices,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=layers[0].local_residual_memcfg,
        )
        x = ttnn.reshape(x, ttnn.Shape((1, 1, batch, 1024)), ttnn.Shape((1, 1, 32, 1024)), skip_padding_fill=True)
        for layer, cache in zip(layers, kv_cache):
            x = layer.decode_forward(
                x, current_pos=current_pos, rotary_pos=rotary_pos, page_table=page_table, kv_cache=cache
            )
        x = layers[0]._norm_input(x, decode=True, site="attn")
        x = ttnn.to_memory_config(x, self.lm_head.config.input_memcfg)
        logits = self.lm_head(x)
        # Common sampling processes a physical tile of rows; only fixed slots
        # are model inputs, and public APIs slice inactive/padded output rows.
        shape = ttnn.Shape((1, 1, 32, self.padded_vocab_size // 4))
        return ttnn.reshape(logits, shape, shape, skip_padding_fill=True)
