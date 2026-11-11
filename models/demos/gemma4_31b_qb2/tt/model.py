# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Gemma4 text autoregression around the accepted TP4 decoder stack.

Weights are converted at construction. Runtime methods consume device tensors;
the generator owns logical lengths, scheduling, trace inputs and host boundaries.
"""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder

HOIST_DECODE_ROPE = os.environ.get("GEMMA4_QB2_HOIST_DECODE_ROPE", "0") == "1"

HF_ID = "google/gemma-4-31B-it"
HF_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"


def checkpoint_path():
    override = os.environ.get("HF_MODEL")
    if override:
        path = Path(override)
        if not path.is_dir():
            raise ValueError(f"HF_MODEL must name a local checkpoint directory: {path}")
        return path
    return Path(snapshot_download(HF_ID, revision=HF_REVISION, local_files_only=True))


class WeightReader:
    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name):
        with safe_open(self.path / self.index[name], framework="pt") as f:
            return f.get_tensor(name)

    def layer(self, index):
        prefix = f"model.language_model.layers.{index}."
        return {k: self.get(k) for k in self.index if k.startswith(prefix)}


class Gemma4Model:
    def __init__(
        self, mesh_device, *, checkpoint=None, layer_indices=None, decode_weights=True, prefill_tile_pad=False
    ):
        """decode_weights=False builds a prefill-only stack; prefill_tile_pad=True pads prefill chunks
        with length-independent kernels (see Decoder.from_state_dict)."""
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Gemma4 31B QB2 requires a [1, 4] mesh")
        self.mesh = mesh_device
        self.activation_dtype = self.logits_dtype = ttnn.bfloat16
        self.checkpoint = Path(checkpoint or checkpoint_path())
        self.config = AutoConfig.from_pretrained(self.checkpoint, local_files_only=True).text_config
        self.layer_indices = (
            list(range(self.config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        )
        if (
            not self.layer_indices
            or len(set(self.layer_indices)) != len(self.layer_indices)
            or any(i < 0 or i >= self.config.num_hidden_layers for i in self.layer_indices)
        ):
            raise ValueError("Layer indices must be distinct valid checkpoint layers")
        self.shared_setup = {}
        reader = WeightReader(self.checkpoint)
        self.layers = []
        for index in self.layer_indices:
            logger.info("Loading Gemma4 layer {}", index)
            self.layers.append(
                Decoder.from_state_dict(
                    reader.layer(index),
                    hf_config=self.config,
                    layer_idx=index,
                    mesh_device=mesh_device,
                    shared_setup=self.shared_setup,
                    decode_weights=decode_weights,
                    prefill_tile_pad=prefill_tile_pad,
                )
            )
        self.ccl = self.shared_setup["ccl"]
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.head_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.head_grid = ttnn.CoreGrid(x=11, y=8)
        weight = reader.get("model.language_model.embed_tokens.weight")
        # The tied embedding has one physical layout for each consuming operation.
        self.embedding = self.tensor(weight, dim=1, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.head = self.tensor(weight.T.contiguous(), dim=1, dtype=ttnn.bfloat8_b)
        self.embedding_scale = torch.tensor(self.config.hidden_size**0.5, dtype=weight.dtype).item()
        norm = reader.get("model.language_model.norm.weight")
        self.norm = self.tensor(norm.reshape(1, 1, 1, -1))
        self.final_norm_mem = self.layers[0]._mem(self.config.hidden_size, 28)
        self.final_norm_rm = self.tensor(norm.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        norm_grid = self.final_norm_mem.shard_spec.grid.bounding_box().end
        self.final_norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[norm_grid.x + 1, norm_grid.y + 1],
            subblock_w=3,
            block_h=1,
            block_w=6,
            inplace=False,
        )
        # Fill setup holes with weights before placing persistent norm owners.
        # All owners and their native views exist before cache allocation/tracing.
        for layer in self.layers:
            layer._prepare_prefill_norm_stats()
        logger.info("Gemma4 model loaded")

    def tensor(self, x, *, dim=None, dtype=None, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            x.contiguous(),
            device=self.mesh,
            dtype=self.activation_dtype if dtype is None else dtype,
            layout=layout,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.mesh) if dim is None else ttnn.ShardTensorToMesh(self.mesh, dim=dim)
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def gather_hidden(self, x):
        return self.layers[0]._gather(x)

    def embed(self, tokens, *, sequence_length=None, batch=None):
        x = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=self.activation_dtype)
        # Prefill owns this newly allocated embedding. Scalar multiplication
        # is elementwise, so it need not retain a second full sequence owner.
        output = x if (sequence_length is not None) else None
        x = ttnn.multiply(x, self.embedding_scale, output_tensor=output)
        del output
        if sequence_length is not None:
            x = ttnn.reshape(
                x,
                ttnn.Shape([1, 1, sequence_length, 1344]),
                ttnn.Shape([1, 1, (sequence_length + 31) // 32 * 32, 1344]),
            )
            if self.layers[0].prefill_input_width(sequence_length) == 5376:
                x = self.gather_hidden(x)
        else:
            x = ttnn.reshape(x, [1, 1, 32, 1344])
            x = self.gather_hidden(x)
            x = ttnn.reshape(x, ttnn.Shape([1, 1, batch, 5376]), ttnn.Shape([1, 1, 32, 5376]))
            x = ttnn.to_memory_config(x, self.layers[0]._mem(5376, 28))
        return x

    def terminal(self, hidden):
        if hidden.shape[-1] == 1344:
            hidden = self.gather_hidden(hidden)
        if hidden.padded_shape[-2] == 32:
            hidden = ttnn.to_memory_config(hidden, self.final_norm_mem)
            hidden = ttnn.rms_norm(
                hidden,
                weight=self.final_norm_rm,
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.compute,
                program_config=self.final_norm_program,
                memory_config=self.final_norm_mem,
            )
            hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
            hidden = ttnn.rms_norm(
                hidden, weight=self.norm, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute
            )
        logits = ttnn.linear(
            hidden,
            self.head,
            core_grid=self.head_grid,
            dtype=self.logits_dtype,
            compute_kernel_config=self.head_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = ttnn.unary_chain(
                logits,
                [
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / cap),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH, 0),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, cap),
                ],
            )
        return logits

    def capture_tap(self, x, start, end):
        """Copy of rows [start, end) of a layer output, as replicated [1,1,end-start,5376] DRAM.

        A tile-aligned window may end past the sequence, inside its tile padding: those rows hold the
        layer's (finite) output for zero input rows, not zeros, and consumers must mask them (the
        drafter's seed selects them to zero). The window is read from the tile-padded view, so no
        slice depends on the exact length. The next layer may consume x in place: always a private copy."""
        if end > x.shape[2]:
            tiled = ttnn.Shape([1, 1, x.padded_shape[2], x.shape[3]])
            x = ttnn.reshape(x, tiled, tiled)
        x = x[:, :, start:end, :] if (start, end) != (0, x.shape[2]) else ttnn.clone(x)
        return self.gather_hidden(x) if x.shape[-1] != self.config.hidden_size else x

    def prefill_device(
        self,
        tokens,
        *,
        sequence_length,
        page_tables,
        kv_cache,
        all_logits=False,
        start_pos=0,
        histories=None,
        taps=None,
        tap_window=None,
    ):
        """taps (speculative drafting only): {layer index: None} is filled with output rows
        [start, end) = tap_window (default: the last min(2048, rows)) of each named layer, before
        the next layer consumes them. See capture_tap for windows that end in tile padding."""
        if len(kv_cache) != len(self.layers):
            raise ValueError("Expected one cache pair per loaded layer")
        x = self.embed(tokens, sequence_length=sequence_length)
        for i, (layer, cache) in enumerate(zip(self.layers, kv_cache)):
            # This hidden stream is private to this invocation. Completed
            # chunks can reuse it once their input reads and KV fills finish.
            x = layer.prefill_forward(
                x,
                page_table=page_tables[i] if i in page_tables else page_tables[layer.kind],
                kv_cache=cache,
                consume_input=True,
                start_pos=start_pos,
                history=histories[i] if histories is not None else None,
                return_history=histories is not None,
            )
            if histories is not None:
                x, histories[i] = x
            if taps is not None and self.layer_indices[i] in taps:
                start, end = tap_window or (max(0, sequence_length - 2048), sequence_length)
                taps[self.layer_indices[i]] = self.capture_tap(x, start, end)
        if not all_logits:
            # An unaligned slice begin untilizes its entire input. Copy
            # the aligned final tile first, then select its logical row.
            tile_start = (sequence_length - 1) // 32 * 32
            x = x[:, :, tile_start:sequence_length, :]
            last_row = sequence_length - 1 - tile_start
            x = x[:, :, last_row : last_row + 1, :]
        return self.terminal(x)

    def decode_device(
        self,
        tokens,
        positions,
        *,
        page_tables,
        kv_cache,
        batch,
        rope_positions=None,
        taps=None,
        row_update_idxs=None,
        row_update_ring=None,
    ):
        """taps, row_update_idxs and row_update_ring serve speculative verify only (rows of one
        user, see Decoder.decode_forward): taps {layer index: None} is filled with DRAM copies of
        those layers' [1,1,batch,5376] outputs."""
        if len(kv_cache) != len(self.layers):
            raise ValueError("Expected one cache pair per loaded layer")
        x = self.embed(tokens, batch=batch)
        # Every layer of a kind reads the same table rows: when hoisted, the first
        # layer of each kind looks them up and later layers reuse the cached tensors.
        rope_tables = {}
        for i, (layer, cache) in enumerate(zip(self.layers, kv_cache)):
            x = layer.decode_forward(
                x,
                positions=positions,
                page_table=page_tables[i] if i in page_tables else page_tables[layer.kind],
                kv_cache=cache,
                rope_positions=rope_positions,
                cyclic_cache=i not in page_tables,
                rope_tables=rope_tables.setdefault(layer.kind, {}) if HOIST_DECODE_ROPE else None,
                row_update_idxs=row_update_idxs,
                row_update_ring=row_update_ring,
            )
            if taps is not None and self.layer_indices[i] in taps:
                taps[self.layer_indices[i]] = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        logits = self.terminal(x)
        # Common samplers own a 32-row tile; inactive padding stays internal.
        shape = ttnn.Shape([1, 1, 32, self.config.vocab_size // 4])
        return ttnn.reshape(logits, shape, shape)
