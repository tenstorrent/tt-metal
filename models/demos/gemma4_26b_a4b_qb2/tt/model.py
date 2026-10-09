# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma4 text stack using the accepted TP4/EP4 decoder policy."""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.demos.gemma4.tt.model import _get_lm_head_program_config
from models.demos.gemma4_26b_a4b_qb2.tt.multichip_decoder import CollectiveBufferPool, MultichipDecoder
from models.demos.gemma4_26b_a4b_qb2.tt.precision_ops import rms_norm
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import assert_precision_matches
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import dtype as policy_dtype
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import dtype_name
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import fidelity as policy_fidelity
from models.demos.gemma4_26b_a4b_qb2.tt.precision_policy import (
    fidelity_name,
    layer_precision_config,
    resolve_precision_config,
)

MODEL_ID = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
PAGE_SIZE = 32


class Checkpoint:
    def __init__(self):
        self.index = json.loads(
            Path(hf_hub_download(MODEL_ID, "model.safetensors.index.json", revision=REVISION)).read_text()
        )["weight_map"]

    def load(self, prefix):
        state = {}
        for shard in sorted({v for k, v in self.index.items() if k.startswith(prefix)}):
            with safe_open(hf_hub_download(MODEL_ID, shard, revision=REVISION), framework="pt", device="cpu") as file:
                for key in file.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix) :]] = file.get_tensor(key)
        return state


class Gemma4Model:
    def __init__(self, mesh_device, *, max_seq_len=None, layer_indices=None, precision_config=None):
        self.mesh = mesh_device
        self.precision_config = resolve_precision_config(precision_config)
        self.precision_observations = {}
        self.config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION).text_config
        if any(int(index) >= self.config.num_hidden_layers for index in self.precision_config["layer_overrides"]):
            raise ValueError("Precision layer override index exceeds model layer count")
        self.max_seq_len = max_seq_len or self.config.max_position_embeddings
        if not 1 <= self.max_seq_len <= self.config.max_position_embeddings:
            raise ValueError("Context exceeds the HF contract")
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("The accepted model requires the 1x4 mesh")
        self.layer_indices = (
            tuple(range(self.config.num_hidden_layers)) if layer_indices is None else tuple(layer_indices)
        )
        self.reduced_probe = len(self.layer_indices) != self.config.num_hidden_layers
        self.pool = CollectiveBufferPool(mesh_device)
        checkpoint = Checkpoint()
        self.layers = []
        self.router_buffers = None
        for index in self.layer_indices:
            state = checkpoint.load(f"model.language_model.layers.{index}.")
            layer = MultichipDecoder.from_state_dict(
                state,
                hf_config=self.config,
                layer_idx=index,
                mesh_device=mesh_device,
                collective_buffer_pool=self.pool,
                precision_config=layer_precision_config(self.precision_config, index, self.config.layer_types[index]),
            )
            router = layer.layer.moe.router
            names = ("bias", "indices", "output", "output_indices")
            if self.router_buffers is None:
                self.router_buffers = {name: getattr(router, name) for name in names}
            else:
                # The gate copies its results to interleaved storage before
                # expert consumers run; serial layers may reuse this scratch.
                for name in names:
                    setattr(router, name, self.router_buffers[name])
            self.layers.append(layer)
            del state
            print(f"LAYER_READY {index}", flush=True)
        embedding = checkpoint.load("model.language_model.embed_tokens.")["weight"]
        self.embedding = self.upload(embedding[None, None], layout=ttnn.ROW_MAJOR_LAYOUT, shard=-1)
        self.head = self.upload(
            embedding.T.contiguous()[None, None],
            dtype=policy_dtype(self.precision_config["model"]["head_weight_dtype"]),
            shard=-1,
        )
        del embedding
        norm = checkpoint.load("model.language_model.norm.")["weight"]
        self.norm = self.upload(norm.float().reshape(1, 1, 1, -1), dtype=ttnn.float32)
        self.embed_scale = float(torch.tensor(self.config.hidden_size**0.5, dtype=torch.bfloat16))
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=policy_fidelity(self.precision_config["model"]["head_fidelity"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.head_program = _get_lm_head_program_config(
            mesh_device, 1, self.config.hidden_size, self.config.vocab_size // 4
        )
        # The 22-tile TP4 hidden shard requires a one-tile K block for the
        # validated BFP4/LoFi decode head.
        self.head_program.in0_block_w = 1
        self.rope_prefill, self.rope_decode = {}, {}
        extent = (self.max_seq_len + 1023) // 1024 * 1024
        rotary = Gemma4TextRotaryEmbedding(self.config)
        for kind in set(self.config.layer_types[i] for i in self.layer_indices):
            cos, sin = rotary(torch.zeros(1, 1, self.config.hidden_size), torch.arange(extent)[None], layer_type=kind)
            self.rope_prefill[kind] = tuple(self.upload(x.unsqueeze(0)) for x in (cos, sin))
            self.rope_decode[kind] = tuple(self.upload(x.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for x in (cos, sin))
        self.zero_hidden = self.upload(torch.zeros(1, 1, 1, self.config.hidden_size))
        self.precision_summary()
        print("MODEL_READY", flush=True)

    def precision_summary(self):
        """Validate requested precision against allocated weights and bound kernels."""
        fixed = {
            "embedding_dtype": dtype_name(self.embedding.dtype),
            "final_norm_weight_dtype": dtype_name(self.norm.dtype),
            "logits_dtype": "bfloat16",
            "sampling_input_dtype": "bfloat16",
            "head_input_dtype": "bfloat16",
            "head_fp32_dest_acc_en": self.compute.fp32_dest_acc_en,
            "head_packer_l1_acc": self.compute.packer_l1_acc,
            "head_math_approx_mode": self.compute.math_approx_mode,
            "activation_dtype": "bfloat16",
            "residual_dtype": "bfloat16",
            "post_attention_residual_dtype": "float32",
            "page_size": PAGE_SIZE,
            "cache_read_alignment": 128,
            "prefill_expert_parallel": 4,
            "decode_tensor_parallel": self.layers[0].tp,
            "collective_topology": "Linear" if self.layers[0].topology == ttnn.Topology.Linear else "unsupported",
        }
        model = {
            "head_weight_dtype": dtype_name(self.head.dtype),
            "head_fidelity": fidelity_name(self.compute.math_fidelity),
            "fixed": fixed,
        }
        assert_precision_matches(model, self.precision_config["model"], "model")
        return {
            "config_id": self.precision_config["config_id"],
            "model": model,
            "layers": {str(index): layer.precision_summary() for index, layer in zip(self.layer_indices, self.layers)},
            "observed_execution_dtypes": dict(self.precision_observations),
            "construction_verified": True,
        }

    def upload(self, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
        return ttnn.from_torch(
            value,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.mesh) if shard is None else ttnn.ShardTensorToMesh(self.mesh, dim=shard)
            ),
        )

    def allocate_cache(self, *, slots, context):
        if not 1 <= slots <= 32 or not 1 <= context <= self.max_seq_len:
            raise ValueError("Invalid cache slots/context")
        # SDPA may read through the next 128-token window, beyond logical length.
        pages_per_slot = (context + 127) // 128 * 4
        table = torch.arange(slots * pages_per_slot, dtype=torch.int32).reshape(slots, pages_per_slot)
        caches = []
        for layer in self.layers:
            cfg = layer.layer.self_attn.config
            shape = (slots * pages_per_slot, cfg.num_key_value_heads, PAGE_SIZE, cfg.head_dim)
            pair = tuple(self.upload(torch.zeros(shape), dtype=layer.kv_cache_dtype) for _ in range(2))
            layer._validate_kv_cache(pair)
            caches.append(pair)
        self.precision_observations["cache_dtypes"] = [[dtype_name(t.dtype) for t in pair] for pair in caches]
        return caches, table

    def embed(self, tokens):
        value = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT)
        value = ttnn.reshape(value, (1, 1, -1, self.config.hidden_size // 4))
        value = self.layers[0].gather(value)
        value = ttnn.mul(value, self.embed_scale)
        if value.dtype != ttnn.bfloat16:
            raise ValueError("Embedding activation must be BF16")
        self.precision_observations["embedding_output"] = dtype_name(value.dtype)
        return value

    def logits(self, hidden):
        if hidden.dtype != ttnn.bfloat16:
            raise ValueError("Decoder output residual must be BF16")
        self.precision_observations["decoder_output"] = dtype_name(hidden.dtype)
        hidden = ttnn.typecast(rms_norm(hidden, self.config.rms_norm_eps, self.norm), ttnn.bfloat16)
        logits = ttnn.linear(
            hidden,
            self.head,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.head_program if hidden.shape[-2] <= 32 else None,
        )
        cap = self.config.final_logit_softcapping
        logits = ttnn.mul(ttnn.tanh(ttnn.mul(logits, 1.0 / cap)), cap) if cap else logits
        if logits.dtype != ttnn.bfloat16:
            raise ValueError("Logits and sampler input must be BF16")
        self.precision_observations["logits_output"] = dtype_name(logits.dtype)
        return logits

    def prefill_forward(self, tokens, *, page_table, kv_cache, user_id=0, return_all_logits=False):
        if len(kv_cache) != len(self.layers):
            raise ValueError("KV cache must contain one pair per model layer")
        hidden = self.embed(tokens)
        for layer_number, (layer, cache, index) in enumerate(zip(self.layers, kv_cache, self.layer_indices)):
            layer_table = page_table[layer_number] if isinstance(page_table, (tuple, list)) else page_table
            hidden = layer.prefill_forward(
                hidden,
                rope_mats=self.rope_prefill[self.config.layer_types[index]],
                page_table=layer_table,
                kv_cache=cache,
                user_id=user_id,
            )
        if not return_all_logits:
            hidden = hidden[:, :, -1:, :]
        return self.logits(hidden)

    def decode_forward(self, tokens, *, current_pos, cache_pos, page_table, kv_cache, batch, active_slots=None):
        # Sampler output is the persistent [1,1,1,32] token input. Only live slots
        # are embedded; the model trace fixes the slot count at request setup.
        if len(kv_cache) != len(self.layers):
            raise ValueError("KV cache must contain one pair per model layer")
        hidden = self.embed(tokens[..., :batch])
        active = tuple(range(batch)) if active_slots is None else tuple(active_slots)
        for layer_number, (layer, cache, index) in enumerate(zip(self.layers, kv_cache, self.layer_indices)):
            layer_table = page_table[layer_number] if isinstance(page_table, (tuple, list)) else page_table
            rope = self.rope_decode[self.config.layer_types[index]]
            if len(active) == batch:
                hidden = layer.decode_forward(
                    hidden,
                    rope_mats=rope,
                    current_pos=current_pos,
                    cache_pos=cache_pos,
                    page_table=layer_table,
                    kv_cache=cache,
                )
            else:
                rows = []
                for slot in range(batch):
                    if slot in active:
                        rows.append(
                            layer.decode_forward(
                                hidden[:, :, slot : slot + 1, :],
                                rope_mats=rope,
                                current_pos=current_pos[:, slot : slot + 1],
                                cache_pos=cache_pos[slot : slot + 1],
                                page_table=layer_table[slot : slot + 1, :],
                                kv_cache=cache,
                            )
                        )
                    else:
                        rows.append(self.zero_hidden)
                hidden = ttnn.concat(rows, dim=2)
        return self.logits(hidden)
