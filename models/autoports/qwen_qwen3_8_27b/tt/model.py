# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 text stack over the accepted TP4 decoder (no vision or host compute)."""
from pathlib import Path
import json
import torch
import ttnn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from .multichip_decoder import MultichipDecoder
from .precision import load_precision, decoder_policy


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / 'model.safetensors.index.json').read_text())['weight_map']

    def read(self, keys):
        result = {}
        for filename in sorted({self.index[k] for k in keys}):
            with safe_open(self.path / filename, framework='pt') as f:
                for key in keys:
                    if self.index[key] == filename:
                        result[key] = f.get_tensor(key)
        return result

    def tensor(self, key):
        return self.read([key])[key]


class Model:
    def __init__(self, checkpoint, mesh_device, *, layer_indices=None, precision_config=None):
        self.mesh = mesh_device
        self.precision_config = load_precision(precision_config)
        self.checkpoint = Checkpoint(checkpoint)
        self.config = AutoConfig.from_pretrained(checkpoint).text_config
        if self.config.max_position_embeddings != self.precision_config['max_context']:
            raise ValueError('Checkpoint context differs from the precision capability contract')
        self.layer_indices = list(range(self.config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        self.layers = []
        for i in self.layer_indices:
            prefix = f'model.language_model.layers.{i}.'
            weights = self.checkpoint.read([k for k in self.checkpoint.index if k.startswith(prefix)])
            print(f'LOAD_LAYER {i}', flush=True)
            self.layers.append(MultichipDecoder.from_state_dict(weights, hf_config=self.config,
                layer_idx=i, mesh_device=mesh_device,
                policy=decoder_policy(self.precision_config, i),
                ccl_workspace=self.layers[0].ar_workspace if self.layers else None))
        self.embedding = self.put(self.checkpoint.tensor('model.language_model.embed_tokens.weight'), shard=1,
                                  layout=ttnn.ROW_MAJOR_LAYOUT)
        head_key = 'model.language_model.embed_tokens.weight' if self.config.tie_word_embeddings else 'lm_head.weight'
        self.head = self.put(self.checkpoint.tensor(head_key).T.contiguous(), shard=1,
                             dtype=getattr(ttnn, self.precision_config['weight_groups']['head']))
        self.head_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.precision_config['compute_fidelities']['head']),
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        self.final_norm = self.put(1 + self.checkpoint.tensor('model.language_model.norm.weight').float())
        # Precomputed text-only MRoPE: the three position axes are identical.
        rope = Qwen3_5TextRotaryEmbedding(self.config)
        with torch.no_grad():
            c, s = rope(torch.zeros(1, 1, self.config.hidden_size),
                        torch.arange(self.config.max_position_embeddings).reshape(1, 1, -1).expand(3, 1, -1))
        self.cos_table, self.sin_table = (self.put(t[0], layout=ttnn.ROW_MAJOR_LAYOUT) for t in (c, s))

    def put(self, value, *, shard=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(value, device=self.mesh, dtype=dtype, layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh) if shard is None else ttnn.ShardTensorToMesh(self.mesh, dim=shard))

    def allocate_cache(self, *, batch_size, context_length):
        return [m.allocate_state(batch_size=batch_size, context_length=context_length) for m in self.layers]

    def embed(self, tokens, *, batch, length, decode=False):
        x = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, [batch, length, 1280])
        if decode and self.layers[0].decode_residual_layout(batch) == 'replicated':
            x = self.layers[0]._gather(x, ('embedding',))
        return x

    def rope(self, positions, *, batch, length):
        return tuple(ttnn.reshape(ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT),
                                  [batch, 1, length, 64]) for table in (self.cos_table, self.sin_table))

    def terminal(self, hidden, *, pack_decode_rows=False):
        packed = pack_decode_rows and len(hidden.shape)==3 and hidden.shape[0]>1 and hidden.shape[1]==1
        if hidden.shape[-1] == 1280:
            # Reuse the accepted gather's compact output instead of unpacking
            # every decode user into a separately padded matmul tile.
            hidden = self.layers[0]._gather(hidden, ('terminal',),compact_output=packed)
        elif packed:
            hidden=ttnn.reshape(hidden,[1,hidden.shape[0],hidden.shape[-1]])
        hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.rms_norm(hidden, weight=self.final_norm, epsilon=self.config.rms_norm_eps,
                              compute_kernel_config=self.layers[0].ckc)
        return ttnn.linear(hidden, self.head, compute_kernel_config=self.head_ckc, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def prefill_forward(self, tokens, *, positions, kv_cache, page_table, batch, length, start_position=0, logits_to_keep=None,
                        prepared_metadata=None, warm_layer_indices=None, return_hidden=False):
        hidden = self.embed(tokens, batch=batch, length=length)
        cos, sin = self.rope(positions, batch=batch, length=length)
        for index, (layer, state) in enumerate(zip(self.layers, kv_cache)):
            if warm_layer_indices is not None and index not in warm_layer_indices:
                continue
            hidden = layer.prefill_forward(hidden, state=state, page_table=page_table,
                cos=cos, sin=sin, start_position=start_position, prepared_metadata=prepared_metadata)
        if return_hidden:
            return hidden
        if logits_to_keep == 0:
            return None
        if logits_to_keep is not None:
            if prepared_metadata is None:
                hidden = hidden[:, -logits_to_keep:, :]
            else:
                hidden = self.select_rows(hidden, prepared_metadata['last_indices'])
        return self.terminal(hidden)

    def select_rows(self, hidden, indices):
        batch, length, width = hidden.shape
        table = ttnn.reshape(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), [batch * length, width])
        return ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT)

    def decode_forward(self, tokens, *, positions, rope_indices, kv_cache, page_table, batch, active_mask=None):
        hidden = self.embed(tokens, batch=batch, length=1, decode=True)
        cos, sin = self.rope(rope_indices, batch=batch, length=1)
        for layer, state in zip(self.layers, kv_cache):
            hidden = layer.decode_forward(hidden, state=state, page_table=page_table,
                                          current_pos=positions, cos=cos, sin=sin, active_mask=active_mask)
        logits = self.terminal(hidden,pack_decode_rows=True)
        return ttnn.reshape(logits, [1, 1, batch, self.config.vocab_size // 4])
