# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full Qwen3.6-27B (DENSE) model in TT-NN — vLLM + tensor-parallel (TP=8).

Adapted from the coder_next template `model.py`, with the MoE/EP construction
dropped (this model is fully dense). The whole stack is TP-sharded on a 1xTP line
mesh; the lm_head is vocab(column)-parallel and all_gathered to full vocab.
"""

import torch
import ttnn
from models.demos.qwen36_27b.tt.mesh_utils import to_torch as mesh_to_torch
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen36_27b.tt.decoder_vllm import TtHybridDecoderLayer
from models.demos.qwen36_27b.tt.deltanet_vllm import TtDeltaNetState
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig


class TtQwen36VllmModel(LightweightModule):
    def __init__(self, device, state_dict, config: Qwen36ModelConfig, dtype=ttnn.bfloat16,
                 dense_tp=True, tp_size=8):
        super().__init__()
        self.device = device
        self.config = config
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.dense_tp = dense_tp
        self.tp_size = tp_size
        config.dense_tp = dense_tp
        config.tp_size = tp_size

        embed_w = state_dict["model.embed_tokens.weight"]
        self.embedding_weight = embed_w  # keep on CPU for lookup

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(TtHybridDecoderLayer(device, state_dict, i, config, dtype=dtype))

        TILE = 32
        norm_w = state_dict["model.norm.weight"]
        dim = norm_w.shape[0]
        torch_norm_w = (norm_w + 1.0).unsqueeze(0).view(1, 1, dim).reshape(1, 1, dim // TILE, TILE)
        norm_mapper = ttnn.ReplicateTensorToMesh(device) if dense_tp else None
        self.final_norm_weight = ttnn.from_torch(
            torch_norm_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=norm_mapper,
        )

        dense_dtype = config.get_dense_dtype(config.weights_dtype)
        lm_head_w = state_dict["lm_head.weight"].T.contiguous()  # [hidden, vocab]
        if self.dense_tp:
            vocab = lm_head_w.shape[1]
            pad_mult = self.tp_size * TILE
            self.vocab_padded = ((vocab + pad_mult - 1) // pad_mult) * pad_mult
            if self.vocab_padded != vocab:
                lm_head_w = torch.nn.functional.pad(lm_head_w, (0, self.vocab_padded - vocab))
            self.lm_head_w = ttnn.from_torch(
                lm_head_w.unsqueeze(0).unsqueeze(0),
                dtype=dense_dtype, layout=ttnn.TILE_LAYOUT, device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
            )
        else:
            self.vocab_padded = config.padded_vocab_size
            self.lm_head_w = ttnn.from_torch(
                lm_head_w.unsqueeze(0).unsqueeze(0),
                dtype=config.weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
            )

        self._build_rope_cache(config)

    def _build_rope_cache(self, config):
        dim = config.rotary_dim
        max_seq = config.max_seq_len
        theta = config.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, freqs)
        self.cos_cache = freqs.cos().reshape(1, 1, max_seq, dim // 2).repeat(1, 1, 1, 2)
        self.sin_cache = freqs.sin().reshape(1, 1, max_seq, dim // 2).repeat(1, 1, 1, 2)

    def get_rope(self, position_ids):
        if isinstance(position_ids, int):
            cos = self.cos_cache[:, :, position_ids:position_ids + 1, :]
            sin = self.sin_cache[:, :, position_ids:position_ids + 1, :]
        else:
            cos = self.cos_cache[:, :, position_ids, :]
            sin = self.sin_cache[:, :, position_ids, :]
        return cos, sin

    def embed(self, token_ids):
        """CPU embedding lookup -> device tensor [1,1,B*S,H] (replicated under TP)."""
        embeddings = self.embedding_weight[token_ids]  # [B, S, H]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        B, S, H = embeddings.shape
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
        return ttnn.from_torch(
            embeddings.reshape(1, 1, B * S, H),
            dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=mapper,
        )

    def rms_norm(self, x, weight, eps=1e-6):
        return ttnn.rms_norm(x, epsilon=eps, weight=weight)

    def _lm_head(self, hidden_states, pad_token_dim=False):
        """final norm + (vocab col-parallel) lm_head + all_gather to full vocab."""
        hidden_states = self.rms_norm(hidden_states, self.final_norm_weight)
        logits = ttnn.linear(hidden_states, self.lm_head_w, compute_kernel_config=self.config.matmul_kcfg())
        if self.dense_tp:
            if pad_token_dim:
                S = int(logits.shape[-2])
                Sp = ((S + 31) // 32) * 32
                if Sp != S:
                    logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, Sp - S), (0, 0)], value=0.0)
                logits = ttnn.all_gather(logits, dim=3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
                if Sp != S:
                    logits = ttnn.slice(logits, [0, 0, 0, 0], [int(logits.shape[0]), int(logits.shape[1]), S, int(logits.shape[-1])])
            else:
                logits = ttnn.all_gather(logits, dim=3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
        return logits

    # ---- vLLM decode (continuous batching, contiguous KV) ----
    def forward_vllm_decode(self, token_ids, page_table, cur_pos, positions, kv_caches, deltanet_state):
        """token_ids [B,1] CPU; cur_pos device int32 [B]; positions CPU int [B];
        deltanet_state persistent [max_batch,...]. Returns logits [1,1,B,vocab_padded]."""
        hidden_states = self.embed(token_ids)  # [1,1,B,H]
        B = token_ids.shape[0]
        rd = self.config.rotary_dim
        pos = positions if isinstance(positions, torch.Tensor) else torch.tensor(positions)
        cos = self.cos_cache[0, 0][pos].reshape(1, B, 1, rd)
        sin = self.sin_cache[0, 0][pos].reshape(1, B, 1, rd)
        rep = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
        cos_t = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin_t = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)

        import os as _os
        _dd = _os.environ.get("QWEN36_DUMP_DEC")  # dump decode-step per-layer acts (first call only)
        _dec = [] if (_dd and not getattr(self, "_dec_dumped", False)) else None
        if _dec is not None:
            _dec.append(("embed", mesh_to_torch(hidden_states).float().reshape(1, -1).clone()))

        for i, layer in enumerate(self.layers):
            lt = self.config.layer_types[i]
            if lt == "full_attention":
                hidden_states, _ = layer(
                    hidden_states, deltanet_state=deltanet_state, cos=cos_t, sin=sin_t,
                    mode="decode", position=cur_pos)
            else:
                hidden_states, _ = layer(hidden_states, deltanet_state=deltanet_state, mode="decode")
            if _dec is not None:
                _dec.append((f"layer{i}:{lt}", mesh_to_torch(hidden_states).float().reshape(1, -1).clone()))

        if _dec is not None:
            torch.save({"names": [n for n, _ in _dec],
                        "acts": torch.stack([a for _, a in _dec], dim=0)}, _dd)
            self._dec_dumped = True
            print(f"[val] dumped {len(_dec)} TT decode-step acts -> {_dd}", flush=True)

        return self._lm_head(hidden_states, pad_token_dim=False)

    def forward_vllm_prefill(self, token_ids, page_table, positions, kv_caches, batch_idx=0):
        """Prefill ONE request (prompt [1,L]). Fills the contiguous per-chip KV
        cache for full_attention layers at row `batch_idx` and runs the prompt
        recurrence in a fresh B=1 DeltaNet state.
        Returns (last-token logits [1,1,1,vocab_padded], temp DeltaNet state B=1)."""
        row = batch_idx
        temp = self.create_deltanet_state(batch=1)
        hidden_states = self.embed(token_ids)  # [1,1,L,H]
        L = token_ids.shape[1]
        pos = positions if isinstance(positions, torch.Tensor) else torch.tensor(positions)
        rd = self.config.rotary_dim
        cos = self.cos_cache[0, 0][pos].reshape(1, 1, L, rd)
        sin = self.sin_cache[0, 0][pos].reshape(1, 1, L, rd)

        import os as _os
        _dump = _os.environ.get("QWEN36_DUMP_ACTS")  # path to save per-layer acts for PCC vs HF oracle
        _acts = [] if _dump else None
        if _dump:
            _acts.append(("embed", mesh_to_torch(hidden_states).float().reshape(1, L, -1).clone()))

        for i, layer in enumerate(self.layers):
            lt = self.config.layer_types[i]
            if lt == "full_attention":
                hidden_states, _ = layer(
                    hidden_states, deltanet_state=temp, cos=cos, sin=sin, kv_cache=None,
                    mode="prefill", page_table=page_table, batch_idx=row,
                    cache_batch=getattr(self, "_vllm_max_batch", 1))
            else:
                hidden_states, _ = layer(hidden_states, deltanet_state=temp, mode="prefill")
            if _dump:
                _acts.append((f"layer{i}:{lt}", mesh_to_torch(hidden_states).float().reshape(1, L, -1).clone()))

        if _dump:
            # Also dump the per-layer final recurrent + conv state for the first few
            # DeltaNet layers, to compare against the HF oracle's per-layer states and
            # localize whether the bug is in the recurrence vs the output projection/norm.
            states = {}
            for li in [0, 1, 2, 4, 5, 6]:
                rs = temp.recurrent_states.get(li)
                cs = temp.conv_states.get(li)
                if rs is not None:
                    states[f"recur{li}"] = mesh_to_torch(rs).float().clone()  # [1,48,128,128]
                if cs is not None:
                    states[f"conv{li}"] = mesh_to_torch(cs).float().clone()   # [1,1,10240,32]
            torch.save({"L": L, "tokens": token_ids.reshape(-1).tolist(),
                        "names": [n for n, _ in _acts],
                        "acts": torch.stack([a for _, a in _acts], dim=0),
                        "states": states}, _dump)
            print(f"[val] dumped {len(_acts)} TT activation tensors -> {_dump}", flush=True)

        last = mesh_to_torch(hidden_states)[:, :, -1:, :]
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
        last_tt = ttnn.from_torch(last, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=mapper)
        logits = self._lm_head(last_tt, pad_token_dim=True)
        return logits, temp

    def create_deltanet_state(self, batch=1):
        return TtDeltaNetState(self.num_layers, self.config.layer_types, self.device, self.config, batch=batch)
