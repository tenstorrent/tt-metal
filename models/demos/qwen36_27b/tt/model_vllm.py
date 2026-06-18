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
    def _decode_host_inputs(self, token_ids, positions):
        """Host-side per-step decode inputs (computed OUTSIDE any trace): embedding
        lookup + RoPE gather. Returns torch tensors (emb [1,1,B,H], cos/sin [1,B,1,rd],
        cur_pos [B] int32)."""
        B = token_ids.shape[0]
        rd = self.config.rotary_dim
        pos = positions if isinstance(positions, torch.Tensor) else torch.as_tensor(positions)
        emb = self.embedding_weight[token_ids].reshape(1, 1, B, -1)
        cos = self.cos_cache[0, 0][pos].reshape(1, B, 1, rd)
        sin = self.sin_cache[0, 0][pos].reshape(1, B, 1, rd)
        return emb, cos, sin, pos.to(torch.int32).reshape(-1)

    def _mk(self, host, dtype, layout=ttnn.TILE_LAYOUT):
        rep = ttnn.ReplicateTensorToMesh(self.device) if self.dense_tp else None
        return ttnn.from_torch(host, dtype=dtype, layout=layout, device=self.device, mesh_mapper=rep)

    def _copy_into(self, buf, host, dtype, layout=ttnn.TILE_LAYOUT):
        tmp = self._mk(host, dtype, layout)
        ttnn.copy(tmp, buf)
        ttnn.deallocate(tmp)

    def _run_decode_layers(self, hidden, cos_t, sin_t, cur_pos, deltanet_state, dump=None):
        """Device-only decode: 64 layers (DeltaNet/GQA, in-place state) + lm_head.
        All inputs are device tensors; traceable. `dump` (eager only) collects per-layer
        host activations for the PCC harness."""
        if dump is not None:
            dump.append(("embed", mesh_to_torch(hidden).float().reshape(1, -1).clone()))
        for i, layer in enumerate(self.layers):
            lt = self.config.layer_types[i]
            if lt == "full_attention":
                hidden, _ = layer(hidden, deltanet_state=deltanet_state, cos=cos_t, sin=sin_t,
                                  mode="decode", position=cur_pos)
            else:
                hidden, _ = layer(hidden, deltanet_state=deltanet_state, mode="decode")
            if dump is not None:
                dump.append((f"layer{i}:{lt}", mesh_to_torch(hidden).float().reshape(1, -1).clone()))
        return self._lm_head(hidden, pad_token_dim=False)

    def _maybe_reseed_conv_hist(self, deltanet_state):
        """A new request's prefill (in-place state path) flags conv_hist stale; re-seed
        the fixed conv_hist buffers in-place from the new conv_state before this decode.
        Skipped for batched (num_seq>1): the sharded conv prep self-seeds per-batch."""
        if getattr(self.config, "batched_decode", False):
            return
        if getattr(deltanet_state, "_reseed_conv_hist", False):
            for i, layer in enumerate(self.layers):
                if self.config.layer_types[i] != "full_attention":
                    layer.token_mixer._seed_conv_hist(deltanet_state)
            deltanet_state._reseed_conv_hist = False

    def forward_vllm_decode(self, token_ids, page_table, cur_pos, positions, kv_caches,
                            deltanet_state, enable_trace=False):
        """token_ids [B,1] CPU; positions [B]; deltanet_state persistent [max_batch,...].
        Returns logits [1,1,B,vocab_padded]."""
        emb, cos, sin, cpos = self._decode_host_inputs(token_ids, positions)
        if enable_trace:
            return self._decode_traced(emb, cos, sin, cpos, deltanet_state)
        # eager
        self._maybe_reseed_conv_hist(deltanet_state)
        hidden = self._mk(emb, self.dtype)
        cos_t = self._mk(cos, ttnn.bfloat16)
        sin_t = self._mk(sin, ttnn.bfloat16)
        cur = self._mk(cpos, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        import os as _os
        _dd = _os.environ.get("QWEN36_DUMP_DEC")
        dump = [] if (_dd and not getattr(self, "_dec_dumped", False)) else None
        logits = self._run_decode_layers(hidden, cos_t, sin_t, cur, deltanet_state, dump=dump)
        # all layers consumed any per-row conv_hist reseed flags this step; clear them.
        if getattr(deltanet_state, "_reseed_rows", None):
            deltanet_state._reseed_rows = set()
        for t in (hidden, cos_t, sin_t, cur):
            ttnn.deallocate(t)
        if dump is not None:
            torch.save({"names": [n for n, _ in dump],
                        "acts": torch.stack([a for _, a in dump], dim=0)}, _dd)
            self._dec_dumped = True
            print(f"[val] dumped {len(dump)} TT decode-step acts -> {_dd}", flush=True)
        return logits

    def _decode_traced(self, emb, cos, sin, cpos, deltanet_state):
        """Captured-trace decode. The 64-layer+lm_head graph is captured once over
        FIXED input buffers and replayed each step (no per-op host dispatch). DeltaNet
        running state (recurrent + conv history) is updated IN-PLACE; the compile+capture
        runs advance it twice, so we snapshot the post-prefill state and restore it after
        capture. cur_pos/cos/sin/embed are refreshed into the fixed buffers each step."""
        dev = self.device
        B = int(emb.shape[2])
        batched = bool(getattr(self.config, "batched_decode", False))
        if getattr(self, "_dec_trace_id", None) is None:
            # seed conv history for all DeltaNet layers so the buffers exist + snapshot-able
            for i, layer in enumerate(self.layers):
                if self.config.layer_types[i] != "full_attention":
                    if batched:
                        layer.token_mixer._seed_conv_hist_sharded(deltanet_state, B)
                    else:
                        layer.token_mixer._seed_conv_hist(deltanet_state)
            deltanet_state.trace_mode = True  # in-place recurrent/conv-state writeback
            deltanet_state._reseed_conv_hist = False  # seeded fresh below
            deltanet_state._reseed_rows = set()  # (batched) seeded fresh; no host reseed in trace
            snap = deltanet_state.snapshot_decode_state()
            # fixed input buffers
            self._tr_hidden = self._mk(emb, self.dtype)
            self._tr_cos = self._mk(cos, ttnn.bfloat16)
            self._tr_sin = self._mk(sin, ttnn.bfloat16)
            self._tr_cur = self._mk(cpos, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            # compile run (eager) to JIT-compile the kernels
            ttnn.deallocate(self._run_decode_layers(self._tr_hidden, self._tr_cos, self._tr_sin, self._tr_cur, deltanet_state))
            # capture
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            self._tr_out = self._run_decode_layers(self._tr_hidden, self._tr_cos, self._tr_sin, self._tr_cur, deltanet_state)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            self._dec_trace_id = tid
            deltanet_state.restore_decode_state(snap)  # undo the 2 setup-run advances
        else:
            self._maybe_reseed_conv_hist(deltanet_state)  # in-place reseed for a new request
            self._copy_into(self._tr_hidden, emb, self.dtype)
            self._copy_into(self._tr_cos, cos, ttnn.bfloat16)
            self._copy_into(self._tr_sin, sin, ttnn.bfloat16)
            self._copy_into(self._tr_cur, cpos, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.execute_trace(dev, self._dec_trace_id, cq_id=0, blocking=False)
        return self._tr_out

    def forward_vllm_prefill(self, token_ids, page_table, positions, kv_caches, batch_idx=0):
        """Prefill ONE request (prompt [1,L]). Fills the contiguous per-chip KV
        cache for full_attention layers at row `batch_idx` and runs the prompt
        recurrence in a fresh B=1 DeltaNet state.
        Returns (last-token logits [1,1,1,vocab_padded], temp DeltaNet state B=1)."""
        row = batch_idx
        temp = self.create_deltanet_state(batch=1, batched=False)  # prefill is GATHERED
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

    def create_deltanet_state(self, batch=1, batched=None):
        return TtDeltaNetState(self.num_layers, self.config.layer_types, self.device, self.config,
                               batch=batch, batched=batched)
