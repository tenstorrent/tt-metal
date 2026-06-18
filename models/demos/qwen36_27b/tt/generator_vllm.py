# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for Qwen3.6-27B (HF arch ``Qwen3_5ForConditionalGeneration``,
model_type ``qwen3_5``) on the Tenstorrent backend.

This is the DENSE counterpart of the coder_next ``Qwen3NextForCausalLM`` adapter:
the model is fully dense (gated MLP, no MoE/experts) and TENSOR-PARALLEL across
TP=8 devices on a 1x8 line mesh (FABRIC_1D, Topology.Linear).

Register in ``vllm/platforms/tt.py::register_tt_models`` with:
    ModelRegistry.register_model(
        "TTQwen3_5ForConditionalGeneration",
        "models.demos.qwen36_27b.tt.generator_vllm:Qwen3_5ForConditionalGeneration")

Hybrid-state handling matches the template: GQA (full_attention) layers use a
contiguous per-request-row KV cache (written at prefill, read at decode); DeltaNet
(linear_attention) layers keep a persistent recurrent + conv STATE indexed by the
runner's stable per-request row (keyed by first KV block id). The paged pool from
``allocate_kv_cache`` is intentionally minimal — the model self-manages caches.
"""

import ttnn

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights_vllm import StreamingStateDict
from models.demos.qwen36_27b.tt.model_vllm import TtQwen36VllmModel


def _hf_config_to_cfg(hf_config, max_seq_len, n_layers=None):
    """Map the HF Qwen3.6-27B config (text_config sub-config) -> Qwen36ModelConfig.

    The HF wrapper config nests the text backbone under ``text_config``; pull
    dims from there. Defaults already match the real model, so only copy what is
    explicitly present.
    """
    cfg = Qwen36ModelConfig()
    tc = getattr(hf_config, "text_config", None) or hf_config

    def _g(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    for attr, hf_key in [
        ("hidden_size", "hidden_size"),
        ("num_hidden_layers", "num_hidden_layers"),
        ("vocab_size", "vocab_size"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_key_value_heads"),
        ("head_dim", "head_dim"),
        ("full_attention_interval", "full_attention_interval"),
        ("intermediate_size", "intermediate_size"),
        ("linear_num_key_heads", "linear_num_key_heads"),
        ("linear_num_value_heads", "linear_num_value_heads"),
        ("linear_key_head_dim", "linear_key_head_dim"),
        ("linear_value_head_dim", "linear_value_head_dim"),
        ("linear_conv_kernel_dim", "linear_conv_kernel_dim"),
        ("partial_rotary_factor", "partial_rotary_factor"),
        ("rms_norm_eps", "rms_norm_eps"),
    ]:
        v = _g(tc, hf_key, None)
        if v is not None:
            setattr(cfg, attr, v)

    # rope_theta lives under text_config.rope_parameters.rope_theta on this config.
    rp = _g(tc, "rope_parameters", None) or _g(tc, "rope_scaling", None)
    theta = _g(rp, "rope_theta", None) if rp is not None else None
    if theta is None:
        theta = _g(tc, "rope_theta", None)
    if theta is not None:
        cfg.rope_theta = float(theta)

    if n_layers:
        cfg.num_hidden_layers = n_layers
    if max_seq_len:
        cfg.max_seq_len = max_seq_len

    # Precision / execution: dense weights bf8 (TP frees DRAM), HiFi4 fp32-accum
    # matmuls, on-device contiguous attention for trace-friendly decode.
    # QWEN36_W_BF16=1 forces bf16 dense weights — used to separate bf8 quantization
    # decay from a real systematic per-layer bug in the PCC-vs-HF oracle sweep.
    import os as _os
    _wdt = ttnn.bfloat16 if _os.environ.get("QWEN36_W_BF16") else ttnn.bfloat8_b
    cfg.weights_dtype = _wdt
    cfg.dense_dtype = _wdt
    cfg.math_fidelity = "HiFi4"
    cfg.ondevice_attn = True
    return cfg


class Qwen3_5ForConditionalGeneration:
    # DeltaNet recurrent state is not prefix-shareable across requests.
    model_capabilities = {
        "supports_prefix_caching": False,
    }

    def __init__(self, model, cfg, mesh_device, max_batch_size):
        self.model = model
        self.cfg = cfg
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        model._vllm_max_batch = max_batch_size
        self._row_of_block = {}
        self._free_rows = list(range(max_batch_size))
        self._decode_inv = {}
        self._decode_B = 0

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations="performance",
    ):
        cfg = _hf_config_to_cfg(hf_config, max_seq_len, n_layers)
        ckpt_path = getattr(hf_config, "_name_or_path", None) or "/home/yito/work/qwen36_27b_hf"
        sd = StreamingStateDict(ckpt_path, config=cfg)
        model = TtQwen36VllmModel(mesh_device, sd, cfg, dense_tp=True, tp_size=8)
        return cls(model, cfg, mesh_device, max_batch_size)

    @property
    def cache_path(self):
        return "/tmp/ttnn_qwen36_cache"

    # ---- cache allocation (GQA paged pool kept minimal; model self-manages) ----
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        import torch
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        shp = list(kv_cache_shape)
        shp[0] = min(int(shp[0]), 8)
        zeros = torch.zeros(tuple(shp), dtype=torch.bfloat16)
        kv = []
        for _ in range(num_layers):
            k = ttnn.from_torch(zeros, device=self.mesh_device, mesh_mapper=rep,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            v = ttnn.from_torch(zeros.clone(), device=self.mesh_device, mesh_mapper=rep,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            kv.append((k, v))
        self.kv_caches = kv
        return kv

    # ---- persistent DeltaNet state + per-request row mapping ----
    def _persistent_state(self):
        st = getattr(self, "_decode_state", None)
        if st is None:
            self._decode_state = self.model.create_deltanet_state(batch=self.max_batch_size)
        return self._decode_state

    def _assign_row(self, block_id):
        if block_id in self._row_of_block:
            return self._row_of_block[block_id]
        row = self._free_rows.pop(0) if self._free_rows else 0
        self._row_of_block[block_id] = row
        return row

    def _reclaim_rows(self, live_blocks):
        for b in [b for b in self._row_of_block if b not in live_blocks]:
            self._free_rows.append(self._row_of_block.pop(b))
        self._free_rows = sorted(set(self._free_rows))

    def _write_persistent_row(self, persistent, row, temp):
        for li in list(temp.recurrent_states.keys()):
            rt = temp.recurrent_states[li]
            ct = temp.conv_states[li]
            persistent.write_recurrent_slot(li, row, rt)
            persistent.write_conv_slot(li, row, ct)
            # Only free the temp buffers if write_*_slot COPIED them (the concat
            # path, max_batch>1). When max_batch==1 _scatter_row returns the temp
            # tensor itself, so it now lives in `persistent`; deallocating it would
            # leave the persistent decode state pointing at freed memory.
            if persistent.recurrent_states[li] is not rt:
                try:
                    ttnn.deallocate(rt)
                except Exception:
                    pass
            if persistent.conv_states[li] is not ct:
                try:
                    ttnn.deallocate(ct)
                except Exception:
                    pass
            # This request wrote a fresh prefill conv_state; drop any stale on-device
            # decode conv history so the first decode step reseeds from it.
            old_hist = persistent.conv_hist.pop(li, None)
            if old_hist:
                for t in old_hist:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass

    # ---- prefill / decode glue ----
    def prefill_forward(self, tokens, page_table=None, kv_cache=None, start_pos=None,
                        prompt_lens=None, enable_trace=False, **kwargs):
        import torch
        from models.demos.qwen36_27b.tt.mesh_utils import to_torch
        tok = tokens if isinstance(tokens, torch.Tensor) else torch.as_tensor(tokens)
        tok = tok.to(torch.long)
        if tok.dim() == 1:
            tok = tok.unsqueeze(0)
        Bn = tok.shape[0]
        plens = prompt_lens if prompt_lens is not None else [tok.shape[1]] * Bn
        plens = [int(x) for x in (plens.tolist() if isinstance(plens, torch.Tensor) else plens)]
        sp = ([0] * Bn) if start_pos is None else [int(x) for x in (
            start_pos.tolist() if isinstance(start_pos, torch.Tensor) else start_pos)]
        pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        persistent = self._persistent_state()

        out_logits = []
        for r in range(Bn):
            L = plens[r]
            prompt = tok[r : r + 1, :L]
            positions = torch.arange(sp[r], sp[r] + L, dtype=torch.long)
            block_id = int(pt[r, 0].item())
            row = self._assign_row(block_id)
            pt_r = ttnn.from_torch(pt[r : r + 1].to(torch.int32), device=self.mesh_device, mesh_mapper=rep)
            logits, temp = self.model.forward_vllm_prefill(prompt, pt_r, positions, kv_cache, batch_idx=row)
            self._write_persistent_row(persistent, row, temp)
            out_logits.append(to_torch(logits).float().reshape(-1)[: self.cfg.vocab_size])
        return torch.stack(out_logits, dim=0).unsqueeze(1)  # [B,1,vocab]

    def decode_forward(self, tokens, page_table=None, kv_cache=None, start_pos=None,
                       enable_trace=False, read_from_device=True, **kwargs):
        import torch
        tok = tokens if isinstance(tokens, torch.Tensor) else torch.as_tensor(tokens)
        tok = tok.to(torch.long).reshape(-1, 1)
        B = tok.shape[0]
        sp = start_pos if isinstance(start_pos, torch.Tensor) else torch.as_tensor(start_pos)
        positions = sp.to(torch.long).reshape(-1)
        pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        nb = pt.shape[1]

        blocks = [int(pt[r, 0].item()) for r in range(B)]
        real = [r for r in range(B) if int(positions[r].item()) > 0]
        self._reclaim_rows({blocks[r] for r in real})

        maxb = self.max_batch_size
        tok_p = torch.zeros(maxb, 1, dtype=torch.long)
        pos_p = torch.zeros(maxb, dtype=torch.long)
        pt_p = torch.full((maxb, nb), -1, dtype=torch.int32)
        inv = {}
        for r in real:
            pr = self._assign_row(blocks[r])
            tok_p[pr] = tok[r]
            pos_p[pr] = positions[r]
            pt_p[pr] = pt[r]
            inv[pr] = r
        self._decode_inv = inv
        self._decode_B = B

        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        cur_pos = ttnn.from_torch(pos_p.to(torch.int32), device=self.mesh_device, mesh_mapper=rep)
        pt_tt = ttnn.from_torch(pt_p, device=self.mesh_device, mesh_mapper=rep)
        state = self._persistent_state()
        logits = self.model.forward_vllm_decode(tok_p, pt_tt, cur_pos, pos_p, kv_cache, state,
                                                enable_trace=enable_trace)
        return logits

    # ---- output plumbing ----
    def read_decode_output(self, tt_out, async_read=False):
        import torch
        from models.demos.qwen36_27b.tt.mesh_utils import to_torch
        host = tt_out.float() if isinstance(tt_out, torch.Tensor) else to_torch(tt_out).float()
        return (host, []) if async_read else host

    def process_decode_output_host(self, tt_out, is_tokens=False):
        import torch
        h = tt_out[0] if isinstance(tt_out, tuple) else tt_out
        h = h if isinstance(h, torch.Tensor) else torch.as_tensor(h)
        V = self.cfg.vocab_size
        hp = h.reshape(h.shape[-2], -1)[:, :V]
        out = torch.zeros(self._decode_B, 1, V, dtype=hp.dtype)
        for pr, r in self._decode_inv.items():
            out[r, 0] = hp[pr]
        return out

    def warmup_model_prefill(self, *args, **kwargs):
        return None

    def warmup_model_decode(self, *args, **kwargs):
        return None


# Alias requested by the porting spec.
Qwen36ForCausalLM = Qwen3_5ForConditionalGeneration
