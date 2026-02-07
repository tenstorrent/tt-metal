# SPDX-FileCopyrightText: © 2026 Tenstorrent AI UL
import torch.nn as nn
import torch.nn.functional as F
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.linear import TTNNLinear


class TTNNGR00TSelfAttention(nn.Module, TTNNModule):
    def __init__(self, config=None, torch_layer=None):
        # Initialize both parents to ensure registry and hardware acceleration
        nn.Module.__init__(self)
        TTNNModule.__init__(self)

        if isinstance(torch_layer, TTNNGR00TSelfAttention):
            return

        self.num_heads = getattr(torch_layer, "num_heads", getattr(config, "num_attention_heads", 16))
        self.num_kv_heads = getattr(
            torch_layer, "num_key_value_heads", getattr(config, "num_key_value_heads", self.num_heads)
        )
        self.hidden_size = getattr(config, "hidden_size", 1152)

        self.tt_q_proj = self.tt_k_proj = self.tt_v_proj = self.tt_o_proj = None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        if torch_layer is not None:
            self._map_weights(torch_layer)

    def _map_weights(self, torch_layer):
        for name, m in torch_layer.named_children():
            lname = name.lower()
            if any(x in lname for x in ["q_proj", "query"]):
                self.tt_q_proj = TTNNLinear.from_torch(m)
            elif any(x in lname for x in ["k_proj", "key"]):
                self.tt_k_proj = TTNNLinear.from_torch(m)
            elif any(x in lname for x in ["v_proj", "value"]):
                self.tt_v_proj = TTNNLinear.from_torch(m)
            elif any(x in lname for x in ["o_proj", "out_proj"]):
                self.tt_o_proj = TTNNLinear.from_torch(m)

    def forward(self, hidden_states, *args, **kwargs):
        if self.tt_q_proj is None:
            return hidden_states, None
        hw_device = self.tt_q_proj.device

        # Projections
        q_w = self.tt_q_proj(hidden_states)
        k_w = self.tt_k_proj(hidden_states)
        v_w = self.tt_v_proj(hidden_states)

        def prepare(wrapped_t, heads):
            # FIX: Check if to_ttnn is a property/tensor or a function
            if hasattr(wrapped_t, "to_ttnn"):
                raw = wrapped_t.to_ttnn
                if callable(raw):
                    raw = raw()  # Only call if it's a function
            else:
                raw = wrapped_t

            t_cpu = ttnn.to_torch(ttnn.from_device(raw))
            b, s, h = t_cpu.shape
            d_head = h // heads
            t_torch = t_cpu.view(b, s, heads, d_head).permute(0, 2, 1, 3).contiguous()
            t_torch = F.pad(t_torch, (0, 24))
            pad_s = (32 - (s % 32)) % 32
            if pad_s > 0:
                t_torch = F.pad(t_torch, (0, 0, 0, pad_s))
            return (
                ttnn.from_torch(t_torch, device=hw_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                b,
                s,
                h,
                d_head,
                pad_s,
            )

        q_raw, b, s, h, d_head, pad_s = prepare(q_w, self.num_heads)
        k_raw, _, _, _, _, _ = prepare(k_w, self.num_kv_heads)
        v_raw, _, _, _, _, _ = prepare(v_w, self.num_kv_heads)

        attn_out_raw = ttnn.transformer.scaled_dot_product_attention(
            q_raw,
            k_raw,
            v_raw,
            attn_mask=None,
            is_causal=False,
            scale=float(d_head**-0.5),
            compute_kernel_config=self.compute_kernel_config,
        )

        out_torch = ttnn.to_torch(ttnn.from_device(attn_out_raw))
        out_merged = out_torch[:, :, :s, :d_head].permute(0, 2, 1, 3).contiguous().view(b, s, h)
        merged_dev = ttnn.from_torch(out_merged, device=hw_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        final_output = self.tt_o_proj(TorchTTNNTensor(merged_dev)) if self.tt_o_proj else TorchTTNNTensor(merged_dev)

        # FORCE SYNC TO CPU: Shield the backbone from hardware SIGFPE during residual add
        res_raw = final_output.to_ttnn
        if callable(res_raw):
            res_raw = res_raw()
        return TorchTTNNTensor(ttnn.to_torch(ttnn.from_device(res_raw))), None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
