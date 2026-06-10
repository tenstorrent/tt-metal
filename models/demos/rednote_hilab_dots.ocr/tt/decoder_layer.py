# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text decoder layer for dots.ocr.

Qwen2DecoderLayer (the dots.ocr text decoder): pre-norm residual block —
``x + attn(input_layernorm(x))`` then ``h + mlp(post_attention_layernorm(h))``
— composing the already brought-up sub-blocks TtTextRMSNorm (eps=1e-6),
TtTextAttention (fused per-chip QKV, 12 Q / 2 KV heads x head_dim 128, fp32
HF rope, fp32 explicit causal core) and TtTextMLP (SwiGLU 1536 -> 8960 ->
1536). Mirrors reference_impl models/tt_transformers/tt/decoder.py
(TransformerBlock): norm -> attention -> residual add -> norm -> mlp ->
residual add, DRAM interleaved throughout.

Numerics: the whole layer runs fp32 by default. The attention path REQUIRES
fp32 (dots.ocr text layer-0 logits reach ±3122 — Qwen2 attention sink; the
bf16 SDPA/bf16 explicit core measures PCC ~0.92, see tt/text_attention.py),
fp32 gammas use the TILE [1, 1, 1, dim] format (fp32 ROW_MAJOR gamma is
misread on device — tt/text_rmsnorm.py), and the MLP simply reuses the same
dtype so no mixed-dtype matmuls or typecasts enter the residual stream.
HiFi4/HiFi2 + fp32 accumulation are configured inside the sub-blocks.

Parallelism plan (ARCHITECTURE.md / inventory notes): decoder_stack
placement=shard — the sub-blocks own the TP layout (column-parallel QKV with
kv_replication=2, row-parallel o_proj/down_proj + sync all-reduce per
tp-guidance; async CCL is the optimization-phase swap). The residual stream
stays REPLICATED on the 1x4 mesh between the sub-blocks' all-reduced
outputs, so the replicated-path TtTextRMSNorm.forward applies and the
residual ttnn.add needs no CCL. On a single device everything degenerates
to the replicated full computation.
"""

import importlib.util
import sys
from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule

# Dir name contains a dot -> not importable as a package; load siblings by path.
_TT_DIR = Path(__file__).resolve().parent


def _load_sibling(stem):
    name = f"dots_ocr_tt_{stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _TT_DIR / f"{stem}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TtTextRMSNorm = _load_sibling("text_rmsnorm").TtTextRMSNorm
TtTextAttention = _load_sibling("text_attention").TtTextAttention
TtTextMLP = _load_sibling("text_mlp").TtTextMLP


class TtDecoderLayer(LightweightModule):
    """dots.ocr text decoder layer: x + attn(ln1(x)); h + mlp(ln2(h)).

    Args:
        mesh_device: ttnn mesh device handle (1xN line; sub-block weights
            TP-sharded per the parallelism plan).
        state_dict: torch tensors keyed input_layernorm.weight,
            self_attn.{q,k,v}_proj.weight/.bias, self_attn.o_proj.weight,
            post_attention_layernorm.weight, mlp.{gate,up,down}_proj.weight
            (HF keys model.layers.N.* with the layer prefix stripped).
        num_heads: Q heads (dots.ocr text: 12, head_dim 128).
        num_kv_heads: KV heads (dots.ocr text: 2, kv_replication=2).
        dtype: on-device weight/activation dtype (fp32 default — the
            attention path is fp32-mandatory, see module docstring).
        eps: RMSNorm epsilon (Qwen2 text decoder uses 1e-6).
    """

    def __init__(self, mesh_device, state_dict, num_heads=12, num_kv_heads=2, dtype=ttnn.float32, eps=1e-6):
        super().__init__()
        self.mesh_device = mesh_device

        self.input_layernorm = TtTextRMSNorm(
            mesh_device, {"weight": state_dict["input_layernorm.weight"]}, dtype=dtype, eps=eps
        )
        self.self_attn = TtTextAttention(
            mesh_device,
            {
                k: state_dict[f"self_attn.{k}"]
                for k in (
                    "q_proj.weight",
                    "q_proj.bias",
                    "k_proj.weight",
                    "k_proj.bias",
                    "v_proj.weight",
                    "v_proj.bias",
                    "o_proj.weight",
                )
            },
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
        )
        self.post_attention_layernorm = TtTextRMSNorm(
            mesh_device, {"weight": state_dict["post_attention_layernorm.weight"]}, dtype=dtype, eps=eps
        )
        self.mlp = TtTextMLP(
            mesh_device,
            {k: state_dict[f"mlp.{k}"] for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")},
            dtype=dtype,
        )

    # Host-side input prep delegates to the attention sub-block (rope tables
    # and the causal mask stay on host per ARCHITECTURE.md hybrid_notes).
    def prepare_rope(self, cos, sin):
        return self.self_attn.prepare_rope(cos, sin)

    def prepare_causal_mask(self, seq):
        return self.self_attn.prepare_causal_mask(seq)

    def forward(self, x_11SH: ttnn.Tensor, rot_mats, causal_mask: ttnn.Tensor) -> ttnn.Tensor:
        """x_11SH: [1, 1, seq, hidden] TILE_LAYOUT, replicated on the mesh.

        rot_mats: (cos, sin) from prepare_rope, [1, hpd, seq, head_dim] fp32.
        causal_mask: from prepare_causal_mask, [1, hpd, seq, seq] fp32.
        Returns: [1, 1, seq, hidden], replicated (both branch outputs are
        all-reduced inside the sub-blocks). The residual stream keeps the
        INPUT dtype so a chained caller controls compounding precision.
        """
        res_dtype = x_11SH.dtype
        attn_in = self.input_layernorm(x_11SH)
        attn_out = self.self_attn(attn_in, rot_mats, causal_mask)
        ttnn.deallocate(attn_in)
        # Residual adds pinned to L1: their consumers are the RMSNorms and the
        # next add — never a matmul — so the L1-interleaved-into-matmul stall
        # (vision tick-23 counter-example) cannot occur; same recipe as the
        # vision_block residual pin (tick-25, adds -50%). Block output stays
        # DRAM (block-output contract).
        h = ttnn.add(x_11SH, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=res_dtype)
        ttnn.deallocate(attn_out)

        ff_in = self.post_attention_layernorm(h)
        ff_out = self.mlp(ff_in)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h, ff_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=res_dtype)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out
