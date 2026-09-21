# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttml

from ttml.models import WeightTyingType
from ttml.models.llama import Llama


class LlamaCompositeKV(Llama):
    """Extends Llama for 2 reasons:
    1. Swaps every block's attention kernel for scaled_dot_product_attention_composite: the
       completer emits non-broadcast masks ((B, 1, S, S)) that only the composite kernel accepts,
       and it makes the decode path (KV-cache slice -> attention) actually work.
    2. Adds weights_ref_hf_dict() -- an HF-keyed export of live ttml ttnn.Tensor handles, shaped
       for tt-transformers' Transformer.update_weights(...). This is the ttml->ttt bridge format
       the remote-rollout weight sync depends on, and it doesn't exist upstream."""

    def __init__(self, config):
        super().__init__(config)
        self.create_name("Llama")

        for block in self.blocks:
            block.attention.sdpa = ttml.ops.attention.scaled_dot_product_attention_composite

    def weights_ref_hf_dict(self) -> dict[str, ttnn.Tensor]:
        """Export this ttml model's parameters as an HF-keyed dict of on-device
        ``ttnn.Tensor`` handles, shaped for tt-transformers'
        ``Transformer.update_weights(hf_state_dict, hf_rope=False)`` (HF
        safetensors dot-keys; HF shapes wrapped in two leading unit dims;
        bf16, TILE, DRAM-interleaved, replicated).

        Q/K row order: both ttml and TTT store Meta-permuted rows for
        Llama-3.2-1B, so the consumer uses ``hf_rope=False`` (no permutation).

        Tied embeddings: with ``weight_tying=Enabled``, ``embed_tokens`` and
        ``lm_head`` point at the same handle; safe because the consumer
        ``ttnn.copy``s into a separate destination and never aliases the source.

        Norms, o_proj and down_proj are live handles into ttml's parameter
        store; do not mutate ttml's parameters between this call and
        ``update_weights``. The fused projections are the exception: ttml
        stores Q, K and V as one ``qkv_linear/weight`` (rows Q, then K, then V)
        and gate/up as one ``w_gate_up/weight`` (rows gate, then up), so
        q/k/v_proj and gate/up_proj are ``ttnn.slice`` copies (newly allocated,
        ~1.2 GB in total for Llama-3.2-1B-Instruct; freed with the dict).

        Single-device assumption: parameters must be replicated across the mesh
        (no DDP/TP shard mapper). The grpo single-device config satisfies this;
        DDP/TP would need a host-side per-parameter concat first, and under TP
        the fused weights are per-rank interleaved, which plain row slices do
        not undo. The shape checks below reject a TP-sharded model.
        """
        cfg = self.config
        assert cfg.weight_tying == WeightTyingType.Enabled, (
            "weights_ref_hf_dict requires weight_tying=Enabled (Llama-3.2-1B/-Instruct "
            f"tie embed_tokens and lm_head). Got weight_tying={cfg.weight_tying!r}."
        )

        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        H = cfg.hidden_size
        head_dim = H // n_heads
        kv_dim = n_kv * head_dim

        params = self.parameters()

        def get(name: str) -> ttnn.Tensor:
            if name not in params:
                raise RuntimeError(
                    f"ttml parameter {name!r} not found; available keys (first 10): " f"{sorted(params.keys())[:10]}"
                )
            return params[name].get_value()

        def rows(weight: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
            return ttnn.slice(weight, [0, 0, start, 0], [1, 1, end, H])

        out: dict[str, ttnn.Tensor] = {}

        # Tied: same handle exposed under both HF keys.
        fc = get("Llama/fc/weight")
        out["model.embed_tokens.weight"] = fc
        out["lm_head.weight"] = fc
        out["model.norm.weight"] = get("Llama/ln_fc/gamma")

        for i in range(len(self.blocks)):
            p = f"Llama/blocks/{i}"

            out[f"model.layers.{i}.input_layernorm.weight"] = get(f"{p}/attention_norm/gamma")
            out[f"model.layers.{i}.post_attention_layernorm.weight"] = get(f"{p}/mlp_norm/gamma")

            qkv = get(f"{p}/attention/qkv_linear/weight")
            qkv_shape = tuple(qkv.shape)
            assert qkv_shape == (1, 1, H + 2 * kv_dim, H), (
                f"qkv_linear shape mismatch at layer {i}: got {qkv_shape}, " f"expected (1, 1, {H + 2 * kv_dim}, {H})"
            )
            out[f"model.layers.{i}.self_attn.q_proj.weight"] = rows(qkv, 0, H)
            out[f"model.layers.{i}.self_attn.k_proj.weight"] = rows(qkv, H, H + kv_dim)
            out[f"model.layers.{i}.self_attn.v_proj.weight"] = rows(qkv, H + kv_dim, H + 2 * kv_dim)
            out[f"model.layers.{i}.self_attn.o_proj.weight"] = get(f"{p}/attention/out_linear/weight")

            gate_up = get(f"{p}/mlp/w_gate_up/weight")
            gate_up_shape = tuple(gate_up.shape)
            assert gate_up_shape[:2] == (1, 1) and gate_up_shape[2] % 2 == 0 and gate_up_shape[3] == H, (
                f"w_gate_up shape mismatch at layer {i}: got {gate_up_shape}, " f"expected (1, 1, 2*I, {H})"
            )
            intermediate = gate_up_shape[2] // 2
            out[f"model.layers.{i}.mlp.gate_proj.weight"] = rows(gate_up, 0, intermediate)
            out[f"model.layers.{i}.mlp.up_proj.weight"] = rows(gate_up, intermediate, 2 * intermediate)
            out[f"model.layers.{i}.mlp.down_proj.weight"] = get(f"{p}/mlp/w2/weight")

        return out
