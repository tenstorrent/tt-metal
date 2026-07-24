# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import ttml

from ttml.models.llama.gqattn import GroupedQueryAttention
from ttml.models.llama.transformer import LlamaBlock
from ttml.models.llama import Llama
from ttml.modules import RunMode


class GroupedQueryAttentionCompositeKV(GroupedQueryAttention):
    def forward_no_kv(self, input: ttml.autograd.Tensor, mask: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        q = self.q_linear(input)
        kv = self.kv_linear(input)

        q_heads, k_heads, v_heads = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kv, self.num_heads, self.num_groups
        )

        q_heads = ttml.ops.rope.rope(q_heads, self.rope_params)
        k_heads = ttml.ops.rope.rope(k_heads, self.rope_params)

        # Composite SDPA supports non-broadcast masks like (B, 1, S, S)
        attention = ttml.ops.attention.scaled_dot_product_attention_composite(q_heads, k_heads, v_heads, mask)
        attention = ttml.ops.multi_head_utils.heads_fusion(attention)

        out = self.out_linear(attention)

        # Match base behavior in training mode
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)

        return out

    def forward_kv(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: ttml.models.KvCache,
        layer_idx: int,
        new_tokens: int,
    ) -> ttml.autograd.Tensor:
        q = self.q_linear(input)
        kv = self.kv_linear(input)

        q_heads, k_heads, v_heads = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kv, self.num_heads, self.num_groups
        )

        token_pos = kv_cache.get_cache_position()
        q_heads = ttml.ops.rope.rope(q_heads, self.rope_params, token_pos)
        k_heads = ttml.ops.rope.rope(k_heads, self.rope_params, token_pos)

        kv_cache.update(layer_idx, k_heads.get_value(), v_heads.get_value(), new_tokens)

        k_cache = kv_cache.get_k_cache(layer_idx)
        v_cache = kv_cache.get_v_cache(layer_idx)

        token_end = [k_cache.shape[0], k_cache.shape[1], mask.shape()[-1], k_cache.shape[3]]
        step = [1, 1, 1, 1]
        k_cache_slice = ttnn.slice(k_cache, [0, 0, 0, 0], token_end, step)
        v_cache_slice = ttnn.slice(v_cache, [0, 0, 0, 0], token_end, step)

        k_cache_to_process = ttml.autograd.create_tensor(k_cache_slice)
        v_cache_to_process = ttml.autograd.create_tensor(v_cache_slice)

        # Change: composite SDPA in decode path
        attention = ttml.ops.attention.scaled_dot_product_attention_composite(
            q_heads, k_cache_to_process, v_cache_to_process, mask
        )
        attention = ttml.ops.multi_head_utils.heads_fusion(attention)
        out = self.out_linear(attention)
        return out


class LlamaBlockCompositeKV(LlamaBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # preserve existing config values
        self.attention = GroupedQueryAttentionCompositeKV(
            embedding_size=self.attention.embedding_size,
            num_heads=self.attention.num_heads,
            num_groups=self.attention.num_groups,
            dropout=self.attention.dropout_prob,
            rope_params=self.attention.rope_params,
            bias_linears=False,  # match your model config if needed
        )


class LlamaCompositeKV(Llama):
    def __init__(self, config):
        super().__init__(config)
        self.create_name("Llama")

        # do NOT rebuild self.blocks; patch attention in-place
        for i, block in enumerate(self.blocks):
            old = block.attention
            block.attention = GroupedQueryAttentionCompositeKV(
                embedding_size=old.embedding_size,
                num_heads=old.num_heads,
                num_groups=old.num_groups,
                dropout=old.dropout_prob,
                rope_params=old.rope_params,
                bias_linears=config.attention_bias,
            )

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

        Most values are live handles into ttml's parameter store; do not mutate
        ttml's parameters between this call and ``update_weights``. The K/V split
        is the exception: ttml fuses K and V into one ``kv_linear/weight``
        (K rows first, then V), so we expose them via two ``ttnn.slice`` calls
        (newly allocated, ~64 MB total for Llama-3.2-1B-Instruct).

        Single-device assumption: parameters must be replicated across the mesh
        (no DDP/TP shard mapper). The grpo single-device config satisfies this;
        DDP/TP would need a host-side per-parameter concat first.
        """
        from ttml.models import WeightTyingType

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

        def t(name: str) -> ttnn.Tensor:
            if name not in params:
                raise RuntimeError(
                    f"ttml parameter {name!r} not found; available keys (first 10): " f"{sorted(params.keys())[:10]}"
                )
            return params[name].get_value()

        out: dict[str, ttnn.Tensor] = {}

        # Tied: same handle exposed under both HF keys.
        fc = t("Llama/fc/weight")
        out["model.embed_tokens.weight"] = fc
        out["lm_head.weight"] = fc
        out["model.norm.weight"] = t("Llama/ln_fc/gamma")

        for i in range(len(self.blocks)):
            p = f"Llama/blocks/{i}"

            out[f"model.layers.{i}.input_layernorm.weight"] = t(f"{p}/attention_norm/gamma")
            out[f"model.layers.{i}.post_attention_layernorm.weight"] = t(f"{p}/mlp_norm/gamma")

            out[f"model.layers.{i}.self_attn.q_proj.weight"] = t(f"{p}/attention/q_linear/weight")
            out[f"model.layers.{i}.self_attn.o_proj.weight"] = t(f"{p}/attention/out_linear/weight")

            kv = t(f"{p}/attention/kv_linear/weight")
            kv_shape = tuple(kv.shape)
            assert kv_shape == (1, 1, 2 * kv_dim, H), (
                f"kv_linear shape mismatch at layer {i}: got {kv_shape}, " f"expected (1, 1, {2 * kv_dim}, {H})"
            )
            out[f"model.layers.{i}.self_attn.k_proj.weight"] = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, kv_dim, H])
            out[f"model.layers.{i}.self_attn.v_proj.weight"] = ttnn.slice(kv, [0, 0, kv_dim, 0], [1, 1, 2 * kv_dim, H])

            out[f"model.layers.{i}.mlp.gate_proj.weight"] = t(f"{p}/mlp/w1/weight")
            out[f"model.layers.{i}.mlp.up_proj.weight"] = t(f"{p}/mlp/w3/weight")
            out[f"model.layers.{i}.mlp.down_proj.weight"] = t(f"{p}/mlp/w2/weight")

        return out
