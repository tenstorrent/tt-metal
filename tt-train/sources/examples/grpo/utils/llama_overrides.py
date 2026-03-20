# e.g. examples/grpo/llama_decode_override.py
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
