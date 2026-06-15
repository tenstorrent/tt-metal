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

    def export_to_hf_dict(self) -> dict[str, ttnn.Tensor]:
        """Export this ttml model's parameters as an HF-keyed dict of on-device
        ``ttnn.Tensor`` handles.

        Output is shaped for direct consumption by tt-transformers'
        ``Transformer.update_weights(hf_state_dict, hf_rope=False)``:

        * keys   -- HF safetensors dot-keys
                    (``model.embed_tokens.weight``, ``lm_head.weight``,
                    ``model.norm.weight``,
                    ``model.layers.{i}.input_layernorm.weight``,
                    ``model.layers.{i}.post_attention_layernorm.weight``,
                    ``model.layers.{i}.self_attn.{q,k,v,o}_proj.weight``,
                    ``model.layers.{i}.mlp.{gate,up,down}_proj.weight``).
        * shapes -- HF Linear ``(out, in)`` / embedding ``(V, H)`` / gamma
                    ``(H,)`` wrapped in two leading unit dims, matching
                    ttml's native 4D storage.
        * dtype  -- ``ttnn.bfloat16``.
        * layout -- ``ttnn.TILE_LAYOUT``.
        * memcfg -- ``ttnn.DRAM_MEMORY_CONFIG`` (interleaved).
        * mesh   -- replicated (single-device assumption; see below).

        **Q/K row order.** ttml's ``GroupedQueryAttentionCompositeKV``
        applies RoPE on Meta-style interleaved pairs (``ttml.ops.rope.rope``),
        and ``safetensors_loader._unpermute_proj_rows`` already converted
        HF -> Meta on load. tt-transformers' Llama-3.2-1B default also uses
        Meta-style RoPE, so the consumer is called with ``hf_rope=False``
        (no row permutation needed). If a future consumer uses HF-style
        RoPE, the caller is responsible for permuting Q/K rows on host
        before re-uploading (Attention.update raises NotImplementedError
        on the row-permutation path today).

        **Tied embeddings.** When ``weight_tying=Enabled`` (the only
        supported case here, and the only case Llama-3.2-1B-Instruct
        uses), ttml stores a single ``Llama/fc/weight``. Both
        ``model.embed_tokens.weight`` and ``lm_head.weight`` in the
        output point at the same ``ttnn.Tensor`` handle. This is safe
        because the consumer's ``update_weights`` does a per-key
        ``ttnn.copy`` into a separate destination buffer and never
        aliases the source.

        **No copy for most values.** Apart from the per-layer K/V split
        (described below), every entry is a handle into ttml's live
        parameter store. Do **not** mutate ttml's parameters between
        calling this and calling ``update_weights``. After
        ``update_weights`` has consumed the dict, the K/V slices can be
        freed; the rest are owned by ttml and stay live.

        **K/V split.** ttml fuses K and V into a single
        ``kv_linear/weight`` of shape ``(1, 1, 2*n_kv*D, H)`` (K rows
        first, then V rows -- see ``safetensors_loader.try_combine_kv``
        which builds it as ``np.concatenate([k, v], axis=0)``). HF
        stores K and V as two separate tensors, so we expose them via
        two ``ttnn.slice`` calls. These slices are *newly allocated*
        tensors (~``2 * n_kv * D * H * sizeof(bf16) * L`` bytes total
        across all layers; ~64 MB for Llama-3.2-1B-Instruct).

        **Single-device assumption.** This method assumes the underlying
        parameter tensors are *replicated* across the mesh (no DDP/TP
        shard mapper applied at upload time). The grpo single-device
        config (``mesh_shape: [1, 1]``, ``enable_ddp: False``) satisfies
        this. Extending to DDP/TP would
        require a host-side ``ttnn.concat_mesh_to_tensor`` (or its ttml
        equivalent) per parameter before exposing the handle; not done
        here.
        """
        from ttml.models import WeightTyingType

        cfg = self.config
        assert cfg.weight_tying == WeightTyingType.Enabled, (
            "export_to_hf_dict requires weight_tying=Enabled (Llama-3.2-1B/-Instruct "
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
