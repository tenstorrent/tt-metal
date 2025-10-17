# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtAttentionTail:
    def __init__(
        self,
        params,
        device,
        cfg,
        dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        self.device = device
        self.params = params
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

    def forward(self, query, key, key_padding_mask, hw_lvl=None):
        B, N, C = query.shape
        _, L, _ = key.shape

        q_proj = ttnn.linear(query, self.params.q.weight, bias=self.params.q.bias)
        q_reshaped = ttnn.reshape(q_proj, (B, N, self.num_heads, C // self.num_heads))
        q = ttnn.permute(q_reshaped, (0, 2, 1, 3))

        k_proj = ttnn.linear(key, self.params.k.weight, bias=self.params.k.bias)
        k_reshaped = ttnn.reshape(k_proj, (B, L, self.num_heads, C // self.num_heads))
        k = ttnn.permute(k_reshaped, (0, 2, 1, 3))

        k_transposed = ttnn.permute(k, (0, 1, 3, 2))
        qk_matmul = ttnn.matmul(q, k_transposed)
        attn = qk_matmul * self.scale

        attn = ttnn.permute(attn, (0, 2, 3, 1))

        new_feats = ttnn.linear(attn, self.params.linear_l1[0].weight, bias=self.params.linear_l1[0].bias)
        new_feats = ttnn.relu(new_feats)
        mask = ttnn.linear(new_feats, self.params.linear[0].weight, bias=self.params.linear[0].bias)
        mask = ttnn.relu(mask)

        return mask


class TtMlp:
    def __init__(
        self,
        params,
        device,
    ):
        super().__init__()
        self.params = params
        self.device = device

    def forward(self, x):
        x = ttnn.linear(x, self.params.fc1.weight, bias=self.params.fc1.bias)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.params.fc2.weight, bias=self.params.fc2.bias)
        return x


class TtSelfAttention:
    def __init__(
        self,
        params,
        device,
        cfg,
        dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        self.params = params
        self.device = device
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

    def forward(self, x):
        B, N, C = x.shape

        qkv = ttnn.linear(x, self.params.qkv.weight, bias=self.params.qkv.bias)
        qkv = ttnn.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))

        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = ttnn.permute(k, (0, 1, 3, 2))
        qk = ttnn.matmul(q, k_t)
        attn = qk * self.scale

        attn = ttnn.softmax(attn, dim=-1)

        attn_v = ttnn.matmul(attn, v)
        attn_v_p = ttnn.permute(attn_v, (0, 2, 1, 3))
        x = ttnn.reshape(attn_v_p, (B, N, C))

        x = ttnn.linear(x, self.params.proj.weight, bias=self.params.proj.bias)

        return x


class TtAttention:
    def __init__(
        self,
        params,
        device,
        cfg,
        dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        self.device = device
        self.params = params
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

    def forward(self, query, key, value, key_padding_mask, hw_lvl):
        B, N, C = query.shape
        _, L, _ = key.shape

        q_proj = ttnn.linear(query, self.params.q.weight, bias=self.params.q.bias)
        q_reshaped = ttnn.reshape(q_proj, (B, N, self.num_heads, C // self.num_heads))
        q = ttnn.permute(q_reshaped, (0, 2, 1, 3))

        k_proj = ttnn.linear(key, self.params.k.weight, bias=self.params.k.bias)
        k_reshaped = ttnn.reshape(k_proj, (B, L, self.num_heads, C // self.num_heads))
        k = ttnn.permute(k_reshaped, (0, 2, 1, 3))

        v_proj = ttnn.linear(value, self.params.v.weight, bias=self.params.v.bias)
        v_reshaped = ttnn.reshape(v_proj, (B, L, self.num_heads, C // self.num_heads))
        v = ttnn.permute(v_reshaped, (0, 2, 1, 3))

        k_transposed = ttnn.permute(k, (0, 1, 3, 2))
        qk_matmul = ttnn.matmul(q, k_transposed)
        attn = qk_matmul * self.scale

        attn = ttnn.permute(attn, (0, 2, 3, 1))

        new_feats = ttnn.linear(attn, self.params.linear_l1[0].weight, bias=self.params.linear_l1[0].bias)
        new_feats = ttnn.relu(new_feats)
        mask = ttnn.linear(new_feats, self.params.linear[0].weight, bias=self.params.linear[0].bias)
        mask = ttnn.relu(mask)

        attn = ttnn.permute(attn, (0, 3, 1, 2))
        attn = ttnn.softmax(attn, dim=-1)

        attn_v = ttnn.matmul(attn, v)
        attn_v_t = ttnn.permute(attn_v, (0, 2, 1, 3))
        x = ttnn.reshape(attn_v_t, (B, N, C))

        x = ttnn.linear(x, self.params.proj.weight, bias=self.params.proj.bias)

        return x, mask


class TtBlock:
    def __init__(
        self, params, device, cfg, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, self_attn=False
    ):
        super().__init__()
        self.device = device
        self.params = params
        self.eps = 1e-06
        self.self_attn = self_attn
        self.attn = TtAttention(
            params.attn,
            device,
            cfg,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TtMlp(
            params=params.mlp,
            device=device,
        )
        if self.self_attn:
            self.self_attention = TtSelfAttention(
                params.self_attention,
                device,
                cfg,
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
            )

    def forward(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        if self.self_attn:
            query = query + self.self_attention.forward(query)
            query = ttnn.layer_norm(
                query,
                weight=self.params.norm3.weight,
                bias=self.params.norm3.bias,
                epsilon=self.eps,
            )

        x, mask = self.attn.forward(query, key, value, key_padding_mask, hw_lvl=hw_lvl)

        query = query + x

        query = ttnn.layer_norm(
            query,
            weight=self.params.head_norm1.weight,
            bias=self.params.head_norm1.bias,
            epsilon=self.eps,
        )

        query = query + self.mlp.forward(query)

        query = ttnn.layer_norm(
            query,
            weight=self.params.head_norm2.weight,
            bias=self.params.head_norm2.bias,
            epsilon=self.eps,
        )

        return query, mask


class TtSegMaskHead:
    def __init__(
        self,
        params,
        device,
        cfg=None,
        d_model=16,
        nhead=2,
        num_encoder_layers=6,
        num_decoder_layers=1,
        dim_feedforward=64,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        normalize_before=False,
        return_intermediate_dec=False,
        self_attn=False,
    ):
        super().__init__()
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        self.blocks = []
        for i in range(num_decoder_layers):
            self.blocks.append(
                TtBlock(
                    params.blocks[i],
                    device,
                    cfg,
                    dim=d_model,
                    num_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    self_attn=self_attn,
                )
            )
        self.attnen = TtAttentionTail(
            params.attnen,
            device,
            cfg,
            d_model,
            num_heads=nhead,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )

    def with_pos_embed(self, tensor, pos):
        if pos is None:
            return tensor
        else:
            return tensor + pos

    def forward(self, memory, mask_memory, pos_memory, query_embed, mask_query, pos_query, hw_lvl):
        masks = []
        inter_query = []
        for i, block in enumerate(self.blocks):
            query_embed, mask = block.forward(
                self.with_pos_embed(query_embed, pos_query),
                self.with_pos_embed(memory, pos_memory),
                memory,
                key_padding_mask=mask_memory,
                hw_lvl=hw_lvl,
            )
            masks.append(mask)
            inter_query.append(query_embed)

        attn = self.attnen.forward(
            self.with_pos_embed(query_embed, pos_query),
            self.with_pos_embed(memory, pos_memory),
            key_padding_mask=mask_memory,
            hw_lvl=hw_lvl,
        )
        return attn, masks, inter_query
