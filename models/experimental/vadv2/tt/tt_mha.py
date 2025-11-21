# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math


def get_high_perf_compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class TtMultiheadAttention:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        init_cfg=None,
        batch_first=False,
    ):
        super().__init__()
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device
        self.batch_first = batch_first
        self.attn_in_proj__weight = params.in_proj.weight
        self.attn_in_proj__bias = params.in_proj.bias
        self.attn_in_proj__weight_permute = ttnn.permute(self.attn_in_proj__weight, (1, 0))
        self.attn_in_proj__bias_squeeze = ttnn.squeeze(self.attn_in_proj__bias, 0)
        self.attn_out_proj_weight = params.out_proj.weight
        self.attn_out_proj_bias = params.out_proj.bias
        self.mem_cfg = ttnn.L1_MEMORY_CONFIG
        self.compute_cfg = get_high_perf_compute_config()
        self.head_dim = embed_dims // num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        batch_first=False,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None and query_pos.shape == key.shape:
            key_pos = query_pos

        # Apply positional encodings
        if query_pos is not None:
            if query_pos.get_layout() != ttnn.TILE_LAYOUT:
                query_pos = ttnn.to_layout(query_pos, ttnn.TILE_LAYOUT)
            query = query + query_pos
        if key_pos is not None:
            if key_pos.get_layout() != ttnn.TILE_LAYOUT:
                key_pos = ttnn.to_layout(key_pos, ttnn.TILE_LAYOUT)
            key = key + key_pos

        # Ensure TILE layout for the core tensors
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)

        attn_mem_cfg = self.mem_cfg
        attn_compute_cfg = self.compute_cfg

        def requires_dram_fallback(err: RuntimeError):
            message = str(err).lower()
            return "out of memory" in message or "circular buffer" in message

        def switch_to_dram():
            nonlocal attn_mem_cfg, attn_compute_cfg
            attn_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
            attn_compute_cfg = None

        def ensure_attn_memory(tensor):
            nonlocal attn_mem_cfg
            tensor_mem_cfg = ttnn.get_memory_config(tensor)
            if tensor_mem_cfg is not None and tensor_mem_cfg == attn_mem_cfg:
                return tensor
            try:
                return ttnn.to_memory_config(tensor, attn_mem_cfg)
            except RuntimeError as err:
                if requires_dram_fallback(err) and attn_mem_cfg != ttnn.DRAM_MEMORY_CONFIG:
                    switch_to_dram()
                    return ttnn.to_memory_config(tensor, attn_mem_cfg)
                if requires_dram_fallback(err):
                    return tensor
                raise

        def matmul_with_attn_mem(lhs, rhs):
            nonlocal attn_mem_cfg
            lhs = ensure_attn_memory(lhs)
            rhs = ensure_attn_memory(rhs)
            try:
                kwargs = {"memory_config": attn_mem_cfg}
                if attn_compute_cfg is not None:
                    kwargs["compute_kernel_config"] = attn_compute_cfg
                return ttnn.matmul(lhs, rhs, **kwargs)
            except RuntimeError as err:
                if requires_dram_fallback(err) and attn_mem_cfg != ttnn.DRAM_MEMORY_CONFIG:
                    switch_to_dram()
                    lhs = ensure_attn_memory(lhs)
                    rhs = ensure_attn_memory(rhs)
                    kwargs = {"memory_config": attn_mem_cfg}
                    if attn_compute_cfg is not None:
                        kwargs["compute_kernel_config"] = attn_compute_cfg
                    return ttnn.matmul(lhs, rhs, **kwargs)
                raise

        def add_with_attn_mem(lhs, rhs):
            nonlocal attn_mem_cfg
            lhs = ensure_attn_memory(lhs)
            rhs = ensure_attn_memory(rhs)
            try:
                return ttnn.add(lhs, rhs, memory_config=attn_mem_cfg)
            except RuntimeError as err:
                if requires_dram_fallback(err) and attn_mem_cfg != ttnn.DRAM_MEMORY_CONFIG:
                    switch_to_dram()
                    lhs = ensure_attn_memory(lhs)
                    rhs = ensure_attn_memory(rhs)
                    return ttnn.add(lhs, rhs, memory_config=attn_mem_cfg)
                raise

        def softmax_device_fallback(tensor, dim):
            nonlocal attn_mem_cfg
            if attn_mem_cfg != ttnn.DRAM_MEMORY_CONFIG:
                switch_to_dram()
                tensor = ensure_attn_memory(tensor)
            max_values = ttnn.max(tensor, dim=dim, keepdim=True)[0]
            max_values = ensure_attn_memory(max_values)
            shifted = ttnn.subtract(tensor, max_values, memory_config=attn_mem_cfg)
            exp_tensor = ttnn.exp(shifted)
            exp_tensor = ensure_attn_memory(exp_tensor)
            sum_tensor = ttnn.sum(exp_tensor, dim=dim, keepdim=True)
            sum_tensor = ensure_attn_memory(sum_tensor)
            return ttnn.divide(exp_tensor, sum_tensor, memory_config=attn_mem_cfg)

        def softmax_with_attn_mem(tensor, dim):
            nonlocal attn_mem_cfg
            tensor = ensure_attn_memory(tensor)

            def call_softmax(tt_tensor):
                kwargs = {"memory_config": attn_mem_cfg, "dim": dim}
                if attn_compute_cfg is not None:
                    kwargs["compute_kernel_config"] = attn_compute_cfg
                return ttnn.softmax(tt_tensor, **kwargs)

            try:
                return call_softmax(tensor)
            except RuntimeError as err:
                if requires_dram_fallback(err):
                    if attn_mem_cfg != ttnn.DRAM_MEMORY_CONFIG:
                        switch_to_dram()
                        tensor = ensure_attn_memory(tensor)
                        try:
                            return call_softmax(tensor)
                        except RuntimeError as err2:
                            if requires_dram_fallback(err2):
                                return softmax_device_fallback(tensor, dim)
                            raise
                    return softmax_device_fallback(tensor, dim)
                raise

        def linear_with_attn_mem(tensor, weight, bias):
            nonlocal attn_mem_cfg
            tensor = ensure_attn_memory(tensor)
            try:
                kwargs = {"memory_config": attn_mem_cfg, "compute_kernel_config": attn_compute_cfg}
                if attn_compute_cfg is None:
                    kwargs.pop("compute_kernel_config")
                return ttnn.linear(
                    tensor,
                    weight,
                    bias=bias,
                    **kwargs,
                )
            except RuntimeError as err:
                if requires_dram_fallback(err) and attn_mem_cfg != ttnn.DRAM_MEMORY_CONFIG:
                    switch_to_dram()
                    tensor = ensure_attn_memory(tensor)
                    kwargs = {"memory_config": attn_mem_cfg, "compute_kernel_config": attn_compute_cfg}
                    if attn_compute_cfg is None:
                        kwargs.pop("compute_kernel_config")
                    return ttnn.linear(
                        tensor,
                        weight,
                        bias=bias,
                        **kwargs,
                    )
                raise

        def concat_heads_with_attn_mem(tensor):
            nonlocal attn_mem_cfg
            tensor = ensure_attn_memory(tensor)
            try:
                return ttnn.transformer.concatenate_heads(tensor, memory_config=attn_mem_cfg)
            except RuntimeError as err:
                if requires_dram_fallback(err) and attn_mem_cfg != ttnn.DRAM_MEMORY_CONFIG:
                    switch_to_dram()
                    tensor = ensure_attn_memory(tensor)
                    return ttnn.transformer.concatenate_heads(tensor, memory_config=attn_mem_cfg)
                raise

        query = ensure_attn_memory(query)
        key = ensure_attn_memory(key)
        value = ensure_attn_memory(value)
        identity = ensure_attn_memory(identity)

        share_qkv = key is query and value is key

        tgt_len, bsz, _ = query.shape
        query_bf = ttnn.permute(query, (1, 0, 2))
        query_bf = ensure_attn_memory(query_bf)

        if share_qkv:
            qkv_states = linear_with_attn_mem(
                query_bf,
                self.attn_in_proj__weight_permute,
                self.attn_in_proj__bias,
            )
            qkv_states = ttnn.unsqueeze(qkv_states, dim=1)
            qkv_states = ensure_attn_memory(qkv_states)
            query_heads, key_heads, value_heads = ttnn.experimental.nlp_create_qkv_heads(
                qkv_states,
                memory_config=attn_mem_cfg,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                transpose_k_heads=True,
            )
            query_heads = ensure_attn_memory(query_heads)
            key_heads = ensure_attn_memory(key_heads)
            value_heads = ensure_attn_memory(value_heads)
            ttnn.deallocate(qkv_states)
        else:
            in_proj_weight = self.attn_in_proj__weight_permute
            in_proj_bias = self.attn_in_proj__bias_squeeze
            q_weight = ttnn.permute(in_proj_weight[: self.embed_dims, :], (1, 0))
            k_weight = ttnn.permute(in_proj_weight[self.embed_dims : 2 * self.embed_dims, :], (1, 0))
            v_weight = ttnn.permute(in_proj_weight[2 * self.embed_dims :, :], (1, 0))

            q_bias = in_proj_bias[: self.embed_dims]
            k_bias = in_proj_bias[self.embed_dims : 2 * self.embed_dims]
            v_bias = in_proj_bias[2 * self.embed_dims :]

            query_proj = linear_with_attn_mem(
                query_bf,
                q_weight,
                bias=q_bias,
            )
            query_proj = ttnn.unsqueeze(query_proj, dim=1)
            query_proj = ensure_attn_memory(query_proj)
            query_heads = ttnn.experimental.nlp_create_qkv_heads(
                query_proj,
                memory_config=attn_mem_cfg,
                num_heads=self.num_heads,
                num_kv_heads=0,
            )[0]
            query_heads = ensure_attn_memory(query_heads)
            ttnn.deallocate(query_proj)

            key_bf = ttnn.permute(key, (1, 0, 2))
            key_bf = ensure_attn_memory(key_bf)
            key_proj = linear_with_attn_mem(
                key_bf,
                k_weight,
                bias=k_bias,
            )
            key_proj = ttnn.unsqueeze(key_proj, dim=1)
            key_proj = ensure_attn_memory(key_proj)
            key_heads = ttnn.experimental.nlp_create_qkv_heads(
                key_proj,
                memory_config=attn_mem_cfg,
                num_heads=self.num_heads,
                num_kv_heads=0,
            )[0]
            key_heads = ttnn.permute(key_heads, [0, 1, 3, 2])
            key_heads = ensure_attn_memory(key_heads)
            ttnn.deallocate(key_proj)

            value_bf = ttnn.permute(value, (1, 0, 2))
            value_bf = ensure_attn_memory(value_bf)
            value_proj = linear_with_attn_mem(
                value_bf,
                v_weight,
                bias=v_bias,
            )
            value_proj = ttnn.unsqueeze(value_proj, dim=1)
            value_proj = ensure_attn_memory(value_proj)
            value_heads = ttnn.experimental.nlp_create_qkv_heads(
                value_proj,
                memory_config=attn_mem_cfg,
                num_heads=self.num_heads,
                num_kv_heads=0,
            )[0]
            value_heads = ensure_attn_memory(value_heads)
            ttnn.deallocate(value_proj)

        query_heads = ensure_attn_memory(query_heads)
        query_heads = ttnn.multiply(query_heads, self.scaling, memory_config=attn_mem_cfg)
        query_heads = ensure_attn_memory(query_heads)
        key_heads = ensure_attn_memory(key_heads)

        attn_weights = matmul_with_attn_mem(query_heads, key_heads)

        if attn_mask is not None:
            if attn_mask.get_layout() != ttnn.TILE_LAYOUT:
                attn_mask = ttnn.to_layout(attn_mask, ttnn.TILE_LAYOUT)
            attn_mask = ensure_attn_memory(attn_mask)
            attn_weights = add_with_attn_mem(attn_weights, attn_mask)

        attn_weights = softmax_with_attn_mem(attn_weights, dim=-1)

        value_heads = ensure_attn_memory(value_heads)
        attn_output = matmul_with_attn_mem(attn_weights, value_heads)

        attn_output = concat_heads_with_attn_mem(attn_output)
        attn_output = linear_with_attn_mem(
            attn_output,
            self.attn_out_proj_weight,
            bias=self.attn_out_proj_bias,
        )

        attn_output = ttnn.permute(attn_output, (1, 0, 2))

        return add_with_attn_mem(attn_output, identity)
