import ttnn

from ..tt.sdpa import sdpa as tt_sdpa


class Attention:
    def __init__(self, mesh_device, hf_config, state_dict, layer_idx):
        self.layer_idx = layer_idx
        self.use_sliding_window = self.layer_idx % 2 == 0
        self.scaling = hf_config.head_dim**-0.5
        self.cache = None
        self.head_dim = hf_config.head_dim
        self.num_heads = hf_config.num_attention_heads
        self.hidden_size = hf_config.hidden_size

        # (['sinks', 'q_proj.weight', 'q_proj.bias', 'k_proj.weight', 'k_proj.bias', 'v_proj.weight', 'v_proj.bias', 'o_proj.weight', 'o_proj.bias'])

        # TODO: Add mesh mapper
        q_proj = state_dict["q_proj.weight"].transpose(-1, -2)
        q_proj_bias = state_dict["q_proj.bias"]  # TODO: unsqueeze?

        k_proj = state_dict["k_proj.weight"].transpose(-1, -2)
        k_proj_bias = state_dict["k_proj.bias"]  # TODO: unsqueeze?

        v_proj = state_dict["v_proj.weight"].transpose(-1, -2)
        v_proj_bias = state_dict["v_proj.bias"]  # TODO: unsqueeze?

        o_proj = state_dict["o_proj.weight"].transpose(-1, -2)
        o_proj_bias = state_dict["o_proj.bias"]  # TODO: unsqueeze?

        sinks = state_dict["sinks"].reshape(1, hf_config.num_attention_heads, 1, 1)

        self.q_proj = ttnn.from_torch(q_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.q_proj_bias = ttnn.from_torch(
            q_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        self.k_proj = ttnn.from_torch(k_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.k_proj_bias = ttnn.from_torch(
            k_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        self.v_proj = ttnn.from_torch(v_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.v_proj_bias = ttnn.from_torch(
            v_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        self.o_proj = ttnn.from_torch(o_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.o_proj_bias = ttnn.from_torch(
            o_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        self.sinks = ttnn.from_torch(sinks, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    def __call__(self, x: ttnn.Tensor, mask, rope_stuff):
        batch_size, seq_len, hidden_size = x.shape

        tt_q = ttnn.matmul(x, self.q_proj) + self.q_proj_bias
        tt_q = ttnn.reshape(tt_q, [1, seq_len * batch_size, -1, self.head_dim])

        tt_k = ttnn.matmul(x, self.k_proj) + self.k_proj_bias
        tt_k = ttnn.reshape(tt_k, [1, seq_len * batch_size, -1, self.head_dim])

        tt_v = ttnn.matmul(x, self.v_proj) + self.v_proj_bias
        tt_v = ttnn.reshape(tt_v, [1, seq_len * batch_size, -1, self.head_dim])

        apply_rope, tt_cos, tt_sin = rope_stuff
        tt_q = apply_rope(tt_q, tt_cos, tt_sin)
        tt_k = apply_rope(tt_k, tt_cos, tt_sin)

        tt_q = ttnn.reshape(tt_q, [batch_size * seq_len, -1, self.num_heads, self.head_dim])
        tt_k = ttnn.reshape(tt_k, [batch_size * seq_len, -1, self.head_dim])
        tt_v = ttnn.reshape(tt_v, [batch_size * seq_len, -1, self.head_dim])

        tt_sdpa_out, self.cache = tt_sdpa(
            tt_q,
            tt_k,
            tt_v,
            self.sinks,
            sm_scale=self.scaling,
            tt_mask=mask,
            tt_cache=self.cache,
        )

        tt_out = ttnn.matmul(tt_sdpa_out, self.o_proj) + self.o_proj_bias
        tt_out = ttnn.reshape(tt_out, [batch_size, seq_len, hidden_size])

        return tt_out
