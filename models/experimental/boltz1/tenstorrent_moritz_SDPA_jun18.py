import torch, ttnn
from torch import nn
from typing import Tuple, Callable, Dict

TRIANGLE_MULT_CHUNK_SIZE = 256
TRANSITION_CHUNK_SIZE = 64
PAIR_WEIGHTED_AVG_CHUNK_SIZE = 64
OUTER_PRODUCT_MEAN_CHUNK_SIZE = 64
USE_FLOAT32 = False

# device = None

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def filter_dict(state_dict: dict, prefix: str, remove: str = "") -> dict:
    if not prefix:
        return state_dict
    prefix += "."
    return {
        key[len(prefix) :].replace(remove, ""): value for key, value in state_dict.items() if key.startswith(prefix)
    }


class Module:
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.state_dict = state_dict
        self.compute_kernel_config = compute_kernel_config

    def torch_to_tt(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
        use_float32: bool = False,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            transform(self.state_dict[key]),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.float32 if USE_FLOAT32 or use_float32 else ttnn.bfloat8_b,
        )

    def torch_to_tt_qkv(
        self,
        weight_tensor: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.t(),
        use_float32: bool = False,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            transform(weight_tensor),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.float32 if USE_FLOAT32 or use_float32 else ttnn.bfloat8_b,
        )


class TriangleMultiplication(Module):
    def __init__(
        self,
        device,
        ending: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.device = device
        super().__init__(state_dict, compute_kernel_config)
        self.ending = ending
        self.in_norm_weight = self.torch_to_tt("norm_in.weight")
        self.in_norm_bias = self.torch_to_tt("norm_in.bias")
        self.in_p = self.torch_to_tt("p_in.weight")
        self.in_g = self.torch_to_tt("g_in.weight")
        self.out_norm_weight = self.torch_to_tt("norm_out.weight")
        self.out_norm_bias = self.torch_to_tt("norm_out.bias")
        self.out_p = self.torch_to_tt("p_out.weight")
        self.out_g = self.torch_to_tt("g_out.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_norm_in = ttnn.layer_norm(
            x,
            weight=self.in_norm_weight,
            bias=self.in_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )

        p_in = ttnn.linear(x_norm_in, self.in_p, compute_kernel_config=self.compute_kernel_config)
        gl_in = ttnn.linear(x_norm_in, self.in_g, compute_kernel_config=self.compute_kernel_config)
        g_in = ttnn.sigmoid_accurate(gl_in)
        x = ttnn.multiply(p_in, g_in)

        dim = int(x.shape[-1] / 2)
        a = ttnn.permute(x[:, :, :, :dim], (0, 3) + ((2, 1) if self.ending else (1, 2)))
        b = ttnn.permute(x[:, :, :, dim:], (0, 3) + ((1, 2) if self.ending else (2, 1)))
        del x
        x_chunks = []
        for chunk_start in range(0, a.shape[2], TRIANGLE_MULT_CHUNK_SIZE):
            chunk_end = min(chunk_start + TRIANGLE_MULT_CHUNK_SIZE, a.shape[2])
            x_chunk = ttnn.matmul(
                a[:, :, chunk_start:chunk_end, :],
                b,
                compute_kernel_config=self.compute_kernel_config,
            )
            x_chunks.append(x_chunk)
        del a, b
        x = ttnn.concat(x_chunks, dim=2)
        del x_chunks
        x = ttnn.permute(x, (0, 2, 3, 1))
        x_norm_out = ttnn.layer_norm(
            x,
            weight=self.out_norm_weight,
            bias=self.out_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )

        """
        x_chunks = []
        for chunk_start in range(0, x_norm_out.shape[1], TRIANGLE_MULT_CHUNK_SIZE):
            chunk_end = min(chunk_start + TRIANGLE_MULT_CHUNK_SIZE, x_norm_out.shape[1])
            x_chunk = ttnn.multiply(
                ttnn.linear(
                    x_norm_out[:, chunk_start:chunk_end, :, :],
                    self.out_p,
                    compute_kernel_config=self.compute_kernel_config,
                ),
                ttnn.sigmoid_accurate(
                    ttnn.linear(
                        x_norm_in[:, chunk_start:chunk_end, :, :],
                        self.out_g,
                        compute_kernel_config=self.compute_kernel_config,
                    )
                ),
            )
            x_chunks.append(x_chunk)
        x = ttnn.concat(x_chunks, dim=1)

        """
        gl_out = ttnn.linear(x_norm_in, self.out_g, compute_kernel_config=self.compute_kernel_config)
        g_out = ttnn.sigmoid_accurate(gl_out)
        p_out = ttnn.linear(x_norm_out, self.out_p, compute_kernel_config=self.compute_kernel_config)
        x = ttnn.multiply(p_out, g_out)

        return x


class TriangleAttention(Module):
    def __init__(
        self,
        device,
        head_dim: int,
        n_heads: int,
        ending: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.device = device
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ending = ending
        self.layer_norm_weight = self.torch_to_tt("layer_norm.weight")
        self.layer_norm_bias = self.torch_to_tt("layer_norm.bias")
        self.bias_weight = self.torch_to_tt("linear.weight")
        self.q_weight = self.torch_to_tt("linear_q.weight")
        self.k_weight = self.torch_to_tt("linear_k.weight")
        self.v_weight = self.torch_to_tt("linear_v.weight")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.g_weight = self.torch_to_tt("linear_g.weight")

        hidden_size = self.n_heads * self.head_dim * 3
        self.qkv_weight_torch = torch.cat(
            [
                self.state_dict["linear_q.weight"].reshape([self.n_heads, self.head_dim, -1]),
                self.state_dict["linear_k.weight"].reshape([self.n_heads, self.head_dim, -1]),
                self.state_dict["linear_v.weight"].reshape([self.n_heads, self.head_dim, -1]),
            ],
            dim=0,
        ).reshape([hidden_size, -1])

        self.qkvg_weight_torch = torch.cat(
            [self.qkv_weight_torch, self.state_dict["linear_g.weight"]],
            dim=0,
        )
        self.qkvg_weight = self.torch_to_tt_qkv(self.qkvg_weight_torch)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))  # THIS CAUSES CACHE -> RESHAPE PROBLEM
        x = ttnn.layer_norm(
            x,
            weight=self.layer_norm_weight,
            bias=self.layer_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        seq_len = x.shape[0]
        padding = -seq_len % 32
        x = ttnn.pad(x, [(0, padding), (0, padding), (0, 0)], 0)
        triangle_bias = ttnn.linear(
            x,
            self.bias_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        triangle_bias = ttnn.reshape(triangle_bias, (1, *triangle_bias.shape))
        triangle_bias = ttnn.permute(triangle_bias, (3, 0, 1, 2))

        qkvg = ttnn.linear(
            x,
            self.qkvg_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
        )

        qkv = qkvg[:, :, : 3 * self.head_dim * self.n_heads]
        qkv = ttnn.unsqueeze(qkv, 0)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_boltz(
            qkv,
            num_heads=self.n_heads,
            num_kv_heads=self.n_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for head in range(0, q.shape[0]):
            head_q = q[head : head + 1, :, :, :]
            head_k = k[head : head + 1, :, :, :]
            head_v = v[head : head + 1, :, :, :]
            head_triangle_bias = triangle_bias[head : head + 1, :, :, :]
            head_o = ttnn.transformer.scaled_dot_product_attention(
                head_q,
                head_k,
                head_v,
                attn_mask=head_triangle_bias,
                is_causal=False,
                scale=self.head_dim**-0.5,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    exp_approx_mode=False,
                    q_chunk_size=256,
                    k_chunk_size=256,
                ),
            )
            if head == 0:
                o = head_o
            else:
                o = ttnn.concat([o, head_o], dim=0)

        o = ttnn.experimental.nlp_concat_heads_boltz(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.squeeze(o, 0)

        g = qkvg[:, :, 3 * self.head_dim * self.n_heads :]
        g = ttnn.squeeze(g, 0)
        g = ttnn.sigmoid_accurate(g)
        o = ttnn.multiply(o, g)
        x = ttnn.linear(
            o,
            self.o_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
        )
        x = x[:seq_len, :seq_len, :]
        if self.ending:
            x = ttnn.permute(x, (1, 0, 2))
        x = ttnn.reshape(x, (1, *x.shape))
        return x


class AttentionPairBias(Module):
    def __init__(
        self,
        device,
        head_dim: int,
        n_heads: int,
        diffusion: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.device = device
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.diffusion = diffusion
        if not diffusion:
            self.norm_s_weight = self.torch_to_tt("norm_s.weight")
            self.norm_s_bias = self.torch_to_tt("norm_s.bias")
        self.q_weight = self.torch_to_tt("proj_q.weight")
        self.q_bias = self.torch_to_tt("proj_q.bias")
        self.k_weight = self.torch_to_tt("proj_k.weight")
        self.v_weight = self.torch_to_tt("proj_v.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        self.z_norm_weight = self.torch_to_tt("proj_z.0.weight")
        self.z_norm_bias = self.torch_to_tt("proj_z.0.bias")
        self.z_weight = self.torch_to_tt("proj_z.1.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")

    def __call__(self, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        if not self.diffusion:
            s = ttnn.layer_norm(
                s,
                weight=self.norm_s_weight,
                bias=self.norm_s_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
        q = ttnn.linear(
            s,
            self.q_weight,
            bias=self.q_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        k = ttnn.linear(s, self.k_weight, compute_kernel_config=self.compute_kernel_config)
        v = ttnn.linear(s, self.v_weight, compute_kernel_config=self.compute_kernel_config)
        q = ttnn.permute(q, (2, 0, 1))
        k = ttnn.permute(k, (2, 0, 1))
        v = ttnn.permute(v, (2, 0, 1))
        q = ttnn.reshape(q, (self.n_heads, self.head_dim, *tuple(q.shape)[1:]))
        k = ttnn.reshape(k, (self.n_heads, self.head_dim, *tuple(k.shape)[1:]))
        v = ttnn.reshape(v, (self.n_heads, self.head_dim, *tuple(v.shape)[1:]))
        q = ttnn.permute(q, (0, 2, 3, 1))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 3, 1))
        if not USE_FLOAT32:
            q = ttnn.clone(q, dtype=ttnn.float32)
            k = ttnn.clone(k, dtype=ttnn.float32)
        a = ttnn.matmul(q, k, compute_kernel_config=self.compute_kernel_config)
        a = ttnn.multiply(a, self.head_dim**-0.5)
        z = ttnn.layer_norm(
            z,
            weight=self.z_norm_weight,
            bias=self.z_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        z = ttnn.linear(z, self.z_weight, compute_kernel_config=self.compute_kernel_config)
        z = ttnn.permute(z, (3, 0, 1, 2))
        if not USE_FLOAT32:
            z = ttnn.clone(z, dtype=ttnn.float32)
        a = ttnn.add(a, z)
        if not USE_FLOAT32:
            a = ttnn.clone(a, dtype=ttnn.bfloat16)
        a = ttnn.softmax(
            a,
            dim=-1,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=self.compute_kernel_config.math_fidelity,
                math_approx_mode=self.compute_kernel_config.math_approx_mode,
                fp32_dest_acc_en=False,
                packer_l1_acc=self.compute_kernel_config.packer_l1_acc,
            ),
            numeric_stable=True,
        )
        o = ttnn.matmul(a, v, compute_kernel_config=self.compute_kernel_config)
        o = ttnn.permute(o, (0, 3, 1, 2))
        o = ttnn.reshape(o, (-1, *tuple(o.shape)[2:]))
        o = ttnn.permute(o, (1, 2, 0))
        g = ttnn.linear(s, self.g_weight, compute_kernel_config=self.compute_kernel_config)
        g = ttnn.sigmoid_accurate(g)
        o = ttnn.multiply(o, g)
        x = ttnn.linear(o, self.o_weight, compute_kernel_config=self.compute_kernel_config)
        return x


class Transition(Module):
    def __init__(
        self,
        device,
        chunking: bool,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        self.device = device
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.fc1_weight = self.torch_to_tt("fc1.weight")
        self.fc2_weight = self.torch_to_tt("fc2.weight")
        self.fc3_weight = self.torch_to_tt("fc3.weight")
        self.chunking = chunking

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        def f(x):
            x_norm = ttnn.layer_norm(
                x,
                weight=self.norm_weight,
                bias=self.norm_bias,
                epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config,
            )
            x_1 = ttnn.linear(
                x_norm,
                self.fc1_weight,
                activation="silu",
                compute_kernel_config=self.compute_kernel_config,
            )
            x_2 = ttnn.linear(
                x_norm,
                self.fc2_weight,
                compute_kernel_config=self.compute_kernel_config,
            )
            x = ttnn.multiply(x_1, x_2)
            x = ttnn.linear(
                x,
                self.fc3_weight,
                compute_kernel_config=self.compute_kernel_config,
            )
            return x

        if not self.chunking:
            x = f(x)
        else:
            for chunk_start in range(0, x.shape[1], TRANSITION_CHUNK_SIZE):
                x_chunk = x[
                    :,
                    chunk_start : min(chunk_start + TRANSITION_CHUNK_SIZE, x.shape[1]),
                    :,
                    :,
                ]
                x_chunk = f(x_chunk)
                if chunk_start == 0:
                    x_out = x_chunk
                else:
                    x_out = ttnn.concat([x_out, x_chunk], dim=1)
            x = x_out
        return x


class PairformerLayer(Module):
    def __init__(
        self,
        device,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.triangle_multiplication_start = TriangleMultiplication(
            device, False, filter_dict(state_dict, "tri_mul_out"), compute_kernel_config
        )
        self.triangle_multiplication_end = TriangleMultiplication(
            device, True, filter_dict(state_dict, "tri_mul_in"), compute_kernel_config
        )
        self.triangle_attention_start = TriangleAttention(
            device,
            tri_att_head_dim,
            tri_att_n_heads,
            False,
            filter_dict(state_dict, "tri_att_start", "mha."),
            compute_kernel_config,
        )
        self.triangle_attention_end = TriangleAttention(
            device,
            tri_att_head_dim,
            tri_att_n_heads,
            True,
            filter_dict(state_dict, "tri_att_end", "mha."),
            compute_kernel_config,
        )
        self.attention_pair_bias = AttentionPairBias(
            device,
            att_head_dim,
            att_n_heads,
            False,
            filter_dict(state_dict, "attention"),
            compute_kernel_config,
        )
        self.transition_z = Transition(device, True, filter_dict(state_dict, "transition_z"), compute_kernel_config)
        self.transition_s = Transition(device, False, filter_dict(state_dict, "transition_s"), compute_kernel_config)

    def __call__(self, s: ttnn.Tensor, z: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        if use_signpost:
            signpost(header="Triangle Multip Start")

        # start = time.time()
        z = ttnn.add(
            z,
            self.triangle_multiplication_start(z),
        )
        # end = time.time()
        # print(f"$$$YF: triangle_multiplication_start time: {end - start:.4f} seconds")

        if use_signpost:
            signpost(header="Triangle Multip End")
        # start = time.time()
        z = ttnn.add(
            z,
            self.triangle_multiplication_end(z),
        )
        # end = time.time()
        # print(f"$$$YF: triangle_multiplication_end time: {end - start:.4f} seconds")

        if use_signpost:
            signpost(header="Triangle Attention Start")
        # start = time.time()
        z = ttnn.add(
            z,
            self.triangle_attention_start(z),
        )
        # end = time.time()
        # print(f"$$$YF: triangle_attention_start time: {end - start:.4f} seconds")

        if use_signpost:
            signpost(header="Triangle Attention End")
        # start = time.time()
        z = ttnn.add(
            z,
            self.triangle_attention_end(z),
        )
        # end = time.time()
        # print(f"$$$YF: triangle_attention_end time: {end - start:.4f} seconds")

        if use_signpost:
            signpost(header="Transition")
        # start = time.time()
        z = ttnn.add(z, self.transition_z(z))
        # end = time.time()
        # print(f"$$$YF: transition time: {end - start:.4f} seconds")

        if use_signpost:
            signpost(header="S Pair Bias")
        # start = time.time()
        s = ttnn.add(
            s,
            self.attention_pair_bias(s, z),
        )
        # end = time.time()
        # print(f"$$$YF: s_attention_pair_bias time: {end - start:.4f} seconds")

        if use_signpost:
            signpost(header="S Transition")
        # start = time.time()
        s = ttnn.add(s, self.transition_s(s))
        # end = time.time()
        # print(f"$$$YF: transition_s# time: {end - start:.4f} seconds")

        return s, z


class Pairformer(Module):
    def __init__(
        self,
        device,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairformerLayer(
                device,
                tri_att_head_dim,
                tri_att_n_heads,
                att_head_dim,
                att_n_heads,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(self, s: ttnn.Tensor, z: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        for block in self.blocks:
            s, z = block(s, z)
        return s, z


class AdaLN(Module):
    def __init__(
        self,
        device,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.device = device
        self.s_norm_weight = self.torch_to_tt("s_norm.weight", use_float32=True)
        self.s_scale_weight = self.torch_to_tt("s_scale.weight")
        self.s_scale_bias = self.torch_to_tt("s_scale.bias")
        self.s_bias_weight = self.torch_to_tt("s_bias.weight")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        if not USE_FLOAT32:
            a = ttnn.clone(a, dtype=ttnn.float32)
            s = ttnn.clone(s, dtype=ttnn.float32)
        a = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        s = ttnn.layer_norm(
            s,
            weight=self.s_norm_weight,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        if not USE_FLOAT32:
            a = ttnn.clone(a, dtype=ttnn.bfloat16)
            s = ttnn.clone(s, dtype=ttnn.bfloat16)
        s_scale = ttnn.linear(
            s,
            self.s_scale_weight,
            bias=self.s_scale_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_scale = ttnn.sigmoid_accurate(s_scale)
        s_bias = ttnn.linear(s, self.s_bias_weight, compute_kernel_config=self.compute_kernel_config)
        a = ttnn.multiply(a, s_scale)
        a = ttnn.add(a, s_bias)
        return a


class ConditionedTransitionBlock(Module):
    def __init__(
        self,
        device,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.device = device
        self.adaln = AdaLN(device, filter_dict(state_dict, "adaln"), compute_kernel_config)
        self.swish_weight = self.torch_to_tt("swish_gate.0.weight")
        self.a_to_b_weight = self.torch_to_tt("a_to_b.weight")
        self.b_to_a_weight = self.torch_to_tt("b_to_a.weight")
        self.output_projection_weight = self.torch_to_tt("output_projection.0.weight")
        self.output_projection_bias = self.torch_to_tt("output_projection.0.bias")

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        a = self.adaln(a, s)
        a_swish = ttnn.linear(a, self.swish_weight, compute_kernel_config=self.compute_kernel_config)
        dim = int(a_swish.shape[-1] / 2)
        a_swish, gates = a_swish[:, :, :dim], a_swish[:, :, dim:]
        gates = ttnn.silu(gates)
        a_swish = ttnn.multiply(gates, a_swish)
        a_b = ttnn.linear(a, self.a_to_b_weight, compute_kernel_config=self.compute_kernel_config)
        b = ttnn.multiply(a_swish, a_b)
        s = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        s = ttnn.sigmoid_accurate(s)
        b_a = ttnn.linear(b, self.b_to_a_weight, compute_kernel_config=self.compute_kernel_config)
        a = ttnn.multiply(s, b_a)
        return a


class DiffusionTransformerLayer(Module):
    def __init__(
        self,
        device,
        dim: int,
        n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.adaln = AdaLN(filter_dict(state_dict, "adaln"), compute_kernel_config)
        self.device = device
        self.attn_pair_bias = AttentionPairBias(
            head_dim=dim // n_heads,
            n_heads=n_heads,
            diffusion=True,
            state_dict=filter_dict(state_dict, "pair_bias_attn"),
            compute_kernel_config=compute_kernel_config,
        )
        self.output_projection_weight = self.torch_to_tt("output_projection_linear.weight")
        self.output_projection_bias = self.torch_to_tt("output_projection_linear.bias")
        self.transition = ConditionedTransitionBlock(
            filter_dict(state_dict, "transition"),
            compute_kernel_config,
        )

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        b = self.adaln(a, s)
        b = self.attn_pair_bias(b, z)
        s_o = ttnn.linear(
            s,
            self.output_projection_weight,
            bias=self.output_projection_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        s_o = ttnn.sigmoid_accurate(s_o)
        b = ttnn.multiply(s_o, b)
        a = ttnn.add(a, b)
        a_t = self.transition(a, s)
        a = ttnn.add(a, a_t)
        return a


class DiffusionTransformer(Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.layers = [
            DiffusionTransformerLayer(
                dim,
                n_heads,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_layers)
        ]
        self.z = None

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        if self.z is None:
            self.z = z
        for layer in self.layers:
            a = layer(a, s, self.z)
        return a


class PairWeightedAveraging(Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.m_norm_weight = self.torch_to_tt("norm_m.weight")
        self.m_norm_bias = self.torch_to_tt("norm_m.bias")
        self.z_norm_weight = self.torch_to_tt("norm_z.weight")
        self.z_norm_bias = self.torch_to_tt("norm_z.bias")
        self.m_weight = self.torch_to_tt("proj_m.weight")
        self.g_weight = self.torch_to_tt("proj_g.weight")
        self.z_weight = self.torch_to_tt("proj_z.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")

    def __call__(self, m: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.reshape(m, tuple(m.shape)[1:])
        z = ttnn.reshape(z, tuple(z.shape)[1:])
        m = ttnn.layer_norm(
            m,
            weight=self.m_norm_weight,
            bias=self.m_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        z = ttnn.layer_norm(
            z,
            weight=self.z_norm_weight,
            bias=self.z_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        for i in range(self.n_heads):
            b = ttnn.linear(
                z,
                self.z_weight[:, i : i + 1],
                compute_kernel_config=self.compute_kernel_config,
            )
            b = ttnn.permute(b, (2, 0, 1))
            w = ttnn.softmax(
                b,
                dim=-1,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=self.compute_kernel_config.math_fidelity,
                    math_approx_mode=self.compute_kernel_config.math_approx_mode,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=self.compute_kernel_config.packer_l1_acc,
                ),
                numeric_stable=True,
            )
            v = ttnn.linear(
                m,
                self.m_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
            )
            o = ttnn.linear(v, w, transpose_a=True, compute_kernel_config=self.compute_kernel_config)
            del v, w
            o = ttnn.permute(o, (0, 2, 1))
            g = ttnn.linear(
                m,
                self.g_weight[:, i * self.head_dim : (i + 1) * self.head_dim],
                compute_kernel_config=self.compute_kernel_config,
            )
            g = ttnn.sigmoid_accurate(g)
            o = ttnn.multiply(o, g)
            del g
            o = ttnn.linear(
                o,
                self.o_weight[i * self.head_dim : (i + 1) * self.head_dim, :],
                compute_kernel_config=self.compute_kernel_config,
            )
            if i == 0:
                o_out = o
            else:
                o_out = ttnn.add(o_out, o)
        o_out = ttnn.reshape(o_out, (1, *o_out.shape))
        return o_out


class OuterProductMean(Module):
    def __init__(
        self,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.a_weight = self.torch_to_tt("proj_a.weight")
        self.b_weight = self.torch_to_tt("proj_b.weight")
        self.o_weight = self.torch_to_tt("proj_o.weight")
        self.o_bias = self.torch_to_tt("proj_o.bias")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.reshape(x, tuple(x.shape)[1:])
        m = ttnn.layer_norm(
            x,
            weight=self.norm_weight,
            bias=self.norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        a = ttnn.linear(m, self.a_weight, compute_kernel_config=self.compute_kernel_config)
        b = ttnn.linear(m, self.b_weight, compute_kernel_config=self.compute_kernel_config)
        S, I, C = a.shape
        _, J, D = b.shape
        chunks = []
        for chunk_start in range(0, I, OUTER_PRODUCT_MEAN_CHUNK_SIZE):
            chunk_end = min(chunk_start + OUTER_PRODUCT_MEAN_CHUNK_SIZE, I)
            a_chunk = a[:, chunk_start:chunk_end, :]
            a_chunk = ttnn.permute(a_chunk, (1, 2, 0))
            a_chunk = ttnn.reshape(a_chunk, (-1, S))
            b = ttnn.reshape(b, (S, -1))
            z = ttnn.matmul(a_chunk, b, compute_kernel_config=self.compute_kernel_config)
            z = ttnn.reshape(z, (chunk_end - chunk_start, C, J, D))
            z = ttnn.permute(z, (0, 2, 1, 3))
            z = ttnn.reshape(z, (*tuple(z.shape)[:2], -1))
            z = ttnn.multiply(z, 1 / S)
            z = ttnn.linear(
                z,
                self.o_weight,
                bias=self.o_bias,
                compute_kernel_config=self.compute_kernel_config,
            )
            chunks.append(z)
        z = ttnn.concat(chunks, dim=0)
        z = ttnn.reshape(z, (1, *z.shape))
        return z


class MSALayer(Module):
    def __init__(
        self,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.msa_transition = Transition(True, filter_dict(state_dict, "msa_transition"), compute_kernel_config)
        self.pair_weighted_averaging = PairWeightedAveraging(
            head_dim=avg_head_dim,
            n_heads=avg_n_heads,
            state_dict=filter_dict(state_dict, "pair_weighted_averaging"),
            compute_kernel_config=compute_kernel_config,
        )
        self.outer_product_mean = OuterProductMean(
            state_dict=filter_dict(state_dict, "outer_product_mean"),
            compute_kernel_config=compute_kernel_config,
        )
        self.triangle_multiplication_start = TriangleMultiplication(
            False, filter_dict(state_dict, "tri_mul_out"), compute_kernel_config
        )
        self.triangle_multiplication_end = TriangleMultiplication(
            True, filter_dict(state_dict, "tri_mul_in"), compute_kernel_config
        )
        self.triangle_attention_start = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            False,
            filter_dict(state_dict, "tri_att_start", "mha."),
            compute_kernel_config,
        )
        self.triangle_attention_end = TriangleAttention(
            tri_att_head_dim,
            tri_att_n_heads,
            True,
            filter_dict(state_dict, "tri_att_end", "mha."),
            compute_kernel_config,
        )
        self.z_transition = Transition(True, filter_dict(state_dict, "z_transition"), compute_kernel_config)

    def __call__(self, z: ttnn.Tensor, m: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        m = ttnn.add(m, self.pair_weighted_averaging(m, z))
        m = ttnn.add(m, self.msa_transition(m))
        z = ttnn.add(z, self.outer_product_mean(m))
        z = ttnn.add(
            z,
            self.triangle_multiplication_start(z),
        )
        z = ttnn.add(
            z,
            self.triangle_multiplication_end(z),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_start(z),
        )
        z = ttnn.add(
            z,
            self.triangle_attention_end(z),
        )
        z = ttnn.add(z, self.z_transition(z))
        return z, m


class MSA(Module):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        state_dict: dict,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.s_weight = self.torch_to_tt("s_proj.weight")
        self.msa_weight = self.torch_to_tt("msa_proj.weight")
        self.blocks = [
            MSALayer(
                avg_head_dim,
                avg_n_heads,
                tri_att_head_dim,
                tri_att_n_heads,
                filter_dict(state_dict, f"layers.{i}"),
                compute_kernel_config,
            )
            for i in range(n_blocks)
        ]

    def __call__(self, z: ttnn.Tensor, m: ttnn.Tensor, emb: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.linear(
            m,
            self.msa_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        m = ttnn.add(
            m,
            ttnn.linear(
                emb,
                self.s_weight,
                compute_kernel_config=self.compute_kernel_config,
            ),
        )
        for block in self.blocks:
            z, m = block(z, m)
        return z


class TorchWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = None
        # global device
        """
        if device is None:
            ttnn.device.EnablePersistentKernelCache()  # be careful, can lead to bugs when profiling etc.
            device = ttnn.open_device(
                device_id=0,
                #dispatch_core_config=ttnn.DispatchCoreConfig(
                #    ttnn.device.DispatchCoreType.ETH, ttnn.DispatchCoreAxis.ROW
                #),
            )
            device.enable_program_cache()
        """
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _from_torch(self, x: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            x,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32 if USE_FLOAT32 else ttnn.bfloat16,
        )

    def _to_torch(self, x: ttnn.Tensor) -> torch.Tensor:
        return torch.Tensor(ttnn.to_torch(x)).to(torch.float32)

    # def __del__(self):
    #    ttnn.DumpDeviceProfiler(self.device)
    #    ttnn.close_device(self.device)


class PairformerModule(TorchWrapper):
    def __init__(
        self,
        device,
        n_blocks: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
        att_head_dim: int,
        att_n_heads: int,
    ):
        self.device = device
        super().__init__()
        self.n_blocks = n_blocks
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads
        self.att_head_dim = att_head_dim
        self.att_n_heads = att_n_heads

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = Pairformer(
            self.device,
            self.n_blocks,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            self.att_head_dim,
            self.att_n_heads,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        pair_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            self._to_torch(x)
            for x in self.module(
                self._from_torch(s),
                self._from_torch(z),
            )
        )


class DiffusionTransformerModule(TorchWrapper):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = DiffusionTransformer(
            self.n_layers,
            self.dim,
            self.n_heads,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor = None,
        to_keys=None,
        multiplicity: int = 1,
        model_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        return self._to_torch(
            self.module(
                self._from_torch(a),
                self._from_torch(s),
                self._from_torch(z) if z is not None else None,
            )
        )


class MSAModule(TorchWrapper):
    def __init__(
        self,
        n_blocks: int,
        avg_head_dim: int,
        avg_n_heads: int,
        tri_att_head_dim: int,
        tri_att_n_heads: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_head_dim = avg_head_dim
        self.avg_n_heads = avg_n_heads
        self.tri_att_head_dim = tri_att_head_dim
        self.tri_att_n_heads = tri_att_n_heads

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.module = MSA(
            self.n_blocks,
            self.avg_head_dim,
            self.avg_n_heads,
            self.tri_att_head_dim,
            self.tri_att_n_heads,
            filter_dict(state_dict, prefix[:-1]),
            self.compute_kernel_config,
        )

    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        m = torch.cat(
            [
                feats["msa"],
                feats["has_deletion"].unsqueeze(-1),
                feats["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        return self._to_torch(
            self.module(
                self._from_torch(z),
                self._from_torch(m),
                self._from_torch(emb),
            )
        )
