import ttnn
import torch
from loguru import logger
import re
import os
import math
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer, CLIPModel

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from typing import Union, List


def convert_from_ttnn(x):
    global device
    if isinstance(x, ttnn._ttnn.tensor.Tensor):
        return ttnn.to_torch(x)
    return x


def save_weight(tensor, filename):
    if isinstance(tensor, ttnn._ttnn.tensor.Tensor):
        tensor = convert_from_ttnn(tensor)
    torch.save(tensor, filename)


def open_ttnn():
    global device
    device = ttnn.open_device(device_id=0, l1_small_size=8192)


def close_ttnn():
    global device
    if device is not None:
        ttnn.close_device(device)


def get_device():
    global device
    return device


def to_ttnn(torch_tensor, dtype=None, layout=ttnn.TILE_LAYOUT):
    global device
    ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=layout, dtype=dtype)
    return ttnn_tensor


def to_torch_shape(ttnn_shape):
    return tuple(ttnn_shape)


# Change dtype of ttnn tensor and optionally reshape
def convert_ttnn_dtype(ttnn_tensor, dtype, new_shape=None):
    # HACK: Can't convert dtype on device
    device = get_device()
    host_tensor = ttnn.from_device(ttnn_tensor)
    host_tensor = ttnn.to_dtype(host_tensor, dtype=dtype)
    if new_shape is not None:
        host_tensor = ttnn.reshape(host_tensor, new_shape)

    return ttnn.to_device(host_tensor, device=device)


def calculate_pcc(tensor, reference_tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    if reference_tensor.dtype == torch.bfloat16:
        reference_tensor = reference_tensor.to(torch.float32)
    np_calculated = tensor.detach().flatten().numpy()
    np_golden = reference_tensor.detach().flatten().numpy()
    pcc = np.corrcoef(np_calculated, np_golden)[0, 1]
    return pcc


def assert_with_pcc(torch_tensor, reference_tensor, pcc_threshold=0.99):
    pcc = calculate_pcc(torch_tensor, reference_tensor)
    assert pcc > pcc_threshold, f"PCC is below threshold: {pcc} < {pcc_threshold}"

    return pcc


class LogAccuracy:
    def __init__(self, output_file):
        self.output_file = output_file

        self.write_header()

        self.cnt = 0

    def write_header(self):
        self.output_file.write(
            f"id,step_name,pcc,ulp_error,relative_difference_between_means,relative_difference_between_std\n"
        )

    def log(self, golden_tensor, calculated_tensor, step_name):
        pcc = calculate_pcc(calculated_tensor, golden_tensor)

        # ULP Error (Mean)
        ulp_error = 0.0  # TODO

        # Relative Difference between Means
        relative_difference_mean = torch.abs(calculated_tensor.mean() - golden_tensor.mean()) / torch.abs(
            golden_tensor.mean()
        )

        # Relative Difference between Standard Deviations
        relative_difference_std = torch.abs(calculated_tensor.std() - golden_tensor.std()) / torch.abs(
            golden_tensor.std()
        )
        self.output_file.write(
            f"{self.cnt},{step_name},{pcc},{ulp_error},{relative_difference_mean},{relative_difference_std}\n"
        )
        self.cnt += 1


def compare_with_reference(tensor, reference_tensor_file, checkpoint_name=None, accuracy_logger=None):
    assert tensor is not None
    if isinstance(tensor, ttnn.Tensor):
        tensor = ttnn.to_torch(tensor)

    with torch.no_grad():
        try:
            reference_tensor = torch.load(reference_tensor_file)
        except FileNotFoundError:
            print(f"⚠️ {checkpoint_name}: Reference tensor not found at {reference_tensor_file}, skipping test")
            return

    if tensor.shape != reference_tensor.shape:
        raise ValueError(f"tensor.shape: {tensor.shape}, reference_tensor.shape: {reference_tensor.shape}")
    assert tensor.shape == reference_tensor.shape

    if accuracy_logger is not None:
        accuracy_logger.log(reference_tensor, tensor, checkpoint_name)

    # pcc_threshold = 0.8
    pcc_threshold = 0.85
    pcc = calculate_pcc(tensor, reference_tensor)

    # pcc = assert_with_pcc(tensor, reference_tensor)

    if checkpoint_name is not None:
        if pcc < pcc_threshold:
            print(f"❌  {checkpoint_name} failed - PCC = {pcc}")
        else:
            print(f"✅  {checkpoint_name} passed - PCC = {pcc}")


class Transformer:
    def __init__(self, state_dict, heads, attention_mask=None, prefix="", accuracy_logger=None):
        self.layers = []
        self.heads = heads
        self.attention_mask = attention_mask
        self.prefix = prefix

        self.accuracy_logger = accuracy_logger

        layer_pattern = re.compile(f"{prefix}\.layers\.(\d+)\.")

        # Count number of layers
        layers_ids = set()
        for k in state_dict.keys():
            re_match = re.search(layer_pattern, k)
            if re_match:
                layers_ids.add(re_match.group(1))

        num_layers = len(layers_ids)

        for i in range(0, num_layers):
            resblock_prefix = f"{prefix}.layers.{i}"

            self.layers.append(
                {
                    "ln_1_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.layer_norm1.weight"], ttnn.bfloat16
                    ),
                    "ln_1_bias": convert_ttnn_dtype(state_dict[f"{resblock_prefix}.layer_norm1.bias"], ttnn.bfloat16),
                    "q_proj_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.q_proj.weight"], ttnn.bfloat16
                    ),
                    "q_proj_bias": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.q_proj.bias"], ttnn.bfloat16
                    ),
                    "k_proj_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.k_proj.weight"], ttnn.bfloat16
                    ),
                    "k_proj_bias": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.k_proj.bias"], ttnn.bfloat16
                    ),
                    "v_proj_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.v_proj.weight"], ttnn.bfloat16
                    ),
                    "v_proj_bias": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.v_proj.bias"], ttnn.bfloat16
                    ),
                    "out_proj_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.out_proj.weight"], ttnn.bfloat16
                    ),
                    "out_proj_bias": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.self_attn.out_proj.bias"], ttnn.bfloat16
                    ),
                    "ln_2_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.layer_norm2.weight"], ttnn.bfloat16
                    ),
                    "ln_2_bias": convert_ttnn_dtype(state_dict[f"{resblock_prefix}.layer_norm2.bias"], ttnn.bfloat16),
                    "mlp_c_fc_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.mlp.fc1.weight"], ttnn.bfloat16
                    ),
                    "mlp_c_fc_bias": convert_ttnn_dtype(state_dict[f"{resblock_prefix}.mlp.fc1.bias"], ttnn.bfloat16),
                    "mlp_c_proj_weight": convert_ttnn_dtype(
                        state_dict[f"{resblock_prefix}.mlp.fc2.weight"], ttnn.bfloat16
                    ),
                    "mlp_c_proj_bias": convert_ttnn_dtype(state_dict[f"{resblock_prefix}.mlp.fc2.bias"], ttnn.bfloat16),
                }
            )

    def forward(self, x):
        def mlp(x, layer):
            # print(f"x.shape: {x.shape}, laywer['mlp_c_fc_weight'].shape: {layer['mlp_c_fc_weight'].shape}, layer['mlp_c_fc_bias'].shape: {layer['mlp_c_fc_bias'].shape}")
            x = ttnn.linear(x, layer["mlp_c_fc_weight"], bias=layer["mlp_c_fc_bias"], transpose_b=True)
            x = ttnn.gelu(x)
            x = ttnn.linear(x, layer["mlp_c_proj_weight"], bias=layer["mlp_c_proj_bias"], transpose_b=True)
            return x

        def multi_head_attention(
            hidden_states,
            fused_qkv_weight,
            fused_qkv_bias,
            self_output_weight,
            self_output_bias,
            attention_mask=None,
            prefix="",
        ):
            seq_length, batch_size, hidden_size = hidden_states.shape

            self._embed_dim = hidden_size
            self._head_dim = hidden_size // self.heads
            self._scale = self._head_dim**-0.5
            self._attention_dropout = 0.0  # Unused

            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            # TODO: No KV-caching for now
            (q_weights, k_weights, v_weights) = fused_qkv_weight
            (q_bias, k_bias, v_bias) = fused_qkv_bias

            # Compute Q, K, V projections
            q = ttnn.linear(hidden_states, q_weights, bias=q_bias, transpose_b=True)
            k = ttnn.linear(hidden_states, k_weights, bias=k_bias, transpose_b=True)
            v = ttnn.linear(hidden_states, v_weights, bias=v_bias, transpose_b=True)

            # Reshape to [batch_size, seq_length, num_heads, head_dim]
            q = ttnn.reshape(q, (seq_length, batch_size * self.heads, self._head_dim))
            k = ttnn.reshape(k, (seq_length, batch_size * self.heads, self._head_dim))
            v = ttnn.reshape(v, (seq_length, batch_size * self.heads, self._head_dim))

            # Transpose to [batch_size, num_heads, seq_length, head_dim] for attention computation
            q = ttnn.transpose(q, 0, 1)
            k = ttnn.transpose(k, 0, 1)
            v = ttnn.transpose(v, 0, 1)

            # Compute attention scores with proper scaling
            scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
            scores = scores * self._scale

            # Apply attention mask if provided (matching PyTorch MHA behavior)
            if attention_mask is not None:
                # Convert attention mask to the right shape and add to scores
                # PyTorch MHA expects mask to be broadcastable to [batch_size, num_heads, seq_len, seq_len]
                scores = scores + attention_mask

            attn_weights = ttnn.softmax(
                scores, dim=-1, numeric_stable=True, compute_kernel_config=compute_kernel_config
            )

            # Apply dropout if needed (currently disabled)
            # attn_weights = ttnn.experimental.dropout(attn_weights, self._attention_dropout)

            # Apply attention weights to values
            attn_output = ttnn.matmul(attn_weights, v)

            # Reshape to [batch_size, seq_length, embed_dim]
            attn_output = ttnn.transpose(attn_output, 0, 1)
            attn_output = ttnn.reshape(attn_output, (seq_length, batch_size, self._embed_dim))

            # Apply output projection
            dense_out = ttnn.linear(
                attn_output,
                self_output_weight,
                bias=self_output_bias,
                compute_kernel_config=compute_kernel_config,
                transpose_b=True,
            )

            return dense_out

        def residual_attention_block(x, layer, i=0):
            # LayerNorm
            residual = x
            x = ttnn.layer_norm(x, weight=layer["ln_1_weight"], bias=layer["ln_1_bias"])

            # Multihead attention / Self-Attention
            # This must be equal to nn.MultiheadAttention(d_model, n_head)(x, x, x, need_weights=False, attn_mask=self.attn_mask)
            x_attn = multi_head_attention(
                x,
                fused_qkv_weight=(layer["q_proj_weight"], layer["k_proj_weight"], layer["v_proj_weight"]),
                fused_qkv_bias=(layer["q_proj_bias"], layer["k_proj_bias"], layer["v_proj_bias"]),
                self_output_weight=layer["out_proj_weight"],
                self_output_bias=layer["out_proj_bias"],
                attention_mask=self.attention_mask,
                prefix=f"{self.prefix}.layers.{i}.attn",
            )  # Vision transformer doesn't use attention mask

            x = residual + x_attn

            # LayerNorm
            x_post_ln_2 = ttnn.layer_norm(x, weight=layer["ln_2_weight"], bias=layer["ln_2_bias"])

            # Multi-Layer Perceptron
            x = x + mlp(x_post_ln_2, layer)

            return x

        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = residual_attention_block(x, layer, i)

        return x


class VisionTransformer:
    def __init__(self, state_dict, prefix="", accuracy_logger=None):
        torch.manual_seed(0)
        self.output_dim = 0
        self.accuracy_logger = accuracy_logger

        conv2_state_dict_name = "vision_model.embeddings.patch_embedding.weight"
        self.vision_width = state_dict[conv2_state_dict_name].shape[0]
        self.patch_size = state_dict[conv2_state_dict_name].shape[-1]
        self.vision_heads = self.vision_width // 64

        self.class_embedding = convert_ttnn_dtype(
            state_dict["vision_model.embeddings.class_embedding"], dtype=ttnn.bfloat16
        )  # TODO: What's this ?
        self.positional_embedding = convert_ttnn_dtype(
            state_dict["vision_model.embeddings.position_embedding.weight"], dtype=ttnn.bfloat16
        )  # TODO: What's this ?

        scale = self.vision_width**-0.5

        self.proj = convert_ttnn_dtype(state_dict["visual_projection.weight"], dtype=ttnn.bfloat16)

        # Weights for convolution layer
        # For sharding; use all cores; strategy = block sharding
        core_grid = ttnn.CoreGrid(x=8, y=8)
        # Error: Physical shard shape (8216, 4) must be tile {32, 32} sized
        # memory_config = ttnn.create_sharded_memory_config(conv1_weights_shape, core_grid, ttnn.ShardStrategy.HEIGHT)
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.conv1_weights = ttnn.to_layout(
            state_dict[conv2_state_dict_name],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
        )
        self.conv1_weights = convert_ttnn_dtype(self.conv1_weights, dtype=ttnn.bfloat16)

        assert self.conv1_weights.dtype == ttnn.bfloat16

        self.ln_pre_weights = state_dict["vision_model.pre_layrnorm.weight"]  # TODO: What's this ?
        self.ln_pre_bias = state_dict["vision_model.pre_layrnorm.bias"]  # TODO: What's this ?

        self.ln_post_weights = state_dict["vision_model.post_layernorm.weight"]  # TODO: What's this ?
        self.ln_post_bias = state_dict["vision_model.post_layernorm.bias"]  # TODO: What's this ?

        self.transformer = Transformer(
            state_dict,
            self.vision_heads,
            attention_mask=None,
            prefix="vision_model.encoder",
            accuracy_logger=self.accuracy_logger,
        )

    def forward(self, x):
        (batch_size, in_channels, height, width) = x.shape

        # Note: ttnn.conv2d uses 'Array of Struct' shape for input tensor:
        # (N, H, W, C_in)
        # whereas torch.nn.Conv2d uses 'Struct of Array' shape for input tensor:
        # (N, C_in, H, W)
        #
        # # Moreover, ttnn.conv2d produces a flattened output tensor:
        # (N, C_in, H, W) -> (1, 1, N * H * W, C_out)
        # whereas torch.nn.Conv2d produces a 4D tensor:
        # (N, C_out, H_out, W_out)

        # Also:
        # ttnn.conv2d only take a tuple for kernel_size and stride

        # Change tensor layout to (N, H, W, C_in)
        x = ttnn.permute(x, [0, 2, 3, 1])  # (N, C_in, H, W) -> (N, H, W, C_in)

        # Note: ttnn.conv2d requires row-major layout for weight tensor
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

        old_memory_config = x.memory_config()
        out_channels = 768
        in_channels = 3
        # torch_small_conv1_weights = torch.rand((out_channels, in_channels, 32, 32), dtype=torch.bfloat16)
        # small_conv1_weights = ttnn.from_torch(torch_small_conv1_weights, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        bias_tensor = ttnn.zeros(
            (1, 1, 1, out_channels), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        assert x.dtype == ttnn.bfloat16
        assert self.conv1_weights.dtype == ttnn.bfloat16
        assert bias_tensor.dtype == ttnn.bfloat16

        x = ttnn.conv2d(
            input_tensor=x,  # should always be named argument
            weight_tensor=self.conv1_weights,
            bias_tensor=bias_tensor,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            padding=(0, 0),
            dilation=(1, 1),
            groups=0,  # No grouped convolution (?)
            device=get_device(),
            return_weights_and_bias=False,
            return_output_dim=False,
        )

        # ERROR: Number of shards along height 7 must not exceed number of cores 2
        output_height = height // self.patch_size
        output_width = width // self.patch_size

        # Check Convolution result
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        host_tensor = ttnn.to_torch(x, dtype=torch.float32)

        assert host_tensor.shape == (1, 1, batch_size * output_height * output_width, out_channels)
        host_tensor = torch.reshape(host_tensor, (batch_size, output_height, output_width, out_channels))

        assert host_tensor.shape == (batch_size, output_height, output_width, out_channels)
        x_reshaped = torch.permute(host_tensor, [0, 3, 1, 2])  # (N, H, W, C_in) -> (N, C_in, H, W)
        assert x_reshaped.shape == (batch_size, out_channels, output_height, output_width)

        compare_with_reference(x_reshaped, "visual.conv1(x).pt", "conv1", accuracy_logger=self.accuracy_logger)

        x = ttnn.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))

        class_embedding = convert_ttnn_dtype(self.class_embedding, x.dtype, (x.shape[0], 1, x.shape[-1]))

        # TODO: See why we use zero tensor here and addition here ?
        zero_tensor = ttnn.zeros(
            shape=(x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=device, layout=ttnn.TILE_LAYOUT
        )

        class_embedding = ttnn.reshape(class_embedding, zero_tensor.shape)
        class_embedding = class_embedding + zero_tensor

        # TODO: Do this in L1 Sharded Memory
        # For now, move data to DRAM
        x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print(f"x.shape: {x.shape}, class_embedding.shape: {class_embedding.shape}")
        # print(f"padded shape = {ttnn.pad_to_tile_shape(x.shape)}")

        # core_grid = ttnn.device.get_device_core_grid(device)
        # padded_shape = ttnn.pad_to_tile_shape(x.shape)
        # print(f"padded shape = {padded_shape}")
        # memory_config = ttnn.create_sharded_memory_config(padded_shape, core_grid=core_grid, strategy=ttnn.ShardStrategy.WIDTH)
        # x = ttnn.reshard(x, memory_config)

        class_embedding = ttnn.reshape(
            class_embedding, (class_embedding.shape[0], class_embedding.shape[1], class_embedding.shape[2])
        )
        class_embedding = ttnn.to_memory_config(class_embedding, memory_config=x.memory_config())

        # ERROR: RuntimeError: bad optional access (???)
        # TODO: Avoid move to host and keep on device
        x = ttnn.concat([class_embedding, x], dim=1, memory_config=None)  # shape = [*, grid ** 2 + 1, width]

        compare_with_reference(
            x, "visual.cat(x, class_embedding).pt", "ttnn.concat", accuracy_logger=self.accuracy_logger
        )

        positional_embedding = convert_ttnn_dtype(self.positional_embedding, x.dtype, (1, x.shape[1], x.shape[2]))
        x = x + positional_embedding

        # LayerNorm
        x = ttnn.layer_norm(x, weight=self.ln_pre_weights, bias=self.ln_pre_bias)

        # Permute
        x = ttnn.permute(x, (1, 0, 2))  # NLD -> LND

        compare_with_reference(
            x, "visual.pre_layer_norm(x).pt", "ttnn.layer_norm", accuracy_logger=self.accuracy_logger
        )

        # Transformer
        x = self.transformer.forward(x)

        compare_with_reference(
            x, "visual.transformer(x).pt", "visual.transformer(x).pt", accuracy_logger=self.accuracy_logger
        )

        # Permute
        x = ttnn.permute(x, (1, 0, 2))  # LND -> NLD

        # LayerNorm
        x = ttnn.layer_norm(x[:, 0, :], weight=self.ln_post_weights, bias=self.ln_post_bias)

        compare_with_reference(x, "visual.ln_post(x).pt", "visual.ln_post(x).pt", accuracy_logger=self.accuracy_logger)

        assert self.proj is not None
        if self.proj is not None:
            self.proj = ttnn.transpose(self.proj, 0, 1)
            x = ttnn.matmul(x, self.proj)

        compare_with_reference(
            x, "visual.transformer-final(x).pt", "visual.transformer-final(x).pt", accuracy_logger=self.accuracy_logger
        )

        return x


class CLIP:
    def __init__(self, state_dict, accuracy_logger=None):
        self.accuracy_logger = accuracy_logger

        self.token_embedding = convert_ttnn_dtype(
            state_dict["text_model.embeddings.token_embedding.weight"], dtype=ttnn.bfloat16
        )
        self.positional_embedding = convert_ttnn_dtype(
            state_dict["text_model.embeddings.position_embedding.weight"], dtype=ttnn.bfloat16
        )

        self.text_projection = convert_ttnn_dtype(state_dict["text_projection.weight"], dtype=ttnn.bfloat16)
        self.context_length = self.positional_embedding.shape[0]
        self.vocab_size = self.token_embedding.shape[0]
        self.transformer_width = state_dict["text_model.final_layer_norm.weight"].shape[0]
        transformer_heads = self.transformer_width // 64

        self.ln_final_weights = state_dict["text_model.final_layer_norm.weight"]
        self.ln_final_bias = state_dict["text_model.final_layer_norm.bias"]

        self.logit_scale = state_dict["logit_scale"].item()

        self.visual = VisionTransformer(state_dict, accuracy_logger)

        self.transformer = Transformer(
            state_dict,
            transformer_heads,
            attention_mask=self.build_attention_mask(),
            prefix="text_model.encoder",
            accuracy_logger=accuracy_logger,
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        def init_weights(shape, std, dtype=None):
            torch_weights = torch.empty(to_torch_shape(shape))
            torch.nn.init.normal_(torch_weights, std=std)
            return ttnn.from_torch(torch_weights, device=device, layout=ttnn.TILE_LAYOUT)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf

        # TODO: Switch this to TTNN
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        mask = ttnn.from_torch(mask, device=get_device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return mask

    def encode_image(self, image):
        return self.visual.forward(image)

    def encode_text(self, tokens):
        tokens = convert_ttnn_dtype(tokens, dtype=ttnn.uint32)

        compare_with_reference(
            self.token_embedding,
            "text.token_embedding.weight.pt",
            "text.token_embedding.weight",
            accuracy_logger=self.accuracy_logger,
        )
        compare_with_reference(
            self.positional_embedding,
            "text.positional_embedding.pt",
            "text.positional_embedding",
            accuracy_logger=self.accuracy_logger,
        )

        x = ttnn.embedding(tokens, weight=self.token_embedding, dtype=ttnn.bfloat16)

        assert x.dtype == ttnn.bfloat16

        compare_with_reference(
            x, "text.token_embedding(text).pt", "ttnn.embedding", accuracy_logger=self.accuracy_logger
        )

        # Add positional embedding
        x = x + self.positional_embedding

        compare_with_reference(
            x,
            "text.token_embedding(text)+positional_embedding.pt",
            "ttnn.embedding",
            accuracy_logger=self.accuracy_logger,
        )

        # Permute
        x = ttnn.permute(x, (1, 0, 2))  # NLD -> LND
        x = self.transformer.forward(x)

        compare_with_reference(
            x, "text.transformer(x).pt", "text.transformer(x).pt", accuracy_logger=self.accuracy_logger
        )

        # Permute back
        x = ttnn.permute(x, (1, 0, 2))  # LND -> NLD

        compare_with_reference(
            x, "text.transformer(x)+permute.pt", "text.transformer(x)+permute.pt", accuracy_logger=self.accuracy_logger
        )

        # LayerNorm
        x = ttnn.layer_norm(x, weight=self.ln_final_weights, bias=self.ln_final_bias)

        compare_with_reference(
            self.ln_final_weights,
            "text.final_layer_norm.weight.pt",
            "text.final_layer_norm.weight",
            accuracy_logger=self.accuracy_logger,
        )
        compare_with_reference(
            self.ln_final_bias,
            "text.final_layer_norm.bias.pt",
            "text.final_layer_norm.bias",
            accuracy_logger=self.accuracy_logger,
        )
        compare_with_reference(x, "text.ln_final(x).pt", "text.final_layer_norm", accuracy_logger=self.accuracy_logger)

        # TODO: Change to TTNN
        # text_projection = ttnn.transpose(self.text_projection, -2, -1)

        torch_tokens = ttnn.to_torch(tokens)
        text_projection = ttnn.to_torch(self.text_projection)
        torch_x = ttnn.to_torch(x)

        torch_x = torch_x[torch.arange(torch_x.shape[0]), torch_tokens.argmax(dim=-1)] @ text_projection.t()

        compare_with_reference(
            torch_x, "text.encode_text(x).pt", "text.encode_text(x).pt", accuracy_logger=self.accuracy_logger
        )
        return ttnn.from_torch(torch_x, device=get_device(), layout=ttnn.TILE_LAYOUT)

        torch_x = torch_x[torch.arange(torch_x.shape[0]), torch_tokens.argmax(dim=-1)] @ text_projection.t()

        return ttnn.from_torch(torch_x, device=get_device(), layout=ttnn.TILE_LAYOUT)

        # TODO: Fix the following
        tokens = ttnn.to_layout(tokens, layout=ttnn.ROW_MAJOR_LAYOUT)  # argmax only support row major

        ttnn_arange = ttnn.arange(x.shape[0], dtype=ttnn.int32)
        ttnn_argmax = ttnn.argmax(tokens, dim=-1)
        ttnn_argmax = convert_ttnn_dtype(ttnn_argmax, ttnn.int32)

        ttnn_argmax = ttnn.to_device(ttnn_argmax, device=get_device())
        ttnn_arange = ttnn.to_device(ttnn_arange, device=get_device())

        index_size = ttnn_arange.shape[0] + ttnn_argmax.shape[0]
        assert len(ttnn_arange.shape) == 1
        assert len(ttnn_argmax.shape) == 1

        ttnn_arange = ttnn.reshape(ttnn_arange, (1, 1, 1, -1))
        ttnn_argmax = ttnn.reshape(ttnn_argmax, (1, 1, 1, -1))

        ttnn_index = ttnn.concat([ttnn_arange, ttnn_argmax], dim=-1)

        ttnn_index = ttnn.to_layout(ttnn_index, layout=ttnn.TILE_LAYOUT)

        ttnn_gather = ttnn.gather(x, dim=0, index=ttnn_index)

        ttnn_index = ttnn.reshape(ttnn_index, [index_size])

        x = ttnn.matmul(ttnn_gather, self.text_projection)

        compare_with_reference(
            x, "text.encode_text(x).pt", "text.encode_text(x).pt", accuracy_logger=self.accuracy_logger
        )

        return x

    def forward(self, image, tokens):
        text_features = self.encode_text(tokens)
        image_features = self.encode_image(image)

        compare_with_reference(
            image_features, "encode_image(image).pt", "encode_image(image).pt", accuracy_logger=self.accuracy_logger
        )
        # compare_with_reference(text_features, "encode_text(tokens).pt", "encode_text(tokens).pt", accuracy_logger=self.accuracy_logger)

        # Normalize features
        norm_image_features = ttnn.operations.moreh.norm(image_features, p=2.0, dim=1, keepdim=True)
        norm_text_features = ttnn.operations.moreh.norm(text_features, p=2.0, dim=1, keepdim=True)

        image_features = ttnn.divide(image_features, norm_image_features)
        text_features = ttnn.divide(text_features, norm_text_features)

        compare_with_reference(
            image_features, "norm_image_features.pt", "norm_image_features.pt", accuracy_logger=self.accuracy_logger
        )
        compare_with_reference(
            text_features, "norm_text_features.pt", "norm_text_features.pt", accuracy_logger=self.accuracy_logger
        )

        # Cosine similarity as logits
        logit_scale = math.exp(self.logit_scale)

        text_features_t = ttnn.transpose(text_features, 0, 1)
        logits_per_image = logit_scale * image_features @ text_features_t
        logits_per_text = ttnn.transpose(logits_per_image, 0, 1)

        return logits_per_image, logits_per_text


def print_state_dict(state_dict):
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            if len(value.shape) == 0:
                print(f"{key}: {value.item()}")
            else:
                print(f"{key}: {value.shape}, {value.dtype}")


def load_model_as_ttnn(model_path):
    state_dict = {}
    with open(model_path, "rb") as model_file:
        logger.info(f"Loading model from {model_path}")

        model = torch.jit.load(model_file, map_location="cpu").eval()
        state_dict = model.state_dict()

        print_state_dict(state_dict)

        # Convert state dict to ttnn tensors
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = to_ttnn(value)
            elif isinstance(value, torch.Size):
                state_dict[key] = ttnn.Size(value)

    return state_dict


def convert_model_to_ttnn(state_dict):
    ttnn_state_dict = {}
    logger.info(f"Converting model to ttnn")

    print_state_dict(state_dict)

    # Convert state dict to ttnn tensor
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = to_ttnn(value)
        elif isinstance(value, torch.Size):
            state_dict[key] = ttnn.Size(value)

    return state_dict


def preprocess_image(image, model_resolution):
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    # Pre-process image on host with torch
    transform_fn = Compose(
        [
            Resize(model_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(model_resolution),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    return transform_fn(image)


if __name__ == "__main__":
    open_ttnn()

    logging_file = open("logging.csv", "w")
    accuracy_logger = LogAccuracy(logging_file)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    state_dict = convert_model_to_ttnn(model.state_dict())

    clip = CLIP(state_dict, accuracy_logger)

    root_dir = "/localdev/nmaurice/CLIP-tt"
    image_path = os.path.join(root_dir, "CLIP.png")

    # Preprocess image
    image = Image.open(image_path)
    image = preprocess_image(image, 224).unsqueeze(0).to("cpu")

    preferred_dtype = ttnn.bfloat16
    tt_image = to_ttnn(image, preferred_dtype)

    prompts = ["a diagram", "a dog", "a cat"]

    # Tokenize text
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokenized_inputs = tokenizer(prompts, padding="max_length", max_length=clip.context_length, return_tensors="pt")
    tokens_pretrained_host = tokenized_inputs["input_ids"]
    tokens_pretrained = ttnn.from_torch(tokens_pretrained_host, device=get_device(), layout=ttnn.TILE_LAYOUT)

    logits_per_image, logits_per_text = clip.forward(tt_image, tokens_pretrained)
    probs = ttnn.softmax(logits_per_image, dim=-1)
    print(f"Label probs: {probs}")

    logging_file.close()

    close_ttnn()
