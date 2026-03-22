# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
import torch
from tests.tt_eager.python_api_testing.sweep_tests.model_tests import TorchConvConv, TorchConvReluConv, BertFeedForward
import transformers
from loguru import logger


def var_global(x, *args, **kwargs):
    dim = kwargs.pop("dim")
    return torch.var(x, dim, keepdim=True)


def std_global(x, *args, **kwargs):
    dim = kwargs.pop("dim")
    return torch.std(x, dim, keepdim=True)


def mean_global(x, *args, **kwargs):
    dim = kwargs.pop("dim")
    return torch.mean(x, dim, keepdim=True)


def prelu(x, *args, **kwargs):
    weight = kwargs.pop("scalar")
    t_weight = torch.ones([1], dtype=x.dtype) * weight
    result = torch.nn.functional.prelu(x, t_weight)
    return result


def max(x, *args, **kwargs):
    dim = kwargs.pop("dim")
    return torch.max(x, dim=dim[0]).values


def min(x, *args, **kwargs):
    dim = kwargs.pop("dim")
    return torch.min(x, dim=dim[0]).values


def eltwise_max(x, y, *args, **kwargs):
    return torch.maximum(x, y)


def eltwise_min(x, y, *args, **kwargs):
    return torch.minimum(x, y)


def embeddings(x, y, *args, **kwargs):
    x = x.int()
    x = torch.clamp(x, min=0, max=y.shape[0] - 1)
    z = torch.nn.functional.embedding(x, y)
    return z


def layernorm_weights_bias(x, y, z, *args, **kwargs):
    w = x.shape[1]
    torch_output_tensor = torch.nn.functional.layer_norm(x, normalized_shape=[w], weight=y, bias=z)
    return torch_output_tensor


def layernorm_weights_bias_residual(x, y, z, w, *args, **kwargs):
    width = x.shape[1]
    torch_output_tensor = torch.nn.functional.layer_norm(x + y, normalized_shape=[width], weight=z, bias=w)
    return torch_output_tensor


def layernorm_noweights(x, *args, **kwargs):
    w = x.shape[1]
    torch_output_tensor = torch.nn.functional.layer_norm(x, normalized_shape=[w])
    return torch_output_tensor


def attention_softmax_nomask(x, *args, **kwargs):
    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax)
    torch_output_tensor = golden_function(
        x,
        head_size=None,
        attention_mask=None,
    )

    return torch_output_tensor


def attention_softmax(x, y, *args, scalar, **kwargs):
    y[y <= 0.50] = 0
    y[y > 0.50] = 1
    if scalar < 0:
        scalar = -scalar

    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax)
    torch_output_tensor = golden_function(
        x,
        head_size=None,
        attention_mask=y,
    )

    return torch_output_tensor


def transformer_concatenate_heads(x, *args, **kwargs):
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    return golden_function(x)


def rmsnorm(hidden_states, weight, epsilon=1e-6, *args, **kwargs):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(weight.dtype)

    return weight * hidden_states


def groupnorm(x, y, z, *args, **kwargs):
    torch_output_tensor = torch.nn.functional.group_norm(x, num_groups=1, weight=y, bias=z)
    return torch_output_tensor


def preprocessing_model_conv_conv(x, *args, **kwargs):
    torch.manual_seed(234)
    num_channels = x.shape[1]

    torch_model = TorchConvConv(num_input_channels=num_channels, num_output_channels=num_channels)
    torch_model.eval()

    torch_input_tensor = x.to(torch.float32)
    output = torch_model(torch_input_tensor)

    return output


def preprocessing_model_conv_relu_conv(x, *args, **kwargs):
    torch.manual_seed(234)
    torch_input_tensor = x.to(torch.float32)
    num_channels = x.shape[1]

    torch_model = TorchConvReluConv(num_input_channels=num_channels, num_output_channels=num_channels)
    torch_model.eval()

    output = torch_model(torch_input_tensor)

    return output


def signbit(x, *args, **kwargs):
    return torch.signbit(x)


def tilize(x, *args, **kwargs):
    return x


def tilize_with_zero_padding(x, *args, **kwargs):
    pad_h = (math.ceil(x.shape[-2] / 32) * 32) - x.shape[-2]
    pad_w = (math.ceil(x.shape[-1] / 32) * 32) - x.shape[-1]
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))


def tilize_with_val_padding(x, output_tensor_shape, pad_value, *args, **kwargs):
    return torch.nn.functional.pad(
        x,
        tuple(j for i in reversed(range(len(x.shape))) for j in (0, output_tensor_shape[i] - x.shape[i])),
        value=pad_value,
    )


def _simulate_bfp_quantization(x, man_bits):
    """Simulate bfp round-trip (float -> bfp -> float) on CPU.

    Args:
        x: Input tensor.
        man_bits: Number of mantissa bits (3 for bfp4, 7 for bfp8).
    """
    orig_shape = x.shape
    x_f32 = x.to(torch.float32).contiguous().reshape(-1, 16)
    x_bits = x_f32.view(torch.int32)

    sign = (x_bits >> 31) & 1
    exp = (x_bits >> 23) & 0xFF
    man = x_bits & 0x7FFFFF

    is_zero = exp == 0
    exp_bfp = torch.clamp(exp - 112, 0, 31)  # rebias: -127 + 15 = -112
    exp_bfp = torch.where(is_zero, torch.zeros_like(exp_bfp), exp_bfp)
    man_full = torch.where(is_zero, torch.zeros_like(man), man | (1 << 23))

    shared_exp = exp_bfp.max(dim=-1, keepdim=True).values
    exp_diff = (shared_exp - exp_bfp).clamp(0, 31)
    man_aligned = man_full >> exp_diff

    # Round to man_bits (banker's rounding)
    shift = 24 - man_bits
    remainder = man_aligned & ((1 << shift) - 1)
    tie = 1 << (shift - 1)
    man_n = man_aligned >> shift
    guard = man_n & 1
    man_n = man_n + ((remainder > tie) | ((remainder == tie) & (guard == 1))).to(torch.int32)
    man_n = man_n.clamp(max=(1 << man_bits) - 1)
    sign = torch.where(man_n == 0, torch.zeros_like(sign), sign)

    # Unpack: normalize mantissa (find leading zeros in man_bits-wide field)
    bfp_zero = man_n == 0
    lz = torch.zeros_like(man_n)
    for i in range(1, man_bits):
        lz = torch.where(
            (man_n >= 1) & (man_n < (1 << (man_bits - i))),
            torch.tensor(i, dtype=torch.int32),
            lz,
        )
    lz = torch.where(bfp_zero, torch.zeros_like(lz), lz)
    man_out = ((man_n << lz) << 1) & ((1 << man_bits) - 1)
    exp_out = shared_exp - lz + 112  # rebias to FP32: +127 - 15 = +112

    result = (sign << 31) | (exp_out << 23) | (man_out << (23 - man_bits))
    result = torch.where(bfp_zero, torch.zeros_like(result), result)
    return result.view(torch.float32).reshape(orig_shape).to(torch.bfloat16)


def eltwise_typecast(x, *args, tt_input_dtype, tt_output_dtype, **kwargs):
    if tt_input_dtype == ttnn.bfloat16 and tt_output_dtype == ttnn.uint16:
        return torch.clamp(x.to(torch.int32), min=0, max=65535)  # due to no uint16 support
    elif tt_input_dtype == ttnn.uint16 and tt_output_dtype == ttnn.bfloat16:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.int32 and tt_output_dtype == ttnn.bfloat16:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat16 and tt_output_dtype == ttnn.int32:
        return x.to(torch.int32)
    elif tt_input_dtype == ttnn.bfloat16 and tt_output_dtype == ttnn.float32:
        return x.to(torch.bfloat16).to(torch.float32)
    elif tt_input_dtype == ttnn.float32 and tt_output_dtype == ttnn.bfloat16:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.float32 and tt_output_dtype == ttnn.uint16:
        return torch.clamp(x.to(torch.int32), min=0, max=65535)  # due to no uint16 support
    elif tt_input_dtype == ttnn.uint16 and tt_output_dtype == ttnn.float32:
        return x.to(torch.float32)
    elif tt_input_dtype == ttnn.float32 and tt_output_dtype == ttnn.int32:
        return x.to(torch.int32)
    elif tt_input_dtype == ttnn.int32 and tt_output_dtype == ttnn.float32:
        return x.to(torch.float32)
    elif tt_input_dtype == ttnn.bfloat8_b and tt_output_dtype == ttnn.uint16:
        x = _simulate_bfp_quantization(x, 7)
        return torch.clamp(x.to(torch.int32), min=0, max=65535)  # due to no uint16 support
    elif tt_input_dtype == ttnn.uint16 and tt_output_dtype == ttnn.bfloat8_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat8_b and tt_output_dtype == ttnn.int32:
        return _simulate_bfp_quantization(x, 7).to(torch.int32)
    elif tt_input_dtype == ttnn.int32 and tt_output_dtype == ttnn.bfloat8_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat16 and tt_output_dtype == ttnn.uint32:
        return torch.relu(x.to(torch.int32))  # due to no uint32 support
    elif tt_input_dtype == ttnn.uint32 and tt_output_dtype == ttnn.bfloat16:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.float32 and tt_output_dtype == ttnn.uint32:
        return torch.relu(x.to(torch.int32))  # due to no uint32 support
    elif tt_input_dtype == ttnn.uint32 and tt_output_dtype == ttnn.float32:
        return x.to(torch.float32)
    elif tt_input_dtype == ttnn.bfloat8_b and tt_output_dtype == ttnn.uint32:
        return torch.relu(_simulate_bfp_quantization(x, 7).to(torch.int32))  # due to no uint32 support
    elif tt_input_dtype == ttnn.uint32 and tt_output_dtype == ttnn.bfloat8_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.uint16 and tt_output_dtype == ttnn.uint32:
        return x.to(torch.int32)
    elif tt_input_dtype == ttnn.uint16 and tt_output_dtype == ttnn.int32:
        return x.to(torch.int32)
    elif tt_input_dtype == ttnn.int32 and tt_output_dtype == ttnn.uint16:
        return torch.clamp(x, min=0, max=65535)
    elif tt_input_dtype == ttnn.uint32 and tt_output_dtype == ttnn.uint16:
        return torch.clamp(x, min=0, max=65535)
    elif tt_input_dtype == ttnn.bfloat8_b and tt_output_dtype == ttnn.bfloat16:
        return _simulate_bfp_quantization(x, 7)
    elif tt_input_dtype == ttnn.bfloat16 and tt_output_dtype == ttnn.bfloat8_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat8_b and tt_output_dtype == ttnn.float32:
        return _simulate_bfp_quantization(x, 7).to(torch.float32)
    elif tt_input_dtype == ttnn.float32 and tt_output_dtype == ttnn.bfloat8_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat4_b and tt_output_dtype == ttnn.uint16:
        x = _simulate_bfp_quantization(x, 3)
        return torch.clamp(x.to(torch.int32), min=0, max=65535)  # due to no uint16 support
    elif tt_input_dtype == ttnn.uint16 and tt_output_dtype == ttnn.bfloat4_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat4_b and tt_output_dtype == ttnn.int32:
        return _simulate_bfp_quantization(x, 3).to(torch.int32)
    elif tt_input_dtype == ttnn.int32 and tt_output_dtype == ttnn.bfloat4_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat4_b and tt_output_dtype == ttnn.uint32:
        return torch.relu(_simulate_bfp_quantization(x, 3).to(torch.int32))  # due to no uint32 support
    elif tt_input_dtype == ttnn.uint32 and tt_output_dtype == ttnn.bfloat4_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat4_b and tt_output_dtype == ttnn.bfloat16:
        return _simulate_bfp_quantization(x, 3)
    elif tt_input_dtype == ttnn.bfloat16 and tt_output_dtype == ttnn.bfloat4_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat4_b and tt_output_dtype == ttnn.float32:
        return _simulate_bfp_quantization(x, 3).to(torch.float32)
    elif tt_input_dtype == ttnn.float32 and tt_output_dtype == ttnn.bfloat4_b:
        return x.to(torch.bfloat16)
    elif tt_input_dtype == ttnn.bfloat4_b and tt_output_dtype == ttnn.bfloat8_b:
        return _simulate_bfp_quantization(x, 3)
    elif tt_input_dtype == ttnn.bfloat8_b and tt_output_dtype == ttnn.bfloat4_b:
        return _simulate_bfp_quantization(x, 7)
    elif tt_output_dtype == ttnn.uint8:
        if tt_input_dtype == ttnn.bfloat4_b:
            x = _simulate_bfp_quantization(x, 3)
        elif tt_input_dtype == ttnn.bfloat8_b:
            x = _simulate_bfp_quantization(x, 7)
        return x.to(torch.uint8)
    elif tt_input_dtype == ttnn.uint8:
        if tt_output_dtype == ttnn.float32:
            return x.to(torch.float32)
        elif tt_output_dtype == ttnn.bfloat16 or tt_output_dtype == ttnn.bfloat8_b or tt_output_dtype == ttnn.bfloat4_b:
            return x.to(torch.bfloat16)
        elif tt_output_dtype == ttnn.int32 or tt_output_dtype == ttnn.uint16 or tt_output_dtype == ttnn.uint32:
            return x.to(torch.int32)
    else:
        return x


def preprocessing_model_bert_1(x, *args, **kwargs):
    torch.manual_seed(234)
    model_name = "phiyodr/bert-large-finetuned-squad2"

    # get torch model
    config = transformers.BertConfig.from_pretrained(model_name)
    model = BertFeedForward(config).eval()
    model = model.to(torch.bfloat16)

    # prepare inputs
    torch_hidden_states = x

    # run model
    torch_output = model(torch_hidden_states)

    return torch_output


def preprocessing_model_bert_2(x, *args, **kwargs):
    torch.manual_seed(234)
    model_name = "phiyodr/bert-large-finetuned-squad2"

    # get torch model
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    model = transformers.models.bert.modeling_bert.BertEncoder(config).eval()

    # prepare inputs
    torch_hidden_states = x.to(torch.float32)
    torch_attention_mask = None

    # run model
    torch_output = model(torch_hidden_states, attention_mask=torch_attention_mask).last_hidden_state

    return torch_output


def preprocessing_model_bert_3(x, *args, **kwargs):
    torch.manual_seed(234)
    model_name = "phiyodr/bert-large-finetuned-squad2"

    # get torch model
    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertAttention(config).eval()
    model = model.to(torch.bfloat16)

    # prepare inputs
    torch_hidden_states = x
    sequence_size = x.shape[1]
    torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)

    # run model
    torch_output, *_ = model(torch_hidden_states, attention_mask=torch_attention_mask)

    return torch_output


def preprocessing_model_bert_4(x, *args, **kwargs):
    torch.manual_seed(0)
    model_name = "phiyodr/bert-large-finetuned-squad2"

    # set parameters
    batch_size = x.shape[0]
    sequence_size = x.shape[1]
    num_hidden_layers = 1

    # get torch model
    config = transformers.BertConfig.from_pretrained(model_name)

    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers
    else:
        logger.warning("Test mismatches when the default number of hidden layers is used")
        return None

    model = transformers.BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    # set inputs
    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = None

    # run model
    torch_output = model(
        torch_input_ids,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    )

    return torch_output.start_logits


def concat_bw(x, y, z, dim, *args, **kwargs):
    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    torch_output_tensor = golden_function(x, y, z, dim=dim)

    return torch_output_tensor
