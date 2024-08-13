# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
    return torch.max(x, dim=dim[0], keepdim=True).values


def min(x, *args, **kwargs):
    dim = kwargs.pop("dim")
    return torch.min(x, dim=dim[0], keepdim=True).values


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
