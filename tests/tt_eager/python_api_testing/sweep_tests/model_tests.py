import torch
import ttnn
from ttnn.tracer import trace, visualize
from ttnn.model_preprocessing import preprocess_model
from IPython.display import SVG
import transformers


torch.manual_seed(0)

# TORCH MODELS ===================================================================


class TorchConvConv(torch.nn.Module):
    def __init__(
        self,
        batch_size=1,
        num_input_channels=64,
        num_output_channels=64,
        kernel_size=(3, 3),
        padding=(1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_input_channels, num_output_channels, kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(num_output_channels, num_output_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class TorchConvReluConv(torch.nn.Module):
    def __init__(
        self,
        batch_size=1,
        num_input_channels=128,
        num_output_channels=128,
        kernel_size=(3, 3),
        padding=(1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_input_channels, num_output_channels, kernel_size, padding=padding)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(num_output_channels, num_output_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class BertFeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = transformers.models.bert.modeling_bert.BertIntermediate(config)
        self.output = transformers.models.bert.modeling_bert.BertOutput(config)

    def forward(self, hidden_states):
        intermediate = self.intermediate(hidden_states)
        hidden_states = self.output(intermediate, hidden_states)
        return hidden_states


# TTNN MODELS ==============================================================


class TTNNConvConv:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.conv1 = parameters.conv1
        self.conv2 = parameters.conv2

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TTNNConvReluConv:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.conv1 = parameters.conv1
        self.conv2 = parameters.conv2

    def __call__(self, x):
        out = self.conv1(x)
        out = ttnn.relu(out)
        out = self.conv2(out)

        return out


# HELPER FUNCTIONS ============================================================


def run_conv_conv(model, torch_input_tensor):
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    input_tensor = model.conv1.copy_input_to_device(input_tensor)
    output_tensor = model(input_tensor)
    output_tensor = model.conv2.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = torch.reshape(output_tensor, torch_input_tensor.shape)
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    return output_tensor


def run_conv_relu_conv(model, torch_input_tensor, device):
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = model.conv1.copy_input_to_device(input_tensor)
    output_tensor = model.conv1(output_tensor)
    output_tensor = model.conv1.copy_output_from_device(output_tensor)
    output_tensor = ttnn.to_device(output_tensor, device=device)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.relu(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = model.conv2.copy_input_to_device(output_tensor)
    output_tensor = model.conv2(output_tensor)
    output_tensor = model.conv2.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = torch.reshape(output_tensor, torch_input_tensor.shape)
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    return output_tensor


# from models.demos.bert.tt import ttnn_bert
# from models.demos.bert.tt import ttnn_optimized_bert
# from ttnn.model_preprocessing import preprocess_model_parameters

# def run_ttnn_bert_model():
#     model_name = "phiyodr/bert-large-finetuned-squad2"

#     config = transformers.BertConfig.from_pretrained(model_name)
#     config.num_hidden_layers = 1

#     batch_size = 8
#     sequence_size = 384

#     parameters = preprocess_model_parameters(
#         initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
#             model_name, config=config
#         ).eval(),
#         custom_preprocessor=bert.custom_preprocessor,
#         device=device,
#     )

#     with trace():
#         input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
#         torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
#         torch_attention_mask = torch.zeros(1, sequence_size) if bert == ttnn_optimized_bert else None


#         ttnn_bert_inputs = bert.preprocess_inputs(
#             input_ids,
#             torch_token_type_ids,
#             torch_attention_mask,
#             device=device,
#         )

#         output = bert.bert_for_question_answering(
#             config,
#             *ttnn_bert_inputs,
#             parameters=parameters,
#         )
#         output = ttnn.from_device(output)

#     return output
