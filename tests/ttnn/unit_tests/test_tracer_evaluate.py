# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers

import ttnn

import networkx as nx


def evaluate(*outputs, input_node_to_tensor_map):
    flattened_graph = ttnn.tracer.get_graph(outputs, flatten=True)
    node_to_tensor_map = input_node_to_tensor_map

    for node in nx.topological_sort(flattened_graph):
        if node in node_to_tensor_map:
            continue

        input_nodes = [None for _ in flattened_graph.in_edges(node)]
        for input_node, _, edge_data in flattened_graph.in_edges(node, data=True):
            input_nodes[edge_data["sink_input_index"]] = input_node

        input_tensors = [node_to_tensor_map[input_node] for input_node in input_nodes]

        operation = flattened_graph.nodes[node]["operation"]
        if isinstance(operation, ttnn.tracer.TorchFunction):
            arg_name_value_pairs = operation.arg_name_value_pairs

            function_args = []
            function_kwargs = {}
            for arg_name, arg in arg_name_value_pairs:
                if isinstance(arg_name, ttnn.tracer.PositionalArgumentName):
                    if isinstance(arg, ttnn.tracer.InputTensorIndex):
                        function_args.append(input_tensors[arg.index])
                    else:
                        function_args.append(arg)
                elif not isinstance(arg_name, ttnn.tracer.PositionalArgumentName):
                    if isinstance(arg, ttnn.tracer.InputTensorIndex):
                        function_kwargs[arg_name] = input_tensors[arg.index]
                    else:
                        function_kwargs[arg_name] = arg

            node_to_tensor_map[node] = operation.function(*function_args, **function_kwargs)

        elif isinstance(operation, ttnn.tracer.TorchParameter):
            node_to_tensor_map[node] = operation.parameter

        elif isinstance(operation, ttnn.tracer.TorchTensor):
            node_to_tensor_map[node] = operation.tensor

        else:
            raise ValueError(f"Unknown operation type: {operation}")

    output_tensors = []
    for node in flattened_graph:
        if flattened_graph.out_degree(node) == 0:
            output_tensors.append(node_to_tensor_map[node])

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


@pytest.mark.requires_fast_runtime_mode_off
def test_evaluate_traced_torch_bert():
    model_name = "bert-base-uncased"
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 3
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()

    with ttnn.tracer.trace():
        input_tensor = torch.randint(0, 1000, (1, 128))
        outputs = model(input_tensor)

    evaluated_output_tensors = evaluate(
        outputs.pooler_output, input_node_to_tensor_map={input_tensor.node: input_tensor}
    )
    evaluated_output_tensor = evaluated_output_tensors[1]
    assert torch.allclose(outputs.pooler_output, evaluated_output_tensor)
