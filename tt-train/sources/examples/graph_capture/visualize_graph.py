# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import json


def visualize(trace_name, path):
    with open(f"{path}/{trace_name}.json", "r") as f:
        trace = json.load(f)
        ttnn.graph.pretty_print(trace)
        ttnn.graph.visualize(trace, file_name=f"{path}/{trace_name}.svg")


if __name__ == "__main__":
    path = "/home/ubuntu/graph_traces"
    visualize("backward_trace", path)
    visualize("forward_trace", path)
