# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Collective-redundancy static analyzer for DiT forward graphs (prototype).

    from dit_analyzer import load_graph, analyze_graph
    from dit_analyzer.report import render_report

    report = analyze_graph(load_graph("example:sd35_block_double_gather"))
    print(render_report(report, states=True))

The package is pure Python and imports no ttnn, so it runs on a laptop against
captured graph dumps. Only :mod:`dit_analyzer.capture` needs a live ttnn.
"""

from __future__ import annotations

import os

from .analysis import analyze_dataflow, run_backward, run_forward
from .builder import GraphBuilder, Value
from .ir import Dist, Graph, Mesh, Node, Placement, TensorSymbol
from .region import Box, RegionSet
from .rules import Finding, Report, run_rules

__all__ = [
    "Box",
    "Dist",
    "Finding",
    "Graph",
    "GraphBuilder",
    "Mesh",
    "Node",
    "Placement",
    "RegionSet",
    "Report",
    "TensorSymbol",
    "Value",
    "analyze_dataflow",
    "analyze_graph",
    "load_graph",
    "run_backward",
    "run_forward",
    "run_rules",
]


def analyze_graph(graph: Graph) -> Report:
    """Forward availability + backward demand + redundancy rules."""
    forward, backward = analyze_dataflow(graph)
    return run_rules(graph, forward, backward)


def load_graph(spec: str) -> Graph:
    """Load ``example:<name>`` or a graph JSON file."""
    if spec.startswith("example:"):
        from .examples import load

        return load(spec.split(":", 1)[1])
    if not os.path.exists(spec):
        raise SystemExit("no such graph: %s (try 'example:<name>', see `ditcheck examples`)" % spec)
    with open(spec) as fh:
        return Graph.from_json(fh.read())
