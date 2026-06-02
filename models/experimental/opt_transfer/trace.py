from dataclasses import dataclass
import torch
import torch.fx as fx


@dataclass
class OpNode:
    name: str
    kind: str  # "linear" | "layer_norm" | "softmax" | call_function name | ...
    inputs: list  # names of producer nodes


class TracedGraph:
    def __init__(self, gm: fx.GraphModule):
        self.gm = gm
        self._nodes = self._summarize()

    def _summarize(self):
        mods = dict(self.gm.named_modules())
        out = []
        for n in self.gm.graph.nodes:
            if n.op == "call_module":
                kind = type(mods[n.target]).__name__.lower()
            elif n.op == "call_function":
                kind = getattr(n.target, "__name__", str(n.target))
            elif n.op == "call_method":
                kind = n.target
            else:
                kind = n.op
            out.append(OpNode(name=n.name, kind=kind, inputs=[a.name for a in n.all_input_nodes]))
        return out

    def nodes(self):
        return list(self._nodes)

    def by_name(self, name: str):
        return next(n for n in self._nodes if n.name == name)

    def summary_json(self):
        return [{"name": n.name, "kind": n.kind, "inputs": n.inputs} for n in self._nodes]


def trace_module(module: torch.nn.Module, example_inputs: tuple) -> TracedGraph:
    gm = fx.symbolic_trace(module)
    return TracedGraph(gm)
