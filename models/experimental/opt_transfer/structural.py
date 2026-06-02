from models.experimental.opt_transfer.schema import FusionProposal, KBEntry, PatternKind
from models.experimental.opt_transfer.trace import TracedGraph


def validate(graph: TracedGraph, proposal: FusionProposal, entry: KBEntry) -> tuple:
    nodes = []
    for name in proposal.matched_nodes:
        try:
            nodes.append(graph.by_name(name))
        except StopIteration:
            return False, f"matched node '{name}' not in graph"

    if entry.pattern_kind == PatternKind.HORIZONTAL_MERGE:
        if any(len(n.inputs) != 1 for n in nodes):
            return False, "horizontal_merge branch has != 1 input"
        shared = {n.inputs[0] for n in nodes}
        if len(shared) != 1:
            return False, f"branches do not share one input: {shared}"
        return True, "ok"

    # CHAIN: matched nodes must form a contiguous producer->consumer chain
    for prev, cur in zip(proposal.matched_nodes, proposal.matched_nodes[1:]):
        if prev not in graph.by_name(cur).inputs:
            return False, f"chain broken: {cur} does not consume {prev}"
    return True, "ok"
