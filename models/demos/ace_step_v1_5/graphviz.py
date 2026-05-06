"""
Minimal stub for `graphviz` python package.

Some TTNN builds import `graphviz` for graph/report utilities even when the user
doesn't use graph rendering. Bring-up environments may not have the external
dependency installed; this stub satisfies the import.
"""

from __future__ import annotations


class Digraph:  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def node(self, *args, **kwargs) -> None:
        return

    def edge(self, *args, **kwargs) -> None:
        return

    def render(self, *args, **kwargs):
        return None


class Graph(Digraph):  # pragma: no cover
    pass
