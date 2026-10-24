"""The PCC pass marks, defined once.

The tool gates correctness at two different tiers and they are deliberately not
the same number:

  * COMPONENT — each ported submodule against its HF reference. Strict, because
    a single module has nothing upstream of it to blame.
  * E2E — the whole chained forward against HF. Looser, because per-component
    error accumulates across the chain, so the composed model is allowed to be
    slightly less exact than any one part of it.

Both tiers used to be restated at each site that needed them, and the e2e tier
drifted: the emit-e2e command gated at 0.95 while the synthesis loop gated at
0.99, so which bar a run had to clear depended on which path produced the demo.
Import from here instead of restating a literal, so the two tiers can only ever
be changed in one place.

These are DEFAULTS. `--pcc-target` and the *_MCP_PCC environment variables
still override them per run; this module only decides what applies when nothing
is passed.
"""

from __future__ import annotations

# Per-component gate. Also stated in the flow diagram's per-component subgraph
# and in README.md's glossary; keep those in step if this ever changes.
COMPONENT_PCC = 0.99

# End-to-end gate for the whole chained forward. README.md's glossary documents
# this tier as the contract, which is why it is the one the drift resolved to.
E2E_PCC = 0.95
