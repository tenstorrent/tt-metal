"""Dump the scout's **DiT stage** graph to JSON, for diffing against a real device run.

Phase A of the whole-pipeline fidelity check (galaxy workstream 3). The scout builds the real
`MiniMaxH3Transformer3DModel` under the metadata shim; this writes that stage's graph out so
`conform_h3_dit.py` can run the *same* model on real silicon and diff the collectives it actually
fires against the ones the shim believes it fires.

Only the DiT stage: it is the stage with every fused kernel and both mesh axes in play, and the
one the analyzer's findings mostly concern.

    cd models/tt_dit/tools/dit_analyzer/scouts
    DITCHECK_DIT_LAYERS=1 DITCHECK_REFINER_LAYERS=1 \\
    PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/models/tt_dit/tools \\
      python3 dump_dit_graph.py 4x8 dit.graph.json

Needs the H3 model code (see ../GALAXY_PLAN.md § "Where the code lives").
"""

import runpy
import sys

preset = sys.argv[1] if len(sys.argv) > 1 else "4x8"
out = sys.argv[2] if len(sys.argv) > 2 else "dit.graph.json"

sys.argv = ["scout_h3_pipeline.py", preset]
ns = runpy.run_path("scout_h3_pipeline.py")
graph = ns["dit_graph"]

with open(out, "w") as fh:
    fh.write(graph.to_json())
print("\nwrote %s: %d nodes (preset %s)" % (out, len(graph.nodes), preset))
