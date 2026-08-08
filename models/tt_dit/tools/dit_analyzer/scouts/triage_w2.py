"""Trace the workstream-2 findings — the classes the Galaxy has never device-checked.

Workstream 1 conformed three findings; the rest of the report (`replicated_stage`,
`overwide_gather`, `participant_shrink`) is un-tainted `likely` but has only ever been
believed. GALAXY_PLAN's operating rule says triage on the laptop first and spend silicon
only on survivors, so this dumps, for each flagged node: what it moves, who consumes it,
and per-device what arrived versus what downstream actually demands. A finding whose
"got" exceeds "needed" for a *modelling* reason (padding the shim counts but consumers
read) is an artifact to fix in the tool; one where the gap is real data is a survivor.

    cd models/tt_dit/tools/dit_analyzer/scouts
    PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/models/tt_dit/tools python3 triage_w2.py

Needs the H3 model code (see ../GALAXY_PLAN.md § "Where the code lives").
"""

import json
import runpy
import sys

sys.argv = ["scout_h3_pipeline.py", "prod"]
ns = runpy.run_path("scout_h3_pipeline.py")  # builds the connected `linked` graph
linked = ns["linked"]

sys.path.insert(0, "models/tt_dit/tools")
from dit_analyzer import analyze_graph  # noqa: E402
from dit_analyzer.analysis import analyze_dataflow  # noqa: E402

rep = analyze_graph(linked)
fwd, bwd = analyze_dataflow(linked)

# The un-conformed classes, by node. The output-head cluster (344/352/356) is left out:
# conform_dit_heads already proved it on silicon at 4x8.
# Fully qualified — a bare "node_92" also matches encoder/layernorm_node_92 and vae/pointwise_node_63.
TARGETS = {
    "overwide_gather": [
        "dit/all_gather_node_274",
        "dit/all_gather_node_296",
        "dit/reduce_scatter_node_302",
    ],
    "replicated_stage (DiT text branch)": [
        "dit/all_gather_node_63",
        "dit/all_gather_node_92",
        "dit/all_gather_node_100",
        "dit/reduce_scatter_node_110",
    ],
}


def sh(sid):
    s = linked.symbols.get(sid)
    return list(s.shape) if s else "?"


def consumers(o):
    return [(n.id, n.op) for n in linked.nodes if o in n.inputs]


def findings_on(nid):
    return [f for f in rep.findings if nid in f.nodes]


for cls, ids in TARGETS.items():
    print("\n\n" + "=" * 100)
    print("CLASS: %s" % cls)
    print("=" * 100)
    for nid in ids:
        for n in linked.nodes:
            if n.id != nid:
                continue
            out = n.outputs[0]
            fo = fwd.final[out]
            dem = bwd.demand.get(out)
            oshape = linked.symbols[out].shape
            print("\n--- %s (%s) mesh_axis=%s" % (n.id, n.op, n.mesh_axis))
            print("    loc:      %s" % (n.loc or "?"))
            print("    input     %s %s dist=%s" % (n.inputs[0], sh(n.inputs[0]), fwd.final[n.inputs[0]].dist))
            print("    output    %s %s dist=%s" % (out, sh(out), fo.dist))
            print("    consumers %s" % consumers(out))
            # Where the claim lives: per device, what landed vs what anything downstream reads.
            for d in list(linked.mesh.devices())[:4]:
                need = dem[d].describe(oshape) if dem and d in dem else "(nothing)"
                print("      dev %2d  got=%s" % (d, fo.regions[d].describe(oshape)))
                print("              needed=%s" % need)
            for f in findings_on(nid):
                print("    [%s/%s] %s" % (f.severity, f.confidence, f.rule))
                for r in f.reason:
                    print("        - %s" % r)
                print("        proof: %s" % json.dumps(f.proof, default=str)[:400])
