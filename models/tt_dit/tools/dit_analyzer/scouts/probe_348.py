"""Trace node_348 in the *connected* pipeline: the DiT output-head TP all_gather of hidden
(participant_shrink, 664 MiB). Dump its proof and the demand its consumers place on it."""
import json
import runpy
import sys

sys.argv = ["scout_h3_pipeline.py", "prod"]
ns = runpy.run_path("scout_h3_pipeline.py")  # builds the connected `linked` graph (prints its report)
linked = ns["linked"]

sys.path.insert(0, "models/tt_dit/tools")
from dit_analyzer import analyze_graph  # noqa: E402
from dit_analyzer.analysis import analyze_dataflow  # noqa: E402

rep = analyze_graph(linked)
fwd, bwd = analyze_dataflow(linked)


def sh(sid):
    s = linked.symbols.get(sid)
    return list(s.shape) if s else "?"


def consumers(o):
    return [(n.id, n.op) for n in linked.nodes if o in n.inputs]


tgt = [n for n in linked.nodes if n.op == "all_gather" and n.loc and "transformer_minimax_h3.py:410" in n.loc]
print("\n\n<<<<<< node_348 — DiT output-head TP all_gather of hidden (@ :410) >>>>>>")
for n in tgt:
    out = n.outputs[0]
    fi, fo = fwd.final[n.inputs[0]], fwd.final[out]
    dem = bwd.demand.get(out)
    devs = list(linked.mesh.devices())
    print("\nNODE %s  mesh_axis=%s" % (n.id, n.mesh_axis))
    print("  input  %s %s dist=%s" % (n.inputs[0], sh(n.inputs[0]), fi.dist))
    print("  output %s %s dist=%s" % (out, sh(out), fo.dist))
    print("  consumers: %s" % consumers(out))
    oshape = linked.symbols[out].shape
    print("  per-device got vs needed-downstream:")
    for d in devs:
        need = dem[d].describe(oshape) if dem and d in dem else "?"
        print("    dev %2d: got=%s  needed=%s" % (d, fo.regions[d].describe(oshape), need))

print("\n<<<<<< participant_shrink / unused / dead findings on node_348 >>>>>>")
for f in rep.findings:
    if any("node_348" in nid for nid in f.nodes):
        print("\n[%s/%s] %s :: %s" % (f.severity, f.confidence, f.rule, f.title))
        for r in f.reason:
            print("  -", r)
        print("  proof:", json.dumps(f.proof, indent=2, default=str))
