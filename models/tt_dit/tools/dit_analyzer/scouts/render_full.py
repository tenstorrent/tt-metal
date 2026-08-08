"""Render the full connected H3 pipeline report (all findings)."""
import runpy
import sys

sys.argv = ["scout_h3_pipeline.py", "prod"]
ns = runpy.run_path("scout_h3_pipeline.py")
linked = ns["linked"]
sys.path.insert(0, "models/tt_dit/tools")
from dit_analyzer import analyze_graph  # noqa: E402
from dit_analyzer.report import render_report  # noqa: E402

print("\n\n<<<<<< FULL REPORT (all findings) >>>>>>")
print(render_report(analyze_graph(linked), top=30, proof=False))
