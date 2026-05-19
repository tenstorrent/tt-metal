# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Plotly HTML report — the diagnostic UI.

Layout (deliberately mirrored by the Dash app so users see the same view
both as a static file and as a live tool):

  +------------------------------------------------------------------+
  |  topbar: run id, model, box, baseline toggle, compare label      |
  +-------------------+---------------------------+------------------+
  | LEFT RAIL         | MAIN PANE                 | RIGHT INSPECTOR  |
  |  nav: block tree  |  tabs:                    |  cluster details |
  |  catalog          |   1 Roofline              |  region label    |
  |   - 9 blocks      |   2 SoL bars              |  ranked blocks   |
  |   - click -> page |   3 Per-RISC stack        |  copy-CLI buttons|
  |                   |   4 Per-block stacks      |                  |
  |                   |   5 Config scatter        |                  |
  |                   |   6 Waterfall             |                  |
  |                   |   7 Cache heatmap         |                  |
  |                   |   8 Kernel hash diff      |                  |
  +-------------------+---------------------------+------------------+

This module owns the *shell*; the charts themselves come from
`perf/charts/*.py`. When a chart is not yet implemented we render a
neutral placeholder so the navigation still works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go

from .ceilings import BoxSpec
from .cluster import Cluster
from .join import JoinedRow

# Charts (some are no-ops at this milestone but the imports keep the
# interface stable).
from .charts.theme import apply_theme


@dataclass
class ReportContext:
    """Everything the report needs to render. Computed once by build_report()."""

    run_id: str
    run_dir: Path
    model_id: str
    box: BoxSpec
    rows: List[JoinedRow]
    clusters: List[Cluster]
    baseline_run_id: Optional[str] = None
    baseline_rows: Optional[List[JoinedRow]] = None
    baseline_clusters: Optional[List[Cluster]] = None
    extra_meta: Dict[str, object] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# HTML scaffolding
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: Inter, -apple-system, system-ui, sans-serif;
       background: #0d1117; color: #c9d1d9; }
.topbar { padding: 10px 20px; background: #161b22; border-bottom: 1px solid #30363d;
          display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.topbar .title { font-size: 16px; font-weight: 600; color: #f0f6fc; }
.topbar .meta { color: #8b949e; font-size: 12px; }
.topbar .meta b { color: #c9d1d9; font-weight: 500; }
.layout { display: grid; grid-template-columns: 240px 1fr 380px; height: calc(100vh - 50px); }
.rail { background: #161b22; border-right: 1px solid #30363d; padding: 12px; overflow: auto; }
.rail h4 { margin: 12px 0 6px; color: #f0f6fc; font-size: 12px; text-transform: uppercase;
           letter-spacing: 0.05em; }
.rail .item { padding: 6px 8px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.rail .item:hover { background: #21262d; }
.rail .item.active { background: #1f6feb33; color: #58a6ff; }
.main { overflow: auto; padding: 0; }
.tabs { display: flex; gap: 4px; padding: 8px 12px; background: #0d1117;
        border-bottom: 1px solid #30363d; position: sticky; top: 0; z-index: 10; }
.tab { padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;
       border: 1px solid transparent; color: #8b949e; }
.tab.active { background: #1f6feb33; color: #58a6ff; border-color: #1f6feb55; }
.tab:hover { background: #21262d; color: #c9d1d9; }
.tab-content { display: none; padding: 16px; }
.tab-content.active { display: block; }
.inspector { background: #161b22; border-left: 1px solid #30363d; padding: 16px;
             overflow: auto; font-size: 12px; }
.inspector h3 { margin: 0 0 12px 0; color: #f0f6fc; font-size: 14px; }
.inspector .empty { color: #8b949e; font-style: italic; }
.kv { display: grid; grid-template-columns: 130px 1fr; gap: 4px 12px; margin-bottom: 12px; }
.kv .k { color: #8b949e; }
.kv .v { color: #c9d1d9; font-family: JetBrains Mono, ui-monospace, monospace; font-size: 11px;
         word-break: break-all; }
.cli { background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
       padding: 6px 8px; font-family: JetBrains Mono, ui-monospace, monospace;
       font-size: 11px; word-break: break-all; }
.copy-btn { background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
            padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; }
.copy-btn:hover { background: #30363d; }
.copy-btn:active { background: #1f6feb33; }
.suggestion { border: 1px solid #30363d; border-radius: 4px; padding: 8px;
              margin-bottom: 8px; background: #0d1117; }
.suggestion .blk { color: #58a6ff; font-weight: 600; font-size: 12px; }
.suggestion .desc { color: #8b949e; font-size: 11px; margin: 4px 0; }
.region-pill { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px;
               font-weight: 600; color: #0d1117; }
.placeholder { border: 1px dashed #30363d; border-radius: 6px; padding: 40px;
               text-align: center; color: #8b949e; }
.catalog-block { padding: 6px 8px; border-radius: 4px; cursor: pointer; font-size: 12px;
                 display: flex; justify-content: space-between; align-items: center; }
.catalog-block:hover { background: #21262d; }
.catalog-block .lvl { color: #8b949e; font-size: 10px; }
"""


def _placeholder_div(label: str) -> str:
    return (
        f'<div class="placeholder"><p style="margin:0;font-size:14px;color:#c9d1d9">{label}</p>'
        f'<p style="margin:8px 0 0 0">Will be rendered by the next chart PR.</p></div>'
    )


def _figure_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def _topbar(ctx: ReportContext) -> str:
    baseline = ""
    if ctx.baseline_run_id:
        baseline = f' <span class="meta">vs baseline <b>{ctx.baseline_run_id}</b></span>'
    return (
        f'<div class="topbar">'
        f'<div><span class="title">tt_hw_planner perf — {ctx.model_id}</span>'
        f' <span class="meta">run <b>{ctx.run_id}</b></span>'
        f' <span class="meta">box <b>{ctx.box.name}</b> mesh '
        f"<b>{ctx.box.mesh_shape[0]}x{ctx.box.mesh_shape[1]}</b></span>"
        f"{baseline}</div>"
        f'<div class="meta">{len(ctx.rows)} ops · {len(ctx.clusters)} clusters</div>'
        f"</div>"
    )


def _left_rail(ctx: ReportContext, blocks: List[str], catalog_entries: List[Tuple[str, int, str]]) -> str:
    items = "".join(
        f'<div class="item" data-block="{b}" onclick="window.ttSelectBlock(this)">{b}</div>' for b in blocks
    )
    catalog_items = "".join(
        (
            f'<div class="catalog-block" data-block-name="{name}" '
            f'onclick="window.ttShowBlockHelp(this)">'
            f'<span>{name}</span><span class="lvl">L{level}</span></div>'
        )
        for (name, level, _desc) in catalog_entries
    )
    return (
        f'<div class="rail">'
        f"<h4>Blocks</h4>"
        f'<div class="item active" data-block="" onclick="window.ttSelectBlock(this)">All</div>'
        f"{items}"
        f"<h4>Optimizer catalog</h4>"
        f"{catalog_items}"
        f"</div>"
    )


def _tabs(charts: Dict[str, str]) -> str:
    """`charts` is an ordered dict {tab_id: chart_html}. The first tab is active."""
    keys = list(charts.keys())
    tabs = "".join(
        f'<div class="tab{(" active" if i == 0 else "")}" data-tab="{k}" '
        f'onclick="window.ttSelectTab(this)">{k.replace("_", " ").title()}</div>'
        for i, k in enumerate(keys)
    )
    panes = "".join(
        f'<div class="tab-content{(" active" if i == 0 else "")}" data-tab-pane="{k}">{html}</div>'
        for i, (k, html) in enumerate(charts.items())
    )
    return f'<div class="main"><div class="tabs">{tabs}</div>{panes}</div>'


def _inspector_placeholder() -> str:
    return (
        '<div class="inspector" id="tt-inspector">'
        "<h3>Inspector</h3>"
        '<div class="empty">Click a point on any chart to populate.</div>'
        "</div>"
    )


def _clusters_json(ctx: ReportContext) -> str:
    payload = []
    for c in ctx.clusters:
        sample = next((r for r in ctx.rows if r.cluster_id == c.cluster_id), None)
        payload.append(
            {
                "cluster_id": c.cluster_id,
                "op_code": c.op_code,
                "args_hash": c.args_hash,
                "compute_kernel_hash": c.compute_kernel_hash,
                "shape_signature": c.shape_signature,
                "math_fidelity": c.math_fidelity,
                "n_calls": c.n_calls,
                "median_device_ns": c.median_device_ns,
                "total_device_ns": c.total_device_ns,
                "pct_of_peak": c.percent_of_peak,
                "mean_fpu_util_pct": c.mean_fpu_util_pct,
                "mean_dram_bw_util_pct": c.mean_dram_bw_util_pct,
                "mean_noc_util_pct": c.mean_noc_util_pct,
                "mean_eth_bw_util_pct": c.mean_eth_bw_util_pct,
                "program_cache_hit_rate": c.program_cache_hit_rate,
                "blocks": c.blocks,
                "arguments_example": c.arguments_example,
                "region": sample.region if sample else None,
                "region_reason": sample.region_reason if sample else None,
                "compute_kernel_source": sample.compute_kernel_source if sample else "",
                "dm_kernel_source": sample.dm_kernel_source if sample else "",
                "brisc_ns": sample.brisc_ns if sample else None,
                "ncrisc_ns": sample.ncrisc_ns if sample else None,
                "trisc0_ns": sample.trisc0_ns if sample else None,
                "trisc1_ns": sample.trisc1_ns if sample else None,
                "trisc2_ns": sample.trisc2_ns if sample else None,
                "erisc_ns": sample.erisc_ns if sample else None,
            }
        )
    return json.dumps(payload)


def _block_help_json(catalog_entries: List[Tuple[str, int, str]]) -> str:
    return json.dumps({name: {"level": level, "description": desc} for name, level, desc in catalog_entries})


# Cross-chart click + tab JS. Kept inline so the report is self-contained.
_JS_TEMPLATE = """
<script>
(function() {
  var clusters = __CLUSTERS_JSON__;
  var blockHelp = __BLOCK_HELP_JSON__;
  var clusterById = {};
  clusters.forEach(function(c) { clusterById[c.cluster_id] = c; });

  function renderInspectorFromCluster(c) {
    var el = document.getElementById('tt-inspector');
    if (!el) return;
    if (!c) {
      el.innerHTML = '<h3>Inspector</h3><div class="empty">Click a point on any chart to populate.</div>';
      return;
    }
    function row(k, v) { return '<div class="k">'+k+'</div><div class="v">'+(v==null?'':v)+'</div>'; }
    function fmtNs(v) { return v==null ? '' : (v >= 1e6 ? (v/1e6).toFixed(2)+' ms' : (v/1e3).toFixed(1)+' us'); }
    function fmtPct(v) { return v==null ? '' : v.toFixed(1)+'%'; }
    var regionLabel = c.region == null ? '?' : c.region;
    var regionColor = ({A:'#2EA043',B:'#FFD93D',C:'#FFA630',D:'#1F6FEB',E:'#F85149',F:'#8B949E','?':'#586069'})[regionLabel] || '#586069';

    var riscRows = '';
    var risc = [['BRISC',c.brisc_ns],['NCRISC',c.ncrisc_ns],['TRISC0',c.trisc0_ns],['TRISC1',c.trisc1_ns],['TRISC2',c.trisc2_ns],['ERISC',c.erisc_ns]];
    var maxR = 0;
    risc.forEach(function(p){ if (p[1]!=null && p[1]>maxR) maxR=p[1]; });
    risc.forEach(function(p){
      if (p[1] == null) return;
      var pct = maxR > 0 ? (100*p[1]/maxR).toFixed(0) : 0;
      var lp = (p[1]===maxR && maxR>0) ? '<b style="color:#f85149">'+p[0]+' (long pole)</b>' : p[0];
      riscRows += '<div style="display:grid;grid-template-columns:80px 1fr 60px;gap:6px;font-family:JetBrains Mono,monospace;font-size:11px;margin-bottom:2px">'+
                  '<div style="color:#8b949e">'+lp+'</div>'+
                  '<div style="background:#21262d;height:10px;border-radius:2px;overflow:hidden"><div style="background:#58a6ff;height:100%;width:'+pct+'%"></div></div>'+
                  '<div style="text-align:right">'+fmtNs(p[1])+'</div></div>';
    });

    var suggestionsHtml = '';
    var regionSuggestions = {
      A: [],
      B: ['math_fidelity_downcast'],
      C: ['math_fidelity_downcast', 'program_config_tuner'],
      D: ['fusion_rewriter'],
      E: ['dram_l1_promoter', 'fusion_rewriter', 'layout_unifier'],
      F: ['trace_capturer', 'cache_warmer'],
      '?': []
    };
    var sugList = regionSuggestions[regionLabel] || [];
    sugList.forEach(function(s) {
      var help = blockHelp[s] || {};
      var cli = 'tt_hw_planner perf apply ' + s + ' --cluster ' + c.cluster_id;
      suggestionsHtml += '<div class="suggestion">'+
        '<div class="blk">'+s+' <span class="lvl" style="color:#8b949e;font-weight:normal">L'+(help.level||'?')+'</span></div>'+
        '<div class="desc">'+(help.description || '')+'</div>'+
        '<div class="cli">'+cli+'</div>'+
        '<button class="copy-btn" onclick="window.ttCopy(this, \\''+cli+'\\')">Copy</button>'+
        '</div>';
    });
    if (suggestionsHtml === '') {
      suggestionsHtml = '<div class="empty">No applicable optimizer block for region '+regionLabel+'.</div>';
    }

    var ks = '';
    ks += row('region', '<span class="region-pill" style="background:'+regionColor+'">'+regionLabel+'</span> '+(c.region_reason||''));
    ks += row('op', c.op_code);
    ks += row('cluster_id', c.cluster_id);
    ks += row('n_calls', c.n_calls);
    ks += row('median', fmtNs(c.median_device_ns));
    ks += row('total', fmtNs(c.total_device_ns));
    ks += row('% of peak', c.pct_of_peak==null?'':c.pct_of_peak.toFixed(1)+'%');
    ks += row('FPU util', fmtPct(c.mean_fpu_util_pct));
    ks += row('DRAM util', fmtPct(c.mean_dram_bw_util_pct));
    ks += row('NoC util', fmtPct(c.mean_noc_util_pct));
    ks += row('ETH util', fmtPct(c.mean_eth_bw_util_pct));
    ks += row('cache hit', c.program_cache_hit_rate==null?'':((100*c.program_cache_hit_rate).toFixed(0))+'%');
    ks += row('args_hash', c.args_hash || '(tracer missing)');
    ks += row('kernel_hash', c.compute_kernel_hash || '(none)');
    ks += row('compute src', c.compute_kernel_source || '(none)');
    ks += row('DM src', c.dm_kernel_source || '(none)');
    ks += row('blocks', (c.blocks||[]).join(', '));

    el.innerHTML =
      '<h3>Cluster ' + c.cluster_id + '</h3>'+
      '<div class="kv">'+ks+'</div>'+
      '<h4 style="margin:10px 0 6px;color:#f0f6fc;font-size:12px">Per-RISC duration</h4>'+
      riscRows+
      '<h4 style="margin:14px 0 6px;color:#f0f6fc;font-size:12px">Suggested optimizer blocks</h4>'+
      suggestionsHtml +
      '<h4 style="margin:14px 0 6px;color:#f0f6fc;font-size:12px">Arguments (model_tracer)</h4>'+
      '<div class="cli">'+JSON.stringify(c.arguments_example||{}, null, 2).replace(/</g,'&lt;')+'</div>';
  }

  window.ttCopy = function(btn, txt) {
    navigator.clipboard.writeText(txt).then(function(){
      var prev = btn.innerText;
      btn.innerText = 'Copied!';
      setTimeout(function(){ btn.innerText = prev; }, 1200);
    });
  };

  window.ttShowBlockHelp = function(el) {
    var name = el.getAttribute('data-block-name');
    var help = blockHelp[name] || {};
    var ins = document.getElementById('tt-inspector');
    if (!ins) return;
    ins.innerHTML =
      '<h3>'+name+' <span style="color:#8b949e;font-size:11px">L'+(help.level||'?')+'</span></h3>'+
      '<div class="empty">'+(help.description || '')+'</div>'+
      '<h4 style="margin:14px 0 6px;color:#f0f6fc;font-size:12px">CLI</h4>'+
      '<div class="cli">tt_hw_planner perf blocks show '+name+'</div>'+
      '<button class="copy-btn" onclick="window.ttCopy(this, \\'tt_hw_planner perf blocks show '+name+'\\')">Copy</button>';
  };

  window.ttSelectTab = function(el) {
    var tab = el.getAttribute('data-tab');
    document.querySelectorAll('.tab').forEach(function(t){ t.classList.toggle('active', t.getAttribute('data-tab')===tab); });
    document.querySelectorAll('.tab-content').forEach(function(p){ p.classList.toggle('active', p.getAttribute('data-tab-pane')===tab); });
    setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 50);
  };

  window.ttSelectBlock = function(el) {
    var block = el.getAttribute('data-block');
    document.querySelectorAll('.rail .item').forEach(function(i){ i.classList.toggle('active', i.getAttribute('data-block')===block); });
    window.ttFilterByBlock = block;
    var evt = new CustomEvent('tt-filter-changed', { detail: { block: block } });
    window.dispatchEvent(evt);
  };

  window.ttSelectCluster = function(cluster_id) {
    renderInspectorFromCluster(clusterById[cluster_id]);
  };

  document.querySelectorAll('.js-plotly-plot').forEach(function(div) {
    div.on && div.on('plotly_click', function(ev) {
      if (!ev || !ev.points || !ev.points.length) return;
      var pt = ev.points[0];
      var cid = pt.customdata && pt.customdata[0];
      if (cid) window.ttSelectCluster(cid);
    });
  });
})();
</script>
"""


def render_report_html(
    ctx: ReportContext, charts: Dict[str, go.Figure], catalog_entries: List[Tuple[str, int, str]]
) -> str:
    """Build the final HTML string from the chart figures + ctx.

    `charts` keys become tab names (lowercase, underscored).
    """
    # Convert figures to divs (one shared plotly.js bundle).
    chart_htmls: Dict[str, str] = {}
    for i, (key, fig) in enumerate(charts.items()):
        if fig is None:
            chart_htmls[key] = _placeholder_div(key.replace("_", " ").title())
        else:
            apply_theme(fig, dark=True)
            chart_htmls[key] = _figure_html(fig, div_id=f"chart-{key}")

    blocks = sorted({r.block_path for r in ctx.rows})
    body = (
        _topbar(ctx)
        + '<div class="layout">'
        + _left_rail(ctx, blocks, catalog_entries)
        + _tabs(chart_htmls)
        + _inspector_placeholder()
        + "</div>"
    )

    js = _JS_TEMPLATE.replace("__CLUSTERS_JSON__", _clusters_json(ctx)).replace(
        "__BLOCK_HELP_JSON__", _block_help_json(catalog_entries)
    )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>tt_hw_planner perf — {ctx.run_id} — {ctx.model_id}</title>"
        '<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>'
        f"<style>{_CSS}</style>"
        "</head><body>"
        f"{body}"
        f"{js}"
        "</body></html>"
    )
    return html


def write_report(
    ctx: ReportContext,
    charts: Dict[str, go.Figure],
    catalog_entries: List[Tuple[str, int, str]],
    path: Optional[Path] = None,
) -> Path:
    if path is None:
        path = ctx.run_dir / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_html(ctx, charts, catalog_entries))
    return path
