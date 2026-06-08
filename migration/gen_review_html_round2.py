# SPDX-License-Identifier: Apache-2.0
# Generates migration/round2_review.html: a tabbed review of the Round-2 mcast_pipe API change
# (drop the MCAST mode knob; infer EXCLUDE/INCLUDE from geometry + the active/recipient count).
#   * Tab "Pipe helper (new)" — the rewritten helper + reference kernels.
#   * One tab per MIGRATED kernel — the Round-2 before/after diff (round-1 Pipe API -> round-2 API).
#   * Tab "conv-WS: why not migrated" — the loopback-inference limitation doc + the Pipe->raw revert.
# Baseline = round-1 tip (b94154a7f1d); After = HEAD. Libs vendored from migration/assets -> offline.
import subprocess, html, json, os, re

REPO = "/localdev/sjovic/tt-metal"
BASE = "b94154a7f1d"  # round-1 tip: the "before" baseline for the Round-2 API change
ASSETS = os.path.join(REPO, "migration", "assets")
OUT = os.path.join(REPO, "migration", "round2_review.html")


def sh(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO, *args], text=True)
    except subprocess.CalledProcessError:
        return ""


def read(path):
    with open(path if os.path.isabs(path) else os.path.join(REPO, path)) as f:
        return f.read()


def asset(name):
    return read(os.path.join(ASSETS, name))


def esc(s):
    return html.escape(s)


HELPER_FILES = [
    ("mcast_pipe.hpp  —  the helper (Round 2)", "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"),
    ("pipe_sender.cpp  —  reference usage (sender)", "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_sender.cpp"),
    ("pipe_receiver.cpp  —  reference usage (receiver)", "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_receiver.cpp"),
    ("pipe_f3_sender.cpp  —  loopback / degenerate", "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_f3_sender.cpp"),
]

# (family, basename, path, note) — the 11 kernels re-migrated to the Round-2 API.
MIGRATED = [
    (
        "matmul",
        "reader_bmm_tile_layout_in0_sender_padding",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        "sender in box, NOT a recipient (num_dests &lt; area) → EXCLUDE inferred",
    ),
    (
        "matmul",
        "reader_bmm_tile_layout_in0_receiver",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        "receiver: num_active_cores=1 (unused on receive)",
    ),
    (
        "matmul",
        "reader_bmm_tile_layout_in1_sender_writer_padding",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        "2× send(): num_active = in1_mcast_num_dests",
    ),
    (
        "matmul",
        "reader_bmm_tile_layout_in1_receiver_writer_padding",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        "2× receive() (in1 + bias)",
    ),
    (
        "conv",
        "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks",
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp",
        "HS 1d weights receiver",
    ),
    (
        "conv",
        "writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks",
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp",
        "BS 2d weights sender: num_active = weights_mcast_num_dests",
    ),
    (
        "conv",
        "writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks",
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp",
        "BS 2d weights receiver",
    ),
    (
        "groupnorm",
        "reader_mcast_receiver_unary_sharded_gn_v2",
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp",
        "reduce receiver (legacy)",
    ),
    (
        "groupnorm",
        "welford_reader_mcast_receiver_unary_sharded_gn_v2",
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp",
        "welford receiver",
    ),
    (
        "reduction",
        "reader_final_topk",
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp",
        "send_signal(): num_active unused; sender out of box → EXCLUDE",
    ),
    (
        "layernorm",
        "reader_mcast_sender_unary_sharded_ln",
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp",
        "phase-1 send_signal(); sender above rect → EXCLUDE",
    ),
]

# The de-migrated kernel — shown in its own explanatory tab.
WS_PATH = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp"
LIMIT_DOC = "helper_design/mcast_pipe/loopback_inference_limitation.md"

CSS = """
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#f6f8fa;color:#1f2328}
header{padding:16px 24px;border-bottom:1px solid #d0d7de;background:#fff}
header h1{margin:0;font-size:18px;font-weight:600}
header .sub{color:#656d76;font-size:13px;margin-top:4px}
.tabs{display:flex;flex-wrap:wrap;gap:4px;padding:10px 18px;background:#fff;border-bottom:1px solid #d0d7de;position:sticky;top:0;z-index:20}
.grp{display:flex;align-items:center;gap:4px;margin-right:14px;margin-bottom:4px}
.grp .lbl{color:#fff;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-right:4px;padding:2px 7px;border-radius:4px}
.grp.t-helper .lbl{background:#6e7781}.grp.t-mig .lbl{background:#1a7f37}.grp.t-gap .lbl{background:#cf222e}
.tab{padding:5px 11px;border:1px solid #d0d7de;border-radius:6px;background:#f6f8fa;color:#1f2328;cursor:pointer;font-size:12px;white-space:nowrap}
.tab:hover{background:#eaeef2;border-color:#afb8c1}
.tab.active{background:#0969da;color:#fff;border-color:#0969da;font-weight:600}
.tab.gap.active{background:#cf222e;border-color:#cf222e}
.panel{display:none;padding:18px 28px;width:100%}
.panel.active{display:block}
.kmeta{font-size:13px;margin:0 0 14px;padding:10px 14px;background:#fff;border:1px solid #d0d7de;border-radius:8px}
.kmeta code{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12px;color:#0550ae}
.kmeta b{font-size:14px}
.tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600;margin-left:8px;vertical-align:middle}
.tag.full{background:#dafbe1;color:#1a7f37;border:1px solid #aceebb}
.tag.gap{background:#ffebe9;color:#cf222e;border:1px solid #ffc1bc}
.filebox{margin:16px 0;border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#fff}
.filebox h3{margin:0;padding:9px 14px;background:#f6f8fa;font-size:13px;border-bottom:1px solid #d0d7de;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}
.filebox pre{margin:0;overflow-x:auto}
.filebox pre code.hljs{padding:14px;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#fff}
.d2h-wrapper{font-size:12.5px}.d2h-file-header{display:none}
.hint{color:#656d76;font-size:12px;margin:0 0 10px}
.doc{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:18px 24px;max-width:980px}
.doc h2{font-size:20px;margin:.2em 0 .4em;border-bottom:1px solid #eaeef2;padding-bottom:.2em}
.doc h3{font-size:16px;margin:1.1em 0 .3em}
.doc code{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12.5px;color:#0550ae}
.doc table{border-collapse:collapse;margin:12px 0;font-size:13px}
.doc th,.doc td{border:1px solid #d0d7de;padding:6px 10px;text-align:left;vertical-align:top}
.doc th{background:#f6f8fa}
.doc ul{margin:.3em 0 .6em;padding-left:1.4em}
.doc li{margin:.15em 0}
.doc .gapcard{border-left:5px solid #cf222e;background:#fff8f7}
"""

JS_TMPL = """
const DIFFS = __DIFFS_JSON__;
const drawn = {};
function show(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector('[data-for="'+id+'"]').classList.add('active');
  if(DIFFS[id] && !drawn[id]){
    const tgt = document.getElementById('d2h_'+id);
    const ui = new Diff2HtmlUI(tgt, DIFFS[id], {
      drawFileList:false, matching:'lines', outputFormat:'side-by-side',
      highlight:true, fileContentToggle:false
    }, hljs);
    ui.draw(); ui.highlightCode();
    drawn[id] = true;
  }
  window.scrollTo(0,0);
}
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.filebox pre code').forEach(el=>hljs.highlightElement(el));
});
"""


def md_to_html(md):
    """Minimal markdown -> HTML for the limitation doc (headings, tables, lists, bold, code)."""

    def inline(s):
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        return s

    out, i, lines = [], 0, md.splitlines()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            out.append(f"<h2>{inline(ln[2:])}</h2>")
        elif ln.startswith("## "):
            out.append(f"<h3>{inline(ln[3:])}</h3>")
        elif ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif (
            ln.lstrip().startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].replace("|", "").strip()) <= set("-: ")
        ):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            hdr, body = rows[0], rows[2:]
            t = ["<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in hdr) + "</tr></thead><tbody>"]
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "".join(out)


# ---- collect diffs (git unified diff, BASE..HEAD) keyed by panel id ----
diffs = {}
for fam, base, path, note in MIGRATED:
    diffs["k_" + base] = sh("diff", BASE, "HEAD", "--", path)
# keyed by PANEL id (show() draws DIFFS[panelId] into #d2h_<panelId>)
diffs["helper_diff_p"] = sh("diff", BASE, "HEAD", "--", "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp")
diffs["wsgap"] = sh("diff", BASE, "HEAD", "--", WS_PATH)
diffs_json = json.dumps(diffs).replace("</", "<\\/")

parts = []
parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
parts.append("<title>mcast_pipe Round 2 review</title>")
parts.append(f"<style>{asset('hljs-github.min.css')}</style>")
parts.append(f"<style>{asset('diff2html.min.css')}</style>")
parts.append(f"<style>{CSS}</style></head><body>")

parts.append(
    "<header><h1>mcast_pipe Round 2 &mdash; drop the multicast-mode knob, infer from geometry + active cores</h1>"
    "<div class='sub'>Baseline = round-1 tip <code>b94154a</code> &nbsp;|&nbsp; After = HEAD &nbsp;|&nbsp; "
    "11 kernels re-migrated to the new API (all mapped tests green); 1 (conv width-sharded) kept raw &amp; "
    "documented. Side-by-side diffs, C++-highlighted.</div></header>"
)

# ---- tab bar ----
tb = ["<div class='tabs'>"]
tb.append(
    "<div class='grp t-helper'><span class='lbl'>helper</span>"
    "<button class='tab active' data-for='helper' onclick=\"show('helper')\">Pipe helper (new)</button>"
    "<button class='tab' data-for='helper_diff_p' onclick=\"show('helper_diff_p')\">helper diff</button></div>"
)
tb.append("<div class='grp t-mig'><span class='lbl'>migrated</span>")
for fam, base, path, note in MIGRATED:
    short = base.replace("reader_bmm_tile_layout_", "").replace("_tiled_col_to_rm_blocks", "")
    short = short.replace("reader_mcast_", "").replace("_unary_sharded", "")
    tb.append(
        f"<button class='tab' data-for='k_{base}' onclick=\"show('k_{base}')\" title='{esc(fam)}'>{esc(short)}</button>"
    )
tb.append("</div>")
tb.append(
    "<div class='grp t-gap'><span class='lbl'>gap</span>"
    "<button class='tab gap' data-for='wsgap' onclick=\"show('wsgap')\">conv-WS: why not migrated</button></div>"
)
tb.append("</div>")
parts.append("".join(tb))

# ---- helper panel ----
hp = ["<div class='panel active' id='helper'>"]
hp.append(
    "<div class='kmeta'>The Round-2 <code>dataflow_kernel_lib::Pipe</code>. The <code>MCAST</code> mode template "
    "param and the <code>McastRect.num_dests</code> field are gone. The caller passes pure-geometry "
    "<code>McastRect</code> + <code>num_active_cores</code> (the recipient/ACK count); EXCLUDE vs INCLUDE_SRC is "
    "inferred at runtime: <code>loopback iff sender_in_rect() &amp;&amp; num_active_cores == area()</code>.</div>"
)
for title, path in HELPER_FILES:
    hp.append(
        f"<div class='filebox'><h3>{esc(title)}</h3><pre><code class='language-cpp'>{esc(read(path))}</code></pre></div>"
    )
hp.append("</div>")
parts.append("".join(hp))

# ---- helper diff panel ----
parts.append(
    "<div class='panel' id='helper_diff_p'>"
    "<div class='kmeta'><b>mcast_pipe.hpp</b> — round-1 API &rarr; round-2 API.</div>"
    "<p class='hint'>Left = round-1 (mode knob + num_dests field), right = round-2 (geometry + active cores).</p>"
    "<div id='d2h_helper_diff_p'></div></div>"
)

# ---- per-kernel diff panels ----
for fam, base, path, note in MIGRATED:
    p = [f"<div class='panel' id='k_{base}'>"]
    p.append(
        f"<div class='kmeta'><b>{esc(base)}</b><span class='tag full'>migrated</span><br>"
        f"<code>{esc(path)}</code><br><span style='color:#656d76'>family:</span> {fam} &nbsp;|&nbsp; {note}</div>"
    )
    p.append(
        "<p class='hint'>Left = round-1 Pipe API, right = round-2 (geometry + num_active_cores). Green = added, red = removed.</p>"
    )
    p.append(f"<div id='d2h_k_{base}'></div></div>")
    parts.append("".join(p))

# ---- conv-WS gap panel: explanation doc + the Pipe->raw revert diff ----
wp = ["<div class='panel' id='wsgap'>"]
wp.append(
    "<div class='kmeta'><b>activation_reader_width_sharded</b><span class='tag gap'>kept raw</span><br>"
    f"<code>{esc(WS_PATH)}</code><br><span style='color:#656d76'>family:</span> conv (width-sharded) &nbsp;|&nbsp; "
    "the one kernel the knob-free Pipe cannot express &mdash; reverted to raw mcast.</div>"
)
wp.append("<div class='doc gapcard'>" + md_to_html(read(LIMIT_DOC)) + "</div>")
wp.append("<p class='hint' style='margin-top:18px'>The Round-2 revert (round-1 Pipe &rarr; raw mcast):</p>")
wp.append("<div id='d2h_wsgap'></div></div>")
parts.append("".join(wp))

parts.append(f"<script>{asset('hljs.min.js')}</script>")
parts.append(f"<script>{asset('hljs-cpp.min.js')}</script>")
parts.append(f"<script>{asset('diff2html-ui-base.min.js')}</script>")
parts.append("<script>" + JS_TMPL.replace("__DIFFS_JSON__", diffs_json) + "</script>")
parts.append("</body></html>")

with open(OUT, "w") as f:
    f.write("".join(parts))
print("wrote", OUT, "(", os.path.getsize(OUT), "bytes )")
