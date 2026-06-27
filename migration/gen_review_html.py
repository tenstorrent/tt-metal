# SPDX-License-Identifier: Apache-2.0
# Generates migration/migration_review.html: a tabbed before/after review of the mcast_pipe rollout.
# Tabs grouped by MIGRATION TIER. Light theme, VSCode-like C++ highlighting (highlight.js),
# GitHub-style side-by-side diffs (diff2html). All libs vendored inline -> fully offline.
# Pure file/git/HTML -- no device code.
import subprocess, html, json, os

REPO = "/localdev/sjovic/tt-metal"
BASE = "90120bf7b7c"  # Phase 1 commit: predates every kernel migration -> the "before" baseline
ASSETS = os.path.join(REPO, "migration", "assets")
OUT = os.path.join(REPO, "migration", "migration_review.html")


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


HELPER_FILES = [
    ("mcast_pipe.hpp  —  the helper", "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"),
    ("pipe_sender.cpp  —  reference usage (sender)", "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_sender.cpp"),
    ("pipe_receiver.cpp  —  reference usage (receiver)", "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_receiver.cpp"),
    ("pipe_f3_sender.cpp  —  loopback / degenerate", "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_f3_sender.cpp"),
]

# Tier metadata shown to the reader.
TIERS = {
    1: {
        "name": "Tier 1 — clean spine",
        "tally": "10 migrated · 1 failed (not shown)",
        "desc": "Canonical two-sided <code>send()</code> / <code>receive()</code> — level-flag staging, "
        "flush fence, EXCLUDE_SRC. The matmul-in0/in1 and conv-weights sender/receiver pairs that "
        "prove the API end-to-end. Smallest API distance, strongest verified tests, smallest diffs. "
        "(The conv-1D weights <i>sender</i> failed here: its handshake ack-count differs from the "
        "mcast rectangle population, which the single-<code>num_dests</code> Pipe can't express.)",
    },
    2: {
        "name": "Tier 2 — refactor-low",
        "tally": "2 migrated · 3 failed (not shown)",
        "desc": "Loopback / multi-rect / mixed counter+flag call sites. Migrated where a single rectangle and "
        "one Pipe verb fit (the welford GN receiver; the conv width-sharded INCLUDE_SRC loopback as a "
        "<b>partial</b> — its R→S counter half stays raw). The GN/welford <i>senders</i> failed here: "
        "they broadcast to a <b>list</b> of rectangles (multi-rect), and <code>writer_local_topk</code> "
        "is unicast-scatter, not a rectangle mcast.",
    },
    3: {
        "name": "Tier 3 — refactor-high",
        "tally": "1 migrated · 3 failed (not shown)",
        "desc": "Multi-phase / counter-staged / chain-link call sites — the hardest API distance. Typically a "
        "single phase migrates and the rest stays raw: here the sharded-LN <i>sender</i> migrates its "
        "phase-1 control-flag broadcast to <code>send_signal()</code> (<b>partial</b>) while the phase-2 "
        "monotone counter stays raw. The LN receiver (shared-cell reset ordering), sdpa "
        "<code>reader_interleaved</code> (legacy raw API + ring/role-flip), and the conv halo reader "
        "(streaming chunked send, deferred) failed here.",
    },
}

# (tier, family, basename, path, status_note)
MIGRATED = [
    (
        1,
        "matmul",
        "reader_bmm_tile_layout_in0_sender_padding",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        "canonical send(); sparsity flag block left raw",
    ),
    (
        1,
        "matmul",
        "reader_bmm_tile_layout_in0_receiver",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        "canonical receive(); sparsity wait_min left raw",
    ),
    (
        1,
        "matmul",
        "reader_bmm_tile_layout_in1_sender_writer_padding",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        "2x send() (in1 + bias)",
    ),
    (
        1,
        "matmul",
        "reader_bmm_tile_layout_in1_receiver_writer_padding",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        "2x receive() (in1 + bias)",
    ),
    (
        1,
        "conv",
        "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks",
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp",
        "2x receive(); count-independent",
    ),
    (
        1,
        "conv",
        "writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks",
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp",
        "2x send(); 2D num_dests==num_cores",
    ),
    (
        1,
        "conv",
        "writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks",
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp",
        "2x receive()",
    ),
    (
        1,
        "groupnorm",
        "reader_mcast_receiver_unary_sharded_gn_v2",
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp",
        "receive(); sem naming flipped",
    ),
    (
        1,
        "reduction",
        "reader_final_topk",
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp",
        "PARTIAL: readiness -> send_signal(); fan-in counter raw",
    ),
    (
        1,
        "deepseek",
        "sampling_kernel",
        "models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp",
        "flag-only loop barrier -> send_signal(1)",
    ),
    (
        2,
        "groupnorm",
        "welford_reader_mcast_receiver_unary_sharded_gn_v2",
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp",
        "receive(); twin of non-welford receiver",
    ),
    (
        2,
        "conv",
        "activation_reader_width_sharded",
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp",
        "PARTIAL: INCLUDE_SRC loopback -> send(); counter half raw; PCC 0.9999992",
    ),
    (
        3,
        "layernorm",
        "reader_mcast_sender_unary_sharded_ln",
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp",
        "PARTIAL: phase-1 flag -> send_signal(); phase-2 counter raw",
    ),
]

CSS = """
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#f6f8fa;color:#1f2328}
header{padding:16px 24px;border-bottom:1px solid #d0d7de;background:#fff}
header h1{margin:0;font-size:18px;font-weight:600}
header .sub{color:#656d76;font-size:13px;margin-top:4px}
.tabs{display:flex;flex-wrap:wrap;gap:4px;padding:10px 18px;background:#fff;border-bottom:1px solid #d0d7de;position:sticky;top:0;z-index:20}
.grp{display:flex;align-items:center;gap:4px;margin-right:14px;margin-bottom:4px}
.grp .lbl{color:#fff;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-right:4px;padding:2px 7px;border-radius:4px}
.grp.t-helper .lbl{background:#6e7781}.grp.t1 .lbl{background:#1a7f37}.grp.t2 .lbl{background:#9a6700}.grp.t3 .lbl{background:#cf222e}
.tab{padding:5px 11px;border:1px solid #d0d7de;border-radius:6px;background:#f6f8fa;color:#1f2328;cursor:pointer;font-size:12px;white-space:nowrap}
.tab:hover{background:#eaeef2;border-color:#afb8c1}
.tab.active{background:#0969da;color:#fff;border-color:#0969da;font-weight:600}
.panel{display:none;padding:18px 28px;max-width:none;width:100%}
.panel.active{display:block}
.tierbox{margin:0 0 16px;padding:12px 16px;border:1px solid #d0d7de;border-left-width:5px;border-radius:8px;background:#fff;font-size:13px}
.tierbox.t1{border-left-color:#1a7f37}.tierbox.t2{border-left-color:#9a6700}.tierbox.t3{border-left-color:#cf222e}
.tierbox .tn{font-size:15px;font-weight:700}
.tierbox .tally{font-size:12px;color:#656d76;margin-left:8px;font-weight:600}
.tierbox .desc{margin-top:6px;color:#424a53}
.tierbox code{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12px;color:#0550ae}
.kmeta{font-size:13px;margin:0 0 14px;padding:10px 14px;background:#fff;border:1px solid #d0d7de;border-radius:8px}
.kmeta code{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12px;color:#0550ae}
.kmeta b{font-size:14px}
.tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600;margin-left:8px;vertical-align:middle}
.tag.partial{background:#fff1cc;color:#9a6700;border:1px solid #f0d58c}
.tag.full{background:#dafbe1;color:#1a7f37;border:1px solid #aceebb}
.filebox{margin:16px 0;border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#fff}
.filebox h3{margin:0;padding:9px 14px;background:#f6f8fa;font-size:13px;border-bottom:1px solid #d0d7de;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}
.filebox pre{margin:0;max-height:none;overflow-x:auto}
.filebox pre code.hljs{padding:14px;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#fff}
.d2h-wrapper{font-size:12.5px}
.d2h-file-header{display:none}
.hint{color:#656d76;font-size:12px;margin:0 0 10px}
.tiergrid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:14px}
"""

JS_TMPL = """
const DIFFS = __DIFFS_JSON__;
const drawn = {};
function show(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector('[data-for="'+id+'"]').classList.add('active');
  if(id.startsWith('k_') && !drawn[id]){
    const base = id.slice(2);
    const tgt = document.getElementById('d2h_'+base);
    const ui = new Diff2HtmlUI(tgt, DIFFS[base], {
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


def esc(s):
    return html.escape(s)


def tier_banner(t):
    d = TIERS[t]
    return (
        f"<div class='tierbox t{t}'><span class='tn'>{esc(d['name'])}</span>"
        f"<span class='tally'>{esc(d['tally'])}</span>"
        f"<div class='desc'>{d['desc']}</div></div>"
    )


# ---- collect diffs (git unified diff, BASE..HEAD per kernel) ----
diffs = {}
for t, fam, base, path, note in MIGRATED:
    diffs[base] = sh("diff", BASE, "HEAD", "--", path)
diffs_json = json.dumps(diffs).replace("</", "<\\/")

parts = []
parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
parts.append("<title>mcast_pipe migration review</title>")
parts.append(f"<style>{asset('hljs-github.min.css')}</style>")
parts.append(f"<style>{asset('diff2html.min.css')}</style>")
parts.append(f"<style>{CSS}</style></head><body>")

parts.append(
    "<header><h1>mcast_pipe rollout &mdash; helper &amp; migrated kernels, grouped by tier</h1>"
    "<div class='sub'>Baseline = Phase-1 commit <code>90120bf</code> (pre-migration) &nbsp;|&nbsp; After = HEAD &nbsp;|&nbsp; "
    "13 kernels migrated across 3 tiers. Side-by-side diffs, C++-highlighted. Tabs are grouped by migration tier.</div></header>"
)

# ---- tab bar (grouped by tier) ----
tb = ["<div class='tabs'>"]
tb.append(
    "<div class='grp t-helper'><span class='lbl'>helper</span>"
    "<button class='tab active' data-for='helper' onclick=\"show('helper')\">Pipe helper</button></div>"
)
by_tier = {1: [], 2: [], 3: []}
for t, fam, base, path, note in MIGRATED:
    by_tier[t].append((fam, base, path, note))
for t in (1, 2, 3):
    tb.append(f"<div class='grp t{t}'><span class='lbl'>Tier {t}</span>")
    for fam, base, path, note in by_tier[t]:
        short = base.replace("reader_bmm_tile_layout_", "").replace("_tiled_col_to_rm_blocks", "")
        short = short.replace("reader_mcast_", "").replace("_unary_sharded", "")
        tb.append(
            f"<button class='tab' data-for='k_{base}' onclick=\"show('k_{base}')\" title='{esc(fam)}'>{esc(short)}</button>"
        )
    tb.append("</div>")
tb.append("</div>")
parts.append("".join(tb))

# ---- helper panel (now also carries the tiers overview) ----
hp = ["<div class='panel active' id='helper'>"]
hp.append(
    "<div class='kmeta'>The proposed kernel-side helper <code>dataflow_kernel_lib::Pipe</code> and its reference "
    "driver kernels. Every migrated kernel was rewritten to call this. The three migration tiers are explained below; "
    "open a tier's tabs above to see each kernel's before/after.</div>"
)
hp.append("<div class='tiergrid'>" + "".join(tier_banner(t) for t in (1, 2, 3)) + "</div>")
for title, path in HELPER_FILES:
    hp.append(
        f"<div class='filebox'><h3>{esc(title)}</h3>"
        f"<pre><code class='language-cpp'>{esc(read(path))}</code></pre></div>"
    )
hp.append("</div>")
parts.append("".join(hp))

# ---- per-kernel diff panels (grouped logically by tier; each carries its tier banner) ----
for t in (1, 2, 3):
    for fam, base, path, note in by_tier[t]:
        partial = "PARTIAL" in note
        tagcls, tagtxt = ("partial", "partial") if partial else ("full", "full")
        p = [f"<div class='panel' id='k_{base}'>"]
        p.append(tier_banner(t))
        p.append(
            f"<div class='kmeta'><b>{esc(base)}</b><span class='tag {tagcls}'>{tagtxt}</span><br>"
            f"<code>{esc(path)}</code><br><span style='color:#656d76'>family:</span> {fam} &nbsp;|&nbsp; {esc(note)}</div>"
        )
        p.append(
            "<p class='hint'>Left = before (Phase-1 baseline), right = after (HEAD). Green = added, red = removed.</p>"
        )
        p.append(f"<div id='d2h_{base}'></div>")
        p.append("</div>")
        parts.append("".join(p))

parts.append(f"<script>{asset('hljs.min.js')}</script>")
parts.append(f"<script>{asset('hljs-cpp.min.js')}</script>")
parts.append(f"<script>{asset('diff2html-ui-base.min.js')}</script>")
parts.append("<script>" + JS_TMPL.replace("__DIFFS_JSON__", diffs_json) + "</script>")
parts.append("</body></html>")

with open(OUT, "w") as f:
    f.write("".join(parts))
print("wrote", OUT, "(", os.path.getsize(OUT), "bytes )")
