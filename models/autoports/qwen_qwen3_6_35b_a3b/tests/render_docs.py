# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Re-derive every mechanical number in the stage docs from the committed artifacts. Idempotent.

Run from the repo root after a full evidence pass:

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/render_docs.py

**Why this exists.** Five consecutive review rounds found doc numbers that disagreed with the
artifacts they cite -- every one of them hand-copied. Anything this script writes cannot drift, which
is the point. It owns:

* README section 3.1 -- the per-family worst-PCC table, from ``pcc.jsonl``;
* README section 3.8 -- the whole decode-SDPA investigation, from the two diagnostic logs;
* README section 5 -- the perf tables and the two analysis bullets, from ``perf_summary.json`` and
  the ``tt-perf-report`` CSVs;
* the scattered derived figures: host throughput, the sparse-matmul derivation, watcher sizes, Tracy
  artifact sizes, the advertised-context table, evidence row counts, suite and probe counts.

``test_docs_match_artifacts`` (CPU-only, in the main suite) re-checks the same derivations and fails
if the committed docs drift from the committed artifacts, so this is not the only line of defence.
Earlier rounds kept these renderers outside the repo, which meant the "generated, cannot drift"
property was not reproducible from a checkout. It is now.

Sections are rewritten in order: 3.1, then 3.8, then 5, then the scattered figures. Later steps
re-read the file, so the order matters.
"""

import csv
import gzip
import json
import pathlib
import re
import shutil
import sys
import tempfile

A = pathlib.Path("models/autoports/qwen_qwen3_6_35b_a3b")
D = A / "doc/functional_decoder"

#: ``--check`` renders into a throwaway copy of the doc tree and diffs, so CI (and the evidence pass,
#: right after a real render) can assert the committed docs already match the artifacts without
#: modifying anything. Everything below writes through these three names, so redirecting them is the
#: whole implementation.
CHECK = "--check" in sys.argv
if CHECK:
    _scratch = pathlib.Path(tempfile.mkdtemp(prefix="render_docs_check_"))
    for _name in ("README.md", "work_log.md", "tracy/README.md"):
        (_scratch / _name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(D / _name, _scratch / _name)
    README = _scratch / "README.md"
    WORKLOG = _scratch / "work_log.md"
    TRACY_README = _scratch / "tracy/README.md"
else:
    README = D / "README.md"
    WORKLOG = D / "work_log.md"
    TRACY_README = D / "tracy/README.md"

report = []

#: Every artifact a full render reads. Checked up front because the provenance logs are *deleted* at
#: the start of the run that rewrites them (``harness.reset_log``), so rendering while the suite is
#: in flight would otherwise read a half-written ``pcc.jsonl`` and quietly publish partial numbers.
#: That happened once; this turns it into a refusal.
REQUIRED = [
    "pcc.jsonl",
    "pcc_real_weights.jsonl",
    "long_context.jsonl",
    "perf_summary.json",
    "perf_host_summary.jsonl",
    "logs/diag_sdpa_decode.txt",
    "logs/diag_decode_sdpa_onmodel.txt",
    "logs/diag_long_decode.txt",
    "logs/probe_ttnn_ops.log",
    "logs/test_suite_main.log",
    "tracy/full_decode/decode_perf_report.csv",
    "tracy/full_prefill/prefill_perf_report.csv",
    "watcher/pytest.log",
]
_missing = [name for name in REQUIRED if not (D / name).exists()]
if _missing:
    raise SystemExit(
        "render_docs: refusing to run, these artifacts are missing (is an evidence run still in "
        "progress?):\n  " + "\n  ".join(_missing)
    )


# =====================================================================================
# section 3.1 -- per-family worst PCC
# =====================================================================================
#: Suffix shown after the family name in the table, so the generated rows keep the annotations that
#: make the table readable. Families not listed here get a bare name.
FAMILY_NOTE = {
    "prefill-fresh-slot[linear]": " (reused slot, no reset)",
    "prefill-fresh-slot[full]": " (reused slot, no reset)",
    "prefill-slot[linear]": " (32 slots)",
    "prefill-slot[full]": " (32 slots)",
    "decode-ragged": " (per-slot positions)",
    "decode-active-slot": " (with `current_pos=-1` peers)",
    "decode-seeded-state": " (random DeltaNet state)",
    "paged-kv": " cache contents",
    "linear": " conv/recurrent state",
}
#: Table order: prefill families, then decode, then traced, then the state/cache rows.
FAMILY_ORDER = [
    "prefill[linear]",
    "prefill[full]",
    "prefill-cont[linear]",
    "prefill-cont[full]",
    "prefill-fresh-slot[linear]",
    "prefill-fresh-slot[full]",
    "prefill-slot[linear]",
    "prefill-slot[full]",
    "decode[linear]",
    "decode[full]",
    "decode-ragged",
    "decode-active-slot",
    "decode-seeded-state",
    "traced-decode[linear]",
    "traced-decode[full]",
    "paged-kv",
    "linear",
]


def pcc_families():
    """``{family: (n, worst_row)}`` from ``pcc.jsonl``, family = the label's first token."""
    fam = {}
    for line in (D / "pcc.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fam.setdefault(r["label"].split(" ")[0], []).append(r)
    return {k: (len(v), min(v, key=lambda x: x["pcc"])) for k, v in fam.items()}


def worst_case_label(family, row):
    """The ``worst case`` cell: the distinguishing part of the label, minus the family token."""
    rest = row["label"][len(family) :].strip()
    if family == "linear":
        return rest
    if family == "paged-kv":
        return rest
    return ", ".join(p for p in rest.split(" ") if not p.startswith("start=0")) or rest


def render_pcc_table():
    fams = pcc_families()
    missing = set(fams) - set(FAMILY_ORDER)
    if missing:
        report.append(f"  WARN     pcc.jsonl has unlisted families {sorted(missing)}")
    lines = ["| family | n | worst PCC | worst case |", "|---|---|---|---|"]
    for family in FAMILY_ORDER:
        if family not in fams:
            report.append(f"  WARN     family {family} absent from pcc.jsonl")
            continue
        n, row = fams[family]
        lines.append(
            f"| `{family}`{FAMILY_NOTE.get(family, '')} | {n} | {row['pcc']:.7f} | "
            f"{worst_case_label(family, row)} |"
        )
    s = README.read_text()
    pat = re.compile(r"(\| family \| n \| worst PCC \| worst case \|\n\|---\|---\|---\|---\|\n)(?:\|.*\n)+", re.M)
    if not pat.search(s):
        report.append("  MISS     section 3.1 table header")
        return
    s = pat.sub(lambda m: "\n".join(lines) + "\n", s)
    overall = min(r["pcc"] for _, r in fams.values())
    worst_family = min(fams.items(), key=lambda kv: kv[1][1]["pcc"])
    s = re.sub(
        r"\*\*Overall minimum: [\d.]+\*\* \(`[^`]+`\)",
        f"**Overall minimum: {overall:.7f}** (`{worst_family[1][1]['label']}`)",
        s,
    )
    README.write_text(s)
    report.append(f"  updated  section 3.1 table ({len(fams)} families, overall {overall:.7f})")


render_pcc_table()


# =====================================================================================
# section 3.8 -- the decode-SDPA investigation, generated from the diagnostic logs
# =====================================================================================
SWEEP = (D / "logs/diag_sdpa_decode.txt").read_text()
ONMODEL = (D / "logs/diag_decode_sdpa_onmodel.txt").read_text()
LONGDEC = (D / "logs/diag_long_decode.txt").read_text()
# ---------------------------------------------------------------- parse the op sweep
head = re.search(r"^DIAG2 +kchunk +maxcore +/head +(.+)$", SWEEP, re.M)
CTX = [int(c) for c in head.group(1).split()]
SHOW = [257, 1024, 4096, 32768, 131072, 262143]
show_i = [CTX.index(c) for c in SHOW]

grid_block = SWEEP.split("op PCC vs exact fp64")[1].split("k-chunks per active core")[0]
rows = {}
for line in grid_block.splitlines():
    m = re.match(r"^DIAG2 +(dynamic|\d+) +(\d+) +(\d+) +(.+)$", line)
    if not m:
        continue
    kc = 0 if m.group(1) == "dynamic" else int(m.group(1))
    rows[(kc, int(m.group(2)))] = (int(m.group(3)), m.group(4).split())

ident_all = re.search(r"IDENT +all contexts bit-identical: (\w+)", SWEEP).group(1)
n_ident = len(re.findall(r"^DIAG2 IDENT ctx=", SWEEP, re.M))

timing = {}
for label, ms, pcc in re.findall(r"^DIAG2 TIME +(.+?) +([\d.]+) ms/call +pcc ([\d.]+)$", SWEEP, re.M):
    timing[label.strip()] = (float(ms), float(pcc))
errored = dict(re.findall(r"^DIAG2 TIME +(.+?) +ERR:(\w+)", SWEEP, re.M))
# The L1 blocker for the *1-core* k1024 candidate specifically. Several settings in the grid fail
# with the same message at different byte counts (k512 at 55 cores/head, for one), so anchor on the
# timing line for the candidate being reported rather than taking the first match in the file.
_k1024 = SWEEP.split("TIME            k1024, 1 core")[1]
l1 = re.search(r"grow to (\d+) B which is beyond max L1 size of (\d+) B", _k1024)
_k2048 = SWEEP.split("TIME            k2048, 1 core")[1]
l1_2048 = re.search(r"grow to (\d+) B which is beyond max L1 size of (\d+) B", _k2048)

# ---------------------------------------------------------------- parse the on-model table
on_labels = re.search(r"^ONMODEL SUMMARY +context +(.+)$", ONMODEL, re.M).group(1)
on_labels = [x.strip() for x in re.split(r" {2,}", on_labels.strip())]
on_rows = []
for line in re.findall(r"^ONMODEL SUMMARY +(\d+) +(.+)$", ONMODEL, re.M):
    on_rows.append((int(line[0]), [float(v) for v in line[1].split()]))
worst = {}
for label, val in re.findall(r"([A-D] [^=]+)=([\d.]+)", ONMODEL.split("WORST-OVER-CONTEXTS")[1]):
    worst[label.strip()] = float(val)

# ---------------------------------------------------------------- parse the position sweep
# `DIAG <pos> <layer_pcc> <tt_vs_fp32> <tt_vs_bf16ctl> <bf16ctl_vs_fp32> <attn_rms>`, with a final
# `token*8` row that is the residual-dilution sensitivity check rather than another position.
long_rows = re.findall(r"^DIAG +(\d+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)\s*$", LONGDEC, re.M)
dilution = re.search(r"^DIAG +(\d+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +token\*8", LONGDEC, re.M)
assert long_rows, "no position-sweep rows parsed"


def cell(kc, mc, i):
    return rows[(kc, mc)][1][i]


def safe_family_table():
    out = [
        "| `k_chunk_size` | " + " | ".join(str(c) for c in SHOW) + " | op time @262144 | verdict |",
        "|---|" + "---|" * (len(SHOW) + 2),
    ]
    for kc, tlabel in [
        (32, "op default (k32, 1 core)"),
        (64, None),
        (128, "k0 dynamic, 1 core"),
        (256, "k256, 1 core"),
        (512, "k512, 1 core"),
    ]:
        if (kc, 1) not in rows:
            continue
        vals = [cell(kc, 1, i) for i in show_i]
        t = f"**{timing[tlabel][0]:.2f} ms**" if tlabel and tlabel in timing else "--"
        note = {
            32: "the op's own default -- worst measured",
            64: "",
            128: "what `k_chunk_size=0` resolves to; shipped in round 2",
            256: "",
            512: "**ships**; largest legal chunk",
        }[kc]
        label = f"**{kc}**" if kc == 512 else str(kc)
        out.append(f"| {label} | " + " | ".join(vals) + f" | {t} | {note} |")
    for kc in (1024, 2048):
        if (kc, 1) in rows and rows[(kc, 1)][1][0].startswith("ERR"):
            out.append(f"| {kc} | " + " | ".join(["L1"] * len(SHOW)) + " | -- | exceeds L1 (see below) |")
    return "\n".join(out)


def multicore_table():
    out = [
        "| `k_chunk_size` | cores/head | " + " | ".join(str(c) for c in SHOW) + " |",
        "|---|---|" + "---|" * len(SHOW),
    ]
    for kc in (256, 512):
        for mc in (1, 2, 8, 16):
            if (kc, mc) not in rows:
                continue
            per_head, vals = rows[(kc, mc)]
            cells = []
            for i in show_i:
                v = vals[i]
                # Bold only structurally wrong cells. Values in the 0.98-0.999 band are the
                # long-context bf16 accumulation floor, which is expected and must not read as a bug.
                cells.append(f"**{v}**" if (not v.startswith("ERR") and float(v) < 0.9) else v)
            out.append(f"| {kc} | {per_head} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def onmodel_table():
    out = ["| context | " + " | ".join(on_labels) + " |", "|---|" + "---|" * len(on_labels)]
    for ctx, vals in on_rows:
        cells = [f"{v:.7f}" if v > 0.99 else f"**{v:.7f}**" for v in vals]
        out.append(f"| {ctx} | " + " | ".join(cells) + " |")
    out.append(
        "| **worst over contexts** | "
        + " | ".join(
            (f"**{worst[l]:.7f}**" if worst[l] > 0.99 else f"**{worst[l]:.7f}**") for l in on_labels if l in worst
        )
        + " |"
    )
    return "\n".join(out)


def timing_table():
    order = [
        ("no config", "= k32, 1 core -- what the layer ran before any sweep"),
        ("op default (k32, 1 core)", "the substituted default, spelled out"),
        ("k0 dynamic, 1 core", "k128; shipped in round 2"),
        ("k256, 1 core", ""),
        ("k512, 1 core", "**ships**"),
        ("k256, 16 cores", "fastest overall -- and unshippable, see above"),
        ("k32, 16 cores", "the op default's chunk at 16 cores"),
    ]
    out = ["| setting | op time @262144 | op PCC @262143 | |", "|---|---|---|---|"]
    for label, note in order:
        if label not in timing:
            continue
        ms, pcc = timing[label]
        out.append(f"| `{label}` | **{ms:.3f} ms** | {pcc:.4f} | {note} |")
    return "\n".join(out)


def position_table():
    out = [
        "| decode position (context) | layer PCC | TTNN attn vs fp32 HF | TTNN attn vs **bf16-operand HF control** | **control vs fp32 HF** | attn RMS |",
        "|---|---|---|---|---|---|",
    ]
    for pos, layer, a1, a2, ctrl, rms in long_rows:
        out.append(f"| {pos} ({int(pos)+1}) | {layer} | {a1} | {a2} | {ctrl} | {rms} |")
    return "\n".join(out)


ship = timing["k512, 1 core"]
# Use the `op default` row, not the `no config` row, for every "vs the op default" ratio. They are
# the same program (the identity control proves it) so the two timings differ only by run-to-run
# variance, but the section distinguishes the rows and the label should follow the row it quotes.
dflt = timing["op default (k32, 1 core)"]
noconfig = timing["no config"]
prev = timing["k0 dynamic, 1 core"]
fastest = timing["k256, 16 cores"]

section = f"""### 3.8 Investigated anomaly: full-attention decode at 262144-token context

`test_longest_decode_context[full]` was the one number in the stage materially below the rest.
Three diagnostics, all kept in `tests/` with their output in `logs/`. The short version: the
variable is **`SDPAProgramConfig::k_chunk_size`** -- how many keys the decode SDPA accumulates per
chunk, i.e. the depth of the sequential bf16 accumulation -- and shipping the largest chunk the L1
allows fixes it, while also making the op {dflt[0]/ship[0]:.1f}x faster than the op's own default
({dflt[0]:.2f} -> {ship[0]:.2f} ms/call at 262144 keys; passing no config at all measures
{noconfig[0]:.2f} ms, the same program within run-to-run variance).

**Read this first: "pass no program config" is not a neutral choice.** The paged decode entry point
substitutes a full config of its own before the device op ever sees one
(`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:122-129`):

```cpp
if (!program_config.has_value()) {{
    program_config = SDPAProgramConfig{{
        input_tensor_q.device()->compute_with_storage_grid_size(),
        std::nullopt,                    // sub_core_grids
        kDefaultDecodeChunkSize,         // q_chunk_size = 32
        kDefaultDecodeChunkSize,         // k_chunk_size = 32
        std::nullopt,                    // exp_approx_mode -> resolved to false
        kDefaultMaxCoresPerHeadBatch}};  // max_cores_per_head_batch = 1
```

Three consequences, all of which invalidate an earlier version of this section:

* the factory's `program_config.has_value() ? max_cores_per_head_batch : num_cores_available`
  (`sdpa_decode_program_factory.cpp:192-193`) is **unreachable from this op**, so the op never runs
  at 55 cores/head no matter what;
* the struct default of 16 (`sdpa_config.hpp:18`) is unreachable too;
* "no config" specifically means **`k_chunk_size = 32`**, which the sweep below shows is the *worst*
  setting measured at long context.

Only the non-paged `scaled_dot_product_attention_decode` leaves the config empty and can reach the
`num_cores_available` branch. This layer calls the paged variant.

`diag_sdpa_decode.py` opens with the identity control that pins this down: no config versus an
explicit config spelling out the substitution above. They are **bit-identical at all {n_ident}
contexts** (`all contexts bit-identical: {ident_all}`, max abs diff 0.0 everywhere) -- so the two are
the same program, and any difference measured against "no config" is attributable to the fields that
actually differ.

**Step 1 -- localise it (`diag_long_decode.py` -> `logs/diag_long_decode.txt`).** Sweep the decode
position over one cache and isolate the **attention branch** (the only position-dependent part of
the layer) by driving the TTNN and HF mixers directly. This is what said the loss is inside
attention rather than in the MoE, the residual or the cache read, and -- via the control column --
that it is not operand precision:

{position_table()}

The last column is a control: HF's own attention math with q/k/v rounded to bf16 and exact
accumulation matches fp32 at **every** context. So **operand precision is not the cause** -- the
device diverges from an exact bf16 reference (column 4) by the same amount as from fp32 (column 3)
at every context.

These are the numbers *after* the fix in step 4. The attention branch still loses accuracy as the
key count grows, but that residual is now a plain accumulation floor rather than a decomposition
bug: at 1 core per head there is no cross-core reduction at all, and the control shows an exact-bf16
reference does not lose it. It dilutes to {long_rows[-1][1]} at the layer level because the attention
branch is one of two summed contributions and the residual dominates.

The `layer PCC` column is not directly comparable to the headline number: this diagnostic seeds its
own cache and token, so it can sweep five positions over one cache in one run, while
`test_longest_decode_context[full]` decodes off a cache built by a real 262143-token prefill.
Different inputs, so different values; what the sweep is for is the *shape* of the curve and the
control column, both of which are input-independent.

(The file's final `token*8` row is a residual-dilution sensitivity check, and it is the cleanest
demonstration that the layer number is dilution rather than accuracy: `input_layernorm` is an RMS
norm, so scaling the input token leaves the attention branch **bit-identical** -- all three attention
columns repeat to the last digit -- while the layer PCC moves {long_rows[-1][1]} ->
{dilution.group(2) if dilution else "n/a"} purely because the residual got 8x larger. It is *not* a
softmax-peaking control.)

**Step 2 -- the 2-D sweep (`diag_sdpa_decode.py` -> `logs/diag_sdpa_decode.txt`).** Drive
`paged_scaled_dot_product_attention_decode` alone (no projections, RoPE, gate, o_proj or MoE) with
random K/V over a paged cache, and sweep `k_chunk_size` **x** `max_cores_per_head_batch` as a grid
rather than one axis at a time. The grid matters: the two interact, because more cores per head
means fewer chunks per core, and an earlier one-axis-at-a-time sweep at a single chunk size is
exactly how this section previously reached the wrong conclusion.

At 1 core per head -- the op's own decomposition -- accuracy is monotone in chunk size at long
context, which is what an accumulation-depth mechanism predicts. At 262144 keys, `k_chunk_size=32`
is 8192 sequential accumulation steps and 512 is 512:

{safe_family_table()}

`k_chunk_size` must be a power of two and a multiple of 32 (`sdpa_decode.cpp:146-151`), and **512 is
the largest legal value here**: 1024 fails to build with
`Statically allocated circular buffers on core range [0-0 - 10-9] grow to {int(l1.group(1)):,} B which is
beyond max L1 size of {int(l1.group(2)):,} B`, and 2048 the same at {int(l1_2048.group(1)):,} B. That is
an op-contract blocker on going further, not a choice.

**Nothing above 1 core per head is usable at every context.** More cores is *more* accurate at long
context and silently wrong at some shorter context, and the boundary moves with the chunk size:

{multicore_table()}

Bold cells are silently wrong answers -- no error, no warning, just a wrong tensor. The unbolded
0.98-0.999 values in the 1-core rows are the accumulation floor, not wrongness. `1` is the only
`max_cores_per_head_batch` value with no such cell anywhere in the grid, and it is also what the op
already does by default; the config pins it so that a later stage has to read why before changing
it. `exp_approx_mode` is bit-identically irrelevant (held-axis rows), and an `8x8` grid at 1
core/head equals `11x10` at 1 core/head to the last digit, so neither is the variable.

**Step 3 -- the cost, measured on the same op** (20 warmed calls at 262144 keys):

{timing_table()}

There is **no accuracy/latency trade-off inside the safe family**: bigger chunks are both faster and
more accurate, so the largest legal chunk wins on both counts. The genuinely fastest setting in the
whole grid is `k256, 16 cores` at {fastest[0]:.2f} ms -- {ship[0]/fastest[0]:.1f}x faster than what
ships -- and it is unshippable because it returns a wrong answer at 257, 1024 and 4096 keys.

**Step 4 -- the on-model decision (`diag_decode_sdpa_onmodel.py` ->
`logs/diag_decode_sdpa_onmodel.txt`).** The op sweep uses random K/V, so the candidates are
re-measured on the **whole decoder layer** against HF, off a real prefilled cache, at five contexts.
One layer is built per context and prefilled once, so every setting decodes from the same cache and
the comparison is same-input:

{onmodel_table()}

D is the fastest setting in the op sweep and it is confirmed unshippable on the real layer, not just
on random K/V. C ships.

**What ships** (`DecoderConfig.decode_sdpa_k_chunk_size = 512`,
`decode_sdpa_max_cores_per_head = 1`): correct at every context measured, best-in-family at the
advertised context, {dflt[0]/ship[0]:.1f}x faster than the op default and
{prev[0]/ship[0]:.2f}x faster than the round-2 setting. At the layer level the advertised-context
decode PCC is now **{[v for c, v in on_rows if c == 262144][0][2]:.7f}**, in line with every other
context, against {[v for c, v in on_rows if c == 262144][0][0]:.7f} for the op default and
{[v for c, v in on_rows if c == 262144][0][1]:.7f} for round 2's.

**Correction, and what it cost.** Two earlier versions of this section were wrong about the
mechanism. The first blamed "whether a program config is passed at all" and handed `optimize` a
per-position config selection. The second blamed `max_cores_per_head_batch` -- a field that does not
differ between the settings it was comparing, because the op already defaults it to 1. That round
even wrote down the disproof and filed it as someone else's bug: its declared control row
("`max_cores=110` should reproduce the no-config row exactly") did *not* reproduce it, and instead of
falsifying the hypothesis the mismatch was recorded as an upstream reproducer. It is fully explained
by the substitution above -- no config is `k32` at **1** core/head, while explicit `max_cores=110`
derives 55, so the two rows differ in cores, not in nothing. The lesson worth carrying: a control
that fails is evidence about your own model first.

**Still worth an upstream issue**, and independent of the above: every `max_cores_per_head_batch`
above 1 makes this op return a **silently wrong** result below some context -- PCC as low as 0.0000
(`k32`, 16 cores/head, 262143 keys) -- rather than refusing to run. `diag_sdpa_decode.py` is a
self-contained reproducer with no model code, and the k-chunks-per-core table it prints alongside the
grid is the starting point for narrowing it.

"""

s = README.read_text()
start = s.index("### 3.8 ")
end = s.index("### 3.9 ")
README.write_text(s[:start] + section + s[end:])
report.append("  updated  section 3.8 (generated from the two diagnostic logs)")

# =====================================================================================
# section 5 -- perf tables and analysis bullets, from perf_summary.json
# =====================================================================================
rows = {(r["kind"], r["mode"]): r for r in json.loads((D / "perf_summary.json").read_text())}
s = README.read_text()
ORDER = [
    ("linear", "prefill", "seq 2048, batch 1"),
    ("full", "prefill", "seq 2048, batch 1"),
    ("linear", "decode", "batch 32, `cur_pos` 4095"),
    ("full", "decode", "batch 32, `cur_pos` 4095"),
]

table = []
for kind, mode, shape in ORDER:
    r = rows[(kind, mode)]
    label = f"{mode} `{kind}`" + (" (traced)" if mode == "decode" else "")
    table.append(
        f"| {label} | {shape} | {r['ops_in_window']} ({r['iters']} iters) | "
        f"**{r['device_kernel_ms_per_iter']:.2f} ms** | {r['op_to_op_gap_ms_per_iter']:.3f} ms | "
        f"{r['host_wall_ms_per_iter']:.2f} ms |"
    )
new_table = "\n".join(table)

# --- replace the table (between its header row and the following blank line) ---
pat = re.compile(
    r"(\| case \| shape \| ops in window \| device kernel / iter \| op-to-op gap / iter \| host wall / iter \|\n"
    r"\|---\|---\|---\|---\|---\|---\|\n)(?:\|.*\n)+",
    re.M,
)
assert pat.search(s), "perf table header not found"
s = pat.sub(lambda m: m.group(1) + new_table + "\n", s)

gap = [
    100 * rows[k]["op_to_op_gap_ms_per_iter"] / rows[k]["device_kernel_ms_per_iter"]
    for k in [(a, b) for a, b, _ in ORDER]
]
host = [
    100 * (rows[k]["host_wall_ms_per_iter"] / rows[k]["device_kernel_ms_per_iter"] - 1)
    for k in [(a, b) for a, b, _ in ORDER]
]

blocks = {}
for kind, mode, _ in ORDER:
    r = rows[(kind, mode)]
    b = r["blocks"]
    total = r["device_kernel_ms_per_iter"]
    blocks[(kind, mode)] = (
        b["mixer_ms_per_iter"],
        b["expert_matmul_ms_per_iter"],
        b["moe_elementwise_ms_per_iter"],
        total,
    )


def pct(x, total):
    return f"{100 * x / total:.1f}%"


split_rows = []
for kind, mode, _ in ORDER:
    mixer, expert, other, total = blocks[(kind, mode)]
    split_rows.append(
        f"| {mode} `{kind}` | {mixer:.2f} ms ({pct(mixer, total)}) | {expert:.2f} ms ({pct(expert, total)}) | "
        f"{other:.2f} ms ({pct(other, total)}) | {total:.2f} ms |"
    )

ctx = rows[("full", "decode")].get("supported_context")
# --- position dependence of the prefill mixer -------------------------------------------------
# The table above is one prefill chunk at abs_pos 0. Chunked SDPA's key length is
# `chunk_start_idx + Sq` (sdpa_program_factory.cpp:216-217), so `full`'s mixer cost grows with
# position while `linear`'s does not. Extrapolating the per-chunk cost to the advertised context and
# comparing against the measured run quantifies what the profiled shape cannot show -- and `linear`
# is the control that makes the extrapolation trustworthy.
_lc = {
    json.loads(line)["label"]: json.loads(line)
    for line in (D / "long_context.jsonl").read_text().splitlines()
    if line.strip()
}
POS_CHUNKS = -(-262143 // rows[("full", "prefill")].get("prefill_chunk_size", 2048))
pos = {}
for _kind in ("linear", "full"):
    _per = rows[(_kind, "prefill")]["device_kernel_ms_per_iter"]
    _row = _lc.get(f"longest-prefill[{_kind}] seq=262143 tail=128", {})
    _meas = _row.get("wall_seconds")
    pos[_kind] = {
        "per_chunk_ms": _per,
        "predicted_s": POS_CHUNKS * _per / 1000,
        "measured_s": _meas,
        "excess_s": (_meas - POS_CHUNKS * _per / 1000) if _meas else None,
        "mixer_pct": 100 * rows[(_kind, "prefill")]["blocks"]["mixer_ms_per_iter"] / _per,
    }
_fx = pos["full"]["excess_s"] or 0.0
_fm = pos["full"]["measured_s"] or 1.0
_full_attn_pct = 100 * (rows[("full", "prefill")]["blocks"]["mixer_ms_per_iter"] * POS_CHUNKS / 1000 + _fx) / _fm

# The advertised-context decode-SDPA cost, read from the sweep rather than hardcoded: it has changed
# with every revision of the shipped config and went stale in the docs each time.
sweep = (D / "logs/diag_sdpa_decode.txt").read_text()
ship_ms = float(re.search(r"^DIAG2 TIME +k512, 1 core +([\d.]+) ms/call", sweep, re.M).group(1))
# the same op at the profiled shape, from the committed perf report, so the comparison names both
_dec = list(csv.DictReader((D / "tracy/full_decode/decode_perf_report.csv").open()))
_sdpa_rows = [r for r in _dec if "SdpaDecode" in r["OP Code"]]
sdpa_profiled = sum(float(r["Device Time"]) for r in _sdpa_rows) / 1000 / rows[("full", "decode")]["iters"]
bullets = f"""Two things worth carrying into the `optimize` stage. Both come from the same three-way split,
which `summarize_perf.py` derives into `perf_summary.json` (`blocks`) by finding the mixer/MoE
boundary structurally — the last `LayerNormDeviceOperation` before the first sparse matmul in an
iteration, i.e. `post_attention_layernorm` — rather than by hand:

| case | token mixer | expert matmuls | MoE dense-intermediate elementwise | total |
|---|---|---|---|---|
{chr(10).join(split_rows)}

* **At the profiled shape the MoE is the whole cost.** (Only at the profiled shape — see the
  position-dependence table below, which is the part that matters for the advertised context.)
  For `full` layers the token mixer is
  {pct(blocks[("full", "prefill")][0], blocks[("full", "prefill")][3])} of prefill and
  {pct(blocks[("full", "decode")][0], blocks[("full", "decode")][3])} of decode. At *this* shape the
  optimisation budget belongs to expert routing, and — less obviously — to the **elementwise work over the
  dense-over-256-expert intermediates**, which is
  {pct(blocks[("full", "prefill")][2], blocks[("full", "prefill")][3])} of prefill and
  {pct(blocks[("full", "decode")][2], blocks[("full", "decode")][3])} of decode, i.e. larger than
  the expert matmuls themselves in decode. `linear` layers add the gated delta rule on top: mixer
  {pct(blocks[("linear", "prefill")][0], blocks[("linear", "prefill")][3])} of prefill and
  {pct(blocks[("linear", "decode")][0], blocks[("linear", "decode")][3])} of decode.
* **Device-bound, not dispatch-bound.** Op-to-op gap is {min(gap):.2f}-{max(gap):.2f}% of device
  time (`gap/device` = {" / ".join(f"{x:.3f}%" for x in gap)} for the four rows in table order), and
  host wall-clock exceeds device kernel time by {min(host):.2f}-{max(host):.2f}%. The traced decode
  has essentially no dispatch overhead left to remove.

**These rows are measured at `supported_context = {ctx}`, not the advertised 262144**
(`perf_summary.json` records it per row). Decode cost grows with `cur_pos`: the decode SDPA alone is
{ship_ms:.2f} ms/call at 262144 keys (batch 1) versus {sdpa_profiled:.2f} ms/iter here at batch 32 (§3.8), so an
advertised-context decode step is roughly {ship_ms - sdpa_profiled:.0f} ms slower than the table shows.
`test_perf.py` explains why the profiled shape is what it is — batch 32 at the full context needs
16 GiB of paged K/V, leaving no room for a profiler buffer.

**And the prefill split above is position-dependent, so "attention is rounding error" is a statement
about the profiled shape only.** The table is *one* prefill chunk at `abs_pos = 0`. Chunked SDPA's
key length is `chunk_start_idx + Sq` (`sdpa_program_factory.cpp:216-217`), so per-chunk attention work
grows linearly with position while the MoE's does not. Extrapolating the per-chunk cost to the
{POS_CHUNKS} chunks of a 262143-token prefill and comparing against the measured run separates the two,
with `linear` as the control — its mixer is position-independent, so its extrapolation should land:

| kind | per chunk | x {POS_CHUNKS} chunks | measured | unexplained by the position-independent model |
|---|---|---|---|---|
| `linear` (control) | {pos["linear"]["per_chunk_ms"]:.2f} ms | {pos["linear"]["predicted_s"]:.2f} s | {pos["linear"]["measured_s"]:.3f} s | {pos["linear"]["excess_s"]:+.2f} s |
| `full` | {pos["full"]["per_chunk_ms"]:.2f} ms | {pos["full"]["predicted_s"]:.2f} s | {pos["full"]["measured_s"]:.3f} s | **{pos["full"]["excess_s"]:+.2f} s** |

The control lands within {abs(pos["linear"]["excess_s"]):.2f} s ({100*abs(pos["linear"]["excess_s"])/pos["linear"]["measured_s"]:.1f}%), which is what makes the `full` row
readable: **{pos["full"]["excess_s"]:.1f} s, {100*_fx/_fm:.0f}% of that prefill, is not explained by position-independent
work**, and the only structural difference between the two kinds is the token mixer. So the `full`
attention path is on the order of **{_full_attn_pct:.0f}% of an advertised-context prefill**, not the
{pos["full"]["mixer_pct"]:.1f}% the profiled row shows. (Part of the excess is per-chunk program creation for
{POS_CHUNKS} distinct `chunk_start_idx` values — §6 limitation 6 — which is also attention-path cost.)
These are cold single-process wall times, not warmed latencies, so treat the {POS_CHUNKS}x figures as an
order-of-magnitude split rather than a benchmark. `optimize` should not read the 1.3% row as
permission to skip prefill attention.
"""

start = s.index("Two things worth carrying into the `optimize` stage")
end = s.index("`linear` prefill issues")
lp, fp = rows[("linear", "prefill")], rows[("full", "prefill")]
tail = (
    f"`linear` prefill issues {lp['ops_in_window'] / fp['ops_in_window']:.1f}x more ops than `full` "
    f"({lp['ops_in_window']} vs {fp['ops_in_window']} in the same window) for the same token count: "
    "the gated delta rule contributes a 32-step Python-driven chunk scan plus the UT transform per\n"
    "2048-token chunk. It is still device-bound, so the op count is a latency risk only at much\n"
    "shorter sequences.\n"
)
end2 = s.index("\n## 6. Known limitations")
s = s[:start] + bullets + "\n" + tail + s[end2:]
README.write_text(s)
report.append(f"  updated  section 5 (gap {min(gap):.3f}-{max(gap):.3f}%, context {ctx})")


# =====================================================================================
# the scattered derived figures
# =====================================================================================
def sub(path, old, new, label):
    s = path.read_text()
    if old == new:
        report.append(f"  same     {label}")
        return
    if old not in s:
        report.append(f"  MISS     {label}  (anchor not found: {old[:60]!r})")
        return
    path.write_text(s.replace(old, new))
    report.append(f"  updated  {label}: {new[:70]!r}")


# ---------------------------------------------------------------- perf section
perf = {(r["kind"], r["mode"]): r for r in json.loads((D / "perf_summary.json").read_text())}
ORDER = [
    ("linear", "prefill", "seq 2048, batch 1"),
    ("full", "prefill", "seq 2048, batch 1"),
    ("linear", "decode", "batch 32, `cur_pos` 4095"),
    ("full", "decode", "batch 32, `cur_pos` 4095"),
]
s = README.read_text()
rows = []
for kind, mode, shape in ORDER:
    r = perf[(kind, mode)]
    label = f"{mode} `{kind}`" + (" (traced)" if mode == "decode" else "")
    rows.append(
        f"| {label} | {shape} | {r['ops_in_window']} ({r['iters']} iters) | "
        f"**{r['device_kernel_ms_per_iter']:.2f} ms** | {r['op_to_op_gap_ms_per_iter']:.3f} ms | "
        f"{r['host_wall_ms_per_iter']:.2f} ms |"
    )
pat = re.compile(
    r"(\| case \| shape \| ops in window \| device kernel / iter \| op-to-op gap / iter \| host wall / iter \|\n"
    r"\|---\|---\|---\|---\|---\|---\|\n)(?:\|.*\n)+",
    re.M,
)
if pat.search(s):
    s = pat.sub(lambda m: m.group(1) + "\n".join(rows) + "\n", s)
    report.append("  updated  perf table")
else:
    report.append("  MISS     perf table header")

# three-way split table
split = []
for kind, mode, _ in ORDER:
    r = perf[(kind, mode)]
    b = r["blocks"]
    tot = r["device_kernel_ms_per_iter"]
    mx, ex, ot = b["mixer_ms_per_iter"], b["expert_matmul_ms_per_iter"], b["moe_elementwise_ms_per_iter"]
    split.append(
        f"| {mode} `{kind}` | {mx:.2f} ms ({100*mx/tot:.1f}%) | {ex:.2f} ms ({100*ex/tot:.1f}%) | "
        f"{ot:.2f} ms ({100*ot/tot:.1f}%) | {tot:.2f} ms |"
    )
pat2 = re.compile(
    r"(\| case \| token mixer \| expert matmuls \| MoE dense-intermediate elementwise \| total \|\n"
    r"\|---\|---\|---\|---\|---\|\n)(?:\|.*\n)+",
    re.M,
)
if pat2.search(s):
    s = pat2.sub(lambda m: m.group(1) + "\n".join(split) + "\n", s)
    report.append("  updated  three-way split table")
else:
    report.append("  MISS     split table header")
README.write_text(s)

# percentages quoted in the two bullets
gaps = [100 * perf[(k, m)]["op_to_op_gap_ms_per_iter"] / perf[(k, m)]["device_kernel_ms_per_iter"] for k, m, _ in ORDER]
hosts = [
    100 * (perf[(k, m)]["host_wall_ms_per_iter"] / perf[(k, m)]["device_kernel_ms_per_iter"] - 1) for k, m, _ in ORDER
]
s = README.read_text()
s = re.sub(
    r"Op-to-op gap is [\d.]+-[\d.]+% of device",
    f"Op-to-op gap is {min(gaps):.2f}-{max(gaps):.2f}% of device",
    s,
)
s = re.sub(
    r"\(`gap/device` = [^)]*\) for the four rows",
    "(`gap/device` = " + " / ".join(f"{g:.3f}%" for g in gaps) + ") for the four rows",
    s,
)
s = re.sub(
    r"host wall-clock exceeds device kernel time by [\d.]+-[\d.]+%",
    f"host wall-clock exceeds device kernel time by {min(hosts):.2f}-{max(hosts):.2f}%",
    s,
)
for kind, mode in (("full", "prefill"), ("full", "decode"), ("linear", "prefill"), ("linear", "decode")):
    pass
README.write_text(s)
report.append(
    f"  updated  gap/host percentages ({min(gaps):.2f}-{max(gaps):.2f}% / {min(hosts):.2f}-{max(hosts):.2f}%)"
)

# ---------------------------------------------------------------- host throughput
host_rows = {
    (r["mode"], r["kind"]): r
    for r in (json.loads(l) for l in (D / "perf_host_summary.jsonl").read_text().splitlines() if l.strip())
}
s = README.read_text()
s = re.sub(
    r"\*\*\d+ tok/s\*\* \(`full`\) / \*\*\d+ tok/s\*\* \(`linear`\) at seq 2048",
    f"**{host_rows[('prefill','full')]['tokens_per_s_host']:.0f} tok/s** (`full`) / "
    f"**{host_rows[('prefill','linear')]['tokens_per_s_host']:.0f} tok/s** (`linear`) at seq 2048",
    s,
)
s = re.sub(
    r"and decode \*\*\d+ tok/s\*\*\n\(`full`\) / \*\*\d+ tok/s\*\* \(`linear`\)",
    f"and decode **{host_rows[('decode','full')]['tokens_per_s_host']:.0f} tok/s**\n"
    f"(`full`) / **{host_rows[('decode','linear')]['tokens_per_s_host']:.0f} tok/s** (`linear`)",
    s,
)
README.write_text(s)
report.append("  updated  host throughput")

# ---------------------------------------------------------------- sparse derivation
fp = list(csv.DictReader((D / "tracy/full_prefill/prefill_perf_report.csv").open()))
sel = [r for r in fp if "2048 x 1024" in r["OP Code"]]
times = [float(r["Device Time"]) for r in sel]
tot, avg = sum(times) / 1000, sum(times) / 1000 / len(times)
per_iter = tot / perf[("full", "prefill")]["iters"]
tflops = 16 * 162.3 * (32 * 2048 * 1024 * 2) / (avg / 1000) / 1e12
s = README.read_text()
s = re.sub(r"appears \*\*\d+ times\*\* with", f"appears **{len(times)} times** with", s)
s = re.sub(r"Those rows sum to [\d.]+ ms,", f"Those rows sum to {tot:.2f} ms,", s)
s = re.sub(
    r"i\.e\. [\d.]+ ms per call and [\d.]+ ms per iteration",
    f"i.e. {avg:.2f} ms per call and {per_iter:.2f} ms per iteration",
    s,
)
s = re.sub(r"FLOP / [\d.]+ ms ~= [\d.]+ TFLOP/s", f"FLOP / {avg:.2f} ms ~= {tflops:.1f} TFLOP/s", s)
s = re.sub(r"runs at \*\*~[\d.]+ TFLOP/s", f"runs at **~{tflops:.1f} TFLOP/s", s)
README.write_text(s)
report.append(f"  updated  sparse derivation ({len(times)} rows, {avg:.2f} ms/call, {tflops:.1f} TFLOP/s)")

# ---------------------------------------------------------------- watcher
wlog = next((D / "watcher").rglob("watcher.log.gz"))
raw = gzip.decompress(wlog.read_bytes()).decode()
lines = raw.splitlines()
dumps = sum(1 for l in lines if l.startswith("Dump #"))
kids = sum(1 for l in lines if "k_ids" in l)
devrows = sum(1 for l in lines if l.startswith("Device "))
passed = re.search(r"(\d+) passed", (D / "watcher/pytest.log").read_text())
s = README.read_text()
s = re.sub(
    r"Result: \*\*\d+ passed, 0 failed\*\* \(`watcher/pytest\.log`\), watcher log \d+ lines / \d+ bytes raw",
    f"Result: **{passed.group(1)} passed, 0 failed** (`watcher/pytest.log`), watcher log {len(lines)} lines / {len(raw)} bytes raw",
    s,
)
s = re.sub(r"`watcher\.log\.gz`, [\d.]+ KB gzipped", f"`watcher.log.gz`, {wlog.stat().st_size/1000:.1f} KB gzipped", s)
s = re.sub(r"\d+ periodic core-status dumps", f"{dumps} periodic core-status dumps", s)
s = re.sub(
    r"holding \d+ `Device \.\.\.` core-state rows and \d+ `k_ids:` kernel-id lines",
    f"holding {devrows} `Device ...` core-state rows and {kids} `k_ids:` kernel-id lines",
    s,
)
README.write_text(s)
w = WORKLOG.read_text()
w = re.sub(
    r"`watcher/` — \d+ passed, \d+-line log \(\d+ dumps\)",
    f"`watcher/` — {passed.group(1)} passed, {len(lines)}-line log ({dumps} dumps)",
    w,
)
WORKLOG.write_text(w)
report.append(f"  updated  watcher ({passed.group(1)} passed, {len(lines)} lines, {dumps} dumps)")

# ---------------------------------------------------------------- tracy artifact sizes / ops-per-iter
sizes = {p.parent.name: p.stat().st_size for p in (D / "tracy").glob("*/[dp]*_ops.csv.gz")}
counts = {}
for kind in ("full", "linear"):
    rws = list(csv.DictReader((D / f"tracy/{kind}_decode/decode_perf_report.csv").open()))
    counts[kind] = len(rws) // perf[(kind, "decode")]["iters"]
if sizes and TRACY_README.exists():
    t = TRACY_README.read_text()
    t = re.sub(
        r"\(`linear_decode` [\d.]+ MB,\n`full_decode` \d+ KB",
        f"(`linear_decode` {sizes['linear_decode']/1e6:.2f} MB,\n`full_decode` {sizes['full_decode']/1e3:.0f} KB",
        t,
    )
    t = re.sub(
        r"records \d+ \(full\) /\n\d+ \(linear\) ops per iteration",
        f"records {counts['full']} (full) /\n{counts['linear']} (linear) ops per iteration",
        t,
    )
    t = re.sub(
        r"\(`full_prefill` \d+ KB, `linear_prefill` \d+ KB\)",
        f"(`full_prefill` {sizes['full_prefill']/1e3:.0f} KB, `linear_prefill` {sizes['linear_prefill']/1e3:.0f} KB)",
        t,
    )
    TRACY_README.write_text(t)
    s = README.read_text()
    s = re.sub(
        r"even gzipped \([\d.]+ MB / \d+ KB\)",
        f"even gzipped ({sizes['linear_decode']/1e6:.2f} MB / {sizes['full_decode']/1e3:.0f} KB)",
        s,
    )
    s = re.sub(
        r"both prefill CSVs \(\d+ KB / \d+ KB\) are committed",
        f"both prefill CSVs ({sizes['linear_prefill']/1e3:.0f} KB / {sizes['full_prefill']/1e3:.0f} KB) are committed",
        s,
    )
    README.write_text(s)
    report.append(f"  updated  tracy sizes {sizes} ops/iter {counts}")

# ---------------------------------------------------------------- advertised-context table
lc = {json.loads(l)["label"]: json.loads(l) for l in (D / "long_context.jsonl").read_text().splitlines() if l.strip()}
s = README.read_text()
for label, pattern in [
    (
        "longest-prefill[linear] seq=262143 tail=128",
        r"(\| `longest-prefill\[linear\]` \| seq_len \*\*262143\*\* \(non-aligned\), tail-128 compared \| )[\d.]+",
    ),
    (
        "longest-prefill[full] seq=262143 tail=128",
        r"(\| `longest-prefill\[full\]` \| seq_len \*\*262143\*\* \(non-aligned\), tail-128 compared \| )[\d.]+",
    ),
    (
        "longest-prefill state conv",
        r"(\| `longest-prefill\[linear\]` carried conv state \| after 262143 tokens \| )[\d.]+",
    ),
    (
        "longest-prefill state recurrent",
        r"(\| `longest-prefill\[linear\]` carried recurrent state \| after 262143 tokens \| )[\d.]+",
    ),
    (
        "longest-decode[linear] pos=262143",
        r"(\| `longest-decode\[linear\]` \| position \*\*262143\*\* after a 262143-token prefill \| )[\d.]+",
    ),
    (
        "longest-decode[full] pos=262143",
        r"(\| `longest-decode\[full\]` \| position \*\*262143\*\* after a 262143-token prefill \| )[\d.]+",
    ),
]:
    if label in lc:
        s = re.sub(pattern, lambda m: m.group(1) + f"{lc[label]['pcc']:.7f}", s)
README.write_text(s)
report.append("  updated  advertised-context table")

# ---------------------------------------------------------------- row counts
n_pcc = sum(1 for l in (D / "pcc.jsonl").read_text().splitlines() if l.strip())
n_rw = sum(1 for l in (D / "pcc_real_weights.jsonl").read_text().splitlines() if l.strip())
n_lc = sum(1 for l in (D / "long_context.jsonl").read_text().splitlines() if l.strip())
s = README.read_text()
s = re.sub(
    r"`pcc\.jsonl` \(\d+\), `pcc_real_weights\.jsonl` \(\d+\),\n`long_context\.jsonl` \(\d+\)",
    f"`pcc.jsonl` ({n_pcc}), `pcc_real_weights.jsonl` ({n_rw}),\n`long_context.jsonl` ({n_lc})",
    s,
)
README.write_text(s)
report.append(f"  updated  row counts ({n_pcc}/{n_rw}/{n_lc})")

# ---------------------------------------------------------------- batched long-context row
batched = next(
    (
        r
        for r in (json.loads(l) for l in (D / "long_context.jsonl").read_text().splitlines() if l.strip())
        if r["label"].startswith("batched-longest-decode")
    ),
    None,
)
if batched:
    s = README.read_text()
    s = re.sub(
        r"(\| `batched-longest-decode\[full\]` \| position \*\*262143\*\*, \*\*batch 2\*\*, slot 1 compared while slot 0 sits at `current_pos = -1` \| )[^|]*",
        lambda m: m.group(1) + f"{batched['pcc']:.7f} ",
        s,
    )
    README.write_text(s)
    report.append(f"  updated  batched long-context row ({batched['pcc']:.7f})")
else:
    report.append("  MISS     batched long-context row not in long_context.jsonl")

print("postprocess_docs:")
print("\n".join(report))


# =====================================================================================
# suite / probe counts
# =====================================================================================
def render_unary_group_total():
    """§7 item 6 quotes the traced-decode unary group total as an exact figure; derive it."""
    dec = list(csv.DictReader((D / "tracy/full_decode/decode_perf_report.csv").open()))
    iters = json.loads((D / "perf_summary.json").read_text())
    iters = next(r for r in iters if r["kind"] == "full" and r["mode"] == "decode")["iters"]
    unary = [r for r in dec if r["OP Code"].startswith("UnaryDeviceOperation")]
    total = sum(float(r["Device Time"]) for r in unary) / 1000 / iters
    per_iter = len(unary) // iters
    s = README.read_text()
    s2 = re.sub(
        r"shows \d+ unary ops/iteration instead of \d+ with the group total unchanged at [\d.]+ ms/iter",
        f"shows {per_iter} unary ops/iteration instead of {per_iter - 1} with the group total unchanged at "
        f"{total:.2f} ms/iter",
        s,
    )
    README.write_text(s2)
    report.append(f"  {'updated ' if s2 != s else 'same    '} unary group total ({per_iter}/iter, {total:.3f} ms/iter)")


render_unary_group_total()


def render_counts():
    """The counts that grew with the suite and were left behind by hand-editing, four rounds running."""
    main = (D / "logs/test_suite_main.log").read_text()
    collected = int(re.search(r"collected (\d+) items", main).group(1))
    passed = int(re.search(r"(\d+) passed", main).group(1))
    summary = main.split("short test summary info")[-1]
    cpu = summary.count("test_reference_math")
    dev = summary.count("test_functional_decoder")
    probe_log = (D / "logs/probe_ttnn_ops.log").read_text()
    probes = int(re.search(r"PROBE SUMMARY (\d+)/(\d+) ok", probe_log).group(2))
    keys = main.count("[layer_pairs] building")
    chunked_pcc = re.search(r"chunk_start_idx_tensor.*?pcc ([\d.]+)", probe_log, re.S)

    s = README.read_text()
    s = re.sub(r"both files in one invocation \(\d+ items\)", f"both files in one invocation ({collected} items)", s)
    skipped = re.search(r"(\d+) skipped", main)
    # State the skip rather than hide it behind "0 failed". The only test that can skip here is the
    # doc-drift check, which stands down while the docs have not been rendered for this run's
    # evidence yet; the pass re-runs it afterwards (`logs/test_docs.log`).
    tally = f"**{passed} passed, 0 failed**"
    if skipped:
        tally += (
            f" (plus {skipped.group(1)} skipped: `test_docs_match_artifacts` defers until the docs are "
            "rendered from this run's artifacts, and `logs/test_docs.log` records it passing after that)"
        )
    s = re.sub(
        r"\*\*\d+ passed, 0 failed\*\*(?: \(plus \d+ skipped:[^)]*\))?: \d+ CPU-only \+ \d+ device cases",
        tally + f": {cpu} CPU-only + {dev} device cases",
        s,
    )
    s = re.sub(
        r"the \d+ device cases share (?:about a dozen|\d+ of them)",
        f"the {dev} device cases share {keys} of them",
        s,
    )
    s = re.sub(r"\| \d+ device op-behaviour probes", f"| {probes} device op-behaviour probes", s)
    if chunked_pcc:
        s = re.sub(r"build \(PCC [\d.]+\)", f"build (PCC {chunked_pcc.group(1)})", s)
    README.write_text(s)

    w = WORKLOG.read_text()
    w = re.sub(
        r"\(\d+ tests: \d+ CPU-only \+ \d+ device\)",
        f"({passed} tests: {cpu} CPU-only + {dev} device)",
        w,
    )
    if chunked_pcc:
        w = re.sub(
            r"ok, pcc [\d.]+ \| the device-tensor offset form",
            f"ok, pcc {chunked_pcc.group(1)} | the device-tensor offset form",
            w,
        )
    WORKLOG.write_text(w)
    report.append(f"  updated  counts ({passed} passed = {cpu} CPU + {dev} device, {probes} probes, {keys} layer keys)")


render_counts()

print("render_docs --check:" if CHECK else "render_docs:")
print("\n".join(report))
if any("MISS" in r for r in report):
    raise SystemExit("at least one anchor was not found")

if CHECK:
    drifted = []
    for name, rendered in (
        ("README.md", README),
        ("work_log.md", WORKLOG),
        ("tracy/README.md", TRACY_README),
    ):
        if (D / name).read_text() != rendered.read_text():
            drifted.append(name)
    shutil.rmtree(README.parent, ignore_errors=True)
    if drifted:
        raise SystemExit(
            "render_docs --check: committed docs disagree with the artifacts in "
            + ", ".join(drifted)
            + " -- run `python models/autoports/qwen_qwen3_6_35b_a3b/tests/render_docs.py`"
        )
    print("render_docs --check: committed docs match the artifacts")
