# ditcheck on the Galaxy — development plan & runbook

Bringing the collective-redundancy analyzer to a full **4×8 Galaxy (32 chips, Ring
topology)**. Everything on device so far ran on a 2×4 Loudbox with Linear links — a
scaled-down proxy. The Galaxy is the real production machine; this plan spends that
silicon on what it uniquely provides: **proof and measurement at true scale and topology.**

Companion status of record: [`DitStaticAnalyzerRoadmap.md`](DitStaticAnalyzerRoadmap.md).

---

## Operating rule — where each machine earns its keep

- **Laptop (free):** dry-run, analyze, trace, and triage every finding. Never spend Galaxy
  time re-deriving a fact the shim already knows.
- **Galaxy (scarce):** every run does one of three things only silicon can — **conform a
  finding, measure a win, or capture a reference.**

---

## The plan — six workstreams, in priority order

**1. Re-conform the proven three, at true scale** — **DONE 2026-08-08, all three green on
4×8 Ring.** The 2×4 could only show 1 of 2 SP rows redundant and TP=2; the Galaxy shows the
real **7 of 8** and **TP=4**, on the Ring fabric the model actually uses.

| finding | 2×4 (Linear) | 4×8 (Ring) | evidence |
|---|---|---|---|
| encoder `replicated_stage` | 1 of 2 SP rows | **7 of 8**, TP=4 | `max\|Δ\|=0` on every TP column |
| audio `unused_gather` (node_360) | 1 of 2 shards | **7 of 8** SP shards | `max\|Δ\|=0`, audio ⊂ shard 0 |
| output-head `participant_shrink` (node_348) | TP=2 | **TP=4**, 3 of 4 copies unread | `max\|Δ\|=0` across all 32 devices |

No verdict changed between the proxy and true scale — only the counts — which is exactly the
bar the watch-out below sets. Three harness defects had to be cleared first; see
[§ What the first Galaxy run cost](#what-the-first-galaxy-run-cost).

**2. Conform the rest of the report** — *in progress; triage done, first survivor class conformed.*
The other classes — `replicated_stage`, `overwide_gather`, `participant_shrink` — are un-tainted
`likely`/`provable` but never device-checked. Trace-then-conform (laptop first); only survivors
reach silicon.

The `prod` report is **23 findings over 15 distinct nodes**. Laptop triage (`scouts/triage_w2.py`)
found no shim artifacts — and one shared root cause. The packed sequence is
`[text 512 | audio 414 | video 37296]` padded to 38400 over SP=8, so shard 0 spans rows
`[0:4800)`: **the whole text and audio block lives on SP column 0.** Everything follows:

| class | nodes | traffic | status |
|---|---|---|---|
| output-head cluster (`unused_gather`, `participant_shrink`, `replicated_stage`) | 344, 352, 356 | — | **conformed** (workstream 1, `conform_dit_heads.py`) |
| `replicated_stage`, DiT text branch | 63, 92, 100, rs_110 | ~477 MiB/fwd | **conformed 4×8 Ring** — `conform_dit_text.py`, both checks `max\|Δ\|=0` |
| `replicated_stage`, encoder | 22, 78, 86, 116, rs_110 | ~588 MiB/fwd | mechanism conformed at stage level (`conform_encoder.py`); per-node intermediate readback would strengthen |
| `overwide_gather` | 274, 296, 302 | ~67 MiB/fwd | **traced, real, not yet on silicon** — see below |

The `overwide_gather` gap is exact, not approximate: on device 0 the spatial matmuls read
`[512:4800)` of a `[0:4800)` shard — they skip precisely the text rows (the "11% more data"); on
device 7 they skip `[38222:38400)`, the 178-row padding tail (the "4%"). The device check that
fits it is a **poison test**: fill the claimed-unread region with a sentinel, run the real fused
AGMM, and show every consumed output is bit-identical — which also pins down that the fused
gather+matmul keeps rows independent.

**3. Validate the whole pipeline against a real run** *(highest value).*
32 chips is the first machine that can run the real `pipeline_minimax_h3.generate()` — the
thing the scout stands in for.
- Drive the real pipeline with a collective logger; diff its actual op/axis/count/shape log
  against the scout's linked graph.
- Exercise `capture.py` on hardware for the first time (its hooks have never run on a device).
- Reconcile any mismatch as a **shim bug, not model waste** (the verify-loop discipline).
- Proves the scout is faithful and retires the "run directly, not via scout" blockers.

**4. From counted bytes to a measured win** *(the proof).*
The tool reports recoverable traffic as byte-counts; only real fabric turns that into measured
latency/bandwidth.
- Profile the flagged collectives on the Ring fabric.
- Implement one fix end-to-end — e.g. run the encoder on a `1×8` submesh instead of replicating
  across SP — and measure: outputs identical, traffic and latency down. One finding becomes a
  landed optimization with a number.

**5. Ring collectives & the soundness gates (11c).**
The shim *models* Ring but has only ever been conformed on Linear. Conform the ring all-gather /
reduce-scatter and the fused ring-joint SDPA at 4×8; then the buffer-liveness and
memory-feasibility gates that need real device state.

**6. Real depth & the config matrix (12).**
Full layer count so aggregate traffic is measured, not extrapolated; `ditcheck matrix` over
topology × resolution × `has_audio`, device-sampled.

**Watch-out that governs all of it:** scale changes the *numbers* (7/8, not 1/2); it must not
change the *verdicts*. A finding that flips real↔false at 4×8 Ring is a shim bug to chase, not a
new result — expect Ring to surface shim gaps, and treat them as tool bugs first.

---

## Running on device (the runbook)

The conform harnesses are self-contained: each sets `FabricConfig.FABRIC_1D`, opens the mesh,
runs real ttnn collectives, reads every device shard back, and diffs against the shim's claim.
You only supply the environment and the mesh shape.

```bash
# environment (adjust ARCH_NAME to the Galaxy's architecture)
export TT_METAL_HOME=/path/to/tt-metal
export PYTHON_ENV_DIR=$TT_METAL_HOME/python_env
export ARCH_NAME=<galaxy arch, e.g. blackhole | wormhole_b0>

# workstream 1 — the fastest credible win (ring is the default now)
python3 models/tt_dit/tools/dit_analyzer/conform_encoder.py   --mesh 4 8 --sp-axis 1 --tp-axis 0
python3 models/tt_dit/tools/dit_analyzer/conform_dit_heads.py --mesh 4 8
# expect: encoder 7 of 8 SP rows redundant · audio 7 of 8 shards · output-head TP=4, all max|Δ|=0

# on the 2x4 Loudbox instead (Linear links):
python3 models/tt_dit/tools/dit_analyzer/conform_encoder.py --mesh 2 4 --topology linear
```

**`--sp-axis 1 --tp-axis 0` is not optional on the Galaxy.** Which axis carries which
parallelism is a property of the *config*, not the mesh: production H3/LTX runs `sp1tp0`, so a
4×8 is SP=8 / TP=4 — that is where 7-of-8 comes from. `conform_encoder` defaults to the 2×4
Loudbox's `sp0tp1`, which on a 4×8 silently measures SP=4 / TP=8 and reports a true but
different finding ("3 of 4"). `conform_dit_heads` derives SP from the larger axis, so it needs
no flag.

If you drive the device through the `tt-device-mcp` broker rather than a direct shell, pass the
same three env vars as `inherited_env` and tag the job owner; the harness command is unchanged.

**The whole-pipeline report** (workstreams 2, 3, 6) comes from the scout — but it needs the H3
model code, which is the branch situation below:

```bash
# from a tree that has BOTH the analyzer tool and the H3 models (see next section)
python3 models/tt_dit/tools/dit_analyzer/scouts/scout_h3_pipeline.py prod   # 768p/5s, packed_seq 38400
python3 models/tt_dit/tools/dit_analyzer/scouts/render_full.py              # all findings, not just top-8
```

Presets: `2x4` (quick smoke), `4x8` (packed_seq 2048), `prod` (production 768p/5s).

---

## What the first Galaxy run cost

Nothing in workstream 1 failed because a finding was wrong. Every failure was the harness or
the box meeting 32 chips for the first time. Recorded so the next workstream doesn't re-pay:

1. **Ring topology needs a ring fabric.** Both harnesses set `FabricConfig.FABRIC_1D` while
   asking `CCLManager` for `Topology.Ring` — a ring hop across the seam then has no route and
   ttnn aborts with `Could not find any forwarding direction from src (M0, D0) to dst (M0, D28)`.
   On the 2×4 this never showed: Linear was the only topology ever run. Fixed — the fabric now
   follows the topology (`FABRIC_1D_RING`), matching every other in-tree Galaxy ring config
   (`gpt_oss`, `gemma3` 6U).
2. **The mesh axes were the Loudbox's, hardcoded.** See the runbook note above.
3. **A probe constant that only modelled SP=2.** `conform_dit_heads` packed `SEQ=4096`, so at
   SP=8 the shard is 512 rows and the audio block `[512:926)` no longer lands in shard 0 — the
   assertion fired. That is a property of the *probe*, not the model: production packs 38400
   rows, where audio sits well inside the first shard. The probe sequence now scales with SP
   (7424 at SP=8, tile-aligned), with `--seq` to override.
4. **The board needed a reset before it would open at all.** Every device init — even a
   single-chip `open_device` — threw `IndexError: unordered_map::at` from
   `Cluster::initialize_ethernet_sockets`, with UMD's `system_health` showing chan 8/9 wrap
   links split three ways (garbage peer id / local peer / down). `tt-smi -glx_reset` cleared it.
   An aborted run wedges an eth core (`Timed out while waiting for active ethernet core ... to
   become active again`) and needs the same reset before the next attempt.

## Where the code lives (the branch story) — read before running the scout

The analyzer **tool** is developed on branch `rsalman-dit-static-analyzer`. That branch carries
the tool, all conform harnesses, the Qwen3-VL **encoder** model, and the generic `layers/`, so
**the conform harnesses (`conform_encoder`, `conform_dit_heads`) run from this branch alone** —
they rebuild what they need from the encoder and generic Linear + CCLManager.

The **whole-pipeline scout** (`scouts/scout_h3_pipeline.py`) imports the DiT, VAE, and audio-VAE
models, which live on the **MiniMax-H3 integration branch (`cglagovich/minimax-h3`)**, *not* on
the analyzer branch. To run the scout you need one working tree that has **both**. Two ways:

1. **Copy the tool into an H3 tree** (the pattern used in development):
   ```bash
   # check out / worktree the H3 branch, then drop the analyzer tool in:
   rsync -a <analyzer-tree>/models/tt_dit/tools/dit_analyzer/ \
            <h3-tree>/models/tt_dit/tools/dit_analyzer/
   ```
2. **Land the analyzer onto the H3 branch** (or main) so tool and models coexist permanently —
   the durable fix, and the recommended next step for shared use.

Until (2) happens, remember to re-sync (1) whenever either side moves — the tool in the H3 tree
is a copy, not the source of truth.

---

## The visual version

A one-glance version of this plan (the six workstreams, the mesh motif, the start-here command)
is published as an Artifact — ask the tool owner for the link, or regenerate from
`scratchpad/galaxy-plan.html`.
