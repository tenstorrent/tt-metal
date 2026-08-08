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

**1. Re-conform the proven three, at true scale** *(start here — hours).*
The 2×4 could only show 1 of 2 SP rows redundant and TP=2; the Galaxy shows the real
**7 of 8** and **TP=4**, on the Ring fabric the model actually uses.
- `conform_encoder.py --mesh 4 8` and `conform_dit_heads.py --mesh 4 8` (they parameterize on
  mesh, and now default to `--topology ring`).
- Confirms the headline numbers (encoder 168 MiB × 7/8, audio & output-head waste) are real,
  not proxy-scaled.

**2. Conform the rest of the report.**
The other classes — `replicated_stage ×10`, `overwide_gather`, `participant_shrink` — are
un-tainted `likely`/`provable` but never device-checked. Trace-then-conform (laptop first);
only survivors reach silicon. Flips the whole report from "shim believes" to **device-proven
at 4×8**.

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
python3 models/tt_dit/tools/dit_analyzer/conform_encoder.py   --mesh 4 8
python3 models/tt_dit/tools/dit_analyzer/conform_dit_heads.py --mesh 4 8
# expect: encoder 7 of 8 SP rows redundant · audio 7 of 8 shards · output-head TP=4, all max|Δ|=0

# on the 2x4 Loudbox instead (Linear links):
python3 models/tt_dit/tools/dit_analyzer/conform_encoder.py --mesh 2 4 --topology linear
```

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
