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

**2. Conform the rest of the report** — **DONE 2026-08-08. Every class is device-proven at 4×8 Ring.**
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
| `overwide_gather` | 274, 296, 302 | ~67 MiB/fwd | **conformed 4×8 Ring** — `conform_overwide.py`, poison test on all 8 devices |

The `overwide_gather` gap is exact, not approximate: on device 0 the spatial matmuls read
`[512:4800)` of a `[0:4800)` shard — they skip precisely the text rows (the "11% more data"); on
device 7 they skip `[38222:38400)`, the 178-row padding tail (the "4%").

**Equality can't conform that class** — the claim is not that two things are equal but that a
region is *never read* — so `conform_overwide.py` **poisons** it: fill exactly the claimed-unread
rows with a sentinel, run the real fused `all_gather_minimal_matmul_async` (the call site the
finding names), and require both halves. Every row the consumers do read came back bit-identical
(`max|Δ|=0`, all 8 affected devices) **and** the poisoned rows' own outputs moved by ~2×10⁶, so the
sentinel demonstrably landed and the null result is evidence rather than a no-op. That also pins
down the property the fix depends on: the fused gather+matmul is row-independent, so trimming the
unread rows cannot perturb anything downstream.

**3. Validate the whole pipeline against a real run** *(highest value).*
32 chips is the first machine that can run the real `pipeline_minimax_h3.generate()` — the
thing the scout stands in for.
- Drive the real pipeline with a collective logger; diff its actual op/axis/count/shape log
  against the scout's linked graph.
- Exercise `capture.py` on hardware for the first time (its hooks have never run on a device).
- Reconcile any mismatch as a **shim bug, not model waste** (the verify-loop discipline).
- Proves the scout is faithful and retires the "run directly, not via scout" blockers.

**4. From counted bytes to a measured win** — *measured 2026-08-08; the number is not the one this
plan assumed.* The encoder-on-a-submesh fix was measured end to end with
`measure_encoder_submesh.py` (Tracy, warm window via signposts, construction outside it). At
production `sp1tp0` the submesh is **4×1**, not the `1×8` written above — TP=4 intact, one SP
column.

| arm | chips | device time/fwd | op-to-op gap/fwd | untraced window/fwd |
|---|---|---|---|---|
| full 4×8, 11×10 grid | 32 | 2824–2834 µs | 5233–5886 µs | 8067–8709 µs |
| submesh 4×1, 11×10 pinned | 4 | 3689–3812 µs | 939–1378 µs | 4751–5067 µs |
| submesh 4×1, 12×10 (real behaviour) | 4 | 3152 µs | 1298 µs | **4450 µs** |

**Outputs are bit-for-bit identical** to the full-mesh SP-column-0 shards on every TP row
(`max|Δ|=0`), so the fix is exact. Splitting device time into kernel and firmware says where the
rest goes — and the split matters, because `DEVICE FW DURATION` alone reads as a compute
regression that isn't one:

| arm | kernel/fwd | FW wait | op-to-op gap | window |
|---|---|---|---|---|
| full, 32 chips | **2539.6 µs** | 294.3 | 5232.8 | 8066.7 |
| submesh, 4 chips | **2527.0 µs** | 1284.5 | 939.1 | 4750.6 |

- **Real compute is identical within 0.5%**, as the analysis requires: every device runs the same
  program on the same shapes either way, and `CORE COUNT` is identical op for op. Per-op kernel
  times match — `Embeddings` 17.68 vs 17.71 µs, `Clone` 29.20 vs 28.40, `MinimalMatmul` 284.17 vs
  281.92.
- The ~41% untraced improvement is **entirely dispatch**: ~4300 µs of op-to-op gap disappears
  (driving 4 devices instead of 32), while ~990 µs of it *relocates* into device firmware wait.
  On the submesh the host keeps up, so each op is picked up immediately and then stalls on its
  data dependency inside FW rather than idling between ops. Same stall, different column.
- `AllGather` per-device time is **unchanged** (597 → 591 µs): each TP group does identical work
  either way, and the saving is *aggregate* traffic — 8 concurrent groups collapsing to 1. `SDPA`
  matches to 0.1 µs.

**What this means for production, which runs traced.** Tracing removes host dispatch — the term the
submesh wins on — so traced, the two arms should converge on kernel time and the fix is **latency-
neutral**, while still freeing **28 of 32 chips** and cutting encoder link traffic 8×. So it is a
*capacity* win, not a latency one: "traffic down ⇒ latency down" does not hold here, and the tool
should not imply that it does. Untraced it also happens to halve the encoder's wall clock.

> **Measurement note, the hard-won kind.** The first pass read `DEVICE FW DURATION` alone and
> concluded per-device compute had regressed ~12%, concentrated in small ops. It had not — FW
> includes time the firmware spends waiting, and on a fast-dispatch configuration that wait grows
> precisely because the host stopped being the bottleneck. **Always split FW into kernel and
> FW-minus-kernel before calling anything a compute regression.**

**A confound worth knowing about for any submesh measurement:** `get_matmul_core_grid` clamps to
11×10 only at ≥32 devices (a BH Galaxy power constraint), so a 4-device submesh silently takes
12×10. That is not a free 20% — it changes the matmul's output sharding, which downstream
elementwise ops inherit, and it moves small-op times in both directions. `--grid clamped` pins it
so the A/B isolates the finding.

**5. Ring collectives & the soundness gates (11c)** — *ring collectives done; the rest is split
between one real task and three that are correctly parked.*

- **Ring all-gather / reduce-scatter at 4×8: DONE 2026-08-08.** `conform_collectives.py --mesh 4 8
  --topology ring --sp-axis 1 --tp-axis 0` → **3/3** (ag_tp, ag_sp, rs_tp) match real ttnn
  per-device shapes. The harness had only ever run Linear on the 2×4 (topology and axes were baked
  in); both are now selectable, and it needed the same `FABRIC_1D_RING` pairing as the others.
- **Fused ring-joint SDPA at 4×8: open.** `conform_block.py` is the harness — it already
  reconciles the fused call into its two hidden K/V sp all-gathers — but it needs a dry-run graph
  regenerated for a 4×8 SD3.5 block, and SD3.5's shapes have to divide 8-way SP. That is a small
  bring-up, not a re-run.
- **The three 11c gates stay parked, deliberately.** The roadmap assessed them and deferred rather
  than faked: *buffer liveness* needs buffer identity from a capture that does not exist (blocker
  29) — a soundness verdict built on shim-modelled ping-pong slots would itself be "the shim
  believes"; *barrier/sync intent* is implementable but does not fire on any current finding (the
  gathers take the persistent-buffer, no-barrier path); *live-bytes / memory feasibility* needs an
  L1/DRAM residency model (blocker 30). Building one just to make the gate emit a verdict is the
  failure mode that note warns against. Worth recording: the fix this tool most wants to
  recommend — the encoder submesh — is **memory-neutral by construction**, since SP carried
  replication rather than shards, so each device holds exactly what it held before.

**6. Real depth & the config matrix (12)** — *depth done 2026-08-08; the matrix is still open.*
The scout now runs production depth at the `prod` preset: **encoder 50** layers
(`MINIMAX_H3_TEXT_ENCODER_LAYER`, the tap `pipeline_minimax_h3` actually builds), **DiT 50**
(`project_block_perf.py: DEFAULT_LAYERS`), refiner 2 (already real). Override per stage with
`DITCHECK_ENC_LAYERS` / `DITCHECK_DIT_LAYERS` / `DITCHECK_REFINER_LAYERS`.

Cost is linear and cheap — 5838 nodes, 512 distinct collectives, **21.7 s** on a laptop, memory
flat — so there was never a reason to report one layer.

**Depth changed the ranking, not just the magnitudes.** That is the result:

| | 1 layer | real depth |
|---|---|---|
| recoverable traffic | 2.7 GiB/forward | **26.3 GiB/forward** |
| findings | 27 | 321 (25 distinct) |
| #1 finding | DiT output-head `participant_shrink`, 664.5 MiB | **encoder `replicated_stage`, 8.2 GiB** |

At one layer the DiT output head looked like the biggest prize. At real depth the **encoder
dominates: 23.5 of 26.3 GiB (89%)** across four call sites in `model_qwen3vl.py`, each repeating
50×; the output head falls to #5. Since that encoder redundancy is 7-of-8 replication, the submesh
fix from workstream 4 removes roughly **20.6 GiB per forward** — which is the magnitude that
workstream was looking for, and it makes the fix already conformed *and* measured the top item.

**The rollup that made this readable (phase 9).** 321 findings behind a top-8 cut showed one
repeat eight times instead of eight distinct problems. `report.rollup_findings` groups on what
makes two findings the same problem — rule, source chain, per-call bytes, verdict — and ranks by
*total* impact, so the report reads "×50 occurrences (same call site, one per layer) = 8.2 GiB
across the stack". 321 → 25 rows, and the ranking finally reflects what is worth fixing first.

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
