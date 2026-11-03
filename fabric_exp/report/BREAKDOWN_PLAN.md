# Where the router cycles go: breakdown plan (phase 1, 2026-11-02)

The report page will show, for main's router and the new router at five test points, how each ERISC's cycles per
packet split across seven shared categories. The data comes from PC samples mapped to source lines.

The user's decisions of 2026-11-02 are applied throughout:
- the "waiting" category is dropped, and the new router's A/B spare time is a separate marker (§1);
- the router rule for multi-router tests (§4.1);
- point 4 runs at the default slots with 1,024 B packets;
- point 3 is exp_2d_4352.yaml's MeshMulticast write.

Phase 1 is offline: the taxonomy, the line maps, the tooling, validation on samples that already exist, the sampling
plan, and a provisional `data/breakdown.json`. Phase 2 is the device runs: the P150 broker job, after the main
session's go. The Galaxy points are now run by a separate Galaxy agent (plan change, 2026-11-02).

**What the page may say.** Where an ERISC's cycles go. Not which ERISC bounds a row, and not why one router is faster
(repo CLAUDE.md).

- Routers poll without stopping, so every ERISC's samples add up to the row's full rate, idle or not.
- The bound side comes only from one-side A/Bs (slack.json, V4_BOUNDS.md), which exist for the new router only.
- Main's bound side is not measured at any of the five points. Its only A/Bs (RESULTS §2) ran on a pre-merge tree at
  2 KiB and are context only.

## 1. Taxonomy (`breakdown/taxonomy.json`)

| id | label | what is in it |
| --- | --- | --- |
| checks | Packet polls and room checks | Per-pass tests for work. Is there a packet: the receive ring's packets-received register, a sender channel's free-slots register. Is there room: the far ring's free slots, downstream admission, transaction-ID reuse. |
| decode | Header and route decode | Locating and reading the header, decoding the route, the command switch, rewriting the header for the next hop. |
| noc | NoC commands | Forward and local delivery (write, scatter, atomic, inline): command-buffer waits, register writes, transaction IDs, the downstream ring's slot and credit-register updates. |
| eth_send | Ethernet send | Header size read, TX-queue status checks and busy spins, TX-queue writes, the far packets-received update, slot advances. |
| credits | Acks, completions and credits | Receiving ERISC: first-level acks, completion proofs, credit messages and frames, slot release. Sending ERISC: reading the far receiver's acks and completions, and crediting producers (workers, upstream routers). |
| loop | Connection and loop overhead | Connection and teardown checks, termination and context switch, round and loop bookkeeping, the pass structure, per-channel state loads and stores, telemetry and trimming hooks. |
| other | Other router code | Init and rare paths that a rule names. |

**Where idle time goes: "waiting" is dropped** (user, 2026-11-02; taxonomy.json `dropped`). Lines that run only on an
empty pass came to 0.0-0.1 cycles/packet on every sample set. A busy pass and an empty pass run the same polling
instructions; an empty pass simply skips the work. So idle time is billed to the per-pass lines it executes (checks,
credits, loop), and a "waiting" bar would mislead.

Its two rules are folded into the categories they share code with:
- the receiver's early return when no packet is ready goes to checks (the readiness test);
- the partial completion group's idle-pass count goes to credits (the credit decision it belongs to).

**Spare time is a separate marker instead.** The new router's spare time per ERISC comes from the one-side delay A/Bs
(slack.json). build_breakdown.py attaches it to each record:
- `spare_cycles`: the A/B slack, or null;
- `spare_label`: text the page can draw as a marker, e.g. "receiving ERISC: ~27 cycles spare";
- `spare_source`: the slack.json record, build and jobs.

The spin ran on every router's E0, or every router's E1, at once, so the value is the row's spare time on that side.
At point 1, v4rxv2i3 has 5.3 spare cycles on E0 and 26.6 on E1, so E1's 51 cycles of credits there include up to ~27
cycles of polling. Where no A/B exists, `spare_cycles` is null and `spare_label` says why:
- **main: "spare time not measured (main has no one-side A/B)"**;
- a new-router proxy on another row or build: "not measured for this build and row".

**Stalls show at the first use.** A stalled load shows where its value is first used, so a category boundary can move
a stall between categories. The rules below put first uses with their loads where the code allows:

- main's packets-received read and its first test are both checks;
- the new router's E0 group read and its first test are both checks;
- the route word's first use is decode.

Where they cannot, a rule note says so:

- the new sender's ack and completion counters are read inside the send's post-send load group, mapped to credits;
  their wait can land on the next send line;
- the header's send type is loaded in decode, and its first use is the command switch, which is also mapped to decode.

**Noise.** Every record carries its window sample count and a binomial standard error (`se_cycles`). The existing sets
have 3,600 to 36,000 window samples per target, which gives ±0.8 to ±1.6 cycles. Records under 3 cycles/packet are
flagged `noise`.

## 2. Line maps and how they are applied

**Files:**
- `breakdown/map_main.json`: 54 rules over fabric_erisc_router.cpp, its speedy path, and the packet-transmission and
  utility headers;
- `breakdown/map_new.json`: 61 rules over router/*.hpp, the kernel, and the packet-transmission header;
- `scripts/bd_map.py`: the engine, with commands `check`, `lines`, `resolve` and `static`;
- `breakdown/sets.json`: sample sets, each with its samples, cache, source layers and map;
- `breakdown/points.json`: which set stands for each point and router;
- `scripts/build_breakdown.py`: writes data/breakdown.json;
- `scripts/bd_phase2_sets.py`: adds phase-2 runs to the sets and points.

**Rule forms.** Each rule names a file (a basename is enough) and optionally a function regex, then one of:
- `match`: a regex over the line text;
- `from` / `to`: text anchors that find a block. The hint line number is documentation only;
- neither: the whole function, or the whole file.

**Resolution.** Rules are resolved against the build's own file versions on every run. A rebuilt variant whose lines
moved still maps. A rule whose anchor is gone is listed as "not resolved, re-map". Rules for builds that lack the code
are marked `optional` or `legacy`, so they do not trigger that warning.

**Classification.** addr2line -i gives each sampled address's inline chain, innermost frame first. The first frame that
a rule matches decides the category. The most specific rule wins: a match rule beats a block, a block beats a whole
function, a function beats a whole file.

- Frames in files no rule names are skipped, so library code takes its caller's category: NoC register writes, stream
  registers, the TX queue, counters.
- `defer` rules pass a frame on to its caller.
- File-level `fallback` rules catch the outer loop. Samples that reach a category only through a fallback are counted
  and listed separately from those a specific rule maps.

**Sources are never the live tree.** Broker jobs swap variant files, including headers that main's router shares, into
the live tree. A swap was active during this work, swapping 11 files and creating 10.

Each set lists source layers instead:
- `variant:DIR`: a variant directory over `git:HEAD`;
- `tgz:FILE`: a snapshot the phase-2 job takes while that router is in the tree;
- `dir:TREE`: a tree.

Main's router files are unchanged in HEAD since the merge of main 6d8a103.

**Build identification.** The image a core runs is found from the launch message's text size and the 256 bytes the
sampler read from L1.

- The text size equals the packed size of the ELF's loadable segments, a new finding: an exact match, where
  pcs_report.py matched on the head alone.
- Candidates are restricted to the config hashes in the run's compile log.
- Builds that share size and head but differ in constants are all mapped. The share of samples whose category differs
  between them is reported as "ambiguity": 0.0% on every set except prlg3ml_hop3 (1.4%).
- The JIT's own `<proc>.elf` carries `-g` (`.debug_line`), so no `-g` twin builds are needed while the run's cache
  exists.

## 3. Validation

### 3.1 On existing samples

All rows are P150. Cycles per packet are the share times the sampled run's own rate. Every target is 100.0% mapped by
specific rules, with 0.0% unmapped.

| set (build, job) | row, rate | ERISC | samples | checks | decode | noc | eth_send | credits | loop |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pcse_v1 (main, one ERISC, job 131) | ring 1-hop bidir 1024 B, 364.1 | 0/6 E1 | 12,267 | 30.9 | 37.2 | 37.3 | 72.0 | 171.8 | 14.9 |
| | | 1/5 E1 | 12,267 | 28.6 | 37.9 | 38.5 | 69.8 | 173.9 | 15.4 |
| b2d (v4rxv2i3, job 358) | 2D 1-hop write 4352 B, 126.9 | 1/5 E0 | 5,557 | 48.8 | | | 51.7 | 16.7 | 9.7 |
| | | 0/6 E1 | 5,557 | 12.4 | 24.2 | 32.8 | | 51.2 | 6.3 |
| b2d | 2D multicast write 4352 B, 141.7 | 4/4 E0 | 36,393 | 35.3 | | | 61.6 | 32.1 | 12.6 |
| | | 4/5 E1 | 36,395 | 15.3 | 20.9 | 58.4 | | 42.1 | 4.9 |
| b2d | 2D 3-hop write, 146.2 | 0/7 E0 | 5,700 | 50.2 | | | 63.3 | 19.5 | 13.2 |
| | | 2/4 E1 | 5,700 | 15.9 | 25.2 | 51.9 | | 48.1 | 5.2 |
| prlg3ml (v4rxg3 + N/S shape, job 459) | 2D 3-hop write, 179.7 | 2/5 E0 | 6,774 | 68.6 | | | 60.8 | 29.7 | 20.6 |
| | | 2/4 E1 | 6,774 | 31.2 | 17.6 | 86.9 | | 33.0 | 10.8 |
| bring (v4rxv2i3) | ring 1-hop write, 140.9 | 1/5 E0 | 3,604 | 28.6 | | | 53.8 | 45.7 | 12.7 |
| | | 0/6 E1 | 3,604 | 5.9 | 18.4 | 32.1 | | 80.6 | 3.9 |
| btor (v4rxv2i3) | Y-torus N/S 1-hop write, 149.2 | 1/8 E0 | 3,781 | 48.2 | | | 51.7 | 42.1 | 7.3 |
| | | 5/11 E1 | 3,780 | 6.3 | 26.9 | 33.2 | | 79.6 | 3.2 |

The 1024 B 1-hop set (bew1024) maps the same way.

**Cross-checks against earlier source-line profiles** (the same samples, other tools):
- b2d ew4352 E0: send 51.7 plus 3.1 of its counter loads (under credits) gives 54.8. V4_BOUNDS §3's send phases add up
  to 54.9.
- prlg3ml hop3 E1: downstream admission 24.1 against V4_ROUTE_PRELOAD §8.2's 16.6 + 6.1. The route lines show 14.5
  against 12.0 plus the header word.
- pcse_v1 (main): the receiver's acks and completions come to ~90 by these rules. RESULTS §98 found ~89 by line.

**The main sample set needed a substitute cache.** The pcse_v1 run's own cache is deleted. Its images match, by packed
size and head, builds of the same config hashes in ~/.tt-metal-cache-cen-main_fw-one: main as shipped, post-merge,
2026-10-29, the same sources as the pcse run on 2026-10-28 22:42, after the 18:42 merge. Those builds' line maps agree
on 100% of the samples.

### 3.2 On code no sample reached (`bd_map.py static`)

This check classifies every instruction of a build's main loop, weighted by instruction rather than by sample. Unmapped
instructions are 0 on every build checked. Specific rules cover:

| build | E0 | E1 |
| --- | --- | --- |
| main 2D mesh, two ERISCs | 95.6% | 97.0% |
| main ring (first-level acks), two ERISCs | 94.4% | 98.1% |
| main speedy VC0 (worker-only routers) | 93.4% | 98.7% |
| v4rxg3 2D, two ERISCs | 96.4% | 98.6% |
| v4rxg3 first-level-ack config, two ERISCs | 97.1% | 99.2% |

The rest is loop setup and the context-switch scratch registers, which fall back to loop overhead.

### 3.3 On the sender pass

On v4txo3b every rule resolves. Its renamed functions are pre-mapped: visit_channel_with_work, visit_vc,
visit_channels, return_slots, and RemoteReceiverChannelProgress's has_room, take_slot and add_free_slots.

**Re-check on the final sender build:**
1. `bd_map.py resolve new --sources variant:fabric_exp/variants/<final>,git:HEAD`.
2. After its samples exist, `bd_map.py check <set>`: any sample that lands on a fallback rule is listed by line. Add a
   rule for each new function, then rebuild.

### 3.4 Debatable lines

Each of these is mapped one way and noted in its rule:

1. Slot release (retire, ring advance, write counter, packets-received decrement) is under credits. It could be loop
   bookkeeping.
2. The header's source channel, kept for the completion credit (main `set_src_chan_id`, the new router's source
   record), is under credits. It is a header read.
3. The new receiver's `if (route.forwards() && !admit(...))` line is under decode. On rows without forwarding it
   carries the route word's first-use stall, ~10 cycles on E1 at point 1. The admission result's branch shares that
   line; the admission work itself is under checks.
4. Main's admission-result lines (`can_send_to_all_local_chip_receivers`) are under checks. They are the first use of
   the downstream credit reads.
5. The new sender's ack and completion loads inside the send are under credits (§1).
6. The command switch in local delivery is under decode in both routers. Main's `switch (noc_send_type)` costs ~15
   cycles on pcse_v1.
7. The forwarding channel's record load and store around a visit is under loop. On forwarding routers it runs once per
   packet: 8.2 on prlg3ml E0.
8. The worker connection semaphores, read in E0's register group, are under loop. Their latency shows at the group's
   first test (checks).
9. Main's TX-queue busy test before a send is under eth_send. It runs only when a packet and room are present.
10. Main's per-pass completion and ack counter reads are under credits. They run every pass, and their idle time lands
    there: 23.9 + 22.6 on pcse_v1's sender side.

## 4. What breakdown.json holds now (provisional)

The file has 68 records: 56 `cycles_breakdown` records (one per point, router, ERISC and category) and 12
`cycles_breakdown_missing` records. **Every filled record is a proxy, marked `provisional: true`.**

| point | new router: set, what it is | E0: router; checks / eth_send / credits / loop; spare | E1: router; checks / decode / noc / credits / loop; spare | bound (A/B) |
| --- | --- | --- | --- | --- |
| 1 2D 1-hop write 4352 B, 400G | b2d ew4352: v4rxv2i3, the right row and link, a pre-lead build; rate 126.9 | chip 1 ch 5; 48.8 / 51.7 / 16.7 / 9.7; ~5 spare | chip 0 ch 6; 12.4 / 24.2 / 32.8 / 51.2 / 6.3; ~27 spare | E0 |
| 2 2D 1-hop write 1024 B, 200G | b2d bew1024: the **P150** row at 1024 B; 116.4 (the Galaxy runs it at 122.9) | chip 1 ch 5; 40.9 / 52.7 / 15.0 / 7.9; not measured | chip 0 ch 6; 12.6 / 23.1 / 31.7 / 42.8 / 6.3; not measured | not measured |
| 3 2D mesh multicast write 4352 B, 400G | b2d mcast (PcsMcast = the census MeshMulticast write row, run longer): v4rxv2i3; 141.7 | chip 4 ch 4; 35.3 / 61.6 / 32.1 / 12.6; ~0 spare | chip 4 ch 5; 15.3 / 20.9 / 58.4 / 42.1 / 4.9; ~26 spare | E0 |
| 4 Galaxy N/S 7 hops | prlg3ml hop3: a **diagnostic** build (v4rxg3 with the N/S routers' forwarding shape) on the P150 3-hop E/W row; 179.7 | chip 0 ch 7; 76.1 / 62.3 / 22.9 / 18.4; not measured | chip 0 ch 6; 27.7 / 20.4 / 87.5 / 32.6 / 11.4; not measured | not measured for the proxy (the Galaxy row is E1-bound, V4_NS_FLOOR §1) |
| 5 Galaxy full-ring multicast 4352 B | none | | | |

- Decode and NoC are 0.0 on every E0 and the send is 0.0 on every E1. Unmapped is zero everywhere.
- **Main: no records at any point,** and its records carry the explicit "spare time not measured" label. The only
  main samples (pcse_v1) are one-ERISC ring bidirectional; they validate the map, not a point.
- **Point 5: no samples of either router.** The nearest new-router samples (ring 1-hop write) are another traffic
  pattern and are not used.
- Point 4 under the rule picks the busiest routers. V4_ROUTE_PRELOAD §8.2 had sampled chip 2 ch 4's E1 and chip 2
  ch 5's E0 (the §3.1 rows), which are within one point and are listed as alternatives.

### 4.1 Which router's ERISC a record shows (user, 2026-11-02)

Per ERISC, the record shows the router whose ERISC sets the test's rate, where an A/B has identified that router;
otherwise the busiest. "Busiest" means the largest share of samples in that ERISC's work category: Ethernet send for
E0, NoC commands for E1.

The existing one-side delay A/Bs (slack.json, V4_BOUNDS §1, V4_NS_FLOOR §1) spin every router's E0, or every router's
E1, at once. They name the bound side, not a router. So today every pick is "busiest", and the record says so.

An A/B that does name a router enters as `ab_targets` (with `ab_source`) in points.json or in a Galaxy manifest
record; it then overrides the busiest-router pick. Each record carries:
- `router_pick`: the rule that picked the router;
- `router_name`: e.g. "chip 4, Ethernet channel 4, E0". The chip is the tt-exalens id, which need not be the test's
  mesh device id;
- `work_share` and `work_category`;
- `router_alternatives`: the other routers of that ERISC within 2 points of the pick's work share. At point 3 there are
  three per ERISC; at point 4, one.

### 4.2 breakdown.json record schema

data/SCHEMA.md points here. Every record has all of SCHEMA.md's required fields:
- `metric` is "cycles_breakdown" and `value` is the cycles per packet;
- `router` is "main" or "new";
- `bound_side` / `bound_status` come from slack.json for the new router where the build and row match, else
  "unknown";
- `same_job_with_main` is false;
- `pair_key` is `bd-p<point>-<ERISC>-<category>`, shared by main's and the new router's record.

Further fields:

| field | meaning |
| --- | --- |
| point, point_label, link | the test point (1-5), its label, "400G" or "200G" |
| erisc | "E0" (sending ERISC) or "E1" (receiving ERISC) |
| category, category_label | a taxonomy id and its label (seven categories; "unmapped" only when nonzero) |
| cycles_per_packet | = value: the category's share of this ERISC's window samples x the sampled run's own rate |
| samples, category_samples, share | window samples of this ERISC (summed over passes), those in the category, the share (mean over passes) |
| se_cycles, noise | binomial standard error in cycles/packet; true under 3 cycles/packet |
| row_rate, rate_source | the sampled run's cycles/packet (mean over passes) and the CSV it came from |
| build, sample_set, target, job, date | the router build or variant, the sample set(s), "chip/chan/E<n>", job(s), date(s) |
| passes | per-pass values when there are two passes (a: main then new; b: new then main), else null |
| provisional | true for phase-1 proxies |
| point_system, point_packet_size_B | present when the sampled system or size differs from the point's |
| router_pick, router_name, work_share, work_category, router_alternatives | §4.1 |
| spare_cycles, spare_label, spare_source | §1: the A/B spare time and the marker text; null for main, labelled "not measured" |
| detail | the largest rules within the category: [{rule, cycles}] |
| coverage, ambiguity, map | shares mapped by specific rules, by fallbacks, and unmapped; the build-ambiguity share; the map file and its hash |

`cycles_breakdown_missing` records (one per ERISC for a point and router without samples) have `value` and `unit`
null, `category` null, the spare fields, and notes saying what is missing and which phase-2 row fills it.

## 5. Phase 2 sampling plan

### 5.1 P150: points 1 and 3, plus the costliest rows (one broker job)

**Script:** `consolidation/bd_p150_runs.sh NEW_VARIANT TAG`. Dry run: `DRY=1 bash ... NEW TAG` prints the steps; it
sources nothing and changes nothing; checked.

**Settings:** main as census `main_fw` (kernel v1, flags 0, firmware keepalive, slot variables unset). The new router
from its variant at the study settings. BW telemetry off (run_cap). Both ERISCs of every router sampled. One cache per
router, kept.

**Order:**
1. pass a: main (2D compile row, ew4352, mcast), then the new router (swap, compile, ew4352, mcast);
2. pass b: the new router (ew4352, mcast);
3. the costliest rows, new router only, 4352 B: 2D multicast scatter; half-ring multicast scatter and ring 1-hop
   scatter (ring compile row first); unidirectional linear multicast write and scatter (1D compile row first); Y-torus
   1-hop N/S scatter (Y-torus compile row first);
4. restore the tree, then pass b for main (ew4352, mcast).

**Source snapshots** (`results/bd_TAG_src_{main,new,main_after}.tgz`) are taken while each router is in the tree. The
job checks that main's are equal in both passes.

**Rows:** pcs3/ew4352.yaml, pcs3/torusns_scatter.yaml, pcs3/compile*.yaml, and the new consolidation/bd_rows/*.yaml.
All new yamls passed a mock-cluster vet (CPU only, `consolidation/bd_mock_vet.sh`): every test reached "Launching
programs".

**Time:** about 12-15 minutes of device time in one broker job:
- 5 compile rows at ~0.5-1 min each;
- 14 sampled rows at ~15-35 s each. Main takes ~1.7x the new router's time. Job 358 ran 19 such rows in ~6 minutes.

**Then ingest:**
1. `bd_phase2_sets.py p150 TAG NEW_VARIANT --job "a09u32 job N"`.
2. `build_breakdown.py --points breakdown/points_phase2.json --md`. Records become the mean of passes a and b, with
   `passes` holding both.
3. For the costliest rows: `build_breakdown.py --points breakdown/points_costliest.json --out
   data/breakdown_costliest.json --md`.

The ingest path was tested end to end with stand-in sets.

**Wait for the final sender build:** the sender pass changes E0. The script takes the variant name, so it is one
command on the final build.

### 5.2 Galaxy: points 2, 4 and 5

The Galaxy agent runs these: pc_sampler.py inside a Slurm job on 110-a nodes, chained after the VC2 line job, both
routers and both ERISCs.

- The new router there is v4rxg3, so its E0 split needs a rerun on the final router. Main's is final.
- `consolidation/bd_glx.sbatch` and `bd_mk_glx_trees.sh` are reference only, not submitted. They show the two-tree
  layout, the 110-a node guard, the kept caches and the source snapshots.

**What I found about pc_sampler.py on the Galaxy:**
- tt-exalens opens the Galaxy from a /data tree's python_env. hang_capture.py did it on bh-glx-110-a02u14 (job 168774)
  in ~13 s, after the test had ended, not beside a running one.
- trees/sls_lead's python_env has tt-exalens 0.4.1 and tt-umd 0.9.12. trees/vc2l_p3's does not, and mk_sls_tree.sh
  copies vc2l_p3's.
- No /data tree has pc_sampler.py yet.
- With ~32 chips the sampler covers many more targets than the P150's 96, so each target gets fewer samples per
  second. Rows need ~15-20 s of traffic for ~1,500+ window samples per target.

#### Galaxy samples: what to save

For each sampled run (point, router, pass), save the following. All paths stay readable from a09u32 (/data).

1. **pc_sampler output:** `<prefix>_samples.bin` and `<prefix>_samples.json`.
   - The .json holds the launch-message layout the mapping needs: per core the kernel config base, each ERISC's text
     offset, the load address (base + offset), the text size, and the first 256 image bytes. It is read at start
     (`layouts`) and after sampling (`layouts_end`).
   - Use pc_sampler.py unchanged, with both ERISCs (the default `--eriscs 0,1`) and all chips. `--chips` takes
     tt-exalens chip ids, which need not equal the mesh device ids.
   - Use `--start-delay-s` of 2 or more and enough traffic for ~10-20 s of sampling.
2. **The run's whole kernel cache (TT_METAL_CACHE), kept, never `rm -rf`'d after the step.** Per router config hash
   `kernels/fabric_erisc_router{,_v2}/<hash>/`, the mapping reads:
   - `{active_erisc,subordinate_active_erisc}/<proc>.elf.xip.elf`, to match the image read from L1;
   - `<proc>.elf`, with the JIT's -g debug info, for addr2line;
   - `named_ct_arg_map_generated.h`, for the config.

   One cache per router per job is fine.
3. **The log of the run that compiled the routers**, with `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`, for the config
   hashes that pick the build. Keep every sampled run's log too.
4. **The rate:** the run's `bandwidth_summary_results_blackhole.csv`, and the per-test CSV as frs_glx.sbatch keeps it.
   The mapping takes cycles/packet = 1.35e9 / avg_packets_per_s. Say so if the device clock was not 1350 MHz.
5. **The router sources exactly as compiled**, one snapshot per tree:
   `tar czf src_<router>.tgz -C <tree> tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric tt_metal/hw/inc tt_metal/hostdevcommon`.
   Archive members must start with `tt_metal/`.
   - **Main must run from a tree with no variant files in it.** Variants replace headers that main's router includes
     (packet transmission, datamover channels, flow control, noc_nonblocking_api.h, dataflow_api*.h, the packet
     header), so a tree with v4rxg3 swapped in does not run main as shipped.
   - Main as shipped: `TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0`, no keepalive override (firmware keepalive),
     no slot variables.
   - The new router: the tree with v4rxg3, kernel v2 slot, `TT_FABRIC_EXP_FLAGS=13129088`, keepalive 1000.
   - Both: `TT_FABRIC_NO_BW_TELEMETRY=1`, default slots, -O3 as the census.
6. **Two passes in reversed router order** if the allocation allows (a: main then new; b: new then main). Otherwise
   one pass, said so.
7. **A manifest** (JSON list, one record per sampled run) for `bd_phase2_sets.py galaxy MANIFEST.json`, with fields:
   - `point` (2, 4 or 5) and `router` ("main" or "new");
   - `pass` ("a" or "b");
   - `samples_prefix`: the absolute path up to `_samples`, e.g. `/data/.../results/a_new_ew1024_samples`;
   - `rate_csv` (absolute);
   - `cache`: the absolute TT_METAL_CACHE directory;
   - `logs`: absolute log paths;
   - `sources`: `["tgz:/data/.../src_new.tgz"]`;
   - `build`: e.g. "v4rxg3", or "main (kernel v1, flags 0, firmware keepalive)";
   - `job` ("Slurm NNNNNN"), `node` and `date`;
   - `test`: the row name, and `packet_size_B`;
   - the routers are picked by the rule in §4.1 (the busiest). Only if an A/B names a specific router, add
     `ab_targets` (`{"E1": "chip/chan/E1"}`) and `ab_source`; nothing else overrides the pick.
8. **Rows:** consolidation/bd_rows/glx_ew1024.yaml (point 2), glx_ns7_1024.yaml (point 4, chip 0 to 28 down column
   0), glx_fullring4352.yaml (point 5), and the compile rows glx_compile_mesh.yaml and glx_compile_ring.yaml.
   - All are mock-vetted on the Galaxy descriptor (§5.1).
   - Equivalent rows of the agent's own are fine; record their names.
   - Point 4 runs at the default slots (4,352 B) with 1,024 B packets (user, 2026-11-02).

## 6. Decisions and open questions

Decided (user, 2026-11-02):
1. **"waiting" is dropped.** Its lines are folded into checks and credits; the A/B spare time is a separate marker,
   and main is labelled "not measured" (§1).
2. **Router choice:** the A/B-identified router, else the busiest, recorded per record (§4.1).
3. **Point 4:** default slots (4,352 B) at 1,024 B packets, as bd_rows/glx_ns7_1024.yaml has it. The new router runs
   that row at ~186 cycles/packet on v4rxv2i3 (2 KiB slots would give ~172-173); main runs it at 222.9.
4. **Point 3:** exp_2d_4352.yaml's MeshMulticast write at 4,352 B, the source of the page's multicast peak (72.4% of
   line rate at 400G). The 69-83% range spans the multicast tests of all topologies. bd_rows/mcast.yaml is that row
   (all-to-all, one link) with more packets.
5. **Bundle:** SCHEMA.md lists breakdown.json and points to §4.2 (the coordinator's side). finalize.py's field check
   does not cover it.

Still open:
- **The debatable lines in §3.4.** Items 1-3 move up to ~10-20 cycles/packet between credits, loop, decode and checks.

## 7. Obstacles met

- **pcse_v1's cache is deleted.** Main's samples are mapped through identical images in another cache (§2, §3.1).
- **The sender-probe caches are deleted** (txp*, job 449), so those samples cannot be mapped. The route-preload
  caches of job 459 (bndp-prlg3ml, -prlrp3ml) were kept and are used.
- **A broker swap was active while I worked.** All sources were read from git HEAD and the variant directories.
- **slotstudy/sls_mock_vet.sh deletes `generated/` in the shared /data tree** at its end. I wrote bd_mock_vet.sh,
  which runs from a symlink farm in the scratchpad instead.
- **No Galaxy samples of either router exist.**

## 8. Files (all new)

| file | what |
| --- | --- |
| report/breakdown/taxonomy.json | the seven categories with definitions and notes, and the dropped "waiting" category with what it folded into |
| report/breakdown/map_main.json, map_new.json | the line maps |
| report/breakdown/sets.json, points.json | phase-1 sample sets and the point-to-set map |
| report/scripts/bd_map.py | the mapping engine and checks: `check SET`, `lines SET`, `resolve MAP --sources`, `static CACHE --map --where` |
| report/scripts/build_breakdown.py | data/breakdown.json (`--md` prints the table) |
| report/scripts/bd_phase2_sets.py | phase-2 runs into sets_phase2.json, points_phase2.json, points_costliest.json |
| report/data/breakdown.json | 68 records, provisional (schema §4.2) |
| consolidation/bd_p150_runs.sh | the P150 broker job (DRY=1 for a dry run) |
| consolidation/bd_rows/*.yaml | the new long rows: P150 and Galaxy (mock-vetted) |
| consolidation/bd_mock_vet.sh | the mock-cluster vet, writing nothing into shared trees |
| consolidation/bd_glx.sbatch, bd_mk_glx_trees.sh | Galaxy reference only (not submitted) |
