# Decode determinism: what is known, how to reproduce (decode agent, for the MoE agent)

Symptom: identical full-model runs are not token-identical (LOG E261007-0818-033, scorecard determinism gate = FAIL entry E261007-0836-176).
Transient whole-ROW glitches (all live users of one mesh row for 2-4 steps; rows 1-3, never row 0 so far; different rows each run), traced and eager.

Ruled out (evidence in mimo-logs/attn_*):
* prefill: first-token logits and KV caches of live slots bit-identical run to run (attn_M8_kvdet).
* ttnn.all_reduce over mesh columns: 96 chained in one trace x 300 replays x {default, ring nl1, linear nl1, ring nl2}: 0 mismatches (attn_N1_ar, tests/test_allreduce_stress.py).
* attention decode (SDPA/paged update) and lm_head: not involved, the first diverging tensor is always a MoE layer `ffn_out`.
* async host->device input copy lifetime (fixed anyway: host tensors kept alive + synchronize, Decoder.set_inputs).
* dispatch/combine num_links=1 (workers 2): still glitches (attn_M15_links1_workers2: L3 row 3, 3 of 100 replays). links=2 workers=1: 0 glitches in 100 replays (attn_M15_links2_workers1: first = None; the default 2/2 showed 2-5 glitching replays per 100, so p~5% that this is luck): MIMO_DISP_WORKERS=1 is the lead, needs a 300+ replay confirmation and a full-model token-identity run. links=1/workers=1: result in attn_M15_links1_workers1.log.

Localization recipe (reproduces in ~8 min on one host, 48-layer model, weights cache warm):
  launch.py ... --env ATTN_STACK_S=2400 --env DEC_REPS=100 --env DEC_TAPS=ffn_out -- python models/demos/blackhole/mimo_v26/tests/test_decode_layer_determinism_traced.py
  (needs MIMO_CKPT, LD_LIBRARY_PATH=/mnt/tt-data/ssinghal/mimo/artifacts/integ_d72e22e2fbf/lib, MIMO_EXPERT_OP=unified; see /tmp-style wrapper in LOG). It prefills 32 GSM8K prompts, runs 5 decode steps,
  captures ONE traced 48-layer decode step with per-layer taps, replays it 100x on identical inputs/KV state and compares live slots of every tap with replay 0.
  Result (attn_M14_ldet_default): 28 layer-taps differ; first = L20 ffn_out rows 1,2 (maxdiff 2.57), later L24 rows 1,2,3 (maxdiff 14.8): 2-5 of 100 replays, ~1e-3 per MoE-layer call.
  Knobs: MIMO_DISP_LINKS / MIMO_DISP_WORKERS (Shared num_links/workers, default 2/2) are env overrides added in tt/moe_prefill.py (decode agent edit, 1 line).
Single-layer replay (tests/test_moe_decode_stress.py, layer 5, 300 replays, fixed input, 8 live/row): stages 99 (full) and 5 (before all-reduce): 0 mismatches, i.e. ONE layer call per replay is
underpowered at ~1e-3 per call: chain many MoE calls per replay (same module, new inputs) or use the 48-layer taps test above.
Suspects (not tested): cross-row semaphores/offset_cumsum (rows 1-3 only), combine init_zeros=False stale rows when a (token,k) is not written, dead-slot spread routing duplicates (a token with the same expert twice?), post_combine_reduce reading comb before combine finished.

# Decode MoE race: hand-over from the moe agent (E261007-0818-033 is the decode agent's localisation)

## What I reproduced (MoE only, no attention, no model)
Test: `models/demos/blackhole/mimo_v26/tests/test_moe_race_stress.py` (commit 5c4b1ee7013+; 4x8, real layers 1-6 chained in ONE trace of `MiMoMoEDecode`, real dump hidden states distinct per mesh row,
8 live slots per row, 3000 replays on identical inputs, every layer output compared bitwise with the first replay, per mesh row). Launch (private lib needed, see below):
`launch.py --worktree <wt> --commit <c> --host <free> --log <log> --env LD_LIBRARY_PATH=/mnt/tt-data/ssinghal/mimo/artifacts/moe_B_9a8fd8582e5/lib --env MIMO_REPS=3000 --env MIMO_ROUTER=exact -- python_env/bin/python -m pytest -x -s -q models/demos/blackhole/mimo_v26/tests/test_moe_race_stress.py`
(runs 90 s after the ~6 min weight build; env knobs MIMO_NL, MIMO_LIVE, MIMO_ROUTER=exact|fast, MIMO_STAGE=99|5|4..., MOE_DISPATCH_SPARSE_MCAST=0, MIMO_DISP_LINKS/WORKERS).

Results (18000 layer-calls per run, logs /mnt/tt-data/ssinghal/mimo-logs/moe_I_*_5c4b1ee7013.log):
| variant | glitching layer-calls | rows | max abs diff |
|---|---|---|---|
| router exact, default (dispatch sparse_mcast ON) | 5 | 1, 2 | 0.0023 .. 0.0242 |
| router exact, MOE_DISPATCH_SPARSE_MCAST=0 (per-expert dispatch path) | 4 | 1, 2, 3 | 0.0023 .. 0.0242 |
| router fast (router_select kernel), default | 2 | 3 and ROW 0 | 0.0020, 0.0242 |
So it reproduces with the MoE alone (~2-5e-4 per layer-call here; the decode agent saw ~1e-3), it is NOT the sparse-multicast dispatch feature, NOT the exact router (fast router also glitches), and row 0 is not immune (one row-0 glitch with the fast router).
The max-diff VALUES repeat exactly across runs/layers/variants (0.0022955, 0.0219879, 0.0241923, 0.0242269, 0.0156631): looks like one deterministic quantity per event, e.g. one slot's expert contribution
missing/duplicated or one expert's rows read stale, not random bit noise. Diffs are small here (hidden of the dump, 8 live slots; the model saw up to 14.8 on massive-activation tokens).

## Stage timings (traced, per layer, host-timed) to know what is in the window
router+pad+spread 0.29 (fast router 0.03-0.1), counts (masked_bincount+offset_cumsum, CCL on cluster axis 0) 0.09, dispatch 0.03, experts (unified op) 0.78 (HiFi3: ~1.0), combine 0.10, post_combine_reduce 0.19, all_reduce 0.05 ms.

## Not yet done / ideas
* Per-stage bitwise stress (MiMoMoEDecode.forward(upto=0..5), outputs kept alive): mask the garbage rows (dispatch/combine buffers are not initialised; compare only valid rows from counts/region offsets) to find the first stage that differs between replays.
* Experts-stage race candidates: unified op reads counts/region offsets on device while offset_cumsum (CCL) may not be complete for rows 1-3? (all consumers are ordered by the program queue, but fabric writes into other chips' buffers are asynchronous: a chip can start dispatch/combine reads before
  a neighbour's remote writes land; look at the semaphores in combine and dispatch end-of-op barriers).
* post_combine_reduce accumulates with packer L1 accumulate in bf16 and `read_tile_value` mailbox reads of the indices: a stale indices page would drop one expert (matches the repeating diff magnitudes).
* The decode agent's finding that attention outputs are identical until the MoE layer and the all_reduce stress is clean stand.
* Libraries: private relink builds with tools/priv_relink.py; moe_B_9a8fd8582e5 has the precise numerics; the dispatch op now has MOE_DISPATCH_SPARSE_MCAST (this branch).

## Sentinel dead-slot HANG (E261007-1202-527) - moe agent findings
* Reproduced MoE-only (tests/test_moe_deep_trace.py: 48 chained MiMoMoEDecode calls in ONE trace, U=1): dead="sentinel" hangs the traced replay (first replay or within a few hundred); dead="spread" 500/500 replays fine; same for dead slots routed to ONE fixed expert (FIXED_DEAD=255, counts non-zero but all traffic on one chip).
  Not fixed by: MOE_DISPATCH_SPARSE_MCAST=0, all_reduce (rs) vs all_gather+sum, expert op mm vs unified, holding every freed MoE buffer alive during capture (hold_begin/hold_end), row-axis barrier all_gathers after counts / dispatch / combine (MIMO_SYNC). 24 layers at U=32 does not hang; 48 layers U=1 does. The decode agent's full-model repro (24 layers) did NOT hang on my runs (mm and unified expert op, sentinel/spread, rs/ag_sum): hang probability depends on depth and imbalance.
* Triage of the 48-layer U=1 sentinel hang (/mnt/tt-data/ssinghal/mimo-logs/moe_K_triage47.out): 2 devices (7, 14) never finish `combine`: reader_combine.cpp:254 sits in the noc_async_write_barrier after the multicast of the per-expert token counters to the untilizer cores
  (untilizer cores wait at reader_untilize.cpp:129 `counter_ready_sem.wait(1)`), the other 6+ devices wait in writer_combine.cpp:381 for the exit-handshake semaphore (combine_devices-1 increments). So it is a combine-internal NoC multicast / handshake hang that needs imbalanced (very uneven or zero) traffic across chips in a deep trace.
* Workaround that works: dead="spread32" (default now): dead/pad slots round-robin over ONE expert per chip (32 experts = 1/8 of the weights, weight 0) so every chip still gets tokens every layer. 48 layers: 1000/1000 replays OK at U=1 and U=32, 500/500 at U=8; live outputs bit-identical to spread at U=1..32.
## Race track result (race agent): the glitch is in the column ttnn.all_reduce, workaround = all_gather + local sum
* tests/test_moe_race_stages.py (MoE-only, real layers 1-6 in one trace, per-stage clones compared on device against replay 0): router/spread/bincount/offsets/counts/region offsets/post_combine_reduce
  outputs bit-identical in all calls; only the all_reduce output (cluster_axis=1, reduce_scatter + all_gather) shows ONE wrong 32-column tile (256..1024 elements) on ONE device, zeros or stale values.
  Rates per layer-call: host 40 sentinel 22/18000, host 47 spread 10/18000 and 8/36000 (no extra clones), host 48 spread 23/72000; reduce_scatter + all_gather as two calls 87/36000, num_links=1 132/36000.
  Every host shows it (not a single bad host); devices vary (cols 0-3 mostly, one col 7), but the glitching LAYER INSTANCES repeat (L2, L4, L5 in the 6-layer chain): per-op buffer placement matters.
* all_gather(dim=0, cluster_axis=1) + ttnn.sum(dim=0): 0/36000 (sentinel, 6000 replays, host 40) and 0/36000 (spread, host 48). Implemented as MiMoMoEDecode._column_reduce (MIMO_MOE_AR=rs restores the old path).
* The stand-alone all_reduce (notes/allreduce_race_repro.py: [1,1,64,4096] bf16, 24 chained in a trace, 3000 replays = 72000 calls, host 47) shows NO glitch: context matters (it needs the MoE ops around it).
* Full 48-layer traced decode (tests/test_decode_layer_determinism_traced.py, now compare-in-trace via one packed tap tensor): the OLD all_reduce path gave 0 glitches in 300 replays (new lib, host 48) and 150 replays (old lib, host 48),
  so the M14 rate (2-5 per 100 on host 30) is not reproduced by this merged tree on 40/48; the MoE-only stress is the sensitive test.
* Test-infrastructure traps found: (1) any device allocation after ttnn.end_trace_capture while the trace is alive (even temporaries) made the next replay hang; (2) a job killed while hung leaves the device dirty (next job hangs
  or fails FW init) until tt-smi -glx_reset; (3) MIMO_DEAD=sentinel hangs decode at >=12-24 layers (decode agent E261007-1202-527).
