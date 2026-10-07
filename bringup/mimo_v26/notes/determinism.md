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
