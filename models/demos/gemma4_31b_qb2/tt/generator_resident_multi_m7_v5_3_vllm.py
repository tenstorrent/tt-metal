# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: multi-user DFlash2 speculation on `batch` (M7). Pre-registration: bench/resident/m7/
M7_BUILD_PREREG_v3.md (C1-C9', §2.3-2.6, §5); LEAD_DECISIONS_M7 M1-M17. The S1 batch class v3
(generator_resident_multi_s3_2_pb_ring_apc_v3_vllm, APC v3: spec_mode passes through, request-keyed layouts, C8)
plus, from S0's protocol layer (propose / verify / SlotState / the S1a async contract) with M7's parts in place of S0's
fakes:

- Bands by live users (prereg v3 §1; v2: the ship rule's c3-4 and c5-8, both on (32, 4)): c1 -> (R 8, M 8),
  c2 -> (8, 4), c3-4 and c5-8 -> (32, 4), c9-16 -> (32, 2): the verify
  program and the D v8 draft block M. Above 16 live users M7 does not speculate (option S, as update 5).
- The T switch (tt/m7_switch.py), one per band (v2; v1 had one per draft block): SPEC drafts for every ring holder
  on every step; PLAIN runs
  option S's program with the taps masked (m5_13 set_taps_enabled, flipped only after a sync), no ingest, async as
  shipped. The gain's s is seeded from bands_u5.json's plain step times (est.) and measured on PLAIN steps; `it` is the
  wall time between two consecutive verify starts with nothing else between them. GEMMA4_M7_SWITCH_PIN (gates only).
- Rings (tt/m7_rings.py): 16 drafter rings to users admitted while <= 16 are live (C2); every assignment zeroes the
  ring first (C9'); a release hands freed rings to ring-less users, cold, after S0 and S1 have released (C6).
- Ingest: every SPEC step's committed rows (verify, ordinary and runner-forced alike) go from the step's taps buffer
  into the user's ring before the next D launch (one ingest launch per 32 rows, set_rows syncs); a step that ran with
  the taps masked ingests nothing. Upkeep still queued when a launch would overwrite its taps buffer runs first (S1a's
  rule, taps_live). PLAIN skips tap upkeep (APC v3 review N1).
- Forced ordinary steps (§5.4): an ordinary decode while SPEC that drops outstanding offers, or is host-sampled. It runs
  option S's program with taps on and its rows are ingested (synced: the step is host-sampled); counted per request in
  forced_ordinary_steps, never a T-switch decision.
- The verify (S0's _verify with the band's program): the T3 device draws are S0's _configure_verify_sampling (row j of
  user u: u's parameters, seed at p + j), configured over all 32 sampler rows (rows past R inert) so R = 8 verifies use
  the served sampler shape; greedy rows keep the greedy mask (S1). A verify with no drafts (an unresolved commit) runs
  S0's draftless (32, 1) launch.
- The async contract (§2.5): S1a's queued steps and stale proposals are kept: a stale proposal that would enter offers
  S1a's one-token placeholder and enters at the next, fresh one. A proposal in hold is never stale (a verify drains).
- The seed (prereg v3 §4): held until H5 (M14). GEMMA4_M7_SEED must be "off"; rings start zeroed and heal in SPEC
  (option (c)). Ingest v5 needs at least one staging slot, so one is built (P = 1, ~28 MB per chip) and never written.
- D flags (sticky att_status words, read after every D launch): any new flag stops speculation for the process (every
  switch pinned PLAIN) with one ERROR line.
- The release line (§2.3), once per request at INFO: "M7 request done req=<plugin request id>: {...}" with exactly
  LINE_KEYS. accept_digest = sha256 of the request's accepted-draft counts of each verify that carried its drafts
  ("a,b,c,"); R, M = the program of its last drafted verify, else of its last ordinary launch; mode_switches = the
  SPEC / PLAIN and band changes while it was live; ring = seeded / cold / none. Plus one INFO line per 1,000 synced
  launches by kind (verify, ingest, D, switch).

v2 (bench/resident/m7/M7_CLASS_V1_REVIEW.md 3976019d: K1, K3, K5; M7_BUILD_PREREG_v3_ADDENDUM_1 D2, D7; v1
generator_resident_multi_m7_vllm.py dcbce9be frozen):
- K1: one frozen serving config, by hash (load_serving_config): tt/m7_serving_config.json, whose sha256 is
  M7_SERVING_CONFIG_SHA256, naming H5's whole report (tt/m7_h5_report.json) by its sha256. The switch parameters
  per draft block (J, dwell, cap; cap "<256" = re-entry off), the seed decision, c1's build and each band's ship
  decision (spec / plain, G-T) come from it, cross-checked against the report's `freezes`; boot fails closed on a
  missing or mismatched file, except under the gate-only GEMMA4_M7_SWITCH_PIN, which runs the provisional parameters
  and logs "unfrozen". One switch per band (c2, c3-4 and c5-8 share block 4's parameters but switch and ship
  separately); a "spec" band runs behind its T switch, a "plain" band never enters SPEC (taps masked), and a gate's
  pin cannot make it. c1 not built (or plain): no (8, 8) engine program and no block-8 D program, and c1 serves
  PLAIN. A verify program or D block that no spec band uses is not built; the boot log lists the blocks built. The
  seed compares on verdict and P only; "measure" needs S-C's resolution (with its evidence hash).
- K3: a step's committed rows are ingested whenever its launch ran with the taps on, whatever the mode is when its
  proposal runs: the verify on which the switch leaves SPEC no longer leaves a hole (PLAIN launches are masked, so
  PLAIN still ingests nothing, N1).
- K5: the release line's key check raises.
- D7: the drafter is tt/m7_device_v3.py (v2's set_launch + read_select; v3: the D capacity from prog.att); draft()
  returns the drafts and the launched program's new flags together.

v3 (bench/resident/m7/M7_CLASS_V2_REVIEW.md 2205cb25: F1, F2, N3; LEAD_DECISIONS_M7 M40; prereg v3 addenda 4-5, the
H6 split with m5b-planner; v2 generator_resident_multi_m7_v2_vllm.py 05dd3748 frozen):
- F1: a verify carries drafts proposed at the previous step's live count, so it no longer takes the program of the
  band the live count is in now (which may be unbuilt, or none above 16). Deterministic rule (M40):
  (1) the drafting band's program, when it is built and still fits (live x M <= R, drafts <= M - 1);
  (2) otherwise the smallest-R built program that fits, never (8, 8);
  (3) otherwise the drafts are dropped for that launch: S0's draftless (32, 1), each row's anchor only, its drafts
  answered with placeholders (the walk accepts none; lossless). Counted in m7_drafts_dropped_on_crossing.
- F2: the gate pin runs the provisional parameters only when no hash is expected; a set hash with a missing file
  raises.
- H6 interim configs (gate-only; prereg v3 addenda 6-7): GEMMA4_M7_SERVING_CONFIG=<path>:<sha256> loads that config
  (its h5_report beside it) with every K1 check, the env's sha in place of the class constant, and logs
  "M7 serving config: H6 interim <sha12>" (plus "; switch pin <pin> (gates only)" under a pin, which moves only the
  switch mode of the spec bands, Q4). Accepted only while M7_SERVING_CONFIG_SHA256 is None (Q1). H7's env proof
  refuses it in the served env, as it does the pins.
- The seed's provenance (prereg v3 addenda 4-5): seed verdict "b" requires seed.d6_judge_sha256 (the D6 judge the
  release rests on; its PASS is H7's and the H6 preflight's to check). Verdict "c": absent or null, ignored.
- N3, g_t: the config's g_t {report, sha256} is read and logged; the class-constant (release) config must carry it
  with a 64-hex sha256 (H7 checks bands against that report, gt/gt_ship.py check_config_bands); interim configs may
  carry null (before G-T).
- G-L0 / G-L1's reference (gate-only): GEMMA4_M7_DUMMY_DRAFT=<token id> (pre-registered 0, <pad>; 6, <unused0>,
  for a re-run; "0" is token 0, not off) with GEMMA4_M7_SWITCH_PIN=spec: every SPEC proposal offers that token in every
  draft column, the tap mask and verify program stay exactly SPEC's, and the drafter and ingest are never launched.
  Refused without the spec pin.
- Path-loaded files for staging and the closure check: tt/m7_serving_config.json, tt/m7_h5_report.json (beside the
  class), and the drafter cache (GEMMA4_M7_DRAFTER_CACHE, a runtime mount).
- M62 (H4b v2 FAIL: 4 alive post-capture buffers, the sampler's all_gather global semaphores): the sampler's first
  eager call runs once before engine.warmup() captures any trace (_prewarm_sampler), so those semaphores exist before
  every capture; S1's _capture_sampling then hits the program cache.
- M62 note 2 (critic, H4B_V2_RESULT_NOTES 362587fb): every L1 guard also holds the floor against M7Device's D v8 and
  ingest v5 CB tops (their cb_tops(), engine.cb_tops' rule), which M1's guard (engine.cb_top() only) never read. Those
  programs launch eagerly and cannot be rebuilt in place, so a floor below their top fails closed.
- The plugin proof line at boot (s1_probe.log_plugin_proof: the vllm_tt_plugin root, its file hashes, PYTHONPATH;
  LEAD_DECISIONS_M7 M52 (3)), as H6's A-arm and R4 classes log it, so each arm's plugin is in its boot log.
- (review N2, kept as v2 has it) a pass through no band (> 16 live) and back is a band change: the band's switch
  restarts its n_r and PLAIN count (§5.1), so G-X sees one reset per excursion.

v4 (LEAD_DECISIONS_M7 M74, M75; m5-verifier's finding on v3 45f6d61b, frozen): F1 rule (3), the drop launch, answers
one column wide. v3 answered the anchor's argmax in column 0 and PLACEHOLDER in every other column at the runner's
1 + K width, but the runner walks with its own num_valid_drafts (model_runner.py:3523-3528, accept_greedy_drafts),
which the class cannot change: a row whose draft 0 equals that argmax was accepted, count 2, and committed
[t1, PLACEHOLDER] (spec_decode.py:576-590; _apply_committed_spec_tokens_to_state writes committed[row, :count],
model_runner.py:3799-3806). Column 0 is the real next token, so no answer at full width can reject a correct draft 0.
The runner verifies only the drafts the answer is wide enough to hold: verified_drafts = argmax_ids.shape[1] - 1
(model_runner.py:3512-3516, "the drafts are trimmed to the width the verify actually answered at"), and with none
accept_greedy_drafts commits column 0 for every row with count 1 (spec_decode.py:555-559, the narrow step's branch).
So the runner sees no valid draft on that launch (option (a) of M75). The drop is launch-wide (one program per launch),
so every row of a drop launch is answered at width 1; the drafter call is padded back to 1 + K by the runner
(_propose_model_drafts, model_runner.py:1835-1850), as after a narrow step. Nothing reads the width elsewhere: no
argmax_ids width check on either path (async_decode.py:66-67 returns argmax_ids as is; walk_spec_acceptance is the
one walk, model_runner.py:3451 and async_decode.py:1096). The drop triggers (the runner trims drafts to
SpecPlan.draft_cap(live), model_runner.py:1537-1567, 7 / 3 / 1): live > 16 with offers outstanding (16 -> 17), and a
config whose intermediate band is plain (c2 spec, c3-4 and c5-8 plain: 2 -> 3 carries 3 drafts, no built program
fits). _prewarm_sampler is byte-identical to v3's (H4b v3 pins it). File:line references: vllm_tt_plugin at the re-pin,
86bb6a47.
- M67: every L1 guard logs the floor against max(engine.cb_top(), each D v8 block's, ingest v5's CB top); the engine
  half rebuilds (M1), the drafter half fails closed (v3), so a passed guard leaves the floor >= that max.

v5 (LEAD_DECISIONS_M7 M88: the H6 tracked session's boot, run_1121_2150, failed before any stage; v4 65818270 frozen):
classes v1-v4 never booted served. vLLM warms prefill before decode, and S3-PB's warmup_model_prefill
(generator_resident_multi_s3_2_pb_vllm.py:84-88) refuses any drafter: "S3-PB serves the plain launch only: a drafter
needs PB3's seed pre-warm". The host harness never called warmup_model_prefill (s1p_harness.make_apc: allocate, then
warmup_model_decode), so no test met it. PB3's seed pre-warm (generator_resident_spec_vllm_pb3.py:1-20) builds every
program class of the single-user DFlash drafter's seed path (DFlashDrafter.seed through _enter, the first tap commit)
before any trace capture, so none is built while the sampling or draft trace is live. M7 has no seed path while the seed
is held (M14; __init__ refuses GEMMA4_M7_SEED != off): its states carry no TapBook (_new_state), so S0's prefill captures
no taps (generator_resident_multi_vllm.py:427-430), _enter never seeds, and the drafter view's reset is a no-op. What M7's
drafter does need is the same rule for its own device programs: they were first built at the first SPEC step, after
engine.warmup and the sampling capture. So:
- warmup_model_prefill runs the chain's warmups (M1's embedding and bucket warmups, S1's, S3-PB's terminal and sampling
  merge) with the drafter set aside, which is what the refusal guards and all that M7's drafter has on that path (no
  book: no taps, no seed; reset a no-op), then restores it;
- then, before warmup_model_decode captures anything, _prewarm_drafter builds every M7Device program once on ring slot
  0: the ring zero (d_ring_zero), each D v8 block's launch with its reads (the top-16, final_h's ttnn.slice, att_status)
  and the selector, one ingest v5 launch (taps launch -1 is live at engine launch count 0: taps_live), then a sync, each D
  program's sticky status row cleared (d_model_v8 clear_status: a warm launch's bits would otherwise mask the same flag
  in serving, M7Device.flags diffs against the last read) and the ring zeroed again (every ring is zeroed on assignment
  anyway, C9'). Counted in m7_drafter_prewarms.
_prewarm_sampler is byte-identical to v3's and v4's.

v5.1 (LEAD_DECISIONS_M7 M95: m5-verifier's v5 interim, GO-with-fixes; v5 23a45dec frozen): three changes to
_prewarm_drafter, nothing else (_prewarm_sampler and _verify byte-identical to v5's):
- under GEMMA4_M7_DUMMY_DRAFT (G-L0's reference) the D and ingest pre-warm launches are skipped, as the pin's contract
  says ("drafter and ingest never launched"); both ring zeros stay (a ring is zeroed on every assignment, so zero_slot
  serves under the pin too);
- _drafter_l1_guard("drafter pre-warm") runs first: the floor is held against the D / ingest CB tops before their first
  launch, failing closed (M62 note 2's rule, at the earliest point it applies);
- a warm launch's D flags are no longer discarded: each block's flags from its warm read are logged at WARNING and counted
  in m7_drafter_prewarm_flags (a cold first launch can set the sticky schedule-miss bit, m5-kernels); the status rows
  are still cleared after, so serving reads its own flags.

v5.2 (LEAD_DECISIONS_M7 M131, M133-M135; critic M131_OPTION_I_FLAGS f88eb0f3 items 3-4; v5.1 47a036ac frozen): on
resident_m5_14 (DEFAULT_PACKAGE), two changes.
- The no-straddle verify cap (option (i)). m5_14 reduces a K2a verify block's tiles in their own chunks, except the
  rows below 16,384 of a block whose rows reach 16,384 (the full layers' flat merge -> group tree switch); its
  engine.verify_cap(R, p, block=M) gives the rows a verify block of (R, M) at p may use: M, or fewer so that its live
  rows end before every listed boundary (M on an M1L program). The class never forms a block past that:
  - at propose, from the anchor p and the drafting band's (R, M): a user drafts min(M - 1, L - 1) (counted in
    m7_verify_cap_truncations when fewer than M - 1); at L = 1 (the anchor alone) it declines the step
    (m7_verify_cap_declines) and enters again on a later fresh step. It never holds with no drafts: under S1a an
    ordinary step then runs async and the next proposal in hold would be stale, which _propose refuses;
  - at verify, again with the program the launch actually takes (an F1 crossing can change (R, M)): a block past
    verify_cap drops the launch's drafts by F1 rule (3), the width-1 answer (m7_verify_cap_drops,
    m7_drafts_dropped_on_cap; lossless; on m5_14 with every spec band a K2a program it cannot fire, kept as the check).
  Drafts are never trimmed at verify: an answer narrower than the runner's num_valid_drafts per row needs placeholder
  columns, which is v3's committed -1 (M74). The cap reaches the engine's schedule as the launch's rows: user u's live
  rows [u M, u M + n + 1) carry p .. p + n, every later row of its block is inert (position -1), and write_rows
  copies them to the device before each launch, so the unit cores end the block at its last live row at runtime,
  in every captured trace (nothing about the cap is baked into a trace).
  Where p is final (critic item 3): the cap reads the anchor position the runner hands the propose, after the step's
  accept walk. Sync: the propose runs after the step's commit. Async (S1a): a verify never overlaps (the plugin
  drains before it builds one), and its blocks are retained proposals of fresh proposes (a stale one offers S1a's
  placeholder, which never reaches a verify); _verify refuses a block that is not the retained offer at exactly its
  p. Traced replay: the positions above are runtime inputs of every replay.
- The c1 policy (M118, M134, M135): c1 never serves (8, 8) SPEC until a fixed (8, 8) exists. One pinned setting, the K1
  serving config's c1_policy (a missing key reads plain, fail closed; the release config carries the user's F2
  answer, an H6 interim config may set f2c):
  - "plain": c1 never speculates (its band's program is (8, 4) on paper, never launched by c1);
  - "f2c": c1 speculates on (8, 4) with D block 4 behind its own T switch (block 4's switch parameters) iff bands.c1
    is "spec" (v5.3, M163: c1_build is then addendum 12's build decision, not H5's block-8 freeze);
  - "b8": reserved for a fixed (8, 8) (F2a/b); this class refuses it.
  Under every value no (8, 8) engine program and no block-8 D program is built. (v5.3: under plain the loader requires
  bands.c1 plain.) The provisional parameters (a gate pin with no config) are plain.
- (M140 D2) log_plugin_proof comes from tt/plugin_proof.py (s1_probe's function, moved unchanged): no module of the
  class chain imports tt/s1_probe.py, which carries the S1 probe's fault injection and is never packaged. The R4
  class keeps its s1_probe import (CorruptSwitch; gate-only, never packaged). The packaged boot class (M41) adds
  ensure_resident_cache and contwarm on top of this one, which subclasses APC v3 directly (packaging, with
  pkg-update).

v5.3 (LEAD_DECISIONS_M7 M159, M163; m5-verifier's v5.2 verdict N1, M7_CLASS_V5_2_REVIEW.md; v5.2 7b50cb71 frozen, GO
from the critic 9c695200 and m5-verifier): the loader's c1 checks, nothing else.
- N1 (M159): under c1_policy plain (or no c1_policy key) a K1 config's bands.c1 must be "plain", whatever c1_build: a
  config that ships c1 SPEC must name f2c, one that serves c1 plain must say so, and a v5.1-style config (bands.c1
  spec, no key) is refused instead of serving c1 plain against its own text. This subsumes v5.2's "c1 is not built:
  bands.c1 must be plain" (plain and b8 only; b8 is refused before it), which is gone.
- M163: c1_build means "c1 is built under the active rule". Under f2c it is addendum 12's build decision (c1_f2c.json,
  written by m7_configs_v2), and the loader no longer requires it to equal H5's freezes.c1 (block 8's, which the f2c
  run trims: false there); a config with bands.c1 "spec" and c1_build false is refused. Under plain c1_build must
  still equal H5's freezes.c1, as in v5.2.
The provisional parameters (a gate pin with no config) never pass the loader's checks and stay plain. Nothing else
changed: the plugin proof (tt/plugin_proof.py, dafab207) and every serving path are v5.2's.
"""

import hashlib
import json
import os
import time
from collections import Counter, namedtuple

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_async_vllm import QueuedSlotState
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_apc_v3_vllm import (
    Gemma4ResidentMultiS32PBRingAPCV3ForCausalLM,
)
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import MAX_USERS, SlotStep
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm import (
    PLACEHOLDER_TOKEN_ID,
    Offer,
    ResidentSpecPolicy,
)
from models.demos.gemma4_31b_qb2.tt.m7_rings import M7SlotTable
from models.demos.gemma4_31b_qb2.tt.m7_switch import PLAIN, SPEC, TSwitch
from models.demos.gemma4_31b_qb2.tt.plugin_proof import log_plugin_proof

RINGS = 16  # drafter ring slots (C2)
Band = namedtuple("Band", "R M name")  # a band's verify program (R, M) and its ship-rule name
# (most live users, band): the ship rule's bands (prereg v3 §1; ADDENDUM_1 K1); c3-4 and c5-8 share the (32, 4) program
# and block 4's switch parameters but switch and ship separately. Above 16 live users: no band (always PLAIN)
BANDS = ((1, Band(8, 8, "c1")), (2, Band(8, 4, "c2")), (4, Band(32, 4, "c3-4")), (8, Band(32, 4, "c5-8")),
         (16, Band(32, 2, "c9-16")))
BAND_NAMES = tuple(band.name for _, band in BANDS)
# v5.2 (M134, M135): the c1 policy; c1's band under each value this class serves (b8, a fixed (8, 8), is refused)
C1_POLICIES = ("plain", "f2c", "b8")
C1_BAND = {"plain": Band(8, 4, "c1"), "f2c": Band(8, 4, "c1")}


def bands_for(c1_policy):
    """BANDS with c1's band of the policy: no (8, 8) under any value this class serves"""
    return tuple((most, C1_BAND[c1_policy] if band.name == "c1" else band) for most, band in BANDS)
K_BY_ROWS = ((1, 7), (8, 3), (16, 1), (32, 1))  # SpecPlan caps (no K = 0: above 16 every row declines)
# PLAIN step seconds per live count until measured (est.: bench/resident/m7/bands_u5.json 0c135c46, ISL 128)
S_SEED = {1: 0.01019, 2: 0.01234, 3: 0.01321, 4: 0.01321, **{c: 0.01433 for c in range(5, 9)},
          **{c: 0.02591 for c in range(9, 17)}}
# the gate-only parameters under GEMMA4_M7_SWITCH_PIN without a frozen config (prereg v3 §5.2-5.3: never shipped)
PROVISIONAL = {"switch": {M: dict(J=16, cap=4096) for M in (2, 4, 8)}, "seed": {"verdict": "c", "P": 0},
               "c1_build": True, "bands": {name: "spec" for name in BAND_NAMES}, "c1_policy": "plain"}
HERE = os.path.dirname(os.path.abspath(__file__))
SERVING_CONFIG = "m7_serving_config.json"
M7_SERVING_CONFIG_SHA256 = "518c6321f43ff96a2b92eff94b8b05e77dd8751b7646b09dad6619f1d8632022"  # set when H6 freezes the serving config (K1); None: only pinned gates run
LINE_KEYS = ("committed_tokens", "seeds", "seeds_dropped", "staging_full", "drafts", "verifies", "verified_drafts",
             "accepted_drafts", "accept_digest", "R", "M", "mode_switches", "forced_ordinary_steps", "ring")
SYNC_LINE_EVERY = 1000


def band_of(live, bands=BANDS):
    """the Band serving `live` users (band[0] = R, band[1] = M), or None (no speculation); v5.2: of `bands`"""
    return next((band for most, band in bands if live <= most), None) if live > 0 else None


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_serving_config(pin, *, expected=M7_SERVING_CONFIG_SHA256, directory=HERE, name=SERVING_CONFIG,
                        release=True):
    """K1: the frozen serving config, checked (module docstring, v2, v3, v5.2). Returns {"switch": {M: {J, cap}},
    "seed", "c1_build", "bands": {name: spec | plain}, "c1_policy", "g_t", "sha256", "h5_sha256", "frozen"}. Raises
    ValueError on anything but a complete, consistent pair of files, unless `pin` (GEMMA4_M7_SWITCH_PIN, gates only)
    is set and no hash is expected (v3, F2: a set hash with a missing file raises): then the provisional parameters,
    frozen False.
    release: the class-constant config (g_t required); False for an H6 interim config (g_t may be null)."""
    path = os.path.join(directory, name)
    if pin and expected is None:
        return {**PROVISIONAL, "g_t": None, "sha256": None, "h5_sha256": None, "frozen": False}
    if expected is None:
        raise ValueError("M7 has no frozen serving config (M7_SERVING_CONFIG_SHA256 is unset): it serves only pinned "
                         "gates (GEMMA4_M7_SWITCH_PIN), never a release")
    if not os.path.isfile(path) or _sha256(path) != expected:
        raise ValueError(f"M7 serving config {path}: missing, or its sha256 is not {expected}")
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("format") != "m7-serving-config/1":
        raise ValueError(f"M7 serving config format {cfg.get('format')!r}")
    report_path = os.path.join(directory, cfg["h5_report"]["file"])
    if not os.path.isfile(report_path) or _sha256(report_path) != cfg["h5_report"]["sha256"]:
        raise ValueError(f"H5 report {report_path}: missing, or its sha256 is not {cfg['h5_report']['sha256']}")
    with open(report_path) as f:
        report = json.load(f)
    freezes = report.get("freezes")
    if report.get("verdict") != "PASS" or report.get("ok") is not True or not freezes or report.get("ship_mode") != "t3":
        raise ValueError(f"H5 report: verdict {report.get('verdict')!r}, ok {report.get('ok')!r}, ship_mode "
                         f"{report.get('ship_mode')!r}, freezes {'present' if freezes else 'missing'}")
    switch = {}
    for M in (2, 4, 8):
        want = {key: freezes[key][str(M)] for key in ("J", "dwell", "cap")}
        got = cfg["switch"][str(M)]
        if {key: got[key] for key in ("J", "dwell", "cap")} != want or want["dwell"] != max(32, want["J"]):
            raise ValueError(f"block {M}: config switch {got} vs H5 freezes {want} (dwell must be max(32, J))")
        if not (want["cap"] == "<256" or isinstance(want["cap"], int)):
            raise ValueError(f"block {M}: cap {want['cap']!r}")
        switch[M] = dict(J=want["J"], cap=want["cap"])
    seed = cfg["seed"]  # compared on verdict and P only (freezes.seed also carries branch, ends, stall, f_s)
    if seed.get("verdict") not in ("b", "c"):
        raise ValueError(f"seed verdict {seed.get('verdict')!r}: only b or c is a release state")
    h5_seed = freezes["seed"]
    if h5_seed.get("verdict") == "measure":  # S-C decides it at H6
        if not release and seed.get("source") != "S-C":
            # an H6 interim config before S-C carries (c) (addendum 6 condition 2; S-C's arms run pinned)
            if (seed["verdict"], seed.get("P")) != ("c", 0):
                raise ValueError(f"seed {seed}: before S-C an interim config carries (c) with P 0 (H5 said measure)")
        else:
            if seed.get("source") != "S-C" or not seed.get("evidence_sha256"):
                raise ValueError("H5's seed verdict is 'measure': the config's must come from S-C, with its evidence "
                                 "hash")
            want_P = h5_seed.get("P") if seed["verdict"] == "b" else 0  # (b): the lowest-gain end's P; (c): none
            if seed.get("P") != want_P:
                raise ValueError(f"seed {seed}: P must be {want_P} for verdict {seed['verdict']} (H5 freezes "
                                 f"{h5_seed})")
    elif (seed.get("verdict"), seed.get("P")) != (h5_seed.get("verdict"), h5_seed.get("P")):
        raise ValueError(f"seed {seed} vs H5 freezes {h5_seed}")
    if seed["verdict"] == "b" and not seed.get("d6_judge_sha256"):  # prereg v3 addenda 4-5 (either D6 judge, M39)
        raise ValueError("seed verdict b: seed.d6_judge_sha256 (the D6 judge the release rests on) is required")
    g_t = cfg.get("g_t")
    if g_t is not None and not (isinstance(g_t, dict) and _is_sha(g_t.get("sha256")) and g_t.get("report")):
        raise ValueError(f"g_t {g_t!r}: {{report, sha256}} of G-T's report, or null before G-T")
    if release and g_t is None:
        raise ValueError("the release serving config must name G-T's report (g_t: report, sha256)")
    c1_build = cfg["c1_build"]
    bands = cfg["bands"]
    if set(bands) != set(BAND_NAMES) or set(bands.values()) - {"spec", "plain"}:
        raise ValueError(f"bands {bands}: one of spec / plain for each of {list(BAND_NAMES)}")
    c1_policy = cfg.get("c1_policy", "plain")  # v5.2 (M135): a missing key is plain (fail closed)
    if c1_policy not in C1_POLICIES:
        raise ValueError(f"c1_policy {c1_policy!r}: one of {list(C1_POLICIES)}")
    if c1_policy == "b8":
        raise ValueError("c1_policy b8 is reserved for a fixed (8, 8) program (F2a/b): this class builds no (8, 8)")
    # v5.3 (M159, m5-verifier N1): under plain bands.c1 must be plain, whatever c1_build (fail closed; this subsumes
    # v5.2's "c1 is not built: bands.c1 must be plain", the only other policy, b8, being refused above)
    if c1_policy == "plain" and bands["c1"] != "plain":
        raise ValueError("c1_policy plain (or no c1_policy key) with bands.c1 spec: a config that ships c1 SPEC must "
                         "name c1_policy f2c; one that serves c1 plain must say bands.c1 plain")
    # v5.3 (M163): c1_build means "c1 is built under the active rule". Under f2c it is addendum 12's decision
    # (c1_f2c.json), never compared with H5's freezes.c1 (block 8's, trimmed on the f2c run), and a c1 that ships SPEC
    # must be built; under plain it is H5's freezes.c1, as before
    if c1_policy == "f2c":
        if not c1_build and bands["c1"] != "plain":
            raise ValueError("c1_policy f2c with c1_build false (addendum 12 did not build c1) and bands.c1 spec")
    elif c1_build is not freezes["c1"]:
        raise ValueError(f"c1_build {c1_build!r} vs H5 freezes c1 {freezes['c1']!r}")
    return {"switch": switch, "seed": {"verdict": seed["verdict"], "P": int(seed.get("P") or 0),
                                       "d6_judge_sha256": seed.get("d6_judge_sha256")},
            "c1_build": c1_build, "bands": dict(bands), "c1_policy": c1_policy, "g_t": g_t, "sha256": expected,
            "h5_sha256": cfg["h5_report"]["sha256"], "frozen": True}


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def interim_config(value):
    """GEMMA4_M7_SERVING_CONFIG=<path>:<sha256> (H6 only) -> (directory, file name, sha256)"""
    path, _, sha = value.rpartition(":")
    if not path or not _is_sha(sha):
        raise ValueError(f"GEMMA4_M7_SERVING_CONFIG={value!r}: expected <path to the config json>:<its sha256>")
    return os.path.dirname(os.path.abspath(path)), os.path.basename(path), sha


class _RingStore:
    """S0's tap-store hooks, for M7: the rings live in the drafter ring and are zeroed when assigned (C9')"""

    slots = RINGS

    def reset(self, tap_slot):
        pass

    def seal(self):
        pass


class Gemma4ResidentMultiM7V53ForCausalLM(Gemma4ResidentMultiS32PBRingAPCV3ForCausalLM):
    DEFAULT_PACKAGE = "resident_m5_14"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log_plugin_proof()  # M52 (3): the plugin root and file hashes, as the A arm and R4 classes log them
        seed = os.environ.get("GEMMA4_M7_SEED", "off")
        if seed != "off":
            raise ValueError(f"GEMMA4_M7_SEED={seed!r}: the seed path is held until H5 (LEAD_DECISIONS_M7 M14)")
        pin = os.environ.get("GEMMA4_M7_SWITCH_PIN") or None
        dummy = os.environ.get("GEMMA4_M7_DUMMY_DRAFT")
        self.dummy_draft = None
        if dummy is not None:  # G-L0 / G-L1's reference (gate-only); "0" is token 0 (<pad>), not off
            if pin != SPEC or not dummy.isdigit():
                raise ValueError(f"GEMMA4_M7_DUMMY_DRAFT={dummy!r} needs a token id and GEMMA4_M7_SWITCH_PIN=spec")
            self.dummy_draft = int(dummy)
            logger.warning("M7 dummy drafts: token {} in every draft column, drafter and ingest not launched (G-L0, "
                           "gates only)", self.dummy_draft)
        self.interim = os.environ.get("GEMMA4_M7_SERVING_CONFIG") or None
        self.serving = self._serving_config(pin)
        if self.serving["frozen"]:
            name = (f"M7 serving config: H6 interim {self.serving['sha256'][:12]}" if self.interim
                    else f"M7 serving config {self.serving['sha256'][:12]}")
            if pin:  # addendum 7 Q4: the pin moves only the switch mode of the spec bands it pins
                name += f"; switch pin {pin} (gates only)"
            logger.info("{}: h5 {}, switch {}, seed {}, c1_build {}, bands {}, c1_policy {}, g_t {}", name,
                        self.serving["h5_sha256"][:12], self.serving["switch"], self.serving["seed"],
                        self.serving["c1_build"], self.serving["bands"], self.serving["c1_policy"],
                        self.serving["g_t"])
        else:
            logger.warning("M7 serving config: unfrozen (GEMMA4_M7_SWITCH_PIN={}, gates only): provisional {}, "
                           "c1_policy {}", pin, self.serving["switch"], self.serving["c1_policy"])
        if self.serving["seed"]["verdict"] != "c":
            raise ValueError(f"seed verdict {self.serving['seed']['verdict']!r}: the seed path is not built (M14)")
        # "spec": behind the T switch; "plain": never SPEC, taps masked. v5.2: c1 speculates only under the f2c
        # policy (on (8, 4), c1_build not consulted); no (8, 8) program under any policy
        self.c1_policy = self.serving["c1_policy"]
        self.bands = bands_for(self.c1_policy)
        self.spec_bands = {band for _, band in self.bands if self.serving["bands"][band.name] == "spec"
                           and (band.name != "c1" or self.c1_policy == "f2c")}
        self.draft_blocks = tuple(sorted({band.M for band in self.spec_bands}))  # D programs: blocks a spec band uses
        self.rings = min(RINGS, self.max_batch_size)
        self.slots = M7SlotTable(self.max_batch_size, self.rings, on_assign=self._ring_assigned, ring_users=RINGS)
        self.switches = {}  # one per band; a band outside spec_bands never speculates, whatever its switch says
        low = 1
        for most, band in self.bands:
            params = self.serving["switch"][band.M]
            cap = params["cap"]
            reentry = isinstance(cap, int) and cap >= 256  # a cap below 256 ("<256", or 64): re-entry off (§5.3)
            sw = TSwitch(s_table={}, J=params["J"], n_r_cap=cap if reentry else 4096, reentry=reentry)
            sw.s.update({(band.R, c): S_SEED[c] for c in range(low, most + 1)})
            self.switches[band], low = sw, most + 1
        logger.info("M7 bands {}: spec {}, D blocks built {}, c1 policy {} (c1 {})", list(BAND_NAMES),
                    sorted(b.name for b in self.spec_bands), list(self.draft_blocks), self.c1_policy,
                    tuple(self.bands[0][1][:2]))
        self.current_band = None
        self.m7_off = None  # the reason speculation stopped for the process (a D flag)
        self.taps_on = None  # the engine's tap mask as last set (None: unknown, e.g. after a rebuild)
        self.want_taps = True
        self.taps_by_launch = {}  # engine launch -> taps on (the last few)
        self.ingest_queue = []  # (state, step, c): committed rows waiting for the ring
        self.accepted = {}  # engine slot -> drafts accepted in the last verify, for the next SPEC iteration
        self.last_start = None  # ("verify" | "plain" | "forced" | "other", seconds, band, live) of the last launch
        self.effective = None  # (band, mode) the last launch ran under
        self.synced = Counter()  # synced launches by kind (the per-1K line)
        self.m7_program = None  # (R, M) of the last launch
        self.draft_band = None  # (R, M) of the band whose propose made the outstanding drafts (F1)
        self.propose_program = None  # v5.2: (R, M) of the propose running now, for the verify cap in _propose
        self.drafter_cb_tops = None  # D v8 / ingest v5 CB tops, read once for the L1 guard (M62 note 2)

    # ------------------------------------------------------------------ construction

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """K = 7 drafted per block of 8, capped by live rows per M7's bands; argmax_ids with narrow decode (sampled,
        penalized and API-feature requests decode without speculation: prereg v3 §2.4, §2.6)"""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if not 1 <= max_num_seqs <= MAX_USERS:
            return SpecReject(reason=f"Resident Gemma4 QB2 speculation serves 1..{MAX_USERS} requests")
        if requested_k != cls.K:
            return SpecReject(
                reason=f"DFlash2 block {cls.BLOCK} drafts exactly {cls.K} tokens; num_speculative_tokens={requested_k}",
                supported_k=(cls.K,),
            )
        caps = tuple((rows, k) for rows, k in K_BY_ROWS if rows < max_num_seqs)
        caps += (next((rows, k) for rows, k in K_BY_ROWS if rows >= max_num_seqs),)
        return SpecPlan(
            effective_k=cls.K,
            lanes_per_request=1,
            extra_bytes_per_seq=0,  # M7's device memory (rings, staging, D programs) is fixed at build (prereg §8)
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
            k_by_rows=caps,
        )

    def _serving_config(self, pin):
        """the class-constant config, or under GEMMA4_M7_SERVING_CONFIG (H6 only) the named interim config, with the
        same checks and its sha in place of the constant"""
        if self.interim:
            if M7_SERVING_CONFIG_SHA256 is not None:  # addendum 7 Q1: a release class refuses an interim config
                raise ValueError("GEMMA4_M7_SERVING_CONFIG (an H6 interim config) on a class with a frozen release "
                                 "config (M7_SERVING_CONFIG_SHA256 set): refused")
            directory, name, sha = interim_config(self.interim)
            return load_serving_config(pin, expected=sha, directory=directory, name=name, release=False)
        return load_serving_config(pin)

    def _engine_block_map(self):
        """m5_13's programs per R: M = 1 everywhere (option S, PLAIN, the draftless verify) plus each speculating band's
        verify block (H1's set B when every band ships SPEC)"""
        blocks = {R: {1} for R in self.ROWS}
        for band in self.spec_bands:
            blocks[band.R].add(band.M)
        return {R: tuple(sorted(b)) for R, b in blocks.items()}

    def _build_engine(self):
        engine = super()._build_engine()
        if not hasattr(engine, "set_taps_enabled"):
            raise ValueError(f"M7 needs resident_m5_13's runtime tap mask; the engine package lacks set_taps_enabled "
                             f"(GEMMA4_RESIDENT_ENGINE_PACKAGE={os.environ.get('GEMMA4_RESIDENT_ENGINE_PACKAGE')})")
        if not callable(getattr(engine, "verify_cap", None)):  # v5.2 (M131)
            raise ValueError(f"M7 v5.2 needs resident_m5_14's verify_cap (the no-straddle verify cap); the engine "
                             f"package lacks it (GEMMA4_RESIDENT_ENGINE_PACKAGE="
                             f"{os.environ.get('GEMMA4_RESIDENT_ENGINE_PACKAGE')})")
        return engine

    def _verify_cap(self, R, M, p):
        """v5.2 (M131): the rows a verify block of program (R, M) at position p may use (the engine's rule)"""
        L = int(self.engine.verify_cap(R, int(p), block=M))
        if not 1 <= L <= M:
            raise ValueError(f"engine.verify_cap({R}, {p}, block={M}) = {L}: outside 1..{M}")
        return L

    def build_taps(self):
        return _RingStore()

    def build_drafter(self):
        """M7Device (tt/m7_device_v3.py) after the engine, before any trace (I3): ring, ingest v5 (staging max(1, P),
        never written while the seed is held), D v8 per speculating band's block with RoPE aliased to the engine's
        sliding pair. No band speculates: nothing is built."""
        if not self.draft_blocks:
            return None
        from models.demos.gemma4_31b_qb2.tt.m7_device_v3 import M7Device

        setup = self.generator.model.shared_setup
        rope = (setup[("rope", "sliding_attention")]["decode_cos"], setup[("rope", "sliding_attention")]["decode_sin"])
        device = M7Device.build(self.mesh, self.engine, blocks=self.draft_blocks,
                                staging_slots=max(1, self.serving["seed"]["P"]), rings=self.rings, rope=rope)
        return self._drafter_view(device)

    def _drafter_view(self, device):
        """S0's DraftSource attributes on the M7 device: tap_layers (checked against the engine's), max_positions (the
        spec limit: a D block at p needs p + M <= capacity) and reset (S0's admission hook; the ring was zeroed when
        assigned)"""
        device.tap_layers = tuple(self.engine.cfg.tap_layers)
        device.max_positions = device.capacity()  # D v8: prog.att.capacity (v1 read p.capacity, which D v8 lacks)
        device.reset = lambda tap_slot: None
        return device

    def _new_state(self, key, slot, tap_slot):
        """S1a's queued state, keeping the ring (S0 drops a tap slot without a TapBook; M7 has none)"""
        state = QueuedSlotState(key, slot, tap_slot, ResidentSpecPolicy(self.K), None)
        state.entry_pending = False
        state.m7 = Counter()
        state.m7_digest = hashlib.sha256()
        state.m7_ring = tap_slot is not None
        state.m7_spec_program = state.m7_plain_program = None
        state.m7_req = None
        return state

    def _ring_assigned(self, ring, state, why):
        """C9': zero every ring when assigned, before any draft (the class has no seed while it is held)"""
        if self.drafter is not None:
            self.drafter.zero(ring)
        state.m7_ring = True
        state.regime = "plain"  # a handed-off ring starts cold: the next SPEC proposal enters
        self.counters[f"m7_rings_{why}"] += 1

    # ------------------------------------------------------------------ modes

    def _switch(self, band):
        """the band's switch; entering a band restarts its n_r and PLAIN count (§5.1: a band change resets them)"""
        sw = None if band is None else self.switches[band]
        if band != self.current_band:
            self.current_band = band
            if sw is not None:
                sw.band(object())
        return sw

    def _spec(self, band):
        """the band ships SPEC (serving config; v5.2: c1 only under f2c), speculation has not stopped, and its switch
        says SPEC"""
        return band in self.spec_bands and self.m7_off is None and self.switches[band].mode == SPEC

    def _effective(self, band, live_states):
        """(program, mode) of this launch; a change counts once per live request (mode_switches: SPEC / PLAIN and band
        program changes, D9; c3-4 -> c5-8 in one mode is the same (32, 4) program and does not count)"""
        now = (None if band is None else (band.R, band.M), SPEC if self._spec(band) else PLAIN)
        if self.effective is not None and now != self.effective:
            for state in live_states:
                state.m7["mode_switches"] += 1
            self.counters["m7_mode_changes"] += 1
        self.effective = now

    def _sync(self):
        import ttnn

        ttnn.synchronize_device(self.mesh)
        if self.drafter is not None:
            self.drafter.synced()

    def _count_synced(self, kind, n=1):
        before = sum(self.synced.values())
        self.synced[kind] += n
        if before // SYNC_LINE_EVERY != (before + n) // SYNC_LINE_EVERY:
            logger.info("M7 synced launches: {}", {k: self.synced[k] for k in ("verify", "ingest", "D", "switch")})

    def _live_states(self, rows):
        return [self.slots.states[s] for s in rows if s is not None and s in self.slots.states]

    # ------------------------------------------------------------------ launches

    def _launch(self, R, M=1):
        """S1a's early upkeep (queued rows whose taps buffer this launch overwrites are ingested first), then the tap
        mask the step wants (flipped only after a sync), then the launch; the launch's mask is remembered so a step
        that ran masked ingests nothing"""
        upcoming = self.engine.launches
        if any(step.launch <= upcoming - 2 for _, step, _ in self.ingest_queue):
            self._flush_ingest(upto=upcoming - 2)
        if self.taps_on is not self.want_taps:
            self._sync()
            self.engine.set_taps_enabled(self.want_taps)
            self.taps_on = self.want_taps
            self._count_synced("switch")
        super()._launch(R, M)
        self.taps_by_launch[upcoming] = self.taps_on
        for old in [k for k in self.taps_by_launch if k < upcoming - 4]:
            del self.taps_by_launch[old]
        self.m7_program = (R, M)

    def _l1_guard(self, reason, before_rebuild=None):
        """S0's guard (it ingests what is queued before a rebuild's warmup rewrites the taps buffers); the tap mask is
        set again before the next launch, since a rebuild's warmup may have reset it (one sync, after a novel prefill)"""
        super()._l1_guard(reason, before_rebuild=before_rebuild)
        self.taps_on = None
        self._drafter_l1_guard(reason)

    def _drafter_cb_tops(self):
        """{program: highest L1 address its CBs occupy} for every D v8 block and ingest v5 (static: built once)"""
        if self.drafter_cb_tops is None:
            tops = {f"D b{b}": max(p.cb_tops().values()) for b, p in sorted(self.drafter.progs.items())}
            tops["ingest"] = max(self.drafter.ingest_prog.cb_tops().values())
            self.drafter_cb_tops = tops
        return self.drafter_cb_tops

    def _drafter_l1_guard(self, reason):
        """M62 note 2: no ttnn L1 buffer inside a D / ingest CB range (the engine's half is M1's guard)"""
        if self.drafter is None:
            return
        floor, tops = self._l1_floor(), self._drafter_cb_tops()
        logger.info("L1 guard after {}: M7 drafter CB tops {} (lowest ttnn L1 buffer {})", reason, tops, floor)
        engine = int(self.engine.cb_top())  # v4, M67: the one number the floor must clear
        logger.info("M7 L1 guard after {}: max CB top {} (engine {}) vs lowest ttnn L1 buffer {}", reason,
                    max(engine, *tops.values()), engine, floor)
        if floor is not None and max(tops.values()) > floor:
            raise RuntimeError(f"L1 buffer at {floor} below the M7 drafter CB tops {tops} after {reason}: D v8 / "
                               f"ingest v5 cannot be rebuilt in place")

    # ------------------------------------------------------------------ prefill

    def prefill_forward(self, *args, **kwargs):
        self.last_start = None  # a prefill between two steps: neither is a timed iteration
        return super().prefill_forward(*args, **kwargs)

    # ------------------------------------------------------------------ ordinary decode

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, *, spec_mode=None, sampling_params=None,
                       **kwargs):
        if spec_mode is not None:
            return super().decode_forward(tokens, start_pos, page_table, kv_cache, spec_mode=spec_mode,
                                          sampling_params=sampling_params, **kwargs)
        positions = [int(p) for p in start_pos.reshape(-1)[: self.max_batch_size]]
        live = sum(p >= 0 for p in positions)
        band = band_of(live, self.bands)
        sw = self._switch(band)
        now = time.perf_counter()
        states = list(self.slots.states.values())
        forced = self._spec(band) and (sampling_params is None or any(s.offer is not None for s in states))
        kind = "other"
        if forced:
            sw.forced_step()
            kind = "forced"
        elif band is not None and not self._spec(band) and self.m7_off is None:
            last = self.last_start
            if last is not None and last[0] == "plain" and last[2:] == (band, live):
                sw.plain_step(now - last[1], band[0], live)
            kind = "plain"
        self.last_start = (kind, now, band, live)
        self.want_taps = self._spec(band)
        self._effective(band, states)
        out = super().decode_forward(tokens, start_pos, page_table, kv_cache, spec_mode=None,
                                     sampling_params=sampling_params, **kwargs)
        for state in self._live_states(self.row_slots):
            state.m7_plain_program = self.m7_program
            if forced:
                state.m7["forced_ordinary_steps"] += 1
        return out

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        """v5 (module docstring): the chain's prefill warmups with the drafter set aside (S3-PB refuses any drafter for
        PB3's seed pre-warm; M7 has no seed path), then, the first time, M7Device's programs before any trace"""
        first = self.generator.pb_warmed is None
        drafter, self.drafter = self.drafter, None
        try:
            super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        finally:
            self.drafter = drafter
        if first and drafter is not None:
            self._prewarm_drafter()

    def _prewarm_drafter(self):
        """every M7Device program once on ring slot 0, before any trace capture (module docstring, v5, v5.1): the guard,
        the ring zero, each D block and the ingest (not under the dummy pin), the status rows cleared, the zero again"""
        self._drafter_l1_guard("drafter pre-warm")
        dev, ring = self.drafter, 0
        t0 = time.perf_counter()
        dev.zero(ring)
        launched = []
        if self.dummy_draft is None:  # the dummy pin: the drafter and ingest are never launched
            position = min(2048, dev.capacity() - max(dev.progs))  # a full 2,048-row window of the zeroed slot
            for block in sorted(dev.progs):
                _, flags = dev.draft(block, [(0, ring, dev.mask_id, position)])
                for b, bits in flags.items():
                    self.counters["m7_drafter_prewarm_flags"] += 1
                    logger.warning("M7 drafter pre-warm: D block {} launch flags {:#x} (status row cleared after)", b,
                                   bits)
            dev.ingest([(0, 0, ring)], self.engine.launches - 1)  # taps_live(-1, 0): live before any launch
            dev.sync()
            dev.synced()
            for block, prog in dev.progs.items():  # sticky: a warm launch's status bits would mask a served flag
                prog.clear_status()
                dev.status_seen[block] = None
            launched = [f"D blocks {sorted(dev.progs)}", "ingest"]
        dev.zero(ring)
        self.counters["m7_drafter_prewarms"] += 1
        logger.info("M7 drafter programs built before any trace: ring zero{} ({:.1f} s)",
                    "".join(f", {x}" for x in launched) or " (dummy pin: D and ingest not launched)",
                    time.perf_counter() - t0)

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        """The chain's warmup, then: every speculating band's verify program covers the serving context (fail closed at
        start)"""
        first = self.prefill_tokens is None
        if first:
            self._prewarm_sampler()  # M62: before engine.warmup()'s trace capture (M1 warmup_model_decode)
        super().warmup_model_decode(kv_cache, enable_trace, **kwargs)
        if first:
            programs = sorted({(band.R, band.M) for band in self.spec_bands})
            short = {(R, M): int(self.engine.capacity_for(R, block=M)) for R, M in programs
                     if int(self.engine.capacity_for(R, block=M)) < self.max_seq_len}
            if short:
                raise ValueError(f"M7 verify programs (R, M): capacity {short} below max_seq_len {self.max_seq_len}")
            self.taps_on = None  # set explicitly (after a sync) before the first served launch

    def _prewarm_sampler(self):
        """M62 (H4b v2's trace-allocation FAIL): the sampler's eager first call creates its all_gather programs, whose
        factories allocate two global semaphores each (ttnn all_gather_unicast_factory.cpp:639-645, L1_SMALL when
        l1_small_size > 0: ccl_common.cpp:160-163) and keep them in the cached program, per op config, not per call.
        M1 runs that first call in _capture_sampling, after engine.warmup() captured the engine's traces, so they were
        allocated after a capture. Here the same eager step runs once before any capture (the engine and M7Device are
        built in allocate_kv_cache_per_layer; the D and ingest programs launch eagerly), with M1's warmup sampling
        config, its seeds and the feed restored after it as _capture_sampling does; the later capture's own eager step
        then hits the program cache and allocates no semaphore."""
        self.generator.configure_sampling(batch=self.max_batch_size, temperature=0)  # = M1 warmup_model_decode's
        entry = {"logits": self.engine.logits, "tokens": self.engine.tokens}
        seeds = self.generator.sampler.seeds_tt_tensor
        import ttnn

        saved = (ttnn.clone(seeds), ttnn.clone(self.engine.tokens))
        self.generator._sample_step(entry)
        ttnn.copy(saved[0], seeds)
        ttnn.copy(saved[1], self.engine.tokens)
        del saved
        self.counters["m7_sampler_prewarms"] += 1

    def _drain_taps(self, only=None):
        """S0 calls this after a launch (inside its device window) and before a rebuild: the queued ingest"""
        self._flush_ingest()

    # ------------------------------------------------------------------ verify

    def _verify(self, tokens, start_pos, kv_cache, spec_mode, num_valid_drafts, slot_remap, sampling_params,
                page_tables_per_layer):
        """S0's argmax_ids verify (generator_resident_multi_vllm._verify) on the band's program: live row b's
        [1, 1 + n_b] block as user block u at launch rows [u M, u M + M). A verify with no drafts (the step after a
        multi-token commit) launches S0's (32, 1). The T switch is fed here: the wall time since the previous verify
        start, when nothing else ran between them, with each user's accepted drafts of that verify."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if spec_mode != "argmax_ids":
            raise ValueError(f"Resident speculation serves argmax_ids verifies only, got {spec_mode!r}")
        batch = self.max_batch_size
        width = tokens.shape[1]
        positions_in = start_pos.reshape(start_pos.shape[0], -1)
        live = [b for b in range(batch) if int(positions_in[b, 0]) >= 0]
        drafts = max(int(num_valid_drafts[b]) for b in live)
        band = band_of(len(live), self.bands)  # where the live count is now (switch, mode); the drafts may be older
        program = self._verify_program(len(live), drafts) if drafts else None
        drop = bool(drafts) and program is None  # F1 (3): no built program fits: this launch verifies no draft
        if drop:
            self.counters["m7_verify_drop_crossings"] += 1
            self.counters["m7_drafts_dropped_on_crossing"] += sum(int(num_valid_drafts[b]) for b in live)
        elif program is not None and any(int(num_valid_drafts[b]) + 1 > self._verify_cap(*program, positions_in[b, 0])
                                         for b in live):
            # v5.2: a block past the cap of the program this launch takes (module docstring): F1 (3)'s drop
            drop, program = True, None
            self.counters["m7_verify_cap_drops"] += 1
            self.counters["m7_drafts_dropped_on_cap"] += sum(int(num_valid_drafts[b]) for b in live)
        R, M = program if program is not None else (max(self.ROWS), 1)
        if len(live) * M > R:
            raise ValueError(f"{len(live)} users x {M} rows do not fit R = {R}")
        sw = self._switch(band)
        now = time.perf_counter()
        last = self.last_start
        aligned = band is not None and program == (band.R, band.M)  # not a crossing: a timed SPEC iteration
        if aligned and last is not None and last[0] == "verify" and last[2:] == (band, len(live)) and self.accepted:
            sw.spec_iteration(now - last[1], dict(self.accepted), R, len(live))
        self.accepted = {}
        self.last_start = ("verify" if aligned else "other", now, band, len(live))
        sources = self._group_tables(page_tables_per_layer)
        rows = self._layout(sources, [int(p) for p in positions_in[:batch, 0]], slot_remap)
        self._write_slot_tables(sources, rows)
        self._guard_users(R, [rows[b] for b in live])
        self.row_slots = rows
        self.want_taps = self._spec(band)
        self._effective(band, self._live_states(rows))
        slots, positions, ids_in, blocks = [0] * R, [-1] * R, [0] * R, []
        for u, b in enumerate(live):
            state = self.slots.states[rows[b]]
            n = int(num_valid_drafts[b])
            block = [int(t) for t in tokens[b, : n + 1]]
            p = int(positions_in[b, 0])
            offer, state.offer = state.offer, None
            if n:
                if offer is None or offer.anchor != block[0] or offer.position != p or offer.drafts[:n] != block[1:]:
                    raise ValueError(f"Verify block {block} at {p} is not the retained proposal {offer}")
                if drop:
                    n, block = 0, block[:1]  # the anchor alone; its drafts are answered with placeholders
                elif p + M > self._spec_limit():
                    raise ValueError(f"Verify block [{p}, {p + M}) passes the limit {self._spec_limit()}")
            for j in range(n + 1):
                slots[u * M + j], positions[u * M + j], ids_in[u * M + j] = rows[b], p + j, block[j]
            blocks.append((u, b, state, n, p, offer))
        self.engine.write_rows(slots, positions, ids_in)
        temps = None if sampling_params is None else self._rows(sampling_params.temperature, batch)
        sampled = temps is not None and any(temps[b] > 0 for b in live)
        if sampled:
            self._configure_verify_sampling(sampling_params, blocks, M, MAX_USERS)
            if self.sampling_trace is None:
                self._capture_sampling()
                self._guard_after_capture()
                self.engine.write_rows(slots, positions, ids_in)
        mode = ("none", sampled)
        if mode != self.mode:
            self.engine.set_mode(feed=mode[0], write_logits=mode[1])
            self.mode = mode
        launch = self.engine.launches
        self.launch_users = [rows[b] for b in live] + [None] * (R // M - len(live))
        self._launch(R, M)
        if sampled:
            self._replay_sampling()
            greedy = [u * M + j for u, b, _, n, _, _ in blocks if temps[b] == 0 for j in range(n + 1)]
            if greedy:
                self._restore_greedy(greedy)
        ids = self.engine.read_tokens(R)
        self._check_status(self._status_words())
        self._count_synced("verify")
        self.feed_stale = True  # the engine's positions are the blocks'; the next ordinary decode reloads
        # v4: a drop launch answers one column, so the runner's walk verifies no draft (module docstring)
        out = torch.full((tokens.shape[0], 1 if drop else width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        for u, b, state, n, p, offer in blocks:
            out[b, : n + 1] = ids[u * M : u * M + n + 1].to(torch.int32)
            # a crossing's dropped drafts are not the request's dropped offer: its next proposal drafts again
            state.step = SlotStep("verify", launch, p, u * M, drafts=n, dropped=offer is not None and n == 0 and not drop)
            state.counters["verifies"] += 1
            state.counters["verified_drafts"] += n
            if n:
                state.m7_spec_program = (R, M)
        self.counters["verifies"] += 1
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out)

    def _verify_program(self, live, drafts):
        """F1 (M40): the verify program for `live` rows carrying up to `drafts` drafts each, among the built ones:
        (1) the drafting band's, if it still fits (live x M <= R, drafts <= M - 1); (2) else the smallest-R built
        program that fits, never (8, 8); (3) else None (the caller drops the drafts)."""
        built = {(band.R, band.M) for band in self.spec_bands}

        def fits(R, M):
            return (R, M) in built and live * M <= R and drafts <= M - 1

        if self.draft_band is not None and fits(*self.draft_band):
            return self.draft_band
        return next(((R, M) for R, M in sorted(built) if (R, M) != (8, 8) and fits(R, M)), None)

    # ------------------------------------------------------------------ propose

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """S1a's propose with M7's drafting: in SPEC every ring holder drafts (block M of its band, M - 1 drafts),
        after every queued committed row is ingested; a stale entry offers S1a's placeholder"""
        from vllm_tt_plugin.spec_decode import DraftOutput

        k = int(num_drafts)
        if k != self.K:
            raise ValueError(f"Expected {self.K} draft columns, got {k}")
        rows = committed_tokens.shape[0]
        ids = torch.full((rows, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        counts = accepted_counts.reshape(-1).tolist()
        tokens, positions = committed_tokens.tolist(), committed_positions.tolist()
        live = [r for r in range(min(rows, len(self.row_slots)))
                if positions[r][0] >= 0 and self.row_slots[r] is not None]
        band = band_of(len(live), self.bands)
        spec = self._spec(band)
        cap = band[1] - 1 if spec else 0
        self.propose_program = (band.R, band.M) if spec else None  # v5.2: the verify cap's program in _propose
        if cap and any(positions[r][counts[r] - 1] + self.BLOCK > self._spec_limit() for r in live):
            cap = 0  # S0's capacity-edge guard
            self.counters["capacity_edge_steps"] += 1
        jobs = []
        for r in live:
            state = self.slots.states.get(self.row_slots[r])
            if state is None:
                continue  # released between the step and its propose
            step, state.step = state.step, None
            if step is None:
                continue
            c = counts[r]
            position = positions[r][c - 1]
            if position != step.position + c:
                raise ValueError(f"Row {r}: anchor position {position} is not the step's {step.position} plus {c}")
            job = self._propose(state, step, c, tokens[r][c - 1], position, cap, spec)
            if job == "placeholder":
                ids[r, 0] = tokens[r][c - 1]
                valid[r] = 1
                self.counters["placeholder_offers"] += 1
            elif job is not None:
                jobs.append((r, state, job))
        self._flush_ingest()
        if jobs:
            block = band[1]
            self.draft_band = (band.R, band.M)  # F1: the verify of these drafts prefers this program
            if self.dummy_draft is not None:  # G-L0's reference: fixed drafts, no D launch
                drafted = [[self.dummy_draft] * (block - 1) for _ in jobs]
            else:
                drafted, flags = self.drafter.draft(block, [(u, state.tap_slot, anchor, position)
                                                            for u, (_, state, (anchor, position)) in enumerate(jobs)])
                self._count_synced("D")
                self._check_drafter(flags)
            for (r, state, (anchor, position)), d in zip(jobs, drafted):
                n = min(cap, self._verify_cap(band.R, block, position) - 1)  # v5.2: the verify cap at this anchor
                if n < 1:
                    raise RuntimeError(f"Engine slot {state.slot}: a job at {position} with no draft under the verify "
                                       f"cap (_propose declines those)")
                if n < cap:
                    self.counters["m7_verify_cap_truncations"] += 1
                drafts = [int(t) for t in d[:n]]
                ids[r, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
                valid[r] = len(drafts)
                state.offer = Offer(anchor, position, drafts)
                state.regime = "hold"
                state.counters["drafts"] += 1
        return DraftOutput(draft_token_ids=ids, num_valid=valid)

    def _propose(self, state, step, c, anchor, position, cap, spec):
        """One request's proposal: (anchor, position) to draft, "placeholder", or None to decline. A verify of its
        drafts records the accepted count (T, accept_digest); in SPEC the step's committed rows queue for the ring."""
        if step.kind == "verify" and step.drafts and state.regime == "hold":
            state.counters["accepted_drafts"] += c - 1
            state.m7_digest.update(f"{c - 1},".encode())
            self.accepted[state.slot] = c - 1
        state.counters["committed_tokens"] += c
        if state.tap_slot is not None and self.taps_by_launch.get(step.launch) and self.dummy_draft is None:  # K3
            self.ingest_queue.append((state, step, c))
        if step.host or step.dropped:
            state.counters["spec_nonspeculable_steps"] += 1
            if step.dropped:
                # the request's own drop only when the runner trimmed its offer under max_model_len (S0)
                own = position + self.K >= self.max_seq_len
                state.counters["spec_offers_dropped" if own else "spec_offers_dropped_cotenant"] += 1
                state.suppressed = state.suppressed or own
            return self._decline(state)
        if not spec:
            return self._decline(state)
        if cap == 0:
            state.counters["spec_rows_over_cap"] += 1
            return self._decline(state)
        if state.tap_slot is None:
            state.counters["spec_no_tap_slot"] += 1
            return self._decline(state)
        if position + self.BLOCK > self._spec_limit():
            state.exhausted = True
        if state.suppressed or state.exhausted:
            return self._decline(state)
        if not state.stale and self._verify_cap(*self.propose_program, position) == 1:
            # v5.2: only the anchor fits before a boundary: decline (never hold with no drafts; module docstring)
            self.counters["m7_verify_cap_declines"] += 1
            return self._decline(state)
        if state.regime == "hold":
            if state.stale:
                raise RuntimeError(f"Engine slot {state.slot}: a stale proposal in hold (a verify is never overlapped)")
            return anchor, position
        if state.stale:
            if not state.entry_pending:
                state.counters["spec_entries_deferred"] += 1
            state.entry_pending = True
            return "placeholder"
        state.entry_pending = False
        state.counters["entries"] += 1
        return anchor, position

    def _decline(self, state, *args, **kwargs):
        if state.entry_pending:
            state.counters["spec_entries_cancelled"] += 1
        state.entry_pending = False
        state.regime = "plain"
        state.counters["declines"] += 1
        return None

    # ------------------------------------------------------------------ ingest and drafter checks

    def _flush_ingest(self, upto=None):
        """The queued committed rows into the rings, by taps launch (oldest first), in ingest launches of <= 32 rows;
        `upto`: only launches <= upto (S1a's early rule). A request released or re-ringed since is skipped."""
        if not self.ingest_queue:
            return
        run = [e for e in self.ingest_queue if upto is None or e[1].launch <= upto]
        self.ingest_queue = [e for e in self.ingest_queue if not (upto is None or e[1].launch <= upto)]
        by_launch = {}
        for state, step, c in run:
            if self.slots.states.get(state.slot) is not state or state.tap_slot is None:
                self.counters["m7_ingest_skipped_released"] += 1
                continue
            by_launch.setdefault(step.launch, []).extend(
                (step.row + j, step.position + j, state.tap_slot) for j in range(c))
        for launch in sorted(by_launch):
            n = self.drafter.ingest(by_launch[launch], launch)
            self._count_synced("ingest", n)
            self.counters["m7_ingest_rows"] += len(by_launch[launch])

    def _check_drafter(self, flags):
        if flags and self.m7_off is None:
            self.m7_off = f"D flags {flags}"
            for sw in self.switches.values():
                sw.pin, sw.mode = PLAIN, PLAIN
            logger.error("M7: the drafter raised new flags {} (block: bits); speculation stops for this process", flags)
            self.counters["m7_off"] += 1

    # ------------------------------------------------------------------ lifecycle

    def release_request(self, slot):
        """APC v3's release (S0 frees the engine slot and ring, S1 drops the key, then released()); the request id is
        read first, for the release line"""
        engine = self.slots.plugin.get(int(slot))
        state = self.slots.states.get(engine) if engine is not None else None
        if state is not None:
            state.m7_req = self._key_reqs.get(state.key)
            self.ingest_queue = [e for e in self.ingest_queue if e[0] is not state]
            self.accepted.pop(state.slot, None)
        super().release_request(slot)

    def released(self, state):
        """C6: after S0 and S1 released the request: its release line, then the freed ring's hand-off"""
        R, M = state.m7_spec_program or state.m7_plain_program or (None, None)
        line = {
            "committed_tokens": state.counters["committed_tokens"],
            "seeds": state.m7["seeds"],
            "seeds_dropped": state.m7["seeds_dropped"],
            "staging_full": state.m7["staging_full"],
            "drafts": state.counters["drafts"],
            "verifies": state.counters["verifies"],
            "verified_drafts": state.counters["verified_drafts"],
            "accepted_drafts": state.counters["accepted_drafts"],
            "accept_digest": state.m7_digest.hexdigest(),
            "R": R,
            "M": M,
            "mode_switches": state.m7["mode_switches"],
            "forced_ordinary_steps": state.m7["forced_ordinary_steps"],
            "ring": "seeded" if state.m7["seeds"] else "cold" if state.m7_ring else "none",
        }
        if tuple(line) != LINE_KEYS:
            raise RuntimeError(f"M7 release line keys {tuple(line)} are not §6's {LINE_KEYS}")
        logger.info("M7 request done req={}: {}", state.m7_req, line)
        self.slots.hand_off()

    def release_persistent_capture(self):
        self.ingest_queue = []
        super().release_persistent_capture()
