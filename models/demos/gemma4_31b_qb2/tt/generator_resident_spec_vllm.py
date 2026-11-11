# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: the resident decode engine with DFlash2 speculative decoding, one user (M2).

Design: bench/resident/M2_DESIGN_v4.md (section numbers below refer to it). On top of the M1 adapter:
  - verify: an `argmax_ids` decode_forward runs the block's n + 1 rows in one launch (R = 8 when n >= 1);
  - propose: after every step, the step's committed tap rows go to the launch-tap ring and history
    (resident_taps), then the request either declines, drafts from the rows the verify committed (hold), or
    seeds from the history window and drafts (entry: the first draft, a re-entry);
  - a host-sampled step or a dropped offer makes the request decline with no seed and no forward (§5.2 step 2);
  - spec_policy decides entry, hold and exit (§7); retries are offered trials.
The drafter seed waits for dflash-builder's shape-independent seed (§8.3); everything else is independent of it.
"""

import gc
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.spec_policy import BUCKET_TRIALS, RecordCandidate, SpecPolicy, SpecPolicyConfig
from models.demos.gemma4_31b_qb2.tt import resident_taps
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import (
    DEFAULT_L1_TOP,
    RAW_WEIGHT_CACHE,
    Gemma4ResidentForCausalLM,
)

PLACEHOLDER_TOKEN_ID = -1  # vllm_tt_plugin.spec_decode.PLACEHOLDER_TOKEN_ID
VERIFY_ROWS = 8


@dataclass
class Offer:
    """The proposal this adapter returned and the runner has not yet verified (C:804-816)."""

    anchor: int
    position: int
    drafts: list


@dataclass
class Step:
    """What the last decode step was, for the propose that follows it."""

    kind: str  # "ordinary" or "verify"
    launch: int  # engine.launches before the launch: its taps are taps_buffer(launch % 2) (m9)
    position: int  # row 0's input position
    drafts: int = 0  # n on a verify
    host: bool = False  # sampling_params is None: the plugin samples on the host (V3-1 (i))
    dropped: bool = False  # an offer was outstanding and the runner sent no verify for it (V3-1 (ii))


# Cost priors (ms) by position: fixed constants, the policy's only cost input (Calibration stays unfed), so decisions
# depend on the request alone. interpolate() is piecewise-linear between knots and clamped (flat) beyond the ends.
# - 128 / 1K / 8K: the M1 engine with logits and the 5 taps (resident-analyst, bench/resident/m2/m1_r8_cost_{1,2}.json):
#   R = 1 10.21 / 12.11 / 12.49, R = 8 14.91 / 15.49 / 15.87.
# - 33,000 / 128,000 / 262,000: resident_m3_1 at full clock, cap 262,144 (bench/resident/VERIFY_M3_1.md §4, sha256
#   a2a20a44efdc9322cd3c28851061c8edf77e601298663d06ca48d6ec884f2c79; data verify_m3_1/perf_resident_m3_1_long_h1.json,
#   b2f3e1a274c732b369e4d0e60fe0458194aaa55e22fe8e5c03c7bf8b6ebf3ce8): R = 1 13.03 / 15.34 / 18.68, R = 8 16.21 / 18.50
#   / 21.90 (lead's ruling on resident-verifier's Q3, 2026-11-11; before it the table ended at 32K, flat beyond).
#   The dev-32k point at 32,768 (R = 1 13.97, R = 8 17.53, an older engine) is dropped: the long-context engine is
#   faster just past 32K (wide full layers; m3_1 itself reads 13.60 at 31K and 13.03 at 33K), and keeping both points
#   would make the prior fall with position.
# - Plus the served host time of a sync step, 0.87 ms (M1 sync 11.08 ms at 128, session_sync1, minus the engine's
#   10.21). Draft: DFlash2 BF8 6.67 + 1 ms host; seed: ~11 ms warm + the window read.
SERVED_HOST_MS = 0.87
PRIOR_PLAIN_MS = tuple((x, ms + SERVED_HOST_MS) for x, ms in (
    (128, 10.21), (1024, 12.11), (8192, 12.49), (33000, 13.03), (128000, 15.34), (262000, 18.68)))
PRIOR_VERIFY_MS = tuple((x, ms + SERVED_HOST_MS) for x, ms in (
    (128, 14.91), (1024, 15.49), (8192, 15.87), (33000, 16.21), (128000, 18.50), (262000, 21.90)))
PRIOR_DRAFT_MS = 7.67
PRIOR_SEED_MS = 11.5


def interpolate(table, position):
    """Piecewise-linear in position over (position, ms) knots, flat outside them."""
    if position <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if position <= x1:
            return y0 + (y1 - y0) * (position - x0) / (x1 - x0)
    return table[-1][1]


# The second-trial rule (lead's ruling (iii) on the M2 item-3 follow-up, 2026-11-11). A request's first trial is
# judged on one verify from an uninformed prior, and a request that failed it was never offered another within its
# length: V4's code request, whose first verify accepts 0 and nearly every later one is high, declined its other
# 254 steps (bench/resident/m2_delta/DEVICE_VALIDATION.md, V4). Now a trial that exits before any hold decision has
# judged the request's own acceptance re-primes the request (a fresh spec_policy, its bucket full) RETRY_AFTER plain
# proposals later, at most RETRY_LIMIT times, each gap RETRY_GROWTH times the last. Counted in the request's own
# proposals, so it stays request-intrinsic and deterministic. Chosen by replay over 13 per-verify traces:
# bench/resident/m2/second_trial_replay.py.
RETRY_AFTER = 16
RETRY_LIMIT = 1
RETRY_GROWTH = 4


def policy_mode():
    """GEMMA4_RESIDENT_SPEC_POLICY: adaptive (spec_policy, the default), force (draft on every speculable step: the
    gates' "policy forced on"), or never (no entry ever: the regression's F2 run, the speculative adapter loaded but
    not speculating). GEMMA4_RESIDENT_SPEC_FORCE=1 is the older spelling of force."""
    mode = os.environ.get("GEMMA4_RESIDENT_SPEC_POLICY")
    if mode is None:
        mode = "force" if os.environ.get("GEMMA4_RESIDENT_SPEC_FORCE", "0") == "1" else "adaptive"
    if mode not in ("adaptive", "force", "never"):
        raise ValueError(f"GEMMA4_RESIDENT_SPEC_POLICY={mode!r}: expected adaptive, force or never")
    return mode


@dataclass
class ResidentSpecPolicy:
    """spec_policy on the resident rail: one record (1, K) verified at width 8, priced at the anchor position.

    Per request (lead's ruling on M2 item 3, 2026-11-11): every request gets a fresh spec_policy, so no state crosses
    requests -- not the exploration bucket, the acceptance prior, the request-length EWMA or a cool-down. A request's
    decisions, and with them its R = 1 / R = 8 mix and its greedy text (an R = 8 verify and an R = 1 decode can
    differ at a near-tie), are then a function of its own tokens and its own acceptance. Before, the bucket was 0 at
    server start and never reset, so the first request after start declined every step for "budget" (e1e).
    The costs are the fixed priors above (PRIOR_PLAIN_MS, PRIOR_VERIFY_MS, PRIOR_DRAFT_MS 7.67, PRIOR_SEED_MS 11.5,
    SERVED_HOST_MS 0.87). Calibration is deliberately left unfed: no wall-clock measurement reaches a decision.
    A failed uninformed trial schedules a retry trial from a fresh policy (RETRY_AFTER, the second-trial rule)."""

    k: int
    mode: str = field(default_factory=policy_mode)

    def __post_init__(self):
        self.force = self.mode == "force"
        self.config = SpecPolicyConfig.from_env()
        self.retry_after, self.retry_limit, self.retry_growth = RETRY_AFTER, RETRY_LIMIT, RETRY_GROWTH
        self.new_request()

    def new_request(self):
        self._fresh()
        self.steps = 0
        self.retries = 0
        self.retry_in = None  # plain proposals until the scheduled retry trial, or None

    def _fresh(self):
        """A fresh spec_policy: the request's start, and a retry's re-prime."""
        self.policy = SpecPolicy(self.config, self.k)
        self.row = self.policy.new_row()
        self.primed = False
        self.informed = False  # a hold decision has judged this policy's run on the request's own acceptance

    def decide(self, position, entering):
        """Whether this proposal drafts. In hold a False is the exit; in the plain regime a True is an entry
        (enter or trial) and the entry cost is the seed."""
        self.steps += 1
        if self.mode != "adaptive":
            return self.mode == "force"
        if entering and self.retry_in is not None:
            self.retry_in -= 1
            if self.retry_in == 0:
                self.retry_in = None
                self.retries += 1
                self._fresh()
        plain = interpolate(PRIOR_PLAIN_MS, position)
        cost = interpolate(PRIOR_VERIFY_MS, position) + PRIOR_DRAFT_MS
        shape = (1, self.k)
        if not self.primed:
            # The request's first decision: its rung's exploration bucket starts full (BUCKET_TRIALS trials, priced
            # as spec_policy prices this rung's trial), so every request gets its first-proposal trial whatever ran
            # before it. Within the request, exits and retries keep spec_policy's rules.
            cap = BUCKET_TRIALS * self.policy.trial_estimate([RecordCandidate(shape, VERIFY_ROWS, cost, PRIOR_SEED_MS)],
                                                             plain)
            self.policy.rung_cap[VERIFY_ROWS] = self.policy.rung_bucket[VERIFY_ROWS] = cap
            self.primed = True
        measured = self.policy.calibration.tokens_per_row(("step", shape, VERIFY_ROWS))
        candidate = RecordCandidate(shape, VERIFY_ROWS, cost, PRIOR_SEED_MS if entering else 0.0,
                                    tokens_per_row=measured)
        fresh = int(getattr(self.row, "spec_steps", 0)) == 0
        speculate = self.policy.decide([candidate], [self.row], plain, self.k, None, fresh=fresh) is not None
        if entering:
            if speculate:
                self.retry_in = None  # entered by spec_policy's own rules: no retry is pending
        elif speculate:
            self.informed = self.informed or self.policy.last_reason not in ("trial", "retry")
        elif not self.informed and self.retries < self.retry_limit:
            self.retry_in = self.retry_after * self.retry_growth ** self.retries  # a failed uninformed trial
        return speculate

    def observe(self, accepted, verified):
        self.policy.observe(self.row, int(accepted), int(verified))

    def plain(self):
        """A step the policy was not asked about (non-speculable, exhausted)."""
        self.steps += 1
        if self.mode == "adaptive":
            self.policy.note_plain_steps()


class Gemma4ResidentSpecForCausalLM(Gemma4ResidentForCausalLM):
    model_capabilities = {
        **Gemma4ResidentForCausalLM.model_capabilities,
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose"],
        "spec_ids_are_sampled": True,
        "supports_async_spec_decode": False,
        "output_tokens_per_step": 1,
    }
    ROWS = (1, VERIFY_ROWS)
    BLOCK = 8
    K = BLOCK - 1

    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None, tap_layers=()):
        super().__init__(generator, max_batch_size, max_seq_len, vllm_config=vllm_config)
        # The drafter and the tap buffers are built after the KV pools and the engine (allocate_kv_cache_per_layer):
        # the engine places each weight at a fixed stride across layers and asserts it, and ~1.2 GB allocated ahead
        # of it fragments the DRAM heap (e1b, 2026-11-10: AssertionError ('qkv_s', [...]) in resident model.py:230).
        self.drafter = None
        self.tap_layers = tuple(tap_layers)
        self.taps = None
        self.book = resident_taps.TapBook()
        self.spec_policy = ResidentSpecPolicy(self.K)
        self.counters = Counter()
        self.draft_trace = None
        self.draft_captures = []
        self._reset_request()

    def _reset_request(self):
        """Per-request speculative state (a start == 0 prefill, a release: m7, V3-8)."""
        self.book.reset()
        self.offer = None
        self.step = None
        self.seeded = False
        self.regime = "plain"  # "plain" (no drafter state, or left speculation) or "hold"
        self.suppressed = False  # V3-1: a dropped offer suppresses retries for the request's life
        self.exhausted = False  # F4: the drafter's positions are spent
        if self.drafter is not None:
            self.drafter.reset()

    # ------------------------------------------------------------------ construction

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        base = super().initialize_vllm_model(
            hf_config, mesh_device, max_batch_size, max_seq_len=max_seq_len, tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
        )
        return cls(base.generator, max_batch_size, max_seq_len, tap_layers=cls.drafter_tap_layers())

    @classmethod
    def drafter_tap_layers(cls):
        """The drafter's tap layers from its checkpoint config alone (no weights), for the engine build."""
        import json
        from pathlib import Path

        from models.demos.gemma4_31b_qb2.tt.dflash import drafter_checkpoint
        from models.demos.gemma4_31b_qb2.tt.dflash2 import DFlash2Drafter

        checkpoint = os.environ.get("GEMMA4_RESIDENT_DRAFTER_CHECKPOINT")
        path = Path(checkpoint or drafter_checkpoint(DFlash2Drafter.REPO, DFlash2Drafter.REVISION))
        probe = object.__new__(DFlash2Drafter)
        probe._configure(json.loads((path / "config.json").read_text()), cls.BLOCK)
        return tuple(probe.tap_layers)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """M1.1's pools and engine first, then the drafter and the tap buffers (see __init__); all before any trace
        capture (§8.1)."""
        cache = super().allocate_kv_cache_per_layer(per_layer_specs)
        self.drafter = self.build_drafter(self.generator.model, self.max_seq_len)
        if tuple(self.drafter.tap_layers) != self.tap_layers:
            raise ValueError(f"Drafter taps {self.drafter.tap_layers} differ from the engine's {self.tap_layers}")
        self.taps = resident_taps.TapBuffers(self.mesh, len(self.tap_layers))
        self._reset_request()
        return cache

    @classmethod
    def build_drafter(cls, model, max_seq_len):
        """DFlash2 BF8, block 8, RoPE for every position a draft can reach (F4: max_seq_len + block + 32)."""
        kind = os.environ.get("GEMMA4_RESIDENT_DRAFTER", "dflash2")
        if kind != "dflash2":
            raise ValueError(f"GEMMA4_RESIDENT_DRAFTER={kind!r}: only dflash2 is wired (z-lab needs a sampler callback)")
        from models.demos.gemma4_31b_qb2.tt.dflash2 import DFlash2Drafter

        drafter = DFlash2Drafter(
            model,
            os.environ.get("GEMMA4_RESIDENT_DRAFTER_CHECKPOINT"),
            block_size=cls.BLOCK,
            max_positions=max_seq_len + cls.BLOCK + 32,
            weight_dtype=ttnn.bfloat8_b,
        )
        if drafter.block != cls.BLOCK or drafter.SEED_ROWS != resident_taps.WINDOW_ROWS:
            raise ValueError(f"Drafter block {drafter.block} / seed rows {drafter.SEED_ROWS} do not match the design")
        return drafter

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """K = 7 at one user. The R = 8 L1 fit is decided at runtime, not here (F5 (b))."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if max_num_seqs != 1:
            return SpecReject(reason="Resident Gemma4 QB2 speculation serves one request (max_num_seqs=1)")
        if requested_k != cls.K:
            return SpecReject(
                reason=f"DFlash2 block {cls.BLOCK} drafts exactly {cls.K} tokens; num_speculative_tokens={requested_k}",
                supported_k=(cls.K,),
            )
        n, page = 5, resident_taps.TILE * resident_taps.TAP_WIDTH * 2
        history = n * 2 * resident_taps.HISTORY_PAGES * page
        ring = resident_taps.RING_SLOTS * n * page
        windows = n * resident_taps.HISTORY_PAGES * page
        return SpecPlan(
            effective_k=cls.K,
            lanes_per_request=1,
            extra_bytes_per_seq=history + ring + windows,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
            k_by_rows=((1, cls.K),),
        )

    def _build_engine(self):
        """M1's engine with the verify rows and the drafter's tap layers."""
        import importlib

        # resident_m2: M1's engine (resident_s3, frozen) plus per-R l1_top and cb_tops() (resident-analyst, F5).
        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", "resident_m2")
        resident = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}.engine")

        setup = self.generator.model.shared_setup
        cfg = resident.EngineConfig(
            weight_cache=os.environ.get("GRES_CACHE", RAW_WEIGHT_CACHE),
            rows=self.ROWS,
            max_users=self.max_batch_size,
            capacity=self.max_seq_len,
            tap_layers=self.tap_layers,
            l1_top=int(os.environ.get("GRES_L1_TOP", DEFAULT_L1_TOP)),
            logits=True,
            embedding="resident",
        )
        pools = resident.KVPools(
            kv=list(self.cache.kv),
            layer_group=list(self.groups),
            scratch=self._group_scratch(),
            rope={
                kind: (setup[("rope", kind)]["decode_cos"], setup[("rope", kind)]["decode_sin"])
                for kind in ("sliding_attention", "full_attention")
            },
        )
        stub = os.environ.get("GEMMA4_RESIDENT_ENGINE", "resident") == "stub"
        logger.info("Building the resident {} engine, rows {}, taps {}", "stub" if stub else "decode", cfg.rows,
                    cfg.tap_layers)
        return (resident.StubEngine if stub else resident.ResidentEngine)(self.mesh, cfg, pools)

    # ------------------------------------------------------------------ prefill

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, *, sampling_params=None,
                        page_tables_per_layer=None, start_pos=None, empty_slots=None, **kwargs):
        """M1.1's prefill (generator_resident_vllm.prefill_forward, same steps) plus the drafter's tap window
        through the generator pass-through into the history. Kept here, not as a hook in the M1.1 adapter, so the
        two verify separately."""
        if len(prompt_lens) != 1:
            raise ValueError("Speculative serving prefills one request at a time")
        start = 0 if start_pos is None else int(start_pos[0])
        end = int(prompt_lens[0])
        slots = [0] if empty_slots is None else [int(s) for s in empty_slots]
        if start == 0:
            self._reset_request()
            self.spec_policy.new_request()
        tables = self._tables(kv_cache, page_tables_per_layer, slots=slots)
        full_layer = next(i for i, layer in enumerate(self.generator.model.layers) if not layer.sliding)
        keys = [int(tables[full_layer][slot, 0]) for slot in slots]
        self.prefill_slot_keys = {s: k for s, k in self.prefill_slot_keys.items() if s not in slots and k not in keys}
        self.prefill_slot_keys.update(zip(slots, keys))
        novel = (kv_cache.batch, kv_cache.capacity, (end,), (start,), False) not in self.generator.prefill_shapes
        if novel:
            self._release_sampling_trace()
            self._release_draft_trace()  # F3
        window = self.drafter.tap_window(end)
        captured = dict.fromkeys(self.tap_layers)
        outputs = self.generator.prefill_forward(
            tokens,
            page_table=tables,
            kv_cache=kv_cache,
            prompt_lens=[end],
            slots=slots,
            return_device=sampling_params is not None,
            start_positions=[start],
            history_keys=keys,
            taps=[captured],
            tap_windows=[window],
        )
        if novel:
            self._l1_guard(f"prefill {(end,)} from {(start,)}")
        self._record_prefill_taps(captured, window, start, end)
        if sampling_params is None:
            return outputs
        return self._sample_prefill(outputs, sampling_params, [end])

    def _record_prefill_taps(self, captured, window, start, end):
        lo = self.generator.tap_capture_range(start, window)
        self.book.note_prefill(end)
        if lo is not None:
            first_page = resident_taps.page_of(lo[0])
            for page in range(first_page, resident_taps.page_of(lo[1] - 1) + 1):
                self.book.check_write(page)
            self.taps.write_prefill({i: captured[layer] for i, layer in enumerate(self.tap_layers)}, first_page)
        if end % resident_taps.TILE:
            # The page holding `end`: its prefill rows enter the ring so that the page's flush keeps them (§4.3).
            page = resident_taps.page_of(end)
            self.taps.insert_page(page, end % resident_taps.TILE, self.book.insert_page(page, end % resident_taps.TILE))

    # ------------------------------------------------------------------ decode and verify

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, *, spec_mode=None, num_valid_drafts=None,
                       accepted_counts=None, **kwargs):
        if spec_mode is not None:
            return self._verify(tokens, start_pos, page_table, kv_cache, spec_mode, num_valid_drafts, **kwargs)
        dropped, self.offer = self.offer is not None, None
        reload = kwargs.get("reload_inputs", True)
        position = int(start_pos.reshape(-1)[0]) if reload else int(self.positions[0])
        launch = self.engine.launches
        out = super().decode_forward(tokens, start_pos, page_table, kv_cache, **kwargs)
        self.step = Step("ordinary", launch, position, host=kwargs.get("sampling_params") is None, dropped=dropped)
        return out

    def _verify(self, tokens, start_pos, page_table, kv_cache, spec_mode, num_valid_drafts, *, sampling_params=None,
                page_tables_per_layer=None, **kwargs):
        """One `argmax_ids` verify of the [1, 1+K] block (§5.3): n + 1 rows in one launch."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if spec_mode != "argmax_ids":
            raise ValueError(f"Resident speculation serves argmax_ids verifies only, got {spec_mode!r}")
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        n = int(num_valid_drafts.reshape(-1)[0])
        block = [int(t) for t in tokens.reshape(tokens.shape[0], -1)[0, : n + 1]]
        positions = [int(p) for p in start_pos.reshape(start_pos.shape[0], -1)[0, : n + 1]]
        p = positions[0]
        if positions != list(range(p, p + n + 1)):
            raise ValueError(f"Verify positions are not consecutive: {positions}")
        if p + n >= self.max_seq_len:
            raise ValueError(f"Verify block [{p}, {p + n}] passes the cache capacity {self.max_seq_len}")
        if n and p + self.BLOCK > self._spec_limit():
            # An R = 8 launch is checked for all 8 rows (p + 8 against the engine's capacity), whatever n is: never
            # offered (`_propose` step 3), so a verify here is a contract breach, refused before the launch flags it.
            raise ValueError(f"Verify block [{p}, {p + self.BLOCK}) passes the R = 8 limit {self._spec_limit()}")
        offer, self.offer = self.offer, None
        if n:
            if offer is None or offer.anchor != block[0] or offer.position != p or offer.drafts[:n] != block[1:]:
                raise ValueError(f"Verify block {block} at {p} is not the retained proposal {offer}")
        tables = self._tables(kv_cache, page_tables_per_layer)
        self.engine.write_tables(self._resident_tables(tables))
        sampled = sampling_params is not None and any(t > 0 for t in self._rows(sampling_params.temperature, 1))
        if sampled:
            self._configure_verify_sampling(sampling_params, p)
            if self.sampling_trace is None:
                # Before the block's rows are written: a guard rebuild restores M1's feed, which these rows replace.
                self._capture_sampling()
                self._guard_after_capture()
        self.engine.write_rows([0] * (n + 1), positions, block)
        mode = ("none", sampled)
        if mode != self.mode:
            self.engine.set_mode(feed=mode[0], write_logits=mode[1])
            self.mode = mode
        launch = self.engine.launches
        if n and VERIFY_ROWS not in self.ROWS:
            # This step's own sampling-capture guard dropped R = 8: the block runs as n + 1 causal R = 1 launches,
            # each row with the seed an R = 8 row would have used. Once per drop; later steps propose nothing.
            ids = self._verify_sequential(block, positions, sampled)
        else:
            self.engine.launch(VERIFY_ROWS if n else 1)
            if sampled:
                self._replay_sampling()
            ids = self.engine.read_tokens(n + 1)
        self._check_status(ttnn.get_device_tensors(self.engine.status)[0].cpu())
        # The engine's positions are the block's; the next ordinary decode reloads (C:4d), never continues.
        self.feed_stale = True
        out = torch.full((tokens.shape[0], self.K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        out[0, : n + 1] = ids.to(torch.int32)
        self.step = Step("verify", launch, p, drafts=n, dropped=offer is not None and n == 0)
        self.counters["verifies"] += 1
        self.counters["verified_drafts"] += n
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out)

    def _configure_verify_sampling(self, sampling_params, position):
        """Rows 0..7 sampled with the request's k / p / temperature; row j's seed is the plain decode's seed at
        position p + j (§6), so committed draws are the plain draws."""
        rows = VERIFY_ROWS

        def first(name, default):
            value = getattr(sampling_params, name, default)
            value = value.tolist() if isinstance(value, torch.Tensor) else value
            return list(value)[0] if isinstance(value, (list, tuple)) else value

        expanded = _Params({name: [first(name, default)] * rows for name, default in _SAMPLING_FIELDS.items()})
        params = self._sampling(expanded, rows, [position + j for j in range(rows)], reset_seed=True)
        self.generator.configure_sampling(batch=rows, **params)
        self.verify_sampling = params

    def _verify_sequential(self, block, positions, sampled):
        """The verify block one row per R = 1 launch (after a mid-step R = 8 drop): row j writes position p + j and
        reads the argmax, or the draw with row j's seed."""
        ids = []
        for j, (token, position) in enumerate(zip(block, positions)):
            self.engine.write_rows([0], [position], [token])
            if sampled:
                row = {k: [v[j]] for k, v in self.verify_sampling.items() if isinstance(v, list)}
                self.generator.configure_sampling(batch=1, reset_seed=True, **row)
            self.engine.launch(1)
            if sampled:
                self._replay_sampling()
            ids.append(int(self.engine.read_tokens(1)[0]))
        return torch.tensor(ids)

    # ------------------------------------------------------------------ propose

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        k = int(num_drafts)
        if k != self.K:
            raise ValueError(f"Expected {self.K} draft columns, got {k}")
        rows = committed_tokens.shape[0]
        ids = torch.full((rows, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        step, self.step = self.step, None
        if step is None:
            return DraftOutput(draft_token_ids=ids, num_valid=valid)
        c = int(accepted_counts.reshape(-1)[0])
        anchor = int(committed_tokens[0, c - 1])
        position = int(committed_positions[0, c - 1])
        if position != step.position + c:
            raise ValueError(f"Anchor position {position} is not the step's row 0 {step.position} plus {c} committed")
        drafts = self._propose(step, c, anchor, position)
        if drafts:
            ids[0, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
            valid[0] = len(drafts)
            self.offer = Offer(anchor, position, list(drafts))
        return DraftOutput(draft_token_ids=ids, num_valid=valid)

    def _propose(self, step, c, anchor, position):
        """The drafts for the anchor at `position`, or [] to decline (§5.2)."""
        if VERIFY_ROWS not in self.ROWS:
            # R = 8 was dropped (F5): no verify can run, so no history is kept either.
            self.counters["spec_r8_unavailable"] += 1
            return self._decline("plain")
        # 1. The step's committed tap rows, in every regime, so the history stays valid.
        self.taps.stage(self.engine.taps_buffer(step.launch), c)
        self.taps.ring_insert(self.book.insert(step.position, c))
        for page in self.book.complete_pages():
            self._flush(page)
        if step.kind == "verify" and step.drafts and self.regime == "hold":
            self.spec_policy.observe(c - 1, step.drafts)
            self.counters["accepted_drafts"] += c - 1
        self.counters["committed_tokens"] += c
        # 2. Non-speculable (V3-1): a host-sampled step, or a dropped offer. A dropped offer is also what the runner
        # sends when it trims an offer to 0 for room under max_model_len (model_runner.py:1963-1967): harmless, the
        # request is ending, but counted the same way.
        if step.host or step.dropped:
            self.counters["spec_nonspeculable_steps"] += 1
            if step.dropped:
                self.counters["spec_offers_dropped"] += 1
                self.suppressed = True
            return self._decline("plain")
        # 3. Near capacity (F4). A verify launches R = 8 rows from the anchor, and the engine flags F_POSITION when
        # p + 8 passes its capacity even when fewer rows are valid (m1l-verifier, 2026-11-11): the last BLOCK - 1
        # positions are plain R = 1 steps. A request-intrinsic rule (its own position), so per-request determinism holds.
        if position + self.BLOCK > self._spec_limit():
            self.exhausted = True
            return self._decline("plain")
        if self.suppressed or self.exhausted:
            return self._decline("plain")
        # 4. The policy, by regime.
        if self.regime == "hold":
            if not self.spec_policy.decide(position, entering=False):
                return self._decline("plain")  # exit: nothing commits; the next entry reseeds
            self.taps.commit(self.drafter.taps, c)
            return self._draft(anchor, position, step.position, c)
        if not self.spec_policy.decide(position, entering=True):
            return self._decline("plain", asked=True)
        self._enter(position)
        return self._draft(anchor, position, position, 0)

    def _spec_limit(self):
        """The bound on anchor + BLOCK for an R = 8 verify: the engine's capacity (EngineConfig.capacity, the position
        bound its launch checks for every launched row; not max_seq_len) and the drafter's RoPE range."""
        return min(int(self.engine.cfg.capacity), int(self.drafter.max_positions))

    def _decline(self, regime, asked=False):
        self.regime = regime
        if not asked:
            self.spec_policy.plain()
        self.counters["declines"] += 1
        return []

    def _flush(self, page):
        self.book.check_write(page)
        self.taps.flush(page, self.book.entries(page))
        self.book.flushed(page)

    def _enter(self, position):
        """Seed the drafter from the history window of every position < `position` (§5.2 step 4)."""
        page = resident_taps.page_of(position - 1)
        if self.book.entries(page):
            self.book.check_write(page)
            self.taps.flush(page, self.book.entries(page))
            self.book.flushed(page, partial=True)
        first, start, valid = resident_taps.window(position)
        start_s = time.perf_counter()
        windows = self.taps.read_window(first)
        self.drafter.seed({layer: windows[i] for i, layer in enumerate(self.tap_layers)}, start, valid)
        self.counters["seeds"] += 1
        self.counters["seed_ms"] += int((time.perf_counter() - start_s) * 1000)
        self.seeded = True

    def _draft(self, anchor, position, committed_start, committed):
        self.drafter.prepare(anchor, position, committed_start, committed)
        for _ in range(2):  # a capture whose L1 guard rebuilt the engine released itself (V3-6): capture again
            if self.draft_trace is None:
                self._capture_draft()
        if self.draft_trace is None:
            raise RuntimeError("The draft trace was released by an engine rebuild twice in a row")
        ttnn.execute_trace(self.mesh, self.draft_trace, cq_id=0, blocking=False)
        drafts = [int(t) for t in self.drafter.read_drafts(anchor)[: self.K]]
        self.regime = "hold"
        self.counters["drafts"] += 1
        return drafts

    # ------------------------------------------------------------------ draft trace (F3, V3-6)

    def _capture_draft(self):
        """The drafter's device step as a trace: eager once (compile, allocate), then captured. Its outputs are
        allocated by the capture and re-bound by it (device_step assigns them), so read_drafts reads the latest."""
        start = time.perf_counter()
        self.drafter.device_step()
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.drafter.device_step()
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self.draft_trace = trace
        self.draft_captures.append(time.perf_counter() - start)
        logger.info("Draft trace captured in {:.1f} ms (host)", self.draft_captures[-1] * 1000)
        self._l1_guard("draft trace capture")

    def _release_draft_trace(self):
        if self.draft_trace is not None:
            ttnn.synchronize_device(self.mesh)
            ttnn.release_trace(self.mesh, self.draft_trace)
            self.draft_trace = None

    def _cb_tops(self):
        """{R: highest L1 address of R's CBs}; resident_s3 has only the maximum."""
        tops = getattr(self.engine, "cb_tops", None)
        return dict(tops()) if tops is not None else {r: self.engine.cb_top() for r in self.ROWS}

    def _l1_guard(self, reason, before_rebuild=None):
        """No persistent ttnn L1 buffer may sit inside any R's CB range (F5). The R = 8 top is the attention
        leader's, which no l1_top lowers, so a floor below it drops the engine to rows (1,) (observable, never
        silent); R = 1 then gets M1's rebuild at the floor if it still does not fit. A rebuild releases the draft
        trace, which re-captures lazily (V3-6)."""
        if VERIFY_ROWS not in self.ROWS:
            rebuilt = []

            def note():
                rebuilt.append(True)
                if before_rebuild is not None:
                    before_rebuild()

            super()._l1_guard(reason, before_rebuild=note)
            if rebuilt:
                self._release_traces_after_rebuild()
                self.counters["engine_rebuilds"] += 1
            return
        floor, tops = self._l1_floor(), self._cb_tops()
        self.l1_checks.append((reason, floor, tops))
        logger.info("L1 guard after {}: lowest ttnn L1 buffer {} (absolute), engine CB tops {}", reason, floor, tops)
        if floor is None or floor >= max(tops.values()):
            return
        if before_rebuild is not None:
            before_rebuild()
        self._drop_r8(reason, floor, tops)

    def _drop_r8(self, reason, floor, tops):
        """Serve R = 1 only from now on; every later proposal declines. Logged and counted, never silent.

        resident_m2's drop_rows releases R = 8's traces and programs and keeps the shared weights and R = 1: no
        weight loop. An engine without it is freed completely before a new one is constructed: close() alone keeps
        its weights alive, and a new weight loop needs a hole-free heap (fixed per-layer stride, IMPLEMENTATION_NOTES)."""
        logger.warning("Resident speculation unavailable after {}: the ttnn L1 floor {} is below the R = 8 CB top {}; "
                       "serving plain decode (rows (1,))", reason, floor, tops.get(VERIFY_ROWS))
        drop = getattr(self.engine, "drop_rows", None)
        self.ROWS = (1,)
        if drop is not None:
            drop((VERIFY_ROWS,))
        else:
            self.engine.close()
            self.engine = None
            gc.collect()
            self.engine = self._build_engine()
            self.engine.write_tables(self._resident_tables(self._tables(self.cache, list(self.cache.page_tables.values()))))
            self.engine.warmup()
        if self.engine.cb_top() > floor:
            self.engine.rebuild(l1_top=floor)
            if self.engine.cb_top() > floor:
                raise RuntimeError(f"Resident engine cannot fit below the ttnn L1 floor {floor} even at R = 1")
        self.mode, self.feed_stale = None, True
        self._release_traces_after_rebuild()
        self.counters["spec_r8_dropped"] += 1
        self.counters["engine_rebuilds"] += 1

    def _release_traces_after_rebuild(self):
        """An engine rebuild allocates after both traces were captured, and a replay can overwrite what was
        allocated after its capture (M1's trace hazard). Both are released and re-captured lazily: the draft trace at
        the next draft, the sampling trace before the next sampled launch (_guard_after_capture covers a rebuild
        triggered by the sampling capture itself)."""
        self._release_sampling_trace()
        self._release_draft_trace()

    def _guard_after_capture(self):
        """M1.1's guard after the lazy sampling capture; when that guard rebuilt the engine (and so released the
        trace just captured), capture again, after the rebuild's allocations, before this step's launch."""
        for _ in range(2):
            super()._guard_after_capture()
            if self.sampling_trace is not None:
                return
            self._capture_sampling()
        raise RuntimeError("The sampling trace was released by an engine rebuild twice in a row")

    # ------------------------------------------------------------------ warmup and lifecycle

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        """M1.1's prefill warmups, then the drafter's tap-capture and seed programs (dflash_warmup.warmup_tap_seed:
        every window row class and seed variant, ~1 s each on a cold kernel cache) at startup rather than on the
        first prompts, before any trace capture (§8.1, VERIFY_SEED_FIX.md). It resets the drafter ring."""
        first = self.embedding_warm_s is None
        super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        if first and self.drafter is not None:
            from models.demos.gemma4_31b_qb2.tt.dflash_warmup import warmup_tap_seed

            start = time.perf_counter()
            warmup_tap_seed(self.generator.model, self.drafter)
            self.seed_warm_s = time.perf_counter() - start
            logger.info("Drafter tap-capture and seed programs built in {:.1f} s", self.seed_warm_s)
            self._reset_request()

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        if self.prefill_tokens is not None:
            return
        if self.taps is not None:
            start = time.perf_counter()
            self.taps.warm()
            logger.info("Tap ring and history programs (with {} fill classes) built in {:.1f} s",
                        resident_taps.HISTORY_PAGES, time.perf_counter() - start)
        super().warmup_model_decode(kv_cache, enable_trace, **kwargs)
        if self.drafter is not None:
            self.drafter.reset()
            self.drafter.prepare(0, 0, 0, 0)
            self._capture_draft()
            self.drafter.reset()
        if self.taps is not None:
            # Traces exist from here on: the tap buffers allocate nothing more (TapBuffers.seal), and the DRAM each
            # chip has left is logged with what the resident selection matrices took (measured in TapBuffers.__init__).
            self.taps.seal()
            free, stats = resident_taps.dram_free_per_chip(self.mesh), self.taps.dram_stats
            taken = stats["free_before_plain_sel"] - stats["free_after_plain_sel"]
            logger.info("DRAM per chip after warmup: {} B free; the resident selection matrices took {} B "
                        "(free without them: {} B)", free, taken, free + taken)
        if os.environ.get("GEMMA4_RESIDENT_TEST_DROP_R8") == "1" and VERIFY_ROWS in self.ROWS:
            # Device test hook (M2 forced-drop case): the guard's R = 8 drop on the real engine, as if the floor had
            # fallen below the R = 8 top; the server then serves plain decode.
            tops = self._cb_tops()
            self._drop_r8("test hook GEMMA4_RESIDENT_TEST_DROP_R8", tops[VERIFY_ROWS] - 1, tops)

    def release_request(self, slot):
        super().release_request(slot)
        if self.counters:
            logger.info("Resident speculation, request done: {}", dict(self.counters))
        self.spec_policy.new_request()
        self._reset_request()

    def release_persistent_capture(self):
        self._release_draft_trace()
        super().release_persistent_capture()


_SAMPLING_FIELDS = {  # the request's fields M1's _sampling reads, with their neutral values
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": None,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "repetition_penalty": 1,
    "enable_log_probs": False,
}


class _Params:
    def __init__(self, values):
        self.__dict__.update(values)
