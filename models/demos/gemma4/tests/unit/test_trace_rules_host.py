# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the decode-trace allocation guard: a device allocation that appears after the
decode trace was captured (and is still alive at the next replay) makes the generator release and
re-capture its decode traces, a bounded number of times."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.demos.gemma4.tt import generator_trace as gt
from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin
from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin as Mixin
from models.demos.gemma4.tt.generator import filter_warmup_sampling_params
from models.demos.gemma4.tt.warmup_isls import ENV_VAR, parse_warmup_isls, run_prefill_ladder, warmup_prefill_isls


class _View:
    """Allocator view with a bottom-up run of ``allocated`` bytes, one free gap, and a top-down block."""

    def __init__(self, allocated, top_down=4096, total=1 << 20):
        self.total_bytes_allocated_per_bank = allocated + top_down
        self.block_table = [
            {"address": "0", "size": str(allocated), "allocated": "yes"},
            {"address": str(allocated), "size": str(total - allocated - top_down), "allocated": "no"},
            {"address": str(total - top_down), "size": str(top_down), "allocated": "yes"},
        ]


def _guard(monkeypatch, levels):
    """A bare mixin instance whose allocator view reports ``levels[buffer_type]`` bytes per bank."""
    monkeypatch.setattr(ttnn, "get_memory_view", lambda mesh, buffer_type: _View(levels[buffer_type]))
    g = ChunkedPrefillPageTableGuardMixin.__new__(ChunkedPrefillPageTableGuardMixin)
    g.model_args = [SimpleNamespace(mesh_device=object())]
    g.model = [SimpleNamespace(sampling=SimpleNamespace(reset_trace=lambda: None))]
    g.trace_ids_decode = defaultdict(lambda: None)
    g.trace_inputs_decode = defaultdict(lambda: None)
    g.trace_output_decode = defaultdict(lambda: None)
    g._g4_trace_alloc_baseline = {}
    g._g4_trace_guard_recaptures = 0
    g._g4_alloc_check_pending = True
    return g


def test_no_growth_no_recapture(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    assert not g._g4_trace_guard_should_recapture("k")  # nothing captured yet
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    assert not g._g4_trace_guard_should_recapture("k")
    levels[ttnn.BufferType.DRAM] = 900  # something freed: still fine, and the baseline follows it down
    g._g4_alloc_check_pending = True
    assert not g._g4_trace_guard_should_recapture("k")
    assert g._g4_trace_alloc_baseline["k"][0] == 900
    levels[ttnn.BufferType.DRAM] = 950  # above the lowest level seen: a post-capture allocation
    g._g4_alloc_check_pending = True
    assert g._g4_trace_guard_should_recapture("k")


@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_growth_triggers_recapture_until_the_limit(monkeypatch, buffer_type):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MAX", "2")
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    levels[buffer_type] += 64
    g._g4_alloc_check_pending = True
    assert g._g4_trace_guard_should_recapture("k")
    g._g4_alloc_check_pending = True
    assert g._g4_trace_guard_should_recapture("k")
    g._g4_alloc_check_pending = True
    assert not g._g4_trace_guard_should_recapture("k")  # limit spent: logged, no more re-captures
    assert g._g4_trace_guard_recaptures == 2  # only the performed re-captures count, never the refused one


def test_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD", "0")
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    levels[ttnn.BufferType.DRAM] += 64
    assert not g._g4_trace_guard_should_recapture("k")


def test_top_down_growth_and_replays_only_do_not_recapture(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    # kernel binaries: top-down, never inside the replay's bottom-up scratch
    monkeypatch.setattr(
        ttnn,
        "get_memory_view",
        lambda mesh, bt: _View(levels[bt], top_down=65536 if bt == ttnn.BufferType.DRAM else 4096),
    )
    g._g4_alloc_check_pending = True
    assert not g._g4_trace_guard_should_recapture("k")
    # bottom-up growth without a prefill in between is not checked (replays allocate nothing persistent)
    levels[ttnn.BufferType.DRAM] += 64
    g._g4_alloc_check_pending = False
    assert not g._g4_trace_guard_should_recapture("k")


def test_warmup_captured_trace_gets_its_baseline_at_the_first_replay(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    assert not g._g4_trace_guard_should_recapture("w")  # no baseline yet: recorded, no re-capture
    assert g._g4_trace_alloc_baseline["w"] == g._g4_alloc_levels()


def test_release_drops_every_trace_and_resets_sampling(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    released, resets = [], []
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: released.append(tid))
    g.model[0].sampling.reset_trace = lambda: resets.append(1)
    g.trace_ids_decode["a"] = {0: 11}
    g.trace_ids_decode["b"] = {0: 12}
    g._prev_decode_batch = 128  # the feedback buffer is found through this key: it must survive the release
    g.trace_output_decode["a"] = [SimpleNamespace()]  # dropped, never freed by the release
    g._g4_trace_alloc_baseline["a"] = (1, 2)
    g._g4_release_decode_traces("test")
    assert sorted(released) == [11, 12]
    assert resets == [1]
    assert not g.trace_ids_decode["a"] and not g.trace_output_decode["a"]
    assert g._g4_trace_alloc_baseline == {}
    assert g._g4_recapture_without_precompile  # the live request's sampler state must not be touched
    assert g._prev_decode_batch == 128


# ===== merged from tests/unit/test_trace_guard_arming_host.py: Host-only: when the decode-trace allocation guard walks the allocator (``ChunkedPrefillPageTableGuardMixin``).
class _Guard(Mixin):
    """The mixin over a stub generator: counted allocator walks, no device."""

    def __init__(self, levels=(100, 10)):
        self.model = [SimpleNamespace(_g4_persistent_alloc_event=None)]
        self.trace_id_prefill = {}
        self._g4_trace_alloc_baseline = {}
        self._g4_trace_guard_recaptures = 0
        self.levels = tuple(levels)
        self.walks = 0

    def _g4_alloc_levels(self):
        self.walks += 1
        return self.levels


def _replays(g, n):
    return [g._g4_trace_guard_should_recapture("k") for _ in range(n)]


def test_prefill_mode_arms_on_every_prefill(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", "prefill")
    g = _Guard()
    assert _replays(g, 1) == [False] and g.walks == 1  # the first replay takes the baseline
    assert _replays(g, 3) == [False] * 3 and g.walks == 2  # armed by default on the first replay after capture
    assert _replays(g, 3) == [False] * 3 and g.walks == 2  # replays alone never walk
    assert g._g4_arm_alloc_check("prefill") is True
    assert _replays(g, 2) == [False] * 2 and g.walks == 3
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={}, chunked=False, multimodal=False) == []


def test_alloc_mode_arms_on_events_and_periodically(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", "alloc")
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_PERIOD", "4")
    g = _Guard()
    _replays(g, 2)  # baseline + the initial pending check
    assert g.walks == 2
    assert g._g4_arm_alloc_check("prefill") is False  # a plain prefill does not arm
    assert _replays(g, 3) == [False] * 3 and g.walks == 2
    assert _replays(g, 1) == [False] and g.walks == 3  # the 4th replay since the last check: periodic
    assert g._g4_alloc_check_reason == "periodic"
    assert _replays(g, 3) == [False] * 3 and g.walks == 3
    # events arm in both modes
    g.model[0]._g4_persistent_alloc_event = "grown"
    reasons = g._g4_arm_after_prefill(prefill_trace_ids_before={}, chunked=False, multimodal=False)
    assert reasons == ["page tables grown"] and g.model[0]._g4_persistent_alloc_event is None
    assert _replays(g, 1) == [False] and g.walks == 4
    g.trace_id_prefill = {(128, 1): 7}
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={}, chunked=False, multimodal=False) == [
        "prefill trace capture"
    ]
    assert _replays(g, 1) == [False] and g.walks == 5
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={(128, 1): 7}, chunked=True, multimodal=True) == [
        "chunked prefill",
        "multimodal prefill",
    ]
    assert _replays(g, 1) == [False] and g.walks == 6
    assert g._g4_arm_after_prefill(prefill_trace_ids_before={(128, 1): 7}, chunked=False, multimodal=False) == []
    assert _replays(g, 1) == [False] and g.walks == 6


def test_alloc_mode_period_zero_is_events_only(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", "alloc")
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_PERIOD", "0")
    g = _Guard()
    _replays(g, 2)
    assert _replays(g, 200) == [False] * 200 and g.walks == 2


def test_decision_is_the_same_in_both_modes(monkeypatch):
    for mode in ("prefill", "alloc"):
        monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", mode)
        monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MAX", "8")
        g = _Guard(levels=(100, 10))
        _replays(g, 2)
        g.levels = (150, 10)  # a persistent allocation grew the bottom-up DRAM level
        g._g4_arm_alloc_check("page tables grown")
        assert _replays(g, 1) == [True], mode  # re-capture
        assert g._g4_trace_guard_recaptures == 1
        g._g4_trace_alloc_baseline["k"] = g.levels  # what the re-capture does
        g.levels = (90, 10)  # lower than the baseline: tracked downwards, no re-capture
        g._g4_arm_alloc_check("page tables grown")
        assert _replays(g, 1) == [False] and g._g4_trace_alloc_baseline["k"] == (90, 10)


def test_default_mode_is_alloc(monkeypatch):
    monkeypatch.delenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", raising=False)
    g = _Guard()
    assert g._g4_guard_arm_mode() == "alloc"
    assert g._g4_arm_alloc_check("prefill") is False  # a plain prefill no longer arms the walk


def test_unknown_mode_is_the_default(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_ARM", "sometimes")
    assert _Guard()._g4_guard_arm_mode() == "alloc"


# ===== merged from tests/unit/test_trace_short_prefill_host.py: Host-only checks of the short-prefill trace opt-in (GEMMA4_TRACE_SHORT_PREFILL, tt/generator_trace.py): the
@pytest.fixture()
def _clean_env(monkeypatch):
    monkeypatch.delenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, raising=False)
    monkeypatch.delenv("GEMMA4_TRACE_SHORT_PREFILL_MAX", raising=False)


@pytest.mark.usefixtures("_clean_env")
def test_opt_in_off_by_default_off_the_column_mesh():
    # no mesh shape (demo / host paths) or a row mesh: off unless GEMMA4_TRACE_SHORT_PREFILL=1
    assert not gt.trace_short_prefill_opt_in()
    assert not gt.trace_short_prefill_opt_in(128, 1)
    assert not gt.trace_short_prefill_opt_in(128, 1, mesh_shape=(1, 4))
    assert not gt.trace_short_prefill_opt_in(128, 1, mesh_shape=(8, 4))


@pytest.mark.usefixtures("_clean_env")
def test_default_on_for_the_blackhole_column_mesh(monkeypatch):
    from models.common.utility_functions import is_blackhole

    on = gt.trace_short_prefill_default((4, 1))
    assert on == bool(is_blackhole())  # the validated 4-chip column mesh, Blackhole only
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "0")
    assert not gt.trace_short_prefill_default((4, 1))
    assert not gt.trace_short_prefill_opt_in(128, 1, mesh_shape=(4, 1))
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "1")
    assert gt.trace_short_prefill_default((1, 4))


@pytest.mark.usefixtures("_clean_env")
def test_opt_in_short_single_user_only(monkeypatch):
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "1")
    assert gt.trace_short_prefill_opt_in()
    assert gt.trace_short_prefill_opt_in(100, 1)  # pads to the 128 bucket
    assert gt.trace_short_prefill_opt_in(128, 1)
    assert not gt.trace_short_prefill_opt_in(129, 1)  # pads to the 1024 bucket (get_padded_prefill_len)
    assert not gt.trace_short_prefill_opt_in(128, 2)  # single user only
    monkeypatch.setenv("GEMMA4_TRACE_SHORT_PREFILL_MAX", "1024")
    assert gt.trace_short_prefill_opt_in(1000, 1)
    assert not gt.trace_short_prefill_opt_in(1025, 1)


def _resolve(enable_trace, seq_len, batch_size=1, can_batch=False, bounded=True):
    model = SimpleNamespace(bounded_sliding_kv_cache=bounded, hidden_size_per_layer_input=0)
    model_args = SimpleNamespace(max_batch_size=32, hidden_size_per_layer_input=0)
    return gt.resolve_gemma4_prefill_trace_enable(
        enable_trace,
        model,
        model_args,
        batch_size=batch_size,
        prefill_seq_lens=[seq_len],
        can_batch_prefill=can_batch,
        empty_slots=None,
    )


@pytest.mark.usefixtures("_clean_env")
def test_bounded_policy_with_and_without_opt_in(monkeypatch):
    assert not _resolve(True, 128)  # bounded: eager by default
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "1")
    assert _resolve(True, 128)  # the 128 bucket is a trace bucket
    assert not _resolve(True, 2048)  # longer prompts stay eager under bounded sliding
    assert not _resolve(True, 128, batch_size=2, can_batch=True)  # batched prefills stay eager
    assert not _resolve(False, 128)  # caller disabled
    assert _resolve(True, 128, bounded=False)  # unbounded: the usual policy


@pytest.mark.usefixtures("_clean_env")
def test_short_prefill_stash_skipped_only_in_the_traced_bounded_short_bucket(monkeypatch):
    # the traced short single-user bucket under bounded sliding records no sliding-tail stash ...
    assert gt.short_prefill_stash_skipped(128, 1, bounded=True, trace_mode=True)
    assert gt.short_prefill_stash_skipped(64, 1, bounded=True, trace_mode=True)
    # ... every eager call keeps it (continuations are eager), as do unbounded, batched and long traced buckets
    assert not gt.short_prefill_stash_skipped(128, 1, bounded=True, trace_mode=False)
    assert not gt.short_prefill_stash_skipped(128, 1, bounded=False, trace_mode=True)
    assert not gt.short_prefill_stash_skipped(128, 2, bounded=True, trace_mode=True)
    assert not gt.short_prefill_stash_skipped(1024, 1, bounded=True, trace_mode=True)
    monkeypatch.setenv("GEMMA4_TRACE_SHORT_PREFILL_MAX", "1024")
    assert gt.short_prefill_stash_skipped(1024, 1, bounded=True, trace_mode=True)


@pytest.mark.usefixtures("_clean_env")
def test_batched_prefill_trace_buckets_default_follows_the_mesh(monkeypatch):
    from models.common.utility_functions import is_blackhole

    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_ENV, raising=False)
    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_BUCKETS_ENV, raising=False)
    # no mesh shape / a row mesh / the Galaxy: off unless GEMMA4_TRACE_BATCHED_PREFILL=1
    assert gt.batched_prefill_trace_buckets() == set()
    assert gt.batched_prefill_trace_buckets((1, 4)) == set()
    assert gt.batched_prefill_trace_buckets((8, 4)) == set()
    assert not gt.batched_prefill_trace_bucket(31, list(range(31)), 32, 128)
    # the validated Blackhole 4-chip column mesh: the 32 x 128 bucket
    assert gt.batched_prefill_trace_buckets((4, 1)) == ({(32, 128)} if is_blackhole() else set())
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_ENV, "0")
    assert gt.batched_prefill_trace_buckets((4, 1)) == set()


@pytest.mark.usefixtures("_clean_env")
def test_batched_prefill_trace_buckets_opt_in(monkeypatch):
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_ENV, "1")
    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_BUCKETS_ENV, raising=False)
    assert gt.batched_prefill_trace_buckets() == {(32, 128)}  # the QB2 32-user ramp bucket
    # a 31-user cold call at 128 tokens pads to 32 (batched_prefill_padded_batch) -> replays the bucket
    assert gt.batched_prefill_trace_bucket(31, list(range(31)), 32, 128)
    assert gt.batched_prefill_trace_bucket(32, None, 32, 128)
    # slots spanning the top of the batch pad to 32 as well, but a thin call (below the 0.4 fill = 13 of 32 users)
    # takes the sequential single-user replays: 4 x ~59 ms beats the ~590 ms 32-row replay (shield 40142241727)
    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_MIN_FILL_ENV, raising=False)
    assert gt.batched_prefill_bucket_min_users(32) == 13
    assert gt.batched_prefill_bucket_min_users(8) == 4
    assert gt.batched_prefill_bucket_min_users(2) == 2
    assert gt.batched_prefill_trace_bucket(13, [0] + list(range(20, 32)), 32, 128)
    assert not gt.batched_prefill_trace_bucket(12, list(range(20, 32)), 32, 128)
    assert not gt.batched_prefill_trace_bucket(4, [28, 29, 30, 31], 32, 128)
    assert not gt.batched_prefill_trace_bucket(2, [0, 31], 32, 128)
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_MIN_FILL_ENV, "0")  # fill 0: every padded call replays
    assert gt.batched_prefill_trace_bucket(2, [0, 31], 32, 128)
    assert gt.batched_prefill_trace_bucket(4, [28, 29, 30, 31], 32, 128)
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_MIN_FILL_ENV, "0.5")
    assert gt.batched_prefill_bucket_min_users(32) == 16
    assert not gt.batched_prefill_trace_bucket(15, list(range(17, 32)), 32, 128)
    assert gt.batched_prefill_trace_bucket(16, list(range(16, 32)), 32, 128)
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_MIN_FILL_ENV, "junk")  # unparsable: the default
    assert gt.batched_prefill_bucket_min_users(32) == 13
    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_MIN_FILL_ENV, raising=False)
    # other shapes keep the sequential single-user replays
    assert not gt.batched_prefill_trace_bucket(2, [0, 1], 32, 128)
    assert not gt.batched_prefill_trace_bucket(31, list(range(31)), 32, 1024)
    assert not gt.batched_prefill_trace_bucket(1, [0], 32, 128)
    assert not gt.batched_prefill_trace_bucket(31, list(range(31)), 32, None)
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_BUCKETS_ENV, "32x128, 16x512,8x4096,3x128,32x100")
    # 8 x 4096 = 32768 virtual tokens is a legal bucket; 3 is not a padded batch, 100 is not a trace ISL
    assert gt.batched_prefill_trace_buckets() == {(32, 128), (16, 512), (8, 4096)}


# ===== merged from tests/unit/test_warmup_prefill_isls.py: Host-only: GEMMA4_WARMUP_PREFILL_ISLS selection and ladder orchestration.
def test_parse_is_tolerant_sorted_and_unique():
    assert parse_warmup_isls("8192, 4096,abc,,4096,-1,0,32768") == [4096, 8192, 32768]
    assert parse_warmup_isls(None) == [] and parse_warmup_isls("") == []


def test_parse_warns_on_invalid_entries():
    # A typo must not disappear silently: each bad entry is named in a warning.
    from loguru import logger

    messages = []
    handle = logger.add(lambda m: messages.append(str(m)), level="WARNING", format="{message}")
    try:
        assert parse_warmup_isls("4096,8l92,-5") == [4096]
    finally:
        logger.remove(handle)
    joined = "\n".join(messages)
    assert "8l92" in joined and "-5" in joined and ENV_VAR in joined


def test_unset_env_warms_nothing_extra():
    assert warmup_prefill_isls(262144, env={}) == []


def test_ladder_is_capped_by_context_and_nothing_is_inferred_as_covered():
    env = {ENV_VAR: "128,512,4096,8192,16384,32768,65536"}
    assert warmup_prefill_isls(16384, env=env) == [128, 512, 4096, 8192, 16384]
    assert warmup_prefill_isls(None, env=env) == [128, 512, 4096, 8192, 16384, 32768, 65536]


class _FakeGenerator:
    def __init__(self, data_parallel=1, paged=True):
        self.data_parallel = data_parallel
        self.paged = paged
        self.calls = []

    def _mock_tokens(self, batch_size, seq_len, kv_cache, model_id):
        return {"tokens": (batch_size, seq_len), "page_table": object() if self.paged else None, "model_id": model_id}


def _prefill_recorder(gen):
    def prefill_forward(**kwargs):
        gen.calls.append(
            (kwargs["model_id_warmup"], kwargs["tokens"][1], kwargs["enable_trace"], kwargs["warmup_prefill"])
        )

    return prefill_forward


def test_ladder_runs_once_per_process_and_per_model():
    gen = _FakeGenerator(data_parallel=2)
    warmed = run_prefill_ladder(
        gen, kv_cache="kv", prefill_forward=_prefill_recorder(gen), ladder=[4096, 8192], chunk=4096
    )
    assert warmed == [4096, 8192]
    assert gen.calls == [
        (0, 4096, False, False),
        (0, 8192, False, False),
        (1, 4096, False, False),
        (1, 8192, False, False),
    ]
    # second (capture) pass: nothing runs again
    assert (
        run_prefill_ladder(gen, kv_cache="kv", prefill_forward=_prefill_recorder(gen), ladder=[4096, 8192], chunk=4096)
        == []
    )
    assert len(gen.calls) == 4


def test_ladder_stops_at_the_chunk_when_paged_attention_is_off():
    gen = _FakeGenerator(paged=False)
    warmed = run_prefill_ladder(
        gen, kv_cache=None, prefill_forward=_prefill_recorder(gen), ladder=[2048, 4096, 8192, 16384], chunk=4096
    )
    assert warmed == [2048, 4096]
    assert [c[1] for c in gen.calls] == [2048, 4096]


# ===== merged from tests/unit/test_warmup_sampling_filter.py: The decode-warmup sampling sweep drops the request shapes a mesh's sampler opts out of
B = 4


PEN = SamplingParams(
    temperature=[1.0] * B,
    top_k=[10] * B,
    top_p=[0.9] * B,
    presence_penalty=[1.2] * B,
    frequency_penalty=[1.2] * B,
    repetition_penalty=[1.5] * B,
)


PLAIN = SamplingParams(
    temperature=[1.0] * B,
    top_k=[10] * B,
    top_p=[0.9] * B,
    presence_penalty=None,
    frequency_penalty=None,
    repetition_penalty=None,
)


GREEDY = SamplingParams(temperature=[0.0] * B, top_k=[1] * B, top_p=[1.0] * B)


SWEEP = [PEN, PLAIN, GREEDY, None]


def test_full_sampler_keeps_the_whole_sweep():
    assert filter_warmup_sampling_params(SWEEP, SimpleNamespace()) == SWEEP
    assert filter_warmup_sampling_params(SWEEP, None) == SWEEP


def test_no_penalties_drops_only_the_penalty_configs():
    sampler = SimpleNamespace(_allow_penalties_sampling=False, _allow_topk_sampling=True)
    assert filter_warmup_sampling_params(SWEEP, sampler) == [PLAIN, GREEDY, None]


def test_no_topk_keeps_only_greedy_and_none():
    sampler = SimpleNamespace(_allow_penalties_sampling=False, _allow_topk_sampling=False)
    assert filter_warmup_sampling_params(SWEEP, sampler) == [GREEDY, None]
