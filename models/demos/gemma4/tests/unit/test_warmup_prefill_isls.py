# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: GEMMA4_WARMUP_PREFILL_ISLS selection and ladder orchestration."""


from models.demos.gemma4.tt.warmup_isls import ENV_VAR, parse_warmup_isls, run_prefill_ladder, warmup_prefill_isls


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
    assert warmup_prefill_isls(262144, already_warmed=[32, 128, 512], env={}) == []


def test_ladder_is_capped_by_context():
    env = {ENV_VAR: "4096,8192,16384,32768,65536"}
    assert warmup_prefill_isls(16384, already_warmed=[], env=env) == [4096, 8192, 16384]
    assert warmup_prefill_isls(None, already_warmed=[], env=env) == [4096, 8192, 16384, 32768, 65536]


def test_already_warmed_is_a_floor_not_set_membership():
    # Everything at or below the longest already-warmed length counts as covered,
    # including lengths that were never explicitly warmed (512 here).
    env = {ENV_VAR: "128,512,4096,8192"}
    assert warmup_prefill_isls(262144, already_warmed=[32, 4096], env=env) == [8192]
    assert warmup_prefill_isls(262144, already_warmed=[32, 128], env=env) == [512, 4096, 8192]


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
