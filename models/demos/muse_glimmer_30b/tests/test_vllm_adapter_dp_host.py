# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only test of the data-parallel vLLM adapter: drives its prefill/decode entries against a fake generator."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.demos.muse_glimmer_30b.tt.generator_vllm_dp import MuseGlimmerDPForCausalLM, _as_list

V = 11


class FakeGenerator:
    """Records calls; mirrors the MuseGlimmerGenerator surface the adapter uses (global slot u -> replica u // B)."""

    def __init__(self, dp=4, B=32, max_seq_len=4096):
        self.dp, self.B = dp, B
        self.global_batch_size = dp * B
        self.models = [SimpleNamespace(args=SimpleNamespace(max_seq_len=max_seq_len, max_top_k=32)) for _ in range(dp)]
        self.prefills = []
        self.decodes = []
        self.reads = []
        self.page_table_dirty = False
        self.traces_released = 0

    @staticmethod
    def _to_sampling_params(sampling, max_top_k, seed=None):
        return dict(sampling or {}, max_top_k=max_top_k)

    def prefill_slot(self, slot, token_ids, sampling_params=None):
        self.prefills.append((slot, slot // self.B, slot % self.B, list(token_ids), sampling_params))
        if sampling_params is None:
            return torch.arange(V, dtype=torch.float32) + slot
        return 500 + slot

    def decode_issue(self, tokens=None, positions=None, sampling=None, sampling_params=None, reload=True, copy_page_table=None):
        self.decodes.append((tokens, positions, sampling, sampling_params, reload, copy_page_table))
        return {"sampling": sampling is not None}

    def decode_read(self, handle, async_read=False):
        self.reads.append(async_read)
        return ("payload", handle), (["ev0", "ev1"] if async_read else None)

    def decode_finish(self, handle, payload):
        assert payload[0] == "payload"
        if handle["sampling"]:
            return 1000 + torch.arange(self.global_batch_size, dtype=torch.int64)
        return torch.arange(self.global_batch_size, dtype=torch.float32).reshape(-1, 1).repeat(1, V)

    def release_traces(self):
        self.traces_released += 1


def make(dp=4, B=32):
    gen = FakeGenerator(dp, B)
    return MuseGlimmerDPForCausalLM._from_generator(gen, kv_handles=["kv"]), gen


def tt_sp(temperature, top_k, top_p):
    return SimpleNamespace(temperature=temperature, top_k=top_k, top_p=top_p)


def prefill(ad, lens, slots, **kw):
    n = len(lens)
    tokens = torch.zeros(n, max(lens), dtype=torch.int64)
    for r, ln in enumerate(lens):
        tokens[r, :ln] = torch.arange(1, 1 + ln) + 1000 * r
    return ad.prefill_forward(
        tokens, None, None, prompt_lens=list(lens), start_pos=[0] * n, empty_slots=slots, **kw
    )


def decode_inputs(rows=128, active=(0, 33, 127)):
    t = torch.zeros(rows, 1, dtype=torch.int32)
    p = torch.full((rows,), -1, dtype=torch.int32)
    for i in active:
        t[i, 0] = 7 + i
        p[i] = 20 + i
    return t, p


def test_capabilities():
    caps = MuseGlimmerDPForCausalLM.model_capabilities
    assert caps["supports_sample_on_device"] is True and caps["max_device_top_k"] == 32
    assert caps["supports_async_decode"] is True
    for k in ("supports_device_penalties", "supports_chunked_prefill", "supports_prefix_caching"):
        assert caps[k] is False
    assert "supports_spec_decode" not in caps
    assert MuseGlimmerDPForCausalLM.decode_input_update_contract == 1
    assert MuseGlimmerDPForCausalLM._HYBRID_KV_CACHE_GROUPS_ENABLED is False
    assert MuseGlimmerDPForCausalLM.get_max_tokens_all_users(max_model_len=8192, max_num_seqs=128) == 128 * 8192


def test_slot_routing_and_prefill_logits():
    ad, gen = make()
    out = prefill(ad, [5, 7], [33, 0])
    assert out.shape == (2, 1, V) and out.dtype == torch.float32
    slot, replica, local, ids, dsp = gen.prefills[0]
    assert (slot, replica, local, dsp) == (33, 1, 1, None) and ids == [1, 2, 3, 4, 5]
    assert gen.prefills[1][:3] == (0, 0, 0) and len(gen.prefills[1][3]) == 7
    assert out[0, 0, 0] == 33.0
    assert ad.allocate_kv_cache() == ["kv"] and ad.allocate_kv_cache_per_layer({}) == ["kv"]


def test_prefill_sampled_returns_tokens():
    ad, gen = make()
    out = prefill(ad, [4], [127], sampling_params=tt_sp([0.7], [20], [0.9]))
    toks, extra = out
    assert extra is None and toks.shape == (1, 1) and toks.dtype == torch.int64 and toks[0, 0] == 627
    assert gen.prefills[0][4]["top_k"] == 20
    assert gen.prefills[0][:3] == (127, 3, 31)


def test_prefill_rejections():
    ad, _ = make()
    tokens = torch.ones(1, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="chunked"):
        ad.prefill_forward(tokens, None, None, prompt_lens=[4], start_pos=[16], empty_slots=[0])
    with pytest.raises(ValueError, match="empty_slots"):
        ad.prefill_forward(tokens, None, None, prompt_lens=[4], start_pos=[0])
    with pytest.raises(ValueError, match="context"):
        ad.prefill_forward(torch.ones(1, 5000, dtype=torch.int64), None, None, prompt_lens=[5000], empty_slots=[0])


def test_prefill_numpy_plugin_args():
    ad, gen = make()
    tokens = torch.arange(1, 22, dtype=torch.int64).reshape(3, 7)
    ad.prefill_forward(
        tokens,
        None,
        None,
        prompt_lens=np.array([5, 7, 6], dtype=np.int64),
        start_pos=np.zeros(3, dtype=np.int32),
        empty_slots=[0, 33, 70],
    )
    assert [(p[0], p[1], p[2]) for p in gen.prefills] == [(0, 0, 0), (33, 1, 1), (70, 2, 6)]
    for r, ln in enumerate([5, 7, 6]):
        assert gen.prefills[r][3] == tokens[r, :ln].tolist()
    with pytest.raises(ValueError, match="chunked"):
        ad.prefill_forward(
            tokens,
            None,
            None,
            prompt_lens=np.array([5, 7, 6], dtype=np.int64),
            start_pos=np.array([0, 16, 0], dtype=np.int32),
            empty_slots=[0, 33, 70],
        )


def test_as_list():
    assert _as_list(None) == [None]
    assert _as_list([1, 2]) == [1, 2] and _as_list((1, 2)) == [1, 2]
    assert _as_list(torch.tensor([[1, 2], [3, 4]])) == [1, 2, 3, 4]
    assert _as_list(np.array([5, 7], dtype=np.int64)) == [5, 7]
    assert _as_list(np.int32(3)) == [3]
    assert _as_list(4) == [4]


GREEDY = dict(temperature=[0.0] * 128, top_k=[1] * 128, top_p=[1.0] * 128)


def test_decode_logits_passthrough():
    ad, gen = make()
    t, p = decode_inputs()
    out, extra = ad.decode_forward(t, None, None, p)
    assert extra is None and out.shape == (128, 1, V) and out.dtype == torch.float32
    assert out[33, 0, 0] == 33.0 and out[1].abs().sum() == 0
    toks, pos, sampling, load, reload, _ = gen.decodes[0]
    assert sampling is None and reload is True
    assert pos[33] == 53 and pos[1] == -1 and toks[33] == 40 and toks[1] == 0
    with pytest.raises(ValueError, match="requires reload_inputs"):
        ad.decode_forward(t, None, None, p, reload_inputs=False)


def test_decode_sampled_pass_through():
    ad, gen = make()
    t, p = decode_inputs()
    ids, extra = ad.decode_forward(t, None, None, p, sampling_params=tt_sp(**GREEDY))
    assert extra is None and ids.shape == (128, 1) and ids.dtype == torch.int64
    assert ids[33, 0] == 1033 and ids[1, 0] == 0 and ids[127, 0] == 1127
    toks, pos, sampling, load, reload, _ = gen.decodes[0]
    assert sampling is not None and load is sampling and reload is True  # first step loads the set
    # no reload flag: cached params, nothing reloaded
    samp = tt_sp([0.8] * 128, [20] * 128, [0.9] * 128)
    ad.decode_forward(t, None, None, p, sampling_params=samp, reload_sampling_params=True)
    assert gen.decodes[1][2]["top_k"] == 20 and gen.decodes[1][3] is not None
    ad.decode_forward(t, None, None, p, sampling_params=tt_sp([0.1] * 128, [5] * 128, [0.5] * 128))
    assert gen.decodes[2][2]["top_k"] == 20 and gen.decodes[2][3] is None  # not re-evaluated without a command
    ad.decode_forward(t, None, None, p, sampling_params=samp, reset_sampling_state=True)
    assert gen.decodes[3][3] is not None
    # a prefill invalidates the loaded set
    prefill(ad, [3], [5])
    ad.decode_forward(t, None, None, p, sampling_params=samp)
    assert gen.decodes[4][3] is not None


def test_decode_no_reload_issues_without_inputs():
    ad, gen = make()
    t, p = decode_inputs()
    ad.decode_forward(t, None, None, p, sampling_params=tt_sp(**GREEDY))
    ids, _ = ad.decode_forward(t, None, None, p, sampling_params=tt_sp(**GREEDY), reload_inputs=False)
    toks, pos, sampling, load, reload, copy_pt = gen.decodes[1]
    assert toks is None and pos is None and reload is False and load is None and copy_pt is False
    assert sampling is not None and ids[33, 0] == 1033 and ids[1, 0] == 0
    # page-table-only refresh, and prefill-dirtied page table
    ad.decode_forward(t, None, None, p, sampling_params=tt_sp(**GREEDY), reload_inputs=False, reload_page_table=True)
    assert gen.decodes[2][4] is False and gen.decodes[2][5] is True
    gen.page_table_dirty = True
    ad.decode_forward(t, None, None, p, sampling_params=tt_sp(**GREEDY), reload_inputs=False)
    assert gen.decodes[3][5] is True
    # unread submission, then explicit read
    handle = ad.decode_forward(t, None, None, p, sampling_params=tt_sp(**GREEDY), reload_inputs=False, read_from_device=False)
    assert gen.reads == [False] * 4 and handle[1] is True
    (res, events) = ad.read_decode_output(handle, async_read=True)
    assert gen.reads[-1] is True and events == ["ev0", "ev1"] and res[1] is True
    out, extra = ad.process_decode_output_host(res, is_tokens=True)
    assert extra is None and out.shape == (128, 1) and out[127, 0] == 1127 and out[2, 0] == 0
    # process also accepts an unread submission (reads blocking)
    out2, _ = ad.process_decode_output_host(handle, is_tokens=True)
    assert torch.equal(out, out2) and gen.reads[-1] is False
    # sample_rows is accepted
    ad.read_decode_output(handle, sample_rows=[0, 33])


def test_decode_flag_validation():
    ad, gen = make()
    t, p = decode_inputs()
    sp = tt_sp(**GREEDY)
    with pytest.raises(ValueError, match="Invalid v1"):
        ad.decode_forward(t, None, None, p, sampling_params=sp, reload_inputs=True, reload_page_table=True)
    with pytest.raises(ValueError, match="Invalid v1"):
        ad.decode_forward(t, None, None, p, sampling_params=sp, reload_inputs=False, reset_sampling_state=True)
    with pytest.raises(ValueError, match="slot remap"):
        ad.decode_forward(t, None, None, p, sampling_params=sp, reload_inputs=False, slot_remap=list(range(128)))
    assert gen.decodes == []


def test_decode_output_phys_permutation_and_gaps():
    ad, gen = make(dp=1, B=4)
    t = torch.tensor([[11], [22], [33], [44]], dtype=torch.int32)
    p = torch.tensor([5, 6, -1, -1], dtype=torch.int32)
    sp = tt_sp([0.0] * 4, [1] * 4, [1.0] * 4)
    out, _ = ad.decode_forward(t, None, None, p, sampling_params=sp, slot_remap=torch.tensor([1, 0, 2, 3]))
    assert out.shape == (4, 1) and out[:, 0].tolist() == [1001, 1000, 0, 0]  # row i reads generator slot phys[i]
    # the permutation persists for resident steps
    out, _ = ad.decode_forward(t, None, None, p, sampling_params=sp, reload_inputs=False)
    assert out[:, 0].tolist() == [1001, 1000, 0, 0]


def test_uniform_sampling_check():
    ad, gen = make()
    t, p = decode_inputs(active=(0, 33))
    temp = [0.5] * 128
    temp[33] = 0.9
    with pytest.raises(NotImplementedError, match="uniform set"):
        ad.decode_forward(t, None, None, p, sampling_params=tt_sp(temp, [20] * 128, [1.0] * 128))
    # rows that are inactive (padding) may differ
    temp = [0.5] * 128
    temp[100] = 0.9
    ad.decode_forward(t, None, None, p, sampling_params=tt_sp(temp, [20] * 128, [1.0] * 128))
    # greedy rows agree whatever their top_p
    ad.decode_forward(
        t, None, None, p, sampling_params=tt_sp([0.0, 0.7] + [0.0] * 126, [20, 1] + [20] * 126, [0.5, 1.0] + [1.0] * 126)
    )


def test_slot_remap_moves_state_map_only():
    ad, gen = make(dp=1, B=4)
    t = torch.tensor([[11], [22], [33], [44]], dtype=torch.int32)
    p = torch.tensor([5, 6, -1, -1], dtype=torch.int32)
    sp = tt_sp([0.0] * 4, [1] * 4, [1.0] * 4)
    # plugin slot 0 now names the state of slot 1 (a swap)
    ad.decode_forward(t, None, None, p, sampling_params=sp, slot_remap=torch.tensor([1, 0, 2, 3]))
    toks, pos, *_ = gen.decodes[0]
    assert toks.tolist() == [22, 11, 0, 0] and pos.tolist() == [6, 5, -1, -1]
    with pytest.raises(RuntimeError, match="permutation"):
        ad.decode_forward(t, None, None, p, sampling_params=sp, slot_remap=[0, 0, 1, 2])


def test_decode_shape_check():
    ad, _ = make()
    with pytest.raises(ValueError, match="128 rows"):
        ad.decode_forward(torch.zeros(4, 1, dtype=torch.int32), None, None, torch.zeros(4, dtype=torch.int32))


def test_initialize_validation():
    cls = MuseGlimmerDPForCausalLM
    mesh = SimpleNamespace(get_num_devices=lambda: 32, shape=(8, 4))
    with pytest.raises(ValueError, match="divisible"):
        cls.initialize_vllm_model(None, mesh, 130, max_seq_len=8192, tt_data_parallel=4)
    with pytest.raises(ValueError, match="per-replica batch"):
        cls.initialize_vllm_model(None, mesh, 132, max_seq_len=8192, tt_data_parallel=4)
    with pytest.raises(ValueError, match="per-replica batch"):
        cls.initialize_vllm_model(None, mesh, 0, max_seq_len=8192, tt_data_parallel=4)
    with pytest.raises(ValueError, match="max_seq_len"):
        cls.initialize_vllm_model(None, mesh, 128, tt_data_parallel=4)
    with pytest.raises(ValueError, match="TP"):
        cls.initialize_vllm_model(
            None, SimpleNamespace(get_num_devices=lambda: 20, shape=(5, 4)), 64, max_seq_len=8192, tt_data_parallel=2
        )
    with pytest.raises(ValueError, match="131072"):
        cls.initialize_vllm_model(None, mesh, 128, max_seq_len=131073, tt_data_parallel=4)


def test_served_seq_len_cap(monkeypatch):
    from models.demos.muse_glimmer_30b.tt.generator_vllm_dp import _served_seq_len

    assert _served_seq_len(1000) == 1024
    monkeypatch.setenv("MUSE_VLLM_MAX_SEQ_LEN", "4000")
    assert _served_seq_len(131072) == 4096
    assert _served_seq_len(1000) == 1024


def test_resolve_dp_env_wins():
    from models.demos.muse_glimmer_30b.tt.generator_vllm_dp import _resolve_dp

    assert _resolve_dp(128, 1, {"TT_DATA_PARALLEL": "4"}) == (4, 128, 32)
    assert _resolve_dp(128, 4, {}) == (4, 128, 32)
    assert _resolve_dp(32, 1, {"TT_DATA_PARALLEL": "0"}) == (1, 32, 32)
    with pytest.raises(ValueError, match="per-replica batch"):
        _resolve_dp(128, 1, {})


def test_resolve_dp_env_wins():
    from models.demos.muse_glimmer_30b.tt.generator_vllm_dp import _resolve_dp

    assert _resolve_dp(128, 1, {"TT_DATA_PARALLEL": "4"}) == (4, 128, 32)
    assert _resolve_dp(128, 4, {}) == (4, 128, 32)
    assert _resolve_dp(32, 1, {"TT_DATA_PARALLEL": "0"}) == (1, 32, 32)
    with pytest.raises(ValueError, match="per-replica batch"):
        _resolve_dp(128, 1, {})
