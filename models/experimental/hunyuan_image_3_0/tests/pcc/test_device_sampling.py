# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device temp / top-p / multinomial via ``ttnn.topk`` + ``ttnn.sampling``.
#
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_device_sampling.py -v -s

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.experimental.hunyuan_image_3_0.tt.device_sampling import (
    can_use_device_sampling,
    sample_logits_ttnn,
    sampling_padded_vocab,
    _sampling_shortlist_width,
)
from models.experimental.hunyuan_image_3_0.ref.generate import SamplingConfig

HUNYUAN_VOCAB = 133120


def test_host_top_k_uses_instruct_1024():
    from models.experimental.hunyuan_image_3_0.tt.device_sampling import _host_top_k

    assert _host_top_k(1024, HUNYUAN_VOCAB) == 1024
    assert _host_top_k(32, HUNYUAN_VOCAB) == 32
    assert _host_top_k(0, HUNYUAN_VOCAB) == HUNYUAN_VOCAB
    assert _host_top_k(200000, HUNYUAN_VOCAB) == HUNYUAN_VOCAB


def test_device_sampling_defaults_on(monkeypatch):
    from models.experimental.hunyuan_image_3_0.tt.device_sampling import (
        device_sampling_enabled,
        ttnn_sampling_op_enabled,
    )

    monkeypatch.delenv("HY_DEVICE_SAMPLING", raising=False)
    monkeypatch.delenv("HY_SAMPLE_DEVICE", raising=False)
    monkeypatch.delenv("HY_TTNN_SAMPLING_OP", raising=False)
    monkeypatch.delenv("HY_TOP_K", raising=False)
    monkeypatch.delenv("HY_TOPK", raising=False)
    assert device_sampling_enabled() is True
    assert ttnn_sampling_op_enabled() is False


def test_ttnn_sampling_op_when_hy_top_k_32(monkeypatch):
    from models.experimental.hunyuan_image_3_0.tt.device_sampling import ttnn_sampling_op_enabled

    monkeypatch.delenv("HY_TTNN_SAMPLING_OP", raising=False)
    monkeypatch.setenv("HY_TOP_K", "32")
    assert ttnn_sampling_op_enabled() is True
    monkeypatch.setenv("HY_TOP_K", "1024")
    assert ttnn_sampling_op_enabled() is False
    monkeypatch.delenv("HY_TOP_K", raising=False)
    monkeypatch.setenv("HY_TOPK", "32")
    assert ttnn_sampling_op_enabled() is True


def test_can_use_device_sampling_allows_instruct_topk(monkeypatch):
    """Device-logits path stays eligible; host shortlist uses Instruct top_k."""
    monkeypatch.delenv("HY_DEVICE_SAMPLING", raising=False)
    monkeypatch.delenv("HY_SAMPLE_DEVICE", raising=False)
    assert can_use_device_sampling(SamplingConfig(do_sample=True, top_k=32, repetition_penalty=1.0))
    assert can_use_device_sampling(SamplingConfig(do_sample=True, top_k=1024, repetition_penalty=1.0))
    assert can_use_device_sampling(SamplingConfig(do_sample=True, top_k=0, repetition_penalty=1.0))
    monkeypatch.setenv("HY_DEVICE_SAMPLING", "0")
    assert can_use_device_sampling(SamplingConfig(do_sample=True, top_k=32, repetition_penalty=1.0)) is False


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def test_sampling_padded_vocab_hunyuan():
    assert sampling_padded_vocab(HUNYUAN_VOCAB) == 262144
    assert sampling_padded_vocab(HUNYUAN_VOCAB) % 32 == 0
    wt = sampling_padded_vocab(HUNYUAN_VOCAB) // 32
    assert wt > 0 and (wt & (wt - 1)) == 0


def test_sampling_shortlist_width_avoids_wt1():
    """BH hangs on W=32 (Wt=1); shortlist pad floor is 64."""
    assert _sampling_shortlist_width(32) == 64
    assert _sampling_shortlist_width(64) == 64
    assert _sampling_shortlist_width(128) == 128
    assert _sampling_shortlist_width(96) == 128


@pytest.mark.parametrize("vocab_size", [128, 2048, HUNYUAN_VOCAB])
def test_sample_logits_ttnn_top1_matches_argmax(device, vocab_size, monkeypatch):
    """With k=1 / p=0 the op is argmax-equivalent (deterministic)."""
    monkeypatch.setenv("HY_TTNN_SAMPLING_OP", "1")
    torch.manual_seed(0)
    B, V = 1, vocab_size
    logits = torch.randn(B, 1, V, dtype=torch.bfloat16) * 0.01
    peak = min(12345, V - 1)
    logits[0, 0, peak] = 50.0

    logits_tt = ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ids = sample_logits_ttnn(
        logits_tt,
        device,
        vocab_size=V,
        batch_size=B,
        temperature=1.0,
        top_p=0.0,
        top_k=1,
        seed=42,
        deallocate_input=True,
        return_device_ids=False,
    )
    assert ids.shape == (B,)
    assert int(ids[0].item()) == peak


def test_sample_logits_ttnn_seed_deterministic(device, monkeypatch):
    """Same seed → same token for a fixed logits vector (temp/top-p path)."""
    monkeypatch.setenv("HY_TTNN_SAMPLING_OP", "1")
    torch.manual_seed(1)
    B, V = 1, 2048
    logits = torch.randn(B, 1, V, dtype=torch.bfloat16)

    def _once(seed: int):
        logits_tt = ttnn.from_torch(
            logits,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return sample_logits_ttnn(
            logits_tt,
            device,
            vocab_size=V,
            batch_size=B,
            temperature=0.8,
            top_p=0.95,
            top_k=32,
            seed=seed,
            deallocate_input=True,
            return_device_ids=False,
        )

    a = _once(2024)
    b = _once(2024)
    c = _once(2025)
    assert torch.equal(a, b)
    if torch.equal(a, c):
        pytest.skip("rare seed collision on multinomial draw")


def test_sample_logits_ttnn_return_device_ids(device, monkeypatch):
    """TTNN op path keeps ids on device for embed (no mandatory D2H)."""
    monkeypatch.setenv("HY_TTNN_SAMPLING_OP", "1")
    torch.manual_seed(0)
    B, V = 1, 128
    logits = torch.randn(B, 1, V, dtype=torch.bfloat16) * 0.01
    peak = 11
    logits[0, 0, peak] = 50.0
    logits_tt = ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    token_tt = sample_logits_ttnn(
        logits_tt,
        device,
        vocab_size=V,
        batch_size=B,
        temperature=1.0,
        top_p=0.0,
        top_k=1,
        seed=1,
        deallocate_input=True,
        return_device_ids=True,
    )
    assert not isinstance(token_tt, torch.Tensor)
    assert list(token_tt.shape)[-1] == 1
    host = ttnn.to_torch(token_tt).view(-1)[:B].long()
    assert int(host[0].item()) == peak
    ttnn.deallocate(token_tt)


def test_device_bookkeeping_append_stop_materialize(device):
    """On-device concat + stop-flag + one-shot materialize (no per-step torch.cat)."""
    from models.experimental.hunyuan_image_3_0.tt.device_sampling import (
        append_token_ids_tt,
        materialize_generated_ids,
        token_hits_stop_tt,
        upload_stop_ids_tt,
    )

    B = 1
    stop_id = 99
    t0 = ttnn.from_torch(
        torch.tensor([[7]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    t1 = ttnn.from_torch(
        torch.tensor([[stop_id]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gen = append_token_ids_tt(None, t0)
    gen = append_token_ids_tt(gen, t1)
    stops = upload_stop_ids_tt(device, [stop_id], B)
    assert token_hits_stop_tt(t1, stops, device, B) == [True]
    assert token_hits_stop_tt(t0, stops, device, B) == [False]
    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
    sequences, new_tokens = materialize_generated_ids(gen, device, B, prefix_ids=prefix, stop_set={stop_id})
    assert new_tokens == [[7, stop_id]]
    assert sequences.tolist() == [[1, 2, 3, 7, stop_id]]
    for t in (t0, t1, gen, stops):
        ttnn.deallocate(t)
