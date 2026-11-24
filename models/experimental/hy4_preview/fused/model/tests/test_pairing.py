# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The pairing of the RoPE dimensions as ONE switch (HY4_ROPE_PAIRING; reference/attention.py rope_pairing): host only.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>

For both pairings: the device's computation (a vector in the device frame, the interleaved cos / sin tables, the
quarter-turn matrix frames.rope_swap: what tt/ and the fused kernels do) equals the reference's apply_rope carried
into the device frame; q . k is the reference's; the default is "interleaved" (the checkpoint's pairing and the
prefill stack's); under "interleaved" no row of a projection moves and the pairs are (2 i, 2 i + 1) of the
checkpoint's order.
"""

import pytest
import torch

from models.experimental.hy4_preview.reference import attention as RA
from models.experimental.hy4_preview.tt import frames


def test_default(monkeypatch):
    """Without HY4_ROPE_PAIRING the process runs the checkpoint's pairing; the variable still selects the other."""
    monkeypatch.delenv(RA.ROPE_PAIRING_ENV, raising=False)
    assert RA.rope_pairing() == "interleaved"
    monkeypatch.setenv(RA.ROPE_PAIRING_ENV, "half_split")
    assert RA.rope_pairing() == "half_split"
    assert RA.rope_pairing("interleaved") == "interleaved"


def test_golden_of_the_other_pairing_is_refused(monkeypatch, tmp_path, expect_error):
    """Every reader of a golden file (hf_harness.load_golden) refuses one recorded under the other pairing."""
    from models.experimental.hy4_preview.reference import hf_harness

    files = {}
    for pairing in RA.ROPE_PAIRINGS:
        files[pairing] = tmp_path / f"{pairing}.pt"
        torch.save({"meta": {"rope_pairing": pairing}, "input_ids": torch.arange(4)}, files[pairing])
    old = tmp_path / "old.pt"  # a file from before the field existed: HF as it is, half-split
    torch.save({"meta": {}, "input_ids": torch.arange(4)}, old)
    for pairing, other in (RA.ROPE_PAIRINGS, RA.ROPE_PAIRINGS[::-1]):
        monkeypatch.setenv(RA.ROPE_PAIRING_ENV, pairing)
        assert hf_harness.load_golden(files[pairing])["meta"]["rope_pairing"] == pairing
        with expect_error(AssertionError, "was recorded with RoPE pairing"):
            hf_harness.load_golden(files[other])
    monkeypatch.delenv(RA.ROPE_PAIRING_ENV, raising=False)
    with expect_error(AssertionError, "was recorded with RoPE pairing"):
        hf_harness.load_golden(old)


def test_prefill_bridge_refuses_half_split(monkeypatch, expect_error):
    """The prefill stack writes interleaved rows only: a half-split decode side next to it is refused before
    anything is built (tt/prefill_bridge.py PrefillBridge.build)."""
    from models.experimental.hy4_preview.tt import prefill_bridge as PB

    class Mesh:
        shape = PB.MESH

    monkeypatch.setenv(RA.ROPE_PAIRING_ENV, "half_split")
    with expect_error(AssertionError, "the prefill stack writes interleaved rows"):
        PB.PrefillBridge(Mesh()).build()


@pytest.mark.parametrize("pairing", RA.ROPE_PAIRINGS)
def test_pairing(pairing, monkeypatch):
    monkeypatch.setenv(RA.ROPE_PAIRING_ENV, pairing)
    cfg = RA.AttentionConfig()
    d = cfg.qk_rope_head_dim
    g = torch.Generator().manual_seed(5)
    positions = torch.tensor([0, 1, 7, 90, 1022, 5118, 100_000, 999_999])
    q, k = torch.randn(8, d, generator=g, dtype=torch.float64), torch.randn(8, d, generator=g, dtype=torch.float64)
    cos, sin = (t.double() for t in RA.rope_cos_sin(positions, cfg))
    ref_q, ref_k = RA.apply_rope(q, cos, sin), RA.apply_rope(k, cos, sin)
    # the device: its frame, its tables, its quarter turn
    cos_d, sin_d = frames.rope_tables(cos, sin)
    swap = frames.rope_swap(d).double()
    dev = lambda x: frames.rope_to_device(x) * cos_d + (frames.rope_to_device(x) @ swap) * sin_d
    assert torch.allclose(dev(q), frames.rope_to_device(ref_q), atol=1e-12)
    assert torch.allclose((dev(q) * dev(k)).sum(-1), (ref_q * ref_k).sum(-1), atol=1e-10)
    assert torch.equal(frames.rope_from_device(frames.rope_to_device(q)), q)
    # every frequency sits on both dims of an interleaved pair of the device's tables
    assert torch.equal(cos_d[:, 0::2], cos_d[:, 1::2]) and torch.equal(sin_d[:, 0::2], sin_d[:, 1::2])
    perm = frames.interleave_perm(d)
    if pairing == "half_split":
        assert perm.tolist() == [v for i in range(d // 2) for v in (i, i + d // 2)]
        assert torch.equal(ref_q, q * cos + RA.rotate_half(q) * sin)
    else:
        assert perm.tolist() == list(range(d))
        # pair (2 i, 2 i + 1) of the checkpoint's own order turns by the angle of frequency i
        i, row = 3, 4
        angle = positions[row].double() * (1.0 / (cfg.rope_theta ** (2 * i / d)))
        x0, x1 = q[row, 2 * i], q[row, 2 * i + 1]
        want = torch.stack((x0 * angle.cos() - x1 * angle.sin(), x1 * angle.cos() + x0 * angle.sin()))
        assert torch.allclose(ref_q[row, 2 * i : 2 * i + 2], want, atol=1e-5)  # (the tables are made in fp32)
        # no row of a projection moves
        w = {
            "kv_a_proj_with_mqa": torch.randn(cfg.kv_lora_rank + d, 16, generator=g),
            "q_b_proj": torch.randn(4 * cfg.qk_head_dim, 16, generator=g),
        }
        out = frames.attention_weights(w, cfg)
        assert all(torch.equal(out[name], w[name]) for name in w)
    # the two pairings are different functions of the same vector
    other = [p for p in RA.ROPE_PAIRINGS if p != pairing][0]
    co, so = (t.double() for t in RA.rope_cos_sin(positions, cfg, other))
    assert not torch.allclose(RA.apply_rope(q, co, so, other), ref_q, atol=1e-3)
