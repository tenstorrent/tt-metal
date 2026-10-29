# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only: the per-grid tuned tables change only the grid they are keyed for.

Matmuls (tt/linear.py `TUNED_CONFIGS`, E9.3): every (role, batch, seq) call that `linear` /
`gated_product` issues is recorded with the table on and off (`TUNED_ON`). On the P150a's 13 x 10
grid the two must be identical; on 12 x 10 exactly the (role, tokens) entries listed in the table
differ, at batch 1 only. No device: the ttnn ops are replaced by recorders and the device is a stub
that reports the grid.

SDPA (tt/attention.py `SDPA_TUNED`, E9.4): the same check on `sdpa_program_config` per (head_dim,
batch, bucket) with `SDPA_TUNED_ON` on and off. Each check has a negative control (a planted 13 x 10
entry must be detected).

    pytest models/demos/embedding_gemma2/tests/perf/test_tuned_config_identity.py
"""

from __future__ import annotations

import pytest

import ttnn
from models.demos.embedding_gemma2.tt import attention as A
from models.demos.embedding_gemma2.tt import linear as L
from models.demos.embedding_gemma2.tt.config import GRID_ENV

P150 = (13, 10)
GALAXY = (12, 10)
# role -> (K, N, gated): the text roles of tt/attention.py, tt/mlp.py, tt/per_layer_input.py, tt/text_encoder.py
ROLES = {
    "qkv_sliding": (512, 2048, False),
    "qkv_full": (512, 3072, False),
    "o_sliding": (1024, 512, False),
    "o_full": (2048, 512, False),
    "up": (512, 2048, False),
    "gate": (512, 2048, True),
    "down": (2048, 512, False),
    "ple_wide": (512, 12288, False),
    "ple_layer": (512, 512, False),
    "ple_gate": (512, 512, True),
    "ple_proj": (512, 512, False),
    "final": (512, 768, False),
}
CONTRACT_SEQS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
# contract buckets (batch 1) + the batched shapes of the test suite
SHAPES = [(1, s) for s in CONTRACT_SEQS] + [(8, 512), (32, 128), (4, 2048), (3, 2048)]


class _Dev:
    def __init__(self, grid):
        self.grid = grid

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(*self.grid)


class _T:
    def __init__(self, shape, dev):
        self.shape = tuple(shape)
        self._dev = dev

    def device(self):
        return self._dev


@pytest.fixture
def recorder(monkeypatch):
    """Replace the ttnn ops `tt/linear.py` issues by recorders; returns the call list."""
    monkeypatch.delenv(GRID_ENV, raising=False)
    calls = []

    def rec_linear(x, w, **kw):
        calls.append(
            ("linear", x.shape, repr(kw.get("program_config")), repr(kw.get("memory_config")), repr(kw.get("dtype")))
        )
        return _T(x.shape[:-1] + (w.shape[-1],), x._dev)

    def rec_mm(x, w, **kw):
        calls.append(
            (
                "minimal_matmul",
                x.shape,
                repr(kw.get("config")),
                repr(kw.get("fused_activation")),
                repr(kw.get("memory_config")),
            )
        )
        return _T(x.shape[:-1] + (w.shape[-1],), x._dev)

    def rec_mul(a, b, **kw):
        calls.append(("multiply", repr(kw.get("input_tensor_a_activations")), repr(kw.get("memory_config"))))
        return _T(a.shape, a._dev)

    monkeypatch.setattr(ttnn, "linear", rec_linear)
    monkeypatch.setattr(ttnn.experimental, "minimal_matmul", rec_mm)
    monkeypatch.setattr(ttnn, "multiply", rec_mul)
    monkeypatch.setattr(ttnn, "reshape", lambda x, shape, *a, **k: _T(shape, x._dev))
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    return calls


def matmul_calls(calls, monkeypatch, grid, tuned_on: bool) -> dict[str, list]:
    monkeypatch.setattr(L, "TUNED_ON", tuned_on)
    dev = _Dev(grid)
    out = {}
    for B, S in SHAPES:
        for role, (K, N, gated) in ROLES.items():
            calls.clear()
            x, w = _T((B, 1, S, K), dev), _T((1, 1, K, N), dev)
            if gated:
                L.gated_product(x, w, _T((B, 1, S, N), dev), memory_config=ttnn.DRAM_MEMORY_CONFIG, role=role)
            else:
                L.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, role=role)
            out[f"{role}@b{B}_s{S}"] = list(calls)
    return out


def differing(calls, monkeypatch, grid) -> set[str]:
    on = matmul_calls(calls, monkeypatch, grid, True)
    off = matmul_calls(calls, monkeypatch, grid, False)
    return {k for k in on if on[k] != off[k]}


def listed(grid) -> set[str]:
    return {f"{role}@b1_s{m}" for role, m in L.TUNED_CONFIGS.get(grid, {})}


def test_matmul_13x10_identical_to_tuned_off(recorder, monkeypatch):
    assert not L.MM_OVERRIDES, "MM_OVERRIDES must be empty outside the in-model search"
    diff = differing(recorder, monkeypatch, P150)
    assert not diff, f"13 x 10 matmul calls change with the tuned table on: {sorted(diff)}"


def test_matmul_12x10_differs_only_in_listed_entries(recorder, monkeypatch):
    diff = differing(recorder, monkeypatch, GALAXY)
    want = listed(GALAXY)
    assert diff == want, f"unlisted entries differ: {sorted(diff - want)}; listed but identical: {sorted(want - diff)}"


def test_identity_check_detects_a_planted_13x10_entry(recorder, monkeypatch):
    """Negative control: the 13 x 10 check is not vacuous."""
    planted = {**L.TUNED_CONFIGS, P150: {("up", 2048): L.MMSpec("2d", in0_block_w=16)}}
    monkeypatch.setattr(L, "TUNED_CONFIGS", planted)
    assert differing(recorder, monkeypatch, P150) == {"up@b1_s2048"}


# ------------------------------------------------------------------------------------------- SDPA
# tt/attention.py `SDPA_TUNED` (E9.4): (head_dim, bucket) -> (q_chunk, k_chunk, grid) per grid, batch 1.
# head_dim 256 = sliding layers, 512 = full layers; the batched shapes are those of the test suite.
SDPA_SHAPES = [(hd, 1, s) for hd in (256, 512) for s in CONTRACT_SEQS] + [
    (hd, b, s) for hd in (256, 512) for b, s in ((8, 512), (32, 128), (4, 2048), (3, 2048))
]


def sdpa_configs(monkeypatch, grid, tuned_on: bool) -> dict[str, tuple]:
    monkeypatch.setattr(A, "SDPA_TUNED_ON", tuned_on)
    out = {}
    for hd, b, s in SDPA_SHAPES:
        pc = A.sdpa_program_config(hd, s, b, 4, ttnn.CoreCoord(*grid))
        g = pc.compute_with_storage_grid_size
        out[f"hd{hd}@b{b}_s{s}"] = (g.x, g.y, pc.q_chunk_size, pc.k_chunk_size, pc.exp_approx_mode)
    return out


def sdpa_differing(monkeypatch, grid) -> set[str]:
    on, off = sdpa_configs(monkeypatch, grid, True), sdpa_configs(monkeypatch, grid, False)
    return {k for k in on if on[k] != off[k]}


def test_sdpa_13x10_identical_to_tuned_off(monkeypatch):
    assert not A.SDPA_OVERRIDES, "SDPA_OVERRIDES must be empty outside the in-model search"
    diff = sdpa_differing(monkeypatch, P150)
    assert not diff, f"13 x 10 SDPA configs change with the tuned table on: {sorted(diff)}"


def test_sdpa_12x10_differs_only_in_listed_entries(monkeypatch):
    diff = sdpa_differing(monkeypatch, GALAXY)
    want = {f"hd{hd}@b1_s{s}" for hd, s in A.SDPA_TUNED.get(GALAXY, {})}
    assert diff == want, f"unlisted entries differ: {sorted(diff - want)}; listed but identical: {sorted(want - diff)}"


def test_sdpa_identity_check_detects_a_planted_13x10_entry(monkeypatch):
    monkeypatch.setattr(A, "SDPA_TUNED", {**A.SDPA_TUNED, P150: {(512, 2048): (128, 128, None)}})
    assert sdpa_differing(monkeypatch, P150) == {"hd512@b1_s2048"}


def test_sliding_map_policy_13x10_unchanged(monkeypatch):
    """The sliding block map buckets and the padding-aware map (tt/masks.py) follow the tuned switch and
    leave 13 x 10 at the default (BLOCK_MAP_MIN_SEQ, no padding-aware map)."""
    from models.demos.embedding_gemma2.tt.masks import BLOCK_MAP_MIN_SEQ

    for on in (True, False):
        monkeypatch.setattr(A, "SDPA_TUNED_ON", on)
        assert A.sliding_map_policy(*P150, BLOCK_MAP_MIN_SEQ) == (BLOCK_MAP_MIN_SEQ, False)
    monkeypatch.setattr(A, "SDPA_TUNED_ON", False)
    assert A.sliding_map_policy(*GALAXY, BLOCK_MAP_MIN_SEQ) == (BLOCK_MAP_MIN_SEQ, False)


# ------------------------------------------------------------------------------- tower attention
# E10.2 (doc/prefill_towers/02-attention/): the vision SDPA table (tt/vision/attention.py `VISION_SDPA_TUNED`,
# switch `VISION_SDPA_TUNED_ON`) and the audio rel-shift choice (tt/audio/attention.py `REL_SHIFT_MM_TUNED`,
# switch `AUDIO_ATTN_TUNED_ON`): the same on / off identity per grid, batch-1 contract shapes plus the batched
# shapes of the tower test suites.
VISION_SDPA_SHAPES = [(1, 2560), (1, 1280), (4, 2560), (8, 1280)]
AUDIO_ATTN_SHAPES = [(1, 256), (1, 768), (4, 256), (1, 1536)]


def vision_sdpa_configs(monkeypatch, grid, tuned_on: bool) -> dict[str, tuple]:
    from models.demos.embedding_gemma2.tt.vision import attention as VA

    monkeypatch.setattr(VA, "VISION_SDPA_TUNED_ON", tuned_on)
    out = {}
    for b, s in VISION_SDPA_SHAPES:
        pc = VA.vision_sdpa_program_config(s, b, ttnn.CoreCoord(*grid))
        g = pc.compute_with_storage_grid_size
        out[f"b{b}_s{s}"] = (g.x, g.y, pc.q_chunk_size, pc.k_chunk_size, pc.max_cores_per_head_batch)
    return out


def vision_sdpa_differing(monkeypatch, grid) -> set[str]:
    on, off = vision_sdpa_configs(monkeypatch, grid, True), vision_sdpa_configs(monkeypatch, grid, False)
    return {k for k in on if on[k] != off[k]}


def test_vision_sdpa_13x10_identical_to_tuned_off(monkeypatch):
    from models.demos.embedding_gemma2.tt.vision import attention as VA

    assert not VA.VISION_SDPA_OVERRIDES and not VA.WINDOWED_OVERRIDES, "search overrides must be empty"
    diff = vision_sdpa_differing(monkeypatch, P150)
    assert not diff, f"13 x 10 vision SDPA configs change with the tuned table on: {sorted(diff)}"


def test_vision_sdpa_12x10_differs_only_in_listed_entries(monkeypatch):
    from models.demos.embedding_gemma2.tt.vision import attention as VA

    diff = vision_sdpa_differing(monkeypatch, GALAXY)
    want = {f"b1_s{s}" for s in VA.VISION_SDPA_TUNED.get(GALAXY, {})}
    assert diff == want, f"unlisted entries differ: {sorted(diff - want)}; listed but identical: {sorted(want - diff)}"


def test_vision_sdpa_identity_check_detects_a_planted_13x10_entry(monkeypatch):
    from models.demos.embedding_gemma2.tt.vision import attention as VA

    planted = {**VA.VISION_SDPA_TUNED, P150: {2560: VA.VisionSdpaSpec(256, 512)}}
    monkeypatch.setattr(VA, "VISION_SDPA_TUNED", planted)
    assert vision_sdpa_differing(monkeypatch, P150) == {"b1_s2560"}


def audio_attn_choices(monkeypatch, grid, tuned_on: bool) -> dict[str, bool]:
    from models.demos.embedding_gemma2.tt.audio import attention as AA

    monkeypatch.setattr(AA, "AUDIO_ATTN_TUNED_ON", tuned_on)
    dev = _Dev(grid)
    return {
        f"b{b}_s{s}_{dt}": AA.rel_shift_by_matmul(dev, s, b, dtype)
        for b, s in AUDIO_ATTN_SHAPES
        for dt, dtype in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32))
    }


def audio_attn_differing(monkeypatch, grid) -> set[str]:
    on, off = audio_attn_choices(monkeypatch, grid, True), audio_attn_choices(monkeypatch, grid, False)
    return {k for k in on if on[k] != off[k]}


def test_audio_attn_13x10_identical_to_tuned_off(monkeypatch):
    from models.demos.embedding_gemma2.tt.audio import attention as AA

    assert not AA.REL_SHIFT_MM_OVERRIDES, "REL_SHIFT_MM_OVERRIDES must be empty outside the in-model search"
    diff = audio_attn_differing(monkeypatch, P150)
    assert not diff, f"13 x 10 audio rel-shift choice changes with the tuned table on: {sorted(diff)}"


def test_audio_attn_12x10_differs_only_in_listed_entries(monkeypatch):
    from models.demos.embedding_gemma2.tt.audio import attention as AA

    diff = audio_attn_differing(monkeypatch, GALAXY)
    want = {f"b1_s{s}_bf16" for s in AA.REL_SHIFT_MM_TUNED.get(GALAXY, set())}
    assert diff == want, f"unlisted entries differ: {sorted(diff - want)}; listed but identical: {sorted(want - diff)}"


def test_audio_attn_identity_check_detects_a_planted_13x10_entry(monkeypatch):
    from models.demos.embedding_gemma2.tt.audio import attention as AA

    monkeypatch.setattr(AA, "REL_SHIFT_MM_TUNED", {**AA.REL_SHIFT_MM_TUNED, P150: {768}})
    assert audio_attn_differing(monkeypatch, P150) == {"b1_s768_bf16"}


def test_audio_rel_shift_select_matches_the_gather_index():
    """P[0, i] (rel_shift_select) picks exactly column rel_shift_index[i, j] for every output column j."""
    import torch

    from models.demos.embedding_gemma2.tt.audio import attention as AA
    from models.demos.embedding_gemma2.tt.audio.config import AudioAttentionConfig

    cfg = AudioAttentionConfig(
        hidden_size=1024,
        num_heads=8,
        head_dim=128,
        chunk_size=12,
        max_past_horizon=12,
        logit_cap=50.0,
        invalid_logit=-1e9,
    )
    X = torch.randn(5, AA.ATTN_BLOCK, AA.REL_WIDTH)
    idx = AA.rel_shift_index(cfg).reshape(1, AA.ATTN_BLOCK, AA.CTX).expand(5, -1, -1).long()
    want = torch.gather(X, -1, idx)
    got = torch.einsum("mic,icj->mij", X, AA.rel_shift_select(cfg)[0])
    assert torch.equal(got, want)
