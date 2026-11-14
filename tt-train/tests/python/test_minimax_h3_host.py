# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the MiniMax-H3 numpy helpers: weight/LoRA layouts, AdaLN selection, masks, FLOPs, and
bit-exact parity of the packed-sequence geometry with tt_dit's (models/tt_dit/pipelines/minimax_h3/packing.py).
No device is opened."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ttml.models.minimax_h3 import flops, packing
from ttml.models.minimax_h3.weights import interleave_halves_per_shard, permute_qk_rows, unpermute_qk_rows

_REPO = Path(__file__).resolve().parents[3]
_EXAMPLE_DIR = _REPO / "tt-train" / "sources" / "examples" / "lora_minimax_h3"

TP_SIZES = (1, 2, 4, 8)
# Small attention geometry with a partial rotary (rotary_dim < head_dim), as in H3 (96 of 128).
ATTN = SimpleNamespace(num_attention_heads=4, attention_head_dim=16, rotary_dim=12)


def _rows(n: int, cols: int = 3) -> np.ndarray:
    """Distinct values that encode their own row index, so any reorder is visible."""
    return np.arange(n * cols, dtype=np.float64).reshape(n, cols)


# --------------------------------------------------------------------------- weights.py


@pytest.mark.parametrize("tp", TP_SIZES)
def test_interleave_halves_per_shard_layout(tp):
    F, cols = 8 * tp, 3
    w = _rows(2 * F, cols)  # checkpoint: [value (F) | gate (F)]
    value, gate = w[:F], w[F:]
    out = interleave_halves_per_shard(w, tp)
    per = F // tp
    for s, shard in enumerate(np.split(out, tp)):  # what each column-parallel device holds
        np.testing.assert_array_equal(shard[:per], gate[s * per : (s + 1) * per])
        np.testing.assert_array_equal(shard[per:], value[s * per : (s + 1) * per])


@pytest.mark.parametrize("rows_2d", [True, False])
def test_permute_qk_rows_roundtrip(rows_2d):
    h, hd, rot = ATTN.num_attention_heads, ATTN.attention_head_dim, ATTN.rotary_dim
    w = _rows(h * hd) if rows_2d else np.arange(hd, dtype=np.float64)
    heads = h if rows_2d else 1
    p = permute_qk_rows(w, heads, hd, rot)
    assert not np.array_equal(p, w)
    np.testing.assert_array_equal(unpermute_qk_rows(p, heads, hd, rot), w)
    perm = packing.qk_channel_permutation(hd, rot)
    np.testing.assert_array_equal(p.reshape(heads, hd, -1), w.reshape(heads, hd, -1)[:, perm])


def test_qk_permutation_keeps_attention_scores():
    """A shared per-head permutation of q and k leaves q . k unchanged; that is why the relayout is safe."""
    rng = np.random.default_rng(0)
    h, hd, rot = ATTN.num_attention_heads, ATTN.attention_head_dim, ATTN.rotary_dim
    q, k = rng.standard_normal((2, h * hd, 5))
    pq, pk = permute_qk_rows(q, h, hd, rot), permute_qk_rows(k, h, hd, rot)
    scores = lambda a, b: np.einsum("hds,hdt->hst", a.reshape(h, hd, -1), b.reshape(h, hd, -1))
    np.testing.assert_allclose(scores(pq, pk), scores(q, k), rtol=1e-12)


def test_qk_channel_permutation_pairs():
    hd, rot = ATTN.attention_head_dim, ATTN.rotary_dim
    perm = packing.qk_channel_permutation(hd, rot)
    assert sorted(perm.tolist()) == list(range(hd))
    half = rot // 2
    for j in range(half):  # rotate-half pair (j, j + rot/2) becomes adjacent (2j, 2j + 1)
        assert (perm[2 * j], perm[2 * j + 1]) == (j, j + half)
    np.testing.assert_array_equal(perm[rot:], np.arange(rot, hd))  # pass-through channels stay put


# --------------------------------------------------------------------------- lora_utils.py (adapter export/import)


@pytest.fixture(scope="module")
def lora_utils():
    if str(_EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLE_DIR))
    import lora_utils

    return lora_utils


@pytest.mark.parametrize("tp", TP_SIZES)
@pytest.mark.parametrize(
    "name, rows",
    [
        ("blocks.0.attn.to_q.lora_B", ATTN.num_attention_heads * ATTN.attention_head_dim),
        ("blocks.0.attn.to_k.lora_B", ATTN.num_attention_heads * ATTN.attention_head_dim),
        ("blocks.0.ff.proj.lora_B", 16),
        ("blocks.0.attn.to_v.lora_B", 8),  # untouched layouts
        ("blocks.0.attn.to_q.lora_A", 4),
        ("blocks.0.ff.out.lora_B", 8),
    ],
)
def test_lora_layout_roundtrip(lora_utils, name, rows, tp):
    hf = _rows(rows * tp)
    device = lora_utils.hf_to_device(name, hf, ATTN, tp=tp)
    np.testing.assert_array_equal(lora_utils.device_to_hf(name, device, ATTN, tp=tp), hf)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_lora_ff_proj_matches_base_weight_layout(lora_utils, tp):
    """ff.proj's lora_B rows must land where the loader puts the base weight's rows."""
    hf = _rows(16 * tp)
    np.testing.assert_array_equal(
        lora_utils.hf_to_device("blocks.3.ff.proj.lora_B", hf, ATTN, tp=tp), interleave_halves_per_shard(hf, tp)
    )


# --------------------------------------------------------------------------- packing.py (AdaLN selection, mask)


@pytest.mark.parametrize("fill", [None, 2])
def test_selection_matrix_gathers(fill):
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 32, size=20)
    table = rng.standard_normal((32, 7))
    sel = packing.selection_matrix(indices, 32, 32, fill=fill)
    assert sel.shape == (32, 32)
    np.testing.assert_array_equal(sel[:20] @ table, table[indices])
    np.testing.assert_array_equal(sel.sum(1)[:20], 1.0)
    if fill is None:
        np.testing.assert_array_equal(sel[20:], 0.0)
    else:
        np.testing.assert_array_equal(sel[20:] @ table, np.broadcast_to(table[fill], (12, 7)))


def test_attention_mask():
    mask = packing.attention_mask(5, 32)
    assert mask.shape == (1, 1, 32, 32)
    np.testing.assert_array_equal(mask[0, 0, :, :5], 1.0)  # every query, padded ones too, sees the real keys
    np.testing.assert_array_equal(mask[0, 0, :, 5:], 0.0)  # nobody attends to padding


# --------------------------------------------------------------------------- flops.py (hand-computed values)

# D=4, I=H*Q=4, F=3, L=2, R=1: block params 112 (q/k/v 48, out 16, qk norms 4, proj 24, out 12, norms 8),
# head params 29, N = 2 * 112 + 29 = 253; attention term 12*L*H*Q*T = 960 at T=10; LoRA rank 1 adds 49 per block.
TINY = SimpleNamespace(
    hidden_size=4,
    num_attention_heads=2,
    attention_head_dim=2,
    ffn_dim=3,
    num_layers=2,
    num_refiner_layers=1,
    video_in_channels=5,
    time_embed_dim=7,
    text_dim=6,
)


@pytest.mark.parametrize("rank, expected", [(0, 6 * 253 + 960), (1, 4 * 253 + 6 * 98 + 960)])
def test_flops_per_token(rank, expected):
    assert flops.calculate_flops_per_token(TINY, seq_len=10, rank=rank) == expected


@pytest.mark.parametrize(
    "rank, expected",
    [
        # per-token * 10 + refiner/embedder on 3 text rows + refiner attention + AdaLN (32 rows, 2 blocks)
        (0, 2478 * 10 + 6 * (116 + 28) * 3 + 3 * 144 + 6 * 38656),
        (1, 2560 * 10 + 2 * (116 + 28) * 3 + 144 + 2 * 38656),
    ],
)
def test_sequence_flops(rank, expected):
    assert flops.calculate_sequence_flops(TINY, seq_len=10, text_len=3, rank=rank) == expected


# --------------------------------------------------------------------------- parity with tt_dit

LAYOUTS = [  # (text tokens, latent frames, latent height, latent width)
    (39, 1, 24, 40),
    (7, 3, 16, 16),
    (128, 6, 48, 84),
    (1, 2, 8, 12),
]


@pytest.fixture(scope="module")
def tt_dit_packing():
    """tt_dit's packing.py, loaded by path: it needs only numpy, torch and PIL, not the tt_dit package."""
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    path = _REPO / "models" / "tt_dit" / "pipelines" / "minimax_h3" / "packing.py"
    spec = importlib.util.spec_from_file_location("tt_dit_minimax_h3_packing", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


def test_constants_match_tt_dit(tt_dit_packing):
    ref = tt_dit_packing
    assert (packing.VIDEO_TAG, packing.TEXT_TAG, packing.AUDIO_TAG) == (
        ref.MINIMAX_H3_VIDEO_TAG,
        ref.MINIMAX_H3_TEXT_TAG,
        ref.MINIMAX_H3_AUDIO_TAG,
    )
    assert packing.MODALITY_NUM == ref.MINIMAX_H3_MODALITY_NUM
    assert packing.AUDIO_CHANNELS == ref.MINIMAX_H3_AUDIO_CHANNELS
    assert packing.TILE == ref.MINIMAX_H3_TILE
    assert packing._ROPE_FRAME_RESCALE == ref._ROPE_FRAME_RESCALE
    assert packing._ROPE_FRAMES_PER_LATENT == ref._ROPE_FRAMES_PER_LATENT
    assert packing._ROPE_SPATIAL_SCALE == ref._ROPE_SPATIAL_SCALE


@pytest.mark.parametrize("n", [1, 31, 32, 33, 4736])
def test_tile_pad_matches_tt_dit(tt_dit_packing, n):
    assert packing.tile_pad(n) == tt_dit_packing.padded_sequence_length(n, sp_factor=1)


def _tt_dit_layout(ref, L, T, H, W):
    import torch

    tags = torch.full((L,), ref.MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return ref.build_packed_sequence(tags, T, H, W, packing.AUDIO_LATENTS, packing.PATCH_SIZE)


@pytest.mark.parametrize("L, T, H, W", LAYOUTS)
def test_packed_sequence_matches_tt_dit(tt_dit_packing, L, T, H, W):
    ours = packing.build_packed_sequence(L, T, H, W)
    ref = _tt_dit_layout(tt_dit_packing, L, T, H, W)
    assert ours.sequence_length == ref.sequence_length
    np.testing.assert_array_equal(ours.position_ids, ref.position_ids.numpy())  # bit-exact float64
    np.testing.assert_array_equal(ours.token_tags, ref.token_tags.numpy())
    np.testing.assert_array_equal(ours.text_indices, ref.text_indices.numpy())
    np.testing.assert_array_equal(ours.audio_indices, ref.audio_indices.numpy())
    np.testing.assert_array_equal(ours.video_indices, ref.video_indices.numpy())


@pytest.mark.parametrize("L, T, H, W", LAYOUTS)
def test_adaln_indices_match_tt_dit(tt_dit_packing, L, T, H, W):
    import torch

    ours = packing.build_packed_sequence(L, T, H, W)
    _, timestep_indices = packing.build_row_timesteps(ours, video_t=0.25, audio_t=0.75)
    ref = tt_dit_packing.adaln_indices(torch.from_numpy(ours.token_tags), torch.from_numpy(timestep_indices))
    np.testing.assert_array_equal(packing.adaln_indices(ours, timestep_indices), ref.numpy())


@pytest.mark.parametrize("T, H, W", [(1, 24, 40), (3, 16, 16)])
def test_patchify_matches_tt_dit(tt_dit_packing, T, H, W):
    import torch

    latents = np.random.default_rng(2).standard_normal((48, T, H, W)).astype(np.float32)
    ref = tt_dit_packing.patchify_video_latents(torch.from_numpy(latents)[None], packing.PATCH_SIZE)
    np.testing.assert_array_equal(packing.patchify_video_latents(latents), ref.numpy())


@pytest.fixture(scope="module")
def tt_dit_attention():
    """tt_dit's rope relayout (rope_channel_permutation, prepare_rope_tables); this imports the tt_dit package."""
    pytest.importorskip("torch")
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from models.tt_dit.models.transformers.minimax_h3 import attention_minimax_h3

    return attention_minimax_h3


def test_qk_permutation_matches_tt_dit(tt_dit_attention):
    for hd, rot in [(128, 96), (ATTN.attention_head_dim, ATTN.rotary_dim)]:
        ref = tt_dit_attention.rope_channel_permutation(hd, rot).numpy()
        np.testing.assert_array_equal(packing.qk_channel_permutation(hd, rot), ref)


@pytest.mark.parametrize("L, T, H, W", LAYOUTS)
def test_rope_tables_match_tt_dit(tt_dit_packing, tt_dit_attention, L, T, H, W):
    head_dim, rope_freq_dim, theta = 128, 16, 10000.0
    ours = packing.build_packed_sequence(L, T, H, W)
    padded = packing.tile_pad(ours.sequence_length)
    cos, sin = packing.rope_tables(ours.position_ids, padded, head_dim, rope_freq_dim, theta)
    ref_layout = _tt_dit_layout(tt_dit_packing, L, T, H, W)
    ref_cos, ref_sin = tt_dit_packing.build_rope_tables(
        ref_layout.position_ids, rope_freq_dim=rope_freq_dim, rope_theta=theta
    )
    ref_cos, ref_sin = tt_dit_attention.prepare_rope_tables(ref_cos, ref_sin, head_dim)
    S = ours.sequence_length
    # Same angles bit for bit; numpy's float32 cos/sin differ from torch's by <= 1 ulp (gone at the bf16 upload).
    np.testing.assert_array_max_ulp(cos[:S], ref_cos.numpy(), maxulp=1)
    np.testing.assert_array_max_ulp(sin[:S], ref_sin.numpy(), maxulp=1)
    np.testing.assert_array_equal(cos[S:], 1.0)  # padded rows pass through
    np.testing.assert_array_equal(sin[S:], 0.0)
