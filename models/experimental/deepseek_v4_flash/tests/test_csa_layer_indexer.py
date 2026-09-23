# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""One real CSA attention layer, with the lightning indexer, against a torch reference.

Decode is one token at a current position. The position is a parameter.
The prefix cache is placed on the device with ``from_torch``, the device runs
a single :meth:`DeepSeekV4Attention.decode` at that position, and the output
comes back with ``to_torch``.

The reference is the modular CSA forward (compressor Ca/Cb pool, indexer
``sum_h ReLU(q · k) * w`` with both scales folded into the head weights, causal
top-k, softmax with the raw sink as an extra logit). It does not call ttnn.

Geometry is deliberately not the production (window 128, top-k 512) pair.
The sliding window is 160 and top-k is 32, so a long position has far more
closed windows than top-k keeps. Positions are 32K, 64K, 128K, 256K and 512K
tokens. The ring and the compressor window are the real projections of the
tokens still in them. Each closed window before that is a shared key.
Replaying every earlier token does not finish, and a random key per window
makes the bf16 top-k disagree with fp32. The indexer keys are built so thirty-two
of them, at the tail of the cache, outscore the rest by a wide margin.

Both sides consume the same RoPE tables and the same dequantized weights. The
device weights are bf4, so the bar is the real-weight decode PCC, not fp32.

Run (ttnn venv)::

    pytest -s models/experimental/deepseek_v4_flash/tests/test_csa_layer_indexer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# TODO: port to ``tests/decode_kv_utils.DecodeLayerKV`` once the lightning indexer is
# re-enabled; it still seeds a dense ``combined`` cache that no longer exists.
pytest.skip("lightning indexer is disabled and this test predates paged-only attention", allow_module_level=True)

import torch  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import comp_pcc  # noqa: E402
from models.experimental.deepseek_v4_flash.tests.test_attention_batching import _rope_half_tables  # noqa: E402
from models.experimental.deepseek_v4_flash.tests.test_attention_real_weights import (  # noqa: E402
    _DEFAULT_MODEL_DIR,
    _WEIGHT_DTYPE,
    _checkpoint_available,
    _rope_rows,
    _w,
    _weight_cache,
)
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.attention import (  # noqa: E402
    DeepSeekV4Attention,
    build_static_layer_cache,
    decode_sdpa_bounds,
    int32_pos_tensor,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader  # noqa: E402
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session  # noqa: E402

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score_dsa is Blackhole-only")

# The ring is shorter than these positions, so it has wrapped and top-k has keys to drop.
_WINDOW = 160
_INDEX_TOPK = 32
_POSITIONS = tuple(k * 1024 for k in (32, 64, 128, 256, 512))
_MASK_NEG = -1.0e9
# bf4 projections; same bar as the real-weight decode test.
PCC_THRESHOLD = 0.95


def _to_tt_decode_row(row: torch.Tensor, device) -> ttnn.Tensor:
    """Width-sharded L1 ``[1, 1, 1, D]``. ``all_gather_for_matmul`` rejects an interleaved tile."""
    packed = row.reshape(1, 1, 1, -1).contiguous()
    return ttnn.from_torch(
        packed,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=width_sharded_l1_config(1, packed.shape[-1], device, tile_height=1),
    )


def _rotate(rope_dim: int) -> torch.Tensor:
    """``[Rd, Rd]`` interleaved ``rotate_half`` (``(x_{2p}, x_{2p+1}) -> (-x_{2p+1}, x_{2p})``)."""
    rot = torch.zeros(rope_dim, rope_dim, dtype=torch.float32)
    for pair in range(rope_dim // 2):
        rot[2 * pair, 2 * pair + 1] = 1.0
        rot[2 * pair + 1, 2 * pair] = -1.0
    return rot


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``nn.Linear`` without bias. ``weight`` is ``[out, in]``."""
    return x.float() @ weight.float().T


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, in fp32."""
    x = x.float()
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * scale * weight.float()


def _rmsnorm_unweighted(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm with no gamma (``q_b_norm``)."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Partial interleaved RoPE on the trailing ``rope_dim`` of the last axis."""
    nope, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    turned = tail * cos + (tail @ rot) * sin
    return torch.cat([nope, turned], dim=-1)


def _host(weight) -> torch.Tensor:
    tensor = weight() if callable(weight) else weight
    return tensor.detach().float().cpu()


class _Compressor:
    """Incremental CSA pool: raw window buffers, bias applied once at close."""

    def __init__(self, cr: int, width: int, bias: torch.Tensor, norm: torch.Tensor, eps: float):
        self.cr = cr
        self.head_dim = width // 2
        self.bias = bias.reshape(cr, width)
        self.norm = norm.reshape(-1)
        self.eps = eps
        self.win_kv = torch.zeros(cr, width)
        self.win_gate = torch.zeros(cr, width)
        self.prev_kv = torch.zeros(cr, width)
        self.prev_gate = torch.full((cr, width), _MASK_NEG)
        self.keys: list[torch.Tensor] = []
        # Long positions store the closed-window keys as one tensor instead of a list.
        self.key_block: torch.Tensor | None = None

    def step(self, hidden: torch.Tensor, kv_w, gate_w, pos: int, cos, sin, rot, rope_dim: int) -> None:
        slot = pos % self.cr
        self.win_kv[slot] = _linear(hidden, kv_w)
        self.win_gate[slot] = _linear(hidden, gate_w)
        if (pos + 1) % self.cr != 0:
            return
        prev_g = self.prev_gate + self.bias
        cur_g = self.win_gate + self.bias
        new_kv = torch.cat([self.prev_kv[:, : self.head_dim], self.win_kv[:, self.head_dim :]], dim=0)
        new_gate = torch.cat([prev_g[:, : self.head_dim], cur_g[:, self.head_dim :]], dim=0)
        weights = torch.softmax(new_gate, dim=0)
        pooled = _rmsnorm((new_kv * weights).sum(0), self.norm, self.eps)
        self.keys.append(_rope(pooled, cos, sin, rot, rope_dim))
        self.prev_kv = self.win_kv.clone()
        self.prev_gate = self.win_gate.clone()


class _TorchCSA:
    """One-token CSA decode in fp32, including the indexer on the last step."""

    def __init__(self, weights: dict, cfg, rot: torch.Tensor):
        self.cfg = cfg
        self.rot = rot
        self.eps = cfg.rms_norm_eps
        self.rope_dim = cfg.qk_rope_head_dim
        self.w = {name: _host(tensor) for name, tensor in weights.items()}
        q_lora, hidden = cfg.q_lora_rank, cfg.hidden_size
        assert self.w["q_a_proj.weight"].shape == (q_lora, hidden), self.w["q_a_proj.weight"].shape
        cr = cfg.compress_rates["compressed_sparse_attention"]
        self.outer = _Compressor(
            cr,
            2 * cfg.head_dim,
            self.w["compressor.position_bias"],
            self.w["compressor.kv_norm.weight"],
            self.eps,
        )
        self.indexer = _Compressor(
            cr,
            2 * cfg.index_head_dim,
            self.w["compressor.indexer.position_bias"],
            self.w["compressor.indexer.kv_norm.weight"],
            self.eps,
        )
        self.sliding = torch.zeros(cfg.sliding_window, cfg.head_dim)
        self.folded = (cfg.index_head_dim**-0.5) * (cfg.index_n_heads**-0.5)
        self.scale = cfg.head_dim**-0.5
        self.dropped = 0

    def update(self, hidden: torch.Tensor, pos: int, q_cos, q_sin, win_cos, win_sin) -> torch.Tensor:
        """Write this token into the ring and the compressor windows. Return normalized ``q_a``."""
        q_a = _rmsnorm(_linear(hidden, self.w["q_a_proj.weight"]), self.w["q_a_norm.weight"], self.eps)
        kv = _rmsnorm(_linear(hidden, self.w["kv_proj.weight"]), self.w["kv_norm.weight"], self.eps)
        self.sliding[pos % self.cfg.sliding_window] = _rope(kv, q_cos, q_sin, self.rot, self.rope_dim)
        self.outer.step(
            hidden,
            self.w["compressor.kv_proj.weight"],
            self.w["compressor.gate_proj.weight"],
            pos,
            win_cos,
            win_sin,
            self.rot,
            self.rope_dim,
        )
        self.indexer.step(
            hidden,
            self.w["compressor.indexer.kv_proj.weight"],
            self.w["compressor.indexer.gate_proj.weight"],
            pos,
            win_cos,
            win_sin,
            self.rot,
            self.rope_dim,
        )
        return q_a

    def output(self, hidden: torch.Tensor, q_a: torch.Tensor, pos: int, q_cos, q_sin, *, sparse: bool) -> torch.Tensor:
        """Attention at ``pos`` over the cache :meth:`update` just wrote, then the grouped projection."""
        cfg = self.cfg
        window = cfg.sliding_window
        compressed = self.outer.key_block
        if compressed is None:
            compressed = torch.stack(self.outer.keys) if self.outer.keys else self.sliding.new_zeros((0, cfg.head_dim))
        n_closed = compressed.shape[0]
        k = torch.cat([self.sliding, compressed], dim=0)
        visible = torch.zeros(k.shape[0], dtype=torch.bool)
        # Dense SDPA keeps sliding slots ``<= pos`` (the whole ring once it is full) and every
        # closed window. The indexer path indexes the whole ring, then only the top-k windows.
        if pos + 1 >= window or sparse:
            visible[:window] = True
        else:
            visible[: pos + 1] = True
        if n_closed and not sparse:
            visible[window : window + n_closed] = True
        if sparse:
            q_idx = _linear(q_a, self.w["compressor.indexer.q_b_proj.weight"])
            q_idx = _rope(q_idx.reshape(cfg.index_n_heads, cfg.index_head_dim), q_cos, q_sin, self.rot, self.rope_dim)
            keys = self.indexer.key_block
            if keys is None:
                keys = torch.stack(self.indexer.keys) if self.indexer.keys else q_idx.new_zeros((0, cfg.index_head_dim))
            take = min(cfg.index_topk, keys.shape[0])
            if take:
                scores = torch.relu(q_idx.float() @ keys.float().T)
                head_w = _linear(hidden, self.w["compressor.indexer.weights_proj.weight"]) * self.folded
                index_scores = (scores * head_w[:, None]).sum(0)
                index_scores[(pos + 1) // self.outer.cr :] = float("-inf")
                selected = torch.topk(index_scores, take).indices
                visible[window : window + n_closed] = False
                visible[window + selected] = True
                self.dropped = int(keys.shape[0] - len(set(int(i) for i in selected)))

        q = _linear(q_a, self.w["q_b_proj.weight"]).reshape(cfg.num_attention_heads, cfg.head_dim)
        q = _rope(_rmsnorm_unweighted(q, self.eps), q_cos, q_sin, self.rot, self.rope_dim)
        logits = (q @ k.float().T) * self.scale
        logits[:, ~visible] = float("-inf")
        sink = self.w["sinks"].reshape(-1, 1)
        probs = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)
        attended = _rope(probs[:, :-1] @ k.float(), q_cos, -q_sin, self.rot, self.rope_dim)

        groups = cfg.o_groups
        in_per = (cfg.num_attention_heads * cfg.head_dim) // groups
        x = attended.reshape(groups, 1, in_per)
        o_a = self.w["o_a_proj.weight"].reshape(groups, cfg.o_lora_rank, in_per).transpose(1, 2)
        mixed = torch.bmm(x, o_a).reshape(-1)
        return _linear(mixed, self.w["o_b_proj.weight"])


def _config(loader: DeepseekV4WeightLoader):
    """Checkpoint attention config, with the short-sequence indexer geometry applied."""
    with (Path(loader.snapshot_dir) / "config.json").open() as fh:
        raw = json.load(fh)
    ratios = raw["compress_ratios"]
    non_zero = sorted({r for r in ratios if r})
    csa_rate, hca_rate = non_zero
    kind = {
        0: "sliding_attention",
        csa_rate: "compressed_sparse_attention",
        hca_rate: "heavily_compressed_attention",
    }
    assert csa_rate == 4, f"CSA compress rate {csa_rate} does not match the {_WINDOW}-token geometry"
    return type(
        "AttnConfig",
        (),
        {
            "hidden_size": raw["hidden_size"],
            "num_attention_heads": raw["num_attention_heads"],
            "head_dim": raw["head_dim"],
            "qk_rope_head_dim": raw["qk_rope_head_dim"],
            "q_lora_rank": raw["q_lora_rank"],
            "o_groups": raw["o_groups"],
            "o_lora_rank": raw["o_lora_rank"],
            "rms_norm_eps": raw["rms_norm_eps"],
            "sliding_window": _WINDOW,
            "index_n_heads": raw["index_n_heads"],
            "index_head_dim": raw["index_head_dim"],
            "index_topk": _INDEX_TOPK,
            "layer_types": [kind[r] for r in ratios],
            "compress_rates": {
                "compressed_sparse_attention": csa_rate,
                "heavily_compressed_attention": hca_rate,
            },
        },
    )()


def _csa_weights(loader: DeepseekV4WeightLoader, layer_idx: int) -> dict:
    keys = DeepSeekV4Model._attn_keys("compressed_sparse_attention")
    return {key: _w(loader, f"layers.{layer_idx}.self_attn.{key}") for key in keys}


def _place(dst: ttnn.Tensor, src: torch.Tensor) -> ttnn.Tensor:
    """Device tensor in ``dst``'s layout and memory, filled from host ``src``.

    ``from_torch`` allocates and writes the buffer. Sharded caches stage an
    interleaved tensor first, the same way :func:`build_static_layer_cache` does,
    then move onto ``dst``'s shard spec.
    """
    packed = src.contiguous()
    mem = dst.memory_config()
    staged = ttnn.from_torch(
        packed,
        dtype=ttnn.bfloat16,
        layout=dst.layout,
        device=dst.device(),
        memory_config=None if mem.is_sharded() else mem,
    )
    if not mem.is_sharded():
        return staged
    placed = ttnn.to_memory_config(staged, mem)
    if placed.buffer_address() != staged.buffer_address():
        ttnn.deallocate(staged)
    return placed


def _seed_cache(cache, reference, window: int) -> None:
    """Device cache becomes the host cache, which holds every token before the decode step."""
    rows, width = cache.combined.shape[2], cache.combined.shape[3]
    combined = torch.zeros(1, 1, rows, width)
    combined[0, 0, :window] = reference.sliding
    closed = reference.outer.key_block
    if closed is None and reference.outer.keys:
        closed = torch.stack(reference.outer.keys)
    if closed is not None:
        combined[0, 0, window : window + closed.shape[0]] = closed
    cache.combined = _place(cache.combined, combined)

    idx_rows, idx_width = cache.idx_key_cache.shape[2], cache.idx_key_cache.shape[3]
    idx = torch.zeros(1, 1, idx_rows, idx_width)
    index_keys = reference.indexer.key_block
    if index_keys is None and reference.indexer.keys:
        index_keys = torch.stack(reference.indexer.keys)
    if index_keys is not None:
        idx[0, 0, : index_keys.shape[0]] = index_keys
    cache.idx_key_cache = _place(cache.idx_key_cache, idx)

    def _window(buf: torch.Tensor) -> torch.Tensor:
        return buf.reshape(buf.shape[0], 1, 1, buf.shape[1])

    cache.win_kv = _place(cache.win_kv, _window(reference.outer.win_kv))
    cache.win_gate = _place(cache.win_gate, _window(reference.outer.win_gate))
    cache.prev_kv = _place(cache.prev_kv, _window(reference.outer.prev_kv))
    cache.prev_gate = _place(cache.prev_gate, _window(reference.outer.prev_gate))
    cache.idx_win_kv = _place(cache.idx_win_kv, _window(reference.indexer.win_kv))
    cache.idx_win_gate = _place(cache.idx_win_gate, _window(reference.indexer.win_gate))
    cache.idx_prev_kv = _place(cache.idx_prev_kv, _window(reference.indexer.prev_kv))
    cache.idx_prev_gate = _place(cache.idx_prev_gate, _window(reference.indexer.prev_gate))


def _window_index(pos: int, cr: int) -> int:
    """Rope-table row of the window a step at ``pos`` belongs to."""
    return max((pos + 1) // cr - 1, 0)


def _hidden_row(index: int, hidden_size: int) -> torch.Tensor:
    """One deterministic hidden row. The seed is the absolute position."""
    generator = torch.Generator()
    generator.manual_seed(index + 1234)
    return torch.randn(hidden_size, generator=generator)


def _project_rows(rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Batched ``nn.Linear`` without bias. ``rows`` is ``[N, in]``, ``weight`` is ``[out, in]``."""
    return rows.float() @ weight.float().T


def _separated_indexer_keys(q_idx: torch.Tensor, head_w: torch.Tensor, n_closed: int, k: int) -> torch.Tensor:
    """``[n_closed, D]`` keys whose top-``k`` set is stable under bf16 scoring.

    A random key per closed window has nearly flat scores, so rounding the query
    changes which 32 windows win. These keys are zero except ``k`` rows at the
    tail. Each of those rows is a growing multiple of one direction that scores
    strictly above zero, so the tail stays selected even when the device query
    is a bf4 projection of this one. A search that stops early misses them.
    """
    q = q_idx.float()
    w = head_w.float().reshape(-1)
    direction = torch.zeros(q.shape[-1])
    for head in range(q.shape[0]):
        sign = 1.0 if float(w[head]) >= 0.0 else -1.0
        direction = direction + q[head] * sign * float(w[head].abs())
    if float(direction.norm()) < 1e-6:
        direction = q.sum(0)
    direction = direction / direction.norm().clamp(min=1e-8)
    probe = (torch.relu(q @ direction) * w).sum()
    if float(probe) <= 0.0:
        direction = -direction
        probe = (torch.relu(q @ direction) * w).sum()
    assert float(probe) > 0.0, "indexer query has no direction with a positive score"

    stride = max(n_closed // (k * 2), 1)
    chosen = [n_closed - 1 - rank * stride for rank in range(k)]
    assert len(set(chosen)) == k and min(chosen) >= 0
    # The unit direction's score depends on this token. A fixed scale of 200 left
    # pos 256K with a margin of ~4 after the bf16 round trip, small enough that
    # the device query can reorder the tail. Grow the scale until the worst
    # selected key still clears the zeros by a wide gap.
    weakest = max(200.0, 50.0 / float(probe))
    keys = torch.zeros(n_closed, q.shape[-1])
    gap = 0.0
    for _ in range(6):
        for rank, idx in enumerate(chosen):
            keys[idx] = direction * (weakest * (1.0 + 0.25 * (k - rank)))
        keys = keys.bfloat16().float()
        assert torch.isfinite(keys).all(), "indexer key scale overflowed bf16"
        scores = (torch.relu(q @ keys.T) * w[:, None]).sum(0)
        picked = torch.topk(scores, k).indices
        assert {int(i) for i in picked} == set(chosen), "separated indexer keys did not occupy the top-k"
        others = scores.clone()
        others[picked] = float("-inf")
        gap = float(scores[picked].min() - others.max())
        if gap > 50.0:
            return keys
        weakest *= 4.0
    raise AssertionError(f"indexer top-k margin {gap} is too small for bf16 scoring")


def _fill_closed_window(comp: _Compressor, rows_kv: torch.Tensor, rows_gate: torch.Tensor) -> None:
    """Both compressor buffers hold the window that just closed, slot 0 first."""
    comp.win_kv = rows_kv.clone()
    comp.win_gate = rows_gate.clone()
    comp.prev_kv = rows_kv.clone()
    comp.prev_gate = rows_gate.clone()


def _seed_prefix(reference: _TorchCSA, pos: int) -> torch.Tensor:
    """Cache state after tokens ``0 .. pos - 1``.

    ``pos`` is a multiple of the compress rate and at least one full ring. The
    sliding ring is the real KV of the last ``window`` tokens, and the
    compressor windows are the real projections of the window that just closed.
    Attention keys are one unit-rms vector per closed window. Indexer keys are
    the separated tail from :func:`_separated_indexer_keys`.

    Returns the hidden row of the token at ``pos``.
    """
    cfg = reference.cfg
    cr = reference.outer.cr
    window = cfg.sliding_window
    assert pos >= window and pos % cr == 0
    ring_pos = torch.arange(pos - window, pos)
    hidden_ring = torch.stack([_hidden_row(int(t), cfg.hidden_size) for t in ring_pos])
    cos_h, sin_h = _rope_half_tables(ring_pos, reference.rope_dim)
    cos, sin = make_rope_table(cos_h, sin_h)
    kv = _rmsnorm(
        _project_rows(hidden_ring, reference.w["kv_proj.weight"]),
        reference.w["kv_norm.weight"],
        reference.eps,
    )
    for i, t in enumerate(ring_pos.tolist()):
        reference.sliding[t % window] = _rope(kv[i], cos[0, 0, i], sin[0, 0, i], reference.rot, reference.rope_dim)

    closed = hidden_ring[-cr:]
    _fill_closed_window(
        reference.outer,
        _project_rows(closed, reference.w["compressor.kv_proj.weight"]),
        _project_rows(closed, reference.w["compressor.gate_proj.weight"]),
    )
    _fill_closed_window(
        reference.indexer,
        _project_rows(closed, reference.w["compressor.indexer.kv_proj.weight"]),
        _project_rows(closed, reference.w["compressor.indexer.gate_proj.weight"]),
    )
    n_closed = pos // cr
    generator = torch.Generator()
    generator.manual_seed(pos)
    attn_keys = torch.randn(n_closed, cfg.head_dim, generator=generator)
    attn_keys = attn_keys * torch.rsqrt(attn_keys.pow(2).mean(dim=-1, keepdim=True) + reference.eps)
    reference.outer.key_block = attn_keys.bfloat16().float()

    hidden_pos = _hidden_row(pos, cfg.hidden_size)
    q_a = _rmsnorm(_linear(hidden_pos, reference.w["q_a_proj.weight"]), reference.w["q_a_norm.weight"], reference.eps)
    q_cos_h, q_sin_h = _rope_half_tables(torch.tensor([pos]), reference.rope_dim)
    q_cos, q_sin = make_rope_table(q_cos_h, q_sin_h)
    q_idx = _rope(
        _linear(q_a, reference.w["compressor.indexer.q_b_proj.weight"]).reshape(cfg.index_n_heads, cfg.index_head_dim),
        q_cos[0, 0, 0],
        q_sin[0, 0, 0],
        reference.rot,
        reference.rope_dim,
    )
    head_w = _linear(hidden_pos, reference.w["compressor.indexer.weights_proj.weight"]) * reference.folded
    reference.indexer.key_block = _separated_indexer_keys(q_idx, head_w, n_closed, cfg.index_topk)
    return hidden_pos


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("pos", _POSITIONS, ids=[f"pos{p // 1024}k" for p in _POSITIONS])
def test_csa_layer_with_indexer_matches_reference(device, reset_seeds, pos: int) -> None:
    """One CSA decode step at ``pos`` matches the torch reference at that same position.

    The prefix cache is placed with ``from_torch`` and the decode output is read
    back with ``to_torch``. The device decodes only the token at ``pos``.
    ``index_sparse`` follows the model rule: on once ``pos + 1`` reaches
    ``compress_rate * index_topk``.
    """
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip("the CSA indexer projections are prefetched")

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    cfg = _config(loader)
    layer_idx = cfg.layer_types.index("compressed_sparse_attention")
    cr = cfg.compress_rates["compressed_sparse_attention"]
    assert cfg.sliding_window == _WINDOW
    assert (_WINDOW + _INDEX_TOPK) % 32 == 0
    assert cr == 4
    assert pos >= _WINDOW and pos % cr == 0
    weights = _csa_weights(loader, layer_idx)

    rope_dim = cfg.qk_rope_head_dim
    reference = _TorchCSA(weights, cfg, _rotate(rope_dim))
    hidden_pos = _seed_prefix(reference, pos)
    q_cos_h, q_sin_h = _rope_half_tables(torch.tensor([pos]), rope_dim)
    wi = _window_index(pos, cr)
    w_cos_h, w_sin_h = _rope_half_tables(torch.tensor([wi * cr]), rope_dim)
    q_cos, q_sin = make_rope_table(q_cos_h, q_sin_h)
    w_cos, w_sin = make_rope_table(w_cos_h, w_sin_h)

    layer_type = "compressed_sparse_attention"
    attn = DeepSeekV4Attention(
        cfg,
        layer_idx,
        weights,
        device,
        cache=_weight_cache(layer_idx),
        weight_dtype=_WEIGHT_DTYPE,
        use_prefetcher=True,
        # The profile depth of 16 pages is 288 KB per core and, stacked with the q_a /
        # kv / indexer rings on core (0, 0), leaves the indexer gate matmul's circular
        # buffers overlapping that L1. Eight is what the single-device layer test uses.
        num_prefetch_pages=8,
    )
    assert attn.compressor is not None and attn.compressor.indexer is not None, "CSA layer did not attach the indexer"
    cache = build_static_layer_cache(
        device,
        cfg.sliding_window,
        layer_type,
        cfg.head_dim,
        pos + 1,
        cfg.compress_rates,
        index_head_dim=cfg.index_head_dim,
    )
    _seed_cache(cache, reference, cfg.sliding_window)

    sparse = (pos + 1) >= cr * cfg.index_topk
    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        attn.prefetch_weights(index_sparse=sparse)
        cos_d, sin_d, neg_sin_d = _rope_rows(q_cos_h, q_sin_h, device)
        cos_win_d, sin_win_d, _ = _rope_rows(w_cos_h, w_sin_h, device)
        mask, sdpa_cur_pos = decode_sdpa_bounds(cfg.sliding_window, layer_type, cr, pos, pos + 1, device)
        out_tt = attn.decode(
            _to_tt_decode_row(hidden_pos, device),
            cos_d,
            sin_d,
            neg_sin_d,
            cos_win_d,
            sin_win_d,
            mask,
            cache,
            int32_pos_tensor(pos % cfg.sliding_window, device),
            int32_pos_tensor(pos, device),
            pool_compressor=(pos + 1) % cr == 0,
            win_slot=int32_pos_tensor(pos % cr, device),
            win_row=int32_pos_tensor(cfg.sliding_window + wi, device),
            sdpa_cur_pos=sdpa_cur_pos,
            index_sparse=sparse,
        )
        got = ttnn.to_torch(out_tt).reshape(-1).float()

    q_a = reference.update(hidden_pos, pos, q_cos[0, 0, 0], q_sin[0, 0, 0], w_cos[0, 0, 0], w_sin[0, 0, 0])
    expected = reference.output(hidden_pos, q_a, pos, q_cos[0, 0, 0], q_sin[0, 0, 0], sparse=sparse)
    n_closed = reference.indexer.key_block.shape[0]
    if sparse:
        assert reference.dropped > 0, "top-k kept every closed window; the indexer cannot change this output"
        logger.info(f"pos={pos} indexer dropped {reference.dropped} of {n_closed} compressed keys")
    passing, message = comp_pcc(expected, got, pcc=PCC_THRESHOLD)
    assert passing, f"CSA decode at pos={pos} vs reference PCC below {PCC_THRESHOLD}: {message}"
    logger.info(f"pos={pos} {message}")
