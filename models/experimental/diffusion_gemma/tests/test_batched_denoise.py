# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched canvas decode tests (#47557).

CPU tests cover the ``DG_BATCH_DECODE`` flag + the batched logits wrapper's ownership/reset logic.

Device tests (``DG_RUN_DEVICE=1``) prove **per-canvas independence / no cross-canvas leakage**:

- ``test_denoise_step_batch_is_row_independent`` — the diffusion-delta decision kernels
  (Gumbel/argmax, entropy, entropy-budget accept, renoise) on a ``[B,1,C,V]`` batch produce
  bit-identical per-row results to running each row as ``[1,1,C,V]``. No checkpoint needed.
- ``test_batched_decode_smoke_matches_standalone`` (also needs ``DG_CKPT`` + ``DG_BATCH_DECODE=1``)
  — the full model-side batched denoise: B=2 committed argmax == two B=1 runs, bit-exact per row.
"""

from __future__ import annotations

import os

import pytest
import torch

DEVICE_GATED = os.environ.get("DG_RUN_DEVICE", "0") == "1"
DG_CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


# ── CPU: flag + wrapper logic ───────────────────────────────────────────
def test_flag_default_off(monkeypatch):
    from models.experimental.diffusion_gemma.tt import batched_decode

    monkeypatch.delenv("DG_BATCH_DECODE", raising=False)
    assert batched_decode.batched_decode_enabled() is False
    for on in ("1", "true", "on", "YES"):
        monkeypatch.setenv("DG_BATCH_DECODE", on)
        assert batched_decode.batched_decode_enabled() is True


def test_run_batched_requires_flag(monkeypatch, expect_error):
    from models.experimental.diffusion_gemma.tt import batched_decode

    monkeypatch.delenv("DG_BATCH_DECODE", raising=False)
    with expect_error(RuntimeError, match="opt-in"):
        batched_decode.run_batched_denoise_block(adapter=None, init_canvas=None, config=None, start_pos=0, batch=2)


def test_batched_logits_fn_loops_and_threads_per_row_state():
    """The wrapper loops the single-canvas adapter per row and keeps per-row prev-logits."""
    from models.experimental.diffusion_gemma.tt.batched_decode import BatchedDenoiseLogitsFn

    calls = []

    class _FakeTensor:
        def __init__(self, tag):
            self.tag = tag
            self.freed = False

        def deallocate(self, _=True):
            self.freed = True

    class _FakeAdapter:
        """Mimics DenoiseLogitsAdapter's prev_logits ownership contract."""

        def __init__(self):
            self.prev_logits = None
            self.q_rope_offset = 0
            self.reset_calls = 0

        def __call__(self, canvas_row, step):
            old = self.prev_logits
            if old is not None:
                old.deallocate(True)
            logits = _FakeTensor(("logits", canvas_row.tag, step))
            self.prev_logits = logits
            calls.append((canvas_row.tag, step))
            return logits

        def reset(self):
            self.reset_calls += 1
            if self.prev_logits is not None:
                self.prev_logits.deallocate(True)
                self.prev_logits = None

    # Patch ttnn.slice/concat used by the wrapper so no device is needed.
    import models.experimental.diffusion_gemma.tt.batched_decode as bd

    def fake_slice(tensor, start, end, **kw):
        return _FakeTensor(("row", start[0]))

    def fake_concat(tensors, dim=0, **kw):
        return _FakeTensor(("concat", tuple(t.tag for t in tensors)))

    orig_slice, orig_concat = bd.ttnn.slice, bd.ttnn.concat
    bd.ttnn.slice, bd.ttnn.concat = fake_slice, fake_concat
    try:
        adapter = _FakeAdapter()
        fn = BatchedDenoiseLogitsFn(adapter, batch=2)

        class _Canvas:
            shape = [2, 1, 8, 1]

        out0 = fn(_Canvas(), 0)
        assert fn.owns_logits(out0) is False  # concat is freeable by the loop
        # after step 0 the wrapper retains one prev-logits per row, detached from the adapter
        assert adapter.prev_logits is None
        assert all(p is not None for p in fn._prev)
        assert calls == [(("row", 0), 0), (("row", 1), 0)]

        prev_after_0 = list(fn._prev)
        fn(_Canvas(), 1)
        # step 1 freed the step-0 per-row logits (self-cond keeps only the immediately previous)
        assert all(p.freed for p in prev_after_0)
        fn.reset()
        assert adapter.reset_calls == 1
    finally:
        bd.ttnn.slice, bd.ttnn.concat = orig_slice, orig_concat


# ── Device: decision-kernel batch independence (no checkpoint) ──────────
#
# NOTE: run the two device tests SEPARATELY (one shared 4-chip mesh): the step test uses the
# module ``device`` fixture, while the smoke test opens its own mesh. Selecting both in one
# session would open two meshes on the same chips.
if DEVICE_GATED:
    import ttnn
    from models.experimental.diffusion_gemma.reference import sampling as S
    from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_step, temperature_at_step


def _to_dev(device, value, *, dtype=None):  # ttnn is imported under DEVICE_GATED
    dtype = ttnn.float32 if dtype is None else dtype
    return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _budget_for_accept_count(entropy: torch.Tensor, count: int) -> float:
    sorted_entropy = torch.sort(entropy, dim=-1).values
    exclusive = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
    return float((exclusive[0, count - 1] + exclusive[0, count]) / 2)


def _structured_logits(length: int, vocab_size: int, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.full((1, length, vocab_size), -4.0, dtype=torch.float32)
    token_ids = torch.randint(0, vocab_size, (length,), generator=g)
    sharpness = 0.25 + 1.75 * torch.rand(length, generator=g)
    logits[0, torch.arange(length), token_ids] = sharpness
    logits += torch.randn(logits.shape, generator=g) * 1.0e-3
    return logits


@pytest.mark.use_module_device
@pytest.mark.skipif(not DEVICE_GATED, reason="set DG_RUN_DEVICE=1")
def test_denoise_step_batch_is_row_independent(device):
    """B=2 denoise_step == two B=1 denoise_steps, bit-exact per row (no cross-canvas leakage)."""
    length = 64
    vocab = 512
    batch = 2
    max_steps = 8
    step = 3
    temperature = temperature_at_step(step, max_steps, 0.8, 0.4)

    logits_rows = [_structured_logits(length, vocab, seed=100 + r) for r in range(batch)]
    noise_rows = [
        torch.randint(0, vocab, (1, length), generator=torch.Generator().manual_seed(200 + r)) for r in range(batch)
    ]
    # A single shared budget (accept count derived from row 0), applied to every row identically.
    budget = _budget_for_accept_count(S.token_entropy(logits_rows[0], temperature=temperature), length // 3)

    # Batched [B,1,C,V].
    logits_b = torch.cat([lr.unsqueeze(1) for lr in logits_rows], dim=0)  # [B,1,C,V]
    noise_b = torch.cat([nr.view(1, 1, length, 1) for nr in noise_rows], dim=0).to(torch.int32)
    res_b = denoise_step(
        _to_dev(device, logits_b),
        temperature=temperature,
        entropy_budget=budget,
        gumbel_noise=None,
        noise_tokens=_to_dev(device, noise_b, dtype=ttnn.uint32),
    )

    def _rows(t, squeeze_last):
        out = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).squeeze(1)
        return out.squeeze(-1) if squeeze_last else out

    b_canvas = _rows(res_b.canvas, True).to(torch.long)
    b_argmax = _rows(res_b.argmax, True).to(torch.long)
    b_sampled = _rows(res_b.sampled, True).to(torch.long)
    b_entropy = _rows(res_b.entropy, True).float()
    b_accept = _rows(res_b.accept_mask, False) > 0.5  # [B,1,C] -> squeeze(1) done -> [B,C]

    for r in range(batch):
        res_i = denoise_step(
            _to_dev(device, logits_rows[r].unsqueeze(1)),
            temperature=temperature,
            entropy_budget=budget,
            gumbel_noise=None,
            noise_tokens=_to_dev(device, noise_rows[r].view(1, 1, length, 1).to(torch.int32), dtype=ttnn.uint32),
        )
        i_canvas = _rows(res_i.canvas, True).to(torch.long)[0]
        i_argmax = _rows(res_i.argmax, True).to(torch.long)[0]
        i_sampled = _rows(res_i.sampled, True).to(torch.long)[0]
        i_entropy = _rows(res_i.entropy, True).float()[0]
        i_accept = (_rows(res_i.accept_mask, False) > 0.5)[0]

        assert torch.equal(b_argmax[r], i_argmax), f"row {r} committed argmax leaked"
        assert torch.equal(b_sampled[r], i_sampled), f"row {r} sampled leaked"
        assert torch.equal(b_accept[r], i_accept), f"row {r} accept mask leaked"
        assert torch.equal(b_canvas[r], i_canvas), f"row {r} renoised canvas leaked"
        assert torch.allclose(b_entropy[r], i_entropy, atol=1e-3), f"row {r} entropy leaked"


# ── Device + checkpoint: full model-side batched denoise ────────────────
@pytest.mark.skipif(not DEVICE_GATED, reason="set DG_RUN_DEVICE=1")
@pytest.mark.skipif(not os.path.isdir(DG_CKPT), reason=f"checkpoint not available at {DG_CKPT}")
@pytest.mark.skipif(
    os.environ.get("DG_BATCH_DECODE", "0").lower() not in ("1", "true", "on", "yes"),
    reason="set DG_BATCH_DECODE=1 to run the opt-in batched decode check",
)
def test_batched_decode_smoke_matches_standalone():
    from models.experimental.diffusion_gemma.demo.batched_decode_smoke import build_arg_parser, run

    argv = [
        "--checkpoint",
        DG_CKPT,
        "--mesh",
        os.environ.get("DG_MESH", "P150x4"),
        "--num-layers",
        os.environ.get("DG_BATCH_NUM_LAYERS", "2"),
        "--max-seq-len",
        "1024",
        "--batch",
        "2",
        "--canvas-length",
        os.environ.get("DG_BATCH_CANVAS", "64"),
        "--max-denoising-steps",
        os.environ.get("DG_BATCH_STEPS", "3"),
        "--mode",
        os.environ.get("DG_BATCH_MODE", "loop"),
        "--local-files-only",
    ]
    args = build_arg_parser().parse_args(argv)
    metrics = run(args)
    assert metrics["batch"] == 2
    assert metrics["all_match"], f"batched != standalone: {metrics['per_row_mismatch_count']}"
