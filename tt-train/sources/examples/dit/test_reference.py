# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the golden DiT reference: run `python test_reference.py`."""

import torch

from reference_torch import (
    DiffusionSchedule,
    dit_tiny,
    patchify,
    sample_ddim,
    training_step,
    unpatchify,
)


def test_patchify_roundtrip():
    x = torch.randn(2, 3, 32, 32)
    tokens = patchify(x, patch=4)
    assert tokens.shape == (2, 1, 64, 48)
    back = unpatchify(tokens, patch=4, channels=3, height=32, width=32)
    assert torch.equal(x, back)
    print("patchify roundtrip OK")


def test_overfit_single_batch():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    patch, num_classes = 4, 10
    model = dit_tiny(in_dim=48, num_tokens=64, num_classes=num_classes)
    schedule = DiffusionSchedule(timesteps=1000)
    images = torch.rand(8, 3, 32, 32, generator=gen) * 2 - 1
    labels = torch.randint(0, num_classes, (8,), generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    first, last = None, None
    for step in range(150):
        opt.zero_grad()
        # fixed sampling seed per step keeps the task deterministic
        loss = training_step(
            model, schedule, images, labels, patch,
            cfg_drop_prob=0.1, null_class=num_classes,
            generator=torch.Generator().manual_seed(step % 4),
        )
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
        if step % 30 == 0:
            print(f"step {step:4d} loss {loss.item():.4f}")
    print(f"first {first:.4f} -> last {last:.4f}")
    assert last < first * 0.5, "did not overfit"
    print("overfit OK")


def test_sampling_runs():
    torch.manual_seed(0)
    model = dit_tiny(in_dim=48, num_tokens=64, num_classes=10)
    schedule = DiffusionSchedule()
    labels = torch.tensor([0, 1, 2, 3])
    imgs = sample_ddim(model, schedule, labels, (3, 32, 32), patch=4, steps=8,
                       cfg_scale=2.0, null_class=10, generator=torch.Generator().manual_seed(1))
    assert imgs.shape == (4, 3, 32, 32) and torch.isfinite(imgs).all()
    print("sampling OK")


if __name__ == "__main__":
    test_patchify_roundtrip()
    test_overfit_single_batch()
    test_sampling_runs()
    print("ALL OK")
