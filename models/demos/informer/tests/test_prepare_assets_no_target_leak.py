# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU test: the checkpoint quality gate must not feed the ground truth to the decoder."""

import numpy as np
import pandas as pd
import torch

from models.demos.informer.reference.eval_utils import default_etth1_splits, iter_windows, resolve_eval_range
from models.demos.informer.reference.torch_reference import TorchInformerModel
from models.demos.informer.scripts import prepare_assets
from models.demos.informer.scripts.prepare_assets import build_default_etth1_config, evaluate_checkpoint_quality


class _RecordingModel(TorchInformerModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.future_values_args = []

    def forward(self, past_values, past_time_features, future_time_features, future_values=None):
        self.future_values_args.append(future_values)
        return super().forward(past_values, past_time_features, future_time_features, future_values)


def test_future_values_change_forward_output():
    torch.manual_seed(0)
    cfg = build_default_etth1_config()
    model = TorchInformerModel(cfg).eval()
    past = torch.randn(1, cfg.seq_len, cfg.enc_in)
    past_t = torch.randn(1, cfg.seq_len, cfg.time_feature_dim)
    future_t = torch.randn(1, cfg.pred_len, cfg.time_feature_dim)
    future = torch.randn(1, cfg.pred_len, cfg.enc_in)
    with torch.no_grad():
        out_zero = model(past, past_t, future_t, None)
        out_truth = model(past, past_t, future_t, future)
    assert not torch.allclose(out_zero, out_truth)


def test_quality_gate_passes_no_future_values():
    torch.manual_seed(0)
    cfg = build_default_etth1_config()
    model = _RecordingModel(cfg)
    n = 24 * 60 * 20 + cfg.seq_len + cfg.pred_len
    values = torch.randn(n, cfg.enc_in)
    time_features = torch.randn(n, cfg.time_feature_dim)
    evaluate_checkpoint_quality(model, values, time_features, cfg, min_improvement=-1e6, min_corr=-1.0)
    eval_start, eval_end = resolve_eval_range(n, split="test", split_cfg=default_etth1_splits())
    expected_windows = len(
        list(iter_windows(eval_end - eval_start, seq_len=cfg.seq_len, pred_len=cfg.pred_len, stride=24, max_windows=8))
    )
    assert expected_windows > 0
    assert len(model.future_values_args) == expected_windows
    assert all(arg is None for arg in model.future_values_args)


def test_training_loop_does_not_pass_future_values(tmp_path, monkeypatch):
    """create_trained_checkpoint calls the model once per training step and then runs the quality gate."""
    recorded = []

    class Recording(TorchInformerModel):
        def forward(self, past_values, past_time_features, future_time_features, future_values=None):
            recorded.append(future_values)
            return super().forward(past_values, past_time_features, future_time_features, future_values)

    monkeypatch.setattr(prepare_assets, "TorchInformerModel", Recording)
    n = 17420
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n, 7)), columns=list("ABCDEFG"))
    df.insert(0, "date", pd.date_range("2016-07-01", periods=n, freq="h"))
    csv_path = tmp_path / "synthetic_etth1.csv"
    df.to_csv(csv_path, index=False)

    train_steps = 2
    prepare_assets.create_trained_checkpoint(
        csv_path,
        tmp_path / "ckpt.pt",
        seed=0,
        force=True,
        train_steps=train_steps,
        train_batch_size=2,
        train_lr=1e-3,
        train_min_improvement=-1e6,
        train_min_corr=-1.0,
    )
    assert len(recorded) > train_steps
    assert all(arg is None for arg in recorded[:train_steps]), "training steps must not see future_values"
    assert all(arg is None for arg in recorded[train_steps:]), "quality gate must not see future_values"
