#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Train a TTML linear regression on synthetic data and compare to scikit-learn.

- Uses robust batching (no shape bugs on the last, smaller batch).
- Reports MSE and R² on a held-out test split.
- Prints learned coefficients/intercept for both models.

Requirements:
    pip install scikit-learn numpy
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Import TTML (adjust path if needed)
# ---------------------------------------------------------------------------
import ttnn  # noqa: E402
import ttml  # noqa: E402


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def make_synthetic_regression(
    n_samples: int,
    n_features: int,
    noise: float,
    test_size: int,
    seed: int = 42,
) -> Split:
    X, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=1,
        noise=noise,
        random_state=seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Simple tail split to keep it deterministic
    x_train, x_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return Split(x_train, y_train, x_test, y_test)


# ---------------------------------------------------------------------------
# TTML model training & inference
# ---------------------------------------------------------------------------
@dataclass
class TTMLConfig:
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.1
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False


def train_ttml_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    cfg: TTMLConfig,
    verbose: bool = True,
):
    """
    Trains TTML linear regression (2D -> 1D generalizes via n_features).
    Shapes TTML expects: [B, 1, 1, n_features] for inputs, [B, 1, 1, 1] for targets.
    """
    model = ttml.models.linear_regression.create_linear_regression_model(n_features, 1)
    loss_fn = ttml.ops.loss.mse_loss
    opt_cfg = ttml.optimizers.SGDConfig.make(
        cfg.lr, cfg.momentum, cfg.weight_decay, cfg.dampening, cfg.nesterov
    )
    opt = ttml.optimizers.SGD(model.parameters(), opt_cfg)
    model.train()

    num_samples = x_train.shape[0]
    indices = np.arange(num_samples)

    for epoch in range(cfg.epochs):
        # Shuffle each epoch
        np.random.shuffle(indices)
        pos = 0
        while pos < num_samples:
            end_pos = min(num_samples, pos + cfg.batch_size)
            batch_idx = indices[pos:end_pos]
            bsz = end_pos - pos

            x_batch = x_train[batch_idx].reshape(bsz, 1, 1, n_features)
            y_batch = y_train[batch_idx].reshape(bsz, 1, 1, 1)

            tt_x = ttml.autograd.Tensor.from_numpy(x_batch.astype(np.float32))
            tt_y = ttml.autograd.Tensor.from_numpy(y_batch.astype(np.float32))
            opt.zero_grad()
            tt_pred = model(tt_x)
            tt_loss = loss_fn(tt_pred, tt_y, ttml.ops.ReduceType.MEAN)
            tt_loss.backward(False)
            opt.step()

            if verbose:
                loss_val = float(tt_loss.to_numpy(ttnn.DataType.FLOAT32))
                print(f"[epoch {epoch+1}/{cfg.epochs}] step_loss={loss_val:.6f}")

            pos = end_pos

    model.eval()
    return model


def predict_ttml(
    model, x: np.ndarray, n_features: int, batch_size: int = 256
) -> np.ndarray:
    """
    Batched prediction to numpy 1D array.
    """
    preds = []
    num_samples = x.shape[0]
    pos = 0
    while pos < num_samples:
        end_pos = min(num_samples, pos + batch_size)
        bsz = end_pos - pos
        x_batch = x[pos:end_pos].reshape(bsz, 1, 1, n_features)
        tt_x = ttml.autograd.Tensor.from_numpy(x_batch.astype(np.float32))
        tt_y = model(tt_x).to_numpy(ttnn.DataType.FLOAT32).reshape(bsz)
        preds.append(tt_y)
        pos = end_pos
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# scikit-learn baseline
# ---------------------------------------------------------------------------
@dataclass
class SklearnResults:
    coef: np.ndarray
    intercept: float
    y_pred: np.ndarray
    mse: float
    r2: float


def fit_sklearn_baseline(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> SklearnResults:
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test).astype(np.float32).reshape(-1)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    coef = lr.coef_.reshape(-1).astype(np.float32)
    intercept = float(lr.intercept_.reshape(()))
    return SklearnResults(coef=coef, intercept=intercept, y_pred=y_pred, mse=mse, r2=r2)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@dataclass
class EvalResults:
    mse: float
    r2: float


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResults:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return EvalResults(
        mse=float(mean_squared_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TTML vs scikit-learn Linear Regression"
    )
    parser.add_argument(
        "--n-samples", type=int, default=512, help="Total samples for synthetic data"
    )
    parser.add_argument("--n-features", type=int, default=2, help="Number of features")
    parser.add_argument(
        "--noise", type=float, default=1.0, help="Noise level for make_regression"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="TTML train batch size"
    )
    parser.add_argument("--epochs", type=int, default=8, help="TTML training epochs")
    parser.add_argument("--test-size", type=int, default=128, help="Hold-out test size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print per-step losses")
    args = parser.parse_args()

    # Build data
    split = make_synthetic_regression(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
    )

    # TTML train
    cfg = TTMLConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=0.1,
    )
    model = train_ttml_linear_regression(
        split.x_train,
        split.y_train,
        n_features=args.n_features,
        cfg=cfg,
        verbose=args.verbose,
    )
    # TTML predict & evaluate
    y_pred_ttml = predict_ttml(
        model, split.x_test, n_features=args.n_features, batch_size=32
    )
    ttml_eval = evaluate(split.y_test, y_pred_ttml)

    # TTML params
    params = model.parameters()
    print(params.keys())
    ttml_w = (
        params["linear/weight"].to_numpy(ttnn.DataType.FLOAT32).reshape(-1)
    )  # shape: [n_features] (no bias)
    ttml_b = params["linear/bias"].to_numpy(ttnn.DataType.FLOAT32).item()

    # sklearn baseline
    sk = fit_sklearn_baseline(split.x_train, split.y_train, split.x_test, split.y_test)

    # Report
    print("\n=== TTML Linear Regression ===")
    print(f"Coefficients: {ttml_w}")
    print(f"Intercept: {ttml_b:.6f}")
    print(f"Test MSE: {ttml_eval.mse:.6f}")
    print(f"Test R²:  {ttml_eval.r2:.6f}")

    print("\n=== scikit-learn LinearRegression ===")
    print(f"Coefficients: {sk.coef}")
    print(f"Intercept:   {sk.intercept:.6f}")
    print(f"Test MSE:    {sk.mse:.6f}")
    print(f"Test R²:     {sk.r2:.6f}")

    # Quick side-by-side summary
    print("\n=== Summary (lower MSE & higher R² is better) ===")
    print(f"MSE  -> TTML: {ttml_eval.mse:.6f} | sklearn: {sk.mse:.6f}")
    print(f"R²   -> TTML: {ttml_eval.r2:.6f}  | sklearn: {sk.r2:.6f}")


if __name__ == "__main__":
    main()
