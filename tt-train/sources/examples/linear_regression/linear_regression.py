#!/usr/bin/env python
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

import numpy

import sys
import os

sys.path.append(f'{os.environ["HOME"]}/tt-metal/build/tt-train/sources/ttml/')
import _ttml

batch_size = 32

# Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X, diabetes_y = datasets.make_regression(n_samples=320, n_features=2, n_targets=1, noise=1, random_state=42)

# Use only one feature diabetes_X = diabetes_X.astype(numpy.float32)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-batch_size].astype(numpy.float32)
diabetes_X_test = diabetes_X[-batch_size:].astype(numpy.float32)

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-batch_size].astype(numpy.float32)
diabetes_y_test = diabetes_y[-batch_size:].astype(numpy.float32)

regr = _ttml.autograd.create_linear_regression_model(2, 1)
loss_function = _ttml.autograd.mse_loss
opt_config = _ttml.autograd.SGDConfig.make(0.01, 0.0, 0.0, 0.0, False)
opt = _ttml.autograd.SGD({"weight": regr.get_weight()}, opt_config)

regr.train()

diabetes_y_pred = None

for _ in range(100):
    pos = 0
    while pos < diabetes_X_train.shape[0]:
        end_pos = min(diabetes_X_train.size, pos + batch_size)

        X_slice = diabetes_X_train[pos:end_pos].reshape([batch_size, 1, 1, 2])
        y_slice = diabetes_y_train[pos:end_pos].reshape([batch_size, 1, 1, 1])

        tt_X_slice = _ttml.autograd.Tensor.from_numpy(X_slice)
        tt_y_slice = _ttml.autograd.Tensor.from_numpy(y_slice)

        opt.zero_grad()
        tt_y_pred = regr(tt_X_slice)
        diabetes_y_pred = tt_y_pred.to_numpy().reshape(batch_size)
        tt_loss = loss_function(tt_y_pred, tt_y_slice, _ttml.autograd.ReduceType.MEAN)
        print(f"{tt_loss.to_numpy()} {tt_loss.dtype()}")
        tt_loss.backward(False)

        opt.step()

        pos = end_pos

regr.eval()

print("Coefficients: \n", regr.get_weight_numpy())
# 938.23786125
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
