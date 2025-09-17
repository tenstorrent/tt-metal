#!/usr/bin/env python
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import sys
import os

sys.path.append(f'{os.environ["HOME"]}/git/tt-metal/tt-train/build/sources/ttml')
import _ttml

batch_size = 32
eval_size = batch_size * 4
epoch_num = 10
# Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
X, Y = datasets.make_regression(n_samples=batch_size * 16, n_features=2, n_targets=1, noise=1, random_state=42)

# Use only one feature diabetes_X = diabetes_X.astype(numpy.float32)

# Split the data into training/testing sets
x_train = X[:-eval_size].astype(np.float32)
x_test = X[-eval_size:].astype(np.float32)

# Split the targets into training/testing sets
y_train = Y[:-eval_size].astype(np.float32)
y_test = Y[-eval_size:].astype(np.float32)

model = _ttml.autograd.create_linear_regression_model(2, 1)
loss_function = _ttml.autograd.mse_loss
opt_config = _ttml.autograd.SGDConfig.make(0.1, 0.0, 0.0, 0.0, False)
opt = _ttml.autograd.SGD(model.parameters(), opt_config)

model.train()


for _ in range(epoch_num):
    pos = 0
    while pos < x_train.shape[0]:
        end_pos = min(x_train.size, pos + batch_size)

        x_slice = x_train[pos:end_pos].reshape([batch_size, 1, 1, 2])
        y_slice = y_train[pos:end_pos].reshape([batch_size, 1, 1, 1])

        tt_x_slice = _ttml.autograd.Tensor.from_numpy(x_slice)
        tt_y_slice = _ttml.autograd.Tensor.from_numpy(y_slice)

        opt.zero_grad()
        tt_y_pred = model(tt_x_slice)
        tt_loss = loss_function(tt_y_pred, tt_y_slice, _ttml.autograd.ReduceType.MEAN)
        print(f"{tt_loss.to_numpy()} {tt_loss.dtype()}")
        tt_loss.backward(False)

        opt.step()

        pos = end_pos

model.eval()

print("Coefficients: \n", model.get_weight_numpy())
# 938.23786125
# The mean squared error

# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
