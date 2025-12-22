# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class HeadDropout(nn.Module):
    def __init__(self, p=0.5):
        super(HeadDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        # If in evaluation mode, return the input as-is
        if not self.training:
            return x

        # Create a binary mask of the same shape as x
        binary_mask = (torch.rand_like(x) > self.p).float()

        # Set dropped values to negative infinity during training
        return x * binary_mask + (1 - binary_mask) * -1e20


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size, stride):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Mole(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, t_dim, time_features, kernel_size, stride):
        super(Mole, self).__init__()

        self.t_dim = t_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_p = 0.1
        self.enc_in = enc_in
        self.time_features = time_features

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.stride = stride
        self.decompsition = series_decomp(self.kernel_size, self.stride)

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len * self.t_dim)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len * self.t_dim)

        self.Linear_Temporal = nn.Sequential(
            nn.Linear(self.time_features, self.t_dim * self.enc_in),
            nn.ReLU(),
            nn.Linear(self.t_dim * self.enc_in, self.t_dim * self.enc_in),
        )
        self.head_dropout = HeadDropout(self.dropout_p)

    def forward(self, x, x_mark):
        # x: [batch_size, seq_len, enc_in]
        ## x_mark: [batch_size, seq_len, time_features]
        # print("x: {}, x_mark: {}".format(x.shape, x_mark.shape))
        x_mark_initial = None
        seasonal_init, trend_init = None, None
        x_mark_initial = x_mark[:, 0]
        # print("x_mark_initial: {}".format(x_mark_initial.shape))
        seasonal_init, trend_init = self.decompsition(x)
        # print("seasonal_init(res): {}, trend_init(avg): {}".format(seasonal_init.shape, trend_init.shape))
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        # print("after permutation, seasonal_init(res): {}, trend_init(avg): {}".format(seasonal_init.shape, trend_init.shape))
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        # print("seasonal_output: {}".format(seasonal_output.shape))
        trend_output = self.Linear_Trend(trend_init)

        # print("trend_output: {}".format(trend_output.shape))
        x = seasonal_output + trend_output

        temporal_out = self.Linear_Temporal(x_mark_initial)
        temporal_out = temporal_out.reshape(-1, self.t_dim)

        # print("temporal_out: {}".format(temporal_out.shape))
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = nn.Softmax(dim=1)(temporal_out)

        x_raw = x.reshape(-1, self.pred_len, self.t_dim)
        temporal_out = temporal_out.unsqueeze(2)

        x = torch.matmul(x_raw, temporal_out)
        x = x.squeeze(2).reshape(-1, self.enc_in, self.pred_len)
        x = x.permute(0, 2, 1)

        return x

    @staticmethod
    def from_random_weights(seq_len, pred_len, enc_in, t_dim, time_features, kernel_size, stride):
        model = Mole(seq_len, pred_len, enc_in, t_dim, time_features, kernel_size, stride)

        new_state_dict = {}
        for name, parameter in model.state_dict().items():
            new_state_dict[name] = parameter

        model.load_state_dict(new_state_dict)

        model.eval()
        return model


class RevIN(nn.Module):
    def __init__(self, enc_in: int, eps=1e-5, affine=True):
        """
        :param enc_in: the number of features or enc_in
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.enc_in = enc_in
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == "denorm":
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(self.enc_in))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


class Rmlp(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, t_dim, time_features):
        super(Rmlp, self).__init__()

        self.t_dim = t_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.time_features = time_features
        self.d_model = 512
        self.dropout_p = 0.1
        self.eps = 1e-5

        self.Linear = nn.Linear(self.seq_len, self.pred_len * self.t_dim)

        self.temporal = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.seq_len)
        )

        self.rev = RevIN(self.enc_in, self.eps)

        self.Linear_Temporal = nn.Sequential(
            nn.Linear(self.time_features, self.t_dim * self.enc_in),
            nn.ReLU(),
            nn.Linear(self.t_dim * self.enc_in, self.t_dim * self.enc_in),
        )
        self.head_dropout = HeadDropout(self.dropout_p)

    def forward(self, x, x_mark):
        # x: [batch_size, seq_len, enc_in]
        ## x_mark: [batch_size, seq_len, time_features]
        x_mark_initial = x_mark[:, 0]
        x = self.rev(x, "norm")
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)

        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.t_dim, self.enc_in)
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = nn.Softmax(dim=1)(temporal_out)
        pred_raw = pred.permute(0, 2, 1).reshape(-1, self.enc_in, self.pred_len, self.t_dim).permute(0, 3, 1, 2)
        pred = pred_raw * temporal_out.unsqueeze(-1)
        pred = pred.sum(dim=1).permute(0, 2, 1)

        pred = self.rev(pred, "denorm")

        return pred

    @staticmethod
    def from_random_weights(seq_len, pred_len, enc_in, t_dim, time_features, kernel_size, stride):
        model = Rmlp(seq_len, pred_len, enc_in, t_dim, time_features)

        new_state_dict = {}
        for name, parameter in model.state_dict().items():
            new_state_dict[name] = parameter

        model.load_state_dict(new_state_dict)

        model.eval()
        return model


class Rlinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, t_dim, time_features):
        super(Rlinear, self).__init__()

        self.t_dim = t_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.time_features = time_features

        self.time_features = time_features
        self.d_model = 512
        self.dropout_p = 0.1
        self.eps = 1e-5

        self.Linear = nn.Linear(self.seq_len, self.pred_len * self.t_dim)

        self.dropout = nn.Dropout(self.dropout_p)
        self.rev = RevIN(self.enc_in)

        self.Linear_Temporal = nn.Sequential(
            nn.Linear(self.time_features, self.t_dim * self.enc_in),
            nn.ReLU(),
            nn.Linear(self.t_dim * self.enc_in, self.t_dim * self.enc_in),
        )
        self.head_dropout = HeadDropout(self.dropout_p)

    def forward(self, x, x_mark):
        # x: [B, L, D]
        x_mark_initial = x_mark[:, 0]

        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.t_dim, self.enc_in)
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = nn.Softmax(dim=1)(temporal_out)

        x = self.rev(x, "norm")
        x = self.dropout(x)

        pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)

        pred_raw = pred.permute(0, 2, 1).reshape(-1, self.enc_in, self.pred_len, self.t_dim).permute(0, 3, 1, 2)
        pred = pred_raw * temporal_out.unsqueeze(-1)
        pred = pred.sum(dim=1).permute(0, 2, 1)

        pred = self.rev(pred, "denorm")

        return pred

    @staticmethod
    def from_random_weights(seq_len, pred_len, enc_in, t_dim, time_features, kernel_size, stride):
        model = Rlinear(seq_len, pred_len, enc_in, t_dim, time_features)

        new_state_dict = {}
        for name, parameter in model.state_dict().items():
            new_state_dict[name] = parameter

        model.load_state_dict(new_state_dict)

        model.eval()
        return model
