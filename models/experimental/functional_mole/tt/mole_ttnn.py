# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class Mole:
    def __init__(self, seq_len, pred_len, enc_in, t_dim, kernel_size, stride, device):
        self.t_dim = t_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.stride = stride
        self.dtype = ttnn.float32
        self.device = device

    def set_parameters(self, torch_model):
        self.linear_season_w = ttnn.from_torch(
            torch_model.Linear_Seasonal.weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_season_b = ttnn.from_torch(
            torch_model.Linear_Seasonal.bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_tread_w = ttnn.from_torch(
            torch_model.Linear_Trend.weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_tread_b = ttnn.from_torch(
            torch_model.Linear_Trend.bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_0_w = ttnn.from_torch(
            torch_model.Linear_Temporal[0].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_0_b = ttnn.from_torch(
            torch_model.Linear_Temporal[0].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_2_w = ttnn.from_torch(
            torch_model.Linear_Temporal[2].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_2_b = ttnn.from_torch(
            torch_model.Linear_Temporal[2].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

    def __call__(self, x, x_mark):
        """
        To conduct the function of mole.
        Args:
            x: [batch_size, seq_len, enc_in]
            x_mark: [batch_size, seq_len, time_features]
        Returns:
            y: [batch_size, pred_len, enc_in]
        """

        "---------------------------- slice --------------------------"
        x_mark_initial = ttnn.slice(
            x_mark, slice_start=(0, 0, 0), slice_end=(x_mark.shape[0], 1, x_mark.shape[2]), slice_step=(1, 1, 1)
        )  ## [batch_size, 1, time_features]

        "---------------------------- mov avg --------------------------"
        x_low_h = ttnn.slice(x, slice_start=(0, 0, 0), slice_end=(x.shape[0], 1, x.shape[2]), slice_step=(1, 1, 1))
        x_high_h = ttnn.slice(
            x, slice_start=(0, x.shape[1] - 1, 0), slice_end=(x.shape[0], x.shape[1], x.shape[2]), slice_step=(1, 1, 1)
        )

        x_low_h_repeat = ttnn.repeat(x_low_h, (1, (self.kernel_size - 1) // 2, 1))
        x_high_h_repeat = ttnn.repeat(x_high_h, (1, (self.kernel_size - 1) // 2, 1))

        x_pad = ttnn.concat([x_low_h_repeat, x, x_high_h_repeat], dim=1)

        x_pad = ttnn.to_layout(x_pad, ttnn.TILE_LAYOUT)
        x_pad = ttnn.typecast(x_pad, ttnn.bfloat16)
        x_pad = ttnn.unsqueeze(x_pad, 2)  ## [batch_size, seq_len_pad, 1, enc_in]
        dims_x_pad = x_pad.shape
        # print("x_pad.shape: ", dims_x_pad)

        avg_x_nch_bf16 = ttnn.avg_pool2d(
            input_tensor=x_pad,
            batch_size=dims_x_pad[0],
            input_h=dims_x_pad[1],
            input_w=dims_x_pad[2],
            channels=dims_x_pad[3],
            kernel_size=[self.kernel_size, 1],
            stride=[self.stride, 1],
            padding=[0, 0, 0, 0],
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        avg_x_nch = ttnn.typecast(avg_x_nch_bf16, self.dtype)
        ## must reshape, otherwise dims doesn't match the infered dims
        trend_init = ttnn.reshape(avg_x_nch, x.shape)  ## [batch_size, seq_len, enc_in]

        # print("trend_init: ",trend_init.shape)
        seasonal_init = x - trend_init  ## [batch_size, seq_len, enc_in]

        trend_init_nch = ttnn.permute(trend_init, (0, 2, 1))  ## [batch_size, enc_in, seq_len]
        seasonal_init_nch = ttnn.permute(seasonal_init, (0, 2, 1))  ## [batch_size, enc_in, seq_len]

        "---------------------------- linear season --------------------------"
        seasonal_init = ttnn.to_layout(seasonal_init_nch, layout=ttnn.TILE_LAYOUT)
        seasonal_output = ttnn.linear(
            seasonal_init, self.linear_season_w, bias=self.linear_season_b, transpose_b=True
        )  ## [batch_size, enc_in, pred_len * t_dim]

        "---------------------------- linear tread --------------------------"
        trend_init_nch = ttnn.to_layout(trend_init_nch, layout=ttnn.TILE_LAYOUT)
        trend_output = ttnn.linear(trend_init_nch, self.linear_tread_w, bias=self.linear_tread_b, transpose_b=True)
        x_raw = seasonal_output + trend_output  ## [batch_size, enc_in, pred_len * t_dim]

        "---------------------------- linear temp --------------------------"
        x_mark_initial = ttnn.to_layout(x_mark_initial, layout=ttnn.TILE_LAYOUT)
        output_temp0 = ttnn.linear(
            x_mark_initial, self.linear_0_w, bias=self.linear_0_b, transpose_b=True
        )  ##  [batch_size, 1, enc_in * t_dim]
        output_temp1 = ttnn.relu(output_temp0)
        output_temp2 = ttnn.linear(
            output_temp1, self.linear_2_w, bias=self.linear_2_b, transpose_b=True
        )  ##  [batch_size, 1, enc_in * t_dim]
        temporal_out = ttnn.reshape(output_temp2, (-1, self.t_dim))  ##  [batch_size* enc_in, t_dim]

        "---------------------------- softmax --------------------------"
        # print("temporal_out: {}".format(temporal_out.shape))
        temporal_out = ttnn.softmax(temporal_out, dim=1)  ## softmax brings a lot of precision loss from 1e-4 upto 1e-3
        temporal_out = ttnn.unsqueeze(temporal_out, dim=2)  ## [batch_size* enc_in, t_dim, 1]
        x_raw = ttnn.reshape(x_raw, (-1, self.pred_len, self.t_dim))  ## [batch_size*enc_in, pred_len, t_dim]

        "---------------------------- matmul --------------------------"
        result = ttnn.matmul(x_raw, temporal_out)  ## [batch_size*enc_in, pred_len, 1]

        result = ttnn.squeeze(result, dim=2)  ## [batch_size*enc_in, pred_len]
        result = ttnn.reshape(result, (-1, self.enc_in, self.pred_len))  ## [batch_size, enc_in, pred_len]
        result = ttnn.permute(result, (0, 2, 1))  ## [batch_size, pred_len, enc_in]

        return result


class Rmlp:
    def __init__(self, seq_len, pred_len, enc_in, t_dim, kernel_size, stride, device):
        self.t_dim = t_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = 512
        self.dropout_p = 0.1
        self.eps = 1e-5

        self.dtype = ttnn.float32
        self.device = device

    def set_parameters(self, torch_model):
        self.rev_w = ttnn.from_torch(
            torch_model.rev.affine_weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.rev_b = ttnn.from_torch(
            torch_model.rev.affine_bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.temporal_0_w = ttnn.from_torch(
            torch_model.temporal[0].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.temporal_0_b = ttnn.from_torch(
            torch_model.temporal[0].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.temporal_2_w = ttnn.from_torch(
            torch_model.temporal[2].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.temporal_2_b = ttnn.from_torch(
            torch_model.temporal[2].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_season_w = ttnn.from_torch(
            torch_model.Linear.weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_season_b = ttnn.from_torch(
            torch_model.Linear.bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_temporal_0_w = ttnn.from_torch(
            torch_model.Linear_Temporal[0].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_temporal_0_b = ttnn.from_torch(
            torch_model.Linear_Temporal[0].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_temporal_2_w = ttnn.from_torch(
            torch_model.Linear_Temporal[2].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_temporal_2_b = ttnn.from_torch(
            torch_model.Linear_Temporal[2].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

    def __call__(self, x, x_mark):
        """
        To conduct the function of mole.
        Args:
            x: [batch_size, seq_len, enc_in]
            x_mark: [batch_size, seq_len, time_features]
        Returns:
            y: [batch_size, pred_len, enc_in]
        """

        "---------------------------- slice --------------------------"
        x_mark_initial = ttnn.slice(
            x_mark, slice_start=(0, 0, 0), slice_end=(x_mark.shape[0], 1, x_mark.shape[2]), slice_step=(1, 1, 1)
        )  ## [batch_size, 1, time_features]

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x_mark_initial = ttnn.to_layout(x_mark_initial, ttnn.TILE_LAYOUT)

        "---------------------------- RevIN norm --------------------------"
        target_reduce_dim = 1
        mean = ttnn.mean(x, dim=target_reduce_dim, keepdim=True)  ## [batch_size, 1, enc_in]
        stdev = ttnn.var(x, dim=target_reduce_dim, keepdim=True)  ## [batch_size, 1, enc_in]
        stdev = stdev + self.eps
        stdev = ttnn.sqrt(stdev)

        norm0 = x - mean  ## [batch_size, seq_len, enc_in]
        norm1 = norm0 / stdev  ## [batch_size, seq_len, enc_in]

        norm = norm1 * self.rev_w + self.rev_b  ## [batch_size, seq_len, enc_in]

        "---------------------------- temporal map --------------------------"
        norm_trans = ttnn.permute(norm, (0, 2, 1))  ## [batch_size, enc_in, seq_len]
        ## [batch_size, enc_in, seq_len] ** [seq_len, d_model] --> [batch_size, enc_in, d_model]
        t_map0 = ttnn.linear(norm_trans, self.temporal_0_w, bias=self.temporal_0_b, transpose_b=True)
        t_map1 = ttnn.relu(t_map0)  ## [batch_size, enc_in, d_model]
        ## [batch_size, enc_in, d_model] ** [d_model, seq_len] --> [batch_size, enc_in, seq_len]
        t_map2 = ttnn.linear(t_map1, self.temporal_2_w, bias=self.temporal_2_b, transpose_b=True)
        t_map3 = ttnn.permute(t_map2, (0, 2, 1))  ## [batch_size, seq_len, enc_in]
        t_map = norm + t_map3

        "---------------------------- linear --------------------------"
        t_map_trans = ttnn.permute(t_map, (0, 2, 1))  ## [batch_size, enc_in, seq_len]
        ## [batch_size, enc_in, seq_len] ** [seq_len, pred_len * t_dim] --> [batch_size, enc_in, pred_len * t_dim]
        pred0 = ttnn.linear(t_map_trans, self.linear_season_w, bias=self.linear_season_b, transpose_b=True)

        "---------------------------- temp linear --------------------------"
        ## [batch_size, 1, time_features] ** [time_features, t_dim * enc_in]  --> [batch_size, 1, t_dim * enc_in]
        temporal_out0 = ttnn.linear(
            x_mark_initial, self.linear_temporal_0_w, bias=self.linear_temporal_0_b, transpose_b=True
        )
        temporal_out1 = ttnn.relu(temporal_out0)
        ## [batch_size, 1, t_dim * enc_in] ** [t_dim * enc_in, t_dim * enc_in] --> [batch_size, 1, t_dim * enc_in]
        temporal_out2 = ttnn.linear(
            temporal_out1, self.linear_temporal_2_w, bias=self.linear_temporal_2_b, transpose_b=True
        )

        "---------------------------- softmax --------------------------"
        temporal_out3 = ttnn.reshape(temporal_out2, (-1, self.t_dim, self.enc_in))  ## [batch_size, t_dim, enc_in]
        temporal_out = ttnn.softmax(temporal_out3, dim=1)  ## [batch_size, t_dim, enc_in]

        "---------------------------- multiply --------------------------"
        pred1 = ttnn.reshape(
            pred0, (-1, self.enc_in, self.pred_len, self.t_dim)
        )  ## [batch_size, enc_in, pred_len, t_dim]
        pred2 = ttnn.permute(pred1, (0, 3, 1, 2))  ## [batch_size, t_dim, enc_in, pred_len]
        pred3 = pred2 * ttnn.unsqueeze(temporal_out, -1)  ## [batch_size, t_dim, enc_in, pred_len]

        "---------------------------- sum --------------------------"
        pred4 = ttnn.sum(pred3, dim=1, keepdim=False)  ## [batch_size, enc_in, pred_len]
        pred = ttnn.permute(pred4, (0, 2, 1))  ## [batch_size, pred_len, enc_in]

        "---------------------------- RevIN denorm --------------------------"
        result0 = pred - self.rev_b  ## [batch_size, pred_len, enc_in]
        result1 = result0 / (self.rev_w + self.eps * self.eps)
        result2 = result1 * stdev
        result = result2 + mean  ## [batch_size, pred_len, enc_in]

        return result


class Rlinear:
    def __init__(self, seq_len, pred_len, enc_in, t_dim, kernel_size, stride, device):
        self.t_dim = t_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = 512
        self.dropout_p = 0.1
        self.eps = 1e-5

        self.dtype = ttnn.float32
        self.device = device

    def set_parameters(self, torch_model):
        self.rev_w = ttnn.from_torch(
            torch_model.rev.affine_weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.rev_b = ttnn.from_torch(
            torch_model.rev.affine_bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_temporal_0_w = ttnn.from_torch(
            torch_model.Linear_Temporal[0].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_temporal_0_b = ttnn.from_torch(
            torch_model.Linear_Temporal[0].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_temporal_2_w = ttnn.from_torch(
            torch_model.Linear_Temporal[2].weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_temporal_2_b = ttnn.from_torch(
            torch_model.Linear_Temporal[2].bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

        self.linear_season_w = ttnn.from_torch(
            torch_model.Linear.weight, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )
        self.linear_season_b = ttnn.from_torch(
            torch_model.Linear.bias, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self.dtype
        )

    def __call__(self, x, x_mark):
        """
        To conduct the function of mole.
        Args:
            x: [batch_size, seq_len, enc_in]
            x_mark: [batch_size, seq_len, time_features]
        Returns:
            y: [batch_size, pred_len, enc_in]
        """

        "---------------------------- slice --------------------------"
        x_mark_initial = ttnn.slice(
            x_mark, slice_start=(0, 0, 0), slice_end=(x_mark.shape[0], 1, x_mark.shape[2]), slice_step=(1, 1, 1)
        )  ## [batch_size, 1, time_features]

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x_mark_initial = ttnn.to_layout(x_mark_initial, ttnn.TILE_LAYOUT)

        "---------------------------- temp linear --------------------------"
        ## [batch_size, 1, time_features] ** [time_features, t_dim * enc_in]  --> [batch_size, 1, t_dim * enc_in]
        temporal_out0 = ttnn.linear(
            x_mark_initial, self.linear_temporal_0_w, bias=self.linear_temporal_0_b, transpose_b=True
        )
        temporal_out1 = ttnn.relu(temporal_out0)
        ## [batch_size, 1, t_dim * enc_in] ** [t_dim * enc_in, t_dim * enc_in] --> [batch_size, 1, t_dim * enc_in]
        temporal_out2 = ttnn.linear(
            temporal_out1, self.linear_temporal_2_w, bias=self.linear_temporal_2_b, transpose_b=True
        )

        "---------------------------- softmax --------------------------"
        temporal_out3 = ttnn.reshape(temporal_out2, (-1, self.t_dim, self.enc_in))  ## [batch_size, t_dim, enc_in]
        temporal_out = ttnn.softmax(temporal_out3, dim=1)  ## [batch_size, t_dim, enc_in]

        "---------------------------- RevIN norm --------------------------"
        target_reduce_dim = 1
        mean = ttnn.mean(x, dim=target_reduce_dim, keepdim=True)  ## [batch_size, 1, enc_in]
        stdev = ttnn.var(x, dim=target_reduce_dim, keepdim=True)  ## [batch_size, 1, enc_in]
        stdev = stdev + self.eps
        stdev = ttnn.sqrt(stdev)

        norm0 = x - mean  ## [batch_size, seq_len, enc_in]
        norm1 = norm0 / stdev  ## [batch_size, seq_len, enc_in]

        norm = norm1 * self.rev_w + self.rev_b  ## [batch_size, seq_len, enc_in]

        "---------------------------- linear --------------------------"
        norm_trans = ttnn.permute(norm, (0, 2, 1))  ## [batch_size, enc_in, seq_len]
        ## [batch_size, enc_in, seq_len] ** [seq_len, pred_len * t_dim] --> [batch_size, enc_in, pred_len * t_dim]
        pred0 = ttnn.linear(norm_trans, self.linear_season_w, bias=self.linear_season_b, transpose_b=True)

        "---------------------------- multiply --------------------------"
        pred1 = ttnn.reshape(
            pred0, (-1, self.enc_in, self.pred_len, self.t_dim)
        )  ## [batch_size, enc_in, pred_len, t_dim]
        pred2 = ttnn.permute(pred1, (0, 3, 1, 2))  ## [batch_size, t_dim, enc_in, pred_len]
        pred3 = pred2 * ttnn.unsqueeze(temporal_out, -1)  ## [batch_size, t_dim, enc_in, pred_len]

        "---------------------------- sum --------------------------"
        pred4 = ttnn.sum(pred3, dim=1, keepdim=False)  ## [batch_size, enc_in, pred_len]
        pred = ttnn.permute(pred4, (0, 2, 1))  ## [batch_size, pred_len, enc_in]

        "---------------------------- RevIN denorm --------------------------"
        result0 = pred - self.rev_b  ## [batch_size, pred_len, enc_in]
        result1 = result0 / (self.rev_w + self.eps * self.eps)
        result2 = result1 * stdev
        result = result2 + mean  ## [batch_size, pred_len, enc_in]

        return result
