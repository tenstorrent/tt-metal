# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

dataset = "Tusimple"
data_root = "models/demos/ufld_v2/demo"  # Need to be modified before running
epoch = 100
batch_size = 32
optimizer = "SGD"
learning_rate = 0.05
weight_decay = 0.0001
momentum = 0.9
scheduler = "multi"
steps = [50, 75]
gamma = 0.1
warmup = "linear"
warmup_iters = 100
backbone = "34"
griding_num = 100
use_aux = False
sim_loss_w = 0.0
shp_loss_w = 0.0
note = ""
log_path = ""
finetune = None
resume = None
test_model = ""
test_work_dir = ""
num_lanes = 4
var_loss_power = 2.0
auto_backup = True
num_row = 56
num_col = 41
train_width = 800
train_height = 320
num_cell_row = 100
num_cell_col = 100
mean_loss_w = 0.05
fc_norm = False
soft_loss = True
cls_loss_col_w = 1.0
cls_ext_col_w = 1.0
mean_loss_col_w = 0.05
eval_mode = "normal"
crop_ratio = 0.8
