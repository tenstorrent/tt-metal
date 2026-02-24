# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch 

def test_layout_for_deepseek_ocr_inputs_cumsum_op(device):
    t1 = torch.randint(0, 2, (1168640,), dtype=torch.int64)
    ttnn_t1 = ttnn.from_torch(t1,dtype=ttnn.int32,layout=ttnn.ROW_MAJOR_LAYOUT,device=device)
    ttnn_t1_tile = ttnn.to_layout(ttnn_t1,layout=ttnn.TILE_LAYOUT)


