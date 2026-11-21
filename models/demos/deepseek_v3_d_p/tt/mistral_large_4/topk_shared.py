# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""topk_shared: a shared-index layer (17-18, 39-...) reuses the latest full layer's top-k of the same chunk.

No compute. The source is the full layer's device top-k (TtIndexer's output: uint32 [1, 1, S/32, k] per chip, chip
(r, t) holding query rows (4 r + t) * S/32 + [0, S/32), pads 0xFFFFFFFF), a persistent buffer the next chunk
overwrites in place. The module copies it into a buffer of
its own (ttnn.clone, as deepseek_v3_d_p's ReuseIndexer hands on the reused indices), so this layer's attention reads
what the source wrote for this chunk whatever happens to the source afterwards.
"""

import ttnn


class TtSharedTopk:
    def __call__(self, src: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.clone(src, memory_config=ttnn.DRAM_MEMORY_CONFIG)
