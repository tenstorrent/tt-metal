# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ModelArgs for d1-3B: LFM2.5-VL-3B's architecture, read from the checkpoint's config.json by lfm25_vl's ModelArgs."""

import os

from models.demos.multimodal.d1_3b.tt.checkpoint import resolve_checkpoint
from models.demos.multimodal.lfm25_vl.tt.model_config import ModelArgs as Lfm25VlModelArgs

PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


class D1ModelArgs(Lfm25VlModelArgs):
    def __init__(self, mesh_device, max_seq_len=PREFILL_BUCKETS[-1], **kwargs):
        os.environ["HF_MODEL"] = str(resolve_checkpoint())
        super().__init__(mesh_device, max_batch_size=1, max_seq_len=max_seq_len, **kwargs)

    def get_max_prefill_chunk_size(self):
        if self.device_name in ("N150", "N300"):
            return 128 * 1024
        return super().get_max_prefill_chunk_size()


def prefill_bucket(num_tokens):
    for bucket in PREFILL_BUCKETS:
        if num_tokens <= bucket:
            return bucket
    raise ValueError(f"{num_tokens} tokens exceed the largest prefill bucket ({PREFILL_BUCKETS[-1]})")
