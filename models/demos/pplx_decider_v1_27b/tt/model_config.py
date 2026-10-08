# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.pplx_decider_v1_27b.tt.decision import DecisionConfig
from models.demos.pplx_decider_v1_27b.tt.weight_mapping import PPLX_DECIDER_HF_MODEL, load_pplx_state_dict


class PplxDeciderModelArgs(Qwen36ModelArgs):
    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=8192, **kwargs):
        # qwen36 defaults HF_MODEL to Qwen3.6-27B, so set ours first. The checkpoint has no mtp.* weights, so the
        # MTP head, which qwen36 builds for dense models by default, must be off.
        os.environ.setdefault("HF_MODEL", PPLX_DECIDER_HF_MODEL)
        super().__init__(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, enable_mtp=False, **kwargs
        )
        self.decision_config = DecisionConfig.from_checkpoint(self.CKPT_DIR)

    def load_state_dict(self):
        return load_pplx_state_dict(self.CKPT_DIR, self.decision_config.token_ids, self.vocab_size, self.dim)
