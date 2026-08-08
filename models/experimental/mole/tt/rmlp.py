# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.experimental.mole.reference.rmlp import RMLPExpert
from models.experimental.mole.tt.rlinear import TtRLinearExpert


class TtRMLPExpert(TtRLinearExpert):
    def __init__(self, config, *, reference_model=None, checkpoint_state_dict=None, runtime_options=None):
        if reference_model is None and checkpoint_state_dict is None:
            reference_model = RMLPExpert(config).eval()
        super().__init__(
            config,
            reference_model=reference_model,
            checkpoint_state_dict=checkpoint_state_dict,
            runtime_options=runtime_options,
        )
