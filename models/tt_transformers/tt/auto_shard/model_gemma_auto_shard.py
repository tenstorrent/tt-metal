# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Activate the Gemma-3 auto-shard stack.

model_auto_shard rebinds model.Transformer and model.LMHead on import; this swaps the decoder block
on top of that, so create_tt_model builds the Gemma block instead of the stock auto-shard one. Import
this module (instead of model_auto_shard) from a Gemma entry point. Importing neither leaves every
existing path untouched -- nothing here runs unless something imports it.

The ModelArgs still has to be the Gemma subclass in models/demos/multimodal/gemma3/tt/model_config.py:
the base class leaves rms_norm_add_unit_offset False and embed_scale None, and neither is inferable
from the HF config. models/demos/multimodal/gemma3/demo/text_demo.py already builds that ModelArgs
and the same model.Transformer this rebinds, so adding an import of this module there is the whole
wiring.
"""

import models.tt_transformers.tt.model as _model
import models.tt_transformers.tt.auto_shard.model_auto_shard  # noqa: F401  (rebinds Transformer + LMHead)
from models.tt_transformers.tt.auto_shard.decoder_gemma_auto_shard import GemmaTransformerBlock
from models.tt_transformers.tt.auto_shard.rmsnorm_auto_shard import RMSNorm

# Transformer.__init__ reads this name when building self.layers (model.py:105).
_model.TransformerBlock = GemmaTransformerBlock


class GemmaTransformer(_model.Transformer):
    """Auto-shard Transformer whose final norm fits Gemma-3's hidden dim in L1.

    Identical to the auto-shard model except the final norm is rebuilt without fp32 accumulation --
    it is the same width as the block norms, so it hits the same L1 ceiling (see rmsnorm_auto_shard).
    """

    def __init__(self, *args, **kwargs):
        state_dict = kwargs["state_dict"]  # create_tt_model passes everything by keyword
        super().__init__(*args, **kwargs)
        self.norm = RMSNorm(
            device=self.mesh_device,
            dim=self.args.dim,
            state_dict=state_dict,
            weight_key="norm",
            axis=None,
            state_dict_prefix=self.args.get_state_dict_prefix("", None),
            eps=self.args.norm_eps,
            add_unit_offset=self.args.rms_norm_add_unit_offset,
            fp32_dest_acc_en=False,
        )


_model.Transformer = GemmaTransformer
