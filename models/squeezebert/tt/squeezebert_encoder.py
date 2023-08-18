from typing import Optional, Tuple, List
import torch
import torch.nn as nn


from models.squeezebert.tt.squeezebert_module import TtSqueezeBertModule
import tt_lib
from dataclasses import dataclass


@dataclass
class TtBaseModelOutput:
    last_hidden_state: tt_lib.tensor.Tensor = None
    hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None


class TtSqueezeBert_Encoder(nn.Module):
    def __init__(self, config, base_address="", state_dict=None, device=None) -> None:
        super().__init__()
        self.config = config
        self.base_address = base_address
        self.state_dict = state_dict
        self.device = device

        assert config.embedding_size == config.hidden_size, (
            "If you want embedding_size != intermediate hidden_size, "
            "please insert a Conv1d layer to adjust the number of channels "
            "before the first SqueezeBertModule."
        )

        self.layers = nn.ModuleList(
            TtSqueezeBertModule(
                self.config,
                f"{self.base_address}.layers.{i}",
                self.state_dict,
                self.device,
            )
            for i in range(self.config.num_hidden_layers)
        )

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor,
        head_mask: List[int],
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if head_mask is None:
            head_mask_is_all_none = True
        elif head_mask.count(None) == len(head_mask):
            head_mask_is_all_none = True
        else:
            head_mask_is_all_none = False
        assert (
            head_mask_is_all_none is True
        ), "head_mask is not yet supported in the SqueezeBert implementation."

        hidden_states = tt_lib.tensor.transpose(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                hidden_states = tt_lib.tensor.transpose(hidden_states)
                all_hidden_states += (hidden_states,)
                hidden_states = tt_lib.tensor.transpose(hidden_states)

            layer_ouput = layer(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_ouput["feature_map"]

            if output_attentions:
                all_attentions += layer_ouput["attention_score"]

        hidden_states = tt_lib.tensor.transpose(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return TtBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
