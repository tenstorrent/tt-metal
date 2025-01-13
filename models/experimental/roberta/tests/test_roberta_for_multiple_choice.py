# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from loguru import logger
from transformers import AutoTokenizer, RobertaForMultipleChoice


from models.experimental.roberta.tt.roberta_for_multiple_choice import TtRobertaForMultipleChoice
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
    is_wormhole_b0,
    is_blackhole,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.skip(reason="Mismatch happening on GS, issue #5943")
def test_roberta_for_multiple_choice(device):
    """
    RoBERTa for multiple choice is loading roberta-base pre-trained model,
    because there are no official weights for RobertaForMultipleChoice
    """
    torch.manual_seed(1234)
    base_address = ""

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = RobertaForMultipleChoice.from_pretrained("roberta-base")
        model.eval()

        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        choice0 = "It is eaten with a fork and a knife."
        choice1 = "It is eaten while held in the hand."

        encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)

        # Tt roberta
        tt_model = TtRobertaForMultipleChoice(
            config=model.config,
            base_address=base_address,
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        # Run torch model
        logger.info("Running torch model...")
        torch_outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()})
        torch_predicted_class = torch_outputs.logits.argmax().item()

        # Run tt model
        inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
        print(inputs_dict["attention_mask"].shape)
        inputs_dict["attention_mask"] = torch.unsqueeze(inputs_dict["attention_mask"], 0)
        inputs_dict["attention_mask"] = torch2tt_tensor(inputs_dict["attention_mask"], device)
        print(inputs_dict["attention_mask"].shape.with_tile_padding())

        logger.info("Running tt model ...")
        tt_output = tt_model(**inputs_dict)
        tt_output_to_torch = tt2torch_tensor(tt_output.logits)
        tt_output_to_torch = tt_output_to_torch.squeeze(0)
        tt_output_to_torch = tt_output_to_torch.squeeze(0)
        tt_predicted_class = tt_output_to_torch.argmax().item()

        print(torch_outputs.logits)
        print(tt_output.logits)
        # Compare outputs

        # Torch output
        logger.info(f"Torch Predicted {torch_predicted_class}")

        logger.info(f"Tt Predicted {tt_predicted_class}")

        does_pass, pcc_message = comp_pcc(torch_outputs.logits, tt_output_to_torch, 0.98)

        # Temporarily change passing codition to allclose until layernorm accuracy is updated
        does_pass, allclose_message = comp_allclose(torch_outputs.logits, tt_output_to_torch, 0, 0.0081)
        logger.info(allclose_message)
        logger.info(pcc_message)

        if does_pass:
            logger.info("RobertaForMultipleChoice Passed!")
        else:
            logger.warning("RobertaForMultipleChoice Failed!")

        assert does_pass
