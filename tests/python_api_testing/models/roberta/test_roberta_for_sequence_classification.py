import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.roberta.roberta_for_sequence_classification import (
    TtRobertaForSequenceClassification,
)
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from transformers import RobertaForSequenceClassification
from transformers import AutoTokenizer


def test_roberta_for_sequence_classification():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    base_address = ""

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion"
        )
        model = RobertaForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion"
        )
        model.eval()

        # Tt roberta
        tt_model = TtRobertaForSequenceClassification(
            config=model.config,
            base_address=base_address,
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        # Run torch model
        logger.info("Running torch model ...")
        torch_output = model(**inputs).logits

        # Run tt model
        logger.info("Running tt model ...")
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, tt_lib.device.GetHost())
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits

        # Compare outputs
        predicted_class_id = torch_output.argmax().item()
        torch_predicted_class = model.config.id2label[predicted_class_id]
        logger.info(f"Torch Predicted {torch_predicted_class}")

        tt_output_torch = tt2torch_tensor(tt_output)
        tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

        tt_predicted_class_id = tt_output_torch.argmax().item()
        tt_predicted_class = tt_model.config.id2label[tt_predicted_class_id]
        logger.info(f"Tt Predicted {tt_predicted_class}")

        does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

        logger.info(comp_allclose(torch_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)

        if does_pass:
            logger.info("RobertaForSequenceClassification Passed!")
        else:
            logger.warning("RobertaForSequenceClassification Failed!")

        assert does_pass


if __name__ == "__main__":
    test_roberta_for_sequence_classification()
