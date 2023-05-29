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
from python_api_testing.models.roberta.roberta_for_token_classification import (
    TtRobertaForTokenClassification,
)
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from transformers import RobertaForTokenClassification
from transformers import AutoTokenizer


def test_roberta_for_token_classification():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    base_address = ""

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            "Jean-Baptiste/roberta-large-ner-english"
        )
        model = RobertaForTokenClassification.from_pretrained(
            "Jean-Baptiste/roberta-large-ner-english"
        )
        model.eval()

        # Tt roberta
        tt_model = TtRobertaForTokenClassification(
            config=model.config,
            base_address=base_address,
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        inputs = tokenizer(
            "HuggingFace is a company based in Paris and New York",
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Run torch model
        logger.info("Running torch model ...")
        torch_output = model(**inputs).logits
        print(torch_output.size())

        # Run tt model
        logger.info("Running tt model ...")
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, tt_lib.device.GetHost())
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits

        # Compare outputs

        # Torch output
        torch_predicted_token_class_ids = torch_output.argmax(-1)
        torch_predicted_tokens_classes = [
            model.config.id2label[t.item()] for t in torch_predicted_token_class_ids[0]
        ]
        logger.info(f"Torch Predicted {torch_predicted_tokens_classes}")

        # Tt ouptut
        tt_output_torch = tt2torch_tensor(tt_output)
        tt_output_torch = tt_output_torch.squeeze(0)

        tt_predicted_token_class_ids = tt_output_torch.argmax(-1)
        tt_predicted_tokens_classes = [
            tt_model.config.id2label[t.item()] for t in tt_predicted_token_class_ids[0]
        ]
        logger.info(f"Tt Predicted {tt_predicted_tokens_classes}")

        does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

        logger.info(comp_allclose(torch_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)

        if does_pass:
            logger.info("RobertaForTokenClassification Passed!")
        else:
            logger.warning("RobertaForTokenClassification Failed!")

        assert does_pass


if __name__ == "__main__":
    test_roberta_for_token_classification()
