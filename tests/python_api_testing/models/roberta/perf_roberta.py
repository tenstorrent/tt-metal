from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import RobertaForSequenceClassification
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, Profiler
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report

from python_api_testing.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.roberta.roberta_for_sequence_classification import (
    TtRobertaForSequenceClassification,
)

BATCH_SIZE = 1


def test_perf():
    profiler = Profiler()
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

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
            base_address="",
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        profiler.start(cpu_key)
        torch_output = model(**inputs).logits
        profiler.end(cpu_key)

        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, tt_lib.device.GetHost())

        profiler.start(first_key)
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        "roberta",
        BATCH_SIZE,
        first_iter_time,
        second_iter_time,
        "Base Emotion",
        cpu_time,
    )
