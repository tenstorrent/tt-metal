import tt_lib
from loguru import logger
from transformers import AutoTokenizer

from models.t5.tt.t5_for_conditional_generation import (
    t5_small_for_conditional_generation,
)
from models.generation_utils import run_generate


def test_demo_t5_small():
    input_sentance = "translate English to German: The house is wonderful."
    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=32)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    output_sentance = run_generate(
        input_sentance,
        tokenizer,
        t5_small_for_conditional_generation,
        device,
        run_tt_model=True,
        log=False,
    )

    logger.info(f"Input sentance: '{input_sentance}'")
    logger.info(f"Tt output: '{output_sentance}'")

    tt_lib.device.CloseDevice(device)
