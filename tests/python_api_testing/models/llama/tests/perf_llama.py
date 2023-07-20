import torch
import pytest
from torch import nn
import tt_lib
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.llama.llama_utils import (
    prepare_llama_input,
    gen_position_ids,
    get_logits_processor,
)
from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)
from models.llama.tt.llama import llama_first_half, llama_second_half

from tests.python_api_testing.models.utility_functions_new import (
    Profiler,
    enable_compile_cache,
    disable_compile_cache,
    prep_report,
)


def run_llama_split_inference(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
    x_inputs=None,
    att_mask=None,
    position_ids=None,
    half=1,
    is_causallm=False,
):
    if half == 1:
        logger.debug("First half of the TT model is invoked!")
        tt_llama_model = llama_first_half(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )
    else:
        logger.debug("Second half of the TT model is invoked!")
        tt_llama_model = llama_second_half(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
            is_causallm,
        )
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )

    # returned type from the model is tuple
    tt_output = tt2torch_tensor(tt_out[0])
    return tt_output


def call_tt_llama_forward_func(
    profiler,
    configuration,
    state_dict,
    base_url,
    max_position_embeddings,
    logits_processor,
    tokenizer,
    input_ids,
    attention_mask,
    first_decoder_start,
    second_decoder_start,
    num_consecutive_decoders,
    is_causallm,
):
    # Disable compile cache
    disable_compile_cache()

    # Perf tests keys
    first_half_first_key = "first_half_first_iter"
    first_half_second_key = "first_half_second_iter"
    second_half_first_key = "second_half_first_iter"
    second_half_second_key = "second_half_second_iter"

    input_ids_padded = input_ids
    attention_mask_padded = attention_mask
    position_ids_padded = gen_position_ids(input_ids_padded)

    # The first half performance measure ========================================
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    logger.debug(f"The first call of the first half started")
    profiler.start(first_half_first_key)
    first_out = run_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=input_ids_padded,
        att_mask=attention_mask_padded,
        position_ids=position_ids_padded,
        half=1,
    )
    profiler.end(first_half_first_key)
    logger.debug(f"The first call of the first half ended")

    # enable cache for the second call
    enable_compile_cache()

    # The second call of the first half
    logger.debug(f"The second call of the first half started")
    profiler.start(first_half_second_key)
    first_out = run_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=input_ids_padded,
        att_mask=attention_mask_padded,
        position_ids=position_ids_padded,
        half=1,
    )
    profiler.end(first_half_second_key)
    logger.debug(f"The second call of the first half ended")

    first_half_first_iter_time = profiler.get(first_half_first_key)
    first_half_second_iter_time = profiler.get(first_half_second_key)
    logger.info(f"First call of the first half: {first_half_first_iter_time}")
    logger.info(f"Second call of the first half: {first_half_second_iter_time}")

    tt_lib.device.CloseDevice(device)
    logger.debug(f"The first call ended")

    # The second half performance measure ======================================
    # Disable compile cache
    disable_compile_cache()

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = first_out

    logger.debug("The first call of the second half started")
    profiler.start(second_half_first_key)
    tt_out = run_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=tt_input,
        att_mask=attention_mask_padded,
        position_ids=position_ids_padded,
        half=2,
        is_causallm=is_causallm,
    )
    profiler.end(second_half_first_key)
    logger.debug(f"The first call of the second half started ended")

    # enable cache for the second call
    enable_compile_cache()

    logger.debug("The second call of the second half started")
    profiler.start(second_half_second_key)
    tt_out = run_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=tt_input,
        att_mask=attention_mask_padded,
        position_ids=position_ids_padded,
        half=2,
        is_causallm=is_causallm,
    )
    profiler.end(second_half_second_key)
    logger.debug(f"The second call of the second half started ended")

    second_half_first_iter_time = profiler.get(second_half_first_key)
    second_half_second_iter_time = profiler.get(second_half_second_key)
    logger.info(f"First call of the second half: {second_half_first_iter_time}")
    logger.info(f"Second call of the second half: {second_half_second_iter_time}")

    tt_lib.device.CloseDevice(device)
    device = None

    return (
        first_half_first_iter_time,
        first_half_second_iter_time,
        second_half_first_iter_time,
        second_half_second_iter_time,
    )


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_max_position_embeddings = 2048
_is_causallm = False
BATCH_SIZE = 1

# how many decoders to use
# number of decoders to be stacked started from the selected id in the original llama model
# e.g. stack 16 consecutive decoders
_num_consecutive_decoders = 16

# decoder id from which decoder stacking starts (the first half of the model)
# e.g. start from 0 add use 3 decoders (0, 1, and 2)
_first_decoder_start = 0

# decoder id from which decoder stacking starts (the second half of the model)
# e.g. start from 16 add use 3 decoders (16, 17, and 18)
_second_decoder_start = _num_consecutive_decoders
# parameters --------------------------------------------------

# prompt = """Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis.
# They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
# Mention the large language model based product mentioned in the paragraph above:"""
prompt = "I believe the meaning of life is to"


@pytest.mark.parametrize(
    "prompt",
    ((prompt),),
)
def test_perf(prompt, use_program_cache):
    profiler = Profiler()
    cpu_key = "ref_key"
    comments = "llama model with two loads (halfs)"

    # set parameters =================================================================
    tokenizer_name = _tokenizer_name
    llama_model_name = _llama_model_name
    is_causallm = _is_causallm
    base_url = _base_url
    max_position_embeddings = _max_position_embeddings

    # how many decoders to use
    first_decoder_start = _first_decoder_start
    second_decoder_start = _second_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders

    # load llama pytorch model ================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        llama_model_name
    )

    hugging_face_reference_model.eval()
    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # generate real input =====================================================
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    logits_processor = get_logits_processor(
        input_ids, hugging_face_reference_model.config
    )

    is_input_padded = True
    input_ids_padded, attention_mask_padded, position_ids_padded = prepare_llama_input(
        prompt, tokenizer, configuration, is_input_padded
    )

    # PyTorch output ===========================================================
    hugging_face_reference_model = hugging_face_reference_model.get_decoder()

    # TT output: call forward() function several times ========================
    with torch.no_grad():
        # call huggingface model
        profiler.start(cpu_key)
        pytorch_out = hugging_face_reference_model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask_padded,
            position_ids=position_ids_padded,
        )
        pytorch_out = pytorch_out.last_hidden_state
        profiler.end(cpu_key)

        # The first TT model call
        (
            first_half_first_iter_time,
            first_half_second_iter_time,
            second_half_first_iter_time,
            second_half_second_iter_time,
        ) = call_tt_llama_forward_func(
            profiler,
            configuration,
            state_dict,
            base_url,
            max_position_embeddings,
            logits_processor,
            tokenizer,
            input_ids_padded,
            attention_mask_padded,
            first_decoder_start,
            second_decoder_start,
            num_consecutive_decoders,
            is_causallm,
        )

    cpu_time = profiler.get(cpu_key)

    prep_report(
        "lammaFirstHalf",
        BATCH_SIZE,
        first_half_first_iter_time,
        first_half_second_iter_time,
        comments,
        cpu_time,
    )

    prep_report(
        "lammaSecondHalf",
        BATCH_SIZE,
        second_half_first_iter_time,
        second_half_second_iter_time,
        comments,
        cpu_time,
    )
