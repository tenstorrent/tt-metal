import pytest
import torch
import tt_lib
from loguru import logger
from transformers import BertForQuestionAnswering, BertTokenizer
from tests.python_api_testing.models.conftest import model_location_generator_
from tests.python_api_testing.models.utility_functions import comp_pcc, profiler
from tests.python_api_testing.models.metal_BERT_large_15.embeddings import PytorchEmbeddings, TtBertEmbeddings


def get_input(tokenizer_name, attention_mask, token_type_ids, batch, seq_len):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    context = batch * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = batch * ["What discipline did Winkelmann create?"]
    bert_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=attention_mask,
        return_token_type_ids=token_type_ids,
        return_tensors="pt",
    )
    return bert_input



@pytest.mark.parametrize(
    "on_weka, model_version, attention_mask, token_type_ids, batch_num, seq_len",
    ((True, "phiyodr/bert-large-finetuned-squad2", True, True, 8, 384),),
    ids=["BERT_LARGE"],
)
def test_embeddings_inference(
    on_weka, model_version, attention_mask, token_type_ids, batch_num, seq_len
):
    model_location_generator = model_location_generator_
    torch.manual_seed(1234)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    if on_weka:
        model_name = str(
            model_location_generator(
                "tt_dnn-models/Bert/BertForQuestionAnswering/models/"
            )
            / model_version
        )
        tokenizer_name = str(
            model_location_generator(
                "tt_dnn-models/Bert/BertForQuestionAnswering/tokenizers/"
            )
            / model_version
        )
    else:
        model_name = model_version
        tokenizer_name = model_version

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    )

    hugging_face_reference_model.eval()
    bert_input = get_input(
        tokenizer_name, attention_mask, token_type_ids, batch_num, seq_len
    )

    base_address = f"bert.embeddings"
    state_dict = hugging_face_reference_model.state_dict()

    cpu_key = "ref_key"
    first_key = "first_iter"
    second_key = "second_iter"
    ref_model = PytorchEmbeddings(hugging_face_reference_model)
    tt_model = TtBertEmbeddings(
        hugging_face_reference_model.config,
        state_dict,
        base_address,
        device
    )
    profiler.start(cpu_key)
    pytorch_out_ref = ref_model(**bert_input)
    profiler.end(cpu_key)
    tt_out = tt_model(**bert_input)
    profiler.start(first_key)
    tt_out = tt_model(**bert_input)
    profiler.end(first_key)
    profiler.start(second_key)
    tt_out = tt_model(**bert_input)
    profiler.end(second_key)

    cpu_iter_time = profiler.get(cpu_key)
    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    passing_pcc, output_pcc = comp_pcc(pytorch_out_ref, tt_out, 0.99)
    logger.info(f"CPU={cpu_iter_time}")
    logger.info(f"First Iteration={first_iter_time}")
    logger.info(f"Second Iteration={second_iter_time}")
    logger.info(f"Out passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")
    assert passing_pcc
    return
