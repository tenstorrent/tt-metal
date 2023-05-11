
import pytest
from loguru import logger
import torch
from transformers import BloomForQuestionAnswering, AutoTokenizer, BloomTokenizerFast, pipeline
import sys
from pathlib import Path
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import time
import random
import json
from libs import tt_lib as ttl
from libs.tt_lib.utils import pad_activation, pad_weight
from utility_functions import enable_binary_cache, enable_compile_cache
from utility_functions import profiler
from utility_functions import disable_binary_cache, disable_compile_cache
import python_api_testing.models.bloom.bloom_qa as bloom_qa


class DataSampler:
    def __init__(self, file_name):
        self.data = {}
        try:
            with open(file_name) as json_file:
                self.data = json.load(json_file)
        except FileNotFoundError:
            print("File not found")

        self.file_name = file_name

    def read(self):
        titles = []

        for topic in self.data["data"]:
            titles.append(topic['title'])

        selection = random.choice(titles)

        for topic in self.data["data"]:
            if topic['title'] != selection:
                continue

            # select paragraph
            total_paragraphs = len(topic['paragraphs'])
            selected_paragraph = random.randint(0, total_paragraphs-1)
            paragraph = topic['paragraphs'][selected_paragraph]

            # select question
            total_questions = len(paragraph['qas'])
            selected_question = random.randint(0, total_questions-1)
            qas = paragraph['qas'][selected_question]
            question = qas['question']

            # get all related answers
            answers = []

            if len(qas['answers']) == 0:
                for answer in qas['plausible_answers']:
                    answers.append(answer['text'])
            else:
                for answer in qas['answers']:
                    answers.append(answer['text'])

            # get context
            context = paragraph['context']

        return {'context': context, 'question': question, 'answers': answers}


    def readn(self, num_samples):
        inputs = []

        for i in range(num_samples):
            sample = self.read()
            inputs.append(sample)

        return inputs


def sample_bloom_input(hugging_face_reference_model, tokenizer, seq_len, attention_mask, token_type_ids, qas_sample, num_samples):

    samples = []

    for i in range(num_samples):

        input_qas = qas_sample.read()
        context = [input_qas['context']]
        question = [input_qas['question']]

        bloom_input = tokenizer.batch_encode_plus(zip(question, context), max_length=seq_len, padding="max_length", truncation=True, return_tensors="pt")
        nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)
        pl_answer = nlp(question = question[0], context=context[0])
        preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
        preprocess_params["max_seq_len"] = seq_len
        input_q = {"context": context[0], "question": question[0]}
        examples = nlp._args_parser(input_q)
        model_input = next(nlp.preprocess(examples[0], **preprocess_params))

        single_input = {
            "data": (
                model_input["input_ids"],
                model_input["attention_mask"] if attention_mask else None,
                model_input["token_type_ids"] if token_type_ids else None
            ),
            "example": model_input["example"],
            "inputs": model_input,
        }

        bloom_input = {}
        bloom_input["input_ids"] = single_input["data"][0]
        bloom_input["attention_mask"] = single_input["data"][1]
        bloom_input["token_type_ids"] = single_input["data"][2]

        sample = {}
        sample["bloom_input"] = bloom_input
        sample["single_input"] = single_input
        sample["nlp"] = nlp
        sample["postprocess_params"] = postprocess_params
        sample["pl_answer"] = pl_answer
        sample["context"] = context
        sample["question"] = question
        sample["answers"] = input_qas["answers"]

        samples.append(sample)

    return samples


def model_location_generator_(rel_path):
    internal_weka_path = Path("/mnt/MLPerf")
    has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

    if has_internal_weka:
        return Path("/mnt/MLPerf") / rel_path
    else:
        return Path("/opt/tt-metal-models") / rel_path


def run_bloom_question_and_answering_inference(model_version, batch, seq_len, on_weka, attention_mask, token_type_ids, pcc, model_location_generator, qas_sample, num_samples):

    torch.manual_seed(1234)

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    if on_weka:
        model_name = "bigscience/bloom-560m"
        tokenizer_name = "BloomTokenizerFast"
    else:
        model_name = model_version
        tokenizer_name = model_version

    model_name = "bigscience/bloom-560m"
    hugging_face_reference_model = BloomForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    hugging_face_reference_model.eval()

    config = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    tt_bloom_qa = bloom_qa.TtBloomForQuestionAnswering(config, state_dict, device)
    #pt_bloom_qa = hugging_bloom_reference_model


    #tt_bloom_model = bloom_qa.TtBloomForQuestionAnswering(hugging_face_reference_model.config, hugging_face_reference_model, seq_len, device)

    profiler.start("processing_of_input")
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    print(f"Sampling random context+question pairs")
    samples = sample_bloom_input(hugging_face_reference_model, tokenizer, seq_len, attention_mask, token_type_ids, qas_sample, num_samples)
    profiler.end("processing_of_input")

    print(f"Running BLOOM model")
    profiler.start("whole_model")
    tt_out_list = tt_bloom_model(samples)
    profiler.end("whole_model", num_samples)

    profiler.start("processing_output_to_string")

    for i in range(len(tt_out_list)):

        single_input = samples[i]["single_input"]
        nlp = samples[i]["nlp"]
        postprocess_params = samples[i]["postprocess_params"]
        tt_out = tt_out_list[i]
        context = samples[i]['context']
        question = samples[i]['question']
        answers = samples[i]['answers']

        tt_out = tt_out.to(host)
        tt_untilized_output = torch.Tensor(tt_out.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(batch, 1, seq_len, -1)

        tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)
        tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)

        tt_res = {
            "start": tt_start_logits,
            "end": tt_end_logits,
            "example": single_input["example"],
            **single_input["inputs"],
        }

        tt_answer = nlp.postprocess([tt_res], **postprocess_params)['answer']

        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Answer from GS: '{tt_answer}'")
        print(f"All valid answers: {answers}\n")

    profiler.end("processing_output_to_string")

    ttl.device.CloseDevice(device)
    profiler.print()

def test_bloom_sample_qas():
    model_version = "bigscience/bloom-560m"
    batch = 1
    seq_len = 384
    on_weka = False
    attention_mask = True
    token_type_ids = True
    pcc = 0.50
    model_location_generator = model_location_generator_
    qas_sample = DataSampler('./tests/python_api_testing/models/bloom/dev-v2.0.json')
    num_samples = 10

    logger.warning("This test uses binary and compile cache. The cache needs to be filled before running this test.")
    run_bloom_question_and_answering_inference(model_version, batch, seq_len, on_weka, attention_mask, token_type_ids, pcc, model_location_generator, qas_sample, num_samples)

if __name__ == "__main__":
    test_bloom_sample_qas()
