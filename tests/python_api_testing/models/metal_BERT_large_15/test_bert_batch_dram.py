import pytest
from loguru import logger
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import sys
from pathlib import Path
import csv
import datetime

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import time
import tt_lib as ttl
from python_api_testing.models.metal_BERT_large_15.embeddings import PytorchEmbeddings
from python_api_testing.models.metal_BERT_large_15.bert_encoder import TtBertEncoder
from tt_lib.utils import pad_activation, pad_weight
from utility_functions import (
    enable_compile_cache,
    enable_compilation_reports,
    disable_compilation_reports,
    enable_memory_reports,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
    disable_compile_cache,
)
from utility_functions import profiler

from python_api_testing.models.metal_BERT_large_15.model_config import get_model_config


class TtBertBatchDram(torch.nn.Module):
    def __init__(self, config, hugging_face_reference_model, device, model_config):
        super().__init__()
        self.device = device
        self.model_config = model_config

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

        self.hidden_states_list = []
        self.tt_attention_mask_list = []

        # So far on CPU until we add embeddings support on device
        self.embeddings = PytorchEmbeddings(hugging_face_reference_model)
        self.get_extended_attention_mask = (
            hugging_face_reference_model.get_extended_attention_mask
        )

        self.encoders = torch.nn.ModuleList(
            [
                TtBertEncoder(config, encoder_idx, state_dict, device, model_config)
                for encoder_idx in range(config.num_hidden_layers)
            ]
        )

        num_classes, hidden_size = state_dict["qa_outputs.weight"].shape

        weight = pad_weight(torch.transpose(state_dict["qa_outputs.weight"], -2, -1))
        weight = (
            ttl.tensor.Tensor(
                weight.reshape(-1).tolist(),
                weight.shape,
                model_config["QA_LINEAR_WEIGHTS_DTYPE"],
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, model_config["QA_LINEAR_WEIGHTS_MEMCFG"])
        )
        bias = pad_weight(state_dict["qa_outputs.bias"])
        bias = (
            ttl.tensor.Tensor(
                bias.reshape(-1).tolist(),
                bias.shape,
                model_config["QA_LINEAR_BIAS_DTYPE"],
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, model_config["QA_LINEAR_BIAS_MEMCFG"])
        )

        # QA linear
        # TODO: Replace with custom op with fused bias?
        def qa_linear_(activation):
            output = ttl.tensor.matmul(
                activation, weight, model_config["QA_LINEAR_OUTPUT_MEMCFG"]
            )
            output_plus_bias = ttl.tensor.bcast(
                output,
                bias,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.H,
                model_config["QA_LINEAR_OUTPUT_MEMCFG"],
            )
            return output_plus_bias

        self.qa_linear = qa_linear_

    def model_preprocessing(self, input_ids, attention_mask=None, token_type_ids=None):
        embeddings = self.embeddings(input_ids, token_type_ids)
        # Convert to tt tensor
        pad_embeddings = pad_activation(embeddings)
        tt_embeddings = (
            ttl.tensor.Tensor(
                pad_embeddings.reshape(-1).tolist(),
                (
                    pad_embeddings.shape[0],
                    1,
                    pad_embeddings.shape[-2],
                    pad_embeddings.shape[-1],
                ),
                self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(self.device, self.model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"])
        )

        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )
            extended_attention_mask = torch.clamp(
                extended_attention_mask, -100000
            )  # Limit neg value that goes into exp
            extended_attention_mask = pad_activation(extended_attention_mask)
            tt_attention_mask = (
                ttl.tensor.Tensor(
                    extended_attention_mask.reshape(-1).tolist(),
                    extended_attention_mask.shape,
                    self.model_config["OP8_SOFTMAX_ATTENTION_MASK_DTYPE"],
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                .to(ttl.tensor.Layout.TILE)
                .to(self.device, self.model_config["OP8_SOFTMAX_ATTENTION_MASK_MEMCFG"])
            )
        else:
            tt_attention_mask = attention_mask
        return tt_embeddings, tt_attention_mask

    def forward(self, PERF_CNT, tt_embeddings, tt_attention_mask=None):
        print(f"Num encoders {len(self.encoders)}")

        for i in range(PERF_CNT):
            print(f"Running BERT model {i}")
            # profiler.start("_run_encoders")
            hidden_states = tt_embeddings
            attention_mask = tt_attention_mask

            for encoder in self.encoders:
                # profiler.start("__one_encoder")
                hidden_states = encoder(hidden_states, attention_mask)
                # profiler.end("__one_encoder")

            # profiler.end("_run_encoders")

            # profiler.start("_qa_linear")
            res = self.qa_linear(hidden_states)
            # profiler.end("_qa_linear")

        return res


def run_bert_question_and_answering_inference(
    model_version,
    batch,
    seq_len,
    on_weka,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_config,
    model_location_generator,
    PERF_CNT,
):
    ttl.profiler.start_profiling("Whole_Test")
    torch.manual_seed(1234)

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(
        device,
        ttl.device.MemoryAllocator.BASIC
        if not model_config["L1_BANKING"]
        else ttl.device.MemoryAllocator.L1_BANKING,
    )
    host = ttl.device.GetHost()

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
    tt_bert_model = TtBertBatchDram(
        hugging_face_reference_model.config,
        hugging_face_reference_model,
        device,
        model_config,
    )

    profiler.start("processing_of_input")

    if real_input:
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
        nlp = pipeline(
            "question-answering",
            model=hugging_face_reference_model,
            tokenizer=tokenizer,
        )
        pl_answer = nlp(question=question, context=context)

        preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
        preprocess_params["max_seq_len"] = seq_len
        input_q = {"context": context, "question": question}
        examples = nlp._args_parser(input_q)

        single_inputs = []
        for i in range(batch):
            model_input = next(nlp.preprocess(examples[0][i], **preprocess_params))
            single_input = {
                "data": (
                    model_input["input_ids"],
                    model_input["attention_mask"] if attention_mask else None,
                    model_input["token_type_ids"] if token_type_ids else None,
                ),
                "example": model_input["example"],
                "inputs": model_input,
            }
            single_inputs.append(single_input)
    else:
        if 1:
            bert_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
        else:
            # batch identical sequences for debugging
            oneseq = [torch.arange(seq_len)] * batch
            bert_input = torch.stack(oneseq)
            bert_input = bert_input.reshape(batch, seq_len)

    tt_bert_input = tt_bert_model.model_preprocessing(**bert_input)
    profiler.end("processing_of_input")

    profiler.start("hugging_face_reference_model")
    pytorch_out = hugging_face_reference_model(**bert_input)
    profiler.end("hugging_face_reference_model")

    print(f"Running BERT model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_out = tt_bert_model(1, *tt_bert_input)
    ttl.device.Synchronize()
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out
    # Recreate inputs since activations were deallocated
    tt_bert_input = tt_bert_model.model_preprocessing(**bert_input)
    print(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_compile_cache()

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    print(f"Running BERT model for perf measurement")

    profiler.start(f"model_run_{PERF_CNT}_times_for_inference")
    tt_out = tt_bert_model(PERF_CNT, *tt_bert_input)
    ttl.device.Synchronize()
    profiler.end(f"model_run_{PERF_CNT}_times_for_inference", PERF_CNT)

    # output postprocessing
    profiler.start("processing_output_to_string")

    tt_untilized_output = torch.Tensor(
        tt_out.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(batch, 1, seq_len, -1)

    tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)

    pt_start_logits = pytorch_out.start_logits.detach()
    pt_end_logits = pytorch_out.end_logits.detach()

    passing_start, output = comp_pcc(pt_start_logits, tt_start_logits, pcc)
    logger.info(f"Start Logits {output}")
    _, output = comp_allclose(
        pt_start_logits, tt_start_logits, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Start Logits {output}")
    if not passing_start:
        logger.error(f"Start Logits PCC < {pcc}")

    passing_end, output = comp_pcc(pt_end_logits, tt_end_logits, pcc)
    logger.info(f"End Logits {output}")
    _, output = comp_allclose(
        pt_end_logits, tt_end_logits, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"End Logits {output}")
    if not passing_end:
        logger.error(f"End Logits PCC < {pcc}")

    if real_input:
        for i in range(batch):
            tt_res = {
                "start": tt_start_logits[i],
                "end": tt_end_logits[i],
                "example": single_inputs[i]["example"],
                **single_inputs[i]["inputs"],
            }

            tt_answer = nlp.postprocess([tt_res], **postprocess_params)
            logger.info(f"TT: {tt_answer}")

            pt_res = {
                "start": pt_start_logits[i],
                "end": pt_end_logits[i],
                "example": single_inputs[i]["example"],
                **single_inputs[i]["inputs"],
            }

            pt_answer = nlp.postprocess([pt_res], **postprocess_params)
            logger.info(f"PT: {pt_answer}")
            logger.info(f"PL: {pl_answer[i]}")

    profiler.end("processing_output_to_string")

    del tt_out

    ttl.device.CloseDevice(device)
    profiler.print()

    ttl.profiler.stop_profiling("Whole_Test")
    # assert profiler.get("whole_model") < 60.0
    assert (
        passing_start and passing_end
    ), f"At least one start or end logits don't meet PCC requirement {pcc}"


@pytest.mark.parametrize(
    "mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["DRAM", "L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka, real_input, attention_mask, token_type_ids, pcc",
    (
        (
            "phiyodr/bert-large-finetuned-squad2",
            9,
            384,
            True,
            True,
            True,
            True,
            0.98,
        ),
    ),
    ids=["BERT_LARGE"],
)
def test_bert_batch_dram(
    model_version,
    batch,
    seq_len,
    on_weka,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    dtype,
    mem_config,
    model_location_generator,
):
    model_config = get_model_config(dtype, mem_config)

    # This test will run BERT-Large once with cache disabled.
    # Then it will enable cache and run BERT-Large PERF_CNT number of times.
    # Performance is reported only for PERF_CNT number of runs.
    PERF_CNT = 1

    disable_compile_cache()
    disable_compilation_reports()

    ttl.profiler.set_profiler_flag(False)
    ttl.profiler.set_profiler_location("tt_metal/tools/profiler/logs/BERT_large_full/")

    run_bert_question_and_answering_inference(
        model_version,
        batch,
        seq_len,
        on_weka,
        real_input,
        attention_mask,
        token_type_ids,
        pcc,
        model_config,
        model_location_generator,
        PERF_CNT,
    )


@pytest.mark.parametrize(
    "mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["DRAM", "L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka, real_input, attention_mask, token_type_ids, pcc",
    (
        (
            "phiyodr/bert-large-finetuned-squad2",
            9,
            384,
            True,
            True,
            True,
            True,
            0.98,
        ),
    ),
    ids=["BERT_LARGE"],
)
def test_bert_batch_dram_with_program_cache(
    use_program_cache,
    model_version,
    batch,
    seq_len,
    on_weka,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    dtype,
    mem_config,
    model_location_generator,
):
    model_config = get_model_config(dtype, mem_config)

    # This test will run BERT-Large once with cache disabled.
    # Then it will enable cache and run BERT-Large PERF_CNT number of times.
    # Performance is reported only for PERF_CNT number of runs.
    PERF_CNT = 1

    disable_compile_cache()
    disable_compilation_reports()

    ttl.profiler.set_profiler_flag(False)
    ttl.profiler.set_profiler_location(
        "tt_metal/tools/profiler/logs/BERT_large_full_with_program_cache/"
    )

    run_bert_question_and_answering_inference(
        model_version,
        batch,
        seq_len,
        on_weka,
        real_input,
        attention_mask,
        token_type_ids,
        pcc,
        model_config,
        model_location_generator,
        PERF_CNT,
    )

    assert ttl.program_cache.num_entries() == 12
