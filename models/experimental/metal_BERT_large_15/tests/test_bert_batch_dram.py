# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from loguru import logger
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

import tt_lib

from models.experimental.metal_BERT_large_15.tt.embeddings import TtEmbeddings
from models.experimental.metal_BERT_large_15.tt.bert_encoder import TtBertEncoder
from models.experimental.metal_BERT_large_15.tt.model_config import get_model_config

from tt_lib.utils import pad_activation, pad_weight
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_compilation_reports,
    comp_pcc,
    comp_allclose,
    disable_persistent_kernel_cache,
    profiler,
)


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

        self.embeddings = TtEmbeddings(
            hugging_face_reference_model,
            device,
            model_config=model_config,
        )

        self.get_extended_attention_mask = hugging_face_reference_model.get_extended_attention_mask

        self.encoders = torch.nn.ModuleList(
            [
                TtBertEncoder(config, encoder_idx, state_dict, device, model_config)
                for encoder_idx in range(config.num_hidden_layers)
            ]
        )

        num_classes, hidden_size = state_dict["qa_outputs.weight"].shape

        weight = pad_weight(torch.transpose(state_dict["qa_outputs.weight"], -2, -1))
        weight = (
            tt_lib.tensor.Tensor(
                weight.reshape(-1).tolist(),
                weight.shape,
                model_config["QA_LINEAR_WEIGHTS_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
            .to(tt_lib.tensor.Layout.TILE)
            .to(device, model_config["QA_LINEAR_WEIGHTS_MEMCFG"])
        )
        bias = pad_weight(state_dict["qa_outputs.bias"])
        bias = (
            tt_lib.tensor.Tensor(
                bias.reshape(-1).tolist(),
                bias.shape,
                model_config["QA_LINEAR_BIAS_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
            .to(tt_lib.tensor.Layout.TILE)
            .to(device, model_config["QA_LINEAR_BIAS_MEMCFG"])
        )

        # QA linear
        # TODO: Replace with custom op with fused bias?
        def qa_linear_(activation):
            output = tt_lib.tensor.matmul(activation, weight, model_config["QA_LINEAR_OUTPUT_MEMCFG"])
            output_plus_bias = tt_lib.tensor.bcast(
                output,
                bias,
                tt_lib.tensor.BcastOpMath.ADD,
                tt_lib.tensor.BcastOpDim.H,
                model_config["QA_LINEAR_OUTPUT_MEMCFG"],
            )
            return output_plus_bias

        self.qa_linear = qa_linear_

    def model_embedding(self, input_ids, token_type_ids=None, position_ids=None):
        tt_embeddings = self.embeddings(input_ids, token_type_ids, position_ids)
        embeddings_shape = tt_embeddings.shape()
        if tt_embeddings.dtype() != self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"]:
            logger.warning("Perf warning: On host conversion of dtype after embeddings")
            embeddings = tt_embeddings.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
            tt_embeddings = (
                tt_lib.tensor.Tensor(
                    embeddings.reshape(-1).tolist(),
                    (
                        embeddings_shape[0],
                        1,
                        embeddings_shape[-2],
                        embeddings_shape[-1],
                    ),
                    # output of embeddings dtype should be same as op1
                    self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                ).to(tt_lib.tensor.Layout.TILE)
                # output config of embeddings should be same as op1_input
                .to(self.device, self.model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"])
            )

        return tt_embeddings

    def model_attention_mask(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
            extended_attention_mask = torch.clamp(
                extended_attention_mask, -100000
            )  # Limit neg value that goes into exp
            extended_attention_mask = pad_activation(extended_attention_mask)
            tt_attention_mask = (
                tt_lib.tensor.Tensor(
                    extended_attention_mask.reshape(-1).tolist(),
                    extended_attention_mask.shape,
                    self.model_config["OP8_SOFTMAX_ATTENTION_MASK_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(self.device, self.model_config["OP8_SOFTMAX_ATTENTION_MASK_MEMCFG"])
            )
        else:
            tt_attention_mask = attention_mask
        return tt_attention_mask

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
                if self.model_config["MOVE_ENCODER_OUTPUT_BOOL"]:
                    hidden_states = tt_lib.tensor.move(hidden_states)
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
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_config,
    model_location_generator,
    PERF_CNT,
    device,
):
    torch.manual_seed(1234)

    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
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

    profiler.end("processing_of_input")

    profiler.start("attention_mask_preprocessing")
    tt_attention_mask = tt_bert_model.model_attention_mask(**bert_input)
    tt_lib.device.Synchronize()
    profiler.end("attention_mask_preprocessing")

    profiler.start("embedding_input_preprocessing")
    tt_embedding_inputs = tt_bert_model.embeddings.preprocess_embedding_inputs(**bert_input)
    tt_lib.device.Synchronize()
    profiler.end("embedding_input_preprocessing")

    profiler.start("hugging_face_reference_model")
    pytorch_out = hugging_face_reference_model(**bert_input)
    profiler.end("hugging_face_reference_model")

    print(f"Running BERT model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_embedding = tt_bert_model.model_embedding(**tt_embedding_inputs)
    tt_out = tt_bert_model(1, tt_embedding, tt_attention_mask)
    tt_lib.device.Synchronize()
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out

    # Recreate inputs since activations were deallocated
    tt_attention_mask = tt_bert_model.model_attention_mask(**bert_input)
    tt_embedding_inputs = tt_bert_model.embeddings.preprocess_embedding_inputs(**bert_input)
    tt_lib.device.Synchronize()
    print(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    print(f"Running BERT model for perf measurement")

    profiler.start(f"model_run_{PERF_CNT}_times_for_inference")
    tt_embedding = tt_bert_model.model_embedding(**tt_embedding_inputs)
    tt_out = tt_bert_model(1, tt_embedding, tt_attention_mask)
    tt_lib.device.Synchronize()
    profiler.end(f"model_run_{PERF_CNT}_times_for_inference", PERF_CNT)

    # output postprocessing
    profiler.start("processing_output_to_string")

    tt_untilized_output = (
        tt_out.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch().reshape(batch, 1, seq_len, -1).to(torch.float32)
    )

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

    passing = passing_start and passing_end

    if real_input:
        if model_config["DEFAULT_DTYPE"] == tt_lib.tensor.DataType.BFLOAT8_B and not passing:
            logger.warning("Skipping post processing due to garbage output in BFP8!")
        else:
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

    profiler.print()

    # assert profiler.get("whole_model") < 60.0

    if model_config["DEFAULT_DTYPE"] == tt_lib.tensor.DataType.BFLOAT8_B and not passing:
        pytest.xfail("PCC is garbage for BFLOAT8_B. Numbers are for perf only!")

    assert passing, f"At least one start or end logits don't meet PCC requirement {pcc}"


@pytest.mark.parametrize(
    "batch, model_config_str",
    (
        (9, "BFLOAT8_B-DRAM"),
        (9, "BFLOAT16-DRAM"),
        (9, "BFLOAT8_B-L1"),
        (9, "BFLOAT16-L1"),
        (9, "MIXED_PRECISION_BATCH9"),
        (8, "MIXED_PRECISION_BATCH8"),
    ),
    ids=[
        "batch_9-BFLOAT8_B-DRAM",
        "batch_9-BFLOAT16-DRAM",
        "batch_9-BFLOAT8_B-L1",
        "batch_9-BFLOAT16-L1",
        "batch_9-MIXED_PRECISION_BATCH9",
        "batch_8-MIXED_PRECISION_BATCH8",
    ],
)
@pytest.mark.parametrize(
    "model_version, seq_len, real_input, attention_mask, token_type_ids, pcc",
    (
        (
            "phiyodr/bert-large-finetuned-squad2",
            384,
            True,
            True,
            True,
            0.97,
        ),
    ),
    ids=["BERT_LARGE"],
)
def test_bert_batch_dram(
    model_version,
    batch,
    seq_len,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_config_str,
    model_location_generator,
    request,
    device,
):
    model_config = get_model_config(model_config_str)

    # This test will run BERT-Large once with cache disabled.
    # Then it will enable cache and run BERT-Large PERF_CNT number of times.
    # Performance is reported only for PERF_CNT number of runs.
    PERF_CNT = 1

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"tt_metal/tools/profiler/logs/BERT_large_full_{request.node.callspec.id}")

    run_bert_question_and_answering_inference(
        model_version,
        batch,
        seq_len,
        real_input,
        attention_mask,
        token_type_ids,
        pcc,
        model_config,
        model_location_generator,
        PERF_CNT,
        device,
    )


@pytest.mark.parametrize(
    "batch, model_config_str",
    (
        (9, "BFLOAT8_B-DRAM"),
        (9, "BFLOAT16-DRAM"),
        (9, "BFLOAT8_B-L1"),
        (9, "BFLOAT16-L1"),
        (9, "MIXED_PRECISION_BATCH9"),
        (8, "MIXED_PRECISION_BATCH8"),
    ),
    ids=[
        "batch_9-BFLOAT8_B-DRAM",
        "batch_9-BFLOAT16-DRAM",
        "batch_9-BFLOAT8_B-L1",
        "batch_9-BFLOAT16-L1",
        "batch_9-MIXED_PRECISION_BATCH9",
        "batch_8-MIXED_PRECISION_BATCH8",
    ],
)
@pytest.mark.parametrize(
    "model_version, seq_len, real_input, attention_mask, token_type_ids, pcc",
    (
        (
            "phiyodr/bert-large-finetuned-squad2",
            384,
            True,
            True,
            True,
            0.97,
        ),
    ),
    ids=["BERT_LARGE"],
)
def test_bert_batch_dram_with_program_cache(
    use_program_cache,
    model_version,
    batch,
    seq_len,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_config_str,
    model_location_generator,
    request,
    device,
):
    model_config = get_model_config(model_config_str)

    # This test will run BERT-Large once with cache disabled.
    # Then it will enable cache and run BERT-Large PERF_CNT number of times.
    # Performance is reported only for PERF_CNT number of runs.
    PERF_CNT = 1

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_full_with_program_cache_{request.node.callspec.id}"
    )

    run_bert_question_and_answering_inference(
        model_version,
        batch,
        seq_len,
        real_input,
        attention_mask,
        token_type_ids,
        pcc,
        model_config,
        model_location_generator,
        PERF_CNT,
        device,
    )

    if batch == 8 and model_config_str == "MIXED_PRECISION_BATCH8":
        assert tt_lib.program_cache.num_entries() == 17

    else:
        assert tt_lib.program_cache.num_entries() == 16
