# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

import time
import ttnn
from models.experimental.bert_large_perf.tt.embeddings import PytorchEmbeddings
from models.experimental.bert_large_perf.tt.bert_encoder import TtBertEncoder
from models.experimental.bert_large_perf.fused_ops.linear import Linear
from tt_lib.utils import pad_activation, pad_weight
from models.utility_functions import (
    enable_persistent_kernel_cache,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
    disable_persistent_kernel_cache,
)
from models.utility_functions import profiler
from models.experimental.bert_large_perf.fused_ops.layernorm import create_var_scaler


class TtBertBatchDram(torch.nn.Module):
    def __init__(self, config, hugging_face_reference_model, var_scaler, device):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

        self.hidden_states_list = []
        self.tt_attention_mask_list = []

        # So far on CPU until we add embeddings support on device
        self.embeddings = PytorchEmbeddings(hugging_face_reference_model)
        self.get_extended_attention_mask = hugging_face_reference_model.get_extended_attention_mask

        self.encoders = torch.nn.ModuleList(
            [
                TtBertEncoder(config, encoder_idx, state_dict, var_scaler, device)
                for encoder_idx in range(config.num_hidden_layers)
            ]
        )

        num_classes, hidden_size = state_dict["qa_outputs.weight"].shape

        weight = pad_weight(state_dict["qa_outputs.weight"])
        weight = (
            ttnn.Tensor(
                weight.reshape(-1).tolist(),
                weight.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        bias = pad_weight(state_dict["qa_outputs.bias"])
        bias = (
            ttnn.Tensor(
                bias.reshape(-1).tolist(),
                bias.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

        # QA linear
        self.qa_linear = Linear(hidden_size, 32, weight, bias, device)

        self.device = device

    def forward(self, PERF_CNT, input_ids, attention_mask=None, token_type_ids=None):
        for i in range(PERF_CNT):
            profiler.start("_calc_embeddings")

            embeddings = self.embeddings(input_ids, token_type_ids)
            if attention_mask is not None:
                extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
                extended_attention_mask = torch.clamp(
                    extended_attention_mask, -100000
                )  # Limit neg value that goes into exp
                extended_attention_mask = pad_activation(extended_attention_mask)
                tt_attention_mask = ttnn.Tensor(
                    extended_attention_mask.reshape(-1).tolist(),
                    extended_attention_mask.shape,
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                ).to(ttnn.TILE_LAYOUT)
                tt_attention_mask = tt_attention_mask.to(self.device)
            else:
                tt_attention_mask = attention_mask

            # Add to list mask
            self.tt_attention_mask_list.append(tt_attention_mask)

            # Convert to ll buda tensor
            pad_embeddings = pad_activation(embeddings)
            tt_embeddings = ttnn.Tensor(
                pad_embeddings.reshape(-1).tolist(),
                (
                    pad_embeddings.shape[0],
                    1,
                    pad_embeddings.shape[-2],
                    pad_embeddings.shape[-1],
                ),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
            tt_embeddings = tt_embeddings.to(self.device)
            hidden_states = tt_embeddings  # pad_embeddings #

            self.hidden_states_list.append(hidden_states)
            profiler.end("_calc_embeddings")

        print(f"Num encoders {len(self.encoders)}")
        tt_out_list = []

        for i in range(PERF_CNT):
            print(f"Running BERT model {i}")
            profiler.start("_run_encoders")

            hidden_states = self.hidden_states_list[i]
            attention_mask = self.tt_attention_mask_list[i]

            for encoder in self.encoders:
                profiler.start("__one_encoder")
                hidden_states = encoder(hidden_states, attention_mask)
                profiler.end("__one_encoder")

            profiler.end("_run_encoders")

            profiler.start("_qa_linear")
            res = self.qa_linear(hidden_states)
            profiler.end("_qa_linear")

            tt_out_list.append(res)

        return tt_out_list


def run_bert_question_and_answering_inference(
    device,
    model_version,
    batch,
    seq_len,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_location_generator,
    PERF_CNT,
):
    torch.manual_seed(1234)

    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    hugging_face_reference_model.eval()
    var_scaler = create_var_scaler(
        seq_len,
        hugging_face_reference_model.config.hidden_size,
        hugging_face_reference_model.config.layer_norm_eps,
        device,
    )
    tt_bert_model = TtBertBatchDram(
        hugging_face_reference_model.config,
        hugging_face_reference_model,
        var_scaler,
        device,
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

    # tt_bert_input = ttnn.Tensor(pad_activation(bert_input).reshape(-1).tolist(), bert_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).to(ttnn.TILE_LAYOUT)
    profiler.start("hugging_face_reference_model")
    pytorch_out = hugging_face_reference_model(**bert_input)
    profiler.end("hugging_face_reference_model")

    print(f"Running BERT model once to fill caches -> disable profiler")
    profiler.disable()

    tt_out_list = tt_bert_model(1, **bert_input)

    # the first inference pass
    tt_out = tt_out_list[0].cpu()
    tt_untilized_output = tt_out.to(ttnn.ROW_MAJOR_LAYOUT).to_torch().reshape(batch, 1, seq_len, -1)

    print(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    print(f"Running BERT model for perf measurement")

    profiler.start("whole_model")
    tt_out_list = tt_bert_model(PERF_CNT, **bert_input)
    profiler.end("whole_model", PERF_CNT)

    # output postprocessing
    for i in range(PERF_CNT):
        profiler.start("processing_output_to_string")

        tt_out = tt_out_list[i].cpu()
        tt_untilized_output = tt_out.to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float().reshape(batch, 1, seq_len, -1)

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
                logger.info(f"PL: {pl_answer}")

        profiler.end("processing_output_to_string")

    del tt_out_list

    profiler.print()

    # assert profiler.get("whole_model") < 60.0
    assert passing_start and passing_end, f"At least one start or end logits don't meet PCC requirement {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, real_input, attention_mask, token_type_ids, pcc",
    (("phiyodr/bert-large-finetuned-squad2", 9, 384, True, True, True, 0.98),),
)
def test_bert_batch_dram(
    device,
    model_version,
    batch,
    seq_len,
    real_input,
    attention_mask,
    token_type_ids,
    pcc,
    model_location_generator,
):
    # This test will run BERT-Large once with cache disabled.
    # Then it will enable cache and run BERT-Large PERF_CNT number of times.
    # Performance is reported only for PERF_CNT number of runs.
    PERF_CNT = 1

    disable_persistent_kernel_cache()

    run_bert_question_and_answering_inference(
        device,
        model_version,
        batch,
        seq_len,
        real_input,
        attention_mask,
        token_type_ids,
        pcc,
        model_location_generator,
        PERF_CNT,
    )
