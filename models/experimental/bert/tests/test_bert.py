# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import pytest
from loguru import logger
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

import ttnn
from models.experimental.bert.tt.embeddings import PytorchEmbeddings
from models.experimental.bert.tt.bert_encoder import TtBertEncoder
from models.experimental.bert.fused_ops.linear import Linear
from tt_lib.utils import pad_activation, pad_weight
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


class TtBertShared(torch.nn.Module):
    @abstractmethod
    def __init__(self, config, hugging_face_reference_model, device):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

        # So far on CPU until we add embeddings support on device
        self.embeddings = PytorchEmbeddings(hugging_face_reference_model)
        self.get_extended_attention_mask = hugging_face_reference_model.get_extended_attention_mask

        self.encoders = torch.nn.ModuleList(
            [TtBertEncoder(config, encoder_idx, state_dict, device) for encoder_idx in range(config.num_hidden_layers)]
        )
        self.device = device

    @abstractmethod
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
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
        hidden_states = tt_embeddings
        for encoder in self.encoders:
            hidden_states = encoder(hidden_states, tt_attention_mask)
        return hidden_states


class TtBertForQuestionAnswering(TtBertShared):
    def __init__(self, config, hugging_face_reference_model, device):
        super().__init__(config, hugging_face_reference_model, device)

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        encoder_output = super().forward(input_ids, attention_mask, token_type_ids)
        return self.qa_linear(encoder_output)


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
):
    torch.manual_seed(1234)

    model_name = str(model_location_generator(model_version, model_subdir="Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    hugging_face_reference_model.eval()
    tt_bert_model = TtBertForQuestionAnswering(
        hugging_face_reference_model.config, hugging_face_reference_model, device
    )

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

    # tt_bert_input = ttnn.Tensor(pad_activation(bert_input).reshape(-1).tolist(), bert_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).to(ttnn.TILE_LAYOUT)
    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    if real_input:
        pytorch_out = hugging_face_reference_model(**bert_input)
        tt_out = tt_bert_model(**bert_input).cpu()
    else:
        pytorch_out = hugging_face_reference_model(bert_input)
        tt_out = tt_bert_model(bert_input).cpu()
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
    assert passing_start and passing_end, f"At least one start or end logits don't meet PCC requirement {pcc}"
    # start_logit_match = (abs(tt_start_logits - pytorch_start_logits) < 0.1).all().item()
    # if not start_logit_match:
    #     print("Start logits don't match")
    # else:
    #     print("Start logits match")

    # end_logit_match = (abs(tt_end_logits - pytorch_end_logits) < 0.1).all().item()

    # if not end_logit_match:
    #     print("End logits don't match")
    # else:
    #     print("End logits match")

    # assert start_logit_match and end_logit_match, "At least one of start or end logits don't match to an absolute difference of 0.1"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, real_input, attention_mask, token_type_ids, pcc",
    (
        ("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, True, True, True, 0.99),
        ("phiyodr/bert-base-finetuned-squad2", 1, 128, True, True, True, 0.99),
        (
            "phiyodr/bert-large-finetuned-squad2",
            1,
            384,
            True,
            True,
            True,
            0.98,
        ),  # Placeholder PCC until issues are resolved
    ),
)
def test_bert_question_and_answering_inference(
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
    # enable_persistent_kernel_cache()
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
    )
