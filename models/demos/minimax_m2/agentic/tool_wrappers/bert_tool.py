# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BERTTool: Wraps TTNN BERT Large for extractive QA (SQuAD2 fine-tuned).

Uses `ttnn_optimized_bert` (no sharding, batch_size=1) for the forward pass.
The HuggingFace pipeline handles tokenization and answer span extraction.
"""

import torch
import transformers
from loguru import logger
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.bert.tt import ttnn_optimized_bert

BERT_MODEL_NAME = "phiyodr/bert-large-finetuned-squad2"
SEQUENCE_SIZE = 384
BATCH_SIZE = 1


class BERTTool:
    """
    TTNN-accelerated BERT Large extractive QA wrapper.

    Given a question and a context passage, returns the extracted answer span.
    """

    def __init__(self, mesh_device):
        self.device = mesh_device

        logger.info("Loading BERT Large QA model...")
        self.hf_model = BertForQuestionAnswering.from_pretrained(BERT_MODEL_NAME, torchscript=False)
        self.hf_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.config = self.hf_model.config

        self.nlp = pipeline("question-answering", model=self.hf_model, tokenizer=self.tokenizer)

        self.parameters = preprocess_model_parameters(
            model_name=f"ttnn_{BERT_MODEL_NAME}_optimized",
            initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
                BERT_MODEL_NAME, torchscript=False
            ).eval(),
            custom_preprocessor=ttnn_optimized_bert.custom_preprocessor,
            device=self.device,
        )
        logger.info("BERT Large QA ready.")

    def _positional_ids(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(self.config.max_position_embeddings, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0)[:, :seq_len]
        return position_ids.expand_as(input_ids)

    def qa(self, question: str, context: str) -> str:
        """
        Extract the answer to *question* from *context*.

        Returns the extracted answer span as a string.
        """
        bert_input = self.tokenizer.batch_encode_plus(
            [(question, context)],
            max_length=SEQUENCE_SIZE,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        position_ids = self._positional_ids(bert_input.input_ids)

        ttnn_inputs = ttnn_optimized_bert.preprocess_inputs(
            bert_input["input_ids"],
            bert_input["token_type_ids"],
            position_ids,
            bert_input["attention_mask"],
            device=self.device,
        )

        tt_output = ttnn_optimized_bert.bert_for_question_answering(
            self.config,
            *ttnn_inputs,
            parameters=self.parameters,
        )

        # On a multi-chip MeshDevice the output tensor lives on N chips.
        # ConcatMeshToTensor gathers it; since inference is replicated (not
        # sharded), both chips produce the same result, so we keep only the
        # first BATCH_SIZE rows.
        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if num_devices > 1 else None
        tt_output = (
            ttnn.to_torch(ttnn.from_device(tt_output), mesh_composer=mesh_composer)[: BATCH_SIZE * num_devices]
            .reshape(BATCH_SIZE * num_devices, 1, SEQUENCE_SIZE, -1)[:BATCH_SIZE]
            .to(torch.float32)
        )

        start_logits = tt_output[..., :, 0].squeeze(1)
        end_logits = tt_output[..., :, 1].squeeze(1)

        # Use HF pipeline postprocessing to decode the answer span
        preprocess_params, _, postprocess_params = self.nlp._sanitize_parameters()
        preprocess_params["max_seq_len"] = SEQUENCE_SIZE
        inputs = self.nlp._args_parser({"context": [context], "question": [question]})
        model_input = next(self.nlp.preprocess(inputs[0][0], **preprocess_params))
        tt_res = {
            "start": start_logits[0],
            "end": end_logits[0],
            "example": model_input["example"],
            **model_input,
        }
        answer = self.nlp.postprocess([tt_res], **postprocess_params)
        return answer.get("answer", "")
