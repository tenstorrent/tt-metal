# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from torch.utils.data import Dataset
from typing import Any
from datasets import load_dataset
from loguru import logger


class SQUADV2Dataset(Dataset):
    """Configurable SQuad-V2 Dataset."""

    def __init__(
        self,
        dataset_question: Any,
        dataset_context: Any,
        dataset_reference: Any,
        tokenizer: Any,
        seq_len: int,
        attention_mask: bool,
        token_type_ids: bool,
    ):
        """Init and preprocess SST-2 dataset.

        Parameters
        ----------
        dataset : Any
            SQUAD-v2 dataset
        tokenizer : Any
            tokenizer object from HuggingFace
        split : str
            Which split to use i.e. ["train", "validation", "test"]
        seq_len : int
            Sequence length
        attention_mask : bool
        token_type_ids : bool
        """

        self.data = []
        for i in range(len(dataset_question)):
            self.data.append(
                (
                    tokenizer.batch_encode_plus(
                        zip(dataset_question[i], dataset_context[i]),
                        max_length=seq_len,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=attention_mask,
                        return_token_type_ids=token_type_ids,
                        return_tensors="pt",
                    ),
                    dataset_reference[i],
                    dataset_question[i],
                    dataset_context[i],
                )
            )

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X = self.data[index]
        return X


def squad_divide_chunks(dataset_question, dataset_context, dataset_reference, batch):
    dataset_question_b = []
    dataset_context_b = []
    dataset_reference_b = []
    for i in range(0, len(dataset_question), batch):
        dataset_question_b.append(dataset_question[i : i + batch])
        dataset_context_b.append(dataset_context[i : i + batch])
        dataset_reference_b.append(dataset_reference[i : i + batch])
    return dataset_question_b, dataset_context_b, dataset_reference_b


def squadv2_1K_samples_input(tokenizer, seq_len, attention_mask, token_type_ids, microbatch=8):
    squadv2_dataset = load_dataset("squad_v2", use_auth_token=False, streaming=True)["validation"]
    # squadv2_dataset = load_dataset("squad_v2", use_auth_token=True, streaming=True)["validation"]

    dataset_iter = iter(squadv2_dataset)
    dataset_question = []
    dataset_context = []
    dataset_reference = []

    for _ in range(2048):
        dataset_sgl = next(dataset_iter)
        if len(dataset_sgl["answers"]["text"]) > 0:
            dataset_question.append(dataset_sgl["question"])
            dataset_context.append(dataset_sgl["context"])
            dataset_reference.append({"answers": dataset_sgl["answers"], "id": dataset_sgl["id"]})
        if len(dataset_question) == 1024:
            logger.info("SQuADv2 1024 samples load ..done")
            break

    dataset_question, dataset_context, dataset_reference = squad_divide_chunks(
        dataset_question, dataset_context, dataset_reference, microbatch
    )
    dataset_processed = SQUADV2Dataset(
        dataset_question,
        dataset_context,
        dataset_reference,
        tokenizer=tokenizer,
        seq_len=seq_len,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    return dataset_processed


def squadv2_answer_decode_batch(
    HF_model, tokenizer, nlp, references, cpu_out, tt_untilized_output, BATCH_SIZE, question, context
):
    tt_predictions = []
    cpu_predictions = []

    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
    preprocess_params["max_seq_len"] = 384
    input_q = {"context": context, "question": question}
    examples = nlp._args_parser(input_q)

    for i in range(BATCH_SIZE):
        logger.info(f"--REF-- {references[i]['answers']['text']}")

        answer_start_scores = cpu_out["start_logits"][i]
        answer_end_scores = cpu_out["end_logits"][i]

        tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)[i]
        tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)[i]

        model_input = next(nlp.preprocess(examples[0][i], **preprocess_params))
        single_input = {
            "data": (
                model_input["input_ids"],
                model_input["attention_mask"],
                model_input["token_type_ids"],
            ),
            "example": model_input["example"],
            "inputs": model_input,
        }

        pt_res = {
            "start": answer_start_scores,
            "end": answer_end_scores,
            "example": single_input["example"],
            **single_input["inputs"],
        }
        cpu_answer_nlp = nlp.postprocess([pt_res], **postprocess_params)["answer"]

        tt_res = {
            "start": tt_start_logits,
            "end": tt_end_logits,
            "example": single_input["example"],
            **single_input["inputs"],
        }
        tt_answer_nlp = nlp.postprocess([tt_res], **postprocess_params)["answer"]

        logger.info(f"--CPU-- {cpu_answer_nlp}")
        logger.info(f"--TT--- {tt_answer_nlp}")
        logger.info(f"=======")
        cpu_predictions.append(
            {"prediction_text": cpu_answer_nlp, "id": references[i]["id"], "no_answer_probability": 0.0}
        )
        tt_predictions.append(
            {"prediction_text": tt_answer_nlp, "id": references[i]["id"], "no_answer_probability": 0.0}
        )

    return cpu_predictions, tt_predictions
