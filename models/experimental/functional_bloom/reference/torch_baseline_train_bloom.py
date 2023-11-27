# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import BloomForCausalLM, BloomTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # The labels are the input IDs shifted by one token
        labels = encoding["input_ids"].clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = self.tokenizer.pad_token_id

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
        }


def preprocess_data(texts, tokenizer, max_length=512):
    return TextDataset(texts, tokenizer, max_length)


if __name__ == "__main__":
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

    sample_data_set = ["Hello World", "Bonjour le mode"]
    dataset = preprocess_data(sample_data_set, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=eval_dataset,
    )

    trainer.train()
