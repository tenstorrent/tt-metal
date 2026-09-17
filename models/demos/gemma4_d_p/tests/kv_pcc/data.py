# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained input, GPU references, and baseline for one KV accuracy test."""

import hashlib
import json
import pathlib

from loguru import logger

ROOT_TRACE = pathlib.Path("/mnt/models/huggingface/gpu_traces/gemma4_d_p")
DEFAULT_DATASET = ROOT_TRACE / "hf-gemma4-31b-36db66e9-262144tok"


class KvPccDataset:
    def __init__(self, directory):
        self.directory = pathlib.Path(directory).expanduser()
        self.baseline_path = self.directory / "baseline.json"
        for filename in ("input.txt", "metadata.json", "index.json", "baseline.json"):
            path = self.directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete KV PCC dataset: missing {path}")
        raw_text = (self.directory / "input.txt").read_bytes()
        self.text = raw_text.decode("utf-8")
        self.input_text_sha256 = hashlib.sha256(raw_text).hexdigest()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.index = json.loads((self.directory / "index.json").read_text())

    def reference_blocks(self, context_len):
        """Select and validate a contiguous reference prefix, retaining source block boundaries."""
        if context_len <= 0 or context_len > len(self.metadata["token_ids"]):
            raise ValueError(f"KV PCC dataset {self.directory} does not cover {context_len} input tokens")
        if self.index["n_layers"] != self.metadata["n_layers"]:
            raise ValueError(f"KV PCC dataset {self.directory}: metadata/index layer counts differ")
        streams = []
        root = self.directory.resolve()
        for layer in range(self.metadata["n_layers"]):
            blocks = sorted(
                self.index["tensor_streams"][f"kv_post_transform_layer_{layer}"]["chunks"],
                key=lambda block: block["row_start"],
            )
            selected, next_row = [], 0
            for block in blocks:
                if next_row == context_len:
                    break
                if block["row_start"] != next_row or block["row_end"] <= next_row:
                    raise ValueError(f"layer {layer}: reference blocks must be contiguous from zero")
                path = self.directory / block["path"]
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"KV reference must be inside its dataset: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing GPU KV reference: {path}")
                selected.append(block)
                next_row = min(block["row_end"], context_len)
            if next_row != context_len:
                raise ValueError(f"layer {layer}: reference does not cover {context_len} tokens")
            streams.append(selected)
        return streams

    def token_ids(self, model_path, context_len, *, model_n_layers):
        """Tokenize this container's text; never substitute or repeat another input."""
        from transformers import AutoTokenizer

        if self.metadata["n_layers"] != model_n_layers:
            raise ValueError("KV reference layer count differs from model")
        if context_len <= 0 or context_len > len(self.metadata["token_ids"]):
            raise ValueError(f"KV PCC dataset {self.directory} does not cover {context_len} input tokens")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokens = tokenizer.encode(self.text)
        if len(tokens) < context_len:
            raise ValueError(f"KV PCC input.txt has {len(tokens)} tokens; requested {context_len}: {self.directory}")
        tokens = tokens[:context_len]
        # Compare actual token IDs with the GPU trace input, not just text hashes.
        if tokens != self.metadata["token_ids"][:context_len]:
            raise ValueError(f"KV PCC input.txt token IDs differ from GPU traces: {self.directory}")
        logger.info(
            f"[kv_pcc] Input: {self.directory / 'input.txt'}; "
            f"token IDs match the GPU reference for the first {context_len} tokens"
        )
        return tokens
