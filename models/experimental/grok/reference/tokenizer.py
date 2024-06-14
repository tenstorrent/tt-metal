# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

#################################################################################################################
# Link: https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
# NOTE: changed the device from CUDA to CPU and dtype to float32
#################################################################################################################

# coding=utf-8
# Copyright 2023 Mistral AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sentencepiece import SentencePieceProcessor
from pathlib import Path
from typing import List


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)
