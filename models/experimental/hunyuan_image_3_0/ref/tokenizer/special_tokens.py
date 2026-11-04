# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Multimodal special-token ID map for HunyuanImage-3.0 T2I sequences.
#
# Mapping to HF upstream (tokenizer_wrapper.py / tokenization_hunyuan_image_3.py)
# -----------------------------------------------------------------------------
#  Step | HF                                           | This port (host)
#  -----+----------------------------------------------+----------------------------------
#   1   | added_tokens_encoder from tokenizer.json     | tokenizer.added_tokens_encoder
#   2   | setup_special_tokens core IDs              | build_special_tokens → SpecialTokens
#   3   | <img_ratio_i> / <img_size_N> lookup          | ratio_token_id / size_token_id
#   4   | validate required multimodal tokens        | validate_special_tokens
#
# Core tokens: <boi>, <eoi>, <img>, <cfg>, <timestep>, <guidance>, ratio/size tokens.
#
# References
# ----------
#   ref/tokenizer/assets/tokenizer.json  — BPE vocab + added-token table
#   ref/tokenizer/chat_template.py       — consumes SpecialTokens during encode
#   ref/tokenizer/hunyuan_tokenizer.py   — HunyuanTokenizer.from_pretrained wiring

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerFast

# Tokens required for text-to-image sequence building (steps 3–5).
CORE_SPECIAL_TOKEN_STRINGS = (
    "<boi>",
    "<eoi>",
    "<img>",
    "<cfg>",
    "<timestep>",
    "<guidance>",
    "<joint_img_sep>",
    "<answer>",
    "</answer>",
    "<recaption>",
    "</recaption>",
    "<think>",
    "</think>",
)

# Bundled checkpoint has <img_ratio_0> … <img_ratio_32> (33 ratios).
NUM_RATIO_TOKENS = 33

# Common image base sizes referenced during encoding.
COMMON_IMAGE_SIZES = (512, 768, 1024, 1280, 2048)


def _require_id(tokenizer: PreTrainedTokenizerFast, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk = tokenizer.unk_token_id
    if token_id is None or (unk is not None and token_id == unk):
        raise KeyError(f"Special token not in vocab: {token!r} -> {token_id}")
    return int(token_id)


@dataclass(frozen=True)
class SpecialTokens:
    """Cached multimodal special-token IDs."""

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    boi_token_id: int
    eoi_token_id: int
    img_token_id: int
    cfg_token_id: int
    timestep_token_id: int
    guidance_token_id: int
    joint_img_sep_token_id: int
    answer_token_id: int
    end_answer_token_id: int
    recaption_token_id: int
    end_recaption_token_id: int
    think_token_id: int
    end_think_token_id: int
    ratio_token_offset: int
    num_ratio_tokens: int
    special_token_map: dict[str, int] = field(repr=False)
    model_version: str = "HunyuanImage-3.0"

    def ratio_token(self, ratio_idx: int) -> str:
        if not 0 <= ratio_idx < self.num_ratio_tokens:
            raise IndexError(f"ratio_idx must be in [0, {self.num_ratio_tokens}), got {ratio_idx}")
        return f"<img_ratio_{ratio_idx}>"

    def ratio_token_id(self, ratio_idx: int) -> int:
        return self.special_token_map[self.ratio_token(ratio_idx)]

    def size_token(self, size: int) -> str:
        return f"<img_size_{size}>"

    def size_token_id(self, size: int) -> int:
        token = self.size_token(size)
        if token not in self.special_token_map:
            raise KeyError(f"Size token not in vocab: {token!r}")
        return self.special_token_map[token]

    def all_ratio_token_ids(self) -> list[int]:
        return [self.ratio_token_id(i) for i in range(self.num_ratio_tokens)]

    def as_dict(self) -> dict[str, int]:
        """Human-readable map for sanity-check printing."""
        out = {
            "bos": self.bos_token_id,
            "eos": self.eos_token_id,
            "pad": self.pad_token_id,
            "boi": self.boi_token_id,
            "eoi": self.eoi_token_id,
            "img": self.img_token_id,
            "cfg": self.cfg_token_id,
            "timestep": self.timestep_token_id,
            "guidance": self.guidance_token_id,
            "joint_img_sep": self.joint_img_sep_token_id,
            "answer": self.answer_token_id,
            "end_answer": self.end_answer_token_id,
            "recaption": self.recaption_token_id,
            "end_recaption": self.end_recaption_token_id,
            "think": self.think_token_id,
            "end_think": self.end_think_token_id,
            "ratio_token_offset (<img_ratio_0>)": self.ratio_token_offset,
        }
        for size in COMMON_IMAGE_SIZES:
            key = f"img_size_{size}"
            token = self.size_token(size)
            if token in self.special_token_map:
                out[key] = self.special_token_map[token]
        return out


def _count_ratio_tokens(special_token_map: dict[str, int]) -> int:
    count = 0
    while f"<img_ratio_{count}>" in special_token_map:
        count += 1
    return count


def build_special_tokens(
    tokenizer: PreTrainedTokenizerFast,
    *,
    model_version: str = "HunyuanImage-3.0",
) -> SpecialTokens:
    special_token_map = dict(tokenizer.added_tokens_encoder)
    num_ratio_tokens = _count_ratio_tokens(special_token_map)
    if num_ratio_tokens == 0:
        raise RuntimeError("No <img_ratio_*> tokens found in bundled tokenizer")

    return SpecialTokens(
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        boi_token_id=_require_id(tokenizer, "<boi>"),
        eoi_token_id=_require_id(tokenizer, "<eoi>"),
        img_token_id=_require_id(tokenizer, "<img>"),
        cfg_token_id=_require_id(tokenizer, "<cfg>"),
        timestep_token_id=_require_id(tokenizer, "<timestep>"),
        guidance_token_id=_require_id(tokenizer, "<guidance>"),
        joint_img_sep_token_id=_require_id(tokenizer, "<joint_img_sep>"),
        answer_token_id=_require_id(tokenizer, "<answer>"),
        end_answer_token_id=_require_id(tokenizer, "</answer>"),
        recaption_token_id=_require_id(tokenizer, "<recaption>"),
        end_recaption_token_id=_require_id(tokenizer, "</recaption>"),
        think_token_id=_require_id(tokenizer, "<think>"),
        end_think_token_id=_require_id(tokenizer, "</think>"),
        ratio_token_offset=_require_id(tokenizer, "<img_ratio_0>"),
        num_ratio_tokens=num_ratio_tokens,
        special_token_map=special_token_map,
        model_version=model_version,
    )


def validate_special_tokens(
    special: SpecialTokens,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, Any]:
    """Sanity-check special-token map (step 2)."""
    errors: list[str] = []

    for name, token_id in special.as_dict().items():
        if token_id is None:
            errors.append(f"{name}: missing id")

    # ratio_token_offset must match ratio_token_id(0)
    if special.ratio_token_id(0) != special.ratio_token_offset:
        errors.append(f"ratio_token_offset mismatch: {special.ratio_token_offset} != ratio_token_id(0)")

    # Ratio IDs should be strictly increasing (contiguous block in vocab).
    ratio_ids = special.all_ratio_token_ids()
    if ratio_ids != sorted(ratio_ids):
        errors.append("ratio token ids are not strictly increasing")
    if len(set(ratio_ids)) != len(ratio_ids):
        errors.append("duplicate ratio token ids")

    # Round-trip token strings for core specials.
    roundtrip: dict[str, str] = {}
    for token_str in CORE_SPECIAL_TOKEN_STRINGS:
        tid = special.special_token_map.get(token_str)
        if tid is None:
            errors.append(f"missing core token: {token_str}")
            continue
        back = tokenizer.convert_ids_to_tokens(tid)
        roundtrip[token_str] = back
        if back != token_str:
            errors.append(f"roundtrip failed for {token_str!r}: got {back!r}")

    size_ids = {}
    for size in COMMON_IMAGE_SIZES:
        try:
            size_ids[size] = special.size_token_id(size)
        except KeyError:
            size_ids[size] = None

    return {
        "model_version": special.model_version,
        "num_ratio_tokens": special.num_ratio_tokens,
        "num_added_tokens": len(special.special_token_map),
        "core_token_map": special.as_dict(),
        "ratio_token_ids_sample": {
            "0": special.ratio_token_id(0),
            "1": special.ratio_token_id(1),
            "32": special.ratio_token_id(min(32, special.num_ratio_tokens - 1)),
        },
        "size_token_ids": size_ids,
        "roundtrip_core_tokens": roundtrip,
        "errors": errors,
        "special_tokens_ok": len(errors) == 0,
    }
