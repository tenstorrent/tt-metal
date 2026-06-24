# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Message list → T2I / I2I token sequence (pretrain template, gen_image + cond paths).
#
# Mapping to HF upstream (hunyuan_image_3/tokenization_hunyuan_image_3.py)
# -------------------------------------------------------------------------
#  Step | HF                                           | This port (host)
#  -----+----------------------------------------------+----------------------------------
#   1   | build message list (user + gen_image)        | apply_chat_template message assembly
#   2   | apply_general_template                       | ChatTemplateEncoder.apply_general_template
#   3   | encode text sections                       | tokenizer.encode per text section
#   4   | encode gen_image section (BOI/ratio/size/  | _encode_gen_image_section
#       |   img placeholders/timestep/EOI)           |
#   5   | gen_image_mask + scatter indices           | TokenizerEncodeOutput side tensors
#   6   | CFG cond/uncond batch stack                | _stack_batch (uncond_p=0 / 1)
#
# Scope: mode='gen_image' (T2I and I2I with cond images) and mode='gen_text' (recaption /
# think AR prefix), sequence_template='pretrain'|'instruct'.
#
# References
# ----------
#   ref/tokenizer/image_info.py      — ImageInfo latent grid metadata
#   ref/tokenizer/special_tokens.py  — BOI/EOI/img/cfg/timestep token IDs
#   ref/tokenizer/hunyuan_tokenizer.py — HunyuanTokenizer public wrapper

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from .image_info import CondImage, ImageInfo, JointImageInfo
from .special_tokens import SpecialTokens


@dataclass
class TokenizerEncodeOutput:
    tokens: torch.Tensor
    gen_image_mask: torch.Tensor | None = None
    vae_image_mask: torch.Tensor | None = None
    vit_image_mask: torch.Tensor | None = None
    gen_timestep_scatter_index: torch.Tensor | None = None
    gen_timestep_r_scatter_index: torch.Tensor | None = None
    cond_timestep_scatter_index: torch.Tensor | None = None
    guidance_scatter_index: torch.Tensor | None = None
    text_slices: list[slice] | None = None
    gen_image_slices: list[slice] | None = None
    vae_image_slices: list[slice] | None = None
    vit_image_slices: list[slice] | None = None
    joint_image_slices: list[slice] | None = None
    all_image_slices: list[slice] | list[list[slice]] | None = None
    real_pos: torch.Tensor | None = None

    def keys(self):
        return (
            "tokens",
            "gen_image_mask",
            "vae_image_mask",
            "vit_image_mask",
            "gen_timestep_scatter_index",
            "gen_timestep_r_scatter_index",
            "cond_timestep_scatter_index",
            "guidance_scatter_index",
            "text_slices",
            "gen_image_slices",
            "vae_image_slices",
            "vit_image_slices",
            "joint_image_slices",
            "all_image_slices",
            "real_pos",
        )


def _parse_extra_token_pos(extra_token_pos: dict, prefix: str, tokens: torch.Tensor):
    start_key, end_key = f"<{prefix}>_start", f"<{prefix}>_end"
    if start_key not in extra_token_pos or end_key not in extra_token_pos:
        return [], None
    image_slices = [slice(start, end + 1) for start, end in zip(extra_token_pos[start_key], extra_token_pos[end_key])]
    if not image_slices:
        return [], None
    image_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for image_slice in image_slices:
        image_mask[image_slice] = True
    return image_slices, image_mask


@dataclass(frozen=True)
class InstructConversation:
    """Minimal hunyuan-image-3 conversation template (upstream conv_templates)."""

    roles: tuple[str, str] = ("User", "Assistant")
    sep: str = "\n\n"
    sep2: str = "<|endoftext|>"

    def role_prefix(self, role: str) -> str:
        return f"{role}: "


class ChatTemplateEncoder:
    """Build Hunyuan T2I / I2I ``input_ids`` from prompts (host-side)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        special: SpecialTokens,
        *,
        sequence_template: str = "pretrain",
    ) -> None:
        self.tokenizer = tokenizer
        self.special = special
        self.sequence_template = sequence_template
        self._instruct_conv = InstructConversation()

    @property
    def bos_token_id(self) -> int:
        return self.special.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.special.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.special.pad_token_id

    @property
    def cfg_token_id(self) -> int:
        return self.special.cfg_token_id

    def encode_text(
        self,
        *texts: str,
        uncond_enabled: bool | list[bool] | None = None,
        uncond_p: float | None = None,
        max_length: int | None = None,
    ) -> list[int]:
        if uncond_enabled is None:
            uncond_enabled = [True] * len(texts)
        elif isinstance(uncond_enabled, bool):
            uncond_enabled = [uncond_enabled] * len(texts)
        do_uncond_drop = uncond_p is not None and random.random() < uncond_p

        text_tokens: list[int] = []
        cum_length = 0
        for text, uncond_flag in zip(texts, uncond_enabled):
            if max_length is not None and cum_length >= max_length:
                break
            text_token = self.tokenizer.encode(text, add_special_tokens=False)
            if uncond_flag and do_uncond_drop:
                text_token = [self.cfg_token_id] * len(text_token)
            if max_length is not None and (cum_length + len(text_token)) > max_length:
                text_token = text_token[: max_length - cum_length]
            text_tokens.extend(text_token)
            cum_length += len(text_token)
        return text_tokens

    def _add_image_meta_info_token(
        self,
        token_seq: list[int],
        token_count: int,
        extra_token_pos: dict[str, list[int]],
        *,
        add_timestep_token: bool,
        add_timestep_r_token: bool,
        add_image_shape_token: bool,
        add_guidance_token: bool,
        base_size: int | None,
        ratio_idx: int | None,
        image_type: str | None,
    ) -> int:
        if add_image_shape_token:
            token_seq.extend([self.special.size_token_id(base_size), self.special.ratio_token_id(ratio_idx)])
            token_count += 2
        if add_timestep_token:
            token_seq.append(self.special.timestep_token_id)
            extra_token_pos["timestep"].append(token_count)
            if image_type == "gen_image":
                extra_token_pos["gen_timestep"].append(token_count)
            elif image_type in ("cond_joint_image", "cond_vae_image"):
                extra_token_pos["cond_timestep"].append(token_count)
            elif image_type is not None:
                raise ValueError(f"Unsupported image_type for timestep: {image_type!r}")
            token_count += 1
        if add_guidance_token:
            token_seq.append(self.special.guidance_token_id)
            extra_token_pos["guidance"].append(token_count)
            token_count += 1
        if add_timestep_r_token:
            tid = self.special.special_token_map.get("<timestep_r>")
            if tid is None:
                raise KeyError("add_timestep_r_token=True but <timestep_r> not in vocab")
            token_seq.append(int(tid))
            extra_token_pos["gen_timestep_r"].append(token_count)
            token_count += 1
        return token_count

    def encode_sequence(
        self,
        template: str,
        token_source: dict[str, list],
        *,
        total_length: int | None = None,
        add_eos: bool | str = True,
        add_pad: bool | str = True,
        add_bos: bool = True,
        drop_last: bool | str = "auto",
        add_timestep_token: bool = False,
        add_timestep_r_token: bool = False,
        add_guidance_token: bool = False,
        add_image_shape_token: bool = False,
    ) -> tuple[list[int], dict[str, list[int]]]:
        keys = template.split("-")
        index_indicator = {k: 0 for k in token_source}
        token_seq: list[int] = []
        token_count = 0
        extra_token_pos: dict[str, list[int]] = defaultdict(list)
        drop_last_break = False

        if add_bos:
            token_seq.append(self.bos_token_id)
            token_count += 1

        for key in keys:
            source = token_source[key][index_indicator[key]]
            index_indicator[key] += 1

            if key == "text":
                token_seq.extend(source)
                extra_token_pos["<text>_start"].append(token_count)
                token_count += len(source)
                extra_token_pos["<text>_end"].append(token_count - 1)

            elif key == "gen_image":
                extra_count = (
                    2
                    + (1 if source.get("timestep", add_timestep_token) else 0)
                    + (1 if source.get("timestep_r", add_timestep_r_token) else 0)
                    + (1 if source.get("guidance", add_guidance_token) else 0)
                    + (2 if source.get("image_shape", add_image_shape_token) else 0)
                )
                if drop_last is True and total_length is not None:
                    if token_count + extra_count + source["length"] > total_length:
                        drop_last_break = True
                        break
                token_seq.append(self.special.boi_token_id)
                extra_token_pos["boi"].append(token_count)
                token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq,
                    token_count,
                    extra_token_pos,
                    add_timestep_token=source.get("timestep", add_timestep_token),
                    add_timestep_r_token=source.get("timestep_r", add_timestep_r_token),
                    add_image_shape_token=source.get("image_shape", add_image_shape_token),
                    add_guidance_token=source.get("guidance", add_guidance_token),
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                token_seq.extend([self.special.img_token_id] * source["length"])
                extra_token_pos["<img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"]
                extra_token_pos["<img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                token_seq.append(self.special.eoi_token_id)
                extra_token_pos["eoi"].append(token_count)
                token_count += 1

            elif key == "cond_joint_image":
                assert isinstance(source["length"], list) and len(source["length"]) == 2
                extra_count = (
                    2
                    + 1
                    + (1 if source.get("timestep", add_timestep_token) else 0)
                    + (2 if source.get("image_shape", add_image_shape_token) else 0)
                )
                if drop_last is True and total_length is not None:
                    if token_count + extra_count + sum(source["length"]) > total_length:
                        drop_last_break = True
                        break
                token_seq.append(self.special.boi_token_id)
                extra_token_pos["boi"].append(token_count)
                token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq,
                    token_count,
                    extra_token_pos,
                    add_timestep_token=source.get("timestep", add_timestep_token),
                    add_timestep_r_token=False,
                    add_image_shape_token=source.get("image_shape", add_image_shape_token),
                    add_guidance_token=False,
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                token_seq.extend([self.special.img_token_id] * source["length"][0])
                extra_token_pos["<vae_img>_start"].append(token_count)
                extra_token_pos["<joint_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"][0]
                extra_token_pos["<vae_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)

                token_seq.append(self.special.joint_img_sep_token_id)
                extra_token_pos["joint_img_sep"].append(token_count)
                token_count += 1

                token_seq.extend([self.special.img_token_id] * source["length"][1])
                extra_token_pos["<vit_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"][1]
                extra_token_pos["<vit_img>_end"].append(token_count - 1)
                extra_token_pos["<joint_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)

                token_seq.append(self.special.eoi_token_id)
                extra_token_pos["eoi"].append(token_count)
                token_count += 1

            elif key == "cond_vae_image":
                extra_count = (
                    2
                    + (1 if source.get("timestep", add_timestep_token) else 0)
                    + (2 if source.get("image_shape", add_image_shape_token) else 0)
                )
                if drop_last is True and total_length is not None:
                    if token_count + extra_count + source["length"] > total_length:
                        drop_last_break = True
                        break
                token_seq.append(self.special.boi_token_id)
                extra_token_pos["boi"].append(token_count)
                token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq,
                    token_count,
                    extra_token_pos,
                    add_timestep_token=source.get("timestep", add_timestep_token),
                    add_timestep_r_token=False,
                    add_image_shape_token=source.get("image_shape", add_image_shape_token),
                    add_guidance_token=False,
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                token_seq.extend([self.special.img_token_id] * source["length"])
                extra_token_pos["<vae_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"]
                extra_token_pos["<vae_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                token_seq.append(self.special.eoi_token_id)
                extra_token_pos["eoi"].append(token_count)
                token_count += 1

            elif key == "cond_vit_image":
                extra_count = 2
                if drop_last is True and total_length is not None:
                    if token_count + extra_count + source["length"] > total_length:
                        drop_last_break = True
                        break
                token_seq.append(self.special.boi_token_id)
                extra_token_pos["boi"].append(token_count)
                token_count += 1
                token_seq.extend([self.special.img_token_id] * source["length"])
                extra_token_pos["<vit_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"]
                extra_token_pos["<vit_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                token_seq.append(self.special.eoi_token_id)
                extra_token_pos["eoi"].append(token_count)
                token_count += 1

            else:
                raise ValueError(f"Unsupported template key: {key!r}")

        if add_eos is True and not drop_last_break:
            token_seq.append(self.eos_token_id)
            extra_token_pos["eos"].append(token_count)
        elif add_eos == "auto" and not drop_last_break:
            if token_seq[-1] != self.eos_token_id and (total_length is None or token_count < total_length):
                token_seq.append(self.eos_token_id)
                extra_token_pos["eos"].append(token_count)

        if total_length is not None:
            pad_num = max(0, total_length - len(token_seq))
            if add_pad and pad_num:
                token_seq.extend([self.pad_token_id] * pad_num)
                extra_token_pos["first_pad"].append(token_count)

        return token_seq, dict(extra_token_pos)

    def encode_general(
        self,
        sections: list[dict[str, Any]],
        *,
        max_token_length: int | None = None,
        add_eos: bool | str = "auto",
        add_pad: bool | str = "auto",
        add_bos: bool = True,
        drop_last: bool | str = "auto",
    ) -> TokenizerEncodeOutput:
        template = "-".join(section["type"] for section in sections)
        token_source: dict[str, list] = defaultdict(list)

        for section in sections:
            if section["type"] == "text":
                text = self.encode_text(
                    section["text"],
                    uncond_enabled=section.get("uncond_enabled"),
                    uncond_p=section.get("uncond_p"),
                    max_length=section.get("max_length"),
                )
                token_source["text"].append(text)
            elif section["type"] == "gen_image":
                token_source["gen_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", False),
                        timestep_r=section.get("add_timestep_r_token", False),
                        guidance=section.get("add_guidance_token", False),
                        image_shape=section.get("add_image_shape_token", False),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )
            elif section["type"] == "cond_joint_image":
                token_source["cond_joint_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", False),
                        image_shape=section.get("add_image_shape_token", False),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )
            elif section["type"] == "cond_vae_image":
                token_source["cond_vae_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", False),
                        image_shape=section.get("add_image_shape_token", False),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )
            elif section["type"] == "cond_vit_image":
                token_source["cond_vit_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", False),
                        image_shape=section.get("add_image_shape_token", False),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )
            else:
                raise ValueError(f"Unsupported section type: {section['type']!r}")

        full_token_seq, extra_token_pos = self.encode_sequence(
            template,
            dict(token_source),
            total_length=max_token_length,
            add_eos=add_eos,
            add_pad=add_pad,
            add_bos=add_bos,
            drop_last=drop_last,
        )
        tokens = torch.tensor(full_token_seq, dtype=torch.long)
        gen_image_slices, gen_image_mask = _parse_extra_token_pos(extra_token_pos, "img", tokens)
        vae_image_slices, vae_image_mask = _parse_extra_token_pos(extra_token_pos, "vae_img", tokens)
        vit_image_slices, vit_image_mask = _parse_extra_token_pos(extra_token_pos, "vit_img", tokens)
        joint_image_slices, _ = _parse_extra_token_pos(extra_token_pos, "joint_img", tokens)
        all_image_slices = [
            slice(start, end + 1)
            for start, end in zip(
                extra_token_pos.get("<all_img>_start", []),
                extra_token_pos.get("<all_img>_end", []),
            )
        ]
        cond_timestep_scatter_index = (
            torch.tensor(extra_token_pos["cond_timestep"], dtype=torch.long)
            if "cond_timestep" in extra_token_pos
            else None
        )
        gen_timestep_scatter_index = (
            torch.tensor(extra_token_pos["gen_timestep"], dtype=torch.long)
            if "gen_timestep" in extra_token_pos
            else None
        )
        gen_timestep_r_scatter_index = (
            torch.tensor(extra_token_pos["gen_timestep_r"], dtype=torch.long)
            if "gen_timestep_r" in extra_token_pos
            else None
        )
        guidance_scatter_index = (
            torch.tensor(extra_token_pos["guidance"], dtype=torch.long) if "guidance" in extra_token_pos else None
        )
        text_slices = (
            [
                slice(start, end + 1)
                for start, end in zip(extra_token_pos["<text>_start"], extra_token_pos["<text>_end"])
            ]
            if "<text>_start" in extra_token_pos
            else []
        )
        real_pos = torch.tensor(
            extra_token_pos.get("first_pad", [tokens.shape[0]]),
            dtype=torch.long,
        )
        return TokenizerEncodeOutput(
            tokens=tokens,
            gen_image_mask=gen_image_mask,
            vae_image_mask=vae_image_mask,
            vit_image_mask=vit_image_mask,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            gen_timestep_r_scatter_index=gen_timestep_r_scatter_index,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            guidance_scatter_index=guidance_scatter_index,
            text_slices=text_slices,
            gen_image_slices=gen_image_slices,
            vae_image_slices=vae_image_slices,
            vit_image_slices=vit_image_slices,
            joint_image_slices=joint_image_slices,
            all_image_slices=all_image_slices,
            real_pos=real_pos,
        )

    def _pretrain_role_format(self) -> tuple[str, str, str, str, str, str, str]:
        return "", "", "", "", "", "", ""

    def _instruct_role_format(self) -> tuple[str, str, str, str, str, str, str]:
        conv = self._instruct_conv
        answer_prefix = self.tokenizer.convert_ids_to_tokens(self.special.answer_token_id)
        answer_suffix = self.tokenizer.convert_ids_to_tokens(self.special.end_answer_token_id)
        return (
            conv.sep,
            conv.role_prefix(conv.roles[0]),
            conv.sep,
            conv.role_prefix(conv.roles[1]),
            conv.sep,
            answer_prefix,
            answer_suffix,
        )

    def apply_general_template(
        self,
        message_list: list[dict[str, Any]],
        *,
        max_length: int | None = None,
        add_assistant_prefix: bool = False,
        bot_task: str = "auto",
        image_base_size: int | None = None,
        uncond_p: float = 0.0,
    ) -> tuple[TokenizerEncodeOutput, list[dict[str, Any]]]:
        if self.sequence_template == "pretrain":
            (
                system_suffix,
                user_prefix,
                user_suffix,
                bot_prefix,
                bot_suffix,
                answer_prefix,
                answer_suffix,
            ) = self._pretrain_role_format()
        elif self.sequence_template == "instruct":
            (
                system_suffix,
                user_prefix,
                user_suffix,
                bot_prefix,
                bot_suffix,
                answer_prefix,
                answer_suffix,
            ) = self._instruct_role_format()
        else:
            raise NotImplementedError(
                f"sequence_template={self.sequence_template!r} not supported; use 'pretrain' or 'instruct'."
            )

        def process_successive_message(
            _message_list,
            _cur_message_idx: int,
            role: str,
            prefix: str,
            suffix: str,
        ) -> tuple[list[dict[str, Any]], int]:
            _sub_sections: list[dict[str, Any]] = []
            uncond_kwargs = dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)
            while _cur_message_idx < len(_message_list) and _message_list[_cur_message_idx]["role"] == role:
                message = _message_list[_cur_message_idx]
                if message["type"] == "text":
                    _sub_sections.append(dict(type="text", text=message["content"], **uncond_kwargs))
                elif message["type"] == "gen_image":
                    info = message["content"]
                    if not isinstance(info, ImageInfo):
                        raise TypeError(f"Expected ImageInfo, got {type(info)}")
                    if role == "assistant" and answer_prefix:
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type="gen_image", **info.meta_info))
                    if role == "assistant" and answer_suffix:
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                elif message["type"] in ("cond_joint_image", "cond_vae_image", "cond_vit_image"):
                    info = message["content"]
                    if not isinstance(info, (ImageInfo, JointImageInfo)):
                        raise TypeError(f"Expected ImageInfo or JointImageInfo, got {type(info)}")
                    _sub_sections.append(dict(type=message["type"], **info.meta_info))
                else:
                    raise ValueError(f"Unknown message type: {message['type']!r}")
                _cur_message_idx += 1
            if _sub_sections:
                _sub_sections.insert(0, dict(type="text", text=prefix))
                _sub_sections.append(dict(type="text", text=suffix))
            return _sub_sections, _cur_message_idx

        if self.sequence_template == "instruct":
            answer_prefix = self.tokenizer.convert_ids_to_tokens(self.special.answer_token_id)
            answer_suffix = self.tokenizer.convert_ids_to_tokens(self.special.end_answer_token_id)
        else:
            answer_prefix = answer_suffix = ""

        sections: list[dict[str, Any]] = []
        cur_message_idx = 0
        final_role: str | None = None
        while cur_message_idx < len(message_list):
            for role, prefix, suffix in (
                ("system", "", system_suffix),
                ("user", user_prefix, user_suffix),
                ("assistant", bot_prefix, bot_suffix),
            ):
                sub_sections, cur_message_idx = process_successive_message(
                    message_list, cur_message_idx, role, prefix, suffix
                )
                sections.extend(sub_sections)
                if sub_sections:
                    final_role = role

        if add_assistant_prefix:
            if bot_task == "img_ratio" and image_base_size is None:
                raise ValueError("image_base_size is required when bot_task='img_ratio'")
            if final_role == "assistant":
                _bot_prefix = ""
                if sections and sections[-1]["type"] == "text" and sections[-1]["text"] == bot_suffix:
                    sections = sections[:-1]
            else:
                _bot_prefix = bot_prefix
            recaption_str = self.tokenizer.convert_ids_to_tokens(self.special.recaption_token_id)
            think_str = self.tokenizer.convert_ids_to_tokens(self.special.think_token_id)
            boi_str = self.tokenizer.convert_ids_to_tokens(self.special.boi_token_id)
            if bot_task == "img_ratio":
                size_str = self.tokenizer.convert_ids_to_tokens(self.special.size_token_id(image_base_size))
                bot_response_prefix = f"{_bot_prefix}{answer_prefix}{boi_str}{size_str}"
            elif bot_task == "think":
                bot_response_prefix = f"{_bot_prefix}{think_str}"
            elif bot_task == "recaption":
                bot_response_prefix = f"{_bot_prefix}{recaption_str}"
            elif bot_task == "image":
                bot_response_prefix = ""
            else:
                bot_response_prefix = _bot_prefix
            sections.append(dict(type="text", text=bot_response_prefix))

        output = self.encode_general(sections=sections, add_eos=False, add_pad=False)
        if max_length is not None and output.tokens.shape[-1] > max_length:
            raise ValueError(f"Encoded length {output.tokens.shape[-1]} exceeds max_length={max_length}")
        return output, sections

    @staticmethod
    def _pad_tensors(tensor_list: list[torch.Tensor], pad_val) -> list[torch.Tensor]:
        max_len = max(int(t.shape[-1]) if t.ndim >= 1 else t.shape[0] for t in tensor_list)
        padded = []
        for t in tensor_list:
            if t.ndim == 1 and t.shape[0] < max_len:
                pad = torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)
                padded.append(torch.cat([t, pad], dim=0))
            else:
                padded.append(t)
        return padded

    def _stack_optional_masks(
        self,
        merged: list[TokenizerEncodeOutput],
        attr: str,
    ) -> torch.Tensor | None:
        masks = [getattr(o, attr) for o in merged]
        if not all(m is not None for m in masks):
            return None
        return torch.stack(
            self._pad_tensors([m.to(torch.bool) for m in masks], False),
            dim=0,
        )

    def _stack_batch(
        self,
        cond_outputs: list[TokenizerEncodeOutput],
        uncond_outputs: list[TokenizerEncodeOutput],
    ) -> TokenizerEncodeOutput:
        merged = cond_outputs + uncond_outputs
        tokens = torch.stack(
            self._pad_tensors([o.tokens for o in merged], self.pad_token_id),
            dim=0,
        )
        return TokenizerEncodeOutput(
            tokens=tokens,
            gen_image_mask=self._stack_optional_masks(merged, "gen_image_mask"),
            vae_image_mask=self._stack_optional_masks(merged, "vae_image_mask"),
            vit_image_mask=self._stack_optional_masks(merged, "vit_image_mask"),
            gen_timestep_scatter_index=_stack_optional_index([o.gen_timestep_scatter_index for o in merged]),
            gen_timestep_r_scatter_index=_stack_optional_index([o.gen_timestep_r_scatter_index for o in merged]),
            cond_timestep_scatter_index=_stack_optional_index([o.cond_timestep_scatter_index for o in merged]),
            guidance_scatter_index=_stack_optional_index([o.guidance_scatter_index for o in merged]),
            text_slices=[o.text_slices for o in merged],
            gen_image_slices=[o.gen_image_slices for o in merged],
            vae_image_slices=[o.vae_image_slices for o in merged],
            vit_image_slices=[o.vit_image_slices for o in merged],
            joint_image_slices=[o.joint_image_slices for o in merged],
            all_image_slices=[o.all_image_slices for o in merged],
            real_pos=torch.stack([o.real_pos for o in merged], dim=0) if merged[0].real_pos is not None else None,
        )

    def apply_chat_template(
        self,
        *,
        batch_prompt: list[str],
        batch_gen_image_info: list[ImageInfo] | None = None,
        mode: str = "gen_image",
        batch_cond_images: list[CondImage] | list[list[CondImage]] | None = None,
        batch_system_prompt: list[str] | None = None,
        batch_cot_text: list[str | None] | None = None,
        max_length: int | None = None,
        cfg_factor: int = 1,
        sequence_template: str | None = None,
        bot_task: str = "auto",
        image_base_size: int | None = None,
        add_assistant_prefix: bool | None = None,
    ) -> dict[str, Any]:
        if mode not in ("gen_image", "gen_text"):
            raise ValueError(f"mode={mode!r} not supported; use 'gen_image' or 'gen_text'")
        if bot_task not in ("auto", "image", "think", "recaption", "img_ratio"):
            raise ValueError(
                f"bot_task={bot_task!r} not supported; " "use one of 'auto', 'image', 'think', 'recaption', 'img_ratio'"
            )
        if sequence_template is not None:
            self.sequence_template = sequence_template
        if mode == "gen_text":
            cfg_factor = 1
        if add_assistant_prefix is None:
            add_assistant_prefix = mode != "gen_image"

        batch_size = len(batch_prompt)
        if batch_system_prompt is None:
            batch_system_prompt = [None] * batch_size
        if batch_cot_text is None:
            batch_cot_text = [None] * batch_size
        if mode == "gen_image":
            if batch_gen_image_info is None:
                raise ValueError("batch_gen_image_info is required when mode='gen_image'")
            if len(batch_gen_image_info) == 1 and batch_size > 1:
                batch_gen_image_info = batch_gen_image_info * batch_size
        if batch_cond_images is not None:
            if len(batch_cond_images) != batch_size:
                raise ValueError(
                    f"batch_cond_images length ({len(batch_cond_images)}) must match batch_size ({batch_size})"
                )
            batch_cond_images = [
                cond_images if isinstance(cond_images, list) else [cond_images] for cond_images in batch_cond_images
            ]
        else:
            batch_cond_images = [[] for _ in range(batch_size)]

        if mode == "gen_image":
            gen_infos_iter = batch_gen_image_info
        else:
            gen_infos_iter = [None] * batch_size

        batch_message_list = []
        for prompt, system_prompt, cot_text, gen_image_info, cond_images in zip(
            batch_prompt,
            batch_system_prompt,
            batch_cot_text,
            gen_infos_iter,
            batch_cond_images,
        ):
            message_list: list[dict[str, Any]] = []
            if system_prompt:
                message_list.append(dict(role="system", type="text", content=system_prompt))
            if len(cond_images) > 0:
                message_list.extend(
                    [
                        dict(role="user", type=cond_image.section_type, content=cond_image.i)
                        for cond_image in cond_images
                    ]
                )
            message_list.append(dict(role="user", type="text", content=prompt))
            if cot_text is not None:
                message_list.append(dict(role="assistant", type="text", content=cot_text))
            if mode == "gen_image":
                message_list.append(dict(role="assistant", type="gen_image", content=gen_image_info))
            batch_message_list.append(message_list)

        cond_outputs: list[TokenizerEncodeOutput] = []
        uncond_outputs: list[TokenizerEncodeOutput] = []
        all_sections: list[list[dict[str, Any]]] = []

        for message_list in batch_message_list:
            out_cond, sections = self.apply_general_template(
                message_list,
                max_length=max_length,
                add_assistant_prefix=add_assistant_prefix,
                bot_task=bot_task,
                image_base_size=image_base_size,
                uncond_p=0.0,
            )
            cond_outputs.append(out_cond)
            all_sections.append(sections)
            if cfg_factor > 1:
                out_uncond, _ = self.apply_general_template(
                    message_list,
                    max_length=max_length,
                    add_assistant_prefix=add_assistant_prefix,
                    bot_task=bot_task,
                    image_base_size=image_base_size,
                    uncond_p=1.0,
                )
                uncond_outputs.append(out_uncond)

        if cfg_factor > 1:
            output = self._stack_batch(cond_outputs, uncond_outputs)
        elif batch_size == 1:
            output = cond_outputs[0]
            output = TokenizerEncodeOutput(
                tokens=output.tokens.unsqueeze(0),
                gen_image_mask=(output.gen_image_mask.unsqueeze(0) if output.gen_image_mask is not None else None),
                vae_image_mask=(output.vae_image_mask.unsqueeze(0) if output.vae_image_mask is not None else None),
                vit_image_mask=(output.vit_image_mask.unsqueeze(0) if output.vit_image_mask is not None else None),
                gen_timestep_scatter_index=(
                    output.gen_timestep_scatter_index.unsqueeze(0)
                    if output.gen_timestep_scatter_index is not None
                    else None
                ),
                gen_timestep_r_scatter_index=(
                    output.gen_timestep_r_scatter_index.unsqueeze(0)
                    if output.gen_timestep_r_scatter_index is not None
                    else None
                ),
                cond_timestep_scatter_index=(
                    output.cond_timestep_scatter_index.unsqueeze(0)
                    if output.cond_timestep_scatter_index is not None
                    else None
                ),
                guidance_scatter_index=(
                    output.guidance_scatter_index.unsqueeze(0) if output.guidance_scatter_index is not None else None
                ),
                text_slices=output.text_slices,
                gen_image_slices=[output.gen_image_slices],
                vae_image_slices=[output.vae_image_slices],
                vit_image_slices=[output.vit_image_slices],
                joint_image_slices=[output.joint_image_slices],
                all_image_slices=[output.all_image_slices],
                real_pos=output.real_pos.unsqueeze(0) if output.real_pos is not None else None,
            )
        else:
            output = self._stack_batch(cond_outputs, [])

        return dict(output=output, sections=all_sections[0] if batch_size == 1 else all_sections)


def _stack_optional_index(items: list[torch.Tensor | None]) -> torch.Tensor | None:
    if not items or items[0] is None:
        return None
    return torch.stack(items, dim=0)
