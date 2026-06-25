# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Host-side public API for HunyuanImage-3.0 T2I tokenization (no TT device).
#
# Mapping to HF upstream (hunyuan_image_3/, not imported at runtime)
# ------------------------------------------------
#  Step | HF                                           | This port (host)
#  -----+----------------------------------------------+----------------------------------
#   1   | HunyuanImage3Config.from_pretrained          | load_config / HunyuanConfig
#   2   | AutoTokenizer.from_pretrained                | load_tokenizer (bundled assets/)
#   3   | HunyuanImage3TokenizerFast.setup_special_tokens | build_special_tokens
#   4   | ImageProcessor.build_gen_image_info          | build_gen_image_info
#   5   | HunyuanImage3TokenizerFast.apply_chat_template | ChatTemplateEncoder.apply_chat_template
#   6   | model forward input assembly                 | prepare_gen_image_inputs
#
# Bundled assets (ref/tokenizer/assets/)
# --------------------------------------
#   config.json            model hyperparameters for the tokenizer stack
#   tokenizer.json         BPE vocab (~24 MB; download from HF, not in git)
#   tokenizer_config.json  HF fast-tokenizer config
#
# References
# ----------
#   ref/tokenizer/chat_template.py     — T2I sequence layout
#   ref/tokenizer/special_tokens.py    — multimodal token ID map
#   ref/tokenizer/gen_image_inputs.py  — host preprocess bundle for device upload
#   HunyuanImage-3.0/hunyuan_image_3/tokenization_hunyuan_image_3.py  — upstream reference

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .chat_template import ChatTemplateEncoder, TokenizerEncodeOutput
from .image_info import CondImage, ImageInfo, build_gen_image_info
from .special_tokens import SpecialTokens, build_special_tokens, validate_special_tokens

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
CONFIG_PATH = ASSETS_DIR / "config.json"
TOKENIZER_DIR = ASSETS_DIR


@dataclass(frozen=True)
class HunyuanConfig:
    """Minimal config fields used by the tokenizer stack."""

    model_version: str
    vocab_size: int
    hidden_size: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    image_base_size: int
    vae_downsample_factor: tuple[int, int]
    patch_size: int
    vae_patch_size: int
    cfg_distilled: bool
    use_meanflow: bool
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> HunyuanConfig:
        vae_factor = config.get("vae_downsample_factor", [8, 8])
        if isinstance(vae_factor, int):
            vae_factor = (vae_factor, vae_factor)
        vit_processor = config.get("vit_processor") or {}
        return cls(
            model_version=config.get("model_version", "HunyuanImage-3.0"),
            vocab_size=int(config["vocab_size"]),
            hidden_size=int(config["hidden_size"]),
            bos_token_id=int(config["bos_token_id"]),
            eos_token_id=int(config["eos_token_id"]),
            pad_token_id=int(config.get("pad_token_id", 128009)),
            image_base_size=int(config.get("image_base_size", 1024)),
            vae_downsample_factor=(int(vae_factor[0]), int(vae_factor[1])),
            patch_size=int(vit_processor.get("patch_size", 16)),
            vae_patch_size=int(config.get("patch_size", 1)),
            cfg_distilled=bool(config.get("cfg_distilled", False)),
            use_meanflow=bool(config.get("use_meanflow", False)),
            raw=config,
        )


def load_config(config_path: Path = CONFIG_PATH) -> HunyuanConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing bundled config: {config_path}")
    with open(config_path) as f:
        return HunyuanConfig.from_dict(json.load(f))


def load_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR) -> PreTrainedTokenizerFast:
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing bundled tokenizer.json under {tokenizer_dir}")
    return AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=False)


class HunyuanTokenizer:
    """Bundled config + HF fast tokenizer + multimodal special-token map + T2I template."""

    def __init__(
        self,
        config: HunyuanConfig,
        tokenizer: PreTrainedTokenizerFast,
        special: SpecialTokens,
        *,
        sequence_template: str = "pretrain",
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.special = special
        self.sequence_template = sequence_template
        self._chat = ChatTemplateEncoder(tokenizer, special, sequence_template=sequence_template)

    @classmethod
    def from_pretrained(cls, *, sequence_template: str = "pretrain") -> HunyuanTokenizer:
        config = load_config()
        tokenizer = load_tokenizer()
        special = build_special_tokens(tokenizer, model_version=config.model_version)
        return cls(config, tokenizer, special, sequence_template=sequence_template)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path | str,
        *,
        sequence_template: str = "instruct",
        tokenizer_dir: Path | str | None = None,
    ) -> HunyuanTokenizer:
        """Load config (and optionally tokenizer) from a checkpoint directory."""
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json under {model_dir}")
        config = load_config(config_path)
        tok_dir = Path(tokenizer_dir) if tokenizer_dir is not None else model_dir
        if not (tok_dir / "tokenizer.json").is_file():
            tok_dir = TOKENIZER_DIR
        tokenizer = load_tokenizer(tok_dir)
        special = build_special_tokens(tokenizer, model_version=config.model_version)
        return cls(config, tokenizer, special, sequence_template=sequence_template)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_tensor(self, text: str, *, add_special_tokens: bool = False) -> Tensor:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_id(self, token: str) -> int:
        if token in self.special.special_token_map:
            return self.special.special_token_map[token]
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id == self.tokenizer.unk_token_id:
            raise KeyError(f"Unknown special token: {token!r}")
        return int(token_id)

    def token_strings(self, token_ids: list[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def build_gen_image_info(
        self,
        image_size: str | tuple[int, int] | list[int],
        *,
        add_guidance_token: bool | None = None,
        add_timestep_r_token: bool | None = None,
    ) -> ImageInfo:
        return build_gen_image_info(
            image_size=image_size,
            image_base_size=self.config.image_base_size,
            vae_downsample_factor=self.config.vae_downsample_factor,
            vae_patch_size=self.config.vae_patch_size,
            add_guidance_token=(self.config.cfg_distilled if add_guidance_token is None else add_guidance_token),
            add_timestep_r_token=(self.config.use_meanflow if add_timestep_r_token is None else add_timestep_r_token),
        )

    def apply_chat_template(
        self,
        prompt: str | list[str],
        *,
        image_size: str | tuple[int, int] | list[int] = 1024,
        cond_images: CondImage | list[CondImage] | list[list[CondImage]] | None = None,
        system_prompt: str | None = None,
        cot_text: str | None = None,
        mode: str = "gen_image",
        bot_task: str = "auto",
        max_length: int | None = None,
        cfg_factor: int | None = None,
        sequence_template: str | None = None,
        image_base_size: int | None = None,
    ) -> dict[str, Any]:
        """Build T2I / I2I token sequence(s) for ``mode='gen_image'`` or recaption prefix for ``mode='gen_text'``.

        Pass ``cond_images`` for image-to-image (user cond block before prompt).
        Pass ``system_prompt`` for instruct-style system block (``en_unified``).
        For ``mode='gen_text'``, set ``bot_task`` to ``'recaption'`` or ``'think'`` to append the
        assistant prefix that starts autoregressive recaption/think generation.
        Returns ``dict(output=TokenizerEncodeOutput, sections=...)`` matching HF layout.
        ``output.tokens`` shape is ``[B, S]`` or ``[2, S]`` when ``cfg_factor=2``.
        """
        batch_prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(batch_prompt)
        if mode == "gen_image":
            gen_infos = [self.build_gen_image_info(image_size) for _ in batch_prompt]
            if cfg_factor is None:
                cfg_factor = 1 if self.config.cfg_distilled else 2
        else:
            gen_infos = None
            cfg_factor = 1
        if sequence_template is None:
            sequence_template = self.sequence_template
        if image_base_size is None:
            image_base_size = self.config.image_base_size

        batch_cond_images = None
        if cond_images is not None:
            if isinstance(cond_images, CondImage):
                batch_cond_images = [[cond_images] for _ in range(batch_size)]
            elif cond_images and isinstance(cond_images[0], CondImage):
                if len(cond_images) == batch_size:
                    batch_cond_images = [[c] for c in cond_images]
                elif len(cond_images) == 1:
                    batch_cond_images = [list(cond_images) for _ in range(batch_size)]
                else:
                    batch_cond_images = [list(cond_images) for _ in range(batch_size)]
            else:
                batch_cond_images = list(cond_images)

        batch_system_prompt = [system_prompt] * batch_size if system_prompt else None
        batch_cot_text = [cot_text] * batch_size if cot_text else None

        return self._chat.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_gen_image_info=gen_infos,
            batch_cond_images=batch_cond_images,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            mode=mode,
            bot_task=bot_task,
            max_length=max_length,
            cfg_factor=cfg_factor,
            sequence_template=sequence_template,
            image_base_size=image_base_size,
        )

    def validate_text(self) -> dict[str, Any]:
        sample = "a cat sitting on a mat"
        ids = self.encode(sample)
        tokens = self.token_strings(ids)
        roundtrip = self.decode(ids)
        return {
            "config_path": str(CONFIG_PATH),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "model_version": self.config.model_version,
            "vocab_size": self.tokenizer.vocab_size,
            "sample_text": sample,
            "sample_ids": ids,
            "sample_tokens": tokens,
            "sample_roundtrip_ok": roundtrip.strip() == sample,
        }

    def validate_special(self) -> dict[str, Any]:
        return validate_special_tokens(self.special, self.tokenizer)

    def validate_t2i_sequence(self, prompt: str = "a cat on a mat", image_size=1024) -> dict[str, Any]:
        """Step 3 sanity-check: assemble full gen_image token layout."""
        result = self.apply_chat_template(prompt, image_size=image_size, cfg_factor=1)
        output: TokenizerEncodeOutput = result["output"]
        tokens = output.tokens[0].tolist()
        info = self.build_gen_image_info(image_size)
        expected_img_tokens = int(info.image_token_length)
        img_count = tokens.count(self.special.img_token_id)
        gen_mask_count = int(output.gen_image_mask[0].sum()) if output.gen_image_mask is not None else 0
        ts_idx = output.gen_timestep_scatter_index[0].tolist() if output.gen_timestep_scatter_index is not None else []
        return {
            "prompt": prompt,
            "image_size": image_size,
            "seq_len": len(tokens),
            "image_token_length": expected_img_tokens,
            "img_placeholder_count": img_count,
            "gen_image_mask_count": gen_mask_count,
            "gen_timestep_scatter_index": ts_idx,
            "has_boi": self.special.boi_token_id in tokens,
            "has_eoi": self.special.eoi_token_id in tokens,
            "has_timestep": self.special.timestep_token_id in tokens,
            "first_20_token_strs": self.token_strings(tokens[:20]),
            "image_block_preview": self.token_strings(
                tokens[
                    max(0, tokens.index(self.special.boi_token_id)) : max(0, tokens.index(self.special.boi_token_id))
                    + 8
                ]
            ),
            "t2i_sequence_ok": (
                img_count == expected_img_tokens
                and gen_mask_count == expected_img_tokens
                and self.special.boi_token_id in tokens
                and self.special.eoi_token_id in tokens
                and len(ts_idx) == 1
            ),
        }

    def validate_cfg(self, prompt: str = "a cat on a mat", image_size=1024) -> dict[str, Any]:
        """Step 4 sanity-check: CFG doubles batch; uncond replaces text, not image block."""
        output = self.apply_chat_template(prompt, image_size=image_size, cfg_factor=2)["output"]
        cfg_id = self.special.cfg_token_id
        boi_id = self.special.boi_token_id
        cond, uncond = output.tokens[0], output.tokens[1]
        boi_idx = cond.tolist().index(boi_id)
        text_cfg_tokens = int((uncond[:boi_idx] == cfg_id).sum())
        image_block_match = torch.equal(cond[boi_idx:], uncond[boi_idx:])
        mask_match = torch.equal(output.gen_image_mask[0], output.gen_image_mask[1])
        scatter_match = torch.equal(output.gen_timestep_scatter_index[0], output.gen_timestep_scatter_index[1])
        return {
            "cfg_batch_size": int(output.tokens.shape[0]),
            "seq_len": int(output.tokens.shape[1]),
            "boi_index": boi_idx,
            "text_cfg_token_count": text_cfg_tokens,
            "image_block_match": image_block_match,
            "gen_image_mask_match": mask_match,
            "gen_timestep_scatter_match": scatter_match,
            "cfg_ok": (
                output.tokens.shape[0] == 2
                and text_cfg_tokens > 0
                and image_block_match
                and mask_match
                and scatter_match
            ),
        }

    def validate(self) -> dict[str, Any]:
        text_report = self.validate_text()
        special_report = self.validate_special()
        t2i_report = self.validate_t2i_sequence()
        cfg_report = self.validate_cfg()
        return {
            **text_report,
            "special_tokens": special_report,
            "t2i_sequence": t2i_report,
            "cfg": cfg_report,
            "all_ok": (
                text_report["sample_roundtrip_ok"]
                and special_report["special_tokens_ok"]
                and t2i_report["t2i_sequence_ok"]
                and cfg_report["cfg_ok"]
            ),
        }


if __name__ == "__main__":
    tok = HunyuanTokenizer.from_pretrained()
    report = tok.validate()

    print("=== Step 1: text encode/decode ===")
    print(f"sample_roundtrip_ok: {report['sample_roundtrip_ok']}")
    print(f"sample_ids: {report['sample_ids']}")

    print("\n=== Step 2: special token map ===")
    sp = report["special_tokens"]
    print(f"special_tokens_ok: {sp['special_tokens_ok']}")

    print("\n=== Step 3: T2I sequence layout ===")
    t2i = report["t2i_sequence"]
    for key in (
        "prompt",
        "image_size",
        "seq_len",
        "image_token_length",
        "img_placeholder_count",
        "gen_image_mask_count",
        "gen_timestep_scatter_index",
        "has_boi",
        "has_eoi",
        "has_timestep",
        "t2i_sequence_ok",
    ):
        print(f"{key}: {t2i[key]}")
    print(f"image_block_preview: {t2i['image_block_preview']}")

    print("\n=== Step 4: CFG + host preprocess ===")
    cfg = report["cfg"]
    for key in (
        "cfg_batch_size",
        "seq_len",
        "text_cfg_token_count",
        "image_block_match",
        "gen_image_mask_match",
        "gen_timestep_scatter_match",
        "cfg_ok",
    ):
        print(f"{key}: {cfg[key]}")

    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_gen_image_inputs

    bundle = prepare_gen_image_inputs(tok, "a cat on a mat", image_size=1024, cfg_factor=2)
    print(
        f"host_bundle: batch={bundle.input_ids.shape[0]} seq={bundle.seq_len} rope_rows={len(bundle.rope_image_info)}"
    )

    print(f"\nall_ok: {report['all_ok']}")
