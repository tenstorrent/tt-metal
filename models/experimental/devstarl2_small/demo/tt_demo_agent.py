# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent variant of ``demo_agent.py``: same interactive agent tools and chat loop, but generation
runs on **TT** via ``TtDevstral2SmallModel`` (Pixtral vision + projector + ``TtMinistral3`` LM),
then LM head and ``SamplingGenerator`` / CPU logits (``demo_devstral2_tt_multimodal`` / ``demo_model_loading_prompt``).

**Per turn:** one full TT prefill of the current chat history (rebuilds KV cache) followed by a
**traced TT decode** for each new token. The single-token decode trace is captured once on the first
turn and reused for every subsequent turn — including across tool-call rounds and ``/image`` switches
— because the decode kernel sequence only depends on shapes (constant for this model), not on the
prompt or KV cache contents.

**Text default:** token-id prefill only (vision stack still loads in ``TtDevstral2SmallModel``).

**Image in-session (no ``--image`` required):** at the ``You:`` prompt use::

    /image /path/to/file.jpg
    What do you see?

Or one line (path + question; quote paths with spaces)::

    /image ./screenshot.png Describe the UI.

**Commands:** ``/image <path> [optional prompt…]``, ``/noimage`` (drop sticky image), ``/clear`` (reset
chat + tools; image stays until ``/noimage``). Optional startup ``--image PATH`` pre-attaches like ``/image``.

``--vision-square-pixels`` / ``--vision-max-edge`` apply to ``/image`` and ``--image`` alike.

**Context budget:** TT KV is allocated for ``--max-context-tokens + max_new_tokens + 2048`` (padded).

Usage (repo root)::

    python models/experimental/devstarl2_small/demo/tt_demo_agent.py --mesh-width 1 \\
        --vision-square-pixels 1540

    python models/experimental/devstarl2_small/demo/tt_demo_agent.py --max-context-tokens 4096 \\
        --lm-head-cpu --text-layers 2
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import ttnn
from PIL import Image
from transformers import AutoProcessor, MistralCommonBackend
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.experimental.devstarl2_small.demo import demo_model_loading_prompt as _mlp
from models.experimental.devstarl2_small.demo.demo_agent import (
    DEFAULT_AGENT_RULES,
    DEFAULT_SYSTEM_PROMPT,
    AgentState,
    ChatConfig,
    execute_tool_call,
    parse_tool_call,
)
from models.experimental.devstarl2_small.devstral_utils import (
    DEFAULT_MODEL_ID,
    TtDecodeTraceContext,
    apply_devstral_hf_trust_patches,
    apply_fp8_dequantize_compat,
    close_devstral_demo_mesh,
    cpu_lm_head_logits_last_token,
    demo_lm_head_max_columns_per_device,
    devstral_supports_on_device_sampling,
    eos_token_ids,
    host_input_ids_to_tt_replicated,
    open_devstral_demo_mesh,
    tt_alloc_decode_input_buffers,
    tt_append_uint32_token,
    tt_capture_decode_trace,
    tt_execute_decode_trace,
    tt_forward_prefill_from_device_ids,
    tt_lm_head_logits_block,
    tt_lm_head_logits_last_token,
    tt_read_decode_traced_hidden,
    tt_read_decode_traced_logits,
    tt_read_decode_traced_token,
    tt_release_decode_trace,
    tt_replicated_ids_to_torch_long,
    tt_sampling_output_token_id,
    tt_update_decode_input_buffers,
)
from models.experimental.devstarl2_small.tt.tt_devstral2_small_model import TtDevstral2SmallModel
from models.experimental.devstarl2_small.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

# Tell the instruct model that Pixtral inputs are real in this demo; otherwise it mimics a text-only agent.
TT_AGENT_MULTIMODAL_SYSTEM_APPEND = """
**Vision (Tenstorrent demo):** Images can be attached with the host `/image` command. When a user message
includes an image (multimodal input), you **do** see the image through the model—answer visual questions
directly. Do not claim you cannot view images or that vision is unavailable in this environment.
""".strip()


@dataclass
class TTAgentConfig(ChatConfig):
    """Extends PyTorch demo config with TT runtime options."""

    mesh_width: int = 1
    text_layers: Optional[int] = None
    lm_head_cpu: bool = False
    lm_head_max_device_cols: Optional[int] = None
    cpu_sampling: bool = False
    max_seq_len: Optional[int] = None
    max_context_tokens: int = 8192
    seed: Optional[int] = None
    vision_image: Optional[Path] = None
    vision_max_edge: int = 0
    vision_square_pixels: Optional[int] = None


@dataclass
class TtAgentRuntime:
    mesh_device: ttnn.MeshDevice
    tokenizer: MistralCommonBackend
    model_args: ModelArgs
    tt_devstral: TtDevstral2SmallModel
    tt_language_model: TtMinistral3Model
    hf_full: torch.nn.Module
    hf_inner: Mistral3Model
    tt_lm_head: Optional[LMHead]
    lm_head_weight_cpu: Optional[torch.Tensor]
    sampling: Optional[SamplingGenerator]
    sampling_empty_slots: Optional[List[int]]
    shared_tt_ccl: TT_CCL
    pad_token_id: int
    pad_row_1d: torch.Tensor
    cfg: TTAgentConfig
    sticky_pil: Optional[Image.Image] = None
    auto_processor: Optional[Any] = None
    image_token_id: Optional[int] = None
    _img_rows_cache: Optional[torch.Tensor] = None
    # Lazily captured single-token decode trace. Shared across all generate calls in the
    # session: the kernel sequence does not depend on the prompt or chat history, only the
    # KV cache state, so one warmup + capture amortizes over every turn of the agent loop.
    decode_trace_ctx: Optional[TtDecodeTraceContext] = None


def _tokenizer_apply_messages(
    tokenizer: MistralCommonBackend,
    messages: List[Dict[str, str]],
) -> torch.LongTensor:
    tokenized = tokenizer.apply_chat_template(
        conversation=messages,
        return_tensors="pt",
        return_dict=True,
        trust_remote_code=True,
    )
    return tokenized["input_ids"]


def _inject_image_into_first_human_user(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """First non-tool ``user`` message becomes Pixtral-style image + text; ``<tool_result>`` rows stay plain."""
    out: List[Dict[str, Any]] = []
    image_done = False
    for m in messages:
        if m["role"] == "user" and not image_done:
            c = m.get("content", "")
            if isinstance(c, str) and c.lstrip().startswith("<tool_result>"):
                out.append(dict(m))
                continue
            image_done = True
            out.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": c}]})
        else:
            out.append(dict(m))
    return out


def _ensure_multimodal_img_rows(
    rt: TtAgentRuntime,
    pixel_values: torch.Tensor,
    image_sizes_list: list[tuple[int, int]],
) -> torch.Tensor:
    if rt._img_rows_cache is not None:
        return rt._img_rows_cache
    pos_vision = _mlp._vision_position_ids_tt(rt.hf_inner, pixel_values, image_sizes_list, rt.mesh_device)
    img_tt = rt.tt_devstral.get_projected_image_features(pixel_values, image_sizes_list, pos_vision)
    ttnn.deallocate(pos_vision)
    img_torch = ttnn.to_torch(img_tt, mesh_composer=ttnn.ConcatMeshToTensor(rt.mesh_device, dim=-1))
    ttnn.deallocate(img_tt)
    while img_torch.dim() > 2:
        img_torch = img_torch.squeeze(0)
    img_rows = img_torch.reshape(-1, img_torch.shape[-1]).contiguous()
    rt._img_rows_cache = img_rows
    return img_rows


def _load_auto_processor(model_id: str) -> Any:
    _extra: dict[str, Any] = {}
    if os.getenv("CI") == "true":
        _extra["local_files_only"] = True
    try:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True, fix_mistral_regex=True, **_extra)
    except (TypeError, ValueError):
        try:
            return AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **_extra)
        except (TypeError, ValueError):
            return AutoProcessor.from_pretrained(model_id, **_extra)


def _load_sticky_image_bundle(
    config: TTAgentConfig,
    hf_full: torch.nn.Module,
    img_path: Path,
) -> tuple[Image.Image, Any, int]:
    img_path = img_path.expanduser()
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")
    pil = _mlp._prepare_vision_image(
        Image.open(img_path).convert("RGB"),
        int(config.vision_max_edge),
        config.vision_square_pixels,
    )
    proc = _load_auto_processor(config.model_id)
    tid = int(hf_full.config.image_token_id)
    return pil, proc, tid


def runtime_attach_image(rt: TtAgentRuntime, config: TTAgentConfig, img_path: Path) -> None:
    pil, proc, tid = _load_sticky_image_bundle(config, rt.hf_full, img_path)
    rt.sticky_pil = pil
    rt.auto_processor = proc
    rt.image_token_id = tid
    rt._img_rows_cache = None


def runtime_clear_image(rt: TtAgentRuntime) -> None:
    rt.sticky_pil = None
    rt.auto_processor = None
    rt.image_token_id = None
    rt._img_rows_cache = None


def _parse_tt_image_command(line: str) -> tuple[Path, str | None] | None:
    """Parse ``/image <path> [optional rest…]`` (prefix is case-insensitive)."""
    if len(line) < 6 or line[:6].lower() != "/image":
        return None
    rest = line[6:].strip()
    if not rest:
        return None
    try:
        parts = shlex.split(rest)
    except ValueError:
        return None
    if not parts:
        return None
    path = Path(parts[0]).expanduser()
    tail = " ".join(parts[1:]).strip()
    return path, tail if tail else None


def load_tt_runtime(config: TTAgentConfig) -> TtAgentRuntime:
    """Open mesh, load weights, build ``TtDevstral2SmallModel`` + LM head + optional ``SamplingGenerator``."""
    os.environ["HF_MODEL"] = config.model_id
    apply_devstral_hf_trust_patches()

    print(f"Loading tokenizer: {config.model_id}")
    _tok_kw: dict[str, Any] = {}
    if os.getenv("CI") == "true":
        _tok_kw["local_files_only"] = True
    try:
        tokenizer = MistralCommonBackend.from_pretrained(
            config.model_id, trust_remote_code=True, fix_mistral_regex=True, **_tok_kw
        )
    except (TypeError, ValueError):
        try:
            tokenizer = MistralCommonBackend.from_pretrained(config.model_id, trust_remote_code=True, **_tok_kw)
        except (TypeError, ValueError):
            tokenizer = MistralCommonBackend.from_pretrained(config.model_id, **_tok_kw)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    else:
        pad_token_id = int(pad_token_id)

    need = int(config.max_context_tokens) + int(config.max_new_tokens) + 2048
    mesh_device = open_devstral_demo_mesh(max(1, min(config.mesh_width, ttnn.get_num_devices())))
    try:
        dtype_tt = ttnn.bfloat16
        if config.max_seq_len is None:
            max_seq = max(4096, need)
            print(f"TT max_seq_len={max_seq} (default cap; need={need}).")
        else:
            if config.max_seq_len < need:
                print(f"Warning: --max-seq-len {config.max_seq_len} < need {need}; using {need} for ModelArgs.")
                max_seq = need
            else:
                max_seq = config.max_seq_len
            print(f"TT max_seq_len={max_seq} (explicit --max-seq-len; need={need}).")
        # Round up to a multiple of 512 so SDPA decode k_chunk_size is always >= 512 (a multiple of 32).
        max_seq = ((max_seq + 511) // 512) * 512

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
        )
        # Multi-chip: PREFILL is gathered to full-dim inside the decoder layer
        # (replicated norm), DECODE leaves residual width-fractured across chips
        # (distributed norm). Single chip: replicated norm for both modes.
        model_args.is_distributed_norm = types.MethodType(
            lambda self, mode: self.is_multichip and mode == Mode.DECODE,
            model_args,
        )

        print("Loading checkpoint via ModelArgs.load_state_dict() …")
        meta_state_dict = model_args.load_state_dict()

        if config.text_layers is not None:
            if config.text_layers < 1 or config.text_layers > model_args.full_model_n_layers:
                raise ValueError(
                    f"--text-layers must be in [1, {model_args.full_model_n_layers}], got {config.text_layers}"
                )
            model_args.n_layers = config.text_layers
            print(f"Warning: using --text-layers {config.text_layers}; quality will not match full depth.")

        hf_full = model_args.cached_hf_model
        if hf_full is None:
            raise RuntimeError("Expected cached HF model after load_state_dict with cache_hf=True.")
        hf_inner = hf_full.model
        if not isinstance(hf_inner, Mistral3Model):
            raise TypeError(f"Expected Mistral3Model, got {type(hf_inner)}")

        text_cfg = model_args.hf_config.text_config
        if not isinstance(text_cfg, Ministral3Config):
            raise TypeError(f"Expected Ministral3Config as text_config, got {type(text_cfg)!r}")

        vision_cfg = hf_full.config.vision_config
        shared_tt_ccl = TT_CCL(mesh_device)
        tt_devstral = TtDevstral2SmallModel(
            mesh_device=mesh_device,
            tt_ccl=shared_tt_ccl,
            model_args=model_args,
            meta_state_dict=meta_state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype_tt),
            dtype=dtype_tt,
            transformation_mats={"decode": None, "prefill": None},
            configuration=model_args,
            vision_config=vision_cfg,
            vision_n_layers=None,
        )
        tt_language_model = tt_devstral.language_model

        emb_layer = hf_inner.get_input_embeddings()
        _pad_dev = emb_layer.weight.device
        pad_row_1d = emb_layer(torch.tensor([[pad_token_id]], device=_pad_dev, dtype=torch.long))[0, 0].detach()

        sticky_pil: Optional[Image.Image] = None
        auto_processor: Optional[Any] = None
        image_token_id: Optional[int] = None
        if config.vision_image is not None:
            img_path = Path(config.vision_image)
            sticky_pil, auto_processor, image_token_id = _load_sticky_image_bundle(config, hf_full, img_path)
            print(f"Multimodal: sticky image {img_path.resolve()} (token id {image_token_id}).")

        sd_prefix = model_args.get_state_dict_prefix("", None)
        out_key = f"{sd_prefix}output.weight"
        if out_key not in meta_state_dict:
            raise RuntimeError(f"Missing {out_key!r} in meta state dict (required for LM head).")

        lm_head_weight_cpu: Optional[torch.Tensor] = None
        tt_lm_head: Optional[LMHead] = None
        if config.lm_head_cpu:
            lm_head_weight_cpu = meta_state_dict[out_key].detach().to(torch.bfloat16).cpu().contiguous()
            print(f"CPU LM head weight shape {tuple(lm_head_weight_cpu.shape)}.")
        else:
            lm_max = demo_lm_head_max_columns_per_device(model_args, cli_cap=config.lm_head_max_device_cols)
            tt_lm_head = LMHead(
                args=model_args,
                mesh_device=mesh_device,
                tt_ccl=shared_tt_ccl,
                dtype=dtype_tt,
                state_dict=meta_state_dict,
                state_dict_prefix=sd_prefix,
                weight_cache_path=model_args.weight_cache_path(dtype_tt),
                max_columns_per_device=lm_max,
            )

        use_device_sampling = (
            not config.lm_head_cpu
            and not config.cpu_sampling
            and tt_lm_head is not None
            and devstral_supports_on_device_sampling(model_args, mesh_device)
        )
        if (
            tt_lm_head is not None
            and not config.cpu_sampling
            and not devstral_supports_on_device_sampling(model_args, mesh_device)
        ):
            print("Warning: using CPU softmax / multinomial on TT logits (on-device sampling unsupported).")

        sampling: Optional[SamplingGenerator] = None
        sampling_empty_slots: Optional[List[int]] = None
        if use_device_sampling:
            sampling = SamplingGenerator(
                args=model_args,
                mesh_device=mesh_device,
                tt_ccl=shared_tt_ccl,
                enable_internal_trace=False,
            )
            sampling_empty_slots = list(range(sampling.tt_sampling.max_batch_size))
            seed_for_params = config.seed
            if not config.do_sample:
                sampling_in = SamplingParams(temperature=0.0, top_k=32, top_p=float(config.top_p), seed=seed_for_params)
            else:
                sampling_in = SamplingParams(
                    temperature=float(config.temperature),
                    top_k=32,
                    top_p=float(config.top_p),
                    seed=seed_for_params,
                )
            formatted_sampling = format_sampling_params(sampling_in, len(sampling_empty_slots))
            sampling.reset_sampling_params(formatted_sampling)
            sampling.seed_manager.reset_seed(formatted_sampling.seed, sampling_empty_slots)

        print("TT agent runtime ready.")
        return TtAgentRuntime(
            mesh_device=mesh_device,
            tokenizer=tokenizer,
            model_args=model_args,
            tt_devstral=tt_devstral,
            tt_language_model=tt_language_model,
            hf_full=hf_full,
            hf_inner=hf_inner,
            tt_lm_head=tt_lm_head,
            lm_head_weight_cpu=lm_head_weight_cpu,
            sampling=sampling,
            sampling_empty_slots=sampling_empty_slots,
            shared_tt_ccl=shared_tt_ccl,
            pad_token_id=pad_token_id,
            pad_row_1d=pad_row_1d,
            cfg=config,
            sticky_pil=sticky_pil,
            auto_processor=auto_processor,
            image_token_id=image_token_id,
            _img_rows_cache=None,
        )
    except Exception:
        close_devstral_demo_mesh(mesh_device)
        raise


def _ensure_decode_trace(rt: TtAgentRuntime, seed_token_id: int, seed_decode_pos: int) -> TtDecodeTraceContext:
    """Lazily build the session-wide single-token decode trace.

    Called once on the first decode step of the first generate call. The captured trace
    is reused across every following call (text or multimodal) because the kernel sequence
    for one decode step does not depend on the prompt — only on the KV cache contents,
    which the per-call prefill rewrites.
    """
    if rt.decode_trace_ctx is not None:
        return rt.decode_trace_ctx
    decode_buffers = tt_alloc_decode_input_buffers(rt.mesh_device)
    tt_update_decode_input_buffers(rt.mesh_device, decode_buffers, int(seed_token_id), int(seed_decode_pos))
    print("Capturing decode trace (warmup + capture)…")
    rt.decode_trace_ctx = tt_capture_decode_trace(
        rt.mesh_device,
        rt.tt_language_model,
        rt.model_args,
        decode_buffers,
        tt_lm_head=None if rt.cfg.lm_head_cpu else rt.tt_lm_head,
        sampling=rt.sampling,
    )
    print("Decode trace ready (reused across all subsequent turns).")
    return rt.decode_trace_ctx


def _sample_from_prefill_out_tt_agent(rt: TtAgentRuntime, tt_out: ttnn.Tensor, last_token_index: int) -> int:
    """Sample a single token id from the prefill hidden-states block at ``last_token_index``."""
    if rt.sampling is not None:
        assert rt.tt_lm_head is not None
        rt.sampling.seed_manager.get_new_values()
        logits_tt = tt_lm_head_logits_block(tt_out, last_token_index, rt.model_args, rt.tt_lm_head)
        sample_result = rt.sampling.sample(logits_tt, enable_trace=False)
        tt_next = sample_result[0] if isinstance(sample_result, tuple) else sample_result
        out = tt_sampling_output_token_id(tt_next, last_token_index % 32)
        ttnn.deallocate(logits_tt)
        return out
    if rt.cfg.lm_head_cpu:
        assert rt.lm_head_weight_cpu is not None
        logits_row = cpu_lm_head_logits_last_token(
            tt_out, last_token_index, rt.mesh_device, rt.lm_head_weight_cpu, int(rt.model_args.vocab_size)
        )
    else:
        assert rt.tt_lm_head is not None
        logits_row = tt_lm_head_logits_last_token(
            tt_out, last_token_index, rt.mesh_device, rt.model_args, rt.tt_lm_head
        )
    if bool(rt.cfg.do_sample):
        probs = torch.softmax(logits_row.float().squeeze(0) / max(float(rt.cfg.temperature), 1e-6), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(logits_row.argmax(dim=-1).item())


def _sample_next_from_decode_trace(rt: TtAgentRuntime) -> int:
    """Read one token id from a freshly replayed decode trace (handles all 3 trace modes)."""
    ctx = rt.decode_trace_ctx
    assert ctx is not None
    if ctx.output_tokens is not None:
        return tt_read_decode_traced_token(ctx, batch_slot=0)
    if ctx.output_logits is not None:
        logits_row = tt_read_decode_traced_logits(ctx, rt.mesh_device, rt.model_args, batch_slot=0)
        if bool(rt.cfg.do_sample):
            probs = torch.softmax(logits_row.float().squeeze(0) / max(float(rt.cfg.temperature), 1e-6), dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(logits_row.argmax(dim=-1).item())
    h_clone = tt_read_decode_traced_hidden(ctx, rt.mesh_device)
    try:
        return _sample_from_prefill_out_tt_agent(rt, h_clone, 0)
    finally:
        ttnn.deallocate(h_clone)


def generate_assistant_text_tt(rt: TtAgentRuntime, messages: List[Dict[str, str]], config: TTAgentConfig) -> str:
    """One prompt-prefill + per-token traced decode (text or sticky-image multimodal).

    Decode trace is captured lazily on the first call of the session and reused for every
    subsequent turn (see :func:`_ensure_decode_trace`).
    """
    pixel_values: Optional[torch.Tensor] = None
    image_sizes: Optional[torch.Tensor] = None
    image_sizes_list: Optional[list[tuple[int, int]]] = None

    if rt.sticky_pil is not None:
        assert rt.auto_processor is not None and rt.image_token_id is not None
        proc_messages = _inject_image_into_first_human_user(messages)
        prompt = rt.auto_processor.apply_chat_template(
            proc_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        proc_out = rt.auto_processor(text=prompt, images=rt.sticky_pil, return_tensors="pt")
        input_ids = proc_out["input_ids"]
        pixel_values = proc_out["pixel_values"]
        image_sizes = proc_out["image_sizes"]
        if not isinstance(image_sizes, torch.Tensor):
            raise TypeError(f"Expected image_sizes tensor from processor, got {type(image_sizes)}")
        image_sizes_list = _mlp._image_sizes_list_from_batch(image_sizes)
    else:
        input_ids = _tokenizer_apply_messages(rt.tokenizer, messages)

    prompt_len = int(input_ids.shape[1])
    if prompt_len > config.max_context_tokens:
        raise RuntimeError(
            f"Prompt is {prompt_len} tokens; exceeds --max-context-tokens ({config.max_context_tokens}). "
            "Use /clear or increase --max-context-tokens (and reload with matching TT budget)."
        )
    need_step = prompt_len + int(config.max_new_tokens) + 2048
    if need_step > int(rt.model_args.max_seq_len):
        raise RuntimeError(
            f"This prompt needs budget ≈{need_step} but ModelArgs max_seq_len={rt.model_args.max_seq_len}. "
            "Use /clear, lower --max-new-tokens, or restart with higher --max-context-tokens / --max-seq-len."
        )

    dev = rt.hf_inner.get_input_embeddings().weight.device
    input_ids = input_ids.to(dev)

    eos_ids = eos_token_ids(rt.hf_full.config, rt.tokenizer)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    if rt.sticky_pil is None:
        # ── Text-only: prefill the prompt once, then traced decode per new token. ──
        gen_sl = prompt_len
        ids_tt_gen = host_input_ids_to_tt_replicated(rt.mesh_device, input_ids)
        try:
            # Step 0: full TT prefill of the prompt → first sampled token.
            tt_out = tt_forward_prefill_from_device_ids(
                ids_tt_gen,
                gen_sl,
                rt.pad_token_id,
                rt.mesh_device,
                rt.tt_language_model,
                rt.model_args,
            )
            next_scalar = _sample_from_prefill_out_tt_agent(rt, tt_out, gen_sl - 1)
            ttnn.deallocate(tt_out)

            done_early = next_scalar in eos_ids or config.max_new_tokens <= 1
            if next_scalar not in eos_ids:
                ids_tt_gen = tt_append_uint32_token(ids_tt_gen, next_scalar, rt.mesh_device)
                gen_sl += 1

            if not done_early:
                ctx = _ensure_decode_trace(rt, seed_token_id=next_scalar, seed_decode_pos=gen_sl - 1)
                for _step in range(1, config.max_new_tokens):
                    decode_pos = gen_sl - 1
                    if rt.sampling is not None:
                        rt.sampling.seed_manager.get_new_values()
                    tt_update_decode_input_buffers(rt.mesh_device, ctx.buffers, int(next_scalar), decode_pos)
                    tt_execute_decode_trace(rt.mesh_device, ctx)
                    next_scalar = _sample_next_from_decode_trace(rt)
                    if next_scalar in eos_ids:
                        break
                    ids_tt_gen = tt_append_uint32_token(ids_tt_gen, next_scalar, rt.mesh_device)
                    gen_sl += 1

            ids_host = tt_replicated_ids_to_torch_long(rt.mesh_device, ids_tt_gen, gen_sl)
            answer_ids = ids_host[prompt_len:]
            return rt.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
        finally:
            ttnn.deallocate(ids_tt_gen)

    # ── Multimodal (sticky image): one prefill from merged embeds, then traced decode. ──
    assert pixel_values is not None and image_sizes is not None and image_sizes_list is not None
    pixel_values = pixel_values.to(torch.bfloat16).to(dev)
    image_sizes = image_sizes.to(dev)
    img_rows = _ensure_multimodal_img_rows(rt, pixel_values, image_sizes_list)
    assert rt.image_token_id is not None
    id_device = input_ids.device
    current_ids = input_ids.clone()

    # Step 0: prefill with image-merged embeddings → first sampled token.
    sl = int(current_ids.shape[1])
    merged = _mlp._merge_image_into_text_embeds(rt.hf_inner, current_ids, img_rows, rt.image_token_id)
    merged_bf = merged.to(torch.bfloat16)
    tt_out = _mlp._tt_prefill_from_merged_embeds(
        current_ids,
        merged_bf,
        rt.pad_row_1d,
        rt.pad_token_id,
        rt.mesh_device,
        rt.tt_language_model,
        rt.model_args,
        sl,
    )
    next_scalar = _sample_from_prefill_out_tt_agent(rt, tt_out, sl - 1)
    ttnn.deallocate(tt_out)

    if next_scalar in eos_ids or config.max_new_tokens <= 1:
        if next_scalar not in eos_ids:
            next_id = torch.tensor([[next_scalar]], device=id_device, dtype=torch.long)
            current_ids = torch.cat([current_ids, next_id], dim=1)
        answer_ids = current_ids[0, prompt_len:]
        return rt.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()

    next_id = torch.tensor([[next_scalar]], device=id_device, dtype=torch.long)
    current_ids = torch.cat([current_ids, next_id], dim=1)

    ctx = _ensure_decode_trace(rt, seed_token_id=next_scalar, seed_decode_pos=int(current_ids.shape[1]) - 1)
    for _step in range(1, config.max_new_tokens):
        decode_pos = int(current_ids.shape[1]) - 1
        if rt.sampling is not None:
            rt.sampling.seed_manager.get_new_values()
        tt_update_decode_input_buffers(rt.mesh_device, ctx.buffers, int(next_scalar), decode_pos)
        tt_execute_decode_trace(rt.mesh_device, ctx)
        next_scalar = _sample_next_from_decode_trace(rt)
        if next_scalar in eos_ids:
            break
        next_id = torch.tensor([[next_scalar]], device=id_device, dtype=torch.long)
        current_ids = torch.cat([current_ids, next_id], dim=1)

    answer_ids = current_ids[0, prompt_len:]
    return rt.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()


def chat_loop_tt(rt: TtAgentRuntime, config: TTAgentConfig) -> None:
    system_content = (f"{config.system_prompt}\n\n{DEFAULT_AGENT_RULES}\n\n{TT_AGENT_MULTIMODAL_SYSTEM_APPEND}").strip()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    state = AgentState()

    print(
        "\n--- Devstral2 Small Agent Demo (Tenstorrent)"
        + ("; image attached (multimodal)" if rt.sticky_pil is not None else "")
        + " ---"
    )
    print(
        "Type 'quit' or 'exit' to stop. '/clear' resets chat + tools (image stays; use '/noimage' to drop). "
        "'/image PATH [prompt…]' attaches an image; optional words after the path are your question this turn.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_content}]
            state = AgentState()
            print("Conversation history and todo state cleared.\n")
            continue
        if user_input.lower() == "/noimage":
            runtime_clear_image(rt)
            print("Cleared sticky image; text-only TT path until you /image again.\n")
            continue
        if len(user_input) >= 6 and user_input[:6].lower() == "/image":
            parsed = _parse_tt_image_command(user_input)
            if parsed is None:
                print("Usage: /image /path/to/image.jpg [optional question…]\n")
                continue
            img_path, same_turn_prompt = parsed
            try:
                runtime_attach_image(rt, config, img_path)
            except FileNotFoundError as exc:
                print(f"{exc}\n")
                continue
            print(f"Attached image: {img_path.resolve()} (--vision-square-pixels / --vision-max-edge apply).\n")
            if same_turn_prompt is None:
                continue
            user_input = same_turn_prompt

        messages.append({"role": "user", "content": user_input})
        final_response: Optional[str] = None
        last_tool_result: Optional[Dict[str, Any]] = None
        last_tool_name: Optional[str] = None
        last_tool_signature: Optional[str] = None
        repeated_tool_calls = 0

        py_cfg = ChatConfig(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            device="cpu",
            system_prompt=config.system_prompt,
            workspace_root=config.workspace_root,
            command_timeout_sec=config.command_timeout_sec,
            max_tool_calls_per_turn=config.max_tool_calls_per_turn,
        )

        for _ in range(config.max_tool_calls_per_turn):
            try:
                assistant_text = generate_assistant_text_tt(rt, messages, config)
            except RuntimeError as exc:
                final_response = f"[TT generation error] {exc}"
                messages.append({"role": "assistant", "content": final_response})
                break

            tool_call = parse_tool_call(assistant_text)

            if tool_call is None:
                final_response = assistant_text
                messages.append({"role": "assistant", "content": assistant_text})
                break

            messages.append({"role": "assistant", "content": assistant_text})
            call_signature = json.dumps(tool_call, sort_keys=True, ensure_ascii=True)
            if call_signature == last_tool_signature:
                repeated_tool_calls += 1
            else:
                repeated_tool_calls = 0
            last_tool_signature = call_signature

            tool_result = execute_tool_call(tool_call, py_cfg, state)
            last_tool_result = tool_result
            last_tool_name = str(tool_call.get("name", ""))
            result_content = f"<tool_result>\n{json.dumps(tool_result, ensure_ascii=True)}\n</tool_result>"
            messages.append({"role": "user", "content": result_content})

            if repeated_tool_calls >= 1:
                if tool_result.get("ok", False):
                    output = str(tool_result.get("output", "")).strip()
                    if output:
                        final_response = output
                    else:
                        final_response = f"Completed `{last_tool_name}` successfully."
                else:
                    err = str(tool_result.get("error", "")).strip()
                    out = str(tool_result.get("output", "")).strip()
                    details = err if err else out
                    final_response = f"`{last_tool_name}` failed." + (f" Details: {details}" if details else "")
                messages.append({"role": "assistant", "content": final_response})
                break

        if final_response is None:
            if last_tool_result is not None:
                if last_tool_result.get("ok", False):
                    output = str(last_tool_result.get("output", "")).strip()
                    final_response = output if output else "Tool execution completed successfully."
                else:
                    err = str(last_tool_result.get("error", "")).strip()
                    out = str(last_tool_result.get("output", "")).strip()
                    details = err if err else out
                    final_response = "Tool execution did not complete cleanly." + (
                        f" Details: {details}" if details else ""
                    )
            else:
                final_response = (
                    "I reached the per-turn tool-call limit before producing a final answer. "
                    "Try narrowing the request."
                )
            messages.append({"role": "assistant", "content": final_response})

        print(f"\nAssistant: {final_response}\n")


def parse_tt_args() -> TTAgentConfig:
    p = argparse.ArgumentParser(
        description="Interactive Devstral agent on Tenstorrent: text + in-session /image multimodal "
        "(optional --image to pre-attach)."
    )
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help=f"HF model id (default {DEFAULT_MODEL_ID})")

    # Agent (same names as demo_agent.py where possible)
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per model call")
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-sample", action="store_true", help="Greedy decoding")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--workspace-root", default=str(Path.cwd()))
    p.add_argument("--command-timeout-sec", type=int, default=20)
    p.add_argument("--max-tool-calls-per-turn", type=int, default=6)

    # TT
    p.add_argument("--mesh-width", type=int, default=1)
    p.add_argument("--text-layers", type=int, default=None)
    p.add_argument("--lm-head-cpu", action="store_true")
    p.add_argument("--lm-head-max-device-cols", type=int, default=None)
    p.add_argument("--cpu-sampling", action="store_true")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="S",
        help="RoPE/grid cap (default: device default vs need from max-context + gen + margin).",
    )
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=8192,
        metavar="N",
        help="Upper bound on **prompt** token length (budget is N + max_new_tokens + 2048, padded).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional: pre-attach image at startup (same as /image in the chat loop).",
    )
    p.add_argument(
        "--vision-max-edge",
        type=int,
        default=0,
        help="Max longest image side (px) before processor (0 = no PIL thumbnail). "
        "Use --vision-square-pixels (e.g. 1540) for fixed square sizing; ignored when square-pixels is set.",
    )
    p.add_argument(
        "--vision-square-pixels",
        type=int,
        default=None,
        help="If set (>0), resize image to S×S with LANCZOS (overrides --vision-max-edge).",
    )

    a = p.parse_args()

    return TTAgentConfig(
        model_id=a.model,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        do_sample=not a.no_sample,
        device="cpu",
        system_prompt=a.system_prompt,
        workspace_root=str(Path(a.workspace_root).resolve()),
        command_timeout_sec=a.command_timeout_sec,
        max_tool_calls_per_turn=a.max_tool_calls_per_turn,
        mesh_width=a.mesh_width,
        text_layers=a.text_layers,
        lm_head_cpu=a.lm_head_cpu,
        lm_head_max_device_cols=a.lm_head_max_device_cols,
        cpu_sampling=a.cpu_sampling,
        max_seq_len=a.max_seq_len,
        max_context_tokens=a.max_context_tokens,
        seed=a.seed,
        vision_image=a.image,
        vision_max_edge=a.vision_max_edge,
        vision_square_pixels=a.vision_square_pixels,
    )


def main() -> None:
    config = parse_tt_args()
    rt = load_tt_runtime(config)
    try:
        chat_loop_tt(rt, config)
    finally:
        if rt.decode_trace_ctx is not None:
            tt_release_decode_trace(rt.mesh_device, rt.decode_trace_ctx)
            rt.decode_trace_ctx = None
        close_devstral_demo_mesh(rt.mesh_device)


if __name__ == "__main__":
    main()
