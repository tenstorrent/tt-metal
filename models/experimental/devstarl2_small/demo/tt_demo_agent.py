# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent variant of ``demo_agent.py``: same interactive agent tools and chat loop, but **text
generation** runs on **TT** (``TtMinistral3Model`` + LM head + ``SamplingGenerator`` / CPU logits),
mirroring ``demo_devstral2_tt_multimodal.py``.

**Scope:** plain-string chat messages (no multimodal images). Tool I/O and workspace rules are
identical to the PyTorch ``demo_agent.py``.

**Context budget:** TT KV is allocated for ``--max-context-tokens + max_new_tokens + 2048`` (padded to
TT rules). If the running chat exceeds ``max_context-tokens`` prompt tokens, the demo errors and
asks you to ``/clear`` or raise the limit.

Usage (repo root)::

    python models/experimental/devstarl2_small/demo/tt_demo_agent.py --mesh-width 1

    # Wormhole-style cap (default max context 8192 prompt tokens unless overridden):
    python models/experimental/devstarl2_small/demo/tt_demo_agent.py --max-context-tokens 4096 \\
        --lm-head-cpu --text-layers 2
"""

from __future__ import annotations

import argparse
import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import ttnn
from transformers import MistralCommonBackend
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
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
    DEVSTRAL_DEMO_BLACKHOLE_DEFAULT_MAX_SEQ_LEN,
    apply_devstral_hf_trust_patches,
    apply_fp8_dequantize_compat,
    cpu_lm_head_logits_last_token,
    demo_lm_head_max_columns_per_device,
    devstral_supports_on_device_sampling,
    devstral_tt_kv_cache_max_seq_len,
    default_devstral_demo_max_seq_len,
    eos_token_ids,
    host_input_ids_to_tt_replicated,
    open_devstral_demo_mesh,
    tt_append_uint32_token,
    tt_forward_prefill_from_device_ids,
    tt_lm_head_logits_block,
    tt_lm_head_logits_last_token,
    tt_replicated_ids_to_torch_long,
    tt_sampling_output_token_id,
)
from models.experimental.devstarl2_small.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()


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


@dataclass
class TtAgentRuntime:
    mesh_device: ttnn.MeshDevice
    tokenizer: MistralCommonBackend
    model_args: ModelArgs
    tt_model: TtMinistral3Model
    hf_full: torch.nn.Module
    hf_inner: Mistral3Model
    tt_lm_head: Optional[LMHead]
    lm_head_weight_cpu: Optional[torch.Tensor]
    sampling: Optional[SamplingGenerator]
    sampling_empty_slots: Optional[List[int]]
    shared_tt_ccl: TT_CCL
    pad_token_id: int
    cfg: TTAgentConfig


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


def load_tt_runtime(config: TTAgentConfig) -> TtAgentRuntime:
    """Open mesh, load weights, build ``TtMinistral3Model`` + LM head + optional ``SamplingGenerator``."""
    os.environ["HF_MODEL"] = config.model_id
    apply_devstral_hf_trust_patches()

    print(f"Loading tokenizer: {config.model_id}")
    tokenizer = MistralCommonBackend.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
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
            max_seq = default_devstral_demo_max_seq_len(mesh_device, need)
            _bh = ttnn.device.is_blackhole(mesh_device)
            print(
                f"TT max_seq_len={max_seq} (device default; blackhole={_bh}"
                + (f", floor={DEVSTRAL_DEMO_BLACKHOLE_DEFAULT_MAX_SEQ_LEN}" if _bh else "")
                + f"; need={need})."
            )
        else:
            if config.max_seq_len < need:
                print(f"Warning: --max-seq-len {config.max_seq_len} < need {need}; using {need} for ModelArgs.")
                max_seq = need
            else:
                max_seq = config.max_seq_len
            print(f"TT max_seq_len={max_seq} (explicit --max-seq-len; need={need}).")

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
        )
        model_args.max_kv_cache_seq_len = devstral_tt_kv_cache_max_seq_len(model_args, need)
        print(
            f"KV cache tensor seq dim={model_args.max_kv_cache_seq_len} "
            f"(RoPE max_seq_len={max_seq}; budget need≈{need})."
        )
        model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)

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
        rope_params = getattr(text_cfg, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = dict(rope_params)

        shared_tt_ccl = TT_CCL(mesh_device)
        tt_model = TtMinistral3Model(
            mesh_device=mesh_device,
            tt_ccl=shared_tt_ccl,
            model_args=model_args,
            meta_state_dict=meta_state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype_tt),
            dtype=dtype_tt,
            transformation_mats={"decode": None, "prefill": None},
            configuration=model_args,
            llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
            original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
            ministral_text_config=text_cfg,
        )

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
            tt_model=tt_model,
            hf_full=hf_full,
            hf_inner=hf_inner,
            tt_lm_head=tt_lm_head,
            lm_head_weight_cpu=lm_head_weight_cpu,
            sampling=sampling,
            sampling_empty_slots=sampling_empty_slots,
            shared_tt_ccl=shared_tt_ccl,
            pad_token_id=pad_token_id,
            cfg=config,
        )
    except Exception:
        ttnn.close_mesh_device(mesh_device)
        raise


def generate_assistant_text_tt(rt: TtAgentRuntime, messages: List[Dict[str, str]], config: TTAgentConfig) -> str:
    """Autoregressive decode on TT (full prefill each new token), same pattern as ``demo_devstral2_tt_multimodal``."""
    input_ids = _tokenizer_apply_messages(rt.tokenizer, messages)
    prompt_len = int(input_ids.shape[1])
    if prompt_len > config.max_context_tokens:
        raise RuntimeError(
            f"Prompt is {prompt_len} tokens; exceeds --max-context-tokens ({config.max_context_tokens}). "
            "Use /clear or increase --max-context-tokens (and reload with matching TT budget)."
        )
    need_step = prompt_len + int(config.max_new_tokens) + 2048
    if need_step > int(rt.model_args.max_kv_cache_seq_len) or need_step > int(rt.model_args.max_seq_len):
        raise RuntimeError(
            f"This prompt needs budget ≈{need_step} but ModelArgs caps are "
            f"max_kv_cache_seq_len={rt.model_args.max_kv_cache_seq_len}, max_seq_len={rt.model_args.max_seq_len}. "
            "Use /clear, lower --max-new-tokens, or restart with higher --max-context-tokens / --max-seq-len."
        )

    dev = rt.hf_inner.get_input_embeddings().weight.device
    input_ids = input_ids.to(dev)

    eos_ids = eos_token_ids(rt.hf_full.config, rt.tokenizer)
    do_sample = bool(config.do_sample)
    gen_sl = prompt_len
    ids_tt_gen = host_input_ids_to_tt_replicated(rt.mesh_device, input_ids)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    try:
        for _step in range(config.max_new_tokens):
            tok_slot = (gen_sl - 1) % 32
            if rt.sampling is not None:
                rt.sampling.seed_manager.get_new_values()
            tt_out = tt_forward_prefill_from_device_ids(
                ids_tt_gen,
                gen_sl,
                rt.pad_token_id,
                rt.mesh_device,
                rt.tt_model,
                rt.model_args,
            )
            if rt.sampling is not None:
                assert rt.tt_lm_head is not None
                logits_tt = tt_lm_head_logits_block(tt_out, gen_sl - 1, rt.model_args, rt.tt_lm_head)
                sample_result = rt.sampling.sample(logits_tt, enable_trace=False)
                tt_next = sample_result[0] if isinstance(sample_result, tuple) else sample_result
                next_scalar = tt_sampling_output_token_id(tt_next, tok_slot)
                ttnn.deallocate(logits_tt)
                ttnn.deallocate(tt_out)
            else:
                if config.lm_head_cpu:
                    assert rt.lm_head_weight_cpu is not None
                    logits_row = cpu_lm_head_logits_last_token(
                        tt_out,
                        gen_sl - 1,
                        rt.mesh_device,
                        rt.lm_head_weight_cpu,
                        int(rt.model_args.vocab_size),
                    )
                else:
                    assert rt.tt_lm_head is not None
                    logits_row = tt_lm_head_logits_last_token(
                        tt_out,
                        gen_sl - 1,
                        rt.mesh_device,
                        rt.model_args,
                        rt.tt_lm_head,
                    )
                ttnn.deallocate(tt_out)
                if do_sample:
                    logits_f = logits_row.float().squeeze(0) / max(float(config.temperature), 1e-6)
                    probs = torch.softmax(logits_f, dim=-1)
                    next_scalar = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_scalar = int(logits_row.argmax(dim=-1).item())

            if next_scalar in eos_ids:
                break
            ids_tt_gen = tt_append_uint32_token(ids_tt_gen, next_scalar, rt.mesh_device)
            gen_sl += 1

        ids_host = tt_replicated_ids_to_torch_long(rt.mesh_device, ids_tt_gen, gen_sl)
        answer_ids = ids_host[prompt_len:]
        return rt.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
    finally:
        ttnn.deallocate(ids_tt_gen)


def chat_loop_tt(rt: TtAgentRuntime, config: TTAgentConfig) -> None:
    system_content = f"{config.system_prompt}\n\n{DEFAULT_AGENT_RULES}".strip()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    state = AgentState()

    print("\n--- Devstral2 Small Agent Demo (Tenstorrent) ---")
    print("Type 'quit' or 'exit' to stop. Use '/clear' to reset conversation history.\n")

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
    p = argparse.ArgumentParser(description="Interactive Devstral agent on Tenstorrent (text TT LM).")
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
    )


def main() -> None:
    config = parse_tt_args()
    rt = load_tt_runtime(config)
    try:
        chat_loop_tt(rt, config)
    finally:
        ttnn.close_mesh_device(rt.mesh_device)


if __name__ == "__main__":
    main()
