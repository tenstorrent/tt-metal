# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent demo: Devstral Small 2 (Mistral3) **text** LM on TT, with **autoregressive generation**
using **TT** ``embed_tokens`` and **TT** ``TtMinistral3RotaryEmbedding`` inside ``TtMinistral3Model`` (via
``forward_prefill``), **TT** ``LMHead`` or ``--lm-head-cpu`` chunked torch logits, and **PyTorch** sampling on
the last-token logits (host). ``SamplingGenerator`` (fully on-device sampling) is not wired here; it needs
trace/param setup per the main ``Transformer`` path.

``--verify`` still builds an HF reference with **HF** embeddings for layer-wise PCC; TT side is full TT prefill.

**Prompting** defaults to the **same** chat template as ``reference/inference.py`` (``inference_fixtures``:
system prompt, user fibonacci/tools task, and tool schemas). Use ``--simple-chat`` for a single user turn
from ``--prompt`` only.

**Generation** repeats full TT **prefill** on the growing sequence (no TT decode/KV-cache yet), then
**TT** ``LMHead`` (or ``--lm-head-cpu``) on the last-token block and **CPU** sampling—matching ``inference.py``
setup requires **full** ``--text-layers`` (omit the flag) and sampling defaults aligned with
``REFERENCE_GENERATE_KWARGS``.

Usage (from repo root)::

    export HF_MODEL=mistralai/Devstral-Small-2-24B-Instruct-2512
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py --seed 0

    # Bring-up: fewer layers + PCC check
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py \\
        --text-layers 1 --verify --max-new-tokens 20

    # Optional: pure HF ``generate()`` baseline (not TT)
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py --hf-generate --seed 0

Embeddings are padded so sequence length is **128**-divisible, **KV-shard tile**-aligned, and (when
``L > 1024``) a multiple of **1024** for the attention ``wo`` chunk reshape; hidden states are sliced
back to the real length for PCC. Logits use **TT** ``LMHead`` on the last-token 32-row block.

If the LM head hits Wormhole L1 circular-buffer limits, use ``--lm-head-cpu`` (chunked torch matmul),
lower shards via ``--lm-head-max-device-cols``, or env ``DEVSTRAL2_LM_HEAD_MAX_COLUMNS_PER_DEVICE``.
"""

from __future__ import annotations

import argparse
import os
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import MistralCommonBackend
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs

try:
    from tests.scripts.common import get_updated_device_params
except ImportError:  # minimal fallback if tests package not on PYTHONPATH
    get_updated_device_params = lambda p: p  # type: ignore[assignment]

from models.experimental.devstarl2_small.reference.inference_fixtures import (
    REFERENCE_GENERATE_KWARGS,
    REFERENCE_MESSAGES,
    REFERENCE_TOOLS,
)

_DEFAULT_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
_DEMO_DIR = Path(__file__).resolve().parent

_ORIGINAL_FP8_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def _dequantize_one_compat(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if scales.ndim == 0:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        scale = scales.to(torch.float32)
        return (quantized_fp32 * scale).to(out_dtype)
    return _ORIGINAL_FP8_DEQUANTIZE_ONE(self, quantized, scales)


Fp8Dequantize._dequantize_one = _dequantize_one_compat


def _text_model_root(multimodal_inner: Mistral3Model):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def _apply_devstral_hf_trust_patches():
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    mc.ModelArgs._set_hf_params = _set_hf_params_trust  # type: ignore[method-assign]

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Demo supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    mc.ModelArgs.get_hf_model_cls = _get_hf_model_cls_devstral_safe  # type: ignore[method-assign]


def _tt_prefill_target_seqlen(seq_len: int, n_kv_heads: int, mesh_cluster_cols: int) -> int:
    """
    Prefill constraints (see ``models/tt_transformers/tt/attention.py`` ``forward_prefill``):

    - ``seq_len % 128 == 0`` (hard assert before QKV).
    - KV fill path: ``(n_kv // mesh_cols) * L // 64`` must be a multiple of the tile size (32), or
      ``interleaved_to_sharded`` fails.
    - When ``L > 1024``, ``wo`` reuses ``[1, L // 1024, 1024, H]``; **L must be a multiple of 1024** or
      the reshape is invalid and the next ``ttnn.linear`` sees the wrong inner dim (e.g. 5120 vs 4096).
    """
    k = n_kv_heads // mesh_cluster_cols
    assert k > 0
    target = seq_len if seq_len % 128 == 0 else seq_len + (128 - (seq_len % 128)) % 128
    for _ in range(32768):
        kv_ok = (k * target // 64) % 32 == 0
        wo_ok = target <= 1024 or (target % 1024 == 0)
        if kv_ok and wo_ok:
            return target
        target += 128
    raise RuntimeError("Could not find L satisfying TT prefill KV + WO chunking constraints.")


def _pad_input_ids_and_positions_for_tt_prefill(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor,
    pad_token_id: int,
    n_kv_heads: int,
    mesh_cluster_cols: int,
) -> tuple[torch.LongTensor, torch.LongTensor, int]:
    """Pad token ids and 2D position ids to a TT-valid prefill length (KV tile + optional WO chunk)."""
    seq_len = int(input_ids.shape[1])
    target = _tt_prefill_target_seqlen(seq_len, n_kv_heads, mesh_cluster_cols)
    if target == seq_len:
        return input_ids, position_ids, seq_len
    pad = target - seq_len
    input_ids_pad = F.pad(input_ids, (0, pad), value=pad_token_id)
    extra = torch.arange(seq_len, target, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0)
    position_ids_pad = torch.cat([position_ids, extra], dim=1)
    return input_ids_pad, position_ids_pad, seq_len


def _open_mesh(mesh_width: int):
    device_params = {
        "trace_region_size": 30000000,
        "num_command_queues": 1,
    }
    mesh_shape = ttnn.MeshShape(1, mesh_width)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **get_updated_device_params(device_params))


def _squeeze_tt_hidden_to_bsh(tt_lm_out: ttnn.Tensor, mesh_device, seq_len_keep: int) -> torch.Tensor:
    tt_h = ttnn.to_torch(tt_lm_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while tt_h.dim() > 3:
        tt_h = tt_h.squeeze(0)
    if tt_h.dim() == 2:
        tt_h = tt_h.unsqueeze(0)
    return tt_h[:, :seq_len_keep, :]


def _eos_token_ids(config, tokenizer=None) -> set[int]:
    """
    Collect EOS ids for stopping TT/HF loops.

    ``Mistral3ForConditionalGeneration`` often leaves ``config.eos_token_id`` unset while
    ``config.text_config.eos_token_id`` (and the tokenizer) still define EOS (e.g. 2). Missing this
    yields an empty set and generation never stops on end-of-sequence, which causes long repetition.
    """
    ids: set[int] = set()

    def _add(eos):
        if eos is None:
            return
        if isinstance(eos, (list, tuple)):
            ids.update(int(x) for x in eos)
        else:
            ids.add(int(eos))

    _add(getattr(config, "eos_token_id", None))
    tc = getattr(config, "text_config", None)
    if tc is not None:
        _add(getattr(tc, "eos_token_id", None))
    gen = getattr(config, "generation_config", None)
    if gen is not None:
        _add(getattr(gen, "eos_token_id", None))
    if tokenizer is not None:
        _add(getattr(tokenizer, "eos_token_id", None))
    return ids


def _tt_forward_prefill_from_ids(
    input_ids: torch.LongTensor,
    pad_token_id: int,
    mesh_device,
    tt_model: TtMinistral3Model,
    seq_len_keep: int,
    model_args: ModelArgs,
) -> ttnn.Tensor:
    """TT ``embed_tokens`` + in-model ``TtMinistral3RotaryEmbedding`` via ``forward_prefill``."""
    device = input_ids.device
    position_ids = torch.arange(int(input_ids.shape[1]), dtype=torch.long, device=device).unsqueeze(0)
    input_ids_pad, position_ids_pad, _ = _pad_input_ids_and_positions_for_tt_prefill(
        input_ids,
        position_ids,
        pad_token_id,
        int(model_args.n_kv_heads),
        int(model_args.cluster_shape[1]),
    )
    ids_tt = ttnn.from_torch(
        input_ids_pad.reshape(1, 1, 1, -1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_tt = ttnn.from_torch(
        position_ids_pad.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_model.forward_prefill(ids_tt, pos_tt)
    assert seq_len_keep <= int(input_ids_pad.shape[1])
    return tt_out


def _tt_prefill_hidden_states_from_ids(
    input_ids: torch.LongTensor,
    pad_token_id: int,
    mesh_device,
    tt_model: TtMinistral3Model,
    seq_len_keep: int,
    model_args: ModelArgs,
) -> torch.Tensor:
    tt_out = _tt_forward_prefill_from_ids(input_ids, pad_token_id, mesh_device, tt_model, seq_len_keep, model_args)
    return _squeeze_tt_hidden_to_bsh(tt_out, mesh_device, seq_len_keep)


def _demo_lm_head_max_columns_per_device(model_args: ModelArgs, cli_cap: int | None = None) -> int:
    """
    ``ModelArgs.max_columns_per_device_lm_head`` can still exceed Wormhole L1 for dram-sharded
    ``ttnn.linear`` (static circular buffers). Grids that already use ~16k caps need a **lower**
    ceiling than 24k—otherwise ``min(default, 24576)`` is a no-op.

    Defaults to **4096** columns per shard unless overridden (more matmul slices, smaller L1).
    """
    default = int(model_args.max_columns_per_device_lm_head)
    if cli_cap is not None:
        cap = max(1024, int(cli_cap))
    else:
        env = os.environ.get("DEVSTRAL2_LM_HEAD_MAX_COLUMNS_PER_DEVICE")
        if env is not None:
            cap = max(1024, int(env.strip()))
        else:
            cap = 4096
    return min(default, cap)


def _cpu_lm_head_logits_last_token(
    tt_hidden_prefill_out: ttnn.Tensor,
    last_token_index: int,
    mesh_device,
    weight_vd: torch.Tensor,
    vocab_size: int,
    chunk_v: int = 4096,
) -> torch.Tensor:
    """Chunked ``h @ W.T`` on host; ``weight_vd`` is ``[vocab, dim]`` like HF ``output.weight``."""
    get_last = (last_token_index // 32) * 32
    h_block = ttnn.slice(
        tt_hidden_prefill_out,
        (0, 0, get_last, 0),
        (1, 1, get_last + 32, tt_hidden_prefill_out.shape[-1]),
    )
    h_block = ttnn.to_memory_config(h_block, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h_t = ttnn.to_torch(h_block, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    ttnn.deallocate(h_block)
    while h_t.dim() > 3:
        h_t = h_t.squeeze(0)
    r = last_token_index % 32
    if h_t.dim() == 2:
        h_row = h_t[r].contiguous()
    else:
        h_row = h_t[0, r].contiguous()
    h_row = h_row.to(torch.float32)
    W = weight_vd.to(torch.float32)
    if W.ndim != 2:
        raise RuntimeError(f"LM head weight must be 2D, got {tuple(W.shape)}")
    d = int(h_row.shape[0])
    if W.shape[1] == d:
        pass
    elif W.shape[0] == d:
        W = W.T
    else:
        raise RuntimeError(f"LM head weight {tuple(W.shape)} incompatible with hidden dim {d}")
    vs = min(int(vocab_size), int(W.shape[0]))
    parts: list[torch.Tensor] = []
    for v0 in range(0, vs, chunk_v):
        v1 = min(v0 + chunk_v, vs)
        parts.append(h_row @ W[v0:v1].T)
    return torch.cat(parts, dim=-1).unsqueeze(0).contiguous()


def _tt_lm_head_logits_last_token(
    tt_hidden_prefill_out: ttnn.Tensor,
    last_token_index: int,
    mesh_device,
    model_args: ModelArgs,
    tt_lm_head: LMHead,
) -> torch.Tensor:
    """Run TT LM head on the 32-row prefill block that contains ``last_token_index``; return ``[1, vocab_size]``."""
    get_last = (last_token_index // 32) * 32
    h_block = ttnn.slice(
        tt_hidden_prefill_out,
        (0, 0, get_last, 0),
        (1, 1, get_last + 32, tt_hidden_prefill_out.shape[-1]),
    )
    lm_head_input_mem_cfg = model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg.is_sharded():
        h_block = ttnn.interleaved_to_sharded(h_block, lm_head_input_mem_cfg)
    logits = tt_lm_head(h_block)
    logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logits_torch = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while logits_torch.dim() > 3:
        logits_torch = logits_torch.squeeze(0)
    r = last_token_index % 32
    if logits_torch.dim() == 2:
        row = logits_torch[r : r + 1, :]
    else:
        row = logits_torch[0, r : r + 1, :]
    vs = int(model_args.vocab_size)
    if row.shape[-1] > vs:
        row = row[..., :vs]
    return row.contiguous()


def main():
    ref_max = int(REFERENCE_GENERATE_KWARGS["max_new_tokens"])
    ref_temp = float(REFERENCE_GENERATE_KWARGS["temperature"])
    ref_do_sample = bool(REFERENCE_GENERATE_KWARGS["do_sample"])

    parser = argparse.ArgumentParser(
        description="Devstral-2 on Tenstorrent: TT Ministral3 (embed+rotary+decoder) + TT LMHead; CPU sampling on logits."
    )
    parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID, help="HF repo id (also set HF_MODEL).")
    parser.add_argument(
        "--prompt",
        default=(
            "Can you implement in Python a method to compute the fibonnaci sequence at the `n`th element "
            "with `n` a parameter passed to the function? Start the sequence from 1."
        ),
        help="Only used with --simple-chat (single user turn).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message for --simple-chat only.",
    )
    parser.add_argument(
        "--simple-chat",
        action="store_true",
        help="Tokenize a minimal chat from --prompt/--system-prompt only. "
        "Default: same messages+tools as reference/inference.py (inference_fixtures).",
    )
    parser.add_argument(
        "--text-layers",
        type=int,
        default=None,
        help="Decoder layers on TT/HF cache after load (default: all). Required for quality matching full inference.",
    )
    parser.add_argument("--mesh-width", type=int, default=1, help="Device mesh width (1 × N).")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="PCC: HF reference (HF prompt embeddings) vs full TT prefill (TT embed + TT rotary).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=ref_max,
        help=f"New tokens after prompt (default {ref_max}, same as reference/inference.py). 0 = skip generation.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help=f"Greedy argmax (default: sample with temperature {ref_temp}, like reference/inference.py).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=ref_temp,
        help="Sampling temperature when not --greedy.",
    )
    parser.add_argument(
        "--hf-generate",
        action="store_true",
        help="Use Hugging Face model.generate() instead of TT stack + TT LM head (baseline only; not TT).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Torch/CUDA RNG seed before generation (recommended for sampling).",
    )
    parser.add_argument(
        "--lm-head-cpu",
        action="store_true",
        help="LM head as chunked torch matmul on CPU (avoids Wormhole L1 static CB limits on TT lm_head).",
    )
    parser.add_argument(
        "--lm-head-max-device-cols",
        type=int,
        default=None,
        help="On-device LMHead: max vocab columns per matmul shard (default 4096; try 2048 if L1 error remains).",
    )
    args = parser.parse_args()

    if args.hf_generate and args.text_layers is not None:
        logger.warning("--hf-generate uses full HF depth; ignoring --text-layers.")
        args.text_layers = None

    os.environ["HF_MODEL"] = args.model_id
    _apply_devstral_hf_trust_patches()

    mesh_device = _open_mesh(max(1, min(args.mesh_width, ttnn.get_num_devices())))
    try:
        dtype_tt = ttnn.bfloat16

        tokenizer = MistralCommonBackend.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
        if args.simple_chat:
            if args.system_prompt:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
                ]
                tokenized = tokenizer.apply_chat_template(
                    conversation=messages,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
                tokenized = tokenizer.apply_chat_template(
                    conversation=messages,
                    return_tensors="pt",
                    return_dict=True,
                )
        else:
            tokenized = tokenizer.apply_chat_template(
                conversation=REFERENCE_MESSAGES,
                tools=REFERENCE_TOOLS,
                return_tensors="pt",
                return_dict=True,
            )

        input_ids = tokenized["input_ids"]
        prompt_len = int(input_ids.shape[1])
        extra_tokens = max(0, args.max_new_tokens)
        # Upper bound padded prefill length: 128-step fixes for KV shard tile height + generation growth.
        max_seq = max(4096, prompt_len + extra_tokens + 2048)

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
        )
        model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)

        logger.info("Loading checkpoint via ModelArgs.load_state_dict() …")
        try:
            meta_state_dict = model_args.load_state_dict()
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint load failed (memory, hub, FP8, etc.): {exc}\n"
                "Ensure HF access, enough RAM, and compatible transformers."
            ) from exc

        if args.text_layers is not None:
            if args.text_layers < 1 or args.text_layers > model_args.full_model_n_layers:
                raise ValueError(
                    f"--text-layers must be in [1, {model_args.full_model_n_layers}], got {args.text_layers}"
                )
            model_args.n_layers = args.text_layers
            if args.max_new_tokens > 0 and not args.hf_generate:
                logger.warning(
                    "Partial --text-layers: TT generation will not match full-model reference/inference.py quality."
                )

        hf_full = model_args.cached_hf_model
        if hf_full is None:
            raise RuntimeError("Expected cached HF model after load_state_dict with cache_hf=True.")
        hf_inner = hf_full.model
        if not isinstance(hf_inner, Mistral3Model):
            raise TypeError(f"Expected Mistral3Model, got {type(hf_inner)}")

        text_cfg = model_args.hf_config.text_config
        if not isinstance(text_cfg, Ministral3Config):
            raise TypeError(f"Demo expects Ministral3Config as text_config, got {type(text_cfg)!r}")
        rope_params = getattr(text_cfg, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = dict(rope_params)

        tt_model = TtMinistral3Model(
            mesh_device=mesh_device,
            tt_ccl=TT_CCL(mesh_device),
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
        lm_head_weight_cpu: torch.Tensor | None = None
        tt_lm_head: LMHead | None = None
        if args.lm_head_cpu:
            lm_head_weight_cpu = meta_state_dict[out_key].detach().to(torch.bfloat16).cpu().contiguous()
            logger.info(
                f"CPU LM head: chunked torch matmul; weight {tuple(lm_head_weight_cpu.shape)} {lm_head_weight_cpu.dtype}."
            )
        else:
            lm_head_max_cols = _demo_lm_head_max_columns_per_device(model_args, cli_cap=args.lm_head_max_device_cols)
            logger.info(
                f"On-device LMHead max columns per shard: {lm_head_max_cols} "
                f"(ModelArgs value {model_args.max_columns_per_device_lm_head})."
            )
            if lm_head_max_cols < int(model_args.max_columns_per_device_lm_head):
                logger.info(
                    "Tune with --lm-head-max-device-cols or DEVSTRAL2_LM_HEAD_MAX_COLUMNS_PER_DEVICE, "
                    "or use --lm-head-cpu if L1 CB errors persist."
                )
            tt_lm_head = LMHead(
                args=model_args,
                mesh_device=mesh_device,
                tt_ccl=TT_CCL(mesh_device),
                dtype=dtype_tt,
                state_dict=meta_state_dict,
                state_dict_prefix=sd_prefix,
                weight_cache_path=model_args.weight_cache_path(dtype_tt),
                max_columns_per_device=lm_head_max_cols,
            )
        _sampling_splits = model_args.num_devices if list(mesh_device.shape) != [1, 1] else 2
        if model_args.vocab_size // _sampling_splits <= 64 * 1024:
            logger.info(
                "Mesh/vocab would allow Transformer on-device sampling; this demo still uses CPU softmax/multinomial on TT logits."
            )

        input_ids = input_ids.to(hf_inner.get_input_embeddings().weight.device)
        seq_len_lm = int(input_ids.shape[1])
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        else:
            pad_token_id = int(pad_token_id)

        target_lm = _tt_prefill_target_seqlen(seq_len_lm, int(model_args.n_kv_heads), int(model_args.cluster_shape[1]))
        if target_lm != seq_len_lm:
            logger.info(
                f"Padded language sequence length {seq_len_lm} → {target_lm} for TT prefill "
                "(128-divisible seq + KV prefill shard tile alignment)."
            )

        logger.info(
            f"TT language model prefill ({model_args.n_layers} decoder layers; TT embed + TtMinistral3RotaryEmbedding); "
            f"prompt template: {'simple-chat' if args.simple_chat else 'inference_fixtures (inference.py)'}."
        )
        tt_lm_torch = _tt_prefill_hidden_states_from_ids(
            input_ids, pad_token_id, mesh_device, tt_model, seq_len_lm, model_args
        )

        logger.info(f"TT prefill hidden states shape (batch, seq, dim): {tuple(tt_lm_torch.shape)}")

        if args.verify:
            text_root = _text_model_root(hf_inner)
            rotary = text_root.rotary_emb
            rotary.eval()
            merged = hf_inner.get_input_embeddings()(input_ids).to(torch.bfloat16)
            position_ids_lm = torch.arange(seq_len_lm, dtype=torch.long, device=merged.device).unsqueeze(0)
            position_embeddings_hf = rotary(merged, position_ids=position_ids_lm)
            causal_mask = create_causal_mask(
                config=text_cfg,
                inputs_embeds=merged,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids_lm,
            )
            hidden = merged
            for layer in text_root.layers[: model_args.n_layers]:
                hidden = layer(
                    hidden_states=hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids_lm,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings_hf,
                )
            ref_out = text_root.norm(hidden)
            tt_cmp = tt_lm_torch
            if tt_cmp.shape != ref_out.shape:
                tt_cmp = tt_cmp.reshape(ref_out.shape)
            pcc_ok, msg = comp_pcc(ref_out, tt_cmp, 0.90)
            logger.info(comp_allclose(ref_out, tt_cmp))
            logger.info(f"PCC (HF ref on HF embeddings vs full TT prefill): {msg}")
            if not pcc_ok:
                logger.warning(f"PCC check did not reach threshold: {msg}")

        if args.max_new_tokens <= 0:
            pass
        elif args.hf_generate:
            gen_device = next(hf_full.parameters()).device
            if args.seed is not None:
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
            prompt_vec = tokenized["input_ids"][0]
            input_ids_gen = tokenized["input_ids"].to(gen_device)
            logger.info(f"HF baseline generate {REFERENCE_GENERATE_KWARGS} on {gen_device} …")
            with torch.inference_mode():
                out = hf_full.generate(input_ids_gen, **REFERENCE_GENERATE_KWARGS)
            seq = out[0]
            answer_text = tokenizer.decode(seq[len(prompt_vec) :].tolist(), skip_special_tokens=False)
            logger.info(f"HF generate baseline ({seq.numel() - prompt_vec.numel()} new tokens):\n{answer_text}")
        else:
            do_sample = ref_do_sample if not args.greedy else False
            if args.seed is not None:
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)

            id_device = input_ids.device
            eos_ids = _eos_token_ids(hf_full.config, tokenizer)
            if not eos_ids:
                logger.warning(
                    "No eos_token_id on config/text_config/tokenizer; TT loop will only stop at --max-new-tokens."
                )
            current_ids = input_ids.clone()
            mode = "greedy" if args.greedy else f"sample (T={args.temperature})"
            lm_mode = "CPU lm_head (chunked torch)" if args.lm_head_cpu else "TT lm_head"
            logger.info(
                f"TT generation: up to {args.max_new_tokens} new tokens, {mode}; "
                f"TT decoder + {lm_mode}; sampling on CPU from last-token logits ({id_device})."
            )
            for _step in range(args.max_new_tokens):
                sl = int(current_ids.shape[1])
                tt_out = _tt_forward_prefill_from_ids(current_ids, pad_token_id, mesh_device, tt_model, sl, model_args)
                if args.lm_head_cpu:
                    assert lm_head_weight_cpu is not None
                    logits_row = _cpu_lm_head_logits_last_token(
                        tt_out, sl - 1, mesh_device, lm_head_weight_cpu, int(model_args.vocab_size)
                    )
                else:
                    assert tt_lm_head is not None
                    logits_row = _tt_lm_head_logits_last_token(tt_out, sl - 1, mesh_device, model_args, tt_lm_head)
                ttnn.deallocate(tt_out)
                if do_sample:
                    probs = torch.softmax(logits_row.float().squeeze(0) / max(args.temperature, 1e-6), dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).view(1, 1)
                else:
                    next_id = logits_row.argmax(dim=-1, keepdim=True)
                next_id = next_id.to(id_device)
                if int(next_id.item()) in eos_ids:
                    break
                current_ids = torch.cat([current_ids, next_id], dim=1)

            answer_ids = current_ids[0, prompt_len:]
            answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
            logger.info(f"TT stack + TT lm_head generated ({answer_ids.numel()} tokens):\n{answer_text}")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
