# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from loguru import logger
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

import ttnn
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from models.experimental.tt_symbiote.core.run_config import DispatchManager, TracedRun
from models.perf.benchmarking_utils import BenchmarkProfiler
from torch import nn
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3NextGatedAttention
from models.experimental.tt_symbiote.modules.gated_deltanet import TTNNGatedDeltaNet
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3MoE
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


QWEN3_MODEL_ID = "Qwen/Qwen3-Coder-Next"

# When > 0, stop after N identical generated token ids in a row (can truncate answers). Default off.
QWEN3_SAME_TOKEN_STREAK_STOP = 0
QWEN3_RNG_SEED = 42


class _LatencyStreamer(BaseStreamer):
    """Lightweight streamer that records wall-clock timestamps for every token batch
    emitted by ``generate()``.  The first ``put()`` call corresponds to the prompt echo
    (or first-token emission); the gap between construction (or explicit ``reset()``) and
    that first call is the **real** Time-to-First-Token (TTFT).
    """

    def __init__(self):
        self._t_start: float = time.perf_counter()
        self._timestamps: List[float] = []

    def reset(self):
        self._t_start = time.perf_counter()
        self._timestamps.clear()

    def put(self, value):
        self._timestamps.append(time.perf_counter())

    def end(self):
        pass

    @property
    def ttft(self) -> float:
        """Seconds from start/reset to first token emission."""
        if not self._timestamps:
            return 0.0
        return self._timestamps[0] - self._t_start

    @property
    def per_token_latencies(self) -> List[float]:
        """Seconds between consecutive token emissions (decode steps)."""
        lats = []
        for i in range(1, len(self._timestamps)):
            lats.append(self._timestamps[i] - self._timestamps[i - 1])
        return lats

    @property
    def total_decode_time(self) -> float:
        """Wall time from first to last token emission (pure decode)."""
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    @property
    def n_decode_steps(self) -> int:
        """Number of decode steps (token emissions after the first)."""
        return max(0, len(self._timestamps) - 1)


def _model_max_context_length(model, tokenizer) -> int:
    """Best-effort maximum total sequence length for this checkpoint."""
    for attr in ("max_position_embeddings", "model_max_length"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    if tokenizer is not None:
        tml = getattr(tokenizer, "model_max_length", None)
        if isinstance(tml, int) and tml > 0 and tml < 10**9:
            return tml
    return 32768


def _compute_max_new_tokens(model, tokenizer, prompt_len: int) -> int:
    """Allow generation until EOS within remaining context (no fixed short cap)."""
    max_ctx = _model_max_context_length(model, tokenizer)
    reserve = 8
    available = max_ctx - int(prompt_len) - reserve
    if available < 1:
        pytest.fail(f"Prompt length {prompt_len} meets or exceeds model context {max_ctx}; cannot generate.")
    return available


def _main_max_new_tokens(available: int, main_max_new_tokens: int) -> int:
    """``main_max_new_tokens`` is the test parameter: >0 caps decode; <=0 uses full ``available``."""
    if main_max_new_tokens <= 0:
        return available
    return min(available, main_max_new_tokens)


# Run only on T3K (symbiote DeviceArch). @run_on_devices reads os.environ["MESH_DEVICE"], not a Python default.
if not (os.environ.get("MESH_DEVICE") or "").strip():
    os.environ["MESH_DEVICE"] = "T3K"
MESH_DEVICE = os.environ["MESH_DEVICE"].strip().upper()
MESH_SHAPE_T3K = (1, 8)


class StopOnConsecutiveSameToken(StoppingCriteria):
    """Stop when the same token id is generated ``streak`` times in a row (kills ``you you you`` tails)."""

    def __init__(self, streak: int = 6):
        self.streak = max(2, int(streak))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        if input_ids.shape[1] < self.streak:
            return torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
        tail = input_ids[:, -self.streak :]
        done = (tail == tail[:, :1]).all(dim=1)
        return done


def _stopping_criteria_list() -> StoppingCriteriaList:
    """Optional degenerate-tail stop; disabled when ``QWEN3_SAME_TOKEN_STREAK_STOP <= 0``."""
    if QWEN3_SAME_TOKEN_STREAK_STOP <= 0:
        return StoppingCriteriaList([])
    return StoppingCriteriaList([StopOnConsecutiveSameToken(QWEN3_SAME_TOKEN_STREAK_STOP)])


def _merge_generate_kw(base: Dict[str, Any]) -> Dict[str, Any]:
    sc = _stopping_criteria_list()
    if len(sc) == 0:
        return base
    return {**base, "stopping_criteria": sc}


def _clone_model_inputs(inputs: dict) -> dict:
    """Deep-enough copy so ``generate`` / warmup cannot alias or resize shared prompt tensors."""
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def _seed_torch_for_generate() -> None:
    """Reset PyTorch RNG before ``generate`` when ``do_sample=True`` (no ``generator=`` kwarg on this model)."""
    torch.manual_seed(QWEN3_RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(QWEN3_RNG_SEED)


def _generate_extra_kw(model, tokenizer) -> Dict[str, Any]:
    """Kwargs for ``generate`` from ``model.generation_config`` / tokenizer (no env overrides)."""
    kw: Dict[str, Any] = {}
    gc = model.generation_config

    eos = getattr(model.config, "eos_token_id", None)
    if eos is None and tokenizer is not None:
        eos = tokenizer.eos_token_id
    if eos is not None:
        kw["eos_token_id"] = eos
    pad = getattr(model.config, "pad_token_id", None)
    if pad is None and tokenizer is not None:
        pad = tokenizer.pad_token_id
    if pad is not None:
        kw["pad_token_id"] = pad

    rp = getattr(gc, "repetition_penalty", None)
    if rp is not None and float(rp) != 1.0:
        kw["repetition_penalty"] = float(rp)

    ngram = getattr(gc, "no_repeat_ngram_size", None)
    if ngram is not None and int(ngram) > 0:
        kw["no_repeat_ngram_size"] = int(ngram)

    do_sample = bool(getattr(gc, "do_sample", False))
    kw["do_sample"] = do_sample
    if do_sample:
        for name, key in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
        ):
            val = getattr(gc, key, None)
            if val is not None:
                kw[name] = val

    return kw


def _get_cached_model_path():
    """Resolve HF cache snapshot path for Qwen3-Coder-Next (avoids network)."""
    if hub := os.environ.get("HF_HUB_CACHE"):
        cache_root = Path(hub) / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    else:
        base = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        cache_root = Path(base) / "hub" / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    snapshots = cache_root / "snapshots"
    if not snapshots.exists():
        return None
    dirs = sorted(snapshots.iterdir())
    return str(dirs[0]) if dirs else None


def _load_qwen3_model():
    """Load tokenizer and model. Uses HF cache if available to avoid network."""
    try:
        cached = _get_cached_model_path()
        load_kw = dict(trust_remote_code=True, torch_dtype="auto", device_map="auto")
        if cached:
            tokenizer = AutoTokenizer.from_pretrained(cached, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(cached, local_files_only=True, **load_kw)
            return tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL_ID, **load_kw)
        return tokenizer, model
    except ModuleNotFoundError as e:
        if "triton" in str(e).lower():
            pytest.skip("Qwen3-Coder-Next requires triton for loading")
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(
            kw in msg
            for kw in (
                "network is unreachable",
                "name resolution",
                "connection",
                "protobuf",
                "does not appear to have",
            )
        ):
            pytest.skip(f"Model unavailable: {e}")
        raise


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_SHAPE_T3K],
    indirect=True,
)
@pytest.mark.parametrize(
    "main_max_new_tokens",
    [256],
    ids=["decode_max_256"],
)
def test_qwen3_coder_next(mesh_device, main_max_new_tokens: int):
    """Test Qwen3-Coder-Next model with TTNN acceleration (MoE + Gated Attention). Runs only on T3K.

    ``main_max_new_tokens``: cap on new tokens for the main ``generate`` (pytest param; add values or use ``<=0`` for full context).
    """
    if MESH_DEVICE != "T3K":
        pytest.skip(f"test_qwen3_coder_next runs only on T3K (MESH_DEVICE={os.environ.get('MESH_DEVICE')!r})")
    tokenizer, model = _load_qwen3_model()

    # All Linears → sharded TTNN except lm_head (always PyTorch for correct full-vocab logits).
    # Still PyTorch: embed_tokens, Qwen3NextRMSNorm, Qwen3NextRotaryEmbedding.
    nn_to_ttnn = {
        Qwen3NextSparseMoeBlock: TTNNQwen3MoE,
        Qwen3NextAttention: TTNNQwen3NextGatedAttention,
        Qwen3NextGatedDeltaNet: TTNNGatedDeltaNet,
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }
    exclude_replacement = {"lm_head"}

    messages = [
        {
            "role": "user",
            "content": "Write a quick sort algorithm.",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    def _to_device(v):
        if not isinstance(v, torch.Tensor):
            return v
        device = next(model.parameters()).device
        v = v.to(device)
        if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return v
        return v.to(torch.bfloat16)

    inputs = {k: _to_device(v) for k, v in inputs.items()}

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.config.eos_token_id

    all_modules = register_module_replacement_dict(
        model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_replacement
    )

    set_device(model, mesh_device, dump_visualization=False)

    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    batch_size = int(inputs["input_ids"].shape[0])
    profiler = BenchmarkProfiler()
    profiler.start("run")

    profiler.start("compile_prefill")
    warm_in = _clone_model_inputs(inputs)
    wk = _merge_generate_kw(_generate_extra_kw(model, tokenizer))
    if wk.get("do_sample", False):
        _seed_torch_for_generate()
    model.generate(**warm_in, max_new_tokens=2, use_cache=True, **wk)
    profiler.end("compile_prefill")

    DispatchManager.clear_timings()
    prompt_len = int(inputs["input_ids"].shape[-1])
    available = _compute_max_new_tokens(model, tokenizer, prompt_len)
    max_new = _main_max_new_tokens(available, main_max_new_tokens)
    logger.info(
        f"Main generate max_new_tokens={max_new} (param main_max_new_tokens={main_max_new_tokens}, "
        f"prompt_len={prompt_len}, context_budget={available})"
    )
    main_in = _clone_model_inputs(inputs)
    mk = _merge_generate_kw(_generate_extra_kw(model, tokenizer))
    if mk.get("do_sample", False):
        _seed_torch_for_generate()

    latency_streamer = _LatencyStreamer()
    latency_streamer.reset()
    profiler.start("inference_generate")
    outputs = model.generate(**main_in, max_new_tokens=max_new, use_cache=True, streamer=latency_streamer, **mk)
    profiler.end("inference_generate")

    profiler.end("run")

    output_ids = outputs[0][prompt_len:].tolist()
    num_new = len(output_ids)
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Qwen3-Coder-Next output ({num_new} new tokens):\n{content}")

    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = 0.0
    full_generation_time = profiler.get_duration("inference_generate")

    # Real TTFT from streamer: wall time from generate() entry to first token emission.
    # This includes the full prefill forward pass (but NOT warmup compile, which ran earlier).
    real_ttft = latency_streamer.ttft
    real_decode_time = latency_streamer.total_decode_time
    n_decode_steps = latency_streamer.n_decode_steps

    total_inference_prefill_time = real_ttft
    total_inference_decode_time = real_decode_time

    avg_time_to_first_token = total_inference_prefill_time / batch_size
    avg_decode_iteration_time = total_inference_decode_time / n_decode_steps if n_decode_steps > 0 else 0.0
    prefill_tok_s = prompt_len / total_inference_prefill_time * batch_size if total_inference_prefill_time > 0 else 0.0
    decode_tok_s_user = (
        n_decode_steps / total_inference_decode_time if n_decode_steps > 0 and total_inference_decode_time > 0 else 0.0
    )
    decode_tok_s = decode_tok_s_user * batch_size

    per_tok_lats = latency_streamer.per_token_latencies

    logger.info("")
    logger.info("=== Performance metrics ===")
    logger.info(
        f"Prompt tokens: {prompt_len} | Generated tokens: {num_new} | Decode steps (streamer): {n_decode_steps}"
    )
    logger.info("==")
    logger.info(f"Prefill compile time (warmup): {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s (included in warmup)")
    logger.info("")
    logger.info(f"Time to First Token (TTFT, measured): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Prefill throughput: {round(prefill_tok_s, 2)} tok/s "
        f"({prompt_len} tokens in {round(total_inference_prefill_time, 3)}s)"
    )
    logger.info("")
    logger.info(
        f"Decode speed: {round(avg_decode_iteration_time * 1000, 2)}ms/tok @ "
        f"{round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Decode wall time: {round(total_inference_decode_time, 2)}s for {n_decode_steps} steps")
    logger.info("")
    logger.info(f"Full generate() wall time: {round(full_generation_time, 2)}s")

    DispatchManager.save_stats_to_file("qwen3_coder_next_timing_stats.csv")
    TracedRun.release_all()
