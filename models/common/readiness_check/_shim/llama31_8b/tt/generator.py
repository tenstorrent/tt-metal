# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Readiness-check shim for Llama 3.1 8B Instruct (tt-transformers backend).

This module exists **only to exercise the readiness-check verification
flow** against an already-working tt-transformers model — it is not a
real auto-ported model. It lives under `models/autoports/` alongside
the actual ported models that the porting pipeline produces.

The shim is a thin wrapper around the existing tt-transformers
`Generator`. The tt-transformers stack is parameterised by env vars
(`HF_MODEL`, `MESH_DEVICE`); this shim sets them and exposes the
`models.common.readiness_check.contract.GeneratorBase` surface so the
readiness runner can drive the model the same way it drives any newly
ported model.

The shim does not introduce new behavior — it only adapts the existing
batched, multi-user-oriented `Generator.prefill_forward_text` /
`decode_forward` into the single-user, host-sampling `generate()` loop
the readiness check needs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import torch

from models.common.readiness_check.contract import GeneratorBase, NextInputFn

#: tt_transformers' `HF_MODEL` env var accepts either a HuggingFace repo id
#: or a local checkpoint directory. We default to the local path so the
#: shim works on boxes without HF authentication (e.g. CI). Override via
#: build_generator(..., hf_model_id=...) when a different checkpoint
#: location is desired.
DEFAULT_HF_MODEL_ID = "/proj_sw/user_dev/llama31-8b-data/Llama-3.1-8B-Instruct"
DEFAULT_MESH_DEVICE = "N150"

# tt_transformers requires prefill length to be a power of two; the floor is
# typically 128 in the demos. Anything below 128 is rounded up.
_MIN_PREFILL_LEN = 128


def _round_up_to_power_of_two(n: int, floor: int = _MIN_PREFILL_LEN) -> int:
    target = max(floor, 1)
    while target < n:
        target *= 2
    return target


#: Writable cache path for tt_transformers weight + tensor caches. We set
#: it explicitly because tt_transformers derives its default from HF_MODEL,
#: and when HF_MODEL is an absolute local path (which it is here, to avoid
#: HF auth), the default ends up writing into the read-only checkpoint
#: directory.
DEFAULT_TT_CACHE_DIR = Path(__file__).resolve().parents[6] / "model_cache" / "llama31_8b_readiness_shim"


def _set_env_for_tt_transformers(hf_model_id: str, mesh_device_label: str) -> None:
    """tt_transformers reads HF_MODEL / MESH_DEVICE / TT_CACHE_PATH during ModelArgs init."""
    os.environ.setdefault("HF_MODEL", hf_model_id)
    os.environ.setdefault("MESH_DEVICE", mesh_device_label)
    if "TT_CACHE_PATH" not in os.environ:
        cache_dir = DEFAULT_TT_CACHE_DIR / mesh_device_label
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TT_CACHE_PATH"] = str(cache_dir)


class Llama31_8BReadinessGenerator(GeneratorBase):
    """
    Readiness-check generator for Llama 3.1 8B Instruct on a single-device
    mesh (default N150). Internally constructs the tt-transformers
    `Generator`, allocates a paged KV cache + page table, and exposes the
    contract surface.
    """

    def __init__(
        self,
        *,
        mesh_device,
        max_new_tokens: int,
        max_prefill_len: int,
        # Physical batch the device is configured for. The readiness check
        # only uses 1 logical user; tt_transformers' demo batch-1 test
        # also runs with max_batch_size=1.
        max_batch_size: int = 1,
        page_block_size: int = 32,
        page_max_num_blocks: int = 1024,
        instruct: bool = True,
        dtype=None,
    ) -> None:
        # Lazy imports — these pull in the entire tt_transformers stack,
        # which only loads cleanly when the shim is actually being built
        # on a host with ttnn + a real mesh device.
        import ttnn
        from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
        from models.tt_transformers.tt.generator import Generator as TTTGenerator
        from models.tt_transformers.tt.model_config import DecodersPrecision

        self._ttnn = ttnn
        self._mesh_device = mesh_device

        self._paged_attention_config = PagedAttentionConfig(
            block_size=page_block_size,
            max_num_blocks=page_max_num_blocks,
        )

        # tt_transformers requires max_seq_len to be a power of 2 (used by
        # the prefill warmup machinery to enumerate supported seq lens).
        requested_max_seq_len = max(_MIN_PREFILL_LEN, max_prefill_len + max_new_tokens)
        self._max_seq_len = _round_up_to_power_of_two(requested_max_seq_len, floor=_MIN_PREFILL_LEN)

        self._model_args, self._model, _tt_kv_cache_single, _ = create_tt_model(
            mesh_device=mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            optimizations=lambda m: DecodersPrecision.performance(m.n_layers, m.model_name),
            max_seq_len=self._max_seq_len,
            paged_attention_config=self._paged_attention_config,
            dtype=dtype if dtype is not None else ttnn.bfloat8_b,
        )
        # Generator.prefill_forward_text indexes kv_cache as
        # kv_cache[submesh_id][layer_id][k_or_v]. create_tt_model returns
        # only the inner [layer_id][k_or_v] list, so we wrap it once for
        # the single-submesh (data_parallel=1) case — matching what
        # simple_text_demo.py builds via prepare_generator_args.
        self._tt_kv_cache = [_tt_kv_cache_single]

        self._page_table = self._build_page_table(global_batch_size=max_batch_size, data_parallel=1)

        self._inner = TTTGenerator(
            model=[self._model],
            model_args=[self._model_args],
            mesh_device=mesh_device,
            processor=getattr(self._model_args, "processor", None),
            tokenizer=self._model_args.tokenizer,
        )

        # GeneratorBase contract surface.
        self.tokenizer = self._model_args.tokenizer

        # Internal state to track whether the next decode step is the first
        # after a fresh prefill (controls reset_batch in decode_forward).
        self._needs_decode_reset = True

    # --- helpers ---------------------------------------------------------

    def _build_page_table(self, *, global_batch_size: int, data_parallel: int) -> torch.Tensor:
        cfg = self._paged_attention_config
        permutation = torch.randperm(cfg.max_num_blocks)
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        return reverse_permutation.reshape(
            global_batch_size,
            cfg.max_num_blocks // (global_batch_size // data_parallel),
        )

    def _prompt_to_tensor(self, prompt_token_ids: List[int]) -> torch.Tensor:
        """Pack prompt as [1, real_len] int32 — tt_transformers' prefill_forward_text
        does its own power-of-two padding internally, so we must not pre-pad."""
        return torch.tensor([list(prompt_token_ids)], dtype=torch.int32)

    # --- GeneratorBase low-level API ------------------------------------

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._inner.prefill_forward_text(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=None,
            enable_trace=kwargs.get("enable_trace", True),
            warmup_prefill=kwargs.get("warmup_prefill", False),
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._inner.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=kwargs.get("enable_trace", True),
            read_from_device=kwargs.get("read_from_device", True),
            sampling_params=None,
            reset_batch=kwargs.get("reset_batch", False),
        )

    # --- GeneratorBase high-level API -----------------------------------

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        **kwargs: Any,
    ) -> List[int]:
        if max_new_tokens <= 0:
            return []

        input_tensor = self._prompt_to_tensor(prompt_token_ids)
        real_prompt_len = len(prompt_token_ids)
        prompt_lens = [real_prompt_len]  # list[int], matching preprocess_inputs_prefill

        # One-shot prefill — no separate warmup pass; the readiness check
        # tolerates the first call being slow.
        prefill_out = self.prefill_forward(
            input_tensor,
            page_table=self._page_table,
            kv_cache=self._tt_kv_cache,
            prompt_lens=prompt_lens,
        )
        # With sampling_params=None, prefill_forward_text returns logits.
        # If a future tt_transformers version starts returning a tuple, the
        # same unwrap as decode handles it.
        prefill_logits = prefill_out[0] if isinstance(prefill_out, tuple) else prefill_out
        first_pred = int(torch.argmax(prefill_logits[0, 0, :], dim=-1).item())
        predictions: List[int] = [first_pred]

        # Decide the next input. If teacher forcing is active, the
        # callback observes our prediction and returns what to feed next.
        feed_token = next_input(0, first_pred) if next_input is not None else first_pred
        current_pos = torch.tensor([real_prompt_len], dtype=torch.long)
        self._needs_decode_reset = True

        for step in range(1, max_new_tokens):
            out_tok = torch.tensor([[int(feed_token)]], dtype=torch.long)
            decode_out = self.decode_forward(
                out_tok,
                current_pos,
                page_table=self._page_table,
                kv_cache=self._tt_kv_cache,
                reset_batch=self._needs_decode_reset,
            )
            self._needs_decode_reset = False

            # tt_transformers' Generator.decode_forward returns a (logits, log_probs)
            # tuple even when sampling_params=None. Logits shape is [batch, 1, vocab].
            logits = decode_out[0] if isinstance(decode_out, tuple) else decode_out
            pred = int(torch.argmax(logits[0, 0, :], dim=-1).item())
            predictions.append(pred)

            feed_token = next_input(step, pred) if next_input is not None else pred
            current_pos = current_pos + 1

        return predictions

    # --- GeneratorBase lifecycle ---------------------------------------

    def reset(self) -> None:
        ttnn = self._ttnn
        # Zero every K/V cache buffer in place.
        for layer in self._model.layers:
            k_cache, v_cache = layer.attention.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)
        # Clear any cached trace-side page table state on the inner Generator.
        if hasattr(self._inner, "prev_page_table"):
            self._inner.prev_page_table = None
        self._needs_decode_reset = True


# --- Factory required by the contract ----------------------------------


def build_generator(
    model_dir: Path | str,
    mesh_device,
    *,
    max_new_tokens: int = 256,
    max_prefill_len: int = 1024,
    hf_model_id: str = DEFAULT_HF_MODEL_ID,
    mesh_device_label: str = DEFAULT_MESH_DEVICE,
    **kwargs: Any,
) -> Llama31_8BReadinessGenerator:
    """
    Required factory: imported by
    `models.common.readiness_check.run_teacher_forcing` via the
    `<model_dir>/tt/generator.py::build_generator` convention.

    `model_dir` is currently unused — this model's identity is encoded in
    the `HF_MODEL` env var that tt_transformers reads — but the parameter
    is kept to satisfy the contract and to make future per-model config
    files (e.g. `<model_dir>/config.json`) easy to wire in.
    """
    del model_dir  # reserved for future per-model config
    _set_env_for_tt_transformers(hf_model_id, mesh_device_label)
    return Llama31_8BReadinessGenerator(
        mesh_device=mesh_device,
        max_new_tokens=max_new_tokens,
        max_prefill_len=max_prefill_len,
        **kwargs,
    )
