# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reading the real checkpoint: safetensors iteration, fp8 dequantization, layer extraction.

**torch only — no ttnn.** This is deliberately on the reference side even though P1's device weight
loading is its main consumer, because it is also what the M1 ground-truth test needs: one loader
means the host check and the device run can never disagree about what the checkpoint says.

The checkpoint is fp8 with **per-tensor scalar** scales (``config.json`` ->
``quantization_config.weight_block_size: null``), so dequantization is one multiply:

    w_bf16 = w_f8e4m3.to(bf16) * weight_scale_inv

Three things about this checkpoint are not guessable and are asserted rather than assumed:

* **Only the projections are quantized.** ``embed_tokens``, both per-layer norms, the final norm and
  ``lm_head`` are stored bf16 with no ``weight_scale_inv``. Multiplying one of those by a scale that
  is not there, or failing to dequantize one that is, are both silent errors — so
  :meth:`CheckpointLoader.dequantized` decides from the presence of the scale key and asserts that
  the dtype it found matches that decision.
* **The text model is under ``model.language_model.``, but ``lm_head`` is not.** ``lm_head.weight``
  sits at the top level. The checkpoint also carries a ``model.vision_tower.*`` stack this bring-up
  does not touch; the prefix filter is what keeps it out.
* **``activation_scale`` is ignored.** It is the static input scale for an fp8 *activation* path.
  This bring-up runs bf16 activations against bf8 weights (the prepared dataformats), so there is no
  activation quantization to scale for. Reading it and applying it would double-scale the matmul.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.mistral_medium_3_5_128b.reference.model_config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE, LayerWeights

#: The prepared checkpoint. ``prepared.yaml`` names this path; ``PREFILL_WEIGHTS`` overrides it.
DEFAULT_WEIGHTS = Path("/mnt/models/mistralai/Mistral-Medium-3.5-128B")

#: Suffixes that are fp8-quantized. Everything else in the text model is stored bf16.
QUANTIZED_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class CheckpointLoader:
    """Lazy reader over a sharded safetensors checkpoint, dequantizing on the way out.

    Nothing is held: each call opens the shard the index points at, reads one tensor and closes it.
    That keeps an 88-layer streaming load inside a few GB of host memory instead of the ~256 GB a
    dequantized bf16 copy of the whole model would need, which is the difference between P1 running
    and P1 not running at all.
    """

    #: Environment variables naming the checkpoint directory, in precedence order.
    #: ``PREFILL_HF_MODEL`` / ``HF_MODEL`` are the acceptance interface's names (``ACCEPTANCE.md``);
    #: ``PREFILL_WEIGHTS`` is this package's own and wins so a developer can point one test
    #: elsewhere without disturbing a verification run's environment.
    PATH_ENV_VARS = ("PREFILL_WEIGHTS", "PREFILL_HF_MODEL", "HF_MODEL")

    @classmethod
    def from_env(cls, config: MistralMediumConfig | None = None) -> "CheckpointLoader":
        """Read the first of :data:`PATH_ENV_VARS` that is set, else :data:`DEFAULT_WEIGHTS`.

        Mirrors :meth:`~.golden.GoldenTrace.from_env` so a run that relocates the checkpoint moves
        the trace and the weights together.
        """
        path = next((os.environ[v] for v in cls.PATH_ENV_VARS if os.environ.get(v)), None)
        return cls(path or DEFAULT_WEIGHTS, config)

    def __init__(self, path: Path | str, config: MistralMediumConfig | None = None):
        self.path = Path(path)
        self.config = config or MistralMediumConfig.from_json()
        index = self.path / "model.safetensors.index.json"
        if not index.exists():
            raise FileNotFoundError(f"no safetensors index at {index}")
        self.weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        self.prefix = self.config.weight_prefix

    # --- raw access ---------------------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.weight_map

    def raw(self, name: str) -> torch.Tensor:
        """One tensor exactly as stored, no dequantization."""
        from safetensors import safe_open

        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{name!r} is not in the checkpoint index ({len(self.weight_map)} keys)")
        with safe_open(self.path / shard, framework="pt") as h:
            return h.get_tensor(name)

    def dequantized(self, name: str, dtype: torch.dtype = REF_DTYPE) -> torch.Tensor:
        """One tensor in ``dtype``, scaled by ``weight_scale_inv`` if the checkpoint has one.

        The presence of the scale key is the only signal used, and the stored dtype is asserted
        against it: an fp8 tensor with no scale, or a bf16 tensor with one, means this loader's
        model of the checkpoint is wrong and the numbers downstream would be quietly off.
        """
        w = self.raw(name)
        scale_key = f"{name[: -len('.weight')]}.weight_scale_inv" if name.endswith(".weight") else None
        if scale_key and self.has(scale_key):
            assert w.dtype == torch.float8_e4m3fn, f"{name} has a weight_scale_inv but is stored {w.dtype}"
            return (w.to(torch.float32) * self.raw(scale_key).to(torch.float32)).to(dtype)
        assert w.dtype != torch.float8_e4m3fn, f"{name} is fp8 but has no weight_scale_inv"
        return w.to(dtype)

    # --- model pieces -------------------------------------------------------------------------
    def layer_prefix(self, layer_idx: int) -> str:
        return f"{self.prefix}layers.{layer_idx}."

    def layer_weights(self, layer_idx: int, dtype: torch.dtype = REF_DTYPE) -> LayerWeights:
        """One layer as a :class:`~.modeling.LayerWeights`, dequantized, HF orientation.

        Same container the random-weight tests use, so the device modules take real and random
        weights through exactly one code path.
        """
        p = self.layer_prefix(layer_idx)
        return LayerWeights(
            **{s.split(".")[-1]: self.dequantized(f"{p}{s}.weight", dtype) for s in LayerWeights._SUFFIXES}
        )

    def embed_tokens(self, dtype: torch.dtype = REF_DTYPE) -> torch.Tensor:
        """``[vocab_size, hidden_size]``, unquantized."""
        return self.dequantized(f"{self.prefix}embed_tokens.weight", dtype)

    def final_norm(self, dtype: torch.dtype = REF_DTYPE) -> torch.Tensor:
        """``[hidden_size]`` — the model's tail norm, ``model.language_model.norm.weight``."""
        return self.dequantized(f"{self.prefix}norm.weight", dtype)

    def lm_head(self, dtype: torch.dtype = REF_DTYPE) -> torch.Tensor:
        """``[vocab_size, hidden_size]``. Top level, *not* under the language-model prefix.

        ``tie_word_embeddings`` is False for this checkpoint, so this is a distinct tensor from
        :meth:`embed_tokens`; the assertion keeps a tied checkpoint from silently reusing the
        embedding as a head.
        """
        assert not self.config.tie_word_embeddings, "tie_word_embeddings=True would need lm_head = embed_tokens"
        return self.dequantized("lm_head.weight", dtype)

    def embed(self, token_ids: torch.Tensor, dtype: torch.dtype = REF_DTYPE) -> torch.Tensor:
        """Gather rows for ``token_ids`` ``[b, s]`` without materializing the 3.2 GB table.

        Used by the host ground-truth test, which needs the embedding of 10240 tokens but has no
        reason to hold all 131072 rows.
        """
        from safetensors import safe_open

        name = f"{self.prefix}embed_tokens.weight"
        shard = self.weight_map[name]
        flat = token_ids.reshape(-1)
        with safe_open(self.path / shard, framework="pt") as h:
            sl = h.get_slice(name)
            rows = torch.stack([sl[int(i) : int(i) + 1][0] for i in flat])
        return rows.to(dtype).reshape(*token_ids.shape, -1)


class CheckpointStateDict:
    """A lazy ``state_dict`` over a :class:`CheckpointLoader`, in the package's stripped naming.

    :class:`~...tt.model.MistralModel` never indexes a state dict directly — it asks
    :func:`~...utils.substate.substate` for one sub-tree at a time (``embed_tokens``,
    ``layers.N``, ``norm``, ``lm_head``) and hands each to the module that owns it. This class
    answers those four requests and nothing else, reading from the checkpoint on each call.

    **That is the whole point: the full checkpoint never exists in host memory.** Dequantized to
    bf16 it is ~250 GB. Here the peak is one layer — nine tensors, ~2.8 GB — which the module
    tilizes and pushes to the mesh, after which the dict falls out of scope and the next layer
    reuses the space. An 88-layer real-weights build costs a constant amount of RAM and, measured
    on this mesh, ~9.3 s per layer end to end — most of which is the caller's tilize and push, not
    the read this class does.

    A key outside the four shapes raises rather than returning an empty dict, because an empty
    dict is the package's "load from the tensor cache instead" signal — a typo would silently
    build a model on uninitialized weights.
    """

    def __init__(self, loader: CheckpointLoader, dtype: torch.dtype = REF_DTYPE):
        self.loader = loader
        self.dtype = dtype

    def __bool__(self) -> bool:
        """Always true: the cache-only load path keys off an empty state dict, and this is not one."""
        return True

    def __repr__(self) -> str:
        return f"CheckpointStateDict({self.loader.path}, dtype={self.dtype})"

    def substate(self, key: str) -> dict[str, torch.Tensor]:
        """The sub-dict under ``key``, read from the checkpoint now. See :func:`...substate`."""
        if key == "embed_tokens":
            return {"weight": self.loader.embed_tokens(self.dtype)}
        if key == "norm":
            return {"weight": self.loader.final_norm(self.dtype)}
        if key == "lm_head":
            return {"weight": self.loader.lm_head(self.dtype)}
        if key.startswith("layers."):
            idx = int(key.split(".", 1)[1])
            p = self.loader.layer_prefix(idx)
            # A full-depth load is many minutes of disk read and tilize with nothing else to show
            # for it; this is the only progress signal a caller gets before the first chunk runs.
            if idx % 8 == 0:
                logger.info(f"[checkpoint] reading layer {idx} of {self.loader.config.num_hidden_layers}")
            return {
                f"{s}.weight": self.loader.dequantized(f"{p}{s}.weight", self.dtype) for s in LayerWeights._SUFFIXES
            }
        raise KeyError(
            f"{key!r} is not a sub-tree of this checkpoint view; expected 'embed_tokens', 'norm', "
            f"'lm_head' or 'layers.<i>'"
        )
