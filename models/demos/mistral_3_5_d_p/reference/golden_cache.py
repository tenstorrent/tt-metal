# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden runner + on-disk cache for the Mistral-Medium-3.5 CPU reference.

Runs :mod:`..reference.model` once and dumps every block's inputs/outputs, keyed on everything that
changes the result. A CPU forward of even a reduced model is expensive relative to a test, and the
full model's is measured in hours, so it must run ONCE and be asserted against — never recomputed
per test.

Both rules the recipe attaches to this cache are honoured:

  * **Key on every field that changes the output.** ``ReferenceCacheKey`` (imported from
    ``deepseek_v3_d_p/utils/transformer_helpers.py``) is frozen and stringifies into the filename, so
    a changed field yields a different file rather than a silently stale hit. Fields that the frozen
    key cannot express (here: hidden/intermediate width, which a reduced-config test varies) go into
    :class:`GoldenSpec` and are folded into ``input_source``, keeping the whole identity in the name.
  * **Assert rather than recompute where a CPU run is expensive.** :func:`load_or_fail` raises on a
    miss instead of quietly running an hours-long forward inside a test; the generator script
    (``scripts/generate_golden_kv_cache.py``) is the one place that computes.

Per-module goldens are cheap and deliberately NOT cached — a test builds those inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from loguru import logger

# Imported, not copied — recipe §3 "CPU golden cache".
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ReferenceCacheKey,
    check_reference_cache_exists,
    load_reference_cache,
    save_reference_cache,
)
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config, reduced_text_config


class _CacheVariant:
    """The two attributes ``transformer_helpers``' cache functions read off a variant."""

    name = "mistral_3_5_d_p"
    ref_cache_env = "MISTRAL_35_HOST_REF_CACHE"


VARIANT = _CacheVariant()


@dataclass(frozen=True)
class GoldenSpec:
    """Everything that defines one golden run. ``num_layers`` / ``isl`` also live on the
    ``ReferenceCacheKey``; ``hidden_size`` / ``intermediate_size`` / ``vocab_size`` / ``seed`` do
    not, so they are folded into the key's ``input_source`` field to keep the filename unique."""

    num_layers: int
    isl: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    seed: int = 0
    weight_type: str = "random"

    @property
    def cache_key(self) -> ReferenceCacheKey:
        return ReferenceCacheKey(
            weight_type=self.weight_type,
            input_source=f"randtok_h{self.hidden_size}_i{self.intermediate_size}_v{self.vocab_size}_s{self.seed}",
            isl_total=self.isl,
            num_layers=self.num_layers,
            n_routed_experts=MistralMedium35Config.NUM_EXPERTS,  # 0 — dense model
            padding_side="right",
        )

    def hf_config(self):
        """The (possibly reduced) HF text config this golden was computed at."""
        return reduced_text_config(
            num_hidden_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
        )


@dataclass
class Golden:
    """One golden run's tensors.

    ``snapshots[i]`` is the residual stream ENTERING layer ``i``, and ``snapshots[-1]`` is the
    post-final-norm hidden state — so a test can PCC any layer's input as well as its output.
    ``kv[i]`` is ``[2, num_kv_heads, isl, head_dim]`` (post-RoPE K stacked over raw V).
    """

    spec: GoldenSpec
    token_ids: torch.Tensor
    snapshots: list
    kv: list
    state_dict: dict

    def layer_kv(self, layer_idx: int):
        stacked = self.kv[layer_idx]
        return stacked[0].unsqueeze(0), stacked[1].unsqueeze(0)


def _state_dict_path(spec: GoldenSpec) -> Path:
    import os

    root = Path(os.environ.get(VARIANT.ref_cache_env, f"/tmp/{VARIANT.name}_transformer_ref_cache"))
    return root / f"{spec.cache_key}.weights.pt"


def compute(spec: GoldenSpec) -> Golden:
    """Run the CPU reference for ``spec``. This is the expensive path — call it from the generator
    script, not from a test."""
    hf_config = spec.hf_config()
    logger.info(
        f"[golden] computing reference: layers={spec.num_layers} isl={spec.isl} "
        f"hidden={spec.hidden_size} inter={spec.intermediate_size} vocab={spec.vocab_size} seed={spec.seed}"
    )
    model = reference.build_reference_model(hf_config, seed=spec.seed)
    torch.manual_seed(spec.seed + 1)
    token_ids = torch.randint(0, hf_config.vocab_size, (1, spec.isl))

    snapshots: list = []
    handles = []

    def make_hook(idx):
        def hook(_module, args, _kwargs, _output):
            snapshots.append((idx, args[0].detach().clone() if args else _kwargs["hidden_states"].detach().clone()))

        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i), with_kwargs=True))
    try:
        out = reference.model_reference_forward(model, token_ids, skip_lm_head=True)
    finally:
        for h in handles:
            h.remove()

    ordered = [t for _, t in sorted(snapshots, key=lambda p: p[0])]
    ordered.append(out.hidden_states.detach().clone())
    kv = [torch.stack([k[0].detach().clone(), v[0].detach().clone()]) for k, v in out.kv]
    return Golden(
        spec=spec,
        token_ids=token_ids,
        snapshots=ordered,
        kv=kv,
        state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
    )


def save(golden: Golden) -> None:
    """Persist a computed golden: snapshots + KV through the shared cache, weights alongside."""
    save_reference_cache(VARIANT, golden.spec.cache_key, golden.snapshots, golden.kv)
    path = _state_dict_path(golden.spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"token_ids": golden.token_ids, "state_dict": golden.state_dict}, path)
    logger.info(f"[golden] saved weights + token ids to {path}")


def exists(spec: GoldenSpec) -> bool:
    return check_reference_cache_exists(VARIANT, spec.cache_key) and _state_dict_path(spec).exists()


def load_or_fail(spec: GoldenSpec) -> Golden:
    """Load a cached golden, or raise. Deliberately does NOT recompute: a silent CPU forward inside
    a device test is how an afternoon disappears (see ``tests/test_mla.py::293``, the same rule)."""
    if not exists(spec):
        raise FileNotFoundError(
            f"no golden for {spec.cache_key}; generate it first:\n"
            f"  python models/demos/mistral_3_5_d_p/scripts/generate_golden_kv_cache.py "
            f"--layers {spec.num_layers} --isl {spec.isl} --hidden {spec.hidden_size} "
            f"--intermediate {spec.intermediate_size} --vocab {spec.vocab_size} --seed {spec.seed}"
        )
    snapshots, kv = load_reference_cache(VARIANT, spec.cache_key)
    blob = torch.load(_state_dict_path(spec), weights_only=True)
    return Golden(spec=spec, token_ids=blob["token_ids"], snapshots=snapshots, kv=kv, state_dict=blob["state_dict"])


def load_or_compute(spec: GoldenSpec, *, allow_compute: bool = False) -> Golden:
    """``load_or_fail``, or compute+save when the caller explicitly opts in (the generator script)."""
    if exists(spec):
        return load_or_fail(spec)
    if not allow_compute:
        return load_or_fail(spec)  # raises with the generator command
    golden = compute(spec)
    save(golden)
    return golden


def inline_golden_weights(state_dict: dict, num_layers: int) -> dict:
    """Rearrange an HF ``Ministral3ForCausalLM`` state dict into the inline-golden weight tree
    (:func:`..reference.model.golden_model`), so the two oracles can be run on one set of weights."""
    return {
        "embed": state_dict["model.embed_tokens.weight"],
        "norm": state_dict["model.norm.weight"],
        "lm_head": state_dict["lm_head.weight"],
        "layers": [
            {
                "q": state_dict[f"model.layers.{i}.self_attn.q_proj.weight"],
                "k": state_dict[f"model.layers.{i}.self_attn.k_proj.weight"],
                "v": state_dict[f"model.layers.{i}.self_attn.v_proj.weight"],
                "o": state_dict[f"model.layers.{i}.self_attn.o_proj.weight"],
                "gate": state_dict[f"model.layers.{i}.mlp.gate_proj.weight"],
                "up": state_dict[f"model.layers.{i}.mlp.up_proj.weight"],
                "down": state_dict[f"model.layers.{i}.mlp.down_proj.weight"],
                "input_layernorm": state_dict[f"model.layers.{i}.input_layernorm.weight"],
                "post_attention_layernorm": state_dict[f"model.layers.{i}.post_attention_layernorm.weight"],
            }
            for i in range(num_layers)
        ],
    }


def layer_state_dict(state_dict: dict, layer_idx: int) -> dict:
    """The HF sub-state for one layer, with the ``model.layers.<i>.`` prefix stripped."""
    prefix = f"model.layers.{layer_idx}."
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


# The golden used by the whole-model tests: a reduced-depth, reduced-width run that a host can
# actually compute, at the model's real head geometry (96 Q / 8 KV / head_dim 128 — the part under
# test). Depth/width are the only reductions; see README "Known gaps".
DEFAULT_GOLDEN = GoldenSpec(num_layers=4, isl=512, hidden_size=1024, intermediate_size=2048, vocab_size=2048)

# The golden used by the SP whole-model device test: sequence length must satisfy
# chunk % (TILE_SIZE * sp) == 0 at sp=4, i.e. a multiple of 128. 512 qualifies.
