# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-torch reference of the Alpamayo 2 Edge inference path: prefill -> CoT decode -> expert flow -> trajectory.

Port of NVlabs `Alpamayo2Edge.sample_trajectories_from_data` with one candidate per prompt row: shared prompt
prefill, CoT sampling with the upstream logits processors (no trajectory codes, no text EOS) that stops one step
after every row emits `<|traj_future_start|>` (the anchor), then ten Euler steps of the non-causal expert over the
CoT cache, decoded by the course-heading unicycle action space. It is the golden for the TT decode, expert, action
and end-to-end tests.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.experimental.alpamayo2_edge.reference.action import (
    ActionInProj,
    action_to_traj,
    euler_sample,
    expert_mask,
    expert_position_ids,
)
from models.experimental.alpamayo2_edge.reference.config import Alpamayo2EdgeConfig, TokenIds, validate_supported
from models.experimental.alpamayo2_edge.reference.inputs import ModelInputs
from models.experimental.alpamayo2_edge.reference.text import KVCache, LanguageModel, TextModel
from models.experimental.alpamayo2_edge.reference.vision import VisionTower
from models.experimental.alpamayo2_edge.reference.weights import load_weights

# Upstream `sample_trajectories_from_data` defaults.
TEMPERATURE = 0.6
TOP_P = 0.98
MAX_NEW_TOKENS = 256


@dataclass
class CotOut:
    tokens: torch.Tensor  # [B, T] generated ids; the last step is the latest row's anchor
    offsets: torch.Tensor  # [B] cache length that ends each row's CoT: prompt length + index of its first anchor + 1
    logits: torch.Tensor  # [B, T, V] fp32 raw LM logits that tokens[:, t] was drawn from


@dataclass
class Prediction:
    xyz: torch.Tensor  # [B, n_waypoints, 3]
    rot: torch.Tensor  # [B, n_waypoints, 3, 3]
    cot_tokens: torch.Tensor  # [B, T]
    actions: torch.Tensor  # [B, n_waypoints, 2] normalized (accel, curvature)


def process_logits(
    logits: torch.Tensor, tokens: TokenIds, temperature: float = TEMPERATURE, top_p: float = TOP_P
) -> torch.Tensor:
    """Upstream CoT logits processors, then HF's temperature and top-p warpers, on a copy of [B, V] logits.

    Trajectory codes and every EOS other than the anchor are banned, so the CoT can only end at the anchor.
    """
    scores = logits.clone()
    scores[:, tokens.history_id0 : tokens.history_id0 + tokens.history_vocab] = -torch.inf
    scores[:, tokens.future_id0 : tokens.future_id0 + tokens.future_vocab] = -torch.inf
    blocked_eos = [eos for eos in (tokens.eos,) if eos != tokens.future_start]
    if blocked_eos:
        scores[:, blocked_eos] = -torch.inf
    if temperature != 1.0:
        scores = scores / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1 - top_p)
        remove[..., -1:] = 0
        scores = scores.masked_fill(remove.scatter(1, sorted_indices, remove), -torch.inf)
    return scores


def find_anchor_offsets(generated: torch.Tensor, anchor: int) -> torch.Tensor:
    """[B, T] generated ids -> [B] index just past each row's first anchor; every row must have one."""
    if generated.ndim != 2 or generated.shape[1] == 0:
        raise ValueError("No <traj_future_start> token found: generated sequences are empty.")
    is_anchor = generated == anchor
    has_anchor = is_anchor.any(dim=1)
    if not has_anchor.all():
        rows = (~has_anchor).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"No <traj_future_start> token found in generated rows {rows}. "
            "Increase max_generation_length or check the prompt and checkpoint tokenizer."
        )
    return is_anchor.int().argmax(dim=1) + 1


def make_forced_cot(tokens: torch.Tensor, anchor: int) -> torch.Tensor:
    """[T] ids -> a teacher-forcing CoT that ends at its first anchor (appended if there is none)."""
    tokens = tokens.long().reshape(-1)
    hits = (tokens == anchor).nonzero()
    if len(hits):
        return tokens[: int(hits[0]) + 1].clone()
    return torch.cat([tokens, tokens.new_tensor([anchor])])


def _first_layers(weights: dict, prefix: str, layers: int) -> dict:
    return {
        k: v for k, v in weights.items() if not k.startswith(prefix) or int(k[len(prefix) :].split(".")[0]) < layers
    }


class ReferenceModel:
    def __init__(
        self,
        cfg: Alpamayo2EdgeConfig,
        vision: VisionTower,
        lm: LanguageModel,
        expert: TextModel,
        action_in_proj: ActionInProj,
        action_out_proj: nn.Linear,
    ):
        validate_supported(cfg)
        self.cfg = cfg
        self.vision = vision
        self.lm = lm
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj

    @classmethod
    def from_pretrained(cls, cfg: Alpamayo2EdgeConfig, dtype: torch.dtype = torch.float32) -> "ReferenceModel":
        md = cfg.model_dir
        vision = VisionTower(
            cfg.vision,
            cfg.projector,
            _first_layers(load_weights(md, "vlm.model.visual."), "encoder.layers.", cfg.vision.layers),
            load_weights(md, "vlm.model.projector."),
            dtype,
        )
        lm = LanguageModel(
            cfg.text,
            _first_layers(load_weights(md, "vlm.model.language_model."), "layers.", cfg.text.layers),
            load_weights(md, "vlm.lm_head."),
            dtype,
        )
        expert = TextModel(
            cfg.expert,
            _first_layers(load_weights(md, "expert.expert."), "layers.", cfg.expert.layers),
            has_embedding=False,
            dtype=dtype,
        )
        in_proj = ActionInProj(cfg.action, cfg.expert.hidden, load_weights(md, "expert.action_in_proj."), dtype)
        out_proj = nn.Linear(cfg.expert.hidden, cfg.action.action_dim)
        out_proj.load_state_dict(load_weights(md, "expert.action_out_proj."))
        out_proj.to(dtype)
        for module in (vision, lm, expert, in_proj, out_proj):
            module.requires_grad_(False)
        return cls(cfg, vision, lm, expert, in_proj, out_proj)

    def _rows(self, inp: ModelInputs, batch: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Samples per prompt, the prompt mask and the M-RoPE text deltas, repeated to `batch` rows."""
        prompts = inp.input_ids.shape[0]
        if batch % prompts:
            raise ValueError(f"batch {batch} is not a multiple of the {prompts} prompts")
        n = batch // prompts
        prompt_mask = inp.attention_mask.repeat_interleave(n, dim=0)
        # Cache offsets count prompt padding, the rope deltas do not.
        rope_deltas = inp.rope_deltas.repeat_interleave(n, dim=0) - prompt_mask.eq(0).sum(dim=-1, keepdim=True)
        return n, prompt_mask, rope_deltas

    @torch.no_grad()
    def prefill(self, inp: ModelInputs) -> tuple[torch.Tensor, KVCache]:
        """Whole prompt with vision embeddings scattered into the image tokens -> (last logits [B, V], cache of S)."""
        if not bool(inp.attention_mask.all()):
            raise ValueError("the reference prefill does not support padded prompts")
        ids = inp.input_ids
        embeds = self.lm.text.embed_tokens(ids)
        image_rows = ids == self.cfg.tokens.image
        vision = self.vision(inp.patches, inp.image_grid_thw)
        if int(image_rows.sum()) != vision.shape[0]:
            raise ValueError(f"{int(image_rows.sum())} image tokens for {vision.shape[0]} vision embeddings")
        embeds = embeds.masked_scatter(image_rows[..., None], vision.to(embeds.dtype))
        cache = KVCache(self.cfg.text.layers)
        hidden = self.lm.text(embeds, inp.position_ids, cache=cache, causal=True)
        return self.lm.logits(hidden[:, -1]).float(), cache

    def _forward_text(self, ids: torch.Tensor, cache: KVCache, rope_deltas: torch.Tensor) -> torch.Tensor:
        """Append [B, s] text ids at the next cache positions; text positions are cache position + delta on T/H/W."""
        positions = torch.arange(cache.seq_len, cache.seq_len + ids.shape[1]) + rope_deltas
        return self.lm.text(self.lm.text.embed_tokens(ids), positions.expand(3, *positions.shape), cache=cache)

    @torch.no_grad()
    def decode(
        self,
        inp: ModelInputs,
        cache: KVCache,
        *,
        forced: torch.Tensor | None = None,
        seed: int = 0,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> CotOut:
        """Sample (or teacher-force) the CoT on the prefill cache, leaving it cropped to the longest CoT.

        Step 0 re-runs the last prompt position, as upstream's prefill of S - 1 tokens plus first generate step
        does. Rows keep decoding past their own anchor until every row has one; one more step then caches the last
        anchor. Hitting `max_new_tokens` on an anchor caches it separately, as upstream does.
        """
        tok = self.cfg.tokens
        anchor = tok.future_start
        prompt_len = inp.input_ids.shape[1]
        if cache.seq_len != prompt_len:
            raise ValueError(f"decode expects the {prompt_len}-row prefill cache, got {cache.seq_len} rows")
        batch = cache.k[0].shape[0]
        n, _, rope_deltas = self._rows(inp, batch)
        if forced is not None and forced.shape[0] != batch:
            raise ValueError(f"forced CoT has {forced.shape[0]} rows for a cache of {batch}")
        generator = torch.Generator().manual_seed(seed)

        cache.crop(prompt_len - 1)
        step_ids = inp.input_ids[:, -1:].repeat_interleave(n, dim=0)
        tokens, logits = [], []
        found = torch.zeros(batch, dtype=torch.bool)
        while True:
            hidden = self._forward_text(step_ids, cache, rope_deltas)
            if found.all():
                break
            step_logits = self.lm.logits(hidden[:, -1]).float()
            if forced is None:
                probs = torch.softmax(process_logits(step_logits, tok), dim=-1)
                next_ids = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            elif len(tokens) < forced.shape[1]:
                next_ids = forced[:, len(tokens)].long()
            else:
                rows = (~found).nonzero(as_tuple=True)[0].tolist()
                raise ValueError(
                    f"No <traj_future_start> token found in generated rows {rows}: "
                    f"the forced CoT ran out after {len(tokens)} tokens."
                )
            tokens.append(next_ids)
            logits.append(step_logits)
            found |= next_ids == anchor
            if len(tokens) >= max_new_tokens:
                break
            step_ids = next_ids[:, None]

        generated = torch.stack(tokens, dim=1)
        offsets = find_anchor_offsets(generated, anchor) + prompt_len
        prefix_len = int(offsets.max())
        if cache.seq_len < prefix_len:
            self._forward_text(generated[:, cache.seq_len - prompt_len : prefix_len - prompt_len], cache, rope_deltas)
        cache.crop(prefix_len)
        return CotOut(tokens=generated, offsets=offsets, logits=torch.stack(logits, dim=1))

    @torch.no_grad()
    def expert_velocity(
        self, cache: KVCache, cot: CotOut, inp: ModelInputs, x: torch.Tensor, t: float | torch.Tensor
    ) -> torch.Tensor:
        """Flow velocity [B, N, 2] at noisy actions x [B, N, 2] and time t, attending to each row's CoT cache."""
        batch, n_waypoints, _ = x.shape
        prefix_len = cache.seq_len
        if prefix_len != int(cot.offsets.max()):
            raise ValueError(f"the cache holds {prefix_len} rows, the CoT ends at {int(cot.offsets.max())}")
        _, prompt_mask, rope_deltas = self._rows(inp, batch)
        positions = expert_position_ids(cot.offsets, rope_deltas, n_waypoints)
        mask = expert_mask(cot.offsets, prompt_mask, prefix_len, n_waypoints)
        t = torch.as_tensor(t, dtype=torch.float32).expand(batch, 1, 1)
        embeds = self.action_in_proj(x, t)
        try:
            hidden = self.expert(embeds, positions, cache=cache, attn_mask=mask, causal=False)
        finally:
            # Every denoising step conditions on the CoT only.
            cache.crop(prefix_len)
        return self.action_out_proj(hidden).reshape(batch, n_waypoints, -1)

    @torch.no_grad()
    def sample_actions(self, cache: KVCache, cot: CotOut, inp: ModelInputs, noise: torch.Tensor) -> torch.Tensor:
        """Euler-integrate the expert flow from `noise` [B, N, 2] (a standard-normal draw) to actions."""
        step_fn = lambda x, t: self.expert_velocity(cache, cot, inp, x, t)
        return euler_sample(step_fn, noise, self.cfg.action.steps)

    @torch.no_grad()
    def infer(
        self,
        inp: ModelInputs,
        *,
        forced_cot: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        seed: int = 0,
    ) -> Prediction:
        """One trajectory per prompt row; `forced_cot` [T] or [B, T] replaces CoT sampling."""
        _, cache = self.prefill(inp)
        if forced_cot is not None and forced_cot.ndim == 1:
            forced_cot = forced_cot[None]
        cot = self.decode(inp, cache, forced=forced_cot, seed=seed)
        batch = cot.tokens.shape[0]
        if noise is None:
            noise = torch.randn(
                batch,
                self.cfg.action.n_waypoints,
                self.cfg.action.action_dim,
                generator=torch.Generator().manual_seed(seed),
            )
        actions = self.sample_actions(cache, cot, inp, noise)
        n = batch // inp.input_ids.shape[0]
        xyz, rot = action_to_traj(
            actions,
            inp.ego_history_xyz.repeat_interleave(n, dim=0),
            inp.ego_history_rot.repeat_interleave(n, dim=0),
            self.cfg.action,
        )
        return Prediction(xyz=xyz, rot=rot, cot_tokens=cot.tokens, actions=actions)
