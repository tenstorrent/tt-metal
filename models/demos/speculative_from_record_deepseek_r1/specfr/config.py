# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathProposal:
    """Draft proposal: paths and optional per-token draft probabilities."""

    paths: list[list[int]]
    draft_probs_per_path: list[list[float]] | None = None


@dataclass(frozen=True)
class EagleConfig:
    """Core speculative decoding controls."""

    # Branch budget per beam per draft step. With draft_branching=temperature_top_p, use <=0 to expand
    # the entire top-p nucleus (no extra subsampling cap).
    top_k: int = 4
    depth: int = 2
    num_steps: int = 2
    # Legacy / unused by engine; draft stochasticity uses ``draft_branching`` + ``draft_*`` below.
    temperature: float = 0.0
    # Cap beam count per draft step (highest draft confidence). Use <=0 to disable pruning (can explode).
    max_paths: int = 16
    verification_mode: str = "batched_single_pass"
    verification_acceptance: str = "argmax"
    # When False (default), one batched forward per round. When True, one forward per path (for backends that do not batch correctly).
    verification_per_path_forward: bool = False
    verbose: bool = False
    verbose_shapes: bool = False
    log_every_steps: int = 0
    random_seed: int | None = None
    debug_proposal_rounds: int = 0
    print_speculated_tokens_per_round: bool = False
    # When True and draft is MTP: take argmax at each depth step (single path of length depth). Else use top_k branching.
    draft_mtp_greedy: bool = False
    # NextNSglangCPUDraftAdapter only: run one decoder forward per depth by batching beams (default True).
    # When False, each beam uses a separate forward (legacy; slower for depth>1 and many paths).
    draft_sglang_cpu_batch_beams: bool = True
    # How the draft picks multiple next-token candidates per beam (``top_k`` = branch count cap).
    # ``top_k``: deterministic ``torch.topk`` on logits (default).
    # ``temperature_top_p``: sample up to ``top_k`` *distinct* tokens from a temperature-scaled softmax,
    # restricted to a minimal top-p nucleus (see ``models_draft.draft_branch_token_ids_from_logits`` docstring).
    draft_branching: str = "top_k"
    draft_temperature: float = 0.6
    draft_top_p: float = 0.95
    # When True and the base provides recorded logits (MTP .pt): log softmax p(drafted token), p_max, argmax per step.
    log_base_confidence: bool = False
    # Trace replay only: print record greedy next, stored base logits/confidence, draft paths per depth, verify outcome.
    log_round_replay_detail: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers for base and draft models."""

    base_model_id: str = "deepseek-ai/DeepSeek-R1-0528"
    draft_model_id: str = "jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0"
    draft_mode: str = "draft_r1"
    trust_remote_code: bool = True
    base_impl: str = "reference"


@dataclass(frozen=True)
class RunConfig:
    """Execution controls."""

    max_new_tokens: int = 32
    batch_size: int = 1
    device: str = "cpu"
    torch_dtype: str = "float32"
