# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from models.demos.speculative_deepseek_r1_broad.base_runtime import cache_seq_len
from models.demos.speculative_deepseek_r1_broad.config import EagleConfig, PathProposal
from models.demos.speculative_deepseek_r1_broad.models_base import DeepSeekBaseAdapter
from models.demos.speculative_deepseek_r1_broad.trace_replay_base import TraceReplayBaseAdapter

logger = logging.getLogger(__name__)


def _finalize_pmax_entropy_pairs(pairs: list[tuple[float, float]]) -> "BaseNextLogitBucketStats":
    if not pairs:
        return BaseNextLogitBucketStats(0, float("nan"), float("nan"))
    n = len(pairs)
    return BaseNextLogitBucketStats(
        count=n,
        mean_p_max=sum(t[0] for t in pairs) / n,
        mean_entropy=sum(t[1] for t in pairs) / n,
    )


def _mean_pmax_entropy_from_logits1d(logits_1d: torch.Tensor) -> tuple[float, float]:
    p = F.softmax(logits_1d.float().reshape(-1), dim=-1)
    p_max = float(p.max().item())
    ent = float((-(p * (p.clamp_min(1e-30)).log())).sum().item())
    return p_max, ent


def draft_per_depth_max_p_max(
    paths: Sequence[Sequence[int]],
    draft_probs_per_path: Sequence[Sequence[float]] | None,
    depth_cap: int,
) -> list[float]:
    """For each depth index ``d``, max drafted-token probability at ``d`` across proposed paths."""
    if depth_cap <= 0:
        return []
    if draft_probs_per_path is None:
        return [float("nan")] * depth_cap
    n = min(len(paths), len(draft_probs_per_path))
    out: list[float] = []
    for d in range(depth_cap):
        vals = [
            float(draft_probs_per_path[i][d])
            for i in range(n)
            if d < len(draft_probs_per_path[i])
        ]
        out.append(max(vals) if vals else float("nan"))
    return out


def finalize_draft_depth_means(sums: list[float], counts: list[int]) -> tuple[float, ...]:
    return tuple((s / c if c > 0 else float("nan")) for s, c in zip(sums, counts))


def format_draft_depth_pmax_summary(
    depth: int,
    dd_by_accept_len: tuple[tuple[float, ...], ...],
) -> str:
    """Human-readable mean max draft probability at each depth, by verification accept count.

    ``dd_by_accept_len[L]`` is the per-depth means for rounds where selected ``accept_len == L``.
    Expected ``len(dd_by_accept_len) == depth + 1`` (``L`` from 0 through ``depth``).
    """
    lines: list[str] = [
        "Draft mean max-p per depth (max across proposed paths at that depth), "
        "by selected accept count after verify:"
    ]
    for accept_len, tup in enumerate(dd_by_accept_len):
        parts: list[str] = []
        for d in range(max(0, depth)):
            v = tup[d] if d < len(tup) else float("nan")
            parts.append(f"d{d}={v:.6f}" if not math.isnan(v) else f"d{d}=n/a")
        lines.append(f"  accept_len=={accept_len}: " + " ".join(parts))
    return "\n".join(lines)


@dataclass(frozen=True)
class BaseNextLogitBucketStats:
    """Mean softmax p_max and entropy of base next-token logits at speculative round start."""

    count: int
    mean_p_max: float
    mean_entropy: float


class DraftAdapter(Protocol):
    def propose_paths(
        self,
        prefix_token_ids: list[int],
        cfg: EagleConfig,
        **kwargs: object,
    ) -> PathProposal | list[list[int]]:
        ...


@dataclass(frozen=True)
class EagleRunStats:
    generated_tokens: int
    proposed_tokens: int
    accepted_tokens: int
    accepted_tokens_percentage: float
    acceptance_rate: float
    selected_path_acceptance_rate: float
    first_token_match_rate: float
    second_token_match_rate: float
    # draft token[p] == base argmax[p] for p = 2, 3, … (same eligibility as second_token); length max(0, depth-2).
    additional_token_match_rates: tuple[float, ...]
    any_accept_rate: float
    total_rounds_with_paths: int
    speculation_rounds_saved_by_multi_accept: int
    # Index L = selected accept_len after verify (L = 0 .. depth).
    base_next_by_accept_len: tuple[BaseNextLogitBucketStats, ...]
    # [L][d] = mean max draft p at depth d when accept_len == L; outer len depth+1, inner len depth.
    draft_depth_mean_pmax_by_accept_len: tuple[tuple[float, ...], ...]
    bonus_tokens_committed: int
    # After max_paths pruning: how many distinct token IDs appear across all kept draft paths per round.
    avg_unique_draft_tokens_per_round: float
    max_unique_draft_tokens_per_round: int
    elapsed_s: float
    tokens_per_s: float


@dataclass(frozen=True)
class EagleGenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    stats: EagleRunStats


class EagleEngine:
    """Non-flattened path-batched EAGLE loop."""

    def __init__(self, *, base: DeepSeekBaseAdapter, draft: DraftAdapter, cfg: EagleConfig) -> None:
        self.base = base
        self.draft = draft
        self.cfg = cfg
        self._draft_torch_gen = torch.Generator(device="cpu")
        if self.cfg.random_seed is not None:
            self._draft_torch_gen.manual_seed(int(self.cfg.random_seed))

    def generate(
        self,
        prompt: str | None = None,
        prefix_token_ids: list[int] | None = None,
        max_new_tokens: int = 0,
    ) -> EagleGenerationResult:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if prefix_token_ids is not None:
            committed = list(prefix_token_ids)
        elif prompt is not None:
            if self.cfg.verbose:
                logger.info("Generate: encoding prompt then prefill (base forward on full prompt)...")
            committed = list(self.base.encode_prompt(prompt))
        else:
            raise ValueError("Provide either prompt or prefix_token_ids")

        prompt_token_ids = committed
        decode_state = self.base.forward_prefill(committed)
        if self.cfg.verbose:
            logger.info("Prefill done. Starting speculative rounds (draft + per-path verification).")
        generated: list[int] = []
        proposed_tokens = 0
        accepted_tokens = 0
        best_path_accept_denom = 0
        best_path_accept_num = 0
        first_token_match_rounds = 0
        second_token_match_rounds = 0
        second_token_match_denom = 0
        rounds_with_paths = 0
        rounds_with_any_accept = 0
        speculation_rounds_saved_by_multi_accept = 0
        depth_i = max(1, int(self.cfg.depth))
        n_accept_buckets = depth_i + 1
        pmax_ent_by_accept: list[list[tuple[float, float]]] = [[] for _ in range(n_accept_buckets)]
        draft_d_sums = [[0.0] * depth_i for _ in range(n_accept_buckets)]
        draft_d_cnts = [[0] * depth_i for _ in range(n_accept_buckets)]
        n_extra_pos = max(0, depth_i - 2)
        extra_token_match_num = [0] * n_extra_pos
        extra_token_match_denom = [0] * n_extra_pos
        bonus_tokens_committed = 0
        unique_draft_sum = 0
        unique_draft_max = 0

        start = time.perf_counter()
        rng = random.Random(self.cfg.random_seed) if self.cfg.random_seed is not None else None
        while len(generated) < max_new_tokens:
            speculative_rounds = max(1, int(self.cfg.num_steps))
            made_progress = False

            for _ in range(speculative_rounds):
                if len(generated) >= max_new_tokens:
                    break

                base_skip_thr = getattr(self.cfg, "base_skip_speculation_p_max", None)
                if base_skip_thr is not None:
                    bpm, _ = _mean_pmax_entropy_from_logits1d(decode_state.next_token_logits)
                    if bpm < base_skip_thr:
                        fallback_token = self.base.decode_state_next_token(decode_state)
                        if self.cfg.log_base_confidence:
                            meta_sk = (
                                decode_state.past_key_values
                                if isinstance(decode_state.past_key_values, dict)
                                else {}
                            )
                            pos_sk = int(meta_sk.get("pos", 0))
                            lt_sk = decode_state.next_token_logits.reshape(-1)
                            conf_sk = TraceReplayBaseAdapter.format_base_token_confidence(
                                lt_sk, fallback_token
                            )
                            print(
                                f"[base_conf] round_skip_spec p_max={bpm:.6f}<{base_skip_thr} "
                                f"pos={pos_sk} token={fallback_token} {conf_sk}",
                                flush=True,
                            )
                        committed.append(fallback_token)
                        generated.append(fallback_token)
                        made_progress = True
                        if len(generated) < max_new_tokens:
                            decode_state = self.base.forward_decode(decode_state, fallback_token)
                        break

                proposal = self.draft.propose_paths(
                    committed,
                    self.cfg,
                    decode_state=decode_state,
                    base_adapter=self.base,
                    draft_torch_generator=self._draft_torch_gen,
                )
                if isinstance(proposal, PathProposal):
                    paths = proposal.paths
                    draft_probs_per_path = proposal.draft_probs_per_path
                else:
                    paths = proposal
                    draft_probs_per_path = None
                if not paths:
                    break

                rounds_with_paths += 1
                unique_this = len({(d, int(p[d])) for p in paths for d in range(len(p))})
                unique_draft_sum += unique_this
                unique_draft_max = max(unique_draft_max, unique_this)
                round_pmax_ent = _mean_pmax_entropy_from_logits1d(decode_state.next_token_logits)
                if self.cfg.log_base_confidence:
                    preamble = getattr(self.base, "log_base_confidence_round_preamble", None)
                    if callable(preamble):
                        preamble(decode_state, rounds_with_paths)
                proposed_tokens += sum(len(path) for path in paths)
                if getattr(self.cfg, "print_speculated_tokens_per_round", False):
                    print(
                        f"[speculated] round={rounds_with_paths} generated={len(generated)} committed_len={len(committed)} paths={paths}"
                    )
                if self.cfg.verbose and self.cfg.verbose_shapes:
                    max_path_len = max((len(path) for path in paths), default=0)
                    logger.info(
                        "round=%d pre-verify committed_len=%d cache_len=%d paths=%d max_path_len=%d proposed_tokens_total=%d",
                        rounds_with_paths,
                        len(committed),
                        cache_seq_len(decode_state.past_key_values),
                        len(paths),
                        max_path_len,
                        proposed_tokens,
                    )
                use_draft_probs = (
                    self.cfg.verification_acceptance == "probabilistic" and draft_probs_per_path is not None
                )
                do_debug_proposals = (
                    getattr(self.cfg, "debug_proposal_rounds", 0) > 0
                    and rounds_with_paths <= getattr(self.cfg, "debug_proposal_rounds", 0)
                )
                return_base_argmax = True
                if self.cfg.verification_mode == "batched_single_pass":
                    verification = self.base.verify_paths_batched_single_pass(
                        committed,
                        paths,
                        acceptance_mode=self.cfg.verification_acceptance,
                        rng=rng,
                        draft_probs_per_path=draft_probs_per_path if use_draft_probs else None,
                        return_base_argmax=return_base_argmax,
                        per_path_forward=getattr(self.cfg, "verification_per_path_forward", False),
                    )
                elif self.cfg.verification_mode == "flattened_tree":
                    verification = self.base.verify_paths_flattened_tree(
                        committed,
                        paths,
                        acceptance_mode=self.cfg.verification_acceptance,
                        rng=rng,
                        draft_probs_per_path=draft_probs_per_path if use_draft_probs else None,
                        return_base_argmax=return_base_argmax,
                    )
                else:
                    verification = self.base.verify_paths_from_decode_state(
                        decode_state,
                        paths,
                        acceptance_mode=self.cfg.verification_acceptance,
                        rng=rng,
                        draft_probs_per_path=draft_probs_per_path if use_draft_probs else None,
                        return_base_argmax=return_base_argmax,
                    )
                if do_debug_proposals:
                    logger.info(
                        "[round %d] speculative paths=%s base_pos0=%s",
                        rounds_with_paths,
                        paths,
                        verification.base_argmax_pos0,
                    )
                    for pidx, path in enumerate(paths):
                        logger.info(
                            "[debug] path%d proposed=%s accepted_len=%d",
                            pidx,
                            path,
                            verification.accepted_prefix_lengths[pidx] if pidx < len(verification.accepted_prefix_lengths) else -1,
                        )
                        if (
                            verification.base_argmax_per_path is not None
                            and pidx < len(verification.base_argmax_per_path)
                        ):
                            base_argmax = verification.base_argmax_per_path[pidx]
                            n = min(len(path), len(base_argmax))
                            match = path[:n] == base_argmax[:n] if n else False
                            logger.info(
                                "[debug] path%d base_argmax=%s prefix_match=%s",
                                pidx,
                                base_argmax,
                                match,
                            )
                if self.cfg.verbose and self.cfg.verbose_shapes:
                    logger.info(
                        "round=%d verify mode=%s batch=%s seq_len=%s accepted_prefixes=%s",
                        rounds_with_paths,
                        self.cfg.verification_mode,
                        verification.verification_batch_size,
                        verification.verification_seq_len,
                        verification.accepted_prefix_lengths,
                    )

                best_idx = -1
                best_len = -1
                for path_idx, accepted_len in enumerate(verification.accepted_prefix_lengths):
                    if accepted_len > best_len:
                        best_len = accepted_len
                        best_idx = path_idx
                if best_len >= 0 and verification.base_argmax_pos0 is not None:
                    for path_idx, accepted_len in enumerate(verification.accepted_prefix_lengths):
                        if accepted_len == best_len and len(verification.proposed_paths[path_idx]) > 0:
                            if verification.proposed_paths[path_idx][0] == verification.base_argmax_pos0:
                                best_idx = path_idx
                                break

                if do_debug_proposals and best_idx >= 0:
                    logger.info(
                        "[round %d] selected path_idx=%d accepted_len=%d committed_tokens=%s",
                        rounds_with_paths,
                        best_idx,
                        best_len,
                        verification.proposed_paths[best_idx][:best_len] if best_len > 0 else [],
                    )

                if getattr(self.cfg, "log_round_replay_detail", False):
                    detail = getattr(self.base, "log_round_replay_detail", None)
                    if callable(detail):
                        detail(
                            round_idx=rounds_with_paths,
                            decode_state=decode_state,
                            paths=paths,
                            draft_probs_per_path=draft_probs_per_path,
                            verification=verification,
                            best_path_idx=best_idx,
                            best_accepted_len=best_len,
                            acceptance_mode=self.cfg.verification_acceptance,
                        )

                if self.cfg.log_base_confidence and best_idx >= 0:
                    sel = verification.proposed_paths[best_idx]
                    print(
                        f"[base_conf] round={rounds_with_paths} verify summary "
                        f"best_path_idx={best_idx} accepted_prefix_len={best_len} draft_path={sel}",
                        flush=True,
                    )

                if 0 <= best_len < n_accept_buckets:
                    pmax_ent_by_accept[best_len].append(round_pmax_ent)
                elif best_len >= n_accept_buckets:
                    pmax_ent_by_accept[-1].append(round_pmax_ent)

                per_depth = draft_per_depth_max_p_max(paths, draft_probs_per_path, depth_i)
                if 0 <= best_len < n_accept_buckets:
                    for d in range(depth_i):
                        v = per_depth[d]
                        if not math.isnan(v):
                            draft_d_sums[best_len][d] += v
                            draft_d_cnts[best_len][d] += 1
                elif best_len >= n_accept_buckets:
                    for d in range(depth_i):
                        v = per_depth[d]
                        if not math.isnan(v):
                            draft_d_sums[-1][d] += v
                            draft_d_cnts[-1][d] += 1

                if best_len > 0:
                    speculation_rounds_saved_by_multi_accept += max(0, best_len - 1)

                if best_idx >= 0:
                    bap = verification.base_argmax_per_path
                    sel_path = verification.proposed_paths[best_idx]
                    if len(sel_path) >= 2:
                        second_token_match_denom += 1
                        if (
                            bap is not None
                            and best_idx < len(bap)
                            and len(bap[best_idx]) >= 2
                            and sel_path[1] == bap[best_idx][1]
                        ):
                            second_token_match_rounds += 1
                    for j, pos in enumerate(range(2, depth_i)):
                        if len(sel_path) > pos:
                            if (
                                bap is not None
                                and best_idx < len(bap)
                                and len(bap[best_idx]) > pos
                            ):
                                extra_token_match_denom[j] += 1
                                if sel_path[pos] == bap[best_idx][pos]:
                                    extra_token_match_num[j] += 1

                if best_idx >= 0 and best_len > 0:
                    rounds_with_any_accept += 1
                    depth_denom = max(1, min(self.cfg.depth, len(verification.proposed_paths[best_idx])))
                    best_path_accept_denom += depth_denom
                    best_path_accept_num += best_len
                    if (
                        verification.base_argmax_pos0 is not None
                        and len(verification.proposed_paths[best_idx]) >= 1
                        and verification.proposed_paths[best_idx][0] == verification.base_argmax_pos0
                    ):
                        first_token_match_rounds += 1
                    accepted = verification.proposed_paths[best_idx][:best_len]
                    if self.cfg.verbose and (
                        self.cfg.log_every_steps <= 0 or (len(generated) % self.cfg.log_every_steps == 0)
                    ):
                        logger.info(
                            "accepted path idx=%d accepted_len=%d/%d generated=%d",
                            best_idx,
                            best_len,
                            len(verification.proposed_paths[best_idx]),
                            len(generated),
                        )
                    for token_id in accepted:
                        if len(generated) >= max_new_tokens:
                            break
                        if self.cfg.log_base_confidence:
                            meta_ds = (
                                decode_state.past_key_values
                                if isinstance(decode_state.past_key_values, dict)
                                else {}
                            )
                            pos_ds = int(meta_ds.get("pos", 0))
                            lt = decode_state.next_token_logits.reshape(-1)
                            conf = TraceReplayBaseAdapter.format_base_token_confidence(lt, token_id)
                            print(
                                f"[base_conf] round={rounds_with_paths} pos={pos_ds} "
                                f"accepted_commit token={token_id} {conf}",
                                flush=True,
                            )
                        committed.append(token_id)
                        generated.append(token_id)
                        accepted_tokens += 1
                        made_progress = True
                        if len(generated) < max_new_tokens:
                            decode_state = self.base.forward_decode(decode_state, token_id)
                    if (
                        verification.bonus_token_per_path is not None
                        and best_idx < len(verification.bonus_token_per_path)
                        and len(generated) < max_new_tokens
                    ):
                        bonus_tid = verification.bonus_token_per_path[best_idx]
                        committed.append(bonus_tid)
                        generated.append(bonus_tid)
                        bonus_tokens_committed += 1
                        made_progress = True
                        if len(generated) < max_new_tokens:
                            decode_state = self.base.forward_decode(decode_state, bonus_tid)
                    if self.cfg.verbose and self.cfg.verbose_shapes:
                        logger.info(
                            "round=%d commit accepted_len=%d (+1 bonus) committed_len=%d cache_len=%d",
                            rounds_with_paths,
                            best_len,
                            len(committed),
                            cache_seq_len(decode_state.past_key_values),
                        )
                    break

                # 0 draft tokens accepted: still commit the bonus (+1) from verification
                if (
                    verification.bonus_token_per_path is not None
                    and best_idx >= 0
                    and best_idx < len(verification.bonus_token_per_path)
                    and len(generated) < max_new_tokens
                ):
                    bonus_tid = verification.bonus_token_per_path[best_idx]
                    committed.append(bonus_tid)
                    generated.append(bonus_tid)
                    bonus_tokens_committed += 1
                    made_progress = True
                    if len(generated) < max_new_tokens:
                        decode_state = self.base.forward_decode(decode_state, bonus_tid)
                    break

                if best_idx >= 0:
                    depth_denom = max(1, min(self.cfg.depth, len(verification.proposed_paths[best_idx])))
                    best_path_accept_denom += depth_denom
                if self.cfg.verbose and (
                    self.cfg.log_every_steps <= 0 or (len(generated) % self.cfg.log_every_steps == 0)
                ):
                    logger.info(
                        "rejected all proposals generated=%d paths=%d",
                        len(generated),
                        len(paths),
                    )
                break

            if made_progress:
                continue

            fallback_token = self.base.decode_state_next_token(decode_state)
            if self.cfg.log_base_confidence:
                meta_fb = (
                    decode_state.past_key_values if isinstance(decode_state.past_key_values, dict) else {}
                )
                pos_fb = int(meta_fb.get("pos", 0))
                lt_fb = decode_state.next_token_logits.reshape(-1)
                conf_fb = TraceReplayBaseAdapter.format_base_token_confidence(lt_fb, fallback_token)
                print(
                    f"[base_conf] round={rounds_with_paths} pos={pos_fb} "
                    f"fallback token={fallback_token} {conf_fb}",
                    flush=True,
                )
            committed.append(fallback_token)
            generated.append(fallback_token)
            if len(generated) < max_new_tokens:
                decode_state = self.base.forward_decode(decode_state, fallback_token)
            if self.cfg.verbose and self.cfg.verbose_shapes:
                logger.info(
                    "fallback committed_len=%d cache_len=%d",
                    len(committed),
                    cache_seq_len(decode_state.past_key_values),
                )

        elapsed_s = max(time.perf_counter() - start, 1e-9)
        tokens_per_s = len(generated) / elapsed_s
        accepted_tokens_percentage = accepted_tokens / proposed_tokens if proposed_tokens > 0 else 0.0
        max_accept_slots = rounds_with_paths * depth_i
        acceptance_rate = accepted_tokens / max_accept_slots if max_accept_slots > 0 else 0.0
        selected_path_acceptance_rate = (
            best_path_accept_num / best_path_accept_denom if best_path_accept_denom > 0 else 0.0
        )
        first_token_match_rate = first_token_match_rounds / rounds_with_paths if rounds_with_paths > 0 else 0.0
        second_token_match_rate = (
            second_token_match_rounds / second_token_match_denom if second_token_match_denom > 0 else 0.0
        )
        any_accept_rate = rounds_with_any_accept / rounds_with_paths if rounds_with_paths > 0 else 0.0
        base_next_by_accept_len = tuple(
            _finalize_pmax_entropy_pairs(pmax_ent_by_accept[i]) for i in range(n_accept_buckets)
        )
        draft_depth_mean_pmax_by_accept_len = tuple(
            finalize_draft_depth_means(draft_d_sums[i], draft_d_cnts[i]) for i in range(n_accept_buckets)
        )
        additional_token_match_rates = tuple(
            (extra_token_match_num[j] / extra_token_match_denom[j])
            if extra_token_match_denom[j] > 0
            else 0.0
            for j in range(n_extra_pos)
        )
        text = self.base.decode_tokens(generated)
        if self.cfg.verbose:
            logger.info(
                "run done tokens=%d tps=%.2f accepted_pct=%.4f acceptance_rate=%.4f "
                "sel_path_rate=%.4f first_match=%.4f second_match=%.4f any_accept_round_rate=%.4f",
                len(generated),
                tokens_per_s,
                accepted_tokens_percentage,
                acceptance_rate,
                selected_path_acceptance_rate,
                first_token_match_rate,
                second_token_match_rate,
                any_accept_rate,
            )

        return EagleGenerationResult(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated,
            generated_text=text,
            stats=EagleRunStats(
                generated_tokens=len(generated),
                proposed_tokens=proposed_tokens,
                accepted_tokens=accepted_tokens,
                accepted_tokens_percentage=accepted_tokens_percentage,
                acceptance_rate=acceptance_rate,
                selected_path_acceptance_rate=selected_path_acceptance_rate,
                first_token_match_rate=first_token_match_rate,
                second_token_match_rate=second_token_match_rate,
                additional_token_match_rates=additional_token_match_rates,
                any_accept_rate=any_accept_rate,
                total_rounds_with_paths=rounds_with_paths,
                speculation_rounds_saved_by_multi_accept=speculation_rounds_saved_by_multi_accept,
                base_next_by_accept_len=base_next_by_accept_len,
                draft_depth_mean_pmax_by_accept_len=draft_depth_mean_pmax_by_accept_len,
                bonus_tokens_committed=bonus_tokens_committed,
                avg_unique_draft_tokens_per_round=(
                    unique_draft_sum / rounds_with_paths if rounds_with_paths > 0 else 0.0
                ),
                max_unique_draft_tokens_per_round=unique_draft_max,
                elapsed_s=elapsed_s,
                tokens_per_s=tokens_per_s,
            ),
        )
