# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Speculative Decoding: Token Verification and Sampling Logic

This module implements the core speculative decoding algorithm:
1. Verify draft tokens against target model
2. Accept/reject based on probability ratios
3. Sample bonus/correction tokens

Reference: https://arxiv.org/abs/2211.17192
"""

import torch


class SpeculativeDecoding:
    def __init__(self, vocab_size, temperature=1.0, top_p=1.0):
        """
        Initialize speculative decoding sampler.

        Args:
            vocab_size: Size of the vocabulary
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling threshold (default: 1.0 = disabled)
        """
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.top_p = top_p

    def verify_tokens(self, draft_logits, target_logits, draft_tokens):
        """
        Verifies K draft tokens in parallel using the target model's logits.

        Uses the standard speculative decoding acceptance criterion:
        - For greedy (temp=0): Accept if argmax(target) == draft_token
        - For sampling (temp>0): Accept with prob = min(1, p_target/p_draft)
        - Otherwise reject and sample from adjusted distribution

        Args:
            draft_logits: Logits from the draft model for K tokens [B, K, vocab_size]
            target_logits: Logits from the target model for K tokens [B, K, vocab_size]
            draft_tokens: The K tokens proposed by the draft model [B, K]

        Returns:
            accepted_tokens: [B, K] boolean mask indicating accepted tokens
            num_accepted: [B] number of accepted tokens per batch
            bonus_tokens_logits: [B, vocab_size] logits for sampling bonus/correction token
        """
        B, K, V = draft_logits.shape
        assert target_logits.shape == (
            B,
            K,
            V,
        ), f"Target logits shape mismatch: {target_logits.shape} vs expected {(B, K, V)}"
        assert draft_tokens.shape == (B, K), f"Draft tokens shape mismatch: {draft_tokens.shape} vs expected {(B, K)}"

        # Determine if using greedy decoding
        is_greedy = self.temperature == 0.0

        # Calculate acceptance probabilities
        draft_probs = torch.softmax(draft_logits / (self.temperature if not is_greedy else 1.0), dim=-1)
        target_probs = torch.softmax(target_logits / (self.temperature if not is_greedy else 1.0), dim=-1)

        # For each token, check if it's accepted
        accepted_mask = torch.zeros((B, K), dtype=torch.bool)
        bonus_tokens_logits = torch.zeros((B, V))
        num_accepted = torch.zeros(B, dtype=torch.long)

        for b in range(B):
            for k in range(K):
                draft_token_id = draft_tokens[b, k].item()

                if is_greedy:
                    # GREEDY: Accept only if target's argmax == draft token
                    target_argmax = torch.argmax(target_logits[b, k]).item()
                    if target_argmax == draft_token_id:
                        accepted_mask[b, k] = True
                        num_accepted[b] += 1
                    else:
                        # Rejected: use target's preferred token
                        # For greedy, just use the target logits at this position
                        bonus_tokens_logits[b] = target_logits[b, k]
                        break  # Stop verification for this sequence
                else:
                    # SAMPLING: Use rejection sampling
                    p_draft_token = draft_probs[b, k, draft_token_id]
                    p_target_token = target_probs[b, k, draft_token_id]

                    # P_accept = min(1, P_target(token) / P_draft(token))
                    acceptance_prob = min(1.0, (p_target_token / (p_draft_token + 1e-10)).item())

                    # Sample whether to accept using rejection sampling
                    if torch.rand(1).item() < acceptance_prob:
                        accepted_mask[b, k] = True
                        num_accepted[b] += 1
                    else:
                        # Token rejected - sample from adjusted distribution
                        # P_adjusted(x) = max(0, P_target(x) - P_draft(x)) / (1 - P_draft(draft_token))
                        adjusted_probs = torch.relu(target_probs[b, k] - draft_probs[b, k])

                        # Normalize by (1 - p_draft(draft_token))
                        normalization = 1.0 - p_draft_token.item()
                        if normalization > 1e-6 and adjusted_probs.sum() > 1e-6:
                            adjusted_probs /= normalization
                        else:
                            # Fallback to target distribution if normalization fails
                            adjusted_probs = target_probs[b, k]

                        # Store logits for bonus token sampling
                        bonus_tokens_logits[b] = torch.log(adjusted_probs + 1e-10)
                        break  # Stop verification for this sequence

        return accepted_mask, num_accepted, bonus_tokens_logits

    def sample_accepted(self, logits, accepted_mask, num_accepted):
        """
        Samples the next token for sequences where all draft tokens were accepted,
        and the bonus token for sequences where a rejection occurred.

        Args:
            logits: [B, K, vocab_size] Target model logits for all positions
            accepted_mask: [B, K] Boolean mask of accepted tokens
            num_accepted: [B] Number of accepted tokens per sequence

        Returns:
            next_tokens: [B] Next token IDs for each sequence
            log_probs: [B] Log probabilities of sampled tokens
        """
        B, K, V = logits.shape
        next_tokens = torch.zeros(B, dtype=torch.long)
        log_probs = torch.zeros(B)

        for b in range(B):
            if num_accepted[b] == K:
                # All K tokens accepted, sample from the K-th target logits
                final_logits = logits[b, K - 1]
            else:
                # Rejection occurred, use the logits at the point of rejection
                final_logits = logits[b, num_accepted[b]]

            # Apply temperature and top_p sampling
            probs = torch.softmax(final_logits / self.temperature, dim=-1)

            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            probs[sorted_indices[sorted_indices_to_remove]] = 0
            if probs.sum() == 0:  # Fallback if all probabilities become zero
                probs = torch.softmax(final_logits / self.temperature, dim=-1)
            probs = probs / probs.sum()  # Re-normalize

            next_token = torch.multinomial(probs, num_samples=1).squeeze(0)
            next_tokens[b] = next_token
            log_probs[b] = torch.log(probs[next_token] + 1e-10)  # Add epsilon for stability

        return next_tokens, log_probs

    def compute_acceptance_rate(self, num_accepted, K):
        """
        Computes the acceptance rate for each sequence and a scalar average.

        Args:
            num_accepted: [B] Number of accepted tokens per sequence
            K: Total number of draft tokens

        Returns:
            acceptance_rate_per_sequence: [B] Acceptance rate for each sequence
            scalar_acceptance_rate: Scalar average acceptance rate
        """
        acceptance_rate_per_sequence = num_accepted.float() / K
        scalar_acceptance_rate = num_accepted.float().sum() / (num_accepted.shape[0] * K)
        return acceptance_rate_per_sequence, scalar_acceptance_rate
