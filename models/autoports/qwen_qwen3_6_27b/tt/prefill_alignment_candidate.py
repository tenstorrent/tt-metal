# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental short-capacity prefill K blocking; no runtime policy mutation."""


def schedule(length, start, temporal_chunk, alignment):
    result = []
    offset = 0
    while offset < length:
        absolute = start + offset
        count = 1 if absolute % alignment else min(temporal_chunk, length - offset)
        result.append((absolute, count))
        offset += count
    return result


def make_decoder(base):
    class PrefillAlignmentDecoder(base):
        @classmethod
        def from_state_dict(cls, state_dict, *, policy=None, **kwargs):
            policy = dict(policy or {})
            chunk = policy.setdefault("candidate_prefill_k_chunk", 128)
            limit = policy.setdefault("candidate_prefill_k_capacity_limit", 2048)
            if chunk not in (32, 64, 128, 256, 512) or limit < 512 or limit % 512:
                raise ValueError("Invalid prefill K chunk or capacity limit")
            # Smaller blocks change SDPA rounding. Bound the experiment to short
            # capacity and leave both long context and all decode numerics intact.
            return super().from_state_dict(state_dict, policy=policy, **kwargs)

        def _attention_chunk(self, state):
            original = super()._attention_chunk(state)
            table = self._attention_page_table(state)
            if original == 512 and table.shape[1] * 32 <= self.policy["candidate_prefill_k_capacity_limit"]:
                return self.policy["candidate_prefill_k_chunk"]
            return original

        def _decode_attention_chunk(self, state, *, decode=False):
            # Do not dispatch through the prefill override. This is the inherited
            # decode rule with its original 512-key baseline, including the
            # unaligned prefill single-token updates (decode=False).
            original = super()._attention_chunk(state)
            if decode and original == 512 and self._attention_page_table(state).shape[1] <= 64:
                chunk = self.policy.get("short_capacity_decode_k_chunk", 512)
                assert chunk in (32, 64, 128, 256, 512)
                return chunk
            return original

    return PrefillAlignmentDecoder
