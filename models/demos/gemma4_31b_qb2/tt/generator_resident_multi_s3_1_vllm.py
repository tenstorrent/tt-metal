# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: S3 on a plain launch only (S3.1), for API parity with the shipped `batch` profile (B32).

The S3 adapter (generator_resident_multi_s3_vllm, unchanged) plus one refusal: `spec_plan` rejects any speculative
config, so a server started with `--speculative-config` fails at admission instead of running S0's argmax_ids-only
speculative launch, which answers HTTP 400 to logprobs, structured outputs, bad_words, allowed_token_ids and min_tokens
(vllm_tt_plugin platform.py _reject_unsupported_speculative_request) and makes vLLM refuse logit_bias and min_p.
On a plain launch the plugin stores no SpecPlan, admission refuses nothing, and those requests take the host-sampled
step (S0 decode_forward with no sampling params -> _host_logits), as penalties do on B32.
B never speculated in the batch profile (policy never; S0's build_drafter returns None): the speculative config came
from the S1 serving spec, a copy of the single-user M2 spec (S1_PLAN_v3 section 1 item 7).
bench/resident/s1/BATCH_PROFILE_SWAP_PLAN_v2_ADDENDUM_1.md, LEAD_DECISIONS_S1 S18.
"""

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_vllm import Gemma4ResidentMultiS3ForCausalLM


class Gemma4ResidentMultiS31ForCausalLM(Gemma4ResidentMultiS3ForCausalLM):
    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import SpecReject

        return SpecReject(reason="The resident batch profile (S3.1) serves a plain launch: it does not speculate, and "
                                 "a speculative launch would refuse logprobs, structured outputs, bad_words, "
                                 "allowed_token_ids, min_tokens, logit_bias and min_p (API parity with B32). "
                                 "Remove --speculative-config.")
